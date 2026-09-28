#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#



import os
import torch
import sys
import cv2
import glob
import uuid
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # for headless environment

from mpl_toolkits.mplot3d import Axes3D
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui


from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.stopper import GaussianStatStopper

from utils.mask_projection_visualization import visualize_mask_projection_with_centers
from scene.view_consistency import compute_view_jaccard, compute_view_jaccard_fast
from scene.view_consistency import gaussian_mask_overlap
from scene.mask_readers import _find_mask_path, _load_binary_mask, _load_binary_mask_from_dir


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def str2bool(v):
    return v.lower() not in ('false', '0', 'no')

def soft_mask_iou_loss(pred_mask, gt_mask):
    pred_mask = pred_mask.clamp(0.0, 1.0)
    gt_mask = gt_mask.clamp(0.0, 1.0)
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum() - intersection
    return 1.0 - (intersection + 1e-6) / (union + 1e-6)

def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from,
             mask_dir=None, mask_binary_threshold=128, mask_invert=False, mask_disabled=False, prune_iter=None, prune_ratio=1.0, 
             cov_threshold=None, hit_ratio=None, threshold_prune_k=None, max_pruning=None, 
             geometric_filtering=True, region_filtering=True, gaussian_merge=True,
             filtering_start=1000, filtering_interval=1000, filtering_end=15000,
             merge_start=0, merge_step=2000, merge_end=5000,
             projection_debug=False, projection_debug_count=6,
             object_weight_loss=True,
             mask_iou_loss=True, mask_iou_weight=0.0, mask_iou_start_iter=0, mask_iou_interval=10):


    if geometric_filtering:
        print("GEOMETRIC_FILTERING ON")
    else:
        print("GEOMETRIC_FILTERING OFF")

    if region_filtering:
        print("REGION_FILTERING ON")
    else:
        print("REGION_FILTERING OFF")
        
    if prune_ratio==0:
        print('PRUNING OFF')
    else:
        print(f'PRUNING: {prune_ratio}')

    if gaussian_merge:
        print("GAUSSIAN_MERGE ON")
    else:
        print("GAUSSIAN_MERGE OFF")

    print("maskdir", mask_dir)
    print("mask_disabled", mask_disabled)
    print("object_weight_loss", object_weight_loss)
    print("mask_iou_loss", mask_iou_loss, "mask_iou_weight", mask_iou_weight)


    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    train_views = scene.getTrainCameras().copy()
    viewpoint_stack = train_views.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    
    opacity_log = []
    shdc_log = []
    iter_log = []


    stopper = GaussianStatStopper(patience=500, min_delta=1e-5)
    print(f"Prune ratio : {prune_ratio}")
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=use_sparse_adam)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # filtering on/off
        if geometric_filtering:
            current_filtering_start = int(filtering_start)
            current_filtering_end = min(int(filtering_end), int(opt.densify_until_iter) - 1)
            current_filtering_interval = max(1, int(filtering_interval))
            should_filter = (
                iteration >= current_filtering_start
                and iteration <= current_filtering_end
                and (iteration - current_filtering_start) % current_filtering_interval == 0
            )
            if should_filter:
                bad_idx = compute_view_jaccard_fast(
                    scene,
                    gaussians,
                    pipe,
                    background,
                    mask_dir=mask_dir if (mask_dir is not None and len(mask_dir) > 0) else None,
                    mask_disabled=mask_disabled,
                    mask_invert=mask_invert,
                    views=train_views,
                    debug_projection=projection_debug,
                    debug_iter=iteration,
                    debug_count=projection_debug_count,
                )
                if len(bad_idx) > 0:
                    filtered_train_views = [v for i, v in enumerate(train_views) if i not in bad_idx]
                    if len(filtered_train_views) > 0:
                        train_views = filtered_train_views
                        viewpoint_stack = train_views.copy()
                        viewpoint_indices = list(range(len(viewpoint_stack)))
                        print(f"[Iter {iteration}] Removed {len(bad_idx)} low-consistency views from training.")
                        print(f"[INFO] Remaining training views: {len(train_views)}")
                    else:
                        print(f"[Iter {iteration}] Skipped geometric filtering because it would remove all training views.")


        if region_filtering:
            if iteration == 1800 and iteration < opt.densify_until_iter:
                from scene.view_consistency import gaussian_view_consistency
                bad_idx = gaussian_view_consistency(
                    scene=scene,
                    gaussians=gaussians,
                    mask_disabled=mask_disabled,
                    mask_dir=mask_dir,
                    mask_invert=mask_invert,
                    threshold=hit_ratio,
                    views=train_views,
                )
                if bad_idx is not None and len(bad_idx) > 0:
                    removed_views = [v for i, v in enumerate(train_views) if i in bad_idx]
                    removed_names = [v.image_name for v in removed_views]
                    filtered_train_views = [v for i, v in enumerate(train_views) if i not in bad_idx]
                    if len(filtered_train_views) > 0:
                        train_views = filtered_train_views
                        viewpoint_stack = train_views.copy()
                        viewpoint_indices = list(range(len(viewpoint_stack)))
                        for i, name in zip(bad_idx, removed_names):
                            print(f"  - View {i:03d}: {name}")
                        print(f"[INFO] Remaining training views: {len(train_views)}")
                    else:
                        print(f"[Iter {iteration}] Skipped region filtering because it would remove all training views.")
            
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = train_views.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=use_sparse_adam)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        
        gt_image = viewpoint_cam.original_image.cuda()
        
        
        use_mask = mask_dir is not None and len(mask_dir) > 0
        loaded_mask = None
        image_mask = None
        if use_mask and not mask_disabled:
            loaded_mask = _load_binary_mask_from_dir(
                mask_dir,
                viewpoint_cam.image_name,
                image.shape[1],
                image.shape[2],
                binary_threshold=mask_binary_threshold,
                invert=mask_invert,
            )
            if loaded_mask is not None:
                image_mask = loaded_mask.unsqueeze(0).expand_as(gt_image)
                gt_image = gt_image * image_mask

        if object_weight_loss and use_mask and (iteration > prune_iter[0]):  
            if mask_disabled:
                mask = torch.ones_like(gt_image)
            else:
                if image_mask is not None:
                    mask = image_mask
                else:
                    mask = torch.ones_like(gt_image)

            diff = torch.abs(image - gt_image) * mask
            out_diff = torch.abs(image - gt_image) * (1 - mask) * 0.01
            Ll1 = (diff.sum() + out_diff.sum()) / (image.numel() / image.shape[0] + 1e-8)

            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)) if FUSED_SSIM_AVAILABLE \
                         else ssim(image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        
        
        else:
            Ll1 = l1_loss(image, gt_image)
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)) if FUSED_SSIM_AVAILABLE \
                         else ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        Lmask_iou = image.new_tensor(0.0)
        if (
            mask_iou_loss
            and mask_iou_weight > 0.0
            and use_mask
            and not mask_disabled
            and loaded_mask is not None
            and iteration >= mask_iou_start_iter
            and mask_iou_interval > 0
            and ((iteration - mask_iou_start_iter) % mask_iou_interval == 0)
        ):
            silhouette_bg = torch.zeros((3), dtype=image.dtype, device="cuda")
            silhouette_color = torch.ones((gaussians.get_xyz.shape[0], 3), dtype=image.dtype, device="cuda")
            silhouette_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                silhouette_bg,
                override_color=silhouette_color,
                use_trained_exp=False,
                separate_sh=False,
            )
            pred_mask = silhouette_pkg["render"].mean(dim=0, keepdim=True)
            gt_mask = loaded_mask.unsqueeze(0).to(device=pred_mask.device, dtype=pred_mask.dtype)
            Lmask_iou = soft_mask_iou_loss(pred_mask, gt_mask)
            loss = loss + mask_iou_weight * Lmask_iou

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.update(10)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

        

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    bad_idx = gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii,
                        mask_dir=mask_dir if use_mask else None,
                        mask_disabled=mask_disabled,
                        scene=scene,
                        viewpoint_camera=viewpoint_cam,
                        iter=iteration,
                        mask_prune_iter=prune_iter,
                        prune_ratio=prune_ratio,
                        pipeline=pipe,            
                        background=background,
                        k=threshold_prune_k,
                        max_pruning=max_pruning
                    )

                    if gaussian_merge:
                        merge_interval = max(1, int(merge_step))
                        if (
                            iteration >= int(merge_start)
                            and iteration < int(merge_end)
                            and (iteration - int(merge_start)) % merge_interval == 0
                        ):
                            merged_params = gaussians.merge_similar_neighbors(
                                color_threshold=0.2,
                                neighbor_radius=0.4,
                                min_group_size=3
                            )
                            
                            gaussians._xyz = nn.Parameter(merged_params['xyz'].requires_grad_(True))
                            gaussians._features_dc = nn.Parameter(merged_params['colors'].requires_grad_(True))
                            gaussians._scaling = nn.Parameter(merged_params['scales'].requires_grad_(True))
                            gaussians._opacity = nn.Parameter(merged_params['opacities'].requires_grad_(True))
                            gaussians._rotation = nn.Parameter(merged_params['rotations'].requires_grad_(True))
                            gaussians._features_rest = nn.Parameter(merged_params['features_rest'].requires_grad_(True))
                            
                            gaussians.xyz_gradient_accum = torch.zeros((gaussians._xyz.shape[0], 1), device="cuda")
                            gaussians.denom = torch.zeros((gaussians._xyz.shape[0], 1), device="cuda")
                            gaussians.max_radii2D = torch.zeros((gaussians._xyz.shape[0]), device="cuda")
                            
                            gaussians.training_setup(opt)        



            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    

    parser.add_argument("--mask_dir", type=str, default="")
    parser.add_argument("--mask_binary_threshold", type=int, default=128)
    parser.add_argument("--mask_invert", action="store_true")
    
    parser.add_argument("--mask_disabled", action="store_true", default=False)


    # view filtering
    parser.add_argument("--geometric_filtering", type=str2bool, default=True) # off = False
    parser.add_argument("--region_filtering", type=str2bool, default=True) # off = False
    parser.add_argument("--gaussian_merge", type=str2bool, default=True)

    # pruning
    parser.add_argument('--prune_ratio', type=float, default=1.0) # off = 0
    parser.add_argument('--prune_iterations', nargs="+", type=int, default=[1000, 3000, 5000]) # when to prune


    # hyperparameter
    parser.add_argument("--cov_threshold", type=float, default=0.2)
    parser.add_argument("--hit_ratio", type=float, default=0.05)
    parser.add_argument("--threshold_prune_k", type=float, default=0.5)
    parser.add_argument("--pruning_max", type=float, default=0.05)
    parser.add_argument("--object_weight_loss", type=str2bool, default=True)
    parser.add_argument("--mask_iou_loss", type=str2bool, default=True)
    parser.add_argument("--mask_iou_weight", type=float, default=1.0)
    parser.add_argument("--mask_iou_start_iter", type=int, default=5000)
    parser.add_argument("--mask_iou_interval", type=int, default=100)

    parser.add_argument("--filtering_start", type=int, default=1000)
    parser.add_argument("--filtering_interval", type=int, default=2000)
    parser.add_argument("--filtering_end", type=int, default=5000)
    parser.add_argument("--projection_debug", action="store_true", default=False)
    parser.add_argument("--projection_debug_count", type=int, default=6)

    # gaussian merging
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step", type=int, default=2000)
    parser.add_argument("--end", type=int, default=5000)


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), 
            args.test_iterations, args.save_iterations, 
            args.checkpoint_iterations, args.start_checkpoint, 
            args.debug_from,
            mask_dir=args.mask_dir if args.mask_dir else None,
            mask_binary_threshold=args.mask_binary_threshold,
            mask_invert=args.mask_invert,
            mask_disabled=args.mask_disabled,
            prune_iter=args.prune_iterations,
            prune_ratio=args.prune_ratio,
            cov_threshold=args.cov_threshold,
            hit_ratio=args.hit_ratio,
            threshold_prune_k=args.threshold_prune_k, 
            max_pruning=args.pruning_max, 
            geometric_filtering=args.geometric_filtering, 
            region_filtering=args.region_filtering,
            gaussian_merge=args.gaussian_merge,
            filtering_start=args.filtering_start,
            filtering_interval=args.filtering_interval,
            filtering_end=args.filtering_end,
            merge_start=args.start,
            merge_step=args.step,
            merge_end=args.end,
            projection_debug=args.projection_debug,
            projection_debug_count=args.projection_debug_count,
            object_weight_loss=args.object_weight_loss,
            mask_iou_loss=args.mask_iou_loss,
            mask_iou_weight=args.mask_iou_weight,
            mask_iou_start_iter=args.mask_iou_start_iter,
            mask_iou_interval=args.mask_iou_interval
            )

    # All done
    print("\nTraining complete.")
