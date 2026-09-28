#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import numpy as np
import cv2
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background,
               train_test_exp, separate_sh, mask_threshold):

    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    mask_path = os.path.join(model_path, name, f"ours_{iteration}", "masks")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)

    silhouette_background = torch.zeros(3, dtype=torch.float32, device="cuda")
    silhouette_color = torch.ones(
        (gaussians.get_xyz.shape[0], 3),
        dtype=gaussians.get_xyz.dtype,
        device=gaussians.get_xyz.device,
    )

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name} set")):
        # === 렌더링 ===
        out = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = out["render"]
        gt = view.original_image[0:3, :, :]
        silhouette_out = render(
            view,
            gaussians,
            pipeline,
            silhouette_background,
            override_color=silhouette_color,
            use_trained_exp=False,
            separate_sh=False,
        )
        silhouette = silhouette_out["render"].mean(dim=0, keepdim=True)

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]
            silhouette = silhouette[..., silhouette.shape[-1] // 2:]

        # === RGB & GT 저장 ===
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))
        
        mask = (silhouette >= mask_threshold).to(rendering.dtype)
        torchvision.utils.save_image(mask, os.path.join(mask_path, f"{idx:05d}.png"))






def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool, separate_sh: bool,
                mask_threshold: float):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter,
                       scene.getTrainCameras(), gaussians, pipeline, background,
                       dataset.train_test_exp, separate_sh, mask_threshold)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter,
                       scene.getTestCameras(), gaussians, pipeline, background,
                       dataset.train_test_exp, separate_sh, mask_threshold)


if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering script with mask output")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--test_list", type=str, default="", help="파일명 목록(한 줄 하나). 여기에 포함된 train 카메라만 test로 렌더")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--mask_threshold", default=0.5, type=float)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print("Rendering model:", args.model_path)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test, False, args.mask_threshold)
