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
import matplotlib
from utils.mask_projection_visualization import visualize_mask_overlap_on_mask, visualize_mask_projection_with_centers

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.mask_readers import _find_mask_path, _load_binary_mask

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        if not hasattr(self, "_comp_state"):
            self._comp_state = {}

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)
        
    
        
        
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii,
                    mask_dir=None,
                    mask_disabled=False,
                    scene=None,
                    iter=None,
                    viewpoint_camera=None,
                    mask_prune_iter=[600,1200,1800],
                    mask_invert=False,
                    prune_ratio=1.0,
                    mask_threshold=0.3,
                    pipeline=None,      
                    background=None,     
                    prev_brightness=None,
                    k=0.5,
                    max_pruning=0.05
                    ):  

        from scene.view_consistency import gaussian_mask_overlap

        self.tmp_radii = radii
        xyz = self.get_xyz.detach()
        num_points = xyz.shape[0]
        device = xyz.device

        # ==========================================================
        # Global mask-based pruning
        # ==========================================================
        if not mask_disabled and mask_dir is not None and mask_prune_iter is not None and iter in mask_prune_iter:
            print(f"[MaskPrune@{iter}] Start global mask-based pruning...")

            overlap_ratio, avg_mask_ratio, overlap_sum, view_count, view_ratios = gaussian_mask_overlap(
                xyz=self.get_xyz,
                scene=scene,
                mask_dir=mask_dir,
                mask_invert=mask_invert,
                mask_disabled=mask_disabled,
                iter=iter
            )


            visualize_mask_overlap_on_mask(
                xyz=self.get_xyz.detach(),
                scene=scene,
                overlap_sum=overlap_sum,
                view_count=view_count,
                mask_dir=mask_dir,
                mask_invert=mask_invert,
                save_path=os.path.join(scene.model_path, f"mask_overlap_iter{iter}.png")
            )

            # ==============================
            # Adaptive Soft + Hard Pruning
            # ==============================
            mean = overlap_ratio.mean()
            std = overlap_ratio.std()

            coverage_scale = np.clip(avg_mask_ratio, 0.1, 1.0)
            mask_weight = 1.0 - 0.5 * (1.0 - coverage_scale)  # 0.5~1.0

            dist_spread = (std / (mean + 1e-6)).clamp(0, 5)
            pruning_strength = torch.tanh(dist_spread).item()  # 0~1
            base_thresh = mean + k * std
            base_thresh = max(base_thresh.item(), 0.002)

            soft_thresh = base_thresh * (1.0 - 0.6 * pruning_strength * mask_weight)
            soft_thresh = np.clip(soft_thresh, 0.001, max_pruning)


            hard_thresh = max(0.6 * mean.item(), 0.002)

            print(f"[MaskPrune@{iter}] mean={mean:.4f}, std={std:.4f}, spread={dist_spread:.2f}, "
                f"strength={pruning_strength:.2f}, mask_ratio={avg_mask_ratio:.2f} → "
                f"soft_thresh={soft_thresh:.4f}, hard_thresh={hard_thresh:.4f}")

            prune_mask = overlap_ratio < soft_thresh
            prune_mask[overlap_ratio < hard_thresh] = True 

            num_prune = prune_mask.sum().item()
            num_keep = (~prune_mask).sum().item()

            if prune_ratio is not None:
                prune_idx = torch.where(prune_mask)[0]

                target_prune = int(num_prune * prune_ratio)

                if target_prune < num_prune:
                    prune_overlap = overlap_ratio[prune_idx]
                    prune_order = torch.argsort(prune_overlap)  


                    to_prune_idx = prune_idx[prune_order[:target_prune]]
                    to_restore_idx = prune_idx[prune_order[target_prune:]]

                    prune_mask[to_restore_idx] = False


                num_prune = prune_mask.sum().item()
                num_keep = (~prune_mask).sum().item()

                print(f"[MaskPrune@{iter}] prune_ratio applied → "
                    f"target_prune={target_prune}, final_prune={num_prune}, keep={num_keep}")

            
            self.prune_points(prune_mask)
            print(f"[MaskPrune@{iter}] pruned={num_prune}, kept={num_keep}")

                

            return None



        # ==========================================================
        # Gradient computation
        # ==========================================================
        grads = self.xyz_gradient_accum / self.denom
        grads = torch.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
        densify_grads = grads.clone()
        grad_norm = torch.norm(densify_grads, dim=1)

        # ==========================================================
        # Local mask-based densification suppression
        # ==========================================================
        if viewpoint_camera is not None and mask_dir is not None:
            xyz = self.get_xyz.detach()
            num_points = xyz.shape[0]
            device = xyz.device
                    
            mask_path = _find_mask_path(mask_dir, viewpoint_camera.image_name)
            if mask_path:
                H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
                mask = _load_binary_mask(mask_path, H, W, invert=mask_invert, device=device)

                uv = viewpoint_camera.project_to_screen(xyz)
                u, v_ = uv[:, 0].long(), uv[:, 1].long()
                v_ = (H - 1) - v_
                valid = (u >= 0) & (u < W) & (v_ >= 0) & (v_ < H)

                mask_vals = torch.zeros(num_points, device=device)
                mask_vals[valid] = mask[v_[valid], u[valid]].float()

                outside_mask = (mask_vals < mask_threshold)
                densify_grads[outside_mask] = 0.0
                grad_norm[outside_mask] = 0.0

        
        
        # ------------- original densification and pruning -----------------
        self.densify_and_clone(densify_grads, max_grad, extent)
        self.densify_and_split(densify_grads, max_grad, extent)


        # ----------------------- original pruning and densification -----------------------
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            
            
        self.prune_points(prune_mask)
        
        
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    # ----------------------- Gaussian Merging -----------------------
    def merge_similar_neighbors(self, color_threshold=0.2, neighbor_radius=0.4, min_group_size=3):

        xyz = self.get_xyz.detach()
        colors = self._features_dc.detach().squeeze(1)
        scales = self.get_scaling.detach()
        opacities = self.get_opacity.detach()
        rotations = self._rotation.detach()
        features_rest = self._features_rest.detach()
        
        N = xyz.shape[0]
        merged = torch.zeros(N, dtype=torch.bool, device=xyz.device)
        
        new_xyz_list = []
        new_colors_list = []
        new_scales_list = []
        new_opacities_list = []
        new_rotations_list = []
        new_features_rest_list = []
        
        avg_scales = scales.mean(dim=1)
        
        print(f"[Merging] Processing {N} gaussians...")
        
        for i in range(N):
            if merged[i]:
                continue
            
            if i % 10000 == 0:
                print(f"  Progress: {i}/{N}")
            
            current_xyz = xyz[i:i+1]
            current_color = colors[i:i+1]
            neighbor_threshold = avg_scales[i] * neighbor_radius
            

            dists = torch.norm(xyz - current_xyz, dim=1)
            is_close = (dists < neighbor_threshold) & (dists > 1e-6)
            
            if is_close.sum() < min_group_size - 1:

                new_xyz_list.append(xyz[i:i+1])
                new_colors_list.append(colors[i:i+1])
                new_scales_list.append(self._scaling[i:i+1])
                new_opacities_list.append(self._opacity[i:i+1])
                new_rotations_list.append(rotations[i:i+1])
                new_features_rest_list.append(features_rest[i:i+1])
                merged[i] = True
                continue
            

            neighbor_indices = torch.where(is_close)[0]
            neighbor_colors = colors[neighbor_indices]
            color_diffs = torch.abs(current_color - neighbor_colors).mean(dim=1)
            similar_mask = color_diffs < color_threshold
            
            if similar_mask.sum() < min_group_size - 1:

                new_xyz_list.append(xyz[i:i+1])
                new_colors_list.append(colors[i:i+1])
                new_scales_list.append(self._scaling[i:i+1])
                new_opacities_list.append(self._opacity[i:i+1])
                new_rotations_list.append(rotations[i:i+1])
                new_features_rest_list.append(features_rest[i:i+1])
                merged[i] = True
                continue
            
            merge_indices = torch.cat([torch.tensor([i], device=xyz.device), 
                                       neighbor_indices[similar_mask]])
            

            merged_xyz = xyz[merge_indices].mean(dim=0, keepdim=True)
            merged_color = colors[merge_indices].mean(dim=0, keepdim=True)
            merged_scale = self._scaling[merge_indices].mean(dim=0, keepdim=True)
            
            merged_opacity_logit = torch.log(
                opacities[merge_indices].sum().clamp(min=0.01, max=0.99) / 
                (1 - opacities[merge_indices].sum().clamp(min=0.01, max=0.99))
            ).unsqueeze(0).unsqueeze(1)
            
            merged_rotation = rotations[merge_indices].mean(dim=0, keepdim=True)
            merged_rotation = merged_rotation / merged_rotation.norm()  # normalize
            
            merged_features_rest = features_rest[merge_indices].mean(dim=0, keepdim=True)
            
            new_xyz_list.append(merged_xyz)
            new_colors_list.append(merged_color)
            new_scales_list.append(merged_scale)
            new_opacities_list.append(merged_opacity_logit)
            new_rotations_list.append(merged_rotation)
            new_features_rest_list.append(merged_features_rest)
            

            merged[merge_indices] = True
            
            if i % 1000 == 0:
                torch.cuda.empty_cache()
        

        new_xyz = torch.cat(new_xyz_list, dim=0)
        new_colors = torch.cat(new_colors_list, dim=0)
        new_scales = torch.cat(new_scales_list, dim=0)
        new_opacities = torch.cat(new_opacities_list, dim=0)
        new_rotations = torch.cat(new_rotations_list, dim=0)
        new_features_rest = torch.cat(new_features_rest_list, dim=0)
        
        print(f"[Merging] {N} → {new_xyz.shape[0]} gaussians ({N - new_xyz.shape[0]} merged)")
        
        return {
            'xyz': new_xyz,
            'colors': new_colors.unsqueeze(1),
            'scales': new_scales,
            'opacities': new_opacities,
            'rotations': new_rotations,
            'features_rest': new_features_rest
        }





