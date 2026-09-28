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

import math
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


    def project_to_screen_self(self, xyz: torch.Tensor):
        """
        Project 3D points (world space) into image pixel coordinates
        using intrinsic/extrinsic derived from the Camera object (COLMAP style).
        """

        device = xyz.device
        W, H = self.image_width, self.image_height

        # === derive fx, fy, cx, cy from FOV ===
        fx = (W / 2.0) / math.tan(self.FoVx / 2.0)
        fy = (H / 2.0) / math.tan(self.FoVy / 2.0)
        cx = W / 2.0
        cy = H / 2.0

        # === convert R,T into torch tensors ===
        R = torch.tensor(self.R, dtype=torch.float32, device=device)
        T = torch.tensor(self.T, dtype=torch.float32, device=device)

        # === world → camera ===
        xyz_cam = (R @ xyz.T + T.unsqueeze(1)).T  # (N,3)
        X, Y, Z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2].clamp(min=1e-6)

        # === camera → pixel (COLMAP style: +Z forward, +Y down) ===
        u = fx * (-X / Z) + cx
        v = fy * (-Y / Z) + cy  # flip Y to align with top-left origin

        u = torch.clamp(u, 0, W - 1)
        v = torch.clamp(v, 0, H - 1)

        #print(f"[Proj2Screen-ColmapLike] u range: {u.min():.1f}~{u.max():.1f}, v range: {v.min():.1f}~{v.max():.1f}, mask size: ({H},{W})")

        return torch.stack([u, v], dim=1)
    
    def project_to_screen(self, xyz):
        """
        Match exactly how diff_gaussian_rasterization projects 3D points.
        Uses tan(fov) scaling (same as rasterizer C++).
        """
        device = xyz.device
        W, H = self.image_width, self.image_height
        
    
        viewmatrix=self.world_view_transform
        projmatrix=self.full_proj_transform
        # world → camera
        ones = torch.ones((xyz.shape[0], 1), device=device)
        xyz_h = torch.cat([xyz, ones], dim=1)
        cam_h = xyz_h @ self.world_view_transform
        cam = cam_h[:, :3]

        # perspective division (z-forward)
        X, Y, Z = cam[:, 0], cam[:, 1], cam[:, 2].clamp(min=1e-6)

        # === rasterizer uses tan(fov) directly ===
        tan_fovx = math.tan(self.FoVx * 0.5)
        tan_fovy = math.tan(self.FoVy * 0.5)

        # normalized by FOV
        x_ndc = X / (Z * tan_fovx)
        y_ndc = Y / (Z * tan_fovy)

        # map [-1,1] → [0,W/H]
        u = (x_ndc * 0.5 + 0.5) * W
        v = (1.0 - (y_ndc * 0.5 + 0.5)) * H

        u = u.clamp(0, W - 1)
        v = v.clamp(0, H - 1)
        #print(f"[Proj2Screen-GSReal] u {u.min():.1f}~{u.max():.1f}, v {v.min():.1f}~{v.max():.1f}, ({H},{W})")

        return torch.stack([u, v], dim=1)




    # def project_to_screen(self, xyz: torch.Tensor):
    #     device = xyz.device
    #     W, H = self.image_width, self.image_height

    #     ones = torch.ones((xyz.shape[0], 1), device=device)


    #     xyz_h = torch.cat([xyz, ones], dim=1)  # (N,4)
    #     #xyz_h[:, :3] *= self.scale
    #     clip_h = xyz_h @ self.full_proj_transform.T  # (N,4)

    #     ndc = clip_h[:, :3] / (clip_h[:, 3:4] + 1e-6)

    #     # u = (ndc[:, 0] * 0.5 + 0.5) * W
    #     # v = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * H  # image y-down

        
    #     X, Y, Z = clip_h[:, 0], clip_h[:, 1], clip_h[:, 2].clamp(min=1e-6)

    #     # === pinhole projection (no NDC) ===
        
    #     fx = (W / 2.0) / math.tan(self.FoVx / 2.0)
    #     fy = (H / 2.0) / math.tan(self.FoVy / 2.0)
    #     cx, cy = W / 2.0, H / 2.0
    #     u = fx * (X / Z) + cx
    #     v = fy * (Y / Z) + cy
        
        
    #     u = u.clamp(0, W - 1)
    #     v = v.clamp(0, H - 1)

    #     print(f"[Proj2Screen-FullPV] u: {u.min():.1f}~{u.max():.1f}, v: {v.min():.1f}~{v.max():.1f}, mask ({H},{W})")
    #     return torch.stack([u, v], dim=1)






        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

