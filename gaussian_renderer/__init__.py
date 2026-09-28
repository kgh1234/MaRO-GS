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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    separate_sh: bool = False,
    override_color=None,
    use_trained_exp: bool = False,
):
    """
    Render the scene.

    Args:
        viewpoint_camera: camera with FoVx/FoVy, image_width/height, transforms, etc.
        pc: GaussianModel
        pipe: pipeline config (sh_degree/prefiltered/debug[/antialiasing])
        bg_color: CUDA tensor (3,) in [0,1]
    """

    # screen-space points to collect grads
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)
    bg = bg_color
    viewmatrix = viewpoint_camera.world_view_transform
    projmatrix = viewpoint_camera.full_proj_transform

    # prefer model's active SH degree if present
    sh_degree = getattr(pc, "active_sh_degree", None)
    if sh_degree is None:
        sh_degree = getattr(pipe, "sh_degree", 0)

    # Build raster settings robustly across versions
    try:
        raster_settings = GaussianRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier=scaling_modifier,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            sh_degree=sh_degree,
            prefiltered=getattr(pipe, "prefiltered", False),
            debug=getattr(pipe, "debug", False),
            antialiasing=getattr(pipe, "antialiasing", False),
            campos=viewpoint_camera.camera_center,
        )
    except TypeError:
        try:
            raster_settings = GaussianRasterizationSettings(
                image_height=H,
                image_width=W,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg,
                scale_modifier=scaling_modifier,
                viewmatrix=viewmatrix,
                projmatrix=projmatrix,
                sh_degree=sh_degree,
                prefiltered=getattr(pipe, "prefiltered", False),
                debug=getattr(pipe, "debug", False),
                campos=viewpoint_camera.camera_center,
            )
        except TypeError:
            raster_settings = GaussianRasterizationSettings(
                image_height=H,
                image_width=W,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg,
                scale_modifier=scaling_modifier,
                viewmatrix=viewmatrix,
                projmatrix=projmatrix,
                sh_degree=sh_degree,
                prefiltered=getattr(pipe, "prefiltered", False),
                debug=getattr(pipe, "debug", False),
            )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # covariance or scale/rot
    scales = None
    rotations = None
    cov3D_precomp = None
    if getattr(pipe, "compute_cov3D_python", False):
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # SH / color handling
    shs = None
    colors_precomp = None
    if override_color is None:
        if getattr(pipe, "convert_SHs_python", False):
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize (some builds return 3 outputs, others return 2)
    if separate_sh:
        rendered_out = rasterizer(
            means3D=means3D,
            means2D=means2D,
            dc=dc,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
    else:
        rendered_out = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

    if isinstance(rendered_out, (list, tuple)) and len(rendered_out) == 3:
        rendered_image, radii, depth_image = rendered_out
    else:
        rendered_image, radii = rendered_out
        depth_image = None

    # Apply exposure (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = (
            torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1)
            + exposure[:3, 3, None, None]
        )

    rendered_image = rendered_image.clamp(0, 1)
    
    

    eps = 1e-8
    vis_mask = (radii > eps).to(rendered_image.dtype)
    if depth_image is not None:
        depth_mask = (depth_image > 0).to(rendered_image.dtype)  # (H, W)
        mask = depth_mask
    else:
        mask = vis_mask

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)


    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        # Training indexes per-Gaussian statistics with this tensor, so it
        # must remain a boolean per-Gaussian visibility mask.
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth_image,                   
        "mask": mask,                            
    }
    return out
