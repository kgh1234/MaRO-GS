

import torch, os
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def visualize_mask_projection_with_centers(xy_proj, mask_img, save_path="debug/mask_check.png", point_size=5):
    """
    Visualize Gaussian 2D projections over mask image.

    Args:
        xy_proj (torch.Tensor): (N,2) projected coordinates (u,v)
        mask_img (torch.Tensor or np.ndarray): [H,W] or [1,H,W]
        save_path (str): output file path
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert mask → numpy grayscale
    if isinstance(mask_img, torch.Tensor):
        mask_np = mask_img.detach().cpu().numpy()
    else:
        mask_np = mask_img

    if mask_np.ndim == 3:
        mask_np = mask_np[0]  # [1,H,W]

    H, W = mask_np.shape
    plt.figure(figsize=(8, 6))
    plt.imshow(mask_np, cmap='gray', origin='upper')

    # Valid points only (inside image)
    u = torch.round(xy_proj[:, 0]).long()
    v = torch.round(xy_proj[:, 1]).long()

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    u_valid = u[valid].cpu().numpy()
    v_valid = v[valid].cpu().numpy()

    plt.scatter(u_valid, v_valid, s=point_size, c='red', alpha=0.7)

    plt.title(f"Mask projection check ({len(u_valid)} / {len(u)} visible)")
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"[Saved] Projection visualization → {save_path}")
    

import torch
import matplotlib.pyplot as plt
import os

@torch.no_grad()
def visualize_mask_pruning_result(
    xyz,
    viewpoint_cam,
    mask_path,
    prune_mask=None,
    invert=False,
    save_path="debug/mask_prune_vis.png",
):
    """
    Visualize Gaussians over mask with color-coded pruning results.

    Args:
        xyz (torch.Tensor): (N, 3) Gaussian centers in world coords.
        viewpoint_cam: Camera object (has .project_to_screen).
        mask_path (str): path to mask image.
        prune_mask (torch.BoolTensor): (N,) True if pruned.
        invert (bool): whether to invert mask.
        save_path (str): output visualization path.
    """

    from scene.mask_readers import _load_binary_mask

    H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # === Load mask ===
    mask = _load_binary_mask(mask_path, H, W, invert=invert).cpu().numpy()

    # === Project Gaussian points to image ===
    uv = viewpoint_cam.project_to_screen(xyz)
    u = uv[:, 0].long()
    v = uv[:, 1].long()

    v = (H - 1) - v

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if valid.sum() == 0:
        print(f"[Vis] Warning: No valid projected points for {save_path}")
        return

    u_valid = u[valid].cpu().numpy()
    v_valid = v[valid].cpu().numpy()

    # === Prepare prune mask ===
    if prune_mask is None:
        prune_mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
    prune_mask_valid = prune_mask[valid].cpu().numpy()

    # === Split kept vs pruned ===
    kept_mask = ~prune_mask_valid
    pruned_mask = prune_mask_valid

    num_kept = kept_mask.sum()
    num_pruned = pruned_mask.sum()

    # === Plot ===
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap="gray", origin="upper")

    # Kept → green / Pruned → red
    plt.scatter(
        u_valid[kept_mask],
        v_valid[kept_mask],
        s=3,
        c="lime",
        alpha=0.7,
        label=f"Kept ({num_kept})",
        edgecolors="none",
    )
    plt.scatter(
        u_valid[pruned_mask],
        v_valid[pruned_mask],
        s=3,
        c="red",
        alpha=0.6,
        label=f"Pruned ({num_pruned})",
        edgecolors="none",
    )

    plt.title(
        f"Mask-based Gaussian Pruning\nKept: {num_kept} | Pruned: {num_pruned} | HxW=({H}x{W})"
    )
    plt.legend(loc="upper right")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[Vis] Saved mask pruning visualization → {save_path}")

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scene.mask_readers import _find_mask_path, _load_binary_mask

@torch.no_grad()
def visualize_mask_overlap_on_mask(
    xyz,
    scene,
    overlap_sum,
    view_count,
    mask_dir,
    mask_invert=False,
    save_path="debug/mask_overlap_heatmap.png",
    cmap="jet"
):
    """
    Visualize Gaussian overlap ratio heatmap over the binary mask background.

    Args:
        xyz (torch.Tensor): (N,3) Gaussian centers in world coordinates.
        scene: Scene object (has getTrainCameras()).
        overlap_sum (torch.Tensor): accumulated mask values per Gaussian.
        view_count (torch.Tensor): number of views where Gaussian was seen.
        mask_dir (str): directory containing binary mask images.
        mask_invert (bool): whether to invert mask pixel values.
        save_path (str): path to save visualization.
        cmap (str): colormap for heat intensity.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = xyz.device
    overlap_ratio = overlap_sum / (view_count + 1e-6)
    overlap_ratio[overlap_ratio.isnan()] = 0.0

    # Pick one representative camera (center view)
    views = scene.getTrainCameras()
    if len(views) == 0:
        print("[Vis] No cameras found in scene.")
        return
    cam = views[len(views) // 2]

    H, W = cam.image_height, cam.image_width
    mask_path = _find_mask_path(mask_dir, cam.image_name)
    if not mask_path:
        print(f"[Vis] No mask found for {cam.image_name}")
        return

    # === Load mask ===
    mask = _load_binary_mask(mask_path, H, W, invert=mask_invert).cpu().numpy()

    # === Project Gaussians ===
    uv = cam.project_to_screen(xyz)
    u = uv[:, 0].detach().cpu().numpy()
    v = uv[:, 1].detach().cpu().numpy()

    v = (H - 1) - v

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u_valid = u[valid]
    v_valid = v[valid]

    overlap_ratio_np = overlap_ratio.detach().cpu().numpy()

    if len(overlap_ratio_np) != len(valid):
        print(f"[Warn] overlap_ratio ({len(overlap_ratio_np)}) vs valid ({len(valid)}) mismatch — trimming to min length")
        min_len = min(len(overlap_ratio_np), len(valid))
        overlap_ratio_np = overlap_ratio_np[:min_len]
        valid = valid[:min_len]
        u_valid = u_valid[:min_len]
        v_valid = v_valid[:min_len]

    overlap_valid = overlap_ratio_np[valid]

    # === Visualization ===
    plt.figure(figsize=(10, 7))
    plt.imshow(mask, cmap='gray', origin='upper')
    sc = plt.scatter(
        u_valid,
        v_valid,
        s=4,
        c=overlap_valid,
        cmap=cmap,
        alpha=0.8,
        vmin=0.0,
        vmax=min(1.0, overlap_valid.max() + 1e-3)
    )

    cbar = plt.colorbar(sc, pad=0.02)
    cbar.set_label("Mask Overlap Ratio (0–1)", fontsize=10)
    plt.title(f"Mask + Gaussian Overlap Heatmap\nH×W=({H}×{W}), points={len(u_valid)}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Vis] Saved mask overlap heatmap → {save_path}")
