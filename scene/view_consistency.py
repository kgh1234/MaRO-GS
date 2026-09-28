import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import random
import numpy as np
matplotlib.use('Agg')  # for headless environment


from gaussian_renderer import render, network_gui
from scene.mask_readers import _find_mask_path, _load_binary_mask, _load_binary_mask_from_dir  


# =========================
# Gaussian Overlap Calulation
# =========================

@torch.no_grad()
def gaussian_mask_overlap(xyz, scene, mask_dir, mask_disabled=False, mask_invert=False, iter=0, views=None):
    """
    Compute per-Gaussian overlap ratio with 2D binary masks across all training views.

    Args:
        xyz (torch.Tensor): (N, 3) Gaussian centers in world coordinates
        scene (Scene): 3DGS Scene object with camera intrinsics/extrinsics
        mask_dir (str): directory containing GT or binary masks (same name as images)
        mask_invert (bool): if True, invert mask colors (object ↔ background)
        mask_disabled (bool): if True, disable mask-based filtering
        iter (int): current training iteration (for logging/saving)
        views (list, optional): cameras to evaluate; defaults to scene.getTrainCameras()
    Returns:
        overlap_ratio (torch.Tensor): (N,) average overlap ratio across visible views
        avg_mask_ratio (float): average object coverage ratio per view
        overlap_sum (torch.Tensor): (N,) accumulated overlaps
        view_count (torch.Tensor): (N,) number of views that saw each Gaussian
        view_ratios (list[float]): list of mean overlap per view
    """
    views = scene.getTrainCameras() if views is None else views
    n_views = len(views)

    overlap_sum = torch.zeros(xyz.shape[0], device=xyz.device)
    view_count = torch.zeros_like(overlap_sum)

    mask_coverage_all = []
    view_ratios = []

    for v_idx, v in enumerate(views):
        H, W = v.image_height, v.image_width

        mask_path = _find_mask_path(mask_dir, v.image_name)
        if not mask_path:
            continue

        if mask_disabled:
            mask = np.ones((H, W), dtype=np.float32)
        else:
            mask = _load_binary_mask(mask_path, H, W, invert=mask_invert).cpu().numpy()
            mask_coverage_all.append(mask.mean())

        uv = v.project_to_screen(xyz)
        u = uv[:, 0].long()
        v_ = uv[:, 1].long()
        v_ = (H - 1) - v_
        valid = (u >= 0) & (u < W) & (v_ >= 0) & (v_ < H)

        if valid.sum() == 0:
            view_ratios.append(0.0)
            continue

        u_idx_img = u[valid].cpu().numpy()
        v_idx_img = v_[valid].cpu().numpy()
        mask_vals = mask[v_idx_img, u_idx_img]

        overlap_sum[valid] += torch.tensor(mask_vals, device=xyz.device, dtype=torch.float32)
        view_count[valid] += 1.0

        mean_overlap_view = float(np.mean(mask_vals))
        mask_coverage_val = float(mask.mean())
        #print(f"[Overlap@{iter}] View {v_idx:03d}: {mean_overlap_view:.4f} mean overlap, ")

        view_ratios.append(mean_overlap_view)

        # print(f"[Overlap@{iter}] View {v_idx:03d}: "
        #     f"mean_overlap={mean_overlap_view:.4f}, "
        #     f"mask_coverage={mask_coverage_val:.4f}, "
        #     f"valid_gaussians={valid.sum().item()}")

    # ===== Compute final average per-Gaussian overlap =====
    overlap_ratio = overlap_sum / (view_count + 1e-6)
    overlap_ratio[torch.isnan(overlap_ratio)] = 0.0

    avg_mask_ratio = float(np.mean(mask_coverage_all)) if len(mask_coverage_all) > 0 else 0.5

    print(f"[MaskOverlap@{iter}] mean={overlap_ratio.mean():.4f}, "
        f"std={overlap_ratio.std():.4f}, avg_mask_ratio={avg_mask_ratio:.4f}")

    return overlap_ratio, avg_mask_ratio, overlap_sum, view_count, view_ratios




# ==================================================
# View Consistency Filtering (Gaussian Mask Overlap)
# ==================================================
@torch.no_grad()
def gaussian_view_consistency(scene, gaussians, mask_dir, mask_disabled=False, mask_invert=False, threshold=None, save_dir=None, debug_views=None, views=None):
    """
    Identify low-outlier (inconsistent) views based on Gaussian–mask overlap and hit ratio.
    Automatically filters low-hit views, saves only those visualizations, and prints lowest 10 hit ratios.
    Returned indices refer to `views` (defaults to scene.getTrainCameras()).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scene.view_consistency import gaussian_mask_overlap, _find_mask_path, _load_binary_mask

    LOW_HIT_THRESHOLD = threshold
    views = scene.getTrainCameras() if views is None else views
    print(f"[GaussianViewConsistency] Checking {len(views)} training views...")

    # === Step 1. Compute global overlap stats ===
    overlap_ratio, avg_mask_ratio, overlap_sum, view_count, view_ratios = gaussian_mask_overlap(
        xyz=gaussians.get_xyz,
        scene=scene,
        mask_dir=mask_dir,
        mask_disabled=mask_disabled,
        mask_invert=mask_invert,
        iter=0,
        views=views,
    )
    mean_overlaps = np.array(view_ratios, dtype=np.float32)

    if save_dir is None:
        save_dir = os.path.join(scene.model_path, "debug")
    os.makedirs(save_dir, exist_ok=True)

    # === Step 2. Select views to visualize ===
    if debug_views is None:
        debug_views = range(len(views))
    xyz = gaussians.get_xyz.detach().to(views[0].world_view_transform.device)

    bad_indices, hit_ratios = [], []

    print("[Debug] Visualizing projection alignment (only low-hit views will be saved)...")
    for idx in debug_views:
        if idx >= len(views):
            continue

        cam = views[idx]
        mask_path = _find_mask_path(mask_dir, cam.image_name)
        if not mask_path or not os.path.exists(mask_path):
            print(f"[WARN] View {idx:03d}: mask not found → {cam.image_name}")
            continue

        H, W = cam.image_height, cam.image_width
        mask = _load_binary_mask(mask_path, H, W, invert=mask_invert)
        if mask is None:
            print(f"[WARN] View {idx:03d}: failed to load mask → {cam.image_name}")
            continue
        mask = mask.cpu().numpy()
        h_mask, w_mask = mask.shape[:2]

        # === Project 3D Gaussians to 2D ===
        uv = cam.project_to_screen(xyz)
        u = uv[:, 0].detach().cpu().numpy()
        v = uv[:, 1].detach().cpu().numpy()

        scale_x, scale_y = w_mask / float(W), h_mask / float(H)
        u = np.round(u * scale_x).astype(np.int32)
        v = np.round(v * scale_y).astype(np.int32)
        v = (h_mask - 1) - v  # flip y-axis (image coordinates)

        valid = (u >= 0) & (u < w_mask) & (v >= 0) & (v < h_mask)
        if valid.sum() == 0:
            continue

        u_valid, v_valid = u[valid], v[valid]
        mask_vals = mask[v_valid, u_valid].astype(np.float32)
        hit_ratio = float(np.mean(mask_vals > 0.5))
        hit_ratios.append((idx, cam.image_name, hit_ratio))

        # === Only visualize low-hit views ===
        if hit_ratio < LOW_HIT_THRESHOLD:
            bad_indices.append(idx)
        #     print(f"[LowHit] View {idx:03d} ({cam.image_name}) → hit_ratio={hit_ratio:.3f} < {LOW_HIT_THRESHOLD}")
        #     fig, ax = plt.subplots(figsize=(6, 5))
        #     ax.imshow(mask, cmap='gray')
        #     ax.scatter(u_valid, v_valid, s=0.5, c='r', alpha=0.3)
        #     ax.set_xlim([0, w_mask])
        #     ax.set_ylim([h_mask, 0])
        #     title = f"{cam.image_name} | hit_ratio={hit_ratio:.3f}"
        #     ax.set_title(title, fontsize=9)
        #     plt.tight_layout()
        #     lowhit_path = os.path.join(save_dir, f"proj_debug_{idx:03d}_LOWHIT.png")
        #     plt.savefig(lowhit_path, dpi=150)
        #     plt.close(fig)
        #     print(f"  [Saved] {lowhit_path}")

    # === Step 3. Summary: top-10 lowest hit ratios ===
    hit_ratios_sorted = sorted(hit_ratios, key=lambda x: x[2])  # sort by hit_ratio
    print("\n[Summary] 🔻 10 lowest hit_ratio views:")
    for rank, (idx, name, hr) in enumerate(hit_ratios_sorted[:10]):
        mark = "⚠️" if hr < LOW_HIT_THRESHOLD else ""
        print(f"  {rank+1:02d}. View {idx:03d} | {name:<25} | hit_ratio={hr:.3f} {mark}")

    # === Step 4. Histogram ===
    # low_hit_vals = [hr for (_, _, hr) in hit_ratios if hr < LOW_HIT_THRESHOLD]
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.hist([hr for (_, _, hr) in hit_ratios], bins=30, color='lightgray', edgecolor='k', alpha=0.6, label="All views")
    # if low_hit_vals:
    #     ax.hist(low_hit_vals, bins=30, color='red', alpha=0.6, label=f"Low hit_ratio (<{LOW_HIT_THRESHOLD})")
    # ax.set_xlabel("Hit Ratio per View")
    # ax.set_ylabel("View Count")
    # ax.set_title("Hit Ratio Distribution (low-hit views in red)")
    # ax.legend()
    # plt.tight_layout()
    # hist_path = os.path.join(save_dir, "view_lowhit_distribution.png")
    # plt.savefig(hist_path, dpi=150)
    # plt.close(fig)
    # print(f"\n[Saved] Low-hit histogram → {hist_path}")
    # print(f"[Done] {len(bad_indices)} low-hit views removed.\n")

    return sorted(set(bad_indices))






#=================================
# Consistency
# =================================

def compute_view_jaccard(scene, gaussians, pipeline, background, threshold=0.2):
    views = scene.getTrainCameras()
    n = len(views)
    visible_sets = []

    for v in views:
        out = render(v, gaussians, pipeline, background)
        vis_mask = out["visibility_filter"] > 0
        visible_ids = torch.nonzero(vis_mask, as_tuple=False).squeeze(-1).cpu().numpy().ravel().tolist()
        visible_sets.append(set(visible_ids))

    jaccard_means = []
    for i in range(n):
        sims = []
        for j in range(n):
            if i == j:
                continue
            inter = len(visible_sets[i] & visible_sets[j])
            union = len(visible_sets[i] | visible_sets[j]) + 1e-6
            sims.append(inter / union)
        mean_sim = sum(sims) / len(sims)
        jaccard_means.append(mean_sim)

    bad_indices = [i for i, score in enumerate(jaccard_means) if score < threshold]
    print(f"[JaccardFilter] {len(bad_indices)}/{n} views flagged (avg sim < {threshold})")

    for i, score in enumerate(jaccard_means):
        if i in bad_indices:
            print(f"* View {i:03d}: {score:.3f} (removed)")
    return bad_indices


def _visible_bool_from_render(render_pkg, num_points, device):
    radii = render_pkg.get("radii", None)
    if radii is not None and radii.numel() == num_points:
        return radii > 0

    visibility = render_pkg["visibility_filter"]
    visible = torch.zeros(num_points, dtype=torch.bool, device=device)
    if visibility.dtype == torch.bool and visibility.numel() == num_points:
        return visibility.to(device)
    ids = visibility.reshape(-1).long().to(device)
    ids = ids[(ids >= 0) & (ids < num_points)]
    visible[ids] = True
    return visible


@torch.no_grad()
def _object_visibility_weights(view, gaussians, pipeline, background, mask_dir=None, mask_disabled=False, mask_invert=False):
    render_pkg = render(view, gaussians, pipeline, background)
    xyz = gaussians.get_xyz.detach()
    device = xyz.device
    weights = _visible_bool_from_render(render_pkg, xyz.shape[0], device).float()

    if mask_dir is None or mask_disabled:
        return weights

    mask = _load_binary_mask_from_dir(
        mask_dir,
        view.image_name,
        int(view.image_height),
        int(view.image_width),
        invert=mask_invert,
        device=device,
    )
    if mask is None:
        return weights

    uv = view.project_to_screen(xyz)
    u = uv[:, 0].long()
    v = uv[:, 1].long()
    H, W = int(view.image_height), int(view.image_width)
    v = (H - 1) - v
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    mask_vals = torch.zeros(xyz.shape[0], dtype=torch.float32, device=device)
    mask_vals[valid] = mask[v[valid], u[valid]].float()
    opacity = gaussians.get_opacity.detach().squeeze(-1)
    return weights * mask_vals * opacity


def _adaptive_low_score_threshold(scores, max_low_fraction=0.25):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size < 3:
        return None, "too_few_views"

    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size < 3:
        return None, "invalid_scores"

    score_min = float(finite_scores.min())
    score_max = float(finite_scores.max())
    if score_max <= score_min:
        return None, "flat_scores"

    sorted_scores = np.sort(finite_scores)
    gaps = sorted_scores[1:] - sorted_scores[:-1]
    split_idx = int(np.argmax(gaps))
    threshold = float((sorted_scores[split_idx] + sorted_scores[split_idx + 1]) * 0.5)

    low_count = split_idx + 1
    high_count = sorted_scores.size - low_count
    if low_count <= 0 or high_count <= 0:
        return None, "empty_cluster"
    low_fraction = low_count / float(sorted_scores.size)
    if low_fraction > max_low_fraction:
        return None, f"low_cluster_not_outlier(low={low_count}, total={sorted_scores.size})"

    total_var = float(np.var(finite_scores)) + 1e-12
    low_var = float(np.var(sorted_scores[:low_count])) if low_count > 1 else 0.0
    high_var = float(np.var(sorted_scores[low_count:])) if high_count > 1 else 0.0
    within_var = (low_count * low_var + high_count * high_var) / float(sorted_scores.size)
    separation = 1.0 - (within_var / total_var)

    if separation <= 0.5:
        return None, f"unimodal(separation={separation:.3f})"

    return threshold, f"largest_gap(separation={separation:.3f}, low={low_count}, high={high_count})"


def _pose_neighbor_indices(views, sample_k):
    centers = []
    for view in views:
        center = getattr(view, "camera_center", None)
        if center is None:
            return None
        centers.append(center.detach().float().cpu())

    centers = torch.stack(centers, dim=0)
    n = centers.shape[0]
    if n <= 1:
        return [[] for _ in range(n)]

    dist = torch.cdist(centers, centers)
    dist.fill_diagonal_(float("inf"))
    neighbor_k = min(sample_k, n - 1)
    return torch.topk(dist, k=neighbor_k, largest=False).indices.tolist()


@torch.no_grad()
def _save_projection_debug(
    scene,
    views,
    scores,
    gaussians,
    mask_dir,
    mask_disabled=False,
    mask_invert=False,
    iteration=0,
    max_views=6,
    max_points=20000,
):
    if mask_dir is None or mask_disabled or max_views <= 0:
        return

    debug_dir = os.path.join(scene.model_path, "debug", f"projection_iter{int(iteration):06d}")
    os.makedirs(debug_dir, exist_ok=True)

    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(scores)
    half = max(1, max_views // 2)
    selected = []
    for idx in list(order[:half]) + list(order[-half:]):
        if int(idx) not in selected:
            selected.append(int(idx))
        if len(selected) >= max_views:
            break

    xyz = gaussians.get_xyz.detach()
    device = xyz.device
    for idx in selected:
        view = views[idx]
        H, W = int(view.image_height), int(view.image_width)
        mask = _load_binary_mask_from_dir(
            mask_dir,
            view.image_name,
            H,
            W,
            invert=mask_invert,
            device=device,
        )
        if mask is None:
            continue

        uv = view.project_to_screen(xyz)
        u = uv[:, 0].long()
        v = uv[:, 1].long()
        v = (H - 1) - v
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        draw_mask = valid

        draw_ids = torch.nonzero(draw_mask, as_tuple=False).squeeze(-1)
        if draw_ids.numel() > max_points:
            perm = torch.randperm(draw_ids.numel(), device=device)[:max_points]
            draw_ids = draw_ids[perm]

        u_draw = u[draw_ids]
        v_draw = v[draw_ids]
        inside = mask[v_draw, u_draw] > 0.5

        mask_np = mask.detach().cpu().numpy()
        u_np = u_draw.detach().cpu().numpy()
        v_np = v_draw.detach().cpu().numpy()
        inside_np = inside.detach().cpu().numpy()

        plt.figure(figsize=(10, 7))
        plt.imshow(mask_np, cmap="gray", origin="upper")
        plt.scatter(u_np[~inside_np], v_np[~inside_np], s=2, c="red", alpha=0.55, label="projected outside mask")
        plt.scatter(u_np[inside_np], v_np[inside_np], s=2, c="lime", alpha=0.55, label="projected inside mask")
        plt.xlim([0, W])
        plt.ylim([H, 0])
        plt.title(
            f"{view.image_name} | score={scores[idx]:.4f} | "
            f"inside={inside_np.sum()}/{inside_np.size} | HxW={H}x{W}"
        )
        plt.legend(loc="upper right", markerscale=4)
        plt.tight_layout()
        safe_name = os.path.splitext(os.path.basename(view.image_name))[0]
        save_path = os.path.join(debug_dir, f"view_{idx:03d}_{safe_name}_score_{scores[idx]:.4f}.png")
        plt.savefig(save_path, dpi=180)
        plt.close()
        print(f"[ProjectionDebug] saved {save_path}")


def compute_view_jaccard_fast(
    scene,
    gaussians,
    pipeline,
    background,
    threshold=None,
    sample_k=20,
    mask_dir=None,
    mask_disabled=False,
    mask_invert=False,
    views=None,
    debug_projection=False,
    debug_iter=0,
    debug_count=6,
):
    views = scene.getTrainCameras() if views is None else views
    n = len(views)
    if n <= 1:
        return []
    view_weights = []
    pose_neighbors = _pose_neighbor_indices(views, sample_k)

    for v in views:
        weights = _object_visibility_weights(
            v,
            gaussians,
            pipeline,
            background,
            mask_dir=mask_dir,
            mask_disabled=mask_disabled,
            mask_invert=mask_invert,
        )
        view_weights.append(weights)

    jaccard_means = []
    for i in range(n):
        sims = []
        if pose_neighbors is not None:
            sample_idx = pose_neighbors[i]
        else:
            sample_idx = random.sample([j for j in range(n) if j != i], min(sample_k, n - 1))
        for j in sample_idx:
            inter = torch.minimum(view_weights[i], view_weights[j]).sum()
            union = torch.maximum(view_weights[i], view_weights[j]).sum()
            sims.append((inter / (union + 1e-6)).item() if union.item() > 0 else 0.0)
        top_k = min(3, len(sims))
        mean_sim = sum(sorted(sims, reverse=True)[:top_k]) / max(top_k, 1)
        jaccard_means.append(mean_sim)

    score_arr = np.array(jaccard_means, dtype=np.float32)
    if debug_projection:
        _save_projection_debug(
            scene,
            views,
            score_arr,
            gaussians,
            mask_dir,
            mask_disabled=mask_disabled,
            mask_invert=mask_invert,
            iteration=debug_iter,
            max_views=debug_count,
        )

    effective_threshold, threshold_reason = _adaptive_low_score_threshold(score_arr)
    bad_indices = [] if effective_threshold is None else [
        i for i, score in enumerate(jaccard_means) if score < effective_threshold
    ]
    mode = "object-weighted" if mask_dir is not None and not mask_disabled else "visibility"
    if effective_threshold is None:
        print(
            f"[FastJaccard:{mode}] 0/{n} views flagged "
            f"(adaptive=no_split, reason={threshold_reason}, mean={score_arr.mean():.4f}) "
            f"[pose_neighbors={pose_neighbors is not None}, sample_k={sample_k}]"
        )
    else:
        print(
            f"[FastJaccard:{mode}] {len(bad_indices)}/{n} views flagged "
            f"(adaptive_thresh={effective_threshold:.4f}, {threshold_reason}, "
            f"mean={score_arr.mean():.4f}) [pose_neighbors={pose_neighbors is not None}, sample_k={sample_k}]"
        )

    return bad_indices
