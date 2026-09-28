#!/usr/bin/env python3
"""
No-Reference Image Quality Assessment Script
─────────────────────────────────────────────────────
Metrics:
  (No-Reference)
  ① CLIP-IQA+  - pyiqa 'clipiqa+'   (0~1,   ↑ higher is better)
  ② MUSIQ      - pyiqa 'musiq'      (0~100, ↑ higher is better)

  (Multi-view consistency)
  ③ MEt3R      - met3r (pair-wise, ↓ lower means better consistency)
     - Computes scores for (i, i+K) pairs at stride K
     - Multiple strides (default 1,2,4) can be computed at once and saved to JSON

Dependencies:
  pip install pyiqa
  pip install git+https://github.com/openai/CLIP.git  # CLIP-IQA+ backend
  pip install git+https://github.com/mohammadasim98/met3r

Notes:
  - MEt3R is most meaningful for aligned image sequences (e.g., frame_000.png, frame_001.png, ...).
  - If OOM occurs, try --met3r_batch 1, --met3r_img_size 256 (or lower).
─────────────────────────────────────────────────────
"""

import os
import csv
import glob
import json
import argparse
import numpy as np
from PIL import Image
import traceback

import torch
from torchvision import transforms


# ═══════════════════════════════════════════════════
# Common Utilities
# ═══════════════════════════════════════════════════

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

def collect_images(directory: str):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(paths)

TO_TENSOR = transforms.ToTensor()

def load_tensor(path: str, device: str):
    """PIL → float32 (1,3,H,W) [0,1]"""
    img = Image.open(path).convert("RGB")
    return TO_TENSOR(img).unsqueeze(0).to(device)


# ═══════════════════════════════════════════════════
# pyiqa Common Computation Function
# ═══════════════════════════════════════════════════

@torch.no_grad()
def compute_pyiqa(metric_name: str, device: str, image_paths):
    """
    Creates a metric using pyiqa.create_metric and returns per-image scores.
    Images that raise an error are skipped.
    """
    try:
        import pyiqa
    except ImportError as e:
        raise ImportError(f"Failed to import pyiqa ({e}). See the README for the install command.") from e

    metric = pyiqa.create_metric(metric_name, device=device)
    metric.eval()

    scores = []
    skipped = 0
    for p in image_paths:
        t = load_tensor(p, device)

        try:
            out = metric(t)
            scores.append(round(float(out.mean().item()), 6))
        except Exception as e:
            skipped += 1
            print(f"    [skip] {os.path.basename(p)}: {e}")

    if skipped:
        print(f"    [info] {metric_name}: {skipped} image(s) skipped")

    del metric
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


# ═══════════════════════════════════════════════════
# MEt3R Computation
# ═══════════════════════════════════════════════════

def _parse_stride_list(s: str):
    """
    "1,2,4" -> [1,2,4]
    """
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k = int(tok)
        if k <= 0:
            raise ValueError("Stride must be positive.")
        out.append(k)
    return sorted(list(dict.fromkeys(out)))  # unique + keep order-ish

@torch.no_grad()
def compute_met3r_for_stride(
    device: str,
    image_paths,
    stride: int,
    batch_size: int = 1,
    img_size: int = 256,
    backbone: str = "mast3r",
    feature_backbone: str = "dino16",
    upsampler: str = "featup",
    distance: str = "cosine",
    freeze: bool = True,
):
    """
    Returns a list of pair-wise MEt3R scores for (i, i+K) pairs at stride K.
    - Input RGB is converted to [-1,1] range
    """
    try:
        from met3r import MEt3R
    except ImportError as e:
        raise ImportError(f"Failed to import met3r ({e}). See the README for the install command.") from e

    n = len(image_paths)
    if n < stride + 1:
        return []

    pairs = [(image_paths[i], image_paths[i + stride]) for i in range(0, n - stride)]

    metric = MEt3R(
        img_size=img_size,                 # None is possible but increases OOM risk
        use_norm=True,
        backbone=backbone,                 # ["mast3r","dust3r","raft"]
        feature_backbone=feature_backbone, # ["dino16","dinov2","maskclip","vit","clip","resnet50"]
        feature_backbone_weights="mhamilton723/FeatUp",
        upsampler=upsampler,               # ["featup","nearest","bilinear","bicubic"]
        distance=distance,                 # ["cosine","lpips","rmse","psnr","mse","ssim"]
        freeze=freeze,
    ).to(device)
    metric.eval()

    from torchvision.transforms import functional as TF

    def _round_down_to_multiple(x, m):
        return (x // m) * m

    def load_tensor_met3r(path: str, patch_size: int = 16, img_size: int = None):
        img = Image.open(path).convert("RGB")

        # 1) If img_size is specified: resize to square (ensure multiple of 16)
        if img_size is not None:
            # Round down if img_size is not a multiple of 16
            size = _round_down_to_multiple(int(img_size), patch_size)
            size = max(size, patch_size)
            img = img.resize((size, size), resample=Image.BILINEAR)

        # 2) If img_size is None: center-crop from original to nearest multiple of 16
        else:
            w, h = img.size
            new_w = _round_down_to_multiple(w, patch_size)
            new_h = _round_down_to_multiple(h, patch_size)
            if new_w < patch_size or new_h < patch_size:
                raise ValueError(f"Image too small after rounding: {(w,h)} -> {(new_w,new_h)}")
            left = (w - new_w) // 2
            top  = (h - new_h) // 2
            img = img.crop((left, top, left + new_w, top + new_h))

        t = TO_TENSOR(img)     # [0,1]
        t = t * 2.0 - 1.0      # [-1,1]
        return t

    scores = []
    for s in range(0, len(pairs), batch_size):
        batch_pairs = pairs[s:s + batch_size]

        imgs = []
        for p1, p2 in batch_pairs:
            t1 = load_tensor_met3r(p1, patch_size=16, img_size=img_size)
            t2 = load_tensor_met3r(p2, patch_size=16, img_size=img_size)
            imgs.append(torch.stack([t1, t2], dim=0))  # (2,3,H,W)

        inputs = torch.stack(imgs, dim=0).to(device)   # (B,2,3,H,W)

        score, *_ = metric(
            images=inputs,
            return_overlap_mask=False,
            return_score_map=False,
            return_projections=False
        )
        score = score.reshape(score.shape[0], -1).mean(dim=1)  # (B,)

        for v in score.detach().cpu().tolist():
            scores.append(round(float(v), 6))

    del metric
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


# ═══════════════════════════════════════════════════
# CSV Helper
# ═══════════════════════════════════════════════════

CSV_FIELDS = [
    "scene", "num_rendered",
    "clip_iqa_mean",   # 0~1,   ↑
    "musiq_mean",      # 0~100, ↑
    "met3r_mean",      # ↓ (average of stride-wise means; see definition below)
]

def append_csv_row(csv_path, row):
    need_header = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if need_header:
            w.writeheader()
        w.writerow({k: ("" if row.get(k) is None else row.get(k, ""))
                    for k in CSV_FIELDS})


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="No-Reference Image Quality Assessment (CLIP-IQA+ / MUSIQ) + MEt3R"
    )
    parser.add_argument("--render_dir",    required=True)
    parser.add_argument("--out_json",      required=True)
    parser.add_argument("--out_csv",       required=True)
    parser.add_argument("--scene_name",    default="")

    parser.add_argument("--skip_clip_iqa", action="store_true")
    parser.add_argument("--skip_musiq",    action="store_true")
    parser.add_argument("--skip_met3r",    action="store_true")

    # MEt3R options
    parser.add_argument("--met3r_strides", type=str, default="1,2,4",
                        help="comma-separated strides, e.g., '1,2,4' (pair: i,i+K)")
    parser.add_argument("--met3r_batch",  type=int, default=1, help="MEt3R batch size (use 1 if OOM)")
    parser.add_argument("--met3r_img_size", type=int, default=256, help="MEt3R img_size (integer recommended)")
    parser.add_argument("--met3r_backbone", type=str, default="mast3r",
                        choices=["mast3r", "dust3r", "raft"])
    parser.add_argument("--met3r_feat", type=str, default="dino16",
                        choices=["dino16","dinov2","maskclip","vit","clip","resnet50"])
    parser.add_argument("--met3r_upsampler", type=str, default="featup",
                        choices=["featup","nearest","bilinear","bicubic"])
    parser.add_argument("--met3r_distance", type=str, default="cosine",
                        choices=["cosine","lpips","rmse","psnr","mse","ssim"])

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device     : {device}")
    print(f"[eval] scene      : {args.scene_name}")
    print(f"[eval] render_dir : {args.render_dir}")

    rendered = collect_images(args.render_dir)
    if not rendered:
        print(f"[eval] ERROR: No rendered images found → {args.render_dir}")
        return
    print(f"[eval] images     : {len(rendered)} images")

    result = {
        "scene"       : args.scene_name,
        "render_dir"  : args.render_dir,
        "num_rendered": len(rendered),
    }

    # ── ① CLIP-IQA+ ──────────────────────────────────
    if not args.skip_clip_iqa:
        print("[eval] ① Computing CLIP-IQA+...")
        try:
            s = compute_pyiqa("clipiqa+", device, rendered)
            result["clip_iqa_mean"]      = round(float(np.mean(s)), 6)
            result["clip_iqa_per_image"] = s
            print(f"[eval]   CLIP-IQA+  = {result['clip_iqa_mean']:.4f}  (↑ 0~1)")
        except Exception as e:
            print(f"[eval]   CLIP-IQA+ failed: {e}")

    # ── ② MUSIQ ──────────────────────────────────────
    if not args.skip_musiq:
        print("[eval] ② Computing MUSIQ...")
        try:
            s = compute_pyiqa("musiq", device, rendered)
            result["musiq_mean"]      = round(float(np.mean(s)), 6)
            result["musiq_per_image"] = s
            print(f"[eval]   MUSIQ      = {result['musiq_mean']:.4f}  (↑ 0~100)")
        except Exception as e:
            print(f"[eval]   MUSIQ failed: {e}")

    # ── ③ MEt3R ───────────────────────────────────────
    if not args.skip_met3r:
        print("[eval] ③ Computing MEt3R... (pair-wise consistency)")
        try:
            strides = _parse_stride_list(args.met3r_strides)
            print(f"[eval]   strides    : {strides} (pair: i, i+K)")

            met3r_all = {}
            met3r_means = {}

            for k in strides:
                print(f"[eval]   - Computing stride={k}...")
                s = compute_met3r_for_stride(
                    device=device,
                    image_paths=rendered,
                    stride=k,
                    batch_size=args.met3r_batch,
                    img_size=args.met3r_img_size,
                    backbone=args.met3r_backbone,
                    feature_backbone=args.met3r_feat,
                    upsampler=args.met3r_upsampler,
                    distance=args.met3r_distance,
                    freeze=True,
                )
                met3r_all[str(k)] = s
                if len(s) > 0:
                    met3r_means[str(k)] = round(float(np.mean(s)), 6)
                    print(f"[eval]     mean={met3r_means[str(k)]:.4f} (n_pairs={len(s)})")
                else:
                    met3r_means[str(k)] = None
                    print(f"[eval]     (skip) Not enough pairs (n_images={len(rendered)})")

            # Store all stride results in JSON
            result["met3r"] = {
                "strides": strides,
                "per_stride_mean": met3r_means,
                "per_stride_scores": met3r_all,
                "settings": {
                    "batch_size": args.met3r_batch,
                    "img_size": args.met3r_img_size,
                    "backbone": args.met3r_backbone,
                    "feature_backbone": args.met3r_feat,
                    "upsampler": args.met3r_upsampler,
                    "distance": args.met3r_distance,
                }
            }

            # For CSV, store a single value:
            # average of per-stride means (valid strides only) as met3r_mean
            valid_means = [v for v in met3r_means.values() if v is not None]
            if valid_means:
                result["met3r_mean"] = round(float(np.mean(valid_means)), 6)
                print(f"[eval]   MEt3R(over strides) = {result['met3r_mean']:.4f}  (↓)")
            else:
                result["met3r_mean"] = None
                print("[eval]   MEt3R: No valid strides")

        except Exception as e:
            print(f"[eval]   MEt3R failed: {e}")
            traceback.print_exc()

    # ── Save JSON ─────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[eval] JSON saved : {args.out_json}")

    # ── Append CSV row ──────────────────────────────────
    append_csv_row(args.out_csv, {
        "scene"        : result["scene"],
        "num_rendered" : result["num_rendered"],
        "clip_iqa_mean": result.get("clip_iqa_mean"),
        "musiq_mean"   : result.get("musiq_mean"),
        "met3r_mean"   : result.get("met3r_mean"),
    })
    print(f"[eval] CSV appended: {args.out_csv}")


if __name__ == "__main__":
    main()