import os
import glob
import cv2
import torch
import numpy as np

# =========================
# Mask Utilities
# =========================
def _stem(path_or_name: str) -> str:
    return os.path.splitext(os.path.basename(path_or_name))[0]

def _find_mask_path(mask_dir: str, image_name_or_path: str):
    stem = _stem(image_name_or_path)
    for ext in ["png", "jpg", "jpeg", "bmp", "webp", "JPG"]:
        cand = os.path.join(mask_dir, f"{stem}.{ext}")
        if os.path.isfile(cand):
            return cand
    # fallback for “mask” token
    for ext in ["png", "jpg", "jpeg", "bmp", "webp", "JPG"]:
        cands = glob.glob(os.path.join(mask_dir, f"{stem}*mask*.{ext}"))
        if cands:
            return cands[0]
    return None

def _find_mask_paths(mask_dir: str, image_name_or_path: str):
    stem = _stem(image_name_or_path)
    candidates = []
    for ext in ["png", "jpg", "jpeg", "bmp", "webp", "JPG"]:
        candidates.extend(glob.glob(os.path.join(mask_dir, f"{stem}*.{ext}")))
    return sorted(set(candidates))

def _load_binary_mask(mask_path: str, H: int, W: int, binary_threshold=32, invert=False, device="cuda"):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #m = cv2.flip(m, 0)
    if m is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")
    
    if m.shape[0] == W and m.shape[1] == H:
        print(f"[MaskFix] Transposing mask (W,H)->(H,W): {mask_path}")
        m = m.T

    if (m.shape[0] != H) or (m.shape[1] != W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    if invert:
        m = 255 - m
    m = (m >= binary_threshold).astype(np.float32)
    return torch.from_numpy(m).to(device)  # (H, W)

def _load_binary_mask_from_dir(mask_dir: str, image_name_or_path: str, H: int, W: int, binary_threshold=32, invert=False, device="cuda"):
    mask_paths = _find_mask_paths(mask_dir, image_name_or_path)
    if not mask_paths:
        return None

    mask_paths = sorted(mask_paths)
    if len(mask_paths) == 2:
        try:
            m1 = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
            m2 = cv2.imread(mask_paths[1], cv2.IMREAD_GRAYSCALE)
        except Exception:
            m1 = m2 = None

        if m1 is not None and m2 is not None:
            if m1.shape[0] == W and m1.shape[1] == H:
                m1 = m1.T
            if m2.shape[0] == W and m2.shape[1] == H:
                m2 = m2.T

            if (m1.shape[1] < W) and (m2.shape[1] < W):
                half_w = W // 2
                left = cv2.resize(m1, (half_w, H), interpolation=cv2.INTER_NEAREST) if (m1.shape[0] != H or m1.shape[1] != half_w) else m1
                right = cv2.resize(m2, (W - half_w, H), interpolation=cv2.INTER_NEAREST) if (m2.shape[0] != H or m2.shape[1] != (W - half_w)) else m2
                combined = np.concatenate([left.astype(np.uint8), right.astype(np.uint8)], axis=1)
                if invert:
                    combined = 255 - combined
                combined = (combined >= binary_threshold).astype(np.float32)
                return torch.from_numpy(combined).to(device)

    combined = None
    for mask_path in mask_paths:
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue

        if m.shape[0] == W and m.shape[1] == H:
            m = m.T

        if (m.shape[0] != H) or (m.shape[1] != W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

        if combined is None:
            combined = m.astype(np.uint8)
        else:
            combined = np.maximum(combined, m.astype(np.uint8))

    if combined is None:
        return None

    if invert:
        combined = 255 - combined

    combined = (combined >= binary_threshold).astype(np.float32)
    return torch.from_numpy(combined).to(device)
