#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import cv2
import numpy as np
import re


def natural_key(s):
    s = str(s)
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []

    render_files = sorted(os.listdir(renders_dir), key=natural_key)
    gt_files = sorted(os.listdir(gt_dir), key=natural_key)

    print(f"[DEBUG] first renders: {render_files[:3]}")
    print(f"[DEBUG] first gts    : {gt_files[:3]}")

    for fname_r, fname_g in zip(render_files, gt_files):
        render = Image.open(Path(renders_dir) / fname_r)
        gt = Image.open(Path(gt_dir) / fname_g)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname_r)
    return renders, gts, image_names


def compute_iou(mask_pred, mask_gt):
    mask_pred = (mask_pred > 127).astype(np.uint8)
    mask_gt = (mask_gt > 127).astype(np.uint8)
    inter = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def evaluate(model_paths, mask_dir):
    """
    model_paths: list of 3DGS scene dirs
    mask_dir: GT mask directory
    """
    full_dict = {}
    per_view_dict = {}
    print("")

    mask_files = sorted(os.listdir(mask_dir), key=natural_key)
    selected_masks = [mask_files[i] for i in range(0, len(mask_files), 8)]
    print(f"Selected {len(selected_masks)} masks (every 8th frame)")
    print(f"[DEBUG] first masks: {mask_files[:3]}")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"
            method = sorted(os.listdir(test_dir))[-1]
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            pred_mask_dir = method_dir / "masks"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims, psnrs, lpipss, mious = [], [], [], []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):

                if idx < len(selected_masks):
                    gt_mask_path = os.path.join(mask_dir, selected_masks[idx])
                    if not os.path.exists(gt_mask_path):
                        print(f"GT mask not found: {gt_mask_path}")
                        continue

                    mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        print(f"Cannot read GT mask {gt_mask_path}")
                        continue

                    h, w = renders[idx].shape[-2], renders[idx].shape[-1]
                    mask = cv2.resize(mask, (w, h)).astype(np.float32) / 255.0
                    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).cuda()
                    mask_t = mask_t.expand_as(renders[idx])
                    render_masked = renders[idx] * mask_t
                    gt_masked = gts[idx] * mask_t
                else:
                    render_masked = renders[idx]
                    gt_masked = gts[idx]

                # === PSNR / SSIM / LPIPS ===
                PSNR = psnr(render_masked, gt_masked)
                if PSNR != float('inf'):
                    ssims.append(ssim(render_masked, gt_masked))
                    psnrs.append(PSNR)
                    lpipss.append(lpips(render_masked, gt_masked, net_type='vgg'))

                base_name = os.path.splitext(image_names[idx])[0]
                pred_mask_path = os.path.join(pred_mask_dir, f"{base_name}.png")
                gt_mask_path = os.path.join(mask_dir, selected_masks[idx] if idx < len(selected_masks) else "")
                if os.path.exists(pred_mask_path) and os.path.exists(gt_mask_path):
                    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
                    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                    if pred_mask is not None and gt_mask is not None:
                        if pred_mask.shape != gt_mask.shape:
                            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                                   interpolation=cv2.INTER_NEAREST)
                        mious.append(compute_iou(pred_mask, gt_mask))
                else:
                    print(f"[WARN] Missing mask pair for {base_name}")

            ssim_mean = torch.tensor(ssims).mean().item() if ssims else 0
            psnr_mean = torch.tensor(psnrs).mean().item() if psnrs else 0
            lpips_mean = torch.tensor(lpipss).mean().item() if lpipss else 0
            miou_mean = np.mean(mious) if mious else 0

            print("SSIM : {:>12.7f}".format(ssim_mean))
            print("PSNR : {:>12.7f}".format(psnr_mean))
            print("LPIPS: {:>12.7f}".format(lpips_mean))
            print("mIoU : {:>12.7f}".format(miou_mean))
            print("")

            full_dict[scene_dir][method].update({
                "SSIM": ssim_mean,
                "PSNR": psnr_mean,
                "LPIPS": lpips_mean,
                "mIoU": float(miou_mean)
            })

            with open(scene_dir + "/results_masked.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)

        except Exception as e:
            print("Unable to compute metrics for model", scene_dir, ":", e)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Metric evaluation (masked version + mIoU)")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[],
                        help="Path(s) to 3DGS model directories")
    parser.add_argument('--mask_dir', '-mask', required=True, type=str, default="",
                        help="Path to GT mask directory")
    args = parser.parse_args()
    evaluate(args.model_paths, args.mask_dir)
