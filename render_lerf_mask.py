# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.

import os
import time
from argparse import ArgumentParser
from os import makedirs

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def _object_name_from_model_path(model_path):
    name = os.path.basename(os.path.normpath(model_path))
    if name.startswith("object_"):
        name = name[len("object_"):]
    return name.replace("_", " ")


def _shared_lerf_output_root(model_path, split_name, tag):
    model_path = os.path.normpath(model_path)
    object_dir = os.path.basename(model_path)
    if object_dir.startswith("object_"):
        return os.path.join(os.path.dirname(model_path), split_name, f"{tag}_text", "test_mask")
    return os.path.join(model_path, split_name, f"{tag}_text", "test_mask")


def _configure_lerf_test_split(dataset):
    has_lerf_test = (
        os.path.isdir(os.path.join(dataset.source_path, "images_train"))
        and os.path.isdir(os.path.join(dataset.source_path, "test_mask"))
    )
    if has_lerf_test:
        dataset.train_split = True
        dataset.eval = True
        print("[LERF] Using gaussian-grouping style test split from images - images_train")


def _filter_lerf_test_views(source_path, views):
    train_dir = os.path.join(source_path, "images_train")
    test_mask_dir = os.path.join(source_path, "test_mask")
    if not (os.path.isdir(train_dir) and os.path.isdir(test_mask_dir)):
        return views

    train_stems = {
        os.path.splitext(name)[0]
        for name in os.listdir(train_dir)
        if not name.startswith(".")
    }
    filtered = [
        view for view in views
        if os.path.splitext(os.path.basename(view.image_name))[0] not in train_stems
    ]
    print(f"[LERF] Filtered test views: {len(filtered)}/{len(views)}")
    return filtered


def _lerf_test_view_id(view, fallback_idx):
    stem = os.path.splitext(os.path.basename(view.image_name))[0]
    if stem.startswith("test_"):
        return stem[len("test_"):]
    return str(fallback_idx)


def _render_object_silhouette(view, gaussians, pipeline, separate_sh):
    black = torch.zeros(3, dtype=torch.float32, device="cuda")
    white = torch.ones((gaussians.get_xyz.shape[0], 3), dtype=torch.float32, device="cuda")
    out = render(
        view,
        gaussians,
        pipeline,
        black,
        override_color=white,
        separate_sh=separate_sh,
    )
    return out["render"].amax(dim=0, keepdim=True).clamp(0, 1)


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    train_test_exp,
    separate_sh,
    save_lerf_test_mask=False,
    object_name=None,
):
    tag = f"ours_{iteration}"
    render_path = os.path.join(model_path, name, tag, "renders")
    gts_path = os.path.join(model_path, name, tag, "gt")
    mask_path = os.path.join(model_path, name, tag, "masks")
    lerf_mask_root = os.path.join(model_path, name, f"{tag}_text", "test_mask")
    shared_lerf_mask_root = _shared_lerf_output_root(model_path, name, tag)

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)
    if save_lerf_test_mask:
        makedirs(lerf_mask_root, exist_ok=True)
        makedirs(shared_lerf_mask_root, exist_ok=True)

    view_id_rows = []
    render_times_ms = []
    for view in views[:5]:
        _ = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}/{tag}")):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        out = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        render_times_ms.append((time.perf_counter() - start_time) * 1000.0)
        rendering = out["render"]
        gt = view.original_image[0:3, :, :]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        mask = _render_object_silhouette(view, gaussians, pipeline, separate_sh)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        render_times_ms.append((time.perf_counter() - start_time) * 1000.0)

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(mask.float(), os.path.join(mask_path, f"{idx:05d}.png"))
        if save_lerf_test_mask:
            view_id = _lerf_test_view_id(view, idx)
            view_id_rows.append((idx, view.image_name, view_id))
            view_mask_dir = os.path.join(lerf_mask_root, view_id)
            shared_view_mask_dir = os.path.join(shared_lerf_mask_root, view_id)
            makedirs(view_mask_dir, exist_ok=True)
            makedirs(shared_view_mask_dir, exist_ok=True)
            mask_filename = f"{object_name}.png"
            torchvision.utils.save_image(mask.float(), os.path.join(view_mask_dir, mask_filename))
            torchvision.utils.save_image(mask.float(), os.path.join(shared_view_mask_dir, mask_filename))

    avg_time_ms = sum(render_times_ms) / len(render_times_ms) if render_times_ms else 0.0
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0
    object_label = object_name if object_name is not None else _object_name_from_model_path(model_path)
    log_line = (
        f"[object render][FPS] split={name}, tag={tag}, object={object_label}, views={len(views)}, "
        f"render_calls={len(render_times_ms)}, avg_render_ms={avg_time_ms:.4f}, fps={fps:.4f}"
    )
    print(log_line)
    with open(os.path.join(model_path, name, tag, "fps.log"), "w", encoding="utf-8") as fp:
        fp.write(log_line + "\n")

    if save_lerf_test_mask:
        metadata_path = os.path.join(model_path, name, tag, "view_ids.csv")
        with open(metadata_path, "w") as fp:
            fp.write("idx,image_name,view_id\n")
            for idx, image_name, view_id in view_id_rows:
                fp.write(f"{idx:05d},{image_name},{view_id}\n")


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, separate_sh: bool):
    with torch.no_grad():
        _configure_lerf_test_split(dataset)
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        object_name = _object_name_from_model_path(dataset.model_path)
        print(f"[LERF] Object mask name: {object_name}.png")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                dataset.train_test_exp,
                separate_sh,
            )

        if not skip_test:
            test_views = _filter_lerf_test_views(dataset.source_path, scene.getTestCameras())
            if len(test_views) == 0:
                print("[WARN] No test cameras found. For LERF, check images_train/test_mask and train_split/eval.")
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                test_views,
                gaussians,
                pipeline,
                background,
                dataset.train_test_exp,
                separate_sh,
                save_lerf_test_mask=True,
                object_name=object_name,
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering script for LERF-style mask outputs")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, False)
