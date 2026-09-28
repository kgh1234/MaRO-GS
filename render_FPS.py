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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import time
import csv

def measure_fps(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, warmup_iters):
    output_dir = os.path.join(model_path, name, f"ours_{iteration}")
    makedirs(output_dir, exist_ok=True)

    render_times = []

    print(f"\n[ Measuring FPS on {name} ]")
    pbar = tqdm(views)

    # Warm-up repeatedly so a short benchmark is not run at the GPU's idle
    # clock.  This remains outside the timed section.
    for i in range(warmup_iters):
        v = views[i % len(views)]
        _ = render(v, gaussians, pipeline, background,
                   use_trained_exp=train_test_exp, separate_sh=separate_sh)
    torch.cuda.synchronize()

    for view in pbar:
        torch.cuda.synchronize()
        start = time.time()

        _ = render(view, gaussians, pipeline, background,
                   use_trained_exp=train_test_exp, separate_sh=separate_sh)

        torch.cuda.synchronize()
        end = time.time()

        render_times.append((end - start) * 1000.0)

    avg_time_ms = sum(render_times) / len(render_times)
    fps = 1000.0 / avg_time_ms

    print(f" → FPS: {fps:.3f}\n")

    with open(os.path.join(output_dir, "fps.txt"), "w") as f:
        f.write(f"{fps}\n")

    return fps



def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_test : bool, separate_sh: bool, warmup_iters: int):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_test:
            views = scene.getTestCameras()
            measure_fps(dataset.model_path, "test", scene.loaded_iter, views,
                            gaussians, pipeline, background, dataset.train_test_exp, separate_sh, warmup_iters)



if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--warmup_iters", default=5, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_test, False, args.warmup_iters)
