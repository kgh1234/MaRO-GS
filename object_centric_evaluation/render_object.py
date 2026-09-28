#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, shutil, tempfile
from os import makedirs
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torchvision

from utils.general_utils import safe_state
from gaussian_renderer import GaussianModel, render
from arguments import ModelParams, PipelineParams
from scene import Scene


def render_set_only_renders(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, out_name=None):
    tag = f"ours_{iteration}"
    render_root = os.path.join(model_path, name, tag, (out_name if out_name else "renders"))
    makedirs(render_root, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}/{tag}")):
        out = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=False)
        img = out["render"]
        if train_test_exp:
            img = img[..., img.shape[-1] // 2:]
        torchvision.utils.save_image(img, os.path.join(render_root, f"{idx:05d}.png"))


def main():
    # ---- CLI wrapper ----
    cli = ArgumentParser("Render from custom camera JSON (novel views only, no GT required)")
    cli.add_argument("--model_path", "-m", required=True, help="Trained model folder (3DGS -m)")
    cli.add_argument("--camera_json", required=True, help="Path to your camera.json (NeRF/3DGS transforms-like)")
    cli.add_argument("--iteration", type=int, default=-1, help="Model iteration to load (e.g., 30000)")
    cli.add_argument("--images_ext", default=".jpg", help="Assume this ext if file_path has no ext (e.g., .jpg/.png)")
    cli.add_argument("--white_background", action="store_true", help="Force white background")
    cli.add_argument("--quiet", action="store_true")
    cli.add_argument("--out_name", default="custom_render", help="Subfolder name under test/ours_xxxxx/")
    cli.add_argument("--ply_path", type=str, default=None, help="Optional: directly specify a .ply path to load instead of model checkpoint")
    cli.add_argument("--sh_degree", type=int, default=3, help="Spherical Harmonics degree used in training (default: 3)")
    cli.add_argument("--resolution", type=float, default=-1, help="-1 keeps ~1600px width cap; 1=full, 2=half, 4=quarter, ...")
    cli.add_argument("--object_number", type=int, default=0, help="Object number for multi-object scenes")
    args_cli = cli.parse_args()

    # ---- check camera.json ----
    if not os.path.isfile(args_cli.camera_json):
        raise FileNotFoundError(f"camera_json not found: {args_cli.camera_json}")

    # ---- temporary transforms ----
    tmpdir = tempfile.mkdtemp(prefix="3dgs_custom_")
    tf_test  = os.path.join(tmpdir, "transforms_test.json")
    tf_train = os.path.join(tmpdir, "transforms_train.json")

    with open(args_cli.camera_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k in ["fl_x", "fl_y", "cx", "cy", "w", "h", "frames"]:
        if k not in data:
            raise ValueError(f"camera_json missing required field: {k}")

    with open(tf_test, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    empty_train = {
        "fl_x": data["fl_x"], "fl_y": data["fl_y"],
        "cx": data["cx"], "cy": data["cy"],
        "w": data["w"], "h": data["h"],
        "camera_angle_x": data.get("camera_angle_x", None),
        "camera_angle_y": data.get("camera_angle_y", None),
        "camera_model": data.get("camera_model", "OPENCV"),
        "frames": []
    }
    with open(tf_train, "w", encoding="utf-8") as f:
        json.dump(empty_train, f, indent=2)

    # ---- Build 3DGS parsers & defaults ----
    parser = ArgumentParser(add_help=False)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--object_number', type=int, default=0, help='Object number for multi-object scenes')

    args = parser.parse_args([])

    # fill args from CLI
    args.source_path = tmpdir
    args.model_path  = args_cli.model_path
    args.images      = args_cli.images_ext
    args.eval        = True
    args.skip_train  = True
    args.skip_test   = False
    args.iteration   = args_cli.iteration
    args.quiet       = args_cli.quiet
    args.depths      = ""
    args.white_background = args_cli.white_background
    args.train_test_exp   = False
    args.data_device = "cpu"
    args.resolution  = args_cli.resolution
    args.downsample  = 1.0
    args.sh_degree   = int(args_cli.sh_degree)
    args.object_number = args_cli.object_number

    print(f"[Info] model_path        : {args.model_path}")
    print(f"[Info] camera_json       : {args_cli.camera_json}")
    print(f"[Info] iteration         : {args.iteration}")
    print(f"[Info] ply_path          : {args_cli.ply_path}")
    print(f"[Info] white background  : {args.white_background}")
    print(f"[Info] object_number     : {args.object_number}")

    os.environ["GS_IMAGES_EXT"] = args.images

    # ---- init & load ---    safe_state(args.quiet)
    with torch.no_grad():
        extracted_model_args = model.extract(args)
        extracted_model_args.sh_degree = args.sh_degree
        g = GaussianModel(extracted_model_args.sh_degree)

        scene_name = os.path.basename(os.path.normpath(extracted_model_args.model_path))

        if args.object_number > 0:
            name = f"test/{scene_name}_{args.object_number}"
        else:
            name = "test"

        if args_cli.ply_path is not None and os.path.isfile(args_cli.ply_path):
            print(f"[Info] Using custom PLY path: {args_cli.ply_path}")
            g.load_ply(args_cli.ply_path)
            sc = Scene(extracted_model_args, g, load_iteration=None, shuffle=False)
            sc.loaded_iter = args_cli.iteration if args_cli.iteration > 0 else 0
        else:
            print("[Info] No custom ply_path provided — loading from model directory")
            sc = Scene(extracted_model_args, g, load_iteration=args.iteration, shuffle=False)

        bg_color   = [1,1,1] if extracted_model_args.white_background else [0,0,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        extracted_pipeline_args = pipeline.extract(args)

        render_set_only_renders(
            model_path=extracted_model_args.model_path,
            name=name,
            iteration=sc.loaded_iter,
            views=sc.getTestCameras(),
            gaussians=g,
            pipeline=extracted_pipeline_args,
            background=background,
            train_test_exp=extracted_model_args.train_test_exp,
            out_name=args_cli.out_name
        )

    # cleanup
    try:
        shutil.rmtree(tmpdir)
    except Exception as e:
        print(f"[Warn] temp dir not removed: {tmpdir} ({e})")


if __name__ == "__main__":
    main()
