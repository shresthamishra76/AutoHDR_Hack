"""
FEGAN Data Preprocessing Pipeline
==================================
Organizes raw image pairs into the directory structure expected by
FEGAN-master's unaligned dataset loader, with configurable hyperparameters.

Usage:
    python preprocess.py                          # uses config.yaml
    python preprocess.py --config my_config.yaml  # custom config
    python preprocess.py --input_dir /path/to/raw --target_size 512  # CLI overrides

The output directory will contain:
    trainA/  trainB/  trainBtoA/
    valA/    valB/    valBtoA/
    testA/   testB/   testBtoA/

Plus a generated `run_train.sh` and `run_test.sh` for launching FEGAN
with the hyperparameters from your config.
"""

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
import numpy as np
from PIL import Image


INTERPOLATION_MODES = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".ppm"}


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Override config values with any CLI arguments that were explicitly set."""
    if args.input_dir:
        cfg["source"]["input_dir"] = args.input_dir
    if args.output_dir:
        cfg["output"]["output_dir"] = args.output_dir
    if args.target_size is not None:
        cfg["preprocessing"]["target_size"] = args.target_size
    if args.train_split is not None:
        cfg["split"]["train"] = args.train_split
    if args.val_split is not None:
        cfg["split"]["val"] = args.val_split
    if args.test_split is not None:
        cfg["split"]["test"] = args.test_split
    if args.seed is not None:
        cfg["split"]["seed"] = args.seed
    if args.naming:
        cfg["source"]["naming"] = args.naming
    if args.output_format:
        cfg["preprocessing"]["output_format"] = args.output_format
    if args.fineSize is not None:
        cfg["fegan"]["fineSize"] = args.fineSize
    if args.loadSize is not None:
        cfg["fegan"]["loadSize"] = args.loadSize
    if args.batchSize is not None:
        cfg["fegan"]["batchSize"] = args.batchSize
    if args.lr is not None:
        cfg["fegan"]["lr"] = args.lr
    if args.niter is not None:
        cfg["fegan"]["niter"] = args.niter
    return cfg


# ── Pair Discovery ──────────────────────────────────────────────────────────


def discover_pairs_uuid_camera(input_dir: str, extensions: set) -> list:
    files = defaultdict(dict)
    pattern = re.compile(r"^(.+)_(original|generated)$", re.IGNORECASE)

    for entry in os.scandir(input_dir):
        if not entry.is_file():
            continue
        name = entry.name
        dot = name.rfind(".")
        if dot < 0:
            continue
        ext = name[dot:].lower()
        if ext not in extensions:
            continue
        stem = name[:dot]
        m = pattern.match(stem)
        if not m:
            continue
        key, role = m.group(1), m.group(2).lower()
        files[key][role] = entry.path

    pairs = []
    for key in sorted(files):
        if "original" in files[key] and "generated" in files[key]:
            pairs.append({
                "id": key,
                "A": files[key]["original"],
                "B": files[key]["generated"],
            })
    return pairs


def discover_pairs_suffix(input_dir: str, suffix_a: str, suffix_b: str, extensions: set) -> list:
    files = defaultdict(dict)

    for entry in os.scandir(input_dir):
        if not entry.is_file():
            continue
        name = entry.name
        dot = name.rfind(".")
        if dot < 0:
            continue
        ext = name[dot:].lower()
        if ext not in extensions:
            continue
        stem = name[:dot]
        if stem.endswith(suffix_a):
            key = stem[: -len(suffix_a)]
            files[key]["A"] = entry.path
        elif stem.endswith(suffix_b):
            key = stem[: -len(suffix_b)]
            files[key]["B"] = entry.path

    pairs = []
    for key in sorted(files):
        if "A" in files[key] and "B" in files[key]:
            pairs.append({"id": key, "A": files[key]["A"], "B": files[key]["B"]})
    return pairs


def discover_pairs_subfolder(input_dir: str, subfolder_a: str, subfolder_b: str, extensions: set) -> list:
    dir_a = os.path.join(input_dir, subfolder_a)
    dir_b = os.path.join(input_dir, subfolder_b)

    files_a = {}
    if os.path.isdir(dir_a):
        for entry in os.scandir(dir_a):
            if entry.is_file():
                name = entry.name
                dot = name.rfind(".")
                if dot >= 0 and name[dot:].lower() in extensions:
                    files_a[name[:dot].lower()] = entry.path

    pairs = []
    if os.path.isdir(dir_b):
        for entry in sorted(os.scandir(dir_b), key=lambda e: e.name):
            if entry.is_file():
                name = entry.name
                dot = name.rfind(".")
                if dot >= 0 and name[dot:].lower() in extensions:
                    key = name[:dot].lower()
                    if key in files_a:
                        pairs.append({"id": name[:dot], "A": files_a[key], "B": entry.path})
    return pairs


def discover_pairs(cfg: dict) -> list:
    src = cfg["source"]
    input_dir = src["input_dir"]
    extensions = {e.lower() for e in src.get("extensions", IMAGE_EXTENSIONS)}
    naming = src.get("naming", "uuid_camera")

    if naming == "uuid_camera":
        return discover_pairs_uuid_camera(input_dir, extensions)
    elif naming == "suffix":
        return discover_pairs_suffix(input_dir, src["suffix_a"], src["suffix_b"], extensions)
    elif naming == "subfolder":
        return discover_pairs_subfolder(input_dir, src["subfolder_a"], src["subfolder_b"], extensions)
    else:
        raise ValueError(f"Unknown naming mode: {naming}")


# ── Image Processing ────────────────────────────────────────────────────────


def _process_single_image(src_path: str, dst_path: str, target_size, interp, fmt, jpeg_quality, min_dim, skip_corrupt):
    """Process one image. Designed to be called in a worker process."""
    try:
        img = Image.open(src_path)
        img.load()
        img = img.convert("RGB")
    except Exception as e:
        if skip_corrupt:
            return False, f"corrupt: {e}"
        raise

    w, h = img.size
    if min(w, h) < min_dim:
        return False, f"too small ({w}x{h})"

    if target_size:
        img = img.resize((target_size, target_size), interp)

    if fmt in ("jpg", "jpeg"):
        img.save(dst_path, "JPEG", quality=jpeg_quality)
    else:
        img.save(dst_path, "PNG")

    return True, None


def _process_pair(pair, dir_a, dir_b, dir_bta, ext, target_size, interp, fmt, jpeg_quality, min_dim, skip_corrupt):
    """Process a single pair (A + B + BtoA link). Called in worker process."""
    name = pair["id"]
    dst_a = os.path.join(dir_a, f"{name}{ext}")
    dst_b = os.path.join(dir_b, f"{name}{ext}")
    dst_bta = os.path.join(dir_bta, f"{name}{ext}")

    ok_a, err_a = _process_single_image(pair["A"], dst_a, target_size, interp, fmt, jpeg_quality, min_dim, skip_corrupt)
    if not ok_a:
        return False, err_a

    ok_b, err_b = _process_single_image(pair["B"], dst_b, target_size, interp, fmt, jpeg_quality, min_dim, skip_corrupt)
    if not ok_b:
        if os.path.exists(dst_a):
            os.remove(dst_a)
        return False, err_b

    if os.path.exists(dst_bta):
        os.remove(dst_bta)
    try:
        os.link(dst_b, dst_bta)
    except OSError:
        import shutil
        shutil.copy2(dst_b, dst_bta)

    return True, None


# ── Split & Write ───────────────────────────────────────────────────────────


def split_pairs(pairs: list, cfg: dict) -> dict:
    split_cfg = cfg["split"]
    rng = np.random.RandomState(split_cfg.get("seed", 42))
    indices = rng.permutation(len(pairs))

    train_frac = split_cfg.get("train", 0.8)
    val_frac = split_cfg.get("val", 0.1)

    n = len(pairs)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return {
        "train": [pairs[i] for i in indices[:n_train]],
        "val": [pairs[i] for i in indices[n_train : n_train + n_val]],
        "test": [pairs[i] for i in indices[n_train + n_val :]],
    }


def write_split(pairs: list, phase: str, output_dir: str, cfg_pre: dict, num_workers: int) -> dict:
    """Write one split using parallel workers."""
    fmt = cfg_pre.get("output_format", "png").lower()
    ext = ".jpg" if fmt in ("jpg", "jpeg") else ".png"

    dir_a = os.path.join(output_dir, f"{phase}A")
    dir_b = os.path.join(output_dir, f"{phase}B")
    dir_bta = os.path.join(output_dir, f"{phase}BtoA")
    os.makedirs(dir_a, exist_ok=True)
    os.makedirs(dir_b, exist_ok=True)
    os.makedirs(dir_bta, exist_ok=True)

    target_size = cfg_pre.get("target_size")
    interp = INTERPOLATION_MODES.get(cfg_pre.get("interpolation", "lanczos"), Image.LANCZOS)
    jpeg_quality = cfg_pre.get("jpeg_quality", 95)
    min_dim = cfg_pre.get("min_dimension", 0)
    skip_corrupt = cfg_pre.get("skip_corrupt", True)

    written = 0
    skipped = 0
    total = len(pairs)
    t0 = time.time()
    last_print = 0

    if num_workers <= 1:
        for i, pair in enumerate(pairs):
            ok, err = _process_pair(pair, dir_a, dir_b, dir_bta, ext,
                                    target_size, interp, fmt, jpeg_quality,
                                    min_dim, skip_corrupt)
            if ok:
                written += 1
            else:
                skipped += 1
            done = written + skipped
            if done - last_print >= 500 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"    {done}/{total} pairs  ({rate:.0f} pairs/s)", flush=True)
                last_print = done
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for pair in pairs:
                fut = pool.submit(_process_pair, pair, dir_a, dir_b, dir_bta, ext,
                                  target_size, interp, fmt, jpeg_quality,
                                  min_dim, skip_corrupt)
                futures[fut] = pair

            for fut in as_completed(futures):
                ok, err = fut.result()
                if ok:
                    written += 1
                else:
                    skipped += 1
                done = written + skipped
                if done - last_print >= 500 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"    {done}/{total} pairs  ({rate:.0f} pairs/s)", flush=True)
                    last_print = done

    return {"written": written, "skipped": skipped}


# ── Script Generation ───────────────────────────────────────────────────────


def generate_train_script(output_dir: str, cfg: dict):
    f = cfg["fegan"]
    fegan_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "training", "FEGAN-master")
    )
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_root_abs = os.path.join(project_root, "results")
    output_dir_abs = os.path.abspath(output_dir)

    fegan_rel = os.path.relpath(fegan_root, output_dir_abs)
    results_rel = os.path.relpath(results_root_abs, output_dir_abs)

    flags = [
        '--dataroot "$SCRIPT_DIR"',
        f"--name {f['name']}",
        f"--model {f['model']}",
        f"--batchSize {f['batchSize']}",
        f"--loadSize {f['loadSize']}",
        f"--fineSize {f['fineSize']}",
        f"--input_nc {f['input_nc']}",
        f"--output_nc {f['output_nc']}",
        f"--ngf {f['ngf']}",
        f"--ndf {f['ndf']}",
        f"--niter {f['niter']}",
        f"--niter_decay {f['niter_decay']}",
        f"--lr {f['lr']}",
        f"--beta1 {f['beta1']}",
        f"--lr_policy {f['lr_policy']}",
        f"--save_epoch_freq {f['save_epoch_freq']}",
        f"--which_direction {f['which_direction']}",
        f"--which_model_netG {f['which_model_netG']}",
        f"--which_model_netD {f['which_model_netD']}",
        f"--upsample_flow {f['upsample_flow']}",
        f"--identity {f['identity']}",
        f"--lambda_G {f['lambda_G']}",
        f"--lambda_gc {f['lambda_gc']}",
        f"--lambda_AB {f['lambda_AB']}",
        f"--lambda_smooth {f['lambda_smooth']}",
        f"--lambda_crossflow {f['lambda_crossflow']}",
        f"--lambda_radial {f['lambda_radial']}",
        f"--lambda_rot {f['lambda_rot']}",
        f"--lambda_metric {f['lambda_metric']}",
        f"--w_edge {f['w_edge']}",
        f"--w_line {f['w_line']}",
        f"--w_grad {f['w_grad']}",
        f"--w_ssim {f['w_ssim']}",
        f"--w_pixel {f['w_pixel']}",
        f"--gpu_ids {f['gpu_ids']}",
        f'--results_root "$SCRIPT_DIR/{results_rel}"',
        "--nThreads 0",
        "--tensorboard",
    ]
    if f.get("use_att"):
        flags.append("--use_att")
    if f.get("no_dropout"):
        flags.append("--no_dropout")
    if f.get("no_flip"):
        flags.append("--no_flip")

    script = (
        '#!/usr/bin/env bash\n'
        '# Auto-generated training script -- edit config.yaml and re-run preprocess.py to update\n'
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        f'cd "$SCRIPT_DIR/{fegan_rel}"\n\n'
        f"python3 train.py \\\n  " + " \\\n  ".join(flags) + "\n"
    )

    path = os.path.join(output_dir, "run_train.sh")
    with open(path, "w") as fh:
        fh.write(script)
    os.chmod(path, 0o755)
    return path


def generate_test_script(output_dir: str, cfg: dict):
    f = cfg["fegan"]
    fegan_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "training", "FEGAN-master")
    )
    output_dir_abs = os.path.abspath(output_dir)

    fegan_rel = os.path.relpath(fegan_root, output_dir_abs)

    flags = [
        '--dataroot "$SCRIPT_DIR"',
        f"--name {f['name']}",
        f"--model {f['model']}",
        f"--batchSize 1",
        f"--loadSize {f['fineSize']}",
        f"--fineSize {f['fineSize']}",
        f"--which_direction {f['which_direction']}",
        f"--which_model_netG {f['which_model_netG']}",
        f"--gpu_ids {f['gpu_ids']}",
        "--no_dropout",
        "--which_epoch latest",
        "--phase test",
    ]
    if f.get("use_att"):
        flags.append("--use_att")

    script = (
        '#!/usr/bin/env bash\n'
        '# Auto-generated test/inference script\n'
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        f'cd "$SCRIPT_DIR/{fegan_rel}"\n\n'
        f"python3 test.py \\\n  " + " \\\n  ".join(flags) + "\n"
    )

    path = os.path.join(output_dir, "run_test.sh")
    with open(path, "w") as fh:
        fh.write(script)
    os.chmod(path, 0o755)
    return path


# ── Main ────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess image pairs for FEGAN training/inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"),
                    help="Path to YAML config file (default: config.yaml)")
    p.add_argument("--input_dir", help="Override source.input_dir")
    p.add_argument("--output_dir", help="Override output.output_dir")
    p.add_argument("--naming", choices=["uuid_camera", "suffix", "subfolder"],
                    help="Override source.naming")
    p.add_argument("--target_size", type=int, help="Override preprocessing.target_size")
    p.add_argument("--output_format", choices=["png", "jpg"], help="Override output format")
    p.add_argument("--train_split", type=float, help="Override split.train fraction")
    p.add_argument("--val_split", type=float, help="Override split.val fraction")
    p.add_argument("--test_split", type=float, help="Override split.test fraction")
    p.add_argument("--seed", type=int, help="Override split.seed")

    p.add_argument("--max_pairs", type=int, help="Limit total pairs (useful for quick test runs)")
    p.add_argument("--workers", type=int, default=None,
                    help="Number of parallel workers (default: CPU count)")

    g = p.add_argument_group("FEGAN hyperparameter overrides")
    g.add_argument("--fineSize", type=int, help="FEGAN crop size")
    g.add_argument("--loadSize", type=int, help="FEGAN initial resize")
    g.add_argument("--batchSize", type=int, help="FEGAN batch size")
    g.add_argument("--lr", type=float, help="FEGAN learning rate")
    g.add_argument("--niter", type=int, help="FEGAN training epochs")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    input_dir = cfg["source"]["input_dir"]
    output_dir = cfg["output"]["output_dir"]

    num_workers = args.workers
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)

    print(f"Source directory : {input_dir}")
    print(f"Output directory : {output_dir}")
    print(f"Naming convention: {cfg['source']['naming']}")
    print(f"Workers          : {num_workers}")
    print()

    if not os.path.isdir(input_dir):
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)

    t_start = time.time()

    print("Discovering image pairs...")
    t0 = time.time()
    pairs = discover_pairs(cfg)
    print(f"  Found {len(pairs)} valid pairs ({time.time() - t0:.1f}s)")

    if args.max_pairs and len(pairs) > args.max_pairs:
        rng = np.random.RandomState(cfg["split"].get("seed", 42))
        indices = rng.choice(len(pairs), args.max_pairs, replace=False)
        pairs = [pairs[i] for i in sorted(indices)]
        print(f"  Limited to {len(pairs)} pairs (--max_pairs)")

    if len(pairs) == 0:
        print("No image pairs found. Check your input_dir and naming convention.")
        sys.exit(1)

    print("Splitting dataset...")
    splits = split_pairs(pairs, cfg)
    for phase, phase_pairs in splits.items():
        print(f"  {phase}: {len(phase_pairs)} pairs")

    print()
    cfg_pre = cfg["preprocessing"]
    if cfg_pre.get("target_size"):
        print(f"Pre-resizing images to {cfg_pre['target_size']}x{cfg_pre['target_size']}")
    print(f"Output format: {cfg_pre.get('output_format', 'png')}")
    print()

    os.makedirs(output_dir, exist_ok=True)
    total_written = 0
    total_skipped = 0

    for phase in ("train", "val", "test"):
        phase_pairs = splits[phase]
        if not phase_pairs:
            continue
        print(f"Processing {phase} split ({len(phase_pairs)} pairs)...")
        t0 = time.time()
        stats = write_split(phase_pairs, phase, output_dir, cfg_pre, num_workers)
        elapsed = time.time() - t0
        total_written += stats["written"]
        total_skipped += stats["skipped"]
        print(f"  Written: {stats['written']}, Skipped: {stats['skipped']} ({elapsed:.1f}s)")

    total_time = time.time() - t_start
    print()
    print(f"Total: {total_written} pairs written, {total_skipped} skipped ({total_time:.1f}s)")

    train_script = generate_train_script(output_dir, cfg)
    test_script = generate_test_script(output_dir, cfg)
    print()
    print(f"Generated training script : {train_script}")
    print(f"Generated inference script: {test_script}")

    resolved_path = os.path.join(output_dir, "resolved_config.yaml")
    with open(resolved_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Saved resolved config     : {resolved_path}")

    print()
    print("Done! Next steps:")
    print(f"  1. Train:     bash {train_script}")
    print(f"  2. Inference: bash {test_script}")
    print(f"  3. Tune hyperparameters in config.yaml and re-run this script")


if __name__ == "__main__":
    main()
