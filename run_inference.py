"""
Simplified FEGAN Inference
==========================
Runs a trained FEGAN generator on a folder of distorted images and saves
rectified outputs into a flat results directory.

Usage:
    python run_inference.py --checkpoint results/my_exp/checkpoints/latest_net_G_AB.pth \
                            --input_dir datasets/fisheye_tiny/testA \
                            --name my_exp

    python run_inference.py --checkpoint results/my_exp/checkpoints/latest_net_G_AB.pth \
                            --input_dir datasets/fisheye_tiny/testA \
                            --gt_dir datasets/fisheye_tiny/testB \
                            --name my_exp

Output:
    results/{name}/
        input/          original distorted images
        output/         model-rectified images
        ground_truth/   (if --gt_dir provided) ground truth images
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "training", "FEGAN-master"))
from models.networks import define_G


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def load_generator(checkpoint_path, ngf=64, which_model="unet_128", use_att=False, norm="instance", gpu_ids=[]):
    flow_nc = 2
    netG = define_G(3, flow_nc, ngf, which_model, norm=norm,
                     use_dropout=False, init_type="xavier",
                     use_att=use_att, gpu_ids=gpu_ids)
    device = torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids else torch.device("cpu")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    netG.load_state_dict(state_dict)
    netG.eval()
    return netG, device


def load_image(path, size=256):
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform(img).unsqueeze(0)


def tensor_to_image(tensor):
    img = tensor[0].cpu().float().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img + 1.0) / 2.0 * 255.0
    return img.clip(0, 255).astype(np.uint8)


def run_inference(netG, image_tensor, device, upsample_flow=2.0):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        real_down = F.interpolate(image_tensor, scale_factor=1.0 / upsample_flow)
        flow = netG(real_down)
        flow = F.interpolate(flow, scale_factor=upsample_flow).permute(0, 2, 3, 1)

        theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).repeat(image_tensor.shape[0], 1, 1)
        grid = F.affine_grid(theta, image_tensor.shape, align_corners=True).to(device)
        fake = F.grid_sample(image_tensor, flow + grid, padding_mode="zeros", align_corners=True)
    return fake


def find_images(directory):
    images = []
    for entry in sorted(Path(directory).iterdir()):
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(entry)
    return images


def main():
    parser = argparse.ArgumentParser(description="Run FEGAN inference on a folder of images.")
    parser.add_argument("--checkpoint", required=True, help="Path to generator checkpoint (.pth)")
    parser.add_argument("--input_dir", required=True, help="Folder of distorted input images")
    parser.add_argument("--gt_dir", default=None, help="Folder of ground truth images (optional)")
    parser.add_argument("--name", default="inference", help="Experiment name (results saved to results/{name}/)")
    parser.add_argument("--results_root", default="./results", help="Root results directory")
    parser.add_argument("--size", type=int, default=256, help="Image size for inference")
    parser.add_argument("--ngf", type=int, default=64, help="Generator filter count (must match training)")
    parser.add_argument("--which_model", default="unet_128", help="Generator architecture (must match training)")
    parser.add_argument("--use_att", action="store_true", help="Use attention (must match training)")
    parser.add_argument("--upsample_flow", type=float, default=2.0, help="Flow upsample ratio (must match training)")
    parser.add_argument("--gpu_ids", type=str, default="-1", help="GPU ids (-1 for CPU)")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if int(x) >= 0]

    results_dir = os.path.join(args.results_root, args.name)
    input_out = os.path.join(results_dir, "input")
    output_out = os.path.join(results_dir, "output")
    gt_out = os.path.join(results_dir, "ground_truth")

    os.makedirs(input_out, exist_ok=True)
    os.makedirs(output_out, exist_ok=True)
    if args.gt_dir:
        os.makedirs(gt_out, exist_ok=True)

    print(f"Loading generator from {args.checkpoint}")
    netG, device = load_generator(
        args.checkpoint, ngf=args.ngf, which_model=args.which_model,
        use_att=args.use_att, gpu_ids=gpu_ids,
    )

    images = find_images(args.input_dir)
    print(f"Found {len(images)} images in {args.input_dir}")

    gt_images = {}
    if args.gt_dir:
        for p in find_images(args.gt_dir):
            gt_images[p.stem.lower()] = p

    for i, img_path in enumerate(images):
        image_tensor = load_image(str(img_path), size=args.size)
        output_tensor = run_inference(netG, image_tensor, device, upsample_flow=args.upsample_flow)

        out_name = img_path.stem + ".png"

        # Save input
        input_img = tensor_to_image(image_tensor)
        Image.fromarray(input_img).save(os.path.join(input_out, out_name))

        # Save output
        output_img = tensor_to_image(output_tensor)
        Image.fromarray(output_img).save(os.path.join(output_out, out_name))

        # Copy ground truth if available
        if args.gt_dir:
            gt_key = img_path.stem.lower()
            if gt_key in gt_images:
                gt_img = Image.open(str(gt_images[gt_key])).convert("RGB")
                gt_img = gt_img.resize((args.size, args.size), Image.BICUBIC)
                gt_img.save(os.path.join(gt_out, out_name))

        print(f"  [{i+1}/{len(images)}] {img_path.name} -> {out_name}")

    print(f"\nResults saved to {results_dir}/")
    print(f"  input/        ({len(images)} images)")
    print(f"  output/       ({len(images)} images)")
    if args.gt_dir:
        print(f"  ground_truth/ ({len(gt_images)} images)")


if __name__ == "__main__":
    main()
