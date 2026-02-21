"""
FEGAN Test Submission
=====================
Runs the trained generator on the test set and produces a submission zip
with corrected images named {image_id}.jpg.

Usage:
    python submit.py --checkpoint results/fisheye_full/checkpoints/latest_net_G_AB.pth \
                     --test_dir /kaggle/input/automatic-lens-correction/test-originals

Output:
    submission/
        {image_id}.jpg    (one per test image)
    submission.zip        (ready to submit)
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "training", "FEGAN-master"))
from models.networks import define_G


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def load_generator(checkpoint_path, ngf=64, which_model="unet_128",
                   use_att=False, norm="instance", gpu_ids=[]):
    flow_nc = 2
    netG = define_G(3, flow_nc, ngf, which_model, norm=norm,
                    use_dropout=False, init_type="xavier",
                    use_att=use_att, gpu_ids=gpu_ids)
    device = torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids else torch.device("cpu")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    netG.load_state_dict(state_dict)
    netG.eval()
    return netG, device


def rectify_image(netG, image_tensor, device, upsample_flow=2.0):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        real_down = F.interpolate(image_tensor, scale_factor=1.0 / upsample_flow)
        flow = netG(real_down)
        flow = F.interpolate(flow, scale_factor=upsample_flow).permute(0, 2, 3, 1)

        theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).repeat(
            image_tensor.shape[0], 1, 1)
        grid = F.affine_grid(theta, image_tensor.shape, align_corners=True).to(device)
        fake = F.grid_sample(image_tensor, flow + grid,
                             padding_mode="zeros", align_corners=True)
    return fake


def main():
    parser = argparse.ArgumentParser(description="Generate submission zip from test images.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to generator checkpoint (.pth)")
    parser.add_argument("--test_dir", required=True,
                        help="Folder of test images "
                             "(default: /kaggle/input/automatic-lens-correction/test-originals)")
    parser.add_argument("--output_dir", default="./submission",
                        help="Directory to write corrected images")
    parser.add_argument("--zip_path", default="./submission.zip",
                        help="Path for the output zip file")
    parser.add_argument("--size", type=int, default=256,
                        help="Image size for inference")
    parser.add_argument("--ngf", type=int, default=64,
                        help="Generator filter count (must match training)")
    parser.add_argument("--which_model", default="unet_128",
                        help="Generator architecture (must match training)")
    parser.add_argument("--use_att", action="store_true",
                        help="Use attention (must match training)")
    parser.add_argument("--upsample_flow", type=float, default=2.0,
                        help="Flow upsample ratio (must match training)")
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="GPU ids (0 for GPU, -1 for CPU)")
    parser.add_argument("--jpeg_quality", type=int, default=95,
                        help="JPEG quality for output images")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if int(x) >= 0]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading generator from {args.checkpoint}")
    netG, device = load_generator(
        args.checkpoint, ngf=args.ngf, which_model=args.which_model,
        use_att=args.use_att, gpu_ids=gpu_ids,
    )

    transform = transforms.Compose([
        transforms.Resize((args.size, args.size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_images = sorted([
        p for p in Path(args.test_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    print(f"Found {len(test_images)} test images in {args.test_dir}")

    output_paths = []
    for i, img_path in enumerate(test_images):
        img = Image.open(str(img_path)).convert("RGB")
        original_size = img.size  # (W, H)

        tensor_in = transform(img).unsqueeze(0)
        tensor_out = rectify_image(netG, tensor_in, device,
                                   upsample_flow=args.upsample_flow)

        arr = tensor_out[0].cpu().float().numpy()
        arr = np.transpose(arr, (1, 2, 0))
        arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        result = Image.fromarray(arr)

        result = result.resize(original_size, Image.BICUBIC)

        image_id = img_path.stem
        out_name = f"{image_id}.jpg"
        out_path = os.path.join(args.output_dir, out_name)
        result.save(out_path, "JPEG", quality=args.jpeg_quality)
        output_paths.append(out_path)

        if (i + 1) % 100 == 0 or (i + 1) == len(test_images):
            print(f"  [{i+1}/{len(test_images)}] {img_path.name} -> {out_name}")

    print(f"\nCorrected images saved to {args.output_dir}/ ({len(output_paths)} images)")

    print(f"Creating {args.zip_path}...")
    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in output_paths:
            zf.write(path, os.path.basename(path))

    zip_size_mb = os.path.getsize(args.zip_path) / (1024 * 1024)
    print(f"Done! {args.zip_path} ({zip_size_mb:.1f} MB, {len(output_paths)} images)")


if __name__ == "__main__":
    main()
