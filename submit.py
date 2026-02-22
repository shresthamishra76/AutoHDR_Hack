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
import cv2

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


_realesrgan_upsampler = None


def get_realesrgan_upsampler(scale=4, device="cpu", half=False):
    """Lazily create and cache the Real-ESRGAN upsampler (created once, reused)."""
    global _realesrgan_upsampler
    if _realesrgan_upsampler is not None:
        return _realesrgan_upsampler

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        raise ImportError(
            "Real-ESRGAN requires: pip install realesrgan basicsr"
        )

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=scale)

    model_url = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.1.0/RealESRGAN_x4plus.pth"
    )

    # Check for a local cached copy first
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "realesrgan")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, f"RealESRGAN_x{scale}plus.pth")

    if not os.path.isfile(model_path):
        print(f"Downloading Real-ESRGAN weights to {model_path}...")
        import urllib.request
        urllib.request.urlretrieve(model_url, model_path)

    _realesrgan_upsampler = RealESRGANer(
        scale=scale, model_path=model_path,
        model=model, half=half, device=device
    )
    return _realesrgan_upsampler


def upsample_realesrgan(image_arr, scale=4, device="cpu"):
    """Upsample using Real-ESRGAN. image_arr is uint8 BGR numpy array."""
    upsampler = get_realesrgan_upsampler(scale=scale, device=device)
    output, _ = upsampler.enhance(image_arr, outscale=scale)
    return output


def check_quality(corrected_arr, image_name):
    """Hard-fail detection: check for catastrophic artifacts in corrected output.

    Checks:
    - Large black/white regions (model collapse or padding artifacts)
    - Very low edge count (blank or washed-out output)
    """
    warnings_list = []

    gray = cv2.cvtColor(corrected_arr, cv2.COLOR_RGB2GRAY)

    # Check for large black regions (padding/zeros from grid_sample)
    black_frac = (gray < 5).sum() / gray.size
    if black_frac > 0.25:
        warnings_list.append(f"  HIGH BLACK REGION ({black_frac:.1%} of pixels < 5)")

    # Check for large white regions (saturation)
    white_frac = (gray > 250).sum() / gray.size
    if white_frac > 0.25:
        warnings_list.append(f"  HIGH WHITE REGION ({white_frac:.1%} of pixels > 250)")

    # Check edge count — very low means washed-out/blank output
    edges = cv2.Canny(gray, 50, 150)
    edge_frac = (edges > 0).sum() / edges.size
    if edge_frac < 0.01:
        warnings_list.append(f"  VERY LOW EDGE COUNT ({edge_frac:.4f})")

    if warnings_list:
        print(f"  WARNING [{image_name}]:")
        for w_msg in warnings_list:
            print(w_msg)
    return warnings_list


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
    parser.add_argument("--size", type=int, default=None,
                        help="(deprecated, images are no longer resized)")
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
    parser.add_argument("--upsampler", type=str, default=None,
                        help="(deprecated, images are no longer resized)")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if int(x) >= 0]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading generator from {args.checkpoint}")
    netG, device = load_generator(
        args.checkpoint, ngf=args.ngf, which_model=args.which_model,
        use_att=args.use_att, gpu_ids=gpu_ids,
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_images = sorted([
        p for p in Path(args.test_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    print(f"Found {len(test_images)} test images in {args.test_dir}")

    output_paths = []
    total_warnings = 0
    for i, img_path in enumerate(test_images):
        img = Image.open(str(img_path)).convert("RGB")

        tensor_in = transform(img).unsqueeze(0)
        tensor_out = rectify_image(netG, tensor_in, device,
                                   upsample_flow=args.upsample_flow)

        arr = tensor_out[0].cpu().float().numpy()
        arr = np.transpose(arr, (1, 2, 0))
        arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

        result = Image.fromarray(arr)

        # Quality check for catastrophic artifacts
        result_arr = np.array(result)
        quality_warnings = check_quality(result_arr, img_path.name)
        if quality_warnings:
            total_warnings += 1

        image_id = img_path.stem
        out_name = f"{image_id}.jpg"
        out_path = os.path.join(args.output_dir, out_name)
        result.save(out_path, "JPEG", quality=args.jpeg_quality)
        output_paths.append(out_path)

        if (i + 1) % 100 == 0 or (i + 1) == len(test_images):
            print(f"  [{i+1}/{len(test_images)}] {img_path.name} -> {out_name}")

    print(f"\nCorrected images saved to {args.output_dir}/ ({len(output_paths)} images)")
    if total_warnings > 0:
        print(f"  {total_warnings} images had quality warnings (see above)")

    print(f"Creating {args.zip_path}...")
    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in output_paths:
            zf.write(path, os.path.basename(path))

    zip_size_mb = os.path.getsize(args.zip_path) / (1024 * 1024)
    print(f"Done! {args.zip_path} ({zip_size_mb:.1f} MB, {len(output_paths)} images)")


if __name__ == "__main__":
    main()
