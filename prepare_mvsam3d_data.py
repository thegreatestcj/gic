"""
Convert PAC-NeRF dataset to MV-SAM3D input format.

PAC-NeRF format:
  data/pacnerf/torus/all_data.json
  165 cameras = 15 timesteps x 11 views

MV-SAM3D format (per timestep):
  output_dir/
    frame_00/
      images/
        0.png, 1.png, ..., 10.png
      torus/
        0_mask.png, 1_mask.png, ..., 10_mask.png

Usage:
  python prepare_mvsam3d_data.py \
      --data_path data/pacnerf/torus \
      --config_path config/pacnerf/torus.json \
      --output_dir data/mvsam3d_torus
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from scene.dataset_readers import readPACNeRFInfo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/pacnerf/torus")
    parser.add_argument("--config_path", type=str, default="config/pacnerf/torus.json")
    parser.add_argument("--output_dir", type=str, default="data/mvsam3d_torus")
    parser.add_argument("--white_background", action="store_true", default=True)
    args = parser.parse_args()

    # Read PAC-NeRF data
    scene_info = readPACNeRFInfo(args.data_path, args.config_path, args.white_background)
    cams = scene_info.train_cameras

    # Group cameras by fid (timestep)
    frames = {}
    for ci in cams:
        fid = ci.fid
        if fid not in frames:
            frames[fid] = []
        frames[fid].append(ci)

    # Sort timesteps
    sorted_fids = sorted(frames.keys())
    print(f"Found {len(sorted_fids)} timesteps, {len(cams)} total cameras")
    print(f"Views per timestep: {[len(frames[f]) for f in sorted_fids[:5]]}...")

    # Also save camera extrinsics for later use
    cam_data = {}

    for frame_idx, fid in enumerate(tqdm(sorted_fids, desc="Converting")):
        views = sorted(frames[fid], key=lambda ci: ci.uid)  # Sort by camera uid for consistent ordering
        frame_dir = Path(args.output_dir) / f"frame_{frame_idx:02d}"
        images_dir = frame_dir / "images"
        mask_dir = frame_dir / "torus"
        images_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        frame_cams = []
        for view_idx, ci in enumerate(views):
            # Save image as RGB PNG
            img = np.array(ci.image)  # (H, W, 3) or (H, W, 4)
            if img.shape[-1] == 4:
                img_rgb = img[:, :, :3]
            else:
                img_rgb = img
            Image.fromarray(img_rgb.astype(np.uint8)).save(images_dir / f"{view_idx}.png")

            # Save mask as RGBA (MV-SAM3D expects alpha channel as mask)
            alpha = np.array(ci.alpha) if hasattr(ci, 'alpha') and ci.alpha is not None else None
            if alpha is not None:
                if alpha.ndim == 3:
                    alpha = alpha[:, :, 0]
                if alpha.max() <= 1.0:
                    alpha = (alpha * 255).astype(np.uint8)
                else:
                    alpha = alpha.astype(np.uint8)
            else:
                # No mask, use all white
                alpha = np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8) * 255

            # MV-SAM3D mask format: RGBA where alpha=255 for object
            rgba = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = img_rgb
            rgba[:, :, 3] = alpha
            Image.fromarray(rgba).save(mask_dir / f"{view_idx}_mask.png")

            # Save camera info
            frame_cams.append({
                'view_idx': view_idx,
                'R': ci.R.tolist(),
                'T': ci.T.tolist(),
                'FovX': ci.FovX,
                'FovY': ci.FovY,
                'width': ci.width,
                'height': ci.height,
                'image_name': ci.image_name,
            })

        cam_data[f"frame_{frame_idx:02d}"] = {
            'fid': fid,
            'cameras': frame_cams,
        }

    # Save camera data
    cam_path = Path(args.output_dir) / "cameras.json"
    with open(cam_path, 'w') as f:
        json.dump(cam_data, f, indent=2)

    print(f"\nDone! Converted {len(sorted_fids)} frames to {args.output_dir}")
    print(f"Camera data saved to {cam_path}")
    print(f"\nTo run MV-SAM3D on frame 0:")
    print(f"  cd /home/fding/MV-SAM3D")
    print(f"  python run_inference_weighted.py \\")
    print(f"      --input_path {os.path.abspath(args.output_dir)}/frame_00 \\")
    print(f"      --mask_prompt torus \\")
    print(f"      --image_names {','.join(str(i) for i in range(len(frames[sorted_fids[0]])))} \\")
    print(f"      --no_stage1_weighting --no_stage2_weighting")


if __name__ == "__main__":
    main()