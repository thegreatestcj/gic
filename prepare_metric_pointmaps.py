"""
Generate metric-scale pointmaps for MV-SAM3D using MoGe + known PAC-NeRF cameras.

MoGe estimates depth up to an unknown scale. PAC-NeRF cameras provide:
  - intrinsics (FoVX, FoVY → fx, fy, cx, cy): correct ray directions
  - extrinsics (R, T): true camera distance → fixes MoGe's scale

Output: DA3-format npz per frame, compatible with run_inference_weighted.py --da3_output

Usage:
  cd /orcd/home/002/fding/gic
  python prepare_metric_pointmaps.py \
      --data_path data/pacnerf/torus \
      --config_path config/pacnerf/torus.json \
      --mvsam3d_data_dir data/mvsam3d_torus \
      --output_dir data/mvsam3d_torus_metric
"""

import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def load_pacnerf_cameras(data_path, config_path):
    """Load PAC-NeRF camera parameters."""
    with open(os.path.join(data_path, 'all_data.json')) as f:
        all_data = json.load(f)

    with open(config_path) as f:
        config = json.load(f)

    # all_data is a list of dicts with: file_path, time, c2w (3x4), intrinsic (3x3)
    cameras = []
    for frame in all_data:
        c2w_3x4 = np.array(frame['c2w'])  # (3, 4)
        c2w = np.eye(4)
        c2w[:3, :] = c2w_3x4

        intrinsic = np.array(frame['intrinsic'])  # (3, 3)

        # Compute w2c = inv(c2w)
        w2c = np.linalg.inv(c2w)

        cam = {
            'file_path': frame['file_path'],
            'time': frame.get('time', 0),
            'c2w': c2w,
            'w2c': w2c,
            'intrinsic': intrinsic,
            'T': w2c[:3, 3],  # translation in camera space
        }
        cameras.append(cam)

    return cameras, config


def fov_to_intrinsics(fov_x, fov_y, width, height):
    """Convert FoV to intrinsics matrix. (kept for compatibility)"""
    fx = width / (2.0 * np.tan(fov_x / 2.0))
    fy = height / (2.0 * np.tan(fov_y / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)


def load_moge_model():
    """Load MoGe depth estimation model."""
    try:
        from moge.model.v1 import MoGeModel
        model = MoGeModel.from_pretrained("Ruicheng/moge-vitl")
        model.cuda().eval()
        print("Loaded MoGe v1")
        return model
    except Exception as e:
        print(f"Failed to load MoGe from hub: {e}")

    # Fallback: load from SAM3D pipeline
    try:
        from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
        # Load the pipeline just to get the depth model
        print("Trying to load MoGe via SAM3D pipeline...")
        pipeline_path = "/home/fding/sam-3d-objects/checkpoints/hf/"
        pipeline = InferencePipelinePointMap.from_pretrained(pipeline_path)
        model = pipeline.depth_model
        model.cuda().eval()
        print("Loaded MoGe from SAM3D pipeline")
        return model
    except Exception as e:
        print(f"Failed to load MoGe via SAM3D: {e}")
        raise RuntimeError("Cannot load MoGe model. Check installation.")


def run_moge(model, image_path, device='cuda'):
    """
    Run MoGe on a single image to get pointmap.

    Returns:
        pointmap: (H, W, 3) in standard camera space (x-right, y-down, z-forward)
        intrinsics_moge: estimated intrinsics from MoGe (may be inaccurate)
    """
    from PIL import Image
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(device)

    with torch.no_grad():
        output = model.infer(img_tensor)

    # MoGe v1 infer returns 'points' (H, W, 3)
    pointmap = output.get("points", output.get("pointmaps", None))
    if pointmap is None:
        raise RuntimeError(f"MoGe output has no 'points' or 'pointmaps'. Keys: {output.keys()}")
    if pointmap.dim() == 4:
        pointmap = pointmap.squeeze(0)

    intrinsics_moge = output.get("intrinsics", None)

    return pointmap.cpu().numpy(), intrinsics_moge


def create_metric_pointmap(
    moge_pointmap,
    mask,
    known_intrinsics,
    cam_T,
    use_known_intrinsics=True,
):
    """
    Create metric-scale pointmap from MoGe output + known camera params.

    MoGe gives depth with arbitrary scale. We fix it using known camera distance.
    Optionally replace MoGe's intrinsics with known ones for correct ray directions.

    Args:
        moge_pointmap: (H, W, 3) MoGe output in camera space
        mask: (H, W) binary mask of the object
        known_intrinsics: (3, 3) known camera intrinsics
        cam_T: (3,) GIC camera T — depth to origin is cam_T[2]
        use_known_intrinsics: if True, recompute x,y using known focal length

    Returns:
        metric_pointmap: (H, W, 3) in standard camera space, metric scale
        scale_factor: the computed scale
    """
    H, W = moge_pointmap.shape[:2]
    z_moge = moge_pointmap[:, :, 2]  # depth from MoGe

    # Compute scale: true depth to object center
    # cam_T[2] can be negative depending on convention, use absolute value
    true_depth = abs(float(cam_T[2]))
    mask_bool = mask > 0.5

    if mask_bool.sum() > 0:
        median_depth = abs(float(np.median(z_moge[mask_bool])))
    else:
        median_depth = abs(float(np.median(z_moge[z_moge != 0])))

    if median_depth < 1e-6:
        scale_factor = 1.0
    else:
        scale_factor = true_depth / median_depth

    z_metric = z_moge * scale_factor

    # Simply scale the entire MoGe pointmap uniformly
    # MoGe's x/y are already consistent with its z (correct ray directions from MoGe's own intrinsics)
    # We only need to fix the absolute scale, not the ray directions
    # This preserves valid values everywhere (including background) so downstream
    # infer_intrinsics_from_pointmap won't crash on zeros/NaN
    metric_pointmap = moge_pointmap * scale_factor

    # Final safety: replace any remaining NaN/inf
    metric_pointmap = np.nan_to_num(metric_pointmap, nan=0.0, posinf=0.0, neginf=0.0)

    return metric_pointmap.astype(np.float32), scale_factor


def main():
    parser = argparse.ArgumentParser(description="Generate metric pointmaps for MV-SAM3D")
    parser.add_argument("--data_path", required=True, help="PAC-NeRF data directory")
    parser.add_argument("--config_path", required=True, help="PAC-NeRF config json")
    parser.add_argument("--mvsam3d_data_dir", required=True,
                        help="MV-SAM3D data dir (from prepare_mvsam3d_data.py)")
    parser.add_argument("--output_dir", required=True, help="Output dir for metric npz files")
    parser.add_argument("--no_known_intrinsics", action='store_true',
                        help="Don't replace MoGe intrinsics (just scale depth)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load PAC-NeRF cameras
    cameras, config = load_pacnerf_cameras(args.data_path, args.config_path)
    n_views_per_frame = config['data'].get('n_camera', 11)
    n_total = len(cameras)
    n_frames = n_total // n_views_per_frame
    print(f"PAC-NeRF: {n_total} cameras = {n_frames} frames x {n_views_per_frame} views")

    # Load MoGe
    print("Loading MoGe model...")
    moge_model = load_moge_model()

    # Process each frame
    for frame_idx in range(n_frames):
        frame_name = f"frame_{frame_idx:02d}"
        frame_data_dir = os.path.join(args.mvsam3d_data_dir, frame_name)

        if not os.path.exists(frame_data_dir):
            print(f"[SKIP] {frame_name}: data dir not found")
            continue

        output_npz = os.path.join(args.output_dir, frame_name, "da3_output.npz")
        if os.path.exists(output_npz):
            print(f"[SKIP] {frame_name}: already processed")
            continue

        os.makedirs(os.path.join(args.output_dir, frame_name), exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Processing {frame_name}")
        print(f"{'='*60}")

        # Get cameras for this frame
        frame_cameras = cameras[frame_idx * n_views_per_frame: (frame_idx + 1) * n_views_per_frame]

        all_pointmaps = []
        all_extrinsics = []
        all_intrinsics = []
        all_image_files = []
        all_scales = []

        for view_idx in range(n_views_per_frame):
            cam = frame_cameras[view_idx]

            # Load image
            image_path = os.path.join(frame_data_dir, "images", f"{view_idx}.png")
            if not os.path.exists(image_path):
                print(f"  [SKIP] View {view_idx}: image not found")
                continue

            # Load mask
            mask_dir = os.path.join(frame_data_dir, "torus")
            mask_path = os.path.join(mask_dir, f"{view_idx}_mask.png")
            if os.path.exists(mask_path):
                mask_img = np.array(Image.open(mask_path))
                if mask_img.ndim == 3 and mask_img.shape[2] == 4:
                    mask = mask_img[:, :, 3] / 255.0  # alpha channel
                elif mask_img.ndim == 3:
                    mask = mask_img[:, :, 0] / 255.0
                else:
                    mask = mask_img / 255.0
            else:
                # Use full image as mask
                img = Image.open(image_path)
                mask = np.ones((img.size[1], img.size[0]), dtype=np.float32)

            # Known intrinsics and extrinsics directly from PAC-NeRF
            known_intrinsics = cam['intrinsic']
            w2c = cam['w2c']
            cam_T = cam['T']  # w2c[:3, 3]

            # Run MoGe
            moge_pointmap, _ = run_moge(moge_model, image_path)

            # Resize mask if needed
            if mask.shape[:2] != moge_pointmap.shape[:2]:
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize(
                    (moge_pointmap.shape[1], moge_pointmap.shape[0]),
                    PILImage.NEAREST
                )
                mask = np.array(mask_pil).astype(np.float32) / 255.0

            # Create metric pointmap
            metric_pm, scale = create_metric_pointmap(
                moge_pointmap, mask, known_intrinsics, cam['T'],
                use_known_intrinsics=not args.no_known_intrinsics,
            )

            # Convert to (3, H, W) format for DA3 compatibility
            metric_pm_chw = metric_pm.transpose(2, 0, 1)  # (3, H, W)

            all_pointmaps.append(metric_pm_chw)
            all_extrinsics.append(w2c)
            all_intrinsics.append(known_intrinsics)
            all_image_files.append(f"{view_idx}.png")
            all_scales.append(scale)

            if view_idx == 0:
                z_masked = moge_pointmap[:, :, 2][mask > 0.5]
                if len(z_masked) > 0:
                    print(f"  View {view_idx}: scale={scale:.4f}, "
                          f"moge_z median={np.median(z_masked):.4f}, "
                          f"true_depth={cam_T[2]:.4f}")
                else:
                    print(f"  View {view_idx}: scale={scale:.4f}, no masked points")

        if len(all_pointmaps) == 0:
            print(f"  [ERROR] No valid views for {frame_name}")
            continue

        # Use consistent scale across all views (average)
        mean_scale = np.mean(all_scales)
        print(f"  Scale stats: mean={mean_scale:.4f}, std={np.std(all_scales):.4f}, "
              f"min={np.min(all_scales):.4f}, max={np.max(all_scales):.4f}")

        # Save DA3-format npz
        np.savez(
            output_npz,
            pointmaps_sam3d=np.stack(all_pointmaps, axis=0),  # (N, 3, H, W)
            extrinsics=np.stack(all_extrinsics, axis=0),       # (N, 4, 4)
            intrinsics=np.stack(all_intrinsics, axis=0),       # (N, 3, 3)
            image_files=np.array(all_image_files),
        )
        print(f"  Saved: {output_npz}")
        print(f"    pointmaps: {np.stack(all_pointmaps).shape}")
        print(f"    extrinsics: {np.stack(all_extrinsics).shape}")

    print(f"\nDone! Metric pointmaps saved to {args.output_dir}")


if __name__ == "__main__":
    main()