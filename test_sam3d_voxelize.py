# test_sam3d_voxelize.py
# Quick sanity check: SAM3D inference -> voxelize -> save point clouds
# Run this BEFORE the full pipeline to verify coordinates are correct.
#
# Usage:
#   python test_sam3d_voxelize.py \
#       --sam3d_config checkpoints/hf/pipeline.yaml \
#       --image path/to/rgba_image.png \
#       --mask path/to/mask.png \
#       --output_dir test_output/

import torch
import numpy as np
import os
import argparse
import trimesh
from pathlib import Path

from sam3d_wrapper import SAM3DWrapper
from train_dynamic_sam3d import voxelize_points


def load_test_image(image_path: str, mask_path: str = None):
    """Load an RGBA image and optional separate mask."""
    from PIL import Image
    
    img = Image.open(image_path)
    img_np = np.array(img)
    
    if img_np.shape[-1] == 4:
        # RGBA image - extract mask from alpha channel
        rgba = img_np
        mask = (rgba[:, :, 3] > 127).astype(np.uint8)
    elif img_np.shape[-1] == 3:
        # RGB image - need separate mask
        if mask_path is None:
            raise ValueError("RGB image provided but no mask path specified")
        mask_img = Image.open(mask_path)
        mask = (np.array(mask_img.convert('L')) > 127).astype(np.uint8)
        rgba = np.concatenate([img_np, mask[:, :, None] * 255], axis=-1)
    else:
        raise ValueError(f"Unexpected image shape: {img_np.shape}")
    
    return rgba, mask


def save_pointcloud(xyz: torch.Tensor, path: str, color=None):
    """Save a point cloud as .ply for visualization."""
    xyz_np = xyz.detach().cpu().numpy()
    if color is None:
        color = np.full((xyz_np.shape[0], 3), [0, 255, 0], dtype=np.uint8)
    elif isinstance(color, (list, tuple)):
        color = np.full((xyz_np.shape[0], 3), color, dtype=np.uint8)
    
    pc = trimesh.PointCloud(xyz_np, colors=color)
    pc.export(path)
    print(f"  Saved {xyz_np.shape[0]} points to {path}")


def main():
    parser = argparse.ArgumentParser(description="Test SAM3D + Voxelization")
    parser.add_argument("--sam3d_config", required=True, help="Path to SAM3D pipeline.yaml")
    parser.add_argument("--sam3d_repo", default=None, help="Path to sam3d-objects repo")
    parser.add_argument("--image", required=True, help="Path to RGBA image")
    parser.add_argument("--mask", default=None, help="Path to mask (if image is RGB)")
    parser.add_argument("--output_dir", default="test_output", help="Output directory")
    parser.add_argument("--grid_size", type=float, default=0.12, help="Voxel grid size")
    parser.add_argument("--density_min_th", type=float, default=0.1)
    parser.add_argument("--density_max_th", type=float, default=0.9)
    # Camera params for testing (identity if not provided)
    parser.add_argument("--cam_R", default=None, help="Camera R as comma-separated 9 floats")
    parser.add_argument("--cam_T", default=None, help="Camera T as comma-separated 3 floats")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ---- 1. Load image ----
    print("Step 1: Loading image...")
    rgba, mask = load_test_image(args.image, args.mask)
    print(f"  Image: {rgba.shape}, Mask sum: {mask.sum()}")
    
    # ---- 2. Camera params ----
    if args.cam_R is not None:
        cam_R = np.array([float(x) for x in args.cam_R.split(',')]).reshape(3, 3)
    else:
        cam_R = np.eye(3)  # identity rotation
    
    if args.cam_T is not None:
        cam_T = np.array([float(x) for x in args.cam_T.split(',')])
    else:
        cam_T = np.zeros(3)  # no translation
    
    print(f"  cam_R:\n{cam_R}")
    print(f"  cam_T: {cam_T}")
    
    # ---- 3. Run SAM3D ----
    print("\nStep 2: Running SAM3D inference...")
    sam3d = SAM3DWrapper(
        config_path=args.sam3d_config,
        sam3d_repo_path=args.sam3d_repo,
    )
    
    output = sam3d.run_inference(rgba, mask)
    xyz_local = sam3d.extract_xyz_from_output(output)
    print(f"  SAM3D output: {xyz_local.shape[0]} gaussians in local space")
    print(f"  Local space range: min={xyz_local.min(0)[0].tolist()}, max={xyz_local.max(0)[0].tolist()}")
    
    # Save local space point cloud
    save_pointcloud(xyz_local, os.path.join(args.output_dir, "01_local_space.ply"), [255, 0, 0])
    
    # ---- 4. Transform to world space ----
    print("\nStep 3: Transforming to world space...")
    xyz_camera = sam3d.local_to_camera(xyz_local, pose_7d=None)
    xyz_world = sam3d.camera_to_world(xyz_camera, cam_R, cam_T)
    print(f"  World space range: min={xyz_world.min(0)[0].tolist()}, max={xyz_world.max(0)[0].tolist()}")
    
    save_pointcloud(xyz_world, os.path.join(args.output_dir, "02_world_space.ply"), [0, 0, 255])
    
    # ---- 5. Voxelize ----
    print("\nStep 4: Voxelizing...")
    vox_result = voxelize_points(
        xyz_world,
        grid_size=args.grid_size,
        density_min_th=args.density_min_th,
        density_max_th=args.density_max_th,
        is_first_frame=True,
    )
    
    vol = vox_result['vol']
    surface = vox_result['surface_pts']
    vol_surface = vox_result['vol_surface']
    
    print(f"  Interior particles (vol): {vol.shape[0]}")
    print(f"  Surface points: {surface.shape[0]}")
    print(f"  Surface particles in vol: {vol_surface.shape[0]}")
    
    save_pointcloud(vol, os.path.join(args.output_dir, "03_vol_interior.ply"), [0, 255, 0])
    save_pointcloud(surface, os.path.join(args.output_dir, "04_surface.ply"), [255, 255, 0])
    
    # ---- 6. Also save the original SAM3D splat for comparison ----
    print("\nStep 5: Saving SAM3D splat...")
    gs = output["gs"]
    splat_path = os.path.join(args.output_dir, "05_sam3d_splat.ply")
    gs.save_ply(splat_path)
    print(f"  Saved to {splat_path}")
    
    # ---- Summary ----
    print("\n" + "=" * 60)
    print("DONE. Check these files in a 3D viewer (e.g., MeshLab):")
    print(f"  {args.output_dir}/01_local_space.ply   - SAM3D output (local)")
    print(f"  {args.output_dir}/02_world_space.ply   - After cam2world transform")
    print(f"  {args.output_dir}/03_vol_interior.ply  - Voxelized interior (MPM particles)")
    print(f"  {args.output_dir}/04_surface.ply       - Extracted surface (gts)")
    print(f"  {args.output_dir}/05_sam3d_splat.ply   - Original SAM3D gaussian splat")
    print()
    print("What to verify:")
    print("  1. 01 -> 02: Does the coordinate transform look correct?")
    print("     (object should be in a reasonable world-space position)")
    print("  2. 02 -> 03: Is the interior properly filled?")
    print("     (03 should be a solid version of 02)")
    print("  3. 04: Are surface points on the actual surface?")
    print("     (should outline the shape of the object)")
    print("=" * 60)


if __name__ == "__main__":
    main()