"""
MV-SAM3D + GIC Physical Parameter Estimation Pipeline

Workflow:
  1. Load MV-SAM3D reconstruction results (pre-computed)
  2. Apply correct coordinate transform: canonical -> camera -> world
     - Scale correction using known camera distance
     - PyTorch3D -> OpenCV flip
     - GIC cam2world
  3. Voxelize world-space points
  4. Run GIC velocity + physical parameter estimation
  5. Render trajectory with MV-SAM3D gaussians

Coordinate transform chain (verified by diagnose_cam2world.py):
  canonical [-0.5, 0.5]
    -> Z-up to Y-up
    -> pose (scale * R + translation) -> PyTorch3D camera space
    -> scale_correction = ||cam_T|| / translation_z
    -> flip x, y  (PyTorch3D -> OpenCV)
    -> GIC cam2world: (xyz - T) @ R.T

Usage:
  # Step 1: Prepare data (login node, no GPU)
  python prepare_mvsam3d_data.py

  # Step 2: Run MV-SAM3D for all frames (GPU node)
  cd /home/fding/MV-SAM3D
  bash /orcd/home/002/fding/gic/run_mvsam3d_all_frames.sh

  # Step 3: Run this script (GPU node)
  python train_dynamic_mvsam3d.py \
      -c config/pacnerf/torus.json \
      -s data/pacnerf/torus \
      -m output/mvsam3d_torus \
      --mvsam3d_dir /home/fding/MV-SAM3D/visualization \
      --camera_json data/mvsam3d_torus/cameras.json \
      --config_file config/pacnerf/torus.json
"""

import os
import sys
import json
import glob
import time
import math
import torch
import torch.nn as nn
import numpy as np
import taichi as ti
import trimesh
import torchvision
import imageio
from tqdm import tqdm, trange
from pathlib import Path
from argparse import ArgumentParser, Namespace

# GIC imports
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readPACNeRFInfo
from scene.cameras import Camera
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from simulator import Simulator
from simulator.estimator import Estimator
from train_dynamic import train, forward, backward, export_result
from train_gs_fixed_pcd import assign_gs_to_pcd
from utils.system_utils import write_particles

# PyTorch3D for quaternion
from pytorch3d.transforms import quaternion_to_matrix


# ============================================================================
# Coordinate Transforms
# ============================================================================

def mvsam3d_to_world(
    xyz_canonical: torch.Tensor,
    pose: dict,
    cam_R: np.ndarray,
    cam_T: np.ndarray,
) -> torch.Tensor:
    """
    Transform points from MV-SAM3D canonical space to GIC world space.

    Uses per-frame scale correction for correct center position.

    Args:
        xyz_canonical: (N, 3) points in canonical space [-0.5, 0.5]
        pose: dict with 'scale' (3,), 'rotation' (4,) wxyz, 'translation' (3,)
        cam_R: (3, 3) GIC camera rotation (transposed convention)
        cam_T: (3,) GIC camera translation
    Returns:
        xyz_world: (N, 3) points in world space
    """
    device = xyz_canonical.device

    # Parse pose
    scale = torch.tensor(pose['scale'], dtype=torch.float32, device=device).flatten()
    rot_quat = torch.tensor(pose['rotation'], dtype=torch.float32, device=device).flatten()
    translation = torch.tensor(pose['translation'], dtype=torch.float32, device=device).flatten()

    # Rotation matrix from quaternion (wxyz)
    R_pose = quaternion_to_matrix(rot_quat.unsqueeze(0)).squeeze(0)  # (3, 3)

    # Step 1: Z-up to Y-up
    Z_UP_TO_Y_UP = torch.tensor(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32, device=device
    )
    xyz_yup = xyz_canonical @ Z_UP_TO_Y_UP.T

    # Step 2: Apply pose -> PyTorch3D camera space
    xyz_scaled = xyz_yup * scale.unsqueeze(0)
    xyz_rotated = xyz_scaled @ R_pose.T
    xyz_p3d = xyz_rotated + translation.unsqueeze(0)

    # Step 3: Per-frame scale correction (correct center position)
    cam_dist = float(np.linalg.norm(cam_T))
    depth_z = translation[2].item()
    if abs(depth_z) > 0.01:
        scale_correction = cam_dist / depth_z
    else:
        scale_correction = 1.0
    xyz_scaled_p3d = xyz_p3d * scale_correction

    # Step 4: PyTorch3D -> OpenCV (flip x, y)
    flip = torch.tensor([-1, -1, 1], dtype=torch.float32, device=device)
    xyz_cv = xyz_scaled_p3d * flip.unsqueeze(0)

    # Step 5: GIC cam2world: pw = (pc - T) @ R.T
    R_cam = torch.tensor(cam_R, dtype=torch.float32, device=device)
    T_cam = torch.tensor(cam_T, dtype=torch.float32, device=device)
    xyz_world = (xyz_cv - T_cam.unsqueeze(0)) @ R_cam.T

    return xyz_world


# ============================================================================
# Load MV-SAM3D Results
# ============================================================================

def find_mvsam3d_output(mvsam3d_dir: str, frame_name: str) -> dict:
    """Find MV-SAM3D output files for a given frame."""
    pattern = os.path.join(mvsam3d_dir, frame_name, "torus", "*", "params.npz")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No MV-SAM3D output found: {pattern}")

    # Use the latest run
    params_path = matches[-1]
    output_dir = os.path.dirname(params_path)

    return {
        'params_path': params_path,
        'ply_path': os.path.join(output_dir, 'result.ply'),
        'glb_path': os.path.join(output_dir, 'result.glb'),
        'output_dir': output_dir,
    }


def load_mvsam3d_pose(params_path: str) -> dict:
    """Load MV-SAM3D pose parameters."""
    data = np.load(params_path)
    return {
        'scale': data['scale'].flatten(),
        'rotation': data['rotation'].flatten(),
        'translation': data['translation'].flatten(),
    }


def load_mvsam3d_gaussians(ply_path: str, device='cuda') -> GaussianModel:
    """Load MV-SAM3D gaussian splat as GaussianModel."""
    gs = GaussianModel(0)
    gs.load_ply(ply_path)
    return gs


# ============================================================================
# Voxelization (same as train_dynamic_sam3d.py)
# ============================================================================

def voxelize_points(
    xyzt: torch.Tensor,
    grid_size: float = 0.02,
    density_min_th: float = 0.05,
    density_max_th: float = 0.5,
    is_first_frame: bool = False,
) -> dict:
    """Voxelize point cloud and extract interior/surface points."""
    device = xyzt.device
    result = {}

    bbox_mins = xyzt.min(dim=0)[0]
    bbox_maxs = xyzt.max(dim=0)[0]
    bbox_bounds = bbox_maxs - bbox_mins
    curr_grid_size = grid_size

    volume_size = ((bbox_bounds / curr_grid_size).int() + 1).tolist()
    volume_size = [max(int(v), 1) for v in volume_size]

    # Build density volume
    density_volume = torch.zeros(volume_size, device=device)
    grid_indices = ((xyzt - bbox_mins) / curr_grid_size).long()
    max_idx = torch.tensor([v - 1 for v in volume_size], device=device).long()
    grid_indices = torch.clamp(grid_indices, min=0)
    grid_indices = torch.min(grid_indices, max_idx.unsqueeze(0))

    ones = torch.ones(grid_indices.shape[0], device=device)
    density_volume.index_put_(
        (grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]),
        ones, accumulate=True,
    )

    max_density = density_volume.max()
    if max_density > 0:
        density_volume = density_volume / max_density

    # Sample particles on half-grid
    half_grid_xyz = torch.stack(torch.meshgrid(
        torch.linspace(0, volume_size[0] - 0.5, 2 * max(volume_size[0] - 1, 1)),
        torch.linspace(0, volume_size[1] - 0.5, 2 * max(volume_size[1] - 1, 1)),
        torch.linspace(0, volume_size[2] - 0.5, 2 * max(volume_size[2] - 1, 1)),
    ), -1).to(bbox_mins) * curr_grid_size + bbox_mins[None, None, None]

    ids_norm = (half_grid_xyz - bbox_mins[None, None, None]) / bbox_bounds[None, None, None] * 2 - 1
    ids_norm = ids_norm[None].flip((-1,))
    density_half = torch.nn.functional.grid_sample(
        density_volume[None, None], ids_norm, mode='bilinear', align_corners=True
    )[0, 0]

    valid_pts = half_grid_xyz[density_half > density_min_th]
    if len(valid_pts) == 0:
        # Fallback: use even lower threshold
        valid_pts = half_grid_xyz[density_half > density_min_th * 0.1]

    delta = (torch.rand_like(valid_pts) * curr_grid_size * 0.5).to(xyzt)
    particles = valid_pts + delta

    # Sample density at particle locations
    ids_norm_p = (particles[None, None] - bbox_mins[None, None, None]) / bbox_bounds[None, None, None] * 2 - 1
    ids_norm_p = ids_norm_p[None].flip((-1,))
    density_particles = torch.nn.functional.grid_sample(
        density_volume[None, None], ids_norm_p, mode='bilinear', align_corners=True
    )[0, 0, 0, 0]

    # Surface and interior
    surface_mask = (density_particles > density_min_th) & (density_particles < density_max_th)
    result['surface_pts'] = particles[surface_mask]

    if is_first_frame:
        interior_mask = density_particles > density_min_th
        vol = particles[interior_mask]
        vol_densities = density_particles[interior_mask]
        vol_surface_mask = density_particles[interior_mask] < density_max_th
        vol_surface = torch.arange(vol_surface_mask.shape[0]).to(vol.device).to(torch.int64)[vol_surface_mask]
        result['vol'] = vol
        result['vol_densities'] = vol_densities
        result['vol_surface'] = vol_surface

    return result


# ============================================================================
# Prepare GT from MV-SAM3D results
# ============================================================================

def prepare_gt_mvsam3d(
    mvsam3d_dir: str,
    camera_json: str,
    phys_args,
    output_path: str = None,
) -> tuple:
    """
    Load MV-SAM3D results and convert to world-space GTs for GIC.

    Returns:
        (gts, vol, vol_densities, grid_size, volume_surface, cam_info, frame0_data)
    """
    grid_size = phys_args.density_grid_size
    density_min_th = phys_args.density_min_th
    density_max_th = phys_args.density_max_th

    # Load camera data
    with open(camera_json) as f:
        cam_data = json.load(f)

    n_frames = getattr(phys_args, 'n_frames', None) or len(cam_data)
    frame_names = sorted(cam_data.keys())[:n_frames]

    gts = []
    vol = None
    vol_densities = None
    vol_surface = None
    frame0_data = {}
    frame0_bbox_size = None  # Record frame 0's bounding box size

    for idx, frame_name in enumerate(tqdm(frame_names, desc="Loading MV-SAM3D results")):
        # Find MV-SAM3D output
        try:
            mvsam3d_out = find_mvsam3d_output(mvsam3d_dir, frame_name)
        except FileNotFoundError:
            print(f"  [SKIP] {frame_name}: no MV-SAM3D output")
            continue

        # Load pose
        pose = load_mvsam3d_pose(mvsam3d_out['params_path'])

        # Load gaussian splat (only need xyz for voxelization)
        gs = load_mvsam3d_gaussians(mvsam3d_out['ply_path'])
        xyz_canonical = gs.get_xyz.detach()  # (N, 3) in [-0.5, 0.5]

        # Get view 0 camera params
        cam0 = cam_data[frame_name]['cameras'][0]
        cam_R = np.array(cam0['R'])
        cam_T = np.array(cam0['T'])

        # Transform to world space (per-frame scale correction for correct position)
        xyz_world = mvsam3d_to_world(xyz_canonical, pose, cam_R, cam_T)

        # Normalize size to match frame 0
        # Per-frame sc gives correct center but inconsistent size
        # Fix: rescale each frame's point cloud around its center to match frame 0's bbox size
        center = xyz_world.mean(dim=0)
        bbox_size = xyz_world.max(dim=0)[0] - xyz_world.min(dim=0)[0]  # (3,)

        if frame0_bbox_size is None:
            # Record frame 0's size as reference
            frame0_bbox_size = bbox_size.clone()
            print(f"  Frame 0 bbox size (reference): {frame0_bbox_size.tolist()}")
        else:
            # Rescale this frame to match frame 0's size, keeping center unchanged
            size_ratio = frame0_bbox_size / bbox_size.clamp(min=1e-6)  # (3,)
            xyz_world = (xyz_world - center) * size_ratio + center
            print(f"  Size ratio: {size_ratio.tolist()}")

        print(f"  {frame_name}: {xyz_world.shape[0]} pts, "
              f"center=[{center[0]:.3f},{center[1]:.3f},{center[2]:.3f}], "
              f"range [{xyz_world.min(0)[0].tolist()}] ~ [{xyz_world.max(0)[0].tolist()}]")

        # Voxelize
        is_first = (idx == 0)
        vox_result = voxelize_points(
            xyz_world, grid_size, density_min_th, density_max_th,
            is_first_frame=is_first,
        )

        gts.append(vox_result['surface_pts'])

        if is_first:
            vol = vox_result['vol']
            vol_densities = vox_result['vol_densities']
            vol_surface = vox_result['vol_surface']

            if output_path:
                os.makedirs(os.path.join(output_path, 'mpm'), exist_ok=True)
                write_particles(vol, 0, output_path, 'static')

            # Save frame 0 data for rendering later
            frame0_data = {
                'ply_path': mvsam3d_out['ply_path'],
                'pose': pose,
                'cam_R': cam_R,
                'cam_T': cam_T,
                'xyz_world': xyz_world.cpu(),
            }

            print(f"  MPM particles: {vol.shape[0]}, surface: {vol_surface.shape[0]}")

    grid_size_tensor = torch.tensor([grid_size])
    return gts, vol, vol_densities, grid_size_tensor, vol_surface, cam_data, frame0_data


# ============================================================================
# Load cameras for estimator
# ============================================================================

def load_cameras_for_estimator(data_path, config_file, phys_args):
    """Load GIC cameras for image-based supervision in estimator."""
    from torchvision.transforms import ToTensor
    scene_info = readPACNeRFInfo(data_path, config_file, True)

    train_cams = []
    for ci in scene_info.train_cameras:
        img_tensor = ToTensor()(ci.image)
        alpha = np.array(ci.alpha) if hasattr(ci, 'alpha') and ci.alpha is not None else None
        if alpha is not None:
            if alpha.ndim == 2:
                alpha = alpha[np.newaxis, :, :]
            elif alpha.ndim == 3 and alpha.shape[2] <= 4:
                alpha = alpha.transpose(2, 0, 1)
            if alpha.shape[0] > 1:
                alpha = alpha[:1]

        cam = Camera(
            colmap_id=ci.uid, R=ci.R, T=ci.T,
            FoVx=ci.FovX, FoVy=ci.FovY,
            image=img_tensor, gt_alpha_mask=alpha,
            image_name=ci.image_name, uid=ci.uid, fid=ci.fid,
        )
        train_cams.append(cam)

    return {
        'train_cams': train_cams,
        'test_cams': [],
        'cameras_extent': scene_info.nerf_normalization['radius'],
    }


# ============================================================================
# Rendering with MV-SAM3D gaussians
# ============================================================================

def render_trajectory(
    output_dir: str,
    frame0_data: dict,
    phys_args,
    estimation_result: dict,
    data_path: str,
    config_file: str,
):
    """Render physics simulation using MV-SAM3D frame 0 gaussians."""
    render_dir = os.path.join(output_dir, 'render_mvsam3d')
    os.makedirs(render_dir, exist_ok=True)

    # Load frame 0 gaussians
    gs = load_mvsam3d_gaussians(frame0_data['ply_path'])
    xyz_canonical = gs.get_xyz.detach()
    pose = frame0_data['pose']
    cam_R = frame0_data['cam_R']
    cam_T = frame0_data['cam_T']

    # Transform gaussians to world space
    xyz_world = mvsam3d_to_world(xyz_canonical, pose, cam_R, cam_T)
    gs._xyz = nn.Parameter(xyz_world)
    print(f"  Gaussians: {gs.get_xyz.shape[0]}")
    print(f"  World range: {gs.get_xyz.min(0)[0].tolist()} ~ {gs.get_xyz.max(0)[0].tolist()}")

    # Load MPM particles
    vol_path = os.path.join(output_dir, 'mpm', 'static_0.ply')
    pcd = trimesh.load_mesh(vol_path)
    vol = torch.from_numpy(np.array(pcd.vertices)).to('cuda', dtype=torch.float32)

    # Build NN mapping: for each gaussian, find nearest MPM particle
    print(f"  Building NN mapping...")
    batch_size = 10000
    n_gs = gs.get_xyz.shape[0]
    nn_indices = torch.zeros(n_gs, dtype=torch.long, device='cuda')
    for i in range(0, n_gs, batch_size):
        end = min(i + batch_size, n_gs)
        dists = torch.cdist(gs.get_xyz[i:end].unsqueeze(0), vol.unsqueeze(0)).squeeze(0)
        nn_indices[i:end] = dists.argmin(dim=1)

    # Run physics simulation
    predict_frames = getattr(phys_args, 'predict_frames', 48)
    est_params = Namespace(**estimation_result)
    simulator = Simulator(est_params, vol)
    xyz0 = vol.detach()

    print(f"  Simulating {predict_frames} frames...")
    simulator.initialize()
    d_xyz_list = []
    for f in tqdm(range(predict_frames), desc="Simulating"):
        xyz = simulator.forward(f)
        if isinstance(xyz, np.ndarray):
            xyz = torch.from_numpy(xyz).float().to('cuda')
        d_xyz_list.append((xyz - xyz0).detach())

    # Get camera for rendering
    scene_info = readPACNeRFInfo(data_path, config_file, True)
    ci = scene_info.train_cameras[0]
    world_view = torch.tensor(
        getWorld2View2(ci.R, ci.T, np.array([0.0, 0.0, 0.0]), 1.0)
    ).transpose(0, 1).float().cuda()
    projection = getProjectionMatrix(
        znear=0.01, zfar=100.0, fovX=ci.FovX, fovY=ci.FovY
    ).transpose(0, 1).float().cuda()
    full_proj = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    camera_center = world_view.inverse()[3, :3]
    bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    # Render frames
    frames_for_video = []
    print(f"  Rendering {predict_frames} frames...")
    with torch.no_grad():
        for f in tqdm(range(predict_frames), desc="Rendering"):
            gs_d_xyz = d_xyz_list[f][nn_indices]

            tanfovx = math.tan(ci.FovX * 0.5)
            tanfovy = math.tan(ci.FovY * 0.5)
            settings = GaussianRasterizationSettings(
                image_height=ci.height, image_width=ci.width,
                tanfovx=tanfovx, tanfovy=tanfovy,
                bg=bg, scale_modifier=1.0,
                viewmatrix=world_view, projmatrix=full_proj,
                sh_degree=gs.active_sh_degree,
                campos=camera_center,
                prefiltered=False, debug=False,
            )
            rasterizer = GaussianRasterizer(raster_settings=settings)

            means3D = gs.get_xyz + gs_d_xyz
            means2D = torch.zeros_like(means3D)
            opacity = gs.get_opacity
            scales = gs.get_scaling.repeat(1, 3)
            rotations = gs.get_rotation
            shs = gs.get_features

            image, depth, alpha, radii = rasterizer(
                means3D=means3D, means2D=means2D,
                shs=shs, colors_precomp=None,
                opacities=opacity, scales=scales,
                rotations=rotations, cov3D_precomp=None,
            )
            image = torch.clamp(image, 0.0, 1.0)
            torchvision.utils.save_image(image, os.path.join(render_dir, f'{f:05d}.png'))
            frame_np = (image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frames_for_video.append(frame_np)

    # Save video
    video_path = os.path.join(render_dir, 'output.mp4')
    imageio.mimwrite(video_path, frames_for_video, fps=phys_args.fps, quality=8)
    print(f"  Video saved to {video_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    start_time = time.time()

    parser = ArgumentParser(description="MV-SAM3D + GIC Pipeline")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--config_file", default='config/pacnerf/torus.json', type=str)
    parser.add_argument("--mvsam3d_dir", required=True, type=str,
                        help="Path to MV-SAM3D visualization directory")
    parser.add_argument("--camera_json", required=True, type=str,
                        help="Path to cameras.json from prepare_mvsam3d_data.py")
    parser.add_argument("--predict_config", default='config/predict/elastic.json', type=str,
                        help="Config for trajectory prediction (optional)")
    parser.add_argument("--skip_training", action='store_true',
                        help="Skip training, only render trajectory with existing results")

    gs_args, phys_args = get_combined_args(parser)
    config_id = phys_args.id
    print(phys_args)
    safe_state(gs_args.quiet)

    dataset = model.extract(gs_args)
    os.makedirs(dataset.model_path, exist_ok=True)
    os.makedirs(os.path.join(dataset.model_path, 'mpm'), exist_ok=True)
    os.makedirs(os.path.join(dataset.model_path, 'gs'), exist_ok=True)
    os.makedirs(os.path.join(dataset.model_path, 'img'), exist_ok=True)
    os.makedirs(os.path.join(dataset.model_path, 'render'), exist_ok=True)

    # Create cfg_args
    cfg_args_path = os.path.join(dataset.model_path, 'cfg_args')
    if not os.path.exists(cfg_args_path):
        from argparse import Namespace as NS
        cfg = NS(
            sh_degree=3, source_path=dataset.source_path, model_path=dataset.model_path,
            config_path=gs_args.config_file, images='images', resolution=-1,
            white_background=True, data_device='cuda', eval=False, num_frame=-1,
            load2gpu_on_the_fly=False, is_blender=False, is_6dof=False, res_scale=1.0,
        )
        with open(cfg_args_path, 'w') as f:
            f.write(str(cfg))

    image_scale = 1.0

    if not gs_args.skip_training:
        # ---- Step 1: Load MV-SAM3D results and convert to world space ----
        print("=" * 60)
        print("Step 1: Load MV-SAM3D Results + World Space Transform")
        print("=" * 60)

        gts, vol, vol_densities, grid_size, volume_surface, cam_data, frame0_data = \
            prepare_gt_mvsam3d(
                gs_args.mvsam3d_dir,
                gs_args.camera_json,
                phys_args,
                output_path=dataset.model_path,
            )

        # Save frame0 data for rendering
        torch.save(frame0_data, os.path.join(dataset.model_path, 'frame0_data.pt'))

        print(f"\nTotal frames: {len(gts)}")
        print(f"MPM particles: {vol.shape[0]}")
        print(f"GT surface points per frame: {[g.shape[0] for g in gts]}")

        # ---- Step 2: Velocity estimation ----
        print("=" * 60)
        print("Step 2: Velocity Estimation")
        print("=" * 60)

        ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)

        estimator = Estimator(
            phys_args, 'float32', gts,
            surface_index=volume_surface,
            init_vol=vol,
            dynamic_scene=None,
            image_scale=image_scale,
            pipeline=pipeline.extract(gs_args),
            image_op=op.extract(gs_args),
        )
        estimator.set_stage(Estimator.velocity_stage)
        losses, e_s = train(estimator, phys_args, phys_args.vel_estimation_frames)
        torch.cuda.empty_cache()

        # ---- Step 3: Physical parameter estimation ----
        print("=" * 60)
        print("Step 3: Physical Parameter Estimation")
        print("=" * 60)

        # Disable image/alpha loss — assign_gs_to_pcd renders are unreliable
        # with MV-SAM3D pipeline, use geometry loss only
        phys_args.img_loss = False
        phys_args.w_img = 0.0
        phys_args.w_alp = 0.0
        print("  [Override] Using geometry loss only (img_loss disabled)")

        scene = assign_gs_to_pcd(
            vol, vol_densities, dataset,
            op.extract(gs_args),
            pipeline.extract(gs_args),
            load_cameras_for_estimator(dataset.source_path, gs_args.config_file, phys_args),
            phys_args.density_grid_size,
        )

        if isinstance(scene.train_cameras, list):
            scene.train_cameras = {1.0: scene.train_cameras}
        if isinstance(scene.test_cameras, list):
            scene.test_cameras = {1.0: scene.test_cameras}

        estimator.set_scene(scene)
        max_f = len(gts)
        estimator.set_stage(Estimator.physical_params_stage)
        losses, e_s = train(estimator, phys_args, max_f)

        # ---- Export results ----
        print("=" * 60)
        print("Results")
        print("=" * 60)
        print(phys_args)
        print(f"Estimated velocity: {estimator.init_vel}")
        export_result(dataset, phys_args, estimator, losses, e_s, config_id)

    # ---- Step 4: Render trajectory ----
    print("=" * 60)
    print("Step 4: Render Trajectory")
    print("=" * 60)

    # Load estimation result
    pred_path = os.path.join(dataset.model_path, f'{config_id}-pred.json')
    with open(pred_path) as f:
        estimation_result = json.load(f)
    print(f"Loaded estimation: {pred_path}")

    # Load frame0 data
    frame0_data_path = os.path.join(dataset.model_path, 'frame0_data.pt')
    frame0_data = torch.load(frame0_data_path, weights_only=False)

    # Load predict config for rendering params
    predict_config_path = gs_args.predict_config
    if os.path.exists(predict_config_path):
        with open(predict_config_path) as f:
            pred_cfg = json.load(f)
        predict_frames = pred_cfg.get('physics', {}).get('predict_frames', 48)
    else:
        predict_frames = 48

    # Override estimation result with predict config params if desired
    # (for testing with hand-tuned params)
    estimation_result['predict_frames'] = predict_frames

    render_trajectory(
        output_dir=dataset.model_path,
        frame0_data=frame0_data,
        phys_args=phys_args,
        estimation_result=estimation_result,
        data_path=dataset.source_path,
        config_file=gs_args.config_file,
    )

    print(f"\nTotal time: {time.time() - start_time:.1f}s")