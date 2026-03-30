# train_dynamic_sam3d.py
# SAM3D + GIC pipeline: replaces train_dynamic.py
# Differences from original:
#   - No Deformable 3DGS training (step 1 removed)
#   - prepare_gt replaced with prepare_gt_sam3d (uses SAM3D point clouds)
#   - Everything else (velocity estimation, physical param estimation) is the same

import torch
import taichi as ti
import torch.nn as nn
import time, os, json
import numpy as np
from tqdm import tqdm, trange
from argparse import ArgumentParser, Namespace
from pathlib import Path

from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from simulator import MPMSimulator, Estimator
from train_gs_fixed_pcd import train_gs_with_fixed_pcd, assign_gs_to_pcd
from utils.system_utils import check_gs_model, draw_curve, write_particles
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

from sam3d_wrapper import SAM3DWrapper

image_scale = 1.0


# ============================================================================
# Data loading - ADAPT THIS TO YOUR DATA FORMAT
# ============================================================================

def load_multiview_data(data_path: str, phys_args) -> dict:
    """
    Load multi-view video data. 
    
    *** ADAPT THIS FUNCTION TO YOUR DATA FORMAT ***
    
    Expected return format:
    {
        'frames': [
            {
                'fid': 0,  # frame index
                'views': [
                    {
                        'image': np.ndarray (H, W, 4) RGBA uint8,
                        'mask': np.ndarray (H, W) binary,
                        'cam_R': np.ndarray (3, 3),
                        'cam_T': np.ndarray (3,),
                        'FovX': float,
                        'FovY': float,
                        'width': int,
                        'height': int,
                        'image_path': str,
                    },
                    ...  # multiple views per frame
                ]
            },
            ...  # multiple frames
        ],
        'fps': float,
    }
    """
    # ---- PLACEHOLDER: replace with your actual data loading ----
    # Example for PAC-NeRF style data:
    from scene.dataset_readers import readPACNeRFInfo, CameraInfo
    from PIL import Image
    
    data_path = Path(data_path)
    frames = []
    
    # TODO: implement for your specific data format
    # This is a skeleton showing the expected structure
    raise NotImplementedError(
        "load_multiview_data() needs to be implemented for your data format.\n"
        "See the docstring for the expected return format.\n"
        "Each frame needs: RGBA images, masks, camera R/T for at least one view."
    )
    
    return {
        'frames': frames,
        'fps': getattr(phys_args, 'fps', 24),
    }


def load_multiview_data_from_gic_dataset(dataset: ModelParams, pipeline, phys_args, config_file: str = None) -> dict:
    """
    Load data from GIC's existing dataset format (PAC-NeRF / Spring-Gaus).
    Directly reads cameras and images from dataset readers, bypassing Scene.
    """
    from scene.dataset_readers import (
        readPACNeRFInfo, readNerfSyntheticInfo, 
        readSpringGausMPMSyntheticInfo, readSpringGausCaptureRealInfo,
        getNerfppNorm, CameraInfo,
    )
    from scene.cameras import Camera
    from PIL import Image
    
    # Detect dataset type and load scene info (same logic as Scene.__init__)
    source_path = dataset.source_path
    config_path = getattr(dataset, 'config_path', None)
    
    if os.path.exists(os.path.join(source_path, "all_data.json")):
        print("Found all_data.json, assuming PAC-NeRF data set!")
        cfg_path = config_file or getattr(dataset, 'config_path', None)
        if cfg_path is None:
            raise ValueError("PAC-NeRF dataset requires config file path. Pass -c config/pacnerf/torus.json")
        scene_info = readPACNeRFInfo(
            source_path, cfg_path,
            dataset.white_background
        )
    elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
        print("Found transforms_train.json, assuming Blender data set!")
        scene_info = readNerfSyntheticInfo(
            source_path, dataset.white_background, dataset.eval
        )
    else:
        raise ValueError(f"Cannot detect dataset type at {source_path}")
    
    # Build Camera objects from CameraInfo
    cam_infos = scene_info.train_cameras
    cameras_extent = scene_info.nerf_normalization["radius"]
    
    # Convert CameraInfo to Camera objects for later use
    def cam_info_to_camera(ci):
        from utils.graphics_utils import getWorld2View2, getProjectionMatrix
        from torchvision.transforms import ToTensor
        img_tensor = ToTensor()(ci.image)
        alpha = None
        if hasattr(ci, 'alpha') and ci.alpha is not None:
            alpha = np.array(ci.alpha)
            if alpha.ndim == 2:
                alpha = alpha[np.newaxis, :, :]  # (1, H, W)
            elif alpha.ndim == 3 and alpha.shape[2] <= 4:
                alpha = alpha.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
            # Ensure single channel
            if alpha.shape[0] > 1:
                alpha = alpha[:1]
        return Camera(
            colmap_id=ci.uid, R=ci.R, T=ci.T,
            FoVx=ci.FovX, FoVy=ci.FovY,
            image=img_tensor, gt_alpha_mask=alpha,
            image_name=ci.image_name, uid=ci.uid,
            fid=ci.fid,
        )
    
    train_cams = [cam_info_to_camera(ci) for ci in cam_infos]
    test_cams = [cam_info_to_camera(ci) for ci in scene_info.test_cameras] if scene_info.test_cameras else []
    
    # Group by frame id
    fids = sorted(set([ci.fid for ci in cam_infos]))
    
    frames = []
    for fid in fids:
        curr_cams = [ci for ci in cam_infos if ci.fid == fid]
        view_data = []
        for ci in curr_cams:
            # Get image as numpy
            img_np = np.array(ci.image)  # (H, W, 3) RGB uint8
            
            # Get alpha mask
            if hasattr(ci, 'alpha') and ci.alpha is not None:
                alpha = np.array(ci.alpha)
                if alpha.ndim == 2:
                    alpha = alpha[:, :, np.newaxis]
                if alpha.max() <= 1.0:
                    alpha = (alpha * 255).astype(np.uint8)
                else:
                    alpha = alpha.astype(np.uint8)
            else:
                # No mask - create from non-black pixels
                alpha = ((img_np.sum(-1, keepdims=True) > 0) * 255).astype(np.uint8)
            
            # Build RGBA
            rgba = np.concatenate([img_np, alpha], axis=-1)  # (H, W, 4)
            mask = (alpha.squeeze(-1) > 127).astype(np.uint8)
            
            view_data.append({
                'image': rgba,
                'mask': mask,
                'cam_R': ci.R,
                'cam_T': ci.T,
                'FovX': ci.FovX,
                'FovY': ci.FovY,
                'width': ci.width,
                'height': ci.height,
                'uid': ci.uid,
            })
        
        frames.append({
            'fid': fid,
            'views': view_data,
        })
    
    cam_info = {
        "train_cams": train_cams,
        "test_cams": test_cams,
        "cameras_extent": cameras_extent,
    }
    
    return {
        'frames': frames,
        'fps': getattr(phys_args, 'fps', 24),
        'cam_info': cam_info,
    }


# ============================================================================
# Core: prepare_gt using SAM3D (replaces original prepare_gt)
# ============================================================================

def voxelize_points(
    xyzt: torch.Tensor,
    grid_size: float,
    density_min_th: float,
    density_max_th: float,
    is_first_frame: bool = False,
) -> dict:
    """
    Voxelize a point cloud into density field, extract interior (vol) and surface (gts).
    This is the core voxelization logic extracted from the original prepare_gt.
    
    Args:
        xyzt: (N, 3) point cloud in world space
        grid_size: voxel grid size
        density_min_th: minimum density threshold for interior points
        density_max_th: maximum density threshold for surface points  
        is_first_frame: if True, also extract vol (MPM particles)
    
    Returns:
        dict with 'surface_pts', and optionally 'vol', 'vol_densities', 'vol_surface'
    """
    result = {}
    
    # Compute bounding box
    bbox_mins = xyzt.min(dim=0)[0] - grid_size
    bbox_maxs = xyzt.max(dim=0)[0] + grid_size
    bbox_bounds = bbox_maxs - bbox_mins
    
    # Build density volume at progressively finer resolutions
    curr_grid_size = grid_size
    volume_size = torch.round(bbox_bounds / curr_grid_size).to(torch.int64) + 1
    bbox_maxs = bbox_mins + (volume_size - 1) * curr_grid_size
    bbox_bounds = bbox_maxs - bbox_mins
    
    # Initialize density volume from points
    density_volume = torch.zeros(volume_size.cpu().numpy().tolist()).to(xyzt)
    ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
    # Clamp indices to valid range
    for d in range(3):
        ids[:, d] = ids[:, d].clamp(0, volume_size[d].item() - 1)
    density_volume[ids[:, 0], ids[:, 1], ids[:, 2]] = 1.0
    
    # Smoothing kernel
    weight = torch.ones((1, 1, 3, 3, 3)).to(xyzt)
    weight = weight / weight.sum()
    
    # Iterative refinement (smooth + threshold)
    for _ in range(20):
        density_volume = torch.nn.functional.conv3d(
            density_volume[None, None], weight=weight, padding='same'
        )[0, 0]
        density_volume[density_volume < 0.5] = 0.0
        # Re-insert original points to maintain structure
        density_volume[ids[:, 0], ids[:, 1], ids[:, 2]] = 1.0
    
    # Final smoothing pass
    density_volume = torch.nn.functional.conv3d(
        density_volume[None, None], weight=weight, padding='same'
    )[0, 0]
    
    # Sample at half-grid resolution for finer extraction
    half_grid_xyz = torch.stack(torch.meshgrid(
        torch.linspace(0, volume_size[0] - 0.5, 2 * (volume_size[0] - 1)),
        torch.linspace(0, volume_size[1] - 0.5, 2 * (volume_size[1] - 1)),
        torch.linspace(0, volume_size[2] - 0.5, 2 * (volume_size[2] - 1)),
    ), -1).to(bbox_mins) * curr_grid_size + bbox_mins[None, None, None]
    
    ids_norm = (half_grid_xyz - bbox_mins[None, None, None]) / bbox_bounds[None, None, None] * 2 - 1
    ids_norm = ids_norm[None].flip((-1,))
    density_half = torch.nn.functional.grid_sample(
        density_volume[None, None], ids_norm, mode='bilinear', align_corners=True
    )[0, 0]
    
    # Add random jitter for uniform sampling
    valid_pts = half_grid_xyz[density_half > 0.5]
    delta = (torch.rand_like(valid_pts) * curr_grid_size * 0.5).to(xyzt)
    particles = valid_pts + delta
    
    # Sample density at particle locations
    ids_norm_p = (particles[None, None] - bbox_mins[None, None, None]) / bbox_bounds[None, None, None] * 2 - 1
    ids_norm_p = ids_norm_p[None].flip((-1,))
    density_particles = torch.nn.functional.grid_sample(
        density_volume[None, None], ids_norm_p, mode='bilinear', align_corners=True
    )[0, 0, 0, 0]
    
    # Extract surface points (between min and max density)
    surface_mask = (density_particles > density_min_th) & (density_particles < density_max_th)
    surface_pts = particles[surface_mask]
    result['surface_pts'] = surface_pts
    
    # Extract interior points (vol) only for first frame
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


def prepare_gt_sam3d(
    sam3d: SAM3DWrapper,
    data: dict,
    phys_args,
    output_path: str = None,
    world_bbox: dict = None,
) -> tuple:
    """
    Replacement for original prepare_gt().
    Uses SAM3D with proper pose transforms (local -> camera -> world).
    No bbox rescale hack — coordinates come from SAM3D pose + camera extrinsics.
    
    Returns:
        Same as original prepare_gt:
        (gts, vol, vol_densities, grid_size, volume_surface, cam_info)
    """
    frames = data['frames']
    grid_size = phys_args.density_grid_size
    density_min_th = phys_args.density_min_th
    density_max_th = phys_args.density_max_th
    
    n_frames = getattr(phys_args, 'n_frames', None) or len(frames)
    n_frames = min(n_frames, len(frames))
    
    gts = []
    vol = None
    vol_densities = None
    vol_surface = None
    
    for idx in tqdm(range(n_frames), desc="SAM3D reconstruction"):
        frame = frames[idx]
        views = frame['views']
        view = views[0]
        
        if idx == 0:
            # First frame: get full output (need splat + pose for rendering)
            result = sam3d.reconstruct_frame_full(
                image=view['image'],
                mask=view['mask'],
                cam_R=view['cam_R'],
                cam_T=view['cam_T'],
            )
            xyz_world = result['xyz_world']
            
            # Save SAM3D splat, pose, and world data for new_trajectory_sam3d.py
            if output_path:
                splat_ply_path = os.path.join(output_path, 'sam3d_frame0.ply')
                result['output']['gs'].save_ply(splat_ply_path)
                print(f"  Saved SAM3D splat to {splat_ply_path}")
        else:
            # Other frames: just get world xyz using proper pose + cam2world
            xyz_world = sam3d.reconstruct_frame(
                image=view['image'],
                mask=view['mask'],
                cam_R=view['cam_R'],
                cam_T=view['cam_T'],
            )
        
        print(f"  Frame {idx}: {xyz_world.shape[0]} points, range [{xyz_world.min(0)[0].tolist()}]-[{xyz_world.max(0)[0].tolist()}]")
        
        # Voxelize
        is_first = (idx == 0)
        vox_result = voxelize_points(
            xyz_world, grid_size, density_min_th, density_max_th,
            is_first_frame=is_first,
        )
        
        # Collect surface points as geometric supervision
        gts.append(vox_result['surface_pts'])
        
        # First frame: extract MPM particles and save world data
        if is_first:
            vol = vox_result['vol']
            vol_densities = vox_result['vol_densities']
            vol_surface = vox_result['vol_surface']
            
            if output_path:
                write_particles(vol, 0, output_path, 'static')
                # Save world-space data and pose for rendering
                torch.save({
                    'xyz_world': xyz_world.cpu(),
                    'pose': {k: v.cpu() if torch.is_tensor(v) else v for k, v in result['pose'].items()},
                    'cam_R': view['cam_R'],
                    'cam_T': view['cam_T'],
                }, os.path.join(output_path, 'sam3d_frame0_world.pt'))
                print(f"  Saved world-space data and pose")
            
            print(f"  MPM particles: {vol.shape[0]}, surface: {vol_surface.shape[0]}")
    
    grid_size_tensor = torch.tensor([grid_size])
    cam_info = data.get('cam_info', None)
    
    return gts, vol, vol_densities, grid_size_tensor, vol_surface, cam_info


from train_dynamic import train, forward, backward, export_result


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    parser = ArgumentParser(description="SAM3D + GIC Physical Parameter Estimation")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--config_file", default='config/torus.json', type=str)
    parser.add_argument("--sam3d_config", default='checkpoints/hf/pipeline.yaml', type=str,
                        help="Path to SAM3D pipeline.yaml")
    parser.add_argument("--sam3d_repo", default=None, type=str,
                        help="Path to sam3d-objects repo root")
    parser.add_argument("--use_gic_data", action='store_true',
                        help="Use GIC's existing dataset format (for testing)")
    
    gs_args, phys_args = get_combined_args(parser)
    config_id = phys_args.id
    print(phys_args)
    safe_state(gs_args.quiet)
    
    dataset = model.extract(gs_args)
    os.makedirs(dataset.model_path, exist_ok=True)
    os.makedirs(os.path.join(dataset.model_path, 'mpm'), exist_ok=True)
    
    # ---- Step 1: SAM3D reconstruction (replaces Deformable 3DGS training) ----
    print("=" * 60)
    print("Step 1: SAM3D Reconstruction")
    print("=" * 60)
    
    sam3d = SAM3DWrapper(
        config_path=gs_args.sam3d_config,
        sam3d_repo_path=gs_args.sam3d_repo,
    )
    
    if gs_args.use_gic_data:
        # Use GIC's existing data format for testing
        data = load_multiview_data_from_gic_dataset(
            dataset, pipeline.extract(gs_args), phys_args,
            config_file=gs_args.config_file,
        )
    else:
        data = load_multiview_data(dataset.source_path, phys_args)
    
    gts, vol, vol_densities, grid_size, volume_surface, cam_info = prepare_gt_sam3d(
        sam3d, data, phys_args, output_path=dataset.model_path,
    )
    
    del sam3d  # Free GPU memory
    torch.cuda.empty_cache()
    
    print(f"Total frames: {len(gts)}")
    print(f"MPM particles: {vol.shape[0]}")
    print(f"GT surface points per frame: {[g.shape[0] for g in gts]}")
    
    # ---- Step 2: Velocity estimation (same as original) ----
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
    
    # ---- Step 3: Physical parameter estimation (same as original) ----
    print("=" * 60)
    print("Step 3: Physical Parameter Estimation")
    print("=" * 60)
    
    # Assign gaussian appearance to MPM particles for rendering supervision
    scene = assign_gs_to_pcd(
        vol, vol_densities, dataset,
        op.extract(gs_args),
        pipeline.extract(gs_args),
        cam_info, phys_args.density_grid_size,
    )
    
    # Fix: ensure train_cameras is a dict keyed by resolution scale
    # assign_gs_to_pcd -> Scene may store cameras as list instead of dict
    if isinstance(scene.train_cameras, list):
        scene.train_cameras = {1.0: scene.train_cameras}
    if isinstance(scene.test_cameras, list):
        scene.test_cameras = {1.0: scene.test_cameras}
    # Also ensure cameras are populated from cam_info if empty
    if not scene.train_cameras or (isinstance(scene.train_cameras, dict) and not scene.train_cameras.get(1.0)):
        scene.train_cameras = {1.0: cam_info['train_cams']}
        scene.test_cameras = {1.0: cam_info.get('test_cams', [])}
    
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
    print(f"Total time: {time.time() - start_time:.1f}s")