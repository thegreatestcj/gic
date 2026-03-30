# new_trajectory_sam3d.py
# Generate physics simulation videos using SAM3D gaussians directly
# Instead of retraining gaussian appearance with train_gs_fixed_pcd,
# we use the original SAM3D output and map MPM particle displacements
# to SAM3D gaussians via nearest neighbor.
#
# Usage:
#   python new_trajectory_sam3d.py \
#       -c config/predict/elastic.json \
#       -s data/pacnerf/torus \
#       -m output/sam3d_torus_v2 \
#       -vid 0 -cid 0

import taichi as ti
import torch
import torch.nn as nn
import numpy as np
import os
import json
import math
import torchvision
import imageio
import trimesh
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from pathlib import Path

from scene.gaussian_model import GaussianModel
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from simulator import Simulator


def load_pcd_file(file_path, iteration=None):
    """Load MPM particles from ply file."""
    path = os.path.join(file_path, "mpm", "static_0.ply")
    pcd = trimesh.load_mesh(path)
    vol = torch.from_numpy(np.array(pcd.vertices)).to('cuda', dtype=torch.float32).contiguous()
    return vol


def read_estimation_result(model_path, config_id):
    """Read physical parameter estimation result."""
    result_path = os.path.join(model_path, f"{config_id}-pred.json")
    with open(result_path, 'r') as f:
        result = json.load(f)
    print(f"Loaded estimation result from {result_path}")
    return result


def load_sam3d_gaussians(model_path, device='cuda'):
    """
    Load SAM3D gaussians and transform to world space using saved pose + camera params.
    """
    ply_path = os.path.join(model_path, 'sam3d_frame0.ply')
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"SAM3D splat not found: {ply_path}\nRe-run train_dynamic_sam3d.py to generate it.")
    
    # Load as GaussianModel (SH degree 0)
    gs = GaussianModel(0)
    gs.load_ply(ply_path)
    
    # Load saved world-space data
    pt_path = os.path.join(model_path, 'sam3d_frame0_world.pt')
    if os.path.exists(pt_path):
        saved = torch.load(pt_path, map_location=device, weights_only=False)
        
        if saved.get('xyz_world') is not None:
            xyz_world = saved['xyz_world'].to(device)
            if xyz_world.shape[0] == gs.get_xyz.shape[0]:
                gs._xyz = nn.Parameter(xyz_world)
                print(f"  Loaded world-space xyz directly ({xyz_world.shape[0]} pts)")
            else:
                # Count mismatch: xyz_world came from SAM3D output points,
                # gs has gaussian splat points. Need to apply pose transform.
                print(f"  xyz_world count ({xyz_world.shape[0]}) != gs count ({gs.get_xyz.shape[0]})")
                print(f"  Applying pose + cam transform to gs.get_xyz")
                xyz_world = None
        else:
            xyz_world = None
        
        # If no direct xyz_world, apply pose + cam2world transform
        if xyz_world is None and saved.get('pose') is not None:
            from sam3d_wrapper import SAM3DWrapper, rotation_6d_to_matrix
            
            pose = saved['pose']
            cam_R = saved['cam_R']
            cam_T = saved['cam_T']
            
            R = pose['rotation'].to(device)
            scale = pose['scale'].to(device)
            translation = pose['translation'].to(device)
            
            xyz_local = gs.get_xyz.detach()  # centered [-0.5, 0.5]
            
            # local -> camera: scale, rotate, translate
            xyz_scaled = xyz_local * scale.unsqueeze(0)
            xyz_rotated = xyz_scaled @ R.T
            xyz_camera = xyz_rotated + translation.unsqueeze(0)
            
            # camera -> world: R_cam, T_cam
            R_cam = torch.tensor(cam_R, dtype=torch.float32, device=device)
            T_cam = torch.tensor(cam_T, dtype=torch.float32, device=device)
            xyz_world = (xyz_camera - T_cam.unsqueeze(0)) @ R_cam
            
            gs._xyz = nn.Parameter(xyz_world)
            print(f"  Applied pose + cam2world transform")
    
    print(f"  XYZ range: {gs.get_xyz.min(0)[0].tolist()} ~ {gs.get_xyz.max(0)[0].tolist()}")
    print(f"  Loaded {gs.get_xyz.shape[0]} SAM3D gaussians")
    return gs


def build_nn_mapping(gs_xyz, vol_xyz):
    """
    Build nearest-neighbor mapping from SAM3D gaussians to MPM particles.
    For each SAM3D gaussian, find the nearest MPM particle.
    
    Args:
        gs_xyz: (N_gs, 3) SAM3D gaussian positions (world space)
        vol_xyz: (N_vol, 3) MPM particle positions (world space, frame 0)
    Returns:
        nn_indices: (N_gs,) index of nearest MPM particle for each gaussian
    """
    # Use batched computation to avoid OOM
    batch_size = 10000
    n_gs = gs_xyz.shape[0]
    nn_indices = torch.zeros(n_gs, dtype=torch.long, device=gs_xyz.device)
    
    for i in range(0, n_gs, batch_size):
        end = min(i + batch_size, n_gs)
        # (batch, 1, 3) - (1, N_vol, 3) -> (batch, N_vol)
        dists = torch.cdist(gs_xyz[i:end].unsqueeze(0), vol_xyz.unsqueeze(0)).squeeze(0)
        nn_indices[i:end] = dists.argmin(dim=1)
    
    return nn_indices


def render_frame(gs, cam_params, bg_color, d_xyz=0.0):
    """Render one frame using diff_gauss."""
    tanfovx = math.tan(cam_params['FoVx'] * 0.5)
    tanfovy = math.tan(cam_params['FoVy'] * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=cam_params['image_height'],
        image_width=cam_params['image_width'],
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=cam_params['world_view_transform'],
        projmatrix=cam_params['full_proj_transform'],
        sh_degree=gs.active_sh_degree,
        campos=cam_params['camera_center'],
        prefiltered=False,
        debug=False,
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means3D = gs.get_xyz + d_xyz
    means2D = torch.zeros_like(means3D, requires_grad=False, device="cuda")
    opacity = gs.get_opacity
    scales = gs.get_scaling.repeat(1, 3)
    rotations = gs.get_rotation
    shs = gs.get_features
    
    rendered_image, depth, alpha, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    
    return rendered_image, alpha


def get_camera_from_scene(dataset, view_id=0):
    """Load camera parameters from dataset, bypassing Scene."""
    from scene.dataset_readers import readPACNeRFInfo, readNerfSyntheticInfo
    import os, glob
    
    source_path = dataset.source_path
    config_path = getattr(dataset, 'config_path', None)
    
    # config_path might point to predict config, we need training config
    # Try to find the right config with 'data' key
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg_test = json.load(f)
        if 'data' not in cfg_test:
            # This is a predict config, look for training config
            # Search in config/pacnerf/ for a matching config
            for candidate in glob.glob('config/pacnerf/*.json'):
                with open(candidate) as f:
                    c = json.load(f)
                if 'data' in c:
                    config_path = candidate
                    break
    
    if os.path.exists(os.path.join(source_path, "all_data.json")):
        scene_info = readPACNeRFInfo(source_path, config_path, dataset.white_background)
    elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
        scene_info = readNerfSyntheticInfo(source_path, dataset.white_background, False)
    else:
        raise ValueError(f"Cannot detect dataset type at {source_path}")
    
    cam_infos = scene_info.train_cameras
    
    # Get first frame views
    fids = [ci.fid for ci in cam_infos]
    min_fid = min(fids)
    first_frame_cams = [ci for ci in cam_infos if ci.fid == min_fid]
    
    ci = first_frame_cams[min(view_id, len(first_frame_cams) - 1)]
    
    # Build camera matrices
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix
    
    world_view = torch.tensor(
        getWorld2View2(ci.R, ci.T, np.array([0.0, 0.0, 0.0]), 1.0)
    ).transpose(0, 1).float().cuda()
    
    fovx = ci.FovX
    fovy = ci.FovY
    projection = getProjectionMatrix(
        znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy
    ).transpose(0, 1).float().cuda()
    
    full_proj = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    camera_center = world_view.inverse()[3, :3]
    
    cam_params = {
        'FoVx': fovx,
        'FoVy': fovy,
        'image_height': ci.height,
        'image_width': ci.width,
        'world_view_transform': world_view,
        'full_proj_transform': full_proj,
        'camera_center': camera_center,
    }
    
    return cam_params


def gen_xyz_list(simulator, frames):
    """Generate particle trajectory from simulator."""
    seq = []
    xyz0 = simulator.vol.detach()
    simulator.initialize()
    print(f"Init velocity: {simulator.vel.tolist()}")
    
    for f in tqdm(range(frames), desc="Simulating"):
        xyz = simulator.forward(f)
        # Ensure xyz is a tensor (simulator may return numpy)
        if isinstance(xyz, np.ndarray):
            xyz = torch.from_numpy(xyz).float().to(xyz0.device)
        d_xyz = (xyz - xyz0).detach()
        seq.append(d_xyz)
    
    return seq


def main():
    parser = ArgumentParser(description="Generate trajectory with SAM3D gaussians")
    parser.add_argument('-vid', '--view_id', type=int, default=0)
    parser.add_argument('-cid', '--config_id', type=int, default=0)
    parser.add_argument('--fps', type=int, default=24)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    gs_args, phys_args = get_combined_args(parser)
    
    phys_args.view_id = gs_args.view_id
    phys_args.config_id = gs_args.config_id
    print(phys_args)
    safe_state(gs_args.quiet)
    dataset = model.extract(gs_args)
    
    output_dir = dataset.model_path
    render_dir = os.path.join(output_dir, 'render_sam3d')
    os.makedirs(render_dir, exist_ok=True)
    
    # ---- 1. Load SAM3D gaussians ----
    print("=" * 60)
    print("Loading SAM3D Gaussians")
    print("=" * 60)
    gs = load_sam3d_gaussians(output_dir)
    
    # ---- 2. Load MPM particles and physical params ----
    print("=" * 60)
    print("Loading Physical Parameters")
    print("=" * 60)
    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.4)
    
    vol = load_pcd_file(output_dir)
    
    # Read estimation result for reference, but use predict config for simulation
    estimation_result = read_estimation_result(output_dir, phys_args.config_id)
    
    # Use phys_args (from predict config) directly for simulation
    # This allows overriding E, nu, vel etc. from the predict config
    # Fill in missing fields from estimation result
    if not hasattr(phys_args, 'density_grid_size'):
        phys_args.density_grid_size = estimation_result.get('density_grid_size', 0.02)
    
    print(f"  Using mat_params: {phys_args.mat_params}")
    print(f"  Using vel: {phys_args.vel}")
    print(f"  Using gravity: {phys_args.gravity}")
    
    # ---- 3. Build nearest-neighbor mapping ----
    print("=" * 60)
    print("Building NN Mapping (SAM3D gaussians -> MPM particles)")
    print("=" * 60)
    nn_indices = build_nn_mapping(gs.get_xyz.detach(), vol)
    print(f"  Mapped {gs.get_xyz.shape[0]} gaussians to {vol.shape[0]} particles")
    
    # ---- 4. Run simulation ----
    print("=" * 60)
    print("Running Physics Simulation")
    print("=" * 60)
    predict_frames = getattr(phys_args, 'predict_frames', 48)
    simulator = Simulator(phys_args, vol)
    d_xyz_list = gen_xyz_list(simulator, predict_frames)
    
    # ---- 5. Get camera ----
    print("=" * 60)
    print("Loading Camera")
    print("=" * 60)
    cam_params = get_camera_from_scene(dataset, view_id=phys_args.view_id)
    
    bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    
    # ---- 6. Render ----
    print("=" * 60)
    print(f"Rendering {predict_frames} frames")
    print("=" * 60)
    
    frames_for_video = []
    with torch.no_grad():
        for f in tqdm(range(predict_frames), desc="Rendering"):
            # Map MPM particle displacement to SAM3D gaussians
            particle_d_xyz = d_xyz_list[f]  # (N_vol, 3)
            gs_d_xyz = particle_d_xyz[nn_indices]  # (N_gs, 3) - each gaussian gets its nearest particle's displacement
            
            # Render
            image, alpha = render_frame(gs, cam_params, bg_color, d_xyz=gs_d_xyz)
            image = torch.clamp(image, 0.0, 1.0)
            
            # Save frame
            torchvision.utils.save_image(image, os.path.join(render_dir, f'{f:05d}.png'))
            
            frame_np = (image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frames_for_video.append(frame_np)
    
    # ---- 7. Save video ----
    video_path = os.path.join(render_dir, 'output.mp4')
    imageio.mimwrite(video_path, frames_for_video, fps=phys_args.fps, quality=8)
    print(f"\nVideo saved to {video_path}")
    print(f"Frames saved to {render_dir}/")


if __name__ == "__main__":
    main()