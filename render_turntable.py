# render_turntable.py
# Render a turntable video from a SAM3D .ply splat file
#
# Usage:
#   python render_turntable.py \
#       --ply test_output_v2/05_sam3d_splat.ply \
#       --output_dir render_output/ \
#       --n_frames 60 \
#       --resolution 512

import torch
import numpy as np
import os
import math
import argparse
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh
import torchvision
import imageio


def make_camera_params(
    R: np.ndarray,
    T: np.ndarray,
    fovx: float,
    fovy: float,
    width: int,
    height: int,
):
    """Build camera matrices in GIC convention."""
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix
    
    world_view = torch.tensor(
        getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)
    ).transpose(0, 1).float().cuda()
    
    projection = getProjectionMatrix(
        znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy
    ).transpose(0, 1).float().cuda()
    
    full_proj = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    camera_center = world_view.inverse()[3, :3]
    
    return {
        'world_view_transform': world_view,
        'projection_matrix': projection,
        'full_proj_transform': full_proj,
        'camera_center': camera_center,
        'FoVx': fovx,
        'FoVy': fovy,
        'image_width': width,
        'image_height': height,
    }


def render_frame(gaussians: GaussianModel, cam: dict, bg_color: torch.Tensor):
    """Render one frame using diff_gauss rasterizer."""
    tanfovx = math.tan(cam['FoVx'] * 0.5)
    tanfovy = math.tan(cam['FoVy'] * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=cam['image_height'],
        image_width=cam['image_width'],
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=cam['world_view_transform'],
        projmatrix=cam['full_proj_transform'],
        sh_degree=gaussians.active_sh_degree,
        campos=cam['camera_center'],
        prefiltered=False,
        debug=False,
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means3D = gaussians.get_xyz
    means2D = torch.zeros_like(means3D, requires_grad=False, device="cuda")
    opacity = gaussians.get_opacity
    scales = gaussians.get_scaling.repeat(1, 3)
    rotations = gaussians.get_rotation
    shs = gaussians.get_features
    
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


def generate_orbit_cameras(
    center: np.ndarray,
    radius: float,
    elevation: float,
    n_frames: int,
    fov: float,
    width: int,
    height: int,
):
    """Generate camera parameters for a turntable orbit."""
    cameras = []
    fovx = fov
    fovy = fov * height / width
    
    for i in range(n_frames):
        angle = 2.0 * np.pi * i / n_frames
        
        # Camera position on orbit
        eye = np.array([
            center[0] + radius * np.cos(angle),
            center[1] + elevation,
            center[2] + radius * np.sin(angle),
        ])
        
        # Look-at matrix
        forward = center - eye
        forward = forward / np.linalg.norm(forward)
        
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # World-to-camera rotation (GIC convention: R is transposed)
        R = np.stack([right, -up, forward], axis=1)  # (3, 3)
        R = R.T  # GIC stores transposed rotation
        
        # Translation
        T = -R @ eye
        
        cam = make_camera_params(R, T, fovx, fovy, width, height)
        cameras.append(cam)
    
    return cameras


def main():
    parser = argparse.ArgumentParser(description="Render turntable video from SAM3D splat")
    parser.add_argument("--ply", required=True, help="Path to .ply gaussian splat file")
    parser.add_argument("--output_dir", default="render_output", help="Output directory")
    parser.add_argument("--n_frames", type=int, default=60, help="Number of frames")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--radius", type=float, default=None, help="Camera orbit radius (auto if None)")
    parser.add_argument("--elevation", type=float, default=None, help="Camera elevation (auto if None)")
    parser.add_argument("--fov", type=float, default=0.8, help="Field of view in radians")
    parser.add_argument("--bg", type=float, default=1.0, help="Background color (0=black, 1=white)")
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'frames'), exist_ok=True)
    
    # Load gaussian splat
    print(f"Loading {args.ply}...")
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(args.ply)
    print(f"  Loaded {gaussians.get_xyz.shape[0]} gaussians")
    
    # Compute scene center and auto radius
    xyz = gaussians.get_xyz.detach()
    center = xyz.mean(dim=0).cpu().numpy()
    extent = (xyz.max(dim=0)[0] - xyz.min(dim=0)[0]).cpu().numpy()
    max_extent = extent.max()
    
    radius = args.radius if args.radius is not None else max_extent * 1.5
    elevation = args.elevation if args.elevation is not None else max_extent * 0.3
    
    print(f"  Center: {center}")
    print(f"  Extent: {extent}")
    print(f"  Orbit radius: {radius:.3f}, elevation: {elevation:.3f}")
    
    # Generate cameras
    width = height = args.resolution
    cameras = generate_orbit_cameras(
        center, radius, elevation, args.n_frames,
        args.fov, width, height,
    )
    
    # Render
    bg_color = torch.tensor([args.bg, args.bg, args.bg], dtype=torch.float32, device="cuda")
    
    frames = []
    print(f"Rendering {args.n_frames} frames...")
    with torch.no_grad():
        for i, cam in enumerate(tqdm(cameras, desc="Rendering")):
            image, alpha = render_frame(gaussians, cam, bg_color)
            image = torch.clamp(image, 0.0, 1.0)
            
            # Save individual frame
            torchvision.utils.save_image(
                image, 
                os.path.join(args.output_dir, 'frames', f'{i:04d}.png')
            )
            
            # Collect for video
            frame_np = (image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frames.append(frame_np)
    
    # Save video
    video_path = os.path.join(args.output_dir, 'turntable.mp4')
    imageio.mimwrite(video_path, frames, fps=30, quality=8)
    print(f"\nVideo saved to {video_path}")
    print(f"Frames saved to {args.output_dir}/frames/")


if __name__ == "__main__":
    main()