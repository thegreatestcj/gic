import os, sys, glob, math, argparse
import numpy as np, torch, torch.nn as nn, imageio
from PIL import Image as PILImage
from tqdm import tqdm
from pytorch3d.transforms import quaternion_to_matrix

sys.path.insert(0, '.')
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readPACNeRFInfo
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

def load_frame(mvsam3d_dir, frame_name):
    pattern = os.path.join(mvsam3d_dir, frame_name, "torus", "*", "params.npz")
    matches = sorted(glob.glob(pattern))
    if not matches: return None, None
    p = np.load(matches[-1])
    pose = {k: p[k].flatten() for k in ['scale','rotation','translation']}
    gs = GaussianModel(0)
    gs.load_ply(matches[-1].replace("params.npz","result.ply"))
    return gs, pose

def canonical_to_world(xyz, pose, cam_R, cam_T, device='cuda'):
    s = torch.tensor(pose['scale'], dtype=torch.float32, device=device)
    q = torch.tensor(pose['rotation'], dtype=torch.float32, device=device)
    t = torch.tensor(pose['translation'], dtype=torch.float32, device=device)
    R = quaternion_to_matrix(q.unsqueeze(0)).squeeze(0)
    # PLY stores coords in sparse tensor [D,H,W] order, NOT [X,Y,Z].
    # Reorder to [W,D,H] = [X,Y,Z] in mesh/canonical format (matches MV-SAM3D's LATENT_TO_MESH).
    LATENT_TO_MESH = torch.tensor([[0,0,1],[1,0,0],[0,1,0]], dtype=torch.float32, device=device)
    xyz = xyz @ LATENT_TO_MESH.T
    xyz = xyz * s.unsqueeze(0) @ R.T + t.unsqueeze(0)
    # Scale correction: MoGe depth is scale-ambiguous, correct using known camera distance
    cam_T_tensor = torch.tensor(cam_T, dtype=torch.float32, device=device)
    scale_correction = torch.norm(cam_T_tensor) / t[2]
    xyz = xyz * scale_correction
    xyz = xyz * torch.tensor([-1,-1,1], dtype=torch.float32, device=device)
    R_cam = torch.tensor(cam_R, dtype=torch.float32, device=device)
    xyz = (xyz - cam_T_tensor.unsqueeze(0)) @ R_cam.T
    return xyz

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvsam3d_dir", required=True)
    parser.add_argument("--data_path", default="data/pacnerf/torus")
    parser.add_argument("--config_file", default="config/pacnerf/torus.json")
    parser.add_argument("--output_dir", default="output/camera_space_render")
    parser.add_argument("--render_cam_id", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    scene_info = readPACNeRFInfo(args.data_path, args.config_file, True)
    all_cams = scene_info.train_cameras

    # Index cameras by (cam_id=uid, frame_time=fid) since all_data.json is unordered
    cam_by_uid_fid = {}
    for c in all_cams:
        cam_by_uid_fid[(c.uid, c.fid)] = c

    # Get sorted frame times
    frame_times = sorted(set(c.fid for c in all_cams))
    print(f"Found {len(frame_times)} unique frame times, using cam_id={args.render_cam_id}")

    # Use cam_id=0 frame_time=0 as the render camera
    ci = cam_by_uid_fid[(args.render_cam_id, frame_times[0])]
    world_view = torch.tensor(
        getWorld2View2(ci.R, ci.T, np.array([0.0,0.0,0.0]), 1.0)
    ).transpose(0,1).float().cuda()
    projection = getProjectionMatrix(
        znear=0.01, zfar=100.0, fovX=ci.FovX, fovY=ci.FovY
    ).transpose(0,1).float().cuda()
    full_proj = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    camera_center = world_view.inverse()[3,:3]
    bg = torch.ones(3, device='cuda')

    # Load background image (r_{cam_id}_-1.png)
    bg_path = os.path.join(args.data_path, "data", f"r_{args.render_cam_id}_-1.png")
    bg_img = np.array(PILImage.open(bg_path))[:, :, :3]
    if bg_img.shape[0] != ci.height or bg_img.shape[1] != ci.width:
        bg_img = np.array(PILImage.fromarray(bg_img).resize((ci.width, ci.height)))
    print(f"Loaded background: {bg_path}")

    frame_names = sorted([os.path.basename(d) for d in glob.glob(os.path.join(args.mvsam3d_dir, "frame_*"))])
    print(f"Found {len(frame_names)} MV-SAM3D frames")

    recon_frames = []
    gt_frames = []

    for fidx, frame_name in enumerate(tqdm(frame_names)):
        gs, pose = load_frame(args.mvsam3d_dir, frame_name)
        if gs is None:
            continue

        # Look up the correct camera: cam_id=render_cam_id, frame_time for this frame
        if fidx >= len(frame_times):
            continue
        fid = frame_times[fidx]
        key = (args.render_cam_id, fid)
        if key not in cam_by_uid_fid:
            print(f"  {frame_name}: no camera found for cam_id={args.render_cam_id} fid={fid}")
            continue
        frame_ci = cam_by_uid_fid[key]

        # GT frame — load raw image with background
        gt_path = os.path.join(args.data_path, "data", f"r_{args.render_cam_id}_{fidx}.png")
        gt_img = np.array(PILImage.open(gt_path))[:, :, :3]
        if gt_img.shape[0] != ci.height or gt_img.shape[1] != ci.width:
            gt_img = np.array(PILImage.fromarray(gt_img).resize((ci.width, ci.height)))
        gt_frames.append(gt_img)

        # Recon frame
        xyz_world = canonical_to_world(gs.get_xyz.detach(), pose, frame_ci.R, frame_ci.T)
        gs._xyz = nn.Parameter(xyz_world)

        with torch.no_grad():
            settings = GaussianRasterizationSettings(
                image_height=ci.height, image_width=ci.width,
                tanfovx=math.tan(ci.FovX*0.5), tanfovy=math.tan(ci.FovY*0.5),
                bg=bg, scale_modifier=1.0,
                viewmatrix=world_view, projmatrix=full_proj,
                sh_degree=gs.active_sh_degree, campos=camera_center,
                prefiltered=False, debug=False,
            )
            rasterizer = GaussianRasterizer(raster_settings=settings)
            scales = gs.get_scaling
            if scales.shape[1] == 1:
                scales = scales.repeat(1,3)
            image, _, _, _ = rasterizer(
                means3D=gs.get_xyz, means2D=torch.zeros_like(gs.get_xyz),
                shs=gs.get_features, colors_precomp=None,
                opacities=gs.get_opacity, scales=scales,
                rotations=gs.get_rotation, cov3D_precomp=None,
            )
            image = torch.clamp(image, 0, 1)

        recon_np = (image.cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
        # Composite: replace white background with scene background
        white_mask = (recon_np > 250).all(axis=2)
        recon_np[white_mask] = bg_img[white_mask]
        recon_frames.append(recon_np)
        rot = pose['rotation']
        world_ext = gs.get_xyz.max(dim=0).values - gs.get_xyz.min(dim=0).values
        print(f"  {frame_name}: scale={pose['scale'][0]:.4f} rot=[{rot[0]:.3f},{rot[1]:.3f},{rot[2]:.3f},{rot[3]:.3f}] t=[{pose['translation'][0]:.3f},{pose['translation'][1]:.3f},{pose['translation'][2]:.3f}] world_ext=[{world_ext[0]:.2f},{world_ext[1]:.2f},{world_ext[2]:.2f}]")

    # Side by side video
    if recon_frames and gt_frames:
        combined = []
        for r, g in zip(recon_frames, gt_frames):
            # Resize if different
            if r.shape != g.shape:
                g = np.array(PILImage.fromarray(g).resize((r.shape[1], r.shape[0])))
            combined.append(np.concatenate([g, r], axis=1))  # GT left, recon right
        imageio.mimwrite(os.path.join(args.output_dir, "comparison.mp4"), combined, fps=8, quality=8)
        print(f"Comparison: {args.output_dir}/comparison.mp4")

    if recon_frames:
        imageio.mimwrite(os.path.join(args.output_dir, "recon.mp4"), recon_frames, fps=8, quality=8)
    if gt_frames:
        imageio.mimwrite(os.path.join(args.output_dir, "gt.mp4"), gt_frames, fps=8, quality=8)

if __name__=="__main__":
    main()