import os, sys, glob, math, argparse, json
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
    matches = sorted(glob.glob(os.path.join(mvsam3d_dir, frame_name, "torus", "*", "params.npz")))
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
    Z2Y = torch.tensor([[1,0,0],[0,0,-1],[0,1,0]], dtype=torch.float32, device=device)
    xyz = xyz @ Z2Y.T
    xyz = xyz * s.unsqueeze(0) @ R.T + t.unsqueeze(0)
    xyz = xyz * torch.tensor([-1,-1,1], dtype=torch.float32, device=device)
    R_cam = torch.tensor(cam_R, dtype=torch.float32, device=device)
    T_cam = torch.tensor(cam_T, dtype=torch.float32, device=device)
    xyz = (xyz - T_cam.unsqueeze(0)) @ R_cam.T
    return xyz

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvsam3d_dir", required=True)
    parser.add_argument("--data_path", default="data/pacnerf/torus")
    parser.add_argument("--config_file", default="config/pacnerf/torus.json")
    parser.add_argument("--output_dir", default="output/camera_space_render")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    scene_info = readPACNeRFInfo(args.data_path, args.config_file, True)
    all_cams = scene_info.train_cameras

    # Load correct camera ordering from cameras.json
    with open('data/mvsam3d_torus/cameras.json') as f:
        cam_json = json.load(f)

    # Render camera = View 0 Frame 0 (from cameras.json)
    cam0 = cam_json['frame_00']['cameras'][0]
    # Find matching CameraInfo from all_cams for FoV and image size
    ci = all_cams[0]  # just for FoV/size, actual R/T from cam_json
    # Camera 0 (uid=0) is view_idx=6 in cameras.json - matches GT r_0_*.png
    render_R = np.array(cam_json['frame_00']['cameras'][0]['R'])
    render_T = np.array(cam_json['frame_00']['cameras'][0]['T'])
    world_view = torch.tensor(
        getWorld2View2(render_R, render_T, np.array([0.0,0.0,0.0]), 1.0)
    ).transpose(0,1).float().cuda()
    projection = getProjectionMatrix(
        znear=0.01, zfar=100.0, fovX=ci.FovX, fovY=ci.FovY
    ).transpose(0,1).float().cuda()
    full_proj = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    camera_center = world_view.inverse()[3,:3]
    bg = torch.ones(3, device='cuda')

    # Load background from GT frame 0: r_0_0.png with torus removed
    gt0 = np.array(PILImage.open(os.path.join(args.data_path, "data", "r_0_-1.png")))[:,:,:3]
    a0 = np.array(PILImage.open(os.path.join(args.data_path, "data", "a_0_0.png")))
    if a0.ndim == 3: a0 = a0[:,:,3] if a0.shape[2]==4 else a0[:,:,0]
    # Get ground color and fill torus area
    # ground_color = gt0[a0 < 30].mean(axis=0).astype(np.uint8)
    bg_clean = gt0
    # bg_clean[a0 > 128] = ground_color
    # print(f"Background ready, ground color: {ground_color}")

    recon_frames = []
    gt_frames = []

    for fidx in tqdm(range(14)):
        frame_name = f"frame_{fidx:02d}"

        # GT
        gt_img = np.array(PILImage.open(os.path.join(args.data_path, "data", f"r_0_{fidx}.png")))[:,:,:3]
        gt_frames.append(gt_img)

        # Load recon
        gs, pose = load_frame(args.mvsam3d_dir, frame_name)
        if gs is None:
            recon_frames.append(bg_clean.copy())
            continue

        # Use cameras.json for correct camera
        cam_f = cam_json[frame_name]['cameras'][0]  # View 0 of this frame
        cam_R = np.array(cam_f['R'])
        cam_T = np.array(cam_f['T'])

        xyz_world = canonical_to_world(gs.get_xyz.detach(), pose, cam_R, cam_T)
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
            if scales.shape[1] == 1: scales = scales.repeat(1,3)
            image, _, _, _ = rasterizer(
                means3D=gs.get_xyz, means2D=torch.zeros_like(gs.get_xyz),
                shs=gs.get_features, colors_precomp=None,
                opacities=gs.get_opacity, scales=scales,
                rotations=gs.get_rotation, cov3D_precomp=None,
            )
            image = torch.clamp(image, 0, 1)

        recon_np = (image.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)

        # White pixels in recon = no torus = replace with background
        white = (recon_np > 250).all(axis=2)
        recon_with_bg = recon_np.copy()
        if bg_clean.shape[:2] == recon_np.shape[:2]:
            recon_with_bg[white] = bg_clean[white]

        recon_frames.append(recon_with_bg)
        print(f"  {frame_name}: scale={pose['scale'][0]:.4f} ty={pose['translation'][1]:.4f}")

    imageio.mimwrite(os.path.join(args.output_dir, "gt.mp4"), gt_frames, fps=8, quality=8)
    imageio.mimwrite(os.path.join(args.output_dir, "recon.mp4"), recon_frames, fps=8, quality=8)
    combined = [np.concatenate([g,r], axis=1) for g,r in zip(gt_frames, recon_frames)]
    imageio.mimwrite(os.path.join(args.output_dir, "comparison.mp4"), combined, fps=8, quality=8)
    for i,(g,r) in enumerate(zip(gt_frames, recon_frames)):
        PILImage.fromarray(g).save(os.path.join(args.output_dir, f"frame_{i:02d}_gt.png"))
        PILImage.fromarray(r).save(os.path.join(args.output_dir, f"frame_{i:02d}_recon.png"))
    print("Done.")

if __name__=="__main__":
    main()
