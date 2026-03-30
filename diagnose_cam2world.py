"""
Diagnose and fix the MV-SAM3D -> world coordinate transform.

We know:
- MV-SAM3D pose maps canonical [-0.5,0.5] to view0's camera space (PyTorch3D convention)
- PyTorch3D: x-left, y-up, z-forward
- GIC stores cam_R (transposed rotation) and cam_T
- GIC cam2world: x_world = (x_cam - T) @ R.T  (row-vector convention)
- MoGe depth is scale-ambiguous -> need scale correction

Plan:
1. Load MV-SAM3D params for both frames
2. Apply pose to get object center in camera space
3. Compute scale correction using known camera distance
4. Try different coordinate transforms to find the right one
5. Compare world coordinates with expected PAC-NeRF bbox
"""

import numpy as np
import json
import glob
import sys
import torch
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix, Transform3d

# ---- Load data ----
with open('data/mvsam3d_torus/cameras.json') as f:
    cam_data = json.load(f)

# Expected: torus world bbox from config
with open('config/pacnerf/torus.json') as f:
    cfg = json.load(f)
xyz_min = np.array(cfg['data']['xyz_min'])  # [-0.5, 0.1, -0.5]
xyz_max = np.array(cfg['data']['xyz_max'])  # [0.5, 1.2, 0.5]
world_center = (xyz_min + xyz_max) / 2  # [0, 0.65, 0]
print(f"Expected world center: {world_center}")
print(f"Expected world bbox: {xyz_min} ~ {xyz_max}\n")

# Z-up to Y-up (SAM3D canonical uses Z-up, GLB uses Y-up)
Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)

for frame_name in ['frame_00', 'frame_12']:
    print(f"{'='*60}")
    print(f"  {frame_name}")
    print(f"{'='*60}")
    
    # Load MV-SAM3D params
    npz_files = glob.glob(f'/home/fding/MV-SAM3D/visualization/{frame_name}/**/*params.npz', recursive=True)
    params = np.load(npz_files[0])
    
    scale = params['scale'].flatten()
    rot_quat = params['rotation'].flatten()  # wxyz
    translation = params['translation'].flatten()
    
    print(f"MV-SAM3D pose:")
    print(f"  scale: {scale}")
    print(f"  rotation (wxyz): {rot_quat}")
    print(f"  translation: {translation}")
    
    # Build pose transform (canonical -> PyTorch3D camera space)
    R_pose = quaternion_to_matrix(torch.tensor(rot_quat).float().unsqueeze(0)).squeeze(0).numpy()
    
    # Object center in canonical = [0, 0, 0]
    # After Z-up to Y-up: still [0, 0, 0]
    # After pose: scale * R @ [0,0,0] + translation = translation
    center_p3d = translation
    print(f"\nObject center in PyTorch3D camera space: {center_p3d}")
    print(f"  distance from camera: {np.linalg.norm(center_p3d):.4f}")
    
    # Get view 0 camera params
    cam0 = cam_data[frame_name]['cameras'][0]
    cam_R = np.array(cam0['R'])  # GIC convention: transposed rotation
    cam_T = np.array(cam0['T'])
    
    print(f"\nGIC camera 0:")
    print(f"  cam_R:\n{cam_R}")
    print(f"  cam_T: {cam_T}")
    print(f"  camera distance from origin: {np.linalg.norm(cam_T):.4f}")
    
    # ---- Scale correction ----
    # MoGe depth is scale-ambiguous. The object should be at ~cam_distance from camera.
    # MV-SAM3D says it's at distance ||translation|| from camera.
    # Scale correction = true_distance / estimated_distance
    
    # For PAC-NeRF torus, camera is at distance 3 from origin, object is near origin
    # So object-camera distance ≈ 3.0
    true_cam_dist = np.linalg.norm(cam_T)  # 3.0
    estimated_dist = center_p3d[2]  # z component (depth in PyTorch3D = forward)
    scale_correction = true_cam_dist / estimated_dist if abs(estimated_dist) > 0.01 else 1.0
    print(f"\nScale correction: {true_cam_dist:.4f} / {estimated_dist:.4f} = {scale_correction:.4f}")
    
    # ---- Try different coordinate transforms ----
    
    # Transform A: PyTorch3D -> OpenCV: flip x,y
    # Then GIC cam2world
    center_cv_A = center_p3d * np.array([-1, -1, 1])
    world_A = (center_cv_A - cam_T) @ cam_R.T
    
    # Transform B: same but with scale correction
    center_cv_B = center_p3d * scale_correction * np.array([-1, -1, 1])
    world_B = (center_cv_B - cam_T) @ cam_R.T
    
    # Transform C: no flip, direct
    center_cv_C = center_p3d
    world_C = (center_cv_C - cam_T) @ cam_R.T
    
    # Transform D: no flip, with scale correction
    center_cv_D = center_p3d * scale_correction
    world_D = (center_cv_D - cam_T) @ cam_R.T
    
    # Transform E: flip only x
    center_cv_E = center_p3d * np.array([-1, 1, 1])
    world_E = (center_cv_E - cam_T) @ cam_R.T
    
    # Transform F: flip only x, with scale correction
    center_cv_F = center_p3d * scale_correction * np.array([-1, 1, 1])
    world_F = (center_cv_F - cam_T) @ cam_R.T
    
    # Transform G: Use R instead of R.T for cam2world
    center_cv_G = center_p3d * np.array([-1, -1, 1])
    world_G = (center_cv_G - cam_T) @ cam_R
    
    # Transform H: Use R with scale correction
    center_cv_H = center_p3d * scale_correction * np.array([-1, -1, 1])
    world_H = (center_cv_H - cam_T) @ cam_R
    
    print(f"\nWorld coordinate results:")
    print(f"  Expected: near {world_center} = [0.0, 0.65, 0.0]")
    print(f"  A (flip xy, R.T):          {world_A}")
    print(f"  B (flip xy, R.T, scaled):  {world_B}")
    print(f"  C (no flip, R.T):          {world_C}")
    print(f"  D (no flip, R.T, scaled):  {world_D}")
    print(f"  E (flip x, R.T):           {world_E}")
    print(f"  F (flip x, R.T, scaled):   {world_F}")
    print(f"  G (flip xy, R):            {world_G}")
    print(f"  H (flip xy, R, scaled):    {world_H}")
    
    # Score each transform by distance to expected center
    transforms = {'A': world_A, 'B': world_B, 'C': world_C, 'D': world_D,
                  'E': world_E, 'F': world_F, 'G': world_G, 'H': world_H}
    
    print(f"\n  Distance to expected center:")
    for name, w in sorted(transforms.items(), key=lambda x: np.linalg.norm(x[1] - world_center)):
        dist = np.linalg.norm(w - world_center)
        in_bbox = np.all(w >= xyz_min) and np.all(w <= xyz_max)
        print(f"    {name}: {dist:.4f} {'✅ in bbox' if in_bbox else ''}")
    
    print()