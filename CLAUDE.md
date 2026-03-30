# CLAUDE.md - Project Context for Claude Code

## Project: MV-SAM3D + GIC Physical Parameter Estimation

### Goal
Replace GIC's Deformable 3DGS training + DeformNetwork with **MV-SAM3D** (multi-view) per-frame reconstruction to get world-space Gaussians, then feed into GIC's MPM physical parameter estimation pipeline.

### Pipeline Overview
```
multi-view frames (11 views × N timesteps)
  → MV-SAM3D (per timestep, 11 views → 1 unified 3D model)
  → cam2world coordinate transform (with scale correction)
  → Frame 0: voxelize → vol (MPM particles)
  → All frames: extract surface → gts[] (geometric supervision)
  → GIC velocity estimation (MPM)
  → GIC physical parameter estimation (MPM + differentiable rendering)
  → Render trajectory video with MV-SAM3D gaussians
```

### Key Files (in /orcd/home/002/fding/gic/)
- `train_dynamic_mvsam3d.py` — **Main pipeline script** (MV-SAM3D → GIC)
- `prepare_mvsam3d_data.py` — Convert PAC-NeRF data to MV-SAM3D format
- `run_mvsam3d_pipeline.sh` — Complete pipeline shell script
- `diagnose_cam2world.py` — Coordinate transform diagnostic (already verified)
- `sam3d_wrapper.py` — SAM3D inference wrapper (single-view, legacy)
- `train_dynamic_sam3d.py` — Single-view SAM3D pipeline (legacy, has scale issues)
- `new_trajectory_sam3d.py` — Render with SAM3D gaussians (legacy)

### Critical Coordinate Transform (VERIFIED)
MV-SAM3D outputs in canonical space [-0.5, 0.5] with a pose (scale, rotation, translation) mapping to view 0's PyTorch3D camera space. The correct world transform is:

```python
# 1. Z-up to Y-up rotation
Z_UP_TO_Y_UP = [[1,0,0],[0,0,-1],[0,1,0]]
xyz_yup = xyz_canonical @ Z_UP_TO_Y_UP.T

# 2. Apply MV-SAM3D pose → PyTorch3D camera space
xyz_p3d = scale * (R_pose @ xyz_yup) + translation

# 3. Scale correction (MoGe depth is scale-ambiguous)
scale_correction = ||cam_T|| / translation_z
xyz_scaled = xyz_p3d * scale_correction

# 4. PyTorch3D → OpenCV (flip x, y)
xyz_cv = xyz_scaled * [-1, -1, 1]

# 5. GIC cam2world
xyz_world = (xyz_cv - cam_T) @ cam_R.T
```

This was verified by `diagnose_cam2world.py` — Transform B gives correct results:
- Frame 00: world center [0.04, 0.60, 0.02] (expected ~[0, 0.65, 0])
- Frame 12: world center [0.12, 0.02, -0.17] (torus fell to ground)

### Data
- **PAC-NeRF dataset**: `data/pacnerf/torus/` — 15 timesteps × 11 views = 165 images
- **MV-SAM3D format**: `data/mvsam3d_torus/frame_XX/` — converted by prepare_mvsam3d_data.py
- **MV-SAM3D results**: `/home/fding/MV-SAM3D/visualization/frame_XX/`
- **Camera data**: `data/mvsam3d_torus/cameras.json`
- **Config**: `config/pacnerf/torus.json` (world bbox: [-0.5,0.5]×[0.1,1.2]×[-0.5,0.5])

### Environment
- **Conda env**: `sam3d-objects` (Python 3.11)
- **GPU**: NVIDIA L40S (44GB, sm_89), partition `mit_normal_gpu`
- **SAM3D repo**: `/home/fding/sam-3d-objects/`
- **MV-SAM3D repo**: `/home/fding/MV-SAM3D/` (checkpoints symlinked from sam-3d-objects)
- **GIC repo**: `/orcd/home/002/fding/gic/`
- **Cluster**: login via `fding@login005`, GPU via `salloc -p mit_normal_gpu -t 03:00:00 -G 1 --mem=32G`
- **taichi**: 1.6.0 (1.2.0 unavailable for Python 3.11)
- **diff_gauss**: custom fork, needs `module load cuda/12.4.0` for compilation

### Why MV-SAM3D Instead of Single-View SAM3D
Single-view SAM3D has a fatal flaw for physics estimation: **MoGe's depth scale is arbitrary and different per frame**. When we rescale each frame independently to match the world bbox, all motion information is lost (every frame maps to the same position). MV-SAM3D solves this because:
1. Multi-view fusion produces one consistent 3D model per timestep
2. The pose comes from view 0's camera space (consistent reference frame)
3. We can use known camera distance (||cam_T|| = 3.0) to correct MoGe's scale

### Current Status
- [x] MV-SAM3D reconstruction for all 15 frames (complete)
- [x] Coordinate transform verified (Transform B)
- [ ] GIC training with MV-SAM3D world coordinates (in progress, fixing minor bugs)
- [ ] Trajectory rendering with MV-SAM3D gaussians
- [ ] Validate estimated physical parameters

### Known Issues / TODOs
1. **Scale inconsistency**: MV-SAM3D scale varies ~26% between frames. Scale correction helps but isn't perfect.
2. **density_min_th/density_max_th**: Config has 0.5/0.8 which may be too strict. Previous tests showed 0.05/0.5 gives more particles (~23k vs ~400).
3. **SH degree**: MV-SAM3D gaussians have SH degree 0 (only f_dc, no f_rest). GaussianModel must be initialized with sh_degree=0.
4. **Gaussian scale field**: MV-SAM3D ply may have only `scale_0` (isotropic) instead of `scale_0,1,2`. Rendering needs `.repeat(1,3)`.

### Run Commands
```bash
# Prepare data (login node, no GPU)
cd /orcd/home/002/fding/gic
python prepare_mvsam3d_data.py

# Run MV-SAM3D (GPU node, from MV-SAM3D dir)
cd /home/fding/MV-SAM3D
bash /orcd/home/002/fding/gic/run_mvsam3d_all_frames.sh

# Run GIC training + rendering (GPU node, from GIC dir)
cd /orcd/home/002/fding/gic
python train_dynamic_mvsam3d.py \
    -c config/pacnerf/torus.json \
    -s data/pacnerf/torus \
    -m output/mvsam3d_torus \
    --mvsam3d_dir /home/fding/MV-SAM3D/visualization \
    --camera_json data/mvsam3d_torus/cameras.json \
    --config_file config/pacnerf/torus.json \
    --predict_config config/predict/elastic.json

# Or run everything with one script
bash run_mvsam3d_pipeline.sh
```