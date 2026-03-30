#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 03:00:00
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -o logs/sam3d_gic_%j.out

# ============================================================
# SAM3D + GIC Pipeline
# Usage:
#   sbatch run_sam3d_gic.sh          (submit as job)
#   bash run_sam3d_gic.sh            (run interactively on GPU node)
# ============================================================

set -e
 
# ---- Config ----
CONDA_ENV="sam3d-objects"
SAM3D_CONFIG="/home/fding/sam-3d-objects/checkpoints/hf/pipeline.yaml"
SAM3D_REPO="/home/fding/sam-3d-objects"
 
TRAIN_CONFIG="config/pacnerf/torus.json"
PREDICT_CONFIG="config/predict/elastic.json"
DATA_PATH="data/pacnerf/torus"
OUTPUT_DIR="output/sam3d_torus_v3"
 
# ---- Setup ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
mkdir -p ${OUTPUT_DIR}/{mpm,gs,img,render,render_sam3d}
mkdir -p logs
 
# Create cfg_args if not exists
if [ ! -f "${OUTPUT_DIR}/cfg_args" ]; then
    python -c "
from argparse import Namespace
cfg = Namespace(
    sh_degree=3, source_path='${DATA_PATH}', model_path='${OUTPUT_DIR}',
    config_path='${TRAIN_CONFIG}', images='images', resolution=-1,
    white_background=True, data_device='cuda', eval=False, num_frame=-1,
    load2gpu_on_the_fly=False, is_blender=False, is_6dof=False, res_scale=1.0,
)
with open('${OUTPUT_DIR}/cfg_args', 'w') as f:
    f.write(str(cfg))
"
    echo "Created cfg_args"
fi
 
# ============================================================
# Step 1-3: Training (SAM3D recon + velocity + physical params)
# ============================================================
echo "============================================================"
echo "Step 1-3: SAM3D + GIC Training"
echo "============================================================"
 
python train_dynamic_sam3d.py \
    -c ${TRAIN_CONFIG} \
    -s ${DATA_PATH} \
    -m ${OUTPUT_DIR} \
    --sam3d_config ${SAM3D_CONFIG} \
    --sam3d_repo ${SAM3D_REPO} \
    --use_gic_data \
    --config_file ${TRAIN_CONFIG}
 
echo "Training complete. Results:"
cat ${OUTPUT_DIR}/0-pred.json
 
# ============================================================
# Step 4: Render trajectory video with SAM3D gaussians
# ============================================================
echo ""
echo "============================================================"
echo "Step 4: Rendering trajectory video"
echo "============================================================"
 
# Generate SAM3D splat files if not already created during training
if [ ! -f "${OUTPUT_DIR}/sam3d_frame0.ply" ]; then
    echo "Generating SAM3D splat files..."
    python -c "
import sys, os, json, torch, numpy as np
sys.path.insert(0, '${SAM3D_REPO}')
sys.path.insert(0, '${SAM3D_REPO}/notebook')
from sam3d_wrapper import SAM3DWrapper
from scene.dataset_readers import readPACNeRFInfo
 
sam3d = SAM3DWrapper(
    config_path='${SAM3D_CONFIG}',
    sam3d_repo_path='${SAM3D_REPO}',
)
 
with open('${TRAIN_CONFIG}') as f:
    cfg = json.load(f)
 
scene_info = readPACNeRFInfo('${DATA_PATH}', '${TRAIN_CONFIG}', True)
ci = scene_info.train_cameras[0]
img_np = np.array(ci.image)
alpha = np.array(ci.alpha)
if alpha.ndim == 2: alpha = alpha[:,:,np.newaxis]
if alpha.max() <= 1.0: alpha = (alpha * 255).astype(np.uint8)
rgba = np.concatenate([img_np, alpha], axis=-1)
mask = (alpha.squeeze(-1) > 127).astype(np.uint8)
 
output = sam3d.run_inference(rgba, mask)
xyz_local = sam3d.extract_xyz_from_output(output)
 
output_dir = '${OUTPUT_DIR}'
output['gs'].save_ply(os.path.join(output_dir, 'sam3d_frame0.ply'))
 
src_min = xyz_local.min(0)[0]
src_range = xyz_local.max(0)[0] - src_min
target_min = torch.tensor(cfg['data']['xyz_min']).float().cuda()
target_max = torch.tensor(cfg['data']['xyz_max']).float().cuda()
target_range = target_max - target_min
margin = target_range * 0.05
 
torch.save({
    'xyz_world': None,
    'rescale_params': {
        'src_min': src_min.cpu(),
        'src_range': src_range.cpu(),
        'target_min': (target_min + margin).cpu(),
        'target_range': (target_range - 2*margin).cpu(),
    }
}, os.path.join(output_dir, 'sam3d_frame0_world.pt'))
print('Done! Saved sam3d_frame0.ply and sam3d_frame0_world.pt')
"
fi
 
# Render with SAM3D gaussians
python new_trajectory_sam3d.py \
    -c ${PREDICT_CONFIG} \
    -s ${DATA_PATH} \
    -m ${OUTPUT_DIR} \
    -vid 0 \
    -cid 0
 
echo ""
echo "============================================================"
echo "All done!"
echo "  Physical params: ${OUTPUT_DIR}/0-pred.json"
echo "  Video:           ${OUTPUT_DIR}/render_sam3d/output.mp4"
echo "  Frames:          ${OUTPUT_DIR}/render_sam3d/*.png"
echo "============================================================"