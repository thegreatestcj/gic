#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 04:00:00
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -o logs/mvsam3d_gic_%j.out

# ============================================================
# MV-SAM3D + GIC Complete Pipeline
#
# Step 1: Prepare data (convert PAC-NeRF to MV-SAM3D format)
# Step 2: Run MV-SAM3D multi-view reconstruction for all frames
# Step 3: Run GIC physical parameter estimation
# Step 4: Render trajectory video
#
# Usage:
#   sbatch run_mvsam3d_pipeline.sh
#   bash run_mvsam3d_pipeline.sh
# ============================================================

et -e
 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam3d-objects
 
GIC_DIR="/orcd/home/002/fding/gic"
MVSAM3D_DIR="/home/fding/MV-SAM3D"
DATA_PATH="data/pacnerf/torus"
CONFIG_PATH="config/pacnerf/torus.json"
MVSAM3D_DATA="data/mvsam3d_torus"
METRIC_DATA="data/mvsam3d_torus_metric"
 
NUM_VIEWS=11
IMAGE_NAMES=$(python -c "print(','.join(str(i) for i in range(${NUM_VIEWS})))")
mkdir -p ${GIC_DIR}/logs
 
# ============================================================
# Step 1: Prepare MV-SAM3D data format (if not done)
# ============================================================
echo "============================================================"
echo "Step 1: Prepare MV-SAM3D data"
echo "============================================================"
 
cd ${GIC_DIR}
 
if [ ! -f "${MVSAM3D_DATA}/cameras.json" ]; then
    python prepare_mvsam3d_data.py \
        --data_path ${DATA_PATH} \
        --config_path ${CONFIG_PATH} \
        --output_dir ${MVSAM3D_DATA}
else
    echo "Data already prepared, skipping."
fi
 
# ============================================================
# Step 2: Generate metric pointmaps (MoGe + known cameras)
# ============================================================
echo ""
echo "============================================================"
echo "Step 2: Generate metric pointmaps"
echo "============================================================"
 
python prepare_metric_pointmaps.py \
    --data_path ${DATA_PATH} \
    --config_path ${CONFIG_PATH} \
    --mvsam3d_data_dir ${MVSAM3D_DATA} \
    --output_dir ${METRIC_DATA}
 
# ============================================================
# Step 3: Run MV-SAM3D with metric pointmaps
# ============================================================
echo ""
echo "============================================================"
echo "Step 3: Run MV-SAM3D with metric pointmaps"
echo "============================================================"
 
cd ${MVSAM3D_DIR}
 
for frame_dir in ${GIC_DIR}/${MVSAM3D_DATA}/frame_*; do
    frame_name=$(basename ${frame_dir})
    da3_npz="${GIC_DIR}/${METRIC_DATA}/${frame_name}/da3_output.npz"
 
    if ls visualization/${frame_name}/torus/*da3*/params.npz 1>/dev/null 2>&1; then
        echo "[SKIP] ${frame_name}: metric output exists"
        continue
    fi
 
    if [ ! -f "${da3_npz}" ]; then
        echo "[SKIP] ${frame_name}: no metric pointmap"
        continue
    fi
 
    echo "  Processing: ${frame_name} (with metric pointmaps)"
    python run_inference_weighted.py \
        --input_path ${frame_dir} \
        --mask_prompt torus \
        --image_names ${IMAGE_NAMES} \
        --no_stage1_weighting --no_stage2_weighting \
        --da3_output ${da3_npz} \
        --seed 42
 
done
 
echo ""
echo "============================================================"
echo "MV-SAM3D with metric pointmaps complete!"
echo "============================================================"