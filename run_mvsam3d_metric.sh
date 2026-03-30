#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 04:00:00
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -o logs/mvsam3d_metric_%j.out
#SBATCH -e logs/mvsam3d_metric_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam3d-objects

GIC_DIR="/orcd/home/002/fding/gic"
MVSAM3D_DIR="/home/fding/MV-SAM3D"
MVSAM3D_DATA="data/mvsam3d_torus"
METRIC_DATA="data/mvsam3d_torus_metric"
IMAGE_NAMES="0,1,2,3,4,5,6,7,8,9,10"
mkdir -p ${GIC_DIR}/logs

cd ${GIC_DIR}

echo "Step 2: Generate metric pointmaps"
python prepare_metric_pointmaps.py \
    --data_path data/pacnerf/torus \
    --config_path config/pacnerf/torus.json \
    --mvsam3d_data_dir ${MVSAM3D_DATA} \
    --output_dir ${METRIC_DATA}

echo "Step 3: Run MV-SAM3D with metric pointmaps"
cd ${MVSAM3D_DIR}

for frame_dir in ${GIC_DIR}/${MVSAM3D_DATA}/frame_*; do
    frame_name=$(basename ${frame_dir})
    da3_npz="${GIC_DIR}/${METRIC_DATA}/${frame_name}/da3_output.npz"

    if [ ! -f "${da3_npz}" ]; then
        echo "[SKIP] ${frame_name}: no metric pointmap"
        continue
    fi

    echo "Processing: ${frame_name}"
    python run_inference_weighted.py \
        --input_path ${frame_dir} \
        --mask_prompt torus \
        --image_names ${IMAGE_NAMES} \
        --no_stage1_weighting --no_stage2_weighting \
        --da3_output ${da3_npz} \
        --seed 42 || echo "FAILED: ${frame_name}"

done

echo "Done."
