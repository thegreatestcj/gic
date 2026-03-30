#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 03:00:00
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -o logs/gic_metric_%j.out
#SBATCH -e logs/gic_metric_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam3d-objects
cd /orcd/home/002/fding/gic

rm -rf output/mvsam3d_torus

python train_dynamic_mvsam3d.py \
    -c config/pacnerf/torus.json \
    -s data/pacnerf/torus \
    -m output/mvsam3d_torus \
    --mvsam3d_dir /home/fding/MV-SAM3D/visualization \
    --camera_json data/mvsam3d_torus/cameras.json \
    --config_file config/pacnerf/torus.json \
    --predict_config config/predict/elastic.json
