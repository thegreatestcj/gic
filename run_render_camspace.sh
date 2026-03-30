#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 01:30:00
#SBATCH -G 1
#SBATCH --mem=16G
#SBATCH -o logs/render_camspace_%j.out
#SBATCH -e logs/render_camspace_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam3d-objects
cd /orcd/home/002/fding/gic

python render_camera_space.py \
    --mvsam3d_dir /home/fding/MV-SAM3D/visualization \
    --output_dir output/camera_space_render
