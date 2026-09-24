#!/bin/bash
#SBATCH --time 1-12:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition maxgpu
##SBATCH --constraint="GPUx1&A100-PCIE-80GB"

#SBATCH --constraint="GPUx4"
#SBATCH --job-name PointCountFM-train
#SBATCH --output /data/dust/user/mmozzani/PointCountFM_private/PointCountFM-train-%j.out

# usage: sbatch train.sh [config]
# To resume a run that hit the time limit, pass its results/<run>/conf.yaml: it carries
# result_path, so the trainer picks up checkpoint.pt instead of starting a new run.

echo "SLURM nodelist: $SLURM_NODELIST"
echo "start time: $(date)"
nvidia-smi --query-gpu=index,name --format=csv,noheader

source ~/.bashrc
cd /data/dust/user/mmozzani/PointCountFM_private/ || exit
source .venv/bin/activate

config=${1:-config/config_T_meanflow.yaml}
echo "config: $config"
python pointcountfm/trainer.py "$config"

echo "end time: $(date)"
