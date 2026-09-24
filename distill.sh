#!/bin/bash
#SBATCH --time 1-12:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition maxgpu
#SBATCH --constraint="GPUx1&A100-PCIE-80GB"

#SBATCH --job-name PointCountFM-distill
#SBATCH --output /data/dust/user/mmozzani/PointCountFM_private/PointCountFM-distill-%j.out

# usage: sbatch distill.sh results/<run>/conf.yaml
# Distils a trained CNF into a 1-step student. The config carries result_path, so the
# trainer resumes: training is a no-op for a finished run, then it generates the
# teacher samples (new_samples.h5, 200 Heun steps, train+val conditions) if they are
# not there yet and trains the student on them. Re-submitting resumes from
# checkpoint_distill.pt and reuses new_samples.h5.

echo "SLURM nodelist: $SLURM_NODELIST"
echo "start time: $(date)"
nvidia-smi --query-gpu=index,name --format=csv,noheader

source ~/.bashrc
cd /data/dust/user/mmozzani/PointCountFM_private/ || exit
source .venv/bin/activate

config=${1:?usage: sbatch distill.sh results/<run>/conf.yaml}
echo "config: $config"
python pointcountfm/trainer.py "$config" --distill

echo "end time: $(date)"
