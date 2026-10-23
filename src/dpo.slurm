#!/bin/bash -e
#SBATCH --job-name=dpo
#SBATCH --account=nn9851k
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=accel
#SBATCH --mem=0
#SBATCH --time=10:00:00

echo "--Node: $(hostname)"

# Set proxy settings for HTTP and HTTPS traffic
export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/

export WANDB_PROJECT=defgen_olmo

SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_arm_nlpl.sif"

export SINGULARITYENV_TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib.real"
export APPTAINERENV_TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib.real"

CMD="python3 dpo.py"

apptainer exec --nv -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF $CMD ${@}