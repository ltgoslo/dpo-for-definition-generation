#!/bin/bash -e
#SBATCH --job-name=greenwashingud
#SBATCH --account=nn9851k
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --partition=accel
#SBATCH --mem=0
#SBATCH --time=5:00:00
#SBATCH --output=/cluster/work/projects/nn9851k/mariiaf/greenwashing-ud-%j.out

SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_arm_numpy2.sif"
CMD="python3 ud_parsing.py"
srun apptainer exec --nv -B /cluster/work/projects/:/cluster/work/projects/ $SIF $CMD
