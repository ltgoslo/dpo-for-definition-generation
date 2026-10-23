#!/bin/bash -e
#SBATCH --job-name=merge_lora
#SBATCH --account=nn9851k
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=small
#SBATCH --mem-per-cpu=18G
#SBATCH --time=0:05:00
#SBATCH --output=/cluster/work/projects/nn9851k/mariiaf/merge_lora-%j.out

SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_amd_nlpl.sif"
CMD="python3 merge_lora.py"
srun apptainer exec --nv -B /cluster/work/projects/:/cluster/work/projects/ $SIF $CMD ${@}
