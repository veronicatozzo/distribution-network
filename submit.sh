#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --mem=15GB
#SBATCH --array=1-36
#SBATCH --gres=gpu:1
#SBATCH --time=0-10:00:00
srun $(head -n $SLURM_ARRAY_TASK_ID jobs.txt | tail -n 1)
