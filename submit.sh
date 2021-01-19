#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=2G
#SBATCH --array=1
#SBATCH --exclude=hpc1,hpc2,hpc3,hpc4,hpc5,hpc6,hpc7,hpc8,hpc9,vine3,vine4,vine6,vine11,vine12,rose7,rose8,rose9,lion17,lion3
#SBATCH --gres=gpu:1
#SBATCH --time=0-10:00:00

module load cuda-10.2

srun $(head -n $SLURM_ARRAY_TASK_ID jobs_setttr.txt | tail -n 1)
