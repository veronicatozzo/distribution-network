#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10GB
#SBATCH --array=40-49
#SBATCH --time=1-00:00:00
srun $(head -n $SLURM_ARRAY_TASK_ID jobs3.txt | tail -n 1)
