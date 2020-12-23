#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1GB
#SBATCH --array=1-12
#SBATCH --time=1-00:00:00
srun $(head -n $SLURM_ARRAY_TASK_ID jobs.txt | tail -n 1)
