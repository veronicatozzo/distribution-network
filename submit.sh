#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=500
#SBATCH --array=1-160
#SBATCH --time=1-00:00:00
srun $(head -n $SLURM_ARRAY_TASK_ID jobs.txt | tail -n 1)
