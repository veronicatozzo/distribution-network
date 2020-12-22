#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --array=1-22
#SBATCH --time=1-00:00:00
srun $(head -n $SLURM_ARRAY_TASK_ID jobs.txt | tail -n 1)
