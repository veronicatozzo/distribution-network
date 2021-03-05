#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=6G
#SBATCH --array=1-20
#SBATCH --gres=gpu:1
#SBATCH --exclude=hpc1,hpc2,hpc3,hpc4,hpc5,hpc6,hpc7,hpc8,hpc9,vine3,vine4,vine6,vine11,vine12,rose7,rose8,rose9,lion17,lion3,rose1
#SBATCH --time=1-10:00:00
#SBATCH --constraint="pascal|turing|volta|maxwell"

module load cuda-10.2

srun $(wandb agent lily/synthetic-moments2/za2zq2ud; wandb agent lily/synthetic-moments2/za2zq2ud; wandb agent lily/synthetic-moments2/za2zq2ud; wandb agent lily/synthetic-moments2/za2zq2ud; wandb agent lily/synthetic-moments2/za2zq2ud; wandb agent lily/synthetic-moments2/za2zq2ud; wandb agent lily/synthetic-moments2/za2zq2ud)
