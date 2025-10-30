#!/usr/bin/bash
#SBATCH --job-name=fm_unet_cifar
#SBATCH --partition=a10
#SBATCH --account=lin491
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6        # Reduced from 12 - plenty for CIFAR-10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G                # Reduced from 48G - sufficient for U-Net
#SBATCH --time=5:00:00           # Reduced from 6:00:00 - 100 epochs ~4-4.5 hours
#SBATCH --output=fm_unet_cifar_%j.out
#SBATCH --error=fm_unet_cifar_%j.err

module load cuda/12.1.1 cudnn/9.2.0.82-12 conda
conda activate /depot/lin491/apps/torch_env

python conditional/fm_cifar_conditional.py