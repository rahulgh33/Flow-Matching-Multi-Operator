#!/usr/bin/bash
#SBATCH --job-name=fm_unet_extended
#SBATCH --partition=a10
#SBATCH --account=lin491
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6        # Efficient for CIFAR-10 data loading
#SBATCH --gres=gpu:1
#SBATCH --mem=32G                # Sufficient for U-Net + CIFAR-10
#SBATCH --time=8:00:00           # Extended time for 150 epochs + fine-tuning
#SBATCH --output=fm_unet_extended_%j.out
#SBATCH --error=fm_unet_extended_%j.err

module load cuda/12.1.1 cudnn/9.2.0.82-12 conda
conda activate /depot/lin491/apps/torch_env

python conditional/fm_cifar_conditional_extended.py