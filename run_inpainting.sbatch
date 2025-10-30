#!/usr/bin/bash
#SBATCH --job-name=fm_inpainting
#SBATCH --partition=a10
#SBATCH --account=lin491
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        # Efficient for CIFAR-10 data loading
#SBATCH --gres=gpu:1
#SBATCH --mem=24G                # Sufficient for U-Net + CIFAR-10
#SBATCH --time=3:00:00           # 50 epochs should complete in ~2.5 hours
#SBATCH --output=fm_inpainting_%j.out
#SBATCH --error=fm_inpainting_%j.err

module load cuda/12.1.1 cudnn/9.2.0.82-12 conda
conda activate /depot/lin491/apps/torch_env

python conditional/fm_cifar_inpainting.py