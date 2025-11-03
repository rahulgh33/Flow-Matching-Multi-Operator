#!/usr/bin/bash
#SBATCH --job-name=fm_generate
#SBATCH --partition=a10
#SBATCH --account=lin491
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2        # Minimal CPU for generation
#SBATCH --gres=gpu:1
#SBATCH --mem=8G                 # Much less memory needed for generation
#SBATCH --time=1:00:00           # Generation is fast
#SBATCH --output=fm_generate_%j.out
#SBATCH --error=fm_generate_%j.err

module load cuda/12.1.1 cudnn/9.2.0.82-12 conda
conda activate /depot/lin491/apps/torch_env

# Generate samples with different step counts
python generate_samples.py --model models/fm_unet_extended_final.pth --steps 15 25 35 50