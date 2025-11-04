#!/usr/bin/bash
#SBATCH --job-name=fm_gen_existing
#SBATCH --partition=a10
#SBATCH --account=lin491
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=fm_gen_existing_%j.out
#SBATCH --error=fm_gen_existing_%j.err

module load cuda/12.1.1 cudnn/9.2.0.82-12 conda
conda activate /depot/lin491/apps/torch_env

python generate_from_existing.py