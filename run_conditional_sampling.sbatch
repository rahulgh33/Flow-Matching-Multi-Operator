#!/usr/bin/bash
#SBATCH --job-name=fm_conditional
#SBATCH --partition=a10
#SBATCH --account=lin491
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        # Increased for data loading
#SBATCH --gres=gpu:1
#SBATCH --mem=16G                # Increased for model + data
#SBATCH --time=4:00:00           # Conservative: 9 generations × ~20min each
#SBATCH --output=fm_conditional_%j.out
#SBATCH --error=fm_conditional_%j.err

module load cuda/12.1.1 cudnn/9.2.0.82-12
module load anaconda
source activate /depot/lin491/apps/torch_env

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Run Bayesian conditional sampling
python conditional/fm_cifar_conditional_sampling.py

# Print completion
echo "Job completed at: $(date)"
