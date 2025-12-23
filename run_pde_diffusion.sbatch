#!/usr/bin/bash
#SBATCH --job-name=fm_pde_diffusion
#SBATCH --partition=a10
#SBATCH --account=lin491
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=fm_pde_diffusion_%j.out
#SBATCH --error=fm_pde_diffusion_%j.err

module load cuda/12.1.1 cudnn/9.2.0.82-12 conda
conda activate /depot/lin491/apps/torch_env

# Create necessary directories
mkdir -p models
mkdir -p results/pde
mkdir -p train_data64
mkdir -p test_data64_cal

echo "🎯 Starting Amir's PDE Flow Matching Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Step 1: Generate PDE data if it doesn't exist
echo "📊 Checking for PDE data..."
if [ ! -f "train_data64/dataset_2d_00000_diff.npz" ]; then
    echo "Generating PDE data..."
    python pde_examples/amir_data_generator.py
else
    echo "PDE data already exists, skipping generation"
fi

# Step 2: Prepare and analyze data
echo "🔍 Preparing data..."
python pde_examples/prepare_amir_data.py

# Step 3: Train Flow Matching on 2D diffusion PDE
echo "🏋️ Training Flow Matching on 2D Diffusion PDE..."
python pde_examples/fm_diffusion_2d.py

echo "✅ PDE Flow Matching training completed!"
echo "Results saved to results/pde/"
echo "Model saved to models/pde_diffusion_2d.pth"