#!/usr/bin/bash
#SBATCH --job-name=fm_pde_simple
#SBATCH --partition=a10
#SBATCH --account=lin491
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=fm_pde_simple_%j.out
#SBATCH --error=fm_pde_simple_%j.err

module load cuda/12.1.1 cudnn/9.2.0.82-12 conda
conda activate /depot/lin491/apps/torch_env

# Create necessary directories
mkdir -p models
mkdir -p results/pde

echo "🎯 Starting Simple PDE Flow Matching Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Check if Amir's data exists
echo "📊 Checking for Amir's PDE data..."
if [ -d "train_data64" ] && [ "$(ls -A train_data64/*_diff.npz 2>/dev/null)" ]; then
    echo "✅ Found Amir's data in train_data64/"
elif [ -d "test_data64_cal" ] && [ "$(ls -A test_data64_cal/*_diff.npz 2>/dev/null)" ]; then
    echo "✅ Found Amir's data in test_data64_cal/"
elif [ -d "test_data64" ] && [ "$(ls -A test_data64/*_diff.npz 2>/dev/null)" ]; then
    echo "✅ Found Amir's data in test_data64/"
else
    echo "❌ No Amir's PDE data found!"
    echo "Please copy Amir's generated .npz files to one of these folders:"
    echo "  - train_data64/"
    echo "  - test_data64_cal/"
    echo "  - test_data64/"
    exit 1
fi

# Train Flow Matching on 2D diffusion PDE (simplified version)
echo "🏋️ Training Flow Matching on 2D Diffusion PDE..."
python pde_examples/fm_diffusion_2d_simple.py

echo "✅ PDE Flow Matching training completed!"
echo "Results saved to results/pde/"
echo "Model saved to models/pde_diffusion_2d.pth"

# List what was created
echo "📁 Created files:"
ls -la results/pde/
ls -la models/pde_*