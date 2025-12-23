#!/usr/bin/bash
#SBATCH --job-name=fm_pde_complete
#SBATCH --partition=a10
#SBATCH --account=lin491
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=fm_pde_complete_%j.out
#SBATCH --error=fm_pde_complete_%j.err

module load cuda/12.1.1 cudnn/9.2.0.82-12 conda
conda activate /depot/lin491/apps/torch_env

# Create necessary directories
mkdir -p models
mkdir -p results/pde

echo "🎯 Starting Complete PDE Flow Matching Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Step 1: Generate sample PDE data (no dependencies needed)
echo "📊 Generating sample PDE data..."
python generate_sample_pde_data.py

# Step 2: Verify data was created
echo "🔍 Verifying data creation..."
if [ -f "test_data64_cal/dataset_2d_00000_diff.npz" ]; then
    echo "✅ Sample data generated successfully"
    ls -la test_data64_cal/ | head -5
else
    echo "❌ Data generation failed!"
    exit 1
fi

# Step 3: Train Flow Matching on 2D diffusion PDE
echo "🏋️ Training Flow Matching on 2D Diffusion PDE..."
python pde_examples/fm_diffusion_2d_simple.py

echo "✅ Complete PDE Flow Matching pipeline finished!"
echo "Results saved to results/pde/"
echo "Model saved to models/pde_diffusion_2d.pth"

# List what was created
echo "📁 Created files:"
ls -la results/pde/
ls -la models/pde_*

echo "🎯 Pipeline Summary:"
echo "- Generated 50 diffusion PDE samples"
echo "- Trained Flow Matching model (2D U-Net)"
echo "- Created visualizations and saved model"
echo "- Ready for inference and analysis!"