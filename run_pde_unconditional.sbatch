#!/usr/bin/bash
#SBATCH --job-name=pde_unconditional
#SBATCH --partition=a10
#SBATCH --account=lin491
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=pde_unconditional_%j.out
#SBATCH --error=pde_unconditional_%j.err

module load cuda/12.1.1 cudnn/9.2.0.82-12 conda
conda activate /depot/lin491/apps/torch_env

echo "🎨 Training Unconditional PDE Generator (as Amir requested)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Create necessary directories
mkdir -p models
mkdir -p results/pde

# Run unconditional generation training
python pde_examples/fm_pde_unconditional.py

echo ""
echo "✅ Unconditional PDE generator training completed!"
echo "📁 Check results in: results/pde/"
echo "🎯 Model saved to: models/pde_unconditional_generator.pth"
echo ""
echo "💡 This model learns to generate realistic PDE solution pairs from noise."
echo "   Ready for inverse problems and inpainting as Amir suggested!"
