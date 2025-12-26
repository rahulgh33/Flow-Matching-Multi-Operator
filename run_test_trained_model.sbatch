#!/bin/bash
#SBATCH --job-name=test_pde_fm
#SBATCH --output=test_pde_fm_%j.out
#SBATCH --error=test_pde_fm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -A standby

echo "🧪 Testing Trained PDE Flow Matching Model"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Activate environment
source /depot/lin491/apps/torch_env/bin/activate

# Run test
python pde_examples/test_trained_model.py

echo ""
echo "✅ Testing completed!"
echo "📁 Check results in: results/pde/testing/"
