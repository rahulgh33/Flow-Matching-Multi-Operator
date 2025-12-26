#!/usr/bin/bash
#SBATCH --job-name=test_pde_fm
#SBATCH --partition=a10
#SBATCH --account=lin491
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=test_pde_fm_%j.out
#SBATCH --error=test_pde_fm_%j.err

module load cuda/12.1.1 cudnn/9.2.0.82-12 conda
conda activate /depot/lin491/apps/torch_env

echo "🧪 Testing Trained PDE Flow Matching Model"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Run test
python pde_examples/test_trained_model.py

echo ""
echo "✅ Testing completed!"
echo "📁 Check results in: results/pde/testing/"
