#!/bin/bash
#SBATCH --job-name=gpt_fm_cifar
#SBATCH --output=logs/gpt_fm_cifar_%j.out
#SBATCH --error=logs/gpt_fm_cifar_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules (adjust for your cluster)
module load python/3.9
module load cuda/11.8

# Activate virtual environment (adjust path as needed)
# source /path/to/your/venv/bin/activate

echo "🎯 Starting GPT-Optimized Flow Matching Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"

# Check GPU availability
nvidia-smi

# Set environment variables for better performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Run the training
echo "🚀 Starting training with GPT optimizations..."
python conditional/fm_cifar_conditional.py

echo "✅ Training completed at $(date)"
echo "📁 Check generated images in current directory"
echo "📊 Key files to examine:"
echo "   - gpt_test_15steps_target.png (our 15-step target)"
echo "   - gpt_test_100steps_baseline.png (diffusion comparison)"
echo "   - gpt_final_optimized.png (best quality showcase)"
echo "   - gpt_optimized_*steps_*.png (speed benchmarks)"

# Optional: Copy important results to a results directory
mkdir -p results/gpt_optimized_$(date +%Y%m%d_%H%M)
cp gpt_*.png results/gpt_optimized_$(date +%Y%m%d_%H%M)/
cp specific_cifar_classes.png results/gpt_optimized_$(date +%Y%m%d_%H%M)/

echo "🏆 Results saved to results/gpt_optimized_$(date +%Y%m%d_%H%M)/"