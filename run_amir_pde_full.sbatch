#!/usr/bin/bash
#SBATCH --job-name=fm_amir_pde_full
#SBATCH --partition=a10
#SBATCH --account=lin491
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=fm_amir_pde_full_%j.out
#SBATCH --error=fm_amir_pde_full_%j.err

module load cuda/12.1.1 cudnn/9.2.0.82-12 conda

# Use your own PDE environment (create with setup_pde_env.sbatch first)
source activate pde_env

# Create necessary directories
mkdir -p models
mkdir -p results/pde
mkdir -p train_data64
mkdir -p test_data64_cal

echo "🎯 Starting Amir's Full PDE Flow Matching Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Dependencies should already be installed in pde_env
echo "🔍 Checking environment setup..."
python -c "
try:
    import fipy; print('✅ FiPy available')
except: print('❌ FiPy not available - run setup_pde_env.sbatch first')
try:
    import tqdm; print('✅ tqdm available')  
except: print('❌ tqdm not available - run setup_pde_env.sbatch first')
try:
    import cupy; print('✅ CuPy available (GPU acceleration)')
except: print('⚠️  CuPy not available (CPU fallback)')
"

# Step 2: Generate high-quality PDE data using Amir's method
echo "🏭 Generating PDE data using Amir's method..."
python pde_examples/amir_data_generator.py

# Step 3: Prepare and analyze the data
echo "🔍 Preparing and analyzing data..."
python pde_examples/prepare_amir_data.py

# Step 4: Train Flow Matching on the high-quality data
echo "🏋️ Training Flow Matching on Amir's PDE data..."
python pde_examples/fm_diffusion_2d.py

echo "✅ Amir's Full PDE Flow Matching pipeline completed!"
echo "Results saved to results/pde/"
echo "Model saved to models/pde_diffusion_2d.pth"

# List what was created
echo "📁 Created files:"
ls -la results/pde/
ls -la models/pde_*
ls -la test_data64_cal/ | head -5

echo "🎯 Pipeline Summary:"
echo "- Used Amir's high-quality PDE solver (fipy + GPU acceleration)"
echo "- Generated 100 samples of diffusion/advection/advection-diffusion"
echo "- Trained Flow Matching model with 2D U-Net architecture"
echo "- Created comprehensive visualizations and analysis"
echo "- Ready for multi-operator applications and inverse problems!"