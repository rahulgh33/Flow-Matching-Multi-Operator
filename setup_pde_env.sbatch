#!/usr/bin/bash
#SBATCH --job-name=setup_pde_env
#SBATCH --partition=a10
#SBATCH --account=lin491
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=setup_pde_env_%j.out
#SBATCH --error=setup_pde_env_%j.err

module load cuda/12.1.1 cudnn/9.2.0.82-12 conda

echo "🏗️ Setting up PDE environment with all dependencies"
echo "Job ID: $SLURM_JOB_ID"

# Create new conda environment with all needed packages
echo "📦 Creating conda environment with PDE dependencies..."
conda create -n pde_env python=3.9 -y

# Activate the new environment
source activate pde_env

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install scientific computing packages
echo "🧮 Installing scientific packages..."
conda install numpy matplotlib scipy -y

# Install PDE-specific packages
echo "🌊 Installing PDE packages..."
conda install -c conda-forge fipy tqdm -y

# Try to install CuPy for GPU acceleration
echo "🚀 Installing CuPy for GPU acceleration..."
conda install -c conda-forge cupy -y || pip install cupy-cuda12x

# Install additional useful packages
conda install -c conda-forge jupyterlab ipython -y

echo "✅ PDE environment setup completed!"
echo "Environment location: $(conda info --envs | grep pde_env)"

# Test the installation
echo "🧪 Testing installations..."
python -c "
import torch; print(f'PyTorch: {torch.__version__}')
import numpy; print(f'NumPy: {numpy.__version__}')
import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')
try:
    import fipy; print('✅ FiPy available')
except: print('❌ FiPy not available')
try:
    import tqdm; print('✅ tqdm available')
except: print('❌ tqdm not available')
try:
    import cupy; print('✅ CuPy available (GPU acceleration)')
except: print('⚠️  CuPy not available (CPU only)')
print(f'CUDA available: {torch.cuda.is_available()}')
"

echo "🎯 To use this environment in future jobs:"
echo "source activate pde_env"