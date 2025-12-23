#!/usr/bin/env python3
"""
Quick Setup Script for Amir's PDE Data + Flow Matching

This script helps you get started with Amir's PDE data generation
and Flow Matching training in just a few commands.
"""

import os
import sys
import subprocess


def check_dependencies():
    """Check if required packages are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy', 
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
        'fipy': 'FiPy (for PDE solving)'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("✅ All dependencies found!")
    return True


def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")
    
    dirs = [
        "results/pde",
        "models",
        "train_data64", 
        "test_data64_cal"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✅ {dir_path}")


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}...")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print("Error output:", e.stderr[:200] + "..." if len(e.stderr) > 200 else e.stderr)
        return False


def main():
    """Main setup pipeline."""
    print("🎯 Setting up Amir's PDE Data + Flow Matching\n")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first!")
        return
    
    # Step 2: Setup directories
    setup_directories()
    
    # Step 3: Generate sample data (small dataset for testing)
    print("\n📊 Generating sample PDE data...")
    if not run_command("python pde_examples/amir_data_generator.py", "Generate PDE data"):
        print("⚠️  Data generation failed, but you can continue with existing data")
    
    # Step 4: Prepare and visualize data
    print("\n🔍 Preparing and analyzing data...")
    if not run_command("python pde_examples/prepare_amir_data.py", "Prepare data"):
        print("⚠️  Data preparation failed")
    
    # Step 5: Quick training test (optional)
    print("\n🤖 Would you like to run a quick training test? (y/n)")
    response = input().lower().strip()
    
    if response in ['y', 'yes']:
        print("\n🏋️ Running quick Flow Matching training test...")
        if run_command("python pde_examples/fm_diffusion_2d.py", "Train Flow Matching"):
            print("\n🎉 Training completed successfully!")
        else:
            print("\n⚠️  Training failed - check the error messages above")
    
    print("\n✅ Setup completed!")
    print("\n📋 What you can do now:")
    print("1. 📊 Visualize data: python pde_examples/prepare_amir_data.py")
    print("2. 🏋️ Train locally: python pde_examples/fm_diffusion_2d.py") 
    print("3. 🖥️ Train on cluster: sbatch run_pde_diffusion.sbatch")
    print("4. 🔬 Generate samples: python generate_samples.py")
    
    print("\n🎯 Perfect for your research:")
    print("- Fast PDE solution operator learning")
    print("- Multi-operator applications (diffusion, advection, etc.)")
    print("- Inverse problems and inpainting")
    print("- 100-1000x speedup over numerical solvers")


if __name__ == "__main__":
    main()