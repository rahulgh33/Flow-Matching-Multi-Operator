#!/usr/bin/env python3
"""
Install PDE Dependencies for Amir's Code
Run this once to set up the environment
"""

import subprocess
import sys


def run_command(cmd):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {cmd}")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Install required packages for Amir's PDE code."""
    
    print("📦 Installing PDE dependencies for Amir's code...")
    
    # Required packages
    packages = [
        "fipy",      # PDE solving library
        "tqdm",      # Progress bars
    ]
    
    # Optional GPU packages
    gpu_packages = [
        "cupy-cuda12x",  # GPU acceleration (adjust CUDA version as needed)
    ]
    
    success_count = 0
    
    # Install required packages
    for package in packages:
        if run_command(f"pip install --user {package}"):
            success_count += 1
    
    # Try to install GPU packages (optional)
    for package in gpu_packages:
        if run_command(f"pip install --user {package}"):
            print(f"🚀 GPU acceleration available with {package}")
        else:
            print(f"⚠️  {package} failed - will use CPU fallback")
    
    print(f"\n📊 Installation Summary:")
    print(f"Required packages: {success_count}/{len(packages)} installed")
    
    if success_count == len(packages):
        print("✅ All required dependencies installed!")
        print("You can now run Amir's PDE code:")
        print("  python pde_examples/amir_data_generator.py")
        print("  python pde_examples/prepare_amir_data.py")
        print("  python pde_examples/fm_diffusion_2d.py")
    else:
        print("❌ Some required packages failed to install")
        print("You may need to use the simplified version instead")
    
    # Test imports
    print("\n🧪 Testing imports...")
    
    try:
        import fipy
        print("✅ fipy imported successfully")
    except ImportError:
        print("❌ fipy import failed")
    
    try:
        import tqdm
        print("✅ tqdm imported successfully")
    except ImportError:
        print("❌ tqdm import failed")
    
    try:
        import cupy
        print("✅ cupy imported successfully (GPU acceleration available)")
    except ImportError:
        print("⚠️  cupy not available (will use CPU)")


if __name__ == "__main__":
    main()