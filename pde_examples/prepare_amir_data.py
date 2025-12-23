"""
Prepare and Analyze Amir's PDE Data for Flow Matching

This script helps you:
1. Generate PDE data using Amir's method
2. Check if data exists and is properly formatted
3. Visualize sample PDE solutions
4. Analyze data statistics
5. Verify data format compatibility with Flow Matching
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm


def check_data_folder(folder_path):
    """Check what PDE data files exist in the folder."""
    print(f"🔍 Checking data folder: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder {folder_path} does not exist!")
        return False, {}
    
    # Count files by PDE type
    diff_files = glob.glob(os.path.join(folder_path, "*_diff.npz"))
    adve_files = glob.glob(os.path.join(folder_path, "*_adve.npz"))
    advdiff_files = glob.glob(os.path.join(folder_path, "*_adve_diff.npz"))
    
    file_counts = {
        'diffusion': len(diff_files),
        'advection': len(adve_files),
        'advection_diffusion': len(advdiff_files)
    }
    
    print(f"Found files:")
    print(f"  - Diffusion: {file_counts['diffusion']} files")
    print(f"  - Advection: {file_counts['advection']} files")
    print(f"  - Advection-Diffusion: {file_counts['advection_diffusion']} files")
    
    if file_counts['diffusion'] == 0:
        print("❌ No diffusion files found!")
        return False, file_counts
    
    print("✅ Data folder looks good!")
    return True, file_counts


def verify_data_format(folder_path, sample_idx=0):
    """Verify that data files have the expected format."""
    print(f"\n🔬 Verifying data format...")
    
    diff_file = os.path.join(folder_path, f"dataset_2d_{sample_idx:05d}_diff.npz")
    
    if not os.path.exists(diff_file):
        print(f"❌ Sample file {diff_file} not found!")
        return False
    
    try:
        data = np.load(diff_file)
        
        # Check required fields
        required_fields = ['init_value', 'last_value', 'diffusivity', 'label']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            print(f"❌ Missing fields: {missing_fields}")
            return False
        
        # Check data shapes
        init_shape = data['init_value'].shape
        final_shape = data['last_value'].shape
        
        print(f"Data format verification:")
        print(f"  - Initial condition shape: {init_shape}")
        print(f"  - Final state shape: {final_shape}")
        print(f"  - Diffusivity: {data['diffusivity']:.4f}")
        print(f"  - Label: {data['label']}")
        
        # Check if shapes are compatible with 64x64
        expected_size = 64 * 64
        if len(init_shape) == 1 and init_shape[0] == expected_size:
            print("  - ✅ Data is flattened 64x64 format")
        elif len(init_shape) == 2 and init_shape == (64, 64):
            print("  - ✅ Data is 2D 64x64 format")
        else:
            print(f"  - ⚠️  Unexpected shape: {init_shape}")
        
        print("✅ Data format verification passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


def visualize_sample_data(folder_path, sample_idx=0):
    """Visualize a sample from each PDE type."""
    print(f"\n📊 Visualizing sample data (index {sample_idx})...")
    
    # File paths
    diff_file = os.path.join(folder_path, f"dataset_2d_{sample_idx:05d}_diff.npz")
    adve_file = os.path.join(folder_path, f"dataset_2d_{sample_idx:05d}_adve.npz")
    advdiff_file = os.path.join(folder_path, f"dataset_2d_{sample_idx:05d}_adve_diff.npz")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    pde_types = [
        ("Diffusion", diff_file),
        ("Advection", adve_file),
        ("Advection-Diffusion", advdiff_file)
    ]
    
    for i, (pde_name, file_path) in enumerate(pde_types):
        if not os.path.exists(file_path):
            print(f"⚠️  {file_path} not found, skipping {pde_name}")
            # Fill with empty plots
            for j in range(3):
                axes[i, j].text(0.5, 0.5, f"{pde_name}\nNot Available", 
                              ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
            continue
        
        try:
            # Load data
            data = np.load(file_path)
            init_state = data['init_value']
            final_state = data['last_value']
            
            # Reshape if needed
            if len(init_state.shape) == 1:
                init_state = init_state.reshape(64, 64)
                final_state = final_state.reshape(64, 64)
            
            # Plot initial condition
            im0 = axes[i, 0].imshow(init_state, origin="lower", cmap="viridis")
            axes[i, 0].set_title(f"{pde_name} - Initial")
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            # Plot final state
            im1 = axes[i, 1].imshow(final_state, origin="lower", cmap="viridis")
            axes[i, 1].set_title(f"{pde_name} - Final")
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
            
            # Plot difference
            diff = final_state - init_state
            im2 = axes[i, 2].imshow(diff, origin="lower", cmap="RdBu_r")
            axes[i, 2].set_title(f"{pde_name} - Change")
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
            
            # Print parameters
            if 'diffusivity' in data and data['diffusivity'] > 0:
                print(f"  {pde_name} - Diffusivity: {data['diffusivity']:.4f}")
            if 'velocity_x' in data and abs(data['velocity_x']) > 1e-6:
                print(f"  {pde_name} - Velocity: ({data['velocity_x']:.2f}, {data['velocity_y']:.2f})")
        
        except Exception as e:
            print(f"⚠️  Error loading {pde_name}: {e}")
            for j in range(3):
                axes[i, j].text(0.5, 0.5, f"{pde_name}\nError: {str(e)[:20]}...", 
                              ha='center', va='center', transform=axes[i, j].transAxes)
    
    plt.tight_layout()
    
    # Create results directory
    os.makedirs("results/pde", exist_ok=True)
    save_path = "results/pde/sample_data_visualization.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved visualization to {save_path}")


def analyze_data_statistics(folder_path, max_samples=100):
    """Analyze statistics of the PDE data."""
    print(f"\n📈 Analyzing data statistics (max {max_samples} samples)...")
    
    diff_files = sorted(glob.glob(os.path.join(folder_path, "*_diff.npz")))
    
    if len(diff_files) == 0:
        print("❌ No diffusion files found for analysis!")
        return
    
    print(f"Analyzing {min(len(diff_files), max_samples)} diffusion files...")
    
    diffusivities = []
    init_stats = []
    final_stats = []
    
    for file_path in tqdm(diff_files[:max_samples], desc="Analyzing files"):
        try:
            data = np.load(file_path)
            
            init_state = data['init_value']
            final_state = data['last_value']
            
            # Flatten if needed
            if len(init_state.shape) > 1:
                init_state = init_state.flatten()
                final_state = final_state.flatten()
            
            diffusivities.append(data['diffusivity'])
            init_stats.append([init_state.min(), init_state.max(), init_state.mean(), init_state.std()])
            final_stats.append([final_state.min(), final_state.max(), final_state.mean(), final_state.std()])
        
        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")
            continue
    
    if len(diffusivities) == 0:
        print("❌ No valid data found for analysis!")
        return
    
    diffusivities = np.array(diffusivities)
    init_stats = np.array(init_stats)
    final_stats = np.array(final_stats)
    
    print(f"\nDiffusivity statistics:")
    print(f"  Range: [{diffusivities.min():.4f}, {diffusivities.max():.4f}]")
    print(f"  Mean: {diffusivities.mean():.4f} ± {diffusivities.std():.4f}")
    
    print(f"\nInitial state statistics:")
    print(f"  Value range: [{init_stats[:, 0].min():.4f}, {init_stats[:, 1].max():.4f}]")
    print(f"  Mean values: {init_stats[:, 2].mean():.4f} ± {init_stats[:, 2].std():.4f}")
    
    print(f"\nFinal state statistics:")
    print(f"  Value range: [{final_stats[:, 0].min():.4f}, {final_stats[:, 1].max():.4f}]")
    print(f"  Mean values: {final_stats[:, 2].mean():.4f} ± {final_stats[:, 2].std():.4f}")
    
    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(diffusivities, bins=20, alpha=0.7, color='blue')
    axes[0].set_title("Diffusivity Distribution")
    axes[0].set_xlabel("Diffusivity")
    axes[0].set_ylabel("Count")
    
    axes[1].hist(init_stats[:, 2], bins=20, alpha=0.7, label="Initial", color='green')
    axes[1].hist(final_stats[:, 2], bins=20, alpha=0.7, label="Final", color='red')
    axes[1].set_title("Mean Value Distribution")
    axes[1].set_xlabel("Mean Value")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    
    axes[2].hist(init_stats[:, 3], bins=20, alpha=0.7, label="Initial", color='green')
    axes[2].hist(final_stats[:, 3], bins=20, alpha=0.7, label="Final", color='red')
    axes[2].set_title("Standard Deviation Distribution")
    axes[2].set_xlabel("Std Dev")
    axes[2].set_ylabel("Count")
    axes[2].legend()
    
    plt.tight_layout()
    save_path = "results/pde/data_statistics.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved statistics to {save_path}")


def generate_data_if_missing():
    """Generate PDE data if it doesn't exist."""
    print("\n🏭 Checking if data generation is needed...")
    
    # Check if any data folders exist
    data_folders = ["train_data64", "test_data64_cal", "test_data64"]
    found_data = False
    
    for folder in data_folders:
        if os.path.exists(folder):
            diff_files = glob.glob(os.path.join(folder, "*_diff.npz"))
            if len(diff_files) > 0:
                print(f"✅ Found existing data in {folder}")
                found_data = True
                break
    
    if not found_data:
        print("❌ No PDE data found! Generating data...")
        
        try:
            # Import and run the data generator
            from amir_data_generator import generate_pde_data
            generate_pde_data(num_datasets=100, train_split=0.9, use_gpu=True)
            print("✅ Data generation completed!")
            return True
        except ImportError:
            print("❌ Could not import data generator!")
            print("Please run: python pde_examples/amir_data_generator.py")
            return False
        except Exception as e:
            print(f"❌ Error generating data: {e}")
            return False
    
    return True


def main():
    """Main data preparation pipeline."""
    print("🔍 Preparing Amir's PDE data for Flow Matching...\n")
    
    # Create results directory
    os.makedirs("results/pde", exist_ok=True)
    
    # Step 1: Check if data needs to be generated
    if not generate_data_if_missing():
        return
    
    # Step 2: Find data folder
    data_folders = ["train_data64", "test_data64_cal", "test_data64"]
    found_folder = None
    
    for folder in data_folders:
        success, file_counts = check_data_folder(folder)
        if success and file_counts['diffusion'] > 0:
            found_folder = folder
            break
    
    if found_folder is None:
        print("\n❌ No valid data found! Please:")
        print("1. Run: python pde_examples/amir_data_generator.py")
        print("2. Or run Amir's original data generation script")
        print("3. Make sure .npz files are created successfully")
        return
    
    print(f"\n✅ Using data from: {found_folder}")
    
    # Step 3: Verify data format
    if not verify_data_format(found_folder):
        print("❌ Data format verification failed!")
        return
    
    # Step 4: Visualize sample data
    print("\n📊 Visualizing sample data...")
    try:
        visualize_sample_data(found_folder, sample_idx=0)
    except Exception as e:
        print(f"⚠️  Could not visualize data: {e}")
    
    # Step 5: Analyze statistics
    print("\n📈 Analyzing data statistics...")
    try:
        analyze_data_statistics(found_folder, max_samples=50)
    except Exception as e:
        print(f"⚠️  Could not analyze statistics: {e}")
    
    print("\n✅ Data preparation completed!")
    print("\n🚀 Next steps:")
    print("1. Run: python pde_examples/fm_diffusion_2d.py")
    print("2. Or submit: sbatch run_pde_diffusion.sbatch")
    print("3. This will train Flow Matching on the diffusion PDE data")
    print("\n💡 Expected results:")
    print("- Fast PDE solution (50 steps vs numerical solver)")
    print("- Learns initial condition → final state mapping")
    print("- Perfect for inverse problems and multi-operator learning!")


if __name__ == "__main__":
    main()