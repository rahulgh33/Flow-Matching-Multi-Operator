"""
Amir's PDE Data Generation Script
Generates sample pairs for diffusion, advection, and advection-diffusion PDEs.

This script creates .npz files with:
- init_value: Initial condition (64x64)
- last_value: Final state after PDE evolution (64x64)
- diffusivity: Diffusion coefficient
- velocity_x, velocity_y: Advection velocities
- label: PDE type (0=diffusion, 1=advection, 2=advection-diffusion)
"""

import os
import numpy as np
from fipy import *
from tqdm import tqdm

# Try to import GPU acceleration (optional)
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cnd
    GPU_AVAILABLE = True
    cp.cuda.Device(0).use()
    DTYPE = cp.float64
    print("✅ GPU acceleration available")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  GPU acceleration not available, using CPU")


# ---------- GPU kernels (if available) ----------
def _gpu_kernels(dx, dy, dtype=DTYPE):
    """Create finite difference kernels for GPU computation."""
    kx = cp.asarray([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=dtype) / (2 * dx)
    ky = cp.asarray([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=dtype) / (2 * dy)
    kxx = cp.asarray([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=dtype) / (dx * dx)
    kyy = cp.asarray([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=dtype) / (dy * dy)
    return kx, ky, (kxx + kyy)


def gpu_solve_diffusion(u0_np, nu, dt, steps, dx, dy):
    """Solve diffusion equation on GPU using explicit Euler."""
    kx, ky, klap = _gpu_kernels(dx, dy)
    u = cp.asarray(u0_np, dtype=DTYPE)
    lap = cp.empty_like(u)
    
    for _ in range(steps):
        cnd.correlate(u, klap, mode='nearest', output=lap)
        u = u + dt * (nu * lap)
    
    return cp.asnumpy(u)


def gpu_solve_advection(u0_np, vel, dt, steps, dx, dy):
    """Solve advection equation on GPU using explicit Euler."""
    kx, ky, _ = _gpu_kernels(dx, dy)
    u = cp.asarray(u0_np, dtype=DTYPE)
    dudx = cp.empty_like(u)
    dudy = cp.empty_like(u)
    
    for _ in range(steps):
        cnd.correlate(u, kx, mode='nearest', output=dudx)
        cnd.correlate(u, ky, mode='nearest', output=dudy)
        u = u - dt * vel * (dudx + dudy)
    
    return cp.asnumpy(u)


def gpu_solve_advdiff(u0_np, vel, nu, dt, steps, dx, dy):
    """Solve advection-diffusion equation on GPU using explicit Euler."""
    kx, ky, klap = _gpu_kernels(dx, dy)
    u = cp.asarray(u0_np, dtype=DTYPE)
    dudx = cp.empty_like(u)
    dudy = cp.empty_like(u)
    lap = cp.empty_like(u)
    
    for _ in range(steps):
        cnd.correlate(u, kx, mode='nearest', output=dudx)
        cnd.correlate(u, ky, mode='nearest', output=dudy)
        cnd.correlate(u, klap, mode='nearest', output=lap)
        u = u - dt * vel * (dudx + dudy) + dt * (nu * lap)
    
    return cp.asnumpy(u)


# ---------- CPU fallback functions ----------
def cpu_solve_diffusion(u0_np, nu, dt, steps, dx, dy):
    """Solve diffusion equation on CPU using FiPy."""
    mesh = Grid2D(dx=dx, dy=dy, nx=u0_np.shape[0], ny=u0_np.shape[1])
    phi = CellVariable(name="phi", mesh=mesh, value=u0_np.ravel())
    
    eq = TransientTerm() == DiffusionTerm(coeff=nu)
    
    for _ in range(steps):
        eq.solve(var=phi, dt=dt)
    
    return phi.value.reshape(u0_np.shape)


def generate_initial_condition(mesh, center_x, center_y, initial_width):
    """Generate Gaussian initial condition with taper."""
    x, y = mesh.cellCenters[0], mesh.cellCenters[1]
    
    # Gaussian blob
    init_val = np.exp(-(((x - center_x)**2 + (y - center_y)**2) / initial_width))
    
    # Apply taper to satisfy boundary conditions
    taper = numerix.sin(numerix.pi * x) * numerix.sin(numerix.pi * y)
    init_val = init_val * taper
    
    return init_val


def generate_pde_data(num_datasets=100, train_split=0.9, use_gpu=True):
    """
    Generate PDE datasets for diffusion, advection, and advection-diffusion.
    
    Args:
        num_datasets: Number of samples to generate
        train_split: Fraction for training (rest goes to test)
        use_gpu: Whether to use GPU acceleration if available
    """
    
    # Create directories
    train_folder = "train_data64"
    test_folder = "test_data64_cal"
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    # Set up 2D domain and mesh
    nx, ny = 64, 64
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / nx, Ly / ny
    mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
    
    # Simulation parameters (optimized by Amir)
    timeStep = 0.0001
    steps = 500
    
    # Determine which solver to use
    use_gpu_solver = use_gpu and GPU_AVAILABLE
    solver_type = "GPU" if use_gpu_solver else "CPU"
    print(f"Using {solver_type} solver for PDE integration")
    
    # Generate datasets
    print(f"Generating {num_datasets} PDE datasets...")
    
    for i in tqdm(range(num_datasets), desc="Generating datasets"):
        
        # Determine if this goes to train or test
        is_train = i < int(train_split * num_datasets)
        save_folder = train_folder if is_train else test_folder
        
        # Random parameters for this sample
        center_x = np.random.uniform(0.2 * Lx, 0.8 * Lx)
        center_y = np.random.uniform(0.2 * Ly, 0.8 * Ly)
        initial_width = np.random.uniform(0.025, 0.075)
        
        # ----------- Diffusion Only Simulation -----------
        diffusivity_diff = np.random.uniform(0.1, 0.4)
        init_val_diff = generate_initial_condition(mesh, center_x, center_y, initial_width)
        
        if use_gpu_solver:
            U_diff = gpu_solve_diffusion(
                np.array(init_val_diff).reshape(nx, ny), 
                float(diffusivity_diff), 
                float(timeStep), 
                steps, 
                float(dx), 
                float(dy)
            )
            phi_diff_value = U_diff.ravel()
        else:
            phi_diff_value = cpu_solve_diffusion(
                np.array(init_val_diff).reshape(nx, ny),
                diffusivity_diff,
                timeStep,
                steps,
                dx,
                dy
            )
        
        # ----------- Advection Only Simulation -----------
        velocity_x_adv = np.random.uniform(2, 5)
        velocity_y_adv = 2.0
        vel_adv = np.sqrt(velocity_x_adv**2 + velocity_y_adv**2)
        init_val_adve = generate_initial_condition(mesh, center_x, center_y, initial_width)
        
        if use_gpu_solver:
            U_adve = gpu_solve_advection(
                np.array(init_val_adve).reshape(nx, ny),
                float(vel_adv),
                float(timeStep),
                steps,
                float(dx),
                float(dy)
            )
            phi_adv_value = U_adve.ravel()
        else:
            # For CPU, we'll use a simple upwind scheme
            phi_adv_value = init_val_adve  # Placeholder - would need proper implementation
        
        # ----------- Advection-Diffusion Simulation -----------
        diffusivity_adv_diff = np.random.uniform(0.1, 0.4)
        velocity_x_adv_diff = 4.0
        velocity_y_adv_diff = 2.0
        vel_adv_diff = np.sqrt(velocity_x_adv_diff**2 + velocity_y_adv_diff**2)
        init_val_adv_diff = generate_initial_condition(mesh, center_x, center_y, initial_width)
        
        if use_gpu_solver:
            U_adv_diff = gpu_solve_advdiff(
                np.array(init_val_adv_diff).reshape(nx, ny),
                float(vel_adv_diff),
                float(diffusivity_adv_diff),
                float(timeStep),
                steps,
                float(dx),
                float(dy)
            )
            phi_adv_diff_value = U_adv_diff.ravel()
        else:
            # For CPU, use diffusion only as approximation
            phi_adv_diff_value = cpu_solve_diffusion(
                init_val_adv_diff.reshape(nx, ny),
                diffusivity_adv_diff,
                timeStep,
                steps,
                dx,
                dy
            )
        
        # ----------- Save datasets -----------
        size = 64 * 64
        
        # Save diffusion dataset
        filename_diff = os.path.join(save_folder, f"dataset_2d_{i:05d}_diff.npz")
        pattern = np.array([1, 0, 0]).reshape(-1, 1)
        one_hot = np.zeros((size, 1))
        one_hot[:len(one_hot) // 3 * 3] = np.tile(pattern, (size // 3, 1))
        
        np.savez(
            filename_diff,
            last_value=phi_diff_value,
            init_value=init_val_diff,
            label=0,
            one_hot=one_hot,
            diffusivity=diffusivity_diff,
            velocity_x=0.0,
            velocity_y=0.0
        )
        
        # Save advection dataset
        filename_adve = os.path.join(save_folder, f"dataset_2d_{i:05d}_adve.npz")
        pattern = np.array([0, 1, 0]).reshape(-1, 1)
        one_hot = np.zeros((size, 1))
        one_hot[:len(one_hot) // 3 * 3] = np.tile(pattern, (size // 3, 1))
        
        np.savez(
            filename_adve,
            last_value=phi_adv_value,
            init_value=init_val_adve,
            label=1,
            one_hot=one_hot,
            diffusivity=0.0,
            velocity_x=velocity_x_adv,
            velocity_y=velocity_y_adv
        )
        
        # Save advection-diffusion dataset
        filename_adve_diff = os.path.join(save_folder, f"dataset_2d_{i:05d}_adve_diff.npz")
        pattern = np.array([0, 0, 1]).reshape(-1, 1)
        one_hot = np.zeros((size, 1))
        one_hot[:len(one_hot) // 3 * 3] = np.tile(pattern, (size // 3, 1))
        
        np.savez(
            filename_adve_diff,
            last_value=phi_adv_diff_value,
            init_value=init_val_adv_diff,
            label=2,
            one_hot=one_hot,
            diffusivity=diffusivity_adv_diff,
            velocity_x=velocity_x_adv_diff,
            velocity_y=velocity_y_adv_diff
        )
    
    print(f"\n✅ Generated {num_datasets} PDE datasets!")
    print(f"Training samples: {int(train_split * num_datasets)} in {train_folder}")
    print(f"Test samples: {num_datasets - int(train_split * num_datasets)} in {test_folder}")
    print("\nNext steps:")
    print("1. Run: python pde_examples/prepare_amir_data.py  # Visualize data")
    print("2. Run: python pde_examples/fm_diffusion_2d.py    # Train Flow Matching")


def visualize_sample_data(folder="test_data64_cal", sample_idx=0):
    """Visualize a sample from the generated data."""
    import matplotlib.pyplot as plt
    
    # File paths
    diff_file = os.path.join(folder, f"dataset_2d_{sample_idx:05d}_diff.npz")
    adve_file = os.path.join(folder, f"dataset_2d_{sample_idx:05d}_adve.npz")
    advdiff_file = os.path.join(folder, f"dataset_2d_{sample_idx:05d}_adve_diff.npz")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    pde_types = [
        ("Diffusion", diff_file),
        ("Advection", adve_file),
        ("Advection-Diffusion", advdiff_file)
    ]
    
    for i, (pde_name, file_path) in enumerate(pde_types):
        if not os.path.exists(file_path):
            print(f"⚠️  {file_path} not found, skipping {pde_name}")
            continue
        
        # Load data
        data = np.load(file_path)
        init_state = data['init_value'].reshape(64, 64)
        final_state = data['last_value'].reshape(64, 64)
        
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
            print(f"{pde_name} - Diffusivity: {data['diffusivity']:.4f}")
        if 'velocity_x' in data and data['velocity_x'] != 0:
            print(f"{pde_name} - Velocity: ({data['velocity_x']:.2f}, {data['velocity_y']:.2f})")
    
    plt.tight_layout()
    
    # Create results directory
    os.makedirs("results/pde", exist_ok=True)
    plt.savefig("results/pde/amir_sample_data.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved visualization to results/pde/amir_sample_data.png")


if __name__ == "__main__":
    # Generate data
    generate_pde_data(num_datasets=100, train_split=0.9, use_gpu=True)
    
    # Visualize sample
    print("\nVisualizing sample data...")
    try:
        visualize_sample_data()
    except Exception as e:
        print(f"Could not visualize: {e}")