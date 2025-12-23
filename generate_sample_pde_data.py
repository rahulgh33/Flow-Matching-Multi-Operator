"""
Generate Sample PDE Data Without Dependencies
Creates minimal diffusion PDE data for Flow Matching testing
"""

import numpy as np
import os


def generate_gaussian_blob(nx=64, ny=64, center_x=None, center_y=None, width=None):
    """Generate a Gaussian blob initial condition."""
    if center_x is None:
        center_x = np.random.uniform(0.2, 0.8)
    if center_y is None:
        center_y = np.random.uniform(0.2, 0.8)
    if width is None:
        width = np.random.uniform(0.025, 0.075)
    
    # Create coordinate grids
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian blob
    init_val = np.exp(-(((X - center_x)**2 + (Y - center_y)**2) / width))
    
    # Apply boundary taper (sin functions)
    taper = np.sin(np.pi * X) * np.sin(np.pi * Y)
    init_val = init_val * taper
    
    return init_val


def simple_diffusion_step(u, diffusivity, dt, dx, dy):
    """Simple explicit diffusion step using finite differences."""
    # Pad for boundary conditions (Neumann = 0)
    u_pad = np.pad(u, 1, mode='edge')
    
    # Compute Laplacian using finite differences
    laplacian = (
        u_pad[2:, 1:-1] + u_pad[:-2, 1:-1] - 2*u_pad[1:-1, 1:-1]
    ) / (dx**2) + (
        u_pad[1:-1, 2:] + u_pad[1:-1, :-2] - 2*u_pad[1:-1, 1:-1]
    ) / (dy**2)
    
    # Update using explicit Euler
    u_new = u + dt * diffusivity * laplacian
    
    return u_new


def solve_diffusion_simple(init_state, diffusivity, steps=500, dt=0.0001):
    """Solve diffusion equation using simple finite differences."""
    nx, ny = init_state.shape
    dx, dy = 1.0/nx, 1.0/ny
    
    u = init_state.copy()
    
    for _ in range(steps):
        u = simple_diffusion_step(u, diffusivity, dt, dx, dy)
    
    return u


def generate_sample_dataset(num_samples=50, save_folder="test_data64_cal"):
    """Generate sample PDE dataset."""
    
    print(f"🏭 Generating {num_samples} sample PDE datasets...")
    
    # Create folder
    os.makedirs(save_folder, exist_ok=True)
    
    for i in range(num_samples):
        # Random parameters
        diffusivity = np.random.uniform(0.1, 0.4)
        center_x = np.random.uniform(0.2, 0.8)
        center_y = np.random.uniform(0.2, 0.8)
        width = np.random.uniform(0.025, 0.075)
        
        # Generate initial condition
        init_state = generate_gaussian_blob(64, 64, center_x, center_y, width)
        
        # Solve diffusion PDE
        final_state = solve_diffusion_simple(init_state, diffusivity)
        
        # Create one-hot encoding (diffusion = [1,0,0])
        size = 64 * 64
        pattern = np.array([1, 0, 0]).reshape(-1, 1)
        one_hot = np.zeros((size, 1))
        one_hot[:len(one_hot) // 3 * 3] = np.tile(pattern, (size // 3, 1))
        
        # Save in Amir's format
        filename = os.path.join(save_folder, f"dataset_2d_{i:05d}_diff.npz")
        
        np.savez(
            filename,
            last_value=final_state.flatten(),  # Final state (flattened)
            init_value=init_state.flatten(),   # Initial condition (flattened)
            label=0,                           # Diffusion PDE type
            one_hot=one_hot,                   # One-hot encoding
            diffusivity=diffusivity,           # Diffusion coefficient
            velocity_x=0.0,                    # No advection
            velocity_y=0.0                     # No advection
        )
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{num_samples} samples")
    
    print(f"✅ Generated {num_samples} PDE samples in {save_folder}/")
    print(f"Files: dataset_2d_00000_diff.npz to dataset_2d_{num_samples-1:05d}_diff.npz")


def visualize_sample(save_folder="test_data64_cal", sample_idx=0):
    """Visualize a sample to verify it looks correct."""
    
    filename = os.path.join(save_folder, f"dataset_2d_{sample_idx:05d}_diff.npz")
    
    if not os.path.exists(filename):
        print(f"❌ Sample file {filename} not found!")
        return
    
    # Load data
    data = np.load(filename)
    init_state = data['init_value'].reshape(64, 64)
    final_state = data['last_value'].reshape(64, 64)
    diffusivity = data['diffusivity']
    
    print(f"\n📊 Sample {sample_idx} verification:")
    print(f"  Diffusivity: {diffusivity:.4f}")
    print(f"  Initial state range: [{init_state.min():.4f}, {init_state.max():.4f}]")
    print(f"  Final state range: [{final_state.min():.4f}, {final_state.max():.4f}]")
    print(f"  Change: {np.abs(final_state - init_state).mean():.4f}")
    
    # Check if diffusion worked (should be more spread out)
    init_peak = init_state.max()
    final_peak = final_state.max()
    print(f"  Peak reduction: {init_peak:.4f} → {final_peak:.4f} (diffusion effect)")
    
    if final_peak < init_peak:
        print("  ✅ Diffusion effect detected (peak reduced)")
    else:
        print("  ⚠️  No clear diffusion effect")


if __name__ == "__main__":
    # Generate sample data
    generate_sample_dataset(num_samples=50, save_folder="test_data64_cal")
    
    # Verify first sample
    visualize_sample("test_data64_cal", 0)