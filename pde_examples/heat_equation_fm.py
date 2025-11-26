"""
Flow Matching for PDE Solution: Heat Equation

Solves the 1D heat equation:
    ∂u/∂t = α ∂²u/∂x²
    
with initial condition u(x, 0) = u₀(x)
and boundary conditions u(0, t) = u(L, t) = 0

Flow Matching learns the mapping: u₀(x) → u(x, T)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class PDEFlowMatchingNet(nn.Module):
    """
    Neural network for Flow Matching on PDE solutions.
    
    Input: (u_t, t) where u_t is the solution at time t
    Output: velocity field v(u_t, t)
    """
    
    def __init__(self, spatial_dim=64, hidden_dim=128):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Spatial processing (1D convolutions)
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv_out = nn.Conv1d(hidden_dim, 1, kernel_size=5, padding=2)
        
        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.norm2 = nn.GroupNorm(8, hidden_dim)
        self.norm3 = nn.GroupNorm(8, hidden_dim)
        
        self.act = nn.SiLU()
        
    def forward(self, u, t):
        """
        Args:
            u: Solution field (batch, 1, spatial_dim)
            t: Time (batch, 1)
        Returns:
            v: Velocity field (batch, 1, spatial_dim)
        """
        # Time embedding
        t_emb = self.time_embed(t)  # (batch, hidden_dim)
        
        # Spatial processing
        h = self.act(self.norm1(self.conv1(u)))
        h = h + t_emb[:, :, None]  # Add time conditioning
        
        h = self.act(self.norm2(self.conv2(h)))
        h = h + t_emb[:, :, None]
        
        h = self.act(self.norm3(self.conv3(h)))
        
        v = self.conv_out(h)
        
        return v


def solve_heat_equation_analytical(u0, x, t, alpha=0.01):
    """
    Analytical solution using Fourier series (for validation).
    
    Args:
        u0: Initial condition (spatial_dim,)
        x: Spatial coordinates
        t: Time
        alpha: Thermal diffusivity
    """
    # Simple exponential decay for Gaussian initial condition
    # This is approximate - real solution needs Fourier series
    return u0 * np.exp(-alpha * t * 10)


def generate_initial_conditions(batch_size, spatial_dim=64, condition_type="gaussian"):
    """
    Generate random initial conditions for heat equation.
    
    Types:
    - gaussian: Single Gaussian bump
    - multi_gaussian: Multiple bumps
    - sine: Sine wave
    - random: Random smooth function
    """
    x = torch.linspace(0, 1, spatial_dim)
    u0_batch = []
    
    for _ in range(batch_size):
        if condition_type == "gaussian":
            # Single Gaussian
            center = 0.3 + 0.4 * torch.rand(1)
            width = 0.05 + 0.1 * torch.rand(1)
            u0 = torch.exp(-((x - center) ** 2) / (2 * width ** 2))
            
        elif condition_type == "multi_gaussian":
            # Multiple Gaussians
            u0 = torch.zeros_like(x)
            num_bumps = torch.randint(2, 4, (1,)).item()
            for _ in range(num_bumps):
                center = torch.rand(1)
                width = 0.05 + 0.05 * torch.rand(1)
                amplitude = 0.5 + 0.5 * torch.rand(1)
                u0 += amplitude * torch.exp(-((x - center) ** 2) / (2 * width ** 2))
                
        elif condition_type == "sine":
            # Sine wave
            freq = 1 + 3 * torch.rand(1)
            phase = 2 * np.pi * torch.rand(1)
            u0 = torch.sin(freq * 2 * np.pi * x + phase)
            u0 = (u0 + 1) / 2  # Normalize to [0, 1]
            
        else:  # random
            # Random smooth function
            coeffs = torch.randn(5)
            u0 = torch.zeros_like(x)
            for i, c in enumerate(coeffs):
                u0 += c * torch.sin((i + 1) * 2 * np.pi * x)
            u0 = (u0 - u0.min()) / (u0.max() - u0.min())
        
        # Apply boundary conditions
        u0[0] = 0
        u0[-1] = 0
        
        u0_batch.append(u0)
    
    return torch.stack(u0_batch).unsqueeze(1)  # (batch, 1, spatial_dim)


def solve_heat_equation_numerical(u0, alpha=0.01, T=1.0, dt=0.001):
    """
    Numerical solution using finite differences (for ground truth).
    
    Args:
        u0: Initial condition (batch, 1, spatial_dim)
        alpha: Thermal diffusivity
        T: Final time
        dt: Time step
    """
    device = u0.device
    batch_size, _, N = u0.shape
    dx = 1.0 / (N - 1)
    
    # Stability condition: dt <= dx^2 / (2*alpha)
    dt = min(dt, dx**2 / (2 * alpha) * 0.5)
    num_steps = int(T / dt)
    
    u = u0.clone()
    
    for _ in range(num_steps):
        # Finite difference: ∂²u/∂x² ≈ (u[i+1] - 2u[i] + u[i-1]) / dx²
        u_xx = (torch.roll(u, -1, dims=2) - 2 * u + torch.roll(u, 1, dims=2)) / (dx ** 2)
        
        # Enforce boundary conditions
        u_xx[:, :, 0] = 0
        u_xx[:, :, -1] = 0
        
        # Forward Euler: u(t+dt) = u(t) + dt * α * ∂²u/∂x²
        u = u + dt * alpha * u_xx
        
        # Enforce boundary conditions
        u[:, :, 0] = 0
        u[:, :, -1] = 0
    
    return u


def train_flow_matching_pde(model, num_epochs=1000, batch_size=32, device='cuda'):
    """Train Flow Matching to learn PDE solution operator."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    losses = []
    
    for epoch in range(num_epochs):
        # Generate random initial conditions
        u0 = generate_initial_conditions(batch_size, condition_type="multi_gaussian").to(device)
        
        # Solve PDE numerically to get ground truth
        uT = solve_heat_equation_numerical(u0, alpha=0.01, T=1.0).to(device)
        
        # Sample random time
        t = torch.rand(batch_size, 1, device=device)
        
        # Flow Matching interpolation
        t_expanded = t.view(-1, 1, 1)
        u_t = (1 - t_expanded) * u0 + t_expanded * uT
        
        # Target velocity
        v_target = uT - u0
        
        # Predict velocity
        v_pred = model(u_t, t)
        
        # Loss
        loss = F.mse_loss(v_pred, v_target)
        
        # Optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
    
    return losses


@torch.no_grad()
def solve_pde_with_flow_matching(model, u0, num_steps=50, device='cuda'):
    """
    Solve PDE using trained Flow Matching model.
    
    Args:
        model: Trained FM model
        u0: Initial condition (batch, 1, spatial_dim)
        num_steps: Number of integration steps
    """
    model.eval()
    u = u0.to(device)
    
    time_steps = torch.linspace(0, 1, num_steps + 1, device=device)
    
    for i in range(num_steps):
        t_curr = time_steps[i].expand(u.size(0), 1)
        t_next = time_steps[i + 1].expand(u.size(0), 1)
        dt = (t_next - t_curr).view(-1, 1, 1)
        
        # Euler step
        v = model(u, t_curr)
        u = u + v * dt
        
        # Enforce boundary conditions
        u[:, :, 0] = 0
        u[:, :, -1] = 0
    
    return u


def visualize_pde_solution(u0, uT_numerical, uT_fm, save_path="pde_solution.png"):
    """Visualize PDE solutions."""
    
    u0 = u0.cpu().numpy()
    uT_numerical = uT_numerical.cpu().numpy()
    uT_fm = uT_fm.cpu().numpy()
    
    num_examples = min(4, u0.shape[0])
    x = np.linspace(0, 1, u0.shape[2])
    
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 3*num_examples))
    
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_examples):
        # Initial condition
        axes[i, 0].plot(x, u0[i, 0], 'b-', linewidth=2)
        axes[i, 0].set_title(f"Initial Condition (t=0)")
        axes[i, 0].set_ylim([-0.1, 1.5])
        axes[i, 0].grid(True, alpha=0.3)
        
        # Numerical solution
        axes[i, 1].plot(x, uT_numerical[i, 0], 'g-', linewidth=2, label='Numerical')
        axes[i, 1].set_title(f"Numerical Solution (t=1)")
        axes[i, 1].set_ylim([-0.1, 1.5])
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].legend()
        
        # Flow Matching solution
        axes[i, 2].plot(x, uT_numerical[i, 0], 'g--', linewidth=2, alpha=0.5, label='Numerical')
        axes[i, 2].plot(x, uT_fm[i, 0], 'r-', linewidth=2, label='Flow Matching')
        axes[i, 2].set_title(f"FM vs Numerical (t=1)")
        axes[i, 2].set_ylim([-0.1, 1.5])
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].legend()
        
        # Error
        error = np.abs(uT_numerical[i, 0] - uT_fm[i, 0])
        axes[i, 2].text(0.5, 1.3, f"Max Error: {error.max():.4f}", 
                       ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PDE solution visualization to {save_path}")


def main():
    """Main training and evaluation pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = PDEFlowMatchingNet(spatial_dim=64, hidden_dim=128).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\nTraining Flow Matching for Heat Equation...")
    losses = train_flow_matching_pde(model, num_epochs=1000, batch_size=32, device=device)
    
    # Test
    print("\nTesting on new initial conditions...")
    u0_test = generate_initial_conditions(4, condition_type="multi_gaussian").to(device)
    
    # Numerical solution (ground truth)
    uT_numerical = solve_heat_equation_numerical(u0_test, alpha=0.01, T=1.0)
    
    # Flow Matching solution
    uT_fm = solve_pde_with_flow_matching(model, u0_test, num_steps=50, device=device)
    
    # Compute error
    error = (uT_numerical - uT_fm).abs().mean().item()
    print(f"Mean absolute error: {error:.6f}")
    
    # Visualize
    import os
    os.makedirs("results/pde", exist_ok=True)
    visualize_pde_solution(u0_test, uT_numerical, uT_fm, "results/pde/heat_equation.png")
    
    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Flow Matching Training Loss")
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig("results/pde/training_loss.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved training loss plot to results/pde/training_loss.png")
    
    print("\n✅ PDE Flow Matching completed!")
    print("Key advantages:")
    print("- Learns solution operator (any initial condition → solution)")
    print("- Fast inference (50 steps vs thousands for numerical solver)")
    print("- Generalizes to unseen initial conditions")


if __name__ == "__main__":
    main()
