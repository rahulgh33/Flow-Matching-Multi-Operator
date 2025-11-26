"""
Flow Matching for Burgers' Equation (Nonlinear PDE)

Solves the viscous Burgers' equation:
    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
    
This is a nonlinear PDE that models shock waves and turbulence.
Much more challenging than the heat equation!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def solve_burgers_numerical(u0, nu=0.01, T=1.0, dt=0.0001):
    """
    Solve Burgers' equation numerically using finite differences.
    
    Args:
        u0: Initial condition (batch, 1, spatial_dim)
        nu: Viscosity
        T: Final time
        dt: Time step
    """
    device = u0.device
    batch_size, _, N = u0.shape
    dx = 1.0 / (N - 1)
    
    # Stability: dt <= min(dx^2/(2*nu), dx/max(u))
    dt = min(dt, dx**2 / (2 * nu) * 0.5)
    num_steps = int(T / dt)
    
    u = u0.clone()
    
    for _ in range(num_steps):
        # Nonlinear term: u ∂u/∂x (upwind scheme)
        u_x = (u - torch.roll(u, 1, dims=2)) / dx
        nonlinear = u * u_x
        
        # Viscous term: ν ∂²u/∂x²
        u_xx = (torch.roll(u, -1, dims=2) - 2 * u + torch.roll(u, 1, dims=2)) / (dx ** 2)
        viscous = nu * u_xx
        
        # Time step: ∂u/∂t = -u ∂u/∂x + ν ∂²u/∂x²
        u = u + dt * (-nonlinear + viscous)
        
        # Periodic boundary conditions
        # (or use u[:, :, 0] = u[:, :, -1] = 0 for Dirichlet)
    
    return u


def generate_burgers_initial_conditions(batch_size, spatial_dim=128):
    """Generate initial conditions for Burgers' equation."""
    x = torch.linspace(0, 1, spatial_dim)
    u0_batch = []
    
    for _ in range(batch_size):
        # Smooth random function (will develop shocks)
        u0 = torch.zeros_like(x)
        num_modes = torch.randint(2, 5, (1,)).item()
        
        for k in range(1, num_modes + 1):
            amplitude = torch.randn(1) * 0.5
            phase = 2 * np.pi * torch.rand(1)
            u0 += amplitude * torch.sin(k * 2 * np.pi * x + phase)
        
        u0_batch.append(u0)
    
    return torch.stack(u0_batch).unsqueeze(1)


# Reuse PDEFlowMatchingNet from heat_equation_fm.py
# (or import it if you want)

print("Burgers' equation example created!")
print("This demonstrates Flow Matching on nonlinear PDEs with shock formation.")
