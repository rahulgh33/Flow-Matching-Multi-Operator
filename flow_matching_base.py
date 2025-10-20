"""Base classes and utilities for Flow Matching implementations."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FlowMatchingBase(nn.Module, ABC):
    """Abstract base class for Flow Matching networks."""
    
    def __init__(self, channels: int, hidden_dim: int):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass with time conditioning."""
        pass


def flow_matching_train_step(
    model: FlowMatchingBase, 
    data_batch: torch.Tensor, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device
) -> float:
    """Generic training step for flow matching models.
    
    Args:
        model: Flow matching network
        data_batch: Batch of real data samples
        optimizer: Model optimizer
        device: Training device
        
    Returns:
        Training loss value
    """
    model.train()
    batch_size = data_batch.size(0)
    
    # Sample noise as starting point (x0)
    noise = torch.randn_like(data_batch)
    
    # Sample random interpolation times
    t = torch.rand(batch_size, 1, device=device)
    
    # Linear interpolation between noise and data
    t_expanded = t.view(-1, 1, 1, 1)
    x_t = (1 - t_expanded) * noise + t_expanded * data_batch
    
    # Target velocity field (constant for linear interpolation)
    target_velocity = data_batch - noise
    
    # Predict velocity field
    predicted_velocity = model(x_t, t)
    
    # Compute MSE loss
    loss = F.mse_loss(predicted_velocity, target_velocity)
    
    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def flow_matching_generate_samples(
    model: FlowMatchingBase, 
    device: torch.device, 
    image_shape: Tuple[int, int, int, int],  # (B, C, H, W)
    num_steps: int = 50
) -> torch.Tensor:
    """Generate samples using Euler ODE solver.
    
    Args:
        model: Trained flow matching network
        device: Generation device
        image_shape: Shape of images to generate (B, C, H, W)
        num_steps: Number of integration steps
        
    Returns:
        Generated samples tensor
    """
    model.eval()
    
    # Start from pure noise
    x = torch.randn(*image_shape, device=device)
    
    # Time discretization
    time_steps = torch.linspace(0, 1, num_steps + 1, device=device)
    dt = 1.0 / num_steps
    
    # Euler integration
    for i in range(num_steps):
        t = time_steps[i].expand(image_shape[0], 1)
        velocity = model(x, t)
        x = x + velocity * dt
    
    return x


def create_time_embedding(input_dim: int, hidden_dim: int) -> nn.Module:
    """Create a standard time embedding network."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim)
    )