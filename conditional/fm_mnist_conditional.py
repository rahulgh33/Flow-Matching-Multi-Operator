"""Conditional Flow Matching implementation for MNIST digit generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flow_matching_base import (
    FlowMatchingBase, 
    flow_matching_train_step, 
    flow_matching_generate_samples,
    create_time_embedding
)


class ConditionalFlowMatchingNet(FlowMatchingBase):
    """Conditional Flow Matching network for MNIST generation with class and time conditioning."""
    
    def __init__(self, channels: int = 1, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__(channels, hidden_dim)
        self.num_classes = num_classes
        
        # Encoder: 28x28 -> 7x7
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1),  # 7x7
            nn.ReLU(inplace=True)
        )
        
        # Time embedding network
        self.time_embedding = create_time_embedding(1, hidden_dim)
        
        # Class embedding network
        self.class_embedding = nn.Sequential(
            nn.Embedding(num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Fusion layer (spatial + time + class)
        self.fusion = nn.Linear(hidden_dim * 7 * 7 + hidden_dim + hidden_dim, hidden_dim * 7 * 7)
        
        # Decoder: 7x7 -> 28x28
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 64, 4, stride=2, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """Forward pass with time and class conditioning.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            t: Time tensor of shape (B, 1)
            class_labels: Class labels tensor of shape (B,)
            
        Returns:
            Predicted velocity field of same shape as x
        """
        batch_size = x.size(0)
        
        # Encode spatial features
        h = self.encoder(x)  # (B, hidden_dim, 7, 7)
        h_flat = h.view(batch_size, -1)  # (B, hidden_dim * 7 * 7)
        
        # Embed time
        t_emb = self.time_embedding(t)  # (B, hidden_dim)
        
        # Embed class
        c_emb = self.class_embedding(class_labels)  # (B, hidden_dim)
        
        # Fuse features, time, and class
        combined = torch.cat([h_flat, t_emb, c_emb], dim=-1)
        fused = self.fusion(combined)
        fused = fused.view(batch_size, self.hidden_dim, 7, 7)
        
        # Decode to velocity field
        velocity = self.decoder(fused)
        return velocity


def conditional_train_step(
    model: ConditionalFlowMatchingNet, 
    data_batch: torch.Tensor, 
    labels_batch: torch.Tensor,
    optimizer: torch.optim.Optimizer, 
    device: torch.device
) -> float:
    """Single training step for conditional flow matching.
    
    Args:
        model: Conditional flow matching network
        data_batch: Batch of real data samples
        labels_batch: Batch of class labels
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
    
    # Predict velocity field with class conditioning
    predicted_velocity = model(x_t, t, labels_batch)
    
    # Compute MSE loss
    loss = F.mse_loss(predicted_velocity, target_velocity)
    
    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def generate_conditional_samples(
    model: ConditionalFlowMatchingNet, 
    device: torch.device, 
    class_labels: torch.Tensor,
    num_steps: int = 50
) -> torch.Tensor:
    """Generate samples conditioned on specific class labels.
    
    Args:
        model: Trained conditional flow matching network
        device: Generation device
        class_labels: Desired class labels tensor of shape (num_samples,)
        num_steps: Number of integration steps
        
    Returns:
        Generated samples tensor of shape (num_samples, 1, 28, 28)
    """
    model.eval()
    num_samples = len(class_labels)
    
    # Start from pure noise
    x = torch.randn(num_samples, 1, 28, 28, device=device)
    
    # Time discretization
    time_steps = torch.linspace(0, 1, num_steps + 1, device=device)
    dt = 1.0 / num_steps
    
    # Euler integration with class conditioning
    for i in range(num_steps):
        t = time_steps[i].expand(num_samples, 1)
        velocity = model(x, t, class_labels)
        x = x + velocity * dt
    
    return x


def create_mnist_data_loader(batch_size: int = 128) -> DataLoader:
    """Create MNIST data loader with proper preprocessing."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)  # Scale to [-1, 1]
    ])
    
    dataset = datasets.MNIST(
        root="./data", 
        train=True, 
        download=True, 
        transform=transform
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def visualize_conditional_samples(samples: torch.Tensor, labels: torch.Tensor, save_path: str = "conditional_mnist_samples.png"):
    """Visualize generated conditional samples with their labels."""
    samples = samples.cpu()
    labels = labels.cpu()
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle("Conditional MNIST Generation", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            ax.imshow(samples[i, 0], cmap="gray", vmin=-1, vmax=1)
            ax.set_title(f"Class: {labels[i].item()}", fontsize=12)
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved conditional samples to {save_path}")


def generate_specific_digits(model: ConditionalFlowMatchingNet, device: torch.device, digits: list):
    """Generate specific digits and visualize them.
    
    Args:
        model: Trained model
        device: Device to run on
        digits: List of digits to generate (e.g., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    """
    # Create labels tensor
    labels = torch.tensor(digits, device=device)
    
    # Generate samples
    samples = generate_conditional_samples(model, device, labels, num_steps=50)
    
    # Visualize
    fig, axes = plt.subplots(1, len(digits), figsize=(2*len(digits), 2))
    if len(digits) == 1:
        axes = [axes]
    
    fig.suptitle("Generated Specific Digits", fontsize=16)
    
    for i, (sample, digit) in enumerate(zip(samples, digits)):
        axes[i].imshow(sample[0].cpu(), cmap="gray", vmin=-1, vmax=1)
        axes[i].set_title(f"Digit: {digit}", fontsize=12)
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig("specific_digits.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Generated digits: {digits}")
    print("Saved to specific_digits.png")


def main():
    """Main training and generation pipeline for conditional MNIST."""
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    train_loader = create_mnist_data_loader(batch_size=128)
    
    # Model and optimizer
    model = ConditionalFlowMatchingNet(channels=1, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    num_epochs = 15  # Increased for better results
    log_interval = 100
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            loss = conditional_train_step(model, data, labels, optimizer, device)
            epoch_losses.append(loss)
            
            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss:.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    # Generate samples for all classes
    print("Generating samples for all classes...")
    all_labels = torch.arange(10, device=device).repeat(2)  # 2 samples per class
    samples = generate_conditional_samples(model, device, all_labels, num_steps=50)
    visualize_conditional_samples(samples, all_labels)
    
    # Generate specific digits
    print("Generating specific digits...")
    generate_specific_digits(model, device, [1, 3, 7, 9])  # Example: generate 1, 3, 7, 9


if __name__ == "__main__":
    main()