"""Flow Matching implementation for MNIST digit generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from flow_matching_base import (
    FlowMatchingBase, 
    flow_matching_train_step, 
    flow_matching_generate_samples,
    create_time_embedding
)

class FlowMatchingNet(FlowMatchingBase):
    """Flow Matching network for MNIST generation with time conditioning."""
    
    def __init__(self, channels: int = 1, hidden_dim: int = 128):
        super().__init__(channels, hidden_dim)
        
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
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 7 * 7 + hidden_dim, hidden_dim * 7 * 7)
        
        # Decoder: 7x7 -> 28x28
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 64, 4, stride=2, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass with time conditioning.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            t: Time tensor of shape (B, 1)
            
        Returns:
            Predicted velocity field of same shape as x
        """
        batch_size = x.size(0)
        
        # Encode spatial features
        h = self.encoder(x)  # (B, hidden_dim, 7, 7)
        h_flat = h.view(batch_size, -1)  # (B, hidden_dim * 7 * 7)
        
        # Embed time
        t_emb = self.time_embedding(t)  # (B, hidden_dim)
        
        # Fuse features and time
        combined = torch.cat([h_flat, t_emb], dim=-1)
        fused = self.fusion(combined)
        fused = fused.view(batch_size, self.hidden_dim, 7, 7)
        
        # Decode to velocity field
        velocity = self.decoder(fused)
        return velocity


# Use shared training and generation functions
train_step = flow_matching_train_step

def generate_samples(
    model: FlowMatchingNet, 
    device: torch.device, 
    num_steps: int = 50, 
    num_samples: int = 16
) -> torch.Tensor:
    """Generate MNIST samples using the shared generation function."""
    return flow_matching_generate_samples(
        model, device, (num_samples, 1, 28, 28), num_steps
    )

def create_data_loader(batch_size: int = 128) -> DataLoader:
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


def visualize_samples(samples: torch.Tensor, save_path: str = "fm_samples.png"):
    """Visualize generated samples in a 4x4 grid."""
    samples = samples.cpu()
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Generated MNIST Samples", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            ax.imshow(samples[i, 0], cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved generated samples to {save_path}")


def main():
    """Main training and generation pipeline."""
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    train_loader = create_data_loader(batch_size=128)
    
    # Model and optimizer
    model = FlowMatchingNet(channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    num_epochs = 5
    log_interval = 100
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            loss = train_step(model, data, optimizer, device)
            epoch_losses.append(loss)
            
            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss:.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    # Generate and visualize samples
    print("Generating samples...")
    samples = generate_samples(model, device, num_steps=50, num_samples=16)
    visualize_samples(samples)


if __name__ == "__main__":
    main()
