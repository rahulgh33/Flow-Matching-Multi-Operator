"""Conditional Flow Matching implementation for CIFAR-10 class generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from flow_matching_base import (
    FlowMatchingBase, 
    create_time_embedding
)

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']


class ConditionalFlowMatchingNetCIFAR(FlowMatchingBase):
    """Conditional Flow Matching network for CIFAR-10 generation with class and time conditioning."""
    
    def __init__(self, channels: int = 3, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__(channels, hidden_dim)
        self.num_classes = num_classes
        
        # Encoder: 32x32 -> 8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, 3, stride=2, padding=1),  # 8x8
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
        self.fusion = nn.Linear(hidden_dim * 8 * 8 + hidden_dim + hidden_dim, hidden_dim * 8 * 8)
        
        # Decoder: 8x8 -> 32x32
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """Forward pass with time and class conditioning."""
        batch_size = x.size(0)
        
        # Encode spatial features
        h = self.encoder(x)  # (B, hidden_dim, 8, 8)
        h_flat = h.view(batch_size, -1)  # (B, hidden_dim * 8 * 8)
        
        # Embed time
        t_emb = self.time_embedding(t)  # (B, hidden_dim)
        
        # Embed class
        c_emb = self.class_embedding(class_labels)  # (B, hidden_dim)
        
        # Fuse features, time, and class
        combined = torch.cat([h_flat, t_emb, c_emb], dim=-1)
        fused = self.fusion(combined)
        fused = fused.view(batch_size, self.hidden_dim, 8, 8)
        
        # Decode to velocity field
        velocity = self.decoder(fused)
        return velocity


def conditional_train_step_cifar(
    model: ConditionalFlowMatchingNetCIFAR, 
    data_batch: torch.Tensor, 
    labels_batch: torch.Tensor,
    optimizer: torch.optim.Optimizer, 
    device: torch.device
) -> float:
    """Single training step for conditional CIFAR-10 flow matching."""
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
def generate_conditional_cifar_samples(
    model: ConditionalFlowMatchingNetCIFAR, 
    device: torch.device, 
    class_labels: torch.Tensor,
    num_steps: int = 50
) -> torch.Tensor:
    """Generate CIFAR-10 samples conditioned on specific class labels."""
    model.eval()
    num_samples = len(class_labels)
    
    # Start from pure noise
    x = torch.randn(num_samples, 3, 32, 32, device=device)
    
    # Time discretization
    time_steps = torch.linspace(0, 1, num_steps + 1, device=device)
    dt = 1.0 / num_steps
    
    # Euler integration with class conditioning
    for i in range(num_steps):
        t = time_steps[i].expand(num_samples, 1)
        velocity = model(x, t, class_labels)
        x = x + velocity * dt
    
    return x


def create_cifar_conditional_data_loader(batch_size: int = 128) -> DataLoader:
    """Create CIFAR-10 data loader with proper preprocessing."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)  # Scale to [-1, 1]
    ])
    
    dataset = datasets.CIFAR10(
        root="./data", 
        train=True, 
        download=True, 
        transform=transform
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def visualize_conditional_cifar_samples(samples: torch.Tensor, labels: torch.Tensor, save_path: str = "conditional_cifar_samples.png"):
    """Visualize generated conditional CIFAR-10 samples with their labels."""
    samples = samples.cpu()
    labels = labels.cpu()
    
    # Convert from [-1, 1] to [0, 1] for visualization
    samples = (samples + 1) / 2
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("Conditional CIFAR-10 Generation", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            # Convert from CHW to HWC format
            img = samples[i].permute(1, 2, 0).clamp(0, 1)
            ax.imshow(img)
            class_name = CIFAR10_CLASSES[labels[i].item()]
            ax.set_title(f"{class_name} ({labels[i].item()})", fontsize=10)
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved conditional CIFAR-10 samples to {save_path}")


def generate_specific_classes(model: ConditionalFlowMatchingNetCIFAR, device: torch.device, class_indices: list):
    """Generate specific CIFAR-10 classes and visualize them."""
    # Create labels tensor
    labels = torch.tensor(class_indices, device=device)
    
    # Generate samples
    samples = generate_conditional_cifar_samples(model, device, labels, num_steps=50)
    
    # Convert for visualization
    samples = (samples.cpu() + 1) / 2
    
    # Visualize
    fig, axes = plt.subplots(1, len(class_indices), figsize=(3*len(class_indices), 3))
    if len(class_indices) == 1:
        axes = [axes]
    
    fig.suptitle("Generated Specific CIFAR-10 Classes", fontsize=16)
    
    for i, (sample, class_idx) in enumerate(zip(samples, class_indices)):
        img = sample.permute(1, 2, 0).clamp(0, 1)
        axes[i].imshow(img)
        class_name = CIFAR10_CLASSES[class_idx]
        axes[i].set_title(f"{class_name}", fontsize=12)
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig("specific_cifar_classes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Generated classes: {[CIFAR10_CLASSES[i] for i in class_indices]}")
    print("Saved to specific_cifar_classes.png")


def main():
    """Main training and generation pipeline for conditional CIFAR-10."""
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    train_loader = create_cifar_conditional_data_loader(batch_size=128)
    
    # Model and optimizer (lower learning rate for CIFAR-10)
    model = ConditionalFlowMatchingNetCIFAR(channels=3, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    num_epochs = 10  # Reduced for demo
    log_interval = 100
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            loss = conditional_train_step_cifar(model, data, labels, optimizer, device)
            epoch_losses.append(loss)
            
            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss:.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    # Generate samples for all classes
    print("Generating samples for all classes...")
    all_labels = torch.arange(10, device=device).repeat(2)  # 2 samples per class
    samples = generate_conditional_cifar_samples(model, device, all_labels, num_steps=50)
    visualize_conditional_cifar_samples(samples, all_labels)
    
    # Generate specific classes
    print("Generating specific classes...")
    # Generate: airplane, cat, dog, ship
    generate_specific_classes(model, device, [0, 3, 5, 8])


if __name__ == "__main__":
    main()