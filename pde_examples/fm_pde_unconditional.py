"""
Unconditional Flow Matching for PDE Data Generation.

Learns to generate realistic PDE solution pairs (initial state, final state)
from pure noise. This is a generative model, not a solver.

Each generated sample has 2 channels:
  - Channel 0: Initial condition (t=0)
  - Channel 1: Final solution (t=final)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flow_matching_base import (
    FlowMatchingBase,
    flow_matching_train_step,
    flow_matching_generate_samples,
    create_time_embedding
)


class PDEDataset(Dataset):
    """Dataset for PDE solution pairs."""
    
    def __init__(self, data_folder="train_data64", pde_type="diff"):
        """
        Args:
            data_folder: Folder containing .npz files
            pde_type: "diff", "adve", or "adve_diff"
        """
        self.data_folder = data_folder
        self.pde_type = pde_type
        
        # Find all files of this type
        self.files = [
            os.path.join(data_folder, f) 
            for f in os.listdir(data_folder) 
            if f.endswith(f"_{pde_type}.npz")
        ]
        
        print(f"Found {len(self.files)} {pde_type} files in {data_folder}")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        Returns:
            sample: (2, 64, 64) tensor with [initial_state, final_state]
        """
        data = np.load(self.files[idx])
        
        # Get initial and final states
        init_val = data['init_value']  # (4096,) flattened
        last_val = data['last_value']  # (64, 64) or (4096,)
        
        # Reshape if needed
        if init_val.shape == (4096,):
            init_val = init_val.reshape(64, 64)
        if last_val.shape == (4096,):
            last_val = last_val.reshape(64, 64)
        
        # Stack into 2-channel image: [initial, final]
        sample = np.stack([init_val, last_val], axis=0)  # (2, 64, 64)
        
        # Convert to tensor and normalize to [-1, 1]
        sample = torch.from_numpy(sample).float()
        sample = sample * 2 - 1  # Assume data is in [0, 1], map to [-1, 1]
        
        return sample


class PDEFlowMatchingNet(FlowMatchingBase):
    """
    Unconditional Flow Matching network for PDE generation.
    
    Generates 2-channel PDE solutions from noise:
      Channel 0: Initial condition
      Channel 1: Final solution
    """
    
    def __init__(self, channels=2, hidden_dim=128):
        super().__init__(channels, hidden_dim)
        
        # Encoder: 64x64 -> 16x16
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 32x32
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1),  # 16x16
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        
        # Time embedding
        self.time_embedding = create_time_embedding(1, hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 16 * 16 + hidden_dim, hidden_dim * 16 * 16),
            nn.SiLU()
        )
        
        # Decoder: 16x16 -> 64x64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, 4, stride=2, padding=1),  # 32x32
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64x64
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, channels, 3, padding=1)
        )
        
        # Initialize output layer to zero for stability
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
    
    def forward(self, x, t):
        """
        Args:
            x: Noisy samples (B, 2, 64, 64)
            t: Time (B, 1)
        Returns:
            Velocity field (B, 2, 64, 64)
        """
        batch_size = x.size(0)
        
        # Encode spatial features
        h = self.encoder(x)  # (B, hidden_dim, 16, 16)
        h_flat = h.view(batch_size, -1)
        
        # Embed time
        t_emb = self.time_embedding(t)  # (B, hidden_dim)
        
        # Fuse
        combined = torch.cat([h_flat, t_emb], dim=-1)
        fused = self.fusion(combined)
        fused = fused.view(batch_size, self.hidden_dim, 16, 16)
        
        # Decode to velocity
        velocity = self.decoder(fused)
        return velocity


# Use shared training function
train_step = flow_matching_train_step


def generate_samples(model, device, num_steps=50, num_samples=16):
    """Generate PDE solution pairs from noise."""
    return flow_matching_generate_samples(
        model, device, (num_samples, 2, 64, 64), num_steps
    )


def visualize_generated_samples(samples, save_path="results/pde/unconditional_samples.png"):
    """
    Visualize generated PDE samples.
    
    Args:
        samples: (N, 2, 64, 64) tensor with generated samples
        save_path: Where to save the visualization
    """
    samples = samples.cpu()
    num_samples = min(16, samples.size(0))
    
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    
    for i in range(num_samples):
        # Initial condition (channel 0)
        ax_init = axes[i // 4, (i % 4) * 2]
        ax_init.imshow(samples[i, 0], cmap='hot', vmin=-1, vmax=1)
        ax_init.axis('off')
        if i < 4:
            ax_init.set_title(f'Sample {i+1}\nInitial', fontsize=8)
        
        # Final solution (channel 1)
        ax_final = axes[i // 4, (i % 4) * 2 + 1]
        ax_final.imshow(samples[i, 1], cmap='hot', vmin=-1, vmax=1)
        ax_final.axis('off')
        if i < 4:
            ax_final.set_title('Final', fontsize=8)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved generated samples to {save_path}")
    plt.close()


def compare_real_vs_generated(real_data, generated_data, save_path="results/pde/real_vs_generated.png"):
    """Compare real PDE data with generated samples."""
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    
    # First 2 rows: Real data
    for i in range(8):
        if i < len(real_data):
            # Initial
            axes[0, i].imshow(real_data[i, 0].cpu(), cmap='hot', vmin=-1, vmax=1)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Real\nInitial', fontsize=10)
            
            # Final
            axes[1, i].imshow(real_data[i, 1].cpu(), cmap='hot', vmin=-1, vmax=1)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('Real\nFinal', fontsize=10)
    
    # Last 2 rows: Generated data
    for i in range(8):
        if i < len(generated_data):
            # Initial
            axes[2, i].imshow(generated_data[i, 0].cpu(), cmap='hot', vmin=-1, vmax=1)
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_ylabel('Generated\nInitial', fontsize=10)
            
            # Final
            axes[3, i].imshow(generated_data[i, 1].cpu(), cmap='hot', vmin=-1, vmax=1)
            axes[3, i].axis('off')
            if i == 0:
                axes[3, i].set_ylabel('Generated\nFinal', fontsize=10)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved comparison to {save_path}")
    plt.close()


def main():
    """Train unconditional Flow Matching on PDE diffusion data."""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    epochs = 500
    lr = 1e-4
    
    print("🏋️ Training Unconditional Flow Matching for PDE Generation")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Epochs: {epochs}")
    
    # Create dataset (focus on diffusion as Amir requested)
    dataset = PDEDataset(data_folder="train_data64", pde_type="diff")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Create model
    model = PDEFlowMatchingNet(channels=2, hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    losses = []
    os.makedirs("results/pde", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("\nTraining unconditional PDE generator...")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        
        for batch in dataloader:
            batch = batch.to(device)
            loss = train_step(model, batch, optimizer, device)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Generate samples
            model.eval()
            with torch.no_grad():
                samples = generate_samples(model, device, num_steps=50, num_samples=16)
                visualize_generated_samples(
                    samples, 
                    save_path=f"results/pde/unconditional_epoch_{epoch}.png"
                )
                
                # Compare with real data
                real_batch = next(iter(dataloader))[:8].to(device)
                compare_real_vs_generated(
                    real_batch, 
                    samples[:8],
                    save_path=f"results/pde/comparison_epoch_{epoch}.png"
                )
            model.train()
    
    # Save final model
    model_path = "models/pde_unconditional_generator.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
    }, model_path)
    print(f"\n✅ Saved model to {model_path}")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Unconditional PDE Generator Training')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/pde/unconditional_training.png', dpi=150, bbox_inches='tight')
    print("✅ Saved training curve")
    
    # Generate final samples
    print("\n🎨 Generating final samples...")
    model.eval()
    with torch.no_grad():
        final_samples = generate_samples(model, device, num_steps=100, num_samples=16)
        visualize_generated_samples(final_samples, save_path="results/pde/final_unconditional_samples.png")
    
    print("\n✅ Unconditional PDE generation training completed!")
    print("\n💡 This model can now generate realistic PDE solution pairs from noise.")
    print("   Next steps: Use this for inverse problems and inpainting as Amir suggested.")


if __name__ == "__main__":
    main()
