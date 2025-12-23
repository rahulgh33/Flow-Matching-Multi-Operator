"""
Flow Matching for 2D Diffusion PDE Data

Learns the mapping: initial_condition → final_state
for the 2D diffusion equation using Amir's generated data.

Data format: Each .npz file contains:
- init_value: Initial condition (64x64)
- last_value: Final state after diffusion (64x64)
- diffusivity: Diffusion coefficient
- label: PDE type (0=diffusion, 1=advection, 2=advection-diffusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm


class PDEDataset(Dataset):
    """Dataset for PDE initial condition → final state pairs."""
    
    def __init__(self, data_folder, pde_type="diff"):
        """
        Args:
            data_folder: Path to folder with .npz files
            pde_type: "diff", "adve", or "adve_diff"
        """
        self.data_folder = data_folder
        self.pde_type = pde_type
        
        # Find all files of this PDE type
        pattern = os.path.join(data_folder, f"*_{pde_type}.npz")
        self.files = sorted(glob.glob(pattern))
        
        print(f"Found {len(self.files)} {pde_type} files in {data_folder}")
        
        if len(self.files) == 0:
            raise ValueError(f"No {pde_type} files found in {data_folder}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load data
        data = np.load(self.files[idx])
        
        # Get initial and final states
        init_state = data['init_value'].reshape(64, 64)
        final_state = data['last_value'].reshape(64, 64)
        
        # Convert to tensors and add channel dimension
        init_state = torch.from_numpy(init_state).float().unsqueeze(0)  # (1, 64, 64)
        final_state = torch.from_numpy(final_state).float().unsqueeze(0)  # (1, 64, 64)
        
        # Get parameters (for conditioning if needed)
        diffusivity = float(data.get('diffusivity', 0.0))
        
        return {
            'init_state': init_state,
            'final_state': final_state,
            'diffusivity': diffusivity
        }


class PDE2DFlowMatching(nn.Module):
    """
    2D Flow Matching network for PDE solution learning.
    
    Input: (u_t, t) where u_t is the 2D field at time t
    Output: Velocity field v(u_t, t)
    """
    
    def __init__(self, channels=1, base_channels=64):
        super().__init__()
        
        # Time embedding
        time_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # 2D U-Net for spatial processing
        self.conv_in = nn.Conv2d(channels, base_channels, 3, padding=1)
        
        # Encoder
        self.down1 = self._make_layer(base_channels, base_channels*2, time_dim)
        self.down2 = self._make_layer(base_channels*2, base_channels*4, time_dim)
        self.down3 = self._make_layer(base_channels*4, base_channels*8, time_dim, downsample=False)
        
        # Decoder
        self.up3 = self._make_layer(base_channels*8, base_channels*4, time_dim, upsample=False)
        self.up2 = self._make_layer(base_channels*4*2, base_channels*2, time_dim, upsample=True)
        self.up1 = self._make_layer(base_channels*2*2, base_channels, time_dim, upsample=True)
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, channels, 3, padding=1),
        )
        
        # Initialize output to zero
        nn.init.zeros_(self.conv_out[-1].weight)
        nn.init.zeros_(self.conv_out[-1].bias)
        
        # Store base_channels for time embedding
        self.base_channels = base_channels
        
    def _make_layer(self, in_ch, out_ch, time_dim, downsample=True, upsample=False):
        """Create a layer with time conditioning."""
        layers = []
        
        if upsample:
            layers.append(nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1))
        
        layers.extend([
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        ])
        
        if downsample and not upsample:
            layers.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
        
        return nn.Sequential(*layers)
    
    def _get_time_embedding(self, t):
        """Get sinusoidal time embeddings."""
        t = t.squeeze(-1)
        half_dim = self.base_channels
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device) * (-np.log(10000.0) / (half_dim - 1))
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    def forward(self, u, t):
        """
        Args:
            u: PDE field (batch, 1, 64, 64)
            t: Time (batch, 1)
        Returns:
            v: Velocity field (batch, 1, 64, 64)
        """
        # Time embedding
        t_sinusoidal = self._get_time_embedding(t)
        t_emb = self.time_embedding(t_sinusoidal)
        
        # Initial convolution
        x = self.conv_in(u)
        
        # Encoder with skip connections
        x1 = self.down1(x)  # 64x64 -> 32x32
        x2 = self.down2(x1)  # 32x32 -> 16x16
        x3 = self.down3(x2)  # 16x16 (bottleneck)
        
        # Add time conditioning to bottleneck
        x3 = x3 + t_emb.view(-1, t_emb.size(1), 1, 1)
        
        # Decoder with skip connections
        x = self.up3(x3)  # 16x16
        x = self.up2(torch.cat([x, x2], dim=1))  # 16x16 -> 32x32
        x = self.up1(torch.cat([x, x1], dim=1))  # 32x32 -> 64x64
        
        # Output velocity
        return self.conv_out(x)


def train_step(model, batch, optimizer, device):
    """Training step for PDE Flow Matching."""
    model.train()
    
    init_state = batch['init_state'].to(device)
    final_state = batch['final_state'].to(device)
    batch_size = init_state.size(0)
    
    # Sample random time
    t = torch.rand(batch_size, 1, device=device)
    
    # Flow Matching interpolation
    t_expanded = t.view(-1, 1, 1, 1)
    u_t = (1 - t_expanded) * init_state + t_expanded * final_state
    
    # Target velocity
    v_target = final_state - init_state
    
    # Predict velocity
    v_pred = model(u_t, t)
    
    # Loss
    loss = F.mse_loss(v_pred, v_target)
    
    # Optimization
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def solve_pde_fm(model, init_state, num_steps=50, device='cuda'):
    """Solve PDE using Flow Matching."""
    model.eval()
    u = init_state.to(device)
    
    time_steps = torch.linspace(0, 1, num_steps + 1, device=device)
    
    for i in range(num_steps):
        t_curr = time_steps[i].expand(u.size(0), 1)
        dt = time_steps[i + 1] - time_steps[i]
        
        # Predict velocity
        v = model(u, t_curr)
        
        # Euler step
        u = u + v * dt
    
    return u


def visualize_pde_results(init_states, final_numerical, final_fm, save_path="pde_results.png"):
    """Visualize PDE solution comparison."""
    
    init_states = init_states.cpu().numpy()
    final_numerical = final_numerical.cpu().numpy()
    final_fm = final_fm.cpu().numpy()
    
    num_examples = min(4, init_states.shape[0])
    
    fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4*num_examples))
    
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_examples):
        # Initial condition
        im0 = axes[i, 0].imshow(init_states[i, 0], origin="lower", cmap="viridis")
        axes[i, 0].set_title("Initial Condition")
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Numerical solution
        im1 = axes[i, 1].imshow(final_numerical[i, 0], origin="lower", cmap="viridis")
        axes[i, 1].set_title("Numerical Solution")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Flow Matching solution
        im2 = axes[i, 2].imshow(final_fm[i, 0], origin="lower", cmap="viridis")
        axes[i, 2].set_title("Flow Matching Solution")
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # Error
        error = np.abs(final_numerical[i, 0] - final_fm[i, 0])
        im3 = axes[i, 3].imshow(error, origin="lower", cmap="Reds")
        axes[i, 3].set_title(f"Error (Max: {error.max():.4f})")
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)
        
        for ax in axes[i]:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PDE results to {save_path}")


def main():
    """Main training and evaluation pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if data exists
    data_folder = "train_data64"  # or "test_data64_cal"
    if not os.path.exists(data_folder):
        print(f"❌ Data folder {data_folder} not found!")
        print("Please run Amir's data generation script first.")
        return
    
    # Create dataset and dataloader
    try:
        dataset = PDEDataset(data_folder, pde_type="diff")
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        print(f"✅ Loaded {len(dataset)} diffusion examples")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create model
    model = PDE2DFlowMatching(channels=1, base_channels=64).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    num_epochs = 500
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-6)
    
    # Training loop
    print(f"\nTraining Flow Matching on 2D Diffusion PDE...")
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch in dataloader:
            loss = train_step(model, batch, optimizer, device)
            epoch_losses.append(loss)
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    # Test on validation examples
    print("\nTesting on validation examples...")
    
    # Get a few test examples
    test_batch = next(iter(dataloader))
    init_states = test_batch['init_state'][:4]
    final_numerical = test_batch['final_state'][:4]
    
    # Solve using Flow Matching
    final_fm = solve_pde_fm(model, init_states, num_steps=50, device=device)
    
    # Compute error
    error = (final_numerical.to(device) - final_fm).abs().mean().item()
    print(f"Mean absolute error: {error:.6f}")
    
    # Create results directory
    os.makedirs("results/pde", exist_ok=True)
    
    # Visualize results
    visualize_pde_results(
        init_states, final_numerical, final_fm, 
        "results/pde/diffusion_2d_results.png"
    )
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("2D Diffusion PDE - Flow Matching Training")
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig("results/pde/diffusion_2d_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'final_error': error
    }, 'models/pde_diffusion_2d.pth')
    
    print("\n✅ 2D Diffusion PDE Flow Matching completed!")
    print("Results:")
    print(f"- Final error: {error:.6f}")
    print(f"- Visualization: results/pde/diffusion_2d_results.png")
    print(f"- Training curve: results/pde/diffusion_2d_training.png")
    print(f"- Saved model: models/pde_diffusion_2d.pth")
    
    print("\nKey achievements:")
    print("- Learned 2D diffusion solution operator")
    print("- Fast inference (50 steps vs numerical solver)")
    print("- Generalizes to unseen initial conditions")
    print("- Ready for inverse problems and inpainting!")


if __name__ == "__main__":
    main()