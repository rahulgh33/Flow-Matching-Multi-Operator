"""
U-Net Flow Matching for CIFAR-10

Full U-Net implementation optimized for fast inference with Flow Matching:
- U-Net architecture with skip connections for better quality
- Multi-scale attention for global context
- Optimized training with EMA and advanced solvers
- Target: 15 steps for fast, high-quality generation
"""

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
from flow_matching_base import FlowMatchingBase

# CIFAR-10 class names for visualization
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']


class ResBlock(nn.Module):
    """ResNet block with time and class conditioning via FiLM."""
    
    def __init__(self, in_channels, out_channels, cond_dim, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Group normalization (better than BatchNorm for small batches)
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # FiLM conditioning
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
            
        self.activation = nn.SiLU()
        
    def forward(self, x, cond):
        h = self.activation(self.norm1(x))
        h = self.conv1(h)
        
        # Apply FiLM conditioning
        scale, shift = self.cond_proj(cond).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        h = self.activation(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Multi-head attention block for U-Net."""
    
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        
        qkv = self.qkv(h).view(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bhij,bhdj->bhdi', attn, v)
        out = out.contiguous().view(B, C, H, W)
        
        return x + self.proj_out(out)


class DownBlock(nn.Module):
    """Downsampling block with ResBlocks and optional attention."""
    
    def __init__(self, in_channels, out_channels, cond_dim, num_layers=2, downsample=True, attention=False):
        super().__init__()
        self.downsample = downsample
        
        # ResNet blocks
        layers = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(ResBlock(in_ch, out_channels, cond_dim))
        self.layers = nn.ModuleList(layers)
        
        # Optional attention
        self.attention = AttentionBlock(out_channels) if attention else None
        
        # Downsampling
        if downsample:
            self.downsample_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        
    def forward(self, x, cond):
        for layer in self.layers:
            x = layer(x, cond)
            
        if self.attention:
            x = self.attention(x)
            
        if self.downsample:
            return self.downsample_conv(x), x  # Return both downsampled and skip
        else:
            return x, x


class UpBlock(nn.Module):
    """Upsampling block with ResBlocks, skip connections, and optional attention."""
    
    def __init__(self, in_channels, out_channels, cond_dim, num_layers=2, upsample=True, attention=False):
        super().__init__()
        self.upsample = upsample
        
        # Upsampling
        if upsample:
            self.upsample_conv = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        
        # ResNet blocks (input has skip connection concatenated)
        layers = []
        for i in range(num_layers):
            in_ch = in_channels * 2 if i == 0 else out_channels  # *2 for skip connection
            layers.append(ResBlock(in_ch, out_channels, cond_dim))
        self.layers = nn.ModuleList(layers)
        
        # Optional attention
        self.attention = AttentionBlock(out_channels) if attention else None
        
    def forward(self, x, skip, cond):
        if self.upsample:
            x = self.upsample_conv(x)
            
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        for layer in self.layers:
            x = layer(x, cond)
            
        if self.attention:
            x = self.attention(x)
            
        return x


class UNetFlowMatching(FlowMatchingBase):
    """
    U-Net architecture optimized for Flow Matching on CIFAR-10.
    
    Features:
    - Full U-Net with skip connections for better quality
    - Multi-scale attention for global context
    - FiLM conditioning throughout the network
    - Optimized for 15-step generation
    """
    
    def __init__(self, channels: int = 3, base_channels: int = 128, num_classes: int = 10):
        super().__init__(channels, base_channels)
        self.num_classes = num_classes
        
        # Time embedding (sinusoidal + MLP)
        time_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Class embedding
        self.class_embedding = nn.Sequential(
            nn.Embedding(num_classes, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, time_dim),
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(channels, base_channels, 3, padding=1)
        
        # U-Net encoder (downsampling path)
        self.down1 = DownBlock(base_channels, base_channels*2, time_dim, attention=False)    # 32x32 -> 16x16
        self.down2 = DownBlock(base_channels*2, base_channels*4, time_dim, attention=False)  # 16x16 -> 8x8  
        self.down3 = DownBlock(base_channels*4, base_channels*8, time_dim, attention=True, downsample=False)  # 8x8 (bottleneck)
        
        # U-Net decoder (upsampling path)
        self.up3 = UpBlock(base_channels*8, base_channels*4, time_dim, attention=True, upsample=False)  # 8x8
        self.up2 = UpBlock(base_channels*4, base_channels*2, time_dim, attention=True)   # 8x8 -> 16x16
        self.up1 = UpBlock(base_channels*2, base_channels, time_dim, attention=False)    # 16x16 -> 32x32
        
        # Output projection
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, channels, 3, padding=1),
        )
        
        # Initialize output layer to zero for stable training
        nn.init.zeros_(self.conv_out[-1].weight)
        nn.init.zeros_(self.conv_out[-1].bias)
        
        # Sinusoidal time embedding
        self.register_buffer('time_embed', self._build_sinusoidal_embedding(base_channels))
        
    def _build_sinusoidal_embedding(self, dim):
        """Create sinusoidal position embeddings."""
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        return emb
        
    def _get_time_embedding(self, t):
        """Get sinusoidal time embeddings."""
        # t is (batch_size, 1), we want (batch_size, dim)
        t = t.squeeze(-1)  # (batch_size,)
        emb = t[:, None] * self.time_embed[None, :]  # (batch_size, dim//2)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (batch_size, dim)
        return emb
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net."""
        # Get embeddings
        t_sinusoidal = self._get_time_embedding(t)
        t_emb = self.time_embedding(t_sinusoidal)
        c_emb = self.class_embedding(class_labels)
        
        # Combine time and class conditioning
        cond = t_emb + c_emb
        
        # Initial convolution
        x = self.conv_in(x)
        
        # Encoder with skip connections
        x1, skip1 = self.down1(x, cond)      # 32x32 -> 16x16
        x2, skip2 = self.down2(x1, cond)     # 16x16 -> 8x8
        x3, skip3 = self.down3(x2, cond)     # 8x8 (bottleneck)
        
        # Decoder with skip connections
        x = self.up3(x3, skip3, cond)        # 8x8
        x = self.up2(x, skip2, cond)         # 8x8 -> 16x16
        x = self.up1(x, skip1, cond)         # 16x16 -> 32x32
        
        # Output
        return self.conv_out(x)


# ============================================================================
# OPTIMIZED SAMPLING AND TRAINING
# ============================================================================

def cosine_grid(N, device):
    """Cosine time grid that concentrates steps near t=0 and t=1."""
    u = torch.linspace(0, 1, N+1, device=device)
    s = torch.sin(0.5 * torch.pi * u)
    return (s - s[0]) / (s[-1] - s[0])


@torch.no_grad()
def sample_heun_optimized(model, x, labels, steps=20, guidance_scale=1.0):
    """Optimized Heun sampler with cosine time grid and optional guidance."""
    ts = cosine_grid(steps, x.device)
    for i in range(steps):
        t0 = ts[i].expand(x.size(0), 1)
        t1 = ts[i+1].expand(x.size(0), 1)
        dt = (t1 - t0).view(-1, 1, 1, 1)
        
        # Heun's method with classifier-free guidance
        if guidance_scale > 1.0:
            # First step
            v0_cond = model(x, t0, labels)
            v0_uncond = model(x, t0, torch.full_like(labels, 10))  # null class
            v0 = v0_uncond + guidance_scale * (v0_cond - v0_uncond)
            
            x_pred = x + v0 * dt
            
            # Second step
            v1_cond = model(x_pred, t1, labels)
            v1_uncond = model(x_pred, t1, torch.full_like(labels, 10))
            v1 = v1_uncond + guidance_scale * (v1_cond - v1_uncond)
        else:
            v0 = model(x, t0, labels)
            x_pred = x + v0 * dt
            v1 = model(x_pred, t1, labels)
        
        x = x + 0.5 * (v0 + v1) * dt
        
        # Stability clamp
        x = torch.clamp(x, -3, 3)
    return x


def unet_train_step(model, data_batch, labels_batch, optimizer, device):
    """
    Optimized training step for U-Net Flow Matching.
    Uses standard linear interpolation with uniform time sampling.
    """
    model.train()
    batch_size = data_batch.size(0)
    
    # Sample noise and time
    noise = torch.randn_like(data_batch)
    t = torch.rand(batch_size, 1, device=device)
    
    # Linear interpolation (standard Flow Matching)
    t_expanded = t.view(-1, 1, 1, 1)
    x_t = (1 - t_expanded) * noise + t_expanded * data_batch
    
    # Target velocity (constant for linear path)
    target_velocity = data_batch - noise
    
    # Predict velocity
    predicted_velocity = model(x_t, t, labels_batch)
    
    # MSE loss
    loss = F.mse_loss(predicted_velocity, target_velocity)
    
    # Optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


class EMAHelper:
    """Exponential Moving Average for improved sample quality."""
    
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])


# ============================================================================
# GENERATION AND EVALUATION
# ============================================================================

@torch.no_grad()
def generate_samples(model, device, class_labels, num_steps=15, solver="heun", guidance_scale=1.0):
    """Generate samples with U-Net Flow Matching and optional guidance."""
    model.eval()
    x = torch.randn(len(class_labels), 3, 32, 32, device=device)
    
    if solver == "heun":
        return sample_heun_optimized(model, x, class_labels, steps=num_steps, guidance_scale=guidance_scale)
    else:
        # Simple Euler method with guidance
        time_steps = torch.linspace(0, 1, num_steps + 1, device=device)
        for i in range(num_steps):
            t_curr = time_steps[i].expand(x.size(0), 1)
            dt = time_steps[i + 1] - time_steps[i]
            
            if guidance_scale > 1.0:
                # Classifier-free guidance
                v_cond = model(x, t_curr, class_labels)
                v_uncond = model(x, t_curr, torch.full_like(class_labels, 10))  # 10 = null class
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = model(x, t_curr, class_labels)
            
            x = x + v * dt
        return x


def create_data_loader(batch_size=64):
    """Create CIFAR-10 data loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)  # Scale to [-1, 1]
    ])
    
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def visualize_samples(samples, labels, save_path="samples.png"):
    """Visualize generated samples with class labels."""
    samples = (samples.cpu() + 1) / 2  # Convert to [0, 1]
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("Flow Matching Results", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            img = samples[i].permute(1, 2, 0).clamp(0, 1)
            ax.imshow(img)
            class_name = CIFAR10_CLASSES[labels[i].item()]
            ax.set_title(f"{class_name}", fontsize=10)
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved samples to {save_path}")


def generate_specific_classes(model: UNetFlowMatching, device: torch.device, class_indices: list):
    """Generate specific CIFAR-10 classes with U-Net Flow Matching."""
    labels = torch.tensor(class_indices, device=device)
    samples = generate_samples(model, device, labels, num_steps=15, solver="heun")
    
    # Convert for visualization
    samples = (samples.cpu() + 1) / 2
    
    # Visualize
    fig, axes = plt.subplots(1, len(class_indices), figsize=(3*len(class_indices), 3))
    if len(class_indices) == 1:
        axes = [axes]
    
    fig.suptitle("Generated CIFAR-10 Classes", fontsize=16)
    
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


# ============================================================================
# MAIN TRAINING AND EVALUATION
# ============================================================================

def main():
    """Main training and evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Clear GPU memory and set conservative memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    
    # Data and model
    train_loader = create_data_loader(batch_size=24)  # Reduced batch for extended training
    model = UNetFlowMatching(channels=3, base_channels=128, num_classes=10).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Extended U-Net Flow Matching Features:")
    print("- Full U-Net with skip connections")
    print("- Multi-scale attention blocks")
    print("- FiLM conditioning throughout")
    print("- EMA weights for stable generation")
    print("- Two-phase training (100 + 50 epochs)")
    print("- Fine-tuning for better quality")
    
    # Extended training setup for better results
    num_epochs = 150  # Extended training
    
    # Two-phase training: normal then fine-tuning
    phase1_epochs = 100
    phase2_epochs = 50
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_epochs, eta_min=5e-5)
    ema_helper = EMAHelper(model, decay=0.999)  # Faster EMA for better tracking
    
    # Two-phase training loop
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Phase transition at epoch 100
        if epoch == phase1_epochs:
            print(f"\n🔄 Switching to fine-tuning phase (epochs {phase1_epochs+1}-{num_epochs})")
            # Lower learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-5
            # Reset scheduler for phase 2
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=1e-6)
        
        # Warmup learning rate for first few epochs only
        if epoch < 5:
            warmup_lr = 2e-4 * (epoch + 1) / 5
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            loss = unet_train_step(model, data, labels, optimizer, device)
            epoch_losses.append(loss)
            ema_helper.update(model)
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss:.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
        
        if epoch >= 5:  # Start scheduler after warmup
            scheduler.step()
    
    # Save the trained model
    print("Saving trained model...")
    import os
    os.makedirs("models", exist_ok=True)
    
    # Save both regular and EMA weights
    torch.save({
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema_helper.shadow,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'loss': avg_loss,
        'model_config': {
            'channels': 3,
            'base_channels': 128,
            'num_classes': 10
        }
    }, 'models/fm_unet_extended_final.pth')
    
    print("Model saved to models/fm_unet_extended_final.pth")
    print("Training completed successfully!")
    print("Use generate_samples.py to generate images from the trained model.")
    
    print("\nU-NET FLOW MATCHING RESULTS:")
    print("Target: 15 steps vs 100 steps = 6.7x speedup")
    print("Architecture: Full U-Net with skip connections and multi-scale attention")
    print("Key features: FiLM conditioning + EMA weights + Heun solver")
    print("Check generated images for quality comparison")


if __name__ == "__main__":
    main()