"""
Optimized Conditional Flow Matching for CIFAR-10

This implementation incorporates several key optimizations for fast, high-quality generation:
1. Straightened path training (α(t) = t²) for smoother ODE integration
2. Beta(0.5,0.5) time sampling for endpoint emphasis
3. EMA weights for improved sample quality
4. Cosine time grid + Heun solver for efficient few-step generation

Target: 15-20 steps vs 100-step diffusion models (5-7x speedup)
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
    """Residual block with FiLM conditioning for time and class information."""
    
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, channels//4), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, channels//4), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
        # FiLM conditioning: Feature-wise Linear Modulation
        self.cond_proj = nn.Linear(cond_dim, channels * 2)
        self.activation = nn.SiLU()  # Swish activation for better gradients
        
    def forward(self, x, cond):
        residual = x
        
        # First conv block
        h = self.activation(self.norm1(x))
        h = self.conv1(h)
        
        # Apply FiLM conditioning
        scale, shift = self.cond_proj(cond).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        # Second conv block
        h = self.activation(self.norm2(h))
        h = self.conv2(h)
        
        return h + residual


class SelfAttention(nn.Module):
    """Multi-head self-attention for capturing global dependencies."""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(min(32, channels//4), channels)
        self.num_heads = 8
        self.head_dim = channels // self.num_heads
        
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        
        qkv = self.qkv(h).view(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bhij,bhdj->bhdi', attn, v)
        out = out.contiguous().view(B, C, H, W)
        
        return x + self.proj_out(out)


class ConditionalFlowMatchingNetCIFAR(FlowMatchingBase):
    """
    Optimized Flow Matching Network for CIFAR-10
    
    Key features:
    - ResNet encoder-decoder (no skip connections for cleaner FM story)
    - Class-conditional generation via FiLM
    - Optimized for few-step generation (15-20 steps)
    """
    
    def __init__(self, channels: int = 3, hidden_dim: int = 160, num_classes: int = 10):
        super().__init__(channels, hidden_dim)
        self.num_classes = num_classes
        
        # Enhanced conditioning embeddings
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4)
        )
        
        self.class_embedding = nn.Sequential(
            nn.Embedding(num_classes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(), 
            nn.Linear(hidden_dim * 2, hidden_dim * 4)
        )
        
        # Encoder: 32x32 -> 8x8
        self.conv_in = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        
        # Down path
        self.down1 = nn.ModuleList([
            ResBlock(hidden_dim, hidden_dim * 4),
            ResBlock(hidden_dim, hidden_dim * 4),
        ])
        self.down1_conv = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1)
        
        self.down2 = nn.ModuleList([
            ResBlock(hidden_dim * 2, hidden_dim * 4),
            ResBlock(hidden_dim * 2, hidden_dim * 4),
        ])
        self.down2_conv = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, stride=2, padding=1)
        
        # Bottleneck with attention
        self.bottleneck = nn.ModuleList([
            ResBlock(hidden_dim * 4, hidden_dim * 4),
            ResBlock(hidden_dim * 4, hidden_dim * 4),
            SelfAttention(hidden_dim * 4),
            ResBlock(hidden_dim * 4, hidden_dim * 4),
        ])
        
        # Up path (NO skip connections - key for clean FM)
        self.up2_conv = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, stride=2, padding=1)
        self.up2 = nn.ModuleList([
            ResBlock(hidden_dim * 2, hidden_dim * 4),
            ResBlock(hidden_dim * 2, hidden_dim * 4),
        ])
        
        self.up1_conv = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, stride=2, padding=1)
        self.up1 = nn.ModuleList([
            ResBlock(hidden_dim, hidden_dim * 4),
            ResBlock(hidden_dim, hidden_dim * 4),
        ])
        
        # Output layer
        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(32, hidden_dim//4), hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, channels, 3, padding=1)
        )
        
        # Zero-initialize final layer for training stability
        nn.init.zeros_(self.conv_out[-1].weight)
        nn.init.zeros_(self.conv_out[-1].bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict velocity field v(x,t,c)."""
        # Embed conditioning information
        t_emb = self.time_embedding(t)
        c_emb = self.class_embedding(class_labels)
        cond = t_emb + c_emb  # Combined conditioning
        
        # Encoder
        h = self.conv_in(x)
        
        for block in self.down1:
            h = block(h, cond)
        h = self.down1_conv(h)
        
        for block in self.down2:
            h = block(h, cond)
        h = self.down2_conv(h)
        
        # Bottleneck
        for block in self.bottleneck:
            if isinstance(block, SelfAttention):
                h = block(h)
            else:
                h = block(h, cond)
        
        # Decoder (no skip connections)
        h = self.up2_conv(h)
        for block in self.up2:
            h = block(h, cond)
            
        h = self.up1_conv(h)
        for block in self.up1:
            h = block(h, cond)
        
        return self.conv_out(h)


# ============================================================================
# OPTIMIZED SAMPLING AND TRAINING
# ============================================================================

def cosine_grid(N, device):
    """Cosine time grid that concentrates steps near t=0 and t=1."""
    u = torch.linspace(0, 1, N+1, device=device)
    s = torch.sin(0.5 * torch.pi * u)
    return (s - s[0]) / (s[-1] - s[0])


@torch.no_grad()
def sample_heun_optimized(model, x, labels, steps=20):
    """Optimized Heun sampler with cosine time grid."""
    ts = cosine_grid(steps, x.device)
    for i in range(steps):
        t0 = ts[i].expand(x.size(0), 1)
        t1 = ts[i+1].expand(x.size(0), 1)
        dt = (t1 - t0).view(-1, 1, 1, 1)
        
        # Heun's method (2nd order ODE solver)
        v0 = model(x, t0, labels)
        x_pred = x + v0 * dt
        v1 = model(x_pred, t1, labels)
        x = x + 0.5 * (v0 + v1) * dt
        
        # Stability clamp
        x = torch.clamp(x, -3, 3)
    return x


def optimized_train_step(model, data_batch, labels_batch, optimizer, device):
    """
    Optimized training step with:
    1. Straightened path (α(t) = t²)
    2. Beta(0.5,0.5) time sampling for endpoint emphasis
    """
    model.train()
    batch_size = data_batch.size(0)
    noise = torch.randn_like(data_batch)
    
    # Beta(0.5,0.5) sampling for U-shaped distribution (endpoint emphasis)
    beta_dist = torch.distributions.Beta(0.5, 0.5)
    t = beta_dist.sample((batch_size, 1)).to(device)
    
    # Straightened path: α(t) = t² makes ODE easier to integrate
    gamma = 2.0
    t_expanded = t.view(-1, 1, 1, 1)
    alpha = t_expanded ** gamma
    dalpha_dt = gamma * (t_expanded ** (gamma - 1) + 1e-6)
    
    # Interpolation and target velocity
    x_t = (1 - alpha) * noise + alpha * data_batch
    target_velocity = dalpha_dt * (data_batch - noise)
    
    # Predict and compute loss
    predicted_velocity = model(x_t, t, labels_batch)
    loss = F.mse_loss(predicted_velocity, target_velocity)
    
    # Optimization step
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
def generate_samples(model, device, class_labels, num_steps=15):
    """Generate samples using optimized settings."""
    model.eval()
    x = torch.randn(len(class_labels), 3, 32, 32, device=device)
    return sample_heun_optimized(model, x, class_labels, steps=num_steps)


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


# ============================================================================
# MAIN TRAINING AND EVALUATION
# ============================================================================

def main():
    """Main training and evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Data and model
    train_loader = create_data_loader(batch_size=64)
    model = ConditionalFlowMatchingNetCIFAR(channels=3, hidden_dim=160, num_classes=10).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Key Optimizations:")
    print("- Straightened path (α(t) = t²)")
    print("- Beta(0.5,0.5) time sampling")
    print("- EMA weights (decay=0.999)")
    print("- Cosine time grid + Heun solver")
    
    # Training setup
    num_epochs = 50
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-1, eta_min=1e-6)
    ema_helper = EMAHelper(model, decay=0.999)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Warmup learning rate for first epoch
        if epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
        elif epoch == 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            loss = optimized_train_step(model, data, labels, optimizer, device)
            epoch_losses.append(loss)
            ema_helper.update(model)
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss:.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
        
        if epoch >= 1:
            scheduler.step()
    
    # Generation with EMA weights
    print("Switching to EMA weights for generation...")
    ema_helper.apply_shadow(model)
    
    # Generate comparison samples
    all_labels = torch.arange(10, device=device).repeat(2)
    
    configs = [
        (15, "Optimized Target (15 steps)"),
        (100, "Baseline (100 steps)")
    ]
    
    for steps, desc in configs:
        print(f"Generating {desc}...")
        if steps == 100:
            # Use simple Euler for baseline comparison
            x = torch.randn(len(all_labels), 3, 32, 32, device=device)
            time_steps = torch.linspace(0, 1, steps + 1, device=device)
            for i in range(steps):
                t_curr = time_steps[i].expand(x.size(0), 1)
                dt = time_steps[i + 1] - time_steps[i]
                v = model(x, t_curr, all_labels)
                x = x + v * dt
            samples = x
        else:
            samples = generate_samples(model, device, all_labels, num_steps=steps)
        
        filename = f"results/{steps}steps.png"
        visualize_samples(samples, all_labels, save_path=filename)
    
    # Restore original weights
    ema_helper.restore(model)
    
    print("\nOPTIMIZATION RESULTS:")
    print("Target: 15 steps vs 100 steps = 6.7x speedup")
    print("Key innovations: Straightened path + Beta sampling + EMA + Cosine grid")
    print("Architecture: ResNet encoder-decoder (no skip connections)")
    print("Check generated images for quality comparison")


if __name__ == "__main__":
    main()