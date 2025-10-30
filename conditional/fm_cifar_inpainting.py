"""
Flow Matching Image Inpainting for CIFAR-10

Fills in masked regions of images using Flow Matching with conditioning on known pixels.
The model learns to generate realistic content in holes while preserving the surrounding context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from flow_matching_base import FlowMatchingBase

# Import some components from the main model
from conditional.fm_cifar_conditional import (
    ResBlock, AttentionBlock, EMAHelper, cosine_grid, sample_heun_optimized
)


class InpaintingDownBlock(nn.Module):
    """Downsampling block for inpainting U-Net."""
    
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


class InpaintingUpBlock(nn.Module):
    """Upsampling block for inpainting U-Net."""
    
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


class InpaintingUNet(FlowMatchingBase):
    """
    U-Net for Flow Matching Inpainting.
    
    Takes as input:
    - Masked image (holes filled with zeros)
    - Binary mask (1 = known pixel, 0 = hole)
    - Time t
    
    Outputs:
    - Velocity field for the entire image
    """
    
    def __init__(self, channels: int = 3, base_channels: int = 128):
        super().__init__(channels, base_channels)
        
        # Input channels: image (3) + mask (1) = 4 total
        input_channels = channels + 1
        
        # Time embedding
        time_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Initial convolution (takes masked image + mask)
        self.conv_in = nn.Conv2d(input_channels, base_channels, 3, padding=1)
        
        # U-Net encoder
        self.down1 = InpaintingDownBlock(base_channels, base_channels*2, time_dim, attention=False)
        self.down2 = InpaintingDownBlock(base_channels*2, base_channels*4, time_dim, attention=False)
        self.down3 = InpaintingDownBlock(base_channels*4, base_channels*8, time_dim, attention=True, downsample=False)
        
        # U-Net decoder
        self.up3 = InpaintingUpBlock(base_channels*8, base_channels*4, time_dim, attention=True, upsample=False)
        self.up2 = InpaintingUpBlock(base_channels*4, base_channels*2, time_dim, attention=True)
        self.up1 = InpaintingUpBlock(base_channels*2, base_channels, time_dim, attention=False)
        
        # Output projection
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, channels, 3, padding=1),
        )
        
        # Initialize output to zero
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
        t = t.squeeze(-1)
        emb = t[:, None] * self.time_embed[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb
    
    def forward(self, masked_image: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inpainting.
        
        Args:
            masked_image: Image with holes filled with zeros (B, 3, H, W)
            mask: Binary mask, 1=known pixel, 0=hole (B, 1, H, W)
            t: Time tensor (B, 1)
            
        Returns:
            Predicted velocity field (B, 3, H, W)
        """
        # Get time embedding
        t_sinusoidal = self._get_time_embedding(t)
        t_emb = self.time_embedding(t_sinusoidal)
        
        # Concatenate masked image and mask as input
        x = torch.cat([masked_image, mask], dim=1)  # (B, 4, H, W)
        
        # Initial convolution
        x = self.conv_in(x)
        
        # U-Net encoder with skip connections
        x1, skip1 = self.down1(x, t_emb)
        x2, skip2 = self.down2(x1, t_emb)
        x3, skip3 = self.down3(x2, t_emb)
        
        # U-Net decoder with skip connections
        x = self.up3(x3, skip3, t_emb)
        x = self.up2(x, skip2, t_emb)
        x = self.up1(x, skip1, t_emb)
        
        # Output velocity field
        return self.conv_out(x)


def create_random_mask(batch_size, height, width, hole_size_range=(8, 16), num_holes_range=(1, 3)):
    """
    Create random rectangular masks for inpainting.
    
    Args:
        batch_size: Number of masks to create
        height, width: Image dimensions
        hole_size_range: (min_size, max_size) for hole dimensions
        num_holes_range: (min_holes, max_holes) per image
        
    Returns:
        Binary mask tensor (batch_size, 1, height, width)
        1 = known pixel, 0 = hole to fill
    """
    masks = torch.ones(batch_size, 1, height, width)
    
    for b in range(batch_size):
        num_holes = np.random.randint(num_holes_range[0], num_holes_range[1] + 1)
        
        for _ in range(num_holes):
            # Random hole size
            hole_h = np.random.randint(hole_size_range[0], hole_size_range[1] + 1)
            hole_w = np.random.randint(hole_size_range[0], hole_size_range[1] + 1)
            
            # Random position (ensure hole fits in image)
            start_h = np.random.randint(0, height - hole_h + 1)
            start_w = np.random.randint(0, width - hole_w + 1)
            
            # Create hole (set to 0)
            masks[b, 0, start_h:start_h+hole_h, start_w:start_w+hole_w] = 0
    
    return masks


def inpainting_train_step(model, data_batch, optimizer, device):
    """
    Training step for inpainting Flow Matching.
    
    Process:
    1. Create random masks for the batch
    2. Apply masks to create masked images
    3. Train Flow Matching to generate full images from masked inputs
    """
    model.train()
    batch_size = data_batch.size(0)
    
    # Create random masks
    masks = create_random_mask(batch_size, 32, 32).to(device)
    
    # Apply masks to create masked images
    masked_images = data_batch * masks  # Holes become zeros
    
    # Sample noise and time for Flow Matching
    noise = torch.randn_like(data_batch)
    t = torch.rand(batch_size, 1, device=device)
    
    # Flow Matching interpolation
    t_expanded = t.view(-1, 1, 1, 1)
    x_t = (1 - t_expanded) * noise + t_expanded * data_batch
    
    # Target velocity (full image)
    target_velocity = data_batch - noise
    
    # Predict velocity using masked image + mask as conditioning
    predicted_velocity = model(masked_images, masks, t)
    
    # Loss only on the masked regions (holes)
    # This encourages the model to focus on filling holes correctly
    hole_mask = (1 - masks)  # 1 where holes are, 0 where known pixels are
    
    # Compute loss weighted by hole regions
    loss_full = F.mse_loss(predicted_velocity, target_velocity, reduction='none')
    loss_holes = (loss_full * hole_mask).sum() / (hole_mask.sum() + 1e-8)
    loss_known = (loss_full * masks).sum() / (masks.sum() + 1e-8)
    
    # Combine losses (focus more on holes, but don't ignore known regions)
    loss = 2.0 * loss_holes + 0.5 * loss_known
    
    # Optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(), loss_holes.item(), loss_known.item()


@torch.no_grad()
def inpaint_image(model, masked_image, mask, device, num_steps=15):
    """
    Inpaint a single masked image using Flow Matching.
    
    Args:
        model: Trained inpainting model
        masked_image: Image with holes (1, 3, H, W)
        mask: Binary mask (1, 1, H, W)
        device: Device to run on
        num_steps: Number of integration steps
        
    Returns:
        Inpainted image (1, 3, H, W)
    """
    model.eval()
    
    # Start from noise
    x = torch.randn_like(masked_image)
    
    # Integration with Heun's method
    time_steps = torch.linspace(0, 1, num_steps + 1, device=device)
    
    for i in range(num_steps):
        t_curr = time_steps[i].expand(1, 1)
        t_next = time_steps[i + 1].expand(1, 1)
        dt = (t_next - t_curr).view(-1, 1, 1, 1)
        
        # Heun's method
        v1 = model(masked_image, mask, t_curr)
        x_pred = x + v1 * dt
        v2 = model(masked_image, mask, t_next)
        x = x + 0.5 * (v1 + v2) * dt
        
        # Constrain known pixels to match the original image
        # This is key for inpainting: preserve known regions
        x = x * (1 - mask) + masked_image * mask
    
    return x


def create_cifar_inpainting_loader(batch_size=32):
    """Create CIFAR-10 data loader for inpainting."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)  # Scale to [-1, 1]
    ])
    
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def visualize_inpainting_results(original, masked, mask, inpainted, save_path="inpainting_results.png"):
    """Visualize inpainting results: original, masked, and inpainted images."""
    # Convert to [0, 1] for visualization
    original = (original.cpu() + 1) / 2
    masked = (masked.cpu() + 1) / 2
    inpainted = (inpainted.cpu() + 1) / 2
    mask = mask.cpu()
    
    batch_size = min(8, original.size(0))
    
    fig, axes = plt.subplots(4, batch_size, figsize=(2*batch_size, 8))
    
    for i in range(batch_size):
        # Original image
        axes[0, i].imshow(original[i].permute(1, 2, 0).clamp(0, 1))
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")
        
        # Masked image
        axes[1, i].imshow(masked[i].permute(1, 2, 0).clamp(0, 1))
        axes[1, i].set_title("Masked")
        axes[1, i].axis("off")
        
        # Mask visualization
        axes[2, i].imshow(mask[i, 0], cmap='gray')
        axes[2, i].set_title("Mask")
        axes[2, i].axis("off")
        
        # Inpainted result
        axes[3, i].imshow(inpainted[i].permute(1, 2, 0).clamp(0, 1))
        axes[3, i].set_title("Inpainted")
        axes[3, i].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved inpainting results to {save_path}")


def main():
    """Main training and evaluation pipeline for inpainting."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Data and model
    train_loader = create_cifar_inpainting_loader(batch_size=32)
    model = InpaintingUNet(channels=3, base_channels=128).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Inpainting Flow Matching Features:")
    print("- U-Net with mask conditioning")
    print("- Hole-focused loss weighting")
    print("- Constrained generation (preserve known pixels)")
    print("- 15-step fast inpainting")
    
    # Training setup
    num_epochs = 50
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    ema_helper = EMAHelper(model, decay=0.9999)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_hole_losses = []
        epoch_known_losses = []
        
        # Warmup
        if epoch < 5:
            warmup_lr = 2e-4 * (epoch + 1) / 5
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        for batch_idx, (data, _) in enumerate(train_loader):  # Ignore labels for inpainting
            data = data.to(device)
            
            loss, hole_loss, known_loss = inpainting_train_step(model, data, optimizer, device)
            epoch_losses.append(loss)
            epoch_hole_losses.append(hole_loss)
            epoch_known_losses.append(known_loss)
            
            ema_helper.update(model)
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}")
                print(f"  Total Loss: {loss:.4f}, Hole Loss: {hole_loss:.4f}, Known Loss: {known_loss:.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_hole_loss = sum(epoch_hole_losses) / len(epoch_hole_losses)
        avg_known_loss = sum(epoch_known_losses) / len(epoch_known_losses)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1} completed:")
        print(f"  Avg Loss: {avg_loss:.4f}, Hole: {avg_hole_loss:.4f}, Known: {avg_known_loss:.4f}")
        print(f"  LR: {current_lr:.2e}")
        
        if epoch >= 5:
            scheduler.step()
    
    # Switch to EMA weights for evaluation
    print("Switching to EMA weights for evaluation...")
    ema_helper.apply_shadow(model)
    
    # Test inpainting on some examples
    print("Testing inpainting...")
    test_data = next(iter(train_loader))[0][:8].to(device)  # Get 8 test images
    
    # Create test masks
    test_masks = create_random_mask(8, 32, 32, hole_size_range=(10, 20), num_holes_range=(1, 2)).to(device)
    test_masked = test_data * test_masks
    
    # Inpaint
    inpainted_results = []
    for i in range(8):
        inpainted = inpaint_image(
            model, 
            test_masked[i:i+1], 
            test_masks[i:i+1], 
            device, 
            num_steps=15
        )
        inpainted_results.append(inpainted)
    
    inpainted_batch = torch.cat(inpainted_results, dim=0)
    
    # Visualize results
    visualize_inpainting_results(
        test_data, test_masked, test_masks, inpainted_batch,
        save_path="results/cifar/inpainting_results.png"
    )
    
    # Restore original weights
    ema_helper.restore(model)
    
    print("\nINPAINTING RESULTS:")
    print("Successfully trained Flow Matching for image inpainting")
    print("Key features: Mask conditioning + hole-focused loss + constrained generation")
    print("Check results/cifar/inpainting_results.png for visual results")


if __name__ == "__main__":
    main()