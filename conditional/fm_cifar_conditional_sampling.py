"""
Conditional Sampling for Flow Matching using Bayesian Posterior Updates

This implements the approach from your mentor:
    ∇_x log p(x|c) = ∇_x log p(x|τ) + ∇_x log p(c|x)
                      ↑ prior          ↑ likelihood

Key advantages:
- No retraining needed - uses existing unconditional/class-conditional model
- Works with sparse observations (not just continuous geometry masks)
- Bayesian framework: prior → posterior via observation guidance
- Flexible: same model for inpainting, super-resolution, conditional generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from conditional.fm_cifar_conditional import (
    UNetFlowMatching, cosine_grid, visualize_samples, CIFAR10_CLASSES
)


def validate_shapes(x_t, observed, mask, t, context=""):
    """Validate tensor shapes for debugging."""
    assert x_t.shape[0] == observed.shape[0] == mask.shape[0] == t.shape[0], \
        f"{context}: Batch size mismatch: x_t={x_t.shape[0]}, observed={observed.shape[0]}, mask={mask.shape[0]}, t={t.shape[0]}"
    assert x_t.shape[1:] == observed.shape[1:] == (3, 32, 32), \
        f"{context}: Image shape mismatch: x_t={x_t.shape}, observed={observed.shape}"
    assert mask.shape[1:] == (1, 32, 32), \
        f"{context}: Mask shape mismatch: {mask.shape}"
    assert t.shape[1] == 1, \
        f"{context}: Time shape mismatch: {t.shape}"


def compute_observation_likelihood_gradient(x_t, observed, mask, t, noise_scale=0.1):
    """
    Compute ∇_x log p(c | x_t) - the likelihood term.
    
    This enforces consistency with observations using a Gaussian likelihood:
    p(c | x) ∝ exp(-||x[mask] - observed[mask]||² / (2σ²))
    
    Args:
        x_t: Current state (B, C, H, W)
        observed: Known pixel values (B, C, H, W)
        mask: Binary mask (B, 1, H, W), 1=observed, 0=unknown
        t: Current time (B, 1)
        noise_scale: σ in the likelihood model
        
    Returns:
        Gradient ∇_x log p(c | x_t)
    """
    # Expand mask to match channels
    mask_expanded = mask.expand_as(x_t)
    
    # Compute gradient: ∇_x log p(c|x) = -(x - observed) / σ² on observed pixels
    grad = torch.zeros_like(x_t)
    grad[mask_expanded > 0.5] = -(x_t[mask_expanded > 0.5] - observed[mask_expanded > 0.5]) / (noise_scale ** 2)
    
    # Time-dependent scaling (stronger guidance near t=1)
    time_weight = t.view(-1, 1, 1, 1)
    grad = grad * time_weight
    
    return grad


def conditional_flow_matching_sample(
    model, 
    observed, 
    mask, 
    class_label,
    device,
    num_steps=30,
    guidance_strength=1.0,
    solver="heun"
):
    """
    Sample from posterior p(x | observations) using Bayesian flow guidance.
    
    This implements:
        dx/dt = v_θ(x_t, t, class) + λ * ∇_x log p(observations | x_t)
                ↑ prior (trained)      ↑ likelihood (computed)
    
    Args:
        model: Trained unconditional/class-conditional Flow Matching model
        observed: Known pixels (B, C, H, W)
        mask: Binary mask (B, 1, H, W), 1=observed, 0=unknown
        class_label: Class conditioning (B,)
        device: Device
        num_steps: Number of integration steps
        guidance_strength: λ - how strongly to enforce observations
        solver: "heun" or "euler"
        
    Returns:
        Generated image consistent with observations
    """
    model.eval()
    batch_size = observed.size(0)
    
    # Ensure all tensors on same device
    observed = observed.to(device)
    mask = mask.to(device)
    class_label = class_label.to(device)
    
    # Start from noise
    x = torch.randn_like(observed)
    
    # Initialize with observations (optional warm start)
    x = x * (1 - mask) + observed * mask
    
    # Debug: print shapes
    if batch_size <= 2:  # Only for small batches
        print(f"  Shape check: x={x.shape}, observed={observed.shape}, mask={mask.shape}")
    
    # Time grid
    if solver == "heun":
        ts = cosine_grid(num_steps, device)
    else:
        ts = torch.linspace(0, 1, num_steps + 1, device=device)
    
    for i in range(num_steps):
        t_curr = ts[i].expand(batch_size, 1).to(device)
        t_next = ts[i + 1].expand(batch_size, 1).to(device)
        dt = (t_next - t_curr).view(-1, 1, 1, 1)
        
        # Validate shapes (only first iteration)
        if i == 0:
            validate_shapes(x, observed, mask, t_curr, context=f"Step {i}")
        
        # Prior velocity: v_θ(x_t, t, class)
        v_prior = model(x, t_curr, class_label)
        
        # Likelihood gradient: ∇_x log p(observations | x_t)
        likelihood_grad = compute_observation_likelihood_gradient(
            x, observed, mask, t_curr
        )
        
        # Posterior velocity: Bayesian combination
        v_posterior = v_prior + guidance_strength * likelihood_grad
        
        if solver == "heun":
            # Heun's method with posterior velocity
            x_euler = x + v_posterior * dt
            
            # Second evaluation
            v_prior_next = model(x_euler, t_next, class_label)
            likelihood_grad_next = compute_observation_likelihood_gradient(
                x_euler, observed, mask, t_next
            )
            v_posterior_next = v_prior_next + guidance_strength * likelihood_grad_next
            
            # Heun update
            x = x + 0.5 * (v_posterior + v_posterior_next) * dt
        else:
            # Euler method
            x = x + v_posterior * dt
        
        # Hard constraint: project onto observations (optional but recommended)
        x = x * (1 - mask) + observed * mask
        
        # Stability clamp
        x = torch.clamp(x, -3, 3)
    
    return x


def create_sparse_observations(images, observation_ratio=0.3, pattern="random"):
    """
    Create sparse observations from images.
    
    Args:
        images: Full images (B, C, H, W)
        observation_ratio: Fraction of pixels to observe
        pattern: "random", "grid", "half", "edges"
        
    Returns:
        observed: Observed pixels (B, C, H, W)
        mask: Binary mask (B, 1, H, W)
    """
    B, C, H, W = images.shape
    device = images.device
    dtype = images.dtype
    
    if pattern == "random":
        # Random sparse pixels
        mask = (torch.rand(B, 1, H, W, device=device, dtype=dtype) < observation_ratio).float()
        
    elif pattern == "grid":
        # Regular grid pattern
        mask = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
        stride = int(np.sqrt(1 / observation_ratio))
        mask[:, :, ::stride, ::stride] = 1.0
        
    elif pattern == "half":
        # Half image (left or top)
        mask = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
        if torch.rand(1, device=device) > 0.5:
            mask[:, :, :, :W//2] = 1.0  # Left half
        else:
            mask[:, :, :H//2, :] = 1.0  # Top half
            
    elif pattern == "edges":
        # Edge pixels only
        mask = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
        mask[:, :, 0, :] = 1.0  # Top
        mask[:, :, -1, :] = 1.0  # Bottom
        mask[:, :, :, 0] = 1.0  # Left
        mask[:, :, :, -1] = 1.0  # Right
    
    # Create observed image
    observed = images * mask
    
    return observed, mask


def visualize_conditional_sampling(original, observed, mask, generated, save_path="conditional_sampling.png"):
    """Visualize conditional sampling results."""
    # Debug: Check inputs before conversion
    print(f"    Viz debug - mask mean: {mask.mean().item():.2%}, "
          f"observed non-zero: {(observed != 0).float().mean().item():.2%}, "
          f"gen vs orig diff: {(generated - original).abs().mean().item():.4f}")
    
    # Convert to [0, 1]
    original = ((original.cpu() + 1) / 2).clamp(0, 1)
    observed_vis = ((observed.cpu() + 1) / 2).clamp(0, 1)
    generated = ((generated.cpu() + 1) / 2).clamp(0, 1)
    mask = mask.cpu()
    
    batch_size = min(8, original.size(0))
    
    fig, axes = plt.subplots(4, batch_size, figsize=(2*batch_size, 8))
    
    # Handle single image case
    if batch_size == 1:
        axes = axes.reshape(4, 1)
    
    for i in range(batch_size):
        # Original
        axes[0, i].imshow(original[i].permute(1, 2, 0))
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")
        
        # Observations (sparse)
        axes[1, i].imshow(observed_vis[i].permute(1, 2, 0))
        axes[1, i].set_title("Observations")
        axes[1, i].axis("off")
        
        # Mask
        axes[2, i].imshow(mask[i, 0], cmap='gray')
        axes[2, i].set_title("Mask")
        axes[2, i].axis("off")
        
        # Generated (posterior sample)
        axes[3, i].imshow(generated[i].permute(1, 2, 0))
        axes[3, i].set_title("Generated")
        axes[3, i].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved conditional sampling results to {save_path}")


def main():
    """Demonstrate Bayesian conditional sampling."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained model
    print("Loading trained model...")
    model_path = "models/fm_unet_extended_final.pth"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Using standard trained model instead...")
        # You can load any trained class-conditional model here
        model = UNetFlowMatching(channels=3, base_channels=128, num_classes=10).to(device)
        # For demo, we'll just use random weights
        print("WARNING: Using untrained model for demonstration")
    else:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = UNetFlowMatching(
            channels=checkpoint['model_config']['channels'],
            base_channels=checkpoint['model_config']['base_channels'],
            num_classes=checkpoint['model_config']['num_classes']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    
    model.eval()
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ])
    dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Get test images
    test_images, test_labels = next(iter(test_loader))
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)
    
    # Create results directory
    os.makedirs("results/cifar", exist_ok=True)
    
    # Test different observation patterns
    patterns = ["random", "half", "grid"]
    guidance_strengths = [0.5, 1.0, 2.0]
    
    for pattern in patterns:
        print(f"\nTesting pattern: {pattern}")
        
        try:
            # Create sparse observations
            observed, mask = create_sparse_observations(
                test_images, 
                observation_ratio=0.3 if pattern == "random" else 0.5,
                pattern=pattern
            )
            
            print(f"  Created observations: observed={observed.shape}, mask={mask.shape}")
            print(f"  Observation ratio: {mask.mean().item():.2%}")
            print(f"  Mask min/max: {mask.min().item():.2f}/{mask.max().item():.2f}")
            print(f"  Observed pixels are non-zero: {(observed.abs().sum() > 0).item()}")
            
            for guidance in guidance_strengths:
                print(f"  Guidance strength: {guidance}")
                
                # Generate using Bayesian conditional sampling
                with torch.no_grad():
                    generated = conditional_flow_matching_sample(
                        model,
                        observed,
                        mask,
                        test_labels,
                        device,
                        num_steps=30,
                        guidance_strength=guidance,
                        solver="heun"
                    )
                
                # Debug: Check if generated is different from original
                diff = (generated - test_images).abs().mean().item()
                print(f"  Difference between generated and original: {diff:.4f}")
                
                # Visualize
                save_path = f"results/cifar/conditional_{pattern}_guidance{guidance}.png"
                visualize_conditional_sampling(
                    test_images, observed, mask, generated, save_path
                )
                
        except Exception as e:
            print(f"  ❌ Error with pattern {pattern}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n✅ Conditional sampling completed!")
    print("Key advantages of this approach:")
    print("  - No retraining needed")
    print("  - Works with sparse observations")
    print("  - Bayesian framework (prior → posterior)")
    print("  - Flexible for any observation pattern")


if __name__ == "__main__":
    main()
