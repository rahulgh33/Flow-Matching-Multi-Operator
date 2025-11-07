"""
Test script to validate tensor shapes in conditional sampling.
Run this before submitting jobs to catch dimension errors early.
"""

import torch
import sys
sys.path.append('conditional')

from fm_cifar_conditional_sampling import (
    create_sparse_observations,
    compute_observation_likelihood_gradient,
    validate_shapes
)

def test_shapes():
    """Test all tensor shapes in the pipeline."""
    print("Testing tensor shapes for conditional sampling...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create dummy data
    B, C, H, W = 4, 3, 32, 32
    images = torch.randn(B, C, H, W, device=device)
    
    print(f"\n1. Input images: {images.shape}")
    assert images.shape == (B, C, H, W), "Input shape mismatch"
    
    # Test observation creation
    patterns = ["random", "grid", "half", "edges"]
    for pattern in patterns:
        print(f"\n2. Testing pattern: {pattern}")
        observed, mask = create_sparse_observations(images, pattern=pattern)
        
        print(f"   Observed: {observed.shape}")
        print(f"   Mask: {mask.shape}")
        print(f"   Observation ratio: {mask.mean().item():.2%}")
        
        assert observed.shape == (B, C, H, W), f"Observed shape mismatch for {pattern}"
        assert mask.shape == (B, 1, H, W), f"Mask shape mismatch for {pattern}"
        assert observed.device == images.device, f"Device mismatch for {pattern}"
        assert mask.device == images.device, f"Device mismatch for {pattern}"
    
    # Test likelihood gradient
    print(f"\n3. Testing likelihood gradient computation")
    x_t = torch.randn(B, C, H, W, device=device)
    t = torch.rand(B, 1, device=device)
    
    grad = compute_observation_likelihood_gradient(x_t, observed, mask, t)
    
    print(f"   Gradient: {grad.shape}")
    assert grad.shape == (B, C, H, W), "Gradient shape mismatch"
    assert grad.device == device, "Gradient device mismatch"
    
    # Test shape validation
    print(f"\n4. Testing shape validation")
    try:
        validate_shapes(x_t, observed, mask, t, context="Test")
        print("   ✅ Shape validation passed")
    except AssertionError as e:
        print(f"   ❌ Shape validation failed: {e}")
        return False
    
    # Test broadcasting
    print(f"\n5. Testing broadcasting operations")
    
    # Mask broadcast
    result = images * mask
    assert result.shape == (B, C, H, W), "Mask broadcast failed"
    print(f"   images * mask: {result.shape} ✅")
    
    # Time broadcast
    dt = t.view(-1, 1, 1, 1)
    result = x_t * dt
    assert result.shape == (B, C, H, W), "Time broadcast failed"
    print(f"   x_t * dt: {result.shape} ✅")
    
    # Combined
    result = x_t * (1 - mask) + observed * mask
    assert result.shape == (B, C, H, W), "Combined broadcast failed"
    print(f"   x_t * (1-mask) + observed * mask: {result.shape} ✅")
    
    print(f"\n✅ All shape tests passed!")
    return True

if __name__ == "__main__":
    success = test_shapes()
    sys.exit(0 if success else 1)
