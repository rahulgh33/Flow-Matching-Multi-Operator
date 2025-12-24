"""
Test model architectures for shape consistency.
Run this before training to catch dimension mismatches early.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_fm_diffusion_2d():
    """Test PDE2DFlowMatching model shapes."""
    print("\n" + "="*60)
    print("Testing PDE2DFlowMatching Model")
    print("="*60)
    
    try:
        from fm_diffusion_2d import PDE2DFlowMatching
        
        # Test with default parameters
        model = PDE2DFlowMatching(channels=1, base_channels=64)
        model.eval()
        
        # Test inputs
        batch_sizes = [1, 4, 16]
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch_size={batch_size}:")
            
            u = torch.randn(batch_size, 1, 64, 64)
            t = torch.rand(batch_size, 1)
            
            print(f"  Input u shape: {tuple(u.shape)}")
            print(f"  Input t shape: {tuple(t.shape)}")
            
            # Forward pass
            with torch.no_grad():
                output = model(u, t)
            
            print(f"  Output shape: {tuple(output.shape)}")
            
            # Validate output shape
            assert output.shape == u.shape, \
                f"Output shape mismatch! Expected {u.shape}, got {output.shape}"
            
            print(f"  ✅ Batch size {batch_size} passed!")
        
        # Test parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")
        
        print("\n✅ All PDE2DFlowMatching tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ PDE2DFlowMatching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_time_embedding_dimensions():
    """Test that time embedding dimensions are consistent."""
    print("\n" + "="*60)
    print("Testing Time Embedding Dimensions")
    print("="*60)
    
    try:
        from fm_diffusion_2d import PDE2DFlowMatching
        
        base_channels_list = [32, 64, 128]
        
        for base_channels in base_channels_list:
            print(f"\nTesting base_channels={base_channels}:")
            
            model = PDE2DFlowMatching(channels=1, base_channels=base_channels)
            
            # Check internal dimensions
            expected_time_emb_input = base_channels * 2  # sin + cos
            expected_time_emb_output = base_channels * 8  # bottleneck dimension
            expected_bottleneck_channels = base_channels * 8
            
            # Get first linear layer input dimension
            first_linear = model.time_embedding[0]
            actual_time_emb_input = first_linear.in_features
            actual_time_emb_output = first_linear.out_features
            
            print(f"  Time embedding input dim: {actual_time_emb_input} (expected: {expected_time_emb_input})")
            print(f"  Time embedding output dim: {actual_time_emb_output} (expected: {expected_time_emb_output})")
            print(f"  Bottleneck channels: {expected_bottleneck_channels}")
            
            assert actual_time_emb_input == expected_time_emb_input, \
                f"Time embedding input mismatch: {actual_time_emb_input} != {expected_time_emb_input}"
            assert actual_time_emb_output == expected_time_emb_output, \
                f"Time embedding output mismatch: {actual_time_emb_output} != {expected_time_emb_output}"
            
            print(f"  ✅ base_channels={base_channels} dimensions correct!")
        
        print("\n✅ All time embedding dimension tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Time embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test that gradients flow properly through the model."""
    print("\n" + "="*60)
    print("Testing Gradient Flow")
    print("="*60)
    
    try:
        from fm_diffusion_2d import PDE2DFlowMatching
        
        model = PDE2DFlowMatching(channels=1, base_channels=64)
        model.train()
        
        # Test input
        u = torch.randn(4, 1, 64, 64, requires_grad=True)
        t = torch.rand(4, 1, requires_grad=True)
        
        # Forward pass
        output = model(u, t)
        
        # Compute loss and backward
        loss = output.mean()
        loss.backward()
        
        # Check gradients
        has_gradients = 0
        no_gradients = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients += 1
            else:
                no_gradients += 1
                print(f"  ⚠️  No gradient: {name}")
        
        print(f"\nGradient Statistics:")
        print(f"  Parameters with gradients: {has_gradients}")
        print(f"  Parameters without gradients: {no_gradients}")
        
        assert has_gradients > 0, "No gradients computed!"
        
        print("\n✅ Gradient flow test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" "*15 + "MODEL SHAPE VALIDATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Time Embedding Dimensions", test_time_embedding_dimensions),
        ("Model Forward Pass", test_fm_diffusion_2d),
        ("Gradient Flow", test_gradient_flow),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "="*70)
    print(" "*25 + "TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "="*70)
        print(" "*15 + "🎉 ALL TESTS PASSED! 🎉")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print(" "*15 + "⚠️  SOME TESTS FAILED ⚠️")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit(main())
