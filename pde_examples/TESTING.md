# Model Shape Validation

## Preventing Dimension Mismatch Errors

This directory includes automated tests to catch dimension mismatch errors before training.

### Quick Start

**Before submitting any training job, run:**

```bash
./validate_models.sh
```

Or directly:

```bash
python pde_examples/test_model_shapes.py
```

### What Gets Tested

1. **Time Embedding Dimensions**: Validates that time embeddings match expected dimensions
2. **Forward Pass**: Tests model with various batch sizes to catch shape errors
3. **Gradient Flow**: Ensures gradients propagate through all layers

### Common Issues Caught

✅ **Matrix multiplication shape mismatches**
- Catches `mat1 and mat2 shapes cannot be multiplied` errors
- Validates layer input/output dimensions

✅ **Tensor broadcasting errors**  
- Catches `size of tensor a (X) must match size of tensor b (Y)` errors
- Validates skip connection dimensions

✅ **Attribute errors**
- Catches `'binOp' object has no attribute 'reshape'` errors
- Validates data type conversions

### Integration with Workflow

**Local Development:**
```bash
# After modifying model architecture
python pde_examples/test_model_shapes.py
```

**Before SLURM Submission:**
```bash
# Add to your .sbatch file
python pde_examples/test_model_shapes.py || exit 1
```

**Git Pre-commit Hook:**
```bash
# Add to .git/hooks/pre-commit
#!/bin/bash
python pde_examples/test_model_shapes.py
```

### Adding New Tests

To test a new model, add a test function to `test_model_shapes.py`:

```python
def test_your_model():
    """Test your model architecture."""
    from your_module import YourModel
    
    model = YourModel()
    
    # Test with dummy input
    x = torch.randn(4, input_channels, height, width)
    output = model(x)
    
    # Add assertions
    assert output.shape == expected_shape
    
    print("✅ Your model test passed!")
    return True
```

### Test Output Example

```
======================================================================
                   MODEL SHAPE VALIDATION TEST SUITE
======================================================================

Testing Time Embedding Dimensions
  ✅ base_channels=64 dimensions correct!
  
Testing PDE2DFlowMatching Model  
  ✅ Batch size 16 passed!
  Model size: 52.09 MB
  
Testing Gradient Flow
  ✅ Gradient flow test passed!

======================================================================
                        🎉 ALL TESTS PASSED! 🎉
======================================================================
```

### Benefits

- **Catch errors early**: Find dimension issues locally before wasting GPU time
- **Fast validation**: Tests complete in seconds
- **Multiple batch sizes**: Ensures model works with various batch sizes
- **Gradient checks**: Verifies training will work properly
- **Clear error messages**: Detailed output for debugging

### Architecture Validation

The tests include assertions in the forward pass:

```python
assert t_sinusoidal.size(1) == self.base_channels * 2, \
    f"Time sinusoidal embedding size mismatch"

assert x3.size(1) == t_emb.size(1), \
    f"Channel mismatch at bottleneck"
```

These catch errors immediately with helpful messages.
