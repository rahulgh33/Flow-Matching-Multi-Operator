# Flow Matching Multi-Operator

A comprehensive PyTorch implementation of Flow Matching for conditional image generation with multi-operator extensions.

## Overview

Flow Matching is a generative modeling technique that learns to transform noise into data by predicting velocity fields. This repository contains clean, well-documented implementations for both unconditional and **conditional** generation across multiple datasets.

## Features

- **Unconditional Generation**: 
  - MNIST digits (28x28 grayscale)
  - CIFAR-10 images (32x32 color)
- **🆕 Conditional Generation**: 
  - Generate specific MNIST digits (0-9)
  - Generate specific CIFAR-10 classes (airplane, car, etc.)
- **Shared Base Classes**: Modular architecture with reusable components
- **Interactive Demo**: Easy-to-use interface for conditional generation
- **Clean Code**: Type hints, comprehensive documentation, and best practices

## Requirements

```
torch
torchvision
matplotlib
```

## Usage

### Unconditional Generation
```bash
# MNIST digits (random)
python fm_mnist.py

# CIFAR-10 images (random)
python fm_cfortan.py
```

### 🆕 Conditional Generation
```bash
# MNIST - generate specific digits
python fm_mnist_conditional.py

# CIFAR-10 - generate specific classes
python fm_cifar_conditional.py

# Interactive demo
python demo_conditional.py
```

### Generate Specific Content
```python
# Generate specific MNIST digits
from fm_mnist_conditional import ConditionalFlowMatchingNet, generate_specific_digits

model = ConditionalFlowMatchingNet()
generate_specific_digits(model, device, [1, 3, 7, 9])  # Generate digits 1, 3, 7, 9

# Generate specific CIFAR-10 classes  
from fm_cifar_conditional import ConditionalFlowMatchingNetCIFAR, generate_specific_classes

model = ConditionalFlowMatchingNetCIFAR()
generate_specific_classes(model, device, [0, 3, 5])  # airplane, cat, dog
```

## Architecture

Both models use an encoder-decoder architecture with time conditioning:
- **Encoder**: Downsamples input images to a latent representation
- **Time Embedding**: Embeds the current time step
- **Fusion**: Combines spatial and temporal features
- **Decoder**: Upsamples back to image space

## Results

Generated samples are saved as PNG files:
- `fm_samples.png` - Unconditional MNIST results
- `fm_cfortan_samples.png` - Unconditional CIFAR-10 results
- `conditional_mnist_samples.png` - Conditional MNIST results
- `conditional_cifar_samples.png` - Conditional CIFAR-10 results
- `specific_digits.png` - User-specified MNIST digits
- `specific_cifar_classes.png` - User-specified CIFAR-10 classes

## Implementation Details

The Flow Matching algorithm:
1. Samples random times t ∈ [0,1]
2. Interpolates between noise x₀ and data x₁: xₜ = (1-t)x₀ + tx₁
3. Predicts velocity field v = x₁ - x₀
4. Uses Euler integration for sampling

## License

MIT License