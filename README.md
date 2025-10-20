# Flow Matching Multi-Operator

PyTorch implementation of Flow Matching for unconditional and conditional image generation on MNIST and CIFAR-10.

## Overview

Flow Matching learns continuous normalizing flows by predicting velocity fields that transform noise into data. This implementation includes:

- **Unconditional generation**: Random sampling from learned data distributions
- **Conditional generation**: Generate specific classes/digits on command
- **Modular architecture**: Shared base classes for easy extension

## Quick Start

```bash
# Install dependencies
pip install torch torchvision matplotlib

# Unconditional generation
python unconditional/fm_mnist.py      # Generates random MNIST digits
python unconditional/fm_cfortan.py    # Generates random CIFAR-10 images

# Conditional generation  
python conditional/fm_mnist_conditional.py    # Train conditional MNIST model
python conditional/fm_cifar_conditional.py    # Train conditional CIFAR-10 model
```

## Conditional Generation

After training, generate specific content:

```python
# Generate specific MNIST digits
from conditional.fm_mnist_conditional import ConditionalFlowMatchingNet, generate_specific_digits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionalFlowMatchingNet().to(device)
# ... load trained weights ...
generate_specific_digits(model, device, [1, 3, 7, 9])

# Generate specific CIFAR-10 classes
from conditional.fm_cifar_conditional import ConditionalFlowMatchingNetCIFAR, generate_specific_classes

model = ConditionalFlowMatchingNetCIFAR().to(device)  
# ... load trained weights ...
generate_specific_classes(model, device, [0, 3, 5])  # airplane, cat, dog
```

## Architecture

Both models use encoder-decoder architectures with time and class conditioning:

1. **Encoder**: Downsamples images to latent representations
2. **Time Embedding**: Sinusoidal embeddings for flow time t ∈ [0,1]  
3. **Class Embedding**: Learned embeddings for conditional generation
4. **Fusion**: Concatenates spatial, temporal, and class features
5. **Decoder**: Upsamples to predict velocity fields

## Algorithm

Flow Matching training:
1. Sample noise x₀ ~ N(0,I) and data x₁ from dataset
2. Sample time t ~ Uniform(0,1)
3. Interpolate: x_t = (1-t)x₀ + tx₁
4. Predict velocity: v_θ(x_t, t, c) ≈ x₁ - x₀
5. Minimize: ||v_θ(x_t, t, c) - (x₁ - x₀)||²

Generation uses Euler integration: x_{t+dt} = x_t + v_θ(x_t, t, c) * dt

## Results

Training generates sample images:
- `fm_samples.png` - Unconditional MNIST
- `fm_cfortan_samples.png` - Unconditional CIFAR-10  
- `conditional_mnist_samples.png` - All MNIST digits (0-9)
- `specific_digits.png` - User-requested digits
- `conditional_cifar_samples.png` - All CIFAR-10 classes
- `specific_cifar_classes.png` - User-requested classes