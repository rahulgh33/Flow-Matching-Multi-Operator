# Flow Matching for Image Generation

A PyTorch implementation of Flow Matching for generating MNIST digits and CIFAR-10 images.

## Overview

Flow Matching is a generative modeling technique that learns to transform noise into data by predicting velocity fields. This repository contains clean, well-documented implementations for both grayscale (MNIST) and color (CIFAR-10) image generation.

## Features

- **MNIST Generation**: Grayscale digit generation (28x28)
- **CIFAR-10 Generation**: Color image generation (32x32)
- **Shared Base Classes**: Modular architecture with reusable components
- **Clean Code**: Type hints, comprehensive documentation, and best practices

## Requirements

```
torch
torchvision
matplotlib
```

## Usage

### MNIST Generation
```bash
python fm_mnist.py
```

### CIFAR-10 Generation
```bash
python fm_cfortan.py
```

## Architecture

Both models use an encoder-decoder architecture with time conditioning:
- **Encoder**: Downsamples input images to a latent representation
- **Time Embedding**: Embeds the current time step
- **Fusion**: Combines spatial and temporal features
- **Decoder**: Upsamples back to image space

## Results

Generated samples are saved as PNG files:
- `fm_samples.png` - MNIST results
- `fm_cfortan_samples.png` - CIFAR-10 results

## Implementation Details

The Flow Matching algorithm:
1. Samples random times t ∈ [0,1]
2. Interpolates between noise x₀ and data x₁: xₜ = (1-t)x₀ + tx₁
3. Predicts velocity field v = x₁ - x₀
4. Uses Euler integration for sampling

## License

MIT License