# Flow Matching Results

This folder contains generated samples from our optimized Flow Matching implementation.

## Key Results

### Speed Comparison
- `15steps_target.png` - Our main result: 15-step generation
- `20steps_balanced.png` - Balanced quality/speed: 20-step generation  
- `12steps_ultrafast.png` - Ultra-fast: 12-step generation
- `100steps_baseline.png` - Baseline comparison: 100-step generation

**Performance**: 15 steps vs 100 steps = **6.7x speedup**

### Quality Showcase
- `final_optimized.png` - Best quality samples with optimized settings
- `specific_cifar_classes.png` - Targeted class generation (airplane, cat, dog, ship)

### Additional Results
- `conditional_cifar_samples.png` - CIFAR-10 conditional generation
- `conditional_mnist_samples.png` - MNIST conditional generation
- `specific_digits.png` - MNIST digit generation
- `fm_samples.png` - General Flow Matching samples

## Key Innovations

1. **Straightened path training** (α(t) = t²) for smoother ODE integration
2. **Beta(0.5,0.5) time sampling** for endpoint emphasis
3. **EMA weights** for improved sample quality
4. **Cosine time grid + Heun solver** for efficient few-step generation

## Architecture

- ResNet encoder-decoder (no skip connections for clean Flow Matching story)
- Class-conditional generation via FiLM conditioning
- ~45M parameters, optimized for few-step generation