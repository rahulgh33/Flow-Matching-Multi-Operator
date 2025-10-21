# Flow Matching Results

Generated samples from optimized Flow Matching implementation.

## CIFAR-10 Results (`cifar/`)

### Speed Comparison
- `15steps.png` - Main result: 15-step generation
- `20steps.png` - 20-step generation  
- `12steps.png` - 12-step generation
- `100steps.png` - Baseline: 100-step generation

**Performance**: 15 steps vs 100 steps = 6.7x speedup

### Additional
- `conditional_samples.png` - All CIFAR-10 classes

## MNIST Results (`mnist/`)

- `conditional_samples.png` - All MNIST digits (0-9)
- `specific_digits.png` - Targeted digit generation

## Key Optimizations

1. Straightened path training (α(t) = t²)
2. Beta(0.5,0.5) time sampling for endpoint emphasis
3. EMA weights for improved quality
4. Cosine time grid + Heun solver for few-step generation