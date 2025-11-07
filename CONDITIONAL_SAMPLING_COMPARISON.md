# Conditional Sampling: Two Approaches

## Overview

This document compares two approaches to conditional image generation with Flow Matching:

1. **Training-Time Conditioning** (`fm_cifar_inpainting.py`) - Original approach
2. **Inference-Time Conditioning** (`fm_cifar_conditional_sampling.py`) - Mentor's approach

---

## Approach 1: Training-Time Conditioning (Original)

### Mathematical Framework:
```
Train: v_θ(x_t, observed, mask, t) to match v* = x_1 - x_0
where x_0 = observed + (1-mask)*noise
```

### Characteristics:
- **Training**: Specialized model for inpainting
- **Input**: Current state + observed pixels + mask
- **Mask**: Continuous geometry (rectangular holes)
- **Flexibility**: Limited - need to retrain for different tasks

### Pros:
✅ Direct learning of inpainting task
✅ Can learn complex inpainting strategies
✅ Potentially better quality for specific mask patterns

### Cons:
❌ Requires retraining for each conditioning task
❌ Limited to continuous geometry masks
❌ Not generalizable to other conditional tasks
❌ Larger model (needs to encode mask handling)

---

## Approach 2: Inference-Time Conditioning (Bayesian)

### Mathematical Framework:
```
Bayes' Rule:
∇_x log p(x|c) = ∇_x log p(x|τ) + ∇_x log p(c|x)
                  ↑ prior          ↑ likelihood

Flow Matching:
dx/dt = v_θ(x_t, t, class) + λ * ∇_x log p(observations | x_t)
        ↑ trained once       ↑ computed at inference
```

### Characteristics:
- **Training**: Standard unconditional/class-conditional model
- **Conditioning**: Added at inference time via likelihood gradient
- **Mask**: Any pattern (sparse, dense, structured)
- **Flexibility**: Same model for all conditional tasks

### Pros:
✅ No retraining needed
✅ Works with sparse observations
✅ Bayesian framework (principled)
✅ Flexible for any observation pattern
✅ Same model for inpainting, super-resolution, etc.
✅ Tunable guidance strength

### Cons:
❌ May require more inference steps
❌ Guidance strength needs tuning
❌ Approximate (depends on likelihood model)

---

## Key Differences

| Aspect | Training-Time | Inference-Time |
|--------|---------------|----------------|
| **Model Training** | Task-specific | Once, general |
| **Observation Type** | Continuous geometry | Sparse or dense |
| **Flexibility** | Low | High |
| **Retraining** | Required | Not required |
| **Bayesian** | No | Yes |
| **Guidance** | Baked in | Tunable |
| **Use Cases** | Single task | Multiple tasks |

---

## Mentor's Feedback Explained

### "Your mask is continuous geometry not sparse"
- **Original**: Rectangular holes (continuous regions)
- **Bayesian**: Can handle random pixels (sparse observations)

### "You would have to retrain the model every time"
- **Original**: New mask pattern → retrain
- **Bayesian**: New observations → just run inference

### "Choose trajectory consistent with observation"
- **Original**: Model learns fixed inpainting strategy
- **Bayesian**: Flow adapts to observations via posterior update

### "Using Bayesian rule, use prior to update posterior"
- **Original**: No explicit prior/posterior separation
- **Bayesian**: Clear separation:
  - Prior: p(x) from trained model
  - Likelihood: p(observations|x) computed
  - Posterior: p(x|observations) via Bayes' rule

---

## When to Use Each Approach

### Use Training-Time Conditioning When:
- You have a specific, well-defined conditioning task
- You need maximum quality for that specific task
- You have resources to train specialized models
- Observation pattern is fixed (e.g., always rectangular holes)

### Use Inference-Time Conditioning When:
- You want flexibility across multiple tasks
- Observations are sparse or varied
- You can't retrain for each new task
- You want a principled Bayesian framework
- You need to experiment with different guidance strengths

---

## Implementation Details

### Training-Time (Original):
```python
# Training
x_0 = observed + (1-mask)*noise
x_1 = complete_image
x_t = (1-t)*x_0 + t*x_1
v* = x_1 - x_0
loss = ||v_θ(x_t, observed, mask, t) - v*||²

# Inference
v = model(x_t, observed, mask, t)
x = integrate(v, num_steps)
```

### Inference-Time (Bayesian):
```python
# Training (standard)
x_0 = noise
x_1 = complete_image
x_t = (1-t)*x_0 + t*x_1
v* = x_1 - x_0
loss = ||v_θ(x_t, t, class) - v*||²

# Inference (with guidance)
v_prior = model(x_t, t, class)
v_likelihood = ∇_x log p(observations | x_t)
v_posterior = v_prior + λ * v_likelihood
x = integrate(v_posterior, num_steps)
```

---

## Conclusion

The **Bayesian inference-time conditioning** approach is more elegant and flexible:

1. **Train once, use everywhere** - no retraining needed
2. **Principled framework** - clear probabilistic interpretation
3. **Flexible observations** - sparse, dense, any pattern
4. **Tunable** - adjust guidance strength at inference

This is the approach used in modern conditional diffusion models (classifier guidance, classifier-free guidance) and represents the state-of-the-art in conditional generation.

Your original approach is valid but specialized. The Bayesian approach is more general and demonstrates deeper understanding of conditional probability flows.

---

## References

- **Conditional Flow Matching**: Song et al., "Score-Based Generative Modeling through SDEs"
- **Classifier Guidance**: Dhariwal & Nichol, "Diffusion Models Beat GANs"
- **Bayesian Inference**: Your mentor's diagram showing ∇log p(x|c) decomposition
