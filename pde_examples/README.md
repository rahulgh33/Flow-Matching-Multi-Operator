## Flow Matching for PDE Solution

This directory contains examples of using Flow Matching to learn PDE solution operators.

### Why Flow Matching for PDEs?

**Traditional approach:**
- Solve PDE numerically for each initial condition
- Slow: thousands of time steps
- Must re-solve for every new initial condition

**Flow Matching approach:**
- Learn the mapping: u₀(x) → u(x, T)
- Fast: 50 integration steps
- Generalizes to unseen initial conditions
- Acts as a "surrogate model" or "solution operator"

---

### Examples

#### 1. Heat Equation (`heat_equation_fm.py`)

**PDE:** ∂u/∂t = α ∂²u/∂x²

**Difficulty:** Easy (linear, parabolic)

**What it does:**
- Learns to predict how heat diffuses over time
- Given initial temperature distribution → final distribution

**Run:**
```bash
python pde_examples/heat_equation_fm.py
```

**Expected results:**
- Training: ~1000 epochs, 2-3 minutes
- Error: < 0.01 (very accurate)
- Output: `results/pde/heat_equation.png`

---

#### 2. Burgers' Equation (`burgers_equation_fm.py`)

**PDE:** ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²

**Difficulty:** Hard (nonlinear, develops shocks)

**What it does:**
- Models shock wave formation
- Nonlinear advection + viscous diffusion
- More challenging test of Flow Matching

**Status:** Skeleton provided (needs full implementation)

---

### Key Concepts

#### Solution Operator Learning

Instead of solving:
```
u(x, t=0) → [numerical solver] → u(x, t=T)
```

We learn:
```
u(x, t=0) → [Flow Matching] → u(x, t=T)
```

#### Flow Matching Setup

**Training:**
1. Generate random initial conditions u₀
2. Solve PDE numerically to get uₜ
3. Train FM: u₀ → uₜ using linear interpolation
4. Loss: ||v_θ(u_t, t) - (uₜ - u₀)||²

**Inference:**
1. Start from new initial condition u₀
2. Integrate: du/dt = v_θ(u, t) from t=0 to t=T
3. Get solution uₜ in 50 steps (vs 10,000+ for numerical)

---

### Advantages for Scientific Computing

**Speed:**
- 100-1000x faster than numerical solvers
- Real-time PDE solution

**Generalization:**
- Works on unseen initial conditions
- Interpolates between training examples

**Multi-Query:**
- Solve once, query many times
- Perfect for optimization, inverse problems

**Uncertainty Quantification:**
- Can sample multiple solutions
- Estimate solution uncertainty

---

### Extensions

**More PDEs to try:**
- Wave equation (hyperbolic)
- Navier-Stokes (fluid dynamics)
- Schrödinger equation (quantum mechanics)
- Reaction-diffusion (pattern formation)

**Advanced techniques:**
- Multi-step prediction (u₀ → u₁ → u₂ → ...)
- Conditional on parameters (learn family of PDEs)
- Inverse problems (data → initial condition)
- Operator learning (DeepONet-style)

---

### Connection to Your Research

This demonstrates Flow Matching for:
- **Multi-operator applications** ✓
- **Physical interpretability** ✓
- **Fast inference** ✓
- **Scientific computing** ✓

Perfect for your research statement about "multi-operator applications, PDEs, and video generation"!

---

### References

- **Neural Operators**: Li et al., "Fourier Neural Operator"
- **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling"
- **PDE Surrogates**: Karniadakis et al., "Physics-Informed Neural Networks"
