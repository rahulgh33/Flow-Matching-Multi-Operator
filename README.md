# Flow Matching Multi-Operator

PyTorch implementation of Flow Matching for unconditional and conditional image generation.

## Quick Start

```bash
# Unconditional generation
python unconditional/fm_mnist.py
python unconditional/fm_cfortan.py

# Conditional generation  
python conditional/fm_mnist_conditional.py
python conditional/fm_cifar_conditional.py
```

## Conditional Generation

Generate specific digits or classes:

```python
from conditional.fm_mnist_conditional import generate_specific_digits
generate_specific_digits(model, device, [1, 3, 7, 9])

from conditional.fm_cifar_conditional import generate_specific_classes  
generate_specific_classes(model, device, [0, 3, 5])  # airplane, cat, dog
```

## Requirements

```bash
pip install torch torchvision matplotlib
```