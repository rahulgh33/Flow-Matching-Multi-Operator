# Request for PDE Packages in torch_env

## Summary
Request to install PDE-related packages in the shared conda environment `/depot/lin491/apps/torch_env` for research on Flow Matching applied to Partial Differential Equations.

## Requested Packages

### Required:
- `fipy` - Professional PDE solving library (conda-forge)
- `tqdm` - Progress bars for long-running computations

### Optional (for performance):
- `cupy` - GPU acceleration for numerical computations

## Installation Commands

```bash
# Activate the shared environment
conda activate /depot/lin491/apps/torch_env

# Install required packages
conda install -c conda-forge fipy tqdm

# Optional: GPU acceleration
conda install -c conda-forge cupy
```

## Research Context
- Working on Flow Matching for PDE solution operators
- Collaboration with Amir on high-quality PDE data generation
- Part of multi-operator applications research
- Enables 100-1000x speedup over traditional numerical solvers

## Alternative
If shared environment modification is not possible, we can create a personal conda environment, but the shared environment would be more convenient for the research group.

## Contact
- Student: Rahul Ghosh (ghosh126)
- Account: lin491
- Research: Flow Matching for Multi-Operator Applications