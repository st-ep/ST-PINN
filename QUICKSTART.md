# ST-PINN PyTorch - Quick Start Guide

## Installation

```bash
# Navigate to project directory
cd /geoelements/Stepan/ut_projects/ST-PINN

# Activate virtual environment
source venv_pytorch/bin/activate

# Verify installation
python test_imports.py
```

## Running Training

### 1. Standard PINN (Burgers Equation)
```bash
python source/Burgers1D_PINN.py
```

### 2. Self-Training PINN (Better Accuracy)
```bash
python source/Burgers1D_STPINN.py
```

### 3. Other PDEs

**Diffusion-Reaction:**
```bash
python source/DiffReact1D_STPINN.py
```

**Diffusion-Sorption:**
```bash
python source/DiffSorb1D_STPINN.py
```

## Quick Tests

```bash
# Test imports and basic functionality (30 seconds)
python test_imports.py

# Test standard PINN training (1 minute)
python test_training.py

# Test self-training mechanism (1 minute)
python test_stpinn.py
```

## Outputs

Training generates:
- **Logs:** `output/log/[experiment]-[timestamp].log`
- **Predictions:** `output/prediction/[experiment]-[timestamp].npy`

## Key Parameters

### Standard PINN
```python
layers = [2, 32, 32, 32, 32, 1]  # Network architecture
adam_it = 20000                   # Training iterations
batch_size = 20000                # Batch size
```

### Self-Training PINN (Additional)
```python
update_freq = 200    # Update pseudo-labels every N iterations
max_rate = 0.06      # Max 6% of points as pseudo-labels
stab_coeff = 9       # Require 10 selections for stability
```

## System Requirements

- **Python:** 3.8+
- **PyTorch:** 2.0+
- **NumPy:** 1.21+
- **GPU:** Optional (CUDA-enabled recommended)
- **Memory:** 8GB RAM minimum

## Performance

On NVIDIA RTX 5090:
- ~75 iterations/second (test configuration)
- ~10 iterations/second (full configuration)
- ~2-3 hours for full training (20,000 iterations)

## Troubleshooting

### Import Errors
```bash
source venv_pytorch/bin/activate
pip install torch numpy
```

### CUDA Not Available
PyTorch will automatically use CPU. Training will be slower but still works.

### Out of Memory
Reduce `batch_size` or use smaller network:
```python
layers = [2, 16, 16, 1]  # Smaller network
batch_size = 5000        # Smaller batch
```

## Documentation

- **Full Testing Report:** [TESTING_REPORT.md](TESTING_REPORT.md)
- **Original Paper:** README.md
- **Code Documentation:** Inline comments in source files

## Contact & Issues

For questions about the PyTorch migration, create an issue on GitHub or contact the development team.

---

**Ready to train!** 🚀