# CuPyRAM

GPU-accelerated Range-dependent Acoustic Model (RAM)

## Overview

CuPyRAM is a high-performance GPU implementation of the Range-dependent Acoustic Model (RAM).

This project is a GPU port of
[PyRAM](https://github.com/marcuskd/pyram), which itself is a Python
adaptation of the original RAM model created by Dr. Michael D. Collins at the
US Naval Research Laboratory. RAM is available from the [Ocean Acoustics
Library](https://oalib-acoustics.org/models-and-software/parabolic-equation).

## Features

- **GPU Acceleration**: Leverages Numba CUDA and CuPy for GPU acceleration
- **Compatible API**: Maintains similar interface to PyRAM for easy migration
- **Validated**: Extensive test suite comparing results against reference implementations

## Requirements

- Python >= 3.8
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.x, 12.x, or 13.x
- CuPy (matching your CUDA version)

## Installation

CuPyRAM requires CuPy, which must match your CUDA version. Install based on your CUDA toolkit version:

```bash
# For CUDA 11.x
pip install cupyram[cuda11]

# For CUDA 12.x
pip install cupyram[cuda12]

# For CUDA 13.x
pip install cupyram[cuda13]

# Or install CuPy separately first, then cupyram
pip install cupy-cuda12x  # or cupy-cuda11x, cupy-cuda13x
pip install cupyram
```

**Check your CUDA version**: Run `nvcc --version` or `nvidia-smi` to determine your CUDA version.

### Optional Dependencies

For running tests (includes PyRAM for validation):

```bash
pip install cupyram[test]
```

**Note**: The test suite compares CuPyRAM results against the original PyRAM
implementation to ensure accuracy.

## Quick Start

```python
from cupyram import CuPyRAM

# Initialize the model
model = CuPyRAM(
    freq=100.0,        # Frequency in Hz
    zs=10.0,           # Source depth in meters
    zr=50.0,           # Receiver depth in meters
    rmax=10000.0,      # Maximum range in meters
    dr=10.0,           # Range step in meters
    # ... other parameters
)

# Run the model
tl = model.run()  # Returns transmission loss array
```

## Performance

Performance testing is ongoing. Initial tests suggest that one data-center
class GPU runs about 30x faster than 1 CPU core.

CuPyRAM is designed for large batches. Performance depends on both available
VRAM and kernel residency: submitting many rays keeps work queued, but register
pressure can still limit the number of warps executing concurrently on each
GPU SM.

The fused-kernel launch configuration is currently tuned and validated on an
NVIDIA A100 SXM4 80 GB. Changing from 256 to 192 threads per block improved a
165,000-ray propagation benchmark from 485.8k to 600.4k ray-steps/s without
increasing VRAM use. Other GPU architectures should retain 192 as a strong
starting point but verify it when performance is critical.

See [GPU performance notes](docs/performance.md) for the benchmark methodology,
hardware-counter results, and portability guidance for Ada and other GPUs.

## Testing

Tests require a CUDA-capable GPU and include validation against the original PyRAM implementation:

```bash
# Install with test dependencies (includes PyRAM for validation)
pip install cupyram[test]

# Run tests
pytest tests/
```

On first run, tests will automatically generate baseline data from PyRAM for
comparison. This ensures CuPyRAM results match the ground truth CPU
implementation.

**Note**: GitHub Actions and standard CI services do not provide GPU support.
If you encounter issues, please include your GPU model and CUDA version when
reporting.

## Differences from PyRAM

CuPyRAM maintains API compatibility where possible, but includes several optimizations:
- Most heavy computations performed on GPU
- Some CPU computations parallelized (Padé coefficients)
- Memory-efficient implementations to maximize batch size

## Citation

If you use CuPyRAM in your research, please cite both this implementation and the original PyRAM:

- PyRAM: https://github.com/marcuskd/pyram
- Original RAM: Collins, M. D. (1993). A split-step Padé solution for the parabolic equation method. Journal of the Acoustical Society of America, 93(4), 1736-1742.

## License

BSD 3-Clause License. See LICENSE file for details.

This project is based on PyRAM by Marcus Donnelly, which adapted the original RAM code by Dr. Michael D. Collins.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Dr. Michael D. Collins for the original RAM implementation
- Marcus Donnelly for the PyRAM Python adaptation
- The Numba CUDA and CuPy development teams for the excellent GPU array library
