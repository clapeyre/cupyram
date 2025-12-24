# CuPyRAM

GPU-accelerated Range-dependent Acoustic Model (RAM) using CuPy.

## Overview

CuPyRAM is a high-performance GPU implementation of the Range-dependent Acoustic Model (RAM), providing significant speedups for underwater acoustic propagation modeling through CUDA acceleration.

This project is a GPU port of [PyRAM](https://github.com/NeptuneProjects/PyRAM), which itself is a Python adaptation of the original RAM model created by Dr. Michael D. Collins at the US Naval Research Laboratory. RAM is available from the [Ocean Acoustics Library](https://oalib-acoustics.org/models-and-software/parabolic-equation).

## Features

- **GPU Acceleration**: Leverages NVIDIA CUDA through CuPy for massive performance improvements
- **Batch Processing**: Efficiently handles multiple acoustic scenarios simultaneously
- **Compatible API**: Maintains similar interface to PyRAM for easy migration
- **Validated**: Extensive test suite comparing results against reference implementations

## Requirements

- Python >= 3.8
- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x or compatible

## Installation

```bash
pip install cupyram
```

**Note**: This package requires a CUDA-capable GPU and the appropriate CUDA drivers. CuPy will be installed as a dependency, which provides the GPU acceleration.

### Optional Dependencies

For running tests (includes PyRAM for validation):

```bash
pip install cupyram[test]
```

**Note**: The test suite compares CuPyRAM results against the original PyRAM implementation to ensure accuracy.

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

CuPyRAM provides significant speedups compared to CPU implementations:
- Single scenarios: 10-50x faster than NumPy/Numba implementations
- Batch processing: 100-500x faster for large batches (depending on GPU)

## Testing

Tests require a CUDA-capable GPU and include validation against the original PyRAM implementation:

```bash
# Install with test dependencies (includes PyRAM for validation)
pip install cupyram[test]

# Run tests
pytest tests/
```

On first run, tests will automatically generate baseline data from PyRAM for comparison. This ensures CuPyRAM results match the ground truth CPU implementation.

**Note**: GitHub Actions and standard CI services do not provide GPU support. If you encounter issues, please include your GPU model and CUDA version when reporting.

## Differences from PyRAM

CuPyRAM maintains API compatibility where possible, but includes several optimizations:
- All computations performed on GPU
- Batch processing support for multiple scenarios
- Memory-efficient implementations for large-scale problems
- Optimized matrix operations using CuPy

## Citation

If you use CuPyRAM in your research, please cite both this implementation and the original PyRAM:

- PyRAM: https://github.com/NeptuneProjects/PyRAM
- Original RAM: Collins, M. D. (1993). A split-step Padé solution for the parabolic equation method. Journal of the Acoustical Society of America, 93(4), 1736-1742.

## License

BSD 3-Clause License. See LICENSE file for details.

This project is based on PyRAM by Marcus Donnelly, which adapted the original RAM code by Dr. Michael D. Collins.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Dr. Michael D. Collins for the original RAM implementation
- Marcus Donnelly for the PyRAM Python adaptation
- The CuPy development team for the excellent GPU array library

