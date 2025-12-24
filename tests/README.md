# CuPyRAM Test Suite

This directory contains the test suite for CuPyRAM, which validates the GPU implementation against multiple benchmarks and the original PyRAM implementation.

## Requirements

- CUDA-capable GPU
- Test dependencies: `pip install cupyram[test]`

The test dependencies include PyRAM, which serves as the ground truth reference implementation.

## Running Tests

```bash
# Install with test dependencies
pip install cupyram[test]

# Run all tests
pytest tests/

# Run specific test categories
pytest tests/ -m pekeris     # Pekeris waveguide tests
pytest tests/ -m munk        # Munk profile tests
pytest tests/ -m wedge       # ASA wedge tests
```

## Baseline Data Generation

The test suite includes `generate_baseline.py`, which creates reference data from PyRAM:
- `baseline_data.npz` is **generated automatically** on first test run
- It is **not included in git** (it's in `.gitignore`)
- This ensures tests always validate against the current PyRAM implementation

You can manually regenerate it:

```bash
python -m tests.generate_baseline
```

## Test Files

- `conftest.py` - Pytest configuration and fixtures (auto-generates baseline data)
- `generate_baseline.py` - Creates PyRAM baseline for validation
- `verify_gpu.py` - Validates CuPyRAM against PyRAM baseline
- `test_pekeris.py` - Pekeris waveguide (analytic solution)
- `test_munk.py` - Munk sound speed profile
- `test_wedge.py` - ASA wedge benchmark
- `test_batching.py` - Batch processing validation
- `test_pade_coefficients.py` - Padé approximation tests
- `test_reference.py` - RAM v1.5 reference comparison
- `pekeris_analytic.py` - Analytic Pekeris solution
- `tl_ref.line` - Reference transmission loss data

## Notes

- Tests require GPU with CUDA support
- First test run will be slower (generates baseline data)
- Baseline data is cached for subsequent runs
- `baseline_data.npz` should not be committed to version control

