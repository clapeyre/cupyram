"""
Pytest configuration and fixtures for functional tests
"""
import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import gpu_ram_numba
sys.path.insert(0, str(Path(__file__).parent.parent))

def pytest_configure(config):
    """Configure pytest with custom markers and setup output directory"""
    config.addinivalue_line("markers", "reference: Original RAM v1.5 reference test")
    config.addinivalue_line("markers", "pekeris: Pekeris waveguide test")
    config.addinivalue_line("markers", "munk: Munk profile test")
    config.addinivalue_line("markers", "wedge: ASA wedge test")
    config.addinivalue_line("markers", "raster: Raster/coordinate integration test")
    config.addinivalue_line("markers", "pade: Pade coefficients test")
    config.addinivalue_line("markers", "phase1: Pekeris waveguide (analytic benchmark)")
    config.addinivalue_line("markers", "phase2: Munk profile (range-independent)")
    config.addinivalue_line("markers", "phase3: ASA wedge (range-dependent bathymetry)")
    config.addinivalue_line("markers", "phase4: Raster/coordinate integration")
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    
    # Create test outputs directory
    output_dir = Path(__file__).parent.parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Set as environment variable for tests to use
    os.environ['TEST_OUTPUT_DIR'] = str(output_dir)


@pytest.fixture(scope="session")
def output_dir():
    """Get test outputs directory"""
    output_path = Path(os.environ.get('TEST_OUTPUT_DIR', 'test_outputs'))
    output_path.mkdir(exist_ok=True)
    return output_path


@pytest.fixture(scope="session")
def test_params():
    """Common test parameters"""
    return {
        'freq': 50.0,
        'dr': 50.0,
        'np_pade': 8
    }


@pytest.fixture(scope="session", autouse=True)
def ensure_baseline_data():
    """
    Ensure baseline_data.npz exists before running tests.
    Generates it using PyRAM if not found.
    """
    baseline_file = Path(__file__).parent / 'baseline_data.npz'
    
    if not baseline_file.exists():
        print("\n⚠️  Baseline data not found. Generating from PyRAM...")
        try:
            from tests.generate_baseline import generate_baseline
            generate_baseline()
            print("✅ Baseline data generated successfully.\n")
        except ImportError as e:
            pytest.skip(f"Cannot generate baseline data: {e}. Install test dependencies: pip install cupyram[test]")
    
    return baseline_file