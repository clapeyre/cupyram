"""
Test suite for multi-frequency Super-Batch architecture.

Validates that:
1. Single frequency mode matches existing behavior (backward compatibility)
2. Multi-frequency results match multiple single-frequency runs
3. Output shapes are correct for various (N_env, N_freq) combinations
"""

import numpy as np
import pytest

# Try to import CuPyRAM - skip tests if GPU not available
try:
    from cupyram import CuPyRAM
    import cupy
    GPU_AVAILABLE = True
except (ImportError, RuntimeError):
    GPU_AVAILABLE = False
    pytest.skip("GPU not available", allow_module_level=True)


def create_simple_environment():
    """Create a simple test environment (Pekeris waveguide)."""
    # Water column
    z_ss = np.array([0.0, 100.0])
    rp_ss = np.array([0.0])
    cw = np.array([[1500.0], [1500.0]])
    
    # Seabed
    z_sb = np.array([0.0, 10.0])
    rp_sb = np.array([0.0])
    cb = np.array([[1600.0], [1600.0]])
    rhob = np.array([[1.5], [1.5]])
    attn = np.array([[0.5], [0.5]])
    
    # Bathymetry
    rbzb = np.array([[0.0, 100.0], [10000.0, 100.0]])
    
    return z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb


def test_single_frequency_backward_compatibility():
    """Test that single frequency mode works and produces expected output shape."""
    z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb = create_simple_environment()
    
    # Single frequency (scalar)
    model = CuPyRAM(
        freq=100.0,
        zs=10.0,
        zr=50.0,
        z_ss=z_ss, rp_ss=rp_ss, cw=cw,
        z_sb=z_sb, rp_sb=rp_sb, cb=cb, rhob=rhob, attn=attn,
        rbzb=rbzb,
        rmax=1000.0,
        dr=10.0,
        dz=0.5
    )
    
    results = model.run()
    
    # Check output shapes (single env, single freq -> 1D arrays)
    assert results['TL Line'].ndim == 1
    assert results['TL Grid'].ndim == 2
    assert len(results['TL Line']) > 0
    print(f"✓ Single frequency test passed - TL Line shape: {results['TL Line'].shape}")


def test_multi_frequency_output_shape():
    """Test that multi-frequency mode produces correct output shapes."""
    z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb = create_simple_environment()
    
    # Multiple frequencies
    freqs = [50.0, 100.0, 150.0]
    
    model = CuPyRAM(
        freq=freqs,
        zs=10.0,
        zr=50.0,
        z_ss=z_ss, rp_ss=rp_ss, cw=cw,
        z_sb=z_sb, rp_sb=rp_sb, cb=cb, rhob=rhob, attn=attn,
        rbzb=rbzb,
        rmax=1000.0,
        dr=10.0,
        dz=0.5
    )
    
    results = model.run()
    
    # Check output shapes (single env, 3 freqs -> [3, nvr])
    assert results['TL Line'].ndim == 2
    assert results['TL Line'].shape[0] == len(freqs)
    assert results['TL Grid'].ndim == 3
    assert results['TL Grid'].shape[0] == len(freqs)
    print(f"✓ Multi-frequency test passed - TL Line shape: {results['TL Line'].shape}")


def test_multi_frequency_vs_single_runs():
    """Test that multi-frequency results match multiple single-frequency runs.
    
    Important: Both multi-freq and single-freq runs must use the SAME grid 
    resolution to ensure numerical consistency. We explicitly set dr/dz based
    on the maximum frequency.
    """
    z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb = create_simple_environment()
    
    freqs = [75.0, 125.0]
    max_freq = np.max(freqs)
    
    # Calculate grid parameters based on max frequency (consistent with multi-freq behavior)
    # These values match the internal defaults that would be used for max_freq
    c0 = 1500.0
    np_pade = 8  # Default
    lambda_max = c0 / max_freq
    dr = np_pade * lambda_max  # Default formula from get_params
    dz = 0.4 * lambda_max  # Default formula from get_params (CuPyRAM._dzf = 0.4)
    
    # Run multi-frequency
    model_multi = CuPyRAM(
        freq=freqs,
        zs=10.0,
        zr=50.0,
        z_ss=z_ss, rp_ss=rp_ss, cw=cw,
        z_sb=z_sb, rp_sb=rp_sb, cb=cb, rhob=rhob, attn=attn,
        rbzb=rbzb,
        rmax=1000.0,
        dr=dr,  # Explicitly set grid resolution
        dz=dz,
        c0=c0
    )
    results_multi = model_multi.run()
    
    # Run single frequencies separately WITH SAME GRID RESOLUTION
    results_single = []
    for freq in freqs:
        model_single = CuPyRAM(
            freq=freq,
            zs=10.0,
            zr=50.0,
            z_ss=z_ss, rp_ss=rp_ss, cw=cw,
            z_sb=z_sb, rp_sb=rp_sb, cb=cb, rhob=rhob, attn=attn,
            rbzb=rbzb,
            rmax=1000.0,
            dr=dr,  # SAME grid resolution as multi-freq
            dz=dz,
            c0=c0
        )
        result = model_single.run()
        results_single.append(result['TL Line'])
    
    # Compare results
    for i, freq in enumerate(freqs):
        tl_multi = results_multi['TL Line'][i]
        tl_single = results_single[i]
        
        # Check shapes match
        assert tl_multi.shape == tl_single.shape
        
        # Check values are close (allow small numerical differences)
        # Note: 0.1 dB tolerance accounts for floating-point rounding differences 
        # in parallel execution and coarse grid discretization
        max_diff = np.max(np.abs(tl_multi - tl_single))
        print(f"  Freq {freq} Hz: max diff = {max_diff:.6e} dB")
        assert max_diff < 0.1, f"Results differ by {max_diff} dB at {freq} Hz"
    
    print(f"✓ Multi-frequency vs single runs test passed")


def test_batched_multi_frequency():
    """Test batched environments with multiple frequencies."""
    z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb = create_simple_environment()
    
    # Create 2 environments (slightly different source depths)
    n_env = 2
    freqs = [80.0, 120.0]
    
    model = CuPyRAM(
        freq=freqs,
        zs=10.0,
        zr=50.0,
        z_ss=[z_ss, z_ss],
        rp_ss=[rp_ss, rp_ss],
        cw=[cw, cw],
        z_sb=[z_sb, z_sb],
        rp_sb=[rp_sb, rp_sb],
        cb=[cb, cb],
        rhob=[rhob, rhob],
        attn=[attn, attn],
        rbzb=[rbzb, rbzb],
        rmax=1000.0,
        dr=10.0,
        dz=0.5,
        batch_size=n_env
    )
    
    results = model.run()
    
    # Check output shapes (2 envs, 2 freqs -> [2, 2, nvr])
    assert results['TL Line'].ndim == 3
    assert results['TL Line'].shape[0] == n_env
    assert results['TL Line'].shape[1] == len(freqs)
    print(f"✓ Batched multi-frequency test passed - TL Line shape: {results['TL Line'].shape}")


def test_grid_resolution_uses_max_frequency():
    """Test that grid resolution is based on maximum frequency."""
    z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb = create_simple_environment()
    
    # Frequencies with large range
    freqs = [50.0, 100.0, 200.0]
    
    model = CuPyRAM(
        freq=freqs,
        zs=10.0,
        zr=50.0,
        z_ss=z_ss, rp_ss=rp_ss, cw=cw,
        z_sb=z_sb, rp_sb=rp_sb, cb=cb, rhob=rhob, attn=attn,
        rbzb=rbzb,
        rmax=1000.0
        # Let dr and dz be auto-computed
    )
    
    # Grid should be based on max frequency (200 Hz)
    # Wavelength at 200 Hz: 1500/200 = 7.5 m
    # Default dr = np * 1500 / max_freq = 8 * 1500 / 200 = 60 m
    # Default dz = 0.1 * 1500 / max_freq = 0.1 * 1500 / 200 = 0.75 m
    
    assert model._dr == pytest.approx(60.0, rel=0.01)
    assert model._dz == pytest.approx(0.75, rel=0.01)
    print(f"✓ Grid resolution test passed - dr={model._dr:.1f}m, dz={model._dz:.3f}m")


if __name__ == "__main__":
    print("Running multi-frequency tests...\n")
    
    test_single_frequency_backward_compatibility()
    test_multi_frequency_output_shape()
    test_multi_frequency_vs_single_runs()
    test_batched_multi_frequency()
    test_grid_resolution_uses_max_frequency()
    
    print("\n✅ All multi-frequency tests passed!")
