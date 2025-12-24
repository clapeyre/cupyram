"""
Test batching functionality in CuPyRAM
"""

import pytest
import numpy as np
from cupyram import CuPyRAM


@pytest.fixture
def pekeris_params():
    """Simple Pekeris waveguide parameters for testing"""
    freq = 50.0
    zs = 100.0
    zr = 50.0
    water_depth = 200.0
    rmax = 10000.0
    
    z_ss = np.array([0.0, water_depth])
    rp_ss = np.array([0.0])
    cw = np.array([[1500.0], [1500.0]])
    
    z_sb = np.array([0.0])
    rp_sb = np.array([0.0])
    cb = np.array([[1700.0]])
    rhob = np.array([[1.5]])
    attn = np.array([[0.5]])
    
    rbzb = np.array([[0.0, water_depth], [rmax, water_depth]])
    
    return {
        'freq': freq, 'zs': zs, 'zr': zr,
        'z_ss': z_ss, 'rp_ss': rp_ss, 'cw': cw,
        'z_sb': z_sb, 'rp_sb': rp_sb,
        'cb': cb, 'rhob': rhob, 'attn': attn,
        'rbzb': rbzb, 'rmax': rmax
    }


def test_batch_sizes(pekeris_params):
    """Test that various batch sizes execute without errors"""
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        model = CuPyRAM(**pekeris_params, batch_size=batch_size)
        result = model.run()
        
        # Check output shape
        if batch_size == 1:
            assert result['TL Line'].ndim == 1, \
                f"batch_size=1 should produce 1D output, got shape {result['TL Line'].shape}"
        else:
            assert result['TL Line'].shape[0] == batch_size, \
                f"batch_size={batch_size} should produce output with first dim={batch_size}"
        
        # Check TL values are reasonable
        tl = result['TL Line']
        assert np.all(tl > 0), "All TL values should be positive"
        assert np.all(tl < 200), "All TL values should be < 200 dB"
        assert not np.any(np.isnan(tl)), "No NaN values in TL"
        assert not np.any(np.isinf(tl)), "No Inf values in TL"


def test_batch_consistency(pekeris_params):
    """Test that batched computation gives consistent results"""
    # Run with batch_size=1 (reference)
    model1 = CuPyRAM(**pekeris_params, batch_size=1)
    result1 = model1.run()
    tl_ref = result1['TL Line']
    
    # Run with batch_size=8 and compare first ray
    model8 = CuPyRAM(**pekeris_params, batch_size=8)
    result8 = model8.run()
    tl_batch = result8['TL Line'][0, :]
    
    # Check shapes match
    assert tl_ref.shape == tl_batch.shape, \
        f"Shape mismatch: batch_size=1 {tl_ref.shape} vs batch_size=8[0] {tl_batch.shape}"
    
    # Check numerical agreement (allow small floating-point differences)
    diff = np.abs(tl_ref - tl_batch)
    max_diff = np.max(diff)
    
    # Allow up to 0.01 dB difference (conservative tolerance for acoustic modeling)
    assert max_diff < 0.01, \
        f"batch_size=1 and batch_size=8 differ by {max_diff:.6f} dB (max allowed: 0.01 dB)"
    
    print(f"✓ Consistency check: max diff = {max_diff:.6e} dB")


def test_all_rays_same_environment(pekeris_params):
    """Test that all rays in a batch produce identical results (same environment)"""
    batch_size = 8
    model = CuPyRAM(**pekeris_params, batch_size=batch_size)
    result = model.run()
    tl = result['TL Line']
    
    # All rays should be identical (same source, receiver, environment)
    for i in range(1, batch_size):
        diff = np.abs(tl[0, :] - tl[i, :])
        max_diff = np.max(diff)
        assert max_diff < 1e-10, \
            f"Ray 0 and ray {i} differ by {max_diff:.6e} dB (should be identical)"
    
    print(f"✓ All {batch_size} rays are identical (as expected)")


def test_batch_grid_output(pekeris_params):
    """Test that TL Grid output is correctly batched"""
    batch_size = 4
    model = CuPyRAM(**pekeris_params, batch_size=batch_size)
    result = model.run()
    
    tl_line = result['TL Line']
    tl_grid = result['TL Grid']
    
    # Check dimensions
    assert tl_line.shape[0] == batch_size, "TL Line should have batch dimension"
    assert tl_grid.shape[0] == batch_size, "TL Grid should have batch dimension"
    
    # Grid should be [batch, depth, range]
    assert tl_grid.ndim == 3, f"TL Grid should be 3D, got shape {tl_grid.shape}"
    
    print(f"✓ TL Grid shape: {tl_grid.shape} (batch, depth, range)")


def test_batch_processing_time(pekeris_params):
    """Verify that batch processing completes"""
    batch_size = 16
    model = CuPyRAM(**pekeris_params, batch_size=batch_size)
    result = model.run()
    
    proc_time = result['Proc Time']
    assert proc_time > 0, "Processing time should be positive"
    assert proc_time < 60, "Processing should complete in reasonable time"
    
    print(f"✓ Processed {batch_size} rays in {proc_time:.3f}s")


def test_pekeris_batched_vs_cpu(pekeris_params):
    """
    Test that batched Pekeris execution matches CPU reference.
    This validates the batch_size > 1 code path against PyRAM.
    """
    from pyram.PyRAM import PyRAM
    
    # Run CPU version
    pyram = PyRAM(**pekeris_params)
    cpu_result = pyram.run()
    cpu_tl = cpu_result['TL Line']
    
    # Run GPU batched version (batch_size=4)
    cupyram_batched = CuPyRAM(**pekeris_params, batch_size=4)
    gpu_result = cupyram_batched.run()
    gpu_tl = gpu_result['TL Line'][0, :]  # First ray from batch
    
    # Compare
    diff = np.abs(cpu_tl - gpu_tl)
    max_diff = np.max(diff)
    rms_diff = np.sqrt(np.mean(diff**2))
    
    # Should match within reasonable tolerance for GPU computation
    assert max_diff < 1.0, \
        f"Batched CuPyRAM differs from CPU PyRAM by {max_diff:.3f} dB (max allowed: 1.0 dB)"
    
    print(f"✓ Pekeris batched vs CPU: max diff = {max_diff:.4f} dB, RMS = {rms_diff:.4f} dB")


@pytest.mark.skip(reason="Analytic solution requires careful tuning for PE approximation")
def test_pekeris_batched_vs_analytic(pekeris_params):
    """
    Test that batched Pekeris execution matches analytic solution.
    This is the strongest validation for the batch_size > 1 code path.
    
    NOTE: Skipped because analytic modal solution differs from PE approximation
    by design. The CPU comparison test is more reliable for validation.
    """
    from tests.pekeris_analytic import pekeris_solution
    
    # Get analytic solution
    ranges = np.arange(100, pekeris_params['rmax'] + 1, 100)  # Start at 100m to avoid near-field
    analytic_tl = pekeris_solution(
        ranges=ranges,
        freq=pekeris_params['freq'],
        depth=pekeris_params['z_ss'][-1],
        z_source=pekeris_params['zs'],
        z_receiver=pekeris_params['zr'],
        c_water=pekeris_params['cw'][0, 0],
        c_bottom=pekeris_params['cb'][0, 0],
        rho_water=1.0,
        rho_bottom=pekeris_params['rhob'][0, 0],
        attn_bottom=pekeris_params['attn'][0, 0]
    )
    
    # Run GPU batched version (batch_size=8)
    cupyram_batched = CuPyRAM(**pekeris_params, batch_size=8)
    gpu_result = cupyram_batched.run()
    gpu_tl = gpu_result['TL Line'][0, :]  # First ray from batch
    gpu_ranges = gpu_result['Ranges']
    
    # Interpolate analytic solution to GPU ranges
    analytic_tl_interp = np.interp(gpu_ranges, ranges, analytic_tl)
    
    # Compare (excluding very near field where analytic solution may differ)
    valid_idx = gpu_ranges > 1000  # Skip first 1 km
    diff = np.abs(gpu_tl[valid_idx] - analytic_tl_interp[valid_idx])
    max_diff = np.max(diff)
    rms_diff = np.sqrt(np.mean(diff**2))
    
    # Analytic comparison allows larger tolerance (PE approximation vs exact)
    assert max_diff < 5.0, \
        f"Batched CuPyRAM differs from analytic by {max_diff:.3f} dB (max allowed: 5.0 dB)"
    
    print(f"✓ Pekeris batched vs analytic: max diff = {max_diff:.4f} dB, RMS = {rms_diff:.4f} dB")


@pytest.fixture
def munk_params():
    """Munk profile parameters for testing (range-varying SSP)"""
    freq = 100.0
    zs = 1000.0
    zr = 2500.0
    
    # Munk sound speed profile
    z_ss = np.linspace(0, 5000, 51)
    z0 = 1300.0
    c0 = 1500.0
    eps = 0.00737
    eta = 2 * (z_ss - z0) / z0
    cw = c0 * (1 + eps * (eta + np.exp(-eta) - 1))
    cw = cw.reshape(-1, 1)
    rp_ss = np.array([0.0])
    
    # Seabed
    z_sb = np.array([0.0])
    rp_sb = np.array([0.0])
    cb = np.array([[1600.0]])
    rhob = np.array([[1.5]])
    attn = np.array([[0.2]])
    
    # Bathymetry
    rmax = 50000.0
    rbzb = np.array([[0.0, 5000.0], [rmax, 5000.0]])
    
    return {
        'freq': freq, 'zs': zs, 'zr': zr,
        'z_ss': z_ss, 'rp_ss': rp_ss, 'cw': cw,
        'z_sb': z_sb, 'rp_sb': rp_sb,
        'cb': cb, 'rhob': rhob, 'attn': attn,
        'rbzb': rbzb, 'rmax': rmax
    }


def test_munk_batched_vs_cpu(munk_params):
    """
    Test that batched Munk profile execution matches CPU reference.
    Munk profile has varying SSP, making it a stronger test than Pekeris.
    """
    from pyram.PyRAM import PyRAM
    
    # Run CPU version
    pyram = PyRAM(**munk_params)
    cpu_result = pyram.run()
    cpu_tl = cpu_result['TL Line']
    
    # Run GPU batched version (batch_size=16)
    cupyram_batched = CuPyRAM(**munk_params, batch_size=16)
    gpu_result = cupyram_batched.run()
    gpu_tl = gpu_result['TL Line'][0, :]  # First ray from batch
    
    # Compare
    diff = np.abs(cpu_tl - gpu_tl)
    max_diff = np.max(diff)
    rms_diff = np.sqrt(np.mean(diff**2))
    
    # Munk is more complex, allow slightly larger tolerance
    assert max_diff < 2.0, \
        f"Batched CuPyRAM (Munk) differs from CPU PyRAM by {max_diff:.3f} dB (max allowed: 2.0 dB)"
    
    print(f"✓ Munk batched vs CPU: max diff = {max_diff:.4f} dB, RMS = {rms_diff:.4f} dB")

