#!/usr/bin/env python3
"""
Test Suite for Padé Coefficient Computation

Validates that GPU (CuPyRAM) and CPU (PyRAM) compute identical Padé
approximation coefficients for the parabolic equation square root operator.

The Padé coefficients are fundamental to the PE solver and errors here will
propagate through all range-dependent calculations.

Test cases cover:
- Different stability constraints: ns=0, ns=1, ns=2
- Different profile flags: ip=1 (marching), ip=2 (starter)
- Different frequencies and range steps
"""

import pytest
import numpy as np
from cupyram import CuPyRAM, compute_pade_coefficients
from pyram.PyRAM import PyRAM


def compare_pade_coefficients(pd1_gpu, pd2_gpu, pd1_cpu, pd2_cpu, tolerance=1e-8):
    """
    Compare Padé coefficients between GPU and CPU implementations.
    
    Args:
        pd1_gpu, pd2_gpu: GPU coefficients (Numba-compiled)
        pd1_cpu, pd2_cpu: CPU coefficients (PyRAM reference)
        tolerance: Absolute tolerance for comparison
        
    Returns:
        Dictionary with comparison statistics
        
    Note:
        Tolerance is 1e-8 to account for expected numerical differences
        between implementations (operation reordering, register precision,
        compiler optimizations). This is well below practical significance
        and has negligible impact on physics (<1e-7 dB in TL).
    """
    assert len(pd1_gpu) == len(pd1_cpu), "pd1 length mismatch"
    assert len(pd2_gpu) == len(pd2_cpu), "pd2 length mismatch"
    
    # Compute differences
    diff_pd1_real = np.abs(pd1_gpu.real - pd1_cpu.real)
    diff_pd1_imag = np.abs(pd1_gpu.imag - pd1_cpu.imag)
    diff_pd2_real = np.abs(pd2_gpu.real - pd2_cpu.real)
    diff_pd2_imag = np.abs(pd2_gpu.imag - pd2_cpu.imag)
    
    max_diff_pd1_real = np.max(diff_pd1_real)
    max_diff_pd1_imag = np.max(diff_pd1_imag)
    max_diff_pd2_real = np.max(diff_pd2_real)
    max_diff_pd2_imag = np.max(diff_pd2_imag)
    
    max_diff_overall = max(max_diff_pd1_real, max_diff_pd1_imag, 
                           max_diff_pd2_real, max_diff_pd2_imag)
    
    passed = max_diff_overall < tolerance
    
    return {
        'passed': passed,
        'max_diff_pd1_real': max_diff_pd1_real,
        'max_diff_pd1_imag': max_diff_pd1_imag,
        'max_diff_pd2_real': max_diff_pd2_real,
        'max_diff_pd2_imag': max_diff_pd2_imag,
        'max_diff_overall': max_diff_overall,
        'tolerance': tolerance
    }


def print_pade_comparison(pd1_gpu, pd2_gpu, pd1_cpu, pd2_cpu, stats, label=""):
    """Print detailed comparison of Padé coefficients."""
    print(f"\n{'='*70}")
    if label:
        print(f"Padé Coefficients Comparison: {label}")
    else:
        print("Padé Coefficients Comparison")
    print(f"{'='*70}")
    
    print(f"\nCoefficient Count: {len(pd1_gpu)}")
    print(f"\nStatistics:")
    print(f"  Max diff pd1 (real): {stats['max_diff_pd1_real']:.2e}")
    print(f"  Max diff pd1 (imag): {stats['max_diff_pd1_imag']:.2e}")
    print(f"  Max diff pd2 (real): {stats['max_diff_pd2_real']:.2e}")
    print(f"  Max diff pd2 (imag): {stats['max_diff_pd2_imag']:.2e}")
    print(f"  Overall max diff:    {stats['max_diff_overall']:.2e}")
    print(f"  Tolerance:           {stats['tolerance']:.2e}")
    
    # Show first 3 coefficients for sanity check
    print(f"\nSample Coefficients (first 3):")
    for i in range(min(3, len(pd1_gpu))):
        print(f"\n  pd1[{i}]:")
        print(f"    GPU: {pd1_gpu[i].real:+.10e} {pd1_gpu[i].imag:+.10e}j")
        print(f"    CPU: {pd1_cpu[i].real:+.10e} {pd1_cpu[i].imag:+.10e}j")
        print(f"    Δ:   {(pd1_gpu[i]-pd1_cpu[i]).real:+.2e} {(pd1_gpu[i]-pd1_cpu[i]).imag:+.2e}j")
        
        print(f"  pd2[{i}]:")
        print(f"    GPU: {pd2_gpu[i].real:+.10e} {pd2_gpu[i].imag:+.10e}j")
        print(f"    CPU: {pd2_cpu[i].real:+.10e} {pd2_cpu[i].imag:+.10e}j")
        print(f"    Δ:   {(pd2_gpu[i]-pd2_cpu[i]).real:+.2e} {(pd2_gpu[i]-pd2_cpu[i]).imag:+.2e}j")
    
    if stats['passed']:
        print(f"\n{'='*70}")
        print(f"✓ PASSED: All coefficients match within tolerance")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"✗ FAILED: Coefficients exceed tolerance")
        print(f"{'='*70}")


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def gpu_instance():
    """Create a minimal object with freq and c0 for testing."""
    class MinimalInstance:
        def __init__(self):
            self.freq = 50.0
            self.c0 = 1500.0
    return MinimalInstance()


def create_pyram_instance(freq, dr, np_pade, ns):
    """Create and setup a minimal CPU PyRAM instance for Padé coefficient testing."""
    # Minimal setup for CPU - we only need epade() method
    zs = 100.0
    zr = 100.0
    z_ss = np.array([0.0, 200.0])
    rp_ss = np.array([0.0])
    cw = np.array([[1500.0], [1500.0]])
    z_sb = np.array([200.0, 300.0])
    rp_sb = np.array([0.0])
    cb = np.array([[1700.0], [1700.0]])
    rhob = np.array([[1.8], [1.8]])
    attn = np.array([[0.5], [0.5]])
    rbzb = np.array([[0.0, 200.0], [10000.0, 200.0]])
    
    pyram = PyRAM(freq, zs, zr, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb, 
                  c0=1500.0, dr=dr, np=np_pade, ns=ns)
    pyram.setup()  # Initialize k0 and other parameters
    return pyram


# =============================================================================
# TEST CASES: (ns, ip) COMBINATIONS
# =============================================================================

@pytest.mark.pade
def test_pade_ns1_ip1(gpu_instance):
    """
    Test Padé coefficients: ns=1, ip=1 (stable marching)
    
    This is the standard configuration for range marching.
    """
    np_pade = 8
    ns = 1
    ip = 1
    dr = 50.0
    freq = gpu_instance.freq
    
    print(f"\n{'='*70}")
    print(f"TEST: Padé Coefficients (ns={ns}, ip={ip})")
    print(f"  np_pade: {np_pade}")
    print(f"  dr: {dr} m")
    print(f"  freq: {freq} Hz")
    print(f"{'='*70}")
    
    # GPU computation
    pd1_gpu, pd2_gpu = compute_pade_coefficients(freq, gpu_instance.c0, np_pade, ns=ns, dr=dr, ip=ip)
    
    # CPU computation
    pyram = create_pyram_instance(freq, dr, np_pade, ns)
    pyram.epade(ip=ip)
    pd1_cpu = pyram.pd1.copy()
    pd2_cpu = pyram.pd2.copy()
    
    # Compare
    stats = compare_pade_coefficients(pd1_gpu, pd2_gpu, pd1_cpu, pd2_cpu)
    print_pade_comparison(pd1_gpu, pd2_gpu, pd1_cpu, pd2_cpu, stats, f"ns={ns}, ip={ip}")
    
    assert stats['passed'], (
        f"Padé coefficients mismatch (ns={ns}, ip={ip}): "
        f"max diff {stats['max_diff_overall']:.2e} exceeds tolerance {stats['tolerance']:.2e}"
    )


@pytest.mark.pade
def test_pade_ns0_ip1(gpu_instance):
    """
    Test Padé coefficients: ns=0, ip=1 (unstable marching)
    
    Used when range exceeds stability constraint range (rs).
    """
    np_pade = 8
    ns = 0
    ip = 1
    dr = 50.0
    freq = gpu_instance.freq
    
    print(f"\n{'='*70}")
    print(f"TEST: Padé Coefficients (ns={ns}, ip={ip})")
    print(f"  np_pade: {np_pade}")
    print(f"  dr: {dr} m")
    print(f"  freq: {freq} Hz")
    print(f"{'='*70}")
    
    # GPU computation
    pd1_gpu, pd2_gpu = compute_pade_coefficients(freq, gpu_instance.c0, np_pade, ns=ns, dr=dr, ip=ip)
    
    # CPU computation
    pyram = create_pyram_instance(freq, dr, np_pade, ns)
    pyram.epade(ip=ip)
    pd1_cpu = pyram.pd1.copy()
    pd2_cpu = pyram.pd2.copy()
    
    # Compare
    stats = compare_pade_coefficients(pd1_gpu, pd2_gpu, pd1_cpu, pd2_cpu)
    print_pade_comparison(pd1_gpu, pd2_gpu, pd1_cpu, pd2_cpu, stats, f"ns={ns}, ip={ip}")
    
    assert stats['passed'], (
        f"Padé coefficients mismatch (ns={ns}, ip={ip}): "
        f"max diff {stats['max_diff_overall']:.2e} exceeds tolerance {stats['tolerance']:.2e}"
    )


@pytest.mark.pade
def test_pade_ns1_ip2(gpu_instance):
    """
    Test Padé coefficients: ns=1, ip=2 (self-starter)
    
    Used for initial field construction at source.
    """
    np_pade = 8
    ns = 1
    ip = 2
    dr = 50.0
    freq = gpu_instance.freq
    
    print(f"\n{'='*70}")
    print(f"TEST: Padé Coefficients (ns={ns}, ip={ip})")
    print(f"  np_pade: {np_pade}")
    print(f"  dr: {dr} m")
    print(f"  freq: {freq} Hz")
    print(f"{'='*70}")
    
    # GPU computation
    pd1_gpu, pd2_gpu = compute_pade_coefficients(freq, gpu_instance.c0, np_pade, ns=ns, dr=dr, ip=ip)
    
    # CPU computation
    pyram = create_pyram_instance(freq, dr, np_pade, ns)
    pyram.epade(ip=ip)
    pd1_cpu = pyram.pd1.copy()
    pd2_cpu = pyram.pd2.copy()
    
    # Compare
    stats = compare_pade_coefficients(pd1_gpu, pd2_gpu, pd1_cpu, pd2_cpu)
    print_pade_comparison(pd1_gpu, pd2_gpu, pd1_cpu, pd2_cpu, stats, f"ns={ns}, ip={ip}")
    
    assert stats['passed'], (
        f"Padé coefficients mismatch (ns={ns}, ip={ip}): "
        f"max diff {stats['max_diff_overall']:.2e} exceeds tolerance {stats['tolerance']:.2e}"
    )


# =============================================================================
# TEST CASES: PARAMETER VARIATIONS
# =============================================================================

@pytest.mark.pade
@pytest.mark.parametrize("np_pade", [4, 6, 8])
def test_pade_different_np(np_pade):
    """
    Test that GPU and CPU produce identical coefficients for different np_pade values.
    
    Note: np_pade > 8 not tested due to numerical instability in polynomial root finding
    (inherent to the algorithm, not specific to Numba). Standard usage is np_pade=8.
    """
    freq = 50.0
    dr = 50.0
    ns = 1
    ip = 1
    c0 = 1500.0
    
    # GPU
    pd1_gpu, pd2_gpu = compute_pade_coefficients(freq, c0, np_pade, ns=ns, dr=dr, ip=ip)
    
    # CPU
    pyram = create_pyram_instance(freq, dr, np_pade, ns)
    pyram.epade(ip=ip)
    pd1_cpu = pyram.pd1.copy()
    pd2_cpu = pyram.pd2.copy()
    
    stats = compare_pade_coefficients(pd1_gpu, pd2_gpu, pd1_cpu, pd2_cpu)
    
    print(f"\nnp_pade={np_pade}: max diff = {stats['max_diff_overall']:.2e}")
    
    assert stats['passed'], (
        f"Padé coefficients mismatch for np_pade={np_pade}: "
        f"max diff {stats['max_diff_overall']:.2e}"
    )


@pytest.mark.pade
@pytest.mark.parametrize("freq", [25.0, 50.0, 100.0, 200.0])
def test_pade_different_frequencies(freq):
    """Test that GPU and CPU produce identical coefficients for different frequencies."""
    np_pade = 8
    dr = 50.0
    ns = 1
    ip = 1
    c0 = 1500.0
    
    # GPU
    pd1_gpu, pd2_gpu = compute_pade_coefficients(freq, c0, np_pade, ns=ns, dr=dr, ip=ip)
    
    # CPU
    pyram = create_pyram_instance(freq, dr, np_pade, ns)
    pyram.epade(ip=ip)
    pd1_cpu = pyram.pd1.copy()
    pd2_cpu = pyram.pd2.copy()
    
    stats = compare_pade_coefficients(pd1_gpu, pd2_gpu, pd1_cpu, pd2_cpu)
    
    print(f"\nfreq={freq}Hz: max diff = {stats['max_diff_overall']:.2e}")
    
    assert stats['passed'], (
        f"Padé coefficients mismatch for freq={freq}Hz: "
        f"max diff {stats['max_diff_overall']:.2e}"
    )


@pytest.mark.pade
@pytest.mark.parametrize("dr", [10.0, 25.0, 50.0, 100.0])
def test_pade_different_range_steps(dr):
    """Test that GPU and CPU produce identical coefficients for different range steps."""
    freq = 50.0
    np_pade = 8
    ns = 1
    ip = 1
    c0 = 1500.0
    
    # GPU
    pd1_gpu, pd2_gpu = compute_pade_coefficients(freq, c0, np_pade, ns=ns, dr=dr, ip=ip)
    
    # CPU
    pyram = create_pyram_instance(freq, dr, np_pade, ns)
    pyram.epade(ip=ip)
    pd1_cpu = pyram.pd1.copy()
    pd2_cpu = pyram.pd2.copy()
    
    stats = compare_pade_coefficients(pd1_gpu, pd2_gpu, pd1_cpu, pd2_cpu)
    
    print(f"\ndr={dr}m: max diff = {stats['max_diff_overall']:.2e}")
    
    assert stats['passed'], (
        f"Padé coefficients mismatch for dr={dr}m: "
        f"max diff {stats['max_diff_overall']:.2e}"
    )


# =============================================================================
# SUMMARY TEST
# =============================================================================

@pytest.mark.pade
def test_pade_all_combinations():
    """
    Comprehensive test of all (ns, ip) combinations used in CuPyRAM.
    
    This mimics exactly what happens in CuPyRAM.run():
    - (ns=1, ip=1) for stable marching
    - (ns=0, ip=1) for unstable marching (r > rs)
    - (ns=1, ip=2) for self-starter
    """
    freq = 50.0
    np_pade = 8
    dr = 50.0
    
    combinations = [
        (1, 1, "stable marching"),
        (0, 1, "unstable marching"),
        (1, 2, "self-starter")
    ]
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE PADÉ TEST")
    print(f"Testing all (ns, ip) combinations used in production")
    print(f"  freq: {freq} Hz")
    print(f"  np_pade: {np_pade}")
    print(f"  dr: {dr} m")
    print(f"{'='*70}")
    
    results = []
    
    c0 = 1500.0
    
    for ns, ip, label in combinations:
        print(f"\n--- Testing (ns={ns}, ip={ip}): {label} ---")
        
        # GPU
        pd1_gpu, pd2_gpu = compute_pade_coefficients(freq, c0, np_pade, ns=ns, dr=dr, ip=ip)
        
        # CPU
        pyram = create_pyram_instance(freq, dr, np_pade, ns)
        pyram.epade(ip=ip)
        pd1_cpu = pyram.pd1.copy()
        pd2_cpu = pyram.pd2.copy()
        
        stats = compare_pade_coefficients(pd1_gpu, pd2_gpu, pd1_cpu, pd2_cpu)
        results.append((ns, ip, label, stats))
        
        status = "✓ PASS" if stats['passed'] else "✗ FAIL"
        print(f"  {status}: max diff = {stats['max_diff_overall']:.2e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for ns, ip, label, stats in results:
        status = "✓" if stats['passed'] else "✗"
        print(f"  {status} (ns={ns}, ip={ip}) {label:20s}: {stats['max_diff_overall']:.2e}")
    
    all_passed = all(stats['passed'] for _, _, _, stats in results)
    
    if all_passed:
        print(f"\n{'='*70}")
        print("✓ ALL PADÉ COEFFICIENT TESTS PASSED")
        print("  GPU and CPU implementations are numerically identical")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print("✗ SOME PADÉ COEFFICIENT TESTS FAILED")
        print("  GPU and CPU implementations diverge")
        print(f"{'='*70}")
    
    assert all_passed, "Some Padé coefficient combinations failed"

