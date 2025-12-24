#!/usr/bin/env python3
"""
Reference Test: Original RAM Test Case

Tests CuPyRAM against the reference solution from the original Fortran RAM code.
This is the test case supplied with RAM v1.5.

Environment: Range-dependent sound speed with sloping bathymetry
Success criteria: Mean TL difference < 0.01 dB from reference
"""

import pytest
import numpy as np
from pathlib import Path
from cupyram import CuPyRAM
from .test_utils import print_section


# Test configuration (from original RAM test case)
FREQ = 50.0
Z_SOURCE = 50.0
Z_RECEIVER = 50.0
RMAX = 50000.0
DR = 500.0
DZ = 2.0
ZMPLT = 500.0
C0 = 1600.0
TL_TOLERANCE = 0.01  # dB - mean absolute difference


@pytest.fixture
def reference_data():
    """Load reference TL data from original RAM output."""
    ref_file = Path(__file__).parent / 'tl_ref.line'
    data = np.loadtxt(ref_file)
    ranges = data[:, 0]
    tl = data[:, 1]
    return ranges, tl


@pytest.mark.reference
def test_ram_reference_case(reference_data, output_dir):
    """
    Test CuPyRAM against original RAM v1.5 reference output.
    
    This is the standard test case that came with the Fortran RAM distribution.
    It features:
    - Range-dependent sound speed (upward refracting profile that changes with range)
    - Sloping bathymetry (200m to 400m)
    """
    
    ref_ranges, ref_tl = reference_data
    
    print_section("RAM REFERENCE TEST CASE")
    print(f"\nTest Configuration (from RAM v1.5):")
    print(f"  Frequency: {FREQ} Hz")
    print(f"  Source depth: {Z_SOURCE} m")
    print(f"  Receiver depth: {Z_RECEIVER} m")
    print(f"  Max range: {RMAX/1000:.1f} km")
    print(f"  Range step: {DR} m")
    print(f"  Depth step: {DZ} m")
    print(f"  Reference c0: {C0} m/s")
    
    # =========================================================================
    # STEP 1: SETUP ENVIRONMENT
    # =========================================================================
    print_section("Step 1: Setup Environment")
    
    # Sound speed profile (changes with range)
    z_ss = np.array([0, 100, 400])
    cw = np.array([[1480, 1530],
                   [1520, 1530],
                   [1530, 1530]])
    rp_ss = np.array([0, 25000])
    
    print(f"\nSound speed profile:")
    print(f"  Depths: {z_ss} m")
    print(f"  Range points: {rp_ss/1000} km")
    print(f"  At r=0km: surface={cw[0,0]} m/s, 100m={cw[1,0]} m/s, 400m={cw[2,0]} m/s")
    print(f"  At r=25km: surface={cw[0,1]} m/s, 100m={cw[1,1]} m/s, 400m={cw[2,1]} m/s")
    
    # Seabed parameters (range-independent)
    z_sb = np.array([0])
    cb = np.array([[1700]])
    rhob = np.array([[1.5]])
    attn = np.array([[0.5]])
    rp_sb = np.array([0])
    
    # Sloping bathymetry
    rbzb = np.array([[0, 200],
                     [40000, 400]])
    
    print(f"\nBathymetry:")
    print(f"  r=0km: {rbzb[0,1]} m")
    print(f"  r=40km: {rbzb[1,1]} m")
    print(f"  Slope: {(rbzb[1,1]-rbzb[0,1])/(rbzb[1,0]/1000):.2f} m/km")
    
    # =========================================================================
    # STEP 2: RUN CUPYRAM
    # =========================================================================
    print_section("Step 2: Run CuPyRAM")
    
    cupyram = CuPyRAM(
        freq=FREQ,
        zs=Z_SOURCE,
        zr=Z_RECEIVER,
        z_ss=z_ss,
        rp_ss=rp_ss,
        cw=cw,
        z_sb=z_sb,
        rp_sb=rp_sb,
        cb=cb,
        rhob=rhob,
        attn=attn,
        rbzb=rbzb,
        rmax=RMAX,
        dr=DR,
        dz=DZ,
        zmplt=ZMPLT,
        c0=C0
    )
    
    print(f"\nRunning CuPyRAM...")
    results = cupyram.run()
    
    tl_cupyram = results['TL Line']
    ranges_cupyram = results['Ranges']
    
    print(f"  Completed in {results['Proc Time']:.3f} seconds")
    print(f"  Output points: {len(tl_cupyram)}")
    print(f"  TL range: [{np.min(tl_cupyram):.2f}, {np.max(tl_cupyram):.2f}] dB")
    
    # =========================================================================
    # STEP 3: COMPARE WITH REFERENCE
    # =========================================================================
    print_section("Step 3: Compare with Reference")
    
    # Check ranges match
    assert np.array_equal(ref_ranges, ranges_cupyram), (
        f"Range arrays don't match! "
        f"Reference: {len(ref_ranges)} points, CuPyRAM: {len(ranges_cupyram)} points"
    )
    print(f"  ✓ Range arrays match ({len(ref_ranges)} points)")
    
    # Calculate differences
    diff = tl_cupyram - ref_tl
    abs_diff = np.abs(diff)
    mean_diff = np.mean(abs_diff)
    max_diff = np.max(abs_diff)
    rms_diff = np.sqrt(np.mean(diff**2))
    
    print(f"\nStatistics:")
    print(f"  Mean absolute difference: {mean_diff:.4f} dB")
    print(f"  Max absolute difference:  {max_diff:.4f} dB")
    print(f"  RMS difference:           {rms_diff:.4f} dB")
    print(f"  Tolerance:                {TL_TOLERANCE:.4f} dB")
    
    # Show sample comparisons
    print(f"\nSample Comparisons (at selected ranges):")
    sample_indices = np.linspace(0, len(ref_ranges)-1, 5, dtype=int)
    for idx in sample_indices:
        r_km = ref_ranges[idx] / 1000
        print(f"  {r_km:6.1f} km: Reference={ref_tl[idx]:6.2f} dB, "
              f"CuPyRAM={tl_cupyram[idx]:6.2f} dB, diff={diff[idx]:+6.3f} dB")
    
    # =========================================================================
    # STEP 4: SAVE RESULTS
    # =========================================================================
    print_section("Step 4: Save Results")
    
    # Save CuPyRAM output in same format as reference
    output_file = output_dir / 'tl_cupyram.line'
    with open(output_file, 'w') as f:
        for i in range(len(ranges_cupyram)):
            f.write(f"{ranges_cupyram[i]}\t{tl_cupyram[i]}\n")
    print(f"  Saved: {output_file}")
    
    # Save comparison
    comparison_file = output_dir / 'tl_reference_comparison.npz'
    np.savez(comparison_file,
             ranges=ref_ranges,
             tl_reference=ref_tl,
             tl_cupyram=tl_cupyram,
             diff=diff)
    print(f"  Saved: {comparison_file}")
    
    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    print_section("REFERENCE TEST FINAL VERDICT")
    
    passed = mean_diff <= TL_TOLERANCE
    
    if passed:
        print(f"\n✓ PASSED: Mean difference {mean_diff:.4f} dB < {TL_TOLERANCE} dB threshold")
        print(f"\n{'='*70}")
        print("✓ REFERENCE TEST PASSED")
        print("  → CuPyRAM matches original RAM v1.5 output")
        print("  → Implementation validated against Fortran reference")
        print(f"{'='*70}")
    else:
        print(f"\n✗ FAILED: Mean difference {mean_diff:.4f} dB > {TL_TOLERANCE} dB threshold")
    
    # Pytest assertion
    assert passed, (
        f"CuPyRAM vs Reference accuracy test failed: "
        f"mean difference {mean_diff:.4f} dB exceeds tolerance {TL_TOLERANCE} dB"
    )
