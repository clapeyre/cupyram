#!/usr/bin/env python3
"""
Phase 3: Range-Dependent Bathymetry - ASA Wedge

Tests energy conservation and bathymetry update logic with shoaling environment.
Environment: Linear upslope from 200m to 10m over 4km (ASA wedge benchmark).

Success criteria:
- GPU vs CPU < 0.5 dB → Energy conservation validated
- No NaN/Inf values → Interface handling correct
"""

import pytest
import numpy as np
from cupyram import CuPyRAM
from .test_utils import (SimulationConfig, run_pyram_reference, compare_results,
                        print_comparison, plot_comparison, save_test_results,
                        print_section, print_test_result, TOLERANCES)
import matplotlib.pyplot as plt

# Test configuration
DEPTH_START = 200.0
DEPTH_MIN = 10.0  # Strong wedge: 200m -> 10m (keeps wedge effect)
RANGE_WEDGE = 4000.0
FREQ = 50.0
Z_SOURCE = 5.0  # Shallow source (stays above 10m minimum depth)
Z_RECEIVER = 3.0  # Shallow receiver (stays above 10m minimum depth)
RMAX = 10000.0
DR = 50.0
NP_PADE = 8
C_WATER = 1500.0

slope_angle = np.arctan((DEPTH_START - DEPTH_MIN) / RANGE_WEDGE) * 180 / np.pi


@pytest.mark.wedge
@pytest.mark.slow
def test_asa_wedge(output_dir):
    """
    Phase 3: ASA Wedge Test
    
    Validates energy conservation and bathymetry handling with range-dependent depth.
    
    Note: DEPTH_MIN must be > max(Z_SOURCE, Z_RECEIVER) to keep source/receiver 
    above the seafloor throughout the entire propagation path.
    """
    
    # Safety check
    assert DEPTH_MIN > max(Z_SOURCE, Z_RECEIVER), (
        f"Minimum depth ({DEPTH_MIN}m) must be greater than "
        f"max(source, receiver) depth ({max(Z_SOURCE, Z_RECEIVER)}m)"
    )
    
    config = SimulationConfig(
        name="Phase 3: ASA Wedge",
        freq=FREQ,
        zs=Z_SOURCE,
        zr=Z_RECEIVER,
        rmax=RMAX,
        dr=DR,
        depth=DEPTH_START,
        c0=C_WATER,
        tolerance=TOLERANCES['phase3_cpu']
    )
    
    print_section("PHASE 3: RANGE-DEPENDENT BATHYMETRY - ASA WEDGE")
    print(config)
    print(f"\nWedge Configuration:")
    print(f"  Initial depth: {DEPTH_START} m")
    print(f"  Final depth: {DEPTH_MIN} m")
    print(f"  Slope range: {RANGE_WEDGE/1000:.1f} km")
    print(f"  Slope angle: {slope_angle:.2f}°")
    
    # =========================================================================
    # STEP 1: CREATE WEDGE BATHYMETRY
    # =========================================================================
    print_section("Step 1: Create Wedge Bathymetry")
    
    num_bathy_points = 200
    r_bathy = np.linspace(0, RMAX, num_bathy_points)
    z_bathy = np.where(
        r_bathy <= RANGE_WEDGE,
        DEPTH_START - (r_bathy / RANGE_WEDGE) * (DEPTH_START - DEPTH_MIN),
        DEPTH_MIN
    )
    z_bathy = np.maximum(z_bathy, DEPTH_MIN)
    
    print(f"\nBathymetry Profile:")
    print(f"  Points: {len(r_bathy)}")
    print(f"  Depth at r=0: {z_bathy[0]:.1f} m")
    print(f"  Depth at r={RMAX/1000:.1f}km: {z_bathy[-1]:.1f} m")
    
    # =========================================================================
    # STEP 2: CPU PYRAM REFERENCE
    # =========================================================================
    print_section("Step 2: CPU PyRAM Reference")
    
    # Calculate default dz
    dz_common = 0.1 * 1500 / FREQ
    
    z_ss = np.array([0.0, DEPTH_START])
    cw = np.array([[C_WATER], [C_WATER]])
    z_sb = np.array([DEPTH_START, DEPTH_START + 100])
    cb = np.array([[1700.0], [1700.0]])
    rhob = np.array([[1.8], [1.8]])
    attn = np.array([[0.5], [0.5]])
    rbzb = np.column_stack((r_bathy, z_bathy))
    
    print(f"Using common dz: {dz_common:.4f} m")
    
    tl_cpu, pyram = run_pyram_reference(
        config, z_ss, cw, z_sb, cb, rhob, attn, rbzb,
        dz=dz_common, np_pade=NP_PADE, ns=1
    )
    
    cpu_nz = pyram.nz
    zmax_exact = pyram._zmax
    
    print(f"\nCPU Grid Configuration:")
    print(f"  nz: {cpu_nz}")
    print(f"  dz: {dz_common:.4f} m")
    print(f"  zmax: {zmax_exact:.2f} m")
    
    # =========================================================================
    # STEP 3: CUPYRAM SIMULATION
    # =========================================================================
    print_section("Step 3: CuPyRAM Simulation")
    
    print(f"\nCuPyRAM Configuration:")
    print(f"  Wedge bathymetry: {DEPTH_START}m -> {DEPTH_MIN}m")
    print(f"  Using dz: {dz_common:.4f} m (matching CPU)")
    
    # Create fresh copies of arrays for CuPyRAM
    z_ss_cupyram = np.array([0.0, DEPTH_START])
    cw_cupyram = np.array([[C_WATER], [C_WATER]])
    z_sb_cupyram = np.array([DEPTH_START, DEPTH_START + 100])
    cb_cupyram = np.array([[1700.0], [1700.0]])
    rhob_cupyram = np.array([[1.8], [1.8]])
    attn_cupyram = np.array([[0.5], [0.5]])
    rbzb_cupyram = np.column_stack((r_bathy, z_bathy))
    
    # Create CuPyRAM instance with same parameters as PyRAM
    sim = CuPyRAM(
        freq=FREQ,
        zs=Z_SOURCE,
        zr=Z_RECEIVER,
        z_ss=z_ss_cupyram,
        rp_ss=np.array([0.0]),
        cw=cw_cupyram,
        z_sb=z_sb_cupyram,
        rp_sb=np.array([0.0]),
        cb=cb_cupyram,
        rhob=rhob_cupyram,
        attn=attn_cupyram,
        rbzb=rbzb_cupyram,
        dz=dz_common,
        dr=DR,
        rmax=RMAX,
        np=NP_PADE,
        ns=1
    )
    
    print(f"\nRunning CuPyRAM simulation...")
    cupyram_results = sim.run()
    
    # =========================================================================
    # ASSERTION 1: CUPYRAM SIMULATION COMPLETED SUCCESSFULLY
    # =========================================================================
    print_section("Verification: CuPyRAM Simulation Execution")
    
    # Extract TL results
    tl_cupyram = cupyram_results['TL Line']
    
    # Check for NaN/Inf values
    nan_count = np.sum(np.isnan(tl_cupyram))
    inf_count = np.sum(np.isinf(tl_cupyram))
    assert nan_count == 0, f"CuPyRAM results contain {nan_count} NaN values"
    assert inf_count == 0, f"CuPyRAM results contain {inf_count} Inf values"
    print(f"  ✓ No NaN or Inf values")
    
    # Check TL values are reasonable
    # Note: Wedge scenarios can have high TL values due to strong shoaling effects
    assert np.min(tl_cupyram) > 0, f"TL values should be positive, got min={np.min(tl_cupyram):.2f} dB"
    assert np.max(tl_cupyram) < 300, f"TL values unreasonably high, got max={np.max(tl_cupyram):.2f} dB"
    print(f"  ✓ TL range: [{np.min(tl_cupyram):.2f}, {np.max(tl_cupyram):.2f}] dB")
    
    print("\n✓ CuPyRAM simulation completed successfully")
    
    # =========================================================================
    # STEP 3B: CUPYRAM BATCHED SIMULATION
    # =========================================================================
    print_section("Step 3b: CuPyRAM Batched Simulation (batch_size=8)")
    
    print(f"\nRunning CuPyRAM with batch_size=8...")
    print(f"  Testing batched execution with range-dependent bathymetry")
    
    # Create fresh copies of arrays for batched CuPyRAM
    z_ss_batched = np.array([0.0, DEPTH_START])
    cw_batched = np.array([[C_WATER], [C_WATER]])
    z_sb_batched = np.array([DEPTH_START, DEPTH_START + 100])
    cb_batched = np.array([[1700.0], [1700.0]])
    rhob_batched = np.array([[1.8], [1.8]])
    attn_batched = np.array([[0.5], [0.5]])
    rbzb_batched = np.column_stack((r_bathy, z_bathy))
    
    sim_batched = CuPyRAM(
        freq=FREQ,
        zs=Z_SOURCE,
        zr=Z_RECEIVER,
        z_ss=z_ss_batched,
        rp_ss=np.array([0.0]),
        cw=cw_batched,
        z_sb=z_sb_batched,
        rp_sb=np.array([0.0]),
        cb=cb_batched,
        rhob=rhob_batched,
        attn=attn_batched,
        rbzb=rbzb_batched,
        dz=dz_common,
        dr=DR,
        rmax=RMAX,
        np=NP_PADE,
        ns=1,
        batch_size=8
    )
    
    cupyram_results_batched = sim_batched.run()
    tl_cupyram_batched = cupyram_results_batched['TL Line'][0, :]  # Extract first ray
    
    print(f"  Batched output shape: {cupyram_results_batched['TL Line'].shape}")
    print(f"  Using first ray from batch: shape {tl_cupyram_batched.shape}")
    
    # Quick comparison: batched vs non-batched
    diff_batched = np.abs(tl_cupyram - tl_cupyram_batched)
    print(f"\nBatched vs Non-batched CuPyRAM:")
    print(f"  Max diff: {np.max(diff_batched):.6f} dB")
    print(f"  RMS diff: {np.sqrt(np.mean(diff_batched**2)):.6f} dB")
    
    if np.max(diff_batched) < 0.01:
        print(f"  ✓ Batched and non-batched results are numerically identical")
    elif np.max(diff_batched) < 1.0:
        print(f"  ✓ Batched and non-batched results match within tolerance")
    
    print("\n✓ CuPyRAM batched simulation completed successfully")
    
    # =========================================================================
    # STEP 4: COMPARISON
    # =========================================================================
    print_section("Step 4: Comparison - CuPyRAM vs CPU")
    
    ranges_m = np.arange(DR, RMAX + DR, DR)
    ranges_km = ranges_m / 1000.0
    
    stats_cpu = compare_results(tl_cupyram, tl_cpu, "CuPyRAM", "CPU")
    passed_cpu = print_comparison(
        stats_cpu, "CuPyRAM", "CPU", TOLERANCES['phase3_cpu'],
        show_samples=True, ranges=ranges_km, result1=tl_cupyram, result2=tl_cpu
    )
    
    # =========================================================================
    # STEP 4B: COMPARISON - CUPYRAM BATCHED vs CPU
    # =========================================================================
    print_section("Step 4b: Comparison - CuPyRAM Batched vs CPU")
    
    stats_batched_cpu = compare_results(tl_cupyram_batched, tl_cpu, "CuPyRAM Batched", "CPU")
    passed_batched_cpu = print_comparison(
        stats_batched_cpu, "CuPyRAM Batched", "CPU", TOLERANCES['phase3_cpu'],
        show_samples=True, ranges=ranges_km, result1=tl_cupyram_batched, result2=tl_cpu
    )
    
    # =========================================================================
    # STEP 5: SAVE & PLOT
    # =========================================================================
    print_section("Step 5: Save Results")
    
    metadata = {
        'test': 'Phase 3 - ASA Wedge',
        'freq': FREQ,
        'depth_start': DEPTH_START,
        'depth_min': DEPTH_MIN,
        'slope_angle': slope_angle,
        'rmax': RMAX,
        'dr': DR,
        'dz': dz_common,
        'np_pade': NP_PADE
    }
    
    save_test_results('phase3_cupyram.npy', tl_cupyram, metadata)
    save_test_results('phase3_cpu.npy', tl_cpu, metadata)
    save_test_results('phase3_bathymetry.npy', np.column_stack((r_bathy, z_bathy)), metadata)
    
    print_section("Step 6: Generate Plots")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(ranges_km, tl_cpu, 'b-', linewidth=2, label='CPU PyRAM', alpha=0.7)
    ax1.plot(ranges_km, tl_cupyram, 'r--', linewidth=2, label='CuPyRAM', alpha=0.7)
    ax1.plot(ranges_km, tl_cupyram_batched, 'g:', linewidth=2, label='CuPyRAM Batched', alpha=0.7)
    ax1.set_xlabel('Range (km)', fontsize=12)
    ax1.set_ylabel('Transmission Loss (dB)', fontsize=12)
    ax1.set_title('Phase 3: ASA Wedge', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    ax1.axvline(x=RANGE_WEDGE/1000, color='gray', linestyle=':', alpha=0.5)
    
    ax2.plot(r_bathy/1000, z_bathy, 'k-', linewidth=2)
    ax2.axhline(y=Z_SOURCE, color='red', linestyle='--', linewidth=1, label='Source')
    ax2.axhline(y=Z_RECEIVER, color='blue', linestyle='--', linewidth=1, label='Receiver')
    ax2.set_xlabel('Range (km)', fontsize=12)
    ax2.set_ylabel('Depth (m)', fontsize=12)
    ax2.set_title('Bathymetry Profile', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase3_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Plot saved: phase3_comparison.png")
    plt.close()
    
    plot_comparison(
        ranges_km,
        {'CuPyRAM': tl_cupyram, 'CuPyRAM Batched': tl_cupyram_batched, 'CPU PyRAM': tl_cpu},
        'Phase 3: Wedge - Detailed Comparison',
        'phase3_comparison_detail.png',
        show_diff=True
    )
    
    # =========================================================================
    # ASSERTION 2: RESULTS MATCH REFERENCE (ACCURACY)
    # =========================================================================
    print_section("PHASE 3 FINAL VERDICT")
    
    print("\nAccuracy Test Results:")
    print_test_result(passed_cpu, f"CuPyRAM vs CPU: max diff = {stats_cpu['max_abs_diff']:.4f} dB")
    print_test_result(passed_batched_cpu, f"CuPyRAM Batched vs CPU: max diff = {stats_batched_cpu['max_abs_diff']:.4f} dB")
    
    # Pytest assertions for accuracy
    assert passed_cpu, (
        f"CuPyRAM vs CPU accuracy test failed: "
        f"max difference {stats_cpu['max_abs_diff']:.4f} dB exceeds "
        f"tolerance {TOLERANCES['phase3_cpu']} dB"
    )
    
    assert passed_batched_cpu, (
        f"CuPyRAM Batched vs CPU accuracy test failed: "
        f"max difference {stats_batched_cpu['max_abs_diff']:.4f} dB exceeds "
        f"tolerance {TOLERANCES['phase3_cpu']} dB"
    )
    
    print(f"\n{'='*70}")
    print("✓ PHASE 3 PASSED")
    print("  → Energy conservation validated")
    print("  → Bathymetry update logic correct")
    print("  → Interface handling validated")
    print("  → Batched execution validated (batch_size=8)")
    print("  → Ready for Phase 4")
    print(f"{'='*70}")

