#!/usr/bin/env python3
"""
Phase 2: Range-Independent Verification - Munk Profile

Tests refraction physics and range marching with depth-dependent sound speed.
Environment: Munk canonical profile (deep water with sound channel).

Success criteria:
- GPU vs CPU < 0.1 dB → Refraction physics validated
- No range drift → Padé coefficients correct
"""

import pytest
import numpy as np
from cupyram import CuPyRAM
from .test_utils import (SimulationConfig, run_pyram_reference, compare_results,
                        print_comparison, plot_comparison, save_test_results,
                        print_section, print_test_result, TOLERANCES)
import matplotlib.pyplot as plt

# =============================================================================
# MUNK PROFILE DEFINITION
# =============================================================================
def munk_profile(z, c0=1500.0, epsilon=0.00737, z_axis=1000.0, B=1300.0):
    """
    Munk canonical sound speed profile for deep ocean.
    
    c(z) = c0 * [1 + ε(η + exp(-η) - 1)]
    where η = 2(z - z_axis) / B
    
    Args:
        z: Depth array (m)
        c0: Reference sound speed at axis (m/s)
        epsilon: Perturbation parameter
        z_axis: Sound channel axis depth (m)
        B: Scale depth (m)
    
    Returns:
        Sound speed array (m/s)
    """
    eta = 2.0 * (z - z_axis) / B
    c = c0 * (1.0 + epsilon * (eta + np.exp(-eta) - 1.0))
    return c


# Test configuration
DEPTH = 5000.0
C0 = 1500.0
FREQ = 25.0
Z_SOURCE = 1000.0
Z_RECEIVER = 1000.0
RMAX = 50000.0
DR = 50.0
NP_PADE = 8

# Munk profile parameters
EPSILON = 0.00737
Z_AXIS = 1000.0
B = 1300.0


@pytest.mark.munk
@pytest.mark.slow
def test_munk_profile(output_dir):
    """
    Phase 2: Munk Profile Test
    
    Validates refraction physics with depth-dependent sound speed.
    Tests Padé coefficients by checking for range drift.
    """
    
    config = SimulationConfig(
        name="Phase 2: Munk Profile",
        freq=FREQ,
        zs=Z_SOURCE,
        zr=Z_RECEIVER,
        rmax=RMAX,
        dr=DR,
        depth=DEPTH,
        c0=C0,
        tolerance=TOLERANCES['phase2_cpu']
    )
    
    print_section("PHASE 2: RANGE-INDEPENDENT VERIFICATION - MUNK PROFILE")
    print(config)
    print(f"\nMunk Profile Parameters:")
    print(f"  c0: {C0} m/s")
    print(f"  ε: {EPSILON}")
    print(f"  z_axis: {Z_AXIS} m (sound channel axis)")
    print(f"  B: {B} m (scale depth)")
    
    # =========================================================================
    # STEP 1: GENERATE MUNK PROFILE
    # =========================================================================
    print_section("Step 1: Generate Munk Profile")
    
    # Calculate default dz
    dz_common = 0.1 * 1500 / FREQ
    
    # Create depth grid
    zmax_target = 6000.0
    nz_target = int(np.floor(zmax_target / dz_common)) - 1
    
    z_grid = np.arange(0, (nz_target + 2) * dz_common, dz_common)
    c_munk = munk_profile(z_grid, C0, EPSILON, Z_AXIS, B)
    
    print(f"\nProfile Configuration:")
    print(f"  dz: {dz_common:.4f} m")
    print(f"  nz: {nz_target}")
    print(f"  zmax: {(nz_target + 2) * dz_common:.2f} m")
    print(f"  Grid points: {len(z_grid)}")
    print(f"  Sound speed range: [{np.min(c_munk):.2f}, {np.max(c_munk):.2f}] m/s")
    print(f"  Sound speed at axis ({Z_AXIS}m): {c_munk[int(Z_AXIS/dz_common)]:.2f} m/s")
    
    # =========================================================================
    # STEP 2: CPU PYRAM REFERENCE
    # =========================================================================
    print_section("Step 2: CPU PyRAM Reference")
    
    # Setup PyRAM with Munk profile
    z_ss_fine = np.linspace(0, DEPTH, 100)
    c_ss_fine = munk_profile(z_ss_fine, C0, EPSILON, Z_AXIS, B)
    cw = c_ss_fine.reshape(-1, 1)
    
    # Seabed parameters
    z_sb = np.array([DEPTH, DEPTH + 100])
    cb = np.array([[1700.0], [1700.0]])
    rhob = np.array([[1.8], [1.8]])
    attn = np.array([[0.5], [0.5]])
    
    # Flat bathymetry
    rbzb = np.array([[0.0, DEPTH], [RMAX, DEPTH]])
    
    # Calculate default dz
    dz_common = 0.1 * 1500 / FREQ
    
    print(f"Using common dz: {dz_common:.4f} m")
    print(f"Sound speed profile: {len(z_ss_fine)} points from 0 to {DEPTH} m")
    
    # Run PyRAM
    tl_cpu, pyram = run_pyram_reference(
        config, z_ss_fine, cw, z_sb, cb, rhob, attn, rbzb,
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
    print(f"  Bathymetry: flat {DEPTH} m")
    print(f"  Using dz: {dz_common:.4f} m (matching CPU)")
    
    # Create fresh copies of arrays for CuPyRAM
    z_ss_cupyram = np.linspace(0, DEPTH, 100)
    c_ss_cupyram = munk_profile(z_ss_cupyram, C0, EPSILON, Z_AXIS, B)
    cw_cupyram = c_ss_cupyram.reshape(-1, 1)
    
    z_sb_cupyram = np.array([DEPTH, DEPTH + 100])
    cb_cupyram = np.array([[1700.0], [1700.0]])
    rhob_cupyram = np.array([[1.8], [1.8]])
    attn_cupyram = np.array([[0.5], [0.5]])
    rbzb_cupyram = np.array([[0.0, DEPTH], [RMAX, DEPTH]])
    
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
    assert np.min(tl_cupyram) > 0, f"TL values should be positive, got min={np.min(tl_cupyram):.2f} dB"
    assert np.max(tl_cupyram) < 200, f"TL values should be < 200 dB, got max={np.max(tl_cupyram):.2f} dB"
    print(f"  ✓ TL range reasonable: [{np.min(tl_cupyram):.2f}, {np.max(tl_cupyram):.2f}] dB")
    
    print(f"\nCuPyRAM Grid Configuration:")
    print(f"  nz: {sim.nz}")
    print(f"  dz: {sim._dz:.4f} m")
    print(f"  zmax: {sim._zmax:.2f} m")
    
    # Verify grids match
    assert sim.nz == cpu_nz, f"CuPyRAM and CPU grids must be identical! CuPyRAM nz={sim.nz}, CPU nz={cpu_nz}"
    print(f"  ✓ Grid matches CPU: nz={sim.nz}")
    
    print("\n✓ CuPyRAM simulation completed successfully")
    
    # =========================================================================
    # STEP 3B: CUPYRAM BATCHED SIMULATION
    # =========================================================================
    print_section("Step 3b: CuPyRAM Batched Simulation (batch_size=16)")
    
    print(f"\nRunning CuPyRAM with batch_size=16...")
    print(f"  Testing batched execution with complex SSP (Munk profile)")
    
    # Create fresh copies of arrays for batched CuPyRAM
    z_ss_batched = np.linspace(0, DEPTH, 100)
    c_ss_batched = munk_profile(z_ss_batched, C0, EPSILON, Z_AXIS, B)
    cw_batched = c_ss_batched.reshape(-1, 1)
    
    z_sb_batched = np.array([DEPTH, DEPTH + 100])
    cb_batched = np.array([[1700.0], [1700.0]])
    rhob_batched = np.array([[1.8], [1.8]])
    attn_batched = np.array([[0.5], [0.5]])
    rbzb_batched = np.array([[0.0, DEPTH], [RMAX, DEPTH]])
    
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
        batch_size=16
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
    # STEP 4: COMPARISON - CUPYRAM vs CPU
    # =========================================================================
    print_section("Step 4: Comparison - CuPyRAM vs CPU")
    
    ranges_m = np.arange(DR, RMAX + DR, DR)
    ranges_km = ranges_m / 1000.0
    
    stats_cpu = compare_results(tl_cupyram, tl_cpu, "CuPyRAM", "CPU")
    passed_cpu = print_comparison(
        stats_cpu, "CuPyRAM", "CPU", TOLERANCES['phase2_cpu'],
        show_samples=True, ranges=ranges_km, result1=tl_cupyram, result2=tl_cpu
    )
    
    # =========================================================================
    # STEP 4B: COMPARISON - CUPYRAM BATCHED vs CPU
    # =========================================================================
    print_section("Step 4b: Comparison - CuPyRAM Batched vs CPU")
    
    stats_batched_cpu = compare_results(tl_cupyram_batched, tl_cpu, "CuPyRAM Batched", "CPU")
    passed_batched_cpu = print_comparison(
        stats_batched_cpu, "CuPyRAM Batched", "CPU", TOLERANCES['phase2_cpu'],
        show_samples=True, ranges=ranges_km, result1=tl_cupyram_batched, result2=tl_cpu
    )
    
    # Check for range drift
    min_len = min(len(tl_cupyram), len(tl_cpu))
    diff = tl_cupyram[:min_len] - tl_cpu[:min_len]
    diff_first_quarter = np.mean(np.abs(diff[:min_len//4]))
    diff_last_quarter = np.mean(np.abs(diff[3*min_len//4:]))
    drift = diff_last_quarter - diff_first_quarter
    
    print(f"\nRange Drift Analysis:")
    print(f"  First quarter mean |diff|: {diff_first_quarter:.4f} dB")
    print(f"  Last quarter mean |diff|: {diff_last_quarter:.4f} dB")
    print(f"  Drift: {drift:+.4f} dB")
    
    no_drift = abs(drift) < 0.5
    if no_drift:
        print(f"  ✓ No significant range drift detected")
    else:
        print(f"  ⚠ Range drift detected - may indicate Padé coefficient issue")
    
    # =========================================================================
    # STEP 5: SAVE RESULTS
    # =========================================================================
    print_section("Step 5: Save Results")
    
    metadata = {
        'test': 'Phase 2 - Munk Profile',
        'freq': FREQ,
        'depth': DEPTH,
        'z_source': Z_SOURCE,
        'z_receiver': Z_RECEIVER,
        'c0': C0,
        'munk_epsilon': EPSILON,
        'munk_z_axis': Z_AXIS,
        'munk_B': B,
        'rmax': RMAX,
        'dr': DR,
        'dz': dz_common,
        'np_pade': NP_PADE
    }
    
    save_test_results('phase2_cupyram.npy', tl_cupyram, metadata)
    save_test_results('phase2_cpu.npy', tl_cpu, metadata)
    
    # Also save the Munk profile
    z_profile = np.linspace(0, DEPTH, 100)
    c_profile = munk_profile(z_profile, C0, EPSILON, Z_AXIS, B)
    np.save(output_dir / 'phase2_munk_profile.npy', np.column_stack((z_profile, c_profile)))
    print(f"  Saved: phase2_munk_profile.npy")
    
    # =========================================================================
    # STEP 6: GENERATE PLOTS
    # =========================================================================
    print_section("Step 6: Generate Plots")
    
    # Comparison plot
    plot_comparison(
        ranges_km,
        {'CuPyRAM': tl_cupyram, 'CuPyRAM Batched': tl_cupyram_batched, 'CPU PyRAM': tl_cpu},
        'Phase 2: Munk Profile - CuPyRAM vs CPU',
        'phase2_comparison.png',
        show_diff=True
    )
    
    # Profile and TL combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sound speed profile
    ax1.plot(c_profile, z_profile, 'b-', linewidth=2)
    ax1.axhline(y=Z_SOURCE, color='red', linestyle='--', label=f'Source ({Z_SOURCE}m)')
    ax1.axhline(y=Z_AXIS, color='green', linestyle='--', label=f'Axis ({Z_AXIS}m)')
    ax1.set_xlabel('Sound Speed (m/s)', fontsize=12)
    ax1.set_ylabel('Depth (m)', fontsize=12)
    ax1.set_title('Munk Sound Speed Profile', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 3000])
    
    # TL comparison
    ax2.plot(ranges_km, tl_cpu, 'b-', linewidth=2, label='CPU PyRAM', alpha=0.7)
    ax2.plot(ranges_km, tl_cupyram, 'r--', linewidth=2, label='CuPyRAM', alpha=0.7)
    ax2.plot(ranges_km, tl_cupyram_batched, 'g:', linewidth=2, label='CuPyRAM Batched', alpha=0.7)
    ax2.set_xlabel('Range (km)', fontsize=12)
    ax2.set_ylabel('Transmission Loss (dB)', fontsize=12)
    ax2.set_title('Transmission Loss Comparison', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase2_profile_and_tl.png', dpi=150, bbox_inches='tight')
    print(f"  Plot saved: phase2_profile_and_tl.png")
    plt.close()
    
    # =========================================================================
    # ASSERTION 2: RESULTS MATCH REFERENCE (ACCURACY)
    # =========================================================================
    print_section("PHASE 2 FINAL VERDICT")
    
    print("\nAccuracy Test Results:")
    print_test_result(passed_cpu, f"CuPyRAM vs CPU: max diff = {stats_cpu['max_abs_diff']:.4f} dB")
    print_test_result(passed_batched_cpu, f"CuPyRAM Batched vs CPU: max diff = {stats_batched_cpu['max_abs_diff']:.4f} dB")
    print_test_result(no_drift, f"Range drift: {drift:+.4f} dB")
    
    # Pytest assertions for accuracy
    assert passed_cpu, (
        f"CuPyRAM vs CPU accuracy test failed: "
        f"max difference {stats_cpu['max_abs_diff']:.4f} dB exceeds "
        f"tolerance {TOLERANCES['phase2_cpu']} dB"
    )
    
    assert passed_batched_cpu, (
        f"CuPyRAM Batched vs CPU accuracy test failed: "
        f"max difference {stats_batched_cpu['max_abs_diff']:.4f} dB exceeds "
        f"tolerance {TOLERANCES['phase2_cpu']} dB"
    )
    
    assert no_drift, (
        f"Range drift detected: {drift:+.4f} dB "
        f"(difference between first and last quarters exceeds 0.5 dB). "
        f"This suggests Padé coefficients may be incorrect."
    )
    
    print(f"\n{'='*70}")
    print("✓ PHASE 2 PASSED")
    print("  → Refraction physics validated")
    print("  → Padé coefficients correct (no range drift)")
    print("  → Range marching algorithm validated")
    print("  → Batched execution validated (batch_size=16)")
    print("  → Ready for Phase 3")
    print(f"{'='*70}")

