#!/usr/bin/env python3
"""
Phase 4: Raster/Coordinate Integration

Tests bilinear interpolation with projected coordinates.
Environment: Synthetic raster where depth = f(x, y) with known analytic form.

Success criteria:
- Bilinear interpolation continuous (no artifacts)
- GPU vs CPU < 1.0 dB
- Depth sampling accurate along ray path
"""

import pytest
import numpy as np
from cupyram import CuPyRAM
from .test_utils import (SimulationConfig, run_pyram_reference, compare_results,
                        print_comparison, plot_comparison, save_test_results,
                        print_section, print_test_result, TOLERANCES)
import matplotlib.pyplot as plt

# Test configuration
FREQ = 50.0
Z_SOURCE = 50.0
Z_RECEIVER = 30.0
RMAX = 10000.0
DR = 50.0
NP_PADE = 8
C_WATER = 1500.0

# Raster configuration (projected coordinates in meters)
Y_MIN, Y_MAX = 0.0, 20000.0  # 20 km northward (enough for 10km propagation from center)
X_MIN, X_MAX = 0.0, 20000.0  # 20 km eastward
RASTER_SIZE = 200

DEPTH_BASE = 100.0
DEPTH_GRADIENT = 0.05  # depth increases 0.05m per meter northward (shallower gradient for 20km)


def synthetic_depth(y, x):
    """Analytic depth function: depth increases linearly with northing"""
    return DEPTH_BASE + (y - Y_MIN) * DEPTH_GRADIENT


@pytest.mark.raster
@pytest.mark.slow
@pytest.mark.skip(reason="GPU-specific raster interpolation not available in current CPU-based CuPyRAM implementation")
def test_raster_coordinate_integration(output_dir):
    """
    Phase 4: Raster/Coordinate Integration Test
    
    Validates bilinear interpolation with projected coordinates and synthetic raster.
    """
    
    config = SimulationConfig(
        name="Phase 4: Raster/Coordinate Integration",
        freq=FREQ,
        zs=Z_SOURCE,
        zr=Z_RECEIVER,
        rmax=RMAX,
        dr=DR,
        depth=200.0,
        c0=C_WATER,
        tolerance=TOLERANCES['phase4_cpu']
    )
    
    print_section("PHASE 4: RASTER/COORDINATE INTEGRATION")
    print(config)
    print(f"\nSynthetic Raster (Projected Coordinates):")
    print(f"  Y (northing): [{Y_MIN/1000:.1f}, {Y_MAX/1000:.1f}] km")
    print(f"  X (easting): [{X_MIN/1000:.1f}, {X_MAX/1000:.1f}] km")
    print(f"  Depth function: {DEPTH_BASE} + (y - {Y_MIN}) × {DEPTH_GRADIENT}")
    
    # =========================================================================
    # STEP 1: CREATE SYNTHETIC RASTER
    # =========================================================================
    print_section("Step 1: Create Synthetic Raster")
    
    y_grid = np.linspace(Y_MIN, Y_MAX, RASTER_SIZE)  # Northing
    x_grid = np.linspace(X_MIN, X_MAX, RASTER_SIZE)  # Easting
    
    bathy_raster = np.zeros((RASTER_SIZE, RASTER_SIZE), dtype=np.float32)
    for i, y in enumerate(y_grid):
        for j, x in enumerate(x_grid):
            bathy_raster[i, j] = synthetic_depth(y, x)
    
    print(f"\nRaster: {RASTER_SIZE}x{RASTER_SIZE}")
    print(f"Depth range: [{np.min(bathy_raster):.1f}, {np.max(bathy_raster):.1f}] m")
    
    # =========================================================================
    # STEP 2: CALCULATE EXPECTED DEPTHS
    # =========================================================================
    print_section("Step 2: Calculate Expected Depths")
    
    source_y = (Y_MIN + Y_MAX) / 2.0  # Center northing
    source_x = (X_MIN + X_MAX) / 2.0  # Center easting
    
    print(f"\nSource: (y={source_y:.1f} m, x={source_x:.1f} m)")
    print(f"Initial depth: {synthetic_depth(source_y, source_x):.1f} m")
    
    # Propagate northward (0° angle)
    angle_rad = 0.0
    dir_y = np.cos(angle_rad)  # North direction
    dir_x = np.sin(angle_rad)  # East direction
    
    num_steps = int(RMAX / DR)
    expected_y = np.zeros(num_steps)  # Northing coordinates
    expected_x = np.zeros(num_steps)  # Easting coordinates
    expected_depths = np.zeros(num_steps)
    
    current_y = source_y
    current_x = source_x
    
    # Projected coordinates: simple addition in meters
    for step in range(num_steps):
        current_y += DR * dir_y
        current_x += DR * dir_x
        
        expected_y[step] = current_y
        expected_x[step] = current_x
        expected_depths[step] = synthetic_depth(current_y, current_x)
    
    print(f"\nExpected ray path:")
    print(f"  Final y: {expected_y[-1]:.1f} m ({expected_y[-1]/1000:.2f} km)")
    print(f"  Final depth: {expected_depths[-1]:.1f} m")
    print(f"  Depth change: {expected_depths[-1] - expected_depths[0]:.1f} m")
    
    # =========================================================================
    # STEP 3: GPU SIMULATION
    # =========================================================================
    print_section("Step 3: GPU Numba Simulation")
    
    gpu_temp = CuPyRAM(freq=FREQ)
    dz_common = gpu_temp.dz
    
    max_depth = np.max(bathy_raster)
    zmax_base = max_depth + 500
    nz_gpu = int(np.floor(zmax_base / dz_common))
    zmax_gpu = (nz_gpu + 2) * dz_common
    
    sim = CuPyRAM(freq=FREQ)
    c_profile = np.full(nz_gpu, C_WATER, dtype=np.float64)
    profiles_gpu = c_profile.reshape(1, -1)
    type_raster = np.zeros((RASTER_SIZE, RASTER_SIZE), dtype=np.float32)
    
    tex_bounds = (X_MIN, Y_MIN, X_MAX, Y_MAX)  # (minx, miny, maxx, maxy)
    sources = np.array([[source_y, source_x]])  # [northing, easting]
    
    print(f"\nGPU Configuration:")
    print(f"  Source: (y={sources[0,0]:.1f} m, x={sources[0,1]:.1f} m)")
    print(f"  Texture bounds: x=[{X_MIN:.1f}, {X_MAX:.1f}] m, y=[{Y_MIN:.1f}, {Y_MAX:.1f}] m")
    print(f"  zmax: {zmax_gpu:.2f} m")
    print(f"  dz: {dz_common:.4f} m")
    print(f"  nz: {nz_gpu}")
    
    sim.prepare_arrays(bathy_raster, type_raster)
    
    print(f"\nRunning GPU simulation...")
    gpu_results = sim.run(
        sources_xy=sources,  # Projected coordinates (meters)
        profiles_table=profiles_gpu,
        rmax=RMAX,
        dr=DR,
        np_pade=NP_PADE,
        dz=dz_common,
        zs=Z_SOURCE,
        zr=Z_RECEIVER,
        tex_bounds=tex_bounds,
        zmax=zmax_gpu,
        lyrw=20,
        rs=RMAX + DR
    )
    
    # =========================================================================
    # ASSERTION 1: GPU SIMULATION COMPLETED SUCCESSFULLY
    # =========================================================================
    print_section("Verification: GPU Simulation Execution")
    
    # Check output shape
    expected_shape = (1, 32, int(RMAX / DR))
    assert gpu_results.shape == expected_shape, (
        f"GPU output shape mismatch: expected {expected_shape}, got {gpu_results.shape}"
    )
    print(f"  ✓ Output shape correct: {gpu_results.shape}")
    
    tl_gpu = gpu_results[0, 0, :]
    
    # Check for NaN/Inf values
    nan_count = np.sum(np.isnan(tl_gpu))
    inf_count = np.sum(np.isinf(tl_gpu))
    assert nan_count == 0, f"GPU results contain {nan_count} NaN values"
    assert inf_count == 0, f"GPU results contain {inf_count} Inf values"
    print(f"  ✓ No NaN or Inf values")
    
    # Check TL values are reasonable
    assert np.min(tl_gpu) > 0, f"TL values should be positive, got min={np.min(tl_gpu):.2f} dB"
    assert np.max(tl_gpu) < 200, f"TL values should be < 200 dB, got max={np.max(tl_gpu):.2f} dB"
    print(f"  ✓ TL range reasonable: [{np.min(tl_gpu):.2f}, {np.max(tl_gpu):.2f}] dB")
    
    print("\n✓ GPU simulation completed successfully")
    
    # =========================================================================
    # STEP 4: VERIFY COORDINATES
    # =========================================================================
    print_section("Step 4: Verify Coordinate Accuracy")
    
    final_in_bounds = (Y_MIN <= expected_y[-1] <= Y_MAX and 
                      X_MIN <= expected_x[-1] <= X_MAX)
    
    print(f"\nExpected final: (y={expected_y[-1]:.1f} m, x={expected_x[-1]:.1f} m)")
    print_test_result(final_in_bounds, "Final position in bounds")
    
    # Check interpolation smoothness
    tl_diff = np.diff(tl_gpu)
    tl_diff_smooth = np.abs(tl_diff) < 5.0
    smooth_percentage = np.sum(tl_diff_smooth) / len(tl_diff) * 100
    
    print(f"\nInterpolation Smoothness:")
    print(f"  Smooth steps: {smooth_percentage:.1f}%")
    print(f"  Max TL jump: {np.max(np.abs(tl_diff)):.2f} dB")
    
    interpolation_ok = smooth_percentage > 95.0
    print_test_result(interpolation_ok, "Bilinear interpolation smooth")
    
    # =========================================================================
    # STEP 5: CPU REFERENCE
    # =========================================================================
    print_section("Step 5: CPU PyRAM Reference")
    
    r_bathy = np.arange(0, RMAX + DR, DR)
    z_bathy_pyram = np.interp(r_bathy, np.arange(len(expected_depths)) * DR, expected_depths)
    
    z_ss = np.array([0.0, max_depth + 100])
    cw = np.array([[C_WATER], [C_WATER]])
    z_sb = np.array([max_depth, max_depth + 100])
    cb = np.array([[1700.0], [1700.0]])
    rhob = np.array([[1.8], [1.8]])
    attn = np.array([[0.5], [0.5]])
    rbzb = np.column_stack((r_bathy, z_bathy_pyram))
    
    print(f"Using dz: {dz_common:.4f} m")
    
    tl_cpu, pyram = run_pyram_reference(
        config, z_ss, cw, z_sb, cb, rhob, attn, rbzb,
        dz=dz_common, np_pade=NP_PADE, ns=1
    )
    
    # =========================================================================
    # STEP 6: COMPARISON
    # =========================================================================
    print_section("Step 6: Comparison - GPU vs CPU")
    
    ranges_m = np.arange(DR, RMAX + DR, DR)
    ranges_km = ranges_m / 1000.0
    
    stats_cpu = compare_results(tl_gpu, tl_cpu, "GPU", "CPU")
    passed_cpu = print_comparison(
        stats_cpu, "GPU", "CPU", TOLERANCES['phase4_cpu'],
        show_samples=True, ranges=ranges_km, result1=tl_gpu, result2=tl_cpu
    )
    
    # =========================================================================
    # STEP 7: SAVE & PLOT
    # =========================================================================
    print_section("Step 7: Save Results")
    
    metadata = {
        'test': 'Phase 4 - Raster Integration',
        'freq': FREQ,
        'y_range': [Y_MIN, Y_MAX],
        'x_range': [X_MIN, X_MAX],
        'depth_base': DEPTH_BASE,
        'depth_gradient': DEPTH_GRADIENT,
        'rmax': RMAX,
        'dr': DR,
        'dz': dz_common,
        'np_pade': NP_PADE,
        'coordinate_system': 'projected (meters)'
    }
    
    save_test_results('phase4_gpu.npy', tl_gpu, metadata)
    save_test_results('phase4_cpu.npy', tl_cpu, metadata)
    save_test_results('phase4_depths_expected.npy', expected_depths, metadata)
    
    np.save(output_dir / 'phase4_raster.npy', bathy_raster)
    np.save(output_dir / 'phase4_coordinates.npy', np.column_stack((expected_y, expected_x)))
    print(f"  Saved: phase4_raster.npy, phase4_coordinates.npy")
    
    print_section("Step 8: Generate Plots")
    
    plot_comparison(
        ranges_km,
        {'GPU Numba': tl_gpu, 'CPU PyRAM': tl_cpu},
        'Phase 4: Raster/Coordinate Integration',
        'phase4_comparison.png',
        show_diff=True
    )
    
    # Raster visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    im = ax1.imshow(bathy_raster, extent=[X_MIN/1000, X_MAX/1000, Y_MAX/1000, Y_MIN/1000],
                   aspect='auto', cmap='viridis')
    ax1.plot(expected_x/1000, expected_y/1000, 'r-', linewidth=2, label='Ray path')
    ax1.plot(source_x/1000, source_y/1000, 'wo', markersize=8, label='Source')
    ax1.set_xlabel('Easting (km)')
    ax1.set_ylabel('Northing (km)')
    ax1.set_title('Bathymetry Raster with Ray Path (Projected)', fontweight='bold')
    ax1.legend()
    plt.colorbar(im, ax=ax1, label='Depth (m)')
    
    ax2.plot(ranges_km, expected_depths, 'b-', linewidth=2)
    ax2.set_xlabel('Range (km)')
    ax2.set_ylabel('Depth (m)')
    ax2.set_title('Expected Depth Along Ray', fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(ranges_km, tl_cpu, 'b-', linewidth=2, label='CPU PyRAM', alpha=0.7)
    ax3.plot(ranges_km, tl_gpu, 'r--', linewidth=2, label='GPU Numba', alpha=0.7)
    ax3.set_xlabel('Range (km)')
    ax3.set_ylabel('Transmission Loss (dB)')
    ax3.set_title('TL Comparison', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    
    min_len = min(len(tl_gpu), len(tl_cpu))
    diff = tl_gpu[:min_len] - tl_cpu[:min_len]
    ax4.plot(ranges_km[:min_len], diff, 'k-', linewidth=1.5)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.fill_between(ranges_km[:min_len], -TOLERANCES['phase4_cpu'], 
                    TOLERANCES['phase4_cpu'], alpha=0.2, color='green',
                    label=f'Tolerance (±{TOLERANCES["phase4_cpu"]} dB)')
    ax4.set_xlabel('Range (km)')
    ax4.set_ylabel('Difference (dB)')
    ax4.set_title('GPU - CPU Difference', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase4_raster_visual.png', dpi=150, bbox_inches='tight')
    print(f"  Plot saved: phase4_raster_visual.png")
    plt.close()
    
    # =========================================================================
    # ASSERTION 2: RESULTS MATCH REFERENCE (ACCURACY)
    # =========================================================================
    print_section("PHASE 4 FINAL VERDICT")
    
    print("\nAccuracy Test Results:")
    print_test_result(final_in_bounds, "Ray path stays in bounds")
    print_test_result(interpolation_ok, f"Interpolation smooth ({smooth_percentage:.1f}%)")
    print_test_result(passed_cpu, f"GPU vs CPU: max diff = {stats_cpu['max_abs_diff']:.4f} dB")
    
    # Assertions for accuracy and correctness
    assert final_in_bounds, "Ray path went out of raster bounds - coordinate transforms incorrect"
    assert interpolation_ok, f"Interpolation artifacts detected - only {smooth_percentage:.1f}% smooth"
    assert passed_cpu, (
        f"GPU vs CPU accuracy test failed: "
        f"max difference {stats_cpu['max_abs_diff']:.4f} dB exceeds "
        f"tolerance {TOLERANCES['phase4_cpu']} dB"
    )
    
    print(f"\n{'='*70}")
    print("✓ PHASE 4 PASSED")
    print("  → Projected coordinate handling correct")
    print("  → Bilinear interpolation smooth")
    print("  → Raster sampling validated")
    print("  → All functional tests complete!")
    print(f"{'='*70}")

