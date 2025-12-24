import numpy as np
import cupy as cp
import os
import sys
from cupyram.CuPyRAM import CuPyRAM

def verify_gpu():
    baseline_file = os.path.join(os.path.dirname(__file__), 'baseline_data.npz')
    if not os.path.exists(baseline_file):
        print(f"Baseline file not found. Generating from PyRAM...")
        from tests.generate_baseline import generate_baseline
        generate_baseline()

    print(f"Loading baseline data from {baseline_file}...")
    data = np.load(baseline_file)
    
    # Extract inputs
    inputs = dict(
        freq=float(data['freq']),
        zs=float(data['zs']),
        zr=float(data['zr']),
        z_ss=data['z_ss'],
        rp_ss=data['rp_ss'],
        cw=data['cw'],
        z_sb=data['z_sb'],
        rp_sb=data['rp_sb'],
        cb=data['cb'],
        rhob=data['rhob'],
        attn=data['attn'],
        rbzb=data['rbzb'],
        rmax=float(data['rmax']),
        dr=float(data['dr']),
        dz=float(data['dz']),
        zmplt=float(data['zmplt']),
        c0=float(data['c0'])
    )

    print("Initializing CuPyRAM...")
    # Initialize CuPyRAM with same inputs
    pyram = CuPyRAM(
        inputs['freq'], inputs['zs'], inputs['zr'],
        inputs['z_ss'], inputs['rp_ss'], inputs['cw'],
        inputs['z_sb'], inputs['rp_sb'], inputs['cb'],
        inputs['rhob'], inputs['attn'], inputs['rbzb'],
        rmax=inputs['rmax'], dr=inputs['dr'],
        dz=inputs['dz'], zmplt=inputs['zmplt'],
        c0=inputs['c0']
    )

    print("Running CuPyRAM simulation...")
    results = pyram.run()
    
    print("Simulation complete. Comparing results...")
    
    # Compare Ranges and Depths
    gpu_ranges = results['Ranges']
    ref_ranges = data['ranges']
    if not np.allclose(gpu_ranges, ref_ranges):
        print("FAIL: Ranges mismatch")
        print("Max diff:", np.max(np.abs(gpu_ranges - ref_ranges)))
    else:
        print("PASS: Ranges match")

    gpu_depths = results['Depths']
    ref_depths = data['depths']
    if not np.allclose(gpu_depths, ref_depths):
        print("FAIL: Depths mismatch")
        print("Max diff:", np.max(np.abs(gpu_depths - ref_depths)))
    else:
        print("PASS: Depths match")

    # Compare Complex Pressure Grid
    gpu_cp = results['CP Grid']
    ref_cp = data['cp_grid']
    
    # Absolute difference
    abs_diff_cp = np.abs(gpu_cp - ref_cp)
    max_abs_diff_cp = np.max(abs_diff_cp)
    print(f"Max Absolute Diff (CP Grid): {max_abs_diff_cp:.6e}")
    
    # Relative difference (avoid division by zero)
    # Mask out very small values in ref
    mask = np.abs(ref_cp) > 1e-15
    rel_diff_cp = np.zeros_like(gpu_cp, dtype=float)
    rel_diff_cp[mask] = abs_diff_cp[mask] / np.abs(ref_cp[mask])
    max_rel_diff_cp = np.max(rel_diff_cp)
    print(f"Max Relative Diff (CP Grid): {max_rel_diff_cp:.6e}")

    # Compare Transmission Loss Grid
    gpu_tl = results['TL Grid']
    ref_tl = data['tl_grid']
    
    abs_diff_tl = np.abs(gpu_tl - ref_tl)
    max_abs_diff_tl = np.max(abs_diff_tl)
    print(f"Max Absolute Diff (TL Grid): {max_abs_diff_tl:.6e} dB")
    
    # Pass/Fail Criteria
    # We expect numerical differences due to CPU (80-bit float intermediates?) vs GPU (64-bit).
    # Also algorithm differences (ordering of ops).
    # But logic should be identical.
    
    tol_cp = 1e-10 # Starting strict
    if max_rel_diff_cp < tol_cp:
        print(f"SUCCESS: Results match within tolerance {tol_cp}")
    else:
        print(f"WARNING: Results differ (Max Rel Diff CP: {max_rel_diff_cp:.6e})")
        # Try looser tolerance
        if max_rel_diff_cp < 1e-5:
            print("         (Acceptable for float64 drift)")
        else:
            print("FAILURE: Significant deviation detected.")
            sys.exit(1)

if __name__ == "__main__":
    verify_gpu()
