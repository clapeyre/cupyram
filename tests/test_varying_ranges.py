"""
Test CuPyRAM with per-ray environments that have varying ranges.

This tests the real-world scenario where different rays sample different
bathymetry/environment ranges (e.g., Cap Corse profiling with rays at 
different angles).
"""

import numpy as np
import pytest
from cupyram import CuPyRAM


def test_varying_range_lengths():
    """
    Test batch with per-ray environments where each ray has different range sampling.
    
    This is a common real-world case: rays at different angles traverse different
    distances and sample environments at different ranges.
    """
    freq = 50.0
    zs = 100.0
    zr = 50.0
    batch_size = 3
    
    # Create per-ray environments with DIFFERENT range lengths
    # Ray 0: 3 range points
    # Ray 1: 5 range points  
    # Ray 2: 4 range points
    
    # Water profiles (varying ranges)
    z_ss_list = [
        np.array([0.0, 100.0, 200.0]),  # Ray 0
        np.array([0.0, 100.0, 200.0]),  # Ray 1
        np.array([0.0, 100.0, 200.0]),  # Ray 2
    ]
    
    rp_ss_list = [
        np.array([0.0, 5000.0, 10000.0]),       # Ray 0: 3 ranges
        np.array([0.0, 2000.0, 5000.0, 8000.0, 12000.0]),  # Ray 1: 5 ranges
        np.array([0.0, 3000.0, 7000.0, 11000.0]),  # Ray 2: 4 ranges
    ]
    
    cw_list = [
        np.full((3, 3), 1500.0),     # Ray 0: 3x3
        np.full((3, 5), 1500.0),     # Ray 1: 3x5 (different!)
        np.full((3, 4), 1500.0),     # Ray 2: 3x4 (different!)
    ]
    
    # Seabed profiles (varying ranges)
    z_sb_list = [
        np.array([0.0, 50.0]),
        np.array([0.0, 50.0]),
        np.array([0.0, 50.0]),
    ]
    
    rp_sb_list = [
        np.array([0.0, 5000.0, 10000.0]),       # Ray 0: 3 ranges
        np.array([0.0, 2000.0, 5000.0, 8000.0, 12000.0]),  # Ray 1: 5 ranges
        np.array([0.0, 3000.0, 7000.0, 11000.0]),  # Ray 2: 4 ranges
    ]
    
    cb_list = [
        np.full((2, 3), 1600.0),     # Ray 0: 2x3
        np.full((2, 5), 1600.0),     # Ray 1: 2x5 (different!)
        np.full((2, 4), 1600.0),     # Ray 2: 2x4 (different!)
    ]
    
    rhob_list = [
        np.full((2, 3), 1.5),
        np.full((2, 5), 1.5),
        np.full((2, 4), 1.5),
    ]
    
    attn_list = [
        np.full((2, 3), 0.5),
        np.full((2, 5), 0.5),
        np.full((2, 4), 0.5),
    ]
    
    # Bathymetry (varying ranges)
    rbzb_list = [
        np.array([[0.0, 200.0], [10000.0, 200.0]]),     # Ray 0
        np.array([[0.0, 200.0], [12000.0, 200.0]]),     # Ray 1
        np.array([[0.0, 200.0], [11000.0, 200.0]]),     # Ray 2
    ]
    
    # This should NOT raise ValueError about inhomogeneous shape
    try:
        model = CuPyRAM(
            freq=freq, zs=zs, zr=zr,
            z_ss=z_ss_list,
            rp_ss=rp_ss_list,  # Different lengths!
            cw=cw_list,
            z_sb=z_sb_list,
            rp_sb=rp_sb_list,  # Different lengths!
            cb=cb_list,
            rhob=rhob_list,
            attn=attn_list,
            rbzb=rbzb_list,
            rmax=10000.0,
            dr=50.0,
            dz=2.0,
            batch_size=batch_size
        )
        
        # Run the model
        result = model.run()
        
        # Verify batched output
        assert result['TL Line'].shape[0] == batch_size
        assert result['TL Grid'].shape[0] == batch_size
        
        print(f"✓ Successfully handled {batch_size} rays with varying range lengths")
        print(f"  Ray 0: {len(rp_ss_list[0])} ranges")
        print(f"  Ray 1: {len(rp_ss_list[1])} ranges")
        print(f"  Ray 2: {len(rp_ss_list[2])} ranges")
        
    except ValueError as e:
        if "inhomogeneous" in str(e):
            pytest.fail(f"CuPyRAM failed to handle varying range lengths: {e}")
        else:
            raise


def test_single_ray_converted_to_batch():
    """
    Test that a single ray (non-list inputs) is correctly converted to batch format.
    
    This ensures backward compatibility with PyRAM API.
    """
    freq = 50.0
    zs = 100.0
    zr = 50.0
    
    # Single environment (NOT lists)
    z_ss = np.array([0.0, 100.0, 200.0])
    rp_ss = np.array([0.0, 5000.0, 10000.0])
    cw = np.full((3, 3), 1500.0)
    
    z_sb = np.array([0.0, 50.0])
    rp_sb = np.array([0.0, 5000.0, 10000.0])
    cb = np.full((2, 3), 1600.0)
    rhob = np.full((2, 3), 1.5)
    attn = np.full((2, 3), 0.5)
    rbzb = np.array([[0.0, 200.0], [10000.0, 200.0]])
    
    # Should work with batch_size=1 (default)
    model = CuPyRAM(
        freq=freq, zs=zs, zr=zr,
        z_ss=z_ss, rp_ss=rp_ss, cw=cw,
        z_sb=z_sb, rp_sb=rp_sb, cb=cb,
        rhob=rhob, attn=attn, rbzb=rbzb,
        rmax=10000.0, dr=50.0, dz=2.0
    )
    
    result = model.run()
    
    # Output should be squeezed (no batch dimension for batch_size=1)
    assert result['TL Line'].ndim == 1, "batch_size=1 should return 1D output"
    assert result['TL Grid'].ndim == 2, "batch_size=1 should return 2D grid"
    
    print("✓ Single ray correctly converted to internal batch format and squeezed on output")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Test 1: Varying Range Lengths (Real-world Cap Corse scenario)")
    print("="*70)
    test_varying_range_lengths()
    
    print("\n" + "="*70)
    print("Test 2: Single Ray Backward Compatibility")
    print("="*70)
    test_single_ray_converted_to_batch()
    
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)


