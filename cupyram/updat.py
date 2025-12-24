"""
CUDA kernel for parallel index updates in CuPyRAM.updat()

Replaces the sequential Python loops for bathymetry, sound speed, and seabed index updates
with a parallel GPU kernel.
"""

import numpy
from numba import cuda
import cupy
import math


@cuda.jit(device=True)
def find_valid_length_1d(arr, max_len):
    """
    Find the valid (non-NaN) length of a 1D array.
    Returns the index of the first NaN, or max_len if no NaN found.
    """
    for i in range(max_len):
        if math.isnan(arr[i]):
            return i
    return max_len


@cuda.jit(device=True)
def find_valid_length_2d(arr, max_len):
    """
    Find the valid (non-NaN) length of a 2D array (Nx2 for bathymetry).
    Returns the index of the first row where first column is NaN.
    """
    for i in range(max_len):
        if math.isnan(arr[i, 0]):
            return i
    return max_len


@cuda.jit
def updat_indices_kernel(
    r, dr, dz, nz_max,
    # Bathymetry: rbzb[batch, max_npt, 2], bt_ind[batch], iz[batch]
    rbzb, bt_ind, iz,
    # Sound speed: rp_ss[batch, max_npt], ss_ind[batch]
    rp_ss, ss_ind,
    # Seabed: rp_sb[batch, max_npt], sb_ind[batch]
    rp_sb, sb_ind,
    # Flags
    rd_bt, rd_ss, rd_sb,
    # Output: need_matrc[batch] - set to 1 if this ray needs matrc update
    need_matrc,
    batch_size
):
    """
    CUDA kernel to update indices for bathymetry, sound speed, and seabed profiles.
    
    Each thread handles one ray (batch element).
    """
    b = cuda.grid(1)
    if b >= batch_size:
        return
    
    need_update = 0  # Local flag for this ray
    
    # ===================================================================
    # 1. Varying Bathymetry (rd_bt)
    # ===================================================================
    if rd_bt:
        # Find valid length of rbzb for this ray
        max_npt_bt = rbzb.shape[1]
        npt_bt = find_valid_length_2d(rbzb[b], max_npt_bt)
        
        if npt_bt >= 2:  # Need at least 2 points
            # Linear search forward
            while (bt_ind[b] < npt_bt - 1) and (r >= rbzb[b, bt_ind[b] + 1, 0]):
                bt_ind[b] += 1
            
            # Store old iz for comparison
            jz = iz[b]
            
            # Interpolate depth at current range
            r_current = r + 0.5 * dr
            r0 = rbzb[b, bt_ind[b], 0]
            r1 = rbzb[b, bt_ind[b] + 1, 0]
            z0 = rbzb[b, bt_ind[b], 1]
            z1 = rbzb[b, bt_ind[b] + 1, 1]
            
            z = z0 + (r_current - r0) * (z1 - z0) / (r1 - r0)
            
            # Compute new iz
            iz_new = int(math.floor(z / dz))
            iz_new = max(1, iz_new)
            iz_new = min(nz_max - 1, iz_new)
            iz[b] = iz_new
            
            # Check if changed
            if iz_new != jz:
                need_update = 1
    
    # ===================================================================
    # 2. Varying Sound Speed Profile (rd_ss)
    # ===================================================================
    if rd_ss:
        # Find valid length of rp_ss for this ray
        max_npt_ss = rp_ss.shape[1]
        npt_ss = find_valid_length_1d(rp_ss[b], max_npt_ss)
        
        if npt_ss >= 2:
            # Store old index for comparison
            ss_ind_o = ss_ind[b]
            
            # Linear search forward
            while (ss_ind[b] < npt_ss - 1) and (r >= rp_ss[b, ss_ind[b] + 1]):
                ss_ind[b] += 1
            
            # Check if changed
            if ss_ind[b] != ss_ind_o:
                need_update = 1
    
    # ===================================================================
    # 3. Varying Seabed Profile (rd_sb)
    # ===================================================================
    if rd_sb:
        # Find valid length of rp_sb for this ray
        max_npt_sb = rp_sb.shape[1]
        npt_sb = find_valid_length_1d(rp_sb[b], max_npt_sb)
        
        if npt_sb >= 2:
            # Store old index for comparison
            sb_ind_o = sb_ind[b]
            
            # Linear search forward
            while (sb_ind[b] < npt_sb - 1) and (r >= rp_sb[b, sb_ind[b] + 1]):
                sb_ind[b] += 1
            
            # Check if changed
            if sb_ind[b] != sb_ind_o:
                need_update = 1
    
    # Store result
    need_matrc[b] = need_update


def updat_indices_cuda(
    r, dr, dz, nz,
    rbzb, bt_ind, iz,
    rp_ss, ss_ind,
    rp_sb, sb_ind,
    rd_bt, rd_ss, rd_sb,
    batch_size
):
    """
    Host-side launcher for the updat indices kernel.
    
    Args:
        r: Current range (scalar float)
        dr, dz: Range and depth steps (scalar floats)
        nz: Maximum depth index (scalar int)
        rbzb: Bathymetry array [batch, max_npt, 2] (CuPy)
        bt_ind: Bathymetry indices [batch] (CuPy, int)
        iz: Depth indices [batch] (CuPy, int)
        rp_ss: Sound speed range points [batch, max_npt] (CuPy)
        ss_ind: Sound speed indices [batch] (CuPy, int)
        rp_sb: Seabed range points [batch, max_npt] (CuPy)
        sb_ind: Seabed indices [batch] (CuPy, int)
        rd_bt, rd_ss, rd_sb: Flags (booleans)
        batch_size: Number of rays (int)
    
    Returns:
        need_matrc: Boolean indicating if any ray needs matrc update
    """
    # Allocate output array
    need_matrc_arr = cupy.zeros(batch_size, dtype=cupy.int32)
    
    # Launch kernel with one thread per ray
    threads_per_block = 256
    blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
    
    updat_indices_kernel[blocks_per_grid, threads_per_block](
        r, dr, dz, nz,
        rbzb, bt_ind, iz,
        rp_ss, ss_ind,
        rp_sb, sb_ind,
        rd_bt, rd_ss, rd_sb,
        need_matrc_arr,
        batch_size
    )
    
    # Check if any ray needs matrc (reduction on GPU)
    need_matrc = bool(cupy.any(need_matrc_arr))
    
    return need_matrc

