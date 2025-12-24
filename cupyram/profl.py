"""
CUDA kernel for profl - Batched profile interpolation
"""

from numba import cuda
import math

@cuda.jit(device=True)
def binary_search(val, arr, count):
    """
    Device function to find index idx such that arr[idx] <= val < arr[idx+1]
    Assumes arr is monotonically increasing.
    """
    if count == 0:
        return 0
    
    # Handle out of bounds
    if val <= arr[0]:
        return 0
    if val >= arr[count - 1]:
        return count - 2  # Return second to last index for extrapolation/clamping

    low = 0
    high = count - 1
    
    while low < high:
        mid = (low + high) // 2
        if arr[mid] <= val:
            if arr[mid + 1] > val:
                return mid
            low = mid + 1
        else:
            high = mid
            
    return low

@cuda.jit(device=True)
def linear_interp(x, x0, x1, y0, y1):
    """Standard linear interpolation"""
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + (y1 - y0) * t

@cuda.jit
def profl_kernel(z_grid, z_ss_batch, cw_prof_batch, z_sb_batch, cb_prof_batch, 
                 rhob_prof_batch, attn_prof_batch,
                 # Outputs
                 cw_out, cb_out, rhob_out, attn_out,
                 # Parameters
                 valid_len_ss, valid_len_sb,
                 attnf, lyrw_lambda, batch_size, nz):
    """
    Batched interpolation kernel.
    Grid: (batch_size, threads_per_ray)
    Output arrays: [Nz+2, Batch] for coalesced memory access
    """
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    
    if b >= batch_size:
        return

    # Valid lengths for this ray
    n_ss = valid_len_ss[b]
    n_sb = valid_len_sb[b]
    
    # Pre-fetch absorbing layer parameters for this ray
    z_sb_last = z_sb_batch[b, n_sb - 1]
    attn_last = attn_prof_batch[b, n_sb - 1]
    z_lyr_1 = z_sb_last + 0.75 * lyrw_lambda[b]
    z_lyr_2 = z_sb_last + lyrw_lambda[b]

    # Iterate over depth grid points assigned to this thread
    for k in range(tid, nz + 2, stride):
        z = z_grid[k]
        
        # --- 1. Water Sound Speed (CW) ---
        # Binary search on z_ss
        idx = binary_search(z, z_ss_batch[b], n_ss)
        
        # Clamp/Interpolate
        # Output arrays are [Nz+2, Batch] - transposed for coalescing
        if z <= z_ss_batch[b, 0]:
            cw_out[k, b] = cw_prof_batch[b, 0]
        elif z >= z_ss_batch[b, n_ss - 1]:
            cw_out[k, b] = cw_prof_batch[b, n_ss - 1]
        else:
            cw_out[k, b] = linear_interp(z, 
                                         z_ss_batch[b, idx], z_ss_batch[b, idx+1],
                                         cw_prof_batch[b, idx], cw_prof_batch[b, idx+1])

        # --- 2. Seabed Parameters (CB, RHOB) ---
        # Binary search on z_sb
        idx_sb = binary_search(z, z_sb_batch[b], n_sb)
        
        # Helper lambda for simple profile interpolation
        # (Can't use actual lambda in Numba, duplicating logic slightly for speed)
        
        # CB
        if z <= z_sb_batch[b, 0]:
            cb_out[k, b] = cb_prof_batch[b, 0]
        elif z >= z_sb_batch[b, n_sb - 1]:
            cb_out[k, b] = cb_prof_batch[b, n_sb - 1]
        else:
            cb_out[k, b] = linear_interp(z, 
                                         z_sb_batch[b, idx_sb], z_sb_batch[b, idx_sb+1],
                                         cb_prof_batch[b, idx_sb], cb_prof_batch[b, idx_sb+1])
            
        # RHOB
        if z <= z_sb_batch[b, 0]:
            rhob_out[k, b] = rhob_prof_batch[b, 0]
        elif z >= z_sb_batch[b, n_sb - 1]:
            rhob_out[k, b] = rhob_prof_batch[b, n_sb - 1]
        else:
            rhob_out[k, b] = linear_interp(z, 
                                           z_sb_batch[b, idx_sb], z_sb_batch[b, idx_sb+1],
                                           rhob_prof_batch[b, idx_sb], rhob_prof_batch[b, idx_sb+1])
            
        # --- 3. Attenuation (ATTN) with Absorbing Layer ---
        # Logic: Normal profile -> Constant last value (75%) -> Linear ramp to attnf (25%)
        
        if z <= z_sb_last:
            # Normal profile interpolation
            if z <= z_sb_batch[b, 0]:
                attn_out[k, b] = attn_prof_batch[b, 0]
            else:
                attn_out[k, b] = linear_interp(z, 
                                               z_sb_batch[b, idx_sb], z_sb_batch[b, idx_sb+1],
                                               attn_prof_batch[b, idx_sb], attn_prof_batch[b, idx_sb+1])
        elif z <= z_lyr_1:
            # First 75% of absorbing layer: Constant last value
            attn_out[k, b] = attn_last
        elif z <= z_lyr_2:
            # Last 25% of absorbing layer: Ramp to attnf
            attn_out[k, b] = linear_interp(z, z_lyr_1, z_lyr_2, attn_last, attnf)
        else:
            # Beyond layer
            attn_out[k, b] = attnf

def profl_cuda_launcher(z_grid, z_ss, cw_prof, z_sb, cb_prof, rhob_prof, attn_prof,
                        cw_out, cb_out, rhob_out, attn_out,
                        lyrw_lambda, attnf=10.0):
    """
    Launcher for the batched profile interpolation.
    Output arrays: [Nz+2, Batch] for coalesced memory access
    """
    batch_size = cw_out.shape[1]  # Second dimension is batch
    nz = cw_out.shape[0] - 2  # First dimension is Nz+2
    
    # Calculate valid lengths (assuming NaNs are padded at the end)
    # This is fast on GPU
    import cupy
    valid_len_ss = cupy.sum(~cupy.isnan(z_ss), axis=1).astype(cupy.int32)
    valid_len_sb = cupy.sum(~cupy.isnan(z_sb), axis=1).astype(cupy.int32)
    
    threads_per_ray = 256
    blocks = batch_size
    
    profl_kernel[blocks, threads_per_ray](
        z_grid, z_ss, cw_prof, z_sb, cb_prof, rhob_prof, attn_prof,
        cw_out, cb_out, rhob_out, attn_out,
        valid_len_ss, valid_len_sb,
        attnf, lyrw_lambda, batch_size, nz
    )