"""
CUDA kernel for outpt - output transmission loss and complex pressure
Parallelized over batch dimension
"""

import cupy
from numba import cuda
import math


@cuda.jit
def outpt_kernel(r, mdr_in, ndr, ndz, tlc_in, f3, u, _dir, ir,
                 tll, tlg, cpl, cpg, mdr_out, tlc_out, batch_size, nvz):
    """
    CUDA kernel for computing transmission loss output.
    Arrays: [Nz+2, Batch] for coalesced memory access.
    
    Each thread handles one ray (batch element).
    Grid: batch_size threads
    """
    b = cuda.grid(1)
    
    if b >= batch_size:
        return
    
    eps = 1e-20
    
    # Local copy of counters (same for all threads)
    mdr = mdr_in
    tlc = tlc_in
    
    mdr += 1
    if mdr == ndr:
        mdr = 0
        tlc += 1
        
        # Compute complex pressure at receiver depth
        # Arrays are [Nz+2, Batch] - transposed indexing
        cpl[b, tlc] = (1.0 - _dir) * f3[ir, b] * u[ir, b] + \
                      _dir * f3[ir + 1, b] * u[ir + 1, b]
        
        # Compute transmission loss at receiver depth
        temp = 10.0 * math.log10(r + eps)
        abs_cpl = abs(cpl[b, tlc])
        tll[b, tlc] = -20.0 * math.log10(abs_cpl + eps) + temp
        
        # Compute transmission loss grid
        for i in range(nvz):
            j = (i + 1) * ndz
            # Arrays are [Nz+2, Batch] - transposed indexing
            cpg = u[j, b] * f3[j, b]
            abs_cpg = abs(cpg)
            tlg[b, i, tlc] = -20.0 * math.log10(abs_cpg + eps) + temp
    
    # Write counters (only first thread updates shared values)
    if b == 0:
        mdr_out[0] = mdr
        tlc_out[0] = tlc


def outpt_cuda(r, mdr, ndr, ndz, tlc, f3, u, _dir, ir, tll, tlg, cpl, cpg, batch_size=1):
    """
    GPU-accelerated output computation using CUDA.
    
    All arrays should be CuPy arrays on the device.
    Launches batch_size threads.
    
    Returns: [mdr, tlc] as numpy array (for compatibility)
    """
    nvz = tlg.shape[1] if tlg.ndim == 3 else tlg.shape[0]
    
    # Create device arrays for output counters
    mdr_out = cupy.zeros(1, dtype=cupy.int32)
    tlc_out = cupy.zeros(1, dtype=cupy.int32)
    
    # Launch configuration: 1D grid of batch_size threads
    threads_per_block = min(256, batch_size)
    blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
    
    outpt_kernel[blocks_per_grid, threads_per_block](
        r, mdr, ndr, ndz, tlc, f3, u, _dir, ir,
        tll, tlg, cpl, cpg, mdr_out, tlc_out, batch_size, nvz
    )
    
    # Copy counters back to host for return
    import numpy
    result = numpy.array([int(mdr_out.get().item()), int(tlc_out.get().item())], dtype=numpy.int64)
    return result

