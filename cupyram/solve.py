"""
GPU version of solve function using Numba CUDA

Direct 1:1 port of the CPU version - exact same sequential algorithm.
No parallelization within solve itself.

The parallelization strategy is at a higher level: CuPyRAM will process
batches of rays in parallel, with each ray running its own sequential solve.
"""

import numpy
import cupy
import nvtx
from numba import cuda, int64, complex128
from contextlib import nullcontext


@cuda.jit
def solve_kernel_single(u, v, s1, s2, s3, r1, r2, r3, iz, nz):
    """
    CUDA kernel for single ray - single Padé step.
    Arrays: [Nz+2, 1] for consistency with batched version.
    """
    
    eps = complex128(1e-30)
    
    # Single Padé step (no j loop)
    # The right side
    for i in range(1, nz + 1):
        v[i, 0] = s1[i, 0] * u[i - 1, 0] + s2[i, 0] * u[i, 0] + s3[i, 0] * u[i + 1, 0] + eps
    
    # The elimination steps
    for i in range(2, iz + 1):
        v[i, 0] -= r1[i, 0] * v[i - 1, 0] + eps
    for i in range(nz - 1, iz + 1, -1):
        v[i, 0] -= r3[i, 0] * v[i + 1, 0] + eps
    
    u[iz + 1, 0] = (v[iz + 1, 0] - r1[iz + 1, 0] * v[iz, 0] - r3[iz + 1, 0] * v[iz + 2, 0]) * \
        r2[iz + 1, 0] + eps
    
    # The back substitution steps
    for i in range(iz, -1, -1):
        u[i, 0] = v[i, 0] - r3[i, 0] * u[i + 1, 0] + eps
    for i in range(iz + 2, nz + 1):
        u[i, 0] = v[i, 0] - r1[i, 0] * u[i - 1, 0] + eps


@cuda.jit
def solve_kernel_batched(u, v, s1, s2, s3, r1, r2, r3, iz_array, nz):
    """
    CUDA kernel for batched rays - single Padé step.
    Each thread processes one ray (batch index b).
    Arrays: [Nz+2, Batch] for coalesced memory access.
    
    Args:
        u, v: [Nz+2, Batch] - solution and workspace arrays
        s1, s2, s3, r1, r2, r3: [Nz+2, Batch] - matrix coefficients
        iz_array: int64[batch] or int64[1] - per-ray bathymetry index
        nz: int - number of depth grid points - 2
    """
    b = cuda.grid(1)  # Batch index
    
    # Early exit if beyond batch size
    if b >= u.shape[1]:  # u is now [Nz+2, Batch]
        return
    
    # Get iz for this ray
    if iz_array.shape[0] == 1:
        # Shared iz for all rays
        iz = iz_array[0]
    else:
        # Per-ray iz
        iz = iz_array[b]
    
    eps = complex128(1e-30)
    
    # Single Padé step (no j loop - called once per Padé coefficient)
    # The right side
    for i in range(1, nz + 1):
        v[i, b] = s1[i, b] * u[i - 1, b] + s2[i, b] * u[i, b] + s3[i, b] * u[i + 1, b] + eps
    
    # The elimination steps
    for i in range(2, iz + 1):
        v[i, b] -= r1[i, b] * v[i - 1, b] + eps
    for i in range(nz - 1, iz + 1, -1):
        v[i, b] -= r3[i, b] * v[i + 1, b] + eps
    
    u[iz + 1, b] = (v[iz + 1, b] - r1[iz + 1, b] * v[iz, b] - r3[iz + 1, b] * v[iz + 2, b]) * \
        r2[iz + 1, b] + eps
    
    # The back substitution steps
    for i in range(iz, -1, -1):
        u[i, b] = v[i, b] - r3[i, b] * u[i + 1, b] + eps
    for i in range(iz + 2, nz + 1):
        u[i, b] = v[i, b] - r1[i, b] * u[i - 1, b] + eps


def solve(u, v, s1, s2, s3, r1, r2, r3, iz, nz):
    """
    GPU version of tridiagonal solver - single Padé step.
    
    Uses MIXED PRECISION: solution vectors (u, v) in double precision (complex128),
    matrix coefficients (s1-s3, r1-r3) in single precision (complex64).
    Arrays: [Nz+2, Batch] for coalesced memory access.
    
    All arrays are CuPy arrays on GPU, wrapped for Numba CUDA inside this function.
    
    Args:
        u: complex128[nz+2, batch] - solution vector (DOUBLE PRECISION, CuPy on GPU)
        v: complex128[nz+2, batch] - temporary workspace (DOUBLE PRECISION, CuPy on GPU)
        s1, s2, s3: complex64[nz+2, batch] - matrix coefficients (SINGLE PRECISION, CuPy on GPU)
        r1, r2, r3: complex64[nz+2, batch] - matrix coefficients (SINGLE PRECISION, CuPy on GPU)
        iz: int64 or array[batch] - bathymetry index (CuPy on GPU)
        nz: int64 - grid parameters
    """
    
    # Wrap CuPy arrays for Numba CUDA (zero-copy)
    with nvtx.annotate("solve_array_setup", color="yellow"):
        u_device = cuda.as_cuda_array(u)
        v_device = cuda.as_cuda_array(v)
        s1_device = cuda.as_cuda_array(s1)
        s2_device = cuda.as_cuda_array(s2)
        s3_device = cuda.as_cuda_array(s3)
        r1_device = cuda.as_cuda_array(r1)
        r2_device = cuda.as_cuda_array(r2)
        r3_device = cuda.as_cuda_array(r3)
        iz_device = cuda.as_cuda_array(iz)
    
    # Detect batch size from array dimensions
    # Arrays are [Nz+2, Batch], so batch_size is in dimension 1
    batch_size = u_device.shape[1]
    threads_per_block = min(256, batch_size)
    blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
    
    # Run batched tridiagonal solver kernel
    with nvtx.annotate("solve_kernel_batched", color="green"):
        solve_kernel_batched[blocks_per_grid, threads_per_block](
            u_device, v_device,
            s1_device, s2_device, s3_device,
            r1_device, r2_device, r3_device,
            iz_device, nz
        )
    
    # Result stays on GPU - caller uses device_arrays['u'] and syncs when needed
