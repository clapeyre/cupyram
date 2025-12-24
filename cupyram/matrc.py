"""
CUDA kernels for matrc - Memory-optimized with interleaved Padé computation
Arrays are [Nz+2, Batch] for coalesced memory access

MIXED PRECISION SUPPORT:
- Environment arrays (f1, f2, f3, alpw, alpb, etc.): float32
- Intermediate matrices (r1-r3, s1-s3, ksq): complex64
- Solution vectors (u, v) remain complex128 (handled in solve.py)
- Reduces memory usage by ~50% with minimal accuracy impact
"""

from numba import cuda
import cupy
import nvtx

@cuda.jit
def init_profiles_kernel(iz_arr, jz_arr, nz, f1, f2, f3, ksq, 
                         alpw, alpb, ksqw, ksqb, rhob, batch_size):
    """
    KERNEL 1: Environment Setup
    Parallelism: Batch x Depth
    Arrays: [Nz+2, Batch] for coalesced access
    """
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    total_points = (nz + 2) * batch_size
    
    for idx in range(tid, total_points, stride):
        # Decode index: arrays are [Nz+2, Batch] -> Fastest dim is Batch
        # Adjacent threads (varying b) access adjacent memory
        b = idx % batch_size
        i = idx // batch_size
        
        # Per-ray parameters
        iz = iz_arr[b]
        jz = jz_arr[b]
        
        # Logic matches original, but parallel per point
        val_f1 = 0.0
        val_f2 = 0.0
        val_f3 = 0.0
        val_ksq = 0.0j
        
        if iz == jz:
            if i <= iz:
                val_f1 = 1.0 / alpw[i, b]
                val_f2 = 1.0
                val_f3 = alpw[i, b]
                val_ksq = ksqw[i, b]
            else:
                val_f1 = rhob[i, b] / alpb[i, b]
                val_f2 = 1.0 / rhob[i, b]
                val_f3 = alpb[i, b]
                val_ksq = ksqb[i, b]
        elif iz > jz:
            if i > jz and i <= iz:
                val_f1 = 1.0 / alpw[i, b]
                val_f2 = 1.0
                val_f3 = alpw[i, b]
                val_ksq = ksqw[i, b]
        else: # iz < jz
            if i > iz and i <= jz:
                val_f1 = rhob[i, b] / alpb[i, b]
                val_f2 = 1.0 / rhob[i, b]
                val_f3 = alpb[i, b]
                val_ksq = ksqb[i, b]
                
        # Write to global memory - coalesced access [i, b]
        # Adjacent threads (varying b) write to adjacent memory locations
        f1[i, b] = val_f1
        f2[i, b] = val_f2
        f3[i, b] = val_f3
        ksq[i, b] = val_ksq


@cuda.jit
def discretize_kernel_single(k0_arr, dz, iz_arr, jz_arr, nz,
                              f1, f2, f3, ksq,
                              r1, r2, r3, s1, s2, s3, pd1_vals, pd2_vals, batch_size):
    """
    KERNEL 2: Galerkin Discretization for single Padé step
    Parallelism: Batch x Depth
    Arrays: [Nz+2, Batch] for coalesced access
    pd1_vals, pd2_vals: [Batch] - Padé coefficients for current j
    """
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    total_threads = batch_size * (nz + 2)
    
    # Constants
    cfact = 0.5 / (dz * dz)
    dfact = 1.0 / 12.0
    
    for idx in range(tid, total_threads, stride):
        # Decode Indices - arrays are [Nz+2, Batch]
        b = idx % batch_size
        i = idx // batch_size
        
        # Bounds Check
        if i < 1 or i > nz:
            continue

        # Per-ray parameters
        iz = iz_arr[b]
        jz = jz_arr[b]
        
        # Determine valid range
        i1 = 1
        i2 = nz
        if iz > jz:
            i1 = jz
            i2 = iz + 1
        elif iz < jz:
            i1 = iz
            i2 = jz + 1
            
        if i < i1 or i > i2:
            continue
            
        # Heavy Math - Galerkin discretization
        c1 = cfact * f1[i, b] * (f2[i - 1, b] + f2[i, b]) * f3[i - 1, b]
        c2 = -cfact * f1[i, b] * (f2[i - 1, b] + 2.0 * f2[i, b] + f2[i + 1, b]) * f3[i, b]
        c3 = cfact * f1[i, b] * (f2[i, b] + f2[i + 1, b]) * f3[i + 1, b]
        
        k_prev = ksq[i - 1, b]
        k_curr = ksq[i, b]
        k_next = ksq[i + 1, b]
        
        d1 = c1 + dfact * (k_prev + k_curr)
        d2 = c2 + dfact * (k_prev + 6.0 * k_curr + k_next)
        d3 = c3 + dfact * (k_curr + k_next)
        
        k0 = k0_arr[b]
        a1 = k0 * k0 / 6.0
        a2 = 2.0 * k0 * k0 / 3.0
        # a3 = a1
        
        pd1_val = pd1_vals[b]
        pd2_val = pd2_vals[b]
        
        # Coalesced Write - arrays are [Nz+2, Batch]
        # Adjacent threads (varying b) write to adjacent memory
        r1[i, b] = a1 + pd2_val * d1
        r2[i, b] = a2 + pd2_val * d2
        r3[i, b] = a1 + pd2_val * d3
        s1[i, b] = a1 + pd1_val * d1
        s2[i, b] = a2 + pd1_val * d2
        s3[i, b] = a1 + pd1_val * d3


@cuda.jit
def decompose_kernel_single(iz_arr, jz_arr, nz,
                             r1, r2, r3, s1, s2, s3, batch_size):
    """
    KERNEL 3: Matrix Decomposition for single Padé step (Gaussian Elimination)
    Parallelism: Batch (one thread per ray)
    Constraint: Serial in Depth (i) due to dependencies r1[i] depends on r1[i-1]
    Arrays: [Nz+2, Batch] for coalesced access
    """
    b = cuda.grid(1)
    if b >= batch_size:
        return
    
    iz = iz_arr[b]
    jz = jz_arr[b]
    
    # Logic setup
    i1 = 1
    i2 = nz
    if iz > jz:
        i1 = jz
        i2 = iz + 1
    elif iz < jz:
        i1 = iz
        i2 = jz + 1
        
    # Forward pass (Dependency chain)
    for i in range(i1, iz + 1):
        rfact = 1.0 / (r2[i, b] - r1[i, b] * r3[i - 1, b])
        r1[i, b] *= rfact
        r3[i, b] *= rfact
        s1[i, b] *= rfact
        s2[i, b] *= rfact
        s3[i, b] *= rfact
    
    # Backward pass (Dependency chain)
    for i in range(i2, iz + 1, -1):
        rfact = 1.0 / (r2[i, b] - r3[i, b] * r1[i + 1, b])
        r1[i, b] *= rfact
        r3[i, b] *= rfact
        s1[i, b] *= rfact
        s2[i, b] *= rfact
        s3[i, b] *= rfact
        
    # Final update
    val = r2[iz + 1, b]
    val -= r1[iz + 1, b] * r3[iz, b]
    val -= r3[iz + 1, b] * r1[iz + 2, b]
    r2[iz + 1, b] = 1.0 / val


def matrc_cuda_init_profiles(iz, jz, nz, f1, f2, f3, ksq, alpw, alpb, ksqw, ksqb,
                              rhob, batch_size=1):
    """
    Initialize environment profiles (called once per range step, outside Padé loop).
    Arrays: [Nz+2, Batch] for coalesced memory access.
    
    Note: iz and jz must be CuPy arrays (CuPyRAM is GPU-only).
    """
    # In CuPyRAM context, iz and jz are always CuPy arrays
    assert isinstance(iz, cupy.ndarray), "iz must be CuPy array"
    assert isinstance(jz, cupy.ndarray), "jz must be CuPy array"

    # Init Profiles (Occupancy: Batch * Nz)
    total_threads = batch_size * (nz + 2)
    block_size = 256
    grid_size = (total_threads + block_size - 1) // block_size
    init_profiles_kernel[grid_size, block_size](
        iz, jz, nz, f1, f2, f3, ksq, alpw, alpb, ksqw, ksqb, rhob, batch_size
    )


def matrc_cuda_single_pade(k0, dz, iz, jz, nz, f1, f2, f3, ksq,
                            r1, r2, r3, s1, s2, s3, pd1_vals, pd2_vals, batch_size=1):
    """
    Compute matrices for single Padé step (called np times per range step).
    Arrays: [Nz+2, Batch] for coalesced memory access.
    pd1_vals, pd2_vals: [Batch] - Padé coefficients for current j.
    
    Note: iz and jz must be CuPy arrays (CuPyRAM is GPU-only).
    """
    # In CuPyRAM context, iz and jz are always CuPy arrays
    assert isinstance(iz, cupy.ndarray), "iz must be CuPy array"
    assert isinstance(jz, cupy.ndarray), "jz must be CuPy array"

    block_size = 256
    
    # 1. Discretize (Occupancy: Batch * Nz)
    total_threads_discretize = batch_size * (nz + 2)
    grid_size_discretize = (total_threads_discretize + block_size - 1) // block_size
    discretize_kernel_single[grid_size_discretize, block_size](
        k0, dz, iz, jz, nz,
        f1, f2, f3, ksq, r1, r2, r3, s1, s2, s3, pd1_vals, pd2_vals, batch_size
    )
    
    # 2. Decompose (Occupancy: Batch)
    total_threads_decompose = batch_size
    grid_size_decompose = (total_threads_decompose + block_size - 1) // block_size
    decompose_kernel_single[grid_size_decompose, block_size](
        iz, jz, nz, r1, r2, r3, s1, s2, s3, batch_size
    )