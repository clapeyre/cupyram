"""
Padé Coefficient Computation for Parabolic Equation Solver

Computes rational approximation coefficients for the square root operator
in the parabolic equation marching algorithm.

Corresponds to PyRAM.epade() and supporting mathematical functions.
"""

import math
import numpy as np
import numba
from concurrent.futures import ThreadPoolExecutor


@numba.jit(nopython=True)
def _deriv(n, sig, alp, dg, dh1, dh2, dh3, _bin, nu):
    """
    Derivatives of the operator function at x=0
    Helper for Padé coefficient computation
    """
    dh1[0] = 0.5 * 1j * sig
    exp1 = -0.5
    dh2[0] = alp
    exp2 = -1
    dh3[0] = -2 * nu
    exp3 = -1
    for i in range(1, n):
        dh1[i] = dh1[i - 1] * exp1
        exp1 -= 1
        dh2[i] = dh2[i - 1] * exp2
        exp2 -= 1
        dh3[i] = -nu * dh3[i - 1] * exp3
        exp3 -= 1
    
    dg[0] = 1
    dg[1] = dh1[0] + dh2[0] + dh3[0]
    for i in range(1, n):
        dg[i + 1] = dh1[i] + dh2[i] + dh3[i]
        for j in range(i):
            dg[i + 1] += _bin[i, j] * (dh1[j] + dh2[j] + dh3[j]) * dg[i - j]
    
    return dg, dh1, dh2, dh3


@numba.jit(nopython=True)
def _pivot(n, i, a, b):
    """
    Pivot rows for numerical stability in Gaussian elimination
    """
    i0 = i
    amp0 = np.abs(a[i, i])
    for j in range(i + 1, n):
        amp = np.abs(a[j, i])
        if amp > amp0:
            i0 = j
            amp0 = amp
    
    if i0 != i:
        b[i0], b[i] = b[i], b[i0]
        for j in range(i, n + 1):
            a[i0, j], a[i, j] = a[i, j], a[i0, j]
    
    return a, b


@numba.jit(nopython=True, fastmath=False)
def _gauss(n, a, b):
    """
    Gaussian elimination with partial pivoting
    """
    # Downward elimination
    for i in range(n):
        if i < n - 1:
            a, b = _pivot(n, i, a, b)
        a[i, i] = 1 / a[i, i]
        b[i] *= a[i, i]
        if i < n - 1:
            for j in range(i + 1, n + 1):
                a[i, j] *= a[i, i]
            for k in range(i + 1, n):
                b[k] -= a[k, i] * b[i]
                for j in range(i + 1, n):
                    a[k, j] -= a[k, i] * a[i, j]
    
    # Back substitution
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            b[i] -= a[i, j] * b[j]
    
    return a, b


@numba.jit(nopython=True)
def _guerre(a, n, z, err, nter):
    """
    Find polynomial root using Laguerre's method
    """
    az = np.zeros(n, dtype=np.complex128)

    azz = np.zeros(n - 1, dtype=np.complex128)
    
    eps = 1e-20
    for i in range(n):
        az[i] = (i + 1) * a[i + 1]
    for i in range(n - 1):
        azz[i] = (i + 1) * az[i + 1]
    
    _iter = 0
    jter = 0
    dz = np.inf
    
    while (np.abs(dz) > err) and (_iter < nter - 1):
        p = a[n - 1] + a[n] * z
        for i in range(n - 2, -1, -1):
            p = a[i] + z * p
        if np.abs(p) < eps:
            return a, z, err
        
        pz = az[n - 2] + az[n - 1] * z
        for i in range(n - 3, -1, -1):
            pz = az[i] + z * pz
        
        pzz = azz[n - 3] + azz[n - 2] * z
        for i in range(n - 4, -1, -1):
            pzz = azz[i] + z * pzz
        
        f = pz / p
        g = f ** 2 - pzz / p
        h = np.sqrt((n - 1) * (n * g - f ** 2))
        amp1 = np.abs(f + h)
        amp2 = np.abs(f - h)
        if amp1 > amp2:
            dz = -n / (f + h)
        else:
            dz = -n / (f - h)
        
        _iter += 1
        
        jter += 1
        if jter == 9:
            jter = 0
            dz *= 1j
        z += dz
        
        if _iter == 100:
            raise ValueError('Laguerre method not converging')
    
    return a, z, err


@numba.jit(nopython=True)
def _fndrt(a, n, z):
    """
    Find all roots of a polynomial
    """
    if n == 1:
        z[0] = -a[0] / a[1]
        return a, z
    
    if n != 2:
        for k in range(n - 1, 1, -1):
            root = 0
            err = 1e-12
            a, root, err = _guerre(a, k + 1, root, err, 1000)
            err = 0
            a, root, err = _guerre(a, k + 1, root, err, 5)
            z[k] = root
            for i in range(k, -1, -1):
                a[i] += root * a[i + 1]
            for i in range(k + 1):
                a[i] = a[i + 1]
    
    z[1] = 0.5 * (-a[1] + np.sqrt(a[1] ** 2 - 4 * a[0] * a[2])) / a[2]
    z[0] = 0.5 * (-a[1] - np.sqrt(a[1] ** 2 - 4 * a[0] * a[2])) / a[2]
    
    return a, z


def compute_pade_coefficients(freq, c0, np_pade, ns=1, dr=50.0, ip=1):
    """
    Calculate Padé approximation coefficients for the parabolic equation
    
    Corresponds to PyRAM.epade() - computes rational approximation coefficients
    for the square root operator in the parabolic equation marching algorithm.
    
    Args:
        freq: Frequency (Hz)
        c0: Reference sound speed (m/s)
        np_pade: Number of Padé terms
        ns: Number of stability constraints (0, 1, or 2)
        dr: Range step (meters)
        ip: Initial profile flag (1=marching, 2=starter)
    
    Returns:
        pd1, pd2: Complex coefficient arrays for the Padé approximation
    """
    n = 2 * np_pade
    _bin = np.zeros((n + 1, n + 1))
    a = np.zeros((n + 1, n + 1), dtype=np.complex128)
    b = np.zeros(n, dtype=np.complex128)
    dg = np.zeros(n + 1, dtype=np.complex128)
    dh1 = np.zeros(n, dtype=np.complex128)
    dh2 = np.zeros(n, dtype=np.complex128)
    dh3 = np.zeros(n, dtype=np.complex128)
    fact = np.zeros(n + 1)
    
    k0 = 2.0 * math.pi * freq / c0
    sig = k0 * dr
    
    if ip == 1:
        nu, alp = 0, 0
    else:
        nu, alp = 1, -0.25
    
    # Factorials
    fact[0] = 1
    for i in range(1, n):
        fact[i] = (i + 1) * fact[i - 1]
    
    # Binomial coefficients
    for i in range(n + 1):
        _bin[i, 0] = 1
        _bin[i, i] = 1
    for i in range(2, n + 1):
        for j in range(1, i):
            _bin[i, j] = _bin[i - 1, j - 1] + _bin[i - 1, j]
    
    # Accuracy constraints
    dg, dh1, dh2, dh3 = _deriv(n, sig, alp, dg, dh1, dh2, dh3, _bin, nu)
    for i in range(n):
        b[i] = dg[i + 1]
    for i in range(n):
        if 2 * i <= n - 1:
            a[i, 2 * i] = fact[i]
        for j in range(i + 1):
            if 2 * j + 1 <= n - 1:
                a[i, 2 * j + 1] = -_bin[i + 1, j + 1] * fact[j] * dg[i - j]
    
    # Stability constraints
    if ns >= 1:
        z1 = -3 + 0j
        b[n - 1] = -1
        for j in range(np_pade):
            a[n - 1, 2 * j] = z1 ** (j + 1)
            a[n - 1, 2 * j + 1] = 0
    
    if ns >= 2:
        z1 = -1.5 + 0j
        b[n - 2] = -1
        for j in range(np_pade):
            a[n - 2, 2 * j] = z1 ** (j + 1)
            a[n - 2, 2 * j + 1] = 0
    
    a, b = _gauss(n, a, b)
    
    dh1[0] = 1
    for j in range(np_pade):
        dh1[j + 1] = b[2 * j]
    dh1, dh2 = _fndrt(dh1, np_pade, dh2)
    pd1 = np.zeros(np_pade, dtype=np.complex128)
    for j in range(np_pade):
        pd1[j] = -1 / dh2[j]
    
    dh1[0] = 1
    for j in range(np_pade):
        dh1[j + 1] = b[2 * j + 1]
    dh1, dh2 = _fndrt(dh1, np_pade, dh2)
    pd2 = np.zeros(np_pade, dtype=np.complex128)
    for j in range(np_pade):
        pd2[j] = -1 / dh2[j]
    
    return pd1, pd2


def compute_pade_coefficients_batch(freq, c0_array, np_pade, ns, dr, ip, max_workers=8):
    """
    Compute Padé coefficients for multiple rays in parallel.
    
    Embarrassingly parallel - each ray's Padé computation is independent.
    Uses ThreadPoolExecutor for CPU parallelization (Padé is CPU-bound).
    
    Args:
        freq: Frequency (Hz) - scalar, same for all rays
        c0_array: Reference sound speeds (m/s) - array [batch_size]
        np_pade: Number of Padé terms - scalar, same for all rays
        ns: Number of stability constraints - scalar, same for all rays
        dr: Range step (meters) - scalar, same for all rays
        ip: Initial profile flag (1=marching, 2=starter) - scalar, same for all rays
        max_workers: Maximum number of CPU threads (default: 8)
    
    Returns:
        pd1, pd2: Complex coefficient arrays [np_pade, batch_size]
    """
    batch_size = len(c0_array)
    
    # Allocate output arrays
    pd1_batch = np.zeros((np_pade, batch_size), dtype=np.complex128)
    pd2_batch = np.zeros((np_pade, batch_size), dtype=np.complex128)
    
    # Parallel computation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(compute_pade_coefficients,
                          freq=freq, c0=c0_array[b], 
                          np_pade=np_pade, ns=ns, 
                          dr=dr, ip=ip)
            for b in range(batch_size)
        ]
        # Collect results
        for b, future in enumerate(futures):
            pd1_b, pd2_b = future.result()
            pd1_batch[:, b] = pd1_b
            pd2_batch[:, b] = pd2_b
    
    return pd1_batch, pd2_batch

