#!/usr/bin/env python3
"""
Analytic solution for the Pekeris waveguide using normal mode theory.

The Pekeris waveguide consists of:
- Isovelocity water layer (depth H, sound speed c1, density rho1)
- Isovelocity bottom half-space (sound speed c2, density rho2, attenuation)

Reference: 
- Jensen, Kuperman, Porter, Schmidt "Computational Ocean Acoustics" (2011), Ch. 5
- Pekeris, C.L. (1948) "Theory of propagation of explosive sound in shallow water"
"""

import numpy as np
from scipy.optimize import fsolve, brentq
from scipy.special import hankel2
import warnings

def find_mode_wavenumbers(freq, depth, c_water, c_bottom, rho_water=1.0, rho_bottom=1.5, 
                         attn_bottom=0.5, max_modes=50):
    """
    Find the horizontal wavenumbers (eigenvalues) for modes in a Pekeris waveguide.
    
    The characteristic equation for Pekeris waveguide modes is:
    tan(gamma * H) = -gamma * rho2 / (kappa * rho1)
    
    where:
    - gamma = sqrt(k1^2 - kr^2)  (vertical wavenumber in water)
    - kappa = sqrt(kr^2 - k2^2)  (vertical decay in bottom)
    - k1 = omega/c1 (wavenumber in water)
    - k2 = omega/c2 (wavenumber in bottom)
    - kr is the horizontal wavenumber we're solving for
    
    Args:
        freq: Frequency (Hz)
        depth: Water depth (m)
        c_water: Sound speed in water (m/s)
        c_bottom: Sound speed in bottom (m/s)
        rho_water: Density in water (g/cm³)
        rho_bottom: Density in bottom (g/cm³)
        attn_bottom: Attenuation in bottom (dB/wavelength)
        max_modes: Maximum number of modes to find
    
    Returns:
        Array of horizontal wavenumbers kr for each mode
    """
    omega = 2 * np.pi * freq
    k1 = omega / c_water  # Water wavenumber
    
    # Include attenuation in bottom wavenumber (makes it complex)
    eta = 1.0 / (40.0 * np.pi * np.log10(np.exp(1.0)))
    k2 = (omega / c_bottom) * (1.0 + 1j * eta * attn_bottom)
    
    # Mode wavenumbers must be between k2 and k1 (for propagating modes)
    # For attenuating bottom, use real parts
    k2_real = np.real(k2)
    
    def characteristic_eqn(kr):
        """Characteristic equation for Pekeris modes"""
        gamma = np.sqrt(k1**2 - kr**2 + 0j)  # Vertical wavenumber in water
        kappa = np.sqrt(kr**2 - k2**2 + 0j)  # Vertical wavenumber in bottom
        
        # Avoid division by zero
        if np.abs(kappa) < 1e-10:
            return 1e10
        
        # Characteristic equation: tan(gamma*H) + gamma*rho2/(kappa*rho1) = 0
        lhs = np.tan(gamma * depth)
        rhs = -gamma * rho_bottom / (kappa * rho_water)
        
        return np.real(lhs - rhs)
    
    # Find modes by searching for zeros between k2 and k1
    # Estimate number of modes
    n_modes_est = int(np.ceil(k1 * depth / np.pi)) + 2
    n_modes_est = min(n_modes_est, max_modes)
    
    kr_modes = []
    
    # Search for each mode separately
    for n in range(n_modes_est):
        # For mode n, gamma*depth is approximately (n+1)*pi
        # So kr ≈ sqrt(k1^2 - ((n+1)*pi/depth)^2)
        gamma_approx = (n + 1) * np.pi / depth
        kr_guess_sq = k1**2 - gamma_approx**2
        
        if kr_guess_sq <= k2_real**2:
            # Mode cutoff reached
            break
        
        kr_guess = np.sqrt(kr_guess_sq)
        
        # Refine using fsolve
        try:
            result = fsolve(characteristic_eqn, kr_guess, full_output=True)
            kr_mode = result[0][0]
            info = result[1]
            
            # Check if converged and in valid range
            if info['fvec'][0]**2 < 1e-10 and k2_real < kr_mode < k1:
                # Check for duplicates
                is_duplicate = any(np.abs(kr_mode - km) < 1e-6 for km in kr_modes)
                if not is_duplicate:
                    kr_modes.append(kr_mode)
        except:
            pass
    
    # Sort by kr (descending - mode 1 has highest kr)
    kr_modes = np.array(sorted(kr_modes, reverse=True))
    
    return kr_modes


def compute_mode_shapes(kr_modes, depth, z_source, z_receiver, k1, dz=0.1):
    """
    Compute the mode shapes (vertical eigenfunctions) at source and receiver depths.
    
    For Pekeris waveguide, mode shapes in water layer are:
    phi_n(z) = sin(gamma_n * z)
    
    where gamma_n = sqrt(k1^2 - kr_n^2)
    
    Args:
        kr_modes: Array of horizontal wavenumbers
        depth: Water depth (m)
        z_source: Source depth (m)
        z_receiver: Receiver depth (m)
        k1: Water wavenumber (2*pi*f/c1)
        dz: Depth resolution for normalization (m)
    
    Returns:
        Tuple of (phi_source, phi_receiver, gamma_modes)
        where phi_source[n] is mode n at source depth
    """
    n_modes = len(kr_modes)
    phi_source = np.zeros(n_modes, dtype=np.complex128)
    phi_receiver = np.zeros(n_modes, dtype=np.complex128)
    gamma_modes = np.zeros(n_modes, dtype=np.complex128)
    
    for n, kr in enumerate(kr_modes):
        gamma = np.sqrt(k1**2 - kr**2 + 0j)
        gamma_modes[n] = gamma
        
        # Mode shape (not normalized yet)
        phi_s = np.sin(gamma * z_source)
        phi_r = np.sin(gamma * z_receiver)
        
        # Normalization: integral of |phi|^2 over depth should equal depth
        # For sin(gamma*z), integral from 0 to H is H/2
        # So normalization factor is sqrt(2/H)
        norm = np.sqrt(2.0 / depth)
        
        phi_source[n] = phi_s * norm
        phi_receiver[n] = phi_r * norm
    
    return phi_source, phi_receiver, gamma_modes


def pekeris_solution(ranges, freq, depth, z_source, z_receiver,
                    c_water=1500.0, c_bottom=1700.0, 
                    rho_water=1.0, rho_bottom=1.5, 
                    attn_bottom=0.5, max_modes=50):
    """
    Compute transmission loss for a Pekeris waveguide using normal mode theory.
    
    The pressure field is:
    p(r, z) = sum_n A_n * phi_n(zs) * phi_n(zr) * H0(kr_n * r)
    
    where H0 is the Hankel function of the second kind (outgoing wave),
    and A_n are amplitude coefficients.
    
    Args:
        ranges: Array of ranges (m)
        freq: Frequency (Hz)
        depth: Water depth (m)
        z_source: Source depth (m)
        z_receiver: Receiver depth (m)
        c_water: Sound speed in water (m/s)
        c_bottom: Sound speed in bottom (m/s)
        rho_water: Density in water (g/cm³)
        rho_bottom: Density in bottom (g/cm³)
        attn_bottom: Attenuation in bottom (dB/wavelength)
        max_modes: Maximum number of modes to compute
    
    Returns:
        Transmission loss array (dB) at each range
    """
    omega = 2 * np.pi * freq
    k1 = omega / c_water
    
    # Find mode wavenumbers
    kr_modes = find_mode_wavenumbers(freq, depth, c_water, c_bottom, 
                                     rho_water, rho_bottom, attn_bottom, max_modes)
    
    if len(kr_modes) == 0:
        raise ValueError("No propagating modes found")
    
    print(f"  Found {len(kr_modes)} propagating modes")
    
    # Compute mode shapes at source and receiver
    phi_s, phi_r, gamma_modes = compute_mode_shapes(kr_modes, depth, z_source, 
                                                     z_receiver, k1)
    
    # Compute pressure field at each range
    pressure = np.zeros(len(ranges), dtype=np.complex128)
    
    # Suppress warnings about Hankel function at r=0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for i, r in enumerate(ranges):
            if r < 1.0:  # Avoid singularity at r=0
                r = 1.0
            
            # Sum over modes
            p_total = 0.0
            for n in range(len(kr_modes)):
                kr = kr_modes[n]
                
                # Hankel function H0^(2)(kr * r)
                # For large arguments, use asymptotic form to avoid overflow
                arg = kr * r
                if arg > 100:
                    # Asymptotic form: H0(z) ~ sqrt(2/(pi*z)) * exp(-i*(z - pi/4))
                    h0 = np.sqrt(2.0 / (np.pi * arg)) * np.exp(-1j * (arg - np.pi/4))
                else:
                    h0 = hankel2(0, arg)
                
                # Mode amplitude coefficient (from source normalization)
                # A_n = -i * k0 / (4 * gamma_n)
                A_n = -1j * k1 / (4.0 * gamma_modes[n])
                
                # Add contribution from this mode
                p_total += A_n * phi_s[n] * phi_r[n] * h0
            
            pressure[i] = p_total
    
    # Convert to transmission loss
    # TL = -20 * log10(|p|) + 10 * log10(r)
    # The cylindrical spreading factor is already in Hankel function,
    # but we normalize to 1m reference
    tl = -20.0 * np.log10(np.abs(pressure) + 1e-30)
    
    # Reference normalization (match RAM convention)
    # Add cylindrical spreading explicitly
    # tl += 10.0 * np.log10(ranges + 1e-30)
    
    return tl


def test_pekeris():
    """Quick test of Pekeris solution"""
    print("Testing Pekeris analytic solution...")
    
    # Test parameters
    freq = 50.0  # Hz
    depth = 200.0  # m
    z_source = 100.0  # m
    z_receiver = 50.0  # m
    c_water = 1500.0  # m/s
    c_bottom = 1700.0  # m/s
    
    # Create range array
    ranges = np.linspace(100, 10000, 100)
    
    # Compute TL
    tl = pekeris_solution(ranges, freq, depth, z_source, z_receiver,
                         c_water, c_bottom)
    
    print(f"  TL range: [{np.min(tl):.1f}, {np.max(tl):.1f}] dB")
    print(f"  Mean TL: {np.mean(tl):.1f} dB")
    
    # Basic sanity checks
    assert np.all(np.isfinite(tl)), "TL contains non-finite values"
    assert np.all(tl > 0), "TL should be positive (loss)"
    assert np.min(tl) < 100, "TL too high (check normalization)"
    
    print("  ✓ Pekeris solution test passed")
    
    return ranges, tl


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Run test
    ranges, tl = test_pekeris()
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ranges / 1000, tl, 'b-', linewidth=2)
    plt.xlabel('Range (km)')
    plt.ylabel('Transmission Loss (dB)')
    plt.title('Pekeris Waveguide - Analytic Solution')
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('pekeris_test.png', dpi=150)
    print("\nPlot saved: pekeris_test.png")