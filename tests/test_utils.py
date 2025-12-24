#!/usr/bin/env python3
"""
Shared test infrastructure for functional test suite.
Provides utilities for comparing GPU results against CPU PyRAM and analytic solutions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from pyram.PyRAM import PyRAM

# Get test output directory
TEST_OUTPUT_DIR = Path(os.environ.get('TEST_OUTPUT_DIR', 'test_outputs'))
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

# Test tolerance thresholds (in dB unless otherwise noted)
TOLERANCES = {
    'phase1_analytic': 0.01,      # dB - vs analytic solution
    'phase1_cpu': 0.01,            # dB - vs PyRAM
    'phase2_cpu': 0.1,             # dB
    'phase3_cpu': 0.5,             # dB - energy conservation is harder
    'phase4_coord': 0.5,           # % - coordinate accuracy
    'phase4_cpu': 1.0              # dB
}

@dataclass
class SimulationConfig:
    """Configuration parameters for a test"""
    name: str
    freq: float
    zs: float
    zr: float
    rmax: float
    dr: float
    depth: float
    c0: float = 1500.0
    tolerance: float = 1.0
    
    def __str__(self):
        return (
                f"Test: {self.name}\n"
                f"  Frequency: {self.freq} Hz\n"
                f"  Source depth: {self.zs} m\n"
                f"  Receiver depth: {self.zr} m\n"
                f"  Max range: {self.rmax/1000:.1f} km\n"
                f"  Range step: {self.dr} m\n"
                f"  Water depth: {self.depth} m")


def compare_results(result1: np.ndarray, result2: np.ndarray, 
                   label1: str = "Result 1", label2: str = "Result 2") -> Dict[str, float]:
    """
    Statistical comparison of two result arrays.
    
    Args:
        result1: First result array (TL in dB)
        result2: Second result array (TL in dB)
        label1: Label for first result
        label2: Label for second result
    
    Returns:
        Dictionary with statistical metrics
    """
    # Ensure same length
    assert len(result1) == len(result2)
    
    # Calculate differences
    diff = result1 - result2
    abs_diff = np.abs(diff)
    
    # Remove any NaN/Inf
    valid = np.isfinite(diff)
    if not np.all(valid):
        print(f"{Colors.YELLOW}Warning: {np.sum(~valid)} non-finite values found{Colors.END}")
        diff = diff[valid]
        abs_diff = abs_diff[valid]
    
    stats = {
        'mean_diff': np.mean(diff),
        'std_diff': np.std(diff),
        'max_abs_diff': np.max(abs_diff),
        'rms_diff': np.sqrt(np.mean(diff**2)),
        'median_diff': np.median(diff),
        'num_samples': len(diff)
    }
    
    return stats


def print_comparison(stats: Dict[str, float], label1: str, label2: str, 
                    tolerance: float, show_samples: bool = True,
                    ranges: Optional[np.ndarray] = None,
                    result1: Optional[np.ndarray] = None,
                    result2: Optional[np.ndarray] = None):
    """
    Pretty-print comparison statistics.
    
    Args:
        stats: Statistics dictionary from compare_results
        label1: Label for first result
        label2: Label for second result
        tolerance: Pass/fail threshold (dB)
        show_samples: Whether to show sample values
        ranges: Range array (km) for sample display
        result1: First result array for sample display
        result2: Second result array for sample display
    """
    print(f"\n{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}Comparison: {label1} vs {label2}{Colors.END}")
    print(f"{Colors.CYAN}{'='*70}{Colors.END}")
    
    print(f"\nStatistical Metrics:")
    print(f"  Mean difference:        {stats['mean_diff']:+8.4f} dB")
    print(f"  Std deviation:           {stats['std_diff']:8.4f} dB")
    print(f"  Median difference:      {stats['median_diff']:+8.4f} dB")
    print(f"  Max absolute difference: {stats['max_abs_diff']:8.4f} dB")
    print(f"  RMS difference:          {stats['rms_diff']:8.4f} dB")
    print(f"  Number of samples:       {stats['num_samples']}")
    
    # Show sample values if requested
    if show_samples and ranges is not None and result1 is not None and result2 is not None:
        print(f"\nSample Comparisons:")
        indices = np.linspace(0, min(len(result1), len(result2)) - 1, 5, dtype=int)
        for idx in indices:
            r_km = ranges[idx] if idx < len(ranges) else idx * 0.05  # Fallback
            diff = result1[idx] - result2[idx]
            print(f"  {r_km:6.2f} km: {label1}={result1[idx]:7.2f} dB, "
                  f"{label2}={result2[idx]:7.2f} dB, diff={diff:+6.2f} dB")
    
    # Pass/Fail verdict
    print(f"\n{Colors.BOLD}Verdict:{Colors.END}")
    if stats['max_abs_diff'] < tolerance:
        print(f"  {Colors.GREEN}✓ PASS{Colors.END}: Max difference {stats['max_abs_diff']:.4f} dB < {tolerance} dB threshold")
        return True
    else:
        print(f"  {Colors.RED}✗ FAIL{Colors.END}: Max difference {stats['max_abs_diff']:.4f} dB > {tolerance} dB threshold")
        return False


def plot_comparison(ranges: np.ndarray, results_dict: Dict[str, np.ndarray],
                   title: str, output_path: str, show_diff: bool = True):
    """
    Create comparison plot with multiple curves.
    
    Args:
        ranges: Range array (km)
        results_dict: Dictionary of {label: result_array}
        title: Plot title
        output_path: Output file name (will be saved in test_outputs/)
        show_diff: Whether to include difference subplot
    """
    # Ensure output goes to test_outputs directory
    output_path = TEST_OUTPUT_DIR / output_path
    if show_diff and len(results_dict) == 2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = None
    
    # Main comparison plot
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    linestyles = ['-', '--', '-.', ':']
    
    for idx, (label, result) in enumerate(results_dict.items()):
        min_len = min(len(ranges), len(result))
        ax1.plot(ranges[:min_len], result[:min_len], 
                color=colors[idx % len(colors)],
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=2, label=label, alpha=0.8)
    
    ax1.set_xlabel('Range (km)', fontsize=12)
    ax1.set_ylabel('Transmission Loss (dB)', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # TL increases downward
    
    # Difference plot if requested
    if ax2 is not None and len(results_dict) == 2:
        labels = list(results_dict.keys())
        r1, r2 = list(results_dict.values())
        min_len = min(len(r1), len(r2), len(ranges))
        diff = r1[:min_len] - r2[:min_len]
        
        ax2.plot(ranges[:min_len], diff, 'k-', linewidth=1.5)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Range (km)', fontsize=12)
        ax2.set_ylabel(f'Difference (dB)\n{labels[0]} - {labels[1]}', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {np.mean(diff):+.3f} dB\nMax: {np.max(np.abs(diff)):.3f} dB'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {output_path}")
    plt.close()


def save_test_results(filepath: str, data: np.ndarray, metadata: Dict[str, Any]):
    """
    Save test results with metadata.
    
    Args:
        filepath: Output .npy file name (will be saved in test_outputs/)
        data: Result array
        metadata: Dictionary of test parameters
    """
    # Ensure output goes to test_outputs directory
    filepath = TEST_OUTPUT_DIR / filepath
    
    # Save data
    np.save(filepath, data)
    
    # Save metadata as separate text file
    meta_path = str(filepath).replace('.npy', '_meta.txt')
    with open(meta_path, 'w') as f:
        f.write("Test Metadata\n")
        f.write("=" * 50 + "\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"  Saved: {filepath}")
    print(f"  Metadata: {meta_path}")


def run_pyram_reference(config: SimulationConfig, z_ss: np.ndarray, cw: np.ndarray,
                       z_sb: np.ndarray, cb: np.ndarray, rhob: np.ndarray,
                       attn: np.ndarray, rbzb: np.ndarray, 
                       dz: float, np_pade: int = 8, ns: int = 1) -> Tuple[np.ndarray, PyRAM]:
    """
    Run CPU PyRAM as reference solution.
    
    Args:
        config: Test configuration
        z_ss: Water sound speed profile depths
        cw: Water sound speed values
        z_sb: Seabed parameter profile depths
        cb: Seabed sound speed values
        rhob: Seabed density values
        attn: Seabed attenuation values
        rbzb: Bathymetry array
        dz: Depth step (m)
        np_pade: Number of Padé terms
        ns: Number of stability constraints
    
    Returns:
        Tuple of (TL array, PyRAM instance)
    """
    print(f"\n{Colors.BLUE}Running CPU PyRAM Reference...{Colors.END}")
    
    # Create PyRAM instance
    pyram = PyRAM(
        freq=config.freq,
        zs=config.zs,
        zr=config.zr,
        z_ss=z_ss,
        rp_ss=np.array([0.0]),
        cw=cw,
        z_sb=z_sb,
        rp_sb=np.array([0.0]),
        cb=cb,
        rhob=rhob,
        attn=attn,
        rbzb=rbzb,
        dz=dz,
        dr=config.dr,
        rmax=config.rmax,
        np=np_pade,
        ns=ns
    )
    
    # Run simulation
    result = pyram.run()
    tl = result['TL Line']
    
    print(f"  CPU Results: shape={tl.shape}, range=[{np.min(tl):.1f}, {np.max(tl):.1f}] dB")
    print(f"  NaN count: {np.sum(np.isnan(tl))}, Inf count: {np.sum(np.isinf(tl))}")
    
    return tl, pyram


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}")
    print(f"{title}")
    print(f"{'='*70}{Colors.END}")


def print_test_result(passed: bool, message: str):
    """Print a test result with color coding."""
    if passed:
        print(f"{Colors.GREEN}✓ PASS:{Colors.END} {message}")
    else:
        print(f"{Colors.RED}✗ FAIL:{Colors.END} {message}")