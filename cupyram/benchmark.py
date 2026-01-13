#!/usr/bin/env python3
"""
GPU Batch Scaling Benchmark for CuPyRAM

CLI tool to test performance scaling with increasing batch sizes.
Automatically detects GPU VRAM and suggests optimal batch sizes.
"""

import numpy as np
import time
import sys
from typing import List, Tuple, Dict, Optional

import click
import cupy
from cupy.cuda.memory import OutOfMemoryError as CuPyOOMError

import nvtx
from cupyram import CuPyRAM


def get_gpu_info() -> Tuple[str, int, int]:
    """
    Get GPU information including name, total VRAM, and available VRAM.
    
    Returns:
        Tuple of (gpu_name, total_bytes, available_bytes)
    """
    device = cupy.cuda.Device()
    
    # Get device name using runtime API
    gpu_name = cupy.cuda.runtime.getDeviceProperties(device.id)['name'].decode('utf-8')
    
    # Get memory info
    mempool = cupy.get_default_memory_pool()
    mempool.free_all_blocks()
    
    free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
    
    return gpu_name, total_bytes, free_bytes


def estimate_memory_per_ray(freq: float, rmax: float, zmax: float = 3000.0) -> int:
    """
    Estimate GPU memory usage per ray in bytes.
    
    This is a rough estimate based on the grid size and typical array allocations.
    CuPyRAM allocates several complex128 arrays for field computation.
    
    Args:
        freq: Frequency in Hz (higher freq = finer grid = more memory)
        rmax: Maximum range in meters
        zmax: Maximum depth in meters
    
    Returns:
        Estimated bytes per ray
    """
    # CuPyRAM computes dr and dz based on frequency
    # Rough estimates (actual values computed by CuPyRAM may vary):
    # dr ≈ c0/(20*freq) where c0≈1500 m/s
    # dz ≈ c0/(10*freq) 
    
    c0 = 1500.0
    dr_estimate = c0 / (20.0 * freq)
    dz_estimate = c0 / (10.0 * freq)
    
    nr = int(rmax / dr_estimate) + 1
    nz = int(zmax / dz_estimate) + 1
    
    # Main arrays in CuPyRAM (complex128 = 16 bytes):
    # - u field: nz × nr (main field array)
    # - pd: multiple arrays for Padé coefficients
    # - Various temporary arrays for computation
    # Conservative estimate: ~5-10 arrays of size nz
    
    bytes_per_complex = 16  # complex128
    field_array_bytes = nz * bytes_per_complex
    
    # Estimate total: field + Padé arrays + workspace + overhead
    # This is conservative to avoid OOM
    estimated_bytes = field_array_bytes * 15  # Safety factor
    
    return estimated_bytes


def generate_batch_sizes(
    freqs: List[float], 
    rmax: float,
    available_vram: int,
    n_steps: int = 9
) -> List[int]:
    """
    Generate smart batch sizes based on available VRAM.
    
    Creates a progression that densely samples the region where GPU 
    saturation typically occurs (50-110% of estimated optimal).
    
    Args:
        freqs: List of frequencies (max freq determines grid resolution)
        rmax: Maximum range in meters
        available_vram: Available GPU memory in bytes
        n_steps: Number of batch size steps to test
    
    Returns:
        List of batch sizes to benchmark
    """
    max_freq = max(freqs) if isinstance(freqs, list) else float(freqs)
    n_freqs = len(freqs) if isinstance(freqs, list) else 1
    
    # Estimate memory per ray
    bytes_per_ray = estimate_memory_per_ray(max_freq, rmax)
    
    # Use 90% of available VRAM as starting estimate (we'll push beyond this)
    usable_vram = int(available_vram * 0.90)
    
    # Account for multi-frequency if applicable
    bytes_per_ray_effective = bytes_per_ray * n_freqs
    
    # Calculate theoretical max batch size
    max_batch_theoretical = usable_vram // bytes_per_ray_effective
    
    # Cap at reasonable maximum (very large batches have diminishing returns)
    max_batch = min(max_batch_theoretical, 50000)
    
    # Ensure minimum
    max_batch = max(max_batch, 128)
    
    # Generate progression that focuses on finding the saturation point
    # Strategy: Start with a few small sizes, then densely sample around 
    # the expected saturation region (60-110% of max_batch)
    
    if max_batch < 512:
        # Very limited VRAM - use small linear steps
        batch_sizes = list(range(32, max_batch + 1, max(32, max_batch // 8)))
    else:
        # Simple strategy: evenly space samples from 40% to 120% of estimate
        # This captures the full performance curve including any non-monotonic behavior
        start_batch = int(max_batch * 0.40)
        end_batch = int(max_batch * 1.20)
        
        # Generate evenly spaced batch sizes
        batch_sizes = np.linspace(start_batch, end_batch, n_steps, dtype=int).tolist()
        
        # Round to multiples of 128 for nice numbers
        batch_sizes = [int(round(b / 128) * 128) for b in batch_sizes]
        
        # Remove duplicates while preserving order
        seen = set()
        batch_sizes = [x for x in batch_sizes if not (x in seen or seen.add(x))]
    
    return batch_sizes


def create_synthetic_environment(rmax: float) -> Dict:
    """
    Create a synthetic but plausible ocean environment for benchmarking.
    
    Args:
        rmax: Maximum range in meters
    
    Returns:
        Dictionary with environment configuration
    """
    # Range sampling points
    r_points = np.arange(0, rmax + 1, 1000)  # Sample every 1 km
    n_points = len(r_points)
    
    # Water column sound speed profile (simple isovelocity)
    z_ss = np.array([0.0, 100.0, 500.0, 1000.0, 2000.0, 3000.0])
    c0 = 1500.0
    cw = np.full((len(z_ss), n_points), c0)
    
    # Bathymetry: constant depth
    bathy_depth = 3000.0
    rbzb = np.column_stack([r_points, np.full(n_points, bathy_depth)])
    
    # Seabed properties
    z_sb = np.array([0.0])
    cb = np.full((1, n_points), 1700.0)      # Seabed sound speed (m/s)
    rhob = np.full((1, n_points), 1.5)       # Seabed density (g/cm³)
    attn = np.full((1, n_points), 0.5)       # Attenuation (dB/wavelength)
    
    return {
        'z_ss': z_ss,
        'rp_ss': r_points,
        'cw': cw,
        'z_sb': z_sb,
        'rp_sb': r_points,
        'cb': cb,
        'rhob': rhob,
        'attn': attn,
        'rbzb': rbzb,
        'c0': c0
    }


def run_batch_benchmark(
    batch_size: int,
    freqs: List[float],
    rmax: float,
    zs: float,
    zr: float,
    zmplt: float,
    max_workers: int
) -> Dict:
    """
    Run CuPyRAM benchmark with synthetic environment.
    
    Args:
        batch_size: Number of rays to process
        freqs: Frequency or list of frequencies
        rmax: Maximum range in meters
        zs: Source depth in meters
        zr: Receiver depth in meters
        zmplt: Maximum output depth in meters
        max_workers: CPU threads for Padé computation
    
    Returns:
        Dictionary with timing breakdown and metadata
    
    Raises:
        CuPyOOMError: If GPU runs out of memory
    """
    env = create_synthetic_environment(rmax)
    
    # Replicate environment for batch
    z_ss_list = [env['z_ss']] * batch_size
    rp_ss_list = [env['rp_ss']] * batch_size
    cw_list = [env['cw']] * batch_size
    z_sb_list = [env['z_sb']] * batch_size
    rp_sb_list = [env['rp_sb']] * batch_size
    cb_list = [env['cb']] * batch_size
    rhob_list = [env['rhob']] * batch_size
    attn_list = [env['attn']] * batch_size
    rbzb_list = [env['rbzb']] * batch_size
    
    # Model instantiation (lightweight - no heavy computation)
    t_total_start = time.time()
    with nvtx.annotate(f"benchmark_batch{batch_size}", color="purple"):
        model = CuPyRAM(
            freqs, zs, zr, z_ss_list, rp_ss_list, cw_list, z_sb_list, rp_sb_list,
            cb_list, rhob_list, attn_list, rbzb_list,
            rmax=rmax,
            zmplt=zmplt,
            compute_grids=False,  # Save VRAM
            batch_size=batch_size,
            max_workers=max_workers,
            c0=env['c0'],
            benchmark_propagation=True  # Enable fine-grained propagation timing
        )
        
        dr_computed = model._dr
        dz_computed = model._dz
        
        # Run model - this does setup + compute + transfer internally
        result = model.run()
    
    t_total = time.time() - t_total_start
    
    # Get timing from model
    t_propagation = model.propagation_time  # Pure GPU propagation kernel time
    
    n_freqs = len(freqs) if isinstance(freqs, list) else 1
    total_fields = batch_size * n_freqs
    
    # Clean up GPU memory
    mempool = cupy.get_default_memory_pool()
    mempool.free_all_blocks()
    
    return {
        'batch_size': batch_size,
        'total_rays': batch_size,
        'total_fields': total_fields,
        'n_freqs': n_freqs,
        'time_total': t_total,
        'time_propagation': t_propagation,  # Pure GPU kernel time
        'dr': dr_computed,
        'dz': dz_computed,
    }


@click.command()
@click.option(
    '--frequency', '-f',
    type=float,
    multiple=True,
    default=[100.0],
    help='Frequency in Hz (can be specified multiple times for multi-frequency mode). Default: 100 Hz',
    show_default=True
)
@click.option(
    '--rmax',
    type=float,
    default=2000.0,
    help='Maximum propagation range in meters',
    show_default=True
)
@click.option(
    '--max-workers',
    type=int,
    default=8,
    help='CPU threads for parallel Padé computation',
    show_default=True
)
@click.option(
    '--n-steps',
    type=int,
    default=9,
    help='Number of batch size steps to test',
    show_default=True
)
@click.option(
    '--source-depth',
    type=float,
    default=10.0,
    help='Source depth in meters',
    show_default=True
)
@click.option(
    '--receiver-depth',
    type=float,
    default=10.0,
    help='Receiver depth in meters',
    show_default=True
)
def benchmark_cli(
    frequency: Tuple[float, ...],
    rmax: float,
    max_workers: int,
    n_steps: int,
    source_depth: float,
    receiver_depth: float
):
    """
    CuPyRAM GPU Batch Scaling Benchmark
    
    Automatically detects GPU VRAM and tests performance scaling with
    increasing batch sizes using a synthetic ocean environment.
    
    Examples:
    
        # Basic benchmark with default 100 Hz:
        cupyram-benchmark
        
        # Multi-frequency benchmark:
        cupyram-benchmark -f 100 -f 125 -f 150 -f 200
        
        # Longer range, more CPU workers:
        cupyram-benchmark --rmax 10000 --max-workers 16
    """
    # Convert frequency tuple to list
    freqs = list(frequency)
    n_freqs = len(freqs)
    
    # Print header
    click.echo("=" * 70)
    click.echo("CuPyRAM GPU Batch Scaling Benchmark")
    click.echo("=" * 70)
    
    # Get GPU info
    try:
        gpu_name, total_vram, available_vram = get_gpu_info()
        click.echo(f"\n🖥️  GPU Information:")
        click.echo(f"  Device: {gpu_name}")
        click.echo(f"  Total VRAM: {total_vram / (1024**3):.2f} GB")
        click.echo(f"  Available VRAM: {available_vram / (1024**3):.2f} GB")
    except Exception as e:
        click.echo(f"\n❌ Error detecting GPU: {e}", err=True)
        sys.exit(1)
    
    # Generate batch sizes
    try:
        batch_sizes = generate_batch_sizes(freqs, rmax, available_vram, n_steps)
        
        # Estimate memory per ray for display
        max_freq = max(freqs)
        bytes_per_ray = estimate_memory_per_ray(max_freq, rmax)
        mb_per_ray = bytes_per_ray / (1024**2)
        
        click.echo(f"\n📊 Benchmark Configuration:")
        if n_freqs > 1:
            click.echo(f"  Frequencies: {freqs} Hz ({n_freqs} frequencies)")
            click.echo(f"    → Multi-frequency mode (super-batch)")
        else:
            click.echo(f"  Frequency: {freqs[0]} Hz")
        click.echo(f"  Max range: {rmax/1000:.1f} km")
        click.echo(f"  Source depth: {source_depth} m")
        click.echo(f"  Receiver depth: {receiver_depth} m")
        click.echo(f"  CPU threads: {max_workers}")
        click.echo(f"\n💾 Memory Estimation:")
        click.echo(f"  Estimated memory per ray: ~{mb_per_ray:.1f} MB")
        if n_freqs > 1:
            click.echo(f"  Effective memory per ray: ~{mb_per_ray * n_freqs:.1f} MB ({n_freqs} frequencies)")
        click.echo(f"  Batch sizes to test: {batch_sizes}")
        click.echo(f"  Estimated optimal: ~{int(available_vram * 0.9 / (bytes_per_ray * n_freqs))}")
        click.echo(f"  (Will push beyond estimate to find actual VRAM limit)")
        
    except Exception as e:
        click.echo(f"\n❌ Error generating batch sizes: {e}", err=True)
        sys.exit(1)
    
    # Run benchmarks
    click.echo("\n" + "=" * 70)
    click.echo("Running Benchmarks")
    click.echo("=" * 70)
    click.echo("\nEach batch uses identical synthetic environment for fair comparison.")
    click.echo("GPU memory is cleared between runs to prevent fragmentation.\n")
    
    results = []
    dr_reported = None
    dz_reported = None
    
    for i, batch_size in enumerate(batch_sizes):
        click.echo(f"[{i+1}/{len(batch_sizes)}] Testing batch size {batch_size}...", nl=False)
        
        try:
            result = run_batch_benchmark(
                batch_size, freqs, rmax, source_depth, receiver_depth,
                zmplt=100.0, max_workers=max_workers
            )
            
            if dr_reported is None:
                dr_reported = result['dr']
                dz_reported = result['dz']
                click.echo(f" ✓ (dr={dr_reported:.2f}m, dz={dz_reported:.4f}m)")
            else:
                click.echo(" ✓")
            
            # Calculate throughput metrics
            # Pure propagation throughput (GPU kernels only - what we care about)
            result['throughput_propagation'] = result['total_fields'] / result['time_propagation']
            result['time_per_field_propagation'] = result['time_propagation'] / result['total_fields']
            
            # Total throughput (including everything - practical metric)
            result['throughput_total'] = result['total_fields'] / result['time_total']
            result['time_per_field_total'] = result['time_total'] / result['total_fields']
            
            # Overhead percentage (everything except pure GPU compute)
            result['overhead_pct'] = ((result['time_total'] - result['time_propagation']) / result['time_total'] * 100.0)
            
            results.append(result)
            
            click.echo(f"    Propagation: {result['time_propagation']:.2f}s | "
                      f"Total: {result['time_total']:.2f}s ({result['overhead_pct']:.0f}% overhead)")
            click.echo(f"    Throughput: {result['throughput_propagation']:.1f} fields/s (GPU) | "
                      f"{result['throughput_total']:.1f} fields/s (total)")
            
        except CuPyOOMError:
            click.echo(" ⚠️  GPU Out of Memory (VRAM limit reached)")
            click.echo(f"    Batch {batch_size} exceeds available VRAM - this is expected.")
            click.echo(f"    Stopping benchmark at VRAM limit.")
            break
        except Exception as e:
            click.echo(f" ❌ Error: {e}")
            click.echo(f"    Skipping batch size {batch_size}")
            continue
    
    if not results:
        click.echo("\n❌ No successful benchmark runs. Check GPU availability.", err=True)
        sys.exit(1)
    
    # Calculate efficiency percentages
    best_throughput_gpu = max(r['throughput_propagation'] for r in results)
    
    for r in results:
        r['efficiency'] = (r['throughput_propagation'] / best_throughput_gpu * 100.0)
    
    # Print results table
    click.echo("\n" + "=" * 85)
    click.echo("Results")
    click.echo("=" * 85)
    
    click.echo(f"\n{'Batch':>6} {'GPU Time':>10} {'GPU Tput':>12} {'Eff.':>7} {'Total Time':>12} {'Total Tput':>12} {'Overhead':>10}")
    click.echo(f"{'Size':>6} {'(s)':>10} {'(fields/s)':>12} {'(%)':>7} {'(s)':>12} {'(fields/s)':>12} {'(%)':>10}")
    click.echo("-" * 85)
    
    for r in results:
        click.echo(f"{r['batch_size']:>6} {r['time_propagation']:>10.2f} "
                  f"{r['throughput_propagation']:>12.1f} {r['efficiency']:>6.1f}% "
                  f"{r['time_total']:>12.2f} {r['throughput_total']:>12.1f} "
                  f"{r['overhead_pct']:>9.0f}%")
    
    # Find and report optimal configuration
    optimal = max(results, key=lambda x: x['throughput_propagation'])
    
    click.echo("\n" + "=" * 85)
    click.echo("✓ Optimal Configuration")
    click.echo("=" * 85)
    
    click.echo(f"\n🚀 Peak GPU Performance:")
    click.echo(f"  Batch size: {optimal['batch_size']}")
    click.echo(f"  GPU throughput: {optimal['throughput_propagation']:.1f} fields/s")
    click.echo(f"  Total throughput: {optimal['throughput_total']:.1f} fields/s")
    click.echo(f"  Time per field: {optimal['time_per_field_propagation']*1000:.2f} ms (GPU only)")
    click.echo(f"  Overhead: {optimal['overhead_pct']:.0f}%")
    
    if dr_reported:
        click.echo(f"\n📐 Grid: dr={dr_reported:.2f}m, dz={dz_reported:.4f}m (~{int(rmax / dr_reported)} steps)")
    
    # Saturation analysis
    if optimal['batch_size'] == max(r['batch_size'] for r in results):
        click.echo(f"\n⚠️  Peak at maximum tested batch size - GPU may not be saturated")
    else:
        click.echo(f"\n✓ GPU saturated at batch size ~{optimal['batch_size']}")
        larger_batches = [r for r in results if r['batch_size'] > optimal['batch_size']]
        if larger_batches:
            first_drop = min(larger_batches, key=lambda x: x['batch_size'])
            drop_pct = (1 - first_drop['throughput_propagation'] / optimal['throughput_propagation']) * 100
            if drop_pct > 5:
                click.echo(f"   Performance drops {drop_pct:.0f}% at batch {first_drop['batch_size']}")
    
    # Simple insights
    avg_overhead = sum(r['overhead_pct'] for r in results) / len(results)
    click.echo(f"\n💡 Average overhead: {avg_overhead:.0f}% (setup, output, transfer)")
    if avg_overhead > 40:
        click.echo(f"   → Use larger batches to reduce overhead impact")
    
    click.echo("\n" + "=" * 85)


if __name__ == "__main__":
    benchmark_cli()
