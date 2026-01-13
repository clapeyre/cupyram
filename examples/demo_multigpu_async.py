#!/usr/bin/env python3
"""
CuPyRAM Multi-GPU Processing Template (Synthetic Data)

This script demonstrates the optimal architecture for high-throughput acoustic 
simulations using CuPyRAM:
1.  **Multi-GPU Distribution**: Distributes batches of emitters across available GPUs.
2.  **Async Preprocessing**: Uses CPU cores to prepare the next batch of data 
    (GIS operations, interpolation) while the GPU computes the current batch.
3.  **Shared Memory**: Zero-copy data sharing between the main process and worker 
    processes to avoid pickling overhead.

This template uses SYNTHETIC data to be runnable out-of-the-box.
"""

import time
import numpy as np
import pickle
import atexit
import sys
import os

# Try importing CuPyRAM
try:
    import cupy as cp
    from cupyram import CuPyRAM, AsyncBatcher
except ImportError as e:
    print(f"Error: {e}")
    print("Please install cupyram first.")
    sys.exit(1)

# Optional dependencies for this specific example (GIS work)
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, box
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("Warning: GeoPandas not found. Using simplified synthetic environment.")


# -----------------------
# PARAMETERS
# -----------------------
FREQS = [60.0] # Hz
ZS = 10.0            # Source depth
ZR = 30.0            # Receiver depth
RMAX = 5000.0        # 5 km range
DR = 50.0            # Range step
NUM_ANGLES = 32      # Radials per emitter
BATCH_SIZE = 200     # Emitters per batch
NUM_EMITTERS = 800   # Total emitters to simulate

# Concurrency settings
MAX_WORKERS = 2      # CPU cores for preprocessing
MAX_PREFETCH = 2     # Batches to prepare in advance


# =============================================================================
# SYNTHETIC DATA GENERATOR
# =============================================================================

def generate_synthetic_environment():
    """
    Creates a toy environment: a 100km x 100km area with random "islands" 
    (shallower bathymetry) and a background ocean.
    """
    print("Generating synthetic environment...")
    
    # 1. Background ocean (deep)
    bounds = (0, 0, 100000, 100000)
    
    # 2. Random "islands" or features (polygons)
    # If we have GeoPandas, we make real polygons. If not, we simulated it.
    if HAS_GEOPANDAS:
        polys = []
        data = []
        
        # Background
        polys.append(box(*bounds))
        data.append({'depth': 2000.0, 'temp_surf': 20.0})
        
        # 20 random features
        np.random.seed(42)
        for _ in range(20):
            x = np.random.uniform(0, 100000)
            y = np.random.uniform(0, 100000)
            r = np.random.uniform(2000, 8000)
            polys.append(Point(x, y).buffer(r))
            data.append({'depth': np.random.uniform(50, 500), 'temp_surf': np.random.uniform(15, 25)})
            
        env_gdf = gpd.GeoDataFrame(data, geometry=polys)
        return env_gdf
    else:
        # Fallback dictionary for pure numpy mode
        return {'bounds': bounds, 'features': []}


def generate_emitters(n=100):
    """Generate random emitter locations."""
    x = np.random.uniform(10000, 90000, n)
    y = np.random.uniform(10000, 90000, n)
    ids = np.arange(n)
    
    if HAS_GEOPANDAS:
        return gpd.GeoDataFrame({'id': ids}, geometry=gpd.points_from_xy(x, y))
    else:
        return [{'id': i, 'x': xi, 'y': yi} for i, xi, yi in zip(ids, x, y)]


# =============================================================================
# WORKER LOGIC (Running in ProcessPool)
# =============================================================================

# Global cache to store unpickled env in worker process
_WORKER_ENV_CACHE = None

def worker_preprocess_batch(batch_id, emitter_batch, shm_name, shm_size):
    """
    The heavy lifting function. This runs on a CPU worker.
    
    It takes raw emitter locations, samples the environment (GIS ops),
    and formats the data for CuPyRAM.
    """
    global _WORKER_ENV_CACHE
    
    # 1. Load Environment (Zero-Copy / Cached)
    if _WORKER_ENV_CACHE is None:
        _WORKER_ENV_CACHE = AsyncBatcher.load_shared_object(shm_name, shm_size)
    env = _WORKER_ENV_CACHE
    
    # 2. Process the batch
    # In a real script, this would involve complex spatial joins, interpolation, etc.
    # Here we simulate the output structure required by CuPyRAM.
    
    tick = time.time()
    
    all_transects_data = []
    emitter_boundaries = []
    
    for i, emitter in enumerate(emitter_batch):
        # Extract coordinates
        if HAS_GEOPANDAS:
            ex, ey = emitter.geometry.x, emitter.geometry.y
            e_id = emitter['id']
        else:
            ex, ey = emitter['x'], emitter['y']
            e_id = emitter['id']
            
        start_idx = len(all_transects_data)
        
        # For each angle, generate a transect
        for angle in np.linspace(0, 360, NUM_ANGLES, endpoint=False):
            
            # --- SIMULATED GIS WORK ---
            # Create a range vector
            ranges = np.arange(0, RMAX, DR)
            
            # Synthetic bathymetry: Deep (2000m) with some sine wave variation
            # In real code: Use gpd.sjoin to sample 'env' at these points
            bathy = 2000.0 - 500.0 * np.sin(ranges / 10000.0)
            
            # Synthetic Sound Speed Profile (Munk profile approximation)
            z_ss = np.linspace(0, 3000, 50)
            
            # Make sound speed vary with range (simulating ocean front)
            # cw must be shape (n_depths, n_ranges)
            cw = np.zeros((len(z_ss), len(ranges)))
            for r_idx in range(len(ranges)):
                cw[:, r_idx] = 1480.0 + 0.005 * (z_ss - 1000.0)**2 + np.sin(ranges / 20000.0)[r_idx] * 10.0
            
            # Seabed properties (constant for demo but fully expanded for consistency)
            n_ranges = len(ranges)
            cb = np.full((1, n_ranges), 1700.0)
            rhob = np.full((1, n_ranges), 1.5)
            attn = np.full((1, n_ranges), 0.5)
            
            # Pack data for CuPyRAM (See CuPyRAM class docstring)
            # Tuple: (angle, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb)
            
            # Range-Bathymetry pairs
            rbzb = np.column_stack((ranges, bathy))
            
            transect_data = {
                'z_ss': z_ss,
                'rp_ss': ranges,
                'cw': cw,
                'z_sb': np.array([0.0]),
                'rp_sb': ranges,
                'cb': cb,
                'rhob': rhob,
                'attn': attn,
                'rbzb': rbzb
            }
            
            all_transects_data.append(transect_data)
            
        end_idx = len(all_transects_data)
        emitter_boundaries.append({
            'emitter_id': e_id,
            'start': start_idx,
            'end': end_idx
        })
        
    proc_time = time.time() - tick
    return batch_id, all_transects_data, emitter_boundaries, proc_time


# =============================================================================
# GPU RUNNER
# =============================================================================

def run_batch_on_gpu(batch_id, transects_data, gpu_id=0):
    """
    Takes the pre-processed data structures and runs CuPyRAM.
    """
    if not transects_data:
        return {}

    with cp.cuda.Device(gpu_id):
        # Unpack the list of dicts into list of arrays (SOA - Structure of Arrays)
        # This is what CuPyRAM expects
        z_ss = [t['z_ss'] for t in transects_data]
        rp_ss = [t['rp_ss'] for t in transects_data]
        cw = [t['cw'] for t in transects_data]
        z_sb = [t['z_sb'] for t in transects_data]
        rp_sb = [t['rp_sb'] for t in transects_data]
        cb = [t['cb'] for t in transects_data]
        rhob = [t['rhob'] for t in transects_data]
        attn = [t['attn'] for t in transects_data]
        rbzb = [t['rbzb'] for t in transects_data]
        
        # Initialize Model
        model = CuPyRAM(
            freq=FREQS, zs=ZS, zr=ZR,
            z_ss=z_ss, rp_ss=rp_ss, cw=cw,
            z_sb=z_sb, rp_sb=rp_sb,
            cb=cb, rhob=rhob, attn=attn,
            rbzb=rbzb,
            rmax=RMAX, dr=DR,
            compute_grids=False, # We provided grids directly
            batch_size=len(transects_data)
        )
        
        # Run Simulation
        result = model.run()
        
        return result['TL Line'] # Transmission Loss at Receiver Depth


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("CuPyRAM Multi-GPU Async Template")
    print("="*60)
    
    # 1. Setup Data
    env = generate_synthetic_environment()
    emitters = generate_emitters(NUM_EMITTERS)
    
    # Convert emitters to list of batches
    emitter_list = []
    if HAS_GEOPANDAS:
        for idx, row in emitters.iterrows():
            emitter_list.append(row)
    else:
        emitter_list = emitters

    batches = [emitter_list[i:i + BATCH_SIZE] for i in range(0, len(emitter_list), BATCH_SIZE)]
    print(f"Prepared {len(batches)} batches of {BATCH_SIZE} emitters.")
    
    # 2. Setup Shared Memory
    print("Moving environment to shared memory...")
    shm_name, shm_size = AsyncBatcher.share_object(env)
    atexit.register(AsyncBatcher.cleanup_shared_object, shm_name)
    
    # 3. Setup Iterator
    # Generator that yields arguments for the worker function
    def args_generator():
        for i, batch in enumerate(batches):
            yield (i, batch, shm_name, shm_size)
            
    # 4. Initialize AsyncBatcher
    batcher = AsyncBatcher(
        input_data_iterator=args_generator(),
        worker_func=worker_preprocess_batch,
        max_prefetch=MAX_PREFETCH,
        max_workers=MAX_WORKERS
    )
    
    # 5. Process Loop
    total_rays = 0
    t_start = time.time()
    
    print("\nStarting processing loop...")
    for batch_id, transects_data, boundaries, prep_time in batcher:
        
        # Simple Round-Robin GPU assignment
        num_gpus = cp.cuda.runtime.getDeviceCount()
        gpu_id = batch_id % num_gpus
        
        print(f"Batch {batch_id}: {len(transects_data)} rays (Prep: {prep_time:.3f}s) -> GPU {gpu_id}")
        
        # Run on GPU
        tl_results = run_batch_on_gpu(batch_id, transects_data, gpu_id)
        
        total_rays += len(transects_data) * len(FREQS)
        
    t_total = time.time() - t_start
    print("\n" + "="*60)
    print(f"Done! Processed {total_rays} rays in {t_total:.2f}s")
    print(f"Throughput: {total_rays/t_total:.1f} rays/sec")

if __name__ == "__main__":
    main()
