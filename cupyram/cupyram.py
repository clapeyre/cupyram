"""
CuPyRAM: Cuda-accelerated Python adaptation of the Range-dependent Acoustic
Model (RAM).  RAM was created by Michael D Collins at the US Naval Research
Laboratory.  This adaptation is of RAM v1.5, available from the Ocean Acoustics
Library at https://oalib-acoustics.org/models-and-software/parabolic-equation

CuPyRAM is a fork of PyRAM, a Python adaptation of the Range-dependent Acoustic
Model (RAM). It is written in pure Python and achieves speeds comparable to
native code by using the Numba library for GPU acceleration.

The CuPyRAM class matches the PyRAM API, and contains methods which largely
correspond to the original Fortran subroutines and functions (including
retaining the same names). The variable names are also mostly the same. However
some of the original code (e.g. subroutine zread) is unnecessary when the same
purpose can be achieved using available Python library functions (e.g. from
NumPy or SciPy) and has therefore been replaced.

A difference in functionality is that sound speed profile updates with range
are decoupled from seabed parameter updates, which provides more flexibility
in specifying the environment (e.g. if the data comes from different sources).

CuPyRAM also provides various conveniences, e.g. automatic calculation of range
and depth steps (though these can be overridden using keyword arguments).
"""

import numpy
import cupy
import nvtx
from time import process_time
from numba import cuda
from tqdm import tqdm
from cupyram.solve import solve
from cupyram.outpt import outpt_cuda
from cupyram.pade import compute_pade_coefficients, compute_pade_coefficients_batch
from cupyram.profl import profl_cuda_launcher
from cupyram.updat import updat_indices_cuda
from cupyram.matrc import matrc_cuda_init_profiles, matrc_cuda_single_pade

# Global flag for fused kernel optimization
# Set to True to use Sum formulation with on-the-fly matrix generation (67% memory savings)
# Set to False to use legacy Product formulation (for validation/comparison)
FUSED_KERNEL = True


class CuPyRAM:

    _np_default = 8
    _dzf = 0.1
    _ndr_default = 1
    _ndz_default = 1
    _ns_default = 1
    _lyrw_default = 20
    _id_default = 0

    @staticmethod
    def _normalize_to_batch(x, batch_size, param_name):
        """
        Normalize input to batched NumPy array with NaN padding for varying lengths.
        
        For backward compatibility with PyRAM API:
        - List of arrays → pad to max length with NaNs → [batch_size, max_len, ...]
        - Single array → tile to [batch_size, ...] (no padding needed)
        
        Returns NumPy array for GPU compatibility and vectorization.
        
        NaN padding enables:
        - Efficient vectorized operations
        - Direct GPU transfer (CuPy compatible)
        - Scalability to billions of rays
        
        Note: Varying range sampling (inhomogeneous shapes) is common in real-world
        scenarios (e.g., rays at different angles traverse different distances).
        """
        if isinstance(x, list):
            if len(x) != batch_size:
                raise ValueError(f"{param_name}: list length {len(x)} != batch_size {batch_size}")
            
            # Convert list elements to numpy arrays
            arrays = [numpy.asarray(item) for item in x]
            
            # Check if all arrays have the same shape (homogeneous)
            shapes = [arr.shape for arr in arrays]
            if len(set(shapes)) == 1:
                # All same shape → stack directly (no padding needed)
                return numpy.stack(arrays, axis=0)
            
            # Inhomogeneous shapes → pad with NaNs
            # Find max shape along each dimension
            ndim = arrays[0].ndim
            max_shape = [max(arr.shape[i] for arr in arrays) for i in range(ndim)]
            
            # Determine dtype (use float for NaN support)
            dtype = arrays[0].dtype
            if numpy.issubdtype(dtype, numpy.integer):
                dtype = numpy.float64  # Convert int to float for NaN support
            elif numpy.issubdtype(dtype, numpy.complexfloating):
                # Complex arrays: use NaN for real and imag parts
                pass  # Keep complex dtype
            
            # Create padded array filled with NaNs
            padded_shape = (batch_size,) + tuple(max_shape)
            padded = numpy.full(padded_shape, numpy.nan, dtype=dtype)
            
            # Copy each array into padded array
            for i, arr in enumerate(arrays):
                # Build slicing tuple for this array's actual shape
                slices = (i,) + tuple(slice(0, s) for s in arr.shape)
                padded[slices] = arr
            
            return padded
        else:
            # Single input → tile for batch
            x_arr = numpy.asarray(x)
            if x_arr.ndim == 0:
                # Scalar → [batch_size]
                return numpy.full(batch_size, x_arr)
            else:
                # Array → add batch dimension and tile
                # [n] → [batch_size, n] or [n,m] → [batch_size, n, m]
                return numpy.tile(x_arr, (batch_size,) + (1,) * x_arr.ndim)
    
    @staticmethod
    def _get_valid_slice(arr):
        """
        Extract valid (non-NaN) portion of a potentially NaN-padded array.
        
        For 1D arrays: returns arr[:valid_len] (trim NaN padding along axis 0)
        For 2D arrays: returns arr[:valid_rows, :valid_cols] (trim NaN padding along both dims)
        
        Returns: (valid_array, valid_length_or_shape)
        """
        # Convert CuPy arrays to NumPy for processing
        if isinstance(arr, cupy.ndarray):
            arr = cupy.asnumpy(arr)
        
        if arr.ndim == 1:
            # 1D: find first NaN
            valid_mask = ~numpy.isnan(arr)
            if numpy.all(valid_mask):
                return arr, len(arr)
            # Find first NaN index
            nan_indices = numpy.where(~valid_mask)[0]
            if len(nan_indices) == 0:
                return arr, len(arr)
            valid_len = nan_indices[0]
            return arr[:valid_len] if valid_len > 0 else arr, valid_len if valid_len > 0 else len(arr)
        elif arr.ndim == 2:
            # 2D: trim NaN padding from both dimensions
            # Check for NaN in rows (first axis)
            row_mask = ~numpy.all(numpy.isnan(arr), axis=1)
            valid_rows = numpy.sum(row_mask)
            
            # Check for NaN in columns (second axis)
            col_mask = ~numpy.all(numpy.isnan(arr), axis=0)
            valid_cols = numpy.sum(col_mask)
            
            if valid_rows == arr.shape[0] and valid_cols == arr.shape[1]:
                # No NaN padding
                return arr, arr.shape[0]
            
            # Return trimmed array
            return arr[:valid_rows, :valid_cols], valid_rows
        else:
            # For higher dimensions, check along first axis
            # Flatten other dims to check for any NaN
            reshaped = arr.reshape(arr.shape[0], -1)
            valid_mask = ~numpy.any(numpy.isnan(reshaped), axis=1)
            if numpy.all(valid_mask):
                return arr, len(arr)
            nan_indices = numpy.where(~valid_mask)[0]
            if len(nan_indices) == 0:
                return arr, len(arr)
            valid_len = nan_indices[0]
            return arr[:valid_len] if valid_len > 0 else arr, valid_len if valid_len > 0 else len(arr)

    def __init__(self, freq, zs, zr, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob,
                 attn, rbzb, **kwargs):
        """
        -------
        args...
        -------
        freq: Frequency (Hz).
        zs: Source depth (m).
        zr: Receiver depth (m).
        z_ss: Water sound speed profile depths (m).
            - Single environment: NumPy 1D array
            - Batched: List of NumPy 1D arrays, one per ray
        rp_ss: Water sound speed profile update ranges (m).
            - Single: NumPy 1D array
            - Batched: List of NumPy 1D arrays, one per ray
        cw: Water sound speed values (m/s).
            - Single: NumPy 2D array, dimensions z_ss.size by rp_ss.size
            - Batched: List of NumPy 2D arrays, one per ray
        z_sb: Seabed parameter profile depths (m).
            - Single: NumPy 1D array
            - Batched: List of NumPy 1D arrays, one per ray
        rp_sb: Seabed parameter update ranges (m).
            - Single: NumPy 1D array
            - Batched: List of NumPy 1D arrays, one per ray
        cb: Seabed sound speed values (m/s).
            - Single: NumPy 2D array, dimensions z_sb.size by rp_sb.size
            - Batched: List of NumPy 2D arrays, one per ray
        rhob: Seabed density values (g/cm3), same structure as cb
        attn: Seabed attenuation values (dB/wavelength), same structure as cb
        rbzb: Bathymetry (m).
            - Single: NumPy 2D array with columns of ranges and depths
            - Batched: List of NumPy 2D arrays, one per ray
        ---------
        kwargs...
        ---------
        np: Number of Pade terms. Defaults to _np_default.
        c0: Reference sound speed (m/s). Defaults to mean of 1st profile.
        dr: Calculation range step (m). Defaults to np times the wavelength.
        dz: Calculation depth step (m). Defaults to _dzf*wavelength.
        ndr: Number of range steps between outputs. Defaults to _ndr_default.
        ndz: Number of depth steps between outputs. Defaults to _ndz_default.
        zmplt: Maximum output depth (m). Defaults to maximum depth in rbzb.
        rmax: Maximum calculation range (m). Defaults to max in rp_ss or rp_sb.
        ns: Number of stability constraints. Defaults to _ns_default.
        rs: Maximum range of the stability constraints (m). Defaults to rmax.
        lyrw: Absorbing layer width (wavelengths). Defaults to _lyrw_default.
        NB: original zmax input not needed due to lyrw.
        id: Integer identifier for this instance.
        batch_size: Number of rays to compute in parallel (GPU batching). Defaults to 1.
            If > 1, all environment parameters must be lists of length batch_size.
        compute_grids: Compute full grid outputs (tlg, cpg). Defaults to True.
            Set to False to save VRAM/RAM by computing only line outputs (tll, cpl).
            When False, tlg and cpg are set to None, and output arrays use 1x1 dummy grids.
        max_workers: Maximum number of CPU threads for parallel Padé coefficient computation.
            Defaults to 8. Set higher for large batches on many-core systems.
            Padé computation is CPU-bound and embarrassingly parallel.
        """

        # GPU array management (CuPyRAM is GPU-only)
        self._batch_size = kwargs.get('batch_size', 1)
        self._compute_grids = kwargs.get('compute_grids', True)  # Compute tlg/cpg grids (can be disabled to save VRAM)
        self._max_workers = kwargs.get('max_workers', 8)  # Parallel Padé computation
        
        # Normalize frequency to array (backward compatible with scalar input)
        if numpy.isscalar(freq):
            self._freqs = numpy.array([freq], dtype=numpy.float64)
        else:
            self._freqs = numpy.array(freq, dtype=numpy.float64).flatten()
        
        self._n_freq = len(self._freqs)
        self._n_env = self._batch_size  # Original batch size = environments
        self._total_batch = self._n_env * self._n_freq  # Total calculations
        
        # Log multi-frequency mode
        if self._n_freq > 1:
            max_freq = numpy.max(self._freqs)
            min_freq = numpy.min(self._freqs)
            print(f"Multi-frequency mode: {self._n_freq} frequencies [{min_freq:.1f}-{max_freq:.1f} Hz]")
            print(f"  N_env={self._n_env}, N_freq={self._n_freq}, N_calc={self._total_batch}")
        
        self._zs, self._zr = zs, zr
        
        # Normalize all inputs to batched numpy arrays [batch_size, ...]
        # Provides backward compatibility with PyRAM API (single arrays → auto-batched)
        z_ss = self._normalize_to_batch(z_ss, self._batch_size, 'z_ss')
        rp_ss = self._normalize_to_batch(rp_ss, self._batch_size, 'rp_ss')
        cw = self._normalize_to_batch(cw, self._batch_size, 'cw')
        z_sb = self._normalize_to_batch(z_sb, self._batch_size, 'z_sb')
        rp_sb = self._normalize_to_batch(rp_sb, self._batch_size, 'rp_sb')
        cb = self._normalize_to_batch(cb, self._batch_size, 'cb')
        rhob = self._normalize_to_batch(rhob, self._batch_size, 'rhob')
        attn = self._normalize_to_batch(attn, self._batch_size, 'attn')
        rbzb = self._normalize_to_batch(rbzb, self._batch_size, 'rbzb')
        
        self.check_inputs(z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb)
        self.get_params(**kwargs)

    @nvtx.annotate("CuPyRAM.run", color="green")
    def run(self):

        """
        Run the model. Sets the following instance variables:
        vr: Calculation ranges (m), NumPy 1D array.
        vz: Calculation depths (m), NumPy 1D array.
        tll: Transmission loss (dB) at receiver depth (zr),
             NumPy 1D array, length vr.size.
        tlg: Transmission loss (dB) grid,
             NumPy 2D array, dimensions vz.size by vr.size.
        proc_time: Processing time (s).
        propagation_time: Pure GPU propagation time (s) - only if benchmark_propagation=True.
        """

        t0 = process_time()

        self.setup()

        nr = int(numpy.round(self._rmax / self._dr)) - 1

        pbar = tqdm(total=nr, desc=f"Running Batch {self._id}", unit="step", mininterval=1.0)

        for rn in range(nr):

            self.updat()

            # Fused matrc-solve step (interleaved Padé computation)
            self._propagate_step()

            self.r = (rn + 2) * self._dr

            # Only compute output if grids are enabled
            if self._compute_grids:
                self.mdr_gpu, self.tlc_gpu = self._outpt()

            # Sync every N steps (e.g., 50 or 100).
            # This makes the progress bar accurate with <0.1% overhead.
            if rn % 50 == 0:
                cuda.synchronize()
            
            pbar.update(1)
        
        # Final sync to ensure timing is correct
        cuda.synchronize()
        pbar.close()

        self.proc_time = process_time() - t0
        
        # Convert output arrays from GPU (CuPy) to CPU (NumPy) for return
        self.tll = cupy.asnumpy(self.tll)
        self.cpl = cupy.asnumpy(self.cpl)
        
        # Only transfer grid outputs if they were computed
        if self._compute_grids:
            self.tlg = cupy.asnumpy(self.tlg)
            self.cpg = cupy.asnumpy(self.cpg)
        else:
            # Set to None to indicate grids were not computed
            self.tlg = None
            self.cpg = None
        
        # Reshape output from [N_calc, ...] to [N_env, N_freq, ...]
        # ALWAYS reshape, regardless of _n_freq value
        # tll: [N_calc, nvr] -> [N_env, N_freq, nvr]
        self.tll = self.tll.reshape(self._n_env, self._n_freq, -1)
        self.cpl = self.cpl.reshape(self._n_env, self._n_freq, -1)
        
        if self._compute_grids:
            # tlg: [N_calc, nvz, nvr] -> [N_env, N_freq, nvz, nvr]
            self.tlg = self.tlg.reshape(self._n_env, self._n_freq, 
                                         self.tlg.shape[1], self.tlg.shape[2])
            self.cpg = self.cpg.reshape(self._n_env, self._n_freq,
                                         self.cpg.shape[1], self.cpg.shape[2])
        
        # PyRAM API compatibility: squeeze dimensions for single env/freq
        if self._n_freq == 1 and self._n_env == 1:
            # Single environment, single frequency: squeeze both dimensions
            self.tll = self.tll[0, 0]
            self.cpl = self.cpl[0, 0]
            if self._compute_grids:
                self.tlg = self.tlg[0, 0]
                self.cpg = self.cpg[0, 0]
        elif self._n_freq == 1:
            # Multiple environments, single frequency: squeeze freq dimension
            self.tll = self.tll[:, 0, :]
            self.cpl = self.cpl[:, 0, :]
            if self._compute_grids:
                self.tlg = self.tlg[:, 0, :, :]
        elif self._n_env == 1:
            # Single environment, multiple frequencies: squeeze env dimension
            self.tll = self.tll[0, :, :]
            self.cpl = self.cpl[0, :, :]
            if self._compute_grids:
                self.tlg = self.tlg[0, :, :, :]
                self.cpg = self.cpg[:, 0, :, :]

        results = {'ID': self._id,
                   'Proc Time': self.proc_time,
                   'Ranges': self.vr,
                   'Depths': self.vz,
                   'TL Grid': self.tlg,
                   'TL Line': self.tll,
                   'CP Grid': self.cpg,
                   'CP Line': self.cpl,
                   'c0': self._c0}

        return results

    def check_inputs(self, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb):
        """
        Validate batched inputs. All inputs are numpy arrays [batch_size, ...] with possible NaN padding.
        """
        self._status_ok = True
        
        # Store NaN-padded arrays (keep as NumPy for validation first)
        self._z_ss = z_ss
        self._rp_ss = rp_ss
        self._cw = cw
        self._z_sb = z_sb
        self._rp_sb = rp_sb
        self._cb = cb
        self._rhob = rhob
        self._attn = attn
        self._rbzb = rbzb
        
        # Validate each ray (check valid portions only)
        for b in range(self._batch_size):
            # Extract valid (non-NaN) portions for validation
            z_ss_b, _ = self._get_valid_slice(z_ss[b])
            rp_ss_b, _ = self._get_valid_slice(rp_ss[b])
            cw_b, _ = self._get_valid_slice(cw[b])
            z_sb_b, _ = self._get_valid_slice(z_sb[b])
            rp_sb_b, _ = self._get_valid_slice(rp_sb[b])
            cb_b, _ = self._get_valid_slice(cb[b])
            rhob_b, _ = self._get_valid_slice(rhob[b])
            attn_b, _ = self._get_valid_slice(attn[b])
            rbzb_b, _ = self._get_valid_slice(rbzb[b])
            
            # Check source/receiver depths
            if not (z_ss_b[0] <= self._zs <= z_ss_b[-1]):
                raise ValueError(f'Ray {b}: Source depth {self._zs}m outside range [{z_ss_b[0]}, {z_ss_b[-1]}]m')
            if not (z_ss_b[0] <= self._zr <= z_ss_b[-1]):
                raise ValueError(f'Ray {b}: Receiver depth {self._zr}m outside range [{z_ss_b[0]}, {z_ss_b[-1]}]m')
            
            # Check water SSP dimensions (using valid portions only)
            num_depths_w = z_ss_b.size
            num_ranges_w = rp_ss_b.size
            cw_dims = cw_b.shape
            if not ((cw_dims[0] == num_depths_w) and (cw_dims[1] == num_ranges_w)):
                raise ValueError(f'Ray {b}: z_ss ({num_depths_w}), rp_ss ({num_ranges_w}), cw {cw_dims} inconsistent')
            
            # Check seabed dimensions
            num_depths_sb = z_sb_b.size
            num_ranges_sb = rp_sb_b.size
            for prof, name in zip([cb_b, rhob_b, attn_b], ['cb', 'rhob', 'attn']):
                if (prof.shape[0] != num_depths_sb) or (prof.shape[1] != num_ranges_sb):
                    raise ValueError(f'Ray {b}: z_sb ({num_depths_sb}), rp_sb ({num_ranges_sb}), {name} {prof.shape} inconsistent')
            
            # Check bathymetry vs sound speed depth
            if rbzb_b[:, 1].max() > z_ss_b[-1]:
                raise ValueError(f'Ray {b}: Max bathy ({rbzb_b[:, 1].max()}m) > max SSP depth ({z_ss_b[-1]}m)')
        
        # Range-dependence flags (use first ray's valid portion)
        rp_ss_0, _ = self._get_valid_slice(self._rp_ss[0])
        rp_sb_0, _ = self._get_valid_slice(self._rp_sb[0])
        rbzb_0, _ = self._get_valid_slice(self._rbzb[0])
        self.rd_ss = rp_ss_0.size > 1
        self.rd_sb = rp_sb_0.size > 1
        self.rd_bt = rbzb_0.shape[0] > 1
        
        # MEMORY OPTIMIZATION: Transfer only light arrays to GPU
        # Heavy arrays (_cw, _cb, _rhob, _attn) are streamed on-demand in profl()
        # This saves massive VRAM for large batch sizes with varying-length profiles
        
        # Light arrays: indices and bathymetry (always needed on GPU)
        self._z_ss = cupy.asarray(self._z_ss)
        self._rp_ss = cupy.asarray(self._rp_ss)
        self._z_sb = cupy.asarray(self._z_sb)
        self._rp_sb = cupy.asarray(self._rp_sb)
        self._rbzb = cupy.asarray(self._rbzb)
        
        # Heavy arrays: environment profiles (keep on CPU, stream to GPU in profl())
        # These remain NumPy arrays: [Batch, Nz_in, Nr]
        # self._cw, self._cb, self._rhob, self._attn stay as NumPy
        # Benefit: Saves GBs of VRAM for large batches with NaN-padded profiles

    def get_params(self, **kwargs):
        """
        Get parameters from keyword arguments.
        All inputs are batched, compute per-ray values.
        """
        self._np = kwargs.get('np', CuPyRAM._np_default)
        
        # Benchmarking mode: time only _propagate_step() calls (pure GPU compute)
        self._benchmark_propagation = kwargs.get('benchmark_propagation', False)
        if self._benchmark_propagation:
            self.propagation_time = 0.0  # Accumulator for pure propagation time

        # Compute per-ray c0 values (always batched now)
        if 'c0' in kwargs:
            # If c0 provided, use it for all rays
            self._c0 = kwargs['c0']
            self._c0_array = numpy.full(self._batch_size, kwargs['c0'])
        else:
            # Compute per-ray c0 from each ray's profile (filter NaN padding)
            # Ensures perfect numerical agreement with CPU
            # Note: self._cw is now kept on CPU (NumPy) for memory optimization
            self._c0_array = numpy.array([
                (numpy.nanmean(self._cw[b, :, 0]) if len(self._cw[b].shape) > 1 else numpy.nanmean(self._cw[b]))
                for b in range(self._batch_size)
            ])
            # Use mean c0 for shared parameters (dr, dz, lambda)
            self._c0 = numpy.mean(self._c0_array)

        # Use maximum frequency for grid resolution (conservative for all frequencies)
        max_freq = numpy.max(self._freqs)
        self._lambda = self._c0 / max_freq

        # dr and dz based on 1500m/s for sensible output steps, using max frequency
        self._dr = kwargs.get('dr', self._np * 1500 / max_freq)
        self._dz = kwargs.get('dz', CuPyRAM._dzf * 1500 / max_freq)
        
        # Log grid resolution for multi-frequency mode
        if self._n_freq > 1:
            print(f"  Grid resolution: dr={self._dr:.1f}m, dz={self._dz:.3f}m (based on max_freq={max_freq:.1f} Hz)")

        self._ndr = kwargs.get('ndr', CuPyRAM._ndr_default)
        self._ndz = kwargs.get('ndz', CuPyRAM._ndz_default)

        # Compute zmplt: maximum bathymetry depth across all rays (filter NaN)
        # After check_inputs(), these are always CuPy arrays
        rbzb_cpu = cupy.asnumpy(self._rbzb)
        rp_ss_cpu = cupy.asnumpy(self._rp_ss)
        rp_sb_cpu = cupy.asnumpy(self._rp_sb)
        
        self._zmplt = kwargs.get('zmplt', 
                                 max(numpy.nanmax(rbzb[:, 1]) for rbzb in rbzb_cpu))

        # Compute rmax: maximum range across all rays (filter NaN)
        rmax_default = max(
            numpy.max([numpy.nanmax(rp_ss), numpy.nanmax(rp_sb), numpy.nanmax(rbzb[:, 0])])
            for rp_ss, rp_sb, rbzb in zip(rp_ss_cpu, rp_sb_cpu, rbzb_cpu)
        )
        self._rmax = kwargs.get('rmax', rmax_default)

        self._ns = kwargs.get('ns', CuPyRAM._ns_default)
        self._rs = kwargs.get('rs', self._rmax + self._dr)

        self._lyrw = kwargs.get('lyrw', CuPyRAM._lyrw_default)

        self._id = kwargs.get('id', CuPyRAM._id_default)

        self.proc_time = None

    @nvtx.annotate("CuPyRAM.setup", color="blue")
    def setup(self):
        """
        Initialize parameters, acoustic field, and matrices.
        All inputs are batched arrays.
        """
        # Extend bathymetry to rmax if needed (per-ray)
        # Note: We need to update the NaN-padded array, potentially growing it
        max_rbzb_len = 0
        extended_rbzb = []
        for i in range(self._batch_size):
            # Get valid (non-NaN) portion of bathymetry
            rbzb_valid, _ = self._get_valid_slice(self._rbzb[i])
            if rbzb_valid[-1, 0] < self._rmax:
                # Extend
                extended = numpy.append(
                    rbzb_valid,
                    numpy.array([[self._rmax, rbzb_valid[-1, 1]]]),
                    axis=0
                )
                extended_rbzb.append(extended)
            else:
                extended_rbzb.append(rbzb_valid)
            max_rbzb_len = max(max_rbzb_len, extended_rbzb[-1].shape[0])
        
        # Re-pad if needed (some arrays may have grown)
        if max_rbzb_len > self._rbzb.shape[1]:
            # Need to re-create with larger padding on CPU, then transfer to GPU
            new_rbzb = numpy.full((self._batch_size, max_rbzb_len, 2), numpy.nan)
            for i in range(self._batch_size):
                new_rbzb[i, :extended_rbzb[i].shape[0], :] = extended_rbzb[i]
            self._rbzb = cupy.asarray(new_rbzb)
        else:
            # Fits in existing padding, update on CPU then transfer to GPU
            rbzb_cpu = cupy.asnumpy(self._rbzb)
            for i in range(self._batch_size):
                rbzb_cpu[i, :extended_rbzb[i].shape[0], :] = extended_rbzb[i]
                # Clear any old data beyond the new valid length
                if extended_rbzb[i].shape[0] < rbzb_cpu.shape[1]:
                    rbzb_cpu[i, extended_rbzb[i].shape[0]:, :] = numpy.nan
            self._rbzb = cupy.asarray(rbzb_cpu)

        self.eta = 1 / (40 * numpy.pi * numpy.log10(numpy.exp(1)))
        self.ib = [0] * self._batch_size  # Bathymetry pair index per ray
        self.mdr_gpu = cupy.array([0], dtype=cupy.int32)  # Output range counter (on GPU)
        self.mdr = 0  # Host copy for compatibility
        self.r = self._dr
        ri = self._zr / self._dz
        self.ir = int(numpy.floor(ri))  # Receiver depth index
        self.dir = ri - self.ir  # Offset
        
        # Adjust seabed depths relative to deepest water profile point (per-ray)
        for i in range(self._batch_size):
            # Get valid portions (filter NaN padding)
            z_ss_valid, _ = self._get_valid_slice(self._z_ss[i])
            z_sb_valid, z_sb_len = self._get_valid_slice(self._z_sb[i])
            
            # Add offset and update valid portion (on GPU)
            z_sb_adjusted = z_sb_valid + z_ss_valid[-1]
            self._z_sb[i, :z_sb_len] = cupy.asarray(z_sb_adjusted)
        
        # Compute zmax_sb from valid portions only
        # After check_inputs(), self._z_sb is always a CuPy array
        z_sb_cpu = cupy.asnumpy(self._z_sb)
        zmax_sb = max(numpy.nanmax(z_sb[:z_sb_len]) 
                      for z_sb, (_, z_sb_len) in 
                      zip(z_sb_cpu, [self._get_valid_slice(z_sb_cpu[i]) for i in range(self._n_env)]))
        
        self._zmax = zmax_sb + self._lyrw * self._lambda
        self.nz = int(numpy.floor(self._zmax / self._dz)) - 1  # Number of depth grid points - 2
        self.nzplt = int(numpy.floor(self._zmplt / self._dz))  # Deepest output grid point
        
        # Initial bathymetry index (per-environment) - use valid portions only
        iz_list = []
        for i in range(self._n_env):
            rbzb_valid, _ = self._get_valid_slice(self._rbzb[i])
            iz_val = int(numpy.floor(rbzb_valid[0, 1] / self._dz))
            iz_list.append(max(1, min(self.nz - 1, iz_val)))
        
        # Create on GPU [N_env]
        self.iz = cupy.array(iz_list, dtype=cupy.int64)

        # === SUPER-BATCH ARCHITECTURE: Split arrays by frequency dependence ===
        # Environment-tied arrays [Nz+2, N_env]: geometry and raw medium properties
        # Field-tied arrays [Nz+2, N_calc]: acoustic fields and frequency-dependent terms
        
        # Environment-tied arrays [Nz+2, N_env] - SINGLE PRECISION
        # Geometry (frequency-independent)
        self.f1 = cupy.zeros([self.nz + 2, self._n_env], dtype=cupy.float32)
        self.f2 = cupy.zeros([self.nz + 2, self._n_env], dtype=cupy.float32)
        self.f3 = cupy.zeros([self.nz + 2, self._n_env], dtype=cupy.float32)
        # Raw medium properties
        self.cw = cupy.zeros([self.nz + 2, self._n_env], dtype=cupy.float32)
        self.cb = cupy.zeros([self.nz + 2, self._n_env], dtype=cupy.float32)
        self.rhob = cupy.zeros([self.nz + 2, self._n_env], dtype=cupy.float32)
        self.attn = cupy.zeros([self.nz + 2, self._n_env], dtype=cupy.float32)
        # Acoustic impedance terms (frequency-independent)
        self.alpw = cupy.zeros([self.nz + 2, self._n_env], dtype=cupy.float32)
        self.alpb = cupy.zeros([self.nz + 2, self._n_env], dtype=cupy.float32)
        
        # Field-tied arrays [Nz+2, N_calc] - Acoustic fields and frequency-dependent terms
        # Solution vectors in DOUBLE PRECISION
        self.u = cupy.zeros([self.nz + 2, self._total_batch], dtype=numpy.complex128)
        self.v = cupy.zeros([self.nz + 2, self._total_batch], dtype=numpy.complex128)
        
        # Wavenumber-dependent arrays in SINGLE PRECISION
        self.ksq = cupy.zeros([self.nz + 2, self._total_batch], dtype=cupy.complex64)
        # CRITICAL: ksqw and ksqb depend on frequency (omega) - must be N_calc
        self.ksqw = cupy.zeros([self.nz + 2, self._total_batch], dtype=cupy.float32)
        self.ksqb = cupy.zeros([self.nz + 2, self._total_batch], dtype=cupy.complex64)
        
        # Workspace arrays [Nz+2, N_calc] - conditional allocation based on kernel type
        if FUSED_KERNEL:
            # Optimized fused kernel: Only 2 workspace arrays
            self.tdma_upper = cupy.zeros([self.nz + 2, self._total_batch], dtype=cupy.complex64)
            self.tdma_rhs = cupy.zeros([self.nz + 2, self._total_batch], dtype=cupy.complex64)
        else:
            # Legacy product formulation: 6 arrays
            self.r1 = cupy.zeros([self.nz + 2, self._total_batch], dtype=cupy.complex64)
            self.r2 = cupy.zeros([self.nz + 2, self._total_batch], dtype=cupy.complex64)
            self.r3 = cupy.zeros([self.nz + 2, self._total_batch], dtype=cupy.complex64)
            self.s1 = cupy.zeros([self.nz + 2, self._total_batch], dtype=cupy.complex64)
            self.s2 = cupy.zeros([self.nz + 2, self._total_batch], dtype=cupy.complex64)
            self.s3 = cupy.zeros([self.nz + 2, self._total_batch], dtype=cupy.complex64)
        
        # Padé coefficients [np, N_calc] - DOUBLE PRECISION for accuracy
        self.pd1 = cupy.zeros([self._np, self._total_batch], dtype=cupy.complex128)
        self.pd2 = cupy.zeros([self._np, self._total_batch], dtype=cupy.complex128)
        
        # Precompute k0 array [N_calc] - broadcast frequencies and c0 values
        # Order: [env0_freq0, env0_freq1, ..., env0_freqN, env1_freq0, ...]
        omegas = 2 * numpy.pi * self._freqs  # [N_freq]
        omegas_tiled = numpy.tile(omegas, self._n_env)  # [N_calc]
        c0_repeated = numpy.repeat(self._c0_array, self._n_freq)  # [N_calc]
        self.k0 = cupy.asarray(omegas_tiled / c0_repeated)
        
        # Cache omega_arr and k0_sq on GPU for profl() (avoids repeated CPU->GPU transfers)
        self._omega_arr_cached = cupy.asarray(omegas_tiled)[None, :]  # [1, N_calc] on GPU
        self._k0_sq_cached = self.k0**2
        if self._k0_sq_cached.ndim == 1:
            self._k0_sq_cached = self._k0_sq_cached[None, :]  # [1, N_calc]
        
        nvr = int(numpy.floor(self._rmax / (self._dr * self._ndr)))
        self._rmax = nvr * self._dr * self._ndr
        nvz = int(numpy.floor(self.nzplt / self._ndz))
        self.vr = numpy.arange(1, nvr + 1) * self._dr * self._ndr
        self.vz = numpy.arange(1, nvz + 1) * self._dz * self._ndz
        
        # Output arrays [N_calc, ...] on GPU
        # Grid outputs (tlg, cpg) can be disabled to save VRAM
        if self._compute_grids:
            nvz_alloc = nvz
        else:
            nvz_alloc = 1
        
        # Allocate output arrays for N_calc (will be reshaped to [N_env, N_freq, ...] in run())
        self.tll = cupy.zeros([self._total_batch, nvr], dtype=numpy.float64)
        self.tlg = cupy.zeros([self._total_batch, nvz_alloc, nvr], dtype=numpy.float64)
        self.cpl = cupy.zeros([self._total_batch, nvr], dtype=numpy.complex128)
        self.cpg = cupy.zeros([self._total_batch, nvz_alloc, nvr], dtype=numpy.complex128)
        
        self.tlc_gpu = cupy.array([-1], dtype=cupy.int32)  # TL output range counter (on GPU)
        self.tlc = -1  # Host copy for compatibility

        # Per-environment profile range indices [N_env] on GPU
        self.ss_ind = cupy.zeros(self._n_env, dtype=cupy.int32)
        self.sb_ind = cupy.zeros(self._n_env, dtype=cupy.int32)
        self.bt_ind = cupy.zeros(self._n_env, dtype=cupy.int32)

        # The initial profiles and starting field
        self.profl()
        self.selfs()  # Initialize acoustic field on GPU
        
        # Only compute output if grids are enabled
        if self._compute_grids:
            self.mdr_gpu, self.tlc_gpu = self._outpt()

        # Compute Padé coefficients per (env, freq) pair
        with nvtx.annotate("compute_pade_batch", color="purple"):
            pd1_list = []
            pd2_list = []
            
            # Order: [env0_freq0, env0_freq1, ..., env0_freqN, env1_freq0, ...]
            for env_idx in range(self._n_env):
                for freq_idx in range(self._n_freq):
                    pd1, pd2 = compute_pade_coefficients(
                        freq=self._freqs[freq_idx],
                        c0=self._c0_array[env_idx],
                        np_pade=self._np, ns=self._ns, 
                        dr=self._dr, ip=1
                    )
                    pd1_list.append(pd1)
                    pd2_list.append(pd2)
            
            # Stack and transfer to GPU: [N_calc, np] -> transpose to [np, N_calc]
            pd1_stacked = numpy.array(pd1_list)  # [N_calc, np]
            pd2_stacked = numpy.array(pd2_list)
            self.pd1 = cupy.asarray(pd1_stacked.T)  # [np, N_calc]
            self.pd2 = cupy.asarray(pd2_stacked.T)

    @nvtx.annotate("CuPyRAM.profl", color="cyan")
    def profl(self):
        """
        Set up profiles. Interpolate per-ray environments on GPU.
        Uses batched CUDA kernel with CPU-GPU streaming for environment data.
        
        MEMORY OPTIMIZATION: Heavy environment arrays (_cw, _cb, _rhob, _attn)
        are stored on CPU and only active profiles are streamed to GPU.
        This saves GBs of VRAM for large batches with varying-length profiles.
        """
        z = cupy.linspace(0, self._zmax, self.nz + 2)
        
        # CPU-GPU STREAMING: Slice active profiles on CPU, transfer small slice to GPU
        # Get active indices from GPU to CPU (N_env size)
        with nvtx.annotate("get_active_indices", color="yellow"):
            ss_ind_cpu = cupy.asnumpy(self.ss_ind)
            sb_ind_cpu = cupy.asnumpy(self.sb_ind)
            batch_indices_cpu = numpy.arange(self._n_env)
        
        # SLICE ON CPU (Host RAM) - environment arrays are still NumPy
        # Extract only the current active profile for each environment: [N_env, Nz_in]
        with nvtx.annotate("slice_profiles_cpu", color="orange"):
            current_cw_prof_cpu = self._cw[batch_indices_cpu, :, ss_ind_cpu]
            current_cb_prof_cpu = self._cb[batch_indices_cpu, :, sb_ind_cpu]
            current_rhob_prof_cpu = self._rhob[batch_indices_cpu, :, sb_ind_cpu]
            current_attn_prof_cpu = self._attn[batch_indices_cpu, :, sb_ind_cpu]
        
        # TRANSFER TO GPU (PCIe) - only the active slice
        with nvtx.annotate("transfer_profiles_to_gpu", color="green"):
            current_cw_prof_gpu = cupy.asarray(current_cw_prof_cpu)
            current_cb_prof_gpu = cupy.asarray(current_cb_prof_cpu)
            current_rhob_prof_gpu = cupy.asarray(current_rhob_prof_cpu)
            current_attn_prof_gpu = cupy.asarray(current_attn_prof_cpu)
        
        # Light arrays already on GPU
        z_ss_gpu = self._z_ss
        z_sb_gpu = self._z_sb
        
        # Pre-calculate absorbing layer width per environment
        lyrw_lambda_arr = cupy.full(self._n_env, self._lyrw * self._lambda)
            
        # Run batched interpolation kernel on GPU (N_env profiles)
        profl_cuda_launcher(
            z, z_ss_gpu, current_cw_prof_gpu, 
            z_sb_gpu, current_cb_prof_gpu, current_rhob_prof_gpu, current_attn_prof_gpu,
            self.cw, self.cb, self.rhob, self.attn,
            lyrw_lambda_arr, attnf=10.0
        )
        
        # === STEP 2: Compute frequency-independent derived quantities (N_env) ===
        # self._c0_array is always NumPy array (created in get_params())
        c0_ray = cupy.asarray(self._c0_array)[None, :]  # [1, N_env]
        
        # Acoustic impedance terms (frequency-independent)
        self.alpw = cupy.sqrt(self.cw / c0_ray)  # [Nz+2, N_env]
        self.alpb = cupy.sqrt(self.rhob * self.cb / c0_ray)  # [Nz+2, N_env]
        
        # === STEP 3: Compute frequency-dependent ksqw, ksqb (N_calc) ===
        if self._n_freq == 1:
            # --- SINGLE FREQUENCY CASE (Legacy/Simple) ---
            # No expansion needed - arrays are already correct shape
            # We just do the math using standard CuPy operations
            
            # Use cached GPU arrays
            omega_arr = self._omega_arr_cached
            k0_sq = self._k0_sq_cached
            
            # Compute ksqw
            # Note: Use x*x instead of x**2 (2-3x faster)
            omega_over_cw = omega_arr / self.cw
            self.ksqw = omega_over_cw * omega_over_cw - k0_sq
            
            # Compute ksqb
            omega_over_cb = omega_arr / self.cb
            term = omega_over_cb * (1 + 1j * self.eta * self.attn)
            self.ksqb = term * term - k0_sq
        else:
            # --- MULTI-FREQUENCY CASE (Optimized) ---
            # Use broadcasting + In-place Float32 Math to avoid VRAM explosion
            with nvtx.annotate("expand_frequency_arrays", color="magenta"):
                # 1. Create Views (No copy, instantaneous)
                cw_view = self.cw.reshape(self.nz + 2, self._n_env, 1)
                cb_view = self.cb.reshape(self.nz + 2, self._n_env, 1)
                attn_view = self.attn.reshape(self.nz + 2, self._n_env, 1)

                # 2. Cast constants to float32/complex64 ONCE to prevent float64 promotion
                # This prevents the creation of massive 1.2GB/2.4GB temporary arrays
                omega_f32 = self._omega_arr_cached.astype(cupy.float32).reshape(1, self._n_env, self._n_freq)
                k0_sq_f32 = self._k0_sq_cached.astype(cupy.float32).reshape(1, self._n_env, self._n_freq)

                # 3. KSQW: Compute directly into output array (In-place)
                # Create a view of the destination array with the broadcast shape
                ksqw_out = self.ksqw.view()
                ksqw_out.shape = (self.nz + 2, self._n_env, self._n_freq)

                # ksqw = (w/cw)^2 - k0^2
                cupy.divide(omega_f32, cw_view, out=ksqw_out)      # Write w/cw directly to output
                cupy.square(ksqw_out, out=ksqw_out)                # Square in-place
                cupy.subtract(ksqw_out, k0_sq_f32, out=ksqw_out)   # Subtract k0^2 in-place

                # 4. KSQB: Compute directly into output array (In-place)
                ksqb_out = self.ksqb.view()
                ksqb_out.shape = (self.nz + 2, self._n_env, self._n_freq)
                
                # We need one temp for the complex term, but we keep it complex64
                # term = (w/cb) * (1 + 1j*eta*attn)
                eta_f32 = cupy.float32(self.eta)
                
                # Use ksqb_out as temporary storage for (w/cb)
                cupy.divide(omega_f32, cb_view, out=ksqb_out.real) # Real part = w/cb
                ksqb_out.imag[:] = 0                               # Reset imag
                
                # Compute term: multiplies (w/cb) by (1 + 1j*eta*attn)
                # This uses a fused kernel in CuPy
                term = ksqb_out * (1.0 + 1j * eta_f32 * attn_view)
                
                # Final calculation: term^2 - k0^2
                cupy.square(term, out=ksqb_out)
                cupy.subtract(ksqb_out, k0_sq_f32, out=ksqb_out)

                print("Optimized F32 broadcasting running!")
        

    @nvtx.annotate("CuPyRAM._propagate_step", color="red")
    def _propagate_step(self):
        """
        Propagation step: advance solution one range step.
        
        Two implementations:
        - FUSED_KERNEL=True: Sum formulation with on-the-fly matrix generation
          (67% memory savings, 8x arithmetic intensity increase)
        - FUSED_KERNEL=False: Legacy product formulation
          (for validation and comparison)
        """
        
        # Start timing if in benchmark mode
        if self._benchmark_propagation:
            t_prop_start = process_time()
        
        if FUSED_KERNEL:
            # === OPTIMIZED: Fused Sum-Padé Kernel with Super-Batch ===
            # Initialize environment (once per range step)
            with nvtx.annotate("init_profiles", color="purple"):
                matrc_cuda_init_profiles(
                    self.iz, self.iz, self.nz, self.f1, self.f2, self.f3, self.ksq,
                    self.alpw, self.alpb, self.ksqw, self.ksqb, self.rhob,
                    batch_size=self._n_env,  # Environment arrays are N_env size
                    n_freqs=self._n_freq     # For ksq expansion to N_calc
                )
            
            # Fused kernel: all Padé terms with on-the-fly matrix generation
            with nvtx.annotate("fused_pade_solve", color="red"):
                from cupyram.fused_kernel import fused_sum_pade_solve
                fused_sum_pade_solve(
                    self.u, self.u,  # In-place operation (u_in = u_out)
                    self.f1, self.f2, self.f3, self.ksq,
                    self.k0, self._dz, self.iz, self.nz,
                    self.pd1, self.pd2,
                    self.tdma_upper, self.tdma_rhs,
                    self._total_batch,  # N_calc calculations
                    self._n_freq        # For broadcast indexing
                )
        else:
            # === LEGACY: Product Formulation with Super-Batch ===
            # Step 1: Init profiles (once per range step, outside Padé loop)
            with nvtx.annotate("init_profiles", color="purple"):
                matrc_cuda_init_profiles(
                    self.iz, self.iz, self.nz, self.f1, self.f2, self.f3, self.ksq,
                    self.alpw, self.alpb, self.ksqw, self.ksqb, self.rhob,
                    batch_size=self._n_env,
                    n_freqs=self._n_freq
                )
            
            # Step 2: Loop over Padé coefficients
            for j in range(self._np):
                # Extract Padé coefficients for this j: [N_calc] slice (COALESCED!)
                pd1_vals = self.pd1[j, :]  # Row access on [np, N_calc] → coalesced
                pd2_vals = self.pd2[j, :]
                
                # Discretize and decompose for this Padé term
                with nvtx.annotate(f"matrc_j{j}", color="orange"):
                    matrc_cuda_single_pade(
                        self.k0, self._dz, self.iz, self.iz, self.nz,
                        self.f1, self.f2, self.f3, self.ksq,
                        self.r1, self.r2, self.r3, self.s1, self.s2, self.s3,
                        pd1_vals, pd2_vals, batch_size=self._total_batch
                    )
                
                # Solve for this Padé term
                with nvtx.annotate(f"solve_j{j}", color="green"):
                    solve(self.u, self.v, self.s1, self.s2, self.s3,
                          self.r1, self.r2, self.r3, self.iz, self.nz)
        
        # End timing if in benchmark mode
        if self._benchmark_propagation:
            cuda.synchronize()  # Ensure GPU work is complete for accurate timing
            self.propagation_time += process_time() - t_prop_start
    
    def _outpt(self):
        """Compute transmission loss outputs on GPU using CUDA kernel.
        
        Returns:
            (mdr_gpu, tlc_gpu): CuPy arrays on GPU (no D2H transfer for performance)
        """
        with nvtx.annotate("outpt_cuda", color="green"):
            # Wrap u for Numba CUDA (zero-copy, u is already CuPy on GPU)
            u_device = cuda.as_cuda_array(self.u)
            
            # Broadcast f3 [N_env] for output [N_calc]
            # f3 is [Nz+2, N_env], need to repeat for N_calc
            f3_expanded = cupy.repeat(self.f3, self._n_freq, axis=1)  # [Nz+2, N_calc]
            
            # Output arrays are already CuPy (allocated in setup)
            # Pass GPU counters to avoid D2H transfers every iteration
            mdr_gpu, tlc_gpu = outpt_cuda(
                self.r, None, self._ndr, self._ndz, None,
                f3_expanded, u_device, self.dir, self.ir,
                self.tll, self.tlg, self.cpl, self.cpg,
                batch_size=self._total_batch,
                mdr_gpu=self.mdr_gpu, tlc_gpu=self.tlc_gpu
            )
        return mdr_gpu, tlc_gpu

    @nvtx.annotate("CuPyRAM.updat", color="yellow")
    def updat(self):
        """
        Update matrices for range-dependent environment.
        Index updates run in parallel on GPU via CUDA kernel (per-environment).
        """
        # Run parallel index updates on GPU (N_env environments)
        with nvtx.annotate("updat_indices_cuda", color="olive"):
            need_matrc = updat_indices_cuda(
                float(self.r), float(self._dr), float(self._dz), int(self.nz),
                self._rbzb, self.bt_ind, self.iz,
                self._rp_ss, self.ss_ind,
                self._rp_sb, self.sb_ind,
                bool(self.rd_bt), bool(self.rd_ss), bool(self.rd_sb),
                self._n_env  # Environment batch size
            )
            # Note: iz is updated in-place on GPU by updat_indices_cuda
        
        # If any environment needs update, recompute profiles
        # Matrices will be computed in _propagate_step()
        if need_matrc:
            self.profl()

        # Turn off the stability constraints (shared across all calculations)
        # Note: This uses single frequency for simplicity (conservative)
        if self.r >= self._rs:
            self._ns = 0
            self._rs = self._rmax + self._dr
            with nvtx.annotate("compute_pade_stability", color="purple"):
                # Recompute for all (env, freq) pairs
                pd1_list = []
                pd2_list = []
                
                for env_idx in range(self._n_env):
                    for freq_idx in range(self._n_freq):
                        pd1, pd2 = compute_pade_coefficients(
                            freq=self._freqs[freq_idx],
                            c0=self._c0_array[env_idx],
                            np_pade=self._np, ns=self._ns, 
                            dr=self._dr, ip=1
                        )
                        pd1_list.append(pd1)
                        pd2_list.append(pd2)
                
                # Update Padé coefficients for all calculations
                pd1_stacked = numpy.array(pd1_list)
                pd2_stacked = numpy.array(pd2_list)
                self.pd1[:, :] = cupy.asarray(pd1_stacked.T)
                self.pd2[:, :] = cupy.asarray(pd2_stacked.T)

    @nvtx.annotate("CuPyRAM.selfs", color="magenta")
    def selfs(self):
        """
        The self-starter. Initialize acoustic field for all calculations.
        Arrays: u [Nz+2, N_calc], alpw [Nz+2, N_env], k0 [N_calc]
        """
        # Conditions for the delta function
        si = self._zs / self._dz
        _is = int(numpy.floor(si))  # Source depth index
        dis = si - _is  # Offset

        # Initialize u for all calculations (same source position for all)
        # Need to broadcast alpw [N_env] for u [N_calc]
        # env_indices maps calc_idx -> env_idx
        env_indices = cupy.arange(self._total_batch) // self._n_freq  # [N_calc]
        
        # Vectorized GPU operation with broadcasting
        # u: [Nz+2, N_calc], k0: [N_calc], alpw: [Nz+2, N_env] -> broadcast via env_indices
        self.u[_is, :] = (1 - dis) * cupy.sqrt(2 * cupy.pi / self.k0) / \
            (self._dz * self.alpw[_is, env_indices])
        self.u[_is + 1, :] = dis * cupy.sqrt(2 * cupy.pi / self.k0) / \
            (self._dz * self.alpw[_is + 1, env_indices])

        # Divide the delta function by (1-X)**2 to get a smooth rhs
        self.pd1[0, :] = 0  # First Padé coefficient for all batches
        self.pd2[0, :] = -1

        # Override np to 1 for initial smoothing
        old_np = self._np
        self._np = 1
        
        # Solve twice for smoothing (using device_arrays - fast path)
        for _ in range(2):
            self._propagate_step()

        # Restore np and apply full operator (1-X)**2*(1+X)**(-1/4)*exp(ci*k0*r*sqrt(1+X))
        self._np = old_np
        
        # Compute Padé coefficients per (env, freq) pair
        with nvtx.annotate("compute_pade_batch_selfs", color="purple"):
            pd1_list = []
            pd2_list = []
            
            # Order: [env0_freq0, env0_freq1, ..., env0_freqN, env1_freq0, ...]
            for env_idx in range(self._n_env):
                for freq_idx in range(self._n_freq):
                    pd1, pd2 = compute_pade_coefficients(
                        freq=self._freqs[freq_idx],
                        c0=self._c0_array[env_idx],
                        np_pade=self._np, ns=self._ns, 
                        dr=self._dr, ip=2
                    )
                    pd1_list.append(pd1)
                    pd2_list.append(pd2)
            
            # Stack and transfer to GPU: [N_calc, np] -> transpose to [np, N_calc]
            pd1_stacked = numpy.array(pd1_list)  # [N_calc, np]
            pd2_stacked = numpy.array(pd2_list)
            self.pd1 = cupy.asarray(pd1_stacked.T)  # [np, N_calc]
            self.pd2 = cupy.asarray(pd2_stacked.T)
        
        # Apply full Padé operator
        self._propagate_step()
