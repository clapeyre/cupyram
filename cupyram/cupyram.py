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
        self._freq, self._zs, self._zr = freq, zs, zr
        
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

            self.mdr, self.tlc = self._outpt()

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
        
        # PyRAM API compatibility: squeeze batch dimension for batch_size=1
        # Input: single arrays → normalized to [1, ...]
        # Output: [1, ...] → squeezed back to single arrays
        if self._batch_size == 1:
            self.tll = self.tll[0] if self.tll.ndim > 1 else self.tll
            self.cpl = self.cpl[0] if self.cpl.ndim > 1 else self.cpl
            
            # Only squeeze grids if they were computed
            if self._compute_grids:
                self.tlg = self.tlg[0] if self.tlg.ndim > 2 else self.tlg
                self.cpg = self.cpg[0] if self.cpg.ndim > 2 else self.cpg

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
        
        # Transfer environment arrays to GPU (after validation)
        self._z_ss = cupy.asarray(self._z_ss)
        self._rp_ss = cupy.asarray(self._rp_ss)
        self._cw = cupy.asarray(self._cw)
        self._z_sb = cupy.asarray(self._z_sb)
        self._rp_sb = cupy.asarray(self._rp_sb)
        self._cb = cupy.asarray(self._cb)
        self._rhob = cupy.asarray(self._rhob)
        self._attn = cupy.asarray(self._attn)
        self._rbzb = cupy.asarray(self._rbzb)

    def get_params(self, **kwargs):
        """
        Get parameters from keyword arguments.
        All inputs are batched, compute per-ray values.
        """
        self._np = kwargs.get('np', CuPyRAM._np_default)

        # Compute per-ray c0 values (always batched now)
        if 'c0' in kwargs:
            # If c0 provided, use it for all rays
            self._c0 = kwargs['c0']
            self._c0_array = numpy.full(self._batch_size, kwargs['c0'])
        else:
            # Compute per-ray c0 from each ray's profile (filter NaN padding)
            # Ensures perfect numerical agreement with CPU
            # After check_inputs(), self._cw is always a CuPy array
            cw_cpu = cupy.asnumpy(self._cw)
            self._c0_array = numpy.array([
                (numpy.nanmean(cw_cpu[b, :, 0]) if len(cw_cpu[b].shape) > 1 else numpy.nanmean(cw_cpu[b]))
                for b in range(self._batch_size)
            ])
            # Use mean c0 for shared parameters (dr, dz, lambda)
            self._c0 = numpy.mean(self._c0_array)

        self._lambda = self._c0 / self._freq

        # dr and dz based on 1500m/s for sensible output steps
        self._dr = kwargs.get('dr', self._np * 1500 / self._freq)
        self._dz = kwargs.get('dz', CuPyRAM._dzf * 1500 / self._freq)

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
        self.mdr = 0  # Output range counter
        self.r = self._dr
        self.omega = 2 * numpy.pi * self._freq
        ri = self._zr / self._dz
        self.ir = int(numpy.floor(ri))  # Receiver depth index
        self.dir = ri - self.ir  # Offset
        
        # Compute per-ray k0 values on CPU then transfer to GPU
        self.k0 = self.omega / self._c0_array  # Array [batch_size]
        self.k0 = cupy.asarray(self.k0)
        
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
                      zip(z_sb_cpu, [self._get_valid_slice(z_sb_cpu[i]) for i in range(self._batch_size)]))
        
        self._zmax = zmax_sb + self._lyrw * self._lambda
        self.nz = int(numpy.floor(self._zmax / self._dz)) - 1  # Number of depth grid points - 2
        self.nzplt = int(numpy.floor(self._zmplt / self._dz))  # Deepest output grid point
        
        # Initial bathymetry index (per-ray) - use valid portions only
        iz_list = []
        for i in range(self._batch_size):
            rbzb_valid, _ = self._get_valid_slice(self._rbzb[i])
            iz_val = int(numpy.floor(rbzb_valid[0, 1] / self._dz))
            iz_list.append(max(1, min(self.nz - 1, iz_val)))
        
        # Create on GPU
        self.iz = cupy.array(iz_list, dtype=cupy.int64)

        # Allocate batched arrays [Nz+2, Batch] for coalesced memory access
        # All arrays transposed: adjacent threads (varying batch) access adjacent memory
        # u and v are GPU-native (CuPy) - solution vectors in DOUBLE PRECISION
        self.u = cupy.zeros([self.nz + 2, self._batch_size], dtype=numpy.complex128)  # GPU (double precision)
        self.v = cupy.zeros([self.nz + 2, self._batch_size], dtype=numpy.complex128)  # GPU (double precision)
        
        # SINGLE PRECISION intermediate arrays (recomputed each step, not accumulated)
        # Memory savings: ~50% reduction in VRAM for these arrays
        self.ksq = cupy.zeros([self.nz + 2, self._batch_size], dtype=cupy.complex64)  # GPU (single precision)
        
        # Matrix coefficients [Nz+2, Batch] - no Np dimension (computed one Padé step at a time)
        self.r1 = cupy.zeros([self.nz + 2, self._batch_size], dtype=cupy.complex64)  # SINGLE PRECISION
        self.r2 = cupy.zeros([self.nz + 2, self._batch_size], dtype=cupy.complex64)  # SINGLE PRECISION
        self.r3 = cupy.zeros([self.nz + 2, self._batch_size], dtype=cupy.complex64)  # SINGLE PRECISION
        self.s1 = cupy.zeros([self.nz + 2, self._batch_size], dtype=cupy.complex64)  # SINGLE PRECISION
        self.s2 = cupy.zeros([self.nz + 2, self._batch_size], dtype=cupy.complex64)  # SINGLE PRECISION
        self.s3 = cupy.zeros([self.nz + 2, self._batch_size], dtype=cupy.complex64)  # SINGLE PRECISION
        
        # Per-ray Padé coefficients [np, batch_size] on GPU (transposed for coalesced access)
        # Keep double precision for accuracy (mathematical constants)
        self.pd1 = cupy.zeros([self._np, self._batch_size], dtype=cupy.complex128)
        self.pd2 = cupy.zeros([self._np, self._batch_size], dtype=cupy.complex128)

        # Per-ray environment arrays [Nz+2, Batch] on GPU - transposed for coalescing
        self.alpw = cupy.zeros([self.nz + 2, self._batch_size], dtype=cupy.float32)  # SINGLE PRECISION
        self.alpb = cupy.zeros([self.nz + 2, self._batch_size], dtype=cupy.float32)  # SINGLE PRECISION
        self.ksqw = cupy.zeros([self.nz + 2, self._batch_size], dtype=cupy.float32)  # SINGLE PRECISION
        self.ksqb = cupy.zeros([self.nz + 2, self._batch_size], dtype=cupy.complex64)  # SINGLE PRECISION
        self.cw = cupy.zeros([self.nz + 2, self._batch_size], dtype=cupy.float32)  # SINGLE PRECISION
        self.cb = cupy.zeros([self.nz + 2, self._batch_size], dtype=cupy.float32)  # SINGLE PRECISION
        self.rhob = cupy.zeros([self.nz + 2, self._batch_size], dtype=cupy.float32)  # SINGLE PRECISION
        self.attn = cupy.zeros([self.nz + 2, self._batch_size], dtype=cupy.float32)  # SINGLE PRECISION
        
        # Batched working arrays [Nz+2, Batch] on GPU
        # Transposed for coalesced memory access in matrc_cuda
        self.f1 = cupy.zeros([self.nz + 2, self._batch_size], dtype=cupy.float32)  # SINGLE PRECISION
        self.f2 = cupy.zeros([self.nz + 2, self._batch_size], dtype=cupy.float32)  # SINGLE PRECISION
        self.f3 = cupy.zeros([self.nz + 2, self._batch_size], dtype=cupy.float32)  # SINGLE PRECISION
        nvr = int(numpy.floor(self._rmax / (self._dr * self._ndr)))
        self._rmax = nvr * self._dr * self._ndr
        nvz = int(numpy.floor(self.nzplt / self._ndz))
        self.vr = numpy.arange(1, nvr + 1) * self._dr * self._ndr
        self.vz = numpy.arange(1, nvz + 1) * self._dz * self._ndz
        
        # Batched output arrays [batch_size, ...] on GPU
        # Grid outputs (tlg, cpg) can be disabled to save VRAM
        if self._compute_grids:
            # Full grid output
            nvz_alloc = nvz
        else:
            # Minimal dummy arrays (1x1 grid) when grids are disabled
            nvz_alloc = 1
        
        # Allocate directly on GPU VRAM
        self.tll = cupy.zeros([self._batch_size, nvr], dtype=numpy.float64)
        self.tlg = cupy.zeros([self._batch_size, nvz_alloc, nvr], dtype=numpy.float64)
        self.cpl = cupy.zeros([self._batch_size, nvr], dtype=numpy.complex128)
        self.cpg = cupy.zeros([self._batch_size, nvz_alloc, nvr], dtype=numpy.complex128)
        
        self.tlc = -1  # TL output range counter

        # Per-ray profile range indices on GPU for parallel updates
        self.ss_ind = cupy.zeros(self._batch_size, dtype=cupy.int32)
        self.sb_ind = cupy.zeros(self._batch_size, dtype=cupy.int32)
        self.bt_ind = cupy.zeros(self._batch_size, dtype=cupy.int32)

        # The initial profiles and starting field
        self.profl()
        self.selfs()  # Initialize acoustic field on GPU
        self.mdr, self.tlc = self._outpt()

        # Compute per-ray Padé coefficients in parallel on CPU, transfer to GPU
        with nvtx.annotate("compute_pade_batch", color="purple"):
            pd1_batch, pd2_batch = compute_pade_coefficients_batch(
                freq=self._freq, c0_array=self._c0_array, 
                np_pade=self._np, ns=self._ns, 
                dr=self._dr, ip=1, max_workers=self._max_workers
            )
            self.pd1 = cupy.asarray(pd1_batch)  # Transfer to GPU: [np, batch]
            self.pd2 = cupy.asarray(pd2_batch)

    @nvtx.annotate("CuPyRAM.profl", color="cyan")
    def profl(self):
        """
        Set up profiles. Interpolate per-ray environments on GPU.
        Uses batched CUDA kernel for parallel interpolation.
        """
        z = cupy.linspace(0, self._zmax, self.nz + 2)
        
        # Select Active Profiles (Batched Fancy Indexing)
        # Convert GPU indices to CPU for indexing (environment arrays already on GPU)
        ss_ind_cpu = cupy.asnumpy(self.ss_ind)
        sb_ind_cpu = cupy.asnumpy(self.sb_ind)
        batch_indices_cpu = numpy.arange(self._batch_size)
        
        # Index GPU arrays by transferring indices to CPU, indexing, then back to GPU
        # This is necessary because CuPy doesn't support advanced indexing with per-element indices
        current_cw_prof = self._cw[batch_indices_cpu, :, ss_ind_cpu]
        current_cb_prof = self._cb[batch_indices_cpu, :, sb_ind_cpu]
        current_rhob_prof = self._rhob[batch_indices_cpu, :, sb_ind_cpu]
        current_attn_prof = self._attn[batch_indices_cpu, :, sb_ind_cpu]
        
        # Ensure everything is on GPU
        z_ss_gpu = cupy.asarray(self._z_ss)
        z_sb_gpu = cupy.asarray(self._z_sb)
        current_cw_prof_gpu = cupy.asarray(current_cw_prof)
        current_cb_prof_gpu = cupy.asarray(current_cb_prof)
        current_rhob_prof_gpu = cupy.asarray(current_rhob_prof)
        current_attn_prof_gpu = cupy.asarray(current_attn_prof)
        
        # Pre-calculate absorbing layer width per ray
        # self._lyrw and self._lambda are scalars, so this is always a float
        lyrw_lambda_arr = cupy.full(self._batch_size, self._lyrw * self._lambda)
            
        # Run batched interpolation kernel on GPU
        profl_cuda_launcher(
            z, z_ss_gpu, current_cw_prof_gpu, 
            z_sb_gpu, current_cb_prof_gpu, current_rhob_prof_gpu, current_attn_prof_gpu,
            self.cw, self.cb, self.rhob, self.attn,
            lyrw_lambda_arr, attnf=10.0
        )
        
        # Compute derived quantities (vectorized on GPU)
        # Arrays are [Nz+2, Batch], need to broadcast k0_sq and c0_ray correctly
        k0_sq = self.k0**2
        if k0_sq.ndim == 1:
            k0_sq = k0_sq[None, :]  # Broadcast to [1, Batch] for [Nz+2, Batch] arrays
            
        # self._c0_array is always NumPy array (created in get_params())
        c0_ray = cupy.asarray(self._c0_array)[None, :]
        
        self.ksqw = (self.omega / self.cw)**2 - k0_sq
        term = (self.omega / self.cb) * (1 + 1j * self.eta * self.attn)
        self.ksqb = term**2 - k0_sq
        self.alpw = cupy.sqrt(self.cw / c0_ray)
        self.alpb = cupy.sqrt(self.rhob * self.cb / c0_ray)

    @nvtx.annotate("CuPyRAM._propagate_step", color="red")
    def _propagate_step(self):
        """
        Fused matrc-solve propagation step.
        Interleaves matrc computation and solve for each Padé coefficient.
        Memory-efficient: computes r1...s3 one Padé term at a time.
        Arrays: [Nz+2, Batch] for coalesced memory access.
        """
        
        # Step 1: Init profiles (once per range step, outside Padé loop)
        with nvtx.annotate("init_profiles", color="purple"):
            matrc_cuda_init_profiles(
                self.iz, self.iz, self.nz, self.f1, self.f2, self.f3, self.ksq,
                self.alpw, self.alpb, self.ksqw, self.ksqb, self.rhob,
                batch_size=self._batch_size
            )
        
        # Step 2: Loop over Padé coefficients
        for j in range(self._np):
            # Extract Padé coefficients for this j: [Batch] slice (COALESCED!)
            pd1_vals = self.pd1[j, :]  # Row access on [np, batch] → coalesced
            pd2_vals = self.pd2[j, :]
            
            # Discretize and decompose for this Padé term
            with nvtx.annotate(f"matrc_j{j}", color="orange"):
                matrc_cuda_single_pade(
                    self.k0, self._dz, self.iz, self.iz, self.nz,
                    self.f1, self.f2, self.f3, self.ksq,
                    self.r1, self.r2, self.r3, self.s1, self.s2, self.s3,
                    pd1_vals, pd2_vals, batch_size=self._batch_size
                )
            
            # Solve for this Padé term
            with nvtx.annotate(f"solve_j{j}", color="green"):
                solve(self.u, self.v, self.s1, self.s2, self.s3,
                      self.r1, self.r2, self.r3, self.iz, self.nz)
    
    def _outpt(self):
        """Compute transmission loss outputs on GPU using CUDA kernel."""
        with nvtx.annotate("outpt_cuda", color="green"):
            # Wrap u for Numba CUDA (zero-copy, u is already CuPy on GPU)
            u_device = cuda.as_cuda_array(self.u)
            
            # Output arrays are already CuPy (allocated in setup)
            result = outpt_cuda(self.r, self.mdr, self._ndr, self._ndz, self.tlc,
                               self.f3, u_device, self.dir, self.ir,
                               self.tll, self.tlg, self.cpl, self.cpg,
                               batch_size=self._batch_size)
        return result

    @nvtx.annotate("CuPyRAM.updat", color="yellow")
    def updat(self):
        """
        Update matrices for range-dependent environment.
        Index updates run in parallel on GPU via CUDA kernel.
        """
        # Run parallel index updates on GPU
        with nvtx.annotate("updat_indices_cuda", color="olive"):
            need_matrc = updat_indices_cuda(
                float(self.r), float(self._dr), float(self._dz), int(self.nz),
                self._rbzb, self.bt_ind, self.iz,
                self._rp_ss, self.ss_ind,
                self._rp_sb, self.sb_ind,
                bool(self.rd_bt), bool(self.rd_ss), bool(self.rd_sb),
                self._batch_size
            )
            # Note: iz is updated in-place on GPU by updat_indices_cuda
        
        # If any ray needs update, recompute profiles
        # Matrices will be computed in _propagate_step()
        if need_matrc:
            self.profl()

        # Turn off the stability constraints (shared across all rays)
        if self.r >= self._rs:
            self._ns = 0
            self._rs = self._rmax + self._dr
            with nvtx.annotate("compute_pade_stability", color="purple"):
                pd1_new, pd2_new = compute_pade_coefficients(
                    freq=self._freq, c0=self._c0, np_pade=self._np,
                    ns=self._ns, dr=self._dr, ip=1
                )
                # Update Padé coefficients for all rays (broadcast to GPU)
                # Shape: [np] → [np, 1] → broadcast to [np, batch_size]
                self.pd1[:, :] = cupy.asarray(pd1_new)[:, None]
                self.pd2[:, :] = cupy.asarray(pd2_new)[:, None]

    @nvtx.annotate("CuPyRAM.selfs", color="magenta")
    def selfs(self):
        """
        The self-starter. Initialize acoustic field for all rays.
        Arrays: [Nz+2, Batch] for coalesced memory access.
        """
        # Conditions for the delta function
        si = self._zs / self._dz
        _is = int(numpy.floor(si))  # Source depth index
        dis = si - _is  # Offset

        # Initialize u for all rays in batch (same source position for all)
        # Vectorized GPU operation - no loops, no transfers
        # u, alpw, k0 are all CuPy arrays on GPU [Nz+2, Batch] or [Batch]
        self.u[_is, :] = (1 - dis) * cupy.sqrt(2 * cupy.pi / self.k0) / \
            (self._dz * self.alpw[_is, :])
        self.u[_is + 1, :] = dis * cupy.sqrt(2 * cupy.pi / self.k0) / \
            (self._dz * self.alpw[_is + 1, :])

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
        
        # Compute per-ray Padé coefficients in parallel on CPU, transfer to GPU
        with nvtx.annotate("compute_pade_batch_selfs", color="purple"):
            pd1_batch, pd2_batch = compute_pade_coefficients_batch(
                freq=self._freq, c0_array=self._c0_array, 
                np_pade=self._np, ns=self._ns, 
                dr=self._dr, ip=2, max_workers=self._max_workers
            )
            self.pd1 = cupy.asarray(pd1_batch)  # Transfer to GPU: [np, batch]
            self.pd2 = cupy.asarray(pd2_batch)
        
        # Apply full Padé operator
        self._propagate_step()
