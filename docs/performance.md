# GPU performance notes

CuPyRAM is intended for large batches of acoustic rays. GPU performance depends
on the compiled kernel's resource use as well as the amount of work submitted;
a large batch can keep every SM assigned while still leaving execution units
underused if too few warps can reside on each SM.

## Fused-kernel launch tuning on A100

The fused Padé kernel was profiled on an **NVIDIA A100 SXM4 80 GB** GPU with
Nsight Compute 2025.2.1. The kernel compiled for compute capability 8.0 using
142 32-bit registers per thread and no local-memory spills.

At 256 threads per block, register pressure permits only one block, or eight
warps, to reside on each A100 SM. Reducing the block size to 192 permits two
blocks, or twelve warps, to reside on each SM without changing the algorithm,
arithmetic, or VRAM allocation.

Hardware counters from a representative 100,000-ray launch were:

| Metric | 256 threads | 192 threads |
|---|---:|---:|
| Registers per thread | 142 | 142 |
| Resident blocks per SM | 1 | 2 |
| Resident warps per SM | 8 | 12 |
| Theoretical occupancy | 12.5% | 18.75% |
| Achieved occupancy | 12.46% | 16.25% |
| Issue slots busy | 32.7% | 39.2% |
| Compute (SM) throughput | 30.5% | 38.5% |
| DRAM throughput | 508 GB/s | 641 GB/s |
| Scheduler cycles with no eligible warp | 67.3% | 60.4% |

The dependency stalls experienced by an individual warp were nearly unchanged.
The gain came from keeping more warps resident so that the scheduler had useful
work available while other warps waited.

An uninstrumented production-length benchmark used 165,000 rays, 2,201 depth
points, 499 range steps over 60 km, and 20 untimed warm-up steps:

| Metric | 256 threads | 192 threads | Change |
|---|---:|---:|---:|
| Propagation time per step | 339.62 ms | 274.83 ms | -19.1% |
| Propagation throughput | 485.8k ray-steps/s | 600.4k ray-steps/s | +23.6% |
| Measured end-to-end time | 205.60 s | 173.73 s | -15.5% |
| GPU memory-pool footprint | 76.53 GiB | 76.53 GiB | unchanged |

The 192-thread default deliberately targets CuPyRAM's large-batch use case.
Small batches can favor smaller blocks because they need more blocks merely to
cover all SMs; PyRAM is generally the more appropriate implementation when the
problem is small enough for that distinction to matter.

## Portability to other GPUs

The value 192 has been benchmarked only on A100. Treat it as a strong default,
not a universal optimum.

There is no immediate architectural reason to expect it to perform poorly on
an RTX 6000 Ada Generation GPU. NVIDIA's Ada tuning guide specifies the same
64K 32-bit register file per SM, with a maximum of 48 resident warps. Assuming
the kernel still compiles to approximately 142 registers per thread:

- 256-thread blocks permit one resident block, or 8 of 48 warps.
- 192-thread blocks permit two resident blocks, or 12 of 48 warps.

That is the same 1.5x increase in resident warps observed on A100. The RTX 6000
Ada also has 142 SMs, so large CuPyRAM batches provide ample blocks to occupy
the device.

Nevertheless, retest the launch size when:

- moving to a different GPU architecture or compute capability;
- changing Numba, `numba-cuda`, CUDA, or driver versions;
- Nsight reports a materially different register count;
- using unusually small batches;
- using MIG or another configuration that changes the available SM count; or
- performance is important enough that a short block-size sweep is justified.

Ada has different FP64 throughput, cache behavior, clocks, and GDDR memory
bandwidth and latency from A100. Those differences can change the best launch
size even though the register-capacity argument remains favorable.

References:

- [NVIDIA Ada GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/)
- [NVIDIA Ada professional GPU architecture whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf)
- [NVIDIA CUDA Best Practices Guide: register pressure](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
