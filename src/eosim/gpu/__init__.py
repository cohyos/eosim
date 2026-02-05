"""
EOSIM GPU Acceleration Module (Stage D).

Provides optional GPU acceleration for compute-intensive operations.
Falls back to CPU implementation if GPU libraries are not available.

Example Usage:
--------------
# Example 1: Check GPU availability and compute Planck radiance
>>> from eosim.gpu import GPUContext, gpu_available
>>> if gpu_available():
...     ctx = GPUContext()
...     temps = ctx.array([[300, 310], [320, 330]])
...     radiance = ctx.planck_radiance(temps, wavelength_um=10.0)
...     print(f"Radiance computed on: {ctx.device_name}")

# Example 2: GPU-accelerated PSF convolution
>>> from eosim.gpu import GPUConvolver
>>> convolver = GPUConvolver()
>>> image = np.random.randn(1024, 1024)
>>> psf = np.ones((11, 11)) / 121
>>> result = convolver.convolve(image, psf)

# Example 3: Batch radiance computation
>>> from eosim.gpu import GPURadiance
>>> gpu_rad = GPURadiance()
>>> temps = np.random.randn(100, 480, 640) * 10 + 300
>>> radiances = gpu_rad.batch_planck(temps, wavelength_um=10.0)
"""

from eosim.gpu.context import (
    gpu_available,
    get_array_module,
    GPUContext,
    to_gpu,
    to_cpu,
    get_device_info,
)
from eosim.gpu.operations import (
    GPUConvolver,
    GPURadiance,
    GPUNoiseGenerator,
)
from eosim.gpu.accelerator import (
    GPUAccelerator,
    AcceleratorConfig,
)

__all__ = [
    # Context
    "gpu_available",
    "get_array_module",
    "GPUContext",
    "to_gpu",
    "to_cpu",
    "get_device_info",
    # Operations
    "GPUConvolver",
    "GPURadiance",
    "GPUNoiseGenerator",
    # Accelerator
    "GPUAccelerator",
    "AcceleratorConfig",
]
