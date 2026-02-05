"""
GPU Context and Device Management for EOSIM.

Provides GPU detection, context management, and CPU fallback.
Supports CuPy (CUDA) with transparent fallback to NumPy.

Example 1: Check GPU availability
    >>> from eosim.gpu import gpu_available, get_device_info
    >>> if gpu_available():
    ...     info = get_device_info()
    ...     print(f"GPU: {info['name']}, Memory: {info['memory_gb']:.1f} GB")
    ... else:
    ...     print("No GPU available, using CPU")

Example 2: Transparent array creation
    >>> from eosim.gpu import GPUContext, to_gpu, to_cpu
    >>> ctx = GPUContext()
    >>> data = ctx.array([[1, 2], [3, 4]], dtype='float32')
    >>> result = ctx.sum(data)
    >>> numpy_result = to_cpu(result)

Example 3: Context manager for GPU operations
    >>> with GPUContext() as ctx:
    ...     a = ctx.array([1, 2, 3])
    ...     b = ctx.array([4, 5, 6])
    ...     c = ctx.add(a, b)
    ...     print(to_cpu(c))
"""

from dataclasses import dataclass
from typing import Optional, Union, Any
import numpy as np
from numpy.typing import NDArray

# Try to import CuPy for CUDA support
_CUPY_AVAILABLE = False
_cupy = None

try:
    import cupy as _cupy
    _CUPY_AVAILABLE = True
except ImportError:
    pass


def gpu_available() -> bool:
    """Check if GPU acceleration is available.

    Returns:
        True if CuPy is installed and a CUDA device is available.
    """
    if not _CUPY_AVAILABLE:
        return False

    try:
        _cupy.cuda.Device(0)
        return True
    except Exception:
        return False


def get_array_module(array: Any = None):
    """Get the appropriate array module (numpy or cupy) for an array.

    Args:
        array: Optional array to check. If None, returns numpy.

    Returns:
        numpy or cupy module
    """
    if array is None:
        return np

    if _CUPY_AVAILABLE and isinstance(array, _cupy.ndarray):
        return _cupy

    return np


def to_gpu(array: NDArray) -> Any:
    """Transfer array to GPU if available.

    Args:
        array: NumPy array to transfer

    Returns:
        CuPy array if GPU available, otherwise original NumPy array
    """
    if gpu_available():
        return _cupy.asarray(array)
    return array


def to_cpu(array: Any) -> NDArray:
    """Transfer array from GPU to CPU.

    Args:
        array: Array (CuPy or NumPy) to transfer

    Returns:
        NumPy array
    """
    if _CUPY_AVAILABLE and isinstance(array, _cupy.ndarray):
        return array.get()
    return np.asarray(array)


def get_device_info() -> dict:
    """Get GPU device information.

    Returns:
        Dictionary with device information, or empty dict if no GPU.
    """
    if not gpu_available():
        return {"available": False}

    try:
        device = _cupy.cuda.Device(0)
        props = device.attributes

        return {
            "available": True,
            "name": device.name if hasattr(device, 'name') else "CUDA Device",
            "compute_capability": device.compute_capability,
            "memory_gb": device.mem_info[1] / (1024**3),
            "memory_free_gb": device.mem_info[0] / (1024**3),
            "multiprocessors": props.get("MultiProcessorCount", "unknown"),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


@dataclass
class GPUMemoryInfo:
    """GPU memory information.

    Attributes:
        total_bytes: Total GPU memory
        free_bytes: Free GPU memory
        used_bytes: Used GPU memory
    """
    total_bytes: int
    free_bytes: int
    used_bytes: int

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024**3)

    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1024**3)

    @property
    def used_gb(self) -> float:
        return self.used_bytes / (1024**3)


class GPUContext:
    """GPU computation context with CPU fallback.

    Provides a unified interface for GPU/CPU array operations.
    Automatically falls back to CPU if GPU is not available.
    """

    def __init__(self, device_id: int = 0, force_cpu: bool = False) -> None:
        """Initialize GPU context.

        Args:
            device_id: CUDA device ID (default 0)
            force_cpu: Force CPU mode even if GPU is available
        """
        self._device_id = device_id
        self._force_cpu = force_cpu
        self._use_gpu = gpu_available() and not force_cpu

        if self._use_gpu:
            self._xp = _cupy
            _cupy.cuda.Device(device_id).use()
        else:
            self._xp = np

    def __enter__(self) -> "GPUContext":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if self._use_gpu:
            self.synchronize()

    @property
    def is_gpu(self) -> bool:
        """Check if using GPU."""
        return self._use_gpu

    @property
    def device_name(self) -> str:
        """Get device name."""
        if self._use_gpu:
            return f"CUDA:{self._device_id}"
        return "CPU"

    @property
    def xp(self):
        """Get array module (numpy or cupy)."""
        return self._xp

    def array(
        self,
        data,
        dtype: Optional[str] = None,
    ):
        """Create array on current device.

        Args:
            data: Input data
            dtype: Data type

        Returns:
            Array on current device
        """
        if dtype:
            return self._xp.array(data, dtype=dtype)
        return self._xp.array(data)

    def zeros(self, shape, dtype: str = "float64"):
        """Create zero array on current device."""
        return self._xp.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype: str = "float64"):
        """Create ones array on current device."""
        return self._xp.ones(shape, dtype=dtype)

    def empty(self, shape, dtype: str = "float64"):
        """Create empty array on current device."""
        return self._xp.empty(shape, dtype=dtype)

    def arange(self, *args, **kwargs):
        """Create arange on current device."""
        return self._xp.arange(*args, **kwargs)

    def linspace(self, *args, **kwargs):
        """Create linspace on current device."""
        return self._xp.linspace(*args, **kwargs)

    def random(self, shape, seed: Optional[int] = None):
        """Create random array on current device."""
        if self._use_gpu:
            if seed is not None:
                _cupy.random.seed(seed)
            return _cupy.random.random(shape)
        rng = np.random.default_rng(seed)
        return rng.random(shape)

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size=None,
        seed: Optional[int] = None,
    ):
        """Create normal-distributed array on current device."""
        if self._use_gpu:
            if seed is not None:
                _cupy.random.seed(seed)
            return _cupy.random.normal(loc, scale, size)
        rng = np.random.default_rng(seed)
        return rng.normal(loc, scale, size)

    def to_device(self, array: NDArray):
        """Transfer array to current device."""
        if self._use_gpu:
            return _cupy.asarray(array)
        return np.asarray(array)

    def to_host(self, array) -> NDArray:
        """Transfer array to host (CPU)."""
        return to_cpu(array)

    def synchronize(self) -> None:
        """Synchronize GPU operations."""
        if self._use_gpu:
            _cupy.cuda.Stream.null.synchronize()

    # Math operations
    def add(self, a, b):
        """Element-wise addition."""
        return self._xp.add(a, b)

    def multiply(self, a, b):
        """Element-wise multiplication."""
        return self._xp.multiply(a, b)

    def exp(self, a):
        """Element-wise exponential."""
        return self._xp.exp(a)

    def log(self, a):
        """Element-wise logarithm."""
        return self._xp.log(a)

    def sum(self, a, axis=None):
        """Sum reduction."""
        return self._xp.sum(a, axis=axis)

    def mean(self, a, axis=None):
        """Mean reduction."""
        return self._xp.mean(a, axis=axis)

    def sqrt(self, a):
        """Element-wise square root."""
        return self._xp.sqrt(a)

    def clip(self, a, a_min, a_max):
        """Clip values."""
        return self._xp.clip(a, a_min, a_max)

    # FFT operations
    def fft2(self, a):
        """2D FFT."""
        return self._xp.fft.fft2(a)

    def ifft2(self, a):
        """2D inverse FFT."""
        return self._xp.fft.ifft2(a)

    def fftshift(self, a):
        """FFT shift."""
        return self._xp.fft.fftshift(a)

    def ifftshift(self, a):
        """Inverse FFT shift."""
        return self._xp.fft.ifftshift(a)

    # Planck radiation
    def planck_radiance(
        self,
        temperature_k,
        wavelength_um: float,
    ):
        """Compute Planck spectral radiance.

        L = (2hc²/λ⁵) × 1/(exp(hc/λkT) - 1)

        Args:
            temperature_k: Temperature array [K]
            wavelength_um: Wavelength [μm]

        Returns:
            Spectral radiance [W/(m²·sr·μm)]
        """
        # Physical constants
        h = 6.62607015e-34  # Planck constant [J·s]
        c = 299792458.0  # Speed of light [m/s]
        k_B = 1.380649e-23  # Boltzmann constant [J/K]

        wavelength_m = wavelength_um * 1e-6

        # Compute radiance
        c1 = 2 * h * c**2
        c2 = h * c / k_B

        T = self._xp.asarray(temperature_k)
        T = self._xp.maximum(T, 1.0)  # Avoid division by zero

        exponent = c2 / (wavelength_m * T)
        exponent = self._xp.minimum(exponent, 700)  # Prevent overflow

        radiance = c1 / (wavelength_m**5) / (self._xp.exp(exponent) - 1)

        # Convert to per-micrometer units
        return radiance * 1e-6

    def memory_info(self) -> Optional[GPUMemoryInfo]:
        """Get GPU memory information.

        Returns:
            GPUMemoryInfo or None if using CPU
        """
        if not self._use_gpu:
            return None

        free, total = _cupy.cuda.Device(self._device_id).mem_info
        return GPUMemoryInfo(
            total_bytes=total,
            free_bytes=free,
            used_bytes=total - free,
        )
