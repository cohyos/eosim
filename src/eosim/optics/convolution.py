"""
FFT-based convolution for optical blur simulation.

Provides efficient PSF convolution using FFT methods with support for
various boundary conditions and optional GPU acceleration.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray
from scipy import fft as scipy_fft
from scipy.ndimage import convolve

from eosim.optics.psf import PSFResult


class BoundaryMode(Enum):
    """Boundary handling modes for convolution."""

    ZERO = "zero"           # Zero padding (default for FFT)
    REFLECT = "reflect"     # Mirror reflection at edges
    WRAP = "wrap"           # Periodic boundary (wraparound)
    NEAREST = "nearest"     # Extend edge values


@dataclass
class ConvolutionResult:
    """Result of PSF convolution.

    Attributes:
        image: Convolved image
        psf_applied: PSF that was applied
        method: Method used ("fft" or "spatial")
        edge_effects_pixels: Estimated edge region affected by PSF
    """

    image: NDArray[np.floating]
    psf_applied: NDArray[np.floating]
    method: str
    edge_effects_pixels: int


def pad_for_fft(
    image: NDArray[np.floating],
    psf: NDArray[np.floating],
    mode: BoundaryMode = BoundaryMode.ZERO,
) -> tuple[NDArray[np.floating], tuple[int, int, int, int]]:
    """Pad image for FFT convolution.

    Args:
        image: Input image
        psf: Point spread function kernel
        mode: Boundary handling mode

    Returns:
        Tuple of (padded_image, padding) where padding is (top, bottom, left, right)
    """
    h, w = image.shape
    kh, kw = psf.shape

    # Pad to avoid circular convolution artifacts
    pad_h = kh // 2
    pad_w = kw // 2

    if mode == BoundaryMode.ZERO:
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    elif mode == BoundaryMode.REFLECT:
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    elif mode == BoundaryMode.WRAP:
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='wrap')
    elif mode == BoundaryMode.NEAREST:
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    else:
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    return padded, (pad_h, pad_h, pad_w, pad_w)


def fft_convolve_2d(
    image: NDArray[np.floating],
    kernel: NDArray[np.floating],
    mode: BoundaryMode = BoundaryMode.ZERO,
) -> NDArray[np.floating]:
    """Convolve image with kernel using FFT.

    FFT-based convolution is O(N log N) vs O(N × K) for spatial,
    making it faster for larger kernels (typically > 11×11).

    Args:
        image: 2D input image
        kernel: 2D convolution kernel (should be normalized to sum=1 for blur)
        mode: Boundary handling mode

    Returns:
        Convolved image (same size as input)
    """
    h, w = image.shape
    kh, kw = kernel.shape

    # Pad image for boundary handling
    padded, (pt, pb, pl, pr) = pad_for_fft(image, kernel, mode)

    # Pad kernel to match image size
    ph, pw = padded.shape
    kernel_padded = np.zeros((ph, pw), dtype=np.float64)

    # Center kernel in padded array
    kh_half, kw_half = kh // 2, kw // 2
    kernel_padded[:kh, :kw] = kernel

    # Shift kernel so center is at (0,0) for proper FFT convolution
    kernel_padded = np.roll(kernel_padded, -kh_half, axis=0)
    kernel_padded = np.roll(kernel_padded, -kw_half, axis=1)

    # FFT convolution: F^-1(F(image) × F(kernel))
    image_fft = scipy_fft.fft2(padded)
    kernel_fft = scipy_fft.fft2(kernel_padded)
    result_fft = image_fft * kernel_fft
    result = np.real(scipy_fft.ifft2(result_fft))

    # Remove padding
    result = result[pt:pt + h, pl:pl + w]

    return result


def spatial_convolve_2d(
    image: NDArray[np.floating],
    kernel: NDArray[np.floating],
    mode: BoundaryMode = BoundaryMode.REFLECT,
) -> NDArray[np.floating]:
    """Convolve image with kernel using spatial domain method.

    Direct spatial convolution is efficient for small kernels (< 11×11)
    and provides better boundary handling options.

    Args:
        image: 2D input image
        kernel: 2D convolution kernel
        mode: Boundary handling mode

    Returns:
        Convolved image (same size as input)
    """
    # Map boundary mode to scipy mode
    scipy_modes = {
        BoundaryMode.ZERO: 'constant',
        BoundaryMode.REFLECT: 'reflect',
        BoundaryMode.WRAP: 'wrap',
        BoundaryMode.NEAREST: 'nearest',
    }
    scipy_mode = scipy_modes.get(mode, 'reflect')

    return convolve(image.astype(np.float64), kernel.astype(np.float64), mode=scipy_mode)


def apply_psf(
    image: NDArray[np.floating],
    psf: Union[PSFResult, NDArray[np.floating]],
    method: str = "auto",
    boundary: BoundaryMode = BoundaryMode.REFLECT,
) -> ConvolutionResult:
    """Apply point spread function to image.

    Args:
        image: 2D input radiance/intensity image
        psf: PSF kernel (PSFResult or 2D array, should sum to 1)
        method: Convolution method - "fft", "spatial", or "auto"
        boundary: Boundary handling mode

    Returns:
        ConvolutionResult with blurred image
    """
    # Extract kernel from PSFResult if needed
    if isinstance(psf, PSFResult):
        kernel = psf.kernel
    else:
        kernel = psf

    # Ensure kernel is normalized
    kernel = kernel / np.sum(kernel)

    # Auto-select method based on kernel size
    if method == "auto":
        kernel_size = max(kernel.shape)
        method = "fft" if kernel_size > 11 else "spatial"

    # Apply convolution
    if method == "fft":
        blurred = fft_convolve_2d(image, kernel, boundary)
    else:
        blurred = spatial_convolve_2d(image, kernel, boundary)

    # Estimate edge effects region
    edge_pixels = max(kernel.shape) // 2

    return ConvolutionResult(
        image=blurred,
        psf_applied=kernel,
        method=method,
        edge_effects_pixels=edge_pixels,
    )


def apply_psf_spectral(
    spectral_cube: NDArray[np.floating],
    psf_per_band: list[NDArray[np.floating]],
    method: str = "auto",
    boundary: BoundaryMode = BoundaryMode.REFLECT,
) -> NDArray[np.floating]:
    """Apply wavelength-dependent PSF to spectral image cube.

    For polychromatic systems, each wavelength may have a slightly
    different PSF due to chromatic aberration.

    Args:
        spectral_cube: 3D array (n_bands, height, width)
        psf_per_band: List of PSF kernels, one per band
        method: Convolution method
        boundary: Boundary handling mode

    Returns:
        Blurred spectral cube
    """
    n_bands, h, w = spectral_cube.shape
    if len(psf_per_band) != n_bands:
        raise ValueError(f"Need {n_bands} PSFs, got {len(psf_per_band)}")

    result = np.zeros_like(spectral_cube)

    for i in range(n_bands):
        conv_result = apply_psf(spectral_cube[i], psf_per_band[i], method, boundary)
        result[i] = conv_result.image

    return result


def apply_psf_field_dependent(
    image: NDArray[np.floating],
    psf_grid: NDArray[np.floating],
    grid_positions: tuple[int, int],
    method: str = "fft",
    boundary: BoundaryMode = BoundaryMode.REFLECT,
) -> NDArray[np.floating]:
    """Apply spatially-varying PSF across field of view.

    Real optical systems have PSF that varies across the field,
    typically being sharper in the center and more aberrated at edges.

    Args:
        image: 2D input image
        psf_grid: 4D array of PSFs (grid_y, grid_x, psf_h, psf_w)
        grid_positions: Number of PSF samples (n_y, n_x)
        method: Convolution method
        boundary: Boundary handling mode

    Returns:
        Blurred image with spatially-varying PSF
    """
    h, w = image.shape
    grid_y, grid_x = grid_positions

    # Verify PSF grid shape
    if psf_grid.shape[:2] != (grid_y, grid_x):
        raise ValueError(f"PSF grid shape {psf_grid.shape[:2]} doesn't match positions {grid_positions}")

    # For simple implementation, divide image into regions and apply local PSF
    result = np.zeros_like(image)
    weights = np.zeros_like(image)

    region_h = h // grid_y
    region_w = w // grid_x

    for iy in range(grid_y):
        for ix in range(grid_x):
            # Get region bounds with overlap for blending
            y_start = max(0, iy * region_h - region_h // 2)
            y_end = min(h, (iy + 1) * region_h + region_h // 2)
            x_start = max(0, ix * region_w - region_w // 2)
            x_end = min(w, (ix + 1) * region_w + region_w // 2)

            # Extract region
            region = image[y_start:y_end, x_start:x_end]

            # Get local PSF
            psf = psf_grid[iy, ix]
            psf = psf / np.sum(psf)

            # Convolve region
            conv_result = apply_psf(region, psf, method, boundary)

            # Create weight mask (cosine taper at edges for blending)
            weight = np.ones((y_end - y_start, x_end - x_start))

            # Add to result with blending
            result[y_start:y_end, x_start:x_end] += conv_result.image * weight
            weights[y_start:y_end, x_start:x_end] += weight

    # Normalize by weights
    result = np.where(weights > 0, result / weights, 0)

    return result


def deconvolve_wiener(
    image: NDArray[np.floating],
    psf: NDArray[np.floating],
    noise_power: float = 0.01,
) -> NDArray[np.floating]:
    """Wiener deconvolution for PSF restoration.

    Attempts to reverse PSF blur using Wiener filter:
    H_wiener = H* / (|H|² + K)

    where K is the noise-to-signal power ratio.

    Args:
        image: Blurred image
        psf: Known PSF kernel
        noise_power: Noise power estimate (regularization parameter)

    Returns:
        Deconvolved image (may have ringing artifacts)
    """
    h, w = image.shape
    kh, kw = psf.shape

    # Pad PSF to image size
    psf_padded = np.zeros((h, w), dtype=np.float64)
    psf_padded[:kh, :kw] = psf

    # Shift PSF center to origin
    psf_padded = np.roll(psf_padded, -kh // 2, axis=0)
    psf_padded = np.roll(psf_padded, -kw // 2, axis=1)

    # Compute FFTs
    image_fft = scipy_fft.fft2(image)
    psf_fft = scipy_fft.fft2(psf_padded)

    # Wiener filter in frequency domain
    # H_wiener = conj(H) / (|H|² + K)
    psf_conj = np.conj(psf_fft)
    psf_power = np.abs(psf_fft) ** 2

    wiener_filter = psf_conj / (psf_power + noise_power)

    # Apply filter
    result_fft = image_fft * wiener_filter
    result = np.real(scipy_fft.ifft2(result_fft))

    return result


def compute_otf(
    psf: NDArray[np.floating],
    output_size: Optional[tuple[int, int]] = None,
) -> NDArray[np.complexfloating]:
    """Compute Optical Transfer Function from PSF.

    OTF = FFT(PSF), with the PSF centered at origin.

    Args:
        psf: Point spread function kernel
        output_size: Optional output size (height, width)

    Returns:
        Complex OTF array
    """
    kh, kw = psf.shape

    if output_size is None:
        output_size = psf.shape

    h, w = output_size

    # Pad PSF to output size
    psf_padded = np.zeros((h, w), dtype=np.float64)
    psf_padded[:kh, :kw] = psf / np.sum(psf)

    # Shift center to origin
    psf_padded = np.roll(psf_padded, -kh // 2, axis=0)
    psf_padded = np.roll(psf_padded, -kw // 2, axis=1)

    # Compute OTF
    otf = scipy_fft.fft2(psf_padded)

    return otf


def apply_otf(
    image: NDArray[np.floating],
    otf: NDArray[np.complexfloating],
) -> NDArray[np.floating]:
    """Apply pre-computed OTF to image.

    Useful when applying the same PSF to many images,
    as OTF only needs to be computed once.

    Args:
        image: Input image
        otf: Pre-computed OTF (must match image size)

    Returns:
        Blurred image
    """
    if image.shape != otf.shape:
        raise ValueError(f"Image shape {image.shape} doesn't match OTF shape {otf.shape}")

    image_fft = scipy_fft.fft2(image)
    result_fft = image_fft * otf
    result = np.real(scipy_fft.ifft2(result_fft))

    return result


class ConvolutionEngine:
    """Efficient convolution engine with caching and optional GPU support.

    Caches OTFs for repeated convolutions with the same PSF,
    and provides consistent interface for CPU/GPU backends.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        cache_size: int = 10,
    ) -> None:
        """Initialize convolution engine.

        Args:
            use_gpu: Whether to use GPU acceleration (requires CuPy)
            cache_size: Maximum number of OTFs to cache
        """
        self.use_gpu = use_gpu
        self.cache_size = cache_size
        self._otf_cache: dict[tuple, NDArray[np.complexfloating]] = {}

        # Try to import CuPy for GPU support
        self._gpu_available = False
        if use_gpu:
            try:
                import cupy as cp
                import cupyx.scipy.fft as cp_fft
                self._cp = cp
                self._cp_fft = cp_fft
                self._gpu_available = True
            except ImportError:
                self._gpu_available = False

    @property
    def gpu_enabled(self) -> bool:
        """Whether GPU is actually being used."""
        return self.use_gpu and self._gpu_available

    def _get_cache_key(
        self,
        psf: NDArray[np.floating],
        image_shape: tuple[int, int],
    ) -> tuple:
        """Generate cache key for PSF + image size."""
        # Use PSF hash and image dimensions
        psf_hash = hash(psf.tobytes())
        return (psf_hash, image_shape)

    def _get_otf(
        self,
        psf: NDArray[np.floating],
        image_shape: tuple[int, int],
    ) -> NDArray[np.complexfloating]:
        """Get OTF from cache or compute it."""
        key = self._get_cache_key(psf, image_shape)

        if key not in self._otf_cache:
            # Compute OTF
            otf = compute_otf(psf, image_shape)

            # Cache management
            if len(self._otf_cache) >= self.cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._otf_cache))
                del self._otf_cache[oldest_key]

            self._otf_cache[key] = otf

        return self._otf_cache[key]

    def convolve(
        self,
        image: NDArray[np.floating],
        psf: NDArray[np.floating],
        boundary: BoundaryMode = BoundaryMode.REFLECT,
    ) -> NDArray[np.floating]:
        """Convolve image with PSF using cached OTF.

        Args:
            image: Input image
            psf: PSF kernel
            boundary: Boundary handling mode

        Returns:
            Blurred image
        """
        if self.gpu_enabled:
            return self._convolve_gpu(image, psf, boundary)
        else:
            return self._convolve_cpu(image, psf, boundary)

    def _convolve_cpu(
        self,
        image: NDArray[np.floating],
        psf: NDArray[np.floating],
        boundary: BoundaryMode,
    ) -> NDArray[np.floating]:
        """CPU convolution with OTF caching."""
        h, w = image.shape
        kh, kw = psf.shape

        # Pad image
        padded, (pt, pb, pl, pr) = pad_for_fft(image, psf, boundary)

        # Get cached OTF
        otf = self._get_otf(psf, padded.shape)

        # Apply OTF
        result = apply_otf(padded, otf)

        # Remove padding
        return result[pt:pt + h, pl:pl + w]

    def _convolve_gpu(
        self,
        image: NDArray[np.floating],
        psf: NDArray[np.floating],
        boundary: BoundaryMode,
    ) -> NDArray[np.floating]:
        """GPU-accelerated convolution."""
        cp = self._cp
        cp_fft = self._cp_fft

        h, w = image.shape
        kh, kw = psf.shape

        # Pad on CPU, then transfer
        padded, (pt, pb, pl, pr) = pad_for_fft(image, psf, boundary)

        # Transfer to GPU
        padded_gpu = cp.asarray(padded)

        # Compute PSF OTF on GPU
        psf_padded = np.zeros(padded.shape, dtype=np.float64)
        psf_padded[:kh, :kw] = psf / np.sum(psf)
        psf_padded = np.roll(psf_padded, -kh // 2, axis=0)
        psf_padded = np.roll(psf_padded, -kw // 2, axis=1)
        psf_gpu = cp.asarray(psf_padded)

        # FFT convolution on GPU
        image_fft = cp_fft.fft2(padded_gpu)
        psf_fft = cp_fft.fft2(psf_gpu)
        result_fft = image_fft * psf_fft
        result_gpu = cp.real(cp_fft.ifft2(result_fft))

        # Transfer back to CPU and remove padding
        result = cp.asnumpy(result_gpu)
        return result[pt:pt + h, pl:pl + w]

    def clear_cache(self) -> None:
        """Clear OTF cache."""
        self._otf_cache.clear()

    def batch_convolve(
        self,
        images: list[NDArray[np.floating]],
        psf: NDArray[np.floating],
        boundary: BoundaryMode = BoundaryMode.REFLECT,
    ) -> list[NDArray[np.floating]]:
        """Convolve multiple images with same PSF.

        Efficient for video frames or multi-band images
        with the same PSF.

        Args:
            images: List of input images (must have same shape)
            psf: PSF kernel
            boundary: Boundary handling

        Returns:
            List of blurred images
        """
        if not images:
            return []

        results = []
        for image in images:
            results.append(self.convolve(image, psf, boundary))

        return results
