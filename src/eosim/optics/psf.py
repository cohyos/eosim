"""
Point Spread Function (PSF) models for EOSIM.

Provides various PSF implementations from simple Gaussian approximations
to full diffraction-limited Airy patterns and aberrated Zernike models.
"""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray
from scipy.special import j1

from eosim.optics.aberrations import (
    AberrationSet,
    compute_wavefront,
    create_pupil_grid,
    strehl_ratio,
)


@dataclass
class PSFResult:
    """Result of PSF computation.

    Attributes:
        kernel: 2D PSF array, normalized to sum=1
        size_pixels: Kernel size in pixels
        sigma_pixels: Effective Gaussian sigma (for blur estimation)
        fwhm_pixels: Full width at half maximum
        strehl_ratio: Optical quality metric (1.0 for diffraction-limited)
        wavelength_um: Wavelength used for computation
    """

    kernel: NDArray[np.floating]
    size_pixels: int
    sigma_pixels: float
    fwhm_pixels: float
    strehl_ratio: float = 1.0
    wavelength_um: float = 10.0

    def __post_init__(self) -> None:
        """Ensure kernel is properly normalized."""
        kernel_sum = np.sum(self.kernel)
        if not np.isclose(kernel_sum, 1.0, rtol=1e-3):
            self.kernel = self.kernel / kernel_sum


class GaussianPSF:
    """Gaussian PSF model - fast approximation to diffraction-limited PSF.

    PSF(r) = (1 / (2πσ²)) × exp(-r² / (2σ²))

    Where σ ≈ 0.44 × λ × F/# in the focal plane.
    """

    def compute(
        self,
        wavelength_um: float,
        f_number: float,
        pixel_pitch_um: float,
        size_pixels: int = 15,
    ) -> PSFResult:
        """Compute Gaussian PSF.

        Args:
            wavelength_um: Wavelength in micrometers
            f_number: Optical f-number (focal length / aperture)
            pixel_pitch_um: Detector pixel pitch in micrometers
            size_pixels: Size of PSF kernel (should be odd)

        Returns:
            PSFResult with normalized Gaussian kernel
        """
        # Ensure odd size for symmetric kernel
        if size_pixels % 2 == 0:
            size_pixels += 1

        # Gaussian sigma in focal plane
        # σ ≈ 0.44 × λ × F/# (approximation to Airy disk core)
        sigma_um = 0.44 * wavelength_um * f_number
        sigma_pixels = sigma_um / pixel_pitch_um

        # FWHM = 2.355 × σ
        fwhm_pixels = 2.355 * sigma_pixels

        # Create coordinate grid
        half = size_pixels // 2
        x = np.arange(-half, half + 1, dtype=np.float64)
        xx, yy = np.meshgrid(x, x)
        r_squared = xx**2 + yy**2

        # Gaussian PSF
        kernel = np.exp(-r_squared / (2 * sigma_pixels**2))

        # Normalize to sum = 1
        kernel = kernel / np.sum(kernel)

        return PSFResult(
            kernel=kernel,
            size_pixels=size_pixels,
            sigma_pixels=sigma_pixels,
            fwhm_pixels=fwhm_pixels,
            strehl_ratio=1.0,
            wavelength_um=wavelength_um,
        )


class AiryPSF:
    """Diffraction-limited Airy disk PSF.

    For a circular aperture:
    PSF(r) = [2 × J₁(x) / x]² where x = π × r / (λ × F/#)

    The first zero occurs at r = 1.22 × λ × F/#
    """

    def compute(
        self,
        wavelength_um: float,
        f_number: float,
        pixel_pitch_um: float,
        size_pixels: int = 31,
    ) -> PSFResult:
        """Compute Airy disk PSF.

        Args:
            wavelength_um: Wavelength in micrometers
            f_number: Optical f-number
            pixel_pitch_um: Detector pixel pitch in micrometers
            size_pixels: Size of PSF kernel (should be odd)

        Returns:
            PSFResult with normalized Airy disk kernel
        """
        # Ensure odd size
        if size_pixels % 2 == 0:
            size_pixels += 1

        # Airy disk radius (first zero)
        airy_radius_um = 1.22 * wavelength_um * f_number
        airy_radius_pixels = airy_radius_um / pixel_pitch_um

        # FWHM of Airy disk ≈ 1.03 × λ × F/#
        fwhm_um = 1.03 * wavelength_um * f_number
        fwhm_pixels = fwhm_um / pixel_pitch_um

        # Equivalent Gaussian sigma
        sigma_pixels = fwhm_pixels / 2.355

        # Create coordinate grid
        half = size_pixels // 2
        x = np.arange(-half, half + 1, dtype=np.float64)
        xx, yy = np.meshgrid(x, x)
        r = np.sqrt(xx**2 + yy**2)

        # Airy disk: [2 × J₁(x) / x]² where x = π × r × pixel_pitch / (λ × F/#)
        # In pixel units: x = π × r / airy_radius_pixels × 1.22
        x_arg = np.pi * r / airy_radius_pixels * 1.22

        # Handle r=0 case (limit is 1)
        kernel = np.ones_like(r)
        nonzero = r > 0
        kernel[nonzero] = (2 * j1(x_arg[nonzero]) / x_arg[nonzero]) ** 2

        # Normalize
        kernel = kernel / np.sum(kernel)

        return PSFResult(
            kernel=kernel,
            size_pixels=size_pixels,
            sigma_pixels=sigma_pixels,
            fwhm_pixels=fwhm_pixels,
            strehl_ratio=1.0,
            wavelength_um=wavelength_um,
        )


class ZernikePSF:
    """PSF with wavefront aberrations modeled by Zernike polynomials.

    Computes PSF via Fourier transform of the pupil function:
    PSF = |FT[P(x,y) × exp(i × 2π × W(x,y))]|²

    Where P is the pupil mask and W is the wavefront error in waves.
    """

    def __init__(
        self,
        aberrations: Optional[Union[AberrationSet, dict[int, float]]] = None,
    ) -> None:
        """Initialize with aberration coefficients.

        Args:
            aberrations: AberrationSet or dict of Noll index to waves
        """
        if aberrations is None:
            self.aberrations = AberrationSet()
        elif isinstance(aberrations, dict):
            self.aberrations = AberrationSet.from_coefficients(aberrations)
        else:
            self.aberrations = aberrations

    def compute(
        self,
        wavelength_um: float,
        f_number: float,
        pixel_pitch_um: float,
        size_pixels: int = 63,
        pupil_samples: int = 128,
    ) -> PSFResult:
        """Compute aberrated PSF via pupil function FFT.

        Args:
            wavelength_um: Wavelength in micrometers
            f_number: Optical f-number
            pixel_pitch_um: Detector pixel pitch in micrometers
            size_pixels: Size of output PSF kernel
            pupil_samples: Number of samples across pupil diameter

        Returns:
            PSFResult with aberrated PSF kernel
        """
        # Compute wavefront on pupil grid
        coeffs = self.aberrations.to_coefficients()
        wavefront, pupil_mask = compute_wavefront(coeffs, pupil_samples)

        # Complex pupil function: P × exp(i × 2π × W)
        pupil_function = pupil_mask.astype(np.complex128)
        pupil_function = pupil_function * np.exp(1j * 2 * np.pi * wavefront)

        # Pad for desired PSF resolution
        # The PSF pixel scale depends on padding
        pad_factor = max(4, size_pixels * 2 // pupil_samples)
        padded_size = pupil_samples * pad_factor
        padded = np.zeros((padded_size, padded_size), dtype=np.complex128)
        start = (padded_size - pupil_samples) // 2
        padded[start : start + pupil_samples, start : start + pupil_samples] = (
            pupil_function
        )

        # FFT to get amplitude in focal plane
        psf_complex = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
        psf_intensity = np.abs(psf_complex) ** 2

        # Extract central region
        center = padded_size // 2
        half = size_pixels // 2
        kernel = psf_intensity[
            center - half : center + half + 1, center - half : center + half + 1
        ]

        # Normalize
        kernel = kernel / np.sum(kernel)

        # Compute metrics
        airy_fwhm_um = 1.03 * wavelength_um * f_number
        fwhm_pixels = airy_fwhm_um / pixel_pitch_um
        sigma_pixels = fwhm_pixels / 2.355
        strehl = self.aberrations.strehl

        return PSFResult(
            kernel=kernel,
            size_pixels=kernel.shape[0],
            sigma_pixels=sigma_pixels,
            fwhm_pixels=fwhm_pixels,
            strehl_ratio=strehl,
            wavelength_um=wavelength_um,
        )


class MeasuredPSF:
    """PSF from measured or imported data."""

    def __init__(
        self,
        kernel: NDArray[np.floating],
        pixel_pitch_um: float,
        wavelength_um: float,
    ) -> None:
        """Initialize with measured PSF data.

        Args:
            kernel: 2D PSF array (will be normalized)
            pixel_pitch_um: Pixel pitch of the measurement
            wavelength_um: Wavelength of measurement
        """
        self._kernel = kernel / np.sum(kernel)
        self._pixel_pitch = pixel_pitch_um
        self._wavelength = wavelength_um

    def compute(
        self,
        wavelength_um: Optional[float] = None,
        pixel_pitch_um: Optional[float] = None,
        size_pixels: Optional[int] = None,
    ) -> PSFResult:
        """Return the measured PSF, optionally resampled.

        Args:
            wavelength_um: Target wavelength (for scaling)
            pixel_pitch_um: Target pixel pitch (for resampling)
            size_pixels: Target size (for cropping/padding)

        Returns:
            PSFResult with measured PSF
        """
        kernel = self._kernel.copy()

        # Resample if pixel pitch differs
        if pixel_pitch_um is not None and pixel_pitch_um != self._pixel_pitch:
            from scipy.ndimage import zoom

            scale = self._pixel_pitch / pixel_pitch_um
            kernel = zoom(kernel, scale, order=3)
            kernel = kernel / np.sum(kernel)

        # Adjust size if needed
        if size_pixels is not None:
            kernel = self._resize_kernel(kernel, size_pixels)

        # Estimate sigma from second moment
        h, w = kernel.shape
        y, x = np.mgrid[: h, : w]
        y = y - h // 2
        x = x - w // 2
        sigma_sq = np.sum(kernel * (x**2 + y**2)) / np.sum(kernel)
        sigma_pixels = np.sqrt(sigma_sq)
        fwhm_pixels = 2.355 * sigma_pixels

        # Estimate Strehl from peak value vs Airy
        # (simplified estimate)
        strehl = min(1.0, kernel.max() / (1.0 / (np.pi * sigma_pixels**2)))

        return PSFResult(
            kernel=kernel,
            size_pixels=kernel.shape[0],
            sigma_pixels=sigma_pixels,
            fwhm_pixels=fwhm_pixels,
            strehl_ratio=strehl,
            wavelength_um=wavelength_um or self._wavelength,
        )

    def _resize_kernel(
        self, kernel: NDArray[np.floating], target_size: int
    ) -> NDArray[np.floating]:
        """Resize kernel to target size via padding or cropping."""
        h, w = kernel.shape
        if h == target_size and w == target_size:
            return kernel

        # Center crop or pad
        if h > target_size:
            # Crop
            start_h = (h - target_size) // 2
            start_w = (w - target_size) // 2
            kernel = kernel[
                start_h : start_h + target_size, start_w : start_w + target_size
            ]
        else:
            # Pad
            result = np.zeros((target_size, target_size), dtype=kernel.dtype)
            start_h = (target_size - h) // 2
            start_w = (target_size - w) // 2
            result[start_h : start_h + h, start_w : start_w + w] = kernel
            kernel = result

        return kernel / np.sum(kernel)


def compute_psf(
    wavelength_um: float,
    f_number: float,
    pixel_pitch_um: float,
    model: str = "airy",
    aberrations: Optional[AberrationSet] = None,
    size_pixels: int = 31,
) -> PSFResult:
    """Convenience function to compute PSF with specified model.

    Args:
        wavelength_um: Wavelength in micrometers
        f_number: Optical f-number
        pixel_pitch_um: Detector pixel pitch in micrometers
        model: PSF model type: "gaussian", "airy", or "zernike"
        aberrations: Aberration coefficients (for Zernike model)
        size_pixels: Size of PSF kernel

    Returns:
        PSFResult with computed PSF
    """
    if model == "gaussian":
        return GaussianPSF().compute(wavelength_um, f_number, pixel_pitch_um, size_pixels)
    elif model == "airy":
        return AiryPSF().compute(wavelength_um, f_number, pixel_pitch_um, size_pixels)
    elif model == "zernike":
        return ZernikePSF(aberrations).compute(
            wavelength_um, f_number, pixel_pitch_um, size_pixels
        )
    else:
        raise ValueError(f"Unknown PSF model: {model}")


def psf_encircled_energy(
    psf: NDArray[np.floating],
    radius_pixels: float,
) -> float:
    """Compute fraction of PSF energy within given radius.

    Args:
        psf: 2D PSF array (normalized)
        radius_pixels: Radius in pixels from center

    Returns:
        Fraction of total energy within radius (0 to 1)
    """
    h, w = psf.shape
    y, x = np.mgrid[:h, :w]
    y = y - h // 2
    x = x - w // 2
    r = np.sqrt(x**2 + y**2)

    mask = r <= radius_pixels
    return float(np.sum(psf[mask]))


def psf_to_ensquared_energy(
    psf: NDArray[np.floating],
    box_size_pixels: int,
) -> float:
    """Compute fraction of PSF energy within a square box.

    Args:
        psf: 2D PSF array (normalized)
        box_size_pixels: Size of box in pixels

    Returns:
        Fraction of total energy within box (0 to 1)
    """
    h, w = psf.shape
    center_h, center_w = h // 2, w // 2
    half = box_size_pixels // 2

    start_h = max(0, center_h - half)
    end_h = min(h, center_h + half + 1)
    start_w = max(0, center_w - half)
    end_w = min(w, center_w + half + 1)

    return float(np.sum(psf[start_h:end_h, start_w:end_w]))
