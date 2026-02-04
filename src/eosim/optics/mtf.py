"""
Modulation Transfer Function (MTF) computation for EOSIM.

Provides functions for computing MTF from PSF and analytical MTF formulas
for various system components (optics, detector, motion).
"""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray


@dataclass
class MTFResult:
    """Result of MTF computation.

    Attributes:
        frequencies: Spatial frequencies in cycles/pixel
        mtf_h: Horizontal MTF values (0 to 1)
        mtf_v: Vertical MTF values (0 to 1)
        mtf_radial: Radially averaged MTF values
        nyquist_frequency: Nyquist frequency (0.5 cy/pixel)
        mtf_at_nyquist: MTF value at Nyquist frequency
        cutoff_frequency: Diffraction cutoff frequency (if applicable)
    """

    frequencies: NDArray[np.floating]
    mtf_h: NDArray[np.floating]
    mtf_v: NDArray[np.floating]
    mtf_radial: NDArray[np.floating]
    nyquist_frequency: float = 0.5
    mtf_at_nyquist: float = 0.0
    cutoff_frequency: Optional[float] = None

    def __post_init__(self) -> None:
        """Compute MTF at Nyquist if not set."""
        if self.mtf_at_nyquist == 0.0:
            idx = np.argmin(np.abs(self.frequencies - self.nyquist_frequency))
            self.mtf_at_nyquist = float(self.mtf_radial[idx])


def compute_mtf_from_psf(
    psf: NDArray[np.floating],
    pixel_pitch_um: Optional[float] = None,
) -> MTFResult:
    """Compute MTF from PSF via Fourier transform.

    MTF = |OTF| = |FFT(PSF)|

    The MTF is the magnitude of the Optical Transfer Function (OTF),
    which is the Fourier transform of the PSF.

    Args:
        psf: 2D PSF array (should be normalized to sum=1)
        pixel_pitch_um: Pixel pitch in micrometers (for frequency scaling)

    Returns:
        MTFResult with computed MTF curves
    """
    # Ensure PSF is normalized
    psf = psf / np.sum(psf)

    # Compute OTF via FFT
    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
    mtf_2d = np.abs(otf)

    # Normalize so MTF(0) = 1
    mtf_2d = mtf_2d / mtf_2d.max()

    # Extract 1D profiles
    h, w = mtf_2d.shape
    center_h, center_w = h // 2, w // 2

    mtf_h = mtf_2d[center_h, center_w:]
    mtf_v = mtf_2d[center_h:, center_w]

    # Frequency axis in cycles/pixel
    n_freq = len(mtf_h)
    frequencies = np.linspace(0, 0.5, n_freq)

    # Radial average
    mtf_radial = _radial_average_mtf(mtf_2d, n_freq)

    return MTFResult(
        frequencies=frequencies,
        mtf_h=mtf_h,
        mtf_v=mtf_v,
        mtf_radial=mtf_radial,
        nyquist_frequency=0.5,
    )


def _radial_average_mtf(
    mtf_2d: NDArray[np.floating],
    n_bins: int,
) -> NDArray[np.floating]:
    """Compute radially averaged MTF.

    Args:
        mtf_2d: 2D MTF array
        n_bins: Number of radial bins

    Returns:
        1D radially averaged MTF
    """
    h, w = mtf_2d.shape
    center_h, center_w = h // 2, w // 2

    # Create radial coordinate array
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)

    # Bin radii (0 to max radius, which is about half the array size)
    max_r = min(center_h, center_w)
    bin_edges = np.linspace(0, max_r, n_bins + 1)

    # Average MTF in each radial bin
    mtf_radial = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if np.any(mask):
            mtf_radial[i] = np.mean(mtf_2d[mask])
        else:
            mtf_radial[i] = mtf_radial[i - 1] if i > 0 else 1.0

    return mtf_radial


def diffraction_mtf(
    frequencies: NDArray[np.floating],
    wavelength_um: float,
    f_number: float,
    pixel_pitch_um: float,
) -> NDArray[np.floating]:
    """Compute diffraction-limited MTF for circular aperture.

    For a circular aperture, the MTF is:
    MTF(f) = (2/π) × [arccos(f/fc) - (f/fc)×√(1-(f/fc)²)]

    where fc = 1 / (λ × F/#) is the cutoff frequency.

    Args:
        frequencies: Spatial frequencies in cycles/pixel
        wavelength_um: Wavelength in micrometers
        f_number: Optical f-number
        pixel_pitch_um: Pixel pitch in micrometers

    Returns:
        MTF values at each frequency
    """
    # Cutoff frequency in cycles/pixel
    # fc = 1 / (λ × F/#) in cycles/μm, then convert to cycles/pixel
    fc_um = 1 / (wavelength_um * f_number)  # cycles/μm
    fc_pixel = fc_um * pixel_pitch_um  # cycles/pixel

    # Normalized frequency
    f_norm = frequencies / fc_pixel

    # MTF formula for circular aperture
    mtf = np.zeros_like(frequencies)
    valid = f_norm < 1.0

    mtf[valid] = (2 / np.pi) * (
        np.arccos(f_norm[valid])
        - f_norm[valid] * np.sqrt(1 - f_norm[valid] ** 2)
    )

    # At f=0, MTF=1
    mtf[frequencies == 0] = 1.0

    return mtf


def detector_mtf(
    frequencies: NDArray[np.floating],
    fill_factor: float = 1.0,
) -> NDArray[np.floating]:
    """Compute detector sampling MTF.

    For a square pixel with 100% fill factor:
    MTF_det(f) = sinc(π × f × d)

    where d is the pixel size (1 pixel) and f is in cycles/pixel.

    Args:
        frequencies: Spatial frequencies in cycles/pixel
        fill_factor: Detector fill factor (0 to 1)

    Returns:
        Detector MTF values
    """
    # For unit pixel pitch, d = √(fill_factor) effective size
    d = np.sqrt(fill_factor)

    # sinc function (note: np.sinc(x) = sin(πx)/(πx))
    mtf = np.abs(np.sinc(frequencies * d))

    return mtf


def motion_mtf(
    frequencies: NDArray[np.floating],
    motion_pixels: float,
) -> NDArray[np.floating]:
    """Compute motion blur MTF.

    For linear motion during integration:
    MTF_motion(f) = sinc(π × f × Δx)

    where Δx is the motion in pixels.

    Args:
        frequencies: Spatial frequencies in cycles/pixel
        motion_pixels: Motion blur extent in pixels

    Returns:
        Motion blur MTF values
    """
    if motion_pixels == 0:
        return np.ones_like(frequencies)

    return np.abs(np.sinc(frequencies * motion_pixels))


def jitter_mtf(
    frequencies: NDArray[np.floating],
    jitter_sigma_pixels: float,
) -> NDArray[np.floating]:
    """Compute random jitter MTF.

    For Gaussian random motion (jitter):
    MTF_jitter(f) = exp(-2 × (π × f × σ)²)

    Args:
        frequencies: Spatial frequencies in cycles/pixel
        jitter_sigma_pixels: RMS jitter in pixels

    Returns:
        Jitter MTF values
    """
    if jitter_sigma_pixels == 0:
        return np.ones_like(frequencies)

    return np.exp(-2 * (np.pi * frequencies * jitter_sigma_pixels) ** 2)


def atmospheric_mtf(
    frequencies: NDArray[np.floating],
    r0_pixels: float,
) -> NDArray[np.floating]:
    """Compute atmospheric turbulence MTF.

    Long-exposure atmospheric MTF based on Fried parameter:
    MTF_atm(f) ≈ exp(-3.44 × (f/fc)^(5/3))

    where fc = 1 / r0 is the atmospheric cutoff.

    Args:
        frequencies: Spatial frequencies in cycles/pixel
        r0_pixels: Fried parameter in pixels

    Returns:
        Atmospheric MTF values
    """
    if r0_pixels == float("inf") or r0_pixels == 0:
        return np.ones_like(frequencies)

    fc = 1 / r0_pixels
    f_norm = frequencies / fc

    return np.exp(-3.44 * f_norm ** (5 / 3))


def system_mtf(
    optics_mtf: NDArray[np.floating],
    detector_mtf: NDArray[np.floating],
    motion_mtf: Optional[NDArray[np.floating]] = None,
    jitter_mtf: Optional[NDArray[np.floating]] = None,
    atmosphere_mtf: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """Compute combined system MTF.

    MTF_system = MTF_optics × MTF_detector × MTF_motion × ...

    Args:
        optics_mtf: Optical system MTF
        detector_mtf: Detector sampling MTF
        motion_mtf: Motion blur MTF (optional)
        jitter_mtf: Jitter MTF (optional)
        atmosphere_mtf: Atmospheric MTF (optional)

    Returns:
        Combined system MTF
    """
    mtf = optics_mtf * detector_mtf

    if motion_mtf is not None:
        mtf = mtf * motion_mtf

    if jitter_mtf is not None:
        mtf = mtf * jitter_mtf

    if atmosphere_mtf is not None:
        mtf = mtf * atmosphere_mtf

    return mtf


def compute_system_mtf(
    wavelength_um: float,
    f_number: float,
    pixel_pitch_um: float,
    fill_factor: float = 1.0,
    motion_pixels: float = 0.0,
    jitter_sigma_pixels: float = 0.0,
    n_frequencies: int = 100,
) -> MTFResult:
    """Compute complete system MTF with all components.

    Args:
        wavelength_um: Wavelength in micrometers
        f_number: Optical f-number
        pixel_pitch_um: Pixel pitch in micrometers
        fill_factor: Detector fill factor
        motion_pixels: Motion blur in pixels
        jitter_sigma_pixels: RMS jitter in pixels
        n_frequencies: Number of frequency samples

    Returns:
        MTFResult with system MTF
    """
    frequencies = np.linspace(0, 0.5, n_frequencies)

    # Individual MTF components
    mtf_opt = diffraction_mtf(frequencies, wavelength_um, f_number, pixel_pitch_um)
    mtf_det = detector_mtf(frequencies, fill_factor)
    mtf_mot = motion_mtf(frequencies, motion_pixels) if motion_pixels > 0 else None
    mtf_jit = jitter_mtf(frequencies, jitter_sigma_pixels) if jitter_sigma_pixels > 0 else None

    # Combined system MTF
    mtf_sys = system_mtf(mtf_opt, mtf_det, mtf_mot, mtf_jit)

    # Cutoff frequency
    fc_um = 1 / (wavelength_um * f_number)
    fc_pixel = fc_um * pixel_pitch_um

    return MTFResult(
        frequencies=frequencies,
        mtf_h=mtf_sys,
        mtf_v=mtf_sys,
        mtf_radial=mtf_sys,
        nyquist_frequency=0.5,
        cutoff_frequency=fc_pixel,
    )


def mtf_frequency_at_threshold(
    frequencies: NDArray[np.floating],
    mtf: NDArray[np.floating],
    threshold: float = 0.5,
) -> float:
    """Find frequency where MTF drops to threshold.

    Args:
        frequencies: Frequency array in cycles/pixel
        mtf: MTF values
        threshold: MTF threshold (default 0.5 for MTF50)

    Returns:
        Frequency at which MTF equals threshold
    """
    # Find crossings
    crossings = np.where(np.diff(np.sign(mtf - threshold)))[0]

    if len(crossings) == 0:
        if mtf[-1] > threshold:
            return frequencies[-1]
        else:
            return 0.0

    # Linear interpolation at first crossing
    i = crossings[0]
    f1, f2 = frequencies[i], frequencies[i + 1]
    m1, m2 = mtf[i], mtf[i + 1]

    if m2 == m1:
        return f1

    return f1 + (threshold - m1) * (f2 - f1) / (m2 - m1)


def mtf50(
    frequencies: NDArray[np.floating],
    mtf: NDArray[np.floating],
) -> float:
    """Compute MTF50 - frequency at 50% MTF.

    Args:
        frequencies: Frequency array
        mtf: MTF values

    Returns:
        MTF50 frequency in same units as input
    """
    return mtf_frequency_at_threshold(frequencies, mtf, 0.5)


def mtf10(
    frequencies: NDArray[np.floating],
    mtf: NDArray[np.floating],
) -> float:
    """Compute MTF10 - frequency at 10% MTF.

    Args:
        frequencies: Frequency array
        mtf: MTF values

    Returns:
        MTF10 frequency in same units as input
    """
    return mtf_frequency_at_threshold(frequencies, mtf, 0.1)


def area_under_mtf(
    frequencies: NDArray[np.floating],
    mtf: NDArray[np.floating],
) -> float:
    """Compute area under MTF curve (image quality metric).

    Args:
        frequencies: Frequency array
        mtf: MTF values

    Returns:
        Area under MTF (higher is better)
    """
    # Use trapezoid (trapz is deprecated in newer numpy)
    try:
        return float(np.trapezoid(mtf, frequencies))
    except AttributeError:
        return float(np.trapz(mtf, frequencies))
