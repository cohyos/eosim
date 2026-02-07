"""
Optical system modeling for electro-optical simulation.

This module provides physically-based optical modeling including:
- Modulation Transfer Function (MTF) modeling
- Point Spread Function (PSF) calculation
- Diffraction-limited optics
- Aberration modeling
- Optical system cascades
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import ndimage
from scipy.special import j1  # Bessel function for Airy disk


class AberrationType(Enum):
    """Common optical aberration types."""
    DEFOCUS = "defocus"
    SPHERICAL = "spherical"
    COMA = "coma"
    ASTIGMATISM = "astigmatism"
    FIELD_CURVATURE = "field_curvature"
    DISTORTION = "distortion"
    CHROMATIC = "chromatic"


@dataclass
class OpticalSystem:
    """
    Optical system parameters for imaging simulation.

    Defines the key parameters that affect image quality through
    diffraction, aberrations, and detector sampling.
    """
    focal_length_mm: float = 100.0  # Effective focal length
    aperture_mm: float = 50.0  # Clear aperture diameter
    wavelength_um: float = 10.0  # Operating wavelength
    pixel_pitch_um: float = 15.0  # Detector pixel size

    # Aberration coefficients (waves RMS)
    defocus_waves: float = 0.0
    spherical_waves: float = 0.0
    coma_waves: float = 0.0
    astigmatism_waves: float = 0.0

    # Obscuration (for reflective systems)
    central_obscuration_ratio: float = 0.0  # 0-1

    # Transmission
    optical_transmission: float = 0.85  # Total optical transmission

    @property
    def f_number(self) -> float:
        """Calculate F-number (focal ratio)."""
        return self.focal_length_mm / self.aperture_mm

    @property
    def diffraction_cutoff(self) -> float:
        """
        Calculate diffraction cutoff frequency in cycles/mm.

        f_cutoff = 1 / (λ × F/#)
        """
        wavelength_mm = self.wavelength_um / 1000.0
        return 1.0 / (wavelength_mm * self.f_number)

    @property
    def nyquist_frequency(self) -> float:
        """
        Calculate Nyquist frequency based on pixel pitch in cycles/mm.

        f_nyquist = 1 / (2 × pixel_pitch)
        """
        pixel_pitch_mm = self.pixel_pitch_um / 1000.0
        return 1.0 / (2.0 * pixel_pitch_mm)

    @property
    def airy_disk_radius_um(self) -> float:
        """
        Calculate Airy disk radius (first null) in micrometers.

        r_airy = 1.22 × λ × F/#
        """
        return 1.22 * self.wavelength_um * self.f_number

    @property
    def strehl_ratio(self) -> float:
        """
        Estimate Strehl ratio from wavefront error.

        Using Maréchal approximation: S ≈ exp(-(2π×σ)²)
        where σ is the RMS wavefront error in waves.
        """
        total_wfe = math.sqrt(
            self.defocus_waves**2 +
            self.spherical_waves**2 +
            self.coma_waves**2 +
            self.astigmatism_waves**2
        )
        return math.exp(-(2 * math.pi * total_wfe)**2)

    @property
    def ifov_mrad(self) -> float:
        """Calculate instantaneous field of view in milliradians."""
        return self.pixel_pitch_um / self.focal_length_mm


def calculate_diffraction_mtf(
    frequencies: np.ndarray,
    optical_system: OpticalSystem,
    obscuration_ratio: Optional[float] = None
) -> np.ndarray:
    """
    Calculate diffraction-limited MTF.

    For an unobscured circular aperture, the MTF is:
    MTF(f) = (2/π) × [arccos(f/f_c) - (f/f_c)×√(1-(f/f_c)²)]

    For an obscured aperture, uses the obscured pupil formula.

    Args:
        frequencies: Spatial frequencies in cycles/mm
        optical_system: Optical system parameters
        obscuration_ratio: Optional override for central obscuration

    Returns:
        MTF values (0-1) at each frequency
    """
    f_cutoff = optical_system.diffraction_cutoff
    eps = obscuration_ratio if obscuration_ratio is not None else optical_system.central_obscuration_ratio

    # Normalized frequency
    v = frequencies / f_cutoff
    v = np.clip(v, 0, 1)

    if eps == 0:
        # Unobscured circular aperture
        mtf = (2 / math.pi) * (np.arccos(v) - v * np.sqrt(1 - v**2))
    else:
        # Obscured aperture (annular pupil)
        # Simplified formula - accurate for small obscurations
        area_ratio = 1 - eps**2
        mtf_unobscured = (2 / math.pi) * (np.arccos(v) - v * np.sqrt(1 - v**2))

        # Correction for central obscuration
        # MTF decreases and has side lobes
        v_eps = v * eps
        mtf_inner = np.where(
            v_eps < 1,
            (2 / math.pi) * (np.arccos(v_eps) - v_eps * np.sqrt(1 - v_eps**2)),
            0
        )
        mtf = (mtf_unobscured - eps**2 * mtf_inner) / area_ratio

    # Set MTF to 0 beyond cutoff
    mtf = np.where(v >= 1, 0, mtf)

    return np.clip(mtf, 0, 1)


def calculate_detector_mtf(
    frequencies: np.ndarray,
    pixel_pitch_um: float
) -> np.ndarray:
    """
    Calculate detector sampling MTF (sinc function).

    MTF_detector(f) = sinc(π × f × d)

    where d is the pixel pitch.

    Args:
        frequencies: Spatial frequencies in cycles/mm
        pixel_pitch_um: Pixel pitch in micrometers

    Returns:
        MTF values (0-1) at each frequency
    """
    pixel_pitch_mm = pixel_pitch_um / 1000.0
    arg = math.pi * frequencies * pixel_pitch_mm

    # sinc(x) = sin(x) / x, with sinc(0) = 1
    with np.errstate(divide='ignore', invalid='ignore'):
        mtf = np.where(arg == 0, 1.0, np.abs(np.sin(arg) / arg))

    return mtf


def calculate_motion_mtf(
    frequencies: np.ndarray,
    blur_length_mm: float
) -> np.ndarray:
    """
    Calculate motion blur MTF.

    MTF_motion(f) = sinc(π × f × L)

    where L is the blur length on the focal plane.

    Args:
        frequencies: Spatial frequencies in cycles/mm
        blur_length_mm: Motion blur length in mm

    Returns:
        MTF values (0-1) at each frequency
    """
    if blur_length_mm <= 0:
        return np.ones_like(frequencies)

    arg = math.pi * frequencies * blur_length_mm

    with np.errstate(divide='ignore', invalid='ignore'):
        mtf = np.where(arg == 0, 1.0, np.abs(np.sin(arg) / arg))

    return mtf


def calculate_jitter_mtf(
    frequencies: np.ndarray,
    jitter_rms_mm: float
) -> np.ndarray:
    """
    Calculate jitter/vibration MTF (Gaussian blur).

    MTF_jitter(f) = exp(-2 × (π × σ × f)²)

    where σ is the RMS jitter amplitude.

    Args:
        frequencies: Spatial frequencies in cycles/mm
        jitter_rms_mm: RMS jitter amplitude in mm on focal plane

    Returns:
        MTF values (0-1) at each frequency
    """
    if jitter_rms_mm <= 0:
        return np.ones_like(frequencies)

    return np.exp(-2 * (math.pi * jitter_rms_mm * frequencies)**2)


def calculate_aberration_mtf(
    frequencies: np.ndarray,
    optical_system: OpticalSystem
) -> np.ndarray:
    """
    Calculate MTF degradation due to aberrations.

    Uses Hopkins' formula for defocus and spherical aberration.

    Args:
        frequencies: Spatial frequencies in cycles/mm
        optical_system: Optical system with aberration coefficients

    Returns:
        MTF values (0-1) at each frequency
    """
    f_cutoff = optical_system.diffraction_cutoff
    v = frequencies / f_cutoff
    v = np.clip(v, 0, 1)

    # Get diffraction MTF as baseline
    mtf_diff = calculate_diffraction_mtf(frequencies, optical_system)

    # Wavefront error reduces MTF
    # Using Strehl approximation scaled by spatial frequency
    total_wfe = math.sqrt(
        optical_system.defocus_waves**2 +
        optical_system.spherical_waves**2 +
        optical_system.coma_waves**2 +
        optical_system.astigmatism_waves**2
    )

    if total_wfe > 0:
        # Wavefront error affects MTF more at lower frequencies
        wfe_factor = np.exp(-((2 * math.pi * total_wfe)**2) * (1 - v**2))
        mtf = mtf_diff * wfe_factor
    else:
        mtf = mtf_diff

    return np.clip(mtf, 0, 1)


@dataclass
class SystemMTF:
    """Complete system MTF model combining all contributors."""
    optical_system: OpticalSystem
    blur_length_mm: float = 0.0  # Motion blur
    jitter_rms_mm: float = 0.0  # Jitter/vibration

    # Additional factors
    atmospheric_mtf: Optional[np.ndarray] = None  # From turbulence model
    electronics_mtf: float = 0.95  # Electronics bandwidth limiting

    def calculate(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Calculate total system MTF.

        System MTF is the product of all individual MTF contributions.

        Args:
            frequencies: Spatial frequencies in cycles/mm

        Returns:
            Total system MTF values
        """
        # Diffraction with aberrations
        mtf_optics = calculate_aberration_mtf(frequencies, self.optical_system)

        # Detector sampling
        mtf_detector = calculate_detector_mtf(frequencies, self.optical_system.pixel_pitch_um)

        # Motion blur
        mtf_motion = calculate_motion_mtf(frequencies, self.blur_length_mm)

        # Jitter
        mtf_jitter = calculate_jitter_mtf(frequencies, self.jitter_rms_mm)

        # Combine all MTFs (multiplicative)
        mtf_total = mtf_optics * mtf_detector * mtf_motion * mtf_jitter * self.electronics_mtf

        # Apply atmospheric MTF if provided
        if self.atmospheric_mtf is not None:
            mtf_atm = np.interp(frequencies,
                               np.linspace(0, frequencies.max(), len(self.atmospheric_mtf)),
                               self.atmospheric_mtf)
            mtf_total *= mtf_atm

        return np.clip(mtf_total, 0, 1)

    def get_effective_resolution(self, threshold: float = 0.1) -> float:
        """
        Get effective resolution where MTF drops below threshold.

        Args:
            threshold: MTF threshold (default 0.1 for limiting resolution)

        Returns:
            Limiting frequency in cycles/mm
        """
        frequencies = np.linspace(0, self.optical_system.nyquist_frequency * 2, 1000)
        mtf = self.calculate(frequencies)

        # Find where MTF drops below threshold
        above_threshold = np.where(mtf >= threshold)[0]
        if len(above_threshold) == 0:
            return 0.0
        return frequencies[above_threshold[-1]]


def calculate_psf(
    optical_system: OpticalSystem,
    size_pixels: int = 64,
    oversampling: int = 4
) -> np.ndarray:
    """
    Calculate the Point Spread Function (PSF).

    For a diffraction-limited system, this is the Airy pattern.
    Aberrations are included via the wavefront error.

    Args:
        optical_system: Optical system parameters
        size_pixels: Output size in detector pixels
        oversampling: Oversampling factor for accurate PSF

    Returns:
        Normalized PSF (sum = 1)
    """
    # Calculate PSF at higher resolution then bin to detector pixels
    full_size = size_pixels * oversampling
    center = full_size // 2

    # Physical size of PSF array
    pixel_pitch_um = optical_system.pixel_pitch_um / oversampling
    x = (np.arange(full_size) - center) * pixel_pitch_um
    y = (np.arange(full_size) - center) * pixel_pitch_um
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)

    # Airy disk parameter
    # r_airy = 1.22 × λ × F/#
    airy_radius = optical_system.airy_disk_radius_um

    # Airy pattern: I(r) = [2×J1(x)/x]² where x = π×r/(λ×F/#)
    with np.errstate(divide='ignore', invalid='ignore'):
        x_airy = math.pi * r / (optical_system.wavelength_um * optical_system.f_number)
        airy = np.where(x_airy == 0, 1.0, (2 * j1(x_airy) / x_airy)**2)

    # Apply Strehl ratio for aberrated systems
    strehl = optical_system.strehl_ratio
    if strehl < 1.0:
        # Add aberration halo (simplified Gaussian model)
        sigma = airy_radius * (1 / strehl - 1)**0.5
        halo = np.exp(-r**2 / (2 * sigma**2))
        psf = strehl * airy + (1 - strehl) * halo
    else:
        psf = airy

    # Apply central obscuration if present
    if optical_system.central_obscuration_ratio > 0:
        eps = optical_system.central_obscuration_ratio
        # Modify PSF for annular aperture (simplified)
        psf *= (1 + eps**2 * np.cos(2 * math.pi * r / airy_radius)) / (1 + eps**2)

    # Normalize
    psf = psf / np.sum(psf)

    # Bin down to detector resolution
    if oversampling > 1:
        psf_binned = psf.reshape(size_pixels, oversampling, size_pixels, oversampling).sum(axis=(1, 3))
        psf_binned = psf_binned / np.sum(psf_binned)
        return psf_binned

    return psf


def apply_psf(
    image: np.ndarray,
    psf: np.ndarray,
    mode: str = 'same'
) -> np.ndarray:
    """
    Apply PSF to an image via convolution.

    Args:
        image: Input image
        psf: Point spread function
        mode: Convolution mode ('same', 'full', 'valid')

    Returns:
        Blurred image
    """
    # Use FFT convolution for efficiency
    from scipy.signal import fftconvolve
    result = fftconvolve(image, psf, mode=mode)
    return result


def apply_mtf_filter(
    image: np.ndarray,
    system_mtf: SystemMTF
) -> np.ndarray:
    """
    Apply MTF degradation to an image in the frequency domain.

    Args:
        image: Input image
        system_mtf: System MTF model

    Returns:
        Degraded image
    """
    height, width = image.shape[:2]

    # Create frequency grid
    fy = np.fft.fftfreq(height)
    fx = np.fft.fftfreq(width)
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    freq_radius = np.sqrt(fx_grid**2 + fy_grid**2)

    # Convert to cycles/mm (assuming 1 pixel = pixel_pitch)
    pixel_pitch_mm = system_mtf.optical_system.pixel_pitch_um / 1000.0
    freq_radius_cpmm = freq_radius / pixel_pitch_mm

    # Calculate MTF at each frequency
    mtf_filter = system_mtf.calculate(freq_radius_cpmm.flatten()).reshape(height, width)

    # Apply in frequency domain
    if len(image.shape) == 3:
        # Color image
        result = np.zeros_like(image, dtype=float)
        for c in range(image.shape[2]):
            img_fft = np.fft.fft2(image[:, :, c])
            img_fft_filtered = img_fft * np.fft.fftshift(mtf_filter)
            result[:, :, c] = np.real(np.fft.ifft2(img_fft_filtered))
    else:
        # Grayscale
        img_fft = np.fft.fft2(image)
        img_fft_filtered = img_fft * np.fft.fftshift(mtf_filter)
        result = np.real(np.fft.ifft2(img_fft_filtered))

    return result


@dataclass
class DistortionModel:
    """Geometric distortion model for lens systems."""
    # Radial distortion coefficients (Brown-Conrady model)
    k1: float = 0.0  # Primary radial
    k2: float = 0.0  # Secondary radial
    k3: float = 0.0  # Tertiary radial

    # Tangential distortion
    p1: float = 0.0
    p2: float = 0.0

    # Image center (normalized, 0 = center)
    cx: float = 0.0
    cy: float = 0.0

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply distortion to an image."""
        height, width = image.shape[:2]
        cy_px = height / 2 + self.cy * height
        cx_px = width / 2 + self.cx * width

        # Create coordinate grids
        y, x = np.ogrid[:height, :width]

        # Normalize coordinates
        x_norm = (x - cx_px) / (width / 2)
        y_norm = (y - cy_px) / (height / 2)
        r2 = x_norm**2 + y_norm**2
        r4 = r2**2
        r6 = r2**3

        # Radial distortion
        radial_factor = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6

        # Tangential distortion
        x_tangent = 2 * self.p1 * x_norm * y_norm + self.p2 * (r2 + 2 * x_norm**2)
        y_tangent = self.p1 * (r2 + 2 * y_norm**2) + 2 * self.p2 * x_norm * y_norm

        # Distorted coordinates
        x_distorted = x_norm * radial_factor + x_tangent
        y_distorted = y_norm * radial_factor + y_tangent

        # Convert back to pixel coordinates
        x_px = x_distorted * (width / 2) + cx_px
        y_px = y_distorted * (height / 2) + cy_px

        # Remap image
        from scipy.ndimage import map_coordinates
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = map_coordinates(
                    image[:, :, c],
                    [y_px, x_px],
                    order=1,
                    mode='constant'
                )
        else:
            result = map_coordinates(image, [y_px, x_px], order=1, mode='constant')

        return result

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Remove distortion from an image (inverse mapping)."""
        # Use negative distortion coefficients for inverse
        inverse_model = DistortionModel(
            k1=-self.k1,
            k2=-self.k2,
            k3=-self.k3,
            p1=-self.p1,
            p2=-self.p2,
            cx=self.cx,
            cy=self.cy
        )
        return inverse_model.apply(image)


@dataclass
class TurbulenceModel:
    """Atmospheric turbulence model affecting optical path."""
    cn2: float = 1e-14  # Refractive index structure constant (m^-2/3)
    path_length_m: float = 1000.0  # Path length through turbulence
    wavelength_um: float = 10.0

    @property
    def fried_parameter(self) -> float:
        """
        Calculate Fried parameter r0 (coherence diameter).

        r0 = 0.185 × (λ²/(Cn² × L))^(3/5)
        """
        wavelength_m = self.wavelength_um * 1e-6
        r0 = 0.185 * (wavelength_m**2 / (self.cn2 * self.path_length_m))**(3/5)
        return r0

    @property
    def seeing_arcsec(self) -> float:
        """Calculate seeing angle (FWHM) in arcseconds."""
        wavelength_m = self.wavelength_um * 1e-6
        r0 = self.fried_parameter
        seeing_rad = 0.98 * wavelength_m / r0
        return seeing_rad * 206265  # Convert to arcsec

    def calculate_mtf(self, frequencies: np.ndarray, aperture_m: float) -> np.ndarray:
        """
        Calculate atmospheric MTF due to turbulence.

        MTF_atm(f) = exp(-3.44 × (λ×f/r0)^(5/3))

        Args:
            frequencies: Spatial frequencies in cycles/rad
            aperture_m: Aperture diameter in meters

        Returns:
            Atmospheric MTF values
        """
        wavelength_m = self.wavelength_um * 1e-6
        r0 = self.fried_parameter

        # Convert to cycles/rad if needed
        # For long-exposure MTF
        arg = 3.44 * (wavelength_m * frequencies / r0)**(5/3)
        mtf = np.exp(-arg)

        return np.clip(mtf, 0, 1)

    def generate_phase_screen(self, size: int, pixel_scale_m: float) -> np.ndarray:
        """
        Generate a random atmospheric phase screen using FFT method.

        Args:
            size: Screen size in pixels
            pixel_scale_m: Physical size per pixel in meters

        Returns:
            Phase screen in radians
        """
        # Kolmogorov power spectrum
        # Φ(f) ∝ f^(-11/3)
        fx = np.fft.fftfreq(size, pixel_scale_m)
        fy = np.fft.fftfreq(size, pixel_scale_m)
        fx_grid, fy_grid = np.meshgrid(fx, fy)
        f = np.sqrt(fx_grid**2 + fy_grid**2)

        # Avoid division by zero
        f[0, 0] = 1

        # Power spectrum (Kolmogorov)
        power = f**(-11/3)
        power[0, 0] = 0

        # Random complex field
        random_phase = np.random.random((size, size)) * 2 * np.pi
        random_amplitude = np.sqrt(power) * np.exp(1j * random_phase)

        # Inverse FFT to get phase screen
        phase_screen = np.real(np.fft.ifft2(random_amplitude))

        # Scale by r0
        r0 = self.fried_parameter
        wavelength_m = self.wavelength_um * 1e-6
        phase_screen *= (pixel_scale_m / r0)**(5/6) * (2 * np.pi / wavelength_m)

        return phase_screen


class OpticalSimulator:
    """
    High-level optical simulation interface.

    Combines all optical effects into a single processing pipeline.
    """

    def __init__(self, optical_system: OpticalSystem):
        self.optical_system = optical_system
        self.system_mtf = SystemMTF(optical_system)
        self.distortion = DistortionModel()
        self.turbulence = None
        self._psf = None

    def set_motion_blur(self, blur_pixels: float, angle_deg: float = 0.0):
        """Set motion blur parameters."""
        # Convert pixels to mm on focal plane
        blur_mm = blur_pixels * self.optical_system.pixel_pitch_um / 1000.0
        self.system_mtf.blur_length_mm = blur_mm

    def set_jitter(self, jitter_rms_pixels: float):
        """Set jitter/vibration RMS in pixels."""
        jitter_mm = jitter_rms_pixels * self.optical_system.pixel_pitch_um / 1000.0
        self.system_mtf.jitter_rms_mm = jitter_mm

    def set_turbulence(self, cn2: float, path_length_m: float):
        """Set atmospheric turbulence parameters."""
        self.turbulence = TurbulenceModel(
            cn2=cn2,
            path_length_m=path_length_m,
            wavelength_um=self.optical_system.wavelength_um
        )

    def set_distortion(self, k1: float = 0.0, k2: float = 0.0, k3: float = 0.0):
        """Set radial distortion coefficients."""
        self.distortion = DistortionModel(k1=k1, k2=k2, k3=k3)

    def get_psf(self, size_pixels: int = 32) -> np.ndarray:
        """Get the current PSF."""
        if self._psf is None or self._psf.shape[0] != size_pixels:
            self._psf = calculate_psf(self.optical_system, size_pixels)
        return self._psf

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all optical effects to an image.

        Processing order:
        1. Geometric distortion
        2. PSF convolution (diffraction + aberrations)
        3. MTF filtering (motion, jitter)
        4. Turbulence effects

        Args:
            image: Input image (grayscale or RGB)

        Returns:
            Optically degraded image
        """
        result = image.astype(float)

        # 1. Apply geometric distortion
        if self.distortion.k1 != 0 or self.distortion.k2 != 0:
            result = self.distortion.apply(result)

        # 2. Apply PSF (diffraction + aberrations)
        psf = self.get_psf(min(32, image.shape[0] // 4))
        result = apply_psf(result, psf)

        # 3. Apply additional MTF effects (motion, jitter)
        if self.system_mtf.blur_length_mm > 0 or self.system_mtf.jitter_rms_mm > 0:
            # Apply motion and jitter via MTF
            result = apply_mtf_filter(result, self.system_mtf)

        # 4. Apply turbulence if enabled
        if self.turbulence is not None:
            # Generate and apply phase screen
            phase_screen = self.turbulence.generate_phase_screen(
                max(image.shape[:2]),
                self.optical_system.pixel_pitch_um * 1e-6
            )
            # Apply as intensity modulation (simplified)
            modulation = 1 + 0.1 * phase_screen[:image.shape[0], :image.shape[1]]
            if len(result.shape) == 3:
                result *= modulation[:, :, np.newaxis]
            else:
                result *= modulation

        return result

    def get_system_performance(self) -> Dict:
        """Get summary of optical system performance."""
        return {
            'f_number': self.optical_system.f_number,
            'diffraction_cutoff_cpmm': self.optical_system.diffraction_cutoff,
            'nyquist_frequency_cpmm': self.optical_system.nyquist_frequency,
            'airy_disk_radius_um': self.optical_system.airy_disk_radius_um,
            'strehl_ratio': self.optical_system.strehl_ratio,
            'ifov_mrad': self.optical_system.ifov_mrad,
            'limiting_resolution_cpmm': self.system_mtf.get_effective_resolution(0.1),
        }


# Convenience factory functions
def create_diffraction_limited_system(
    focal_length_mm: float,
    f_number: float,
    wavelength_um: float,
    pixel_pitch_um: float
) -> OpticalSystem:
    """Create a diffraction-limited optical system."""
    aperture_mm = focal_length_mm / f_number
    return OpticalSystem(
        focal_length_mm=focal_length_mm,
        aperture_mm=aperture_mm,
        wavelength_um=wavelength_um,
        pixel_pitch_um=pixel_pitch_um
    )


def create_aberrated_system(
    focal_length_mm: float,
    f_number: float,
    wavelength_um: float,
    pixel_pitch_um: float,
    wfe_rms_waves: float = 0.1
) -> OpticalSystem:
    """Create an optical system with specified wavefront error."""
    system = create_diffraction_limited_system(
        focal_length_mm, f_number, wavelength_um, pixel_pitch_um
    )
    # Distribute WFE among aberration types
    system.defocus_waves = wfe_rms_waves * 0.5
    system.spherical_waves = wfe_rms_waves * 0.3
    system.coma_waves = wfe_rms_waves * 0.15
    system.astigmatism_waves = wfe_rms_waves * 0.05
    return system
