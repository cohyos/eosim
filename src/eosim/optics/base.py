"""
Base classes and interfaces for optical system modeling.

Provides the OpticsModel abstract base class and related configuration
dataclasses for defining optical system parameters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from eosim.optics.aberrations import AberrationSet
from eosim.optics.psf import (
    PSFResult,
    GaussianPSF,
    AiryPSF,
    ZernikePSF,
    compute_psf,
    psf_encircled_energy,
)
from eosim.optics.mtf import (
    MTFResult,
    compute_mtf_from_psf,
    compute_system_mtf,
    mtf50,
    mtf10,
)
from eosim.optics.convolution import (
    BoundaryMode,
    ConvolutionEngine,
    ConvolutionResult,
    apply_psf,
)


class PSFModel(Enum):
    """Available PSF computation models."""

    GAUSSIAN = "gaussian"   # Fast Gaussian approximation
    AIRY = "airy"          # Diffraction-limited Airy disk
    ZERNIKE = "zernike"    # Full Zernike aberration model


@dataclass
class OpticsParameters:
    """Configuration parameters for optical system.

    Attributes:
        focal_length_mm: Focal length in millimeters
        aperture_diameter_mm: Entrance pupil diameter in millimeters
        pixel_pitch_um: Detector pixel pitch in micrometers
        wavelength_um: Reference wavelength in micrometers
        transmittance: Optical system transmittance (0 to 1)
        vignetting_factor: Edge vignetting factor (0 to 1, 1=no vignetting)
        aberrations: Optical aberration coefficients
        psf_model: PSF computation model to use
        psf_size_pixels: Size of PSF kernel in pixels
    """

    focal_length_mm: float
    aperture_diameter_mm: float
    pixel_pitch_um: float
    wavelength_um: float = 10.0
    transmittance: float = 0.9
    vignetting_factor: float = 1.0
    aberrations: Optional[AberrationSet] = None
    psf_model: PSFModel = PSFModel.AIRY
    psf_size_pixels: int = 31

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.focal_length_mm <= 0:
            raise ValueError("Focal length must be positive")
        if self.aperture_diameter_mm <= 0:
            raise ValueError("Aperture diameter must be positive")
        if self.pixel_pitch_um <= 0:
            raise ValueError("Pixel pitch must be positive")
        if self.wavelength_um <= 0:
            raise ValueError("Wavelength must be positive")
        if not 0 <= self.transmittance <= 1:
            raise ValueError("Transmittance must be between 0 and 1")

    @property
    def f_number(self) -> float:
        """Compute f-number (focal ratio)."""
        return self.focal_length_mm / self.aperture_diameter_mm

    @property
    def fov_per_pixel_rad(self) -> float:
        """Compute instantaneous field of view per pixel in radians."""
        return (self.pixel_pitch_um / 1000) / self.focal_length_mm

    @property
    def fov_per_pixel_mrad(self) -> float:
        """Compute IFOV per pixel in milliradians."""
        return self.fov_per_pixel_rad * 1000

    @property
    def airy_radius_pixels(self) -> float:
        """Compute Airy disk radius in pixels."""
        airy_radius_um = 1.22 * self.wavelength_um * self.f_number
        return airy_radius_um / self.pixel_pitch_um

    @property
    def diffraction_cutoff_cy_per_pixel(self) -> float:
        """Diffraction cutoff frequency in cycles/pixel."""
        fc_um = 1 / (self.wavelength_um * self.f_number)
        return fc_um * self.pixel_pitch_um

    @property
    def is_diffraction_limited(self) -> bool:
        """Check if system is approximately diffraction-limited."""
        if self.aberrations is None:
            return True
        return self.aberrations.strehl > 0.8


@dataclass
class OpticsResult:
    """Result of optical system processing.

    Attributes:
        image: Processed image after optical effects
        psf: Point spread function used
        mtf: Modulation transfer function
        focal_plane_irradiance: Peak irradiance in W/m²
        effective_transmittance: Combined transmittance with vignetting
    """

    image: NDArray[np.floating]
    psf: PSFResult
    mtf: MTFResult
    focal_plane_irradiance: Optional[float] = None
    effective_transmittance: float = 1.0


class OpticsModel(ABC):
    """Abstract base class for optical system models.

    Defines the interface for applying optical effects to radiance images,
    including PSF blur, transmittance, and vignetting.
    """

    @abstractmethod
    def compute_psf(self) -> PSFResult:
        """Compute the point spread function.

        Returns:
            PSFResult with normalized PSF kernel
        """
        pass

    @abstractmethod
    def compute_mtf(self) -> MTFResult:
        """Compute the modulation transfer function.

        Returns:
            MTFResult with frequency response curves
        """
        pass

    @abstractmethod
    def apply(
        self,
        radiance_image: NDArray[np.floating],
    ) -> OpticsResult:
        """Apply optical effects to radiance image.

        Args:
            radiance_image: 2D array of scene radiance [W/m²/sr/μm]

        Returns:
            OpticsResult with processed image and metrics
        """
        pass

    @abstractmethod
    def radiance_to_irradiance(
        self,
        radiance: Union[float, NDArray[np.floating]],
    ) -> Union[float, NDArray[np.floating]]:
        """Convert scene radiance to focal plane irradiance.

        E_fp = π × L × τ / (4 × F#²)

        Args:
            radiance: Scene radiance in W/m²/sr/μm

        Returns:
            Focal plane irradiance in W/m²/μm
        """
        pass


class SimpleOpticsModel(OpticsModel):
    """Standard optical system model with configurable PSF.

    Implements a typical imaging optics chain:
    1. Scene radiance to focal plane irradiance
    2. PSF blur via FFT convolution
    3. Transmittance and vignetting
    """

    def __init__(
        self,
        params: OpticsParameters,
        use_gpu: bool = False,
    ) -> None:
        """Initialize optical system model.

        Args:
            params: Optical system parameters
            use_gpu: Whether to use GPU acceleration for convolution
        """
        self.params = params
        self._convolution_engine = ConvolutionEngine(use_gpu=use_gpu)
        self._cached_psf: Optional[PSFResult] = None
        self._cached_mtf: Optional[MTFResult] = None

    def compute_psf(self) -> PSFResult:
        """Compute PSF based on configured model."""
        if self._cached_psf is not None:
            return self._cached_psf

        model_name = self.params.psf_model.value

        psf = compute_psf(
            wavelength_um=self.params.wavelength_um,
            f_number=self.params.f_number,
            pixel_pitch_um=self.params.pixel_pitch_um,
            model=model_name,
            aberrations=self.params.aberrations,
            size_pixels=self.params.psf_size_pixels,
        )

        self._cached_psf = psf
        return psf

    def compute_mtf(self) -> MTFResult:
        """Compute MTF from PSF."""
        if self._cached_mtf is not None:
            return self._cached_mtf

        psf = self.compute_psf()
        mtf = compute_mtf_from_psf(psf.kernel, self.params.pixel_pitch_um)

        # Store cutoff frequency
        mtf.cutoff_frequency = self.params.diffraction_cutoff_cy_per_pixel

        self._cached_mtf = mtf
        return mtf

    def radiance_to_irradiance(
        self,
        radiance: Union[float, NDArray[np.floating]],
    ) -> Union[float, NDArray[np.floating]]:
        """Convert scene radiance to focal plane irradiance.

        Uses the radiometric transfer equation:
        E_fp = π × L × τ / (4 × F#²)
        """
        tau = self.params.transmittance
        f_num = self.params.f_number

        # Radiometric transfer: E = π L τ / (4 F#²)
        irradiance = np.pi * radiance * tau / (4 * f_num ** 2)

        return irradiance

    def apply_vignetting(
        self,
        image: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Apply vignetting (field-dependent falloff).

        Uses cos⁴ vignetting model modulated by vignetting_factor.

        Args:
            image: Input image

        Returns:
            Image with vignetting applied
        """
        if self.params.vignetting_factor >= 1.0:
            return image

        h, w = image.shape
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2

        # Normalized radial distance from center
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_max = np.sqrt(cx ** 2 + cy ** 2)
        r_norm = r / r_max

        # cos⁴ vignetting scaled by vignetting factor
        # At r=0: vignetting=1, at edges: vignetting=vignetting_factor
        vignetting = 1 - (1 - self.params.vignetting_factor) * r_norm ** 2
        vignetting = np.clip(vignetting, self.params.vignetting_factor, 1.0)

        return image * vignetting

    def apply(
        self,
        radiance_image: NDArray[np.floating],
    ) -> OpticsResult:
        """Apply complete optical chain to radiance image.

        Steps:
        1. Convert radiance to focal plane irradiance
        2. Apply PSF blur
        3. Apply transmittance and vignetting

        Args:
            radiance_image: Scene radiance in W/m²/sr/μm

        Returns:
            OpticsResult with processed image and metrics
        """
        # Get PSF and MTF
        psf = self.compute_psf()
        mtf = self.compute_mtf()

        # Convert to irradiance
        irradiance = self.radiance_to_irradiance(radiance_image)

        # Apply PSF blur
        conv_result = apply_psf(
            irradiance,
            psf.kernel,
            method="auto",
            boundary=BoundaryMode.REFLECT,
        )

        # Apply vignetting
        processed = self.apply_vignetting(conv_result.image)

        # Compute peak irradiance
        peak_irradiance = float(np.max(irradiance))

        return OpticsResult(
            image=processed,
            psf=psf,
            mtf=mtf,
            focal_plane_irradiance=peak_irradiance,
            effective_transmittance=self.params.transmittance * self.params.vignetting_factor,
        )

    def invalidate_cache(self) -> None:
        """Clear cached PSF and MTF (call after parameter changes)."""
        self._cached_psf = None
        self._cached_mtf = None
        self._convolution_engine.clear_cache()


@dataclass
class OpticsQualityMetrics:
    """Quality metrics for optical system characterization.

    Attributes:
        strehl_ratio: Strehl ratio (1.0 for diffraction-limited)
        mtf_at_nyquist: MTF value at Nyquist frequency
        mtf50: Spatial frequency at 50% MTF (cy/pixel)
        mtf10: Spatial frequency at 10% MTF (cy/pixel)
        encircled_energy_50: Radius for 50% encircled energy (pixels)
        encircled_energy_80: Radius for 80% encircled energy (pixels)
        psf_fwhm: PSF full width at half maximum (pixels)
        diffraction_cutoff: Diffraction cutoff frequency (cy/pixel)
    """

    strehl_ratio: float
    mtf_at_nyquist: float
    mtf50: float
    mtf10: float
    encircled_energy_50: float
    encircled_energy_80: float
    psf_fwhm: float
    diffraction_cutoff: float


def compute_optics_metrics(
    optics: OpticsModel,
) -> OpticsQualityMetrics:
    """Compute comprehensive optical quality metrics.

    Args:
        optics: Optical system model

    Returns:
        OpticsQualityMetrics with all quality indicators
    """
    psf = optics.compute_psf()
    mtf = optics.compute_mtf()

    # Find radii for encircled energy targets
    radii = np.linspace(0, psf.size_pixels // 2, 50)
    ee_values = [psf_encircled_energy(psf.kernel, r) for r in radii]

    def find_radius_for_ee(target: float) -> float:
        for r, ee in zip(radii, ee_values):
            if ee >= target:
                return float(r)
        return float(radii[-1])

    ee_50_radius = find_radius_for_ee(0.5)
    ee_80_radius = find_radius_for_ee(0.8)

    return OpticsQualityMetrics(
        strehl_ratio=psf.strehl_ratio,
        mtf_at_nyquist=mtf.mtf_at_nyquist,
        mtf50=mtf50(mtf.frequencies, mtf.mtf_radial),
        mtf10=mtf10(mtf.frequencies, mtf.mtf_radial),
        encircled_energy_50=ee_50_radius,
        encircled_energy_80=ee_80_radius,
        psf_fwhm=psf.fwhm_pixels,
        diffraction_cutoff=mtf.cutoff_frequency or 0.0,
    )


def create_optics_model(
    focal_length_mm: float,
    aperture_diameter_mm: float,
    pixel_pitch_um: float,
    wavelength_um: float = 10.0,
    model: str = "airy",
    aberrations: Optional[AberrationSet] = None,
    use_gpu: bool = False,
) -> SimpleOpticsModel:
    """Factory function to create optical system model.

    Args:
        focal_length_mm: Focal length in mm
        aperture_diameter_mm: Aperture diameter in mm
        pixel_pitch_um: Pixel pitch in micrometers
        wavelength_um: Reference wavelength in micrometers
        model: PSF model ("gaussian", "airy", or "zernike")
        aberrations: Optional aberration coefficients
        use_gpu: Whether to use GPU acceleration

    Returns:
        Configured SimpleOpticsModel
    """
    psf_model = PSFModel(model)

    params = OpticsParameters(
        focal_length_mm=focal_length_mm,
        aperture_diameter_mm=aperture_diameter_mm,
        pixel_pitch_um=pixel_pitch_um,
        wavelength_um=wavelength_um,
        aberrations=aberrations,
        psf_model=psf_model,
    )

    return SimpleOpticsModel(params, use_gpu=use_gpu)
