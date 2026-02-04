"""
Optical system modeling for EOSIM.

This module provides comprehensive optical system simulation including:
- Point Spread Function (PSF) models: Gaussian, Airy, Zernike
- Modulation Transfer Function (MTF) computation
- Wavefront aberrations via Zernike polynomials
- FFT-based PSF convolution with GPU acceleration support

Example usage:
    >>> from eosim.optics import create_optics_model, AberrationSet
    >>>
    >>> # Create diffraction-limited optics
    >>> optics = create_optics_model(
    ...     focal_length_mm=100,
    ...     aperture_diameter_mm=50,  # f/2
    ...     pixel_pitch_um=15,
    ...     wavelength_um=10.0,
    ... )
    >>>
    >>> # Compute PSF and MTF
    >>> psf = optics.compute_psf()
    >>> mtf = optics.compute_mtf()
    >>> print(f"PSF FWHM: {psf.fwhm_pixels:.2f} pixels")
    >>> print(f"MTF at Nyquist: {mtf.mtf_at_nyquist:.3f}")
    >>>
    >>> # Apply to radiance image
    >>> result = optics.apply(radiance_image)
"""

# Aberrations - Zernike polynomials and wavefront errors
from eosim.optics.aberrations import (
    AberrationSet,
    NOLL_TO_NM,
    ZERNIKE_NAMES,
    noll_to_nm,
    nm_to_noll,
    zernike_polynomial,
    zernike_noll,
    compute_wavefront,
    rms_wavefront,
    peak_to_valley,
    strehl_ratio,
    strehl_from_coefficients,
    create_pupil_grid,
    defocus_from_distance,
)

# PSF models
from eosim.optics.psf import (
    PSFResult,
    GaussianPSF,
    AiryPSF,
    ZernikePSF,
    MeasuredPSF,
    compute_psf,
    psf_encircled_energy,
    psf_to_ensquared_energy,
)

# MTF computation
from eosim.optics.mtf import (
    MTFResult,
    compute_mtf_from_psf,
    diffraction_mtf,
    detector_mtf,
    motion_mtf,
    jitter_mtf,
    atmospheric_mtf,
    system_mtf,
    compute_system_mtf,
    mtf_frequency_at_threshold,
    mtf50,
    mtf10,
    area_under_mtf,
)

# Convolution
from eosim.optics.convolution import (
    BoundaryMode,
    ConvolutionResult,
    ConvolutionEngine,
    fft_convolve_2d,
    spatial_convolve_2d,
    apply_psf,
    apply_psf_spectral,
    apply_psf_field_dependent,
    deconvolve_wiener,
    compute_otf,
    apply_otf,
)

# Base classes and models
from eosim.optics.base import (
    PSFModel,
    OpticsParameters,
    OpticsResult,
    OpticsModel,
    SimpleOpticsModel,
    OpticsQualityMetrics,
    compute_optics_metrics,
    create_optics_model,
)

__all__ = [
    # Aberrations
    "AberrationSet",
    "NOLL_TO_NM",
    "ZERNIKE_NAMES",
    "noll_to_nm",
    "nm_to_noll",
    "zernike_polynomial",
    "zernike_noll",
    "compute_wavefront",
    "rms_wavefront",
    "peak_to_valley",
    "strehl_ratio",
    "strehl_from_coefficients",
    "create_pupil_grid",
    "defocus_from_distance",
    # PSF
    "PSFResult",
    "GaussianPSF",
    "AiryPSF",
    "ZernikePSF",
    "MeasuredPSF",
    "compute_psf",
    "psf_encircled_energy",
    "psf_to_ensquared_energy",
    # MTF
    "MTFResult",
    "compute_mtf_from_psf",
    "diffraction_mtf",
    "detector_mtf",
    "motion_mtf",
    "jitter_mtf",
    "atmospheric_mtf",
    "system_mtf",
    "compute_system_mtf",
    "mtf_frequency_at_threshold",
    "mtf50",
    "mtf10",
    "area_under_mtf",
    # Convolution
    "BoundaryMode",
    "ConvolutionResult",
    "ConvolutionEngine",
    "fft_convolve_2d",
    "spatial_convolve_2d",
    "apply_psf",
    "apply_psf_spectral",
    "apply_psf_field_dependent",
    "deconvolve_wiener",
    "compute_otf",
    "apply_otf",
    # Base
    "PSFModel",
    "OpticsParameters",
    "OpticsResult",
    "OpticsModel",
    "SimpleOpticsModel",
    "OpticsQualityMetrics",
    "compute_optics_metrics",
    "create_optics_model",
]
