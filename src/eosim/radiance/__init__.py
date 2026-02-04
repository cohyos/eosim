"""Radiance module: blackbody calculations and surface radiance computation."""

from eosim.radiance.planck import (
    spectral_radiance,
    spectral_exitance,
    total_radiance,
    total_exitance,
    band_radiance,
    band_exitance,
    wien_peak_wavelength,
    temperature_from_peak,
    contrast_temperature,
    radiance_difference,
)
from eosim.radiance.surface import (
    SurfaceProperties,
    IncidentRadiation,
    surface_spectral_radiance,
    surface_band_radiance,
    apparent_temperature,
    emissivity_from_radiance,
    radiance_contrast,
    temperature_contrast,
    SurfaceRadianceModel,
)

__all__ = [
    # Planck functions
    "spectral_radiance",
    "spectral_exitance",
    "total_radiance",
    "total_exitance",
    "band_radiance",
    "band_exitance",
    "wien_peak_wavelength",
    "temperature_from_peak",
    "contrast_temperature",
    "radiance_difference",
    # Surface radiance
    "SurfaceProperties",
    "IncidentRadiation",
    "surface_spectral_radiance",
    "surface_band_radiance",
    "apparent_temperature",
    "emissivity_from_radiance",
    "radiance_contrast",
    "temperature_contrast",
    "SurfaceRadianceModel",
]
