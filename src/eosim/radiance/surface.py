"""
Surface radiance calculations for EOSIM.

Computes leaving radiance from surfaces considering thermal emission
and reflection of incident radiation.
"""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from eosim.core.spectral import SpectralBand
from eosim.core.compat import integrate_trapz
from eosim.radiance.planck import spectral_radiance, band_radiance


@dataclass
class SurfaceProperties:
    """Radiometric properties of a surface.

    Attributes:
        emissivity: Spectral or broadband emissivity (0-1)
        temperature_K: Surface temperature in Kelvin
        reflectance: Reflectance (computed from emissivity if not provided)
    """

    emissivity: Union[float, NDArray[np.floating]]
    temperature_K: float
    reflectance: Optional[Union[float, NDArray[np.floating]]] = None

    def __post_init__(self) -> None:
        """Compute reflectance from Kirchhoff's law if not provided."""
        if self.reflectance is None:
            # For opaque surfaces: ε + ρ = 1
            self.reflectance = 1.0 - np.asarray(self.emissivity)


@dataclass
class IncidentRadiation:
    """Incident radiation on a surface.

    Attributes:
        downwelling_radiance: Downwelling sky/atmospheric radiance [W/(m²·sr·μm)]
        solar_irradiance: Direct solar irradiance [W/(m²·μm)]
        diffuse_irradiance: Diffuse sky irradiance [W/(m²·μm)]
        reflected_radiance: Radiance from surrounding surfaces [W/(m²·sr·μm)]
    """

    downwelling_radiance: Union[float, NDArray[np.floating]] = 0.0
    solar_irradiance: Union[float, NDArray[np.floating]] = 0.0
    diffuse_irradiance: Union[float, NDArray[np.floating]] = 0.0
    reflected_radiance: Union[float, NDArray[np.floating]] = 0.0


def surface_spectral_radiance(
    wavelength_um: Union[float, NDArray[np.floating]],
    surface: SurfaceProperties,
    incident: Optional[IncidentRadiation] = None,
) -> Union[float, NDArray[np.floating]]:
    """Compute spectral leaving radiance from a surface.

    L_leaving = ε × L_bb(T) + ρ × L_reflected

    For thermal IR (MWIR/LWIR), emission dominates and reflection
    of downwelling sky radiance is the main reflected component.

    Args:
        wavelength_um: Wavelength(s) in micrometers
        surface: Surface properties (emissivity, temperature)
        incident: Incident radiation (optional)

    Returns:
        Leaving spectral radiance in W/(m²·sr·μm)
    """
    # Thermal emission component
    L_blackbody = spectral_radiance(wavelength_um, surface.temperature_K)
    L_emitted = np.asarray(surface.emissivity) * L_blackbody

    if incident is None:
        return L_emitted

    # Reflected component (Lambertian assumption)
    L_incident = np.asarray(incident.downwelling_radiance)
    L_incident = L_incident + np.asarray(incident.reflected_radiance)

    # Add solar reflection for daytime visible/SWIR
    # Convert irradiance to equivalent radiance (divide by π for Lambertian)
    if np.any(incident.solar_irradiance > 0):
        L_incident = L_incident + np.asarray(incident.solar_irradiance) / np.pi
    if np.any(incident.diffuse_irradiance > 0):
        L_incident = L_incident + np.asarray(incident.diffuse_irradiance) / np.pi

    L_reflected = np.asarray(surface.reflectance) * L_incident

    return L_emitted + L_reflected


def surface_band_radiance(
    band: SpectralBand,
    surface: SurfaceProperties,
    incident: Optional[IncidentRadiation] = None,
    n_samples: int = 50,
) -> float:
    """Compute band-integrated leaving radiance from a surface.

    Args:
        band: Spectral band for integration
        surface: Surface properties
        incident: Incident radiation (optional)
        n_samples: Number of wavelength samples

    Returns:
        Band-integrated radiance in W/(m²·sr)
    """
    wavelengths = np.linspace(band.lambda_min_um, band.lambda_max_um, n_samples)
    spectral_L = surface_spectral_radiance(wavelengths, surface, incident)
    return float(integrate_trapz(spectral_L, wavelengths))


def apparent_temperature(
    measured_radiance: float,
    emissivity: float,
    band: SpectralBand,
    background_T: float = 300.0,
    n_samples: int = 50,
) -> float:
    """Compute apparent temperature correcting for emissivity.

    Inverts: L_measured = ε × L_bb(T_true) + (1-ε) × L_bb(T_background)

    Args:
        measured_radiance: Measured band radiance [W/(m²·sr)]
        emissivity: Surface emissivity
        band: Spectral band
        background_T: Background/environment temperature [K]
        n_samples: Samples for band integration

    Returns:
        Corrected apparent temperature in Kelvin
    """
    # Remove reflected background contribution
    L_background = band_radiance(band, background_T, n_samples)
    L_emitted = measured_radiance - (1 - emissivity) * L_background

    # Correct for emissivity
    L_blackbody = L_emitted / emissivity

    # Invert Planck function using Newton-Raphson
    T = background_T
    for _ in range(20):
        L = band_radiance(band, T, n_samples)
        dT = 0.1
        L_plus = band_radiance(band, T + dT, n_samples)
        dL_dT = (L_plus - L) / dT

        delta = (L_blackbody - L) / max(dL_dT, 1e-20)
        delta = np.clip(delta, -50, 50)
        T = T + delta

        if abs(delta) < 0.01:
            break

    return T


def emissivity_from_radiance(
    measured_radiance: float,
    surface_T: float,
    background_T: float,
    band: SpectralBand,
    n_samples: int = 50,
) -> float:
    """Estimate emissivity from measured radiance and known temperatures.

    L = ε × L_bb(T_surface) + (1-ε) × L_bb(T_background)
    ε = (L - L_background) / (L_surface - L_background)

    Args:
        measured_radiance: Measured band radiance [W/(m²·sr)]
        surface_T: Known surface temperature [K]
        background_T: Background temperature [K]
        band: Spectral band
        n_samples: Samples for integration

    Returns:
        Estimated emissivity (0-1)
    """
    L_surface = band_radiance(band, surface_T, n_samples)
    L_background = band_radiance(band, background_T, n_samples)

    denom = L_surface - L_background
    if abs(denom) < 1e-20:
        return 1.0  # Temperatures equal, can't determine

    emissivity = (measured_radiance - L_background) / denom
    return float(np.clip(emissivity, 0.0, 1.0))


def radiance_contrast(
    target: SurfaceProperties,
    background: SurfaceProperties,
    band: SpectralBand,
    incident: Optional[IncidentRadiation] = None,
    n_samples: int = 50,
) -> float:
    """Compute radiance contrast between target and background.

    ΔL = L_target - L_background

    Args:
        target: Target surface properties
        background: Background surface properties
        band: Spectral band
        incident: Incident radiation (same for both)
        n_samples: Samples for integration

    Returns:
        Radiance contrast in W/(m²·sr)
    """
    L_target = surface_band_radiance(band, target, incident, n_samples)
    L_background = surface_band_radiance(band, background, incident, n_samples)
    return L_target - L_background


def temperature_contrast(
    target: SurfaceProperties,
    background: SurfaceProperties,
    band: SpectralBand,
    n_samples: int = 50,
) -> float:
    """Compute equivalent temperature contrast (ΔT).

    The temperature difference that would produce the same radiance
    contrast for a blackbody at the background temperature.

    Args:
        target: Target surface properties
        background: Background surface properties
        band: Spectral band
        n_samples: Samples for integration

    Returns:
        Temperature contrast in Kelvin
    """
    delta_L = radiance_contrast(target, background, band, None, n_samples)

    # Compute dL/dT at background temperature
    T_bg = background.temperature_K
    L1 = band_radiance(band, T_bg, n_samples)
    L2 = band_radiance(band, T_bg + 1.0, n_samples)
    dL_dT = L2 - L1

    if abs(dL_dT) < 1e-20:
        return 0.0

    return delta_L / dL_dT


class SurfaceRadianceModel:
    """Model for computing surface radiance across an image.

    Handles arrays of surface temperatures and properties for
    efficient computation of full image radiance.
    """

    def __init__(
        self,
        band: SpectralBand,
        n_spectral_samples: int = 50,
    ) -> None:
        """Initialize surface radiance model.

        Args:
            band: Spectral band for computation
            n_spectral_samples: Number of wavelength samples
        """
        self.band = band
        self.n_samples = n_spectral_samples
        self._wavelengths = np.linspace(
            band.lambda_min_um, band.lambda_max_um, n_spectral_samples
        )

    def compute_radiance_image(
        self,
        temperature_map: NDArray[np.floating],
        emissivity: Union[float, NDArray[np.floating]] = 1.0,
        background_radiance: float = 0.0,
    ) -> NDArray[np.floating]:
        """Compute band radiance for a 2D temperature map.

        Args:
            temperature_map: 2D array of surface temperatures [K]
            emissivity: Scalar or 2D array of emissivity values
            background_radiance: Background/sky radiance for reflection [W/(m²·sr)]

        Returns:
            2D array of band-integrated radiance [W/(m²·sr)]
        """
        # Compute spectral radiance at all wavelengths for all pixels
        # Shape: (n_wavelengths, height, width)
        spectral_L = spectral_radiance(self._wavelengths, temperature_map)

        # Apply emissivity
        emissivity = np.asarray(emissivity)
        if emissivity.ndim == 0:
            L_emitted = emissivity * spectral_L
        else:
            L_emitted = emissivity[np.newaxis, :, :] * spectral_L

        # Add reflected background
        reflectance = 1.0 - emissivity
        if np.ndim(reflectance) == 0:
            L_reflected = reflectance * background_radiance
        else:
            L_reflected = reflectance * background_radiance

        L_total = L_emitted + L_reflected

        # Integrate over wavelength (axis 0)
        return integrate_trapz(L_total, self._wavelengths, axis=0)
