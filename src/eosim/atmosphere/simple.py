"""
Simple atmosphere models for EOSIM.

Provides Beer-Lambert and empirical atmosphere models for
quick calculations without full radiative transfer.
"""

from dataclasses import dataclass
from typing import Union
import numpy as np
from numpy.typing import NDArray

from eosim.core.constants import CONSTANTS
from eosim.core.spectral import SpectralBand, BAND_MWIR, BAND_LWIR
from eosim.atmosphere.base import (
    AtmosphereModel,
    AtmosphereFidelity,
    PathGeometry,
    AtmosphereConditions,
    AtmosphereResult,
)


class BeerLambertAtmosphere(AtmosphereModel):
    """Simple Beer-Lambert atmosphere model.

    Uses wavelength-dependent extinction coefficients with
    visibility scaling for aerosol effects.

    τ(λ, R) = exp(-β(λ) × R)

    where β is the extinction coefficient and R is path length.
    """

    # Approximate extinction coefficients [1/km] at standard visibility (23 km)
    # Based on typical clear atmosphere values
    _EXTINCTION_COEFFS = {
        # Visible (0.4-0.7 μm): Rayleigh + aerosol dominated
        0.4: 0.20,
        0.5: 0.15,
        0.55: 0.12,
        0.6: 0.10,
        0.7: 0.08,
        # NIR (0.7-1.0 μm): Aerosol dominated
        0.8: 0.07,
        0.9: 0.065,
        1.0: 0.06,
        # SWIR (1.0-2.5 μm): Water vapor absorption bands
        1.2: 0.055,
        1.4: 0.25,  # H2O absorption
        1.6: 0.06,
        1.9: 0.35,  # H2O absorption
        2.2: 0.065,
        # MWIR (3-5 μm): CO2 at 4.3 μm
        3.0: 0.04,
        3.5: 0.035,
        4.0: 0.045,
        4.3: 0.30,  # CO2 absorption
        4.8: 0.04,
        5.0: 0.045,
        # LWIR (8-14 μm): Ozone at 9.6 μm
        8.0: 0.03,
        9.0: 0.035,
        9.6: 0.12,  # O3 absorption
        10.0: 0.025,
        11.0: 0.022,
        12.0: 0.025,
        14.0: 0.035,
    }

    def __init__(self, reference_visibility_km: float = 23.0) -> None:
        """Initialize Beer-Lambert model.

        Args:
            reference_visibility_km: Visibility for reference extinction values
        """
        self.reference_visibility = reference_visibility_km
        self._wavelengths = np.array(sorted(self._EXTINCTION_COEFFS.keys()))
        self._extinctions = np.array([self._EXTINCTION_COEFFS[w] for w in self._wavelengths])

    @property
    def fidelity(self) -> AtmosphereFidelity:
        return AtmosphereFidelity.SIMPLE

    def _get_extinction(
        self,
        wavelength_um: Union[float, NDArray[np.floating]],
        conditions: AtmosphereConditions,
    ) -> Union[float, NDArray[np.floating]]:
        """Get extinction coefficient at wavelength, scaled by visibility."""
        # Interpolate base extinction
        beta_ref = np.interp(wavelength_um, self._wavelengths, self._extinctions)

        # Scale by visibility (inverse relationship)
        # β ∝ 3.912 / V where V is visibility in km
        visibility_scale = self.reference_visibility / conditions.visibility_km
        beta = beta_ref * visibility_scale

        # Humidity correction for water vapor bands
        if conditions.relative_humidity > 0.5:
            humidity_factor = 1 + (conditions.relative_humidity - 0.5) * 0.5
            # Apply extra absorption at water vapor wavelengths
            wavelength_um = np.atleast_1d(wavelength_um)
            h2o_mask = ((wavelength_um > 1.3) & (wavelength_um < 1.5)) | \
                       ((wavelength_um > 1.8) & (wavelength_um < 2.0)) | \
                       ((wavelength_um > 5.5) & (wavelength_um < 7.5))
            if isinstance(beta, np.ndarray):
                beta[h2o_mask] *= humidity_factor
            elif np.any(h2o_mask):
                beta *= humidity_factor

        return beta

    def compute(
        self,
        wavelength_um: Union[float, NDArray[np.floating]],
        path: PathGeometry,
        conditions: AtmosphereConditions,
    ) -> AtmosphereResult:
        """Compute atmospheric transmission using Beer-Lambert law."""
        wavelength_um = np.atleast_1d(wavelength_um)

        # Get extinction coefficient [1/km]
        beta = self._get_extinction(wavelength_um, conditions)

        # Path length in km
        R_km = path.slant_range_m / 1000.0

        # Beer-Lambert transmission
        transmission = np.exp(-beta * R_km)

        # Simple path radiance estimate (thermal emission from atmosphere)
        # Using effective atmospheric temperature
        T_atm = conditions.temperature_K - 20  # Approximate atmospheric temp
        avg_emissivity = 1 - np.mean(transmission)

        # Approximate path radiance as graybody at atmospheric temperature
        from eosim.radiance.planck import spectral_radiance
        L_blackbody = spectral_radiance(wavelength_um, T_atm)
        path_radiance = avg_emissivity * L_blackbody * (1 - transmission)

        # Sky radiance (downwelling at surface)
        T_sky = conditions.temperature_K - 30  # Effective sky temperature
        L_sky = 0.9 * spectral_radiance(wavelength_um, T_sky)

        if len(wavelength_um) == 1:
            return AtmosphereResult(
                transmission=float(transmission[0]),
                path_radiance=float(path_radiance[0]),
                sky_radiance=float(L_sky[0]),
            )

        return AtmosphereResult(
            transmission=transmission,
            path_radiance=path_radiance,
            sky_radiance=L_sky,
        )


class ConstantAtmosphere(AtmosphereModel):
    """Constant transmission atmosphere model.

    Useful for testing and quick estimates with known transmission.
    """

    def __init__(
        self,
        transmission: float = 0.8,
        path_radiance: float = 0.0,
    ) -> None:
        """Initialize with constant values.

        Args:
            transmission: Constant transmission (0-1)
            path_radiance: Constant path radiance [W/(m²·sr·μm)]
        """
        self._transmission = transmission
        self._path_radiance = path_radiance

    @property
    def fidelity(self) -> AtmosphereFidelity:
        return AtmosphereFidelity.SIMPLE

    def compute(
        self,
        wavelength_um: Union[float, NDArray[np.floating]],
        path: PathGeometry,
        conditions: AtmosphereConditions,
    ) -> AtmosphereResult:
        """Return constant atmospheric values."""
        wavelength_um = np.atleast_1d(wavelength_um)
        n = len(wavelength_um)

        if n == 1:
            return AtmosphereResult(
                transmission=self._transmission,
                path_radiance=self._path_radiance,
            )

        return AtmosphereResult(
            transmission=np.full(n, self._transmission),
            path_radiance=np.full(n, self._path_radiance),
            sky_radiance=np.zeros(n),
        )


@dataclass
class AtmosphericWindowTransmission:
    """Typical transmission values for atmospheric windows."""

    # Standard clear atmosphere, sea level to 1 km, 23 km visibility
    VISIBLE: float = 0.85
    NIR: float = 0.88
    SWIR_1: float = 0.82  # 1.0-1.35 μm
    SWIR_2: float = 0.78  # 1.5-1.8 μm
    MWIR: float = 0.90  # 3-5 μm
    LWIR: float = 0.88  # 8-14 μm


def estimate_transmission(
    wavelength_um: float,
    range_km: float,
    visibility_km: float = 23.0,
) -> float:
    """Quick transmission estimate for given wavelength and range.

    Args:
        wavelength_um: Wavelength in micrometers
        range_km: Path length in kilometers
        visibility_km: Visibility in kilometers

    Returns:
        Estimated transmission (0-1)
    """
    model = BeerLambertAtmosphere()
    path = PathGeometry(ground_range_m=range_km * 1000)
    conditions = AtmosphereConditions(visibility_km=visibility_km)
    result = model.compute(wavelength_um, path, conditions)
    return float(result.transmission)


def estimate_band_transmission(
    band: SpectralBand,
    range_km: float,
    visibility_km: float = 23.0,
    n_samples: int = 20,
) -> float:
    """Quick band-averaged transmission estimate.

    Args:
        band: Spectral band
        range_km: Path length in kilometers
        visibility_km: Visibility in kilometers
        n_samples: Number of spectral samples

    Returns:
        Band-averaged transmission (0-1)
    """
    model = BeerLambertAtmosphere()
    path = PathGeometry(ground_range_m=range_km * 1000)
    conditions = AtmosphereConditions(visibility_km=visibility_km)
    result = model.compute_band(band, path, conditions, n_samples)
    return result.transmission


def koschmieder_visibility_to_extinction(visibility_km: float) -> float:
    """Convert visibility to extinction coefficient using Koschmieder relation.

    β = 3.912 / V

    where β is extinction coefficient [1/km] and V is visibility [km].

    Args:
        visibility_km: Meteorological visibility [km]

    Returns:
        Extinction coefficient [1/km] at ~550 nm
    """
    return 3.912 / visibility_km


def extinction_to_visibility(beta_per_km: float) -> float:
    """Convert extinction coefficient to visibility.

    Args:
        beta_per_km: Extinction coefficient [1/km]

    Returns:
        Visibility [km]
    """
    return 3.912 / beta_per_km
