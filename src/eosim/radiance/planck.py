"""
Planck blackbody radiation calculations for EOSIM.

Provides functions for computing spectral radiance, radiant exitance,
and band-integrated quantities for blackbody sources.
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray

from eosim.core.constants import CONSTANTS
from eosim.core.spectral import SpectralBand, WavelengthGrid


def spectral_radiance(
    wavelength_um: Union[float, NDArray[np.floating]],
    temperature_K: Union[float, NDArray[np.floating]],
) -> Union[float, NDArray[np.floating]]:
    """Compute spectral radiance of a blackbody using Planck's Law.

    L(λ,T) = (2hc²/λ⁵) × 1/(exp(hc/λkT) - 1)

    Args:
        wavelength_um: Wavelength(s) in micrometers
        temperature_K: Temperature(s) in Kelvin. Can be scalar or array
            matching wavelength shape for spatially-varying temperature.

    Returns:
        Spectral radiance L(λ,T) in W/(m²·sr·μm)
    """
    wavelength_m = np.asarray(wavelength_um, dtype=np.float64) * 1e-6
    temperature = np.asarray(temperature_K, dtype=np.float64)

    c1L = CONSTANTS.c1L  # 2hc² [W·m²]
    c2 = CONSTANTS.c2    # hc/k [m·K]

    # Handle broadcasting for wavelength × temperature grids
    if wavelength_m.ndim == 1 and temperature.ndim > 0 and temperature.ndim != wavelength_m.ndim:
        wavelength_m = wavelength_m[:, np.newaxis]

    exponent = c2 / (wavelength_m * temperature)

    # Avoid overflow for large exponents (low T, short λ)
    with np.errstate(over="ignore", divide="ignore"):
        exp_term = np.exp(np.minimum(exponent, 700))
        radiance_per_m = c1L / (wavelength_m**5 * (exp_term - 1))

    # Handle extreme cases
    radiance_per_m = np.where(exponent > 700, 0.0, radiance_per_m)
    radiance_per_m = np.where(np.isnan(radiance_per_m), 0.0, radiance_per_m)

    # Convert from per meter to per micrometer
    return radiance_per_m * 1e-6


def spectral_exitance(
    wavelength_um: Union[float, NDArray[np.floating]],
    temperature_K: Union[float, NDArray[np.floating]],
) -> Union[float, NDArray[np.floating]]:
    """Compute spectral radiant exitance (hemispherical emission).

    M(λ,T) = π × L(λ,T)

    Args:
        wavelength_um: Wavelength(s) in micrometers
        temperature_K: Temperature(s) in Kelvin

    Returns:
        Spectral exitance M(λ,T) in W/(m²·μm)
    """
    return np.pi * spectral_radiance(wavelength_um, temperature_K)


def total_radiance(temperature_K: float) -> float:
    """Compute total (wavelength-integrated) blackbody radiance.

    L_total = σT⁴/π

    Args:
        temperature_K: Temperature in Kelvin

    Returns:
        Total radiance in W/(m²·sr)
    """
    return CONSTANTS.sigma * temperature_K**4 / np.pi


def total_exitance(temperature_K: float) -> float:
    """Compute total radiant exitance using Stefan-Boltzmann law.

    M = σT⁴

    Args:
        temperature_K: Temperature in Kelvin

    Returns:
        Total radiant exitance in W/m²
    """
    return CONSTANTS.sigma * temperature_K**4


def band_radiance(
    band: SpectralBand,
    temperature_K: Union[float, NDArray[np.floating]],
    n_samples: int = 100,
) -> Union[float, NDArray[np.floating]]:
    """Compute band-integrated radiance for a blackbody.

    Args:
        band: Spectral band for integration
        temperature_K: Temperature(s) in Kelvin
        n_samples: Number of wavelength samples for integration

    Returns:
        Band-integrated radiance in W/(m²·sr)
    """
    wavelengths = np.linspace(band.lambda_min_um, band.lambda_max_um, n_samples)
    spectral_L = spectral_radiance(wavelengths, temperature_K)

    # Trapezoidal integration
    if spectral_L.ndim == 1:
        return float(np.trapz(spectral_L, wavelengths))
    else:
        # Integrate along wavelength axis (axis 0)
        return np.trapz(spectral_L, wavelengths, axis=0)


def band_exitance(
    band: SpectralBand,
    temperature_K: Union[float, NDArray[np.floating]],
    n_samples: int = 100,
) -> Union[float, NDArray[np.floating]]:
    """Compute band-integrated radiant exitance.

    Args:
        band: Spectral band for integration
        temperature_K: Temperature(s) in Kelvin
        n_samples: Number of wavelength samples

    Returns:
        Band-integrated exitance in W/m²
    """
    return np.pi * band_radiance(band, temperature_K, n_samples)


def wien_peak_wavelength(temperature_K: float) -> float:
    """Compute peak emission wavelength using Wien's displacement law.

    λ_peak = b / T where b ≈ 2897.8 μm·K

    Args:
        temperature_K: Temperature in Kelvin

    Returns:
        Peak wavelength in micrometers
    """
    WIEN_CONSTANT = 2897.8  # μm·K
    return WIEN_CONSTANT / temperature_K


def temperature_from_peak(wavelength_um: float) -> float:
    """Compute temperature from peak emission wavelength.

    T = b / λ_peak

    Args:
        wavelength_um: Peak wavelength in micrometers

    Returns:
        Temperature in Kelvin
    """
    WIEN_CONSTANT = 2897.8  # μm·K
    return WIEN_CONSTANT / wavelength_um


def contrast_temperature(
    radiance: Union[float, NDArray[np.floating]],
    band: SpectralBand,
    reference_T: float = 300.0,
    n_samples: int = 50,
) -> Union[float, NDArray[np.floating]]:
    """Compute apparent/contrast temperature from measured radiance.

    Uses Newton-Raphson iteration to invert band-integrated Planck function.

    Args:
        radiance: Measured band radiance in W/(m²·sr)
        band: Spectral band
        reference_T: Initial temperature guess [K]
        n_samples: Samples for band integration

    Returns:
        Apparent temperature in Kelvin
    """
    radiance = np.asarray(radiance)
    T = np.full_like(radiance, reference_T, dtype=np.float64)

    # Newton-Raphson iteration
    for _ in range(20):
        L = band_radiance(band, T, n_samples)
        # Numerical derivative
        dT = 0.1
        L_plus = band_radiance(band, T + dT, n_samples)
        dL_dT = (L_plus - L) / dT

        # Update with damping
        delta = (radiance - L) / np.maximum(dL_dT, 1e-20)
        delta = np.clip(delta, -50, 50)  # Limit step size
        T = T + delta

        # Check convergence
        if np.all(np.abs(delta) < 0.01):
            break

    return T


def radiance_difference(
    temperature_K: Union[float, NDArray[np.floating]],
    background_T: float,
    band: SpectralBand,
    n_samples: int = 100,
) -> Union[float, NDArray[np.floating]]:
    """Compute radiance difference (contrast) against background.

    ΔL = L(T_target) - L(T_background)

    Args:
        temperature_K: Target temperature(s) in Kelvin
        background_T: Background temperature in Kelvin
        band: Spectral band
        n_samples: Samples for integration

    Returns:
        Radiance difference in W/(m²·sr)
    """
    L_target = band_radiance(band, temperature_K, n_samples)
    L_background = band_radiance(band, background_T, n_samples)
    return L_target - L_background


def spectral_radiance_ratio(
    wavelength1_um: float,
    wavelength2_um: float,
    temperature_K: float,
) -> float:
    """Compute ratio of spectral radiances at two wavelengths.

    Useful for two-color pyrometry.

    Args:
        wavelength1_um: First wavelength [μm]
        wavelength2_um: Second wavelength [μm]
        temperature_K: Temperature [K]

    Returns:
        Radiance ratio L(λ1)/L(λ2)
    """
    L1 = spectral_radiance(wavelength1_um, temperature_K)
    L2 = spectral_radiance(wavelength2_um, temperature_K)
    return L1 / L2
