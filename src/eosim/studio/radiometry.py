"""
Radiometric modeling for thermal infrared simulation.

This module provides physically-based radiometric calculations including:
- Planck blackbody radiation
- Spectral radiance integration
- Atmospheric transmission modeling
- Temperature-to-radiance conversion for various spectral bands
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

# Physical constants
PLANCK_H = 6.62607015e-34  # Planck constant (J·s)
SPEED_OF_LIGHT = 2.99792458e8  # Speed of light (m/s)
BOLTZMANN_K = 1.380649e-23  # Boltzmann constant (J/K)
STEFAN_BOLTZMANN = 5.670374419e-8  # Stefan-Boltzmann constant (W/m²/K⁴)

# First and second radiation constants
C1 = 2 * math.pi * PLANCK_H * SPEED_OF_LIGHT**2  # 3.7418e-16 W·m²
C2 = PLANCK_H * SPEED_OF_LIGHT / BOLTZMANN_K  # 0.014388 m·K


class SpectralBand(Enum):
    """Standard infrared spectral bands."""
    VNIR = "vnir"      # Visible/Near-IR: 0.4-1.0 µm
    SWIR = "swir"      # Short-wave IR: 1.0-2.5 µm
    MWIR = "mwir"      # Mid-wave IR: 3.0-5.0 µm
    LWIR = "lwir"      # Long-wave IR: 8.0-14.0 µm
    VLWIR = "vlwir"    # Very long-wave IR: 14.0-30.0 µm


# Band wavelength ranges in micrometers
BAND_WAVELENGTHS: Dict[SpectralBand, Tuple[float, float]] = {
    SpectralBand.VNIR: (0.4, 1.0),
    SpectralBand.SWIR: (1.0, 2.5),
    SpectralBand.MWIR: (3.0, 5.0),
    SpectralBand.LWIR: (8.0, 14.0),
    SpectralBand.VLWIR: (14.0, 30.0),
}


@dataclass
class AtmosphericConditions:
    """Atmospheric conditions affecting radiometric transmission."""
    temperature_k: float = 288.15  # Ambient temperature (15°C)
    pressure_hpa: float = 1013.25  # Sea level pressure
    relative_humidity: float = 0.5  # 0-1
    visibility_km: float = 23.0  # Meteorological visibility
    altitude_m: float = 0.0  # Observer altitude
    path_length_m: float = 1000.0  # Slant path length to target

    # Aerosol model
    aerosol_type: str = "rural"  # rural, urban, maritime, desert

    # Molecular absorbers (column amounts)
    water_vapor_g_cm2: float = 1.0  # Precipitable water
    ozone_atm_cm: float = 0.3  # Ozone column
    co2_ppm: float = 420.0  # CO2 concentration


@dataclass
class SpectralResponse:
    """Detector spectral response function."""
    wavelengths_um: np.ndarray  # Wavelength sample points
    response: np.ndarray  # Relative response (0-1)
    quantum_efficiency: float = 0.7  # Peak QE

    def __post_init__(self):
        if isinstance(self.wavelengths_um, list):
            self.wavelengths_um = np.array(self.wavelengths_um)
        if isinstance(self.response, list):
            self.response = np.array(self.response)


# Standard detector spectral responses
def create_mwir_response() -> SpectralResponse:
    """Create typical InSb MWIR detector response."""
    wavelengths = np.linspace(3.0, 5.5, 100)
    # InSb cutoff around 5.5 µm at 77K
    response = np.ones_like(wavelengths)
    response[wavelengths < 3.0] = 0
    response[wavelengths > 5.3] = np.exp(-((wavelengths[wavelengths > 5.3] - 5.3) / 0.1)**2)
    return SpectralResponse(wavelengths, response, quantum_efficiency=0.8)


def create_lwir_response() -> SpectralResponse:
    """Create typical MCT LWIR detector response."""
    wavelengths = np.linspace(7.0, 14.5, 100)
    # MCT cutoff adjustable, typically 10-14 µm
    response = np.ones_like(wavelengths)
    response[wavelengths < 7.5] = np.exp(-((7.5 - wavelengths[wavelengths < 7.5]) / 0.3)**2)
    response[wavelengths > 13.5] = np.exp(-((wavelengths[wavelengths > 13.5] - 13.5) / 0.5)**2)
    return SpectralResponse(wavelengths, response, quantum_efficiency=0.65)


DETECTOR_RESPONSES: Dict[SpectralBand, SpectralResponse] = {
    SpectralBand.MWIR: create_mwir_response(),
    SpectralBand.LWIR: create_lwir_response(),
}


def planck_radiance(wavelength_um: float, temperature_k: float) -> float:
    """
    Calculate spectral radiance using Planck's law.

    Args:
        wavelength_um: Wavelength in micrometers
        temperature_k: Temperature in Kelvin

    Returns:
        Spectral radiance in W/(m²·sr·µm)
    """
    if temperature_k <= 0 or wavelength_um <= 0:
        return 0.0

    # Convert wavelength to meters
    wavelength_m = wavelength_um * 1e-6

    # Planck function
    # L = (2hc²/λ⁵) × 1/(exp(hc/λkT) - 1)
    try:
        exponent = C2 / (wavelength_m * temperature_k)
        if exponent > 700:  # Prevent overflow
            return 0.0
        denominator = math.exp(exponent) - 1
        if denominator <= 0:
            return 0.0

        radiance = (C1 / wavelength_m**5) / denominator
        # Convert from W/(m²·sr·m) to W/(m²·sr·µm)
        radiance *= 1e-6
        return radiance
    except (OverflowError, ZeroDivisionError):
        return 0.0


def planck_radiance_array(wavelengths_um: np.ndarray, temperature_k: float) -> np.ndarray:
    """
    Vectorized Planck radiance calculation.

    Args:
        wavelengths_um: Array of wavelengths in micrometers
        temperature_k: Temperature in Kelvin

    Returns:
        Array of spectral radiances in W/(m²·sr·µm)
    """
    if temperature_k <= 0:
        return np.zeros_like(wavelengths_um)

    wavelengths_m = wavelengths_um * 1e-6

    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        exponent = C2 / (wavelengths_m * temperature_k)
        # Clip to prevent overflow
        exponent = np.clip(exponent, 0, 700)
        denominator = np.exp(exponent) - 1
        denominator = np.where(denominator <= 0, np.inf, denominator)

        radiance = (C1 / wavelengths_m**5) / denominator
        radiance *= 1e-6  # Convert to per µm
        radiance = np.where(np.isfinite(radiance), radiance, 0)

    return radiance


def integrate_band_radiance(
    temperature_k: float,
    band: SpectralBand,
    spectral_response: Optional[SpectralResponse] = None,
    n_samples: int = 100
) -> float:
    """
    Integrate spectral radiance over a wavelength band.

    Args:
        temperature_k: Target temperature in Kelvin
        band: Spectral band to integrate
        spectral_response: Optional detector response function
        n_samples: Number of integration samples

    Returns:
        In-band radiance in W/(m²·sr)
    """
    lambda_min, lambda_max = BAND_WAVELENGTHS[band]
    wavelengths = np.linspace(lambda_min, lambda_max, n_samples)

    # Get spectral radiance
    radiance = planck_radiance_array(wavelengths, temperature_k)

    # Apply detector response if provided
    if spectral_response is not None:
        response = np.interp(wavelengths, spectral_response.wavelengths_um,
                            spectral_response.response, left=0, right=0)
        radiance *= response * spectral_response.quantum_efficiency

    # Integrate using trapezoidal rule
    d_lambda = (lambda_max - lambda_min) / (n_samples - 1)
    # Use trapezoid for NumPy 2.0+, fall back to trapz for older versions
    try:
        integrated = np.trapezoid(radiance, dx=d_lambda)
    except AttributeError:
        integrated = np.trapz(radiance, dx=d_lambda)

    return integrated


def wien_peak_wavelength(temperature_k: float) -> float:
    """
    Calculate peak emission wavelength using Wien's displacement law.

    Args:
        temperature_k: Temperature in Kelvin

    Returns:
        Peak wavelength in micrometers
    """
    if temperature_k <= 0:
        return float('inf')
    # Wien's constant: 2897.8 µm·K
    return 2897.8 / temperature_k


def stefan_boltzmann_radiance(temperature_k: float) -> float:
    """
    Calculate total hemispherical radiance using Stefan-Boltzmann law.

    Args:
        temperature_k: Temperature in Kelvin

    Returns:
        Total radiance in W/(m²·sr) (Lambertian assumption: M/π)
    """
    if temperature_k <= 0:
        return 0.0
    exitance = STEFAN_BOLTZMANN * temperature_k**4  # W/m²
    return exitance / math.pi  # W/(m²·sr) for Lambertian surface


def apparent_temperature(
    radiance: float,
    band: SpectralBand,
    emissivity: float = 1.0
) -> float:
    """
    Convert measured radiance to apparent (brightness) temperature.

    Uses Newton-Raphson iteration to invert the Planck function.

    Args:
        radiance: Measured in-band radiance in W/(m²·sr)
        band: Spectral band
        emissivity: Target emissivity (0-1)

    Returns:
        Apparent temperature in Kelvin
    """
    if radiance <= 0 or emissivity <= 0:
        return 0.0

    # Correct for emissivity
    radiance_corrected = radiance / emissivity

    # Initial guess using Wien approximation
    lambda_min, lambda_max = BAND_WAVELENGTHS[band]
    lambda_center = (lambda_min + lambda_max) / 2

    # Wien approximation: T ≈ C2 / (λ × ln(C1/(λ⁵×L)))
    wavelength_m = lambda_center * 1e-6
    radiance_per_m = radiance_corrected * 1e6  # Convert to per meter

    try:
        arg = C1 / (wavelength_m**5 * radiance_per_m * math.pi)
        if arg <= 1:
            return 1000.0  # Very hot, return high temperature
        T_guess = C2 / (wavelength_m * math.log(arg))
    except (ValueError, ZeroDivisionError):
        T_guess = 300.0  # Default guess

    # Newton-Raphson refinement
    T = max(T_guess, 100.0)
    for _ in range(20):
        L_calc = integrate_band_radiance(T, band)
        if L_calc <= 0:
            break

        # Numerical derivative
        dT = 0.1
        L_plus = integrate_band_radiance(T + dT, band)
        dL_dT = (L_plus - L_calc) / dT

        if abs(dL_dT) < 1e-20:
            break

        error = L_calc - radiance_corrected
        T_new = T - error / dL_dT

        if abs(T_new - T) < 0.01:
            break
        T = max(T_new, 50.0)  # Prevent negative temperature

    return T


@dataclass
class AtmosphericTransmission:
    """Atmospheric transmission model results."""
    total_transmission: float  # 0-1
    path_radiance: float  # W/(m²·sr) atmospheric emission
    molecular_transmission: float
    aerosol_transmission: float

    # Individual absorber transmissions
    h2o_transmission: float = 1.0
    co2_transmission: float = 1.0
    o3_transmission: float = 1.0


def calculate_atmospheric_transmission(
    conditions: AtmosphericConditions,
    band: SpectralBand
) -> AtmosphericTransmission:
    """
    Calculate atmospheric transmission using simplified MODTRAN-like model.

    This is a simplified model based on empirical fits to MODTRAN/LOWTRAN
    calculations. For accurate results, use actual MODTRAN.

    Args:
        conditions: Atmospheric conditions
        band: Spectral band

    Returns:
        AtmosphericTransmission with transmission and path radiance
    """
    path_km = conditions.path_length_m / 1000.0

    # Molecular absorption coefficients (km⁻¹) - simplified
    if band == SpectralBand.MWIR:
        # MWIR: CO2 at 4.3 µm, H2O throughout
        h2o_coeff = 0.1 * conditions.water_vapor_g_cm2
        co2_coeff = 0.05 * (conditions.co2_ppm / 400.0)
        o3_coeff = 0.0
    elif band == SpectralBand.LWIR:
        # LWIR: H2O at 6.3 µm edge, O3 at 9.6 µm
        h2o_coeff = 0.05 * conditions.water_vapor_g_cm2
        co2_coeff = 0.02 * (conditions.co2_ppm / 400.0)
        o3_coeff = 0.03 * (conditions.ozone_atm_cm / 0.3)
    else:
        h2o_coeff = 0.02 * conditions.water_vapor_g_cm2
        co2_coeff = 0.01
        o3_coeff = 0.01

    # Calculate molecular transmissions
    h2o_trans = math.exp(-h2o_coeff * path_km)
    co2_trans = math.exp(-co2_coeff * path_km)
    o3_trans = math.exp(-o3_coeff * path_km)
    molecular_trans = h2o_trans * co2_trans * o3_trans

    # Aerosol scattering/absorption
    # Visibility-based extinction coefficient
    # β ≈ 3.91 / V (km⁻¹) at 550 nm, scale for IR
    if band == SpectralBand.MWIR:
        wavelength_scale = 0.3  # Less scattering at longer wavelengths
    elif band == SpectralBand.LWIR:
        wavelength_scale = 0.15
    else:
        wavelength_scale = 1.0

    if conditions.visibility_km > 0:
        beta_vis = 3.91 / conditions.visibility_km
        beta_ir = beta_vis * wavelength_scale
        aerosol_trans = math.exp(-beta_ir * path_km)
    else:
        aerosol_trans = 0.0

    # Total transmission
    total_trans = molecular_trans * aerosol_trans

    # Path radiance (atmospheric thermal emission)
    # Simplified: assume atmosphere at ambient temperature
    atm_radiance = integrate_band_radiance(conditions.temperature_k, band)
    # Path radiance = L_atm × (1 - τ) for isothermal atmosphere
    path_rad = atm_radiance * (1 - total_trans)

    return AtmosphericTransmission(
        total_transmission=total_trans,
        path_radiance=path_rad,
        molecular_transmission=molecular_trans,
        aerosol_transmission=aerosol_trans,
        h2o_transmission=h2o_trans,
        co2_transmission=co2_trans,
        o3_transmission=o3_trans
    )


def target_contrast(
    target_temp_k: float,
    background_temp_k: float,
    band: SpectralBand,
    target_emissivity: float = 0.9,
    background_emissivity: float = 0.95,
    atmosphere: Optional[AtmosphericConditions] = None
) -> float:
    """
    Calculate apparent thermal contrast between target and background.

    Args:
        target_temp_k: Target temperature in Kelvin
        background_temp_k: Background temperature in Kelvin
        band: Spectral band
        target_emissivity: Target emissivity
        background_emissivity: Background emissivity
        atmosphere: Optional atmospheric conditions

    Returns:
        Thermal contrast (ΔL/L_background)
    """
    # Calculate radiances
    L_target = integrate_band_radiance(target_temp_k, band) * target_emissivity
    L_background = integrate_band_radiance(background_temp_k, band) * background_emissivity

    # Apply atmospheric effects if provided
    if atmosphere is not None:
        atm = calculate_atmospheric_transmission(atmosphere, band)
        L_target = L_target * atm.total_transmission + atm.path_radiance
        L_background = L_background * atm.total_transmission + atm.path_radiance

    # Calculate contrast
    if L_background > 0:
        return (L_target - L_background) / L_background
    return 0.0


@dataclass
class RadiometricImage:
    """
    Radiometrically calibrated image data.

    Stores both the raw digital counts and calibrated radiance values.
    """
    radiance: np.ndarray  # W/(m²·sr) per pixel
    temperature: np.ndarray  # Apparent temperature (K) per pixel
    band: SpectralBand
    integration_time_s: float
    timestamp: float = 0.0

    # Calibration metadata
    gain: float = 1.0  # DN per W/(m²·sr)
    offset: float = 0.0  # DN offset

    def to_digital_counts(self, bits: int = 14) -> np.ndarray:
        """Convert radiance to digital counts."""
        max_dn = 2**bits - 1
        dn = self.radiance * self.gain + self.offset
        return np.clip(dn, 0, max_dn).astype(np.uint16)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.radiance.shape


class Radiometer:
    """
    Radiometric calculator for thermal imaging simulation.

    Handles conversion between physical temperatures and sensor signals.
    """

    def __init__(
        self,
        band: SpectralBand = SpectralBand.LWIR,
        spectral_response: Optional[SpectralResponse] = None
    ):
        self.band = band
        self.spectral_response = spectral_response or DETECTOR_RESPONSES.get(band)

        # Build lookup table for fast temperature-radiance conversion
        self._build_lut()

    def _build_lut(self, t_min: float = 200.0, t_max: float = 500.0, n_points: int = 1000):
        """Build lookup table for temperature to radiance conversion."""
        self._lut_temps = np.linspace(t_min, t_max, n_points)
        self._lut_radiances = np.array([
            integrate_band_radiance(t, self.band, self.spectral_response)
            for t in self._lut_temps
        ])

    def temperature_to_radiance(self, temperature_k: np.ndarray) -> np.ndarray:
        """Convert temperature map to radiance using LUT interpolation."""
        return np.interp(temperature_k, self._lut_temps, self._lut_radiances)

    def radiance_to_temperature(self, radiance: np.ndarray) -> np.ndarray:
        """Convert radiance map to apparent temperature using LUT interpolation."""
        return np.interp(radiance, self._lut_radiances, self._lut_temps)

    def apply_atmospheric_effects(
        self,
        radiance: np.ndarray,
        conditions: AtmosphericConditions
    ) -> np.ndarray:
        """Apply atmospheric transmission and path radiance to image."""
        atm = calculate_atmospheric_transmission(conditions, self.band)
        return radiance * atm.total_transmission + atm.path_radiance

    def calculate_netd(
        self,
        target_temp_k: float,
        noise_equivalent_power: float,
        aperture_m: float,
        focal_length_m: float,
        pixel_pitch_m: float,
        integration_time_s: float
    ) -> float:
        """
        Calculate Noise Equivalent Temperature Difference (NETD).

        Args:
            target_temp_k: Target temperature
            noise_equivalent_power: NEP in W/√Hz
            aperture_m: Lens aperture diameter
            focal_length_m: Focal length
            pixel_pitch_m: Detector pixel size
            integration_time_s: Integration time

        Returns:
            NETD in Kelvin
        """
        # Calculate dL/dT at target temperature
        dT = 0.1
        L1 = integrate_band_radiance(target_temp_k - dT/2, self.band, self.spectral_response)
        L2 = integrate_band_radiance(target_temp_k + dT/2, self.band, self.spectral_response)
        dL_dT = (L2 - L1) / dT

        if dL_dT <= 0:
            return float('inf')

        # F-number
        f_number = focal_length_m / aperture_m

        # Solid angle subtended by pixel
        pixel_solid_angle = (pixel_pitch_m / focal_length_m)**2

        # Noise equivalent radiance
        # NEL = NEP / (A_pixel × Ω × τ_int)
        a_pixel = pixel_pitch_m**2
        nel = noise_equivalent_power / (a_pixel * pixel_solid_angle * math.sqrt(integration_time_s))

        # NETD = NEL / (dL/dT)
        netd = nel / dL_dT

        return netd


# Convenience functions for common operations
def celsius_to_kelvin(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def kelvin_to_celsius(kelvin: float) -> float:
    """Convert Kelvin to Celsius."""
    return kelvin - 273.15


def fahrenheit_to_kelvin(fahrenheit: float) -> float:
    """Convert Fahrenheit to Kelvin."""
    return (fahrenheit - 32) * 5/9 + 273.15


def create_temperature_map(
    shape: Tuple[int, int],
    base_temp_k: float = 300.0,
    hotspots: Optional[List[Tuple[int, int, float, float]]] = None
) -> np.ndarray:
    """
    Create a synthetic temperature map for testing.

    Args:
        shape: (height, width) of output
        base_temp_k: Background temperature
        hotspots: List of (x, y, radius, delta_temp) tuples

    Returns:
        Temperature map in Kelvin
    """
    temp_map = np.full(shape, base_temp_k, dtype=np.float32)

    if hotspots:
        y_coords, x_coords = np.ogrid[:shape[0], :shape[1]]
        for x, y, radius, delta_t in hotspots:
            distance = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
            mask = distance < radius
            # Gaussian falloff
            temp_map[mask] += delta_t * np.exp(-(distance[mask] / (radius/2))**2)

    return temp_map
