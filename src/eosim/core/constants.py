"""
Physical constants for electro-optical and infrared simulation.

All constants use SI units unless otherwise noted. Values are from
CODATA 2018 recommended values where applicable.
"""

from dataclasses import dataclass
from typing import Final
import numpy as np


@dataclass(frozen=True)
class PhysicalConstants:
    """Fundamental physical constants for radiometric calculations.

    All values in SI units (meters, kilograms, seconds, Kelvin, Watts, etc.)

    Attributes:
        c: Speed of light in vacuum [m/s]
        h: Planck constant [J·s]
        k_B: Boltzmann constant [J/K]
        sigma: Stefan-Boltzmann constant [W/(m²·K⁴)]
        c1: First radiation constant 2πhc² [W·m²/sr]
        c2: Second radiation constant hc/k_B [m·K]
        c1L: First radiation constant for spectral radiance [W·m²/(sr·m)]
        sun_temperature: Effective blackbody temperature of sun [K]
        earth_sun_distance: Mean Earth-Sun distance [m]
        solar_constant: Solar irradiance at Earth [W/m²]
    """

    # Fundamental constants (CODATA 2018)
    c: float = 299792458.0  # Speed of light [m/s]
    h: float = 6.62607015e-34  # Planck constant [J·s]
    k_B: float = 1.380649e-23  # Boltzmann constant [J/K]

    # Derived constants
    @property
    def sigma(self) -> float:
        """Stefan-Boltzmann constant [W/(m²·K⁴)]."""
        return 2 * np.pi**5 * self.k_B**4 / (15 * self.h**3 * self.c**2)

    @property
    def c1(self) -> float:
        """First radiation constant 2πhc² [W·m²/sr]."""
        return 2 * np.pi * self.h * self.c**2

    @property
    def c2(self) -> float:
        """Second radiation constant hc/k_B [m·K]."""
        return self.h * self.c / self.k_B

    @property
    def c1L(self) -> float:
        """First radiation constant for spectral radiance 2hc² [W·m²/(sr·m)]."""
        return 2 * self.h * self.c**2

    # Solar/astronomical constants
    sun_temperature: float = 5778.0  # Effective temperature [K]
    earth_sun_distance: float = 1.496e11  # Mean distance [m] (1 AU)
    solar_constant: float = 1361.0  # Solar irradiance at Earth [W/m²]

    # Earth constants
    earth_radius: float = 6.371e6  # Mean Earth radius [m]

    def __post_init__(self) -> None:
        """Validate constants after initialization."""
        pass


# Singleton instance for convenience
CONSTANTS: Final[PhysicalConstants] = PhysicalConstants()


# Spectral band boundaries [micrometers]
@dataclass(frozen=True)
class SpectralBandLimits:
    """Standard spectral band wavelength limits in micrometers.

    Based on common EO/IR sensor definitions and atmospheric windows.
    """

    # Visible
    VIS_MIN: float = 0.38
    VIS_MAX: float = 0.70

    # Near-infrared
    NIR_MIN: float = 0.70
    NIR_MAX: float = 1.0

    # Short-wave infrared
    SWIR_MIN: float = 1.0
    SWIR_MAX: float = 2.5

    # Mid-wave infrared (atmospheric window)
    MWIR_MIN: float = 3.0
    MWIR_MAX: float = 5.0

    # Long-wave infrared (atmospheric window)
    LWIR_MIN: float = 8.0
    LWIR_MAX: float = 14.0

    # Very long-wave infrared
    VLWIR_MIN: float = 14.0
    VLWIR_MAX: float = 30.0


BAND_LIMITS: Final[SpectralBandLimits] = SpectralBandLimits()


# Common material emissivities (typical values at thermal IR wavelengths)
MATERIAL_EMISSIVITY: Final[dict[str, float]] = {
    # Natural surfaces
    "water": 0.96,
    "ice": 0.97,
    "snow_fresh": 0.99,
    "snow_aged": 0.82,
    "soil_dry": 0.92,
    "soil_wet": 0.95,
    "sand_dry": 0.90,
    "sand_wet": 0.95,
    "grass_green": 0.98,
    "grass_dry": 0.90,
    "vegetation": 0.97,
    "forest_conifer": 0.97,
    "forest_deciduous": 0.95,
    "rock_granite": 0.90,
    "rock_basalt": 0.93,
    "rock_limestone": 0.92,

    # Man-made surfaces
    "concrete": 0.92,
    "asphalt": 0.95,
    "brick_red": 0.93,
    "glass": 0.94,
    "paint_white": 0.90,
    "paint_black": 0.98,
    "aluminum_polished": 0.05,
    "aluminum_anodized": 0.77,
    "aluminum_oxidized": 0.25,
    "steel_polished": 0.07,
    "steel_oxidized": 0.80,
    "steel_rusted": 0.69,
    "copper_polished": 0.03,
    "copper_oxidized": 0.78,
    "rubber": 0.95,
    "plastic_black": 0.95,
    "plastic_white": 0.84,

    # Vehicle surfaces
    "vehicle_paint_dark": 0.92,
    "vehicle_paint_light": 0.90,
    "tank_armor": 0.85,
    "aircraft_skin": 0.20,
    "tire": 0.94,

    # Reference
    "blackbody": 1.00,
    "graybody_typical": 0.95,
}


# Typical atmospheric transmission windows (approximate transmittance)
# For quick reference; detailed modeling uses RAF-tran or similar
ATMOSPHERIC_WINDOWS: Final[dict[str, tuple[float, float, float]]] = {
    # name: (lambda_min_um, lambda_max_um, typical_transmittance)
    "visible": (0.38, 0.70, 0.80),
    "nir": (0.70, 1.0, 0.85),
    "swir_1": (1.0, 1.35, 0.80),
    "swir_2": (1.45, 1.8, 0.75),
    "swir_3": (2.0, 2.5, 0.70),
    "mwir": (3.0, 5.0, 0.85),
    "lwir": (8.0, 14.0, 0.80),
}
