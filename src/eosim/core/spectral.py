"""
Spectral wavelength and band handling for EOSIM.

Provides classes for representing spectral bands, wavelength grids,
and spectral quantities used throughout the simulation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from eosim.core.constants import BAND_LIMITS, CONSTANTS


class SpectralRegion(Enum):
    """Standard spectral regions for EO/IR sensors."""

    VIS = "visible"
    NIR = "near_infrared"
    SWIR = "short_wave_infrared"
    MWIR = "mid_wave_infrared"
    LWIR = "long_wave_infrared"
    VLWIR = "very_long_wave_infrared"

    @classmethod
    def from_wavelength(cls, wavelength_um: float) -> "SpectralRegion":
        """Determine spectral region from wavelength in micrometers."""
        if wavelength_um < BAND_LIMITS.VIS_MIN:
            raise ValueError(f"Wavelength {wavelength_um} μm is below VIS range")
        elif wavelength_um <= BAND_LIMITS.VIS_MAX:
            return cls.VIS
        elif wavelength_um <= BAND_LIMITS.NIR_MAX:
            return cls.NIR
        elif wavelength_um <= BAND_LIMITS.SWIR_MAX:
            return cls.SWIR
        elif wavelength_um <= BAND_LIMITS.MWIR_MAX:
            return cls.MWIR
        elif wavelength_um <= BAND_LIMITS.LWIR_MAX:
            return cls.LWIR
        elif wavelength_um <= BAND_LIMITS.VLWIR_MAX:
            return cls.VLWIR
        else:
            raise ValueError(f"Wavelength {wavelength_um} μm is above VLWIR range")


@dataclass
class SpectralBand:
    """Represents a spectral band with wavelength limits.

    Attributes:
        name: Identifier for the band
        lambda_min_um: Minimum wavelength [micrometers]
        lambda_max_um: Maximum wavelength [micrometers]
        lambda_center_um: Center wavelength (computed if not provided)
    """

    name: str
    lambda_min_um: float
    lambda_max_um: float
    lambda_center_um: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate and compute derived values."""
        if self.lambda_min_um >= self.lambda_max_um:
            raise ValueError(
                f"lambda_min ({self.lambda_min_um}) must be less than "
                f"lambda_max ({self.lambda_max_um})"
            )
        if self.lambda_min_um <= 0:
            raise ValueError("Wavelengths must be positive")

        if self.lambda_center_um is None:
            # Use geometric mean for center wavelength (better for broad bands)
            self.lambda_center_um = np.sqrt(self.lambda_min_um * self.lambda_max_um)

    @property
    def bandwidth_um(self) -> float:
        """Bandwidth in micrometers."""
        return self.lambda_max_um - self.lambda_min_um

    @property
    def region(self) -> SpectralRegion:
        """Determine the spectral region for this band."""
        return SpectralRegion.from_wavelength(self.lambda_center_um)

    @property
    def lambda_min_m(self) -> float:
        """Minimum wavelength in meters."""
        return self.lambda_min_um * 1e-6

    @property
    def lambda_max_m(self) -> float:
        """Maximum wavelength in meters."""
        return self.lambda_max_um * 1e-6

    @property
    def lambda_center_m(self) -> float:
        """Center wavelength in meters."""
        return self.lambda_center_um * 1e-6

    def contains(self, wavelength_um: float) -> bool:
        """Check if wavelength falls within this band."""
        return self.lambda_min_um <= wavelength_um <= self.lambda_max_um

    @classmethod
    def from_region(cls, region: SpectralRegion) -> "SpectralBand":
        """Create a standard band for the given spectral region."""
        limits = {
            SpectralRegion.VIS: (BAND_LIMITS.VIS_MIN, BAND_LIMITS.VIS_MAX),
            SpectralRegion.NIR: (BAND_LIMITS.NIR_MIN, BAND_LIMITS.NIR_MAX),
            SpectralRegion.SWIR: (BAND_LIMITS.SWIR_MIN, BAND_LIMITS.SWIR_MAX),
            SpectralRegion.MWIR: (BAND_LIMITS.MWIR_MIN, BAND_LIMITS.MWIR_MAX),
            SpectralRegion.LWIR: (BAND_LIMITS.LWIR_MIN, BAND_LIMITS.LWIR_MAX),
            SpectralRegion.VLWIR: (BAND_LIMITS.VLWIR_MIN, BAND_LIMITS.VLWIR_MAX),
        }
        lmin, lmax = limits[region]
        return cls(name=region.value, lambda_min_um=lmin, lambda_max_um=lmax)


# Standard predefined bands
BAND_VIS = SpectralBand.from_region(SpectralRegion.VIS)
BAND_NIR = SpectralBand.from_region(SpectralRegion.NIR)
BAND_SWIR = SpectralBand.from_region(SpectralRegion.SWIR)
BAND_MWIR = SpectralBand.from_region(SpectralRegion.MWIR)
BAND_LWIR = SpectralBand.from_region(SpectralRegion.LWIR)


@dataclass
class WavelengthGrid:
    """Represents a discrete wavelength grid for spectral calculations.

    Attributes:
        wavelengths_um: Array of wavelengths in micrometers
        delta_lambda_um: Wavelength spacing (for uniform grids)
    """

    wavelengths_um: NDArray[np.floating]
    delta_lambda_um: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate and compute derived values."""
        self.wavelengths_um = np.asarray(self.wavelengths_um, dtype=np.float64)

        if len(self.wavelengths_um) < 1:
            raise ValueError("Wavelength grid must have at least one point")

        if not np.all(np.diff(self.wavelengths_um) > 0):
            raise ValueError("Wavelengths must be strictly increasing")

        if len(self.wavelengths_um) > 1 and self.delta_lambda_um is None:
            # Check if uniform spacing
            diffs = np.diff(self.wavelengths_um)
            if np.allclose(diffs, diffs[0], rtol=1e-6):
                self.delta_lambda_um = float(diffs[0])

    @property
    def n_wavelengths(self) -> int:
        """Number of wavelength points."""
        return len(self.wavelengths_um)

    @property
    def wavelengths_m(self) -> NDArray[np.floating]:
        """Wavelengths in meters."""
        return self.wavelengths_um * 1e-6

    @property
    def lambda_min_um(self) -> float:
        """Minimum wavelength [μm]."""
        return float(self.wavelengths_um[0])

    @property
    def lambda_max_um(self) -> float:
        """Maximum wavelength [μm]."""
        return float(self.wavelengths_um[-1])

    @property
    def is_uniform(self) -> bool:
        """Check if grid has uniform spacing."""
        return self.delta_lambda_um is not None

    @classmethod
    def uniform(
        cls,
        lambda_min_um: float,
        lambda_max_um: float,
        n_points: int,
    ) -> "WavelengthGrid":
        """Create a uniformly spaced wavelength grid."""
        wavelengths = np.linspace(lambda_min_um, lambda_max_um, n_points)
        return cls(wavelengths_um=wavelengths)

    @classmethod
    def from_band(cls, band: SpectralBand, n_points: int = 100) -> "WavelengthGrid":
        """Create a wavelength grid spanning a spectral band."""
        return cls.uniform(band.lambda_min_um, band.lambda_max_um, n_points)

    @classmethod
    def logarithmic(
        cls,
        lambda_min_um: float,
        lambda_max_um: float,
        n_points: int,
    ) -> "WavelengthGrid":
        """Create a logarithmically spaced wavelength grid.

        Useful for broad spectral ranges where equal resolution is desired
        in log-wavelength space.
        """
        wavelengths = np.logspace(
            np.log10(lambda_min_um), np.log10(lambda_max_um), n_points
        )
        return cls(wavelengths_um=wavelengths)


@dataclass
class SpectralQuantity:
    """A quantity that varies with wavelength.

    Attributes:
        wavelengths_um: Wavelength grid [μm]
        values: Spectral values at each wavelength
        units: String describing the units of the values
    """

    wavelengths_um: NDArray[np.floating]
    values: NDArray[np.floating]
    units: str = ""

    def __post_init__(self) -> None:
        """Validate arrays have same shape."""
        self.wavelengths_um = np.asarray(self.wavelengths_um, dtype=np.float64)
        self.values = np.asarray(self.values, dtype=np.float64)

        if self.wavelengths_um.shape != self.values.shape:
            raise ValueError(
                f"Wavelength shape {self.wavelengths_um.shape} doesn't match "
                f"values shape {self.values.shape}"
            )

    def interpolate(self, wavelengths_um: NDArray[np.floating]) -> NDArray[np.floating]:
        """Interpolate values to new wavelength grid."""
        return np.interp(wavelengths_um, self.wavelengths_um, self.values)

    def integrate(
        self,
        lambda_min_um: Optional[float] = None,
        lambda_max_um: Optional[float] = None,
    ) -> float:
        """Integrate over wavelength range using trapezoidal rule.

        Args:
            lambda_min_um: Lower bound (default: first wavelength)
            lambda_max_um: Upper bound (default: last wavelength)

        Returns:
            Integrated value [units × μm]
        """
        if lambda_min_um is None:
            lambda_min_um = float(self.wavelengths_um[0])
        if lambda_max_um is None:
            lambda_max_um = float(self.wavelengths_um[-1])

        mask = (self.wavelengths_um >= lambda_min_um) & (
            self.wavelengths_um <= lambda_max_um
        )
        return float(np.trapz(self.values[mask], self.wavelengths_um[mask]))

    def band_average(self, band: SpectralBand) -> float:
        """Compute average value over a spectral band."""
        integral = self.integrate(band.lambda_min_um, band.lambda_max_um)
        return integral / band.bandwidth_um


def planck_radiance(
    wavelength_um: Union[float, NDArray[np.floating]],
    temperature_K: float,
) -> Union[float, NDArray[np.floating]]:
    """Compute spectral radiance of a blackbody using Planck's Law.

    Args:
        wavelength_um: Wavelength(s) in micrometers
        temperature_K: Temperature in Kelvin

    Returns:
        Spectral radiance L(λ,T) in W/(m²·sr·μm)
    """
    wavelength_m = np.asarray(wavelength_um) * 1e-6

    # Planck's law: L(λ,T) = (2hc²/λ⁵) × 1/(exp(hc/λkT) - 1)
    c1L = CONSTANTS.c1L  # 2hc² [W·m²]
    c2 = CONSTANTS.c2  # hc/k [m·K]

    # Compute in W/(m²·sr·m) then convert to W/(m²·sr·μm)
    exponent = c2 / (wavelength_m * temperature_K)

    # Avoid overflow for large exponents
    with np.errstate(over="ignore"):
        exp_term = np.exp(exponent)

    # For very large exponents, use approximation
    if np.any(exponent > 700):
        exp_term = np.where(exponent > 700, np.inf, exp_term)

    radiance_per_m = c1L / (wavelength_m**5 * (exp_term - 1))

    # Convert from per meter to per micrometer
    radiance_per_um = radiance_per_m * 1e-6

    return radiance_per_um


def wien_displacement(temperature_K: float) -> float:
    """Compute peak wavelength using Wien's displacement law.

    Args:
        temperature_K: Temperature in Kelvin

    Returns:
        Peak wavelength in micrometers
    """
    # Wien's constant b = 2897.8 μm·K
    WIEN_CONSTANT = 2897.8
    return WIEN_CONSTANT / temperature_K


def stefan_boltzmann(temperature_K: float) -> float:
    """Compute total radiant exitance using Stefan-Boltzmann law.

    Args:
        temperature_K: Temperature in Kelvin

    Returns:
        Total radiant exitance M in W/m²
    """
    return CONSTANTS.sigma * temperature_K**4


def band_integrated_radiance(
    band: SpectralBand,
    temperature_K: float,
    n_points: int = 100,
) -> float:
    """Compute band-integrated radiance for a blackbody.

    Args:
        band: Spectral band for integration
        temperature_K: Temperature in Kelvin
        n_points: Number of wavelength points for numerical integration

    Returns:
        Band-integrated radiance in W/(m²·sr)
    """
    grid = WavelengthGrid.from_band(band, n_points)
    spectral_radiance = planck_radiance(grid.wavelengths_um, temperature_K)
    # Integrate using trapezoidal rule (result in W/(m²·sr))
    return float(np.trapz(spectral_radiance, grid.wavelengths_um))
