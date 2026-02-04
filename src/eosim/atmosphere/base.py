"""
Atmosphere module base classes for EOSIM.

Defines the interface for atmospheric transmission and path radiance models.
All atmosphere implementations should inherit from AtmosphereModel.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from eosim.core.spectral import SpectralBand


class AtmosphereFidelity(Enum):
    """Fidelity level for atmosphere modeling."""

    NONE = "none"  # No atmospheric effects
    SIMPLE = "simple"  # Beer-Lambert extinction only
    STANDARD = "standard"  # LUT-based with path radiance
    HIGH = "high"  # Full radiative transfer (RAF-tran)


@dataclass
class PathGeometry:
    """Geometry of an atmospheric path.

    Attributes:
        ground_range_m: Horizontal distance [m]
        altitude_start_m: Starting altitude (target) [m]
        altitude_end_m: Ending altitude (sensor) [m]
        zenith_angle_deg: View zenith angle [degrees]
        azimuth_angle_deg: View azimuth angle [degrees]
    """

    ground_range_m: float
    altitude_start_m: float = 0.0
    altitude_end_m: float = 1000.0
    zenith_angle_deg: float = 0.0
    azimuth_angle_deg: float = 0.0

    @property
    def slant_range_m(self) -> float:
        """Compute slant path length."""
        dh = abs(self.altitude_end_m - self.altitude_start_m)
        return np.sqrt(self.ground_range_m ** 2 + dh ** 2)

    @property
    def zenith_angle_rad(self) -> float:
        """Zenith angle in radians."""
        return np.radians(self.zenith_angle_deg)

    @classmethod
    def from_slant_range(
        cls,
        slant_range_m: float,
        zenith_angle_deg: float,
        sensor_altitude_m: float = 1000.0,
    ) -> "PathGeometry":
        """Create path geometry from slant range and zenith angle."""
        zenith_rad = np.radians(zenith_angle_deg)
        ground_range = slant_range_m * np.sin(zenith_rad)
        dh = slant_range_m * np.cos(zenith_rad)

        return cls(
            ground_range_m=ground_range,
            altitude_start_m=sensor_altitude_m - dh,
            altitude_end_m=sensor_altitude_m,
            zenith_angle_deg=zenith_angle_deg,
        )

    @classmethod
    def vertical(cls, altitude_m: float) -> "PathGeometry":
        """Create vertical (nadir) path geometry."""
        return cls(
            ground_range_m=0.0,
            altitude_start_m=0.0,
            altitude_end_m=altitude_m,
            zenith_angle_deg=0.0,
        )


@dataclass
class AtmosphereConditions:
    """Atmospheric conditions for modeling.

    Attributes:
        visibility_km: Horizontal visibility [km]
        temperature_K: Ground temperature [K]
        pressure_hPa: Surface pressure [hPa]
        relative_humidity: Relative humidity (0-1)
        aerosol_type: Aerosol model type
        water_vapor_column: Total precipitable water [g/cm²]
        ozone_column: Total ozone column [DU]
    """

    visibility_km: float = 23.0  # Standard visibility
    temperature_K: float = 288.15  # Standard temperature
    pressure_hPa: float = 1013.25  # Standard pressure
    relative_humidity: float = 0.5
    aerosol_type: str = "rural"  # rural, urban, maritime, desert
    water_vapor_column: Optional[float] = None  # Computed from RH if None
    ozone_column: float = 300.0  # Dobson units


@dataclass
class AtmosphereResult:
    """Result of atmospheric transmission calculation.

    Attributes:
        transmission: Path transmission (0-1), scalar or spectral
        path_radiance: Atmospheric path radiance [W/(m²·sr·μm)]
        sky_radiance: Downwelling sky radiance [W/(m²·sr·μm)]
        spherical_albedo: Atmospheric spherical albedo (for adjacency)
    """

    transmission: Union[float, NDArray[np.floating]]
    path_radiance: Union[float, NDArray[np.floating]] = 0.0
    sky_radiance: Union[float, NDArray[np.floating]] = 0.0
    spherical_albedo: Union[float, NDArray[np.floating]] = 0.0

    @property
    def is_spectral(self) -> bool:
        """Check if results are spectral (vs broadband)."""
        return isinstance(self.transmission, np.ndarray)


class AtmosphereModel(ABC):
    """Abstract base class for atmosphere models.

    All atmosphere implementations provide:
    1. Path transmission τ(λ)
    2. Path radiance L_path(λ)
    3. Sky radiance L_sky(λ) (downwelling)
    """

    @property
    @abstractmethod
    def fidelity(self) -> AtmosphereFidelity:
        """Return the fidelity level of this model."""
        pass

    @abstractmethod
    def compute(
        self,
        wavelength_um: Union[float, NDArray[np.floating]],
        path: PathGeometry,
        conditions: AtmosphereConditions,
    ) -> AtmosphereResult:
        """Compute atmospheric effects for given path and conditions.

        Args:
            wavelength_um: Wavelength(s) in micrometers
            path: Path geometry (range, altitude, angle)
            conditions: Atmospheric conditions

        Returns:
            AtmosphereResult with transmission and radiances
        """
        pass

    def compute_band(
        self,
        band: SpectralBand,
        path: PathGeometry,
        conditions: AtmosphereConditions,
        n_samples: int = 20,
    ) -> AtmosphereResult:
        """Compute band-averaged atmospheric effects.

        Args:
            band: Spectral band
            path: Path geometry
            conditions: Atmospheric conditions
            n_samples: Number of spectral samples

        Returns:
            Band-averaged AtmosphereResult
        """
        wavelengths = np.linspace(band.lambda_min_um, band.lambda_max_um, n_samples)
        result = self.compute(wavelengths, path, conditions)

        # Band-average the results
        return AtmosphereResult(
            transmission=float(np.mean(result.transmission)),
            path_radiance=float(np.mean(result.path_radiance)),
            sky_radiance=float(np.mean(result.sky_radiance)),
            spherical_albedo=float(np.mean(result.spherical_albedo)),
        )

    def at_sensor_radiance(
        self,
        surface_radiance: Union[float, NDArray[np.floating]],
        wavelength_um: Union[float, NDArray[np.floating]],
        path: PathGeometry,
        conditions: AtmosphereConditions,
    ) -> Union[float, NDArray[np.floating]]:
        """Compute at-sensor radiance from surface radiance.

        L_sensor = τ × L_surface + L_path

        Args:
            surface_radiance: Surface leaving radiance [W/(m²·sr·μm)]
            wavelength_um: Wavelength(s)
            path: Path geometry
            conditions: Atmospheric conditions

        Returns:
            At-sensor radiance [W/(m²·sr·μm)]
        """
        result = self.compute(wavelength_um, path, conditions)
        return result.transmission * surface_radiance + result.path_radiance


class NoAtmosphere(AtmosphereModel):
    """Trivial atmosphere model with no atmospheric effects.

    Useful for debugging and as a baseline.
    """

    @property
    def fidelity(self) -> AtmosphereFidelity:
        return AtmosphereFidelity.NONE

    def compute(
        self,
        wavelength_um: Union[float, NDArray[np.floating]],
        path: PathGeometry,
        conditions: AtmosphereConditions,
    ) -> AtmosphereResult:
        """Return unit transmission and zero path radiance."""
        wavelength_um = np.atleast_1d(wavelength_um)
        n = len(wavelength_um)

        if n == 1:
            return AtmosphereResult(transmission=1.0)
        else:
            return AtmosphereResult(
                transmission=np.ones(n),
                path_radiance=np.zeros(n),
                sky_radiance=np.zeros(n),
            )
