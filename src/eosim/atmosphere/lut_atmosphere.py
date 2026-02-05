"""
EOSIM LUT-Based Atmosphere Model (Stage B Enhancement).

Provides look-up table (LUT) based atmospheric transmission using
pre-computed values for various conditions and wavelengths.

Example 1: Standard atmosphere with different visibility conditions
    >>> from eosim.atmosphere import LUTAtmosphere, PathGeometry, AtmosphereConditions
    >>> model = LUTAtmosphere()
    >>> path = PathGeometry(ground_range_m=5000, altitude_end_m=1000)
    >>> clear = AtmosphereConditions(visibility_km=50.0)
    >>> hazy = AtmosphereConditions(visibility_km=5.0)
    >>> result_clear = model.compute(10.0, path, clear)
    >>> result_hazy = model.compute(10.0, path, hazy)
    >>> print(f"Clear: {result_clear.transmission:.3f}, Hazy: {result_hazy.transmission:.3f}")

Example 2: Spectral transmission across LWIR band
    >>> wavelengths = np.linspace(8, 14, 50)
    >>> result = model.compute(wavelengths, path, AtmosphereConditions())
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(wavelengths, result.transmission)
    >>> plt.xlabel('Wavelength (um)'); plt.ylabel('Transmission')

Example 3: Different altitude paths
    >>> ground_path = PathGeometry(ground_range_m=5000, altitude_end_m=100)
    >>> high_path = PathGeometry(ground_range_m=5000, altitude_end_m=10000)
    >>> r1 = model.compute(10.0, ground_path, AtmosphereConditions())
    >>> r2 = model.compute(10.0, high_path, AtmosphereConditions())
    >>> print(f"Low alt: {r1.transmission:.3f}, High alt: {r2.transmission:.3f}")
"""

from dataclasses import dataclass, field
from typing import Union, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from eosim.core.compat import integrate_trapz
from eosim.atmosphere.base import (
    AtmosphereModel,
    AtmosphereFidelity,
    PathGeometry,
    AtmosphereConditions,
    AtmosphereResult,
)
from eosim.radiance.planck import spectral_radiance


@dataclass
class LUTAtmosphereData:
    """Pre-computed LUT data for atmosphere model.

    Attributes:
        wavelength_grid: Wavelength points [um]
        range_grid: Range points [km]
        altitude_grid: Altitude points [km]
        visibility_grid: Visibility points [km]
        transmission_lut: 4D transmission array [wavelength, range, altitude, visibility]
        path_radiance_lut: 4D path radiance array
    """
    wavelength_grid: NDArray = field(default_factory=lambda: np.array([]))
    range_grid: NDArray = field(default_factory=lambda: np.array([]))
    altitude_grid: NDArray = field(default_factory=lambda: np.array([]))
    visibility_grid: NDArray = field(default_factory=lambda: np.array([]))
    transmission_lut: NDArray = field(default_factory=lambda: np.array([]))
    path_radiance_lut: NDArray = field(default_factory=lambda: np.array([]))


class LUTAtmosphere(AtmosphereModel):
    """Look-Up Table based atmosphere model.

    Uses pre-computed transmission and path radiance values interpolated
    across wavelength, range, altitude, and visibility dimensions.

    Based on MODTRAN-style atmospheric characterization with:
    - Molecular absorption (H2O, CO2, O3, CH4, N2O)
    - Rayleigh scattering
    - Aerosol extinction and scattering
    - Path thermal emission
    """

    # Wavelength grid [um]
    _WAVELENGTH_GRID = np.array([
        0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.9, 2.2,
        3.0, 3.5, 4.0, 4.3, 4.8, 5.0,
        8.0, 9.0, 9.6, 10.0, 11.0, 12.0, 14.0
    ])

    # Range grid [km]
    _RANGE_GRID = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])

    # Altitude grid [km]
    _ALTITUDE_GRID = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])

    # Visibility grid [km]
    _VISIBILITY_GRID = np.array([2.0, 5.0, 10.0, 23.0, 50.0])

    def __init__(self, aerosol_type: str = "rural") -> None:
        """Initialize LUT atmosphere model.

        Args:
            aerosol_type: Aerosol model ("rural", "urban", "maritime", "desert")
        """
        self.aerosol_type = aerosol_type
        self._lut_data = self._build_lut()
        self._transmission_interp = self._create_interpolator(self._lut_data.transmission_lut)

    @property
    def fidelity(self) -> AtmosphereFidelity:
        return AtmosphereFidelity.STANDARD

    def _build_lut(self) -> LUTAtmosphereData:
        """Build the look-up table with pre-computed values."""
        nw = len(self._WAVELENGTH_GRID)
        nr = len(self._RANGE_GRID)
        na = len(self._ALTITUDE_GRID)
        nv = len(self._VISIBILITY_GRID)

        transmission = np.zeros((nw, nr, na, nv))
        path_radiance = np.zeros((nw, nr, na, nv))

        for iw, w in enumerate(self._WAVELENGTH_GRID):
            for ir, r in enumerate(self._RANGE_GRID):
                for ia, a in enumerate(self._ALTITUDE_GRID):
                    for iv, v in enumerate(self._VISIBILITY_GRID):
                        tau, Lp = self._compute_transmission_at_point(w, r, a, v)
                        transmission[iw, ir, ia, iv] = tau
                        path_radiance[iw, ir, ia, iv] = Lp

        return LUTAtmosphereData(
            wavelength_grid=self._WAVELENGTH_GRID,
            range_grid=self._RANGE_GRID,
            altitude_grid=self._ALTITUDE_GRID,
            visibility_grid=self._VISIBILITY_GRID,
            transmission_lut=transmission,
            path_radiance_lut=path_radiance,
        )

    def _compute_transmission_at_point(
        self,
        wavelength_um: float,
        range_km: float,
        altitude_km: float,
        visibility_km: float,
    ) -> tuple[float, float]:
        """Compute transmission at a single LUT point.

        Args:
            wavelength_um: Wavelength [um]
            range_km: Slant range [km]
            altitude_km: Sensor altitude [km]
            visibility_km: Visibility [km]

        Returns:
            Tuple of (transmission, path_radiance)
        """
        # Molecular absorption coefficient
        beta_mol = self._molecular_absorption(wavelength_um, altitude_km)

        # Rayleigh scattering
        beta_rayleigh = self._rayleigh_scattering(wavelength_um, altitude_km)

        # Aerosol extinction
        beta_aerosol = self._aerosol_extinction(wavelength_um, visibility_km)

        # Total extinction
        beta_total = beta_mol + beta_rayleigh + beta_aerosol

        # Beer-Lambert transmission
        transmission = np.exp(-beta_total * range_km)

        # Path radiance (thermal + scattered)
        T_atm = 288.0 - 6.5 * altitude_km  # Simple lapse rate
        avg_emis = 1 - np.sqrt(transmission)
        L_path = avg_emis * spectral_radiance(wavelength_um, max(T_atm, 200.0))

        return float(transmission), float(L_path)

    def _molecular_absorption(self, wavelength_um: float, altitude_km: float) -> float:
        """Compute molecular absorption coefficient [1/km]."""
        # Scale factor for altitude (pressure decrease)
        scale_height = 8.5  # km
        altitude_factor = np.exp(-altitude_km / scale_height)

        # Wavelength-dependent absorption coefficients [1/km at sea level]
        # Based on major absorbers: H2O, CO2, O3

        # H2O bands
        h2o_coeff = 0.0
        if 1.3 < wavelength_um < 1.5:
            h2o_coeff = 0.2 * np.exp(-((wavelength_um - 1.4) / 0.05) ** 2)
        elif 1.8 < wavelength_um < 2.0:
            h2o_coeff = 0.3 * np.exp(-((wavelength_um - 1.9) / 0.05) ** 2)
        elif 2.5 < wavelength_um < 3.0:
            h2o_coeff = 0.4 * np.exp(-((wavelength_um - 2.7) / 0.1) ** 2)
        elif 5.5 < wavelength_um < 7.5:
            h2o_coeff = 0.5 * np.exp(-((wavelength_um - 6.3) / 0.5) ** 2)

        # CO2 bands
        co2_coeff = 0.0
        if 4.2 < wavelength_um < 4.5:
            co2_coeff = 0.25 * np.exp(-((wavelength_um - 4.3) / 0.05) ** 2)
        elif 14.5 < wavelength_um < 16.0:
            co2_coeff = 0.3 * np.exp(-((wavelength_um - 15.0) / 0.3) ** 2)

        # O3 band
        o3_coeff = 0.0
        if 9.4 < wavelength_um < 9.8:
            o3_coeff = 0.1 * np.exp(-((wavelength_um - 9.6) / 0.1) ** 2)

        return (h2o_coeff + co2_coeff + o3_coeff) * altitude_factor

    def _rayleigh_scattering(self, wavelength_um: float, altitude_km: float) -> float:
        """Compute Rayleigh scattering coefficient [1/km]."""
        # Rayleigh scattering scales as lambda^-4
        lambda_ref = 0.55  # Reference wavelength
        beta_ref = 0.0116  # Rayleigh coefficient at 550nm, sea level [1/km]

        # Scale with altitude
        scale_height = 8.5
        altitude_factor = np.exp(-altitude_km / scale_height)

        return beta_ref * (lambda_ref / wavelength_um) ** 4 * altitude_factor

    def _aerosol_extinction(self, wavelength_um: float, visibility_km: float) -> float:
        """Compute aerosol extinction coefficient [1/km]."""
        # Koschmieder relation for visual range
        beta_550 = 3.912 / visibility_km

        # Wavelength dependence varies by aerosol type
        # Angstrom exponent
        if self.aerosol_type == "rural":
            angstrom = 1.3  # Smaller particles
        elif self.aerosol_type == "urban":
            angstrom = 1.5  # Small soot particles
        elif self.aerosol_type == "maritime":
            angstrom = 0.3  # Large sea salt particles
        elif self.aerosol_type == "desert":
            angstrom = 0.0  # Large dust particles
        else:
            angstrom = 1.0

        # Angstrom relation: beta(lambda) = beta(550nm) * (lambda/550nm)^-alpha
        return beta_550 * (0.55 / wavelength_um) ** angstrom

    def _create_interpolator(self, lut: NDArray) -> RegularGridInterpolator:
        """Create interpolator for LUT data."""
        return RegularGridInterpolator(
            (self._WAVELENGTH_GRID, self._RANGE_GRID,
             self._ALTITUDE_GRID, self._VISIBILITY_GRID),
            lut,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def compute(
        self,
        wavelength_um: Union[float, NDArray[np.floating]],
        path: PathGeometry,
        conditions: AtmosphereConditions,
    ) -> AtmosphereResult:
        """Compute atmospheric transmission using LUT interpolation."""
        wavelength_um = np.atleast_1d(wavelength_um)
        n = len(wavelength_um)

        # Clamp inputs to grid bounds
        range_km = np.clip(path.slant_range_m / 1000.0, 0.1, 50.0)
        altitude_km = np.clip(path.altitude_end_m / 1000.0, 0.0, 20.0)
        visibility_km = np.clip(conditions.visibility_km, 2.0, 50.0)

        # Build interpolation points
        points = np.column_stack([
            wavelength_um,
            np.full(n, range_km),
            np.full(n, altitude_km),
            np.full(n, visibility_km),
        ])

        # Interpolate transmission
        transmission = self._transmission_interp(points)
        transmission = np.clip(transmission, 0.0, 1.0)

        # Compute path radiance
        T_atm = conditions.temperature_K - 20
        avg_emis = 1 - np.sqrt(np.mean(transmission))
        L_blackbody = spectral_radiance(wavelength_um, T_atm)
        path_radiance = avg_emis * L_blackbody * (1 - transmission)

        # Sky radiance
        T_sky = conditions.temperature_K - 30
        sky_radiance = 0.9 * spectral_radiance(wavelength_um, T_sky)

        if n == 1:
            return AtmosphereResult(
                transmission=float(transmission[0]),
                path_radiance=float(path_radiance[0]),
                sky_radiance=float(sky_radiance[0]),
            )

        return AtmosphereResult(
            transmission=transmission,
            path_radiance=path_radiance,
            sky_radiance=sky_radiance,
        )


class AerosolModel:
    """Aerosol optical properties model.

    Provides wavelength-dependent extinction, single scatter albedo,
    and asymmetry parameter for different aerosol types.
    """

    # Aerosol optical properties at 550nm
    _AEROSOL_PROPERTIES = {
        # (extinction_eff, single_scatter_albedo, asymmetry, angstrom_exp)
        "rural": (0.85, 0.92, 0.65, 1.3),
        "urban": (0.70, 0.85, 0.68, 1.5),
        "maritime": (0.75, 0.99, 0.72, 0.3),
        "desert": (0.90, 0.95, 0.75, 0.0),
        "continental": (0.80, 0.90, 0.65, 1.2),
        "biomass": (0.65, 0.88, 0.60, 1.8),
    }

    def __init__(self, aerosol_type: str = "rural") -> None:
        """Initialize aerosol model.

        Args:
            aerosol_type: Type of aerosol model
        """
        if aerosol_type not in self._AEROSOL_PROPERTIES:
            aerosol_type = "rural"

        props = self._AEROSOL_PROPERTIES[aerosol_type]
        self.aerosol_type = aerosol_type
        self.extinction_efficiency = props[0]
        self.single_scatter_albedo = props[1]
        self.asymmetry_parameter = props[2]
        self.angstrom_exponent = props[3]

    def extinction(
        self,
        wavelength_um: Union[float, NDArray],
        aod_550: float = 0.1,
    ) -> Union[float, NDArray]:
        """Compute aerosol extinction at wavelength.

        Args:
            wavelength_um: Wavelength(s) [um]
            aod_550: Aerosol optical depth at 550nm

        Returns:
            Aerosol extinction coefficient [1/km] assuming 1km scale height
        """
        # Angstrom law
        aod = aod_550 * (0.55 / wavelength_um) ** self.angstrom_exponent
        # Convert AOD to extinction (assume 1 km scale height)
        return aod

    def scattering(
        self,
        wavelength_um: Union[float, NDArray],
        aod_550: float = 0.1,
    ) -> Union[float, NDArray]:
        """Compute aerosol scattering coefficient."""
        ext = self.extinction(wavelength_um, aod_550)
        return ext * self.single_scatter_albedo

    def phase_function(
        self,
        cos_theta: Union[float, NDArray],
    ) -> Union[float, NDArray]:
        """Henyey-Greenstein phase function.

        Args:
            cos_theta: Cosine of scattering angle

        Returns:
            Phase function value (normalized)
        """
        g = self.asymmetry_parameter
        return (1 - g**2) / (4 * np.pi * (1 + g**2 - 2 * g * cos_theta) ** 1.5)


class TurbulenceModel:
    """Atmospheric turbulence model for scintillation effects.

    Models Cn² (structure constant) profiles and computes
    turbulence-induced blur and scintillation.
    """

    def __init__(
        self,
        cn2_ground: float = 1e-14,
        turbulence_profile: str = "hufnagel_valley",
    ) -> None:
        """Initialize turbulence model.

        Args:
            cn2_ground: Ground-level Cn² [m^-2/3]
            turbulence_profile: Profile model
        """
        self.cn2_ground = cn2_ground
        self.profile = turbulence_profile

    def cn2_profile(self, altitude_m: float) -> float:
        """Compute Cn² at altitude using Hufnagel-Valley model.

        Args:
            altitude_m: Altitude [m]

        Returns:
            Cn² value [m^-2/3]
        """
        h = altitude_m / 1000.0  # km

        if self.profile == "hufnagel_valley":
            # HV 5/7 model
            cn2 = (
                0.00594 * (27 / 1000) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000)
                + 2.7e-16 * np.exp(-h / 1500)
                + self.cn2_ground * np.exp(-h / 100)
            )
        else:
            # Simple exponential decay
            cn2 = self.cn2_ground * np.exp(-altitude_m / 1000)

        return cn2

    def fried_parameter(
        self,
        wavelength_um: float,
        path_length_m: float,
        zenith_angle_deg: float = 0.0,
    ) -> float:
        """Compute Fried parameter r0.

        Args:
            wavelength_um: Wavelength [um]
            path_length_m: Path length [m]
            zenith_angle_deg: Zenith angle [deg]

        Returns:
            Fried parameter r0 [m]
        """
        k = 2 * np.pi / (wavelength_um * 1e-6)  # wavenumber [1/m]
        sec_z = 1 / np.cos(np.radians(zenith_angle_deg))

        # Integrate Cn² along path
        n_points = 100
        heights = np.linspace(0, path_length_m, n_points)
        cn2_values = np.array([self.cn2_profile(h) for h in heights])
        cn2_integral = integrate_trapz(cn2_values, heights)

        r0 = (0.423 * k**2 * sec_z * cn2_integral) ** (-3/5)
        return r0

    def seeing_blur_fwhm(
        self,
        wavelength_um: float,
        path_length_m: float,
    ) -> float:
        """Compute seeing-limited blur FWHM.

        Args:
            wavelength_um: Wavelength [um]
            path_length_m: Path length [m]

        Returns:
            FWHM of seeing blur [arcsec]
        """
        r0 = self.fried_parameter(wavelength_um, path_length_m)
        # Long-exposure seeing FWHM
        lambda_m = wavelength_um * 1e-6
        fwhm_rad = 0.98 * lambda_m / r0
        return np.degrees(fwhm_rad) * 3600  # arcseconds

    def scintillation_index(
        self,
        wavelength_um: float,
        path_length_m: float,
        aperture_m: float = 0.1,
    ) -> float:
        """Compute scintillation index (intensity variance).

        Args:
            wavelength_um: Wavelength [um]
            path_length_m: Path length [m]
            aperture_m: Aperture diameter [m]

        Returns:
            Scintillation index (dimensionless)
        """
        k = 2 * np.pi / (wavelength_um * 1e-6)

        # Integrate Cn² along path with weighting
        n_points = 100
        heights = np.linspace(0, path_length_m, n_points)
        dz = path_length_m / n_points

        sigma_i2 = 0.0
        for i, h in enumerate(heights):
            cn2 = self.cn2_profile(h)
            z = h  # Distance from source
            L = path_length_m
            weight = (z / L) ** (5/6) * (1 - z / L) ** (5/6)
            sigma_i2 += cn2 * weight * dz

        sigma_i2 *= 1.23 * k ** (7/6) * path_length_m ** (11/6)

        # Aperture averaging
        if aperture_m > 0:
            D = aperture_m
            fresnel_zone = np.sqrt(wavelength_um * 1e-6 * path_length_m)
            if D > fresnel_zone:
                sigma_i2 *= (fresnel_zone / D) ** (7/3)

        return float(sigma_i2)
