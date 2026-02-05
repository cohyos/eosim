"""
Thermal solver for EOSIM.

Implements energy balance equations to compute surface temperatures
from environmental conditions and material properties.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from eosim.core.constants import CONSTANTS
from eosim.scene.materials import Material, ThermalProperties


@dataclass
class EnvironmentalConditions:
    """Environmental conditions for thermal balance.

    All radiation values are irradiances in W/m².
    """

    # Solar radiation
    solar_direct_irradiance: float = 0.0  # Direct solar [W/m²]
    solar_diffuse_irradiance: float = 0.0  # Diffuse sky [W/m²]
    solar_zenith_angle_deg: float = 45.0  # Solar zenith angle [degrees]

    # Thermal radiation
    sky_temperature_K: float = 250.0  # Effective sky temperature [K]
    ground_temperature_K: float = 288.0  # Surrounding ground temperature [K]

    # Atmospheric conditions
    air_temperature_K: float = 288.15  # Air temperature [K]
    wind_speed_m_s: float = 1.0  # Wind speed [m/s]
    relative_humidity: float = 0.5  # Relative humidity (0-1)
    visibility_km: float = 23.0  # Atmospheric visibility [km]

    @property
    def solar_zenith_angle_rad(self) -> float:
        """Solar zenith angle in radians."""
        return np.radians(self.solar_zenith_angle_deg)

    @property
    def cos_solar_zenith(self) -> float:
        """Cosine of solar zenith angle."""
        return np.cos(self.solar_zenith_angle_rad)

    @property
    def sky_irradiance(self) -> float:
        """Downwelling longwave (thermal) sky irradiance [W/m²]."""
        return CONSTANTS.sigma * self.sky_temperature_K ** 4


@dataclass
class SurfaceState:
    """State of a surface element.

    Attributes:
        temperature_K: Surface temperature [K]
        absorbed_solar: Absorbed solar power [W/m²]
        emitted_thermal: Emitted thermal power [W/m²]
        net_radiation: Net radiative flux [W/m²]
        convective_flux: Convective heat flux [W/m²]
        conductive_flux: Conductive heat flux [W/m²]
    """

    temperature_K: float
    absorbed_solar: float = 0.0
    emitted_thermal: float = 0.0
    net_radiation: float = 0.0
    convective_flux: float = 0.0
    conductive_flux: float = 0.0


class ThermalSolver(ABC):
    """Abstract base class for thermal solvers."""

    @abstractmethod
    def solve_steady_state(
        self,
        material: Material,
        conditions: EnvironmentalConditions,
        surface_tilt_deg: float = 0.0,
    ) -> SurfaceState:
        """Solve for steady-state surface temperature.

        Args:
            material: Surface material properties
            conditions: Environmental conditions
            surface_tilt_deg: Surface tilt from horizontal [degrees]

        Returns:
            Surface state with equilibrium temperature
        """
        pass


class SimpleEnergyBalance(ThermalSolver):
    """Simple energy balance solver.

    Solves the steady-state energy balance equation:
    Q_absorbed = Q_emitted + Q_convection

    Assumes:
    - Lambertian surface
    - No conduction to substrate
    - Simplified view factors
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance_K: float = 0.01,
    ) -> None:
        """Initialize solver.

        Args:
            max_iterations: Maximum Newton-Raphson iterations
            tolerance_K: Temperature convergence tolerance [K]
        """
        self.max_iterations = max_iterations
        self.tolerance_K = tolerance_K

    def solve_steady_state(
        self,
        material: Material,
        conditions: EnvironmentalConditions,
        surface_tilt_deg: float = 0.0,
    ) -> SurfaceState:
        """Solve for steady-state temperature using Newton-Raphson."""
        emissivity = float(material.emissivity.values[0])
        solar_absorptivity = 1.0 - float(material.reflectance.values[0])

        # Surface orientation effects
        tilt_rad = np.radians(surface_tilt_deg)
        cos_incidence = max(0, np.cos(conditions.solar_zenith_angle_rad - tilt_rad))

        # Absorbed solar radiation
        Q_solar_direct = solar_absorptivity * conditions.solar_direct_irradiance * cos_incidence
        Q_solar_diffuse = solar_absorptivity * conditions.solar_diffuse_irradiance * 0.5 * (1 + np.cos(tilt_rad))
        Q_solar = Q_solar_direct + Q_solar_diffuse

        # Absorbed thermal (longwave) radiation
        # From sky (upper hemisphere)
        sky_view = 0.5 * (1 + np.cos(tilt_rad))
        Q_sky = emissivity * sky_view * CONSTANTS.sigma * conditions.sky_temperature_K ** 4

        # From ground (lower hemisphere)
        ground_view = 0.5 * (1 - np.cos(tilt_rad))
        Q_ground = emissivity * ground_view * CONSTANTS.sigma * conditions.ground_temperature_K ** 4

        Q_absorbed = Q_solar + Q_sky + Q_ground

        # Initial guess: air temperature
        T = conditions.air_temperature_K

        # Newton-Raphson iteration
        for _ in range(self.max_iterations):
            # Emitted thermal radiation
            Q_emitted = emissivity * CONSTANTS.sigma * T ** 4

            # Convective heat transfer
            h_conv = self._convection_coefficient(T, conditions)
            Q_conv = h_conv * (T - conditions.air_temperature_K)

            # Residual (should be zero at equilibrium)
            R = Q_absorbed - Q_emitted - Q_conv

            # Derivative of residual with respect to T
            dR_dT = -4 * emissivity * CONSTANTS.sigma * T ** 3 - h_conv

            # Newton update
            dT = -R / dR_dT
            T = T + dT

            if abs(dT) < self.tolerance_K:
                break

        # Final state
        Q_emitted_final = emissivity * CONSTANTS.sigma * T ** 4
        Q_conv_final = h_conv * (T - conditions.air_temperature_K)

        return SurfaceState(
            temperature_K=T,
            absorbed_solar=Q_solar,
            emitted_thermal=Q_emitted_final,
            net_radiation=Q_absorbed - Q_emitted_final,
            convective_flux=Q_conv_final,
        )

    def _convection_coefficient(
        self,
        surface_T: float,
        conditions: EnvironmentalConditions,
    ) -> float:
        """Compute convective heat transfer coefficient.

        Uses empirical correlation combining free and forced convection.

        Args:
            surface_T: Surface temperature [K]
            conditions: Environmental conditions

        Returns:
            Convection coefficient h [W/(m²·K)]
        """
        # Forced convection (wind)
        # McAdams correlation for flat plate
        v = max(conditions.wind_speed_m_s, 0.1)  # Avoid zero
        h_forced = 5.7 + 3.8 * v

        # Free (natural) convection
        dT = abs(surface_T - conditions.air_temperature_K)
        # Simplified correlation
        h_free = 1.3 * dT ** 0.25 if dT > 0 else 0

        # Combined (approximate)
        return np.sqrt(h_forced ** 2 + h_free ** 2)


class TransientThermalSolver:
    """Transient (time-dependent) thermal solver.

    Solves the 1D heat equation for a thin slab using explicit
    finite difference method.
    """

    def __init__(
        self,
        dt_seconds: float = 60.0,
        n_layers: int = 5,
    ) -> None:
        """Initialize solver.

        Args:
            dt_seconds: Time step [seconds]
            n_layers: Number of layers for 1D discretization
        """
        self.dt = dt_seconds
        self.n_layers = n_layers
        self.steady_solver = SimpleEnergyBalance()

    def solve_diurnal(
        self,
        material: Material,
        conditions_series: list[EnvironmentalConditions],
        initial_temperature_K: float = 288.0,
        surface_tilt_deg: float = 0.0,
    ) -> list[SurfaceState]:
        """Solve transient thermal problem over a time series.

        Args:
            material: Surface material
            conditions_series: Time series of environmental conditions
            initial_temperature_K: Initial temperature
            surface_tilt_deg: Surface tilt angle

        Returns:
            Time series of surface states
        """
        thermal = material.thermal
        emissivity = float(material.emissivity.values[0])
        solar_abs = 1.0 - float(material.reflectance.values[0])

        # Initialize temperature profile (1D through material)
        dz = thermal.thickness / self.n_layers
        T = np.full(self.n_layers, initial_temperature_K)

        # Thermal diffusivity
        alpha = thermal.thermal_diffusivity

        # Stability check for explicit method
        # dt <= dz² / (2 * alpha)
        dt_stable = 0.4 * dz ** 2 / alpha
        dt = min(self.dt, dt_stable)

        # Coefficient for explicit scheme
        r = alpha * dt / dz ** 2

        results = []
        tilt_rad = np.radians(surface_tilt_deg)

        for conditions in conditions_series:
            # Surface boundary condition (energy balance)
            cos_inc = max(0, np.cos(conditions.solar_zenith_angle_rad - tilt_rad))
            Q_solar = solar_abs * (
                conditions.solar_direct_irradiance * cos_inc +
                conditions.solar_diffuse_irradiance * 0.5 * (1 + np.cos(tilt_rad))
            )

            # Sky and ground thermal radiation
            sky_view = 0.5 * (1 + np.cos(tilt_rad))
            ground_view = 1 - sky_view
            Q_in_lw = (
                emissivity * sky_view * CONSTANTS.sigma * conditions.sky_temperature_K ** 4 +
                emissivity * ground_view * CONSTANTS.sigma * conditions.ground_temperature_K ** 4
            )

            # Emitted radiation
            Q_emit = emissivity * CONSTANTS.sigma * T[0] ** 4

            # Convection
            h = self.steady_solver._convection_coefficient(T[0], conditions)
            Q_conv = h * (T[0] - conditions.air_temperature_K)

            # Net flux at surface
            Q_net = Q_solar + Q_in_lw - Q_emit - Q_conv

            # Surface temperature update (explicit)
            T_new = T.copy()

            # Surface layer with heat flux
            T_new[0] = T[0] + dt * (
                Q_net / (thermal.density * thermal.specific_heat * dz) +
                alpha * (T[1] - T[0]) / dz ** 2
            )

            # Interior layers
            for i in range(1, self.n_layers - 1):
                T_new[i] = T[i] + r * (T[i + 1] - 2 * T[i] + T[i - 1])

            # Bottom boundary (insulated or constant)
            T_new[-1] = T[-1] + r * (T[-2] - T[-1])

            T = T_new

            # Record state
            results.append(SurfaceState(
                temperature_K=T[0],
                absorbed_solar=Q_solar,
                emitted_thermal=Q_emit,
                net_radiation=Q_net,
                convective_flux=Q_conv,
            ))

        return results


def estimate_temperature(
    emissivity: float,
    solar_absorptivity: float,
    solar_irradiance: float,
    air_temperature_K: float,
    wind_speed_m_s: float = 2.0,
) -> float:
    """Quick temperature estimate using simplified energy balance.

    Useful for rough estimates without full material definition.

    Args:
        emissivity: Surface emissivity
        solar_absorptivity: Solar absorptivity
        solar_irradiance: Total incident solar [W/m²]
        air_temperature_K: Air temperature [K]
        wind_speed_m_s: Wind speed [m/s]

    Returns:
        Estimated surface temperature [K]
    """
    # Absorbed solar
    Q_solar = solar_absorptivity * solar_irradiance

    # Convection coefficient (simplified)
    h = 5.7 + 3.8 * wind_speed_m_s

    # Initial guess
    T = air_temperature_K

    # Simple iteration
    for _ in range(50):
        Q_emit = emissivity * CONSTANTS.sigma * T ** 4
        Q_conv = h * (T - air_temperature_K)
        residual = Q_solar - Q_emit - Q_conv

        # Derivative
        dR_dT = -4 * emissivity * CONSTANTS.sigma * T ** 3 - h
        dT = -residual / dR_dT
        T = T + dT

        if abs(dT) < 0.01:
            break

    return T


def compute_nedt_temperature(
    noise_e: float,
    responsivity_e_per_K: float,
) -> float:
    """Compute Noise Equivalent Differential Temperature (NEΔT).

    NEΔT = noise / (dSignal/dT)

    Args:
        noise_e: RMS noise in electrons
        responsivity_e_per_K: Signal change per Kelvin [e⁻/K]

    Returns:
        NEΔT in Kelvin
    """
    if responsivity_e_per_K <= 0:
        return float("inf")
    return noise_e / responsivity_e_per_K
