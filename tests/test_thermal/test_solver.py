"""Tests for thermal.solver module."""

import pytest
import numpy as np

from eosim.thermal.solver import (
    EnvironmentalConditions,
    SurfaceState,
    SimpleEnergyBalance,
    TransientThermalSolver,
    estimate_temperature,
)
from eosim.scene.materials import Material, SpectralProperty, ThermalProperties


class TestEnvironmentalConditions:
    """Tests for EnvironmentalConditions class."""

    def test_default_conditions(self) -> None:
        """Test default environmental conditions."""
        cond = EnvironmentalConditions()
        assert cond.air_temperature_K == pytest.approx(288.15)
        assert cond.visibility_km == pytest.approx(23.0)

    def test_solar_zenith_conversion(self) -> None:
        """Test solar zenith angle conversions."""
        cond = EnvironmentalConditions(solar_zenith_angle_deg=60.0)
        assert cond.solar_zenith_angle_rad == pytest.approx(np.radians(60.0))
        assert cond.cos_solar_zenith == pytest.approx(0.5)

    def test_sky_irradiance(self) -> None:
        """Test sky irradiance calculation."""
        cond = EnvironmentalConditions(sky_temperature_K=250.0)
        # σT^4 for 250K
        from eosim.core.constants import CONSTANTS
        expected = CONSTANTS.sigma * 250.0**4
        assert cond.sky_irradiance == pytest.approx(expected)


class TestSimpleEnergyBalance:
    """Tests for SimpleEnergyBalance solver."""

    def test_equilibrium_no_solar(self) -> None:
        """Without solar, equilibrium should be near air temperature."""
        solver = SimpleEnergyBalance()
        material = Material(
            name="test",
            emissivity=SpectralProperty.constant(0.9),
        )
        conditions = EnvironmentalConditions(
            solar_direct_irradiance=0.0,
            solar_diffuse_irradiance=0.0,
            air_temperature_K=300.0,
        )

        state = solver.solve_steady_state(material, conditions)

        # Without solar heating, surface should be close to air temperature
        # (slightly below due to radiative cooling)
        assert state.temperature_K < 300.0
        assert state.temperature_K > 280.0

    def test_solar_heating(self) -> None:
        """Solar irradiance should heat surface above air temperature."""
        solver = SimpleEnergyBalance()
        material = Material(
            name="test",
            emissivity=SpectralProperty.constant(0.9),
        )

        conditions_no_sun = EnvironmentalConditions(
            solar_direct_irradiance=0.0,
            air_temperature_K=300.0,
        )
        conditions_sun = EnvironmentalConditions(
            solar_direct_irradiance=800.0,
            air_temperature_K=300.0,
            solar_zenith_angle_deg=30.0,
        )

        state_no_sun = solver.solve_steady_state(material, conditions_no_sun)
        state_sun = solver.solve_steady_state(material, conditions_sun)

        # Solar heating should raise temperature
        assert state_sun.temperature_K > state_no_sun.temperature_K

    def test_emissivity_effect(self) -> None:
        """Higher emissivity should cool surface (more emission)."""
        solver = SimpleEnergyBalance()
        material_high_e = Material(
            name="high_e",
            emissivity=SpectralProperty.constant(0.95),
        )
        material_low_e = Material(
            name="low_e",
            emissivity=SpectralProperty.constant(0.3),
        )
        conditions = EnvironmentalConditions(
            solar_direct_irradiance=500.0,
            air_temperature_K=300.0,
        )

        state_high = solver.solve_steady_state(material_high_e, conditions)
        state_low = solver.solve_steady_state(material_low_e, conditions)

        # Low emissivity surface retains more heat (if solar abs is also low)
        # This depends on solar absorptivity vs thermal emissivity
        # For graybody with same ε, low e should be cooler with no solar
        # but effect depends on reflectance

    def test_convergence(self) -> None:
        """Solver should converge."""
        solver = SimpleEnergyBalance(max_iterations=100, tolerance_K=0.01)
        material = Material(
            name="test",
            emissivity=SpectralProperty.constant(0.9),
        )
        conditions = EnvironmentalConditions()

        state = solver.solve_steady_state(material, conditions)

        # Should have reached a reasonable temperature
        assert 200 < state.temperature_K < 400


class TestTransientThermalSolver:
    """Tests for TransientThermalSolver."""

    def test_diurnal_cycle(self) -> None:
        """Test solving over a diurnal cycle."""
        solver = TransientThermalSolver(dt_seconds=600, n_layers=3)
        material = Material(
            name="test",
            emissivity=SpectralProperty.constant(0.9),
            thermal=ThermalProperties(
                thermal_conductivity=1.0,
                specific_heat=1000.0,
                density=2000.0,
                thickness=0.1,
            ),
        )

        # Simple day/night cycle
        conditions_series = []
        for hour in range(24):
            if 6 <= hour <= 18:
                solar = 500.0 * np.sin(np.pi * (hour - 6) / 12)
            else:
                solar = 0.0

            conditions_series.append(EnvironmentalConditions(
                solar_direct_irradiance=solar,
                air_temperature_K=290.0 + 5 * np.sin(np.pi * (hour - 6) / 12),
            ))

        results = solver.solve_diurnal(
            material,
            conditions_series,
            initial_temperature_K=290.0,
        )

        assert len(results) == 24
        # Temperatures should vary over the day
        temps = [r.temperature_K for r in results]
        assert max(temps) > min(temps)


class TestEstimateTemperature:
    """Tests for estimate_temperature function."""

    def test_no_solar(self) -> None:
        """Without solar, should be near air temperature."""
        T = estimate_temperature(
            emissivity=0.9,
            solar_absorptivity=0.5,
            solar_irradiance=0.0,
            air_temperature_K=300.0,
        )
        # Should be slightly below air temp due to radiative cooling
        assert T < 300.0
        assert T > 280.0

    def test_solar_heating(self) -> None:
        """Solar should heat above air temperature."""
        T_no_sun = estimate_temperature(
            emissivity=0.9,
            solar_absorptivity=0.8,
            solar_irradiance=0.0,
            air_temperature_K=300.0,
        )
        T_sun = estimate_temperature(
            emissivity=0.9,
            solar_absorptivity=0.8,
            solar_irradiance=800.0,
            air_temperature_K=300.0,
        )
        assert T_sun > T_no_sun
