"""Tests for atmosphere module."""

import pytest
import numpy as np

from eosim.atmosphere.base import (
    PathGeometry,
    AtmosphereConditions,
    AtmosphereResult,
    NoAtmosphere,
)
from eosim.atmosphere.simple import (
    BeerLambertAtmosphere,
    ConstantAtmosphere,
    estimate_transmission,
    estimate_band_transmission,
    koschmieder_visibility_to_extinction,
)
from eosim.core.spectral import BAND_LWIR, BAND_MWIR


class TestPathGeometry:
    """Tests for PathGeometry class."""

    def test_slant_range_horizontal(self) -> None:
        """Test slant range for horizontal path."""
        path = PathGeometry(
            ground_range_m=1000.0,
            altitude_start_m=0.0,
            altitude_end_m=0.0,
        )
        assert path.slant_range_m == pytest.approx(1000.0)

    def test_slant_range_vertical(self) -> None:
        """Test slant range for vertical path."""
        path = PathGeometry(
            ground_range_m=0.0,
            altitude_start_m=0.0,
            altitude_end_m=1000.0,
        )
        assert path.slant_range_m == pytest.approx(1000.0)

    def test_slant_range_diagonal(self) -> None:
        """Test slant range for diagonal path (3-4-5 triangle)."""
        path = PathGeometry(
            ground_range_m=3000.0,
            altitude_start_m=0.0,
            altitude_end_m=4000.0,
        )
        assert path.slant_range_m == pytest.approx(5000.0)

    def test_from_slant_range(self) -> None:
        """Test creating path from slant range."""
        path = PathGeometry.from_slant_range(
            slant_range_m=5000.0,
            zenith_angle_deg=45.0,
            sensor_altitude_m=1000.0,
        )
        assert path.slant_range_m == pytest.approx(5000.0, rel=0.01)

    def test_vertical_path(self) -> None:
        """Test vertical (nadir) path creation."""
        path = PathGeometry.vertical(altitude_m=500.0)
        assert path.zenith_angle_deg == 0.0
        assert path.altitude_end_m == 500.0


class TestAtmosphereConditions:
    """Tests for AtmosphereConditions class."""

    def test_default_conditions(self) -> None:
        """Test default atmospheric conditions."""
        cond = AtmosphereConditions()
        assert cond.visibility_km == 23.0
        assert cond.temperature_K == pytest.approx(288.15)
        assert cond.pressure_hPa == pytest.approx(1013.25)


class TestNoAtmosphere:
    """Tests for NoAtmosphere model."""

    def test_unit_transmission(self) -> None:
        """NoAtmosphere should have unit transmission."""
        model = NoAtmosphere()
        path = PathGeometry(ground_range_m=10000.0)
        conditions = AtmosphereConditions()

        result = model.compute(10.0, path, conditions)

        assert result.transmission == 1.0
        assert result.path_radiance == 0.0

    def test_array_wavelength(self) -> None:
        """Should handle array wavelengths."""
        model = NoAtmosphere()
        path = PathGeometry(ground_range_m=1000.0)
        conditions = AtmosphereConditions()

        wavelengths = np.array([8.0, 10.0, 12.0])
        result = model.compute(wavelengths, path, conditions)

        assert len(result.transmission) == 3
        assert np.all(result.transmission == 1.0)


class TestBeerLambertAtmosphere:
    """Tests for BeerLambertAtmosphere model."""

    def test_transmission_decreases_with_range(self) -> None:
        """Transmission should decrease with path length."""
        model = BeerLambertAtmosphere()
        conditions = AtmosphereConditions()

        path_short = PathGeometry(ground_range_m=1000.0)
        path_long = PathGeometry(ground_range_m=10000.0)

        result_short = model.compute(10.0, path_short, conditions)
        result_long = model.compute(10.0, path_long, conditions)

        assert result_long.transmission < result_short.transmission

    def test_transmission_in_valid_range(self) -> None:
        """Transmission should be between 0 and 1."""
        model = BeerLambertAtmosphere()
        path = PathGeometry(ground_range_m=5000.0)
        conditions = AtmosphereConditions()

        wavelengths = np.linspace(0.5, 14.0, 50)
        result = model.compute(wavelengths, path, conditions)

        assert np.all(result.transmission >= 0)
        assert np.all(result.transmission <= 1)

    def test_visibility_effect(self) -> None:
        """Lower visibility should reduce transmission."""
        model = BeerLambertAtmosphere()
        path = PathGeometry(ground_range_m=5000.0)

        cond_clear = AtmosphereConditions(visibility_km=50.0)
        cond_hazy = AtmosphereConditions(visibility_km=5.0)

        result_clear = model.compute(10.0, path, cond_clear)
        result_hazy = model.compute(10.0, path, cond_hazy)

        assert result_hazy.transmission < result_clear.transmission

    def test_atmospheric_windows(self) -> None:
        """Transmission should be higher in atmospheric windows."""
        model = BeerLambertAtmosphere()
        path = PathGeometry(ground_range_m=5000.0)
        conditions = AtmosphereConditions()

        # LWIR window (10 μm) vs CO2 absorption (4.3 μm)
        result_window = model.compute(10.0, path, conditions)
        result_absorb = model.compute(4.3, path, conditions)

        # Window should have higher transmission
        assert result_window.transmission > result_absorb.transmission

    def test_path_radiance_positive(self) -> None:
        """Path radiance should be non-negative."""
        model = BeerLambertAtmosphere()
        path = PathGeometry(ground_range_m=5000.0)
        conditions = AtmosphereConditions()

        result = model.compute(10.0, path, conditions)

        assert result.path_radiance >= 0


class TestConstantAtmosphere:
    """Tests for ConstantAtmosphere model."""

    def test_constant_values(self) -> None:
        """Should return constant transmission."""
        model = ConstantAtmosphere(transmission=0.75, path_radiance=1.0)
        path = PathGeometry(ground_range_m=1000.0)
        conditions = AtmosphereConditions()

        result = model.compute(10.0, path, conditions)

        assert result.transmission == 0.75
        assert result.path_radiance == 1.0

    def test_range_independent(self) -> None:
        """Constant model should ignore path length."""
        model = ConstantAtmosphere(transmission=0.8)
        conditions = AtmosphereConditions()

        result1 = model.compute(10.0, PathGeometry(ground_range_m=1000.0), conditions)
        result2 = model.compute(10.0, PathGeometry(ground_range_m=10000.0), conditions)

        assert result1.transmission == result2.transmission


class TestHelperFunctions:
    """Tests for atmosphere helper functions."""

    def test_estimate_transmission(self) -> None:
        """Test quick transmission estimate."""
        trans = estimate_transmission(10.0, 5.0, visibility_km=23.0)
        assert 0 < trans < 1

    def test_estimate_band_transmission(self) -> None:
        """Test band-averaged transmission estimate."""
        trans = estimate_band_transmission(BAND_LWIR, 5.0)
        assert 0 < trans < 1

    def test_koschmieder_relation(self) -> None:
        """Test Koschmieder visibility-extinction relation."""
        # Standard visibility 23 km should give ~0.17/km at 550nm
        beta = koschmieder_visibility_to_extinction(23.0)
        assert beta == pytest.approx(3.912 / 23.0)

        # Very clear (100 km)
        beta_clear = koschmieder_visibility_to_extinction(100.0)
        assert beta_clear < beta
