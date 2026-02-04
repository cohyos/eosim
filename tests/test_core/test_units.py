"""Tests for core.units module."""

import pytest
import numpy as np

from eosim.core.units import (
    Unit,
    UnitDimension,
    UnitRegistry,
    Quantity,
    ureg,
    wavelength,
    temperature,
    distance,
    angle,
    celsius_to_kelvin,
    kelvin_to_celsius,
    fahrenheit_to_kelvin,
    kelvin_to_fahrenheit,
)


class TestUnitRegistry:
    """Tests for UnitRegistry."""

    def test_get_meter(self) -> None:
        """Test getting meter unit."""
        m = ureg.get("m")
        assert m.symbol == "m"
        assert m.dimension == UnitDimension.LENGTH
        assert m.to_si == 1.0

    def test_get_kilometer(self) -> None:
        """Test kilometer conversion factor."""
        km = ureg.get("km")
        assert km.to_si == 1e3

    def test_get_micrometer(self) -> None:
        """Test micrometer conversion factor."""
        um = ureg.get("um")
        assert um.to_si == 1e-6

    def test_attribute_access(self) -> None:
        """Test attribute-style access to units."""
        m = ureg.meter
        assert m.symbol == "m"
        deg = ureg.degree
        assert deg.symbol == "deg"

    def test_unknown_unit_error(self) -> None:
        """Test error for unknown unit."""
        with pytest.raises(ValueError, match="Unknown unit"):
            ureg.get("unknown_unit")


class TestQuantity:
    """Tests for Quantity."""

    def test_create_quantity(self) -> None:
        """Test creating a quantity."""
        q = Quantity(magnitude=10.0, unit=ureg.get("m"))
        assert q.magnitude == 10.0
        assert q.unit.symbol == "m"

    def test_from_unit_str(self) -> None:
        """Test creating quantity from unit string."""
        q = Quantity.from_unit_str(5.0, "km")
        assert q.magnitude == 5.0
        assert q.unit.symbol == "km"

    def test_unit_conversion(self) -> None:
        """Test unit conversion."""
        q = Quantity.from_unit_str(1.0, "km")
        q_m = q.to("m")
        assert q_m.magnitude == pytest.approx(1000.0)
        assert q_m.unit.symbol == "m"

    def test_to_si(self) -> None:
        """Test conversion to SI."""
        q = Quantity.from_unit_str(5.0, "um")
        q_si = q.to_si()
        assert q_si.magnitude == pytest.approx(5e-6)

    def test_magnitude_shorthand(self) -> None:
        """Test .m property for magnitude."""
        q = Quantity.from_unit_str(10.0, "K")
        assert q.m == 10.0

    def test_add_same_dimension(self) -> None:
        """Test adding quantities of same dimension."""
        q1 = Quantity.from_unit_str(1.0, "km")
        q2 = Quantity.from_unit_str(500.0, "m")
        result = q1 + q2
        assert result.magnitude == pytest.approx(1.5)
        assert result.unit.symbol == "km"

    def test_subtract_same_dimension(self) -> None:
        """Test subtracting quantities of same dimension."""
        q1 = Quantity.from_unit_str(1.0, "km")
        q2 = Quantity.from_unit_str(200.0, "m")
        result = q1 - q2
        assert result.magnitude == pytest.approx(0.8)

    def test_multiply_by_scalar(self) -> None:
        """Test multiplying quantity by scalar."""
        q = Quantity.from_unit_str(5.0, "m")
        result = q * 2.0
        assert result.magnitude == pytest.approx(10.0)
        assert result.unit.symbol == "m"

    def test_divide_by_scalar(self) -> None:
        """Test dividing quantity by scalar."""
        q = Quantity.from_unit_str(10.0, "m")
        result = q / 2.0
        assert result.magnitude == pytest.approx(5.0)

    def test_incompatible_dimension_error(self) -> None:
        """Test error when adding incompatible dimensions."""
        q1 = Quantity.from_unit_str(1.0, "m")
        q2 = Quantity.from_unit_str(1.0, "K")
        with pytest.raises(ValueError, match="Cannot add"):
            q1 + q2

    def test_array_magnitude(self) -> None:
        """Test quantity with array magnitude."""
        q = Quantity(magnitude=np.array([1.0, 2.0, 3.0]), unit=ureg.get("um"))
        assert len(q.magnitude) == 3
        q_m = q.to("m")
        assert q_m.magnitude[0] == pytest.approx(1e-6)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_wavelength_function(self) -> None:
        """Test wavelength convenience function."""
        w = wavelength(10.0, "um")
        assert w.magnitude == 10.0
        assert w.unit.dimension == UnitDimension.LENGTH

    def test_temperature_function(self) -> None:
        """Test temperature convenience function."""
        t = temperature(300.0, "K")
        assert t.magnitude == 300.0
        assert t.unit.dimension == UnitDimension.TEMPERATURE

    def test_distance_function(self) -> None:
        """Test distance convenience function."""
        d = distance(1000.0, "m")
        assert d.magnitude == 1000.0

    def test_angle_function(self) -> None:
        """Test angle convenience function."""
        a = angle(45.0, "deg")
        assert a.magnitude == 45.0
        a_rad = a.to("rad")
        assert a_rad.magnitude == pytest.approx(np.pi / 4)


class TestTemperatureConversions:
    """Tests for temperature conversion functions."""

    def test_celsius_to_kelvin(self) -> None:
        """Test Celsius to Kelvin conversion."""
        assert celsius_to_kelvin(0.0) == pytest.approx(273.15)
        assert celsius_to_kelvin(100.0) == pytest.approx(373.15)
        assert celsius_to_kelvin(-273.15) == pytest.approx(0.0)

    def test_kelvin_to_celsius(self) -> None:
        """Test Kelvin to Celsius conversion."""
        assert kelvin_to_celsius(273.15) == pytest.approx(0.0)
        assert kelvin_to_celsius(373.15) == pytest.approx(100.0)

    def test_fahrenheit_to_kelvin(self) -> None:
        """Test Fahrenheit to Kelvin conversion."""
        assert fahrenheit_to_kelvin(32.0) == pytest.approx(273.15)
        assert fahrenheit_to_kelvin(212.0) == pytest.approx(373.15)

    def test_kelvin_to_fahrenheit(self) -> None:
        """Test Kelvin to Fahrenheit conversion."""
        assert kelvin_to_fahrenheit(273.15) == pytest.approx(32.0)
        assert kelvin_to_fahrenheit(373.15) == pytest.approx(212.0)

    def test_round_trip_celsius(self) -> None:
        """Test Celsius round-trip conversion."""
        original = 25.0
        k = celsius_to_kelvin(original)
        back = kelvin_to_celsius(k)
        assert back == pytest.approx(original)

    def test_round_trip_fahrenheit(self) -> None:
        """Test Fahrenheit round-trip conversion."""
        original = 98.6
        k = fahrenheit_to_kelvin(original)
        back = kelvin_to_fahrenheit(k)
        assert back == pytest.approx(original)
