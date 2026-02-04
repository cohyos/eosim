"""
Unit handling for EOSIM.

Provides lightweight unit-checked quantities for radiometric calculations.
Uses a simple registry pattern for common EO/IR units without heavy dependencies.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union
import numpy as np
from numpy.typing import NDArray


class UnitDimension(Enum):
    """Physical dimensions for unit checking."""

    DIMENSIONLESS = "dimensionless"
    LENGTH = "length"
    MASS = "mass"
    TIME = "time"
    TEMPERATURE = "temperature"
    ANGLE = "angle"
    SOLID_ANGLE = "solid_angle"
    POWER = "power"
    ENERGY = "energy"
    RADIANCE = "radiance"
    IRRADIANCE = "irradiance"
    SPECTRAL_RADIANCE = "spectral_radiance"
    SPECTRAL_IRRADIANCE = "spectral_irradiance"


@dataclass
class Unit:
    """Represents a physical unit with conversion to SI base."""

    symbol: str
    dimension: UnitDimension
    to_si: float = 1.0  # Multiplier to convert to SI base unit
    si_symbol: str = ""  # SI base unit symbol

    def __post_init__(self) -> None:
        if not self.si_symbol:
            self.si_symbol = self.symbol


class UnitRegistry:
    """Registry of common units for EO/IR simulation."""

    def __init__(self) -> None:
        self._units: dict[str, Unit] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default units."""
        # Length units
        self.register(Unit("m", UnitDimension.LENGTH, 1.0, "m"))
        self.register(Unit("km", UnitDimension.LENGTH, 1e3, "m"))
        self.register(Unit("cm", UnitDimension.LENGTH, 1e-2, "m"))
        self.register(Unit("mm", UnitDimension.LENGTH, 1e-3, "m"))
        self.register(Unit("um", UnitDimension.LENGTH, 1e-6, "m"))
        self.register(Unit("nm", UnitDimension.LENGTH, 1e-9, "m"))

        # Temperature units (Kelvin is base, but we support Celsius)
        self.register(Unit("K", UnitDimension.TEMPERATURE, 1.0, "K"))
        # Note: Celsius conversion is not linear, handled specially

        # Time units
        self.register(Unit("s", UnitDimension.TIME, 1.0, "s"))
        self.register(Unit("ms", UnitDimension.TIME, 1e-3, "s"))
        self.register(Unit("us", UnitDimension.TIME, 1e-6, "s"))

        # Angle units
        self.register(Unit("rad", UnitDimension.ANGLE, 1.0, "rad"))
        self.register(Unit("deg", UnitDimension.ANGLE, np.pi / 180, "rad"))
        self.register(Unit("mrad", UnitDimension.ANGLE, 1e-3, "rad"))
        self.register(Unit("urad", UnitDimension.ANGLE, 1e-6, "rad"))

        # Solid angle units
        self.register(Unit("sr", UnitDimension.SOLID_ANGLE, 1.0, "sr"))

        # Power/Energy units
        self.register(Unit("W", UnitDimension.POWER, 1.0, "W"))
        self.register(Unit("mW", UnitDimension.POWER, 1e-3, "W"))
        self.register(Unit("uW", UnitDimension.POWER, 1e-6, "W"))
        self.register(Unit("J", UnitDimension.ENERGY, 1.0, "J"))

        # Radiometric units
        # Radiance: W/(m²·sr)
        self.register(Unit("W/(m^2*sr)", UnitDimension.RADIANCE, 1.0, "W/(m^2*sr)"))
        self.register(Unit("W/m^2/sr", UnitDimension.RADIANCE, 1.0, "W/(m^2*sr)"))

        # Irradiance: W/m²
        self.register(Unit("W/m^2", UnitDimension.IRRADIANCE, 1.0, "W/m^2"))

        # Spectral radiance: W/(m²·sr·μm)
        self.register(
            Unit(
                "W/(m^2*sr*um)",
                UnitDimension.SPECTRAL_RADIANCE,
                1.0,
                "W/(m^2*sr*um)",
            )
        )

        # Spectral irradiance: W/(m²·μm)
        self.register(
            Unit("W/(m^2*um)", UnitDimension.SPECTRAL_IRRADIANCE, 1.0, "W/(m^2*um)")
        )

        # Dimensionless
        self.register(Unit("", UnitDimension.DIMENSIONLESS, 1.0, ""))
        self.register(Unit("1", UnitDimension.DIMENSIONLESS, 1.0, ""))

    def register(self, unit: Unit) -> None:
        """Register a unit in the registry."""
        self._units[unit.symbol] = unit

    def get(self, symbol: str) -> Unit:
        """Get a unit by symbol."""
        if symbol not in self._units:
            raise ValueError(f"Unknown unit: {symbol}")
        return self._units[symbol]

    def __getattr__(self, name: str) -> Unit:
        """Allow attribute-style access to units."""
        if name.startswith("_"):
            raise AttributeError(name)
        # Handle common unit names
        name_map = {
            "meter": "m",
            "meters": "m",
            "kilometer": "km",
            "micrometer": "um",
            "nanometer": "nm",
            "kelvin": "K",
            "second": "s",
            "seconds": "s",
            "millisecond": "ms",
            "microsecond": "us",
            "radian": "rad",
            "radians": "rad",
            "degree": "deg",
            "degrees": "deg",
            "steradian": "sr",
            "steradians": "sr",
            "watt": "W",
            "watts": "W",
            "joule": "J",
            "joules": "J",
        }
        symbol = name_map.get(name, name)
        return self.get(symbol)


# Global unit registry instance
ureg = UnitRegistry()


@dataclass
class Quantity:
    """A physical quantity with magnitude and unit.

    Provides lightweight unit checking for EOSIM without heavy dependencies.
    """

    magnitude: Union[float, NDArray[np.floating]]
    unit: Unit

    def __post_init__(self) -> None:
        """Ensure magnitude is proper type."""
        if isinstance(self.magnitude, (list, tuple)):
            self.magnitude = np.asarray(self.magnitude, dtype=np.float64)

    @classmethod
    def from_unit_str(
        cls,
        magnitude: Union[float, NDArray[np.floating]],
        unit_str: str,
    ) -> "Quantity":
        """Create quantity from unit string."""
        unit = ureg.get(unit_str)
        return cls(magnitude=magnitude, unit=unit)

    def to(self, target_unit: Union[Unit, str]) -> "Quantity":
        """Convert to another unit of the same dimension."""
        if isinstance(target_unit, str):
            target_unit = ureg.get(target_unit)

        if self.unit.dimension != target_unit.dimension:
            raise ValueError(
                f"Cannot convert {self.unit.dimension.value} to "
                f"{target_unit.dimension.value}"
            )

        # Convert through SI base
        si_magnitude = self.magnitude * self.unit.to_si
        new_magnitude = si_magnitude / target_unit.to_si

        return Quantity(magnitude=new_magnitude, unit=target_unit)

    def to_si(self) -> "Quantity":
        """Convert to SI base unit for this dimension."""
        si_unit = Unit(
            symbol=self.unit.si_symbol,
            dimension=self.unit.dimension,
            to_si=1.0,
            si_symbol=self.unit.si_symbol,
        )
        return Quantity(
            magnitude=self.magnitude * self.unit.to_si,
            unit=si_unit,
        )

    @property
    def m(self) -> Union[float, NDArray[np.floating]]:
        """Return just the magnitude (shorthand)."""
        return self.magnitude

    def __repr__(self) -> str:
        if isinstance(self.magnitude, np.ndarray):
            mag_str = f"array(shape={self.magnitude.shape})"
        else:
            mag_str = f"{self.magnitude:.6g}"
        return f"Quantity({mag_str}, {self.unit.symbol})"

    def __add__(self, other: "Quantity") -> "Quantity":
        if not isinstance(other, Quantity):
            raise TypeError("Can only add Quantity to Quantity")
        if self.unit.dimension != other.unit.dimension:
            raise ValueError("Cannot add quantities with different dimensions")
        # Convert other to same unit
        other_converted = other.to(self.unit)
        return Quantity(
            magnitude=self.magnitude + other_converted.magnitude,
            unit=self.unit,
        )

    def __sub__(self, other: "Quantity") -> "Quantity":
        if not isinstance(other, Quantity):
            raise TypeError("Can only subtract Quantity from Quantity")
        if self.unit.dimension != other.unit.dimension:
            raise ValueError("Cannot subtract quantities with different dimensions")
        other_converted = other.to(self.unit)
        return Quantity(
            magnitude=self.magnitude - other_converted.magnitude,
            unit=self.unit,
        )

    def __mul__(self, other: Union[float, int, "Quantity"]) -> "Quantity":
        if isinstance(other, (int, float)):
            return Quantity(magnitude=self.magnitude * other, unit=self.unit)
        elif isinstance(other, Quantity):
            # Simple scalar multiplication - doesn't track compound units
            return Quantity(
                magnitude=self.magnitude * other.magnitude,
                unit=Unit(
                    f"({self.unit.symbol})*({other.unit.symbol})",
                    UnitDimension.DIMENSIONLESS,  # Simplified
                ),
            )
        raise TypeError(f"Cannot multiply Quantity by {type(other)}")

    def __rmul__(self, other: Union[float, int]) -> "Quantity":
        return self.__mul__(other)

    def __truediv__(self, other: Union[float, int, "Quantity"]) -> "Quantity":
        if isinstance(other, (int, float)):
            return Quantity(magnitude=self.magnitude / other, unit=self.unit)
        elif isinstance(other, Quantity):
            return Quantity(
                magnitude=self.magnitude / other.magnitude,
                unit=Unit(
                    f"({self.unit.symbol})/({other.unit.symbol})",
                    UnitDimension.DIMENSIONLESS,
                ),
            )
        raise TypeError(f"Cannot divide Quantity by {type(other)}")


# Convenience functions for creating quantities
def wavelength(value: float, unit: str = "um") -> Quantity:
    """Create a wavelength quantity."""
    return Quantity.from_unit_str(value, unit)


def temperature(value: float, unit: str = "K") -> Quantity:
    """Create a temperature quantity."""
    return Quantity.from_unit_str(value, unit)


def distance(value: float, unit: str = "m") -> Quantity:
    """Create a distance quantity."""
    return Quantity.from_unit_str(value, unit)


def angle(value: float, unit: str = "rad") -> Quantity:
    """Create an angle quantity."""
    return Quantity.from_unit_str(value, unit)


def celsius_to_kelvin(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def kelvin_to_celsius(kelvin: float) -> float:
    """Convert Kelvin to Celsius."""
    return kelvin - 273.15


def fahrenheit_to_kelvin(fahrenheit: float) -> float:
    """Convert Fahrenheit to Kelvin."""
    return (fahrenheit - 32) * 5 / 9 + 273.15


def kelvin_to_fahrenheit(kelvin: float) -> float:
    """Convert Kelvin to Fahrenheit."""
    return (kelvin - 273.15) * 9 / 5 + 32
