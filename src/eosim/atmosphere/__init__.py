"""Atmosphere module: transmission, path radiance, and scattering models."""

from eosim.atmosphere.base import (
    AtmosphereFidelity,
    PathGeometry,
    AtmosphereConditions,
    AtmosphereResult,
    AtmosphereModel,
    NoAtmosphere,
)
from eosim.atmosphere.simple import (
    BeerLambertAtmosphere,
    ConstantAtmosphere,
    AtmosphericWindowTransmission,
    estimate_transmission,
    estimate_band_transmission,
    koschmieder_visibility_to_extinction,
    extinction_to_visibility,
)

__all__ = [
    # Base classes
    "AtmosphereFidelity",
    "PathGeometry",
    "AtmosphereConditions",
    "AtmosphereResult",
    "AtmosphereModel",
    "NoAtmosphere",
    # Simple models
    "BeerLambertAtmosphere",
    "ConstantAtmosphere",
    "AtmosphericWindowTransmission",
    "estimate_transmission",
    "estimate_band_transmission",
    "koschmieder_visibility_to_extinction",
    "extinction_to_visibility",
]
