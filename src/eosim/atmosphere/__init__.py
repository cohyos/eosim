"""
EOSIM Atmosphere Module - Stage B Enhanced

Provides atmospheric transmission, path radiance, scattering, and adjacency models.

Example Usage:
--------------
# Example 1: Beer-Lambert transmission with different visibility
>>> from eosim.atmosphere import BeerLambertAtmosphere, PathGeometry, AtmosphereConditions
>>> model = BeerLambertAtmosphere()
>>> path = PathGeometry(ground_range_m=5000, altitude_end_m=1000)
>>> clear = AtmosphereConditions(visibility_km=50)
>>> result = model.compute(10.0, path, clear)
>>> print(f"Transmission: {result.transmission:.3f}")

# Example 2: LUT-based atmosphere with aerosol types
>>> from eosim.atmosphere import LUTAtmosphere
>>> rural_model = LUTAtmosphere(aerosol_type="rural")
>>> urban_model = LUTAtmosphere(aerosol_type="urban")
>>> result_rural = rural_model.compute(0.55, path, AtmosphereConditions())
>>> result_urban = urban_model.compute(0.55, path, AtmosphereConditions())

# Example 3: Turbulence effects
>>> from eosim.atmosphere import TurbulenceModel
>>> turb = TurbulenceModel(cn2_ground=1e-14)
>>> seeing = turb.seeing_blur_fwhm(wavelength_um=10.0, path_length_m=5000)
>>> print(f"Seeing blur: {seeing:.1f} arcsec")
"""

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
from eosim.atmosphere.lut_atmosphere import (
    LUTAtmosphere,
    LUTAtmosphereData,
    AerosolModel,
    TurbulenceModel,
)
from eosim.atmosphere.adjacency import (
    AdjacencyModel,
    AdjacencyResult,
    ContrastTransmission,
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
    # LUT-based models (Stage B)
    "LUTAtmosphere",
    "LUTAtmosphereData",
    "AerosolModel",
    "TurbulenceModel",
    # Adjacency effects (Stage B)
    "AdjacencyModel",
    "AdjacencyResult",
    "ContrastTransmission",
]
