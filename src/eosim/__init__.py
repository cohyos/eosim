"""
EOSIM - Electro-Optical/Infrared Simulation Framework

A PC-based, open-source electro-optical/infrared simulation framework that
generates synthetic image/video output from configurable FPA-based passive
sensor models across visible and infrared bands.
"""

__version__ = "0.1.0"

from eosim.core.constants import PhysicalConstants
from eosim.core.spectral import SpectralBand, WavelengthGrid

__all__ = [
    "__version__",
    "PhysicalConstants",
    "SpectralBand",
    "WavelengthGrid",
]
