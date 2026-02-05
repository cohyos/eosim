"""
EOSIM - Electro-Optical/Infrared Simulation Framework

A PC-based, open-source electro-optical/infrared simulation framework that
generates synthetic image/video output from configurable FPA-based passive
sensor models across visible and infrared bands.

Quick Start:
------------
>>> from eosim.examples import example_vehicle_on_road
>>> result = example_vehicle_on_road()
>>> print(f"Generated {result.sensor_type} image at {result.metadata['range_m']}m")

>>> from eosim.pipeline import quick_simulation
>>> import numpy as np
>>> temps = 300 + 10 * np.random.randn(480, 640)
>>> image = quick_simulation(temps, sensor_type="lwir")

Library Usage (Objects, Sensors, Scenarios):
--------------------------------------------
>>> from eosim.library import get_object, get_sensor, create_scenario, run_scenario
>>> # List available objects and sensors
>>> from eosim.library import list_objects, list_sensors
>>> print(list_objects())  # ['f16', 'f35', 'm1_abrams', ...]
>>> print(list_sensors())  # ['mx15', 'mx20', 'toplite_iii', ...]
>>>
>>> # Create and run a scenario
>>> scenario = create_scenario(sensor="mx15", target="f16", range_km=10)
>>> result = run_scenario(scenario)

Video Processing:
-----------------
>>> from eosim.video import process_video, VideoSimulationConfig, VideoProcessingMode
>>> config = VideoSimulationConfig(mode=VideoProcessingMode.THERMAL_ESTIMATE)
>>> result = process_video("input.mp4", config, "output.mp4")

Modules:
--------
- core: Physical constants, spectral utilities
- sensor: FPA, noise models, ADC, metrics
- optics: PSF, MTF, aberrations
- radiance: Planck functions, surface radiance
- atmosphere: Transmission models
- pipeline: Simulation engine and effects
- output: File writers and formats
- video: Video-to-simulation processing
- library: Object library, sensor specs, scenario system
- examples: 15+ example scenarios
"""

__version__ = "0.1.0"

# Core
from eosim.core.constants import PhysicalConstants, CONSTANTS
from eosim.core.spectral import SpectralBand, WavelengthGrid

# Pipeline (main entry points)
from eosim.pipeline.simulation import (
    SimulationPipeline,
    SimulationConfig,
    SimulationMode,
    SceneInput,
    PipelineResult,
    create_pipeline,
    quick_simulation,
)

# Sensor
from eosim.sensor.fpa import DetectorType, FPAGeometry
from eosim.sensor.base import create_sensor_model

__all__ = [
    # Version
    "__version__",
    # Core
    "PhysicalConstants",
    "CONSTANTS",
    "SpectralBand",
    "WavelengthGrid",
    # Pipeline
    "SimulationPipeline",
    "SimulationConfig",
    "SimulationMode",
    "SceneInput",
    "PipelineResult",
    "create_pipeline",
    "quick_simulation",
    # Sensor
    "DetectorType",
    "FPAGeometry",
    "create_sensor_model",
]
