"""
EOSIM Studio - Video Creator Interface for Sensor Simulation.

EOSIM Studio provides a video-production-style interface for creating
simulated sensor imagery. Think of it as a virtual camera system where:

- Scene = The 3D world with objects and terrain
- Camera = The sensor (IR, visible, etc.)
- Timeline = Animation and playback control
- Render = Export to video formats
- Terrain = Geographic terrain with elevation and land cover
- Weather = Atmospheric conditions (clouds, precipitation, fog, storms)

Advanced Sensor Physics (Phase 1-3):
- Radiometry = Planck blackbody, spectral radiance, atmospheric transmission
- Optics = MTF, PSF, diffraction, aberrations
- Sensor = FPA detector physics, quantum efficiency, well capacity
- AGC = Automatic gain control, CLAHE, histogram processing
- Noise = Shot noise, readout noise, 1/f noise, fixed pattern noise
- Thermal = Material properties, diurnal cycles, engine signatures
- Gimbal = Servo dynamics, rate limiting, jitter, stabilization
- Symbology = HUD overlays, track gates, MIL-STD symbols

Example:
    >>> from eosim.studio import Studio
    >>> studio = Studio()
    >>> studio.launch()

    # Or with terrain and weather:
    >>> scene = Scene("Desert Mission")
    >>> scene.set_terrain("mojave_desert", radius_m=10000, detail="medium")
    >>> scene.set_weather("sandstorm", storm_intensity=0.7)

    # Advanced sensor simulation:
    >>> from eosim.studio import Radiometer, FPADetector, OpticalSimulator
    >>> radiometer = Radiometer(band=SpectralBand.LWIR)
    >>> detector = FPADetector(create_hd_cooled_lwir())
"""

from eosim.studio.project import Project, create_new_project, create_demo_project
from eosim.studio.scene import Scene, SceneObject, Position3D, Orientation3D
from eosim.studio.camera import Camera, CameraPreset, CameraPath, LensType, SpectrumMode
from eosim.studio.timeline import Timeline, Keyframe, PlaybackState
from eosim.studio.renderer import Renderer, RenderedFrame
from eosim.studio.terrain import (
    TerrainProvider, TerrainConfig, TerrainData, GeoLocation,
    DetailLevel, LandCoverType, create_terrain
)
from eosim.studio.weather import (
    WeatherSystem, WeatherConditions, WeatherPreset, WEATHER_PRESETS,
    CloudType, PrecipitationType, FogType, StormType, WindSpeed,
    create_weather
)

# Phase 1: Core Sensor Physics
from eosim.studio.radiometry import (
    Radiometer, SpectralBand, SpectralResponse, AtmosphericConditions,
    AtmosphericTransmission, RadiometricImage,
    planck_radiance, integrate_band_radiance, calculate_atmospheric_transmission,
    celsius_to_kelvin, kelvin_to_celsius
)
from eosim.studio.optics import (
    OpticalSystem, OpticalSimulator, SystemMTF, DistortionModel, TurbulenceModel,
    calculate_diffraction_mtf, calculate_detector_mtf, calculate_psf, apply_psf,
    create_diffraction_limited_system, create_aberrated_system
)
from eosim.studio.sensor_physics import (
    FPADetector, FPAConfiguration, DetectorMaterial, DetectorResponse,
    TDIDetector, MicrobolometerDetector,
    create_hd_cooled_mwir, create_hd_cooled_lwir, create_vga_uncooled, create_hd_uncooled
)

# Phase 2: Image Processing
from eosim.studio.agc import (
    AGCProcessor, AGCParameters, AGCMode, PolarityMode,
    NUCProcessor, BadPixelCorrector, HistogramAnalyzer
)
from eosim.studio.noise_models import (
    NoiseGenerator, NoiseParameters, NoiseType,
    SpatialNoiseFilter, TemporalNoiseFilter,
    calculate_snr, estimate_noise_from_image
)
from eosim.studio.thermal_dynamics import (
    ThermalSolver, DiurnalCycleSimulator, VehicleThermalModel, HeatSource,
    ThermalMaterialProperties, MATERIAL_DATABASE, SolarPosition, AmbientConditions,
    create_tank_thermal_model, create_truck_thermal_model, create_aircraft_thermal_model
)

# Phase 3: Platform Integration
from eosim.studio.gimbal import (
    GimbalController, GimbalState, GimbalLimits, GimbalMode, ScanPattern,
    ServoParameters, JitterParameters, TrackState,
    create_flir_turret, create_targeting_pod, create_surveillance_gimbal
)
from eosim.studio.symbology import (
    SymbologyRenderer, SymbolColor, ReticleType, TrackGateType,
    TargetDesignation, SensorStatus, PlatformStatus, GimbalStatus, ScreenPosition,
    create_default_symbology
)

__all__ = [
    # Project
    "Project",
    "create_new_project",
    "create_demo_project",
    # Scene
    "Scene",
    "SceneObject",
    "Position3D",
    "Orientation3D",
    # Camera
    "Camera",
    "CameraPreset",
    "CameraPath",
    "LensType",
    "SpectrumMode",
    # Timeline
    "Timeline",
    "Keyframe",
    "PlaybackState",
    # Renderer
    "Renderer",
    "RenderedFrame",
    # Terrain
    "TerrainProvider",
    "TerrainConfig",
    "TerrainData",
    "GeoLocation",
    "DetailLevel",
    "LandCoverType",
    "create_terrain",
    # Weather
    "WeatherSystem",
    "WeatherConditions",
    "WeatherPreset",
    "WEATHER_PRESETS",
    "CloudType",
    "PrecipitationType",
    "FogType",
    "StormType",
    "WindSpeed",
    "create_weather",
    # Phase 1: Radiometry
    "Radiometer",
    "SpectralBand",
    "SpectralResponse",
    "AtmosphericConditions",
    "AtmosphericTransmission",
    "RadiometricImage",
    "planck_radiance",
    "integrate_band_radiance",
    "calculate_atmospheric_transmission",
    "celsius_to_kelvin",
    "kelvin_to_celsius",
    # Phase 1: Optics
    "OpticalSystem",
    "OpticalSimulator",
    "SystemMTF",
    "DistortionModel",
    "TurbulenceModel",
    "calculate_diffraction_mtf",
    "calculate_detector_mtf",
    "calculate_psf",
    "apply_psf",
    "create_diffraction_limited_system",
    "create_aberrated_system",
    # Phase 1: Sensor Physics
    "FPADetector",
    "FPAConfiguration",
    "DetectorMaterial",
    "DetectorResponse",
    "TDIDetector",
    "MicrobolometerDetector",
    "create_hd_cooled_mwir",
    "create_hd_cooled_lwir",
    "create_vga_uncooled",
    "create_hd_uncooled",
    # Phase 2: AGC
    "AGCProcessor",
    "AGCParameters",
    "AGCMode",
    "PolarityMode",
    "NUCProcessor",
    "BadPixelCorrector",
    "HistogramAnalyzer",
    # Phase 2: Noise
    "NoiseGenerator",
    "NoiseParameters",
    "NoiseType",
    "SpatialNoiseFilter",
    "TemporalNoiseFilter",
    "calculate_snr",
    "estimate_noise_from_image",
    # Phase 2: Thermal Dynamics
    "ThermalSolver",
    "DiurnalCycleSimulator",
    "VehicleThermalModel",
    "HeatSource",
    "ThermalMaterialProperties",
    "MATERIAL_DATABASE",
    "SolarPosition",
    "AmbientConditions",
    "create_tank_thermal_model",
    "create_truck_thermal_model",
    "create_aircraft_thermal_model",
    # Phase 3: Gimbal
    "GimbalController",
    "GimbalState",
    "GimbalLimits",
    "GimbalMode",
    "ScanPattern",
    "ServoParameters",
    "JitterParameters",
    "TrackState",
    "create_flir_turret",
    "create_targeting_pod",
    "create_surveillance_gimbal",
    # Phase 3: Symbology
    "SymbologyRenderer",
    "SymbolColor",
    "ReticleType",
    "TrackGateType",
    "TargetDesignation",
    "SensorStatus",
    "PlatformStatus",
    "GimbalStatus",
    "ScreenPosition",
    "create_default_symbology",
]


def launch():
    """Launch EOSIM Studio GUI.

    Example:
        >>> from eosim.studio import launch
        >>> launch()
    """
    from eosim.studio.studio_gui import launch_studio
    launch_studio()
