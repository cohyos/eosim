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

Example:
    >>> from eosim.studio import Studio
    >>> studio = Studio()
    >>> studio.launch()

    # Or with terrain and weather:
    >>> scene = Scene("Desert Mission")
    >>> scene.set_terrain("mojave_desert", radius_m=10000, detail="medium")
    >>> scene.set_weather("sandstorm", storm_intensity=0.7)
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
]


def launch():
    """Launch EOSIM Studio GUI.

    Example:
        >>> from eosim.studio import launch
        >>> launch()
    """
    from eosim.studio.studio_gui import launch_studio
    launch_studio()
