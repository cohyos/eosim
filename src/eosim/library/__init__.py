"""
EOSIM Library Module.

Provides pre-defined objects, sensors, and scenarios for EO/IR simulation.

Components:
- Object Library: 3D models of aircraft, ships, vehicles, missiles, etc.
- Sensor Library: Real-world EO/IR sensor specifications (MX-15, MX-20, TopLite, etc.)
- Scenario System: Pre-configured sensor vs target engagement scenarios
- Platform System: 6DOF sensor platform motion with gimbal control
"""

from eosim.library.objects import (
    ObjectLibrary,
    ObjectCategory,
    Object3D,
    AircraftObject,
    ShipObject,
    VehicleObject,
    MissileObject,
    LauncherObject,
    PersonObject,
    get_object,
    list_objects,
    list_categories,
)

from eosim.library.sensors import (
    SensorLibrary,
    SensorSpec,
    SensorType,
    get_sensor,
    list_sensors,
    create_sensor_from_spec,
)

from eosim.library.scenarios import (
    Scenario,
    ScenarioBuilder,
    EngagementGeometry,
    TargetTrack,
    EnvironmentType,
    BackgroundType,
    EnvironmentConditions,
    Position3D,
    Velocity3D,
    ScenarioOutput,
    create_scenario,
    run_scenario,
    list_predefined_scenarios,
    get_predefined_scenario,
)

from eosim.library.platform import (
    # Platform types and modes
    PlatformType,
    GimbalMode,
    TrackingMode,
    # State classes
    Orientation3D,
    AngularVelocity3D,
    PlatformState,
    GimbalState,
    # Configuration
    GimbalLimits,
    StabilizationParams,
    TrackingParams,
    # Controllers
    GimbalController,
    TrajectoryGenerator,
    Waypoint,
    # Main platform class
    SensorPlatform,
    create_platform,
    # Motion effects
    apply_motion_blur,
    apply_jitter,
    apply_platform_motion_effects,
)

__all__ = [
    # Objects
    "ObjectLibrary",
    "ObjectCategory",
    "Object3D",
    "AircraftObject",
    "ShipObject",
    "VehicleObject",
    "MissileObject",
    "LauncherObject",
    "PersonObject",
    "get_object",
    "list_objects",
    "list_categories",
    # Sensors
    "SensorLibrary",
    "SensorSpec",
    "SensorType",
    "get_sensor",
    "list_sensors",
    "create_sensor_from_spec",
    # Scenarios
    "Scenario",
    "ScenarioBuilder",
    "EngagementGeometry",
    "TargetTrack",
    "EnvironmentType",
    "BackgroundType",
    "EnvironmentConditions",
    "Position3D",
    "Velocity3D",
    "ScenarioOutput",
    "create_scenario",
    "run_scenario",
    "list_predefined_scenarios",
    "get_predefined_scenario",
    # Platform (6DOF)
    "PlatformType",
    "GimbalMode",
    "TrackingMode",
    "Orientation3D",
    "AngularVelocity3D",
    "PlatformState",
    "GimbalState",
    "GimbalLimits",
    "StabilizationParams",
    "TrackingParams",
    "GimbalController",
    "TrajectoryGenerator",
    "Waypoint",
    "SensorPlatform",
    "create_platform",
    # Motion effects
    "apply_motion_blur",
    "apply_jitter",
    "apply_platform_motion_effects",
]
