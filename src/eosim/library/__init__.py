"""
EOSIM Library Module.

Provides pre-defined objects, sensors, and scenarios for EO/IR simulation.

Components:
- Object Library: 3D models of aircraft, ships, vehicles, missiles, etc.
- Sensor Library: Real-world EO/IR sensor specifications (MX-15, MX-20, TopLite, etc.)
- Scenario System: Pre-configured sensor vs target engagement scenarios
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
]
