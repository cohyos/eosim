"""
EOSIM Target Library - Stage A

Provides detailed thermal and radiometric signature models for common targets
including vehicles, aircraft, people, buildings, and more.

Each target model includes:
- Geometric shape and dimensions
- Temperature distribution (base + internal variation)
- Emissivity maps by material
- Hot spots (engines, exhausts, windows)
- Time-varying effects (engine warm-up, cooling)

Example Usage:
--------------
# Example 1: Create a vehicle target and render its thermal signature
>>> from eosim.targets import VehicleTarget, TargetRenderer
>>> vehicle = VehicleTarget.sedan(engine_state="running")
>>> renderer = TargetRenderer(resolution=(480, 640), gsd_m=0.1)
>>> temp_map, emis_map = renderer.render(vehicle, position=(240, 320))

# Example 2: Create an aircraft with hot exhaust plume
>>> from eosim.targets import AircraftTarget
>>> aircraft = AircraftTarget.fighter_jet(throttle=0.8, altitude_m=5000)
>>> signature = aircraft.get_signature(aspect_angle_deg=45)

# Example 3: Generate a crowd of people with varying temperatures
>>> from eosim.targets import PersonTarget, TargetGroup
>>> crowd = TargetGroup([
...     PersonTarget.standing(activity="walking") for _ in range(10)
... ])
>>> crowd.randomize_positions(area=(100, 100, 400, 500))
"""

from eosim.targets.base import (
    Target,
    TargetSignature,
    TargetRenderer,
    TargetGroup,
    MaterialProperties,
)
from eosim.targets.vehicles import (
    VehicleTarget,
    TruckTarget,
    TankTarget,
)
from eosim.targets.aircraft import (
    AircraftTarget,
    HelicopterTarget,
    DroneTarget,
)
from eosim.targets.people import (
    PersonTarget,
    CrowdGenerator,
)
from eosim.targets.structures import (
    BuildingTarget,
    IndustrialTarget,
    BridgeTarget,
    StorageTankTarget,
)
from eosim.targets.natural import (
    AnimalTarget,
    VegetationTarget,
    WaterBodyTarget,
    TerrainTarget,
)

__all__ = [
    # Base classes
    "Target",
    "TargetSignature",
    "TargetRenderer",
    "TargetGroup",
    "MaterialProperties",
    # Vehicles
    "VehicleTarget",
    "TruckTarget",
    "TankTarget",
    # Aircraft
    "AircraftTarget",
    "HelicopterTarget",
    "DroneTarget",
    # People
    "PersonTarget",
    "CrowdGenerator",
    # Structures
    "BuildingTarget",
    "IndustrialTarget",
    "BridgeTarget",
    "StorageTankTarget",
    # Natural
    "AnimalTarget",
    "VegetationTarget",
    "WaterBodyTarget",
    "TerrainTarget",
]
