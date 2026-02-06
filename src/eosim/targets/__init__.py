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

Enhanced Features (shapes module):
- Vehicle thermal zones (engine 370K, cabin 320K, wheels 340K, exhaust 400K)
- Humanoid body model (head 308K, hands 305K, torso 295K, legs 298K)
- Aircraft exhaust plume gradients (500K at nozzle falling to 300K)

Example Usage:
--------------
# Example 1: Create a vehicle target with enhanced thermal zones
>>> from eosim.targets import VehicleTarget, TargetRenderer
>>> vehicle = VehicleTarget.sedan(engine_state="running", speed_kmh=60)
>>> signature = vehicle.get_signature(use_enhanced_shape=True)
>>> print(f"Engine zone temp: {signature.max_temperature:.0f}K")

# Example 2: Create an aircraft with exhaust plume gradient
>>> from eosim.targets import AircraftTarget
>>> aircraft = AircraftTarget.fighter_jet(throttle=0.8, afterburner=True)
>>> signature = aircraft.get_signature(aspect_angle_deg=180)  # Rear view
>>> print(f"Plume temp: {signature.max_temperature:.0f}K")

# Example 3: Generate people with anatomical thermal zones
>>> from eosim.targets import PersonTarget
>>> person = PersonTarget.standing(activity="walking")
>>> signature = person.get_signature(use_enhanced_shape=True)
>>> print(f"Head temp: {signature.max_temperature:.0f}K")  # ~308K

# Example 4: Use enhanced shapes directly
>>> from eosim.targets.shapes import create_vehicle_signature, create_person_signature
>>> temp_map, emis_map, mask = create_vehicle_signature("sedan", engine_running=True)
>>> temp_map, emis_map, mask = create_person_signature("standing", "walking")
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
# Enhanced shape classes for realistic thermal signatures
from eosim.targets.shapes import (
    ThermalZone,
    ThermalZoneConfig,
    EnhancedVehicleShape,
    HumanoidShape,
    AircraftWithPlumeShape,
    create_vehicle_signature,
    create_person_signature,
    create_aircraft_signature,
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
    # Enhanced shapes
    "ThermalZone",
    "ThermalZoneConfig",
    "EnhancedVehicleShape",
    "HumanoidShape",
    "AircraftWithPlumeShape",
    "create_vehicle_signature",
    "create_person_signature",
    "create_aircraft_signature",
]
