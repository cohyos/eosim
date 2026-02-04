"""
EOSIM Example Scenarios.

This module provides 10 comprehensive example scenarios demonstrating
different objects, backgrounds, sensors, and imaging modes.

Examples:
---------
1. Vehicle on Road (LWIR) - Hot vehicle on cool road background
2. Person in Forest (MWIR) - Human target with vegetation clutter
3. Aircraft Against Sky (MWIR) - Fast-moving aircraft with motion blur
4. Ship on Ocean (LWIR) - Maritime surveillance scenario
5. Building Thermal (LWIR) - Structural thermal inspection
6. Wildlife Tracking (MWIR) - Animal detection in natural habitat
7. Industrial Monitoring (LWIR) - Hot machinery detection
8. Night Vision (SWIR) - Low-light surveillance with SWIR
9. Solar Panel Inspection (LWIR) - Defect detection in solar arrays
10. Urban Surveillance (Visible/SWIR) - Multi-spectral urban scene

Each example returns a simulation result with digital image and metadata.
"""

from eosim.examples.scenarios import (
    # Individual examples
    example_vehicle_on_road,
    example_person_in_forest,
    example_aircraft_sky,
    example_ship_ocean,
    example_building_thermal,
    example_wildlife_tracking,
    example_industrial_monitoring,
    example_night_vision_swir,
    example_solar_panel_inspection,
    example_urban_surveillance,
    # Utility functions
    run_all_examples,
    list_examples,
    get_example_by_name,
    # Scene generators
    create_vehicle_scene,
    create_person_scene,
    create_building_scene,
    create_industrial_scene,
)

__all__ = [
    "example_vehicle_on_road",
    "example_person_in_forest",
    "example_aircraft_sky",
    "example_ship_ocean",
    "example_building_thermal",
    "example_wildlife_tracking",
    "example_industrial_monitoring",
    "example_night_vision_swir",
    "example_solar_panel_inspection",
    "example_urban_surveillance",
    "run_all_examples",
    "list_examples",
    "get_example_by_name",
    "create_vehicle_scene",
    "create_person_scene",
    "create_building_scene",
    "create_industrial_scene",
]
