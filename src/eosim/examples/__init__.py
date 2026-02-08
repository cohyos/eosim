"""
EOSIM Example Scenarios.

This module provides 25+ comprehensive example scenarios demonstrating
different objects, backgrounds, sensors, and imaging modes.

Basic Examples (10):
--------------------
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

Library Examples (10) - Using Object/Sensor Library:
----------------------------------------------------
11. MX-15 vs F-16 - Air-to-air fighter engagement
12. MX-20 Convoy - Multi-vehicle surveillance
13. Sniper vs Tank - Ground vehicle targeting
14. TopLite Patrol - Helicopter patrol scenario
15. Naval Surveillance - Ship detection
16. SAM Site Detection - SAM launcher detection
17. Night Personnel - Personnel detection at night
18. UAV Tracking - Drone tracking
19. Missile Detection - Cruise missile detection
20. Helicopter Engagement - Attack helicopter targeting

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

from eosim.examples.library_examples import (
    # Library-based examples
    example_mx15_vs_f16,
    example_mx20_convoy,
    example_sniper_vs_tank,
    example_toplite_patrol,
    example_naval_surveillance,
    example_sam_site_detection,
    example_night_personnel,
    example_uav_tracking,
    example_missile_detection,
    example_helicopter_engagement,
    # 6DOF Platform Motion Examples
    example_helicopter_orbit,
    example_fixed_wing_patrol,
    example_ground_vehicle_surveillance,
    example_naval_ship_tracking,
    example_tripod_static,
    # Utility functions
    list_library_examples,
    get_library_example,
)

__all__ = [
    # Basic examples
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
    # Library examples
    "example_mx15_vs_f16",
    "example_mx20_convoy",
    "example_sniper_vs_tank",
    "example_toplite_patrol",
    "example_naval_surveillance",
    "example_sam_site_detection",
    "example_night_personnel",
    "example_uav_tracking",
    "example_missile_detection",
    "example_helicopter_engagement",
    # 6DOF Platform Motion Examples
    "example_helicopter_orbit",
    "example_fixed_wing_patrol",
    "example_ground_vehicle_surveillance",
    "example_naval_ship_tracking",
    "example_tripod_static",
    "list_library_examples",
    "get_library_example",
]
