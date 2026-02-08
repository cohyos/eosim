"""
EOSIM Library Examples.

Demonstrates usage of the object library, sensor library, and scenario system.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
from numpy.typing import NDArray

from eosim.examples.scenarios import ExampleResult


def example_mx15_vs_f16(seed: int = 42) -> ExampleResult:
    """
    Example: MX-15 Targeting F-16 Fighter

    Demonstrates using the sensor and object library to create
    a realistic air-to-air engagement scenario.

    Sensor: L3Harris MX-15 (MWIR, 3-5um)
    Target: F-16 Fighting Falcon at 10km
    Aspect: Side view (90 degrees)

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.library import create_scenario, run_scenario

    scenario = create_scenario(
        sensor="mx15",
        target="f16",
        range_km=10.0,
        aspect_deg=90.0,
        altitude_m=5000.0,
        environment="clear_day",
        background="sky",
        name="MX-15 vs F-16",
        seed=seed,
    )

    result = run_scenario(scenario)

    return ExampleResult(
        name="mx15_vs_f16",
        description="MX-15 sensor targeting F-16 at 10km",
        digital_image=result.digital_image,
        temperature_map=result.temperature_map,
        sensor_type="MWIR",
        metadata={
            **result.metadata,
            "detection_metrics": result.detection_metrics,
        },
    )


def example_mx20_convoy(seed: int = 42) -> ExampleResult:
    """
    Example: MX-20 Surveilling Vehicle Convoy

    Multi-target scenario with MX-20 sensor observing
    a convoy of military vehicles.

    Sensor: L3Harris MX-20 (MWIR, 3-5um)
    Targets: M1 Abrams, 2x Military Trucks
    Range: 8km

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.library import ScenarioBuilder, run_scenario, EnvironmentType, BackgroundType

    scenario = (
        ScenarioBuilder()
        .set_name("MX-20 Convoy Surveillance")
        .set_sensor("mx20", altitude_m=3000)
        .add_target("m1_abrams", position_km=(8.0, 0.0, 0.0), heading_deg=90)
        .add_target("military_truck", position_km=(8.1, 0.1, 0.0), heading_deg=90)
        .add_target("military_truck", position_km=(8.2, 0.0, 0.0), heading_deg=90)
        .set_environment(env_type=EnvironmentType.DESERT_DAY)
        .set_background(BackgroundType.DESERT)
        .set_seed(seed)
        .build()
    )

    result = run_scenario(scenario)

    return ExampleResult(
        name="mx20_convoy",
        description="MX-20 sensor observing 3-vehicle convoy at 8km",
        digital_image=result.digital_image,
        temperature_map=result.temperature_map,
        sensor_type="MWIR",
        metadata={
            **result.metadata,
            "detection_metrics": result.detection_metrics,
        },
    )


def example_sniper_vs_tank(seed: int = 42) -> ExampleResult:
    """
    Example: Sniper ATP vs T-90 Tank

    Air-to-ground targeting scenario with Sniper ATP
    pod targeting a T-90 main battle tank.

    Sensor: Lockheed Martin Sniper ATP (MWIR)
    Target: T-90 Main Battle Tank
    Range: 12km
    Aspect: 45 degrees (front-quarter)

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.library import create_scenario, run_scenario

    scenario = create_scenario(
        sensor="sniper_atp",
        target="t90",
        range_km=12.0,
        aspect_deg=45.0,
        altitude_m=6000.0,
        environment="clear_day",
        background="terrain",
        name="Sniper ATP vs T-90",
        seed=seed,
    )

    result = run_scenario(scenario)

    return ExampleResult(
        name="sniper_vs_tank",
        description="Sniper ATP targeting T-90 tank at 12km",
        digital_image=result.digital_image,
        temperature_map=result.temperature_map,
        sensor_type="MWIR",
        metadata={
            **result.metadata,
            "detection_metrics": result.detection_metrics,
        },
    )


def example_toplite_patrol(seed: int = 42) -> ExampleResult:
    """
    Example: TopLite III Helicopter Patrol

    Low-altitude helicopter patrol scenario with TopLite III
    sensor detecting a technical vehicle.

    Sensor: Rafael TopLite III (MWIR)
    Target: Technical (armed pickup truck)
    Range: 3km

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.library import create_scenario, run_scenario

    scenario = create_scenario(
        sensor="toplite_iii",
        target="pickup_technical",
        range_km=3.0,
        aspect_deg=60.0,
        altitude_m=500.0,
        environment="clear_day",
        background="urban",
        name="TopLite Patrol",
        seed=seed,
    )

    result = run_scenario(scenario)

    return ExampleResult(
        name="toplite_patrol",
        description="TopLite III helicopter patrol at 500m altitude",
        digital_image=result.digital_image,
        temperature_map=result.temperature_map,
        sensor_type="MWIR",
        metadata={
            **result.metadata,
            "detection_metrics": result.detection_metrics,
        },
    )


def example_naval_surveillance(seed: int = 42) -> ExampleResult:
    """
    Example: Naval Surveillance with MX-25

    Long-range naval surveillance scenario with MX-25
    detecting surface vessels.

    Sensor: L3Harris MX-25 (MWIR)
    Targets: Frigate, Patrol Boat
    Range: 25km

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.library import ScenarioBuilder, run_scenario, EnvironmentType, BackgroundType

    scenario = (
        ScenarioBuilder()
        .set_name("Naval Surveillance")
        .set_sensor("mx25", altitude_m=3000)
        .add_target("frigate", position_km=(25.0, 0.0, 0.0), heading_deg=270)
        .add_target("patrol_boat", position_km=(22.0, 2.0, 0.0), heading_deg=290)
        .set_environment(env_type=EnvironmentType.MARITIME)
        .set_background(BackgroundType.WATER)
        .set_seed(seed)
        .build()
    )

    result = run_scenario(scenario)

    return ExampleResult(
        name="naval_surveillance",
        description="MX-25 naval surveillance at 25km",
        digital_image=result.digital_image,
        temperature_map=result.temperature_map,
        sensor_type="MWIR",
        metadata={
            **result.metadata,
            "detection_metrics": result.detection_metrics,
        },
    )


def example_sam_site_detection(seed: int = 42) -> ExampleResult:
    """
    Example: SAM Site Detection

    Detecting a SAM launcher site using FLIR sensor.

    Sensor: FLIR Star SAFIRE 380-HD (MWIR)
    Target: S-400 SAM Launcher
    Range: 15km

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.library import create_scenario, run_scenario

    scenario = create_scenario(
        sensor="star_safire_380hd",
        target="s400",
        range_km=15.0,
        aspect_deg=75.0,
        altitude_m=4000.0,
        environment="clear_day",
        background="terrain",
        name="SAM Site Detection",
        seed=seed,
    )

    result = run_scenario(scenario)

    return ExampleResult(
        name="sam_site_detection",
        description="Star SAFIRE detecting S-400 launcher at 15km",
        digital_image=result.digital_image,
        temperature_map=result.temperature_map,
        sensor_type="MWIR",
        metadata={
            **result.metadata,
            "detection_metrics": result.detection_metrics,
        },
    )


def example_night_personnel(seed: int = 42) -> ExampleResult:
    """
    Example: Night Personnel Detection

    Night-time personnel detection using uncooled thermal.

    Sensor: Generic Uncooled Microbolometer (LWIR)
    Target: Standing Soldier
    Range: 1km

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.library import create_scenario, run_scenario

    scenario = create_scenario(
        sensor="generic_uncooled",
        target="soldier_standing",
        range_km=1.0,
        aspect_deg=0.0,
        altitude_m=100.0,
        environment="clear_night",
        background="terrain",
        name="Night Personnel Detection",
        seed=seed,
    )

    result = run_scenario(scenario)

    return ExampleResult(
        name="night_personnel",
        description="Uncooled thermal detecting personnel at 1km",
        digital_image=result.digital_image,
        temperature_map=result.temperature_map,
        sensor_type="LWIR",
        metadata={
            **result.metadata,
            "detection_metrics": result.detection_metrics,
        },
    )


def example_uav_tracking(seed: int = 42) -> ExampleResult:
    """
    Example: UAV Tracking

    Tracking a medium-altitude UAV with MX-15.

    Sensor: L3Harris MX-15 (MWIR)
    Target: Bayraktar TB2 UAV
    Range: 8km

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.library import create_scenario, run_scenario

    scenario = create_scenario(
        sensor="mx15",
        target="bayraktar_tb2",
        range_km=8.0,
        aspect_deg=120.0,  # Rear-quarter
        altitude_m=2000.0,
        environment="clear_day",
        background="sky",
        name="UAV Tracking",
        seed=seed,
    )

    result = run_scenario(scenario)

    return ExampleResult(
        name="uav_tracking",
        description="MX-15 tracking Bayraktar TB2 at 8km",
        digital_image=result.digital_image,
        temperature_map=result.temperature_map,
        sensor_type="MWIR",
        metadata={
            **result.metadata,
            "detection_metrics": result.detection_metrics,
        },
    )


def example_missile_detection(seed: int = 42) -> ExampleResult:
    """
    Example: Missile Detection

    Detecting an in-flight cruise missile.

    Sensor: L3Harris MX-20 (MWIR)
    Target: Tomahawk Cruise Missile
    Range: 20km

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.library import ScenarioBuilder, run_scenario, EnvironmentType, BackgroundType

    scenario = (
        ScenarioBuilder()
        .set_name("Missile Detection")
        .set_sensor("mx20", altitude_m=5000)
        .add_target(
            "tomahawk",
            range_km=20.0,
            aspect_deg=30.0,
            velocity_ms=(200, 0, 0),
            heading_deg=270,
        )
        .set_environment(env_type=EnvironmentType.CLEAR_DAY)
        .set_background(BackgroundType.SKY)
        .set_seed(seed)
        .build()
    )

    result = run_scenario(scenario)

    return ExampleResult(
        name="missile_detection",
        description="MX-20 detecting Tomahawk cruise missile at 20km",
        digital_image=result.digital_image,
        temperature_map=result.temperature_map,
        sensor_type="MWIR",
        metadata={
            **result.metadata,
            "detection_metrics": result.detection_metrics,
        },
    )


def example_helicopter_engagement(seed: int = 42) -> ExampleResult:
    """
    Example: Helicopter Engagement

    Air-to-air engagement against attack helicopter.

    Sensor: L3Harris MX-15 (MWIR)
    Target: AH-64 Apache
    Range: 6km

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.library import create_scenario, run_scenario

    scenario = create_scenario(
        sensor="mx15",
        target="ah64",
        range_km=6.0,
        aspect_deg=45.0,
        altitude_m=2000.0,
        environment="clear_day",
        background="sky",
        name="Helicopter Engagement",
        seed=seed,
    )

    result = run_scenario(scenario)

    return ExampleResult(
        name="helicopter_engagement",
        description="MX-15 engaging AH-64 Apache at 6km",
        digital_image=result.digital_image,
        temperature_map=result.temperature_map,
        sensor_type="MWIR",
        metadata={
            **result.metadata,
            "detection_metrics": result.detection_metrics,
        },
    )


# =============================================================================
# 6DOF Platform Motion Examples
# =============================================================================

def example_helicopter_orbit(seed: int = 42) -> ExampleResult:
    """
    Example: Helicopter Orbiting Target with 6DOF Motion

    Demonstrates 6DOF sensor platform capabilities with a helicopter
    orbiting around a ground target while the gimbal tracks it.

    Platform: Rotary wing at 500m altitude
    Sensor: TopLite III (MWIR)
    Target: T-90 Tank
    Orbit: 2km radius around target

    Returns:
        ExampleResult with simulated thermal image including motion effects
    """
    from eosim.library import (
        ScenarioBuilder, run_scenario,
        EnvironmentType, BackgroundType, Position3D,
    )

    # Target at origin
    target_pos = Position3D(0, 0, 0, "m")

    scenario = (
        ScenarioBuilder()
        .set_name("Helicopter Orbit - 6DOF Demo")
        .set_sensor("toplite_iii", position=Position3D(2000, 0, 500, "m"))
        .add_target("t90", position=target_pos, heading_deg=45)
        .set_environment(env_type=EnvironmentType.CLEAR_DAY)
        .set_background(BackgroundType.TERRAIN)
        # Configure 6DOF platform
        .set_platform(
            platform_type="rotary_wing",
            speed_ms=40.0,
            heading_deg=90.0,
        )
        .set_gimbal_track(target_idx=0)  # Track the T-90
        .set_orbit(center=target_pos, radius_m=2000.0)
        .set_seed(seed)
        .build()
    )

    result = run_scenario(scenario, verbose=True)

    return ExampleResult(
        name="helicopter_orbit",
        description="Helicopter orbiting T-90 with gimbal tracking at 2km",
        digital_image=result.digital_image,
        temperature_map=result.temperature_map,
        sensor_type="MWIR",
        metadata={
            **result.metadata,
            "detection_metrics": result.detection_metrics,
            "platform_motion": "6DOF orbit with target tracking",
        },
    )


def example_fixed_wing_patrol(seed: int = 42) -> ExampleResult:
    """
    Example: Fixed-Wing Patrol with Waypoint Navigation

    Demonstrates waypoint-based trajectory with a fixed-wing aircraft
    flying a patrol pattern over a convoy.

    Platform: Fixed wing at 3000m altitude, 150 m/s
    Sensor: MX-15 (MWIR)
    Targets: Military convoy (3 vehicles)

    Returns:
        ExampleResult with simulated thermal image including motion effects
    """
    from eosim.library import (
        ScenarioBuilder, run_scenario,
        EnvironmentType, BackgroundType, Position3D,
    )

    scenario = (
        ScenarioBuilder()
        .set_name("Fixed-Wing Patrol - Waypoint Demo")
        .set_sensor("mx15", position=Position3D(0, -5000, 3000, "m"))
        .add_target("m1_abrams", position_km=(0, 0, 0), heading_deg=0, name="lead")
        .add_target("military_truck", position_km=(0.1, 0, 0), heading_deg=0, name="truck1")
        .add_target("military_truck", position_km=(0.2, 0, 0), heading_deg=0, name="truck2")
        .set_environment(env_type=EnvironmentType.DESERT_DAY)
        .set_background(BackgroundType.DESERT)
        # Configure 6DOF platform - fixed wing
        .set_platform(
            platform_type="fixed_wing",
            speed_ms=150.0,
            heading_deg=0.0,
            orientation_deg=(0.0, -5.0, 0.0),  # Slight nose-down pitch
        )
        .set_gimbal_track(target_idx=0)  # Track lead vehicle
        # Add waypoints for patrol pattern
        .add_waypoint(Position3D(0, 0, 3000, "m"), time_s=0.0, heading_deg=0)
        .add_waypoint(Position3D(0, 5000, 3000, "m"), time_s=30.0, heading_deg=0)
        .add_waypoint(Position3D(3000, 5000, 3000, "m"), time_s=50.0, heading_deg=90)
        .add_waypoint(Position3D(3000, 0, 3000, "m"), time_s=80.0, heading_deg=180)
        .set_seed(seed)
        .build()
    )

    result = run_scenario(scenario, verbose=True)

    return ExampleResult(
        name="fixed_wing_patrol",
        description="Fixed-wing patrol over convoy with waypoints",
        digital_image=result.digital_image,
        temperature_map=result.temperature_map,
        sensor_type="MWIR",
        metadata={
            **result.metadata,
            "detection_metrics": result.detection_metrics,
            "platform_motion": "6DOF waypoint patrol",
        },
    )


def example_ground_vehicle_surveillance(seed: int = 42) -> ExampleResult:
    """
    Example: Ground Vehicle with Stabilized Sensor

    Demonstrates ground vehicle platform with stabilized gimbal
    observing personnel.

    Platform: Ground vehicle (stationary with engine vibration)
    Sensor: Catherine XP (LWIR)
    Target: Personnel group

    Returns:
        ExampleResult with simulated thermal image including jitter effects
    """
    from eosim.library import (
        ScenarioBuilder, run_scenario,
        EnvironmentType, BackgroundType, Position3D,
    )

    scenario = (
        ScenarioBuilder()
        .set_name("Ground Vehicle Surveillance - Stabilization Demo")
        .set_sensor("catherine_xp", position=Position3D(0, 0, 3, "m"))  # 3m height
        .add_target("soldier_standing", position_km=(0.5, 0, 0), name="target1")
        .add_target("soldier_prone", position_km=(0.52, 0.02, 0), name="target2")
        .add_target("civilian", position_km=(0.48, -0.01, 0), name="target3")
        .set_environment(env_type=EnvironmentType.CLEAR_NIGHT)
        .set_background(BackgroundType.TERRAIN)
        # Configure 6DOF platform - ground vehicle (stationary)
        .set_platform(
            platform_type="ground_vehicle",
            speed_ms=0.0,  # Stationary
            heading_deg=0.0,
        )
        .set_gimbal_track(target_idx=0)
        .set_seed(seed)
        .build()
    )

    result = run_scenario(scenario, verbose=True)

    return ExampleResult(
        name="ground_vehicle_surveillance",
        description="Ground vehicle observing personnel with stabilized sensor",
        digital_image=result.digital_image,
        temperature_map=result.temperature_map,
        sensor_type="LWIR",
        metadata={
            **result.metadata,
            "detection_metrics": result.detection_metrics,
            "platform_motion": "6DOF ground vehicle (stabilized)",
        },
    )


def example_naval_ship_tracking(seed: int = 42) -> ExampleResult:
    """
    Example: Naval Platform Ship Tracking

    Demonstrates naval platform motion (ship roll/pitch) while
    tracking another vessel.

    Platform: Naval surface vessel with ship motion
    Sensor: MX-20 (MWIR)
    Target: Patrol boat

    Returns:
        ExampleResult with simulated thermal image including ship motion effects
    """
    from eosim.library import (
        ScenarioBuilder, run_scenario,
        EnvironmentType, BackgroundType, Position3D,
    )

    scenario = (
        ScenarioBuilder()
        .set_name("Naval Ship Tracking - Sea Motion Demo")
        .set_sensor("mx20", position=Position3D(0, 0, 20, "m"))  # 20m mast height
        .add_target("patrol_boat", position_km=(5, 0, 0), heading_deg=270,
                   velocity_ms=(10, 0, 0))  # Moving target
        .set_environment(env_type=EnvironmentType.MARITIME)
        .set_background(BackgroundType.WATER)
        # Configure 6DOF platform - naval surface
        .set_platform(
            platform_type="naval_surface",
            speed_ms=8.0,  # ~15 knots
            heading_deg=45.0,
            orientation_deg=(2.0, 1.0, 45.0),  # Slight roll/pitch from waves
        )
        .set_gimbal_track(target_idx=0)
        .set_seed(seed)
        .build()
    )

    result = run_scenario(scenario, verbose=True)

    return ExampleResult(
        name="naval_ship_tracking",
        description="Naval platform tracking patrol boat with ship motion",
        digital_image=result.digital_image,
        temperature_map=result.temperature_map,
        sensor_type="MWIR",
        metadata={
            **result.metadata,
            "detection_metrics": result.detection_metrics,
            "platform_motion": "6DOF naval with sea state effects",
        },
    )


def example_tripod_static(seed: int = 42) -> ExampleResult:
    """
    Example: Tripod-Mounted Static Sensor

    Demonstrates a tripod-mounted sensor with minimal motion effects
    for comparison against moving platforms.

    Platform: Fixed tripod (minimal vibration)
    Sensor: Sophie MF (LWIR)
    Target: Vehicle at close range

    Returns:
        ExampleResult with simulated thermal image (minimal motion effects)
    """
    from eosim.library import (
        ScenarioBuilder, run_scenario,
        EnvironmentType, BackgroundType, Position3D,
    )

    scenario = (
        ScenarioBuilder()
        .set_name("Tripod Static - Reference Demo")
        .set_sensor("sophie_mf", position=Position3D(0, 0, 1.5, "m"))  # 1.5m tripod
        .add_target("civilian_car", position_km=(0.3, 0, 0), heading_deg=90)
        .set_environment(env_type=EnvironmentType.CLEAR_DAY)
        .set_background(BackgroundType.URBAN)
        # Configure 6DOF platform - tripod (minimal motion)
        .set_platform(
            platform_type="tripod",
            speed_ms=0.0,
            heading_deg=0.0,
        )
        .set_gimbal_track(target_idx=0)
        .set_seed(seed)
        .build()
    )

    result = run_scenario(scenario, verbose=True)

    return ExampleResult(
        name="tripod_static",
        description="Tripod-mounted sensor with minimal motion effects",
        digital_image=result.digital_image,
        temperature_map=result.temperature_map,
        sensor_type="LWIR",
        metadata={
            **result.metadata,
            "detection_metrics": result.detection_metrics,
            "platform_motion": "6DOF tripod (reference)",
        },
    )


# Registry of library examples
LIBRARY_EXAMPLES = {
    "mx15_vs_f16": example_mx15_vs_f16,
    "mx20_convoy": example_mx20_convoy,
    "sniper_vs_tank": example_sniper_vs_tank,
    "toplite_patrol": example_toplite_patrol,
    "naval_surveillance": example_naval_surveillance,
    "sam_site_detection": example_sam_site_detection,
    "night_personnel": example_night_personnel,
    "uav_tracking": example_uav_tracking,
    "missile_detection": example_missile_detection,
    "helicopter_engagement": example_helicopter_engagement,
    # 6DOF Platform Motion Examples
    "helicopter_orbit": example_helicopter_orbit,
    "fixed_wing_patrol": example_fixed_wing_patrol,
    "ground_vehicle_surveillance": example_ground_vehicle_surveillance,
    "naval_ship_tracking": example_naval_ship_tracking,
    "tripod_static": example_tripod_static,
}


def list_library_examples() -> List[str]:
    """List all library example names."""
    return list(LIBRARY_EXAMPLES.keys())


def get_library_example(name: str, seed: int = 42) -> ExampleResult:
    """Run a library example by name."""
    if name not in LIBRARY_EXAMPLES:
        raise ValueError(f"Unknown example: {name}. Available: {list_library_examples()}")
    return LIBRARY_EXAMPLES[name](seed=seed)
