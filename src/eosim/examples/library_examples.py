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
}


def list_library_examples() -> List[str]:
    """List all library example names."""
    return list(LIBRARY_EXAMPLES.keys())


def get_library_example(name: str, seed: int = 42) -> ExampleResult:
    """Run a library example by name."""
    if name not in LIBRARY_EXAMPLES:
        raise ValueError(f"Unknown example: {name}. Available: {list_library_examples()}")
    return LIBRARY_EXAMPLES[name](seed=seed)
