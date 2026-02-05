"""
EOSIM Example Scenarios.

Provides 10 comprehensive simulation examples covering:
- Different target types (vehicles, people, aircraft, ships, buildings)
- Various backgrounds (road, forest, sky, ocean, urban)
- Multiple sensor types (LWIR, MWIR, SWIR, Visible)
- Different imaging conditions (static, motion, day, night)
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
import numpy as np
from numpy.typing import NDArray


@dataclass
class ExampleResult:
    """Result from an example simulation.

    Attributes:
        name: Example name
        description: Brief description
        digital_image: Output digital image [DN]
        temperature_map: Input temperature map [K]
        sensor_type: Sensor type used
        metadata: Additional metadata
    """
    name: str
    description: str
    digital_image: NDArray[np.integer]
    temperature_map: NDArray[np.floating]
    sensor_type: str
    metadata: Dict[str, Any]


# ============================================================================
# Scene Generation Utilities
# ============================================================================

def create_background(
    shape: tuple[int, int],
    temperature: float,
    noise_std: float = 1.0,
    seed: Optional[int] = None,
) -> NDArray[np.floating]:
    """Create uniform background with noise.

    Args:
        shape: Image shape (height, width)
        temperature: Mean temperature [K]
        noise_std: Temperature noise standard deviation
        seed: Random seed

    Returns:
        Temperature map
    """
    rng = np.random.default_rng(seed)
    return temperature + rng.normal(0, noise_std, shape)


def add_rectangle(
    image: NDArray,
    center: tuple[int, int],
    size: tuple[int, int],
    value: float,
) -> NDArray:
    """Add rectangular region to image.

    Args:
        image: Input image
        center: (y, x) center position
        size: (height, width) of rectangle
        value: Value to set

    Returns:
        Modified image
    """
    result = image.copy()
    cy, cx = center
    h, w = size
    y0 = max(0, cy - h // 2)
    y1 = min(image.shape[0], cy + h // 2)
    x0 = max(0, cx - w // 2)
    x1 = min(image.shape[1], cx + w // 2)
    result[y0:y1, x0:x1] = value
    return result


def add_circle(
    image: NDArray,
    center: tuple[int, int],
    radius: int,
    value: float,
) -> NDArray:
    """Add circular region to image.

    Args:
        image: Input image
        center: (y, x) center position
        radius: Circle radius
        value: Value to set

    Returns:
        Modified image
    """
    result = image.copy()
    cy, cx = center
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    mask = (y - cy)**2 + (x - cx)**2 <= radius**2
    result[mask] = value
    return result


def add_gradient(
    image: NDArray,
    direction: str = "horizontal",
    delta: float = 5.0,
) -> NDArray:
    """Add temperature gradient to image.

    Args:
        image: Input image
        direction: "horizontal" or "vertical"
        delta: Temperature change across image

    Returns:
        Modified image
    """
    result = image.copy()
    h, w = image.shape
    if direction == "horizontal":
        gradient = np.linspace(-delta/2, delta/2, w)
        result += gradient[np.newaxis, :]
    else:
        gradient = np.linspace(-delta/2, delta/2, h)
        result += gradient[:, np.newaxis]
    return result


# ============================================================================
# Scene Generators
# ============================================================================

def create_vehicle_scene(
    shape: tuple[int, int] = (480, 640),
    road_temp: float = 295.0,
    vehicle_temp: float = 320.0,
    ambient_temp: float = 290.0,
    seed: Optional[int] = None,
) -> tuple[NDArray, NDArray]:
    """Create scene with vehicle on road.

    Args:
        shape: Image shape
        road_temp: Road surface temperature [K]
        vehicle_temp: Vehicle temperature [K]
        ambient_temp: Ambient/vegetation temperature [K]
        seed: Random seed

    Returns:
        Tuple of (temperature_map, emissivity_map)
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    # Background - road and surrounding vegetation
    temp = np.full(shape, ambient_temp, dtype=np.float64)
    emissivity = np.full(shape, 0.95, dtype=np.float64)

    # Road (center horizontal strip)
    road_y0, road_y1 = h // 3, 2 * h // 3
    temp[road_y0:road_y1, :] = road_temp + rng.normal(0, 2, (road_y1 - road_y0, w))
    emissivity[road_y0:road_y1, :] = 0.92  # Asphalt

    # Vehicle (hot rectangle)
    veh_center = (h // 2, w // 2)
    veh_size = (h // 6, w // 4)
    temp = add_rectangle(temp, veh_center, veh_size, vehicle_temp)
    emissivity = add_rectangle(emissivity, veh_center, veh_size, 0.85)

    # Hot engine area
    engine_center = (veh_center[0], veh_center[1] - veh_size[1] // 3)
    temp = add_rectangle(temp, engine_center, (veh_size[0] // 2, veh_size[1] // 3),
                        vehicle_temp + 30)

    # Hot exhaust
    exhaust_center = (veh_center[0], veh_center[1] + veh_size[1] // 2)
    temp = add_circle(temp, exhaust_center, 10, vehicle_temp + 50)

    return temp, emissivity


def create_person_scene(
    shape: tuple[int, int] = (480, 640),
    body_temp: float = 305.0,
    background_temp: float = 290.0,
    vegetation_variation: float = 3.0,
    seed: Optional[int] = None,
) -> tuple[NDArray, NDArray]:
    """Create scene with person in vegetation.

    Args:
        shape: Image shape
        body_temp: Human body surface temperature [K]
        background_temp: Vegetation temperature [K]
        vegetation_variation: Temperature variation in background
        seed: Random seed

    Returns:
        Tuple of (temperature_map, emissivity_map)
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    # Vegetation background with texture
    temp = background_temp + rng.normal(0, vegetation_variation, shape)
    # Add low-frequency variation
    from scipy.ndimage import gaussian_filter
    variation = gaussian_filter(rng.normal(0, 5, shape), sigma=20)
    temp += variation
    emissivity = np.full(shape, 0.97, dtype=np.float64)  # Vegetation

    # Person (torso + head)
    center_y, center_x = h // 2, w // 2

    # Torso
    torso_h, torso_w = h // 4, h // 8
    temp = add_rectangle(temp, (center_y, center_x), (torso_h, torso_w), body_temp)
    emissivity = add_rectangle(emissivity, (center_y, center_x), (torso_h, torso_w), 0.98)

    # Head (warmer)
    head_center = (center_y - torso_h // 2 - h // 16, center_x)
    temp = add_circle(temp, head_center, h // 20, body_temp + 3)
    emissivity = add_circle(emissivity, head_center, h // 20, 0.98)

    return temp, emissivity


def create_building_scene(
    shape: tuple[int, int] = (480, 640),
    wall_temp: float = 295.0,
    window_temp: float = 288.0,
    hvac_temp: float = 310.0,
    seed: Optional[int] = None,
) -> tuple[NDArray, NDArray]:
    """Create building thermal inspection scene.

    Args:
        shape: Image shape
        wall_temp: Building wall temperature [K]
        window_temp: Window temperature [K]
        hvac_temp: HVAC equipment temperature [K]
        seed: Random seed

    Returns:
        Tuple of (temperature_map, emissivity_map)
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    # Building facade (sky at top)
    temp = np.full(shape, 270.0, dtype=np.float64)  # Cold sky
    emissivity = np.full(shape, 0.95, dtype=np.float64)

    # Building (lower 80%)
    building_top = h // 5
    temp[building_top:, :] = wall_temp + rng.normal(0, 1, (h - building_top, w))
    emissivity[building_top:, :] = 0.92  # Concrete/brick

    # Windows (grid pattern)
    win_h, win_w = h // 10, w // 12
    for row in range(3):
        for col in range(5):
            wy = building_top + h // 8 + row * (h // 4)
            wx = w // 8 + col * (w // 6)
            temp = add_rectangle(temp, (wy, wx), (win_h, win_w), window_temp)
            emissivity = add_rectangle(emissivity, (wy, wx), (win_h, win_w), 0.85)

    # Rooftop HVAC unit
    hvac_center = (building_top + h // 20, w // 2)
    temp = add_rectangle(temp, hvac_center, (h // 15, w // 8), hvac_temp)

    # Heat leak at corner (thermal bridge)
    leak_center = (h * 2 // 3, w // 10)
    temp = add_circle(temp, leak_center, h // 15, wall_temp + 8)

    return temp, emissivity


def create_industrial_scene(
    shape: tuple[int, int] = (480, 640),
    background_temp: float = 295.0,
    hot_equipment_temp: float = 350.0,
    pipe_temp: float = 330.0,
    seed: Optional[int] = None,
) -> tuple[NDArray, NDArray]:
    """Create industrial monitoring scene.

    Args:
        shape: Image shape
        background_temp: Background structure temperature [K]
        hot_equipment_temp: Hot machinery temperature [K]
        pipe_temp: Hot pipe temperature [K]
        seed: Random seed

    Returns:
        Tuple of (temperature_map, emissivity_map)
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    # Industrial background
    temp = background_temp + rng.normal(0, 2, shape)
    emissivity = np.full(shape, 0.90, dtype=np.float64)

    # Large hot tank/vessel
    tank_center = (h // 2, w // 4)
    temp = add_circle(temp, tank_center, h // 4, hot_equipment_temp)
    emissivity = add_circle(emissivity, tank_center, h // 4, 0.85)

    # Hot pipes (horizontal lines)
    for i in range(3):
        pipe_y = h // 4 + i * (h // 4)
        temp[pipe_y - 5:pipe_y + 5, w // 3:] = pipe_temp + rng.normal(0, 3, (10, w - w // 3))
        emissivity[pipe_y - 5:pipe_y + 5, w // 3:] = 0.80

    # Hot spot (potential problem area)
    hotspot = (h * 3 // 4, w * 3 // 4)
    temp = add_circle(temp, hotspot, 15, hot_equipment_temp + 50)

    return temp, emissivity


# ============================================================================
# Example Simulations
# ============================================================================

def example_vehicle_on_road(seed: int = 42) -> ExampleResult:
    """
    Example 1: Vehicle on Road (LWIR)

    Simulates a thermal infrared view of a hot vehicle (engine running)
    on a road surface with surrounding vegetation. Demonstrates thermal
    contrast between vehicle, road, and vegetation.

    Sensor: LWIR HgCdTe (8-12 μm)
    Scene: Vehicle at 320K on 295K road, 290K vegetation
    Range: 500m

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.pipeline import create_pipeline, SceneInput

    # Create scene
    temp_map, emissivity_map = create_vehicle_scene(seed=seed)

    # Setup pipeline
    pipeline = create_pipeline(sensor_type="lwir", seed=seed)

    # Run simulation
    scene = SceneInput(
        temperature_map=temp_map,
        emissivity_map=emissivity_map,
        background_temperature=270.0,
    )
    result = pipeline.run(scene, range_m=500.0)

    return ExampleResult(
        name="vehicle_on_road",
        description="Hot vehicle on road (LWIR at 500m)",
        digital_image=result.digital_image,
        temperature_map=temp_map,
        sensor_type="LWIR",
        metadata={
            "range_m": 500.0,
            "band": "8-12 μm",
            "target": "vehicle",
            "background": "road/vegetation",
        },
    )


def example_person_in_forest(seed: int = 42) -> ExampleResult:
    """
    Example 2: Person in Forest (MWIR)

    Simulates detection of a human target in vegetation clutter.
    Human body heat (~32°C surface) against cooler vegetation.

    Sensor: MWIR HgCdTe (3-5 μm)
    Scene: Person at 305K in 290K vegetation
    Range: 200m

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.pipeline import create_pipeline, SceneInput

    # Create scene
    temp_map, emissivity_map = create_person_scene(seed=seed)

    # Setup pipeline
    pipeline = create_pipeline(sensor_type="mwir", seed=seed)

    # Run simulation
    scene = SceneInput(
        temperature_map=temp_map,
        emissivity_map=emissivity_map,
    )
    result = pipeline.run(scene, range_m=200.0)

    return ExampleResult(
        name="person_in_forest",
        description="Human detection in vegetation (MWIR at 200m)",
        digital_image=result.digital_image,
        temperature_map=temp_map,
        sensor_type="MWIR",
        metadata={
            "range_m": 200.0,
            "band": "3-5 μm",
            "target": "person",
            "background": "forest/vegetation",
        },
    )


def example_aircraft_sky(seed: int = 42) -> ExampleResult:
    """
    Example 3: Aircraft Against Sky (MWIR)

    Fast-moving aircraft against cold sky background.
    Demonstrates motion blur from platform/target movement.

    Sensor: MWIR InSb (3-5 μm)
    Scene: Hot aircraft (350K exhaust) against 230K sky
    Range: 5000m with motion blur

    Returns:
        ExampleResult with simulated image including motion blur
    """
    from eosim.pipeline import create_pipeline, SceneInput, MotionBlurParams, apply_motion_blur

    rng = np.random.default_rng(seed)
    h, w = 480, 640

    # Cold sky background
    temp_map = np.full((h, w), 230.0, dtype=np.float64) + rng.normal(0, 2, (h, w))
    emissivity_map = np.full((h, w), 0.95, dtype=np.float64)

    # Aircraft (simplified as hot elongated shape)
    ac_center = (h // 2, w // 2)
    ac_body_temp = 320.0
    ac_exhaust_temp = 380.0

    # Fuselage
    temp_map = add_rectangle(temp_map, ac_center, (30, 150), ac_body_temp)
    emissivity_map = add_rectangle(emissivity_map, ac_center, (30, 150), 0.85)

    # Wings
    temp_map = add_rectangle(temp_map, ac_center, (100, 40), ac_body_temp - 10)

    # Exhaust (very hot)
    exhaust_pos = (ac_center[0], ac_center[1] + 80)
    temp_map = add_circle(temp_map, exhaust_pos, 15, ac_exhaust_temp)

    # Setup pipeline
    pipeline = create_pipeline(sensor_type="mwir", seed=seed)

    scene = SceneInput(
        temperature_map=temp_map,
        emissivity_map=emissivity_map,
        background_temperature=230.0,
    )
    result = pipeline.run(scene, range_m=5000.0)

    # Apply motion blur
    motion_params = MotionBlurParams(velocity_pixels_per_frame=8.0, angle_deg=5.0)
    blurred = apply_motion_blur(result.digital_image.astype(float), motion_params)
    digital_image = np.clip(blurred, 0, 16383).astype(np.uint16)

    return ExampleResult(
        name="aircraft_sky",
        description="Aircraft against sky with motion blur (MWIR at 5km)",
        digital_image=digital_image,
        temperature_map=temp_map,
        sensor_type="MWIR",
        metadata={
            "range_m": 5000.0,
            "band": "3-5 μm",
            "target": "aircraft",
            "background": "sky",
            "motion_blur": True,
        },
    )


def example_ship_ocean(seed: int = 42) -> ExampleResult:
    """
    Example 4: Ship on Ocean (LWIR)

    Maritime surveillance scenario with vessel on water.
    Water has distinct thermal properties from ship structure.

    Sensor: LWIR HgCdTe (8-12 μm)
    Scene: Ship at 305K on 288K water
    Range: 2000m

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.pipeline import create_pipeline, SceneInput

    rng = np.random.default_rng(seed)
    h, w = 480, 640

    # Ocean background (cold water with slight variation)
    water_temp = 288.0
    temp_map = water_temp + rng.normal(0, 0.5, (h, w))

    # Add wave-like texture
    for freq in [0.02, 0.05, 0.1]:
        x = np.arange(w)
        wave = 0.5 * np.sin(2 * np.pi * freq * x + rng.uniform(0, 2*np.pi))
        temp_map += wave[np.newaxis, :]

    emissivity_map = np.full((h, w), 0.96, dtype=np.float64)  # Water

    # Ship (simplified hull shape)
    ship_temp = 305.0
    ship_center = (h // 2, w // 2)

    # Hull
    hull_h, hull_w = 60, 200
    temp_map = add_rectangle(temp_map, ship_center, (hull_h, hull_w), ship_temp)
    emissivity_map = add_rectangle(emissivity_map, ship_center, (hull_h, hull_w), 0.90)

    # Superstructure (warmer)
    super_center = (ship_center[0] - 20, ship_center[1] - 30)
    temp_map = add_rectangle(temp_map, super_center, (40, 60), ship_temp + 5)

    # Funnel (hot)
    funnel_center = (ship_center[0] - 40, ship_center[1])
    temp_map = add_rectangle(temp_map, funnel_center, (30, 20), ship_temp + 30)

    # Setup pipeline
    pipeline = create_pipeline(sensor_type="lwir", seed=seed)

    scene = SceneInput(
        temperature_map=temp_map,
        emissivity_map=emissivity_map,
    )
    result = pipeline.run(scene, range_m=2000.0)

    return ExampleResult(
        name="ship_ocean",
        description="Ship on ocean (LWIR at 2km)",
        digital_image=result.digital_image,
        temperature_map=temp_map,
        sensor_type="LWIR",
        metadata={
            "range_m": 2000.0,
            "band": "8-12 μm",
            "target": "ship",
            "background": "ocean",
        },
    )


def example_building_thermal(seed: int = 42) -> ExampleResult:
    """
    Example 5: Building Thermal Inspection (LWIR)

    Structural thermal inspection showing heat leaks,
    windows, and HVAC equipment.

    Sensor: LWIR Microbolometer (8-14 μm)
    Scene: Building facade with thermal features
    Range: 50m (close inspection)

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.pipeline import create_pipeline, SceneInput
    from eosim.sensor.fpa import DetectorType

    # Create scene
    temp_map, emissivity_map = create_building_scene(seed=seed)

    # Setup pipeline with microbolometer
    pipeline = create_pipeline(sensor_type="lwir", seed=seed)
    # Override with microbolometer for building inspection
    from eosim.sensor.base import create_sensor_model
    pipeline.sensor_model = create_sensor_model(
        detector_type=DetectorType.MICROBOLOMETER,
        spectral_band_um=(8.0, 14.0),
        seed=seed,
    )

    scene = SceneInput(
        temperature_map=temp_map,
        emissivity_map=emissivity_map,
    )
    result = pipeline.run(scene, range_m=50.0)

    return ExampleResult(
        name="building_thermal",
        description="Building thermal inspection (LWIR at 50m)",
        digital_image=result.digital_image,
        temperature_map=temp_map,
        sensor_type="LWIR_Microbolometer",
        metadata={
            "range_m": 50.0,
            "band": "8-14 μm",
            "target": "building",
            "background": "sky",
            "application": "thermal_inspection",
        },
    )


def example_wildlife_tracking(seed: int = 42) -> ExampleResult:
    """
    Example 6: Wildlife Tracking (MWIR)

    Animal detection in natural habitat for wildlife
    monitoring and conservation.

    Sensor: MWIR HgCdTe (3-5 μm)
    Scene: Deer-sized animal in forest clearing
    Range: 300m

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.pipeline import create_pipeline, SceneInput

    rng = np.random.default_rng(seed)
    h, w = 480, 640

    # Forest/grassland background
    background_temp = 288.0
    temp_map = background_temp + rng.normal(0, 3, (h, w))

    # Add vegetation texture
    from scipy.ndimage import gaussian_filter
    texture = gaussian_filter(rng.normal(0, 4, (h, w)), sigma=15)
    temp_map += texture

    emissivity_map = np.full((h, w), 0.96, dtype=np.float64)

    # Animal (warm body)
    animal_temp = 308.0  # Body surface temp
    animal_center = (h // 2 + 50, w // 2)

    # Body (elliptical)
    body_h, body_w = 40, 80
    temp_map = add_rectangle(temp_map, animal_center, (body_h, body_w), animal_temp)
    emissivity_map = add_rectangle(emissivity_map, animal_center, (body_h, body_w), 0.98)

    # Head
    head_center = (animal_center[0] - 10, animal_center[1] + body_w // 2 + 15)
    temp_map = add_circle(temp_map, head_center, 15, animal_temp + 2)

    # Legs (slightly cooler)
    for dx in [-25, 25]:
        leg_center = (animal_center[0] + 30, animal_center[1] + dx)
        temp_map = add_rectangle(temp_map, leg_center, (30, 8), animal_temp - 5)

    # Setup pipeline
    pipeline = create_pipeline(sensor_type="mwir", seed=seed)

    scene = SceneInput(
        temperature_map=temp_map,
        emissivity_map=emissivity_map,
    )
    result = pipeline.run(scene, range_m=300.0)

    return ExampleResult(
        name="wildlife_tracking",
        description="Wildlife detection in forest (MWIR at 300m)",
        digital_image=result.digital_image,
        temperature_map=temp_map,
        sensor_type="MWIR",
        metadata={
            "range_m": 300.0,
            "band": "3-5 μm",
            "target": "wildlife",
            "background": "forest",
            "application": "conservation",
        },
    )


def example_industrial_monitoring(seed: int = 42) -> ExampleResult:
    """
    Example 7: Industrial Monitoring (LWIR)

    Condition monitoring of industrial equipment showing
    hot machinery, pipes, and potential fault detection.

    Sensor: LWIR HgCdTe (8-12 μm)
    Scene: Industrial facility with hot equipment
    Range: 100m

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.pipeline import create_pipeline, SceneInput

    # Create scene
    temp_map, emissivity_map = create_industrial_scene(seed=seed)

    # Setup pipeline
    pipeline = create_pipeline(sensor_type="lwir", seed=seed)

    scene = SceneInput(
        temperature_map=temp_map,
        emissivity_map=emissivity_map,
    )
    result = pipeline.run(scene, range_m=100.0)

    return ExampleResult(
        name="industrial_monitoring",
        description="Industrial equipment monitoring (LWIR at 100m)",
        digital_image=result.digital_image,
        temperature_map=temp_map,
        sensor_type="LWIR",
        metadata={
            "range_m": 100.0,
            "band": "8-12 μm",
            "target": "industrial_equipment",
            "background": "facility",
            "application": "condition_monitoring",
        },
    )


def example_night_vision_swir(seed: int = 42) -> ExampleResult:
    """
    Example 8: Night Vision (SWIR)

    Low-light surveillance using SWIR imaging.
    SWIR can see through glass and in low light.
    Uses reflected illumination (moonlight, airglow, artificial light).

    Sensor: InGaAs SWIR (0.9-1.7 μm)
    Scene: Urban area at night
    Range: 100m

    Returns:
        ExampleResult with simulated SWIR image
    """
    from eosim.pipeline import create_pipeline, SceneInput

    rng = np.random.default_rng(seed)
    h, w = 480, 640

    # SWIR night scene uses reflected illumination, not thermal emission
    # Radiance is in W/(m²·sr) - typical night SWIR scene ~0.001-0.1 W/(m²·sr)
    # Background radiance from moonlight/airglow
    background_radiance = 0.01  # W/(m²·sr)
    radiance_map = background_radiance * (1 + 0.1 * rng.normal(0, 1, (h, w)))
    radiance_map = np.maximum(radiance_map, 0)

    # Building (lower reflectance)
    building_region = (slice(h//4, 3*h//4), slice(w//4, 3*w//4))
    radiance_map[building_region] *= 0.5

    # Windows (illuminated from inside - high radiance)
    for wy in range(h//4 + 30, 3*h//4 - 30, 60):
        for wx in range(w//4 + 40, 3*w//4 - 40, 80):
            radiance_map = add_rectangle(radiance_map, (wy, wx), (25, 35), 0.1)

    # Street lights (bright)
    for lx in [w//6, w//2, 5*w//6]:
        radiance_map = add_circle(radiance_map, (3*h//4 + 30, lx), 20, 0.5)

    # Person (moderate reflectance)
    person_center = (3*h//4, w//3)
    radiance_map = add_rectangle(radiance_map, person_center, (50, 20), 0.02)

    # Create temperature map for metadata (not used for radiance-based input)
    temp_map = 290.0 * np.ones((h, w))  # Placeholder

    # Setup pipeline
    pipeline = create_pipeline(sensor_type="swir", seed=seed)

    scene = SceneInput(
        radiance_map=radiance_map,
    )
    result = pipeline.run(scene, range_m=100.0)

    return ExampleResult(
        name="night_vision_swir",
        description="Night surveillance scene (SWIR at 100m)",
        digital_image=result.digital_image,
        temperature_map=temp_map,
        sensor_type="SWIR",
        metadata={
            "range_m": 100.0,
            "band": "0.9-1.7 μm",
            "target": "urban_scene",
            "background": "night",
            "application": "surveillance",
        },
    )


def example_solar_panel_inspection(seed: int = 42) -> ExampleResult:
    """
    Example 9: Solar Panel Inspection (LWIR)

    Defect detection in solar photovoltaic arrays.
    Hot cells indicate faults or degradation.

    Sensor: LWIR HgCdTe (8-12 μm)
    Scene: Solar panel array with defective cells
    Range: 30m (drone inspection)

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.pipeline import create_pipeline, SceneInput

    rng = np.random.default_rng(seed)
    h, w = 480, 640

    # Background (roof/ground)
    background_temp = 305.0
    temp_map = np.full((h, w), background_temp, dtype=np.float64)
    emissivity_map = np.full((h, w), 0.90, dtype=np.float64)

    # Solar panel array
    panel_temp = 320.0  # Normal operating temp
    cell_h, cell_w = 30, 50
    gap = 5

    # Create grid of cells
    for row in range(8):
        for col in range(10):
            cy = 50 + row * (cell_h + gap)
            cx = 70 + col * (cell_w + gap)

            # Normal cell
            cell_temp = panel_temp + rng.normal(0, 2)

            # Some cells are defective (hot spots)
            if rng.random() < 0.1:  # 10% defect rate
                cell_temp += rng.uniform(15, 40)

            temp_map = add_rectangle(temp_map, (cy, cx), (cell_h, cell_w), cell_temp)
            emissivity_map = add_rectangle(emissivity_map, (cy, cx), (cell_h, cell_w), 0.85)

    # Add frame (cooler metal)
    frame_temp = panel_temp - 5

    # Setup pipeline
    pipeline = create_pipeline(sensor_type="lwir", seed=seed)

    scene = SceneInput(
        temperature_map=temp_map,
        emissivity_map=emissivity_map,
    )
    result = pipeline.run(scene, range_m=30.0)

    return ExampleResult(
        name="solar_panel_inspection",
        description="Solar panel defect detection (LWIR at 30m)",
        digital_image=result.digital_image,
        temperature_map=temp_map,
        sensor_type="LWIR",
        metadata={
            "range_m": 30.0,
            "band": "8-12 μm",
            "target": "solar_panels",
            "background": "roof",
            "application": "defect_detection",
        },
    )


def example_urban_surveillance(seed: int = 42) -> ExampleResult:
    """
    Example 10: Urban Surveillance (Visible/SWIR fusion)

    Multi-sensor urban monitoring combining visible
    and thermal information.

    Sensor: Si CMOS visible (0.4-0.7 μm)
    Scene: Urban street scene
    Range: 200m

    Returns:
        ExampleResult with simulated visible image
    """
    from eosim.pipeline import create_pipeline, SceneInput, VignetteParams, apply_vignetting

    rng = np.random.default_rng(seed)
    h, w = 480, 640

    # Visible scene - reflectance based
    # Using temperature as proxy for radiance/reflectance
    base_level = 1000.0  # Arbitrary units

    # Sky (bright)
    temp_map = np.full((h, w), base_level * 1.5, dtype=np.float64)

    # Ground/road (medium)
    temp_map[h//2:, :] = base_level * 0.8 + rng.normal(0, 20, (h//2, w))
    emissivity_map = np.full((h, w), 0.5, dtype=np.float64)

    # Buildings
    for bx, bw, bh in [(100, 150, 200), (350, 120, 180), (520, 100, 220)]:
        by = h // 2 - bh
        temp_map[by:h//2, bx:bx+bw] = base_level * 0.6 + rng.normal(0, 10, (bh, bw))
        emissivity_map[by:h//2, bx:bx+bw] = 0.4

        # Windows
        for wy in range(by + 20, h//2 - 20, 40):
            for wx in range(bx + 15, bx + bw - 15, 30):
                temp_map = add_rectangle(temp_map, (wy, wx), (20, 15), base_level * 0.3)

    # Vehicles
    for vx in [180, 400]:
        vy = h * 3 // 4
        temp_map = add_rectangle(temp_map, (vy, vx), (30, 60), base_level * 0.7)

    # People
    for px in [250, 320, 480]:
        py = h * 3 // 4 - 20
        temp_map = add_rectangle(temp_map, (py, px), (50, 15), base_level * 0.5)

    # Setup pipeline
    pipeline = create_pipeline(sensor_type="visible", seed=seed)

    scene = SceneInput(
        temperature_map=temp_map,
        emissivity_map=emissivity_map,
    )
    result = pipeline.run(scene, range_m=200.0)

    # Apply vignetting (common in visible cameras)
    vignette_params = VignetteParams(strength=0.2)
    vignetted = apply_vignetting(result.digital_image.astype(float), vignette_params)
    digital_image = np.clip(vignetted, 0, 65535).astype(np.uint16)

    return ExampleResult(
        name="urban_surveillance",
        description="Urban street scene (Visible at 200m)",
        digital_image=digital_image,
        temperature_map=temp_map,
        sensor_type="Visible",
        metadata={
            "range_m": 200.0,
            "band": "0.4-0.7 μm",
            "target": "urban_scene",
            "background": "city",
            "application": "surveillance",
        },
    )


# ============================================================================
# Realistic Examples (with proper thermal signatures)
# ============================================================================

def example_f16_500m(seed: int = 42) -> ExampleResult:
    """
    Example 11: F-16 Fighter at 500m Range (MWIR)

    Realistic F-16 thermal signature at close range with:
    - Hot exhaust nozzle and plume
    - Aerodynamically heated leading edges
    - Cold airframe skin at altitude
    - Detailed thermal features visible

    Sensor: MWIR HgCdTe (3-5 μm) - better for hot targets
    Scene: F-16 at 500m range, 1km altitude
    Background: Cold sky (~230K)

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.pipeline import create_pipeline, SceneInput
    from eosim.examples.realistic_scenes import create_f16_at_range

    # Create realistic F-16 scene at 500m
    temp_map, emissivity_map, scene_meta = create_f16_at_range(
        shape=(480, 640),
        range_km=0.5,  # 500 meters
        altitude_km=1.0,  # Lower altitude for close range
        seed=seed,
    )

    # Setup pipeline (MWIR better for detecting hot exhaust)
    pipeline = create_pipeline(sensor_type="mwir", seed=seed)

    scene = SceneInput(
        temperature_map=temp_map,
        emissivity_map=emissivity_map,
        background_temperature=230.0,  # Cold sky
    )
    result = pipeline.run(scene, range_m=500.0)

    return ExampleResult(
        name="f16_500m",
        description="F-16 fighter jet at 500m range (MWIR)",
        digital_image=result.digital_image,
        temperature_map=temp_map,
        sensor_type="MWIR",
        metadata={
            "range_m": 500.0,
            "band": "3-5 μm",
            "target": "F-16",
            "aspect": "side",
            "altitude_km": 1.0,
            "background": "cold_sky",
            **scene_meta,
        },
    )


def example_realistic_vehicle(seed: int = 42) -> ExampleResult:
    """
    Example 12: Realistic Vehicle Thermal Signature (LWIR)

    Improved vehicle thermal model with:
    - Temperature gradients across surfaces
    - Hot engine compartment
    - Hot exhaust
    - Cool windows (reflecting sky)
    - Warm tires (friction heating)
    - Textured background

    Sensor: LWIR HgCdTe (8-12 μm)
    Scene: Vehicle at 500m range
    Background: Road and vegetation

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.pipeline import create_pipeline, SceneInput
    from eosim.examples.realistic_scenes import create_realistic_vehicle, VehicleThermalParams

    params = VehicleThermalParams(
        body_temp_K=320.0,  # Hot from sun
        engine_temp_K=365.0,
        exhaust_temp_K=410.0,
        tire_temp_K=325.0,
        window_temp_K=275.0,  # Reflects cold sky
        road_temp_K=310.0,
        vegetation_temp_K=295.0,
    )

    temp_map, emissivity_map = create_realistic_vehicle(
        shape=(480, 640),
        params=params,
        vehicle_type="sedan",
        seed=seed,
    )

    pipeline = create_pipeline(sensor_type="lwir", seed=seed)

    scene = SceneInput(
        temperature_map=temp_map,
        emissivity_map=emissivity_map,
        background_temperature=270.0,
    )
    result = pipeline.run(scene, range_m=500.0)

    return ExampleResult(
        name="realistic_vehicle",
        description="Realistic vehicle with thermal gradients (LWIR at 500m)",
        digital_image=result.digital_image,
        temperature_map=temp_map,
        sensor_type="LWIR",
        metadata={
            "range_m": 500.0,
            "band": "8-12 μm",
            "target": "vehicle",
            "vehicle_type": "sedan",
            "background": "road/vegetation",
            "realistic": True,
        },
    )


def example_realistic_person(seed: int = 42) -> ExampleResult:
    """
    Example 13: Realistic Person Thermal Signature (MWIR)

    Improved human thermal model with:
    - Hot face (exposed skin)
    - Cooler clothing
    - Warm hands
    - Proper body articulation
    - Textured vegetation background

    Sensor: MWIR HgCdTe (3-5 μm)
    Scene: Person at 200m range
    Background: Vegetation/forest

    Returns:
        ExampleResult with simulated thermal image
    """
    from eosim.pipeline import create_pipeline, SceneInput
    from eosim.examples.realistic_scenes import create_realistic_person, PersonThermalParams

    params = PersonThermalParams(
        face_temp_K=307.0,
        hands_temp_K=302.0,
        clothing_temp_K=297.0,
        hair_temp_K=300.0,
        background_temp_K=288.0,
    )

    temp_map, emissivity_map = create_realistic_person(
        shape=(480, 640),
        params=params,
        pose="standing",
        seed=seed,
    )

    pipeline = create_pipeline(sensor_type="mwir", seed=seed)

    scene = SceneInput(
        temperature_map=temp_map,
        emissivity_map=emissivity_map,
    )
    result = pipeline.run(scene, range_m=200.0)

    return ExampleResult(
        name="realistic_person",
        description="Realistic person with thermal features (MWIR at 200m)",
        digital_image=result.digital_image,
        temperature_map=temp_map,
        sensor_type="MWIR",
        metadata={
            "range_m": 200.0,
            "band": "3-5 μm",
            "target": "person",
            "pose": "standing",
            "background": "vegetation",
            "realistic": True,
        },
    )


# ============================================================================
# Utility Functions
# ============================================================================

# Registry of all examples
EXAMPLES = {
    "vehicle_on_road": example_vehicle_on_road,
    "person_in_forest": example_person_in_forest,
    "aircraft_sky": example_aircraft_sky,
    "ship_ocean": example_ship_ocean,
    "building_thermal": example_building_thermal,
    "wildlife_tracking": example_wildlife_tracking,
    "industrial_monitoring": example_industrial_monitoring,
    "night_vision_swir": example_night_vision_swir,
    "solar_panel_inspection": example_solar_panel_inspection,
    "urban_surveillance": example_urban_surveillance,
    # Realistic examples with improved thermal signatures
    "f16_500m": example_f16_500m,
    "realistic_vehicle": example_realistic_vehicle,
    "realistic_person": example_realistic_person,
}


def list_examples() -> list[str]:
    """List all available example names.

    Returns:
        List of example names
    """
    return list(EXAMPLES.keys())


def get_example_by_name(name: str, seed: int = 42) -> ExampleResult:
    """Get example by name.

    Args:
        name: Example name
        seed: Random seed

    Returns:
        ExampleResult
    """
    if name not in EXAMPLES:
        raise ValueError(f"Unknown example: {name}. Available: {list_examples()}")
    return EXAMPLES[name](seed=seed)


def run_all_examples(seed: int = 42) -> Dict[str, ExampleResult]:
    """Run all examples.

    Args:
        seed: Random seed

    Returns:
        Dictionary mapping example name to result
    """
    results = {}
    for name, func in EXAMPLES.items():
        print(f"Running example: {name}...")
        results[name] = func(seed=seed)
    return results
