"""
Realistic thermal scene generation for EOSIM.

Provides utilities and example scenes with physically-based thermal signatures
including gradients, texture, and proper hot/cold features.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, distance_transform_edt
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


# =============================================================================
# Thermal Texture Utilities
# =============================================================================

def add_thermal_texture(
    image: NDArray,
    scales: list[float] = [30, 10, 3],
    amplitudes: list[float] = [2.0, 1.0, 0.3],
    seed: Optional[int] = None,
) -> NDArray:
    """Add multi-scale thermal texture to an image.

    Real thermal imagery has texture at multiple spatial scales from
    large-scale environmental variations to fine sensor noise.

    Args:
        image: Input temperature/radiance image
        scales: Gaussian blur sigma for each scale
        amplitudes: Temperature variation amplitude for each scale
        seed: Random seed

    Returns:
        Image with added thermal texture
    """
    rng = np.random.default_rng(seed)
    result = image.copy()

    for sigma, amp in zip(scales, amplitudes):
        noise = rng.normal(0, amp, image.shape)
        if sigma > 0:
            noise = gaussian_filter(noise, sigma=sigma)
        result += noise

    return result


def create_gradient_blob(
    shape: Tuple[int, int],
    center: Tuple[int, int],
    size: Tuple[int, int],
    core_temp: float,
    edge_temp: float,
    falloff: float = 2.0,
    aspect_ratio: float = 1.0,
    rotation_deg: float = 0.0,
) -> Tuple[NDArray, NDArray]:
    """Create a blob with temperature gradient from center to edge.

    Args:
        shape: Image shape (h, w)
        center: Center position (y, x)
        size: Size (height, width) of the blob
        core_temp: Temperature at center
        edge_temp: Temperature at edge
        falloff: Gradient falloff exponent (higher = sharper edge)
        aspect_ratio: Width/height ratio for elliptical blobs
        rotation_deg: Rotation angle in degrees

    Returns:
        Tuple of (temperature_map, mask)
    """
    h, w = shape
    cy, cx = center
    sy, sx = size[0] / 2, size[1] / 2

    y, x = np.ogrid[:h, :w]

    # Apply rotation
    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Centered coordinates
    dy = y - cy
    dx = x - cx

    # Rotated coordinates
    dy_rot = dy * cos_t - dx * sin_t
    dx_rot = dy * sin_t + dx * cos_t

    # Normalized distance (elliptical)
    dist = np.sqrt((dy_rot / sy) ** 2 + (dx_rot / (sx * aspect_ratio)) ** 2)

    # Temperature gradient
    t = np.clip(dist, 0, 1) ** falloff
    temp = core_temp * (1 - t) + edge_temp * t

    # Mask
    mask = dist < 1.0

    return temp, mask


def create_smooth_shape(
    shape: Tuple[int, int],
    vertices: list[Tuple[int, int]],
    smoothing: float = 3.0,
) -> NDArray:
    """Create a smooth shape mask from vertices.

    Args:
        shape: Image shape (h, w)
        vertices: List of (y, x) vertex positions
        smoothing: Gaussian smoothing sigma

    Returns:
        Smooth mask (0-1)
    """
    from PIL import Image, ImageDraw

    # Create polygon mask
    img = Image.new('L', (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)
    # Convert (y, x) to (x, y) for PIL
    poly = [(x, y) for y, x in vertices]
    draw.polygon(poly, fill=255)

    mask = np.array(img, dtype=np.float64) / 255.0

    # Smooth edges
    if smoothing > 0:
        mask = gaussian_filter(mask, sigma=smoothing)

    return mask


# =============================================================================
# Aircraft Thermal Model
# =============================================================================

@dataclass
class AircraftThermalParams:
    """Thermal parameters for aircraft."""

    # Airframe temperatures
    skin_temp_K: float = 280.0  # Base skin temperature (cold at altitude)
    leading_edge_temp_K: float = 300.0  # Aerodynamic heating
    cockpit_temp_K: float = 295.0  # Heated cockpit

    # Engine/exhaust
    engine_inlet_temp_K: float = 320.0
    exhaust_nozzle_temp_K: float = 450.0  # Visible nozzle
    exhaust_plume_temp_K: float = 600.0  # Hot core of plume
    plume_length_factor: float = 2.0  # Plume length relative to aircraft

    # Environmental
    sky_temp_K: float = 230.0  # Cold sky background

    # Altitude effects
    altitude_km: float = 5.0


def create_f16_side_aspect(
    shape: Tuple[int, int] = (480, 640),
    params: Optional[AircraftThermalParams] = None,
    position: Optional[Tuple[int, int]] = None,
    scale: float = 1.0,
    heading_right: bool = True,
    seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray]:
    """Create realistic F-16 thermal signature (side aspect).

    Models thermal features:
    - Cold airframe skin at altitude
    - Hot exhaust nozzle and plume
    - Warm leading edges (aerodynamic heating)
    - Cockpit canopy
    - Engine inlet

    Args:
        shape: Image shape (h, w)
        params: Aircraft thermal parameters
        position: Center position (y, x), defaults to image center
        scale: Scale factor for aircraft size
        heading_right: Aircraft heading right (True) or left (False)
        seed: Random seed

    Returns:
        Tuple of (temperature_map, emissivity_map)
    """
    if params is None:
        params = AircraftThermalParams()

    rng = np.random.default_rng(seed)
    h, w = shape

    if position is None:
        position = (h // 2, w // 2)

    cy, cx = position

    # F-16 approximate dimensions (pixels, will be scaled)
    # Total length ~15m, wingspan ~10m, height ~5m
    fuselage_length = int(180 * scale)
    fuselage_height = int(25 * scale)
    wing_span = int(40 * scale)  # Visible portion in side view
    wing_chord = int(60 * scale)
    tail_height = int(50 * scale)

    # Direction multiplier
    direction = 1 if heading_right else -1

    # Initialize with cold sky
    temp = np.full(shape, params.sky_temp_K, dtype=np.float64)
    emissivity = np.full(shape, 0.95, dtype=np.float64)

    y, x = np.ogrid[:h, :w]

    # Helper to add component with gradient
    def add_component(temp_map, emis_map, mask, core_temp, edge_temp, emis=0.9):
        """Add thermal component with gradient."""
        if not np.any(mask > 0):
            return

        # Distance from edge for gradient
        dist = distance_transform_edt(mask > 0.5)
        max_dist = max(dist.max(), 1)
        gradient = dist / max_dist

        component_temp = edge_temp + (core_temp - edge_temp) * gradient
        component_temp += rng.normal(0, 1, shape) * mask  # Add noise

        blend = mask
        temp_map[:] = temp_map * (1 - blend) + component_temp * blend
        emis_map[:] = emis_map * (1 - blend) + emis * blend

    # 1. FUSELAGE (main body)
    fuselage_y = cy
    fuselage_x = cx

    # Fuselage shape (tapered cylinder)
    nose_x = fuselage_x - direction * fuselage_length // 2
    tail_x = fuselage_x + direction * fuselage_length // 2

    # Create fuselage mask
    dx = (x - fuselage_x) * direction
    dy = y - fuselage_y

    # Tapered fuselage (wider in middle, narrower at nose/tail)
    fuselage_width = fuselage_height * (1 - 0.3 * np.abs(dx) / (fuselage_length / 2))
    fuselage_width = np.maximum(fuselage_width, fuselage_height * 0.4)

    fuselage_mask = (np.abs(dx) < fuselage_length / 2) & (np.abs(dy) < fuselage_width / 2)
    fuselage_mask = gaussian_filter(fuselage_mask.astype(float), sigma=2)

    add_component(temp, emissivity, fuselage_mask,
                  params.skin_temp_K + 5, params.skin_temp_K - 5, emis=0.85)

    # 2. NOSE (slightly warmer due to aerodynamic heating)
    nose_center = (cy, cx - direction * fuselage_length * 0.4)
    nose_temp, nose_mask = create_gradient_blob(
        shape, nose_center,
        (fuselage_height * 0.8, fuselage_height * 1.2),
        params.leading_edge_temp_K, params.skin_temp_K,
        falloff=1.5
    )
    nose_mask = nose_mask.astype(float) * 0.8
    add_component(temp, emissivity, nose_mask,
                  params.leading_edge_temp_K, params.skin_temp_K, emis=0.85)

    # 3. COCKPIT CANOPY (warmer, different emissivity)
    cockpit_x = cx - direction * fuselage_length * 0.25
    cockpit_y = cy - fuselage_height * 0.6
    cockpit_temp, cockpit_mask = create_gradient_blob(
        shape, (int(cockpit_y), int(cockpit_x)),
        (int(fuselage_height * 0.8), int(fuselage_height * 1.5)),
        params.cockpit_temp_K + 5, params.cockpit_temp_K - 5,
        falloff=2.0, aspect_ratio=2.0
    )
    cockpit_mask = cockpit_mask.astype(float) * 0.9
    add_component(temp, emissivity, cockpit_mask,
                  params.cockpit_temp_K, params.cockpit_temp_K - 10, emis=0.6)  # Glass

    # 4. WING (side view shows thickness)
    wing_y = cy + fuselage_height * 0.3
    wing_x = cx - direction * fuselage_length * 0.05

    # Wing cross-section
    wing_dx = (x - wing_x) * direction
    wing_dy = y - wing_y
    wing_mask = (np.abs(wing_dx) < wing_chord / 2) & (np.abs(wing_dy) < wing_span / 2)
    wing_mask = wing_mask & (wing_dy > -wing_span * 0.3)  # Mostly below fuselage
    wing_mask = gaussian_filter(wing_mask.astype(float), sigma=2)

    # Leading edge is warmer
    wing_leading = (wing_dx < -wing_chord * 0.3) & (np.abs(wing_dy) < wing_span / 2)
    wing_leading = gaussian_filter(wing_leading.astype(float), sigma=3)

    add_component(temp, emissivity, wing_mask,
                  params.skin_temp_K, params.skin_temp_K - 8, emis=0.85)
    add_component(temp, emissivity, wing_leading * 0.5,
                  params.leading_edge_temp_K - 10, params.skin_temp_K, emis=0.85)

    # 5. VERTICAL TAIL
    tail_x = cx + direction * fuselage_length * 0.35
    tail_y = cy - fuselage_height * 0.5

    tail_dx = (x - tail_x) * direction
    tail_dy = y - tail_y
    tail_mask = (np.abs(tail_dx) < fuselage_height * 0.6) & (tail_dy < 0) & (tail_dy > -tail_height)
    # Taper the tail
    tail_mask = tail_mask & (np.abs(tail_dx) < fuselage_height * 0.6 * (1 + tail_dy / tail_height))
    tail_mask = gaussian_filter(tail_mask.astype(float), sigma=2)

    add_component(temp, emissivity, tail_mask,
                  params.skin_temp_K - 3, params.skin_temp_K - 10, emis=0.85)

    # 6. HORIZONTAL STABILIZER
    stab_y = cy + fuselage_height * 0.2
    stab_x = tail_x - direction * fuselage_height * 0.5
    stab_dx = (x - stab_x) * direction
    stab_dy = y - stab_y
    stab_mask = (np.abs(stab_dx) < fuselage_height * 1.2) & (np.abs(stab_dy) < wing_span * 0.3)
    stab_mask = gaussian_filter(stab_mask.astype(float), sigma=2)

    add_component(temp, emissivity, stab_mask * 0.7,
                  params.skin_temp_K - 5, params.skin_temp_K - 12, emis=0.85)

    # 7. ENGINE INLET (side of fuselage, warm)
    inlet_x = cx - direction * fuselage_length * 0.1
    inlet_y = cy + fuselage_height * 0.3
    inlet_temp, inlet_mask = create_gradient_blob(
        shape, (int(inlet_y), int(inlet_x)),
        (int(fuselage_height * 0.6), int(fuselage_height * 0.8)),
        params.engine_inlet_temp_K, params.skin_temp_K + 10,
        falloff=2.5
    )
    add_component(temp, emissivity, inlet_mask.astype(float) * 0.6,
                  params.engine_inlet_temp_K, params.skin_temp_K + 10, emis=0.9)

    # 8. EXHAUST NOZZLE (very hot!)
    nozzle_x = cx + direction * fuselage_length * 0.48
    nozzle_y = cy
    nozzle_temp, nozzle_mask = create_gradient_blob(
        shape, (int(nozzle_y), int(nozzle_x)),
        (int(fuselage_height * 0.7), int(fuselage_height * 0.5)),
        params.exhaust_nozzle_temp_K, params.exhaust_nozzle_temp_K - 100,
        falloff=1.5
    )
    add_component(temp, emissivity, nozzle_mask.astype(float),
                  params.exhaust_nozzle_temp_K, params.exhaust_nozzle_temp_K - 80, emis=0.95)

    # 9. EXHAUST PLUME (extends behind aircraft)
    plume_length = int(fuselage_length * params.plume_length_factor)
    plume_start_x = nozzle_x + direction * fuselage_height * 0.3

    # Plume expands and cools with distance
    plume_dx = (x - plume_start_x) * direction
    plume_dy = y - nozzle_y

    # Plume region (cone expanding backward)
    plume_width = fuselage_height * 0.3 + np.abs(plume_dx) * 0.15
    in_plume = (plume_dx > 0) & (plume_dx < plume_length) & (np.abs(plume_dy) < plume_width)

    # Temperature decreases with distance
    plume_dist = np.clip(plume_dx / plume_length, 0, 1)
    plume_temp_field = (params.exhaust_plume_temp_K * (1 - plume_dist ** 0.5) +
                        params.sky_temp_K * plume_dist ** 0.5)

    # Core is hotter than edges
    plume_core = np.exp(-2 * (plume_dy / np.maximum(plume_width, 1)) ** 2)
    plume_temp_field = plume_temp_field * plume_core + params.sky_temp_K * (1 - plume_core)

    plume_mask = in_plume.astype(float) * plume_core
    plume_mask = gaussian_filter(plume_mask, sigma=3)

    # Blend plume (semi-transparent hot gas)
    plume_blend = plume_mask * 0.7
    temp = temp * (1 - plume_blend) + plume_temp_field * plume_blend

    # 10. Add fine thermal texture/noise
    aircraft_mask = (temp > params.sky_temp_K + 5).astype(float)
    texture = gaussian_filter(rng.normal(0, 2, shape), sigma=2) * aircraft_mask
    temp += texture

    # Add sensor noise to everything
    temp += rng.normal(0, 0.5, shape)

    return temp, emissivity


# =============================================================================
# Improved Ground Vehicle
# =============================================================================

@dataclass
class VehicleThermalParams:
    """Thermal parameters for ground vehicle."""

    body_temp_K: float = 315.0  # Sun-heated metal
    engine_temp_K: float = 360.0  # Hot engine compartment
    exhaust_temp_K: float = 400.0  # Exhaust pipe/muffler
    tire_temp_K: float = 320.0  # Friction-heated tires
    window_temp_K: float = 280.0  # Glass reflects cold sky

    road_temp_K: float = 305.0  # Warm asphalt
    vegetation_temp_K: float = 295.0  # Cooler vegetation

    ambient_temp_K: float = 295.0


def create_realistic_vehicle(
    shape: Tuple[int, int] = (480, 640),
    params: Optional[VehicleThermalParams] = None,
    position: Optional[Tuple[int, int]] = None,
    scale: float = 1.0,
    vehicle_type: str = "sedan",  # sedan, suv, truck
    seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray]:
    """Create realistic ground vehicle thermal signature.

    Args:
        shape: Image shape
        params: Vehicle thermal parameters
        position: Center position
        scale: Scale factor
        vehicle_type: Type of vehicle
        seed: Random seed

    Returns:
        Tuple of (temperature_map, emissivity_map)
    """
    if params is None:
        params = VehicleThermalParams()

    rng = np.random.default_rng(seed)
    h, w = shape

    if position is None:
        position = (h // 2, w // 2)

    cy, cx = position

    # Vehicle dimensions based on type
    dims = {
        "sedan": (60, 140, 35),   # height, length, cabin_height
        "suv": (75, 150, 45),
        "truck": (80, 180, 40),
    }
    veh_h, veh_l, cabin_h = [int(d * scale) for d in dims.get(vehicle_type, dims["sedan"])]

    # Initialize with background
    temp = np.full(shape, params.vegetation_temp_K, dtype=np.float64)
    emissivity = np.full(shape, 0.95, dtype=np.float64)

    # Add background texture
    temp = add_thermal_texture(temp, scales=[40, 15, 5], amplitudes=[3, 1.5, 0.5], seed=seed)

    # Road strip
    road_y0, road_y1 = h // 3, 2 * h // 3
    road = params.road_temp_K + gaussian_filter(rng.normal(0, 2, (road_y1 - road_y0, w)), sigma=8)
    temp[road_y0:road_y1, :] = road
    emissivity[road_y0:road_y1, :] = 0.92

    y, x = np.ogrid[:h, :w]

    # 1. MAIN BODY (with gradients)
    body_temp, body_mask = create_gradient_blob(
        shape, position, (veh_h, veh_l),
        params.body_temp_K + 5, params.body_temp_K - 10,
        falloff=2.5, aspect_ratio=veh_l / veh_h
    )
    body_mask = body_mask.astype(float)
    body_mask = gaussian_filter(body_mask, sigma=2)

    temp = temp * (1 - body_mask) + body_temp * body_mask
    emissivity = emissivity * (1 - body_mask) + 0.85 * body_mask

    # 2. ROOF (slightly cooler, different angle to sky)
    roof_y = cy - veh_h * 0.3
    roof_temp, roof_mask = create_gradient_blob(
        shape, (int(roof_y), cx), (int(cabin_h * 0.4), int(veh_l * 0.5)),
        params.body_temp_K - 5, params.body_temp_K - 15,
        falloff=2.0, aspect_ratio=2.5
    )
    roof_mask = roof_mask.astype(float) * 0.8
    temp = temp * (1 - roof_mask) + roof_temp * roof_mask

    # 3. ENGINE COMPARTMENT (hot, front of vehicle)
    engine_x = cx - veh_l * 0.35
    engine_temp, engine_mask = create_gradient_blob(
        shape, (cy, int(engine_x)), (int(veh_h * 0.7), int(veh_l * 0.25)),
        params.engine_temp_K, params.body_temp_K + 20,
        falloff=1.8
    )
    engine_mask = engine_mask.astype(float) * 0.9
    temp = temp * (1 - engine_mask) + engine_temp * engine_mask
    emissivity = emissivity * (1 - engine_mask) + 0.9 * engine_mask

    # 4. EXHAUST (hot spot at rear underside)
    exhaust_x = cx + veh_l * 0.4
    exhaust_y = cy + veh_h * 0.35
    exhaust_temp, exhaust_mask = create_gradient_blob(
        shape, (int(exhaust_y), int(exhaust_x)), (int(veh_h * 0.2), int(veh_h * 0.3)),
        params.exhaust_temp_K, params.exhaust_temp_K - 60,
        falloff=1.5
    )
    exhaust_mask = exhaust_mask.astype(float)
    temp = temp * (1 - exhaust_mask) + exhaust_temp * exhaust_mask

    # 5. WINDOWS (cold - reflect sky)
    window_positions = [
        (cy - veh_h * 0.2, cx - veh_l * 0.15, (cabin_h * 0.5, veh_l * 0.12)),  # Front windshield
        (cy - veh_h * 0.2, cx + veh_l * 0.1, (cabin_h * 0.5, veh_l * 0.08)),   # Rear window
        (cy - veh_h * 0.15, cx - veh_l * 0.02, (cabin_h * 0.35, veh_l * 0.06)), # Side window
    ]

    for wy, wx, (wh, ww) in window_positions:
        win_temp, win_mask = create_gradient_blob(
            shape, (int(wy), int(wx)), (int(wh), int(ww)),
            params.window_temp_K + 3, params.window_temp_K - 3,
            falloff=3.0, aspect_ratio=ww/wh if wh > 0 else 1
        )
        win_mask = win_mask.astype(float) * 0.85
        temp = temp * (1 - win_mask) + win_temp * win_mask
        emissivity = emissivity * (1 - win_mask) + 0.6 * win_mask  # Glass emissivity

    # 6. TIRES (warm from friction)
    tire_positions = [
        (cy + veh_h * 0.35, cx - veh_l * 0.3),  # Front left
        (cy + veh_h * 0.35, cx + veh_l * 0.25), # Rear left
    ]
    tire_radius = int(veh_h * 0.25)

    for ty, tx in tire_positions:
        tire_temp, tire_mask = create_gradient_blob(
            shape, (int(ty), int(tx)), (tire_radius, tire_radius),
            params.tire_temp_K, params.tire_temp_K - 15,
            falloff=2.0
        )
        tire_mask = tire_mask.astype(float) * 0.7
        temp = temp * (1 - tire_mask) + tire_temp * tire_mask
        emissivity = emissivity * (1 - tire_mask) + 0.92 * tire_mask

    # 7. Add thermal noise
    noise = rng.normal(0, 0.8, shape)
    temp += noise

    return temp, emissivity


# =============================================================================
# Improved Person Thermal Signature
# =============================================================================

@dataclass
class PersonThermalParams:
    """Thermal parameters for human."""

    face_temp_K: float = 307.0  # Exposed skin (face)
    hands_temp_K: float = 303.0  # Hands (cooler than face)
    clothing_temp_K: float = 298.0  # Clothed body
    hair_temp_K: float = 301.0  # Hair (insulating)

    background_temp_K: float = 290.0


def create_realistic_person(
    shape: Tuple[int, int] = (480, 640),
    params: Optional[PersonThermalParams] = None,
    position: Optional[Tuple[int, int]] = None,
    scale: float = 1.0,
    pose: str = "standing",  # standing, walking, sitting
    seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray]:
    """Create realistic human thermal signature.

    Args:
        shape: Image shape
        params: Person thermal parameters
        position: Center position
        scale: Scale factor
        pose: Person pose
        seed: Random seed

    Returns:
        Tuple of (temperature_map, emissivity_map)
    """
    if params is None:
        params = PersonThermalParams()

    rng = np.random.default_rng(seed)
    h, w = shape

    if position is None:
        position = (h // 2, w // 2)

    cy, cx = position

    # Person dimensions
    person_height = int(120 * scale)
    torso_width = int(35 * scale)
    head_radius = int(15 * scale)

    # Initialize background
    temp = np.full(shape, params.background_temp_K, dtype=np.float64)
    temp = add_thermal_texture(temp, scales=[25, 8, 2], amplitudes=[3, 1.5, 0.5], seed=seed)
    emissivity = np.full(shape, 0.95, dtype=np.float64)

    y, x = np.ogrid[:h, :w]

    # 1. TORSO (clothed)
    torso_y = cy + person_height * 0.1
    torso_temp, torso_mask = create_gradient_blob(
        shape, (int(torso_y), cx), (int(person_height * 0.45), torso_width),
        params.clothing_temp_K + 3, params.clothing_temp_K - 5,
        falloff=2.0, aspect_ratio=0.6
    )
    torso_mask = gaussian_filter(torso_mask.astype(float), sigma=2)
    temp = temp * (1 - torso_mask) + torso_temp * torso_mask
    emissivity = emissivity * (1 - torso_mask) + 0.95 * torso_mask

    # 2. HEAD (with face hot spot)
    head_y = cy - person_height * 0.35
    head_temp, head_mask = create_gradient_blob(
        shape, (int(head_y), cx), (head_radius * 2, int(head_radius * 1.6)),
        params.hair_temp_K, params.hair_temp_K - 5,
        falloff=2.5
    )
    head_mask = head_mask.astype(float)
    temp = temp * (1 - head_mask) + head_temp * head_mask

    # Face (front of head, warmer)
    face_y = head_y + head_radius * 0.1
    face_temp, face_mask = create_gradient_blob(
        shape, (int(face_y), cx), (int(head_radius * 1.2), int(head_radius * 0.9)),
        params.face_temp_K, params.face_temp_K - 8,
        falloff=2.0
    )
    face_mask = face_mask.astype(float) * 0.9
    temp = temp * (1 - face_mask) + face_temp * face_mask
    emissivity = emissivity * (1 - face_mask) + 0.98 * face_mask

    # 3. ARMS
    for arm_side in [-1, 1]:
        arm_x = cx + arm_side * torso_width * 0.6
        arm_y = cy + person_height * 0.05

        arm_temp, arm_mask = create_gradient_blob(
            shape, (int(arm_y), int(arm_x)),
            (int(person_height * 0.35), int(torso_width * 0.25)),
            params.clothing_temp_K, params.clothing_temp_K - 6,
            falloff=2.5
        )
        arm_mask = arm_mask.astype(float) * 0.8
        temp = temp * (1 - arm_mask) + arm_temp * arm_mask

        # Hands (exposed, warm)
        hand_y = arm_y + person_height * 0.2
        hand_temp, hand_mask = create_gradient_blob(
            shape, (int(hand_y), int(arm_x)),
            (int(head_radius * 0.6), int(head_radius * 0.5)),
            params.hands_temp_K, params.hands_temp_K - 5,
            falloff=2.0
        )
        hand_mask = hand_mask.astype(float) * 0.7
        temp = temp * (1 - hand_mask) + hand_temp * hand_mask
        emissivity = emissivity * (1 - hand_mask) + 0.98 * hand_mask

    # 4. LEGS
    for leg_side in [-0.3, 0.3]:
        leg_x = cx + leg_side * torso_width
        leg_y = cy + person_height * 0.45

        leg_temp, leg_mask = create_gradient_blob(
            shape, (int(leg_y), int(leg_x)),
            (int(person_height * 0.4), int(torso_width * 0.35)),
            params.clothing_temp_K - 2, params.clothing_temp_K - 8,
            falloff=2.5
        )
        leg_mask = leg_mask.astype(float) * 0.75
        temp = temp * (1 - leg_mask) + leg_temp * leg_mask

    # 5. Add thermal noise
    person_mask = temp > params.background_temp_K + 3
    noise = rng.normal(0, 0.5, shape)
    temp += noise

    return temp, emissivity


# =============================================================================
# Scene with Multiple Elements
# =============================================================================

def create_f16_at_range(
    shape: Tuple[int, int] = (480, 640),
    range_km: float = 10.0,
    altitude_km: float = 5.0,
    seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray, dict]:
    """Create F-16 scene at specified range.

    Applies appropriate scaling and atmospheric effects for range.

    Args:
        shape: Image shape
        range_km: Range to target in km
        altitude_km: Aircraft altitude in km
        seed: Random seed

    Returns:
        Tuple of (temperature_map, emissivity_map, metadata)
    """
    # Scale aircraft size based on range (assume 25mm focal length, 15μm pixel)
    # At 1km, F-16 (15m) subtends ~150 pixels
    # At 10km, ~15 pixels
    reference_range_km = 1.0
    reference_scale = 1.0
    scale = reference_scale * (reference_range_km / range_km)

    # Adjust thermal parameters for altitude
    params = AircraftThermalParams(
        altitude_km=altitude_km,
        # Colder at altitude
        skin_temp_K=260.0 + 5 * (5 - altitude_km),  # Colder at higher altitude
        leading_edge_temp_K=280.0 + 10 * (5 - altitude_km),
        # Sky gets colder with altitude
        sky_temp_K=220.0 - 3 * altitude_km,
    )

    temp, emis = create_f16_side_aspect(
        shape=shape,
        params=params,
        scale=scale,
        seed=seed,
    )

    metadata = {
        "range_km": range_km,
        "altitude_km": altitude_km,
        "scale": scale,
        "aircraft_type": "F-16",
        "aspect": "side",
    }

    return temp, emis, metadata
