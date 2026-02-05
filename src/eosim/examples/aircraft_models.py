"""
Detailed aircraft models for EOSIM.

Provides realistic aircraft shapes and rendering for both thermal and visible bands.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path


@dataclass
class F16Geometry:
    """F-16 Fighting Falcon geometry definition.

    Approximate dimensions (meters):
    - Length: 15.06 m
    - Wingspan: 9.96 m
    - Height: 4.88 m
    """
    length_m: float = 15.06
    wingspan_m: float = 9.96
    height_m: float = 4.88

    # Component positions (normalized 0-1 along length, from nose)
    nose_length: float = 0.15
    cockpit_start: float = 0.12
    cockpit_end: float = 0.28
    intake_start: float = 0.25
    intake_end: float = 0.40
    wing_start: float = 0.35
    wing_end: float = 0.65
    tail_start: float = 0.75
    nozzle_start: float = 0.92


def create_f16_silhouette(
    shape: Tuple[int, int],
    center: Tuple[int, int],
    scale_pixels_per_meter: float,
    heading_right: bool = True,
    bank_angle_deg: float = 0.0,
) -> Tuple[NDArray, dict]:
    """Create detailed F-16 silhouette mask with component regions.

    Args:
        shape: Image shape (height, width)
        center: Aircraft center position (y, x)
        scale_pixels_per_meter: Pixels per meter for sizing
        heading_right: Aircraft heading right (True) or left
        bank_angle_deg: Bank angle in degrees (positive = right wing down)

    Returns:
        Tuple of (silhouette_mask, component_masks_dict)
    """
    h, w = shape
    cy, cx = center
    geom = F16Geometry()

    # Scale factors
    length_px = geom.length_m * scale_pixels_per_meter
    wingspan_px = geom.wingspan_m * scale_pixels_per_meter
    height_px = geom.height_m * scale_pixels_per_meter

    direction = 1 if heading_right else -1

    # Create coordinate grids
    y, x = np.ogrid[:h, :w]

    # Transform to aircraft-relative coordinates (nose at 0, tail at 1)
    dx = (x - cx) * direction
    dy = y - cy

    # Normalize to aircraft length
    x_norm = (dx / length_px) + 0.5  # 0 = nose, 1 = tail

    # Initialize masks
    masks = {
        'fuselage': np.zeros(shape, dtype=bool),
        'nose': np.zeros(shape, dtype=bool),
        'cockpit': np.zeros(shape, dtype=bool),
        'canopy': np.zeros(shape, dtype=bool),
        'intake': np.zeros(shape, dtype=bool),
        'wing_left': np.zeros(shape, dtype=bool),
        'wing_right': np.zeros(shape, dtype=bool),
        'horizontal_stab': np.zeros(shape, dtype=bool),
        'vertical_tail': np.zeros(shape, dtype=bool),
        'nozzle': np.zeros(shape, dtype=bool),
        'exhaust_plume': np.zeros(shape, dtype=bool),
    }

    # 1. FUSELAGE - tapered cylinder shape
    # Width varies along length
    def fuselage_width(x_n):
        """Fuselage half-width as function of normalized position."""
        # Nose taper
        if isinstance(x_n, np.ndarray):
            width = np.zeros_like(x_n)
            # Nose cone (0 to 0.15)
            nose_mask = (x_n >= 0) & (x_n < 0.15)
            width[nose_mask] = 0.3 * np.power(np.maximum(x_n[nose_mask] / 0.15, 0), 0.7)
            # Main fuselage (0.15 to 0.85)
            main_mask = (x_n >= 0.15) & (x_n < 0.85)
            width[main_mask] = 0.3 + 0.15 * np.sin(np.pi * (x_n[main_mask] - 0.15) / 0.7)
            # Tail taper (0.85 to 1.0)
            tail_mask = x_n >= 0.85
            width[tail_mask] = 0.35 * (1 - (x_n[tail_mask] - 0.85) / 0.15)
            return width
        else:
            if x_n < 0.15:
                return 0.3 * (x_n / 0.15) ** 0.7
            elif x_n < 0.85:
                return 0.3 + 0.15 * np.sin(np.pi * (x_n - 0.15) / 0.7)
            else:
                return 0.35 * (1 - (x_n - 0.85) / 0.15)

    fuselage_half_width = fuselage_width(x_norm) * height_px
    in_fuselage = (x_norm >= 0) & (x_norm <= 1.0) & (np.abs(dy) < fuselage_half_width)
    masks['fuselage'] = in_fuselage

    # 2. NOSE CONE
    masks['nose'] = in_fuselage & (x_norm < 0.15)

    # 3. COCKPIT area (beneath canopy)
    cockpit_width = 0.25 * height_px
    in_cockpit = ((x_norm >= 0.12) & (x_norm < 0.30) &
                  (dy > -cockpit_width * 0.3) & (dy < cockpit_width * 0.8))
    masks['cockpit'] = in_cockpit & in_fuselage

    # 4. CANOPY (bubble on top)
    canopy_cx = 0.20
    canopy_length = 0.12 * length_px
    canopy_height = 0.20 * height_px
    canopy_dist = np.sqrt(((x_norm - canopy_cx) * length_px / canopy_length) ** 2 +
                          ((dy + height_px * 0.15) / canopy_height) ** 2)
    masks['canopy'] = (canopy_dist < 1) & (dy < 0)

    # 5. AIR INTAKE (side-mounted on F-16)
    intake_y_offset = height_px * 0.15  # Below centerline
    intake_width = height_px * 0.25
    intake_length = 0.15 * length_px
    in_intake = ((x_norm >= 0.25) & (x_norm < 0.40) &
                 (np.abs(dy - intake_y_offset) < intake_width))
    masks['intake'] = in_intake

    # 6. WINGS (delta shape with strakes)
    # Wing leading edge sweep
    wing_le_x = 0.30  # Leading edge start
    wing_te_x = 0.70  # Trailing edge
    wing_root_y = height_px * 0.3
    wing_tip_y = wingspan_px * 0.5

    # Left wing (below in side view, but visible with bank)
    wing_span_visible = wing_tip_y * (1 - 0.7 * abs(bank_angle_deg) / 90)

    # Simplified wing shape
    in_wing_region = (x_norm >= wing_le_x) & (x_norm <= wing_te_x)
    wing_local_x = (x_norm - wing_le_x) / (wing_te_x - wing_le_x)

    # Leading edge sweep line
    wing_le_y = wing_root_y + wing_local_x * (wing_span_visible - wing_root_y)
    # Trailing edge
    wing_te_y = wing_root_y + wing_local_x * 0.3 * (wing_span_visible - wing_root_y)

    in_left_wing = in_wing_region & (dy > wing_te_y) & (dy < wing_le_y)
    in_right_wing = in_wing_region & (dy < -wing_te_y) & (dy > -wing_le_y)

    masks['wing_left'] = in_left_wing
    masks['wing_right'] = in_right_wing

    # 7. HORIZONTAL STABILIZER
    stab_x_start = 0.78
    stab_x_end = 0.95
    stab_span = wingspan_px * 0.25
    in_stab_region = (x_norm >= stab_x_start) & (x_norm <= stab_x_end)
    stab_local_x = np.clip((x_norm - stab_x_start) / (stab_x_end - stab_x_start), 0, 1)
    stab_y_extent = stab_span * (1 - stab_local_x * 0.5)

    masks['horizontal_stab'] = in_stab_region & (np.abs(dy) < stab_y_extent) & (np.abs(dy) > height_px * 0.2)

    # 8. VERTICAL TAIL
    tail_x_start = 0.70
    tail_x_end = 0.95
    tail_height = height_px * 1.5
    in_tail_region = (x_norm >= tail_x_start) & (x_norm <= tail_x_end)
    tail_local_x = np.clip((x_norm - tail_x_start) / (tail_x_end - tail_x_start), 0, 1)
    tail_y_top = -height_px * 0.3 - tail_height * (1 - tail_local_x * 0.3)
    tail_width = height_px * 0.15 * (1 - tail_local_x * 0.5)

    masks['vertical_tail'] = in_tail_region & (dy < -height_px * 0.2) & (dy > tail_y_top) & (np.abs(dx - cx) < tail_width + np.abs((dy + height_px * 0.3) * 0.1))

    # Simplified vertical tail
    vt_cx = 0.85
    vt_width = 0.08 * length_px
    vt_height = 1.2 * height_px
    vt_dist_x = np.abs((x_norm - vt_cx) * length_px)
    in_vtail = (vt_dist_x < vt_width) & (dy < -height_px * 0.1) & (dy > -vt_height)
    masks['vertical_tail'] = in_vtail

    # 9. EXHAUST NOZZLE
    nozzle_radius = height_px * 0.35
    nozzle_cx = 0.96
    nozzle_dist = np.sqrt(((x_norm - nozzle_cx) * length_px) ** 2 + dy ** 2)
    masks['nozzle'] = (nozzle_dist < nozzle_radius) & (x_norm > 0.92)

    # 10. EXHAUST PLUME
    plume_start_x = 1.0
    plume_length = 0.8  # Relative to aircraft length
    in_plume_x = (x_norm > plume_start_x) & (x_norm < plume_start_x + plume_length)
    plume_local_x = np.clip((x_norm - plume_start_x) / plume_length, 0, 1)
    plume_radius = nozzle_radius * (1 + plume_local_x * 2)  # Expands
    masks['exhaust_plume'] = in_plume_x & (np.abs(dy) < plume_radius)

    # Combine all solid parts for silhouette
    silhouette = (masks['fuselage'] | masks['canopy'] |
                  masks['wing_left'] | masks['wing_right'] |
                  masks['horizontal_stab'] | masks['vertical_tail'])

    # Smooth edges
    silhouette = gaussian_filter(silhouette.astype(float), sigma=1.5) > 0.3

    return silhouette, masks


def render_f16_visible(
    shape: Tuple[int, int],
    center: Tuple[int, int],
    scale_pixels_per_meter: float,
    heading_right: bool = True,
    lighting_angle_deg: float = 45.0,
    sky_brightness: float = 0.7,
    aircraft_albedo: float = 0.3,
    seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Render F-16 in visible band with realistic shading.

    Args:
        shape: Image shape (height, width)
        center: Aircraft center (y, x)
        scale_pixels_per_meter: Scale factor
        heading_right: Heading direction
        lighting_angle_deg: Sun angle from horizon
        sky_brightness: Sky background brightness (0-1)
        aircraft_albedo: Aircraft surface reflectance
        seed: Random seed

    Returns:
        Tuple of (rgb_image, silhouette_mask, component_masks)
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    # Get silhouette and component masks
    silhouette, masks = create_f16_silhouette(
        shape, center, scale_pixels_per_meter, heading_right
    )

    # Initialize RGB image with sky gradient
    rgb = np.zeros((h, w, 3), dtype=np.float64)

    # Sky gradient (blue, darker at top)
    y_norm = np.linspace(0, 1, h)[:, np.newaxis]
    sky_r = sky_brightness * (0.5 + 0.3 * y_norm)
    sky_g = sky_brightness * (0.6 + 0.3 * y_norm)
    sky_b = sky_brightness * (0.9 + 0.1 * y_norm)

    rgb[:, :, 0] = sky_r.squeeze()[:, np.newaxis] * np.ones(w)
    rgb[:, :, 1] = sky_g.squeeze()[:, np.newaxis] * np.ones(w)
    rgb[:, :, 2] = sky_b.squeeze()[:, np.newaxis] * np.ones(w)

    # Add some clouds/variation
    cloud_noise = gaussian_filter(rng.normal(0, 0.05, shape), sigma=30)
    for c in range(3):
        rgb[:, :, c] += cloud_noise

    # Aircraft base color (gray with slight blue tint - typical fighter paint)
    aircraft_color = np.array([0.45, 0.47, 0.50]) * aircraft_albedo

    # Compute simple shading based on surface normals approximation
    # Use gradient of silhouette distance as proxy for surface orientation
    from scipy.ndimage import distance_transform_edt, sobel

    dist = distance_transform_edt(silhouette)
    dist_normalized = dist / max(dist.max(), 1)

    # Gradient gives approximate surface normal
    grad_y = sobel(dist_normalized, axis=0)
    grad_x = sobel(dist_normalized, axis=1)

    # Lighting direction
    light_angle_rad = np.radians(lighting_angle_deg)
    light_dir = np.array([np.cos(light_angle_rad), -np.sin(light_angle_rad)])

    # Diffuse shading
    shading = 0.5 + 0.5 * (grad_x * light_dir[0] + grad_y * light_dir[1])
    shading = np.clip(shading, 0.2, 1.0)

    # Apply aircraft color with shading
    silhouette_float = silhouette.astype(float)
    for c in range(3):
        aircraft_layer = aircraft_color[c] * shading * silhouette_float
        rgb[:, :, c] = rgb[:, :, c] * (1 - silhouette_float) + aircraft_layer

    # Add component details
    # Canopy - darker, reflective
    canopy_mask = masks['canopy'].astype(float)
    canopy_mask = gaussian_filter(canopy_mask, sigma=1)
    canopy_color = np.array([0.15, 0.18, 0.22])  # Dark tinted glass
    for c in range(3):
        rgb[:, :, c] = rgb[:, :, c] * (1 - canopy_mask * 0.8) + canopy_color[c] * canopy_mask * 0.8

    # Canopy glint (sun reflection)
    cy, cx = center
    glint_y = cy - int(scale_pixels_per_meter * 2)
    glint_x = cx - int(scale_pixels_per_meter * 1) * (1 if heading_right else -1)
    y, x = np.ogrid[:h, :w]
    glint_dist = np.sqrt((y - glint_y)**2 + (x - glint_x)**2)
    glint = np.exp(-glint_dist**2 / (scale_pixels_per_meter * 2)**2) * canopy_mask
    for c in range(3):
        rgb[:, :, c] += glint * 0.5

    # Intake - dark opening
    intake_mask = masks['intake'].astype(float)
    intake_mask = gaussian_filter(intake_mask, sigma=1)
    intake_color = np.array([0.08, 0.08, 0.10])
    for c in range(3):
        rgb[:, :, c] = rgb[:, :, c] * (1 - intake_mask * 0.9) + intake_color[c] * intake_mask * 0.9

    # Nozzle - metallic
    nozzle_mask = masks['nozzle'].astype(float)
    nozzle_mask = gaussian_filter(nozzle_mask, sigma=1)
    nozzle_color = np.array([0.25, 0.22, 0.20])
    for c in range(3):
        rgb[:, :, c] = rgb[:, :, c] * (1 - nozzle_mask * 0.8) + nozzle_color[c] * nozzle_mask * 0.8

    # Exhaust plume - hot glow (visible in afterburner)
    plume_mask = masks['exhaust_plume'].astype(float)
    plume_mask = gaussian_filter(plume_mask, sigma=3)
    # Afterburner colors: bright orange/yellow core
    direction = 1 if heading_right else -1
    plume_x_norm = ((np.arange(w) - cx) * direction / (scale_pixels_per_meter * 15)).clip(0, 1)
    plume_intensity = plume_mask * (1 - plume_x_norm[np.newaxis, :]) ** 2
    rgb[:, :, 0] += plume_intensity * 0.8  # Red
    rgb[:, :, 1] += plume_intensity * 0.5  # Green (orange)
    rgb[:, :, 2] += plume_intensity * 0.1  # Blue (minimal)

    # Add panel lines / surface detail
    detail_noise = gaussian_filter(rng.normal(0, 0.02, shape), sigma=2) * silhouette_float
    for c in range(3):
        rgb[:, :, c] += detail_noise

    # Add markings (simplified - just darker patches for insignia areas)
    # US Air Force typically has insignia on fuselage and wings

    # Clip to valid range
    rgb = np.clip(rgb, 0, 1)

    return rgb, silhouette, masks


def render_f16_thermal_detailed(
    shape: Tuple[int, int],
    center: Tuple[int, int],
    scale_pixels_per_meter: float,
    heading_right: bool = True,
    airframe_temp_K: float = 280.0,
    exhaust_temp_K: float = 700.0,
    sky_temp_K: float = 230.0,
    seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray]:
    """Render detailed F-16 thermal signature.

    Args:
        shape: Image shape
        center: Aircraft center
        scale_pixels_per_meter: Scale
        heading_right: Heading direction
        airframe_temp_K: Base airframe temperature
        exhaust_temp_K: Exhaust temperature
        sky_temp_K: Sky background temperature
        seed: Random seed

    Returns:
        Tuple of (temperature_map, emissivity_map)
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    # Get masks
    silhouette, masks = create_f16_silhouette(
        shape, center, scale_pixels_per_meter, heading_right
    )

    # Initialize with sky temperature
    temp = np.full(shape, sky_temp_K, dtype=np.float64)
    emissivity = np.full(shape, 0.95, dtype=np.float64)

    # Add sky texture
    temp += gaussian_filter(rng.normal(0, 2, shape), sigma=20)

    # Fuselage - base temperature with variation
    fuselage_temp = airframe_temp_K + gaussian_filter(rng.normal(0, 3, shape), sigma=5)
    fuselage_mask = masks['fuselage'].astype(float)
    fuselage_mask = gaussian_filter(fuselage_mask, sigma=1)
    temp = temp * (1 - fuselage_mask) + fuselage_temp * fuselage_mask
    emissivity = emissivity * (1 - fuselage_mask) + 0.85 * fuselage_mask

    # Nose - slightly warmer (aerodynamic heating)
    nose_temp = airframe_temp_K + 15
    nose_mask = masks['nose'].astype(float)
    nose_mask = gaussian_filter(nose_mask, sigma=1)
    temp = temp * (1 - nose_mask) + nose_temp * nose_mask

    # Canopy - reflects sky (appears cold) or shows cockpit (warm)
    canopy_temp = sky_temp_K + 20  # Mix of sky reflection and interior
    canopy_mask = masks['canopy'].astype(float)
    canopy_mask = gaussian_filter(canopy_mask, sigma=1)
    temp = temp * (1 - canopy_mask) + canopy_temp * canopy_mask
    emissivity = emissivity * (1 - canopy_mask) + 0.6 * canopy_mask  # Glass

    # Wings
    wing_temp = airframe_temp_K - 5
    for wing_key in ['wing_left', 'wing_right']:
        wing_mask = masks[wing_key].astype(float)
        wing_mask = gaussian_filter(wing_mask, sigma=1)
        temp = temp * (1 - wing_mask) + wing_temp * wing_mask

    # Leading edges - warmer (aerodynamic heating)
    # Approximate by eroding wing masks
    from scipy.ndimage import binary_erosion
    for wing_key in ['wing_left', 'wing_right']:
        wing_mask = masks[wing_key]
        le_mask = wing_mask & ~binary_erosion(wing_mask, iterations=3)
        le_mask = gaussian_filter(le_mask.astype(float), sigma=2)
        le_temp = airframe_temp_K + 25
        temp = temp * (1 - le_mask) + le_temp * le_mask

    # Tail surfaces
    for tail_key in ['horizontal_stab', 'vertical_tail']:
        tail_mask = masks[tail_key].astype(float)
        tail_mask = gaussian_filter(tail_mask, sigma=1)
        tail_temp = airframe_temp_K - 8
        temp = temp * (1 - tail_mask) + tail_temp * tail_mask

    # Nozzle - very hot!
    nozzle_temp = exhaust_temp_K * 0.7  # Nozzle slightly cooler than plume core
    nozzle_mask = masks['nozzle'].astype(float)
    nozzle_mask = gaussian_filter(nozzle_mask, sigma=1)
    temp = temp * (1 - nozzle_mask) + nozzle_temp * nozzle_mask
    emissivity = emissivity * (1 - nozzle_mask) + 0.9 * nozzle_mask

    # Exhaust plume - hot with gradient
    plume_mask = masks['exhaust_plume'].astype(float)
    direction = 1 if heading_right else -1
    cy, cx = center

    # Temperature decreases with distance from nozzle
    plume_x = (np.arange(w) - cx) * direction
    plume_dist_norm = np.clip(plume_x / (scale_pixels_per_meter * 12), 0, 1)
    plume_temp_field = exhaust_temp_K * (1 - plume_dist_norm ** 0.5) + sky_temp_K * plume_dist_norm ** 0.5

    # Core is hotter than edges
    y = np.arange(h)[:, np.newaxis]
    plume_core = np.exp(-((y - cy) ** 2) / (scale_pixels_per_meter * 3) ** 2)
    plume_temp_field = plume_temp_field[np.newaxis, :] * plume_core + sky_temp_K * (1 - plume_core)

    plume_mask = gaussian_filter(plume_mask, sigma=2)
    temp = temp * (1 - plume_mask * 0.8) + plume_temp_field * plume_mask * 0.8

    # Add thermal noise
    temp += rng.normal(0, 1, shape)

    return temp, emissivity


def generate_f16_video_frames(
    n_frames: int = 30,
    shape: Tuple[int, int] = (480, 640),
    start_position: Tuple[int, int] = None,
    velocity_pixels_per_frame: Tuple[float, float] = (5.0, 0.0),
    scale_pixels_per_meter: float = 10.0,
    mode: str = "visible",  # "visible" or "thermal"
    seed: Optional[int] = None,
) -> List[NDArray]:
    """Generate video frames of F-16 in flight.

    Args:
        n_frames: Number of frames to generate
        shape: Frame shape
        start_position: Starting position (y, x), defaults to left side
        velocity_pixels_per_frame: Movement per frame (dy, dx)
        scale_pixels_per_meter: Scale factor
        mode: "visible" for RGB or "thermal" for temperature map
        seed: Random seed

    Returns:
        List of frames (RGB or grayscale depending on mode)
    """
    h, w = shape

    if start_position is None:
        # Start from left side, heading right
        start_position = (h // 2, -50)

    frames = []

    for i in range(n_frames):
        # Current position
        cy = int(start_position[0] + i * velocity_pixels_per_frame[0])
        cx = int(start_position[1] + i * velocity_pixels_per_frame[1])

        # Ensure aircraft is at least partially visible
        if cx < -100 or cx > w + 100:
            continue

        if mode == "visible":
            rgb, _, _ = render_f16_visible(
                shape=shape,
                center=(cy, cx),
                scale_pixels_per_meter=scale_pixels_per_meter,
                heading_right=True,
                seed=seed + i if seed else None,
            )
            frames.append((rgb * 255).astype(np.uint8))
        else:
            temp, _ = render_f16_thermal_detailed(
                shape=shape,
                center=(cy, cx),
                scale_pixels_per_meter=scale_pixels_per_meter,
                heading_right=True,
                seed=seed + i if seed else None,
            )
            frames.append(temp)

    return frames


def save_video(
    frames: List[NDArray],
    output_path: str,
    fps: int = 15,
) -> None:
    """Save frames as video file.

    Args:
        frames: List of frames (RGB uint8 or grayscale float)
        output_path: Output file path (.mp4 or .avi)
        fps: Frames per second
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV (cv2) required for video output. Install with: pip install opencv-python")
        # Fallback: save as individual PNGs
        from PIL import Image
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            if frame.ndim == 2:
                # Normalize thermal to 8-bit
                frame_norm = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
                img = Image.fromarray(frame_norm, mode='L')
            else:
                img = Image.fromarray(frame)
            img.save(output_dir / f"frame_{i:04d}.png")
        print(f"Saved {len(frames)} frames to {output_dir}")
        return

    # Determine codec and output format
    output_path = str(output_path)
    if output_path.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Get frame dimensions
    if frames[0].ndim == 3:
        h, w, _ = frames[0].shape
        is_color = True
    else:
        h, w = frames[0].shape
        is_color = False

    # Create video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), is_color)

    for frame in frames:
        if frame.ndim == 2:
            # Thermal - normalize and convert to BGR
            frame_norm = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
            frame_bgr = cv2.applyColorMap(frame_norm, cv2.COLORMAP_INFERNO)
        else:
            # RGB to BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Saved video: {output_path}")
