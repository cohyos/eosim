"""
EOSIM Object Library.

Provides a collection of 3D object models for EO/IR simulation including:
- Aircraft (fighters, bombers, helicopters, UAVs, commercial)
- Ships (naval vessels, cargo, patrol boats)
- Ground vehicles (tanks, APCs, trucks, cars)
- Missiles and rockets
- Launchers (SAM, artillery, MLRS)
- Personnel (soldiers, civilians)
- Buildings and structures

Each object includes:
- Geometric dimensions
- Thermal signature characteristics
- Emissivity properties
- Hot spot definitions
- Aspect-dependent signatures
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any, Callable
import numpy as np
from numpy.typing import NDArray


class ObjectCategory(Enum):
    """Categories of objects in the library."""
    AIRCRAFT = "aircraft"
    HELICOPTER = "helicopter"
    UAV = "uav"
    SHIP = "ship"
    VEHICLE = "vehicle"
    TANK = "tank"
    MISSILE = "missile"
    LAUNCHER = "launcher"
    PERSON = "person"
    BUILDING = "building"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class ThermalProfile:
    """Thermal signature profile for an object.

    Attributes:
        base_temperature_k: Nominal surface temperature
        emissivity: Surface emissivity (0-1)
        hot_spots: List of (name, rel_pos, size, delta_T, shape) tuples
        aspect_variation: Temperature variation with aspect angle
        altitude_correction: Temperature change per km altitude
        speed_correction: Temperature change per 100 m/s
    """
    base_temperature_k: float = 300.0
    emissivity: float = 0.85
    hot_spots: List[Tuple[str, Tuple[float, float], float, float, str]] = field(default_factory=list)
    aspect_variation: Dict[str, float] = field(default_factory=dict)
    altitude_correction_k_per_km: float = -6.5  # Lapse rate
    speed_correction_k_per_100ms: float = 5.0   # Aerodynamic heating


@dataclass
class GeometricDimensions:
    """Physical dimensions of an object.

    Attributes:
        length_m: Length in meters
        width_m: Width/wingspan in meters
        height_m: Height in meters
        rcs_m2: Radar cross section (for reference)
    """
    length_m: float
    width_m: float
    height_m: float
    rcs_m2: Optional[float] = None

    @property
    def frontal_area_m2(self) -> float:
        """Approximate frontal area."""
        return self.width_m * self.height_m * 0.7

    @property
    def side_area_m2(self) -> float:
        """Approximate side area."""
        return self.length_m * self.height_m * 0.8

    @property
    def top_area_m2(self) -> float:
        """Approximate top-down area."""
        return self.length_m * self.width_m * 0.7


@dataclass
class Object3D:
    """Base class for 3D objects in the library.

    Attributes:
        id: Unique identifier
        name: Display name
        category: Object category
        dimensions: Physical dimensions
        thermal: Thermal signature profile
        description: Detailed description
        source: Data source/reference
        silhouette_func: Function to generate silhouette
    """
    id: str
    name: str
    category: ObjectCategory
    dimensions: GeometricDimensions
    thermal: ThermalProfile
    description: str = ""
    source: str = "estimated"
    country: str = "unknown"
    silhouette_func: Optional[Callable] = None

    def get_signature(
        self,
        resolution: Tuple[int, int] = (64, 64),
        aspect_angle_deg: float = 0.0,
        elevation_angle_deg: float = 0.0,
        speed_ms: float = 0.0,
        altitude_km: float = 0.0,
    ) -> Tuple[NDArray, NDArray]:
        """Generate thermal signature at given viewing angle.

        Args:
            resolution: Output resolution (height, width)
            aspect_angle_deg: Horizontal viewing angle (0=front, 90=side, 180=rear)
            elevation_angle_deg: Vertical viewing angle
            speed_ms: Object speed in m/s
            altitude_km: Altitude in km

        Returns:
            Tuple of (temperature_map, emissivity_map)
        """
        h, w = resolution

        # Create base shape
        temp_map = np.zeros((h, w), dtype=np.float64)
        emis_map = np.zeros((h, w), dtype=np.float64)

        # Use silhouette function if available
        if self.silhouette_func is not None:
            mask = self.silhouette_func(resolution, aspect_angle_deg)
        else:
            mask = self._create_default_silhouette(resolution, aspect_angle_deg)

        # Compute base temperature with corrections
        base_temp = self.thermal.base_temperature_k
        base_temp += altitude_km * self.thermal.altitude_correction_k_per_km
        base_temp += (speed_ms / 100) * self.thermal.speed_correction_k_per_100ms

        # Apply aspect-dependent variation
        aspect_key = self._get_aspect_key(aspect_angle_deg)
        if aspect_key in self.thermal.aspect_variation:
            base_temp += self.thermal.aspect_variation[aspect_key]

        # Fill base temperature
        temp_map[mask] = base_temp
        emis_map[mask] = self.thermal.emissivity

        # Apply hot spots
        for name, rel_pos, rel_size, delta_t, shape in self.thermal.hot_spots:
            self._apply_hot_spot(
                temp_map, mask, rel_pos, rel_size,
                base_temp + delta_t, shape
            )

        return temp_map, emis_map

    def _create_default_silhouette(
        self,
        resolution: Tuple[int, int],
        aspect_angle_deg: float,
    ) -> NDArray:
        """Create default rectangular silhouette."""
        h, w = resolution
        mask = np.zeros((h, w), dtype=bool)

        # Simple rectangular mask with margins
        margin = 0.15
        y0 = int(h * margin)
        y1 = int(h * (1 - margin))
        x0 = int(w * margin)
        x1 = int(w * (1 - margin))

        mask[y0:y1, x0:x1] = True
        return mask

    def _get_aspect_key(self, angle_deg: float) -> str:
        """Get aspect key from angle."""
        angle = angle_deg % 360
        if angle < 45 or angle >= 315:
            return "front"
        elif angle < 135:
            return "right"
        elif angle < 225:
            return "rear"
        else:
            return "left"

    def _apply_hot_spot(
        self,
        temp_map: NDArray,
        mask: NDArray,
        rel_pos: Tuple[float, float],
        rel_size: float,
        temperature: float,
        shape: str,
    ) -> None:
        """Apply a hot spot to the temperature map."""
        h, w = temp_map.shape
        cy = int(rel_pos[0] * h)
        cx = int(rel_pos[1] * w)
        radius = int(rel_size * min(h, w) / 2)
        radius = max(1, radius)

        yy, xx = np.ogrid[:h, :w]
        if shape == "circle":
            spot_mask = (yy - cy)**2 + (xx - cx)**2 <= radius**2
        elif shape == "ellipse_h":
            spot_mask = ((yy - cy) / radius)**2 + ((xx - cx) / (radius * 2))**2 <= 1
        elif shape == "ellipse_v":
            spot_mask = ((yy - cy) / (radius * 2))**2 + ((xx - cx) / radius)**2 <= 1
        else:  # rectangle
            spot_mask = (np.abs(yy - cy) <= radius) & (np.abs(xx - cx) <= radius)

        # Only apply within object mask
        combined_mask = spot_mask & mask
        temp_map[combined_mask] = temperature


# =============================================================================
# Aircraft Objects
# =============================================================================

@dataclass
class AircraftObject(Object3D):
    """Aircraft-specific object with additional properties."""
    max_speed_ms: float = 0.0
    cruise_speed_ms: float = 0.0
    ceiling_m: float = 0.0
    engine_type: str = "jet"
    num_engines: int = 1


def _create_f16_silhouette(resolution: Tuple[int, int], aspect_deg: float) -> NDArray:
    """Create F-16 silhouette."""
    h, w = resolution
    mask = np.zeros((h, w), dtype=bool)

    # Center coordinates
    cy, cx = h // 2, w // 2

    # Aspect adjustment (simplified)
    if 45 < aspect_deg < 135 or 225 < aspect_deg < 315:
        # Side view
        # Fuselage
        fuse_h = int(h * 0.12)
        fuse_w = int(w * 0.7)
        mask[cy - fuse_h:cy + fuse_h, cx - fuse_w//2:cx + fuse_w//2] = True

        # Tail
        tail_w = int(w * 0.15)
        tail_h = int(h * 0.25)
        mask[cy - tail_h:cy, cx + fuse_w//2 - tail_w:cx + fuse_w//2 + 5] = True

        # Canopy
        canopy_h = int(h * 0.08)
        canopy_w = int(w * 0.15)
        mask[cy - fuse_h - canopy_h:cy - fuse_h, cx - fuse_w//4:cx - fuse_w//4 + canopy_w] = True
    else:
        # Front/rear view
        # Fuselage
        fuse_r = int(min(h, w) * 0.1)
        yy, xx = np.ogrid[:h, :w]
        mask = (yy - cy)**2 + (xx - cx)**2 <= fuse_r**2

        # Wings
        wing_span = int(w * 0.4)
        wing_h = int(h * 0.05)
        mask[cy - wing_h:cy + wing_h, cx - wing_span:cx + wing_span] = True

        # Tail
        tail_h = int(h * 0.2)
        tail_w = int(w * 0.03)
        mask[cy - tail_h:cy, cx - tail_w:cx + tail_w] = True

    return mask


def _create_f35_silhouette(resolution: Tuple[int, int], aspect_deg: float) -> NDArray:
    """Create F-35 silhouette."""
    h, w = resolution
    mask = np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2

    if 45 < aspect_deg < 135 or 225 < aspect_deg < 315:
        # Side view - stealthier profile
        fuse_h = int(h * 0.1)
        fuse_w = int(w * 0.65)
        mask[cy - fuse_h:cy + fuse_h, cx - fuse_w//2:cx + fuse_w//2] = True

        # Angled tail
        for i in range(int(h * 0.2)):
            tw = int(w * 0.08) - i // 2
            if tw > 0:
                mask[cy - fuse_h - i, cx + fuse_w//2 - tw:cx + fuse_w//2] = True
    else:
        # Front view
        fuse_r = int(min(h, w) * 0.12)
        yy, xx = np.ogrid[:h, :w]
        mask = (yy - cy)**2 + (xx - cx)**2 <= fuse_r**2

        wing_span = int(w * 0.35)
        wing_h = int(h * 0.04)
        mask[cy - wing_h:cy + wing_h, cx - wing_span:cx + wing_span] = True

    return mask


def _create_helicopter_silhouette(resolution: Tuple[int, int], aspect_deg: float) -> NDArray:
    """Create generic helicopter silhouette."""
    h, w = resolution
    mask = np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2

    # Main body
    body_h = int(h * 0.2)
    body_w = int(w * 0.4)
    mask[cy - body_h//2:cy + body_h//2, cx - body_w//2:cx + body_w//2] = True

    # Tail boom
    boom_h = int(h * 0.08)
    boom_w = int(w * 0.35)
    mask[cy - boom_h//2:cy + boom_h//2, cx + body_w//2:cx + body_w//2 + boom_w] = True

    # Main rotor disc (top view aspect)
    if abs(aspect_deg - 90) < 30 or abs(aspect_deg - 270) < 30:
        rotor_r = int(min(h, w) * 0.4)
        yy, xx = np.ogrid[:h, :w]
        rotor_mask = (yy - (cy - body_h//2))**2 + (xx - cx)**2 <= rotor_r**2
        # Thin rotor in side view
        rotor_mask &= np.abs(yy - (cy - body_h//2)) < 3
        mask |= rotor_mask

    return mask


# =============================================================================
# Ship Objects
# =============================================================================

@dataclass
class ShipObject(Object3D):
    """Ship-specific object."""
    displacement_tons: float = 0.0
    max_speed_knots: float = 0.0
    draft_m: float = 0.0


def _create_destroyer_silhouette(resolution: Tuple[int, int], aspect_deg: float) -> NDArray:
    """Create destroyer silhouette."""
    h, w = resolution
    mask = np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2

    # Hull
    hull_h = int(h * 0.15)
    hull_w = int(w * 0.75)
    mask[cy:cy + hull_h, cx - hull_w//2:cx + hull_w//2] = True

    # Superstructure
    super_h = int(h * 0.2)
    super_w = int(w * 0.25)
    mask[cy - super_h:cy, cx - super_w//2:cx + super_w//2] = True

    # Bridge/mast
    mast_h = int(h * 0.15)
    mast_w = int(w * 0.05)
    mask[cy - super_h - mast_h:cy - super_h, cx - mast_w:cx + mast_w] = True

    # Bow taper
    for i in range(int(w * 0.1)):
        taper = i // 2
        mask[cy + taper:cy + hull_h - taper, cx - hull_w//2 - i] = False

    return mask


def _create_carrier_silhouette(resolution: Tuple[int, int], aspect_deg: float) -> NDArray:
    """Create aircraft carrier silhouette."""
    h, w = resolution
    mask = np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2

    # Flight deck
    deck_h = int(h * 0.4)
    deck_w = int(w * 0.85)
    mask[cy - deck_h//4:cy + deck_h, cx - deck_w//2:cx + deck_w//2] = True

    # Island superstructure
    island_h = int(h * 0.25)
    island_w = int(w * 0.1)
    island_x = cx + int(w * 0.25)
    mask[cy - deck_h//4 - island_h:cy - deck_h//4, island_x:island_x + island_w] = True

    return mask


# =============================================================================
# Vehicle Objects
# =============================================================================

@dataclass
class VehicleObject(Object3D):
    """Ground vehicle object."""
    max_speed_kmh: float = 0.0
    armor_mm: float = 0.0
    crew: int = 0


def _create_tank_silhouette(resolution: Tuple[int, int], aspect_deg: float) -> NDArray:
    """Create tank silhouette."""
    h, w = resolution
    mask = np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2

    # Hull
    hull_h = int(h * 0.3)
    hull_w = int(w * 0.6)
    mask[cy:cy + hull_h, cx - hull_w//2:cx + hull_w//2] = True

    # Turret
    turret_r = int(min(h, w) * 0.15)
    yy, xx = np.ogrid[:h, :w]
    turret_mask = (yy - cy)**2 + (xx - cx)**2 <= turret_r**2
    mask |= turret_mask

    # Gun barrel
    barrel_h = int(h * 0.04)
    barrel_w = int(w * 0.25)
    mask[cy - barrel_h:cy + barrel_h, cx - turret_r - barrel_w:cx - turret_r] = True

    # Tracks
    track_h = int(h * 0.08)
    mask[cy + hull_h - track_h:cy + hull_h + track_h, cx - hull_w//2:cx + hull_w//2] = True

    return mask


def _create_truck_silhouette(resolution: Tuple[int, int], aspect_deg: float) -> NDArray:
    """Create military truck silhouette."""
    h, w = resolution
    mask = np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2

    # Cab
    cab_h = int(h * 0.25)
    cab_w = int(w * 0.2)
    mask[cy - cab_h:cy + int(h*0.1), cx - w//3:cx - w//3 + cab_w] = True

    # Cargo bed
    bed_h = int(h * 0.2)
    bed_w = int(w * 0.5)
    mask[cy - bed_h:cy + int(h*0.12), cx - w//3 + cab_w:cx - w//3 + cab_w + bed_w] = True

    # Wheels
    wheel_r = int(min(h, w) * 0.08)
    yy, xx = np.ogrid[:h, :w]
    for wx in [cx - w//4, cx, cx + w//4]:
        wheel_mask = (yy - (cy + int(h*0.1)))**2 + (xx - wx)**2 <= wheel_r**2
        mask |= wheel_mask

    return mask


# =============================================================================
# Missile Objects
# =============================================================================

@dataclass
class MissileObject(Object3D):
    """Missile/rocket object."""
    max_speed_mach: float = 0.0
    range_km: float = 0.0
    warhead_kg: float = 0.0
    guidance: str = "unguided"


def _create_missile_silhouette(resolution: Tuple[int, int], aspect_deg: float) -> NDArray:
    """Create missile silhouette."""
    h, w = resolution
    mask = np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2

    # Body
    body_r = int(min(h, w) * 0.08)
    body_w = int(w * 0.6)

    yy, xx = np.ogrid[:h, :w]
    # Cylindrical body
    body_mask = (np.abs(yy - cy) <= body_r) & (np.abs(xx - cx) <= body_w // 2)
    mask |= body_mask

    # Nose cone
    for i in range(int(w * 0.15)):
        r = body_r - (i * body_r) // int(w * 0.15)
        if r > 0:
            mask[cy - r:cy + r, cx - body_w//2 - i] = True

    # Fins
    fin_h = int(h * 0.15)
    fin_w = int(w * 0.05)
    for dy in [-1, 1]:
        mask[cy + dy * body_r:cy + dy * (body_r + fin_h),
             cx + body_w//2 - fin_w:cx + body_w//2 + fin_w] = True

    return mask


# =============================================================================
# Launcher Objects
# =============================================================================

@dataclass
class LauncherObject(Object3D):
    """Missile launcher object."""
    num_tubes: int = 1
    reload_time_s: float = 0.0
    traverse_deg: float = 360.0
    elevation_deg: Tuple[float, float] = (0.0, 90.0)


def _create_sam_launcher_silhouette(resolution: Tuple[int, int], aspect_deg: float) -> NDArray:
    """Create SAM launcher silhouette."""
    h, w = resolution
    mask = np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2

    # Vehicle base
    base_h = int(h * 0.2)
    base_w = int(w * 0.5)
    mask[cy + int(h*0.1):cy + int(h*0.1) + base_h, cx - base_w//2:cx + base_w//2] = True

    # Launcher rail (angled)
    rail_h = int(h * 0.35)
    rail_w = int(w * 0.08)
    for i in range(rail_h):
        offset = i // 2
        mask[cy + int(h*0.1) - i, cx - rail_w//2 + offset:cx + rail_w//2 + offset] = True

    # Missiles on rail
    missile_w = int(w * 0.25)
    missile_h = int(h * 0.05)
    for i in range(3):
        my = cy - int(h * 0.1) - i * (missile_h + 5)
        mx = cx + i * 3
        mask[my:my + missile_h, mx:mx + missile_w] = True

    return mask


# =============================================================================
# Person Objects
# =============================================================================

@dataclass
class PersonObject(Object3D):
    """Human figure object."""
    posture: str = "standing"  # standing, crouching, prone
    equipment_kg: float = 0.0


def _create_person_silhouette(resolution: Tuple[int, int], aspect_deg: float) -> NDArray:
    """Create person silhouette."""
    h, w = resolution
    mask = np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2

    # Head
    head_r = int(min(h, w) * 0.08)
    yy, xx = np.ogrid[:h, :w]
    head_y = cy - int(h * 0.25)
    head_mask = (yy - head_y)**2 + (xx - cx)**2 <= head_r**2
    mask |= head_mask

    # Torso
    torso_h = int(h * 0.25)
    torso_w = int(w * 0.2)
    mask[head_y + head_r:head_y + head_r + torso_h, cx - torso_w//2:cx + torso_w//2] = True

    # Legs
    leg_h = int(h * 0.3)
    leg_w = int(w * 0.07)
    for dx in [-1, 1]:
        lx = cx + dx * int(w * 0.05)
        mask[head_y + head_r + torso_h:head_y + head_r + torso_h + leg_h,
             lx - leg_w//2:lx + leg_w//2] = True

    # Arms
    arm_h = int(h * 0.2)
    arm_w = int(w * 0.05)
    for dx in [-1, 1]:
        ax = cx + dx * (torso_w//2 + arm_w//2)
        mask[head_y + head_r:head_y + head_r + arm_h, ax - arm_w//2:ax + arm_w//2] = True

    return mask


# =============================================================================
# Object Library Database
# =============================================================================

# Aircraft
AIRCRAFT_OBJECTS = {
    # US Aircraft
    "f16": AircraftObject(
        id="f16",
        name="F-16 Fighting Falcon",
        category=ObjectCategory.AIRCRAFT,
        dimensions=GeometricDimensions(15.06, 9.96, 4.88, 1.2),
        thermal=ThermalProfile(
            base_temperature_k=280.0,
            emissivity=0.85,
            hot_spots=[
                ("exhaust", (0.5, 0.9), 0.15, 300, "ellipse_h"),
                ("engine_bay", (0.5, 0.75), 0.12, 80, "ellipse_h"),
                ("leading_edge", (0.5, 0.2), 0.08, 30, "ellipse_h"),
            ],
            aspect_variation={"rear": 100, "front": -20, "left": 50, "right": 50},
        ),
        description="Single-engine multirole fighter aircraft",
        country="USA",
        max_speed_ms=600,
        cruise_speed_ms=250,
        ceiling_m=15240,
        engine_type="turbofan",
        num_engines=1,
        silhouette_func=_create_f16_silhouette,
    ),
    "f35": AircraftObject(
        id="f35",
        name="F-35 Lightning II",
        category=ObjectCategory.AIRCRAFT,
        dimensions=GeometricDimensions(15.67, 10.67, 4.38, 0.005),
        thermal=ThermalProfile(
            base_temperature_k=275.0,
            emissivity=0.75,  # Lower due to RAM coating
            hot_spots=[
                ("exhaust", (0.5, 0.92), 0.12, 250, "ellipse_h"),
                ("engine_bay", (0.5, 0.78), 0.1, 60, "ellipse_h"),
            ],
            aspect_variation={"rear": 80, "front": -30},
        ),
        description="Fifth-generation stealth multirole fighter",
        country="USA",
        max_speed_ms=535,
        cruise_speed_ms=250,
        ceiling_m=15240,
        engine_type="turbofan",
        num_engines=1,
        silhouette_func=_create_f35_silhouette,
    ),
    "f15": AircraftObject(
        id="f15",
        name="F-15 Eagle",
        category=ObjectCategory.AIRCRAFT,
        dimensions=GeometricDimensions(19.43, 13.05, 5.63, 10.0),
        thermal=ThermalProfile(
            base_temperature_k=285.0,
            emissivity=0.85,
            hot_spots=[
                ("exhaust_l", (0.45, 0.9), 0.12, 320, "ellipse_h"),
                ("exhaust_r", (0.55, 0.9), 0.12, 320, "ellipse_h"),
                ("engine_bay", (0.5, 0.75), 0.15, 100, "ellipse_h"),
            ],
        ),
        description="Twin-engine air superiority fighter",
        country="USA",
        max_speed_ms=720,
        cruise_speed_ms=280,
        ceiling_m=20000,
        engine_type="turbofan",
        num_engines=2,
    ),
    "f18": AircraftObject(
        id="f18",
        name="F/A-18 Super Hornet",
        category=ObjectCategory.AIRCRAFT,
        dimensions=GeometricDimensions(18.38, 13.62, 4.88, 1.0),
        thermal=ThermalProfile(
            base_temperature_k=282.0,
            emissivity=0.85,
            hot_spots=[
                ("exhaust_l", (0.45, 0.88), 0.11, 290, "ellipse_h"),
                ("exhaust_r", (0.55, 0.88), 0.11, 290, "ellipse_h"),
            ],
        ),
        description="Twin-engine carrier-capable multirole fighter",
        country="USA",
        max_speed_ms=540,
        cruise_speed_ms=250,
        ceiling_m=15240,
        engine_type="turbofan",
        num_engines=2,
    ),
    "b2": AircraftObject(
        id="b2",
        name="B-2 Spirit",
        category=ObjectCategory.AIRCRAFT,
        dimensions=GeometricDimensions(21.03, 52.43, 5.18, 0.0001),
        thermal=ThermalProfile(
            base_temperature_k=270.0,
            emissivity=0.65,  # Stealth coating
            hot_spots=[
                ("exhaust", (0.5, 0.85), 0.15, 150, "ellipse_h"),
            ],
        ),
        description="Stealth strategic bomber",
        country="USA",
        max_speed_ms=280,
        cruise_speed_ms=250,
        ceiling_m=15200,
        engine_type="turbofan",
        num_engines=4,
    ),
    "c130": AircraftObject(
        id="c130",
        name="C-130 Hercules",
        category=ObjectCategory.AIRCRAFT,
        dimensions=GeometricDimensions(29.79, 40.41, 11.66, 80.0),
        thermal=ThermalProfile(
            base_temperature_k=290.0,
            emissivity=0.88,
            hot_spots=[
                ("engine_1", (0.35, 0.35), 0.08, 200, "circle"),
                ("engine_2", (0.35, 0.45), 0.08, 200, "circle"),
                ("engine_3", (0.65, 0.45), 0.08, 200, "circle"),
                ("engine_4", (0.65, 0.35), 0.08, 200, "circle"),
            ],
        ),
        description="Four-engine turboprop military transport",
        country="USA",
        max_speed_ms=180,
        cruise_speed_ms=150,
        ceiling_m=10060,
        engine_type="turboprop",
        num_engines=4,
    ),

    # Russian Aircraft
    "su27": AircraftObject(
        id="su27",
        name="Su-27 Flanker",
        category=ObjectCategory.AIRCRAFT,
        dimensions=GeometricDimensions(21.94, 14.70, 5.93, 10.0),
        thermal=ThermalProfile(
            base_temperature_k=288.0,
            emissivity=0.87,
            hot_spots=[
                ("exhaust_l", (0.45, 0.92), 0.13, 350, "ellipse_h"),
                ("exhaust_r", (0.55, 0.92), 0.13, 350, "ellipse_h"),
            ],
        ),
        description="Twin-engine supermaneuverable fighter",
        country="Russia",
        max_speed_ms=700,
        cruise_speed_ms=280,
        ceiling_m=18500,
        engine_type="turbofan",
        num_engines=2,
    ),
    "su35": AircraftObject(
        id="su35",
        name="Su-35 Flanker-E",
        category=ObjectCategory.AIRCRAFT,
        dimensions=GeometricDimensions(21.90, 15.30, 5.90, 2.0),
        thermal=ThermalProfile(
            base_temperature_k=285.0,
            emissivity=0.85,
            hot_spots=[
                ("exhaust_l", (0.45, 0.91), 0.12, 330, "ellipse_h"),
                ("exhaust_r", (0.55, 0.91), 0.12, 330, "ellipse_h"),
            ],
        ),
        description="4++ generation multirole fighter",
        country="Russia",
        max_speed_ms=720,
        cruise_speed_ms=300,
        ceiling_m=18000,
        engine_type="turbofan",
        num_engines=2,
    ),
    "mig29": AircraftObject(
        id="mig29",
        name="MiG-29 Fulcrum",
        category=ObjectCategory.AIRCRAFT,
        dimensions=GeometricDimensions(17.37, 11.36, 4.73, 3.0),
        thermal=ThermalProfile(
            base_temperature_k=283.0,
            emissivity=0.86,
            hot_spots=[
                ("exhaust_l", (0.45, 0.9), 0.11, 310, "ellipse_h"),
                ("exhaust_r", (0.55, 0.9), 0.11, 310, "ellipse_h"),
            ],
        ),
        description="Twin-engine air superiority fighter",
        country="Russia",
        max_speed_ms=700,
        cruise_speed_ms=280,
        ceiling_m=18000,
        engine_type="turbofan",
        num_engines=2,
    ),

    # European Aircraft
    "eurofighter": AircraftObject(
        id="eurofighter",
        name="Eurofighter Typhoon",
        category=ObjectCategory.AIRCRAFT,
        dimensions=GeometricDimensions(15.96, 10.95, 5.28, 1.0),
        thermal=ThermalProfile(
            base_temperature_k=280.0,
            emissivity=0.84,
            hot_spots=[
                ("exhaust_l", (0.45, 0.9), 0.11, 300, "ellipse_h"),
                ("exhaust_r", (0.55, 0.9), 0.11, 300, "ellipse_h"),
            ],
        ),
        description="Twin-engine canard-delta fighter",
        country="Europe",
        max_speed_ms=680,
        cruise_speed_ms=280,
        ceiling_m=19812,
        engine_type="turbofan",
        num_engines=2,
    ),
    "rafale": AircraftObject(
        id="rafale",
        name="Dassault Rafale",
        category=ObjectCategory.AIRCRAFT,
        dimensions=GeometricDimensions(15.27, 10.80, 5.34, 1.0),
        thermal=ThermalProfile(
            base_temperature_k=278.0,
            emissivity=0.83,
            hot_spots=[
                ("exhaust_l", (0.45, 0.89), 0.10, 280, "ellipse_h"),
                ("exhaust_r", (0.55, 0.89), 0.10, 280, "ellipse_h"),
            ],
        ),
        description="Twin-engine multirole fighter",
        country="France",
        max_speed_ms=560,
        cruise_speed_ms=260,
        ceiling_m=15240,
        engine_type="turbofan",
        num_engines=2,
    ),

    # Chinese Aircraft
    "j20": AircraftObject(
        id="j20",
        name="Chengdu J-20",
        category=ObjectCategory.AIRCRAFT,
        dimensions=GeometricDimensions(20.30, 13.50, 4.45, 0.01),
        thermal=ThermalProfile(
            base_temperature_k=275.0,
            emissivity=0.72,
            hot_spots=[
                ("exhaust_l", (0.45, 0.93), 0.11, 270, "ellipse_h"),
                ("exhaust_r", (0.55, 0.93), 0.11, 270, "ellipse_h"),
            ],
        ),
        description="Fifth-generation stealth fighter",
        country="China",
        max_speed_ms=680,
        cruise_speed_ms=300,
        ceiling_m=20000,
        engine_type="turbofan",
        num_engines=2,
    ),
}

# Helicopters
HELICOPTER_OBJECTS = {
    "ah64": AircraftObject(
        id="ah64",
        name="AH-64 Apache",
        category=ObjectCategory.HELICOPTER,
        dimensions=GeometricDimensions(17.73, 14.63, 4.66, 5.0),
        thermal=ThermalProfile(
            base_temperature_k=300.0,
            emissivity=0.88,
            hot_spots=[
                ("exhaust_l", (0.4, 0.55), 0.1, 250, "circle"),
                ("exhaust_r", (0.6, 0.55), 0.1, 250, "circle"),
                ("engine_l", (0.4, 0.5), 0.08, 150, "circle"),
                ("engine_r", (0.6, 0.5), 0.08, 150, "circle"),
            ],
        ),
        description="Twin-engine attack helicopter",
        country="USA",
        max_speed_ms=80,
        cruise_speed_ms=55,
        ceiling_m=6400,
        engine_type="turboshaft",
        num_engines=2,
        silhouette_func=_create_helicopter_silhouette,
    ),
    "uh60": AircraftObject(
        id="uh60",
        name="UH-60 Black Hawk",
        category=ObjectCategory.HELICOPTER,
        dimensions=GeometricDimensions(19.76, 16.36, 5.13, 6.0),
        thermal=ThermalProfile(
            base_temperature_k=298.0,
            emissivity=0.87,
            hot_spots=[
                ("exhaust_l", (0.4, 0.6), 0.09, 220, "circle"),
                ("exhaust_r", (0.6, 0.6), 0.09, 220, "circle"),
            ],
        ),
        description="Medium-lift utility helicopter",
        country="USA",
        max_speed_ms=80,
        cruise_speed_ms=60,
        ceiling_m=5790,
        engine_type="turboshaft",
        num_engines=2,
        silhouette_func=_create_helicopter_silhouette,
    ),
    "mi24": AircraftObject(
        id="mi24",
        name="Mi-24 Hind",
        category=ObjectCategory.HELICOPTER,
        dimensions=GeometricDimensions(21.35, 17.30, 6.50, 15.0),
        thermal=ThermalProfile(
            base_temperature_k=305.0,
            emissivity=0.89,
            hot_spots=[
                ("exhaust_l", (0.35, 0.45), 0.12, 280, "circle"),
                ("exhaust_r", (0.65, 0.45), 0.12, 280, "circle"),
            ],
        ),
        description="Large attack helicopter",
        country="Russia",
        max_speed_ms=83,
        cruise_speed_ms=60,
        ceiling_m=4500,
        engine_type="turboshaft",
        num_engines=2,
        silhouette_func=_create_helicopter_silhouette,
    ),
    "ka52": AircraftObject(
        id="ka52",
        name="Ka-52 Alligator",
        category=ObjectCategory.HELICOPTER,
        dimensions=GeometricDimensions(16.00, 14.50, 4.93, 8.0),
        thermal=ThermalProfile(
            base_temperature_k=302.0,
            emissivity=0.86,
            hot_spots=[
                ("exhaust_l", (0.4, 0.5), 0.1, 260, "circle"),
                ("exhaust_r", (0.6, 0.5), 0.1, 260, "circle"),
            ],
        ),
        description="Coaxial rotor attack helicopter",
        country="Russia",
        max_speed_ms=83,
        cruise_speed_ms=60,
        ceiling_m=5500,
        engine_type="turboshaft",
        num_engines=2,
    ),
}

# UAVs
UAV_OBJECTS = {
    "mq9": AircraftObject(
        id="mq9",
        name="MQ-9 Reaper",
        category=ObjectCategory.UAV,
        dimensions=GeometricDimensions(11.00, 20.12, 3.81, 1.0),
        thermal=ThermalProfile(
            base_temperature_k=290.0,
            emissivity=0.82,
            hot_spots=[
                ("exhaust", (0.5, 0.85), 0.1, 150, "ellipse_h"),
                ("engine", (0.5, 0.75), 0.08, 80, "circle"),
            ],
        ),
        description="Medium-altitude long-endurance UAV",
        country="USA",
        max_speed_ms=130,
        cruise_speed_ms=85,
        ceiling_m=15240,
        engine_type="turboprop",
        num_engines=1,
    ),
    "rq4": AircraftObject(
        id="rq4",
        name="RQ-4 Global Hawk",
        category=ObjectCategory.UAV,
        dimensions=GeometricDimensions(14.50, 39.90, 4.70, 0.1),
        thermal=ThermalProfile(
            base_temperature_k=275.0,
            emissivity=0.80,
            hot_spots=[
                ("exhaust", (0.5, 0.9), 0.12, 120, "ellipse_h"),
            ],
        ),
        description="High-altitude surveillance UAV",
        country="USA",
        max_speed_ms=175,
        cruise_speed_ms=155,
        ceiling_m=18300,
        engine_type="turbofan",
        num_engines=1,
    ),
    "bayraktar_tb2": AircraftObject(
        id="bayraktar_tb2",
        name="Bayraktar TB2",
        category=ObjectCategory.UAV,
        dimensions=GeometricDimensions(6.50, 12.00, 2.20, 0.3),
        thermal=ThermalProfile(
            base_temperature_k=295.0,
            emissivity=0.84,
            hot_spots=[
                ("exhaust", (0.5, 0.8), 0.08, 100, "circle"),
            ],
        ),
        description="Medium-altitude tactical UAV",
        country="Turkey",
        max_speed_ms=65,
        cruise_speed_ms=40,
        ceiling_m=8200,
        engine_type="piston",
        num_engines=1,
    ),
    "shahed136": AircraftObject(
        id="shahed136",
        name="Shahed-136",
        category=ObjectCategory.UAV,
        dimensions=GeometricDimensions(3.50, 2.50, 0.80, 0.05),
        thermal=ThermalProfile(
            base_temperature_k=305.0,
            emissivity=0.88,
            hot_spots=[
                ("engine", (0.5, 0.7), 0.15, 120, "circle"),
            ],
        ),
        description="Loitering munition / kamikaze drone",
        country="Iran",
        max_speed_ms=50,
        cruise_speed_ms=40,
        ceiling_m=4000,
        engine_type="piston",
        num_engines=1,
    ),
}

# Ships
SHIP_OBJECTS = {
    "arleigh_burke": ShipObject(
        id="arleigh_burke",
        name="Arleigh Burke-class Destroyer",
        category=ObjectCategory.SHIP,
        dimensions=GeometricDimensions(155.0, 20.0, 44.0, 5000.0),
        thermal=ThermalProfile(
            base_temperature_k=295.0,
            emissivity=0.90,
            hot_spots=[
                ("funnel", (0.35, 0.45), 0.08, 80, "rectangle"),
                ("engine_exhaust", (0.5, 0.5), 0.1, 120, "ellipse_h"),
            ],
        ),
        description="Guided missile destroyer",
        country="USA",
        displacement_tons=9200,
        max_speed_knots=30,
        draft_m=9.4,
        silhouette_func=_create_destroyer_silhouette,
    ),
    "nimitz": ShipObject(
        id="nimitz",
        name="Nimitz-class Carrier",
        category=ObjectCategory.SHIP,
        dimensions=GeometricDimensions(333.0, 77.0, 74.0, 100000.0),
        thermal=ThermalProfile(
            base_temperature_k=298.0,
            emissivity=0.88,
            hot_spots=[
                ("island", (0.3, 0.65), 0.1, 50, "rectangle"),
                ("catapults", (0.5, 0.2), 0.12, 40, "ellipse_h"),
            ],
        ),
        description="Nuclear-powered aircraft carrier",
        country="USA",
        displacement_tons=100000,
        max_speed_knots=30,
        draft_m=11.9,
        silhouette_func=_create_carrier_silhouette,
    ),
    "type45": ShipObject(
        id="type45",
        name="Type 45 Destroyer",
        category=ObjectCategory.SHIP,
        dimensions=GeometricDimensions(152.0, 21.2, 48.0, 4000.0),
        thermal=ThermalProfile(
            base_temperature_k=293.0,
            emissivity=0.88,
            hot_spots=[
                ("funnel", (0.4, 0.5), 0.07, 70, "rectangle"),
            ],
        ),
        description="Air defence destroyer",
        country="UK",
        displacement_tons=7350,
        max_speed_knots=29,
        draft_m=7.4,
    ),
    "kirov": ShipObject(
        id="kirov",
        name="Kirov-class Battlecruiser",
        category=ObjectCategory.SHIP,
        dimensions=GeometricDimensions(252.0, 28.5, 59.0, 20000.0),
        thermal=ThermalProfile(
            base_temperature_k=300.0,
            emissivity=0.89,
            hot_spots=[
                ("funnel_fore", (0.35, 0.35), 0.08, 90, "rectangle"),
                ("funnel_aft", (0.35, 0.6), 0.08, 90, "rectangle"),
                ("reactor", (0.5, 0.45), 0.1, 60, "ellipse_h"),
            ],
        ),
        description="Nuclear-powered battlecruiser",
        country="Russia",
        displacement_tons=24300,
        max_speed_knots=32,
        draft_m=9.1,
    ),
    "frigate": ShipObject(
        id="frigate",
        name="Generic Frigate",
        category=ObjectCategory.SHIP,
        dimensions=GeometricDimensions(120.0, 15.0, 35.0, 2000.0),
        thermal=ThermalProfile(
            base_temperature_k=292.0,
            emissivity=0.89,
            hot_spots=[
                ("funnel", (0.4, 0.5), 0.06, 60, "rectangle"),
            ],
        ),
        description="General purpose frigate",
        country="Generic",
        displacement_tons=3500,
        max_speed_knots=28,
        draft_m=5.0,
    ),
    "patrol_boat": ShipObject(
        id="patrol_boat",
        name="Fast Patrol Boat",
        category=ObjectCategory.SHIP,
        dimensions=GeometricDimensions(40.0, 8.0, 12.0, 100.0),
        thermal=ThermalProfile(
            base_temperature_k=300.0,
            emissivity=0.87,
            hot_spots=[
                ("exhaust", (0.5, 0.7), 0.1, 100, "circle"),
            ],
        ),
        description="Fast attack craft",
        country="Generic",
        displacement_tons=250,
        max_speed_knots=40,
        draft_m=2.5,
    ),
}

# Ground Vehicles
VEHICLE_OBJECTS = {
    "m1_abrams": VehicleObject(
        id="m1_abrams",
        name="M1A2 Abrams",
        category=ObjectCategory.TANK,
        dimensions=GeometricDimensions(9.83, 3.66, 2.44, 5.0),
        thermal=ThermalProfile(
            base_temperature_k=310.0,
            emissivity=0.90,
            hot_spots=[
                ("engine", (0.6, 0.75), 0.15, 150, "rectangle"),
                ("exhaust", (0.8, 0.85), 0.08, 200, "circle"),
                ("gun_barrel", (0.5, 0.15), 0.05, 30, "ellipse_h"),
            ],
        ),
        description="Main battle tank",
        country="USA",
        max_speed_kmh=67,
        armor_mm=600,
        crew=4,
        silhouette_func=_create_tank_silhouette,
    ),
    "t90": VehicleObject(
        id="t90",
        name="T-90 Main Battle Tank",
        category=ObjectCategory.TANK,
        dimensions=GeometricDimensions(9.53, 3.78, 2.23, 4.0),
        thermal=ThermalProfile(
            base_temperature_k=315.0,
            emissivity=0.91,
            hot_spots=[
                ("engine", (0.6, 0.8), 0.15, 180, "rectangle"),
                ("exhaust", (0.85, 0.9), 0.1, 250, "circle"),
            ],
        ),
        description="Russian main battle tank",
        country="Russia",
        max_speed_kmh=60,
        armor_mm=550,
        crew=3,
        silhouette_func=_create_tank_silhouette,
    ),
    "leopard2": VehicleObject(
        id="leopard2",
        name="Leopard 2A7",
        category=ObjectCategory.TANK,
        dimensions=GeometricDimensions(9.97, 3.75, 2.64, 5.0),
        thermal=ThermalProfile(
            base_temperature_k=308.0,
            emissivity=0.89,
            hot_spots=[
                ("engine", (0.6, 0.78), 0.14, 140, "rectangle"),
                ("exhaust", (0.8, 0.88), 0.08, 190, "circle"),
            ],
        ),
        description="German main battle tank",
        country="Germany",
        max_speed_kmh=68,
        armor_mm=650,
        crew=4,
        silhouette_func=_create_tank_silhouette,
    ),
    "bradley": VehicleObject(
        id="bradley",
        name="M2 Bradley IFV",
        category=ObjectCategory.VEHICLE,
        dimensions=GeometricDimensions(6.55, 3.60, 2.98, 3.0),
        thermal=ThermalProfile(
            base_temperature_k=305.0,
            emissivity=0.88,
            hot_spots=[
                ("engine", (0.7, 0.8), 0.12, 120, "rectangle"),
                ("exhaust", (0.85, 0.9), 0.08, 150, "circle"),
            ],
        ),
        description="Infantry fighting vehicle",
        country="USA",
        max_speed_kmh=66,
        armor_mm=30,
        crew=3,
    ),
    "bmp3": VehicleObject(
        id="bmp3",
        name="BMP-3 IFV",
        category=ObjectCategory.VEHICLE,
        dimensions=GeometricDimensions(7.14, 3.15, 2.65, 2.5),
        thermal=ThermalProfile(
            base_temperature_k=310.0,
            emissivity=0.89,
            hot_spots=[
                ("engine", (0.65, 0.75), 0.12, 130, "rectangle"),
                ("exhaust", (0.8, 0.88), 0.08, 170, "circle"),
            ],
        ),
        description="Russian infantry fighting vehicle",
        country="Russia",
        max_speed_kmh=72,
        armor_mm=35,
        crew=3,
    ),
    "humvee": VehicleObject(
        id="humvee",
        name="HMMWV (Humvee)",
        category=ObjectCategory.VEHICLE,
        dimensions=GeometricDimensions(4.57, 2.16, 1.83, 1.0),
        thermal=ThermalProfile(
            base_temperature_k=300.0,
            emissivity=0.87,
            hot_spots=[
                ("engine", (0.5, 0.25), 0.15, 80, "rectangle"),
                ("exhaust", (0.5, 0.85), 0.06, 100, "circle"),
            ],
        ),
        description="Light utility vehicle",
        country="USA",
        max_speed_kmh=113,
        armor_mm=0,
        crew=4,
    ),
    "military_truck": VehicleObject(
        id="military_truck",
        name="Military Cargo Truck",
        category=ObjectCategory.VEHICLE,
        dimensions=GeometricDimensions(8.50, 2.50, 3.20, 5.0),
        thermal=ThermalProfile(
            base_temperature_k=305.0,
            emissivity=0.88,
            hot_spots=[
                ("engine", (0.5, 0.15), 0.12, 100, "rectangle"),
                ("exhaust", (0.6, 0.85), 0.06, 130, "circle"),
            ],
        ),
        description="6x6 military cargo truck",
        country="Generic",
        max_speed_kmh=90,
        armor_mm=0,
        crew=2,
        silhouette_func=_create_truck_silhouette,
    ),
    "pickup_technical": VehicleObject(
        id="pickup_technical",
        name="Technical (Armed Pickup)",
        category=ObjectCategory.VEHICLE,
        dimensions=GeometricDimensions(5.30, 1.90, 1.80, 0.5),
        thermal=ThermalProfile(
            base_temperature_k=298.0,
            emissivity=0.86,
            hot_spots=[
                ("engine", (0.5, 0.2), 0.12, 70, "rectangle"),
            ],
        ),
        description="Armed civilian pickup truck",
        country="Generic",
        max_speed_kmh=120,
        armor_mm=0,
        crew=4,
    ),
    "civilian_car": VehicleObject(
        id="civilian_car",
        name="Civilian Sedan",
        category=ObjectCategory.VEHICLE,
        dimensions=GeometricDimensions(4.50, 1.80, 1.50, 0.3),
        thermal=ThermalProfile(
            base_temperature_k=305.0,
            emissivity=0.85,
            hot_spots=[
                ("engine", (0.5, 0.2), 0.15, 60, "rectangle"),
                ("exhaust", (0.5, 0.9), 0.05, 80, "circle"),
            ],
        ),
        description="Standard civilian automobile",
        country="Generic",
        max_speed_kmh=180,
        armor_mm=0,
        crew=5,
    ),
}

# Missiles
MISSILE_OBJECTS = {
    "aim120": MissileObject(
        id="aim120",
        name="AIM-120 AMRAAM",
        category=ObjectCategory.MISSILE,
        dimensions=GeometricDimensions(3.66, 0.178, 0.178),
        thermal=ThermalProfile(
            base_temperature_k=350.0,
            emissivity=0.90,
            hot_spots=[
                ("motor", (0.5, 0.7), 0.3, 800, "ellipse_h"),
                ("plume", (0.5, 0.95), 0.2, 1500, "ellipse_h"),
            ],
        ),
        description="Medium-range air-to-air missile",
        country="USA",
        max_speed_mach=4.0,
        range_km=160,
        warhead_kg=23,
        guidance="active_radar",
        silhouette_func=_create_missile_silhouette,
    ),
    "aim9x": MissileObject(
        id="aim9x",
        name="AIM-9X Sidewinder",
        category=ObjectCategory.MISSILE,
        dimensions=GeometricDimensions(3.02, 0.127, 0.127),
        thermal=ThermalProfile(
            base_temperature_k=340.0,
            emissivity=0.88,
            hot_spots=[
                ("motor", (0.5, 0.65), 0.25, 700, "ellipse_h"),
                ("plume", (0.5, 0.92), 0.18, 1400, "ellipse_h"),
            ],
        ),
        description="Short-range air-to-air missile",
        country="USA",
        max_speed_mach=2.5,
        range_km=35,
        warhead_kg=9.4,
        guidance="infrared",
        silhouette_func=_create_missile_silhouette,
    ),
    "r77": MissileObject(
        id="r77",
        name="R-77 (AA-12 Adder)",
        category=ObjectCategory.MISSILE,
        dimensions=GeometricDimensions(3.60, 0.20, 0.20),
        thermal=ThermalProfile(
            base_temperature_k=345.0,
            emissivity=0.89,
            hot_spots=[
                ("motor", (0.5, 0.68), 0.28, 750, "ellipse_h"),
                ("plume", (0.5, 0.94), 0.2, 1450, "ellipse_h"),
            ],
        ),
        description="Russian air-to-air missile",
        country="Russia",
        max_speed_mach=4.5,
        range_km=110,
        warhead_kg=22,
        guidance="active_radar",
        silhouette_func=_create_missile_silhouette,
    ),
    "tomahawk": MissileObject(
        id="tomahawk",
        name="BGM-109 Tomahawk",
        category=ObjectCategory.MISSILE,
        dimensions=GeometricDimensions(5.56, 0.52, 0.52),
        thermal=ThermalProfile(
            base_temperature_k=320.0,
            emissivity=0.85,
            hot_spots=[
                ("engine", (0.5, 0.6), 0.15, 200, "ellipse_h"),
                ("exhaust", (0.5, 0.85), 0.1, 400, "circle"),
            ],
        ),
        description="Cruise missile",
        country="USA",
        max_speed_mach=0.74,
        range_km=2500,
        warhead_kg=450,
        guidance="inertial_gps_tercom",
        silhouette_func=_create_missile_silhouette,
    ),
    "kalibr": MissileObject(
        id="kalibr",
        name="3M-54 Kalibr",
        category=ObjectCategory.MISSILE,
        dimensions=GeometricDimensions(6.20, 0.53, 0.53),
        thermal=ThermalProfile(
            base_temperature_k=325.0,
            emissivity=0.86,
            hot_spots=[
                ("engine", (0.5, 0.55), 0.15, 220, "ellipse_h"),
                ("exhaust", (0.5, 0.88), 0.12, 420, "circle"),
            ],
        ),
        description="Russian cruise missile",
        country="Russia",
        max_speed_mach=0.8,
        range_km=2500,
        warhead_kg=400,
        guidance="inertial_gps",
        silhouette_func=_create_missile_silhouette,
    ),
    "scud": MissileObject(
        id="scud",
        name="Scud-B (SS-1)",
        category=ObjectCategory.MISSILE,
        dimensions=GeometricDimensions(11.25, 0.88, 0.88),
        thermal=ThermalProfile(
            base_temperature_k=400.0,
            emissivity=0.92,
            hot_spots=[
                ("motor", (0.5, 0.5), 0.35, 1000, "ellipse_h"),
                ("plume", (0.5, 0.9), 0.25, 2000, "ellipse_h"),
            ],
        ),
        description="Short-range ballistic missile",
        country="Russia",
        max_speed_mach=5.0,
        range_km=300,
        warhead_kg=985,
        guidance="inertial",
        silhouette_func=_create_missile_silhouette,
    ),
    "atacms": MissileObject(
        id="atacms",
        name="MGM-140 ATACMS",
        category=ObjectCategory.MISSILE,
        dimensions=GeometricDimensions(3.98, 0.61, 0.61),
        thermal=ThermalProfile(
            base_temperature_k=380.0,
            emissivity=0.91,
            hot_spots=[
                ("motor", (0.5, 0.55), 0.3, 900, "ellipse_h"),
                ("plume", (0.5, 0.92), 0.22, 1800, "ellipse_h"),
            ],
        ),
        description="Army tactical missile",
        country="USA",
        max_speed_mach=3.0,
        range_km=300,
        warhead_kg=227,
        guidance="inertial_gps",
        silhouette_func=_create_missile_silhouette,
    ),
}

# Launchers
LAUNCHER_OBJECTS = {
    "patriot": LauncherObject(
        id="patriot",
        name="MIM-104 Patriot",
        category=ObjectCategory.LAUNCHER,
        dimensions=GeometricDimensions(9.00, 2.70, 3.50),
        thermal=ThermalProfile(
            base_temperature_k=300.0,
            emissivity=0.88,
            hot_spots=[
                ("generator", (0.6, 0.8), 0.1, 80, "rectangle"),
                ("electronics", (0.4, 0.5), 0.08, 40, "rectangle"),
            ],
        ),
        description="SAM launcher vehicle",
        country="USA",
        num_tubes=4,
        reload_time_s=600,
        traverse_deg=360,
        elevation_deg=(0, 90),
        silhouette_func=_create_sam_launcher_silhouette,
    ),
    "s400": LauncherObject(
        id="s400",
        name="S-400 Triumf TEL",
        category=ObjectCategory.LAUNCHER,
        dimensions=GeometricDimensions(12.00, 3.20, 4.50),
        thermal=ThermalProfile(
            base_temperature_k=305.0,
            emissivity=0.89,
            hot_spots=[
                ("engine", (0.7, 0.85), 0.12, 100, "rectangle"),
                ("electronics", (0.3, 0.4), 0.08, 35, "rectangle"),
            ],
        ),
        description="Russian SAM launcher",
        country="Russia",
        num_tubes=4,
        reload_time_s=900,
        traverse_deg=360,
        elevation_deg=(0, 90),
        silhouette_func=_create_sam_launcher_silhouette,
    ),
    "himars": LauncherObject(
        id="himars",
        name="M142 HIMARS",
        category=ObjectCategory.LAUNCHER,
        dimensions=GeometricDimensions(7.00, 2.40, 3.20),
        thermal=ThermalProfile(
            base_temperature_k=302.0,
            emissivity=0.87,
            hot_spots=[
                ("engine", (0.5, 0.2), 0.12, 90, "rectangle"),
                ("electronics", (0.5, 0.6), 0.08, 30, "rectangle"),
            ],
        ),
        description="Rocket artillery system",
        country="USA",
        num_tubes=6,
        reload_time_s=300,
        traverse_deg=180,
        elevation_deg=(0, 60),
    ),
    "bm21": LauncherObject(
        id="bm21",
        name="BM-21 Grad",
        category=ObjectCategory.LAUNCHER,
        dimensions=GeometricDimensions(7.35, 2.40, 3.09),
        thermal=ThermalProfile(
            base_temperature_k=298.0,
            emissivity=0.88,
            hot_spots=[
                ("engine", (0.5, 0.15), 0.12, 85, "rectangle"),
            ],
        ),
        description="Multiple rocket launcher",
        country="Russia",
        num_tubes=40,
        reload_time_s=600,
        traverse_deg=120,
        elevation_deg=(0, 55),
    ),
    "m777": LauncherObject(
        id="m777",
        name="M777 Howitzer",
        category=ObjectCategory.LAUNCHER,
        dimensions=GeometricDimensions(10.70, 2.77, 2.26),
        thermal=ThermalProfile(
            base_temperature_k=295.0,
            emissivity=0.86,
            hot_spots=[
                ("barrel_tip", (0.5, 0.1), 0.08, 150, "circle"),  # After firing
            ],
        ),
        description="Towed 155mm howitzer",
        country="USA",
        num_tubes=1,
        reload_time_s=20,
        traverse_deg=45,
        elevation_deg=(-5, 72),
    ),
}

# Personnel
PERSON_OBJECTS = {
    "soldier_standing": PersonObject(
        id="soldier_standing",
        name="Standing Soldier",
        category=ObjectCategory.PERSON,
        dimensions=GeometricDimensions(0.50, 0.50, 1.80),
        thermal=ThermalProfile(
            base_temperature_k=305.0,
            emissivity=0.97,
            hot_spots=[
                ("face", (0.15, 0.5), 0.12, 5, "circle"),
                ("hands", (0.55, 0.3), 0.06, 3, "circle"),
                ("hands_r", (0.55, 0.7), 0.06, 3, "circle"),
            ],
        ),
        description="Standing infantry soldier",
        posture="standing",
        equipment_kg=30,
        silhouette_func=_create_person_silhouette,
    ),
    "soldier_prone": PersonObject(
        id="soldier_prone",
        name="Prone Soldier",
        category=ObjectCategory.PERSON,
        dimensions=GeometricDimensions(1.80, 0.50, 0.40),
        thermal=ThermalProfile(
            base_temperature_k=303.0,
            emissivity=0.96,
            hot_spots=[
                ("face", (0.5, 0.1), 0.1, 4, "circle"),
            ],
        ),
        description="Prone infantry soldier",
        posture="prone",
        equipment_kg=30,
    ),
    "civilian": PersonObject(
        id="civilian",
        name="Civilian Person",
        category=ObjectCategory.PERSON,
        dimensions=GeometricDimensions(0.45, 0.45, 1.70),
        thermal=ThermalProfile(
            base_temperature_k=306.0,
            emissivity=0.98,
            hot_spots=[
                ("face", (0.12, 0.5), 0.12, 5, "circle"),
            ],
        ),
        description="Civilian in normal clothing",
        posture="standing",
        equipment_kg=0,
        silhouette_func=_create_person_silhouette,
    ),
    "crowd": PersonObject(
        id="crowd",
        name="Group of People (5)",
        category=ObjectCategory.PERSON,
        dimensions=GeometricDimensions(3.0, 3.0, 1.80),
        thermal=ThermalProfile(
            base_temperature_k=305.5,
            emissivity=0.97,
            hot_spots=[
                ("faces", (0.2, 0.5), 0.25, 4, "rectangle"),
            ],
        ),
        description="Small group of people",
        posture="standing",
        equipment_kg=0,
    ),
}


# =============================================================================
# Object Library Interface
# =============================================================================

class ObjectLibrary:
    """Central registry for all objects."""

    _objects: Dict[str, Object3D] = {}
    _initialized: bool = False

    @classmethod
    def _initialize(cls) -> None:
        """Initialize the library with all objects."""
        if cls._initialized:
            return

        # Register all objects
        all_objects = {
            **AIRCRAFT_OBJECTS,
            **HELICOPTER_OBJECTS,
            **UAV_OBJECTS,
            **SHIP_OBJECTS,
            **VEHICLE_OBJECTS,
            **MISSILE_OBJECTS,
            **LAUNCHER_OBJECTS,
            **PERSON_OBJECTS,
        }

        for obj_id, obj in all_objects.items():
            cls._objects[obj_id] = obj

        cls._initialized = True

    @classmethod
    def get(cls, object_id: str) -> Object3D:
        """Get object by ID."""
        cls._initialize()
        if object_id not in cls._objects:
            raise ValueError(f"Unknown object: {object_id}. Available: {list(cls._objects.keys())}")
        return cls._objects[object_id]

    @classmethod
    def list_all(cls) -> List[str]:
        """List all available object IDs."""
        cls._initialize()
        return list(cls._objects.keys())

    @classmethod
    def list_by_category(cls, category: ObjectCategory) -> List[str]:
        """List objects by category."""
        cls._initialize()
        return [
            obj_id for obj_id, obj in cls._objects.items()
            if obj.category == category
        ]

    @classmethod
    def search(cls, query: str) -> List[str]:
        """Search objects by name or description."""
        cls._initialize()
        query_lower = query.lower()
        results = []
        for obj_id, obj in cls._objects.items():
            if (query_lower in obj_id.lower() or
                query_lower in obj.name.lower() or
                query_lower in obj.description.lower()):
                results.append(obj_id)
        return results

    @classmethod
    def get_info(cls, object_id: str) -> Dict[str, Any]:
        """Get detailed info about an object."""
        obj = cls.get(object_id)
        return {
            "id": obj.id,
            "name": obj.name,
            "category": obj.category.value,
            "dimensions": {
                "length_m": obj.dimensions.length_m,
                "width_m": obj.dimensions.width_m,
                "height_m": obj.dimensions.height_m,
            },
            "thermal": {
                "base_temp_k": obj.thermal.base_temperature_k,
                "emissivity": obj.thermal.emissivity,
                "hot_spots": len(obj.thermal.hot_spots),
            },
            "description": obj.description,
            "country": obj.country,
        }


# Convenience functions
def get_object(object_id: str) -> Object3D:
    """Get object by ID."""
    return ObjectLibrary.get(object_id)


def list_objects(category: Optional[ObjectCategory] = None) -> List[str]:
    """List available objects, optionally filtered by category."""
    if category is not None:
        return ObjectLibrary.list_by_category(category)
    return ObjectLibrary.list_all()


def list_categories() -> List[str]:
    """List all object categories."""
    return [cat.value for cat in ObjectCategory]
