"""
Enhanced Target Shape Library for EOSIM.

Provides realistic thermal signature shapes for targets with:
- Distinct thermal zones (engine, cabin, wheels, exhaust for vehicles)
- Humanoid body models with skin/clothing differentiation
- Aircraft exhaust plumes with thermal gradients

Example Usage:
--------------
# Example 1: Vehicle with thermal zones
>>> from eosim.targets.shapes import EnhancedVehicleShape
>>> shape = EnhancedVehicleShape(
...     length_m=4.5, width_m=1.8, height_m=1.4,
...     engine_running=True, speed_kmh=60
... )
>>> temp_map, emis_map, mask = shape.render((64, 64), aspect_angle_deg=90)

# Example 2: Humanoid person with body segment temperatures
>>> from eosim.targets.shapes import HumanoidShape
>>> person = HumanoidShape(pose="standing", activity="walking")
>>> temp_map, emis_map, mask = person.render((64, 32))

# Example 3: Aircraft with exhaust plume gradient
>>> from eosim.targets.shapes import AircraftWithPlumeShape
>>> aircraft = AircraftWithPlumeShape(
...     throttle=0.8, afterburner=True, altitude_m=8000
... )
>>> temp_map, emis_map, mask = aircraft.render((64, 64), aspect_angle_deg=180)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum
import numpy as np
from numpy.typing import NDArray


class ThermalZone(Enum):
    """Thermal zone identifiers for distinct temperature regions."""
    ENGINE = "engine"
    CABIN = "cabin"
    WHEELS = "wheels"
    EXHAUST = "exhaust"
    BODY = "body"
    # Person zones
    HEAD = "head"
    FACE = "face"
    HANDS = "hands"
    TORSO = "torso"
    LEGS = "legs"
    FEET = "feet"
    # Aircraft zones
    FUSELAGE = "fuselage"
    WINGS = "wings"
    NOZZLE = "nozzle"
    PLUME = "plume"


@dataclass
class ThermalZoneConfig:
    """Configuration for a thermal zone.

    Attributes:
        temperature_k: Base temperature in Kelvin
        emissivity: Surface emissivity (0-1)
        gradient_falloff: Temperature falloff rate at edges
        noise_std_k: Standard deviation of temperature noise
    """
    temperature_k: float
    emissivity: float = 0.9
    gradient_falloff: float = 0.0
    noise_std_k: float = 1.0


# Default thermal zone temperatures based on user requirements
DEFAULT_VEHICLE_ZONES = {
    ThermalZone.ENGINE: ThermalZoneConfig(370.0, 0.85, 0.3, 5.0),
    ThermalZone.CABIN: ThermalZoneConfig(320.0, 0.92, 0.1, 2.0),
    ThermalZone.WHEELS: ThermalZoneConfig(340.0, 0.95, 0.2, 3.0),
    ThermalZone.EXHAUST: ThermalZoneConfig(400.0, 0.80, 0.5, 8.0),
    ThermalZone.BODY: ThermalZoneConfig(300.0, 0.90, 0.05, 1.5),
}

DEFAULT_PERSON_ZONES = {
    ThermalZone.HEAD: ThermalZoneConfig(308.0, 0.98, 0.1, 0.5),
    ThermalZone.FACE: ThermalZoneConfig(307.0, 0.98, 0.1, 0.3),
    ThermalZone.HANDS: ThermalZoneConfig(305.0, 0.98, 0.15, 0.5),
    ThermalZone.TORSO: ThermalZoneConfig(295.0, 0.95, 0.05, 1.0),
    ThermalZone.LEGS: ThermalZoneConfig(298.0, 0.95, 0.08, 1.0),
    ThermalZone.FEET: ThermalZoneConfig(300.0, 0.95, 0.1, 0.8),
}

DEFAULT_AIRCRAFT_ZONES = {
    ThermalZone.FUSELAGE: ThermalZoneConfig(280.0, 0.85, 0.05, 2.0),
    ThermalZone.WINGS: ThermalZoneConfig(275.0, 0.85, 0.1, 1.5),
    ThermalZone.NOZZLE: ThermalZoneConfig(500.0, 0.75, 0.2, 10.0),
    ThermalZone.PLUME: ThermalZoneConfig(400.0, 0.3, 0.8, 15.0),
}


def create_gradient_falloff(
    shape: Tuple[int, int],
    center: Tuple[int, int],
    radius: float,
    falloff_rate: float = 0.5,
) -> NDArray:
    """Create a radial gradient falloff pattern.

    Args:
        shape: (height, width) of output
        center: (y, x) center of gradient
        radius: Base radius
        falloff_rate: Rate of temperature falloff (0-1)

    Returns:
        2D array with values from 1.0 (center) to 0.0 (edge)
    """
    h, w = shape
    cy, cx = center
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)

    # Sigmoid-based falloff for smooth transition
    falloff = 1.0 / (1.0 + np.exp((dist - radius) / (radius * falloff_rate + 0.1)))
    return falloff.astype(np.float64)


def create_linear_gradient(
    shape: Tuple[int, int],
    start_pos: Tuple[int, int],
    end_pos: Tuple[int, int],
    start_temp: float,
    end_temp: float,
) -> NDArray:
    """Create a linear temperature gradient between two points.

    Args:
        shape: (height, width) of output
        start_pos: (y, x) start position
        end_pos: (y, x) end position
        start_temp: Temperature at start
        end_temp: Temperature at end

    Returns:
        2D temperature gradient array
    """
    h, w = shape
    y0, x0 = start_pos
    y1, x1 = end_pos

    # Direction vector
    dy = y1 - y0
    dx = x1 - x0
    length = np.sqrt(dy**2 + dx**2)
    if length < 1:
        return np.full(shape, start_temp, dtype=np.float64)

    # Normalized direction
    dy_n, dx_n = dy / length, dx / length

    # Project each point onto the gradient line
    yy, xx = np.ogrid[:h, :w]
    proj = (yy - y0) * dy_n + (xx - x0) * dx_n

    # Normalize to 0-1 range
    t = np.clip(proj / length, 0, 1)

    # Interpolate temperature
    return start_temp + (end_temp - start_temp) * t


class EnhancedVehicleShape:
    """Enhanced vehicle shape with distinct thermal zones.

    Creates realistic thermal signatures with:
    - Engine compartment (front, 370K when running)
    - Cabin/interior (320K, varies with occupancy)
    - Wheels (340K, increases with speed)
    - Exhaust (rear, 400K when engine running)
    - Body panels (300K ambient-matched)
    """

    def __init__(
        self,
        length_m: float = 4.5,
        width_m: float = 1.8,
        height_m: float = 1.4,
        engine_running: bool = False,
        speed_kmh: float = 0.0,
        ambient_k: float = 290.0,
        occupants: int = 1,
        engine_position: str = "front",  # "front", "rear", "mid"
        vehicle_type: str = "sedan",  # "sedan", "suv", "truck", "tank"
        zone_configs: Optional[dict] = None,
    ):
        """Initialize enhanced vehicle shape.

        Args:
            length_m: Vehicle length in meters
            width_m: Vehicle width in meters
            height_m: Vehicle height in meters
            engine_running: Whether engine is running
            speed_kmh: Current speed in km/h
            ambient_k: Ambient temperature in Kelvin
            occupants: Number of occupants (affects cabin temp)
            engine_position: Location of engine
            vehicle_type: Type of vehicle
            zone_configs: Custom thermal zone configurations
        """
        self.length_m = length_m
        self.width_m = width_m
        self.height_m = height_m
        self.engine_running = engine_running
        self.speed_kmh = speed_kmh
        self.ambient_k = ambient_k
        self.occupants = occupants
        self.engine_position = engine_position
        self.vehicle_type = vehicle_type

        # Initialize zone configs with defaults, then apply custom
        self.zones = dict(DEFAULT_VEHICLE_ZONES)
        if zone_configs:
            self.zones.update(zone_configs)

        # Adjust temperatures based on state
        self._adjust_temperatures()

    def _adjust_temperatures(self) -> None:
        """Adjust zone temperatures based on vehicle state."""
        if not self.engine_running:
            # Engine off - everything closer to ambient
            self.zones[ThermalZone.ENGINE] = ThermalZoneConfig(
                self.ambient_k + 5, 0.85, 0.3, 2.0
            )
            self.zones[ThermalZone.EXHAUST] = ThermalZoneConfig(
                self.ambient_k, 0.80, 0.5, 1.0
            )
        else:
            # Engine running - adjust based on speed (load)
            engine_temp = 350 + self.speed_kmh * 0.3  # Higher load = higher temp
            exhaust_temp = 380 + self.speed_kmh * 0.4
            self.zones[ThermalZone.ENGINE] = ThermalZoneConfig(
                min(engine_temp, 420), 0.85, 0.3, 5.0
            )
            self.zones[ThermalZone.EXHAUST] = ThermalZoneConfig(
                min(exhaust_temp, 500), 0.80, 0.5, 8.0
            )

        # Wheel temperature increases with speed (friction)
        wheel_temp = 300 + self.speed_kmh * 0.5
        self.zones[ThermalZone.WHEELS] = ThermalZoneConfig(
            min(wheel_temp, 380), 0.95, 0.2, 3.0
        )

        # Cabin temperature based on occupants
        cabin_base = 295 + self.occupants * 8  # Body heat
        if self.engine_running:
            cabin_base += 5  # HVAC/heater
        self.zones[ThermalZone.CABIN] = ThermalZoneConfig(
            min(cabin_base, 320), 0.92, 0.1, 2.0
        )

        # Body temperature tracks ambient with solar loading
        self.zones[ThermalZone.BODY] = ThermalZoneConfig(
            self.ambient_k + 10, 0.90, 0.05, 1.5
        )

    def render(
        self,
        resolution: Tuple[int, int] = (64, 64),
        aspect_angle_deg: float = 0.0,
        elevation_angle_deg: float = 0.0,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Render the vehicle thermal signature.

        Args:
            resolution: Output (height, width)
            aspect_angle_deg: Viewing angle (0=front, 90=side, 180=rear)
            elevation_angle_deg: Elevation angle

        Returns:
            Tuple of (temperature_map, emissivity_map, mask)
        """
        h, w = resolution

        # Initialize maps
        temp_map = np.zeros((h, w), dtype=np.float64)
        emis_map = np.zeros((h, w), dtype=np.float64)
        mask = np.zeros((h, w), dtype=bool)
        zone_map = np.zeros((h, w), dtype=np.int32)  # Track which zone each pixel belongs to

        # Create vehicle outline
        self._create_body_outline(mask, resolution, aspect_angle_deg)

        # Apply body base temperature
        body_cfg = self.zones[ThermalZone.BODY]
        temp_map[mask] = body_cfg.temperature_k
        emis_map[mask] = body_cfg.emissivity

        # Apply thermal zones
        self._apply_engine_zone(temp_map, emis_map, mask, resolution, aspect_angle_deg)
        self._apply_cabin_zone(temp_map, emis_map, mask, resolution, aspect_angle_deg)
        self._apply_wheel_zones(temp_map, emis_map, mask, resolution, aspect_angle_deg)
        self._apply_exhaust_zone(temp_map, emis_map, mask, resolution, aspect_angle_deg)

        # Add noise for realism
        noise = np.random.normal(0, 2.0, (h, w))
        temp_map = np.where(mask, temp_map + noise, 0)

        return temp_map, emis_map, mask

    def _create_body_outline(
        self,
        mask: NDArray,
        resolution: Tuple[int, int],
        aspect_angle_deg: float,
    ) -> None:
        """Create the vehicle body outline."""
        h, w = resolution
        angle_rad = np.radians(aspect_angle_deg % 360)

        # Adjust apparent dimensions based on view angle
        cos_a = abs(np.cos(angle_rad))
        sin_a = abs(np.sin(angle_rad))

        # Main body (lower section)
        body_h = int(h * 0.45)
        body_w = int(w * 0.85)
        y_center = h // 2 + int(h * 0.15)
        x_center = w // 2

        y0 = y_center - body_h // 2
        y1 = y_center + body_h // 2
        x0 = x_center - body_w // 2
        x1 = x_center + body_w // 2

        mask[max(0,y0):min(h,y1), max(0,x0):min(w,x1)] = True

        # Cabin (upper section)
        if self.vehicle_type in ("sedan", "suv"):
            cabin_h = int(h * 0.30)
            cabin_w = int(w * 0.55)
            cabin_y = y0 - int(cabin_h * 0.3)
            cabin_x0 = x_center - cabin_w // 2
            cabin_x1 = x_center + cabin_w // 2

            if cabin_y >= 0:
                mask[cabin_y:y0+5, cabin_x0:cabin_x1] = True

        elif self.vehicle_type == "truck":
            # Cab section
            cab_h = int(h * 0.35)
            cab_w = int(w * 0.25)
            cab_y = y0 - int(cab_h * 0.5)
            cab_x0 = x0

            if cab_y >= 0:
                mask[cab_y:y0+5, cab_x0:cab_x0+cab_w] = True

    def _apply_engine_zone(
        self,
        temp_map: NDArray,
        emis_map: NDArray,
        mask: NDArray,
        resolution: Tuple[int, int],
        aspect_angle_deg: float,
    ) -> None:
        """Apply engine thermal zone."""
        h, w = resolution
        cfg = self.zones[ThermalZone.ENGINE]

        # Engine position
        if self.engine_position == "front":
            cx = int(w * 0.15)
        elif self.engine_position == "rear":
            cx = int(w * 0.85)
        else:
            cx = w // 2

        cy = h // 2 + int(h * 0.1)
        radius = int(min(h, w) * 0.15)

        # Create engine zone with gradient falloff
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        engine_mask = (dist <= radius * 1.5) & mask

        # Temperature gradient from center outward
        gradient = create_gradient_falloff((h, w), (cy, cx), radius, cfg.gradient_falloff)
        engine_temp = cfg.temperature_k * gradient + self.zones[ThermalZone.BODY].temperature_k * (1 - gradient)

        temp_map[engine_mask] = engine_temp[engine_mask]
        emis_map[engine_mask] = cfg.emissivity

    def _apply_cabin_zone(
        self,
        temp_map: NDArray,
        emis_map: NDArray,
        mask: NDArray,
        resolution: Tuple[int, int],
        aspect_angle_deg: float,
    ) -> None:
        """Apply cabin/interior thermal zone."""
        h, w = resolution
        cfg = self.zones[ThermalZone.CABIN]

        # Cabin is in center-upper area (windows visible as warmer)
        cabin_cy = h // 2 - int(h * 0.1)
        cabin_cx = w // 2

        # Elliptical cabin zone
        cabin_ry = int(h * 0.12)
        cabin_rx = int(w * 0.20)

        yy, xx = np.ogrid[:h, :w]
        cabin_mask = ((yy - cabin_cy)**2 / cabin_ry**2 + (xx - cabin_cx)**2 / cabin_rx**2 <= 1) & mask

        gradient = create_gradient_falloff((h, w), (cabin_cy, cabin_cx), cabin_rx, cfg.gradient_falloff)
        cabin_temp = cfg.temperature_k * gradient + self.zones[ThermalZone.BODY].temperature_k * (1 - gradient)

        temp_map[cabin_mask] = cabin_temp[cabin_mask]
        emis_map[cabin_mask] = cfg.emissivity

    def _apply_wheel_zones(
        self,
        temp_map: NDArray,
        emis_map: NDArray,
        mask: NDArray,
        resolution: Tuple[int, int],
        aspect_angle_deg: float,
    ) -> None:
        """Apply wheel thermal zones."""
        h, w = resolution
        cfg = self.zones[ThermalZone.WHEELS]

        # Four wheel positions
        wheel_positions = [
            (int(h * 0.75), int(w * 0.18)),  # Front-left
            (int(h * 0.75), int(w * 0.82)),  # Rear-left
        ]

        # For side view, only show near-side wheels
        wheel_radius = int(min(h, w) * 0.08)

        yy, xx = np.ogrid[:h, :w]

        for wy, wx in wheel_positions:
            dist = np.sqrt((yy - wy)**2 + (xx - wx)**2)
            wheel_mask = (dist <= wheel_radius) & mask

            # Wheel temperature with gradient
            gradient = create_gradient_falloff((h, w), (wy, wx), wheel_radius, cfg.gradient_falloff)
            wheel_temp = cfg.temperature_k * gradient + self.zones[ThermalZone.BODY].temperature_k * (1 - gradient)

            temp_map[wheel_mask] = wheel_temp[wheel_mask]
            emis_map[wheel_mask] = cfg.emissivity

    def _apply_exhaust_zone(
        self,
        temp_map: NDArray,
        emis_map: NDArray,
        mask: NDArray,
        resolution: Tuple[int, int],
        aspect_angle_deg: float,
    ) -> None:
        """Apply exhaust thermal zone."""
        if not self.engine_running:
            return

        h, w = resolution
        cfg = self.zones[ThermalZone.EXHAUST]

        # Exhaust is at rear, low position
        exhaust_cy = h // 2 + int(h * 0.25)
        exhaust_cx = int(w * 0.9)

        # Small hot spot for exhaust
        exhaust_radius = int(min(h, w) * 0.06)

        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((yy - exhaust_cy)**2 + (xx - exhaust_cx)**2)
        exhaust_mask = dist <= exhaust_radius * 2

        # Strong gradient falloff for exhaust
        gradient = create_gradient_falloff((h, w), (exhaust_cy, exhaust_cx), exhaust_radius, cfg.gradient_falloff)

        # Only apply where mask is True or just outside for heat dissipation
        temp_map[exhaust_mask] = np.maximum(
            temp_map[exhaust_mask],
            cfg.temperature_k * gradient[exhaust_mask]
        )
        emis_map[exhaust_mask & mask] = cfg.emissivity

        # Update mask to include visible exhaust plume
        mask[exhaust_mask] = True


class HumanoidShape:
    """Enhanced humanoid shape with anatomically correct thermal zones.

    Creates realistic human thermal signatures with:
    - Head/face (308K, highest exposed skin temp)
    - Hands (305K, often exposed)
    - Torso (295K through clothing)
    - Legs (298K through clothing)
    - Proper body proportions for different poses
    """

    # Body segment proportions (fraction of total height)
    PROPORTIONS = {
        "head": 0.13,
        "neck": 0.03,
        "torso": 0.30,
        "upper_arm": 0.15,
        "lower_arm": 0.13,
        "hand": 0.05,
        "upper_leg": 0.22,
        "lower_leg": 0.20,
        "foot": 0.04,
    }

    def __init__(
        self,
        height_m: float = 1.75,
        pose: str = "standing",
        activity: str = "idle",
        clothing_coverage: float = 0.8,
        ambient_k: float = 290.0,
        zone_configs: Optional[dict] = None,
    ):
        """Initialize humanoid shape.

        Args:
            height_m: Person height in meters
            pose: Body pose ("standing", "walking", "running", "sitting", "prone")
            activity: Activity level affecting temperature
            clothing_coverage: Fraction of body covered (0-1)
            ambient_k: Ambient temperature
            zone_configs: Custom thermal zone configurations
        """
        self.height_m = height_m
        self.pose = pose
        self.activity = activity
        self.clothing_coverage = clothing_coverage
        self.ambient_k = ambient_k

        # Initialize zone configs
        self.zones = dict(DEFAULT_PERSON_ZONES)
        if zone_configs:
            self.zones.update(zone_configs)

        # Adjust for activity level
        self._adjust_for_activity()

    def _adjust_for_activity(self) -> None:
        """Adjust temperatures based on activity level."""
        activity_temp_delta = {
            "sleeping": -1.0,
            "sitting": 0.0,
            "idle": 0.5,
            "standing": 1.0,
            "walking": 2.0,
            "walking_fast": 3.0,
            "running": 5.0,
            "sprinting": 7.0,
        }

        delta = activity_temp_delta.get(self.activity, 0.0)

        # Apply delta to exposed areas (head, hands)
        for zone in [ThermalZone.HEAD, ThermalZone.FACE, ThermalZone.HANDS]:
            if zone in self.zones:
                cfg = self.zones[zone]
                self.zones[zone] = ThermalZoneConfig(
                    cfg.temperature_k + delta,
                    cfg.emissivity,
                    cfg.gradient_falloff,
                    cfg.noise_std_k,
                )

    def render(
        self,
        resolution: Tuple[int, int] = (64, 32),
        aspect_angle_deg: float = 0.0,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Render the humanoid thermal signature.

        Args:
            resolution: Output (height, width)
            aspect_angle_deg: Viewing angle

        Returns:
            Tuple of (temperature_map, emissivity_map, mask)
        """
        h, w = resolution

        temp_map = np.zeros((h, w), dtype=np.float64)
        emis_map = np.zeros((h, w), dtype=np.float64)
        mask = np.zeros((h, w), dtype=bool)

        if self.pose == "standing":
            self._render_standing(temp_map, emis_map, mask, resolution, aspect_angle_deg)
        elif self.pose == "walking":
            self._render_walking(temp_map, emis_map, mask, resolution, aspect_angle_deg)
        elif self.pose == "sitting":
            self._render_sitting(temp_map, emis_map, mask, resolution, aspect_angle_deg)
        elif self.pose == "prone":
            self._render_prone(temp_map, emis_map, mask, resolution, aspect_angle_deg)
        else:
            self._render_standing(temp_map, emis_map, mask, resolution, aspect_angle_deg)

        # Add subtle noise
        noise = np.random.normal(0, 0.5, (h, w))
        temp_map = np.where(mask, temp_map + noise, 0)

        return temp_map, emis_map, mask

    def _render_standing(
        self,
        temp_map: NDArray,
        emis_map: NDArray,
        mask: NDArray,
        resolution: Tuple[int, int],
        aspect_angle_deg: float,
    ) -> None:
        """Render standing pose."""
        h, w = resolution
        cx = w // 2

        # Head (ellipse at top)
        head_h = int(h * self.PROPORTIONS["head"])
        head_w = int(w * 0.4)
        head_cy = int(h * 0.07)

        yy, xx = np.ogrid[:h, :w]
        head_mask = ((yy - head_cy)**2 / (head_h/2)**2 + (xx - cx)**2 / (head_w/2)**2) <= 1

        cfg = self.zones[ThermalZone.HEAD]
        temp_map[head_mask] = cfg.temperature_k
        emis_map[head_mask] = cfg.emissivity
        mask[head_mask] = True

        # Face region (slightly warmer center of head)
        face_mask = ((yy - head_cy)**2 / (head_h/3)**2 + (xx - cx)**2 / (head_w/3)**2) <= 1
        cfg_face = self.zones[ThermalZone.FACE]
        temp_map[face_mask] = cfg_face.temperature_k

        # Neck
        neck_y0 = int(h * 0.12)
        neck_y1 = int(h * 0.16)
        neck_w = int(w * 0.15)
        mask[neck_y0:neck_y1, cx-neck_w//2:cx+neck_w//2] = True
        temp_map[neck_y0:neck_y1, cx-neck_w//2:cx+neck_w//2] = self.zones[ThermalZone.HEAD].temperature_k - 1
        emis_map[neck_y0:neck_y1, cx-neck_w//2:cx+neck_w//2] = 0.98

        # Torso
        torso_y0 = int(h * 0.16)
        torso_y1 = int(h * 0.50)
        torso_w = int(w * 0.50)

        torso_mask = np.zeros_like(mask)
        torso_mask[torso_y0:torso_y1, cx-torso_w//2:cx+torso_w//2] = True

        cfg_torso = self.zones[ThermalZone.TORSO]
        temp_map[torso_mask] = cfg_torso.temperature_k
        emis_map[torso_mask] = cfg_torso.emissivity
        mask[torso_mask] = True

        # Arms (on sides of torso)
        arm_w = int(w * 0.12)
        arm_y0 = int(h * 0.18)
        arm_y1 = int(h * 0.45)

        # Left arm
        left_arm_x = cx - torso_w//2 - arm_w
        if left_arm_x >= 0:
            mask[arm_y0:arm_y1, left_arm_x:left_arm_x+arm_w] = True
            temp_map[arm_y0:arm_y1, left_arm_x:left_arm_x+arm_w] = cfg_torso.temperature_k + 1
            emis_map[arm_y0:arm_y1, left_arm_x:left_arm_x+arm_w] = cfg_torso.emissivity

        # Right arm
        right_arm_x = cx + torso_w//2
        if right_arm_x + arm_w <= w:
            mask[arm_y0:arm_y1, right_arm_x:right_arm_x+arm_w] = True
            temp_map[arm_y0:arm_y1, right_arm_x:right_arm_x+arm_w] = cfg_torso.temperature_k + 1
            emis_map[arm_y0:arm_y1, right_arm_x:right_arm_x+arm_w] = cfg_torso.emissivity

        # Hands (at end of arms, exposed skin)
        hand_y0 = int(h * 0.43)
        hand_y1 = int(h * 0.50)
        hand_w = int(w * 0.10)

        cfg_hands = self.zones[ThermalZone.HANDS]

        # Left hand
        if left_arm_x >= 0:
            mask[hand_y0:hand_y1, left_arm_x:left_arm_x+hand_w] = True
            temp_map[hand_y0:hand_y1, left_arm_x:left_arm_x+hand_w] = cfg_hands.temperature_k
            emis_map[hand_y0:hand_y1, left_arm_x:left_arm_x+hand_w] = cfg_hands.emissivity

        # Right hand
        if right_arm_x + hand_w <= w:
            mask[hand_y0:hand_y1, right_arm_x:right_arm_x+hand_w] = True
            temp_map[hand_y0:hand_y1, right_arm_x:right_arm_x+hand_w] = cfg_hands.temperature_k
            emis_map[hand_y0:hand_y1, right_arm_x:right_arm_x+hand_w] = cfg_hands.emissivity

        # Legs
        legs_y0 = int(h * 0.50)
        legs_y1 = int(h * 0.92)
        leg_w = int(w * 0.18)
        leg_gap = int(w * 0.08)

        cfg_legs = self.zones[ThermalZone.LEGS]

        # Left leg
        left_leg_x = cx - leg_gap//2 - leg_w
        mask[legs_y0:legs_y1, left_leg_x:left_leg_x+leg_w] = True
        temp_map[legs_y0:legs_y1, left_leg_x:left_leg_x+leg_w] = cfg_legs.temperature_k
        emis_map[legs_y0:legs_y1, left_leg_x:left_leg_x+leg_w] = cfg_legs.emissivity

        # Right leg
        right_leg_x = cx + leg_gap//2
        mask[legs_y0:legs_y1, right_leg_x:right_leg_x+leg_w] = True
        temp_map[legs_y0:legs_y1, right_leg_x:right_leg_x+leg_w] = cfg_legs.temperature_k
        emis_map[legs_y0:legs_y1, right_leg_x:right_leg_x+leg_w] = cfg_legs.emissivity

        # Feet
        feet_y0 = int(h * 0.92)
        feet_y1 = h
        foot_w = int(w * 0.15)

        cfg_feet = self.zones[ThermalZone.FEET]

        mask[feet_y0:feet_y1, left_leg_x-2:left_leg_x+foot_w] = True
        temp_map[feet_y0:feet_y1, left_leg_x-2:left_leg_x+foot_w] = cfg_feet.temperature_k
        emis_map[feet_y0:feet_y1, left_leg_x-2:left_leg_x+foot_w] = cfg_feet.emissivity

        mask[feet_y0:feet_y1, right_leg_x:right_leg_x+foot_w+2] = True
        temp_map[feet_y0:feet_y1, right_leg_x:right_leg_x+foot_w+2] = cfg_feet.temperature_k
        emis_map[feet_y0:feet_y1, right_leg_x:right_leg_x+foot_w+2] = cfg_feet.emissivity

    def _render_walking(
        self,
        temp_map: NDArray,
        emis_map: NDArray,
        mask: NDArray,
        resolution: Tuple[int, int],
        aspect_angle_deg: float,
    ) -> None:
        """Render walking pose (legs apart)."""
        # Start with standing, then adjust leg positions
        self._render_standing(temp_map, emis_map, mask, resolution, aspect_angle_deg)

        # Walking has wider leg stance - already handled by standing pose
        # Could add arm swing animation here for future enhancement

    def _render_sitting(
        self,
        temp_map: NDArray,
        emis_map: NDArray,
        mask: NDArray,
        resolution: Tuple[int, int],
        aspect_angle_deg: float,
    ) -> None:
        """Render sitting pose."""
        h, w = resolution
        cx = w // 2

        # Head
        head_h = int(h * 0.18)
        head_w = int(w * 0.35)
        head_cy = int(h * 0.12)

        yy, xx = np.ogrid[:h, :w]
        head_mask = ((yy - head_cy)**2 / (head_h/2)**2 + (xx - cx)**2 / (head_w/2)**2) <= 1

        cfg = self.zones[ThermalZone.HEAD]
        temp_map[head_mask] = cfg.temperature_k
        emis_map[head_mask] = cfg.emissivity
        mask[head_mask] = True

        # Torso (more compact for sitting)
        torso_y0 = int(h * 0.22)
        torso_y1 = int(h * 0.55)
        torso_w = int(w * 0.55)

        cfg_torso = self.zones[ThermalZone.TORSO]
        mask[torso_y0:torso_y1, cx-torso_w//2:cx+torso_w//2] = True
        temp_map[torso_y0:torso_y1, cx-torso_w//2:cx+torso_w//2] = cfg_torso.temperature_k
        emis_map[torso_y0:torso_y1, cx-torso_w//2:cx+torso_w//2] = cfg_torso.emissivity

        # Legs (horizontal, bent at knees)
        legs_y0 = int(h * 0.55)
        legs_y1 = int(h * 0.75)

        cfg_legs = self.zones[ThermalZone.LEGS]
        mask[legs_y0:legs_y1, int(w*0.15):int(w*0.85)] = True
        temp_map[legs_y0:legs_y1, int(w*0.15):int(w*0.85)] = cfg_legs.temperature_k
        emis_map[legs_y0:legs_y1, int(w*0.15):int(w*0.85)] = cfg_legs.emissivity

        # Lower legs (hanging down)
        lower_legs_y0 = int(h * 0.75)
        lower_legs_y1 = h
        leg_w = int(w * 0.12)

        mask[lower_legs_y0:lower_legs_y1, int(w*0.25):int(w*0.25)+leg_w] = True
        mask[lower_legs_y0:lower_legs_y1, int(w*0.65):int(w*0.65)+leg_w] = True
        temp_map[lower_legs_y0:lower_legs_y1, int(w*0.25):int(w*0.25)+leg_w] = cfg_legs.temperature_k
        temp_map[lower_legs_y0:lower_legs_y1, int(w*0.65):int(w*0.65)+leg_w] = cfg_legs.temperature_k
        emis_map[lower_legs_y0:lower_legs_y1, int(w*0.25):int(w*0.25)+leg_w] = cfg_legs.emissivity
        emis_map[lower_legs_y0:lower_legs_y1, int(w*0.65):int(w*0.65)+leg_w] = cfg_legs.emissivity

    def _render_prone(
        self,
        temp_map: NDArray,
        emis_map: NDArray,
        mask: NDArray,
        resolution: Tuple[int, int],
        aspect_angle_deg: float,
    ) -> None:
        """Render prone (lying down) pose."""
        h, w = resolution
        cy = h // 2

        # Body as elongated ellipse
        body_ry = int(h * 0.25)
        body_rx = int(w * 0.45)

        yy, xx = np.ogrid[:h, :w]
        body_mask = ((yy - cy)**2 / body_ry**2 + (xx - w//2)**2 / body_rx**2) <= 1

        # Apply torso temperature to main body
        cfg_torso = self.zones[ThermalZone.TORSO]
        temp_map[body_mask] = cfg_torso.temperature_k
        emis_map[body_mask] = cfg_torso.emissivity
        mask[body_mask] = True

        # Head at one end
        head_cx = int(w * 0.1)
        head_ry = int(h * 0.15)
        head_rx = int(w * 0.08)

        head_mask = ((yy - cy)**2 / head_ry**2 + (xx - head_cx)**2 / head_rx**2) <= 1

        cfg_head = self.zones[ThermalZone.HEAD]
        temp_map[head_mask] = cfg_head.temperature_k
        emis_map[head_mask] = cfg_head.emissivity
        mask[head_mask] = True


class AircraftWithPlumeShape:
    """Enhanced aircraft shape with exhaust plume thermal gradients.

    Creates realistic aircraft thermal signatures with:
    - Fuselage at ambient-adjusted temperature
    - Wing leading edge heating from aerodynamic friction
    - Engine nozzle hot spots (500K+)
    - Exhaust plume with gradient (500K at nozzle falling to 300K)
    """

    def __init__(
        self,
        length_m: float = 16.0,
        wingspan_m: float = 11.0,
        height_m: float = 5.0,
        throttle: float = 0.5,
        afterburner: bool = False,
        altitude_m: float = 5000.0,
        speed_mach: float = 0.8,
        num_engines: int = 2,
        zone_configs: Optional[dict] = None,
    ):
        """Initialize aircraft shape with plume.

        Args:
            length_m: Aircraft length
            wingspan_m: Wingspan
            height_m: Height
            throttle: Throttle setting (0-1)
            afterburner: Afterburner active
            altitude_m: Altitude in meters
            speed_mach: Speed in Mach number
            num_engines: Number of engines
            zone_configs: Custom thermal zone configurations
        """
        self.length_m = length_m
        self.wingspan_m = wingspan_m
        self.height_m = height_m
        self.throttle = np.clip(throttle, 0, 1)
        self.afterburner = afterburner
        self.altitude_m = altitude_m
        self.speed_mach = speed_mach
        self.num_engines = num_engines

        # Calculate atmospheric temperature at altitude
        self.ambient_k = max(220.0, 288.15 - 0.0065 * altitude_m)

        # Initialize zone configs
        self.zones = dict(DEFAULT_AIRCRAFT_ZONES)
        if zone_configs:
            self.zones.update(zone_configs)

        self._calculate_temperatures()

    def _calculate_temperatures(self) -> None:
        """Calculate zone temperatures based on flight conditions."""
        # Aerodynamic heating from speed
        recovery_factor = 0.9
        gamma = 1.4
        temp_ratio = 1 + recovery_factor * (gamma - 1) / 2 * self.speed_mach**2
        aero_heating = self.ambient_k * (temp_ratio - 1)

        # Fuselage temperature
        fuselage_temp = self.ambient_k + aero_heating * 0.3
        self.zones[ThermalZone.FUSELAGE] = ThermalZoneConfig(
            fuselage_temp, 0.85, 0.05, 2.0
        )

        # Wing leading edge (more heating)
        wing_temp = self.ambient_k + aero_heating * 0.7
        self.zones[ThermalZone.WINGS] = ThermalZoneConfig(
            wing_temp, 0.85, 0.1, 1.5
        )

        # Engine nozzle temperature
        base_nozzle_temp = 400 + self.throttle * 300
        if self.afterburner:
            base_nozzle_temp += 400
        self.zones[ThermalZone.NOZZLE] = ThermalZoneConfig(
            min(base_nozzle_temp, 900), 0.75, 0.2, 10.0
        )

        # Exhaust plume (gradient will be applied during render)
        plume_peak_temp = 300 + self.throttle * 300
        if self.afterburner:
            plume_peak_temp += 300
        self.zones[ThermalZone.PLUME] = ThermalZoneConfig(
            min(plume_peak_temp, 700), 0.3, 0.8, 15.0
        )

    def render(
        self,
        resolution: Tuple[int, int] = (64, 64),
        aspect_angle_deg: float = 0.0,
        elevation_angle_deg: float = 0.0,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Render the aircraft thermal signature with exhaust plume.

        Args:
            resolution: Output (height, width)
            aspect_angle_deg: Viewing angle (0=front, 90=side, 180=rear)
            elevation_angle_deg: Elevation angle

        Returns:
            Tuple of (temperature_map, emissivity_map, mask)
        """
        h, w = resolution

        temp_map = np.zeros((h, w), dtype=np.float64)
        emis_map = np.zeros((h, w), dtype=np.float64)
        mask = np.zeros((h, w), dtype=bool)

        # Create aircraft shape
        self._create_aircraft_body(temp_map, emis_map, mask, resolution, aspect_angle_deg)

        # Apply engine nozzles
        self._apply_engine_nozzles(temp_map, emis_map, mask, resolution, aspect_angle_deg)

        # Apply exhaust plumes with gradient
        self._apply_exhaust_plumes(temp_map, emis_map, mask, resolution, aspect_angle_deg)

        # Add noise
        noise = np.random.normal(0, 3.0, (h, w))
        temp_map = np.where(mask, temp_map + noise, 0)

        return temp_map, emis_map, mask

    def _create_aircraft_body(
        self,
        temp_map: NDArray,
        emis_map: NDArray,
        mask: NDArray,
        resolution: Tuple[int, int],
        aspect_angle_deg: float,
    ) -> None:
        """Create the aircraft fuselage and wings."""
        h, w = resolution
        cx, cy = w // 2, h // 2

        angle_factor = abs(np.cos(np.radians(aspect_angle_deg)))

        # Fuselage (ellipse)
        fuse_ry = int(h * 0.12)
        fuse_rx = int(w * 0.40)

        yy, xx = np.ogrid[:h, :w]
        fuselage = ((yy - cy)**2 / fuse_ry**2 + (xx - cx)**2 / fuse_rx**2) <= 1

        cfg_fuse = self.zones[ThermalZone.FUSELAGE]
        temp_map[fuselage] = cfg_fuse.temperature_k
        emis_map[fuselage] = cfg_fuse.emissivity
        mask[fuselage] = True

        # Wings (visible based on aspect angle)
        if angle_factor > 0.3:
            wing_span = int(h * 0.4 * angle_factor)
            wing_chord = int(w * 0.15)
            wing_x = int(w * 0.35)

            # Left wing
            wing_y0 = cy - wing_span
            wing_y1 = cy - fuse_ry
            if wing_y0 >= 0:
                mask[wing_y0:wing_y1, wing_x:wing_x+wing_chord] = True

            # Right wing
            wing_y0 = cy + fuse_ry
            wing_y1 = cy + wing_span
            if wing_y1 <= h:
                mask[wing_y0:wing_y1, wing_x:wing_x+wing_chord] = True

            cfg_wing = self.zones[ThermalZone.WINGS]
            # Apply wing temperatures with leading edge heating
            for wy in range(max(0, cy - wing_span), min(h, cy + wing_span)):
                if mask[wy, wing_x]:
                    # Leading edge is hotter
                    for wx in range(wing_x, min(w, wing_x + wing_chord)):
                        if mask[wy, wx]:
                            edge_factor = 1.0 - (wx - wing_x) / wing_chord * 0.5
                            temp_map[wy, wx] = cfg_wing.temperature_k * edge_factor + self.ambient_k * (1 - edge_factor)
                            emis_map[wy, wx] = cfg_wing.emissivity

        # Tail
        tail_h = int(h * 0.08)
        tail_w = int(w * 0.06)
        tail_x = int(w * 0.82)

        mask[cy-tail_h:cy+tail_h, tail_x:tail_x+tail_w] = True
        temp_map[cy-tail_h:cy+tail_h, tail_x:tail_x+tail_w] = cfg_fuse.temperature_k
        emis_map[cy-tail_h:cy+tail_h, tail_x:tail_x+tail_w] = cfg_fuse.emissivity

    def _apply_engine_nozzles(
        self,
        temp_map: NDArray,
        emis_map: NDArray,
        mask: NDArray,
        resolution: Tuple[int, int],
        aspect_angle_deg: float,
    ) -> None:
        """Apply engine nozzle hot spots."""
        h, w = resolution
        cy = h // 2

        cfg = self.zones[ThermalZone.NOZZLE]
        nozzle_radius = int(min(h, w) * 0.05)
        nozzle_x = int(w * 0.88)

        yy, xx = np.ogrid[:h, :w]

        if self.num_engines == 1:
            positions = [cy]
        elif self.num_engines == 2:
            offset = int(h * 0.12)
            positions = [cy - offset, cy + offset]
        else:
            offset = int(h * 0.10)
            positions = [cy - 2*offset, cy - offset, cy + offset, cy + 2*offset]

        for nozzle_y in positions:
            dist = np.sqrt((yy - nozzle_y)**2 + (xx - nozzle_x)**2)
            nozzle_mask = dist <= nozzle_radius

            # Nozzle has gradient from center
            gradient = create_gradient_falloff((h, w), (nozzle_y, nozzle_x), nozzle_radius, 0.3)

            temp_map[nozzle_mask] = cfg.temperature_k * gradient[nozzle_mask]
            emis_map[nozzle_mask] = cfg.emissivity
            mask[nozzle_mask] = True

    def _apply_exhaust_plumes(
        self,
        temp_map: NDArray,
        emis_map: NDArray,
        mask: NDArray,
        resolution: Tuple[int, int],
        aspect_angle_deg: float,
    ) -> None:
        """Apply exhaust plumes with thermal gradient."""
        h, w = resolution
        cy = h // 2

        cfg_plume = self.zones[ThermalZone.PLUME]
        cfg_nozzle = self.zones[ThermalZone.NOZZLE]

        # Plume length depends on throttle
        plume_length = int(w * 0.15 * self.throttle)
        if self.afterburner:
            plume_length = int(plume_length * 1.8)

        plume_start_x = int(w * 0.90)
        plume_end_x = min(w, plume_start_x + plume_length)
        plume_base_width = int(h * 0.04)

        if self.num_engines == 1:
            positions = [cy]
        elif self.num_engines == 2:
            offset = int(h * 0.12)
            positions = [cy - offset, cy + offset]
        else:
            offset = int(h * 0.10)
            positions = [cy - 2*offset, cy - offset, cy + offset, cy + 2*offset]

        for plume_cy in positions:
            for px in range(plume_start_x, plume_end_x):
                # Linear gradient from nozzle temperature to ambient
                t = (px - plume_start_x) / max(1, plume_length)
                plume_temp = cfg_nozzle.temperature_k * (1 - t) + self.ambient_k * t

                # Plume width expands with distance
                current_width = int(plume_base_width * (1 + t * 2))

                y0 = max(0, plume_cy - current_width // 2)
                y1 = min(h, plume_cy + current_width // 2)

                # Add turbulent variation
                for py in range(y0, y1):
                    # Plume intensity falls off radially
                    radial_dist = abs(py - plume_cy) / max(1, current_width / 2)
                    radial_factor = max(0, 1 - radial_dist**2)

                    pixel_temp = plume_temp * radial_factor + self.ambient_k * (1 - radial_factor)

                    # Add turbulence
                    turbulence = np.random.normal(0, cfg_plume.noise_std_k * t)
                    pixel_temp += turbulence

                    if px < w:
                        temp_map[py, px] = max(temp_map[py, px], pixel_temp)
                        emis_map[py, px] = cfg_plume.emissivity
                        mask[py, px] = True


# Convenience functions for creating common shapes

def create_vehicle_signature(
    vehicle_type: str = "sedan",
    engine_running: bool = True,
    speed_kmh: float = 60.0,
    resolution: Tuple[int, int] = (64, 64),
    aspect_angle_deg: float = 90.0,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Create a vehicle thermal signature with realistic thermal zones.

    Args:
        vehicle_type: Type of vehicle ("sedan", "suv", "truck", "tank")
        engine_running: Whether engine is running
        speed_kmh: Vehicle speed in km/h
        resolution: Output resolution
        aspect_angle_deg: Viewing angle

    Returns:
        Tuple of (temperature_map, emissivity_map, mask)
    """
    dimensions = {
        "sedan": (4.5, 1.8, 1.4),
        "suv": (4.8, 2.0, 1.8),
        "truck": (12.0, 2.5, 4.0),
        "tank": (9.5, 3.7, 2.4),
    }

    length, width, height = dimensions.get(vehicle_type, (4.5, 1.8, 1.4))

    shape = EnhancedVehicleShape(
        length_m=length,
        width_m=width,
        height_m=height,
        engine_running=engine_running,
        speed_kmh=speed_kmh,
        vehicle_type=vehicle_type,
    )

    return shape.render(resolution, aspect_angle_deg)


def create_person_signature(
    pose: str = "standing",
    activity: str = "walking",
    resolution: Tuple[int, int] = (64, 32),
    aspect_angle_deg: float = 0.0,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Create a person thermal signature with anatomical thermal zones.

    Args:
        pose: Body pose ("standing", "walking", "sitting", "prone")
        activity: Activity level
        resolution: Output resolution
        aspect_angle_deg: Viewing angle

    Returns:
        Tuple of (temperature_map, emissivity_map, mask)
    """
    shape = HumanoidShape(pose=pose, activity=activity)
    return shape.render(resolution, aspect_angle_deg)


def create_aircraft_signature(
    aircraft_type: str = "fighter",
    throttle: float = 0.7,
    afterburner: bool = False,
    altitude_m: float = 8000.0,
    resolution: Tuple[int, int] = (64, 64),
    aspect_angle_deg: float = 180.0,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Create an aircraft thermal signature with exhaust plume gradient.

    Args:
        aircraft_type: Type of aircraft ("fighter", "airliner", "turboprop")
        throttle: Throttle setting (0-1)
        afterburner: Afterburner active
        altitude_m: Altitude in meters
        resolution: Output resolution
        aspect_angle_deg: Viewing angle (180 = rear view for plume)

    Returns:
        Tuple of (temperature_map, emissivity_map, mask)
    """
    configs = {
        "fighter": (16.0, 11.0, 5.0, 2, 0.9),
        "airliner": (60.0, 50.0, 15.0, 2, 0.82),
        "turboprop": (20.0, 25.0, 6.0, 2, 0.45),
    }

    length, wingspan, height, engines, mach = configs.get(
        aircraft_type, (16.0, 11.0, 5.0, 2, 0.9)
    )

    shape = AircraftWithPlumeShape(
        length_m=length,
        wingspan_m=wingspan,
        height_m=height,
        throttle=throttle,
        afterburner=afterburner,
        altitude_m=altitude_m,
        speed_mach=mach,
        num_engines=engines,
    )

    return shape.render(resolution, aspect_angle_deg)


__all__ = [
    # Enums and configs
    "ThermalZone",
    "ThermalZoneConfig",
    # Default configurations
    "DEFAULT_VEHICLE_ZONES",
    "DEFAULT_PERSON_ZONES",
    "DEFAULT_AIRCRAFT_ZONES",
    # Utility functions
    "create_gradient_falloff",
    "create_linear_gradient",
    # Shape classes
    "EnhancedVehicleShape",
    "HumanoidShape",
    "AircraftWithPlumeShape",
    # Convenience functions
    "create_vehicle_signature",
    "create_person_signature",
    "create_aircraft_signature",
]
