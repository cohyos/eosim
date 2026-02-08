"""
EOSIM Scenario System.

Provides a framework for creating and running sensor vs target engagement scenarios:
- Engagement geometry (range, aspect, elevation)
- Target tracks (moving targets with time-varying positions)
- Multi-target scenarios
- Sensor-target pairing with automatic parameter calculation
- Pre-defined scenario templates

Example Usage:
-------------
# Quick scenario
scenario = create_scenario(
    sensor="mx15",
    target="f16",
    range_km=10.0,
    aspect_deg=90,  # Side view
)
result = run_scenario(scenario)

# Complex multi-target scenario
builder = ScenarioBuilder()
builder.set_sensor("mx20")
builder.add_target("t90", position_km=(5.0, 0.0, 0.0))
builder.add_target("bmp3", position_km=(5.5, 0.2, 0.0))
builder.set_environment(temperature_k=295, visibility_km=10)
scenario = builder.build()
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any, Union, Callable
import numpy as np
from numpy.typing import NDArray

from eosim.library.objects import (
    ObjectLibrary,
    Object3D,
    ObjectCategory,
    get_object,
)
from eosim.library.sensors import (
    SensorLibrary,
    SensorSpec,
    get_sensor,
    create_sensor_from_spec,
)


class EnvironmentType(Enum):
    """Pre-defined environment types."""
    CLEAR_DAY = "clear_day"
    CLEAR_NIGHT = "clear_night"
    OVERCAST = "overcast"
    HAZE = "haze"
    FOG = "fog"
    RAIN = "rain"
    DESERT_DAY = "desert_day"
    ARCTIC = "arctic"
    MARITIME = "maritime"
    URBAN = "urban"


class BackgroundType(Enum):
    """Background scene types."""
    SKY = "sky"
    TERRAIN = "terrain"
    URBAN = "urban"
    FOREST = "forest"
    DESERT = "desert"
    WATER = "water"
    SNOW = "snow"
    MIXED = "mixed"


@dataclass
class Position3D:
    """3D position in meters or kilometers.

    Attributes:
        x: East-West position (positive = East)
        y: North-South position (positive = North)
        z: Altitude (positive = Up)
        unit: "m" or "km"
    """
    x: float
    y: float
    z: float
    unit: str = "m"

    def to_meters(self) -> "Position3D":
        """Convert to meters."""
        if self.unit == "km":
            return Position3D(
                self.x * 1000,
                self.y * 1000,
                self.z * 1000,
                "m"
            )
        return self

    def to_km(self) -> "Position3D":
        """Convert to kilometers."""
        if self.unit == "m":
            return Position3D(
                self.x / 1000,
                self.y / 1000,
                self.z / 1000,
                "km"
            )
        return self

    @property
    def range_m(self) -> float:
        """Range from origin in meters."""
        pos = self.to_meters()
        return np.sqrt(pos.x**2 + pos.y**2 + pos.z**2)

    @property
    def range_km(self) -> float:
        """Range from origin in kilometers."""
        return self.range_m / 1000

    @property
    def azimuth_deg(self) -> float:
        """Azimuth angle from North (degrees, clockwise)."""
        pos = self.to_meters()
        return np.rad2deg(np.arctan2(pos.x, pos.y)) % 360

    @property
    def elevation_deg(self) -> float:
        """Elevation angle (degrees, positive = up)."""
        pos = self.to_meters()
        ground_range = np.sqrt(pos.x**2 + pos.y**2)
        return np.rad2deg(np.arctan2(pos.z, ground_range))


@dataclass
class Velocity3D:
    """3D velocity vector.

    Attributes:
        vx: East-West velocity (positive = East)
        vy: North-South velocity (positive = North)
        vz: Vertical velocity (positive = Up)
        unit: "m/s" or "km/h" or "knots"
    """
    vx: float
    vy: float
    vz: float
    unit: str = "m/s"

    def to_ms(self) -> "Velocity3D":
        """Convert to m/s."""
        if self.unit == "km/h":
            scale = 1000 / 3600
        elif self.unit == "knots":
            scale = 0.514444
        else:
            scale = 1.0
        return Velocity3D(
            self.vx * scale,
            self.vy * scale,
            self.vz * scale,
            "m/s"
        )

    @property
    def speed_ms(self) -> float:
        """Speed magnitude in m/s."""
        vel = self.to_ms()
        return np.sqrt(vel.vx**2 + vel.vy**2 + vel.vz**2)

    @property
    def heading_deg(self) -> float:
        """Heading from North (degrees, clockwise)."""
        vel = self.to_ms()
        return np.rad2deg(np.arctan2(vel.vx, vel.vy)) % 360


@dataclass
class EngagementGeometry:
    """Defines sensor-to-target geometry.

    Attributes:
        range_m: Slant range to target in meters
        aspect_angle_deg: Target aspect angle (0=nose, 90=side, 180=tail)
        elevation_angle_deg: Elevation angle (0=level, +ve=above)
        sensor_altitude_m: Sensor platform altitude
        target_altitude_m: Target altitude
        look_down_angle_deg: Sensor depression angle
    """
    range_m: float
    aspect_angle_deg: float = 90.0
    elevation_angle_deg: float = 0.0
    sensor_altitude_m: float = 1000.0
    target_altitude_m: float = 0.0
    look_down_angle_deg: Optional[float] = None

    def __post_init__(self):
        if self.look_down_angle_deg is None:
            # Calculate from altitudes and range
            alt_diff = self.sensor_altitude_m - self.target_altitude_m
            if self.range_m > 0:
                self.look_down_angle_deg = np.rad2deg(np.arcsin(
                    min(1.0, alt_diff / self.range_m)
                ))
            else:
                self.look_down_angle_deg = 0.0

    @property
    def ground_range_m(self) -> float:
        """Horizontal ground range."""
        return np.sqrt(
            self.range_m**2 -
            (self.sensor_altitude_m - self.target_altitude_m)**2
        )

    @property
    def range_km(self) -> float:
        """Slant range in kilometers."""
        return self.range_m / 1000

    @classmethod
    def from_positions(
        cls,
        sensor_pos: Position3D,
        target_pos: Position3D,
        target_heading_deg: float = 0.0,
    ) -> "EngagementGeometry":
        """Create geometry from sensor and target positions.

        Args:
            sensor_pos: Sensor position
            target_pos: Target position
            target_heading_deg: Target heading (0=North)

        Returns:
            EngagementGeometry
        """
        sp = sensor_pos.to_meters()
        tp = target_pos.to_meters()

        # Calculate range
        dx = tp.x - sp.x
        dy = tp.y - sp.y
        dz = tp.z - sp.z
        range_m = np.sqrt(dx**2 + dy**2 + dz**2)

        # Calculate bearing from sensor to target
        bearing_deg = np.rad2deg(np.arctan2(dx, dy)) % 360

        # Aspect angle = difference between target heading and sensor bearing
        # 0 = target nose toward sensor, 180 = tail toward sensor
        aspect_angle_deg = (bearing_deg - target_heading_deg + 180) % 360

        # Elevation angle
        ground_range = np.sqrt(dx**2 + dy**2)
        elevation_angle_deg = np.rad2deg(np.arctan2(dz, ground_range))

        return cls(
            range_m=range_m,
            aspect_angle_deg=aspect_angle_deg,
            elevation_angle_deg=elevation_angle_deg,
            sensor_altitude_m=sp.z,
            target_altitude_m=tp.z,
        )


@dataclass
class TargetTrack:
    """A target with position and optional trajectory.

    Attributes:
        object_id: ID from object library
        initial_position: Starting position
        velocity: Velocity vector (optional for stationary targets)
        heading_deg: Target heading/orientation
        name: Optional custom name
        state: Additional state parameters (engine_on, etc.)
    """
    object_id: str
    initial_position: Position3D
    velocity: Optional[Velocity3D] = None
    heading_deg: float = 0.0
    name: Optional[str] = None
    state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.name is None:
            self.name = self.object_id

    def get_position_at_time(self, t_seconds: float) -> Position3D:
        """Get target position at a specific time.

        Args:
            t_seconds: Time in seconds from start

        Returns:
            Position at time t
        """
        if self.velocity is None:
            return self.initial_position

        p = self.initial_position.to_meters()
        v = self.velocity.to_ms()

        return Position3D(
            x=p.x + v.vx * t_seconds,
            y=p.y + v.vy * t_seconds,
            z=p.z + v.vz * t_seconds,
            unit="m"
        )

    def get_heading_at_time(self, t_seconds: float) -> float:
        """Get target heading at a specific time.

        For now, assumes constant heading. Could be extended for maneuvering.
        """
        return self.heading_deg

    @property
    def object(self) -> Object3D:
        """Get the object specification."""
        return get_object(self.object_id)


@dataclass
class EnvironmentConditions:
    """Environmental conditions for the scenario.

    Attributes:
        ambient_temperature_k: Ambient air temperature
        sky_temperature_k: Effective sky temperature
        ground_temperature_k: Ground surface temperature
        visibility_km: Atmospheric visibility
        humidity_percent: Relative humidity
        wind_speed_ms: Wind speed
        time_of_day: Time (0-24 hours)
        cloud_cover: Cloud coverage fraction (0-1)
    """
    ambient_temperature_k: float = 290.0
    sky_temperature_k: float = 230.0
    ground_temperature_k: float = 295.0
    visibility_km: float = 23.0  # Standard clear
    humidity_percent: float = 50.0
    wind_speed_ms: float = 5.0
    time_of_day: float = 12.0  # Noon
    cloud_cover: float = 0.0

    @classmethod
    def from_type(cls, env_type: EnvironmentType) -> "EnvironmentConditions":
        """Create conditions from environment type."""
        presets = {
            EnvironmentType.CLEAR_DAY: cls(
                ambient_temperature_k=295,
                sky_temperature_k=250,
                ground_temperature_k=305,
                visibility_km=23,
                time_of_day=12,
            ),
            EnvironmentType.CLEAR_NIGHT: cls(
                ambient_temperature_k=285,
                sky_temperature_k=220,
                ground_temperature_k=280,
                visibility_km=23,
                time_of_day=2,
            ),
            EnvironmentType.OVERCAST: cls(
                ambient_temperature_k=288,
                sky_temperature_k=270,
                ground_temperature_k=290,
                visibility_km=15,
                cloud_cover=0.8,
            ),
            EnvironmentType.HAZE: cls(
                ambient_temperature_k=295,
                sky_temperature_k=260,
                visibility_km=8,
                humidity_percent=70,
            ),
            EnvironmentType.FOG: cls(
                ambient_temperature_k=283,
                sky_temperature_k=275,
                visibility_km=1.0,
                humidity_percent=95,
            ),
            EnvironmentType.RAIN: cls(
                ambient_temperature_k=285,
                sky_temperature_k=280,
                visibility_km=5,
                humidity_percent=90,
                cloud_cover=1.0,
            ),
            EnvironmentType.DESERT_DAY: cls(
                ambient_temperature_k=315,
                sky_temperature_k=240,
                ground_temperature_k=340,
                visibility_km=20,
                humidity_percent=20,
            ),
            EnvironmentType.ARCTIC: cls(
                ambient_temperature_k=250,
                sky_temperature_k=210,
                ground_temperature_k=255,
                visibility_km=30,
                humidity_percent=60,
            ),
            EnvironmentType.MARITIME: cls(
                ambient_temperature_k=290,
                sky_temperature_k=245,
                ground_temperature_k=288,  # Sea surface
                visibility_km=15,
                humidity_percent=80,
            ),
            EnvironmentType.URBAN: cls(
                ambient_temperature_k=298,
                sky_temperature_k=255,
                ground_temperature_k=305,
                visibility_km=10,
            ),
        }
        return presets.get(env_type, cls())


@dataclass
class ScenarioOutput:
    """Output from scenario simulation.

    Attributes:
        digital_image: Sensor output image
        temperature_map: Scene temperature map
        target_pixels: Dictionary of target name -> pixel coordinates
        metadata: Additional scenario metadata
        detection_metrics: Target detection metrics
        frames: List of frames for video scenarios
    """
    digital_image: NDArray[np.integer]
    temperature_map: NDArray[np.floating]
    target_pixels: Dict[str, Tuple[int, int]]
    metadata: Dict[str, Any]
    detection_metrics: Optional[Dict[str, float]] = None
    frames: Optional[List[NDArray]] = None

    @property
    def has_video(self) -> bool:
        """Whether this output contains video frames."""
        return self.frames is not None and len(self.frames) > 0


@dataclass
class PlatformConfig:
    """Configuration for 6DOF sensor platform motion.

    Attributes:
        platform_type: Type of platform (fixed_wing, rotary_wing, etc.)
        velocity: Initial platform velocity
        orientation: Initial platform orientation
        gimbal_mode: Gimbal operating mode
        track_target_idx: Index of target to track (-1 for no tracking)
        orbit_center: Center point for orbit mode
        orbit_radius_m: Orbit radius
        waypoints: List of waypoints for trajectory
    """
    platform_type: str = "fixed_wing"
    velocity: Optional[Velocity3D] = None
    orientation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # roll, pitch, yaw
    gimbal_mode: str = "stabilized"
    track_target_idx: int = 0  # Track first target by default
    orbit_center: Optional[Position3D] = None
    orbit_radius_m: float = 5000.0
    waypoints: List[Any] = field(default_factory=list)


@dataclass
class Scenario:
    """Complete scenario definition.

    Combines sensor, targets, environment, and simulation parameters.
    """
    name: str
    sensor_id: str
    targets: List[TargetTrack]
    sensor_position: Position3D
    environment: EnvironmentConditions
    background: BackgroundType = BackgroundType.TERRAIN

    # Simulation parameters
    resolution: Tuple[int, int] = (480, 640)
    duration_s: float = 0.0  # 0 = single frame
    frame_rate_hz: float = 30.0
    seed: Optional[int] = None

    # Output options
    compute_detection: bool = True
    save_intermediate: bool = False

    # 6DOF Platform configuration
    platform_config: Optional[PlatformConfig] = None

    @property
    def sensor(self) -> SensorSpec:
        """Get sensor specification."""
        return get_sensor(self.sensor_id)

    @property
    def n_targets(self) -> int:
        """Number of targets."""
        return len(self.targets)

    @property
    def is_video(self) -> bool:
        """Whether this is a video scenario."""
        return self.duration_s > 0

    @property
    def n_frames(self) -> int:
        """Number of frames for video scenarios."""
        if not self.is_video:
            return 1
        return int(self.duration_s * self.frame_rate_hz)

    def get_geometry(self, target_idx: int = 0, time_s: float = 0.0) -> EngagementGeometry:
        """Get engagement geometry for a target at a specific time."""
        if target_idx >= len(self.targets):
            raise IndexError(f"Target index {target_idx} out of range")

        track = self.targets[target_idx]
        target_pos = track.get_position_at_time(time_s)
        target_heading = track.get_heading_at_time(time_s)

        return EngagementGeometry.from_positions(
            self.sensor_position,
            target_pos,
            target_heading,
        )


class ScenarioBuilder:
    """Builder for creating scenarios step by step."""

    def __init__(self):
        self._name: str = "scenario"
        self._sensor_id: Optional[str] = None
        self._sensor_position: Position3D = Position3D(0, 0, 1000, "m")
        self._targets: List[TargetTrack] = []
        self._environment: EnvironmentConditions = EnvironmentConditions()
        self._background: BackgroundType = BackgroundType.TERRAIN
        self._resolution: Tuple[int, int] = (480, 640)
        self._duration_s: float = 0.0
        self._frame_rate_hz: float = 30.0
        self._seed: Optional[int] = None
        self._platform_config: Optional[PlatformConfig] = None

    def set_name(self, name: str) -> "ScenarioBuilder":
        """Set scenario name."""
        self._name = name
        return self

    def set_sensor(
        self,
        sensor_id: str,
        position: Optional[Position3D] = None,
        altitude_m: Optional[float] = None,
    ) -> "ScenarioBuilder":
        """Set the sensor.

        Args:
            sensor_id: Sensor ID from library
            position: Full position (optional)
            altitude_m: Simple altitude setting (ignored if position given)
        """
        self._sensor_id = sensor_id
        if position is not None:
            self._sensor_position = position
        elif altitude_m is not None:
            self._sensor_position = Position3D(0, 0, altitude_m, "m")
        return self

    def add_target(
        self,
        object_id: str,
        position_km: Optional[Tuple[float, float, float]] = None,
        position: Optional[Position3D] = None,
        range_km: Optional[float] = None,
        aspect_deg: float = 90.0,
        heading_deg: float = 0.0,
        velocity_ms: Optional[Tuple[float, float, float]] = None,
        name: Optional[str] = None,
        **state,
    ) -> "ScenarioBuilder":
        """Add a target to the scenario.

        Args:
            object_id: Object ID from library
            position_km: Position as (x, y, z) in km
            position: Full Position3D object
            range_km: Simple range setting (creates position from range/aspect)
            aspect_deg: Aspect angle (used with range_km)
            heading_deg: Target heading
            velocity_ms: Velocity as (vx, vy, vz) in m/s
            name: Custom name for target
            **state: Additional state parameters
        """
        if position is not None:
            pos = position
        elif position_km is not None:
            pos = Position3D(*position_km, "km")
        elif range_km is not None:
            # Create position from range and aspect
            range_m = range_km * 1000
            aspect_rad = np.deg2rad(aspect_deg)
            # Simple geometry: target at range, aspect angle
            x = range_m * np.sin(aspect_rad)
            y = range_m * np.cos(aspect_rad)
            z = 0.0
            pos = Position3D(x, y, z, "m")
        else:
            pos = Position3D(5000, 0, 0, "m")  # Default 5km east

        vel = None
        if velocity_ms is not None:
            vel = Velocity3D(*velocity_ms, "m/s")

        track = TargetTrack(
            object_id=object_id,
            initial_position=pos,
            velocity=vel,
            heading_deg=heading_deg,
            name=name,
            state=state,
        )
        self._targets.append(track)
        return self

    def set_environment(
        self,
        env_type: Optional[EnvironmentType] = None,
        temperature_k: Optional[float] = None,
        visibility_km: Optional[float] = None,
        **kwargs,
    ) -> "ScenarioBuilder":
        """Set environment conditions.

        Args:
            env_type: Pre-defined environment type
            temperature_k: Override ambient temperature
            visibility_km: Override visibility
            **kwargs: Additional EnvironmentConditions parameters
        """
        if env_type is not None:
            self._environment = EnvironmentConditions.from_type(env_type)
        else:
            self._environment = EnvironmentConditions(**kwargs)

        if temperature_k is not None:
            self._environment.ambient_temperature_k = temperature_k
        if visibility_km is not None:
            self._environment.visibility_km = visibility_km

        return self

    def set_background(self, background: BackgroundType) -> "ScenarioBuilder":
        """Set background type."""
        self._background = background
        return self

    def set_resolution(self, height: int, width: int) -> "ScenarioBuilder":
        """Set output resolution."""
        self._resolution = (height, width)
        return self

    def set_video(
        self,
        duration_s: float,
        frame_rate_hz: float = 30.0,
    ) -> "ScenarioBuilder":
        """Configure video output.

        Args:
            duration_s: Video duration in seconds
            frame_rate_hz: Frame rate
        """
        self._duration_s = duration_s
        self._frame_rate_hz = frame_rate_hz
        return self

    def set_seed(self, seed: int) -> "ScenarioBuilder":
        """Set random seed for reproducibility."""
        self._seed = seed
        return self

    def set_platform(
        self,
        platform_type: str = "fixed_wing",
        velocity_ms: Optional[Tuple[float, float, float]] = None,
        speed_ms: Optional[float] = None,
        heading_deg: float = 0.0,
        orientation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "ScenarioBuilder":
        """Configure 6DOF sensor platform motion.

        Args:
            platform_type: Platform type ("fixed_wing", "rotary_wing", "ground_vehicle", etc.)
            velocity_ms: Full velocity vector (vx, vy, vz) in m/s
            speed_ms: Speed magnitude (used with heading if velocity not specified)
            heading_deg: Heading in degrees (0=North, 90=East)
            orientation_deg: Platform orientation as (roll, pitch, yaw) in degrees

        Returns:
            Self for chaining
        """
        if velocity_ms is not None:
            vel = Velocity3D(*velocity_ms, "m/s")
        elif speed_ms is not None:
            heading_rad = np.deg2rad(heading_deg)
            vel = Velocity3D(
                speed_ms * np.sin(heading_rad),
                speed_ms * np.cos(heading_rad),
                0,
                "m/s"
            )
        else:
            vel = None

        self._platform_config = PlatformConfig(
            platform_type=platform_type,
            velocity=vel,
            orientation_deg=orientation_deg,
        )
        return self

    def set_gimbal_track(self, target_idx: int = 0) -> "ScenarioBuilder":
        """Set gimbal to track a specific target.

        Args:
            target_idx: Index of target to track (0-based)

        Returns:
            Self for chaining
        """
        if self._platform_config is None:
            self._platform_config = PlatformConfig()
        self._platform_config.gimbal_mode = "track_target"
        self._platform_config.track_target_idx = target_idx
        return self

    def set_gimbal_point(self, azimuth_deg: float, elevation_deg: float) -> "ScenarioBuilder":
        """Set gimbal to point at fixed angles.

        Args:
            azimuth_deg: Azimuth angle (0=forward, positive=right)
            elevation_deg: Elevation angle (negative=down)

        Returns:
            Self for chaining
        """
        if self._platform_config is None:
            self._platform_config = PlatformConfig()
        self._platform_config.gimbal_mode = "position"
        return self

    def set_orbit(
        self,
        center: Position3D,
        radius_m: float = 5000.0,
    ) -> "ScenarioBuilder":
        """Set platform to orbit around a point.

        Args:
            center: Center of orbit
            radius_m: Orbit radius in meters

        Returns:
            Self for chaining
        """
        if self._platform_config is None:
            self._platform_config = PlatformConfig()
        self._platform_config.orbit_center = center
        self._platform_config.orbit_radius_m = radius_m
        return self

    def add_waypoint(
        self,
        position: Position3D,
        time_s: Optional[float] = None,
        heading_deg: Optional[float] = None,
    ) -> "ScenarioBuilder":
        """Add a waypoint to the platform trajectory.

        Args:
            position: Waypoint position
            time_s: Optional time to reach waypoint
            heading_deg: Optional heading at waypoint

        Returns:
            Self for chaining
        """
        if self._platform_config is None:
            self._platform_config = PlatformConfig()

        waypoint = {
            "position": position,
            "time_s": time_s,
            "heading_deg": heading_deg,
        }
        self._platform_config.waypoints.append(waypoint)
        return self

    def build(self) -> Scenario:
        """Build the scenario."""
        if self._sensor_id is None:
            raise ValueError("Sensor must be set")
        if not self._targets:
            raise ValueError("At least one target must be added")

        return Scenario(
            name=self._name,
            sensor_id=self._sensor_id,
            targets=self._targets,
            sensor_position=self._sensor_position,
            environment=self._environment,
            background=self._background,
            resolution=self._resolution,
            duration_s=self._duration_s,
            frame_rate_hz=self._frame_rate_hz,
            seed=self._seed,
            platform_config=self._platform_config,
        )


# =============================================================================
# Scenario Execution
# =============================================================================

def _create_background_scene(
    resolution: Tuple[int, int],
    background: BackgroundType,
    environment: EnvironmentConditions,
    seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray]:
    """Create background temperature and emissivity maps."""
    rng = np.random.default_rng(seed)
    h, w = resolution

    # Base background temperature
    base_temps = {
        BackgroundType.SKY: environment.sky_temperature_k,
        BackgroundType.TERRAIN: environment.ground_temperature_k,
        BackgroundType.URBAN: environment.ground_temperature_k + 5,
        BackgroundType.FOREST: environment.ambient_temperature_k - 2,
        BackgroundType.DESERT: environment.ground_temperature_k + 10,
        BackgroundType.WATER: environment.ground_temperature_k - 5,
        BackgroundType.SNOW: environment.ambient_temperature_k - 10,
        BackgroundType.MIXED: environment.ground_temperature_k,
    }

    base_emis = {
        BackgroundType.SKY: 0.95,
        BackgroundType.TERRAIN: 0.93,
        BackgroundType.URBAN: 0.90,
        BackgroundType.FOREST: 0.97,
        BackgroundType.DESERT: 0.92,
        BackgroundType.WATER: 0.96,
        BackgroundType.SNOW: 0.85,
        BackgroundType.MIXED: 0.94,
    }

    base_temp = base_temps.get(background, environment.ground_temperature_k)
    base_e = base_emis.get(background, 0.93)

    # Add texture
    noise_std = 2.0 if background != BackgroundType.SKY else 1.0
    temp_map = base_temp + rng.normal(0, noise_std, (h, w))

    # Add low-frequency variation
    from scipy.ndimage import gaussian_filter
    variation = gaussian_filter(rng.normal(0, 5, (h, w)), sigma=30)
    temp_map += variation

    emissivity_map = np.full((h, w), base_e, dtype=np.float64)
    emissivity_map += rng.normal(0, 0.02, (h, w))
    emissivity_map = np.clip(emissivity_map, 0.5, 1.0)

    return temp_map.astype(np.float64), emissivity_map


def _render_target_at_position(
    temp_map: NDArray,
    emis_map: NDArray,
    target: TargetTrack,
    pixel_center: Tuple[int, int],
    pixels_per_meter: float,
    aspect_angle_deg: float,
    elevation_angle_deg: float,
) -> Tuple[int, int]:
    """Render a target into the scene maps.

    Returns:
        Actual pixel center of placed target
    """
    obj = target.object

    # Calculate target size in pixels
    # Use projected dimensions based on aspect
    aspect_rad = np.deg2rad(aspect_angle_deg)

    # Simple projection: side view shows length, front view shows width
    proj_length = abs(np.cos(aspect_rad)) * obj.dimensions.width_m + \
                  abs(np.sin(aspect_rad)) * obj.dimensions.length_m
    proj_height = obj.dimensions.height_m

    target_h = max(8, int(proj_height * pixels_per_meter))
    target_w = max(8, int(proj_length * pixels_per_meter))

    # Get target signature
    target_temp, target_emis = obj.get_signature(
        resolution=(target_h, target_w),
        aspect_angle_deg=aspect_angle_deg,
        elevation_angle_deg=elevation_angle_deg,
        speed_ms=target.velocity.speed_ms if target.velocity else 0,
        altitude_km=target.initial_position.to_km().z,
    )

    # Place in scene
    h, w = temp_map.shape
    cy, cx = pixel_center
    ty, tx = target_h // 2, target_w // 2

    # Clamp to scene bounds
    y0 = max(0, cy - ty)
    y1 = min(h, cy + ty)
    x0 = max(0, cx - tx)
    x1 = min(w, cx + tx)

    # Source region
    sy0 = ty - (cy - y0)
    sy1 = sy0 + (y1 - y0)
    sx0 = tx - (cx - x0)
    sx1 = sx0 + (x1 - x0)

    if y1 > y0 and x1 > x0:
        src_temp = target_temp[sy0:sy1, sx0:sx1]
        src_emis = target_emis[sy0:sy1, sx0:sx1]

        # Only place non-zero pixels (object mask)
        mask = src_temp > 0
        if mask.any():
            temp_map[y0:y1, x0:x1] = np.where(mask, src_temp, temp_map[y0:y1, x0:x1])
            emis_map[y0:y1, x0:x1] = np.where(mask, src_emis, emis_map[y0:y1, x0:x1])

    return (cy, cx)


def run_scenario(
    scenario: Scenario,
    verbose: bool = False,
    apply_motion_effects: bool = True,
) -> ScenarioOutput:
    """Run a scenario simulation.

    Args:
        scenario: Scenario definition
        verbose: Print progress
        apply_motion_effects: Whether to apply platform motion effects (blur, jitter)

    Returns:
        ScenarioOutput with simulation results
    """
    from eosim.pipeline import create_pipeline, SceneInput
    from eosim.library.platform import (
        SensorPlatform,
        PlatformType,
        GimbalMode,
        Waypoint,
        apply_platform_motion_effects,
    )

    if verbose:
        print(f"Running scenario: {scenario.name}")
        print(f"  Sensor: {scenario.sensor.name}")
        print(f"  Targets: {scenario.n_targets}")
        print(f"  Frames: {scenario.n_frames}")

    # Get sensor spec and create pipeline
    sensor_spec = scenario.sensor
    band = sensor_spec.detector_ir.spectral_band_um

    # Determine sensor type from band
    if band[1] <= 2.5:
        sensor_type = "swir"
    elif band[1] <= 5.5:
        sensor_type = "mwir"
    else:
        sensor_type = "lwir"

    pipeline = create_pipeline(
        sensor_type=sensor_type,
        seed=scenario.seed,
    )

    # Create 6DOF platform if configured
    platform = None
    if scenario.platform_config is not None:
        pc = scenario.platform_config

        # Map string to PlatformType enum
        try:
            platform_type = PlatformType(pc.platform_type)
        except ValueError:
            platform_type = PlatformType.FIXED_WING

        # Create platform
        platform = SensorPlatform(
            platform_type=platform_type,
            initial_position=scenario.sensor_position,
            initial_velocity=pc.velocity,
            seed=scenario.seed,
        )

        # Configure gimbal tracking
        if pc.gimbal_mode == "track_target" and scenario.targets:
            track_idx = min(pc.track_target_idx, len(scenario.targets) - 1)
            target = scenario.targets[track_idx]
            platform.gimbal.track_point(
                target.initial_position,
                target.velocity,
            )
        elif pc.gimbal_mode == "position":
            platform.gimbal.set_mode(GimbalMode.POSITION)
        elif pc.gimbal_mode == "stabilized":
            platform.gimbal.set_mode(GimbalMode.STABILIZED)

        # Set up orbit if configured
        if pc.orbit_center is not None:
            platform.set_orbit(
                center=pc.orbit_center,
                radius_m=pc.orbit_radius_m,
                altitude_m=scenario.sensor_position.to_meters().z,
                speed_ms=pc.velocity.speed_ms if pc.velocity else 50.0,
            )

        # Add waypoints if configured
        for wp_dict in pc.waypoints:
            wp = Waypoint(
                position=wp_dict["position"],
                time_s=wp_dict.get("time_s"),
                heading_deg=wp_dict.get("heading_deg"),
            )
            platform.trajectory.add_waypoint(wp)

        if verbose:
            print(f"  Platform: {platform_type.value}")
            print(f"  Gimbal mode: {pc.gimbal_mode}")

    # Calculate pixels per meter based on range and FOV
    # Use first target for reference
    geometry = scenario.get_geometry(0, 0)
    fov_deg = sensor_spec.optics_ir.fov_wide_deg
    fov_rad = np.deg2rad(fov_deg)
    scene_width_m = 2 * geometry.range_m * np.tan(fov_rad / 2)
    pixels_per_meter = scenario.resolution[1] / scene_width_m

    if verbose:
        print(f"  Range: {geometry.range_km:.1f} km")
        print(f"  Scene width: {scene_width_m:.0f} m")
        print(f"  Scale: {pixels_per_meter:.2f} px/m")

    # Create background
    temp_map, emis_map = _create_background_scene(
        scenario.resolution,
        scenario.background,
        scenario.environment,
        scenario.seed,
    )

    # Track target pixel locations
    target_pixels: Dict[str, Tuple[int, int]] = {}

    # Render targets
    h, w = scenario.resolution
    for i, track in enumerate(scenario.targets):
        geom = scenario.get_geometry(i, 0)

        # Calculate pixel position
        # For now, simple centering with offset for multiple targets
        offset_x = (i - (scenario.n_targets - 1) / 2) * w // (scenario.n_targets + 2)
        cx = w // 2 + int(offset_x)
        cy = h // 2

        actual_pos = _render_target_at_position(
            temp_map, emis_map,
            track,
            (cy, cx),
            pixels_per_meter,
            geom.aspect_angle_deg,
            geom.elevation_angle_deg,
        )
        target_pixels[track.name] = actual_pos

    # Run simulation
    scene = SceneInput(
        temperature_map=temp_map,
        emissivity_map=emis_map,
        background_temperature=scenario.environment.sky_temperature_k,
    )

    result = pipeline.run(scene, range_m=geometry.range_m)
    digital_image = result.digital_image

    # Apply platform motion effects if configured
    platform_state_history = None
    if platform is not None and apply_motion_effects:
        # Update platform state for a few time steps to build history
        dt = 1.0 / scenario.frame_rate_hz
        for _ in range(10):
            platform.update(dt)

        # Store platform state history
        platform_state_history = [{
            "time_s": state.time_s,
            "position": (state.position.x, state.position.y, state.position.z),
            "orientation": (
                state.orientation.roll_deg,
                state.orientation.pitch_deg,
                state.orientation.yaw_deg,
            ),
            "gimbal": (
                platform.gimbal.state.azimuth_deg,
                platform.gimbal.state.elevation_deg,
            ),
        } for state in platform._state_history[-5:]]

        # Apply motion blur and jitter
        # Use integration time from detector spec (convert ms to seconds)
        integration_time = sensor_spec.detector_ir.integration_time_ms / 1000.0
        digital_image = apply_platform_motion_effects(
            digital_image,
            platform,
            integration_time_s=integration_time,
        )

    # Calculate detection metrics if requested
    detection_metrics = None
    if scenario.compute_detection:
        detection_metrics = _compute_detection_metrics(
            digital_image,
            target_pixels,
            temp_map,
            scenario.environment,
        )

    # Build metadata
    metadata = {
        "scenario_name": scenario.name,
        "sensor_id": scenario.sensor_id,
        "sensor_name": sensor_spec.name,
        "n_targets": scenario.n_targets,
        "range_km": geometry.range_km,
        "aspect_deg": geometry.aspect_angle_deg,
        "background": scenario.background.value,
        "environment": {
            "ambient_temp_k": scenario.environment.ambient_temperature_k,
            "visibility_km": scenario.environment.visibility_km,
        },
    }

    # Add platform info to metadata
    if platform is not None:
        metadata["platform"] = {
            "type": platform.platform_type.value,
            "gimbal_mode": platform.gimbal.mode.value,
            "position": (
                platform.state.position.x,
                platform.state.position.y,
                platform.state.position.z,
            ),
            "gimbal_angles": (
                platform.gimbal.state.azimuth_deg,
                platform.gimbal.state.elevation_deg,
            ),
        }
        if platform_state_history:
            metadata["platform_history"] = platform_state_history

    return ScenarioOutput(
        digital_image=digital_image,
        temperature_map=temp_map,
        target_pixels=target_pixels,
        metadata=metadata,
        detection_metrics=detection_metrics,
    )


def _compute_detection_metrics(
    image: NDArray,
    target_pixels: Dict[str, Tuple[int, int]],
    temp_map: NDArray,
    environment: EnvironmentConditions,
) -> Dict[str, float]:
    """Compute basic detection metrics."""
    metrics = {}

    for name, (cy, cx) in target_pixels.items():
        h, w = image.shape

        # Extract target region (approximate)
        margin = 20
        y0, y1 = max(0, cy - margin), min(h, cy + margin)
        x0, x1 = max(0, cx - margin), min(w, cx + margin)

        if y1 <= y0 or x1 <= x0:
            continue

        target_region = image[y0:y1, x0:x1].astype(float)

        # Background region (ring around target)
        bg_margin = 40
        by0, by1 = max(0, cy - bg_margin), min(h, cy + bg_margin)
        bx0, bx1 = max(0, cx - bg_margin), min(w, cx + bg_margin)
        bg_region = image[by0:by1, bx0:bx1].astype(float)

        # Create mask to exclude target
        bg_mask = np.ones_like(bg_region, dtype=bool)
        ty0, ty1 = margin - (cy - y0), margin + (y1 - cy)
        tx0, tx1 = margin - (cx - x0), margin + (x1 - cx)
        if 0 <= ty0 < bg_mask.shape[0] and 0 <= tx0 < bg_mask.shape[1]:
            bg_mask[ty0:ty1, tx0:tx1] = False

        bg_values = bg_region[bg_mask]

        if len(bg_values) > 0:
            target_signal = np.mean(target_region)
            bg_mean = np.mean(bg_values)
            bg_std = np.std(bg_values)

            if bg_std > 0:
                snr = (target_signal - bg_mean) / bg_std
            else:
                snr = 0

            metrics[f"{name}_snr"] = float(snr)
            metrics[f"{name}_contrast"] = float(target_signal - bg_mean)
            metrics[f"{name}_signal"] = float(target_signal)
            metrics[f"{name}_background"] = float(bg_mean)

    return metrics


# =============================================================================
# Convenience Functions
# =============================================================================

def create_scenario(
    sensor: str,
    target: str,
    range_km: float = 10.0,
    aspect_deg: float = 90.0,
    altitude_m: float = 1000.0,
    environment: Union[EnvironmentType, str] = EnvironmentType.CLEAR_DAY,
    background: Union[BackgroundType, str] = BackgroundType.TERRAIN,
    name: Optional[str] = None,
    seed: Optional[int] = None,
) -> Scenario:
    """Quick scenario creation with common parameters.

    Args:
        sensor: Sensor ID from library
        target: Target object ID from library
        range_km: Range to target in kilometers
        aspect_deg: Target aspect angle
        altitude_m: Sensor altitude
        environment: Environment type
        background: Background type
        name: Scenario name
        seed: Random seed

    Returns:
        Configured Scenario
    """
    if isinstance(environment, str):
        environment = EnvironmentType(environment)
    if isinstance(background, str):
        background = BackgroundType(background)

    builder = ScenarioBuilder()
    builder.set_name(name or f"{sensor}_vs_{target}")
    builder.set_sensor(sensor, altitude_m=altitude_m)
    builder.add_target(target, range_km=range_km, aspect_deg=aspect_deg)
    builder.set_environment(env_type=environment)
    builder.set_background(background)
    builder.set_seed(seed)

    return builder.build()


def list_predefined_scenarios() -> List[str]:
    """List available pre-defined scenario templates."""
    return [
        "air_to_air_fighter",
        "air_to_ground_vehicle",
        "air_to_ground_convoy",
        "helicopter_patrol",
        "naval_surveillance",
        "ground_vehicle_recon",
        "border_patrol",
        "search_and_rescue",
        "industrial_inspection",
    ]


def get_predefined_scenario(
    name: str,
    seed: Optional[int] = None,
) -> Scenario:
    """Get a pre-defined scenario template.

    Args:
        name: Scenario template name
        seed: Random seed

    Returns:
        Configured Scenario
    """
    templates = {
        "air_to_air_fighter": lambda: (
            ScenarioBuilder()
            .set_name("Air-to-Air: Fighter Intercept")
            .set_sensor("mx20", altitude_m=10000)
            .add_target("f16", range_km=15, aspect_deg=30, heading_deg=270,
                       velocity_ms=(200, 0, 0))
            .set_environment(env_type=EnvironmentType.CLEAR_DAY)
            .set_background(BackgroundType.SKY)
            .set_seed(seed)
            .build()
        ),
        "air_to_ground_vehicle": lambda: (
            ScenarioBuilder()
            .set_name("Air-to-Ground: Vehicle Target")
            .set_sensor("sniper_atp", altitude_m=5000)
            .add_target("m1_abrams", range_km=8, aspect_deg=45)
            .set_environment(env_type=EnvironmentType.CLEAR_DAY)
            .set_background(BackgroundType.TERRAIN)
            .set_seed(seed)
            .build()
        ),
        "air_to_ground_convoy": lambda: (
            ScenarioBuilder()
            .set_name("Air-to-Ground: Convoy")
            .set_sensor("mx15", altitude_m=3000)
            .add_target("military_truck", position_km=(5, 0, 0))
            .add_target("military_truck", position_km=(5.1, 0.05, 0))
            .add_target("humvee", position_km=(5.2, 0, 0))
            .set_environment(env_type=EnvironmentType.DESERT_DAY)
            .set_background(BackgroundType.DESERT)
            .set_seed(seed)
            .build()
        ),
        "helicopter_patrol": lambda: (
            ScenarioBuilder()
            .set_name("Helicopter Patrol")
            .set_sensor("toplite_iii", altitude_m=500)
            .add_target("pickup_technical", range_km=2, aspect_deg=90)
            .set_environment(env_type=EnvironmentType.CLEAR_DAY)
            .set_background(BackgroundType.URBAN)
            .set_seed(seed)
            .build()
        ),
        "naval_surveillance": lambda: (
            ScenarioBuilder()
            .set_name("Naval Surveillance")
            .set_sensor("mx20", altitude_m=2000)
            .add_target("frigate", range_km=20, aspect_deg=270)
            .add_target("patrol_boat", range_km=18, aspect_deg=260)
            .set_environment(env_type=EnvironmentType.MARITIME)
            .set_background(BackgroundType.WATER)
            .set_seed(seed)
            .build()
        ),
        "ground_vehicle_recon": lambda: (
            ScenarioBuilder()
            .set_name("Ground Vehicle Reconnaissance")
            .set_sensor("catherine_xp", altitude_m=0)
            .add_target("t90", range_km=3, aspect_deg=45)
            .set_environment(env_type=EnvironmentType.CLEAR_NIGHT)
            .set_background(BackgroundType.TERRAIN)
            .set_seed(seed)
            .build()
        ),
        "border_patrol": lambda: (
            ScenarioBuilder()
            .set_name("Border Patrol")
            .set_sensor("star_safire_380hd", altitude_m=1000)
            .add_target("civilian_car", range_km=5, aspect_deg=90)
            .add_target("soldier_standing", range_km=4, aspect_deg=80)
            .set_environment(env_type=EnvironmentType.CLEAR_NIGHT)
            .set_background(BackgroundType.TERRAIN)
            .set_seed(seed)
            .build()
        ),
        "search_and_rescue": lambda: (
            ScenarioBuilder()
            .set_name("Search and Rescue")
            .set_sensor("ultra_8500", altitude_m=500)
            .add_target("civilian", range_km=1, aspect_deg=90)
            .set_environment(env_type=EnvironmentType.OVERCAST)
            .set_background(BackgroundType.FOREST)
            .set_seed(seed)
            .build()
        ),
        "industrial_inspection": lambda: (
            ScenarioBuilder()
            .set_name("Industrial Thermal Inspection")
            .set_sensor("sophie_mf", altitude_m=0)
            .add_target("civilian_car", range_km=0.1, aspect_deg=90,
                       name="hot_equipment")  # Placeholder
            .set_environment(env_type=EnvironmentType.URBAN)
            .set_background(BackgroundType.URBAN)
            .set_seed(seed)
            .build()
        ),
    }

    if name not in templates:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(templates.keys())}")

    return templates[name]()
