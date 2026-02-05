"""
EOSIM Sensor Platform Motion System.

Provides 6DOF (6 Degrees of Freedom) motion capabilities for sensor platforms:
- 3 translational DOFs: X, Y, Z position
- 3 rotational DOFs: Roll, Pitch, Yaw orientation

Includes:
- Platform dynamics (aircraft, helicopter, ground vehicle, naval, fixed)
- Gimbal/turret control with slew rates and limits
- Line-of-sight (LOS) stabilization
- Target tracking modes
- Trajectory and waypoint following
- Motion-induced effects (blur, jitter)

Example Usage:
--------------
>>> from eosim.library.platform import SensorPlatform, PlatformType, GimbalController
>>>
>>> # Create an aircraft platform
>>> platform = SensorPlatform(
...     platform_type=PlatformType.FIXED_WING,
...     initial_position=Position3D(0, 0, 5000, "m"),
...     initial_velocity=Velocity3D(100, 0, 0, "m/s"),
... )
>>>
>>> # Set gimbal to track a target
>>> platform.gimbal.track_point(Position3D(10000, 500, 0, "m"))
>>>
>>> # Update platform state
>>> platform.update(dt=0.033)  # 30 Hz update
>>> los = platform.get_line_of_sight()
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any, Callable, Union
import numpy as np
from numpy.typing import NDArray

from eosim.library.scenarios import Position3D, Velocity3D


class PlatformType(Enum):
    """Types of sensor platforms."""
    FIXED_WING = "fixed_wing"        # Aircraft (fighter, transport, UAV)
    ROTARY_WING = "rotary_wing"      # Helicopter
    GROUND_VEHICLE = "ground_vehicle" # Tank, truck, etc.
    NAVAL_SURFACE = "naval_surface"   # Ship
    NAVAL_SUBSURFACE = "naval_subsurface"  # Submarine periscope
    TRIPOD = "tripod"                 # Fixed ground-based
    HANDHELD = "handheld"             # Soldier-carried
    SATELLITE = "satellite"           # Space-based
    TETHERED = "tethered"             # Aerostat, tethered drone


class GimbalMode(Enum):
    """Gimbal operating modes."""
    STOWED = "stowed"                # Gimbal stowed/off
    RATE = "rate"                    # Manual rate control
    POSITION = "position"            # Point to fixed angles
    STABILIZED = "stabilized"        # Inertially stabilized
    TRACK_POINT = "track_point"      # Track geographic point
    TRACK_TARGET = "track_target"    # Track moving target
    SCAN = "scan"                    # Scanning pattern
    SEARCH = "search"                # Search pattern


class TrackingMode(Enum):
    """Target tracking modes."""
    CENTROID = "centroid"            # Track thermal centroid
    CORRELATION = "correlation"       # Image correlation tracking
    EDGE = "edge"                    # Edge tracking
    SCENE_LOCK = "scene_lock"        # Lock to scene features
    GPS = "gps"                      # Track GPS coordinates
    PREDICTED = "predicted"          # Predictive tracking


@dataclass
class Orientation3D:
    """3D orientation in Euler angles.

    Uses aerospace convention: roll-pitch-yaw (RPY)
    - Roll: Rotation about forward axis (positive = right wing down)
    - Pitch: Rotation about right axis (positive = nose up)
    - Yaw: Rotation about down axis (positive = nose right/clockwise from above)

    All angles in degrees.
    """
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    def to_radians(self) -> Tuple[float, float, float]:
        """Convert to radians."""
        return (
            np.deg2rad(self.roll_deg),
            np.deg2rad(self.pitch_deg),
            np.deg2rad(self.yaw_deg),
        )

    def to_rotation_matrix(self) -> NDArray:
        """Convert to 3x3 rotation matrix (body to world)."""
        r, p, y = self.to_radians()

        # Roll matrix (about X)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(r), -np.sin(r)],
            [0, np.sin(r), np.cos(r)],
        ])

        # Pitch matrix (about Y)
        Ry = np.array([
            [np.cos(p), 0, np.sin(p)],
            [0, 1, 0],
            [-np.sin(p), 0, np.cos(p)],
        ])

        # Yaw matrix (about Z)
        Rz = np.array([
            [np.cos(y), -np.sin(y), 0],
            [np.sin(y), np.cos(y), 0],
            [0, 0, 1],
        ])

        # Combined rotation: R = Rz @ Ry @ Rx
        return Rz @ Ry @ Rx

    @classmethod
    def from_rotation_matrix(cls, R: NDArray) -> "Orientation3D":
        """Create from rotation matrix."""
        # Extract Euler angles (may have gimbal lock issues near pitch = ±90°)
        pitch = np.arcsin(-R[2, 0])

        if np.abs(np.cos(pitch)) > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Gimbal lock
            roll = 0
            yaw = np.arctan2(-R[0, 1], R[1, 1])

        return cls(
            roll_deg=np.rad2deg(roll),
            pitch_deg=np.rad2deg(pitch),
            yaw_deg=np.rad2deg(yaw),
        )

    def __add__(self, other: "Orientation3D") -> "Orientation3D":
        """Add orientations (simple angle addition)."""
        return Orientation3D(
            roll_deg=self.roll_deg + other.roll_deg,
            pitch_deg=self.pitch_deg + other.pitch_deg,
            yaw_deg=self.yaw_deg + other.yaw_deg,
        )


@dataclass
class AngularVelocity3D:
    """Angular velocity in degrees per second."""
    roll_rate_dps: float = 0.0
    pitch_rate_dps: float = 0.0
    yaw_rate_dps: float = 0.0

    def to_radians_per_sec(self) -> Tuple[float, float, float]:
        """Convert to radians per second."""
        return (
            np.deg2rad(self.roll_rate_dps),
            np.deg2rad(self.pitch_rate_dps),
            np.deg2rad(self.yaw_rate_dps),
        )

    @property
    def magnitude_dps(self) -> float:
        """Total angular rate magnitude."""
        return np.sqrt(
            self.roll_rate_dps**2 +
            self.pitch_rate_dps**2 +
            self.yaw_rate_dps**2
        )


@dataclass
class PlatformState:
    """Complete state of a sensor platform at a point in time.

    Attributes:
        time_s: Time since scenario start
        position: 3D position in world frame
        velocity: 3D velocity in world frame
        acceleration: 3D acceleration in world frame
        orientation: Platform orientation (roll, pitch, yaw)
        angular_velocity: Platform angular rates
    """
    time_s: float = 0.0
    position: Position3D = field(default_factory=lambda: Position3D(0, 0, 0, "m"))
    velocity: Velocity3D = field(default_factory=lambda: Velocity3D(0, 0, 0, "m/s"))
    acceleration: Velocity3D = field(default_factory=lambda: Velocity3D(0, 0, 0, "m/s"))
    orientation: Orientation3D = field(default_factory=Orientation3D)
    angular_velocity: AngularVelocity3D = field(default_factory=AngularVelocity3D)


@dataclass
class GimbalState:
    """State of a gimbal/turret.

    Gimbal angles are relative to platform body frame.
    - Azimuth: Rotation in horizontal plane (0 = forward, positive = right)
    - Elevation: Rotation in vertical plane (0 = level, negative = look down)
    """
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    azimuth_rate_dps: float = 0.0
    elevation_rate_dps: float = 0.0

    def to_unit_vector(self) -> NDArray:
        """Convert gimbal angles to unit vector in body frame."""
        az_rad = np.deg2rad(self.azimuth_deg)
        el_rad = np.deg2rad(self.elevation_deg)

        # Line of sight unit vector
        return np.array([
            np.cos(el_rad) * np.cos(az_rad),  # Forward
            np.cos(el_rad) * np.sin(az_rad),  # Right
            -np.sin(el_rad),                   # Down
        ])


@dataclass
class GimbalLimits:
    """Physical limits of gimbal motion.

    Attributes:
        azimuth_min_deg: Minimum azimuth angle
        azimuth_max_deg: Maximum azimuth angle
        elevation_min_deg: Minimum elevation (most negative = look down)
        elevation_max_deg: Maximum elevation (positive = look up)
        azimuth_rate_max_dps: Maximum azimuth slew rate
        elevation_rate_max_dps: Maximum elevation slew rate
        azimuth_accel_max_dps2: Maximum azimuth acceleration
        elevation_accel_max_dps2: Maximum elevation acceleration
    """
    azimuth_min_deg: float = -180.0
    azimuth_max_deg: float = 180.0
    elevation_min_deg: float = -120.0
    elevation_max_deg: float = 30.0
    azimuth_rate_max_dps: float = 60.0
    elevation_rate_max_dps: float = 60.0
    azimuth_accel_max_dps2: float = 180.0
    elevation_accel_max_dps2: float = 180.0

    def clamp_position(self, az: float, el: float) -> Tuple[float, float]:
        """Clamp gimbal angles to limits."""
        az_clamped = np.clip(az, self.azimuth_min_deg, self.azimuth_max_deg)
        el_clamped = np.clip(el, self.elevation_min_deg, self.elevation_max_deg)
        return az_clamped, el_clamped

    def clamp_rates(self, az_rate: float, el_rate: float) -> Tuple[float, float]:
        """Clamp gimbal rates to limits."""
        az_rate_clamped = np.clip(az_rate, -self.azimuth_rate_max_dps, self.azimuth_rate_max_dps)
        el_rate_clamped = np.clip(el_rate, -self.elevation_rate_max_dps, self.elevation_rate_max_dps)
        return az_rate_clamped, el_rate_clamped


@dataclass
class StabilizationParams:
    """Line-of-sight stabilization parameters.

    Attributes:
        enabled: Whether stabilization is active
        bandwidth_hz: Stabilization loop bandwidth
        jitter_rejection_db: High-frequency jitter rejection
        residual_jitter_urad: Residual LOS jitter RMS
    """
    enabled: bool = True
    bandwidth_hz: float = 10.0
    jitter_rejection_db: float = 40.0
    residual_jitter_urad: float = 10.0


@dataclass
class TrackingParams:
    """Target tracking parameters.

    Attributes:
        mode: Tracking mode
        gate_size_pixels: Tracking gate size
        update_rate_hz: Track update rate
        coast_time_s: Time to coast before losing track
        acceleration_limit_g: Maximum target acceleration assumption
    """
    mode: TrackingMode = TrackingMode.CENTROID
    gate_size_pixels: int = 64
    update_rate_hz: float = 30.0
    coast_time_s: float = 2.0
    acceleration_limit_g: float = 9.0


class GimbalController:
    """Controller for gimbal/turret motion.

    Implements gimbal dynamics, pointing control, and stabilization.
    """

    def __init__(
        self,
        limits: Optional[GimbalLimits] = None,
        stabilization: Optional[StabilizationParams] = None,
        tracking: Optional[TrackingParams] = None,
    ):
        """Initialize gimbal controller.

        Args:
            limits: Gimbal physical limits
            stabilization: Stabilization parameters
            tracking: Tracking parameters
        """
        self.limits = limits or GimbalLimits()
        self.stabilization = stabilization or StabilizationParams()
        self.tracking = tracking or TrackingParams()

        # Current state
        self.state = GimbalState()
        self.mode = GimbalMode.STABILIZED

        # Commanded values
        self._cmd_azimuth_deg: float = 0.0
        self._cmd_elevation_deg: float = 0.0
        self._cmd_az_rate_dps: float = 0.0
        self._cmd_el_rate_dps: float = 0.0

        # Track point (in world coordinates)
        self._track_point: Optional[Position3D] = None
        self._track_velocity: Optional[Velocity3D] = None

        # Scan parameters
        self._scan_center_az: float = 0.0
        self._scan_center_el: float = 0.0
        self._scan_width_az: float = 30.0
        self._scan_width_el: float = 20.0
        self._scan_rate_dps: float = 10.0
        self._scan_phase: float = 0.0

    def set_mode(self, mode: GimbalMode) -> None:
        """Set gimbal operating mode."""
        self.mode = mode

        if mode == GimbalMode.STOWED:
            self._cmd_azimuth_deg = 0.0
            self._cmd_elevation_deg = 0.0

    def command_position(self, azimuth_deg: float, elevation_deg: float) -> None:
        """Command gimbal to specific position.

        Args:
            azimuth_deg: Commanded azimuth angle
            elevation_deg: Commanded elevation angle
        """
        self.mode = GimbalMode.POSITION
        az, el = self.limits.clamp_position(azimuth_deg, elevation_deg)
        self._cmd_azimuth_deg = az
        self._cmd_elevation_deg = el

    def command_rate(self, az_rate_dps: float, el_rate_dps: float) -> None:
        """Command gimbal slew rates.

        Args:
            az_rate_dps: Azimuth rate in degrees per second
            el_rate_dps: Elevation rate in degrees per second
        """
        self.mode = GimbalMode.RATE
        az_rate, el_rate = self.limits.clamp_rates(az_rate_dps, el_rate_dps)
        self._cmd_az_rate_dps = az_rate
        self._cmd_el_rate_dps = el_rate

    def track_point(
        self,
        point: Position3D,
        velocity: Optional[Velocity3D] = None,
    ) -> None:
        """Set gimbal to track a geographic point.

        Args:
            point: Point to track in world coordinates
            velocity: Optional velocity for moving targets
        """
        self.mode = GimbalMode.TRACK_POINT
        self._track_point = point
        self._track_velocity = velocity

    def set_scan_pattern(
        self,
        center_az: float = 0.0,
        center_el: float = -30.0,
        width_az: float = 60.0,
        width_el: float = 30.0,
        rate_dps: float = 15.0,
    ) -> None:
        """Configure scanning pattern.

        Args:
            center_az: Scan center azimuth
            center_el: Scan center elevation
            width_az: Azimuth scan width
            width_el: Elevation scan width
            rate_dps: Scan rate
        """
        self.mode = GimbalMode.SCAN
        self._scan_center_az = center_az
        self._scan_center_el = center_el
        self._scan_width_az = width_az
        self._scan_width_el = width_el
        self._scan_rate_dps = rate_dps
        self._scan_phase = 0.0

    def update(
        self,
        dt: float,
        platform_state: PlatformState,
        rng: Optional[np.random.Generator] = None,
    ) -> GimbalState:
        """Update gimbal state.

        Args:
            dt: Time step in seconds
            platform_state: Current platform state
            rng: Random number generator for jitter

        Returns:
            Updated gimbal state
        """
        if rng is None:
            rng = np.random.default_rng()

        # Calculate commanded angles based on mode
        if self.mode == GimbalMode.STOWED:
            target_az = 0.0
            target_el = 0.0

        elif self.mode == GimbalMode.RATE:
            target_az = self.state.azimuth_deg + self._cmd_az_rate_dps * dt
            target_el = self.state.elevation_deg + self._cmd_el_rate_dps * dt

        elif self.mode == GimbalMode.POSITION:
            target_az = self._cmd_azimuth_deg
            target_el = self._cmd_elevation_deg

        elif self.mode in (GimbalMode.TRACK_POINT, GimbalMode.TRACK_TARGET):
            target_az, target_el = self._compute_track_angles(platform_state)

        elif self.mode == GimbalMode.SCAN:
            target_az, target_el = self._compute_scan_angles(dt)

        elif self.mode == GimbalMode.STABILIZED:
            # Maintain current inertial pointing
            target_az = self.state.azimuth_deg
            target_el = self.state.elevation_deg

        else:
            target_az = self.state.azimuth_deg
            target_el = self.state.elevation_deg

        # Apply gimbal limits
        target_az, target_el = self.limits.clamp_position(target_az, target_el)

        # Compute required rates
        az_error = target_az - self.state.azimuth_deg
        el_error = target_el - self.state.elevation_deg

        # Handle azimuth wrap-around
        if az_error > 180:
            az_error -= 360
        elif az_error < -180:
            az_error += 360

        # Rate-limited slew
        max_az_change = self.limits.azimuth_rate_max_dps * dt
        max_el_change = self.limits.elevation_rate_max_dps * dt

        az_change = np.clip(az_error, -max_az_change, max_az_change)
        el_change = np.clip(el_error, -max_el_change, max_el_change)

        # Update state
        new_az = self.state.azimuth_deg + az_change
        new_el = self.state.elevation_deg + el_change

        # Normalize azimuth to [-180, 180]
        while new_az > 180:
            new_az -= 360
        while new_az < -180:
            new_az += 360

        # Apply limits again
        new_az, new_el = self.limits.clamp_position(new_az, new_el)

        # Calculate actual rates
        az_rate = az_change / dt if dt > 0 else 0
        el_rate = el_change / dt if dt > 0 else 0

        # Add jitter if stabilization is enabled
        if self.stabilization.enabled:
            jitter_std_deg = self.stabilization.residual_jitter_urad * 1e-6 * 180 / np.pi
            az_jitter = rng.normal(0, jitter_std_deg)
            el_jitter = rng.normal(0, jitter_std_deg)
            new_az += az_jitter
            new_el += el_jitter

        # Update state
        self.state = GimbalState(
            azimuth_deg=new_az,
            elevation_deg=new_el,
            azimuth_rate_dps=az_rate,
            elevation_rate_dps=el_rate,
        )

        return self.state

    def _compute_track_angles(self, platform_state: PlatformState) -> Tuple[float, float]:
        """Compute gimbal angles to track point."""
        if self._track_point is None:
            return self.state.azimuth_deg, self.state.elevation_deg

        # Get target position (possibly updated with velocity)
        target_pos = self._track_point.to_meters()
        if self._track_velocity is not None:
            vel = self._track_velocity.to_ms()
            target_pos = Position3D(
                target_pos.x + vel.vx * platform_state.time_s,
                target_pos.y + vel.vy * platform_state.time_s,
                target_pos.z + vel.vz * platform_state.time_s,
                "m"
            )

        # Vector from platform to target in world frame
        plat_pos = platform_state.position.to_meters()
        dx = target_pos.x - plat_pos.x
        dy = target_pos.y - plat_pos.y
        dz = target_pos.z - plat_pos.z

        # Transform to body frame
        R = platform_state.orientation.to_rotation_matrix()
        R_inv = R.T  # Transpose = inverse for rotation matrix
        body_vec = R_inv @ np.array([dx, dy, dz])

        # Convert to gimbal angles
        # Body frame: X = forward, Y = right, Z = down
        fwd, right, down = body_vec

        ground_dist = np.sqrt(fwd**2 + right**2)
        if ground_dist < 1e-6:
            azimuth = 0.0
        else:
            azimuth = np.rad2deg(np.arctan2(right, fwd))

        elevation = np.rad2deg(np.arctan2(-down, ground_dist))

        return azimuth, elevation

    def _compute_scan_angles(self, dt: float) -> Tuple[float, float]:
        """Compute gimbal angles for scan pattern."""
        # Update scan phase
        self._scan_phase += self._scan_rate_dps * dt / self._scan_width_az

        # Simple raster scan
        # Horizontal sweep with vertical steps
        az_offset = self._scan_width_az / 2 * np.sin(2 * np.pi * self._scan_phase)

        # Step elevation every half azimuth cycle
        n_el_steps = 5
        el_step = int(self._scan_phase * 2) % n_el_steps
        el_offset = -self._scan_width_el / 2 + (el_step / n_el_steps) * self._scan_width_el

        return self._scan_center_az + az_offset, self._scan_center_el + el_offset

    def get_line_of_sight_world(self, platform_state: PlatformState) -> NDArray:
        """Get line of sight unit vector in world coordinates.

        Args:
            platform_state: Current platform state

        Returns:
            Unit vector pointing along sensor LOS in world frame
        """
        # Get LOS in body frame
        los_body = self.state.to_unit_vector()

        # Transform to world frame
        R = platform_state.orientation.to_rotation_matrix()
        los_world = R @ los_body

        return los_world


@dataclass
class Waypoint:
    """A waypoint for trajectory planning.

    Attributes:
        position: 3D position
        velocity: Optional velocity at waypoint
        time_s: Optional time to reach waypoint
        heading_deg: Optional heading at waypoint
        name: Optional waypoint name
    """
    position: Position3D
    velocity: Optional[Velocity3D] = None
    time_s: Optional[float] = None
    heading_deg: Optional[float] = None
    name: str = ""


class TrajectoryGenerator:
    """Generates platform trajectories from waypoints or motion profiles."""

    def __init__(self, platform_type: PlatformType):
        """Initialize trajectory generator.

        Args:
            platform_type: Type of platform (affects dynamics)
        """
        self.platform_type = platform_type
        self.waypoints: List[Waypoint] = []
        self._current_waypoint_idx = 0

        # Platform dynamic limits based on type
        self.limits = self._get_default_limits()

    def _get_default_limits(self) -> Dict[str, float]:
        """Get default dynamic limits for platform type."""
        limits_by_type = {
            PlatformType.FIXED_WING: {
                "max_speed_ms": 300.0,
                "max_accel_g": 3.0,
                "max_climb_rate_ms": 50.0,
                "max_roll_rate_dps": 60.0,
                "max_pitch_rate_dps": 30.0,
                "max_yaw_rate_dps": 15.0,
                "max_bank_deg": 60.0,
            },
            PlatformType.ROTARY_WING: {
                "max_speed_ms": 80.0,
                "max_accel_g": 1.5,
                "max_climb_rate_ms": 15.0,
                "max_roll_rate_dps": 40.0,
                "max_pitch_rate_dps": 40.0,
                "max_yaw_rate_dps": 60.0,
                "max_bank_deg": 30.0,
            },
            PlatformType.GROUND_VEHICLE: {
                "max_speed_ms": 30.0,
                "max_accel_g": 0.5,
                "max_climb_rate_ms": 5.0,
                "max_roll_rate_dps": 10.0,
                "max_pitch_rate_dps": 10.0,
                "max_yaw_rate_dps": 30.0,
                "max_bank_deg": 15.0,
            },
            PlatformType.NAVAL_SURFACE: {
                "max_speed_ms": 15.0,
                "max_accel_g": 0.2,
                "max_climb_rate_ms": 0.0,
                "max_roll_rate_dps": 5.0,
                "max_pitch_rate_dps": 3.0,
                "max_yaw_rate_dps": 5.0,
                "max_bank_deg": 20.0,
            },
            PlatformType.TRIPOD: {
                "max_speed_ms": 0.0,
                "max_accel_g": 0.0,
                "max_climb_rate_ms": 0.0,
                "max_roll_rate_dps": 0.0,
                "max_pitch_rate_dps": 0.0,
                "max_yaw_rate_dps": 0.0,
                "max_bank_deg": 0.0,
            },
        }
        return limits_by_type.get(self.platform_type, limits_by_type[PlatformType.FIXED_WING])

    def add_waypoint(self, waypoint: Waypoint) -> None:
        """Add a waypoint to the trajectory."""
        self.waypoints.append(waypoint)

    def set_waypoints(self, waypoints: List[Waypoint]) -> None:
        """Set all waypoints."""
        self.waypoints = waypoints
        self._current_waypoint_idx = 0

    def clear_waypoints(self) -> None:
        """Clear all waypoints."""
        self.waypoints = []
        self._current_waypoint_idx = 0

    def get_state_at_time(self, t: float, initial_state: PlatformState) -> PlatformState:
        """Get interpolated platform state at given time.

        Args:
            t: Time in seconds
            initial_state: Initial platform state

        Returns:
            Interpolated platform state
        """
        if not self.waypoints:
            # No waypoints - maintain constant velocity
            vel = initial_state.velocity.to_ms()
            pos = initial_state.position.to_meters()

            new_pos = Position3D(
                pos.x + vel.vx * t,
                pos.y + vel.vy * t,
                pos.z + vel.vz * t,
                "m"
            )

            # Update heading based on velocity
            if vel.vx != 0 or vel.vy != 0:
                yaw = np.rad2deg(np.arctan2(vel.vx, vel.vy))
            else:
                yaw = initial_state.orientation.yaw_deg

            return PlatformState(
                time_s=t,
                position=new_pos,
                velocity=initial_state.velocity,
                acceleration=Velocity3D(0, 0, 0, "m/s"),
                orientation=Orientation3D(
                    roll_deg=initial_state.orientation.roll_deg,
                    pitch_deg=initial_state.orientation.pitch_deg,
                    yaw_deg=yaw,
                ),
                angular_velocity=AngularVelocity3D(),
            )

        # Find relevant waypoints for interpolation
        # Simple linear interpolation between waypoints
        total_waypoints = len(self.waypoints)

        # Assign times to waypoints if not specified
        waypoint_times = []
        for i, wp in enumerate(self.waypoints):
            if wp.time_s is not None:
                waypoint_times.append(wp.time_s)
            else:
                # Estimate time based on distance and speed
                if i == 0:
                    waypoint_times.append(0.0)
                else:
                    prev_pos = self.waypoints[i-1].position.to_meters()
                    curr_pos = wp.position.to_meters()
                    dist = np.sqrt(
                        (curr_pos.x - prev_pos.x)**2 +
                        (curr_pos.y - prev_pos.y)**2 +
                        (curr_pos.z - prev_pos.z)**2
                    )
                    speed = self.limits["max_speed_ms"] * 0.7  # Cruise at 70% max
                    dt = dist / max(speed, 1.0)
                    waypoint_times.append(waypoint_times[-1] + dt)

        # Find segment
        segment_idx = 0
        for i in range(len(waypoint_times) - 1):
            if t >= waypoint_times[i] and t < waypoint_times[i + 1]:
                segment_idx = i
                break
        else:
            segment_idx = max(0, len(waypoint_times) - 2)

        # Interpolate
        t0 = waypoint_times[segment_idx]
        t1 = waypoint_times[min(segment_idx + 1, total_waypoints - 1)]

        if t1 == t0:
            alpha = 0.0
        else:
            alpha = (t - t0) / (t1 - t0)
            alpha = np.clip(alpha, 0.0, 1.0)

        wp0 = self.waypoints[segment_idx]
        wp1 = self.waypoints[min(segment_idx + 1, total_waypoints - 1)]

        p0 = wp0.position.to_meters()
        p1 = wp1.position.to_meters()

        # Linear position interpolation
        new_pos = Position3D(
            p0.x + alpha * (p1.x - p0.x),
            p0.y + alpha * (p1.y - p0.y),
            p0.z + alpha * (p1.z - p0.z),
            "m"
        )

        # Velocity from position difference
        if t1 > t0:
            vx = (p1.x - p0.x) / (t1 - t0)
            vy = (p1.y - p0.y) / (t1 - t0)
            vz = (p1.z - p0.z) / (t1 - t0)
        else:
            vx, vy, vz = 0, 0, 0

        new_vel = Velocity3D(vx, vy, vz, "m/s")

        # Heading from velocity
        if vx != 0 or vy != 0:
            yaw = np.rad2deg(np.arctan2(vx, vy))
        else:
            yaw = wp0.heading_deg if wp0.heading_deg else initial_state.orientation.yaw_deg

        return PlatformState(
            time_s=t,
            position=new_pos,
            velocity=new_vel,
            acceleration=Velocity3D(0, 0, 0, "m/s"),
            orientation=Orientation3D(roll_deg=0, pitch_deg=0, yaw_deg=yaw),
            angular_velocity=AngularVelocity3D(),
        )


class SensorPlatform:
    """Complete 6DOF sensor platform with gimbal.

    Combines platform dynamics, gimbal control, and trajectory planning.
    """

    def __init__(
        self,
        platform_type: PlatformType = PlatformType.FIXED_WING,
        initial_position: Optional[Position3D] = None,
        initial_velocity: Optional[Velocity3D] = None,
        initial_orientation: Optional[Orientation3D] = None,
        gimbal_limits: Optional[GimbalLimits] = None,
        seed: Optional[int] = None,
    ):
        """Initialize sensor platform.

        Args:
            platform_type: Type of platform
            initial_position: Starting position
            initial_velocity: Starting velocity
            initial_orientation: Starting orientation
            gimbal_limits: Gimbal physical limits
            seed: Random seed for noise generation
        """
        self.platform_type = platform_type
        self.rng = np.random.default_rng(seed)

        # Initialize state
        self.state = PlatformState(
            time_s=0.0,
            position=initial_position or Position3D(0, 0, 1000, "m"),
            velocity=initial_velocity or Velocity3D(0, 0, 0, "m/s"),
            orientation=initial_orientation or Orientation3D(),
        )

        # Create gimbal controller
        self.gimbal = GimbalController(
            limits=gimbal_limits,
            stabilization=StabilizationParams(),
            tracking=TrackingParams(),
        )

        # Create trajectory generator
        self.trajectory = TrajectoryGenerator(platform_type)

        # Platform vibration/disturbance parameters
        self.vibration_params = self._get_default_vibration()

        # History for motion blur calculation
        self._state_history: List[PlatformState] = []
        self._max_history_length = 100

    def _get_default_vibration(self) -> Dict[str, float]:
        """Get default vibration parameters for platform type."""
        vibration_by_type = {
            PlatformType.FIXED_WING: {
                "linear_rms_m": 0.01,
                "angular_rms_deg": 0.1,
                "frequency_hz": 20.0,
            },
            PlatformType.ROTARY_WING: {
                "linear_rms_m": 0.05,
                "angular_rms_deg": 0.5,
                "frequency_hz": 15.0,
            },
            PlatformType.GROUND_VEHICLE: {
                "linear_rms_m": 0.1,
                "angular_rms_deg": 1.0,
                "frequency_hz": 5.0,
            },
            PlatformType.NAVAL_SURFACE: {
                "linear_rms_m": 0.2,
                "angular_rms_deg": 2.0,
                "frequency_hz": 0.5,
            },
            PlatformType.TRIPOD: {
                "linear_rms_m": 0.001,
                "angular_rms_deg": 0.01,
                "frequency_hz": 50.0,
            },
        }
        return vibration_by_type.get(self.platform_type, vibration_by_type[PlatformType.FIXED_WING])

    def update(self, dt: float) -> PlatformState:
        """Update platform and gimbal state.

        Args:
            dt: Time step in seconds

        Returns:
            Updated platform state
        """
        new_time = self.state.time_s + dt

        # Get trajectory state
        if self.trajectory.waypoints:
            self.state = self.trajectory.get_state_at_time(new_time, self.state)
        else:
            # Simple integration
            vel = self.state.velocity.to_ms()
            pos = self.state.position.to_meters()

            self.state = PlatformState(
                time_s=new_time,
                position=Position3D(
                    pos.x + vel.vx * dt,
                    pos.y + vel.vy * dt,
                    pos.z + vel.vz * dt,
                    "m"
                ),
                velocity=self.state.velocity,
                acceleration=self.state.acceleration,
                orientation=self.state.orientation,
                angular_velocity=self.state.angular_velocity,
            )

        # Add platform vibration
        self._add_vibration(dt)

        # Update gimbal
        self.gimbal.update(dt, self.state, self.rng)

        # Store history
        self._state_history.append(self.state)
        if len(self._state_history) > self._max_history_length:
            self._state_history.pop(0)

        return self.state

    def _add_vibration(self, dt: float) -> None:
        """Add platform vibration/disturbance to state."""
        vib = self.vibration_params

        # Random vibration (simplified - could use proper spectral model)
        pos = self.state.position.to_meters()
        orient = self.state.orientation

        # Linear vibration
        lin_noise = vib["linear_rms_m"] * self.rng.normal(0, 1, 3)

        # Angular vibration
        ang_noise = vib["angular_rms_deg"] * self.rng.normal(0, 1, 3)

        self.state = PlatformState(
            time_s=self.state.time_s,
            position=Position3D(
                pos.x + lin_noise[0],
                pos.y + lin_noise[1],
                pos.z + lin_noise[2],
                "m"
            ),
            velocity=self.state.velocity,
            acceleration=self.state.acceleration,
            orientation=Orientation3D(
                roll_deg=orient.roll_deg + ang_noise[0],
                pitch_deg=orient.pitch_deg + ang_noise[1],
                yaw_deg=orient.yaw_deg + ang_noise[2],
            ),
            angular_velocity=self.state.angular_velocity,
        )

    def get_line_of_sight(self) -> NDArray:
        """Get current line of sight unit vector in world frame."""
        return self.gimbal.get_line_of_sight_world(self.state)

    def get_look_point(self, range_m: float = 10000.0) -> Position3D:
        """Get the point the sensor is looking at.

        Args:
            range_m: Assumed range to look point

        Returns:
            Position of look point in world coordinates
        """
        los = self.get_line_of_sight()
        pos = self.state.position.to_meters()

        return Position3D(
            pos.x + los[0] * range_m,
            pos.y + los[1] * range_m,
            pos.z + los[2] * range_m,
            "m"
        )

    def point_at(self, target: Position3D) -> None:
        """Point sensor at a target position.

        Args:
            target: Target position to point at
        """
        self.gimbal.track_point(target)

    def set_orbit(
        self,
        center: Position3D,
        radius_m: float,
        altitude_m: float,
        speed_ms: float,
        clockwise: bool = True,
    ) -> None:
        """Set platform on an orbit around a point.

        Args:
            center: Center of orbit
            radius_m: Orbit radius
            altitude_m: Orbit altitude
            speed_ms: Orbit speed
            clockwise: Orbit direction
        """
        # Generate orbit waypoints
        n_waypoints = 36  # Every 10 degrees
        waypoints = []

        c = center.to_meters()
        direction = 1 if clockwise else -1
        circumference = 2 * np.pi * radius_m
        orbit_time = circumference / max(speed_ms, 1)

        for i in range(n_waypoints + 1):  # +1 to close the loop
            angle = direction * 2 * np.pi * i / n_waypoints
            x = c.x + radius_m * np.cos(angle)
            y = c.y + radius_m * np.sin(angle)
            z = altitude_m

            heading = np.rad2deg(angle + np.pi/2 * direction)  # Tangent to circle

            wp = Waypoint(
                position=Position3D(x, y, z, "m"),
                time_s=orbit_time * i / n_waypoints,
                heading_deg=heading,
                name=f"orbit_{i}",
            )
            waypoints.append(wp)

        self.trajectory.set_waypoints(waypoints)

        # Point gimbal at center
        self.gimbal.track_point(Position3D(c.x, c.y, 0, "m"))

    def get_motion_blur_vector(self, integration_time_s: float = 0.01) -> Tuple[float, float]:
        """Calculate motion blur vector during integration time.

        Args:
            integration_time_s: Sensor integration time

        Returns:
            Blur vector (dx_pixels, dy_pixels) - approximate
        """
        if len(self._state_history) < 2:
            return (0.0, 0.0)

        # Get LOS change over integration time
        current_los = self.get_line_of_sight()

        # Find state from integration_time ago
        history_dt = 0.0
        old_state = self.state
        for state in reversed(self._state_history[:-1]):
            history_dt = self.state.time_s - state.time_s
            if history_dt >= integration_time_s:
                old_state = state
                break

        if history_dt < 1e-6:
            return (0.0, 0.0)

        old_los = self.gimbal.get_line_of_sight_world(old_state)

        # Angular change
        los_change = current_los - old_los

        # Convert to approximate pixel motion (depends on FOV, simplified here)
        # Assume 1 degree FOV = ~100 pixels
        fov_factor = 100.0 / np.deg2rad(1.0)

        dx_pixels = los_change[0] * fov_factor
        dy_pixels = los_change[1] * fov_factor

        return (dx_pixels, dy_pixels)


# =============================================================================
# Convenience Functions
# =============================================================================

def apply_motion_blur(
    image: NDArray,
    blur_vector: Tuple[float, float],
    strength: float = 1.0,
) -> NDArray:
    """Apply directional motion blur to an image.

    Args:
        image: Input image (2D array)
        blur_vector: Blur direction and magnitude (dx_pixels, dy_pixels)
        strength: Blur strength multiplier (0-1)

    Returns:
        Blurred image
    """
    from scipy.ndimage import convolve

    dx, dy = blur_vector
    blur_length = np.sqrt(dx**2 + dy**2) * strength

    if blur_length < 0.5:
        return image

    # Create motion blur kernel
    kernel_size = max(3, int(blur_length * 2) | 1)  # Odd size
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)

    # Draw line in kernel direction
    center = kernel_size // 2
    for i in range(kernel_size):
        t = (i - center) / max(blur_length, 1)
        if abs(t) <= 1:
            px = int(center + t * dx * strength)
            py = int(center + t * dy * strength)
            if 0 <= px < kernel_size and 0 <= py < kernel_size:
                kernel[py, px] = 1.0

    # Normalize kernel
    if kernel.sum() > 0:
        kernel /= kernel.sum()
    else:
        kernel[center, center] = 1.0

    # Apply blur
    blurred = convolve(image.astype(np.float64), kernel, mode='nearest')

    return blurred.astype(image.dtype)


def apply_jitter(
    image: NDArray,
    jitter_std_pixels: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> NDArray:
    """Apply random spatial jitter to an image.

    Simulates LOS jitter by shifting the image slightly.

    Args:
        image: Input image (2D array)
        jitter_std_pixels: Standard deviation of jitter in pixels
        rng: Random number generator

    Returns:
        Jittered image
    """
    from scipy.ndimage import shift

    if rng is None:
        rng = np.random.default_rng()

    if jitter_std_pixels < 0.1:
        return image

    # Random shift
    dy = rng.normal(0, jitter_std_pixels)
    dx = rng.normal(0, jitter_std_pixels)

    # Apply shift
    shifted = shift(image.astype(np.float64), [dy, dx], mode='nearest')

    return shifted.astype(image.dtype)


def apply_platform_motion_effects(
    image: NDArray,
    platform: "SensorPlatform",
    integration_time_s: float = 0.01,
    apply_blur: bool = True,
    apply_jitter_effect: bool = True,
) -> NDArray:
    """Apply all platform motion effects to an image.

    Args:
        image: Input image
        platform: Sensor platform
        integration_time_s: Sensor integration time
        apply_blur: Whether to apply motion blur
        apply_jitter_effect: Whether to apply jitter

    Returns:
        Image with motion effects applied
    """
    result = image.copy()

    # Apply motion blur
    if apply_blur:
        blur_vec = platform.get_motion_blur_vector(integration_time_s)
        result = apply_motion_blur(result, blur_vec)

    # Apply jitter
    if apply_jitter_effect:
        jitter_std = platform.vibration_params["angular_rms_deg"] * 10  # deg to ~pixels
        result = apply_jitter(result, jitter_std, platform.rng)

    return result


def create_platform(
    platform_type: Union[PlatformType, str],
    position: Optional[Position3D] = None,
    velocity: Optional[Velocity3D] = None,
    altitude_m: Optional[float] = None,
    speed_ms: Optional[float] = None,
    heading_deg: float = 0.0,
    seed: Optional[int] = None,
) -> SensorPlatform:
    """Create a sensor platform with simplified parameters.

    Args:
        platform_type: Platform type
        position: Full position (optional)
        velocity: Full velocity (optional)
        altitude_m: Altitude (used if position not given)
        speed_ms: Speed (used if velocity not given)
        heading_deg: Initial heading
        seed: Random seed

    Returns:
        Configured SensorPlatform
    """
    if isinstance(platform_type, str):
        platform_type = PlatformType(platform_type)

    # Build position
    if position is None:
        alt = altitude_m or 1000.0
        position = Position3D(0, 0, alt, "m")

    # Build velocity from speed and heading
    if velocity is None:
        spd = speed_ms or 0.0
        heading_rad = np.deg2rad(heading_deg)
        velocity = Velocity3D(
            spd * np.sin(heading_rad),
            spd * np.cos(heading_rad),
            0,
            "m/s"
        )

    # Build orientation
    orientation = Orientation3D(yaw_deg=heading_deg)

    return SensorPlatform(
        platform_type=platform_type,
        initial_position=position,
        initial_velocity=velocity,
        initial_orientation=orientation,
        seed=seed,
    )
