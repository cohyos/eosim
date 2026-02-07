"""
Gimbal dynamics simulation for electro-optical targeting systems.

This module provides physically-based gimbal modeling including:
- Two-axis (azimuth/elevation) gimbal kinematics
- Servo motor dynamics with rate limiting
- Line-of-sight stabilization
- Jitter and vibration modeling
- Track mode simulation
- Slew commands and rate control
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np


class GimbalMode(Enum):
    """Gimbal operating modes."""
    STOW = "stow"               # Stowed/parked position
    MANUAL = "manual"           # Manual rate control
    POSITION = "position"       # Position command mode
    TRACK = "track"             # Target tracking mode
    SCAN = "scan"               # Scan pattern mode
    STABILIZED = "stabilized"   # Platform stabilized
    SLAVE = "slave"             # Slaved to external source


class ScanPattern(Enum):
    """Standard scan patterns."""
    RASTER = "raster"           # Raster/TV scan
    SPIRAL = "spiral"           # Expanding spiral
    ROSETTE = "rosette"         # Rosette pattern
    SECTOR = "sector"           # Sector scan
    STEP_STARE = "step_stare"   # Step and stare
    RANDOM = "random"           # Random search


@dataclass
class GimbalLimits:
    """Gimbal mechanical limits."""
    # Azimuth limits (degrees)
    az_min_deg: float = -180.0
    az_max_deg: float = 180.0
    az_rate_max_deg_s: float = 60.0    # Max slew rate
    az_accel_max_deg_s2: float = 120.0  # Max acceleration

    # Elevation limits (degrees)
    el_min_deg: float = -30.0
    el_max_deg: float = 90.0
    el_rate_max_deg_s: float = 60.0
    el_accel_max_deg_s2: float = 120.0

    # Roll limits (for 3-axis gimbals)
    roll_min_deg: float = -10.0
    roll_max_deg: float = 10.0
    roll_rate_max_deg_s: float = 30.0

    # Gimbal lock zones
    elevation_gimbal_lock_deg: float = 85.0  # Near-zenith limit


@dataclass
class ServoParameters:
    """Servo system parameters."""
    # Control loop gains
    kp_position: float = 10.0    # Position proportional gain
    ki_position: float = 1.0     # Position integral gain
    kd_position: float = 5.0     # Position derivative gain

    kp_rate: float = 5.0         # Rate loop proportional gain
    ki_rate: float = 0.5         # Rate loop integral gain

    # Motor characteristics
    torque_constant: float = 1.0  # N·m/A
    motor_inertia: float = 0.01   # kg·m²
    load_inertia: float = 0.1     # kg·m²
    damping: float = 0.1          # N·m·s/rad
    friction_static: float = 0.5  # N·m static friction
    friction_coulomb: float = 0.3 # N·m Coulomb friction

    # Bandwidth
    bandwidth_hz: float = 10.0    # Closed-loop bandwidth
    natural_frequency_hz: float = 15.0
    damping_ratio: float = 0.7

    # Noise
    encoder_resolution_deg: float = 0.001  # Encoder resolution
    encoder_noise_deg: float = 0.0005      # Encoder noise RMS


@dataclass
class JitterParameters:
    """Line-of-sight jitter parameters."""
    # Base jitter (always present)
    base_jitter_urad: float = 50.0    # RMS jitter in microradians

    # Vibration-induced jitter
    vibration_freq_hz: float = 30.0    # Primary vibration frequency
    vibration_amplitude_urad: float = 100.0

    # Wind/aerodynamic jitter
    aero_jitter_urad: float = 20.0     # Wind buffet jitter

    # Temporal characteristics
    jitter_bandwidth_hz: float = 100.0  # Jitter spectrum cutoff


@dataclass
class GimbalState:
    """Current gimbal state."""
    # Position (degrees)
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    roll_deg: float = 0.0

    # Rates (degrees/second)
    azimuth_rate_deg_s: float = 0.0
    elevation_rate_deg_s: float = 0.0
    roll_rate_deg_s: float = 0.0

    # Accelerations (degrees/second²)
    azimuth_accel_deg_s2: float = 0.0
    elevation_accel_deg_s2: float = 0.0

    # Operating mode
    mode: GimbalMode = GimbalMode.MANUAL

    # Status flags
    at_limit_az: bool = False
    at_limit_el: bool = False
    near_gimbal_lock: bool = False
    tracking: bool = False

    # Internal servo state
    _integral_error_az: float = 0.0
    _integral_error_el: float = 0.0
    _last_error_az: float = 0.0
    _last_error_el: float = 0.0


@dataclass
class TrackState:
    """Target tracking state."""
    target_az_deg: float = 0.0
    target_el_deg: float = 0.0
    target_range_m: float = 1000.0

    # Track quality
    track_quality: float = 1.0    # 0-1, 1 = perfect track
    track_lock: bool = False
    frames_in_track: int = 0

    # Predicted position (for lead computation)
    target_az_rate_deg_s: float = 0.0
    target_el_rate_deg_s: float = 0.0

    # Track error
    error_az_deg: float = 0.0
    error_el_deg: float = 0.0


class GimbalController:
    """
    Gimbal servo controller simulation.

    Implements realistic servo dynamics including rate limiting,
    acceleration limiting, and closed-loop control.
    """

    def __init__(
        self,
        limits: Optional[GimbalLimits] = None,
        servo: Optional[ServoParameters] = None,
        jitter: Optional[JitterParameters] = None
    ):
        self.limits = limits or GimbalLimits()
        self.servo = servo or ServoParameters()
        self.jitter = jitter or JitterParameters()
        self.state = GimbalState()
        self.track_state = TrackState()

        # Commanded values
        self.cmd_az_deg = 0.0
        self.cmd_el_deg = 0.0
        self.cmd_az_rate = 0.0
        self.cmd_el_rate = 0.0

        # Scan pattern state
        self.scan_pattern = ScanPattern.RASTER
        self.scan_phase = 0.0
        self.scan_center_az = 0.0
        self.scan_center_el = 0.0
        self.scan_width_deg = 30.0
        self.scan_height_deg = 20.0
        self.scan_rate_deg_s = 10.0

        # Platform motion (for stabilization)
        self.platform_az_deg = 0.0
        self.platform_el_deg = 0.0
        self.platform_roll_deg = 0.0

    def set_mode(self, mode: GimbalMode):
        """Set gimbal operating mode."""
        self.state.mode = mode
        if mode == GimbalMode.STOW:
            # Command to stow position
            self.cmd_az_deg = 0.0
            self.cmd_el_deg = self.limits.el_min_deg

    def command_position(self, az_deg: float, el_deg: float):
        """Command gimbal to a position."""
        self.state.mode = GimbalMode.POSITION
        self.cmd_az_deg = np.clip(az_deg, self.limits.az_min_deg, self.limits.az_max_deg)
        self.cmd_el_deg = np.clip(el_deg, self.limits.el_min_deg, self.limits.el_max_deg)

    def command_rate(self, az_rate_deg_s: float, el_rate_deg_s: float):
        """Command gimbal rates (manual mode)."""
        self.state.mode = GimbalMode.MANUAL
        self.cmd_az_rate = np.clip(az_rate_deg_s,
                                   -self.limits.az_rate_max_deg_s,
                                   self.limits.az_rate_max_deg_s)
        self.cmd_el_rate = np.clip(el_rate_deg_s,
                                   -self.limits.el_rate_max_deg_s,
                                   self.limits.el_rate_max_deg_s)

    def set_track_target(
        self,
        az_deg: float,
        el_deg: float,
        range_m: float = 1000.0
    ):
        """Set target for tracking."""
        self.state.mode = GimbalMode.TRACK
        self.track_state.target_az_deg = az_deg
        self.track_state.target_el_deg = el_deg
        self.track_state.target_range_m = range_m

    def set_scan_pattern(
        self,
        pattern: ScanPattern,
        center_az: float,
        center_el: float,
        width_deg: float = 30.0,
        height_deg: float = 20.0,
        rate_deg_s: float = 10.0
    ):
        """Configure scan pattern."""
        self.state.mode = GimbalMode.SCAN
        self.scan_pattern = pattern
        self.scan_center_az = center_az
        self.scan_center_el = center_el
        self.scan_width_deg = width_deg
        self.scan_height_deg = height_deg
        self.scan_rate_deg_s = rate_deg_s
        self.scan_phase = 0.0

    def set_platform_attitude(
        self,
        az_deg: float,
        el_deg: float,
        roll_deg: float = 0.0
    ):
        """Update platform attitude for stabilization."""
        self.platform_az_deg = az_deg
        self.platform_el_deg = el_deg
        self.platform_roll_deg = roll_deg

    def update(self, dt: float) -> GimbalState:
        """
        Update gimbal state for one time step.

        Args:
            dt: Time step in seconds

        Returns:
            Updated gimbal state
        """
        if self.state.mode == GimbalMode.STOW:
            self._update_position_mode(dt)

        elif self.state.mode == GimbalMode.MANUAL:
            self._update_rate_mode(dt)

        elif self.state.mode == GimbalMode.POSITION:
            self._update_position_mode(dt)

        elif self.state.mode == GimbalMode.TRACK:
            self._update_track_mode(dt)

        elif self.state.mode == GimbalMode.SCAN:
            self._update_scan_mode(dt)

        elif self.state.mode == GimbalMode.STABILIZED:
            self._update_stabilized_mode(dt)

        # Apply limits
        self._apply_limits()

        # Check gimbal lock
        self.state.near_gimbal_lock = (
            self.state.elevation_deg > self.limits.elevation_gimbal_lock_deg
        )

        return self.state

    def _update_rate_mode(self, dt: float):
        """Update for rate command mode."""
        # Apply rate commands directly (with acceleration limiting)
        az_rate_error = self.cmd_az_rate - self.state.azimuth_rate_deg_s
        el_rate_error = self.cmd_el_rate - self.state.elevation_rate_deg_s

        # Limit acceleration
        max_az_accel = self.limits.az_accel_max_deg_s2 * dt
        max_el_accel = self.limits.el_accel_max_deg_s2 * dt

        az_rate_change = np.clip(az_rate_error, -max_az_accel, max_az_accel)
        el_rate_change = np.clip(el_rate_error, -max_el_accel, max_el_accel)

        self.state.azimuth_rate_deg_s += az_rate_change
        self.state.elevation_rate_deg_s += el_rate_change

        # Integrate position
        self.state.azimuth_deg += self.state.azimuth_rate_deg_s * dt
        self.state.elevation_deg += self.state.elevation_rate_deg_s * dt

    def _update_position_mode(self, dt: float):
        """Update for position command mode using PID control."""
        # Position errors
        error_az = self.cmd_az_deg - self.state.azimuth_deg
        error_el = self.cmd_el_deg - self.state.elevation_deg

        # Handle azimuth wraparound
        if error_az > 180:
            error_az -= 360
        elif error_az < -180:
            error_az += 360

        # PID control for azimuth
        p_az = self.servo.kp_position * error_az
        self.state._integral_error_az += error_az * dt
        i_az = self.servo.ki_position * self.state._integral_error_az
        d_az = self.servo.kd_position * (error_az - self.state._last_error_az) / dt
        self.state._last_error_az = error_az

        cmd_rate_az = p_az + i_az + d_az

        # PID control for elevation
        p_el = self.servo.kp_position * error_el
        self.state._integral_error_el += error_el * dt
        i_el = self.servo.ki_position * self.state._integral_error_el
        d_el = self.servo.kd_position * (error_el - self.state._last_error_el) / dt
        self.state._last_error_el = error_el

        cmd_rate_el = p_el + i_el + d_el

        # Limit rates
        cmd_rate_az = np.clip(cmd_rate_az,
                              -self.limits.az_rate_max_deg_s,
                              self.limits.az_rate_max_deg_s)
        cmd_rate_el = np.clip(cmd_rate_el,
                              -self.limits.el_rate_max_deg_s,
                              self.limits.el_rate_max_deg_s)

        # Apply rate with acceleration limiting
        max_az_accel = self.limits.az_accel_max_deg_s2 * dt
        max_el_accel = self.limits.el_accel_max_deg_s2 * dt

        rate_change_az = np.clip(cmd_rate_az - self.state.azimuth_rate_deg_s,
                                 -max_az_accel, max_az_accel)
        rate_change_el = np.clip(cmd_rate_el - self.state.elevation_rate_deg_s,
                                 -max_el_accel, max_el_accel)

        self.state.azimuth_rate_deg_s += rate_change_az
        self.state.elevation_rate_deg_s += rate_change_el

        # Integrate position
        self.state.azimuth_deg += self.state.azimuth_rate_deg_s * dt
        self.state.elevation_deg += self.state.elevation_rate_deg_s * dt

    def _update_track_mode(self, dt: float):
        """Update for target tracking mode."""
        # Update target rates (for prediction)
        new_error_az = self.track_state.target_az_deg - self.state.azimuth_deg
        new_error_el = self.track_state.target_el_deg - self.state.elevation_deg

        # Track quality affects response
        quality = self.track_state.track_quality

        # Command position with lead
        lead_time = 0.1  # Prediction horizon
        predicted_az = (self.track_state.target_az_deg +
                       self.track_state.target_az_rate_deg_s * lead_time)
        predicted_el = (self.track_state.target_el_deg +
                       self.track_state.target_el_rate_deg_s * lead_time)

        # Blend predicted with current based on track quality
        self.cmd_az_deg = (quality * predicted_az +
                          (1 - quality) * self.track_state.target_az_deg)
        self.cmd_el_deg = (quality * predicted_el +
                          (1 - quality) * self.track_state.target_el_deg)

        # Use position mode update
        self._update_position_mode(dt)

        # Update track error
        self.track_state.error_az_deg = new_error_az
        self.track_state.error_el_deg = new_error_el

        # Update track state
        error_mag = math.sqrt(new_error_az**2 + new_error_el**2)
        if error_mag < 1.0:  # Within 1 degree
            self.track_state.track_lock = True
            self.track_state.frames_in_track += 1
            self.state.tracking = True
        else:
            self.track_state.track_lock = False
            self.track_state.frames_in_track = 0
            self.state.tracking = False

    def _update_scan_mode(self, dt: float):
        """Update for scan pattern mode."""
        self.scan_phase += dt * self.scan_rate_deg_s / self.scan_width_deg

        if self.scan_pattern == ScanPattern.RASTER:
            # Raster scan
            line = int(self.scan_phase) % 10
            phase_in_line = (self.scan_phase % 1.0)

            if line % 2 == 0:
                az_offset = (phase_in_line - 0.5) * self.scan_width_deg
            else:
                az_offset = (0.5 - phase_in_line) * self.scan_width_deg

            el_offset = ((line / 10.0) - 0.5) * self.scan_height_deg

        elif self.scan_pattern == ScanPattern.SPIRAL:
            # Expanding spiral
            r = (self.scan_phase % 1.0) * self.scan_width_deg / 2
            theta = self.scan_phase * 2 * math.pi * 5
            az_offset = r * math.cos(theta)
            el_offset = r * math.sin(theta) * self.scan_height_deg / self.scan_width_deg

        elif self.scan_pattern == ScanPattern.ROSETTE:
            # Rosette pattern
            t = self.scan_phase * 2 * math.pi
            az_offset = self.scan_width_deg / 2 * math.sin(t) * math.cos(5 * t)
            el_offset = self.scan_height_deg / 2 * math.sin(t) * math.sin(5 * t)

        elif self.scan_pattern == ScanPattern.SECTOR:
            # Sector scan (horizontal sweep)
            phase = (self.scan_phase % 2.0)
            if phase < 1.0:
                az_offset = (phase - 0.5) * self.scan_width_deg
            else:
                az_offset = (1.5 - phase) * self.scan_width_deg
            el_offset = 0.0

        else:
            az_offset = 0.0
            el_offset = 0.0

        # Command scan position
        self.cmd_az_deg = self.scan_center_az + az_offset
        self.cmd_el_deg = self.scan_center_el + el_offset

        self._update_position_mode(dt)

    def _update_stabilized_mode(self, dt: float):
        """Update for platform stabilization mode."""
        # Compensate for platform motion
        # Keep inertial LOS constant
        inertial_az = self.state.azimuth_deg + self.platform_az_deg
        inertial_el = self.state.elevation_deg + self.platform_el_deg

        # Maintain current inertial pointing
        self.cmd_az_deg = inertial_az - self.platform_az_deg
        self.cmd_el_deg = inertial_el - self.platform_el_deg

        self._update_position_mode(dt)

    def _apply_limits(self):
        """Apply gimbal position and rate limits."""
        # Position limits
        self.state.at_limit_az = False
        self.state.at_limit_el = False

        if self.state.azimuth_deg < self.limits.az_min_deg:
            self.state.azimuth_deg = self.limits.az_min_deg
            self.state.azimuth_rate_deg_s = max(0, self.state.azimuth_rate_deg_s)
            self.state.at_limit_az = True
        elif self.state.azimuth_deg > self.limits.az_max_deg:
            self.state.azimuth_deg = self.limits.az_max_deg
            self.state.azimuth_rate_deg_s = min(0, self.state.azimuth_rate_deg_s)
            self.state.at_limit_az = True

        if self.state.elevation_deg < self.limits.el_min_deg:
            self.state.elevation_deg = self.limits.el_min_deg
            self.state.elevation_rate_deg_s = max(0, self.state.elevation_rate_deg_s)
            self.state.at_limit_el = True
        elif self.state.elevation_deg > self.limits.el_max_deg:
            self.state.elevation_deg = self.limits.el_max_deg
            self.state.elevation_rate_deg_s = min(0, self.state.elevation_rate_deg_s)
            self.state.at_limit_el = True

        # Rate limits
        self.state.azimuth_rate_deg_s = np.clip(
            self.state.azimuth_rate_deg_s,
            -self.limits.az_rate_max_deg_s,
            self.limits.az_rate_max_deg_s
        )
        self.state.elevation_rate_deg_s = np.clip(
            self.state.elevation_rate_deg_s,
            -self.limits.el_rate_max_deg_s,
            self.limits.el_rate_max_deg_s
        )

    def get_jitter(self) -> Tuple[float, float]:
        """
        Get current line-of-sight jitter.

        Returns:
            (az_jitter_deg, el_jitter_deg) jitter values
        """
        # Base jitter (Gaussian)
        base_az = np.random.randn() * self.jitter.base_jitter_urad * 1e-6 * 180 / math.pi
        base_el = np.random.randn() * self.jitter.base_jitter_urad * 1e-6 * 180 / math.pi

        # Vibration component
        t = self.scan_phase  # Use scan phase as time proxy
        vib_az = (math.sin(2 * math.pi * self.jitter.vibration_freq_hz * t) *
                 self.jitter.vibration_amplitude_urad * 1e-6 * 180 / math.pi)
        vib_el = (math.cos(2 * math.pi * self.jitter.vibration_freq_hz * t) *
                 self.jitter.vibration_amplitude_urad * 1e-6 * 180 / math.pi)

        # Aerodynamic jitter
        aero_az = np.random.randn() * self.jitter.aero_jitter_urad * 1e-6 * 180 / math.pi
        aero_el = np.random.randn() * self.jitter.aero_jitter_urad * 1e-6 * 180 / math.pi

        total_az = base_az + vib_az + aero_az
        total_el = base_el + vib_el + aero_el

        return (total_az, total_el)

    def get_line_of_sight(
        self,
        include_jitter: bool = True
    ) -> Tuple[float, float]:
        """
        Get current line of sight direction.

        Args:
            include_jitter: Whether to include jitter effects

        Returns:
            (azimuth_deg, elevation_deg) in body frame
        """
        az = self.state.azimuth_deg
        el = self.state.elevation_deg

        if include_jitter:
            jitter_az, jitter_el = self.get_jitter()
            az += jitter_az
            el += jitter_el

        return (az, el)

    def get_inertial_los(
        self,
        platform_heading_deg: float = 0.0,
        platform_pitch_deg: float = 0.0,
        platform_roll_deg: float = 0.0
    ) -> Tuple[float, float]:
        """
        Get line of sight in inertial (world) coordinates.

        Args:
            platform_heading_deg: Platform heading (yaw)
            platform_pitch_deg: Platform pitch
            platform_roll_deg: Platform roll

        Returns:
            (azimuth_deg, elevation_deg) in world frame
        """
        # Simplified transformation (proper would use rotation matrices)
        az_body, el_body = self.get_line_of_sight(include_jitter=True)

        # Add platform attitude
        az_world = az_body + platform_heading_deg
        el_world = el_body + platform_pitch_deg

        # Normalize azimuth
        while az_world > 180:
            az_world -= 360
        while az_world < -180:
            az_world += 360

        return (az_world, el_world)

    def get_status(self) -> Dict:
        """Get gimbal status summary."""
        return {
            'mode': self.state.mode.value,
            'azimuth_deg': self.state.azimuth_deg,
            'elevation_deg': self.state.elevation_deg,
            'az_rate_deg_s': self.state.azimuth_rate_deg_s,
            'el_rate_deg_s': self.state.elevation_rate_deg_s,
            'at_limit_az': self.state.at_limit_az,
            'at_limit_el': self.state.at_limit_el,
            'near_gimbal_lock': self.state.near_gimbal_lock,
            'tracking': self.state.tracking,
            'track_lock': self.track_state.track_lock if self.state.mode == GimbalMode.TRACK else False,
        }


# Pre-configured gimbal types
def create_flir_turret() -> GimbalController:
    """Create a typical FLIR turret gimbal."""
    limits = GimbalLimits(
        az_min_deg=-180,
        az_max_deg=180,
        az_rate_max_deg_s=60,
        az_accel_max_deg_s2=120,
        el_min_deg=-30,
        el_max_deg=90,
        el_rate_max_deg_s=60,
        el_accel_max_deg_s2=120
    )

    servo = ServoParameters(
        kp_position=15.0,
        ki_position=2.0,
        kd_position=8.0,
        bandwidth_hz=15.0
    )

    jitter = JitterParameters(
        base_jitter_urad=30,
        vibration_amplitude_urad=50
    )

    return GimbalController(limits, servo, jitter)


def create_targeting_pod() -> GimbalController:
    """Create an aircraft targeting pod gimbal."""
    limits = GimbalLimits(
        az_min_deg=-180,
        az_max_deg=180,
        az_rate_max_deg_s=120,
        az_accel_max_deg_s2=300,
        el_min_deg=-160,  # Can look backward/up
        el_max_deg=20,
        el_rate_max_deg_s=120,
        el_accel_max_deg_s2=300
    )

    servo = ServoParameters(
        kp_position=20.0,
        ki_position=3.0,
        kd_position=10.0,
        bandwidth_hz=25.0
    )

    jitter = JitterParameters(
        base_jitter_urad=20,
        vibration_amplitude_urad=100,  # Aircraft vibration
        aero_jitter_urad=50
    )

    return GimbalController(limits, servo, jitter)


def create_surveillance_gimbal() -> GimbalController:
    """Create a surveillance/ISR gimbal."""
    limits = GimbalLimits(
        az_min_deg=-180,
        az_max_deg=180,
        az_rate_max_deg_s=30,  # Slower for stable imagery
        az_accel_max_deg_s2=60,
        el_min_deg=-90,  # Nadir looking
        el_max_deg=30,
        el_rate_max_deg_s=30,
        el_accel_max_deg_s2=60
    )

    servo = ServoParameters(
        kp_position=10.0,
        ki_position=1.0,
        kd_position=5.0,
        bandwidth_hz=10.0
    )

    jitter = JitterParameters(
        base_jitter_urad=10,  # High stability
        vibration_amplitude_urad=20
    )

    return GimbalController(limits, servo, jitter)
