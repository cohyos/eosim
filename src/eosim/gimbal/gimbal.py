"""
Gimbal servo dynamics and scan pattern generation for EO sensor platforms.

Provides:
- 2-axis servo dynamics with PID control
- Rate and acceleration limits
- Scan patterns (raster, spiral, rosette, sector)
- Target tracking with LOS stabilization
- Jitter and disturbance modeling
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List

import numpy as np
from numpy.typing import NDArray


class ScanPattern(Enum):
    """Available scan patterns."""

    RASTER = "raster"
    SPIRAL = "spiral"
    ROSETTE = "rosette"
    SECTOR = "sector"
    STARE = "stare"


@dataclass
class ServoParameters:
    """Servo axis parameters.

    Attributes:
        max_rate_deg_s: Maximum slew rate (deg/s)
        max_accel_deg_s2: Maximum acceleration (deg/s^2)
        bandwidth_hz: Servo bandwidth (-3dB)
        damping_ratio: Servo damping ratio (zeta)
        position_limit_deg: Maximum position magnitude (deg)
        backlash_deg: Mechanical backlash (deg)
    """

    max_rate_deg_s: float = 60.0
    max_accel_deg_s2: float = 200.0
    bandwidth_hz: float = 5.0
    damping_ratio: float = 0.7
    position_limit_deg: float = 180.0
    backlash_deg: float = 0.01

    def __post_init__(self) -> None:
        if self.max_rate_deg_s <= 0:
            raise ValueError("max_rate_deg_s must be positive")
        if self.max_accel_deg_s2 <= 0:
            raise ValueError("max_accel_deg_s2 must be positive")
        if self.bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive")
        if self.damping_ratio <= 0:
            raise ValueError("damping_ratio must be positive")

    @property
    def natural_frequency_rad_s(self) -> float:
        """Natural frequency in rad/s from bandwidth and damping."""
        wn = 2 * np.pi * self.bandwidth_hz
        return wn


@dataclass
class PIDGains:
    """PID controller gains.

    Attributes:
        kp: Proportional gain
        ki: Integral gain
        kd: Derivative gain
        integral_limit: Anti-windup integral limit (deg)
    """

    kp: float = 15.0
    ki: float = 1.0
    kd: float = 5.0
    integral_limit: float = 10.0


@dataclass
class GimbalState:
    """Current gimbal state.

    Attributes:
        azimuth_deg: Azimuth position (deg)
        elevation_deg: Elevation position (deg)
        az_rate_deg_s: Azimuth rate (deg/s)
        el_rate_deg_s: Elevation rate (deg/s)
        time_s: Current simulation time (s)
    """

    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    az_rate_deg_s: float = 0.0
    el_rate_deg_s: float = 0.0
    time_s: float = 0.0


@dataclass
class JitterModel:
    """Line-of-sight jitter disturbance model.

    Attributes:
        rms_jitter_urad: RMS jitter amplitude (microradians)
        bandwidth_hz: Jitter bandwidth (Hz)
        psd_type: Power spectral density shape ('white', 'pink', 'brownian')
    """

    rms_jitter_urad: float = 10.0
    bandwidth_hz: float = 100.0
    psd_type: str = "white"

    def generate_jitter(
        self,
        duration_s: float,
        sample_rate_hz: float,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[NDArray, NDArray]:
        """Generate time-domain jitter sequence.

        Args:
            duration_s: Duration in seconds
            sample_rate_hz: Sample rate in Hz
            rng: Random number generator

        Returns:
            Tuple of (az_jitter_urad, el_jitter_urad) arrays
        """
        if rng is None:
            rng = np.random.default_rng()

        n_samples = int(duration_s * sample_rate_hz)
        sigma = self.rms_jitter_urad

        if self.psd_type == "white":
            az_jitter = rng.normal(0, sigma, n_samples)
            el_jitter = rng.normal(0, sigma, n_samples)

        elif self.psd_type == "pink":
            # 1/f noise via filtering white noise
            az_white = rng.normal(0, sigma, n_samples)
            el_white = rng.normal(0, sigma, n_samples)
            az_jitter = _pink_filter(az_white)
            el_jitter = _pink_filter(el_white)
            # Re-scale to match desired RMS
            if np.std(az_jitter) > 0:
                az_jitter *= sigma / np.std(az_jitter)
            if np.std(el_jitter) > 0:
                el_jitter *= sigma / np.std(el_jitter)

        elif self.psd_type == "brownian":
            # Random walk (integrated white noise)
            az_jitter = np.cumsum(rng.normal(0, sigma / np.sqrt(n_samples), n_samples))
            el_jitter = np.cumsum(rng.normal(0, sigma / np.sqrt(n_samples), n_samples))
            az_jitter -= np.mean(az_jitter)
            el_jitter -= np.mean(el_jitter)
            if np.std(az_jitter) > 0:
                az_jitter *= sigma / np.std(az_jitter)
            if np.std(el_jitter) > 0:
                el_jitter *= sigma / np.std(el_jitter)
        else:
            raise ValueError(f"Unknown PSD type: {self.psd_type}")

        return az_jitter, el_jitter


class ServoAxis:
    """Single-axis servo controller with PID and rate/acceleration limits.

    Models a second-order servo system with configurable dynamics,
    PID control, and physical constraints.
    """

    def __init__(
        self,
        params: Optional[ServoParameters] = None,
        gains: Optional[PIDGains] = None,
    ) -> None:
        """Initialize servo axis.

        Args:
            params: Servo physical parameters
            gains: PID controller gains
        """
        self.params = params or ServoParameters()
        self.gains = gains or PIDGains()

        # State
        self._position_deg: float = 0.0
        self._rate_deg_s: float = 0.0
        self._command_deg: float = 0.0

        # PID internals
        self._integral: float = 0.0
        self._prev_error: float = 0.0

    @property
    def position_deg(self) -> float:
        """Current position in degrees."""
        return self._position_deg

    @property
    def rate_deg_s(self) -> float:
        """Current rate in deg/s."""
        return self._rate_deg_s

    @property
    def position_error_deg(self) -> float:
        """Current position error in degrees."""
        return self._command_deg - self._position_deg

    def set_command(self, position_deg: float) -> None:
        """Set commanded position.

        Args:
            position_deg: Commanded position in degrees
        """
        self._command_deg = np.clip(
            position_deg,
            -self.params.position_limit_deg,
            self.params.position_limit_deg,
        )

    def update(self, dt: float) -> float:
        """Update servo state for one time step.

        Args:
            dt: Time step in seconds

        Returns:
            Current position in degrees
        """
        if dt <= 0:
            return self._position_deg

        # Position error
        error = self._command_deg - self._position_deg

        # PID control law
        self._integral += error * dt
        self._integral = np.clip(
            self._integral,
            -self.gains.integral_limit,
            self.gains.integral_limit,
        )

        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        # Control output (desired acceleration)
        accel = (
            self.gains.kp * error
            + self.gains.ki * self._integral
            + self.gains.kd * derivative
        )

        # Acceleration limit
        accel = np.clip(
            accel,
            -self.params.max_accel_deg_s2,
            self.params.max_accel_deg_s2,
        )

        # Update rate with acceleration
        new_rate = self._rate_deg_s + accel * dt

        # Rate limit
        new_rate = np.clip(
            new_rate,
            -self.params.max_rate_deg_s,
            self.params.max_rate_deg_s,
        )

        # Update position
        new_position = self._position_deg + new_rate * dt

        # Position limit
        new_position = np.clip(
            new_position,
            -self.params.position_limit_deg,
            self.params.position_limit_deg,
        )

        # If hit position limit, zero the rate
        if abs(new_position) >= self.params.position_limit_deg:
            new_rate = 0.0

        self._rate_deg_s = new_rate
        self._position_deg = new_position

        return self._position_deg

    def reset(self, position_deg: float = 0.0) -> None:
        """Reset servo to given position.

        Args:
            position_deg: Initial position
        """
        self._position_deg = position_deg
        self._rate_deg_s = 0.0
        self._command_deg = position_deg
        self._integral = 0.0
        self._prev_error = 0.0


class GimbalController:
    """Two-axis gimbal controller with servo dynamics.

    Provides azimuth/elevation pointing control with configurable
    servo dynamics, PID control, and scan pattern generation.

    Example:
        >>> from eosim.gimbal import GimbalController
        >>> gimbal = GimbalController()
        >>> gimbal.command_position(30.0, 15.0)  # Az, El
        >>> for i in range(100):
        ...     state = gimbal.update(0.01)
        ...     print(f"Az={state.azimuth_deg:.2f}, El={state.elevation_deg:.2f}")
    """

    def __init__(
        self,
        az_params: Optional[ServoParameters] = None,
        el_params: Optional[ServoParameters] = None,
        az_gains: Optional[PIDGains] = None,
        el_gains: Optional[PIDGains] = None,
        jitter: Optional[JitterModel] = None,
    ) -> None:
        """Initialize gimbal controller.

        Args:
            az_params: Azimuth servo parameters
            el_params: Elevation servo parameters
            az_gains: Azimuth PID gains
            el_gains: Elevation PID gains
            jitter: LOS jitter disturbance model
        """
        self._az_servo = ServoAxis(az_params, az_gains)
        self._el_servo = ServoAxis(el_params, el_gains)
        self._jitter = jitter
        self._time_s: float = 0.0
        self._jitter_rng = np.random.default_rng()

    @property
    def state(self) -> GimbalState:
        """Current gimbal state."""
        return GimbalState(
            azimuth_deg=self._az_servo.position_deg,
            elevation_deg=self._el_servo.position_deg,
            az_rate_deg_s=self._az_servo.rate_deg_s,
            el_rate_deg_s=self._el_servo.rate_deg_s,
            time_s=self._time_s,
        )

    @property
    def azimuth_error_deg(self) -> float:
        """Current azimuth pointing error."""
        return self._az_servo.position_error_deg

    @property
    def elevation_error_deg(self) -> float:
        """Current elevation pointing error."""
        return self._el_servo.position_error_deg

    def command_position(
        self,
        azimuth_deg: float,
        elevation_deg: float,
    ) -> None:
        """Command gimbal to position.

        Args:
            azimuth_deg: Commanded azimuth (deg)
            elevation_deg: Commanded elevation (deg)
        """
        self._az_servo.set_command(azimuth_deg)
        self._el_servo.set_command(elevation_deg)

    def update(self, dt: float) -> GimbalState:
        """Update gimbal dynamics for one time step.

        Args:
            dt: Time step in seconds

        Returns:
            Updated gimbal state
        """
        self._az_servo.update(dt)
        self._el_servo.update(dt)
        self._time_s += dt

        return self.state

    def simulate(
        self,
        duration_s: float,
        dt: float,
    ) -> List[GimbalState]:
        """Simulate gimbal over a time period.

        Args:
            duration_s: Simulation duration in seconds
            dt: Time step in seconds

        Returns:
            List of GimbalState at each time step
        """
        n_steps = int(duration_s / dt)
        states = []

        for _ in range(n_steps):
            state = self.update(dt)
            states.append(state)

        return states

    def get_los_with_jitter(self) -> Tuple[float, float]:
        """Get line-of-sight angles including jitter.

        Returns:
            (azimuth_deg, elevation_deg) with jitter applied
        """
        az = self._az_servo.position_deg
        el = self._el_servo.position_deg

        if self._jitter is not None:
            sigma_deg = self._jitter.rms_jitter_urad * 1e-6 * (180 / np.pi)
            az += self._jitter_rng.normal(0, sigma_deg)
            el += self._jitter_rng.normal(0, sigma_deg)

        return az, el

    def reset(
        self,
        azimuth_deg: float = 0.0,
        elevation_deg: float = 0.0,
    ) -> None:
        """Reset gimbal to position.

        Args:
            azimuth_deg: Initial azimuth
            elevation_deg: Initial elevation
        """
        self._az_servo.reset(azimuth_deg)
        self._el_servo.reset(elevation_deg)
        self._time_s = 0.0


class ScanPatternGenerator:
    """Generates gimbal scan patterns.

    Produces commanded azimuth/elevation angle sequences for various
    scan geometries.

    Example:
        >>> from eosim.gimbal import ScanPatternGenerator, ScanPattern
        >>> gen = ScanPatternGenerator()
        >>> az, el, t = gen.generate(
        ...     ScanPattern.RASTER,
        ...     duration_s=10.0,
        ...     sample_rate_hz=100,
        ...     fov_width_deg=10.0,
        ...     fov_height_deg=8.0,
        ... )
    """

    def generate(
        self,
        pattern: ScanPattern,
        duration_s: float,
        sample_rate_hz: float,
        fov_width_deg: float = 10.0,
        fov_height_deg: float = 8.0,
        center_az_deg: float = 0.0,
        center_el_deg: float = 0.0,
        scan_rate_hz: float = 1.0,
        n_petals: int = 5,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Generate scan pattern.

        Args:
            pattern: Scan pattern type
            duration_s: Pattern duration (s)
            sample_rate_hz: Output sample rate (Hz)
            fov_width_deg: Horizontal scan extent (deg)
            fov_height_deg: Vertical scan extent (deg)
            center_az_deg: Pattern center azimuth (deg)
            center_el_deg: Pattern center elevation (deg)
            scan_rate_hz: Scan frequency (Hz)
            n_petals: Number of petals for rosette pattern

        Returns:
            Tuple of (azimuth_deg, elevation_deg, time_s) arrays
        """
        n_samples = int(duration_s * sample_rate_hz)
        t = np.linspace(0, duration_s, n_samples)

        if pattern == ScanPattern.RASTER:
            az, el = self._raster(
                t, fov_width_deg, fov_height_deg, scan_rate_hz
            )
        elif pattern == ScanPattern.SPIRAL:
            az, el = self._spiral(
                t, fov_width_deg, fov_height_deg, scan_rate_hz
            )
        elif pattern == ScanPattern.ROSETTE:
            az, el = self._rosette(
                t, fov_width_deg, fov_height_deg, scan_rate_hz, n_petals
            )
        elif pattern == ScanPattern.SECTOR:
            az, el = self._sector(
                t, fov_width_deg, fov_height_deg, scan_rate_hz
            )
        elif pattern == ScanPattern.STARE:
            az = np.zeros(n_samples)
            el = np.zeros(n_samples)
        else:
            raise ValueError(f"Unknown scan pattern: {pattern}")

        az += center_az_deg
        el += center_el_deg

        return az, el, t

    def _raster(
        self,
        t: NDArray,
        width: float,
        height: float,
        freq: float,
    ) -> Tuple[NDArray, NDArray]:
        """Raster scan: horizontal lines with vertical step.

        Args:
            t: Time array
            width: Horizontal extent (deg)
            height: Vertical extent (deg)
            freq: Horizontal scan frequency (Hz)

        Returns:
            (azimuth, elevation) arrays
        """
        # Horizontal: triangle wave
        period = 1.0 / freq
        phase = (t % period) / period
        az = width * (2 * np.abs(2 * phase - 1) - 1) / 2

        # Vertical: slow linear progression
        duration = t[-1] - t[0] if len(t) > 1 else 1.0
        el = height * (t / duration - 0.5)

        return az, el

    def _spiral(
        self,
        t: NDArray,
        width: float,
        height: float,
        freq: float,
    ) -> Tuple[NDArray, NDArray]:
        """Spiral scan: expanding outward from center.

        Args:
            t: Time array
            width: Max horizontal extent (deg)
            height: Max vertical extent (deg)
            freq: Angular frequency (Hz)

        Returns:
            (azimuth, elevation) arrays
        """
        duration = t[-1] - t[0] if len(t) > 1 else 1.0
        theta = 2 * np.pi * freq * t
        # Radius grows linearly with time
        r = t / duration

        az = (width / 2) * r * np.cos(theta)
        el = (height / 2) * r * np.sin(theta)

        return az, el

    def _rosette(
        self,
        t: NDArray,
        width: float,
        height: float,
        freq: float,
        n_petals: int,
    ) -> Tuple[NDArray, NDArray]:
        """Rosette scan: flower-like pattern.

        Args:
            t: Time array
            width: Max horizontal extent (deg)
            height: Max vertical extent (deg)
            freq: Angular frequency (Hz)
            n_petals: Number of petals

        Returns:
            (azimuth, elevation) arrays
        """
        theta = 2 * np.pi * freq * t
        r = np.cos(n_petals * theta)

        az = (width / 2) * r * np.cos(theta)
        el = (height / 2) * r * np.sin(theta)

        return az, el

    def _sector(
        self,
        t: NDArray,
        width: float,
        height: float,
        freq: float,
    ) -> Tuple[NDArray, NDArray]:
        """Sector scan: back-and-forth in azimuth with fixed elevation.

        Args:
            t: Time array
            width: Azimuth extent (deg)
            height: Elevation extent (deg) - used for step between passes
            freq: Scan frequency (Hz)

        Returns:
            (azimuth, elevation) arrays
        """
        period = 1.0 / freq
        phase = (t % period) / period
        az = width * (2 * np.abs(2 * phase - 1) - 1) / 2

        # Step elevation after each full azimuth sweep
        n_steps = 8
        sweep_idx = (t * freq).astype(int) % n_steps
        el = height * (sweep_idx / max(n_steps - 1, 1) - 0.5)

        return az, el


class TargetTracker:
    """Gimbal-level target tracker.

    Generates gimbal commands to track a moving target based
    on measured LOS errors from a sensor tracker.

    Example:
        >>> from eosim.gimbal import TargetTracker, GimbalController
        >>> gimbal = GimbalController()
        >>> tracker = TargetTracker(gimbal)
        >>> tracker.track(target_az_deg=45.0, target_el_deg=10.0, dt=0.01)
    """

    def __init__(
        self,
        gimbal: GimbalController,
        track_bandwidth_hz: float = 2.0,
    ) -> None:
        """Initialize target tracker.

        Args:
            gimbal: Gimbal controller to drive
            track_bandwidth_hz: Tracking loop bandwidth (Hz)
        """
        self._gimbal = gimbal
        self._bandwidth_hz = track_bandwidth_hz
        self._tracking: bool = False
        self._target_az: float = 0.0
        self._target_el: float = 0.0

    @property
    def is_tracking(self) -> bool:
        """Whether tracker is actively tracking."""
        return self._tracking

    @property
    def tracking_error_deg(self) -> float:
        """RMS tracking error (degrees)."""
        az_err = self._gimbal.azimuth_error_deg
        el_err = self._gimbal.elevation_error_deg
        return np.sqrt(az_err**2 + el_err**2)

    def track(
        self,
        target_az_deg: float,
        target_el_deg: float,
        dt: float,
    ) -> GimbalState:
        """Update tracking to follow target.

        Args:
            target_az_deg: Target azimuth (deg)
            target_el_deg: Target elevation (deg)
            dt: Time step (s)

        Returns:
            Updated gimbal state
        """
        self._tracking = True
        self._target_az = target_az_deg
        self._target_el = target_el_deg

        self._gimbal.command_position(target_az_deg, target_el_deg)
        return self._gimbal.update(dt)

    def break_track(self) -> None:
        """Stop tracking."""
        self._tracking = False

    def designate(
        self,
        azimuth_deg: float,
        elevation_deg: float,
    ) -> None:
        """Designate a new track point.

        Args:
            azimuth_deg: Designated azimuth (deg)
            elevation_deg: Designated elevation (deg)
        """
        self._target_az = azimuth_deg
        self._target_el = elevation_deg
        self._gimbal.command_position(azimuth_deg, elevation_deg)
        self._tracking = True


def _pink_filter(white: NDArray) -> NDArray:
    """Apply 1/f filter to white noise to create pink noise.

    Uses simple IIR filtering approach.

    Args:
        white: White noise input

    Returns:
        Pink noise output
    """
    # Voss-McCartney algorithm approximation
    n = len(white)
    result = np.zeros(n)
    b = [0.02109238, 0.07113478, 0.68873558]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]

    # Simple 3rd-order IIR filter
    x_hist = [0.0, 0.0, 0.0]
    y_hist = [0.0, 0.0, 0.0]

    for i in range(n):
        x_hist = [white[i]] + x_hist[:2]
        y = (
            b[0] * x_hist[0]
            + b[1] * x_hist[1]
            + b[2] * x_hist[2]
            - a[1] * y_hist[0]
            - a[2] * y_hist[1]
            - a[3] * y_hist[2]
        )
        y_hist = [y] + y_hist[:2]
        result[i] = y

    return result


def create_gimbal(
    max_rate_deg_s: float = 60.0,
    max_accel_deg_s2: float = 200.0,
    bandwidth_hz: float = 5.0,
    jitter_urad: float = 0.0,
) -> GimbalController:
    """Factory function to create a configured gimbal.

    Args:
        max_rate_deg_s: Maximum slew rate (deg/s)
        max_accel_deg_s2: Maximum acceleration (deg/s^2)
        bandwidth_hz: Servo bandwidth (Hz)
        jitter_urad: RMS jitter (microradians), 0 to disable

    Returns:
        Configured GimbalController
    """
    params = ServoParameters(
        max_rate_deg_s=max_rate_deg_s,
        max_accel_deg_s2=max_accel_deg_s2,
        bandwidth_hz=bandwidth_hz,
    )

    jitter = None
    if jitter_urad > 0:
        jitter = JitterModel(rms_jitter_urad=jitter_urad)

    return GimbalController(
        az_params=params,
        el_params=params,
        jitter=jitter,
    )


def create_flir_turret() -> GimbalController:
    """Create a typical FLIR turret gimbal configuration.

    Returns:
        GimbalController configured for a FLIR turret
        (e.g., AN/AAQ-28 LITENING style)
    """
    az_params = ServoParameters(
        max_rate_deg_s=120.0,
        max_accel_deg_s2=400.0,
        bandwidth_hz=8.0,
        position_limit_deg=180.0,
    )
    el_params = ServoParameters(
        max_rate_deg_s=60.0,
        max_accel_deg_s2=200.0,
        bandwidth_hz=6.0,
        position_limit_deg=90.0,
    )
    jitter = JitterModel(rms_jitter_urad=5.0)

    return GimbalController(
        az_params=az_params,
        el_params=el_params,
        jitter=jitter,
    )


def create_surveillance_scanner() -> GimbalController:
    """Create a surveillance/search scanner gimbal.

    Returns:
        GimbalController configured for wide-area search
    """
    params = ServoParameters(
        max_rate_deg_s=30.0,
        max_accel_deg_s2=100.0,
        bandwidth_hz=3.0,
        position_limit_deg=180.0,
    )

    return GimbalController(
        az_params=params,
        el_params=params,
    )
