"""
EOSIM Gimbal Servo Dynamics Module.

Provides gimbal servo simulation including 2-axis PID-controlled
servos, scan pattern generation, target tracking, and LOS jitter.

Example 1: Basic gimbal positioning
    >>> from eosim.gimbal import GimbalController
    >>> gimbal = GimbalController()
    >>> gimbal.command_position(30.0, 15.0)
    >>> for _ in range(100):
    ...     state = gimbal.update(0.01)
    >>> print(f"Az={state.azimuth_deg:.1f}, El={state.elevation_deg:.1f}")

Example 2: Generate raster scan pattern
    >>> from eosim.gimbal import ScanPatternGenerator, ScanPattern
    >>> gen = ScanPatternGenerator()
    >>> az, el, t = gen.generate(
    ...     ScanPattern.RASTER,
    ...     duration_s=10.0,
    ...     sample_rate_hz=100,
    ...     fov_width_deg=10.0,
    ...     fov_height_deg=8.0,
    ... )

Example 3: Track a target
    >>> from eosim.gimbal import GimbalController, TargetTracker
    >>> gimbal = GimbalController()
    >>> tracker = TargetTracker(gimbal)
    >>> state = tracker.track(target_az_deg=45.0, target_el_deg=10.0, dt=0.01)

Example 4: Create pre-configured FLIR turret
    >>> from eosim.gimbal import create_flir_turret
    >>> gimbal = create_flir_turret()
"""

from eosim.gimbal.gimbal import (
    ScanPattern,
    ServoParameters,
    PIDGains,
    GimbalState,
    JitterModel,
    ServoAxis,
    GimbalController,
    ScanPatternGenerator,
    TargetTracker,
    create_gimbal,
    create_flir_turret,
    create_surveillance_scanner,
)

__all__ = [
    "ScanPattern",
    "ServoParameters",
    "PIDGains",
    "GimbalState",
    "JitterModel",
    "ServoAxis",
    "GimbalController",
    "ScanPatternGenerator",
    "TargetTracker",
    "create_gimbal",
    "create_flir_turret",
    "create_surveillance_scanner",
]
