"""Tests for the Gimbal servo dynamics module."""

import numpy as np
import pytest

from eosim.gimbal import (
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


class TestServoParameters:
    """Tests for ServoParameters dataclass."""

    def test_default_parameters(self):
        params = ServoParameters()
        assert params.max_rate_deg_s == 60.0
        assert params.max_accel_deg_s2 == 200.0
        assert params.bandwidth_hz == 5.0

    def test_natural_frequency(self):
        params = ServoParameters(bandwidth_hz=10.0)
        assert params.natural_frequency_rad_s == pytest.approx(
            2 * np.pi * 10.0, rel=1e-6
        )

    def test_invalid_rate(self):
        with pytest.raises(ValueError):
            ServoParameters(max_rate_deg_s=0)

    def test_invalid_bandwidth(self):
        with pytest.raises(ValueError):
            ServoParameters(bandwidth_hz=-1.0)


class TestServoAxis:
    """Tests for single-axis servo controller."""

    def test_step_response(self):
        """Servo should settle to commanded position."""
        servo = ServoAxis()
        servo.set_command(10.0)

        # Simulate for 5 seconds at 100 Hz (allow full settling)
        for _ in range(500):
            servo.update(0.01)

        # Should have settled close to command
        assert abs(servo.position_deg - 10.0) < 1.5

    def test_rate_limiting(self):
        """Servo should respect max rate."""
        params = ServoParameters(max_rate_deg_s=10.0)
        servo = ServoAxis(params)
        servo.set_command(90.0)

        # Take one step
        servo.update(0.01)

        # Rate should not exceed limit
        assert abs(servo.rate_deg_s) <= params.max_rate_deg_s + 0.01

    def test_position_limiting(self):
        """Servo should respect position limits."""
        params = ServoParameters(position_limit_deg=45.0)
        servo = ServoAxis(params)
        servo.set_command(100.0)

        for _ in range(500):
            servo.update(0.01)

        assert abs(servo.position_deg) <= 45.0

    def test_reset(self):
        servo = ServoAxis()
        servo.set_command(30.0)
        for _ in range(100):
            servo.update(0.01)

        servo.reset(5.0)
        assert servo.position_deg == 5.0
        assert servo.rate_deg_s == 0.0

    def test_zero_dt_noop(self):
        servo = ServoAxis()
        servo.set_command(10.0)
        pos = servo.update(0.0)
        assert pos == 0.0  # Should not move


class TestGimbalController:
    """Tests for 2-axis gimbal controller."""

    def test_basic_positioning(self):
        gimbal = GimbalController()
        gimbal.command_position(20.0, 10.0)

        # Simulate settling for 5 seconds
        for _ in range(500):
            state = gimbal.update(0.01)

        assert abs(state.azimuth_deg - 20.0) < 1.5
        assert abs(state.elevation_deg - 10.0) < 1.5

    def test_state_tracking(self):
        gimbal = GimbalController()
        state = gimbal.state

        assert state.azimuth_deg == 0.0
        assert state.elevation_deg == 0.0
        assert state.time_s == 0.0

        gimbal.update(0.1)
        state = gimbal.state
        assert state.time_s == pytest.approx(0.1)

    def test_simulate(self):
        gimbal = GimbalController()
        gimbal.command_position(15.0, 5.0)
        states = gimbal.simulate(1.0, 0.01)

        assert len(states) == 100
        # Should be moving toward commanded position
        final = states[-1]
        assert abs(final.azimuth_deg) > 0  # Should have moved

    def test_errors(self):
        gimbal = GimbalController()
        gimbal.command_position(30.0, 15.0)

        assert gimbal.azimuth_error_deg == pytest.approx(30.0)
        assert gimbal.elevation_error_deg == pytest.approx(15.0)

    def test_los_with_jitter(self):
        """Jitter should add noise to LOS angles."""
        jitter = JitterModel(rms_jitter_urad=100.0)
        gimbal = GimbalController(jitter=jitter)

        # Sample multiple jitter-affected LOS readings
        readings = [gimbal.get_los_with_jitter() for _ in range(100)]
        az_vals = [r[0] for r in readings]

        # Should have some variance from jitter
        assert np.std(az_vals) > 0

    def test_reset(self):
        gimbal = GimbalController()
        gimbal.command_position(30.0, 15.0)
        gimbal.simulate(1.0, 0.01)

        gimbal.reset(5.0, -3.0)
        state = gimbal.state
        assert state.azimuth_deg == 5.0
        assert state.elevation_deg == -3.0
        assert state.time_s == 0.0


class TestJitterModel:
    """Tests for LOS jitter generation."""

    def test_white_jitter(self):
        jitter = JitterModel(rms_jitter_urad=50.0, psd_type="white")
        az, el = jitter.generate_jitter(1.0, 1000.0, rng=np.random.default_rng(42))

        assert len(az) == 1000
        assert len(el) == 1000
        # RMS should be roughly correct
        assert abs(np.std(az) - 50.0) < 20.0

    def test_pink_jitter(self):
        jitter = JitterModel(rms_jitter_urad=50.0, psd_type="pink")
        az, el = jitter.generate_jitter(1.0, 1000.0, rng=np.random.default_rng(42))

        assert len(az) == 1000
        # RMS should be rescaled to target
        assert abs(np.std(az) - 50.0) < 15.0

    def test_brownian_jitter(self):
        jitter = JitterModel(rms_jitter_urad=50.0, psd_type="brownian")
        az, el = jitter.generate_jitter(1.0, 1000.0, rng=np.random.default_rng(42))

        assert len(az) == 1000

    def test_invalid_psd_type(self):
        jitter = JitterModel(psd_type="invalid")
        with pytest.raises(ValueError, match="Unknown PSD type"):
            jitter.generate_jitter(1.0, 100.0)


class TestScanPatternGenerator:
    """Tests for scan pattern generation."""

    def setup_method(self):
        self.gen = ScanPatternGenerator()

    def test_raster_scan(self):
        az, el, t = self.gen.generate(
            ScanPattern.RASTER,
            duration_s=5.0,
            sample_rate_hz=100,
            fov_width_deg=10.0,
            fov_height_deg=8.0,
        )

        assert len(az) == 500
        assert len(el) == 500
        assert len(t) == 500
        # Azimuth should oscillate within FOV
        assert np.max(az) <= 6.0
        assert np.min(az) >= -6.0

    def test_spiral_scan(self):
        az, el, t = self.gen.generate(
            ScanPattern.SPIRAL,
            duration_s=5.0,
            sample_rate_hz=100,
            fov_width_deg=10.0,
        )

        assert len(az) == 500
        # Spiral should start near center
        assert abs(az[0]) < 1.0
        assert abs(el[0]) < 1.0

    def test_rosette_scan(self):
        az, el, t = self.gen.generate(
            ScanPattern.ROSETTE,
            duration_s=5.0,
            sample_rate_hz=100,
            n_petals=5,
        )

        assert len(az) == 500

    def test_sector_scan(self):
        az, el, t = self.gen.generate(
            ScanPattern.SECTOR,
            duration_s=5.0,
            sample_rate_hz=100,
        )

        assert len(az) == 500

    def test_stare(self):
        az, el, t = self.gen.generate(
            ScanPattern.STARE,
            duration_s=2.0,
            sample_rate_hz=100,
            center_az_deg=15.0,
            center_el_deg=5.0,
        )

        assert len(az) == 200
        np.testing.assert_allclose(az, 15.0)
        np.testing.assert_allclose(el, 5.0)

    def test_center_offset(self):
        az, el, t = self.gen.generate(
            ScanPattern.RASTER,
            duration_s=2.0,
            sample_rate_hz=100,
            center_az_deg=45.0,
            center_el_deg=20.0,
        )

        assert np.mean(az) == pytest.approx(45.0, abs=6.0)


class TestTargetTracker:
    """Tests for gimbal-level target tracking."""

    def test_track_target(self):
        gimbal = GimbalController()
        tracker = TargetTracker(gimbal)

        assert not tracker.is_tracking

        # Track a target for 5 seconds
        for _ in range(500):
            tracker.track(30.0, 15.0, 0.01)

        assert tracker.is_tracking
        assert tracker.tracking_error_deg < 5.0

    def test_designate(self):
        gimbal = GimbalController()
        tracker = TargetTracker(gimbal)

        tracker.designate(45.0, 10.0)
        assert tracker.is_tracking

    def test_break_track(self):
        gimbal = GimbalController()
        tracker = TargetTracker(gimbal)

        tracker.track(30.0, 15.0, 0.01)
        tracker.break_track()

        assert not tracker.is_tracking


class TestFactoryFunctions:
    """Tests for module factory functions."""

    def test_create_gimbal(self):
        gimbal = create_gimbal(
            max_rate_deg_s=90.0,
            jitter_urad=20.0,
        )
        assert isinstance(gimbal, GimbalController)

    def test_create_gimbal_no_jitter(self):
        gimbal = create_gimbal(jitter_urad=0.0)
        assert isinstance(gimbal, GimbalController)

    def test_create_flir_turret(self):
        gimbal = create_flir_turret()
        assert isinstance(gimbal, GimbalController)

        # FLIR should be responsive - simulate for 5 seconds
        gimbal.command_position(10.0, 5.0)
        gimbal.simulate(5.0, 0.01)
        state = gimbal.state
        assert abs(state.azimuth_deg - 10.0) < 5.0

    def test_create_surveillance_scanner(self):
        gimbal = create_surveillance_scanner()
        assert isinstance(gimbal, GimbalController)
