"""
Tests for EOSIM Library module.

Tests the object library, sensor library, and scenario system.
"""

import pytest
import numpy as np


class TestObjectLibrary:
    """Tests for object library functionality."""

    def test_list_objects(self):
        """Test listing all objects."""
        from eosim.library import list_objects
        objects = list_objects()
        assert len(objects) > 40  # Should have many objects
        assert "f16" in objects
        assert "m1_abrams" in objects
        assert "tomahawk" in objects

    def test_list_by_category(self):
        """Test listing objects by category."""
        from eosim.library import list_objects, ObjectCategory

        aircraft = list_objects(ObjectCategory.AIRCRAFT)
        assert "f16" in aircraft
        assert "f35" in aircraft
        assert "f15" in aircraft

        ships = list_objects(ObjectCategory.SHIP)
        assert "frigate" in ships
        assert "nimitz" in ships

        missiles = list_objects(ObjectCategory.MISSILE)
        assert "aim120" in missiles
        assert "tomahawk" in missiles

    def test_get_object(self):
        """Test getting a specific object."""
        from eosim.library import get_object

        f16 = get_object("f16")
        assert f16.name == "F-16 Fighting Falcon"
        assert f16.dimensions.length_m == pytest.approx(15.06, rel=0.01)
        assert f16.country == "USA"

    def test_object_signature(self):
        """Test generating object thermal signature."""
        from eosim.library import get_object

        f16 = get_object("f16")
        temp_map, emis_map = f16.get_signature(
            resolution=(64, 64),
            aspect_angle_deg=90.0,
        )

        assert temp_map.shape == (64, 64)
        assert emis_map.shape == (64, 64)
        assert np.any(temp_map > 0)  # Should have non-zero temperatures

    def test_list_categories(self):
        """Test listing categories."""
        from eosim.library import list_categories

        categories = list_categories()
        assert "aircraft" in categories
        assert "ship" in categories
        assert "missile" in categories
        assert "vehicle" in categories


class TestSensorLibrary:
    """Tests for sensor library functionality."""

    def test_list_sensors(self):
        """Test listing all sensors."""
        from eosim.library import list_sensors
        sensors = list_sensors()
        assert len(sensors) >= 15
        assert "mx15" in sensors
        assert "mx20" in sensors
        assert "sniper_atp" in sensors

    def test_get_sensor(self):
        """Test getting a specific sensor."""
        from eosim.library import get_sensor

        mx15 = get_sensor("mx15")
        assert mx15.name == "MX-15"
        assert mx15.manufacturer == "L3Harris"
        assert mx15.detector_ir.width_pixels == 1280
        assert mx15.detector_ir.height_pixels == 1024

    def test_sensor_performance(self):
        """Test sensor performance specs."""
        from eosim.library import get_sensor

        mx20 = get_sensor("mx20")
        assert mx20.performance.detection_range_vehicle_km > 20
        assert mx20.has_laser_designator is True

    def test_sensor_info(self):
        """Test getting sensor info."""
        from eosim.library import SensorLibrary

        info = SensorLibrary.get_info("mx15")
        assert info["name"] == "MX-15"
        assert info["manufacturer"] == "L3Harris"
        assert "detector_ir" in info
        assert "performance" in info


class TestScenarioSystem:
    """Tests for scenario system functionality."""

    def test_create_scenario(self):
        """Test creating a basic scenario."""
        from eosim.library import create_scenario

        scenario = create_scenario(
            sensor="mx15",
            target="f16",
            range_km=10.0,
            aspect_deg=90.0,
        )

        assert scenario.name == "mx15_vs_f16"
        assert scenario.sensor_id == "mx15"
        assert len(scenario.targets) == 1
        assert scenario.targets[0].object_id == "f16"

    def test_scenario_geometry(self):
        """Test scenario geometry calculations."""
        from eosim.library import create_scenario

        scenario = create_scenario(
            sensor="mx15",
            target="f16",
            range_km=10.0,
            aspect_deg=90.0,
        )

        geom = scenario.get_geometry()
        assert geom.range_km == pytest.approx(10.0, rel=0.01)

    def test_scenario_builder(self):
        """Test scenario builder."""
        from eosim.library import ScenarioBuilder, EnvironmentType, BackgroundType

        scenario = (
            ScenarioBuilder()
            .set_name("Test Scenario")
            .set_sensor("mx20", altitude_m=5000)
            .add_target("m1_abrams", position_km=(8.0, 0.0, 0.0))
            .add_target("t90", position_km=(9.0, 0.0, 0.0))
            .set_environment(env_type=EnvironmentType.CLEAR_DAY)
            .set_background(BackgroundType.TERRAIN)
            .build()
        )

        assert scenario.name == "Test Scenario"
        assert len(scenario.targets) == 2
        assert scenario.sensor_id == "mx20"

    def test_predefined_scenarios(self):
        """Test predefined scenario templates."""
        from eosim.library import list_predefined_scenarios, get_predefined_scenario

        templates = list_predefined_scenarios()
        assert len(templates) >= 5
        assert "air_to_air_fighter" in templates

        scenario = get_predefined_scenario("air_to_air_fighter")
        assert scenario.sensor_id == "mx20"

    def test_run_scenario(self):
        """Test running a scenario (integration test)."""
        from eosim.library import create_scenario, run_scenario

        scenario = create_scenario(
            sensor="generic_mwir_hd",
            target="civilian_car",
            range_km=2.0,
            seed=42,
        )

        result = run_scenario(scenario)

        assert result.digital_image.shape == (480, 640)
        assert result.temperature_map.shape == (480, 640)
        assert "civilian_car" in result.target_pixels
        assert result.detection_metrics is not None


class TestEngagementGeometry:
    """Tests for engagement geometry calculations."""

    def test_position_calculations(self):
        """Test position calculations."""
        from eosim.library import Position3D

        pos = Position3D(1000, 2000, 500, "m")
        assert pos.range_m == pytest.approx(np.sqrt(1000**2 + 2000**2 + 500**2))

        pos_km = pos.to_km()
        assert pos_km.x == pytest.approx(1.0)
        assert pos_km.y == pytest.approx(2.0)
        assert pos_km.z == pytest.approx(0.5)

    def test_geometry_from_positions(self):
        """Test creating geometry from positions."""
        from eosim.library import Position3D, EngagementGeometry

        sensor_pos = Position3D(0, 0, 1000, "m")
        target_pos = Position3D(5000, 0, 0, "m")

        geom = EngagementGeometry.from_positions(
            sensor_pos, target_pos, target_heading_deg=0
        )

        assert geom.range_m == pytest.approx(np.sqrt(5000**2 + 1000**2), rel=0.01)


class TestEnvironmentConditions:
    """Tests for environment conditions."""

    def test_environment_presets(self):
        """Test environment presets."""
        from eosim.library import EnvironmentConditions, EnvironmentType

        clear_day = EnvironmentConditions.from_type(EnvironmentType.CLEAR_DAY)
        assert clear_day.visibility_km == 23.0
        assert clear_day.ambient_temperature_k > 290

        fog = EnvironmentConditions.from_type(EnvironmentType.FOG)
        assert fog.visibility_km == 1.0
        assert fog.humidity_percent > 90


class TestPlatformSystem:
    """Tests for 6DOF sensor platform system."""

    def test_create_platform(self):
        """Test creating a sensor platform."""
        from eosim.library import create_platform, PlatformType, Position3D

        platform = create_platform(
            platform_type="fixed_wing",
            altitude_m=5000,
            speed_ms=100,
            heading_deg=45,
        )

        assert platform.platform_type == PlatformType.FIXED_WING
        assert platform.state.position.z == pytest.approx(5000, rel=0.01)
        assert platform.state.velocity.speed_ms == pytest.approx(100, rel=0.01)

    def test_platform_state_update(self):
        """Test platform state update."""
        from eosim.library import create_platform

        platform = create_platform(
            platform_type="fixed_wing",
            altitude_m=1000,
            speed_ms=50,
            heading_deg=0,  # North
        )

        initial_y = platform.state.position.y

        # Update platform for 1 second
        for _ in range(30):
            platform.update(dt=1/30)

        # Should have moved roughly 50m north (with some vibration noise)
        assert platform.state.position.y > initial_y + 40

    def test_gimbal_controller(self):
        """Test gimbal controller functionality."""
        from eosim.library import GimbalController, GimbalLimits, GimbalMode, PlatformState

        gimbal = GimbalController(
            limits=GimbalLimits(
                azimuth_min_deg=-180,
                azimuth_max_deg=180,
                elevation_min_deg=-90,
                elevation_max_deg=30,
            )
        )

        # Test position command
        gimbal.command_position(azimuth_deg=45, elevation_deg=-30)
        assert gimbal.mode == GimbalMode.POSITION

        # Update gimbal
        platform_state = PlatformState()
        for _ in range(100):
            gimbal.update(dt=0.033, platform_state=platform_state)

        # Should be close to commanded position
        assert gimbal.state.azimuth_deg == pytest.approx(45, abs=1)
        assert gimbal.state.elevation_deg == pytest.approx(-30, abs=1)

    def test_gimbal_limits(self):
        """Test gimbal limit enforcement."""
        from eosim.library import GimbalLimits

        limits = GimbalLimits(
            azimuth_min_deg=-90,
            azimuth_max_deg=90,
            elevation_min_deg=-60,
            elevation_max_deg=10,
        )

        # Test clamping
        az, el = limits.clamp_position(180, -90)
        assert az == 90  # Clamped to max
        assert el == -60  # Clamped to min

    def test_trajectory_generator(self):
        """Test trajectory waypoint system."""
        from eosim.library import (
            TrajectoryGenerator, Waypoint, PlatformType, PlatformState,
            Position3D, Velocity3D,
        )

        traj = TrajectoryGenerator(PlatformType.FIXED_WING)

        # Add waypoints
        traj.add_waypoint(Waypoint(
            position=Position3D(0, 0, 1000, "m"),
            time_s=0.0,
        ))
        traj.add_waypoint(Waypoint(
            position=Position3D(1000, 0, 1000, "m"),
            time_s=10.0,
        ))

        initial_state = PlatformState(
            position=Position3D(0, 0, 1000, "m"),
            velocity=Velocity3D(100, 0, 0, "m/s"),
        )

        # Get state at halfway
        state = traj.get_state_at_time(5.0, initial_state)
        assert state.position.x == pytest.approx(500, rel=0.1)

    def test_platform_with_orbit(self):
        """Test platform orbit functionality."""
        from eosim.library import create_platform, Position3D

        platform = create_platform(
            platform_type="rotary_wing",
            altitude_m=500,
            speed_ms=40,
        )

        # Set orbit
        center = Position3D(0, 0, 0, "m")
        platform.set_orbit(
            center=center,
            radius_m=1000,
            altitude_m=500,
            speed_ms=40,
        )

        # Should have waypoints
        assert len(platform.trajectory.waypoints) > 0

    def test_orientation_to_rotation_matrix(self):
        """Test orientation to rotation matrix conversion."""
        from eosim.library import Orientation3D

        # Identity - no rotation
        orient = Orientation3D(roll_deg=0, pitch_deg=0, yaw_deg=0)
        R = orient.to_rotation_matrix()
        assert R[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert R[1, 1] == pytest.approx(1.0, abs=1e-6)
        assert R[2, 2] == pytest.approx(1.0, abs=1e-6)

        # 90 degree yaw
        orient = Orientation3D(roll_deg=0, pitch_deg=0, yaw_deg=90)
        R = orient.to_rotation_matrix()
        # After 90 deg yaw, x-axis points in original y direction
        assert R[1, 0] == pytest.approx(1.0, abs=1e-6)

    def test_motion_blur_function(self):
        """Test motion blur application."""
        from eosim.library import apply_motion_blur

        # Create test image
        image = np.zeros((100, 100), dtype=np.float64)
        image[45:55, 45:55] = 100  # Bright square

        # Apply blur
        blurred = apply_motion_blur(image, blur_vector=(5, 0), strength=1.0)

        # Blurred image should be wider than original
        assert blurred.shape == image.shape
        # Center should still be bright but spread out
        assert blurred[50, 50] > 0

    def test_apply_jitter(self):
        """Test jitter application."""
        from eosim.library import apply_jitter

        # Create test image
        image = np.zeros((100, 100), dtype=np.float64)
        image[50, 50] = 100

        rng = np.random.default_rng(42)

        # Apply jitter
        jittered = apply_jitter(image, jitter_std_pixels=2.0, rng=rng)

        assert jittered.shape == image.shape

    def test_scenario_with_platform(self):
        """Test running scenario with 6DOF platform."""
        from eosim.library import ScenarioBuilder, run_scenario, EnvironmentType, BackgroundType

        scenario = (
            ScenarioBuilder()
            .set_name("6DOF Test")
            .set_sensor("generic_mwir_hd", altitude_m=1000)
            .add_target("civilian_car", range_km=2.0)
            .set_environment(env_type=EnvironmentType.CLEAR_DAY)
            .set_background(BackgroundType.TERRAIN)
            .set_platform(
                platform_type="fixed_wing",
                speed_ms=100,
                heading_deg=0,
            )
            .set_gimbal_track(target_idx=0)
            .set_seed(42)
            .build()
        )

        assert scenario.platform_config is not None
        assert scenario.platform_config.platform_type == "fixed_wing"
        assert scenario.platform_config.gimbal_mode == "track_target"

        # Run scenario
        result = run_scenario(scenario)

        # Should have platform info in metadata
        assert "platform" in result.metadata
        assert result.metadata["platform"]["type"] == "fixed_wing"
