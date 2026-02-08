"""Tests for EOSIM Enhanced Target Shapes Library."""

import pytest
import numpy as np

from eosim.targets.shapes import (
    ThermalZone,
    ThermalZoneConfig,
    EnhancedVehicleShape,
    HumanoidShape,
    AircraftWithPlumeShape,
    create_vehicle_signature,
    create_person_signature,
    create_aircraft_signature,
    create_gradient_falloff,
    create_linear_gradient,
    DEFAULT_VEHICLE_ZONES,
    DEFAULT_PERSON_ZONES,
    DEFAULT_AIRCRAFT_ZONES,
)
from eosim.targets import VehicleTarget, PersonTarget, AircraftTarget


class TestThermalZoneConfig:
    """Tests for ThermalZoneConfig."""

    def test_default_values(self):
        """Test default thermal zone config values."""
        config = ThermalZoneConfig(temperature_k=350.0)
        assert config.temperature_k == 350.0
        assert config.emissivity == 0.9
        assert config.gradient_falloff == 0.0
        assert config.noise_std_k == 1.0

    def test_custom_values(self):
        """Test custom thermal zone config values."""
        config = ThermalZoneConfig(
            temperature_k=400.0,
            emissivity=0.85,
            gradient_falloff=0.5,
            noise_std_k=5.0,
        )
        assert config.temperature_k == 400.0
        assert config.emissivity == 0.85
        assert config.gradient_falloff == 0.5
        assert config.noise_std_k == 5.0


class TestDefaultZones:
    """Tests for default thermal zone configurations."""

    def test_vehicle_zones_exist(self):
        """Test that all vehicle zones are defined."""
        required_zones = [
            ThermalZone.ENGINE,
            ThermalZone.CABIN,
            ThermalZone.WHEELS,
            ThermalZone.EXHAUST,
            ThermalZone.BODY,
        ]
        for zone in required_zones:
            assert zone in DEFAULT_VEHICLE_ZONES

    def test_vehicle_zone_temperatures(self):
        """Test vehicle zone temperature values match requirements."""
        # Engine: 370K when running
        assert DEFAULT_VEHICLE_ZONES[ThermalZone.ENGINE].temperature_k == 370.0
        # Cabin: 320K with occupants
        assert DEFAULT_VEHICLE_ZONES[ThermalZone.CABIN].temperature_k == 320.0
        # Wheels: 340K
        assert DEFAULT_VEHICLE_ZONES[ThermalZone.WHEELS].temperature_k == 340.0
        # Exhaust: 400K when running
        assert DEFAULT_VEHICLE_ZONES[ThermalZone.EXHAUST].temperature_k == 400.0
        # Body: 300K ambient
        assert DEFAULT_VEHICLE_ZONES[ThermalZone.BODY].temperature_k == 300.0

    def test_person_zones_exist(self):
        """Test that all person zones are defined."""
        required_zones = [
            ThermalZone.HEAD,
            ThermalZone.FACE,
            ThermalZone.HANDS,
            ThermalZone.TORSO,
            ThermalZone.LEGS,
            ThermalZone.FEET,
        ]
        for zone in required_zones:
            assert zone in DEFAULT_PERSON_ZONES

    def test_person_zone_temperatures(self):
        """Test person zone temperature values match requirements."""
        # Head: 308K
        assert DEFAULT_PERSON_ZONES[ThermalZone.HEAD].temperature_k == 308.0
        # Hands: 305K
        assert DEFAULT_PERSON_ZONES[ThermalZone.HANDS].temperature_k == 305.0
        # Torso: 295K (through clothing)
        assert DEFAULT_PERSON_ZONES[ThermalZone.TORSO].temperature_k == 295.0
        # Legs: 298K (through clothing)
        assert DEFAULT_PERSON_ZONES[ThermalZone.LEGS].temperature_k == 298.0

    def test_aircraft_zones_exist(self):
        """Test that all aircraft zones are defined."""
        required_zones = [
            ThermalZone.FUSELAGE,
            ThermalZone.WINGS,
            ThermalZone.NOZZLE,
            ThermalZone.PLUME,
        ]
        for zone in required_zones:
            assert zone in DEFAULT_AIRCRAFT_ZONES


class TestGradientFunctions:
    """Tests for gradient utility functions."""

    def test_gradient_falloff_center(self):
        """Test that gradient falloff is 1.0 at center."""
        gradient = create_gradient_falloff((100, 100), (50, 50), 20.0, 0.5)
        # Center should be close to 1.0
        assert gradient[50, 50] > 0.9

    def test_gradient_falloff_edges(self):
        """Test that gradient falloff decreases at edges."""
        gradient = create_gradient_falloff((100, 100), (50, 50), 10.0, 0.5)
        # Edges should be lower than center
        assert gradient[0, 0] < gradient[50, 50]
        assert gradient[99, 99] < gradient[50, 50]

    def test_linear_gradient(self):
        """Test linear gradient creation."""
        gradient = create_linear_gradient(
            (100, 100),
            (50, 10),  # Start
            (50, 90),  # End
            500.0,     # Start temp
            300.0,     # End temp
        )
        # Temperature should decrease from start to end
        assert gradient[50, 10] > gradient[50, 90]
        # Check approximate values
        assert gradient[50, 10] > 400  # Near start temp
        assert gradient[50, 90] < 400  # Near end temp


class TestEnhancedVehicleShape:
    """Tests for EnhancedVehicleShape."""

    def test_default_creation(self):
        """Test default vehicle shape creation."""
        shape = EnhancedVehicleShape()
        temp_map, emis_map, mask = shape.render((64, 64))

        assert temp_map.shape == (64, 64)
        assert emis_map.shape == (64, 64)
        assert mask.shape == (64, 64)
        assert mask.any()  # Should have some True values

    def test_engine_running_increases_temperature(self):
        """Test that running engine increases temperature."""
        shape_off = EnhancedVehicleShape(engine_running=False)
        shape_on = EnhancedVehicleShape(engine_running=True)

        temp_off, _, _ = shape_off.render((64, 64))
        temp_on, _, _ = shape_on.render((64, 64))

        # Running engine should have higher max temperature
        assert temp_on.max() > temp_off.max()

    def test_speed_affects_wheel_temperature(self):
        """Test that speed affects wheel temperature."""
        shape_slow = EnhancedVehicleShape(engine_running=True, speed_kmh=20)
        shape_fast = EnhancedVehicleShape(engine_running=True, speed_kmh=100)

        # Fast vehicle should have higher wheel temperature in config
        assert shape_fast.zones[ThermalZone.WHEELS].temperature_k > \
               shape_slow.zones[ThermalZone.WHEELS].temperature_k

    def test_vehicle_types(self):
        """Test different vehicle types."""
        for vehicle_type in ["sedan", "suv", "truck", "tank"]:
            shape = EnhancedVehicleShape(vehicle_type=vehicle_type)
            temp_map, emis_map, mask = shape.render((64, 64))
            assert temp_map.shape == (64, 64)
            assert mask.any()

    def test_custom_zone_configs(self):
        """Test custom zone configurations."""
        custom_zones = {
            ThermalZone.ENGINE: ThermalZoneConfig(450.0, 0.8)
        }
        shape = EnhancedVehicleShape(
            engine_running=True,
            zone_configs=custom_zones,
        )
        # Custom engine temp should be used (though adjusted for running state)
        assert shape.zones[ThermalZone.ENGINE] is not None


class TestHumanoidShape:
    """Tests for HumanoidShape."""

    def test_default_creation(self):
        """Test default humanoid shape creation."""
        shape = HumanoidShape()
        temp_map, emis_map, mask = shape.render((64, 32))

        assert temp_map.shape == (64, 32)
        assert emis_map.shape == (64, 32)
        assert mask.shape == (64, 32)
        assert mask.any()

    def test_standing_pose(self):
        """Test standing pose rendering."""
        shape = HumanoidShape(pose="standing")
        temp_map, emis_map, mask = shape.render((64, 32))

        # Should have vertical elongated shape
        # Head should be at top (lower y values are higher temps)
        head_region = temp_map[:16, :]
        assert head_region.max() > 300  # Head should be warm

    def test_sitting_pose(self):
        """Test sitting pose rendering."""
        shape = HumanoidShape(pose="sitting")
        temp_map, emis_map, mask = shape.render((64, 32))
        assert mask.any()

    def test_prone_pose(self):
        """Test prone pose rendering."""
        shape = HumanoidShape(pose="prone")
        temp_map, emis_map, mask = shape.render((64, 32))
        assert mask.any()

    def test_activity_affects_temperature(self):
        """Test that activity level affects temperature."""
        idle = HumanoidShape(activity="idle")
        running = HumanoidShape(activity="running")

        temp_idle, _, _ = idle.render((64, 32))
        temp_run, _, _ = running.render((64, 32))

        # Running should be warmer (higher activity increases temps)
        assert temp_run.max() > temp_idle.max()

    def test_head_is_warmest(self):
        """Test that head is warmest exposed area."""
        shape = HumanoidShape(pose="standing")
        temp_map, _, mask = shape.render((64, 32))

        # Get maximum temperature (should be from head region)
        max_temp = temp_map.max()
        # Head zone should be ~308K
        assert max_temp > 305


class TestAircraftWithPlumeShape:
    """Tests for AircraftWithPlumeShape."""

    def test_default_creation(self):
        """Test default aircraft shape creation."""
        shape = AircraftWithPlumeShape()
        temp_map, emis_map, mask = shape.render((64, 64))

        assert temp_map.shape == (64, 64)
        assert emis_map.shape == (64, 64)
        assert mask.shape == (64, 64)
        assert mask.any()

    def test_rear_view_shows_plume(self):
        """Test that rear view (180 deg) shows exhaust plume."""
        shape = AircraftWithPlumeShape(throttle=0.8)
        temp_map, _, _ = shape.render((64, 64), aspect_angle_deg=180)

        # Rear portion should have high temperatures (nozzle/plume)
        rear_section = temp_map[:, 50:]
        assert rear_section.max() > 400  # Hot nozzle

    def test_afterburner_increases_temperature(self):
        """Test that afterburner significantly increases temperature."""
        shape_normal = AircraftWithPlumeShape(throttle=0.8, afterburner=False)
        shape_ab = AircraftWithPlumeShape(throttle=0.8, afterburner=True)

        temp_normal, _, _ = shape_normal.render((64, 64), aspect_angle_deg=180)
        temp_ab, _, _ = shape_ab.render((64, 64), aspect_angle_deg=180)

        # Afterburner should produce much higher temps
        assert temp_ab.max() > temp_normal.max()

    def test_throttle_affects_plume(self):
        """Test that throttle affects plume characteristics."""
        shape_low = AircraftWithPlumeShape(throttle=0.2)
        shape_high = AircraftWithPlumeShape(throttle=0.9)

        temp_low, _, _ = shape_low.render((64, 64), aspect_angle_deg=180)
        temp_high, _, _ = shape_high.render((64, 64), aspect_angle_deg=180)

        # Higher throttle = hotter plume
        assert temp_high.max() > temp_low.max()

    def test_altitude_affects_ambient(self):
        """Test that altitude affects ambient temperature."""
        shape_low = AircraftWithPlumeShape(altitude_m=1000)
        shape_high = AircraftWithPlumeShape(altitude_m=10000)

        # Higher altitude = colder ambient
        assert shape_high.ambient_k < shape_low.ambient_k


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_vehicle_signature_sedan(self):
        """Test vehicle signature creation."""
        temp_map, emis_map, mask = create_vehicle_signature(
            vehicle_type="sedan",
            engine_running=True,
            speed_kmh=60,
        )
        assert temp_map.shape == (64, 64)
        assert temp_map.max() > 300

    def test_create_vehicle_signature_types(self):
        """Test different vehicle types."""
        for vtype in ["sedan", "suv", "truck", "tank"]:
            temp_map, emis_map, mask = create_vehicle_signature(vehicle_type=vtype)
            assert temp_map.shape == (64, 64)
            assert mask.any()

    def test_create_person_signature(self):
        """Test person signature creation."""
        temp_map, emis_map, mask = create_person_signature(
            pose="standing",
            activity="walking",
        )
        assert temp_map.shape == (64, 32)
        assert temp_map.max() > 300

    def test_create_aircraft_signature(self):
        """Test aircraft signature creation."""
        temp_map, emis_map, mask = create_aircraft_signature(
            aircraft_type="fighter",
            throttle=0.8,
            afterburner=True,
        )
        assert temp_map.shape == (64, 64)
        assert temp_map.max() > 500  # Afterburner hot


class TestIntegrationWithTargetClasses:
    """Tests for integration with existing target classes."""

    def test_vehicle_target_enhanced_shape(self):
        """Test VehicleTarget with enhanced shape."""
        vehicle = VehicleTarget.sedan(engine_state="running", speed_kmh=60)
        sig_enhanced = vehicle.get_signature(use_enhanced_shape=True)
        sig_legacy = vehicle.get_signature(use_enhanced_shape=False)

        # Both should produce valid signatures
        assert sig_enhanced.temperature_map.shape == (64, 64)
        assert sig_legacy.temperature_map.shape == (64, 64)

        # Enhanced should have distinct thermal zones
        assert sig_enhanced.max_temperature > 300

    def test_person_target_enhanced_shape(self):
        """Test PersonTarget with enhanced humanoid shape."""
        person = PersonTarget.standing(activity="walking")
        sig_enhanced = person.get_signature(use_enhanced_shape=True)
        sig_legacy = person.get_signature(use_enhanced_shape=False)

        # Both should produce valid signatures
        assert sig_enhanced.temperature_map.shape[0] > 0
        assert sig_legacy.temperature_map.shape[0] > 0

        # Head should be warmest in enhanced (around 308K)
        assert sig_enhanced.max_temperature > 305

    def test_aircraft_target_enhanced_shape(self):
        """Test AircraftTarget with exhaust plume."""
        jet = AircraftTarget.fighter_jet(throttle=0.9, afterburner=True)
        sig_enhanced = jet.get_signature(use_enhanced_shape=True, aspect_angle_deg=180)
        sig_legacy = jet.get_signature(use_enhanced_shape=False, aspect_angle_deg=180)

        # Both should produce valid signatures
        assert sig_enhanced.temperature_map.shape == (64, 64)
        assert sig_legacy.temperature_map.shape == (64, 64)

        # Enhanced with afterburner should be very hot
        assert sig_enhanced.max_temperature > 500

    def test_vehicle_aspect_angles(self):
        """Test vehicle at different viewing angles."""
        vehicle = VehicleTarget.sedan(engine_state="running")

        for angle in [0, 45, 90, 135, 180]:
            sig = vehicle.get_signature(
                use_enhanced_shape=True,
                aspect_angle_deg=angle,
            )
            assert sig.temperature_map.shape == (64, 64)
            assert sig.max_temperature > 290


class TestThermalGradients:
    """Tests for thermal gradient behavior."""

    def test_exhaust_plume_gradient(self):
        """Test that exhaust plume has decreasing temperature gradient."""
        shape = AircraftWithPlumeShape(throttle=0.8)
        temp_map, _, _ = shape.render((64, 128), aspect_angle_deg=180)

        # Temperature should decrease from nozzle (rear) toward ambient
        # Check that max temp is near the rear
        max_idx = np.unravel_index(np.argmax(temp_map), temp_map.shape)
        # Max should be in the rear half
        assert max_idx[1] > 60  # More than halfway to the right

    def test_engine_zone_gradient(self):
        """Test that engine zone has gradient falloff."""
        shape = EnhancedVehicleShape(engine_running=True)
        temp_map, _, mask = shape.render((64, 64))

        # Engine zone should have gradient (not uniform hot spot)
        # Find the hottest region
        hot_pixels = temp_map > 350
        if hot_pixels.any():
            # Should have some pixels with intermediate temperatures
            warm_pixels = (temp_map > 320) & (temp_map < 350)
            assert warm_pixels.any()  # Gradient exists
