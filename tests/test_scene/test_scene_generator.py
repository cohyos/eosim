"""Tests for EOSIM Scene Generator (Stage C)."""

import pytest
import numpy as np

from eosim.scene import (
    SceneGenerator,
    SceneConfig,
    SceneResult,
    EnvironmentConfig,
    BackgroundConfig,
    TimeOfDay,
    TerrainType,
    WeatherCondition,
    SceneTemplate,
    RandomizationConfig,
    create_military_scenario,
    create_surveillance_scenario,
)
from eosim.targets import VehicleTarget, PersonTarget, TankTarget, TargetGroup


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig."""

    def test_default_values(self):
        """Test default environment configuration."""
        config = EnvironmentConfig()
        assert config.time_of_day == TimeOfDay.MIDDAY
        assert config.ambient_temperature_k == 295.0
        assert config.weather == WeatherCondition.CLEAR

    def test_solar_elevation_from_time(self):
        """Test solar elevation derived from time of day."""
        dawn = EnvironmentConfig(time_of_day=TimeOfDay.DAWN)
        midday = EnvironmentConfig(time_of_day=TimeOfDay.MIDDAY)
        night = EnvironmentConfig(time_of_day=TimeOfDay.NIGHT)

        assert dawn.solar_elevation_deg < midday.solar_elevation_deg
        assert night.solar_elevation_deg < 0

    def test_explicit_solar_elevation(self):
        """Test explicit solar elevation override."""
        config = EnvironmentConfig(
            time_of_day=TimeOfDay.DAWN,
            solar_elevation_deg=45.0,
        )
        assert config.solar_elevation_deg == 45.0


class TestBackgroundConfig:
    """Tests for BackgroundConfig."""

    def test_default_values(self):
        """Test default background configuration."""
        config = BackgroundConfig()
        assert config.terrain_type == TerrainType.GRASS
        assert 0.8 < config.emissivity <= 1.0

    def test_terrain_types(self):
        """Test different terrain types."""
        for terrain in TerrainType:
            config = BackgroundConfig(terrain_type=terrain)
            assert config.terrain_type == terrain


class TestSceneConfig:
    """Tests for SceneConfig."""

    def test_default_values(self):
        """Test default scene configuration."""
        config = SceneConfig()
        assert config.resolution == (480, 640)
        assert config.gsd_m == 0.5
        assert config.environment is not None
        assert config.background is not None


class TestSceneGenerator:
    """Tests for SceneGenerator."""

    def test_basic_generation(self):
        """Test basic scene generation."""
        config = SceneConfig(resolution=(100, 100), gsd_m=1.0)
        gen = SceneGenerator(config)

        result = gen.generate()

        assert isinstance(result, SceneResult)
        assert result.temperature_map.shape == (100, 100)
        assert result.emissivity_map.shape == (100, 100)

    def test_add_single_target(self):
        """Test adding a single target."""
        config = SceneConfig(resolution=(100, 100), gsd_m=0.5)
        gen = SceneGenerator(config)

        vehicle = VehicleTarget.sedan(engine_state="running")
        gen.add_target(vehicle, position=(50, 50))

        result = gen.generate()

        # Should have at least one target
        assert result.target_mask.any()
        assert result.metadata["num_targets"] == 1

    def test_add_multiple_targets(self):
        """Test adding multiple targets."""
        config = SceneConfig(resolution=(200, 200), gsd_m=0.5)
        gen = SceneGenerator(config)

        gen.add_target(VehicleTarget.sedan(), position=(50, 50))
        gen.add_target(VehicleTarget.suv(), position=(150, 150))

        result = gen.generate()

        assert result.metadata["num_targets"] == 2

    def test_add_target_group(self):
        """Test adding a target group."""
        config = SceneConfig(resolution=(200, 200), gsd_m=0.5)
        gen = SceneGenerator(config)

        people = [PersonTarget.standing() for _ in range(5)]
        group = TargetGroup(people)
        group.randomize_positions(area=(20, 20, 180, 180))
        gen.add_target_group(group)

        result = gen.generate()

        assert result.metadata["num_targets"] == 5

    def test_add_random_targets(self):
        """Test adding targets at random positions."""
        config = SceneConfig(resolution=(200, 200), gsd_m=0.5)
        gen = SceneGenerator(config)

        vehicles = [VehicleTarget.sedan() for _ in range(3)]
        gen.add_random_targets(vehicles)

        result = gen.generate()

        assert result.metadata["num_targets"] == 3
        assert result.target_mask.any()

    def test_terrain_temperature(self):
        """Test different terrain types affect background temperature."""
        desert_config = SceneConfig(
            resolution=(50, 50),
            background=BackgroundConfig(terrain_type=TerrainType.DESERT),
        )
        water_config = SceneConfig(
            resolution=(50, 50),
            background=BackgroundConfig(terrain_type=TerrainType.WATER),
        )

        desert_gen = SceneGenerator(desert_config)
        water_gen = SceneGenerator(water_config)

        desert_result = desert_gen.generate()
        water_result = water_gen.generate()

        # Desert should be warmer than water
        assert desert_result.temperature_map.mean() > water_result.temperature_map.mean()

    def test_time_of_day_effect(self):
        """Test time of day affects scene temperature."""
        midday_config = SceneConfig(
            resolution=(50, 50),
            environment=EnvironmentConfig(time_of_day=TimeOfDay.MIDDAY),
        )
        night_config = SceneConfig(
            resolution=(50, 50),
            environment=EnvironmentConfig(time_of_day=TimeOfDay.NIGHT),
        )

        midday_gen = SceneGenerator(midday_config)
        night_gen = SceneGenerator(night_config)

        midday_result = midday_gen.generate()
        night_result = night_gen.generate()

        # Midday should be warmer
        assert midday_result.temperature_map.mean() > night_result.temperature_map.mean()

    def test_method_chaining(self):
        """Test method chaining for add_target."""
        config = SceneConfig(resolution=(100, 100))
        gen = SceneGenerator(config)

        result = (
            gen
            .add_target(VehicleTarget.sedan(), (30, 30))
            .add_target(VehicleTarget.suv(), (70, 70))
            .generate()
        )

        assert result.metadata["num_targets"] == 2

    def test_target_labels(self):
        """Test that targets have unique labels."""
        config = SceneConfig(resolution=(200, 200), gsd_m=0.5)
        gen = SceneGenerator(config)

        gen.add_target(VehicleTarget.sedan(), (50, 50))
        gen.add_target(VehicleTarget.suv(), (150, 150))

        result = gen.generate()

        unique_labels = np.unique(result.target_labels)
        # Should have 0 (background) plus labels for each target
        assert len(unique_labels) >= 2


class TestSceneTemplate:
    """Tests for SceneTemplate."""

    def test_urban_template(self):
        """Test urban environment template."""
        gen = SceneTemplate.urban_environment()
        assert gen.config.background.terrain_type == TerrainType.URBAN

    def test_desert_template(self):
        """Test desert environment template."""
        gen = SceneTemplate.desert_environment()
        assert gen.config.background.terrain_type == TerrainType.DESERT
        assert gen.config.environment.ambient_temperature_k > 300

    def test_night_template(self):
        """Test night scenario template."""
        gen = SceneTemplate.night_scenario()
        assert gen.config.environment.time_of_day == TimeOfDay.NIGHT

    def test_generate_batch(self):
        """Test batch generation from template."""
        template = SceneTemplate.urban_environment(resolution=(50, 50))

        def target_factory():
            return [VehicleTarget.sedan()]

        scenes = SceneTemplate.generate_batch(
            template, count=3, target_factory=target_factory
        )

        assert len(scenes) == 3
        for scene in scenes:
            assert scene.temperature_map.shape == (50, 50)


class TestScenarioFunctions:
    """Tests for scenario creation functions."""

    def test_military_scenario(self):
        """Test military scenario creation."""
        gen = create_military_scenario(
            num_vehicles=2,
            num_people=3,
            terrain_type="desert",
            time_of_day="midday",
            resolution=(100, 100),
        )

        result = gen.generate()

        assert result.shape == (100, 100)
        # Should have vehicles + people
        assert result.metadata["num_targets"] == 5

    def test_surveillance_scenario(self):
        """Test surveillance scenario creation."""
        gen = create_surveillance_scenario(
            num_targets=3,
            target_type="vehicle",
            resolution=(100, 100),
        )

        result = gen.generate()

        assert result.shape == (100, 100)
        assert result.metadata["num_targets"] == 3

    def test_mixed_surveillance(self):
        """Test mixed target surveillance scenario."""
        gen = create_surveillance_scenario(
            num_targets=4,
            target_type="mixed",
            resolution=(100, 100),
        )

        result = gen.generate()

        assert result.metadata["num_targets"] == 4


class TestSceneResult:
    """Tests for SceneResult."""

    def test_result_properties(self):
        """Test SceneResult properties."""
        gen = SceneGenerator(SceneConfig(resolution=(50, 50)))
        result = gen.generate()

        assert result.shape == (50, 50)
        assert result.temperature_map.dtype == np.float64
        assert result.emissivity_map.dtype == np.float64
        assert result.target_mask.dtype == bool

    def test_metadata(self):
        """Test result metadata."""
        gen = SceneGenerator(SceneConfig())
        result = gen.generate()

        assert "config" in result.metadata
        assert "num_targets" in result.metadata
        assert "background_temp_k" in result.metadata


class TestIntegration:
    """Integration tests for scene generation."""

    def test_complete_scene_workflow(self):
        """Test complete scene generation workflow."""
        # Configure scene
        config = SceneConfig(
            resolution=(200, 300),
            gsd_m=0.5,
            sensor_type="lwir",
            environment=EnvironmentConfig(
                time_of_day=TimeOfDay.AFTERNOON,
                ambient_temperature_k=298.0,
                weather=WeatherCondition.CLEAR,
            ),
            background=BackgroundConfig(
                terrain_type=TerrainType.URBAN,
                temperature_variation_k=3.0,
            ),
            random_seed=42,
        )

        # Create generator
        gen = SceneGenerator(config)

        # Add various targets
        gen.add_target(VehicleTarget.sedan(engine_state="running"), (50, 100))
        gen.add_target(TankTarget(engine_state="idle"), (150, 200))

        people = [PersonTarget.standing(activity="walking") for _ in range(3)]
        gen.add_random_targets(people, area=(80, 80, 120, 220))

        # Generate
        result = gen.generate()

        # Verify result
        assert result.shape == (200, 300)
        assert result.target_mask.any()
        assert result.metadata["num_targets"] == 5

        # Check temperature ranges
        assert result.temperature_map.min() > 250
        assert result.temperature_map.max() < 500

        # Check emissivity ranges - background emissivity (exclude targets which may have 0)
        bg_emis = result.emissivity_map[~result.target_mask]
        assert bg_emis.min() > 0.7
        assert result.emissivity_map.max() <= 1.0

    def test_reproducibility(self):
        """Test scene generation reproducibility with seed for background."""
        config1 = SceneConfig(resolution=(50, 50), random_seed=123)
        config2 = SceneConfig(resolution=(50, 50), random_seed=123)

        gen1 = SceneGenerator(config1)
        result1 = gen1.generate()

        gen2 = SceneGenerator(config2)
        result2 = gen2.generate()

        # Background (no targets) should be identical with same seed
        np.testing.assert_array_equal(
            result1.temperature_map,
            result2.temperature_map,
        )
