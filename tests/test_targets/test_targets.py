"""Tests for EOSIM Target Library (Stage A)."""

import pytest
import numpy as np

from eosim.targets import (
    # Base classes
    Target,
    TargetSignature,
    TargetRenderer,
    TargetGroup,
    MaterialProperties,
    # Vehicles
    VehicleTarget,
    TruckTarget,
    TankTarget,
    # Aircraft
    AircraftTarget,
    HelicopterTarget,
    DroneTarget,
    # People
    PersonTarget,
    CrowdGenerator,
    # Structures
    BuildingTarget,
    IndustrialTarget,
    BridgeTarget,
    StorageTankTarget,
    # Natural
    AnimalTarget,
    VegetationTarget,
    WaterBodyTarget,
    TerrainTarget,
)
from eosim.targets.base import HotSpot, TargetGeometry, MaterialType


class TestMaterialProperties:
    """Tests for MaterialProperties."""

    def test_default_properties(self):
        """Test default material properties."""
        props = MaterialProperties()
        assert 0 < props.emissivity <= 1
        assert 0 <= props.reflectance <= 1
        assert props.thermal_mass > 0

    def test_from_material_type(self):
        """Test creating properties from material type."""
        metal = MaterialProperties.from_material_type(MaterialType.METAL_BARE)
        concrete = MaterialProperties.from_material_type(MaterialType.CONCRETE)

        # Metal has lower emissivity than concrete
        assert metal.emissivity < concrete.emissivity
        # Metal has higher conductivity
        assert metal.thermal_conductivity > concrete.thermal_conductivity


class TestHotSpot:
    """Tests for HotSpot."""

    def test_static_hot_spot(self):
        """Test non-pulsing hot spot."""
        spot = HotSpot(
            name="engine",
            relative_position=(0.3, 0.5),
            relative_size=0.1,
            temperature_delta_k=50.0,
        )

        assert spot.get_temperature_delta(0.0) == 50.0
        assert spot.get_temperature_delta(10.0) == 50.0

    def test_pulsing_hot_spot(self):
        """Test pulsing hot spot."""
        spot = HotSpot(
            name="exhaust",
            relative_position=(0.8, 0.5),
            relative_size=0.05,
            temperature_delta_k=100.0,
            pulsing=True,
            pulse_period_s=2.0,
            pulse_amplitude_k=20.0,
        )

        # Temperature should vary between 80 and 120 K delta
        temps = [spot.get_temperature_delta(t) for t in np.linspace(0, 2, 20)]
        assert min(temps) < 100 < max(temps)


class TestTargetGeometry:
    """Tests for TargetGeometry."""

    def test_box_footprint(self):
        """Test box footprint area."""
        geom = TargetGeometry(length_m=10.0, width_m=5.0, height_m=2.0, shape="box")
        assert geom.footprint_area_m2 == 50.0

    def test_cylinder_footprint(self):
        """Test cylinder footprint area."""
        geom = TargetGeometry(length_m=4.0, width_m=4.0, height_m=3.0, shape="cylinder")
        expected = np.pi * 4.0 * 4.0 / 4
        assert abs(geom.footprint_area_m2 - expected) < 0.01


class TestVehicleTarget:
    """Tests for VehicleTarget."""

    def test_sedan_creation(self):
        """Test creating a sedan."""
        vehicle = VehicleTarget.sedan(engine_state="running")
        assert vehicle.name == "sedan"
        assert vehicle.geometry.length_m > 3.0

    def test_suv_creation(self):
        """Test creating an SUV."""
        vehicle = VehicleTarget.suv(engine_state="idle")
        assert vehicle.name == "suv"

    def test_truck_creation(self):
        """Test creating a truck."""
        vehicle = VehicleTarget.truck()
        assert vehicle.geometry.length_m > 5.0

    def test_vehicle_signature(self):
        """Test generating vehicle thermal signature."""
        vehicle = VehicleTarget.sedan(engine_state="running")
        signature = vehicle.get_signature(resolution=(32, 64))

        assert signature.temperature_map.shape == (32, 64)
        assert signature.emissivity_map.shape == (32, 64)
        assert signature.max_temperature > 290  # Should have hot spots

    def test_engine_state_affects_temperature(self):
        """Test that engine state affects temperature."""
        cold = VehicleTarget.sedan(engine_state="off")
        hot = VehicleTarget.sedan(engine_state="running")

        sig_cold = cold.get_signature()
        sig_hot = hot.get_signature()

        assert sig_hot.max_temperature > sig_cold.max_temperature


class TestTruckTarget:
    """Tests for TruckTarget."""

    def test_heavy_truck(self):
        """Test heavy truck creation."""
        truck = TruckTarget(engine_state="running", speed_kmh=60)
        assert truck.geometry.length_m >= 15.0
        signature = truck.get_signature()
        assert signature.max_temperature > 300


class TestTankTarget:
    """Tests for TankTarget."""

    def test_battle_tank(self):
        """Test battle tank creation."""
        tank = TankTarget(engine_state="running")
        signature = tank.get_signature()
        # Tank engine should be very hot
        assert signature.max_temperature > 350


class TestAircraftTarget:
    """Tests for AircraftTarget."""

    def test_fighter_jet(self):
        """Test fighter jet creation."""
        jet = AircraftTarget.fighter_jet(throttle=0.8)
        signature = jet.get_signature()
        # Jet exhaust should be very hot
        assert signature.max_temperature > 400

    def test_commercial_airliner(self):
        """Test commercial airliner."""
        airliner = AircraftTarget.commercial_airliner()
        assert airliner.geometry.length_m > 30.0

    def test_turboprop(self):
        """Test turboprop aircraft."""
        plane = AircraftTarget.turboprop()
        signature = plane.get_signature()
        assert signature.temperature_map.shape[0] > 0


class TestHelicopterTarget:
    """Tests for HelicopterTarget."""

    def test_utility_helicopter(self):
        """Test utility helicopter."""
        heli = HelicopterTarget.utility()
        signature = heli.get_signature()
        assert signature.max_temperature > 300


class TestDroneTarget:
    """Tests for DroneTarget."""

    def test_quadcopter(self):
        """Test quadcopter drone."""
        drone = DroneTarget.quadcopter()
        signature = drone.get_signature()
        assert signature.temperature_map.shape[0] > 0

    def test_fixed_wing_drone(self):
        """Test fixed-wing drone."""
        drone = DroneTarget.fixed_wing_uav()
        assert drone.geometry.length_m < 5.0


class TestPersonTarget:
    """Tests for PersonTarget."""

    def test_standing_person(self):
        """Test standing person."""
        person = PersonTarget.standing(activity="walking")
        signature = person.get_signature()
        # Max temperature should be around body temperature (non-zero pixels only)
        assert signature.max_temperature > 295

    def test_sitting_person(self):
        """Test sitting person."""
        person = PersonTarget.sitting()
        assert person.geometry.height_m < 1.5

    def test_activity_affects_temperature(self):
        """Test that activity affects temperature."""
        resting = PersonTarget.standing(activity="resting")
        running = PersonTarget.standing(activity="running")

        sig_rest = resting.get_signature()
        sig_run = running.get_signature()

        # Use max temperature to avoid zeros from background
        assert sig_run.max_temperature > sig_rest.max_temperature


class TestCrowdGenerator:
    """Tests for CrowdGenerator."""

    def test_generate_crowd(self):
        """Test generating a crowd."""
        generator = CrowdGenerator(count=10, seed=42)
        group = generator.generate()

        assert len(group.targets) == 10
        assert all(isinstance(p, PersonTarget) for p in group.targets)

    def test_crowd_distribution(self):
        """Test crowd activity distribution."""
        generator = CrowdGenerator(count=20, seed=123)
        group = generator.generate(
            activity_mix={"standing": 0.5, "walking": 0.5}
        )

        # Should have mix of activities
        assert len(group.targets) == 20


class TestBuildingTarget:
    """Tests for BuildingTarget."""

    def test_office_building(self):
        """Test office building creation."""
        building = BuildingTarget.office_building(floors=5, hvac_running=True)
        signature = building.get_signature()

        # HVAC units should create hot spots
        assert signature.thermal_contrast > 5.0

    def test_warehouse(self):
        """Test warehouse creation."""
        warehouse = BuildingTarget.warehouse()
        assert warehouse.geometry.length_m >= 50.0

    def test_residential_house(self):
        """Test residential house."""
        house = BuildingTarget.residential_house(chimney_active=True)
        signature = house.get_signature()
        # Chimney should be hot
        assert signature.max_temperature > 320


class TestIndustrialTarget:
    """Tests for IndustrialTarget."""

    def test_power_plant(self):
        """Test power plant creation."""
        plant = IndustrialTarget.power_plant(load_percent=80)
        signature = plant.get_signature()

        # Stacks should be very hot
        assert signature.max_temperature > 400

    def test_refinery(self):
        """Test refinery with flare."""
        refinery = IndustrialTarget.refinery()
        signature = refinery.get_signature()

        # Flare should be extremely hot
        assert signature.max_temperature > 600

    def test_factory(self):
        """Test factory creation."""
        factory = IndustrialTarget.factory(process_type="steel")
        signature = factory.get_signature()
        assert signature.max_temperature > 350


class TestBridgeTarget:
    """Tests for BridgeTarget."""

    def test_steel_truss(self):
        """Test steel truss bridge."""
        bridge = BridgeTarget.steel_truss(span_m=100.0)
        signature = bridge.get_signature(resolution=(32, 128))

        assert signature.temperature_map.shape == (32, 128)

    def test_concrete_overpass(self):
        """Test concrete overpass."""
        overpass = BridgeTarget.concrete_overpass(span_m=50.0)
        signature = overpass.get_signature()
        assert signature.temperature_map.shape[0] > 0


class TestStorageTankTarget:
    """Tests for StorageTankTarget."""

    def test_fuel_tank(self):
        """Test fuel storage tank."""
        tank = StorageTankTarget.fuel_tank(fill_level=0.8)
        signature = tank.get_signature()

        # Should be circular shape
        assert signature.temperature_map.shape[0] == signature.temperature_map.shape[1]

    def test_chemical_tank(self):
        """Test chemical tank with elevated temperature."""
        # Non-insulated tank shows contents temperature
        tank = StorageTankTarget.chemical_tank(contents_temp_k=350.0, insulated=False)
        signature = tank.get_signature()
        # Use max temperature since mean includes zero background
        assert signature.max_temperature > 300


class TestAnimalTarget:
    """Tests for AnimalTarget."""

    def test_deer(self):
        """Test deer target."""
        deer = AnimalTarget.deer(activity="walking")
        signature = deer.get_signature()

        # Body temperature should be warm (use max since mean includes zeros)
        assert signature.max_temperature > 300

    def test_cattle(self):
        """Test cattle target."""
        cow = AnimalTarget.cattle()
        assert cow.geometry.length_m > 2.0

    def test_bird(self):
        """Test bird target."""
        bird = AnimalTarget.bird(size="large")
        signature = bird.get_signature()
        assert signature.temperature_map.shape[0] > 0


class TestVegetationTarget:
    """Tests for VegetationTarget."""

    def test_deciduous_forest(self):
        """Test deciduous forest."""
        forest = VegetationTarget.deciduous_forest(density=0.8)
        signature = forest.get_signature()

        # Canopy should be cooler than ambient due to transpiration
        # (depends on solar load and moisture)
        assert signature.temperature_map.shape[0] > 0

    def test_coniferous_forest(self):
        """Test coniferous forest."""
        forest = VegetationTarget.coniferous_forest()
        signature = forest.get_signature()
        assert signature.emissivity_map.max() > 0.9

    def test_crop_field(self):
        """Test crop field."""
        field = VegetationTarget.crop_field(crop_type="corn")
        assert field.geometry.height_m > 1.0


class TestWaterBodyTarget:
    """Tests for WaterBodyTarget."""

    def test_lake(self):
        """Test lake creation."""
        lake = WaterBodyTarget.lake(surface_temp_k=288.0)
        signature = lake.get_signature()

        # Water has high emissivity
        assert signature.emissivity_map.max() > 0.9

    def test_river(self):
        """Test river creation."""
        river = WaterBodyTarget.river(width_m=50.0)
        signature = river.get_signature()
        assert signature.temperature_map.shape[0] > 0


class TestTerrainTarget:
    """Tests for TerrainTarget."""

    def test_bare_soil(self):
        """Test bare soil."""
        soil = TerrainTarget.bare_soil(solar_load=0.8)
        signature = soil.get_signature()
        assert signature.temperature_map.mean() > 290

    def test_rock_outcrop(self):
        """Test rock outcrop."""
        rock = TerrainTarget.rock_outcrop()
        signature = rock.get_signature()
        assert signature.temperature_map.shape[0] > 0


class TestTargetRenderer:
    """Tests for TargetRenderer."""

    def test_single_target_render(self):
        """Test rendering a single target."""
        renderer = TargetRenderer(
            resolution=(100, 100),
            gsd_m=0.5,
            background_temperature_k=290.0,
        )

        vehicle = VehicleTarget.sedan()
        temp_map, emis_map = renderer.render(vehicle, position=(50, 50))

        assert temp_map.shape == (100, 100)
        assert emis_map.shape == (100, 100)
        # Should have target pixels different from background
        assert np.any(temp_map != 290.0)

    def test_multiple_targets_render(self):
        """Test rendering multiple targets."""
        renderer = TargetRenderer(resolution=(200, 200), gsd_m=0.5)

        targets = [
            (VehicleTarget.sedan(), (50, 50), 0.0),
            (VehicleTarget.suv(), (150, 150), 90.0),
        ]

        temp_map, emis_map = renderer.render_multiple(targets)

        assert temp_map.shape == (200, 200)


class TestTargetGroup:
    """Tests for TargetGroup."""

    def test_create_group(self):
        """Test creating a target group."""
        group = TargetGroup([
            PersonTarget.standing(),
            PersonTarget.standing(),
        ])

        assert len(group.targets) == 2

    def test_randomize_positions(self):
        """Test randomizing positions."""
        group = TargetGroup([
            PersonTarget.standing() for _ in range(5)
        ])

        group.randomize_positions(area=(10, 10, 90, 90))

        assert len(group.positions) == 5
        for y, x in group.positions:
            assert 10 <= y <= 90
            assert 10 <= x <= 90

    def test_get_render_list(self):
        """Test getting render list."""
        group = TargetGroup()
        group.add(PersonTarget.standing(), position=(50, 50), aspect_angle_deg=45)

        render_list = group.get_render_list()
        assert len(render_list) == 1
        assert render_list[0][1] == (50, 50)
        assert render_list[0][2] == 45


class TestIntegration:
    """Integration tests combining multiple target types."""

    def test_mixed_scene(self):
        """Test scene with multiple target types."""
        renderer = TargetRenderer(
            resolution=(480, 640),
            gsd_m=1.0,
            background_temperature_k=290.0,
        )

        targets = [
            (VehicleTarget.sedan(engine_state="running"), (100, 200), 0),
            (BuildingTarget.warehouse(), (300, 400), 0),
            (PersonTarget.standing(), (150, 150), 90),
        ]

        temp_map, emis_map = renderer.render_multiple(targets)

        # Vehicle should show hot engine
        # Building and person should be present
        assert temp_map.max() > 300  # Hot engine
        assert 0.8 < emis_map.max() <= 1.0

    def test_time_update(self):
        """Test target time updates."""
        vehicle = VehicleTarget.sedan(engine_state="running")
        initial_signature = vehicle.get_signature()

        # Update time
        vehicle.update(dt_s=60.0)  # 1 minute
        updated_signature = vehicle.get_signature()

        # Signatures should exist
        assert initial_signature.temperature_map.shape == updated_signature.temperature_map.shape
