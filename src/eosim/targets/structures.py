"""
EOSIM Structure Targets - Buildings, Infrastructure, and Installations.

Provides target models for buildings, power plants, industrial facilities,
bridges, and other man-made structures with realistic thermal signatures.

Example 1: Building with HVAC exhaust
    >>> from eosim.targets import BuildingTarget
    >>> building = BuildingTarget.office_building(floors=5, hvac_running=True)
    >>> signature = building.get_signature(resolution=(128, 96))
    >>> print(f"Roof temp: {signature.temperature_map[0:20, :].mean():.1f} K")

Example 2: Industrial facility with hot process equipment
    >>> from eosim.targets import IndustrialTarget
    >>> plant = IndustrialTarget.power_plant(load_percent=80)
    >>> signature = plant.get_signature(aspect_angle_deg=45)
    >>> print(f"Stack temp: {signature.max_temperature:.0f} K")

Example 3: Bridge with solar heating differential
    >>> from eosim.targets import BridgeTarget
    >>> bridge = BridgeTarget.steel_truss(span_m=200, solar_load=0.8)
    >>> signature = bridge.get_signature()
    >>> print(f"Thermal contrast: {signature.thermal_contrast:.1f} K")
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np
from numpy.typing import NDArray

from eosim.targets.base import (
    Target,
    TargetSignature,
    TargetGeometry,
    HotSpot,
    MaterialProperties,
    MaterialType,
)


class BuildingType(Enum):
    """Types of buildings."""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    WAREHOUSE = "warehouse"


class RoofType(Enum):
    """Roof construction types."""
    FLAT = "flat"
    PITCHED = "pitched"
    METAL = "metal"
    GREEN = "green"  # Vegetated roof


@dataclass
class HVACUnit:
    """HVAC exhaust unit configuration.

    Attributes:
        relative_position: Position on roof (0-1)
        power_kw: Cooling/heating power
        running: Whether unit is active
        exhaust_temp_delta_k: Temperature above ambient when running
    """
    relative_position: tuple[float, float] = (0.5, 0.5)
    power_kw: float = 50.0
    running: bool = True
    exhaust_temp_delta_k: float = 15.0


@dataclass
class WindowConfig:
    """Window configuration for buildings.

    Attributes:
        window_fraction: Fraction of wall area that is windows
        interior_temp_k: Interior temperature
        shading: Whether windows have shading (reduces emission)
    """
    window_fraction: float = 0.3
    interior_temp_k: float = 295.0
    shading: bool = False


class BuildingTarget(Target):
    """Building target with realistic thermal signature.

    Models thermal signatures including:
    - Roof heating/cooling
    - HVAC exhaust units
    - Window thermal emission
    - Wall material properties
    - Shadow effects
    """

    def __init__(
        self,
        name: str,
        geometry: TargetGeometry,
        building_type: BuildingType = BuildingType.COMMERCIAL,
        roof_type: RoofType = RoofType.FLAT,
        num_floors: int = 1,
        hvac_units: Optional[list[HVACUnit]] = None,
        window_config: Optional[WindowConfig] = None,
        base_temperature_k: float = 300.0,
        ambient_temperature_k: float = 290.0,
        solar_load: float = 0.5,
    ) -> None:
        """Initialize building target.

        Args:
            name: Building identifier
            geometry: Physical dimensions
            building_type: Type of building
            roof_type: Roof construction
            num_floors: Number of floors
            hvac_units: HVAC unit configurations
            window_config: Window thermal config
            base_temperature_k: Base surface temperature
            ambient_temperature_k: Ambient temperature
            solar_load: Solar irradiance factor (0-1)
        """
        super().__init__(name, geometry, base_temperature_k, ambient_temperature_k)
        self.building_type = building_type
        self.roof_type = roof_type
        self.num_floors = num_floors
        self.hvac_units = hvac_units or []
        self.window_config = window_config or WindowConfig()
        self.solar_load = solar_load

        # Set up materials
        self._materials["roof"] = self._get_roof_material()
        self._materials["wall"] = MaterialProperties.from_material_type(MaterialType.CONCRETE)
        self._materials["window"] = MaterialProperties.from_material_type(MaterialType.GLASS)

        # Add HVAC hot spots
        for i, hvac in enumerate(self.hvac_units):
            if hvac.running:
                self.add_hot_spot(HotSpot(
                    name=f"hvac_{i}",
                    relative_position=hvac.relative_position,
                    relative_size=0.08,
                    temperature_delta_k=hvac.exhaust_temp_delta_k,
                    shape="circle",
                ))

    def _get_roof_material(self) -> MaterialProperties:
        """Get material properties based on roof type."""
        if self.roof_type == RoofType.METAL:
            return MaterialProperties(
                emissivity=0.25,
                reflectance=0.6,
                thermal_mass=450,
                thermal_conductivity=50,
                solar_absorptance=0.4,
            )
        elif self.roof_type == RoofType.GREEN:
            return MaterialProperties.from_material_type(MaterialType.VEGETATION)
        else:  # Flat or pitched (asphalt/tar)
            return MaterialProperties(
                emissivity=0.92,
                reflectance=0.15,
                thermal_mass=920,
                thermal_conductivity=0.75,
                solar_absorptance=0.85,
            )

    def get_signature(
        self,
        resolution: tuple[int, int] = (64, 64),
        aspect_angle_deg: float = 0.0,
        elevation_angle_deg: float = 0.0,
    ) -> TargetSignature:
        """Generate building thermal signature."""
        h, w = resolution

        # Create base temperature map
        roof_material = self._materials["roof"]

        # Roof temperature depends on solar load and material
        solar_heating = self.solar_load * 30 * roof_material.solar_absorptance
        roof_temp = self.ambient_temperature_k + solar_heating

        # Green roofs are cooler due to evapotranspiration
        if self.roof_type == RoofType.GREEN:
            roof_temp -= 10

        # Metal roofs can be hotter or cooler depending on conditions
        if self.roof_type == RoofType.METAL and self.solar_load < 0.3:
            roof_temp = self.ambient_temperature_k - 5  # Radiative cooling at night

        temperature_map = np.full((h, w), roof_temp, dtype=np.float64)
        emissivity_map = np.full((h, w), roof_material.emissivity, dtype=np.float64)

        # Create building footprint mask
        mask = self._create_base_shape(resolution)

        # Apply mask (background is 0)
        temperature_map = np.where(mask, temperature_map, 0)
        emissivity_map = np.where(mask, emissivity_map, 0)

        # Add temperature variation across roof
        # Edges are cooler, center is warmer
        y_grid, x_grid = np.meshgrid(
            np.linspace(-1, 1, h),
            np.linspace(-1, 1, w),
            indexing='ij'
        )
        edge_distance = 1 - np.maximum(np.abs(y_grid), np.abs(x_grid))
        temperature_variation = edge_distance * 3  # Up to 3K warmer in center
        temperature_map = np.where(mask, temperature_map + temperature_variation, 0)

        # Apply HVAC hot spots
        temperature_map = self._apply_hot_spots(temperature_map, roof_temp)

        return TargetSignature(
            temperature_map=temperature_map,
            emissivity_map=emissivity_map,
            geometry=self.geometry,
            aspect_angle_deg=aspect_angle_deg,
            elevation_angle_deg=elevation_angle_deg,
        )

    @classmethod
    def office_building(
        cls,
        floors: int = 3,
        hvac_running: bool = True,
        length_m: float = 40.0,
        width_m: float = 25.0,
    ) -> "BuildingTarget":
        """Create a typical office building."""
        hvac_units = []
        if hvac_running:
            # Multiple rooftop units
            for pos in [(0.3, 0.3), (0.3, 0.7), (0.7, 0.3), (0.7, 0.7)]:
                hvac_units.append(HVACUnit(
                    relative_position=pos,
                    power_kw=30.0,
                    running=True,
                    exhaust_temp_delta_k=12.0,
                ))

        return cls(
            name="office_building",
            geometry=TargetGeometry(
                length_m=length_m,
                width_m=width_m,
                height_m=floors * 3.5,
                shape="box",
            ),
            building_type=BuildingType.COMMERCIAL,
            roof_type=RoofType.FLAT,
            num_floors=floors,
            hvac_units=hvac_units,
            solar_load=0.6,
        )

    @classmethod
    def warehouse(
        cls,
        length_m: float = 100.0,
        width_m: float = 50.0,
    ) -> "BuildingTarget":
        """Create a warehouse with metal roof."""
        return cls(
            name="warehouse",
            geometry=TargetGeometry(
                length_m=length_m,
                width_m=width_m,
                height_m=10.0,
                shape="box",
            ),
            building_type=BuildingType.WAREHOUSE,
            roof_type=RoofType.METAL,
            num_floors=1,
            hvac_units=[HVACUnit(
                relative_position=(0.5, 0.9),
                power_kw=100.0,
                running=True,
                exhaust_temp_delta_k=20.0,
            )],
            solar_load=0.5,
        )

    @classmethod
    def residential_house(
        cls,
        chimney_active: bool = False,
    ) -> "BuildingTarget":
        """Create a residential house."""
        building = cls(
            name="residential_house",
            geometry=TargetGeometry(
                length_m=15.0,
                width_m=12.0,
                height_m=8.0,
                shape="box",
            ),
            building_type=BuildingType.RESIDENTIAL,
            roof_type=RoofType.PITCHED,
            num_floors=2,
            hvac_units=[],
            solar_load=0.5,
        )

        if chimney_active:
            building.add_hot_spot(HotSpot(
                name="chimney",
                relative_position=(0.2, 0.8),
                relative_size=0.05,
                temperature_delta_k=80.0,
                shape="circle",
            ))

        return building


class IndustrialTarget(Target):
    """Industrial facility with process heat signatures.

    Models thermal signatures from:
    - Smokestacks and exhaust
    - Process equipment
    - Cooling towers
    - Storage tanks
    """

    def __init__(
        self,
        name: str,
        geometry: TargetGeometry,
        process_temp_k: float = 400.0,
        stack_temp_k: float = 450.0,
        num_stacks: int = 1,
        cooling_towers: int = 0,
        load_percent: float = 100.0,
        ambient_temperature_k: float = 290.0,
    ) -> None:
        """Initialize industrial target.

        Args:
            name: Facility name
            geometry: Physical dimensions
            process_temp_k: Process area temperature
            stack_temp_k: Stack exhaust temperature
            num_stacks: Number of smokestacks
            cooling_towers: Number of cooling towers
            load_percent: Operating load (0-100)
            ambient_temperature_k: Ambient temperature
        """
        super().__init__(name, geometry, process_temp_k, ambient_temperature_k)
        self.process_temp_k = process_temp_k
        self.stack_temp_k = stack_temp_k
        self.num_stacks = num_stacks
        self.cooling_towers = cooling_towers
        self.load_percent = load_percent

        # Add stack hot spots
        stack_positions = self._compute_stack_positions()
        for i, pos in enumerate(stack_positions):
            self.add_hot_spot(HotSpot(
                name=f"stack_{i}",
                relative_position=pos,
                relative_size=0.04,
                temperature_delta_k=(stack_temp_k - ambient_temperature_k) * (load_percent / 100),
                shape="circle",
            ))

        # Add cooling tower hot spots (warm, not hot)
        for i in range(cooling_towers):
            self.add_hot_spot(HotSpot(
                name=f"cooling_tower_{i}",
                relative_position=(0.7, 0.2 + i * 0.2),
                relative_size=0.1,
                temperature_delta_k=15.0 * (load_percent / 100),
                shape="circle",
            ))

    def _compute_stack_positions(self) -> list[tuple[float, float]]:
        """Compute stack positions based on number."""
        if self.num_stacks == 1:
            return [(0.3, 0.5)]
        elif self.num_stacks == 2:
            return [(0.3, 0.35), (0.3, 0.65)]
        else:
            positions = []
            for i in range(self.num_stacks):
                x = 0.2 + (0.6 * i / (self.num_stacks - 1)) if self.num_stacks > 1 else 0.5
                positions.append((0.3, x))
            return positions

    def get_signature(
        self,
        resolution: tuple[int, int] = (64, 64),
        aspect_angle_deg: float = 0.0,
        elevation_angle_deg: float = 0.0,
    ) -> TargetSignature:
        """Generate industrial facility thermal signature."""
        h, w = resolution

        # Base temperature varies with load
        load_factor = self.load_percent / 100
        base_temp = self.ambient_temperature_k + (self.process_temp_k - self.ambient_temperature_k) * 0.3 * load_factor

        temperature_map = np.full((h, w), base_temp, dtype=np.float64)
        emissivity_map = np.full((h, w), 0.85, dtype=np.float64)

        # Create facility footprint
        mask = self._create_base_shape(resolution)
        temperature_map = np.where(mask, temperature_map, 0)
        emissivity_map = np.where(mask, emissivity_map, 0)

        # Add process heat gradient
        y_grid = np.linspace(0, 1, h)
        process_heat = np.outer(1 - y_grid, np.ones(w)) * 20 * load_factor
        temperature_map = np.where(mask, temperature_map + process_heat, 0)

        # Apply hot spots (stacks, cooling towers)
        temperature_map = self._apply_hot_spots(temperature_map, base_temp)

        return TargetSignature(
            temperature_map=temperature_map,
            emissivity_map=emissivity_map,
            geometry=self.geometry,
            aspect_angle_deg=aspect_angle_deg,
            elevation_angle_deg=elevation_angle_deg,
        )

    @classmethod
    def power_plant(
        cls,
        load_percent: float = 80.0,
    ) -> "IndustrialTarget":
        """Create a power plant."""
        return cls(
            name="power_plant",
            geometry=TargetGeometry(
                length_m=200.0,
                width_m=150.0,
                height_m=60.0,
                shape="box",
            ),
            process_temp_k=500.0,
            stack_temp_k=550.0,
            num_stacks=3,
            cooling_towers=2,
            load_percent=load_percent,
        )

    @classmethod
    def refinery(cls) -> "IndustrialTarget":
        """Create an oil refinery."""
        target = cls(
            name="refinery",
            geometry=TargetGeometry(
                length_m=300.0,
                width_m=200.0,
                height_m=40.0,
                shape="box",
            ),
            process_temp_k=600.0,
            stack_temp_k=700.0,
            num_stacks=5,
            cooling_towers=4,
            load_percent=90.0,
        )

        # Add flare stack
        target.add_hot_spot(HotSpot(
            name="flare",
            relative_position=(0.1, 0.9),
            relative_size=0.03,
            temperature_delta_k=800.0,
            pulsing=True,
            pulse_period_s=2.0,
            pulse_amplitude_k=100.0,
        ))

        return target

    @classmethod
    def factory(
        cls,
        process_type: str = "manufacturing",
    ) -> "IndustrialTarget":
        """Create a factory."""
        process_temps = {
            "manufacturing": 350.0,
            "steel": 800.0,
            "chemical": 450.0,
            "food": 320.0,
        }
        return cls(
            name=f"{process_type}_factory",
            geometry=TargetGeometry(
                length_m=100.0,
                width_m=60.0,
                height_m=15.0,
                shape="box",
            ),
            process_temp_k=process_temps.get(process_type, 350.0),
            stack_temp_k=process_temps.get(process_type, 350.0) + 50,
            num_stacks=2,
            cooling_towers=1,
            load_percent=75.0,
        )


class BridgeTarget(Target):
    """Bridge structure with thermal signature.

    Models:
    - Deck and support structure
    - Material-dependent heating/cooling
    - Traffic-induced heating
    """

    def __init__(
        self,
        name: str,
        geometry: TargetGeometry,
        bridge_type: str = "steel_truss",
        deck_material: MaterialType = MaterialType.CONCRETE,
        solar_load: float = 0.5,
        traffic_heat: float = 0.0,
        ambient_temperature_k: float = 290.0,
    ) -> None:
        """Initialize bridge target.

        Args:
            name: Bridge name
            geometry: Span and width
            bridge_type: "steel_truss", "concrete", "suspension"
            deck_material: Deck surface material
            solar_load: Solar heating factor (0-1)
            traffic_heat: Heat from traffic (0-1)
            ambient_temperature_k: Ambient temperature
        """
        super().__init__(name, geometry, ambient_temperature_k, ambient_temperature_k)
        self.bridge_type = bridge_type
        self.deck_material = deck_material
        self.solar_load = solar_load
        self.traffic_heat = traffic_heat

        self._materials["deck"] = MaterialProperties.from_material_type(deck_material)

        if bridge_type == "steel_truss":
            self._materials["structure"] = MaterialProperties.from_material_type(MaterialType.METAL_BARE)
        else:
            self._materials["structure"] = MaterialProperties.from_material_type(MaterialType.CONCRETE)

    def get_signature(
        self,
        resolution: tuple[int, int] = (32, 128),
        aspect_angle_deg: float = 0.0,
        elevation_angle_deg: float = 0.0,
    ) -> TargetSignature:
        """Generate bridge thermal signature."""
        h, w = resolution

        deck_material = self._materials["deck"]
        structure_material = self._materials["structure"]

        # Deck temperature
        solar_heating = self.solar_load * 25 * deck_material.solar_absorptance
        traffic_heating = self.traffic_heat * 10
        deck_temp = self.ambient_temperature_k + solar_heating + traffic_heating

        temperature_map = np.full((h, w), deck_temp, dtype=np.float64)
        emissivity_map = np.full((h, w), deck_material.emissivity, dtype=np.float64)

        # Add structure (truss elements along edges)
        if self.bridge_type == "steel_truss":
            # Top and bottom edges are cooler steel
            structure_temp = self.ambient_temperature_k + self.solar_load * 15
            edge_height = max(2, h // 6)

            temperature_map[:edge_height, :] = structure_temp
            temperature_map[-edge_height:, :] = structure_temp
            emissivity_map[:edge_height, :] = structure_material.emissivity
            emissivity_map[-edge_height:, :] = structure_material.emissivity

        # Add expansion joints (cooler gaps)
        num_joints = max(1, int(self.geometry.length_m / 50))
        for i in range(1, num_joints + 1):
            joint_x = int(w * i / (num_joints + 1))
            temperature_map[:, joint_x:joint_x+2] = self.ambient_temperature_k - 2
            emissivity_map[:, joint_x:joint_x+2] = 0.95

        return TargetSignature(
            temperature_map=temperature_map,
            emissivity_map=emissivity_map,
            geometry=self.geometry,
            aspect_angle_deg=aspect_angle_deg,
            elevation_angle_deg=elevation_angle_deg,
        )

    @classmethod
    def steel_truss(
        cls,
        span_m: float = 100.0,
        solar_load: float = 0.5,
    ) -> "BridgeTarget":
        """Create a steel truss bridge."""
        return cls(
            name="steel_truss_bridge",
            geometry=TargetGeometry(
                length_m=span_m,
                width_m=15.0,
                height_m=8.0,
                shape="box",
            ),
            bridge_type="steel_truss",
            deck_material=MaterialType.CONCRETE,
            solar_load=solar_load,
        )

    @classmethod
    def concrete_overpass(
        cls,
        span_m: float = 50.0,
    ) -> "BridgeTarget":
        """Create a concrete overpass."""
        return cls(
            name="concrete_overpass",
            geometry=TargetGeometry(
                length_m=span_m,
                width_m=12.0,
                height_m=5.0,
                shape="box",
            ),
            bridge_type="concrete",
            deck_material=MaterialType.ASPHALT,
            solar_load=0.6,
        )


class StorageTankTarget(Target):
    """Storage tank (fuel, chemical, water) target.

    Models cylindrical tanks with contents-dependent thermal signature.
    """

    def __init__(
        self,
        name: str,
        diameter_m: float = 20.0,
        height_m: float = 15.0,
        contents: str = "fuel",
        fill_level: float = 0.7,
        contents_temp_k: float = 300.0,
        ambient_temperature_k: float = 290.0,
        insulated: bool = False,
    ) -> None:
        """Initialize storage tank.

        Args:
            name: Tank identifier
            diameter_m: Tank diameter
            height_m: Tank height
            contents: "fuel", "chemical", "water", "empty"
            fill_level: Fill level (0-1)
            contents_temp_k: Temperature of contents
            ambient_temperature_k: Ambient temperature
            insulated: Whether tank is insulated
        """
        geometry = TargetGeometry(
            length_m=diameter_m,
            width_m=diameter_m,
            height_m=height_m,
            shape="cylinder",
        )
        super().__init__(name, geometry, contents_temp_k, ambient_temperature_k)
        self.contents = contents
        self.fill_level = fill_level
        self.contents_temp_k = contents_temp_k
        self.insulated = insulated

    def get_signature(
        self,
        resolution: tuple[int, int] = (64, 64),
        aspect_angle_deg: float = 0.0,
        elevation_angle_deg: float = 0.0,
    ) -> TargetSignature:
        """Generate tank thermal signature (top-down view)."""
        h, w = resolution

        # Circular tank shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r = min(h, w) // 2 - 2
        mask = (yy - cy)**2 + (xx - cx)**2 <= r**2

        # Tank roof temperature
        if self.contents == "empty":
            roof_temp = self.ambient_temperature_k + 5  # Slightly warm from solar
        elif self.insulated:
            roof_temp = self.ambient_temperature_k + 2
        else:
            # Tank roof reflects contents temperature
            roof_temp = self.ambient_temperature_k + (self.contents_temp_k - self.ambient_temperature_k) * 0.5 * self.fill_level

        temperature_map = np.where(mask, roof_temp, 0.0)
        emissivity_map = np.where(mask, 0.9, 0.0)

        # Add floating roof gap (warm ring near edge for some tanks)
        if self.contents == "fuel" and self.fill_level > 0.1:
            inner_r = int(r * 0.9)
            ring_mask = ((yy - cy)**2 + (xx - cx)**2 <= r**2) & ((yy - cy)**2 + (xx - cx)**2 >= inner_r**2)
            temperature_map = np.where(ring_mask, roof_temp + 5, temperature_map)

        return TargetSignature(
            temperature_map=temperature_map,
            emissivity_map=emissivity_map,
            geometry=self.geometry,
            aspect_angle_deg=aspect_angle_deg,
            elevation_angle_deg=elevation_angle_deg,
        )

    @classmethod
    def fuel_tank(
        cls,
        diameter_m: float = 30.0,
        fill_level: float = 0.8,
    ) -> "StorageTankTarget":
        """Create a fuel storage tank."""
        return cls(
            name="fuel_tank",
            diameter_m=diameter_m,
            height_m=15.0,
            contents="fuel",
            fill_level=fill_level,
            contents_temp_k=295.0,  # Near ambient
        )

    @classmethod
    def chemical_tank(
        cls,
        contents_temp_k: float = 350.0,
        insulated: bool = True,
    ) -> "StorageTankTarget":
        """Create a chemical storage tank."""
        return cls(
            name="chemical_tank",
            diameter_m=10.0,
            height_m=12.0,
            contents="chemical",
            fill_level=0.6,
            contents_temp_k=contents_temp_k,
            insulated=insulated,
        )
