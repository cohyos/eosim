"""
EOSIM Natural Targets - Animals, Vegetation, and Natural Features.

Provides target models for wildlife, vegetation, water bodies, and
other natural features with realistic thermal signatures.

Example 1: Wildlife thermal detection
    >>> from eosim.targets import AnimalTarget
    >>> deer = AnimalTarget.deer(activity="resting")
    >>> signature = deer.get_signature(resolution=(32, 48))
    >>> print(f"Body temp: {signature.mean_temperature:.1f} K")

Example 2: Forest canopy with variable density
    >>> from eosim.targets import VegetationTarget, TargetRenderer
    >>> forest = VegetationTarget.deciduous_forest(density=0.8, solar_load=0.6)
    >>> renderer = TargetRenderer(resolution=(480, 640), gsd_m=1.0)
    >>> temp_map, emis_map = renderer.render(forest, position=(240, 320))

Example 3: Water body with thermal stratification
    >>> from eosim.targets import WaterBodyTarget
    >>> lake = WaterBodyTarget.lake(surface_temp_k=288.0, depth_m=5.0)
    >>> signature = lake.get_signature()
    >>> print(f"Shore contrast: {signature.thermal_contrast:.1f} K")
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


class AnimalType(Enum):
    """Types of animals."""
    MAMMAL_LARGE = "mammal_large"  # Deer, elk, cattle
    MAMMAL_MEDIUM = "mammal_medium"  # Dogs, coyotes
    MAMMAL_SMALL = "mammal_small"  # Rabbits, rodents
    BIRD_LARGE = "bird_large"  # Geese, turkeys
    BIRD_SMALL = "bird_small"  # Songbirds
    REPTILE = "reptile"


class VegetationType(Enum):
    """Types of vegetation."""
    DECIDUOUS = "deciduous"
    CONIFEROUS = "coniferous"
    GRASSLAND = "grassland"
    SHRUB = "shrub"
    CROP = "crop"


@dataclass
class AnimalPhysiology:
    """Physiological properties affecting thermal signature.

    Attributes:
        core_temp_k: Core body temperature
        metabolic_rate_w: Base metabolic heat production
        fur_insulation: Insulation factor (higher = cooler surface)
        activity_multiplier: Activity-based metabolic increase
    """
    core_temp_k: float = 311.0  # ~38C
    metabolic_rate_w: float = 100.0
    fur_insulation: float = 0.5
    activity_multiplier: float = 1.0


class AnimalTarget(Target):
    """Animal target with physiological thermal modeling.

    Models:
    - Core body temperature
    - Metabolic heat production
    - Fur/feather insulation
    - Activity-dependent temperature
    - Breathing/exhale signatures
    """

    def __init__(
        self,
        name: str,
        geometry: TargetGeometry,
        animal_type: AnimalType = AnimalType.MAMMAL_LARGE,
        physiology: Optional[AnimalPhysiology] = None,
        activity: str = "resting",
        ambient_temperature_k: float = 290.0,
    ) -> None:
        """Initialize animal target.

        Args:
            name: Animal identifier
            geometry: Physical dimensions
            animal_type: Type of animal
            physiology: Physiological parameters
            activity: Current activity level
            ambient_temperature_k: Ambient temperature
        """
        self.animal_type = animal_type
        self.physiology = physiology or self._default_physiology()
        self.activity = activity

        # Compute effective surface temperature
        surface_temp = self._compute_surface_temperature(ambient_temperature_k)
        super().__init__(name, geometry, surface_temp, ambient_temperature_k)

        # Add head hot spot (warmer)
        self.add_hot_spot(HotSpot(
            name="head",
            relative_position=(0.15, 0.5),
            relative_size=0.15,
            temperature_delta_k=3.0,
            shape="ellipse",
        ))

        # Add breathing signature for mammals
        if animal_type in (AnimalType.MAMMAL_LARGE, AnimalType.MAMMAL_MEDIUM):
            self.add_hot_spot(HotSpot(
                name="breath",
                relative_position=(0.05, 0.5),
                relative_size=0.05,
                temperature_delta_k=8.0,
                shape="circle",
                pulsing=True,
                pulse_period_s=3.0,  # Breathing rate
                pulse_amplitude_k=4.0,
            ))

    def _default_physiology(self) -> AnimalPhysiology:
        """Get default physiology for animal type."""
        physio_map = {
            AnimalType.MAMMAL_LARGE: AnimalPhysiology(311.0, 200.0, 0.6, 1.0),
            AnimalType.MAMMAL_MEDIUM: AnimalPhysiology(311.5, 50.0, 0.5, 1.0),
            AnimalType.MAMMAL_SMALL: AnimalPhysiology(312.0, 5.0, 0.4, 1.0),
            AnimalType.BIRD_LARGE: AnimalPhysiology(314.0, 30.0, 0.7, 1.0),
            AnimalType.BIRD_SMALL: AnimalPhysiology(315.0, 2.0, 0.6, 1.0),
            AnimalType.REPTILE: AnimalPhysiology(295.0, 1.0, 0.1, 1.0),  # Ectotherm
        }
        return physio_map.get(self.animal_type, AnimalPhysiology())

    def _compute_surface_temperature(self, ambient_k: float) -> float:
        """Compute surface temperature based on physiology."""
        activity_factors = {
            "resting": 1.0,
            "walking": 1.3,
            "running": 2.0,
            "sleeping": 0.9,
        }
        activity_mult = activity_factors.get(self.activity, 1.0)

        # Surface temp is between core and ambient, affected by insulation
        core = self.physiology.core_temp_k
        insulation = self.physiology.fur_insulation

        # Higher insulation = cooler surface
        surface = ambient_k + (core - ambient_k) * (1 - insulation * 0.8)

        # Activity increases surface temperature
        surface += (activity_mult - 1) * 5

        return surface

    def get_signature(
        self,
        resolution: tuple[int, int] = (48, 64),
        aspect_angle_deg: float = 0.0,
        elevation_angle_deg: float = 0.0,
    ) -> TargetSignature:
        """Generate animal thermal signature."""
        h, w = resolution

        # Create body shape (elliptical)
        cy, cx = h // 2, w // 2
        ry, rx = h // 2 - 2, w // 2 - 2
        yy, xx = np.ogrid[:h, :w]
        mask = ((yy - cy) / max(ry, 1))**2 + ((xx - cx) / max(rx, 1))**2 <= 1

        # Base temperature map
        temperature_map = np.where(mask, self.base_temperature_k, 0.0)
        emissivity_map = np.where(mask, 0.98, 0.0)  # Skin/fur emissivity

        # Add body temperature gradient (warmer in center)
        dist_from_center = np.sqrt(((yy - cy) / max(ry, 1))**2 + ((xx - cx) / max(rx, 1))**2)
        temp_gradient = (1 - dist_from_center) * 2  # Up to 2K warmer in center
        temperature_map = np.where(mask, temperature_map + temp_gradient, 0)

        # Apply hot spots
        temperature_map = self._apply_hot_spots(temperature_map, self.base_temperature_k)

        # Add legs (for quadrupeds, seen from above)
        if self.animal_type in (AnimalType.MAMMAL_LARGE, AnimalType.MAMMAL_MEDIUM):
            leg_temp = self.base_temperature_k - 3  # Legs are cooler
            # Front legs
            temperature_map[h//4:h//3, w//4:w//4+3] = leg_temp
            temperature_map[h//4:h//3, 3*w//4-3:3*w//4] = leg_temp
            # Rear legs
            temperature_map[2*h//3:3*h//4, w//4:w//4+3] = leg_temp
            temperature_map[2*h//3:3*h//4, 3*w//4-3:3*w//4] = leg_temp

        return TargetSignature(
            temperature_map=temperature_map,
            emissivity_map=emissivity_map,
            geometry=self.geometry,
            aspect_angle_deg=aspect_angle_deg,
            elevation_angle_deg=elevation_angle_deg,
        )

    @classmethod
    def deer(
        cls,
        activity: str = "walking",
    ) -> "AnimalTarget":
        """Create a deer target."""
        return cls(
            name="deer",
            geometry=TargetGeometry(
                length_m=1.8,
                width_m=0.6,
                height_m=1.2,
                shape="ellipsoid",
            ),
            animal_type=AnimalType.MAMMAL_LARGE,
            physiology=AnimalPhysiology(
                core_temp_k=311.5,
                metabolic_rate_w=150.0,
                fur_insulation=0.55,
            ),
            activity=activity,
        )

    @classmethod
    def cattle(
        cls,
        activity: str = "resting",
    ) -> "AnimalTarget":
        """Create a cattle/cow target."""
        return cls(
            name="cattle",
            geometry=TargetGeometry(
                length_m=2.5,
                width_m=1.0,
                height_m=1.5,
                shape="ellipsoid",
            ),
            animal_type=AnimalType.MAMMAL_LARGE,
            physiology=AnimalPhysiology(
                core_temp_k=311.5,
                metabolic_rate_w=300.0,
                fur_insulation=0.45,
            ),
            activity=activity,
        )

    @classmethod
    def dog(
        cls,
        activity: str = "walking",
    ) -> "AnimalTarget":
        """Create a dog target."""
        return cls(
            name="dog",
            geometry=TargetGeometry(
                length_m=0.8,
                width_m=0.3,
                height_m=0.5,
                shape="ellipsoid",
            ),
            animal_type=AnimalType.MAMMAL_MEDIUM,
            activity=activity,
        )

    @classmethod
    def bird(
        cls,
        size: str = "large",
    ) -> "AnimalTarget":
        """Create a bird target."""
        if size == "large":
            return cls(
                name="bird_large",
                geometry=TargetGeometry(
                    length_m=0.6,
                    width_m=0.25,
                    height_m=0.3,
                    shape="ellipsoid",
                ),
                animal_type=AnimalType.BIRD_LARGE,
            )
        return cls(
            name="bird_small",
            geometry=TargetGeometry(
                length_m=0.15,
                width_m=0.08,
                height_m=0.1,
                shape="ellipsoid",
            ),
            animal_type=AnimalType.BIRD_SMALL,
        )


class VegetationTarget(Target):
    """Vegetation target with canopy thermal modeling.

    Models:
    - Canopy temperature vs ambient
    - Evapotranspiration cooling
    - Solar heating
    - Density-dependent signature
    """

    def __init__(
        self,
        name: str,
        geometry: TargetGeometry,
        vegetation_type: VegetationType = VegetationType.DECIDUOUS,
        canopy_density: float = 0.7,
        leaf_area_index: float = 4.0,
        solar_load: float = 0.5,
        soil_moisture: float = 0.5,
        ambient_temperature_k: float = 290.0,
    ) -> None:
        """Initialize vegetation target.

        Args:
            name: Vegetation identifier
            geometry: Physical dimensions
            vegetation_type: Type of vegetation
            canopy_density: Canopy closure (0-1)
            leaf_area_index: LAI for transpiration
            solar_load: Solar irradiance factor
            soil_moisture: Soil moisture affecting transpiration
            ambient_temperature_k: Ambient temperature
        """
        # Vegetation temp is typically cooler than air due to transpiration
        canopy_temp = self._compute_canopy_temperature(
            ambient_temperature_k, solar_load, soil_moisture, leaf_area_index
        )
        super().__init__(name, geometry, canopy_temp, ambient_temperature_k)

        self.vegetation_type = vegetation_type
        self.canopy_density = canopy_density
        self.leaf_area_index = leaf_area_index
        self.solar_load = solar_load
        self.soil_moisture = soil_moisture

        self._materials["canopy"] = MaterialProperties.from_material_type(MaterialType.VEGETATION)
        self._materials["soil"] = MaterialProperties.from_material_type(MaterialType.SOIL)

    def _compute_canopy_temperature(
        self,
        ambient_k: float,
        solar_load: float,
        soil_moisture: float,
        lai: float,
    ) -> float:
        """Compute canopy temperature accounting for transpiration."""
        # Solar heating
        solar_heating = solar_load * 15

        # Transpiration cooling (depends on moisture and LAI)
        transpiration_cooling = soil_moisture * lai * 2

        return ambient_k + solar_heating - transpiration_cooling

    def get_signature(
        self,
        resolution: tuple[int, int] = (64, 64),
        aspect_angle_deg: float = 0.0,
        elevation_angle_deg: float = 0.0,
    ) -> TargetSignature:
        """Generate vegetation thermal signature."""
        h, w = resolution

        canopy_material = self._materials["canopy"]
        soil_material = self._materials["soil"]

        # Create base canopy temperature
        temperature_map = np.full((h, w), self.base_temperature_k, dtype=np.float64)
        emissivity_map = np.full((h, w), canopy_material.emissivity, dtype=np.float64)

        # Add canopy gaps showing warmer soil/ground
        if self.canopy_density < 1.0:
            gap_fraction = 1.0 - self.canopy_density
            rng = np.random.default_rng(42)
            gap_mask = rng.random((h, w)) > self.canopy_density

            # Soil is typically warmer than canopy in daytime
            soil_temp = self.ambient_temperature_k + self.solar_load * 20
            temperature_map = np.where(gap_mask, soil_temp, temperature_map)
            emissivity_map = np.where(gap_mask, soil_material.emissivity, emissivity_map)

        # Add natural temperature variation
        noise = np.random.default_rng(123).normal(0, 1.5, (h, w))
        temperature_map += noise

        # Smooth the result slightly
        from scipy.ndimage import gaussian_filter
        temperature_map = gaussian_filter(temperature_map, sigma=1.0)

        return TargetSignature(
            temperature_map=temperature_map,
            emissivity_map=emissivity_map,
            geometry=self.geometry,
            aspect_angle_deg=aspect_angle_deg,
            elevation_angle_deg=elevation_angle_deg,
        )

    @classmethod
    def deciduous_forest(
        cls,
        density: float = 0.8,
        solar_load: float = 0.5,
        size_m: float = 100.0,
    ) -> "VegetationTarget":
        """Create deciduous forest patch."""
        return cls(
            name="deciduous_forest",
            geometry=TargetGeometry(
                length_m=size_m,
                width_m=size_m,
                height_m=20.0,
                shape="box",
            ),
            vegetation_type=VegetationType.DECIDUOUS,
            canopy_density=density,
            leaf_area_index=5.0,
            solar_load=solar_load,
            soil_moisture=0.6,
        )

    @classmethod
    def coniferous_forest(
        cls,
        density: float = 0.9,
    ) -> "VegetationTarget":
        """Create coniferous forest patch."""
        return cls(
            name="coniferous_forest",
            geometry=TargetGeometry(
                length_m=100.0,
                width_m=100.0,
                height_m=25.0,
                shape="box",
            ),
            vegetation_type=VegetationType.CONIFEROUS,
            canopy_density=density,
            leaf_area_index=7.0,
            solar_load=0.4,  # Less solar penetration
            soil_moisture=0.5,
        )

    @classmethod
    def grassland(
        cls,
        solar_load: float = 0.7,
    ) -> "VegetationTarget":
        """Create grassland patch."""
        return cls(
            name="grassland",
            geometry=TargetGeometry(
                length_m=200.0,
                width_m=200.0,
                height_m=0.5,
                shape="box",
            ),
            vegetation_type=VegetationType.GRASSLAND,
            canopy_density=0.95,
            leaf_area_index=2.0,
            solar_load=solar_load,
            soil_moisture=0.4,
        )

    @classmethod
    def crop_field(
        cls,
        crop_type: str = "corn",
    ) -> "VegetationTarget":
        """Create agricultural crop field."""
        lai_map = {"corn": 4.0, "wheat": 3.0, "soybeans": 5.0}
        height_map = {"corn": 2.5, "wheat": 1.0, "soybeans": 1.2}

        return cls(
            name=f"{crop_type}_field",
            geometry=TargetGeometry(
                length_m=200.0,
                width_m=200.0,
                height_m=height_map.get(crop_type, 1.5),
                shape="box",
            ),
            vegetation_type=VegetationType.CROP,
            canopy_density=0.85,
            leaf_area_index=lai_map.get(crop_type, 3.0),
            solar_load=0.7,
            soil_moisture=0.6,
        )


class WaterBodyTarget(Target):
    """Water body target with surface thermal modeling.

    Models:
    - Surface temperature
    - Shore/edge effects
    - Wind-driven mixing
    - Diurnal temperature variation
    """

    def __init__(
        self,
        name: str,
        geometry: TargetGeometry,
        surface_temp_k: float = 290.0,
        depth_m: float = 2.0,
        shore_temp_k: Optional[float] = None,
        wind_speed_mps: float = 2.0,
        ambient_temperature_k: float = 290.0,
    ) -> None:
        """Initialize water body.

        Args:
            name: Water body name
            geometry: Surface dimensions
            surface_temp_k: Water surface temperature
            depth_m: Average depth
            shore_temp_k: Temperature at shore (None = use gradient)
            wind_speed_mps: Wind speed affecting mixing
            ambient_temperature_k: Air temperature
        """
        super().__init__(name, geometry, surface_temp_k, ambient_temperature_k)
        self.surface_temp_k = surface_temp_k
        self.depth_m = depth_m
        self.shore_temp_k = shore_temp_k or (ambient_temperature_k + 2)
        self.wind_speed_mps = wind_speed_mps

        self._materials["water"] = MaterialProperties.from_material_type(MaterialType.WATER)

    def get_signature(
        self,
        resolution: tuple[int, int] = (64, 64),
        aspect_angle_deg: float = 0.0,
        elevation_angle_deg: float = 0.0,
    ) -> TargetSignature:
        """Generate water body thermal signature."""
        h, w = resolution
        water_material = self._materials["water"]

        # Create elliptical water body
        cy, cx = h // 2, w // 2
        ry, rx = h // 2 - 2, w // 2 - 2
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt(((yy - cy) / max(ry, 1))**2 + ((xx - cx) / max(rx, 1))**2)
        mask = dist <= 1.0

        # Temperature gradient from center to shore
        shore_gradient = dist * (self.shore_temp_k - self.surface_temp_k)
        temperature_map = np.where(mask, self.surface_temp_k + shore_gradient, 0.0)

        # Add surface ripple variation (wind-dependent)
        ripple_amplitude = self.wind_speed_mps * 0.3  # More wind = more variation
        ripples = np.sin(xx * 0.5) * np.cos(yy * 0.3) * ripple_amplitude
        temperature_map = np.where(mask, temperature_map + ripples, 0.0)

        emissivity_map = np.where(mask, water_material.emissivity, 0.0)

        return TargetSignature(
            temperature_map=temperature_map,
            emissivity_map=emissivity_map,
            geometry=self.geometry,
            aspect_angle_deg=aspect_angle_deg,
            elevation_angle_deg=elevation_angle_deg,
        )

    @classmethod
    def lake(
        cls,
        surface_temp_k: float = 288.0,
        depth_m: float = 10.0,
        size_m: float = 500.0,
    ) -> "WaterBodyTarget":
        """Create a lake."""
        return cls(
            name="lake",
            geometry=TargetGeometry(
                length_m=size_m,
                width_m=size_m * 0.7,
                height_m=0.0,
                shape="ellipsoid",
            ),
            surface_temp_k=surface_temp_k,
            depth_m=depth_m,
            wind_speed_mps=3.0,
        )

    @classmethod
    def pond(
        cls,
        surface_temp_k: float = 292.0,
    ) -> "WaterBodyTarget":
        """Create a small pond."""
        return cls(
            name="pond",
            geometry=TargetGeometry(
                length_m=30.0,
                width_m=20.0,
                height_m=0.0,
                shape="ellipsoid",
            ),
            surface_temp_k=surface_temp_k,
            depth_m=2.0,
            wind_speed_mps=1.0,
        )

    @classmethod
    def river(
        cls,
        width_m: float = 50.0,
        length_m: float = 500.0,
        flow_temp_k: float = 286.0,
    ) -> "WaterBodyTarget":
        """Create a river segment."""
        return cls(
            name="river",
            geometry=TargetGeometry(
                length_m=length_m,
                width_m=width_m,
                height_m=0.0,
                shape="box",
            ),
            surface_temp_k=flow_temp_k,
            depth_m=3.0,
            wind_speed_mps=0.5,  # Flow reduces effective wind
        )


class TerrainTarget(Target):
    """Terrain features with thermal signatures.

    Models rocks, bare soil, sand, and other ground features.
    """

    def __init__(
        self,
        name: str,
        geometry: TargetGeometry,
        terrain_type: str = "soil",
        solar_load: float = 0.6,
        moisture: float = 0.3,
        ambient_temperature_k: float = 290.0,
    ) -> None:
        """Initialize terrain target.

        Args:
            name: Terrain identifier
            geometry: Dimensions
            terrain_type: "soil", "rock", "sand", "gravel"
            solar_load: Solar irradiance factor
            moisture: Soil moisture (affects cooling)
            ambient_temperature_k: Ambient temperature
        """
        material_map = {
            "soil": MaterialType.SOIL,
            "rock": MaterialType.CONCRETE,  # Similar thermal properties
            "sand": MaterialType.SOIL,
            "gravel": MaterialType.CONCRETE,
        }
        mat_type = material_map.get(terrain_type, MaterialType.SOIL)
        material = MaterialProperties.from_material_type(mat_type)

        # Compute surface temperature
        solar_heating = solar_load * 35 * material.solar_absorptance
        evap_cooling = moisture * 8
        surface_temp = ambient_temperature_k + solar_heating - evap_cooling

        super().__init__(name, geometry, surface_temp, ambient_temperature_k)
        self.terrain_type = terrain_type
        self.solar_load = solar_load
        self.moisture = moisture
        self._materials["surface"] = material

    def get_signature(
        self,
        resolution: tuple[int, int] = (64, 64),
        aspect_angle_deg: float = 0.0,
        elevation_angle_deg: float = 0.0,
    ) -> TargetSignature:
        """Generate terrain thermal signature."""
        h, w = resolution
        material = self._materials["surface"]

        temperature_map = np.full((h, w), self.base_temperature_k, dtype=np.float64)
        emissivity_map = np.full((h, w), material.emissivity, dtype=np.float64)

        # Add natural variation
        rng = np.random.default_rng(456)
        variation = rng.normal(0, 2.0, (h, w))

        # Spatial correlation
        from scipy.ndimage import gaussian_filter
        variation = gaussian_filter(variation, sigma=3.0)
        temperature_map += variation

        return TargetSignature(
            temperature_map=temperature_map,
            emissivity_map=emissivity_map,
            geometry=self.geometry,
            aspect_angle_deg=aspect_angle_deg,
            elevation_angle_deg=elevation_angle_deg,
        )

    @classmethod
    def bare_soil(
        cls,
        solar_load: float = 0.7,
        moisture: float = 0.2,
    ) -> "TerrainTarget":
        """Create bare soil patch."""
        return cls(
            name="bare_soil",
            geometry=TargetGeometry(
                length_m=50.0,
                width_m=50.0,
                height_m=0.0,
                shape="box",
            ),
            terrain_type="soil",
            solar_load=solar_load,
            moisture=moisture,
        )

    @classmethod
    def rock_outcrop(cls) -> "TerrainTarget":
        """Create rocky outcrop."""
        return cls(
            name="rock_outcrop",
            geometry=TargetGeometry(
                length_m=20.0,
                width_m=15.0,
                height_m=3.0,
                shape="custom",
            ),
            terrain_type="rock",
            solar_load=0.8,
            moisture=0.0,
        )

    @classmethod
    def sandy_area(
        cls,
        solar_load: float = 0.9,
    ) -> "TerrainTarget":
        """Create sandy terrain."""
        return cls(
            name="sandy_area",
            geometry=TargetGeometry(
                length_m=100.0,
                width_m=100.0,
                height_m=0.0,
                shape="box",
            ),
            terrain_type="sand",
            solar_load=solar_load,
            moisture=0.05,
        )
