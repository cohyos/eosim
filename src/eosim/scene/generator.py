"""
EOSIM Scene Generator (Stage C).

Provides tools for generating complete simulation scenes with targets,
backgrounds, atmospheric effects, and environmental conditions.

Example 1: Create a simple scene with a vehicle on a road
    >>> from eosim.scene import SceneGenerator, SceneConfig
    >>> from eosim.targets import VehicleTarget
    >>> config = SceneConfig(
    ...     resolution=(480, 640),
    ...     gsd_m=0.5,
    ...     sensor_type="lwir",
    ... )
    >>> generator = SceneGenerator(config)
    >>> vehicle = VehicleTarget.sedan(engine_state="running")
    >>> scene = generator.add_target(vehicle, position=(240, 320))
    >>> result = generator.generate()
    >>> print(f"Scene shape: {result.temperature_map.shape}")

Example 2: Generate a multi-target military scenario
    >>> from eosim.scene import create_military_scenario
    >>> scene = create_military_scenario(
    ...     num_vehicles=5,
    ...     num_people=10,
    ...     terrain_type="desert",
    ...     time_of_day="midday",
    ... )
    >>> result = scene.generate()

Example 3: Create a scene from a template with randomization
    >>> from eosim.scene import SceneTemplate, RandomizationConfig
    >>> template = SceneTemplate.urban_environment()
    >>> randomizer = RandomizationConfig(
    ...     position_jitter_m=10,
    ...     temperature_jitter_k=5,
    ...     num_targets_range=(3, 8),
    ... )
    >>> scenes = template.generate_batch(count=10, randomizer=randomizer)
"""

from dataclasses import dataclass, field
from typing import Optional, Union
from enum import Enum
import numpy as np
from numpy.typing import NDArray

from eosim.targets.base import Target, TargetRenderer, TargetGroup, TargetSignature


class TimeOfDay(Enum):
    """Time of day presets."""
    DAWN = "dawn"
    MORNING = "morning"
    MIDDAY = "midday"
    AFTERNOON = "afternoon"
    DUSK = "dusk"
    NIGHT = "night"


class TerrainType(Enum):
    """Terrain type presets."""
    GRASS = "grass"
    DESERT = "desert"
    URBAN = "urban"
    FOREST = "forest"
    WATER = "water"
    SNOW = "snow"
    ASPHALT = "asphalt"


class WeatherCondition(Enum):
    """Weather condition presets."""
    CLEAR = "clear"
    PARTLY_CLOUDY = "partly_cloudy"
    OVERCAST = "overcast"
    RAIN = "rain"
    FOG = "fog"


@dataclass
class EnvironmentConfig:
    """Environmental conditions for scene generation.

    Attributes:
        time_of_day: Time of day preset or solar_elevation_deg
        solar_elevation_deg: Sun elevation angle (overrides time_of_day)
        solar_azimuth_deg: Sun azimuth angle
        ambient_temperature_k: Ambient air temperature
        weather: Weather condition
        wind_speed_mps: Wind speed affecting convection
        humidity: Relative humidity (0-1)
    """
    time_of_day: TimeOfDay = TimeOfDay.MIDDAY
    solar_elevation_deg: Optional[float] = None
    solar_azimuth_deg: float = 180.0
    ambient_temperature_k: float = 295.0
    weather: WeatherCondition = WeatherCondition.CLEAR
    wind_speed_mps: float = 2.0
    humidity: float = 0.5

    def __post_init__(self):
        """Set solar elevation from time of day if not specified."""
        if self.solar_elevation_deg is None:
            time_to_elevation = {
                TimeOfDay.DAWN: 5.0,
                TimeOfDay.MORNING: 30.0,
                TimeOfDay.MIDDAY: 60.0,
                TimeOfDay.AFTERNOON: 45.0,
                TimeOfDay.DUSK: 5.0,
                TimeOfDay.NIGHT: -15.0,
            }
            self.solar_elevation_deg = time_to_elevation.get(self.time_of_day, 45.0)


@dataclass
class BackgroundConfig:
    """Background configuration for scene generation.

    Attributes:
        terrain_type: Type of terrain
        base_temperature_k: Background temperature (None = compute from environment)
        emissivity: Background emissivity
        temperature_variation_k: Random temperature variation amplitude
        texture_scale: Scale of spatial texture variation
    """
    terrain_type: TerrainType = TerrainType.GRASS
    base_temperature_k: Optional[float] = None
    emissivity: float = 0.95
    temperature_variation_k: float = 2.0
    texture_scale: float = 10.0


@dataclass
class SceneConfig:
    """Configuration for scene generation.

    Attributes:
        resolution: Output resolution (height, width)
        gsd_m: Ground sample distance in meters
        sensor_type: Target sensor type
        environment: Environmental conditions
        background: Background configuration
        random_seed: Random seed for reproducibility
    """
    resolution: tuple[int, int] = (480, 640)
    gsd_m: float = 0.5
    sensor_type: str = "lwir"
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    random_seed: Optional[int] = None


@dataclass
class SceneResult:
    """Result of scene generation.

    Attributes:
        temperature_map: 2D temperature distribution [K]
        emissivity_map: 2D emissivity distribution [0-1]
        target_mask: Binary mask of target locations
        target_labels: Integer labels for each target
        metadata: Additional scene information
    """
    temperature_map: NDArray
    emissivity_map: NDArray
    target_mask: NDArray
    target_labels: NDArray
    metadata: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.temperature_map.shape


class SceneGenerator:
    """Generates complete thermal scenes with targets and backgrounds.

    Combines targets, backgrounds, and environmental effects into
    a single simulation scene.
    """

    # Terrain thermal properties (temperature above ambient, emissivity)
    _TERRAIN_PROPERTIES = {
        TerrainType.GRASS: (5.0, 0.96),
        TerrainType.DESERT: (25.0, 0.92),
        TerrainType.URBAN: (15.0, 0.93),
        TerrainType.FOREST: (2.0, 0.97),
        TerrainType.WATER: (-3.0, 0.96),
        TerrainType.SNOW: (-10.0, 0.99),
        TerrainType.ASPHALT: (20.0, 0.93),
    }

    def __init__(self, config: Optional[SceneConfig] = None) -> None:
        """Initialize scene generator.

        Args:
            config: Scene configuration
        """
        self.config = config or SceneConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

        # Target storage
        self._targets: list[tuple[Target, tuple[int, int], float]] = []
        self._target_groups: list[TargetGroup] = []

        # Create target renderer
        self._renderer = TargetRenderer(
            resolution=self.config.resolution,
            gsd_m=self.config.gsd_m,
            background_temperature_k=self._compute_background_temp(),
            background_emissivity=self.config.background.emissivity,
        )

    def _compute_background_temp(self) -> float:
        """Compute background temperature from environment."""
        if self.config.background.base_temperature_k is not None:
            return self.config.background.base_temperature_k

        ambient = self.config.environment.ambient_temperature_k
        terrain_props = self._TERRAIN_PROPERTIES.get(
            self.config.background.terrain_type,
            (5.0, 0.95)
        )

        # Solar heating based on elevation
        solar_elev = self.config.environment.solar_elevation_deg or 45.0
        solar_factor = max(0, np.sin(np.radians(solar_elev)))

        # Wind cooling
        wind_cooling = self.config.environment.wind_speed_mps * 0.5

        base_delta = terrain_props[0]
        return ambient + base_delta * solar_factor - wind_cooling

    def add_target(
        self,
        target: Target,
        position: tuple[int, int],
        aspect_angle_deg: float = 0.0,
    ) -> "SceneGenerator":
        """Add a target to the scene.

        Args:
            target: Target to add
            position: (y, x) position in pixels
            aspect_angle_deg: Viewing angle

        Returns:
            Self for method chaining
        """
        self._targets.append((target, position, aspect_angle_deg))
        return self

    def add_target_group(self, group: TargetGroup) -> "SceneGenerator":
        """Add a target group to the scene.

        Args:
            group: TargetGroup to add

        Returns:
            Self for method chaining
        """
        self._target_groups.append(group)
        return self

    def add_random_targets(
        self,
        targets: list[Target],
        area: Optional[tuple[int, int, int, int]] = None,
        min_separation: int = 30,
    ) -> "SceneGenerator":
        """Add targets at random positions.

        Args:
            targets: List of targets to add
            area: (y_min, x_min, y_max, x_max) placement area
            min_separation: Minimum separation in pixels

        Returns:
            Self for method chaining
        """
        h, w = self.config.resolution
        if area is None:
            margin = int(min(h, w) * 0.1)
            area = (margin, margin, h - margin, w - margin)

        group = TargetGroup(targets)
        group.randomize_positions(area, min_separation)
        self._target_groups.append(group)
        return self

    def generate(self) -> SceneResult:
        """Generate the complete scene.

        Returns:
            SceneResult with temperature and emissivity maps
        """
        h, w = self.config.resolution

        # Generate background
        temp_map, emis_map = self._generate_background()

        # Create target mask and labels
        target_mask = np.zeros((h, w), dtype=bool)
        target_labels = np.zeros((h, w), dtype=np.int32)
        label_counter = 1

        # Render individual targets
        for target, position, aspect in self._targets:
            self._render_target(
                temp_map, emis_map, target_mask, target_labels,
                target, position, aspect, label_counter
            )
            label_counter += 1

        # Render target groups
        for group in self._target_groups:
            for target, position, aspect in group.get_render_list():
                self._render_target(
                    temp_map, emis_map, target_mask, target_labels,
                    target, position, aspect, label_counter
                )
                label_counter += 1

        # Apply environmental effects
        temp_map = self._apply_environmental_effects(temp_map)

        return SceneResult(
            temperature_map=temp_map,
            emissivity_map=emis_map,
            target_mask=target_mask,
            target_labels=target_labels,
            metadata={
                "config": self.config,
                "num_targets": label_counter - 1,
                "background_temp_k": self._compute_background_temp(),
            },
        )

    def _generate_background(self) -> tuple[NDArray, NDArray]:
        """Generate background temperature and emissivity maps."""
        h, w = self.config.resolution
        bg_config = self.config.background

        # Base temperature
        base_temp = self._compute_background_temp()

        # Add spatial texture
        texture = self._generate_texture(h, w, bg_config.texture_scale)
        temp_variation = texture * bg_config.temperature_variation_k

        temp_map = np.full((h, w), base_temp, dtype=np.float64) + temp_variation

        # Emissivity map
        terrain_props = self._TERRAIN_PROPERTIES.get(
            bg_config.terrain_type, (0, 0.95)
        )
        emis_map = np.full((h, w), terrain_props[1], dtype=np.float64)

        # Add slight emissivity variation
        emis_variation = self._rng.normal(0, 0.01, (h, w))
        emis_map = np.clip(emis_map + emis_variation, 0.8, 1.0)

        return temp_map, emis_map

    def _generate_texture(
        self,
        height: int,
        width: int,
        scale: float,
    ) -> NDArray:
        """Generate spatial texture using multi-scale noise."""
        texture = np.zeros((height, width))

        # Multi-scale Perlin-like noise
        for octave in range(4):
            freq = 2 ** octave / scale
            amplitude = 1 / (2 ** octave)

            # Generate noise at this frequency
            noise = self._rng.normal(0, 1, (
                int(height * freq) + 2,
                int(width * freq) + 2,
            ))

            # Interpolate to full resolution
            from scipy.ndimage import zoom
            factor_h = height / noise.shape[0]
            factor_w = width / noise.shape[1]
            interpolated = zoom(noise, (factor_h, factor_w), order=1)

            # Crop to exact size
            interpolated = interpolated[:height, :width]

            texture += amplitude * interpolated

        return texture

    def _render_target(
        self,
        temp_map: NDArray,
        emis_map: NDArray,
        target_mask: NDArray,
        target_labels: NDArray,
        target: Target,
        position: tuple[int, int],
        aspect: float,
        label: int,
    ) -> None:
        """Render a single target into the scene maps."""
        h, w = temp_map.shape

        # Get target signature
        target_size = (
            max(8, int(target.geometry.length_m / self.config.gsd_m)),
            max(8, int(target.geometry.width_m / self.config.gsd_m)),
        )
        signature = target.get_signature(resolution=target_size, aspect_angle_deg=aspect)

        # Compute placement bounds
        th, tw = signature.shape
        cy, cx = position
        y0 = max(0, cy - th // 2)
        y1 = min(h, cy + th // 2)
        x0 = max(0, cx - tw // 2)
        x1 = min(w, cx + tw // 2)

        sy0 = max(0, th // 2 - cy)
        sy1 = sy0 + (y1 - y0)
        sx0 = max(0, tw // 2 - cx)
        sx1 = sx0 + (x1 - x0)

        if y1 <= y0 or x1 <= x0 or sy1 <= sy0 or sx1 <= sx0:
            return

        # Extract target region
        target_temp = signature.temperature_map[sy0:sy1, sx0:sx1]
        target_emis = signature.emissivity_map[sy0:sy1, sx0:sx1]

        # Create mask for non-zero pixels
        mask = target_temp > 0

        # Place target
        if mask.any():
            temp_map[y0:y1, x0:x1] = np.where(mask, target_temp, temp_map[y0:y1, x0:x1])
            emis_map[y0:y1, x0:x1] = np.where(mask, target_emis, emis_map[y0:y1, x0:x1])
            target_mask[y0:y1, x0:x1] |= mask
            target_labels[y0:y1, x0:x1] = np.where(mask, label, target_labels[y0:y1, x0:x1])

    def _apply_environmental_effects(self, temp_map: NDArray) -> NDArray:
        """Apply environmental effects to temperature map."""
        env = self.config.environment

        # Night cooling
        if env.time_of_day == TimeOfDay.NIGHT:
            temp_map -= 5.0

        # Rain cooling
        if env.weather == WeatherCondition.RAIN:
            temp_map -= 3.0

        # Fog - reduce contrast
        if env.weather == WeatherCondition.FOG:
            mean_temp = np.mean(temp_map)
            temp_map = 0.8 * temp_map + 0.2 * mean_temp

        return temp_map


@dataclass
class RandomizationConfig:
    """Configuration for scene randomization.

    Attributes:
        position_jitter_m: Position jitter in meters
        temperature_jitter_k: Temperature jitter in Kelvin
        num_targets_range: Range for number of targets (min, max)
        aspect_range: Range for aspect angles (min, max)
        randomize_environment: Whether to randomize environment
    """
    position_jitter_m: float = 5.0
    temperature_jitter_k: float = 3.0
    num_targets_range: tuple[int, int] = (1, 5)
    aspect_range: tuple[float, float] = (0.0, 360.0)
    randomize_environment: bool = True


class SceneTemplate:
    """Pre-defined scene templates for common scenarios."""

    @staticmethod
    def urban_environment(
        resolution: tuple[int, int] = (480, 640),
        gsd_m: float = 0.5,
    ) -> SceneGenerator:
        """Create an urban environment template."""
        config = SceneConfig(
            resolution=resolution,
            gsd_m=gsd_m,
            background=BackgroundConfig(
                terrain_type=TerrainType.URBAN,
                temperature_variation_k=5.0,
            ),
        )
        return SceneGenerator(config)

    @staticmethod
    def desert_environment(
        resolution: tuple[int, int] = (480, 640),
        gsd_m: float = 1.0,
    ) -> SceneGenerator:
        """Create a desert environment template."""
        config = SceneConfig(
            resolution=resolution,
            gsd_m=gsd_m,
            environment=EnvironmentConfig(
                ambient_temperature_k=310.0,
                humidity=0.2,
            ),
            background=BackgroundConfig(
                terrain_type=TerrainType.DESERT,
                temperature_variation_k=8.0,
            ),
        )
        return SceneGenerator(config)

    @staticmethod
    def night_scenario(
        resolution: tuple[int, int] = (480, 640),
        gsd_m: float = 0.5,
    ) -> SceneGenerator:
        """Create a night-time scenario template."""
        config = SceneConfig(
            resolution=resolution,
            gsd_m=gsd_m,
            environment=EnvironmentConfig(
                time_of_day=TimeOfDay.NIGHT,
                ambient_temperature_k=285.0,
            ),
            background=BackgroundConfig(
                temperature_variation_k=3.0,
            ),
        )
        return SceneGenerator(config)

    @staticmethod
    def generate_batch(
        template: SceneGenerator,
        count: int,
        randomizer: Optional[RandomizationConfig] = None,
        target_factory=None,
    ) -> list[SceneResult]:
        """Generate batch of scenes from template.

        Args:
            template: Base scene generator
            count: Number of scenes to generate
            randomizer: Randomization configuration
            target_factory: Callable returning list of targets

        Returns:
            List of SceneResults
        """
        results = []
        rng = np.random.default_rng()

        for i in range(count):
            # Create new generator with same config but different seed
            config = SceneConfig(
                resolution=template.config.resolution,
                gsd_m=template.config.gsd_m,
                sensor_type=template.config.sensor_type,
                environment=template.config.environment,
                background=template.config.background,
                random_seed=rng.integers(0, 2**31),
            )
            gen = SceneGenerator(config)

            # Add targets
            if target_factory is not None:
                targets = target_factory()
                gen.add_random_targets(targets)

            results.append(gen.generate())

        return results


def create_military_scenario(
    num_vehicles: int = 3,
    num_people: int = 5,
    terrain_type: str = "grass",
    time_of_day: str = "midday",
    resolution: tuple[int, int] = (480, 640),
    gsd_m: float = 0.5,
) -> SceneGenerator:
    """Create a military scenario with vehicles and personnel.

    Args:
        num_vehicles: Number of military vehicles
        num_people: Number of personnel
        terrain_type: Terrain type string
        time_of_day: Time of day string
        resolution: Output resolution
        gsd_m: Ground sample distance

    Returns:
        Configured SceneGenerator
    """
    from eosim.targets import VehicleTarget, TankTarget, PersonTarget

    # Parse enums
    terrain = TerrainType.GRASS
    for t in TerrainType:
        if t.value == terrain_type:
            terrain = t
            break

    time = TimeOfDay.MIDDAY
    for t in TimeOfDay:
        if t.value == time_of_day:
            time = t
            break

    config = SceneConfig(
        resolution=resolution,
        gsd_m=gsd_m,
        environment=EnvironmentConfig(time_of_day=time),
        background=BackgroundConfig(terrain_type=terrain),
    )
    gen = SceneGenerator(config)

    # Add vehicles
    vehicles = []
    rng = np.random.default_rng()
    for _ in range(num_vehicles):
        if rng.random() < 0.3:
            vehicles.append(TankTarget(engine_state="running"))
        else:
            vehicles.append(VehicleTarget.truck())
    gen.add_random_targets(vehicles)

    # Add personnel
    people = [PersonTarget.standing(activity="walking") for _ in range(num_people)]
    gen.add_random_targets(people)

    return gen


def create_surveillance_scenario(
    num_targets: int = 3,
    target_type: str = "vehicle",
    resolution: tuple[int, int] = (720, 1280),
    gsd_m: float = 0.25,
) -> SceneGenerator:
    """Create a surveillance scenario.

    Args:
        num_targets: Number of targets
        target_type: Type of targets ("vehicle", "person", "mixed")
        resolution: Output resolution
        gsd_m: Ground sample distance

    Returns:
        Configured SceneGenerator
    """
    from eosim.targets import VehicleTarget, PersonTarget

    config = SceneConfig(
        resolution=resolution,
        gsd_m=gsd_m,
        background=BackgroundConfig(
            terrain_type=TerrainType.URBAN,
        ),
    )
    gen = SceneGenerator(config)

    # Create targets based on type
    rng = np.random.default_rng()
    targets = []
    for _ in range(num_targets):
        if target_type == "vehicle":
            targets.append(VehicleTarget.sedan(engine_state="running"))
        elif target_type == "person":
            targets.append(PersonTarget.standing())
        else:  # mixed
            if rng.random() < 0.5:
                targets.append(VehicleTarget.sedan())
            else:
                targets.append(PersonTarget.standing())

    gen.add_random_targets(targets)

    return gen
