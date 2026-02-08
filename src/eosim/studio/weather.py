"""
Weather system for EOSIM Studio.

Provides atmospheric and weather conditions that affect sensor simulation:
- Cloud types and coverage
- Precipitation (rain, snow, hail)
- Fog and mist
- Storms (thunderstorms, sandstorms, blizzards)
- Wind conditions
- Visibility effects on thermal and visible imaging
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict, List, Any
import math


class CloudType(Enum):
    """Cloud classification types."""
    CLEAR = "clear"                 # No clouds
    CIRRUS = "cirrus"               # High, thin, wispy clouds
    CIRROSTRATUS = "cirrostratus"   # High, thin layer
    CIRROCUMULUS = "cirrocumulus"   # High, small puffy clouds
    ALTOSTRATUS = "altostratus"     # Mid-level gray layer
    ALTOCUMULUS = "altocumulus"     # Mid-level puffy clouds
    STRATUS = "stratus"             # Low, uniform gray layer
    STRATOCUMULUS = "stratocumulus" # Low, lumpy layer
    CUMULUS = "cumulus"             # Fair weather puffy clouds
    CUMULONIMBUS = "cumulonimbus"   # Thunderstorm clouds
    NIMBOSTRATUS = "nimbostratus"   # Dark rain clouds


class PrecipitationType(Enum):
    """Precipitation types."""
    NONE = "none"
    DRIZZLE = "drizzle"         # Light, fine drops
    RAIN_LIGHT = "rain_light"   # Light rain
    RAIN_MODERATE = "rain_moderate"  # Moderate rain
    RAIN_HEAVY = "rain_heavy"   # Heavy rain
    RAIN_TORRENTIAL = "rain_torrential"  # Torrential downpour
    SNOW_LIGHT = "snow_light"   # Light snow
    SNOW_MODERATE = "snow_moderate"  # Moderate snow
    SNOW_HEAVY = "snow_heavy"   # Heavy snow/blizzard
    SLEET = "sleet"             # Mixed rain and snow
    HAIL = "hail"               # Hail
    FREEZING_RAIN = "freezing_rain"  # Freezing rain


class FogType(Enum):
    """Fog and mist types."""
    NONE = "none"
    MIST = "mist"               # Light, visibility > 1km
    FOG_LIGHT = "fog_light"     # Visibility 500m - 1km
    FOG_MODERATE = "fog_moderate"  # Visibility 200m - 500m
    FOG_DENSE = "fog_dense"     # Visibility < 200m
    FOG_FREEZING = "fog_freezing"  # Freezing fog
    HAZE = "haze"               # Dry haze, dust
    SMOKE = "smoke"             # Smoke haze


class StormType(Enum):
    """Storm types."""
    NONE = "none"
    THUNDERSTORM = "thunderstorm"   # Lightning and thunder
    SANDSTORM = "sandstorm"         # Desert sandstorm
    DUST_STORM = "dust_storm"       # Dust storm
    BLIZZARD = "blizzard"           # Snow blizzard
    ICE_STORM = "ice_storm"         # Freezing rain storm
    TROPICAL_STORM = "tropical_storm"  # Tropical storm
    HURRICANE = "hurricane"          # Hurricane/typhoon


class WindSpeed(Enum):
    """Wind speed categories (Beaufort scale simplified)."""
    CALM = "calm"               # 0-5 km/h
    LIGHT = "light"             # 5-20 km/h
    MODERATE = "moderate"       # 20-40 km/h
    STRONG = "strong"           # 40-60 km/h
    GALE = "gale"               # 60-90 km/h
    STORM = "storm"             # 90-120 km/h
    HURRICANE = "hurricane"     # > 120 km/h


# Cloud properties: (altitude_m, opacity, thermal_emission_K, visible_brightness)
CLOUD_PROPERTIES = {
    CloudType.CLEAR: (0, 0.0, 0, 1.0),
    CloudType.CIRRUS: (8000, 0.1, 220, 0.95),
    CloudType.CIRROSTRATUS: (7000, 0.2, 225, 0.90),
    CloudType.CIRROCUMULUS: (6500, 0.15, 223, 0.92),
    CloudType.ALTOSTRATUS: (4000, 0.5, 250, 0.70),
    CloudType.ALTOCUMULUS: (3500, 0.4, 255, 0.75),
    CloudType.STRATUS: (1500, 0.7, 275, 0.50),
    CloudType.STRATOCUMULUS: (1800, 0.6, 270, 0.55),
    CloudType.CUMULUS: (2000, 0.5, 268, 0.80),
    CloudType.CUMULONIMBUS: (3000, 0.9, 260, 0.20),
    CloudType.NIMBOSTRATUS: (2500, 0.85, 265, 0.25),
}

# Precipitation properties: (intensity_mm_h, visibility_reduction, thermal_noise_K)
PRECIPITATION_PROPERTIES = {
    PrecipitationType.NONE: (0, 1.0, 0),
    PrecipitationType.DRIZZLE: (0.5, 0.9, 0.5),
    PrecipitationType.RAIN_LIGHT: (2.5, 0.8, 1.0),
    PrecipitationType.RAIN_MODERATE: (7.5, 0.6, 2.0),
    PrecipitationType.RAIN_HEAVY: (20, 0.4, 4.0),
    PrecipitationType.RAIN_TORRENTIAL: (50, 0.2, 8.0),
    PrecipitationType.SNOW_LIGHT: (1, 0.7, 1.5),
    PrecipitationType.SNOW_MODERATE: (4, 0.5, 3.0),
    PrecipitationType.SNOW_HEAVY: (10, 0.2, 6.0),
    PrecipitationType.SLEET: (5, 0.5, 2.5),
    PrecipitationType.HAIL: (15, 0.3, 5.0),
    PrecipitationType.FREEZING_RAIN: (5, 0.5, 2.0),
}

# Fog properties: (visibility_m, thermal_attenuation, visible_color_shift)
FOG_PROPERTIES = {
    FogType.NONE: (50000, 1.0, (0, 0, 0)),
    FogType.MIST: (2000, 0.95, (10, 10, 15)),
    FogType.FOG_LIGHT: (800, 0.85, (20, 20, 30)),
    FogType.FOG_MODERATE: (350, 0.70, (40, 40, 50)),
    FogType.FOG_DENSE: (100, 0.50, (80, 80, 90)),
    FogType.FOG_FREEZING: (150, 0.55, (70, 75, 85)),
    FogType.HAZE: (5000, 0.90, (15, 12, 8)),
    FogType.SMOKE: (1000, 0.75, (30, 25, 20)),
}

# Storm properties: (visibility_m, wind_speed_kmh, thermal_noise_K, danger_level)
STORM_PROPERTIES = {
    StormType.NONE: (50000, 0, 0, 0),
    StormType.THUNDERSTORM: (2000, 60, 5.0, 3),
    StormType.SANDSTORM: (100, 80, 10.0, 4),
    StormType.DUST_STORM: (500, 50, 6.0, 3),
    StormType.BLIZZARD: (50, 90, 8.0, 5),
    StormType.ICE_STORM: (300, 40, 4.0, 4),
    StormType.TROPICAL_STORM: (1000, 100, 7.0, 4),
    StormType.HURRICANE: (500, 150, 12.0, 5),
}


@dataclass
class WeatherConditions:
    """Complete weather state for a scene."""
    # Cloud conditions
    cloud_type: CloudType = CloudType.CLEAR
    cloud_coverage: float = 0.0  # 0.0 to 1.0 (0% to 100%)
    cloud_base_altitude: float = 2000.0  # meters

    # Precipitation
    precipitation: PrecipitationType = PrecipitationType.NONE
    precipitation_intensity: float = 1.0  # Multiplier on base intensity

    # Fog/visibility
    fog_type: FogType = FogType.NONE
    fog_density: float = 1.0  # Multiplier on base density

    # Storms
    storm_type: StormType = StormType.NONE
    storm_intensity: float = 1.0  # 0.0 to 1.0

    # Wind
    wind_speed: WindSpeed = WindSpeed.CALM
    wind_direction: float = 0.0  # Degrees, 0 = North, 90 = East

    # Atmospheric conditions
    temperature_c: float = 20.0  # Ambient temperature in Celsius
    humidity_percent: float = 50.0  # Relative humidity
    pressure_hpa: float = 1013.25  # Atmospheric pressure

    # Time-varying effects
    lightning_active: bool = False
    lightning_frequency: float = 0.0  # Flashes per minute

    def __post_init__(self):
        # Clamp values
        self.cloud_coverage = max(0.0, min(1.0, self.cloud_coverage))
        self.precipitation_intensity = max(0.0, min(2.0, self.precipitation_intensity))
        self.fog_density = max(0.0, min(2.0, self.fog_density))
        self.storm_intensity = max(0.0, min(1.0, self.storm_intensity))
        self.humidity_percent = max(0.0, min(100.0, self.humidity_percent))
        self.wind_direction = self.wind_direction % 360

    @property
    def visibility_m(self) -> float:
        """Calculate effective visibility in meters."""
        # Start with base visibility
        visibility = 50000.0  # Clear day

        # Apply fog effect
        fog_props = FOG_PROPERTIES.get(self.fog_type, (50000, 1.0, (0, 0, 0)))
        visibility = min(visibility, fog_props[0] / self.fog_density)

        # Apply precipitation effect
        precip_props = PRECIPITATION_PROPERTIES.get(self.precipitation, (0, 1.0, 0))
        visibility *= precip_props[1]

        # Apply storm effect
        storm_props = STORM_PROPERTIES.get(self.storm_type, (50000, 0, 0, 0))
        visibility = min(visibility, storm_props[0])

        # Apply cloud reduction at ground level if overcast
        if self.cloud_coverage > 0.8:
            visibility *= 0.9

        return max(50.0, visibility)  # Minimum 50m visibility

    @property
    def thermal_attenuation(self) -> float:
        """Calculate thermal signal attenuation factor (0-1)."""
        attenuation = 1.0

        # Fog attenuation
        fog_props = FOG_PROPERTIES.get(self.fog_type, (50000, 1.0, (0, 0, 0)))
        attenuation *= fog_props[1]

        # Rain/snow attenuation
        precip_props = PRECIPITATION_PROPERTIES.get(self.precipitation, (0, 1.0, 0))
        attenuation *= precip_props[1]

        # Storm attenuation
        if self.storm_type != StormType.NONE:
            storm_props = STORM_PROPERTIES.get(self.storm_type, (50000, 0, 0, 0))
            attenuation *= max(0.3, 1.0 - self.storm_intensity * 0.5)

        return attenuation

    @property
    def thermal_noise_k(self) -> float:
        """Calculate additional thermal noise in Kelvin."""
        noise = 0.0

        # Precipitation noise
        precip_props = PRECIPITATION_PROPERTIES.get(self.precipitation, (0, 1.0, 0))
        noise += precip_props[2] * self.precipitation_intensity

        # Storm noise
        storm_props = STORM_PROPERTIES.get(self.storm_type, (50000, 0, 0, 0))
        noise += storm_props[2] * self.storm_intensity

        return noise

    @property
    def wind_speed_kmh(self) -> float:
        """Get wind speed in km/h."""
        speed_ranges = {
            WindSpeed.CALM: 2.5,
            WindSpeed.LIGHT: 12.5,
            WindSpeed.MODERATE: 30.0,
            WindSpeed.STRONG: 50.0,
            WindSpeed.GALE: 75.0,
            WindSpeed.STORM: 105.0,
            WindSpeed.HURRICANE: 140.0,
        }
        return speed_ranges.get(self.wind_speed, 0.0)


@dataclass
class WeatherPreset:
    """A named weather preset."""
    name: str
    description: str
    conditions: WeatherConditions


# Weather presets for quick selection
WEATHER_PRESETS: Dict[str, WeatherPreset] = {
    "clear_day": WeatherPreset(
        name="Clear Day",
        description="Sunny, clear skies with excellent visibility",
        conditions=WeatherConditions(
            cloud_type=CloudType.CLEAR,
            cloud_coverage=0.0,
            temperature_c=25.0,
            humidity_percent=40.0,
        )
    ),
    "partly_cloudy": WeatherPreset(
        name="Partly Cloudy",
        description="Fair weather with scattered clouds",
        conditions=WeatherConditions(
            cloud_type=CloudType.CUMULUS,
            cloud_coverage=0.4,
            temperature_c=22.0,
            humidity_percent=55.0,
            wind_speed=WindSpeed.LIGHT,
        )
    ),
    "overcast": WeatherPreset(
        name="Overcast",
        description="Heavy cloud cover, gray skies",
        conditions=WeatherConditions(
            cloud_type=CloudType.STRATUS,
            cloud_coverage=0.9,
            temperature_c=18.0,
            humidity_percent=70.0,
            wind_speed=WindSpeed.LIGHT,
        )
    ),
    "light_rain": WeatherPreset(
        name="Light Rain",
        description="Steady light rain with overcast skies",
        conditions=WeatherConditions(
            cloud_type=CloudType.NIMBOSTRATUS,
            cloud_coverage=1.0,
            precipitation=PrecipitationType.RAIN_LIGHT,
            temperature_c=15.0,
            humidity_percent=85.0,
            wind_speed=WindSpeed.LIGHT,
        )
    ),
    "heavy_rain": WeatherPreset(
        name="Heavy Rain",
        description="Heavy rainfall with reduced visibility",
        conditions=WeatherConditions(
            cloud_type=CloudType.NIMBOSTRATUS,
            cloud_coverage=1.0,
            precipitation=PrecipitationType.RAIN_HEAVY,
            temperature_c=14.0,
            humidity_percent=95.0,
            wind_speed=WindSpeed.MODERATE,
        )
    ),
    "thunderstorm": WeatherPreset(
        name="Thunderstorm",
        description="Active thunderstorm with lightning and heavy rain",
        conditions=WeatherConditions(
            cloud_type=CloudType.CUMULONIMBUS,
            cloud_coverage=0.95,
            precipitation=PrecipitationType.RAIN_HEAVY,
            storm_type=StormType.THUNDERSTORM,
            storm_intensity=0.8,
            temperature_c=18.0,
            humidity_percent=90.0,
            wind_speed=WindSpeed.STRONG,
            lightning_active=True,
            lightning_frequency=6.0,
        )
    ),
    "fog_morning": WeatherPreset(
        name="Morning Fog",
        description="Dense morning fog, limited visibility",
        conditions=WeatherConditions(
            cloud_type=CloudType.STRATUS,
            cloud_coverage=0.3,
            fog_type=FogType.FOG_MODERATE,
            temperature_c=12.0,
            humidity_percent=95.0,
            wind_speed=WindSpeed.CALM,
        )
    ),
    "dense_fog": WeatherPreset(
        name="Dense Fog",
        description="Very dense fog with severely limited visibility",
        conditions=WeatherConditions(
            cloud_type=CloudType.CLEAR,
            cloud_coverage=0.0,
            fog_type=FogType.FOG_DENSE,
            fog_density=1.2,
            temperature_c=8.0,
            humidity_percent=100.0,
            wind_speed=WindSpeed.CALM,
        )
    ),
    "snow_light": WeatherPreset(
        name="Light Snow",
        description="Light snowfall with cold temperatures",
        conditions=WeatherConditions(
            cloud_type=CloudType.STRATUS,
            cloud_coverage=0.9,
            precipitation=PrecipitationType.SNOW_LIGHT,
            temperature_c=-5.0,
            humidity_percent=80.0,
            wind_speed=WindSpeed.LIGHT,
        )
    ),
    "blizzard": WeatherPreset(
        name="Blizzard",
        description="Severe blizzard with heavy snow and high winds",
        conditions=WeatherConditions(
            cloud_type=CloudType.NIMBOSTRATUS,
            cloud_coverage=1.0,
            precipitation=PrecipitationType.SNOW_HEAVY,
            storm_type=StormType.BLIZZARD,
            storm_intensity=0.9,
            temperature_c=-15.0,
            humidity_percent=85.0,
            wind_speed=WindSpeed.GALE,
        )
    ),
    "sandstorm": WeatherPreset(
        name="Sandstorm",
        description="Desert sandstorm with severely reduced visibility",
        conditions=WeatherConditions(
            cloud_type=CloudType.CLEAR,
            cloud_coverage=0.0,
            storm_type=StormType.SANDSTORM,
            storm_intensity=0.8,
            fog_type=FogType.HAZE,
            fog_density=1.5,
            temperature_c=38.0,
            humidity_percent=15.0,
            wind_speed=WindSpeed.STRONG,
        )
    ),
    "haze": WeatherPreset(
        name="Hazy",
        description="Hazy conditions with reduced visibility",
        conditions=WeatherConditions(
            cloud_type=CloudType.CLEAR,
            cloud_coverage=0.1,
            fog_type=FogType.HAZE,
            temperature_c=28.0,
            humidity_percent=60.0,
            wind_speed=WindSpeed.CALM,
        )
    ),
    "tropical_storm": WeatherPreset(
        name="Tropical Storm",
        description="Tropical storm with high winds and heavy rain",
        conditions=WeatherConditions(
            cloud_type=CloudType.CUMULONIMBUS,
            cloud_coverage=1.0,
            precipitation=PrecipitationType.RAIN_TORRENTIAL,
            storm_type=StormType.TROPICAL_STORM,
            storm_intensity=0.85,
            temperature_c=26.0,
            humidity_percent=95.0,
            wind_speed=WindSpeed.STORM,
            lightning_active=True,
            lightning_frequency=4.0,
        )
    ),
    "freezing_fog": WeatherPreset(
        name="Freezing Fog",
        description="Freezing fog with ice formation",
        conditions=WeatherConditions(
            cloud_type=CloudType.STRATUS,
            cloud_coverage=0.5,
            fog_type=FogType.FOG_FREEZING,
            temperature_c=-8.0,
            humidity_percent=100.0,
            wind_speed=WindSpeed.CALM,
        )
    ),
    "night_clear": WeatherPreset(
        name="Clear Night",
        description="Clear night sky with good visibility",
        conditions=WeatherConditions(
            cloud_type=CloudType.CLEAR,
            cloud_coverage=0.0,
            temperature_c=12.0,
            humidity_percent=50.0,
            wind_speed=WindSpeed.CALM,
        )
    ),
}


class WeatherSystem:
    """Manages weather conditions and effects for rendering."""

    def __init__(self):
        self._conditions = WeatherConditions()
        self._time = 0.0  # For time-varying effects

    @property
    def conditions(self) -> WeatherConditions:
        return self._conditions

    @conditions.setter
    def conditions(self, value: WeatherConditions):
        self._conditions = value

    def set_preset(self, preset_name: str) -> bool:
        """Set weather from a preset."""
        if preset_name in WEATHER_PRESETS:
            preset = WEATHER_PRESETS[preset_name]
            # Create a copy of the preset conditions
            self._conditions = WeatherConditions(
                cloud_type=preset.conditions.cloud_type,
                cloud_coverage=preset.conditions.cloud_coverage,
                cloud_base_altitude=preset.conditions.cloud_base_altitude,
                precipitation=preset.conditions.precipitation,
                precipitation_intensity=preset.conditions.precipitation_intensity,
                fog_type=preset.conditions.fog_type,
                fog_density=preset.conditions.fog_density,
                storm_type=preset.conditions.storm_type,
                storm_intensity=preset.conditions.storm_intensity,
                wind_speed=preset.conditions.wind_speed,
                wind_direction=preset.conditions.wind_direction,
                temperature_c=preset.conditions.temperature_c,
                humidity_percent=preset.conditions.humidity_percent,
                pressure_hpa=preset.conditions.pressure_hpa,
                lightning_active=preset.conditions.lightning_active,
                lightning_frequency=preset.conditions.lightning_frequency,
            )
            return True
        return False

    def get_presets(self) -> Dict[str, str]:
        """Get available presets with descriptions."""
        return {name: preset.description for name, preset in WEATHER_PRESETS.items()}

    def update(self, dt: float):
        """Update time-varying weather effects."""
        self._time += dt

    def get_sky_color(self) -> Tuple[int, int, int]:
        """Get sky color based on weather conditions."""
        # Base sky color (clear day)
        r, g, b = 135, 206, 235  # Light blue

        # Adjust for cloud coverage
        cloud_props = CLOUD_PROPERTIES.get(self._conditions.cloud_type, (0, 0, 0, 1.0))
        cloud_opacity = cloud_props[1] * self._conditions.cloud_coverage
        brightness = cloud_props[3]

        # Darken for clouds
        r = int(r * (1 - cloud_opacity * 0.5) * brightness)
        g = int(g * (1 - cloud_opacity * 0.5) * brightness)
        b = int(b * (1 - cloud_opacity * 0.4) * brightness)

        # Apply fog color shift
        fog_props = FOG_PROPERTIES.get(self._conditions.fog_type, (50000, 1.0, (0, 0, 0)))
        fog_shift = fog_props[2]
        r = min(255, r + int(fog_shift[0] * self._conditions.fog_density))
        g = min(255, g + int(fog_shift[1] * self._conditions.fog_density))
        b = min(255, b + int(fog_shift[2] * self._conditions.fog_density))

        # Adjust for storms
        if self._conditions.storm_type != StormType.NONE:
            darkness = self._conditions.storm_intensity * 0.6
            r = int(r * (1 - darkness))
            g = int(g * (1 - darkness))
            b = int(b * (1 - darkness))

        return (max(0, r), max(0, g), max(0, b))

    def get_ambient_temperature_k(self) -> float:
        """Get ambient temperature in Kelvin."""
        return self._conditions.temperature_c + 273.15

    def apply_visibility_effect(self, distance_m: float) -> float:
        """Calculate visibility factor for an object at given distance."""
        visibility = self._conditions.visibility_m

        # Exponential falloff
        if distance_m <= 0:
            return 1.0

        # Visibility factor using Beer-Lambert law
        extinction = 3.0 / visibility  # Extinction coefficient
        factor = math.exp(-extinction * distance_m)

        return max(0.0, min(1.0, factor))

    def apply_thermal_effects(self, temperature_k: float, distance_m: float) -> float:
        """Apply weather effects to thermal reading."""
        # Apply attenuation
        attenuation = self._conditions.thermal_attenuation
        ambient_k = self.get_ambient_temperature_k()

        # Mix with ambient based on attenuation
        temp = temperature_k * attenuation + ambient_k * (1 - attenuation)

        # Add noise
        noise = self._conditions.thermal_noise_k
        if noise > 0:
            temp += np.random.normal(0, noise)

        # Distance attenuation
        vis_factor = self.apply_visibility_effect(distance_m)
        temp = temp * vis_factor + ambient_k * (1 - vis_factor)

        return temp

    def is_lightning_flash(self, time_sec: float) -> bool:
        """Check if a lightning flash should occur at this time."""
        if not self._conditions.lightning_active:
            return False

        freq = self._conditions.lightning_frequency
        if freq <= 0:
            return False

        # Simple stochastic model
        prob = freq / 60.0  # Per second probability
        return np.random.random() < prob

    def get_precipitation_particles(self, width: int, height: int) -> Optional[NDArray]:
        """Generate precipitation particle positions for rendering."""
        if self._conditions.precipitation == PrecipitationType.NONE:
            return None

        # Number of particles based on intensity
        precip_props = PRECIPITATION_PROPERTIES.get(self._conditions.precipitation, (0, 1.0, 0))
        intensity = precip_props[0] * self._conditions.precipitation_intensity

        if intensity <= 0:
            return None

        # Particles per frame
        n_particles = int(intensity * width * height / 50000)
        n_particles = min(n_particles, 2000)  # Cap for performance

        if n_particles <= 0:
            return None

        # Generate random particle positions
        particles = np.zeros((n_particles, 4), dtype=np.float32)
        particles[:, 0] = np.random.uniform(0, width, n_particles)   # x
        particles[:, 1] = np.random.uniform(0, height, n_particles)  # y

        # Particle size and velocity based on type
        if "snow" in self._conditions.precipitation.value:
            particles[:, 2] = np.random.uniform(2, 5, n_particles)   # size
            particles[:, 3] = np.random.uniform(1, 3, n_particles)   # velocity
        elif "hail" in self._conditions.precipitation.value:
            particles[:, 2] = np.random.uniform(4, 8, n_particles)   # size
            particles[:, 3] = np.random.uniform(8, 15, n_particles)  # velocity
        else:  # rain
            particles[:, 2] = np.random.uniform(1, 3, n_particles)   # size
            particles[:, 3] = np.random.uniform(5, 12, n_particles)  # velocity

        return particles


def create_weather(preset: str = "clear_day") -> WeatherConditions:
    """Create weather conditions from a preset.

    Args:
        preset: Preset name (e.g., "clear_day", "thunderstorm", "fog_morning")

    Returns:
        WeatherConditions for the specified preset
    """
    system = WeatherSystem()
    if system.set_preset(preset):
        return system.conditions
    return WeatherConditions()  # Default clear conditions
