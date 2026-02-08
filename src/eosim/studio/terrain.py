"""
Geographic terrain system for EOSIM Studio.

Provides terrain import based on geographic location with:
- Digital Terrain Model (DTM) / elevation data
- Land cover classification (vegetation, urban, water, desert, etc.)
- Configurable detail levels to manage performance
- Procedural terrain generation for demo/testing
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict, List, Any
import math


class DetailLevel(Enum):
    """Terrain detail level - affects grid resolution and feature density."""
    LOW = "low"        # 100m grid, basic features
    MEDIUM = "medium"  # 30m grid, moderate features
    HIGH = "high"      # 10m grid, detailed features


class LandCoverType(Enum):
    """Land cover classification types."""
    WATER_DEEP = "water_deep"           # Deep ocean/sea
    WATER_SHALLOW = "water_shallow"     # Shallow water, coastal
    WATER_RIVER = "water_river"         # Rivers and streams

    SAND_DESERT = "sand_desert"         # Sandy desert
    ROCK_DESERT = "rock_desert"         # Rocky desert
    SALT_FLAT = "salt_flat"             # Salt flats

    GRASS_SHORT = "grass_short"         # Short grass, savanna
    GRASS_TALL = "grass_tall"           # Tall grass, prairie
    SHRUB = "shrub"                     # Shrubland

    FOREST_DECIDUOUS = "forest_deciduous"  # Deciduous forest
    FOREST_CONIFER = "forest_conifer"      # Coniferous forest
    FOREST_TROPICAL = "forest_tropical"    # Tropical forest

    URBAN_LOW = "urban_low"             # Suburban, low density
    URBAN_MEDIUM = "urban_medium"       # Urban, medium density
    URBAN_HIGH = "urban_high"           # Urban, high density (downtown)
    INDUSTRIAL = "industrial"           # Industrial areas

    FARMLAND = "farmland"               # Agricultural land
    SNOW_ICE = "snow_ice"               # Snow and ice
    BARE_ROCK = "bare_rock"             # Exposed rock
    WETLAND = "wetland"                 # Marshes, swamps


# Thermal and visible properties for each land cover type
LAND_COVER_PROPERTIES = {
    # Type: (thermal_temp_day_K, thermal_temp_night_K, visible_color_RGB, roughness)
    LandCoverType.WATER_DEEP: (285, 287, (20, 40, 100), 0.0),
    LandCoverType.WATER_SHALLOW: (288, 286, (40, 80, 140), 0.0),
    LandCoverType.WATER_RIVER: (286, 285, (50, 90, 130), 0.0),

    LandCoverType.SAND_DESERT: (320, 280, (210, 190, 140), 0.2),
    LandCoverType.ROCK_DESERT: (315, 285, (160, 140, 110), 0.4),
    LandCoverType.SALT_FLAT: (318, 282, (230, 230, 220), 0.1),

    LandCoverType.GRASS_SHORT: (300, 288, (140, 160, 80), 0.3),
    LandCoverType.GRASS_TALL: (298, 290, (100, 140, 60), 0.5),
    LandCoverType.SHRUB: (302, 288, (120, 130, 70), 0.6),

    LandCoverType.FOREST_DECIDUOUS: (295, 292, (60, 100, 40), 0.8),
    LandCoverType.FOREST_CONIFER: (293, 291, (40, 80, 50), 0.9),
    LandCoverType.FOREST_TROPICAL: (298, 295, (30, 90, 30), 0.95),

    LandCoverType.URBAN_LOW: (305, 295, (150, 140, 130), 0.4),
    LandCoverType.URBAN_MEDIUM: (310, 298, (130, 130, 130), 0.5),
    LandCoverType.URBAN_HIGH: (315, 302, (110, 110, 120), 0.6),
    LandCoverType.INDUSTRIAL: (320, 305, (140, 130, 120), 0.5),

    LandCoverType.FARMLAND: (302, 290, (160, 170, 90), 0.3),
    LandCoverType.SNOW_ICE: (268, 265, (240, 245, 250), 0.1),
    LandCoverType.BARE_ROCK: (310, 285, (140, 130, 120), 0.7),
    LandCoverType.WETLAND: (295, 293, (80, 110, 70), 0.4),
}


@dataclass
class GeoLocation:
    """Geographic location with latitude and longitude."""
    latitude: float   # Degrees, -90 to 90
    longitude: float  # Degrees, -180 to 180
    name: str = ""

    def __post_init__(self):
        # Clamp values to valid ranges
        self.latitude = max(-90, min(90, self.latitude))
        self.longitude = ((self.longitude + 180) % 360) - 180

    def distance_to(self, other: 'GeoLocation') -> float:
        """Calculate approximate distance in meters using Haversine formula."""
        R = 6371000  # Earth radius in meters

        lat1, lat2 = math.radians(self.latitude), math.radians(other.latitude)
        dlat = math.radians(other.latitude - self.latitude)
        dlon = math.radians(other.longitude - self.longitude)

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    def offset_meters(self, east_m: float, north_m: float) -> 'GeoLocation':
        """Create new location offset by meters."""
        # Approximate degrees per meter
        lat_deg_per_m = 1 / 111320
        lon_deg_per_m = 1 / (111320 * math.cos(math.radians(self.latitude)))

        return GeoLocation(
            latitude=self.latitude + north_m * lat_deg_per_m,
            longitude=self.longitude + east_m * lon_deg_per_m,
            name=f"{self.name}_offset"
        )


@dataclass
class TerrainConfig:
    """Configuration for terrain import."""
    center: GeoLocation
    radius_m: float = 5000.0  # Radius in meters
    detail_level: DetailLevel = DetailLevel.LOW

    # Time of day affects thermal properties
    time_of_day: float = 12.0  # Hours (0-24), noon default

    # Optional bounds override
    bounds_override: Optional[Tuple[float, float, float, float]] = None  # (min_x, min_y, max_x, max_y)

    @property
    def grid_resolution(self) -> float:
        """Get grid cell size in meters based on detail level."""
        resolutions = {
            DetailLevel.LOW: 100.0,
            DetailLevel.MEDIUM: 30.0,
            DetailLevel.HIGH: 10.0,
        }
        return resolutions[self.detail_level]

    @property
    def grid_size(self) -> Tuple[int, int]:
        """Get grid dimensions (rows, cols)."""
        diameter = self.radius_m * 2
        cells = int(diameter / self.grid_resolution)
        return (cells, cells)


@dataclass
class TerrainData:
    """Terrain data for a region."""
    config: TerrainConfig

    # Elevation data (meters above sea level)
    elevation: NDArray  # Shape: (rows, cols)

    # Land cover classification
    land_cover: NDArray  # Shape: (rows, cols), dtype=int (LandCoverType enum values)

    # Computed thermal properties
    thermal_map: Optional[NDArray] = None  # Shape: (rows, cols), temperatures in K

    # Computed visible colors
    visible_map: Optional[NDArray] = None  # Shape: (rows, cols, 3), RGB colors

    # Metadata
    min_elevation: float = 0.0
    max_elevation: float = 0.0
    dominant_cover: Optional[LandCoverType] = None

    def __post_init__(self):
        self.min_elevation = float(self.elevation.min())
        self.max_elevation = float(self.elevation.max())

        # Find dominant land cover
        unique, counts = np.unique(self.land_cover, return_counts=True)
        if len(unique) > 0:
            dominant_idx = unique[np.argmax(counts)]
            for lc in LandCoverType:
                if lc.value == dominant_idx or hash(lc) % 100 == dominant_idx:
                    self.dominant_cover = lc
                    break

    def get_elevation_at(self, x: float, y: float) -> float:
        """Get interpolated elevation at local coordinates (meters from center)."""
        rows, cols = self.elevation.shape
        radius = self.config.radius_m

        # Convert to grid coordinates
        col = int((x + radius) / self.config.grid_resolution)
        row = int((radius - y) / self.config.grid_resolution)  # Y is north

        # Clamp to bounds
        row = max(0, min(rows - 1, row))
        col = max(0, min(cols - 1, col))

        return float(self.elevation[row, col])

    def get_land_cover_at(self, x: float, y: float) -> LandCoverType:
        """Get land cover type at local coordinates."""
        rows, cols = self.land_cover.shape
        radius = self.config.radius_m

        col = int((x + radius) / self.config.grid_resolution)
        row = int((radius - y) / self.config.grid_resolution)

        row = max(0, min(rows - 1, row))
        col = max(0, min(cols - 1, col))

        cover_val = self.land_cover[row, col]

        # Map integer back to enum
        for lc in LandCoverType:
            if hash(lc) % 100 == cover_val:
                return lc

        return LandCoverType.GRASS_SHORT  # Default

    def compute_thermal_map(self, time_of_day: Optional[float] = None):
        """Compute thermal temperature map based on land cover and time."""
        if time_of_day is None:
            time_of_day = self.config.time_of_day

        rows, cols = self.land_cover.shape
        self.thermal_map = np.zeros((rows, cols), dtype=np.float32)

        # Interpolate between day and night temperatures based on time
        # Simple sinusoidal model: max at 14:00, min at 04:00
        hour_offset = (time_of_day - 14) % 24
        day_factor = 0.5 * (1 + math.cos(math.pi * hour_offset / 12))

        for lc in LandCoverType:
            props = LAND_COVER_PROPERTIES.get(lc)
            if props:
                temp_day, temp_night = props[0], props[1]
                temp = temp_day * day_factor + temp_night * (1 - day_factor)

                mask = self.land_cover == (hash(lc) % 100)
                self.thermal_map[mask] = temp

        # Add elevation-based temperature adjustment (-6.5°C per 1000m)
        lapse_rate = 0.0065  # K per meter
        self.thermal_map -= (self.elevation - self.min_elevation) * lapse_rate

    def compute_visible_map(self):
        """Compute visible color map based on land cover."""
        rows, cols = self.land_cover.shape
        self.visible_map = np.zeros((rows, cols, 3), dtype=np.uint8)

        for lc in LandCoverType:
            props = LAND_COVER_PROPERTIES.get(lc)
            if props:
                color = props[2]
                mask = self.land_cover == (hash(lc) % 100)
                self.visible_map[mask] = color

        # Add shading based on elevation gradient (simple hillshade)
        if self.elevation.max() > self.elevation.min():
            # Compute gradient
            gy, gx = np.gradient(self.elevation)

            # Light from northwest
            light_dir = np.array([-1, 1, 2])
            light_dir = light_dir / np.linalg.norm(light_dir)

            # Surface normal
            norm = np.sqrt(gx**2 + gy**2 + 1)
            nx, ny, nz = -gx/norm, -gy/norm, 1/norm

            # Dot product for shading
            shade = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
            shade = np.clip(shade, 0.3, 1.0)

            # Apply shading
            for c in range(3):
                self.visible_map[:, :, c] = (self.visible_map[:, :, c] * shade).astype(np.uint8)


class TerrainProvider:
    """Provides terrain data for geographic locations.

    In a real implementation, this would fetch data from:
    - SRTM/ASTER for elevation
    - ESA WorldCover / NLCD for land cover
    - OpenStreetMap for features

    For demo purposes, generates procedural terrain.
    """

    # Predefined location presets
    LOCATION_PRESETS = {
        "mojave_desert": GeoLocation(35.0, -116.0, "Mojave Desert, CA"),
        "persian_gulf": GeoLocation(26.5, 51.5, "Persian Gulf"),
        "central_europe": GeoLocation(50.0, 10.0, "Central Europe"),
        "korean_peninsula": GeoLocation(37.5, 127.0, "Korean Peninsula"),
        "sahara": GeoLocation(25.0, 10.0, "Sahara Desert"),
        "arctic": GeoLocation(75.0, 0.0, "Arctic"),
        "pacific_islands": GeoLocation(15.0, 145.0, "Pacific Islands"),
        "amazon": GeoLocation(-3.0, -60.0, "Amazon Basin"),
        "himalaya": GeoLocation(28.0, 84.0, "Himalaya Mountains"),
        "great_plains": GeoLocation(40.0, -100.0, "Great Plains, USA"),
    }

    def __init__(self):
        self._cache: Dict[str, TerrainData] = {}

    def get_preset_locations(self) -> Dict[str, GeoLocation]:
        """Get available preset locations."""
        return self.LOCATION_PRESETS.copy()

    def load_terrain(self, config: TerrainConfig) -> TerrainData:
        """Load or generate terrain data for the given configuration."""
        # Check cache
        cache_key = f"{config.center.latitude:.4f}_{config.center.longitude:.4f}_{config.radius_m}_{config.detail_level.value}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Generate procedural terrain based on location
        terrain = self._generate_procedural_terrain(config)

        # Compute derived maps
        terrain.compute_thermal_map()
        terrain.compute_visible_map()

        # Cache result
        self._cache[cache_key] = terrain

        return terrain

    def _generate_procedural_terrain(self, config: TerrainConfig) -> TerrainData:
        """Generate procedural terrain based on geographic location."""
        rows, cols = config.grid_size

        # Determine terrain type based on latitude and location name
        lat = config.center.latitude
        lon = config.center.longitude

        # Base terrain type from latitude/location
        if abs(lat) > 60:
            base_type = "arctic"
        elif abs(lat) < 25 and -30 < lon < 60:
            base_type = "desert"
        elif abs(lat) < 15 and lon > 100:
            base_type = "tropical"
        elif -10 < lat < 10 and -80 < lon < -40:
            base_type = "jungle"
        elif 30 < lat < 50 and -130 < lon < -70:
            base_type = "plains"
        else:
            base_type = "temperate"

        # Check for coastal areas (simplified)
        is_coastal = (abs(lon) > 170) or (20 < lon < 60 and lat < 30)

        # Generate elevation using Perlin-like noise
        elevation = self._generate_elevation(rows, cols, config, base_type)

        # Generate land cover
        land_cover = self._generate_land_cover(rows, cols, elevation, base_type, is_coastal)

        return TerrainData(
            config=config,
            elevation=elevation,
            land_cover=land_cover,
        )

    def _generate_elevation(self, rows: int, cols: int, config: TerrainConfig,
                           base_type: str) -> NDArray:
        """Generate elevation data using multi-octave noise."""
        elevation = np.zeros((rows, cols), dtype=np.float32)

        # Base elevation by terrain type
        base_heights = {
            "arctic": (0, 500),
            "desert": (200, 800),
            "tropical": (0, 200),
            "jungle": (50, 400),
            "plains": (300, 600),
            "temperate": (100, 800),
        }
        base_min, base_max = base_heights.get(base_type, (0, 500))

        # Generate multi-octave noise
        np.random.seed(int(abs(config.center.latitude * 1000 + config.center.longitude * 100)) % (2**31))

        for octave in range(4):
            freq = 2 ** octave
            amp = 1.0 / (octave + 1)

            # Create noise at this frequency
            noise_rows = max(2, rows // (8 // freq))
            noise_cols = max(2, cols // (8 // freq))
            noise = np.random.randn(noise_rows, noise_cols) * amp

            # Upsample to full resolution
            from scipy.ndimage import zoom
            try:
                scale_r = rows / noise_rows
                scale_c = cols / noise_cols
                upsampled = zoom(noise, (scale_r, scale_c), order=1)
                elevation += upsampled[:rows, :cols]
            except ImportError:
                # Fallback without scipy
                elevation += np.random.randn(rows, cols) * amp * 0.5

        # Normalize and scale
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 0.001)
        elevation = base_min + elevation * (base_max - base_min)

        return elevation

    def _generate_land_cover(self, rows: int, cols: int, elevation: NDArray,
                            base_type: str, is_coastal: bool) -> NDArray:
        """Generate land cover classification."""
        land_cover = np.zeros((rows, cols), dtype=np.int32)

        # Land cover distributions by terrain type
        if base_type == "arctic":
            primary = LandCoverType.SNOW_ICE
            secondary = LandCoverType.BARE_ROCK
            tertiary = LandCoverType.WATER_SHALLOW
        elif base_type == "desert":
            primary = LandCoverType.SAND_DESERT
            secondary = LandCoverType.ROCK_DESERT
            tertiary = LandCoverType.SHRUB
        elif base_type == "tropical" or base_type == "jungle":
            primary = LandCoverType.FOREST_TROPICAL
            secondary = LandCoverType.GRASS_TALL
            tertiary = LandCoverType.WETLAND
        elif base_type == "plains":
            primary = LandCoverType.GRASS_SHORT
            secondary = LandCoverType.FARMLAND
            tertiary = LandCoverType.SHRUB
        else:  # temperate
            primary = LandCoverType.FOREST_DECIDUOUS
            secondary = LandCoverType.GRASS_SHORT
            tertiary = LandCoverType.FARMLAND

        # Assign based on elevation percentiles
        elev_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 0.001)

        # Low elevation - primary type
        low_mask = elev_norm < 0.4
        land_cover[low_mask] = hash(primary) % 100

        # Medium elevation - secondary type
        mid_mask = (elev_norm >= 0.4) & (elev_norm < 0.7)
        land_cover[mid_mask] = hash(secondary) % 100

        # High elevation - tertiary type or bare rock
        high_mask = elev_norm >= 0.7
        land_cover[high_mask] = hash(tertiary) % 100

        # Very high elevation - bare rock or snow
        very_high = elev_norm >= 0.9
        if base_type == "arctic":
            land_cover[very_high] = hash(LandCoverType.SNOW_ICE) % 100
        else:
            land_cover[very_high] = hash(LandCoverType.BARE_ROCK) % 100

        # Add water features
        if is_coastal:
            # Add ocean on one side
            water_cols = cols // 4
            land_cover[:, :water_cols] = hash(LandCoverType.WATER_DEEP) % 100
            land_cover[:, water_cols:water_cols+5] = hash(LandCoverType.WATER_SHALLOW) % 100

        # Add some rivers (lowest elevation paths)
        if rows > 20 and cols > 20:
            # Simple river from high to low
            river_start = np.unravel_index(np.argmax(elevation), elevation.shape)
            river_end = np.unravel_index(np.argmin(elevation), elevation.shape)

            # Draw a rough path
            for t in np.linspace(0, 1, 50):
                r = int(river_start[0] + t * (river_end[0] - river_start[0]))
                c = int(river_start[1] + t * (river_end[1] - river_start[1]))
                r = max(0, min(rows-1, r))
                c = max(0, min(cols-1, c))
                land_cover[r, c] = hash(LandCoverType.WATER_RIVER) % 100

        # Add random urban areas in temperate/plains
        if base_type in ["temperate", "plains"]:
            n_cities = 2 if rows > 50 else 1
            for _ in range(n_cities):
                cr = np.random.randint(rows // 4, 3 * rows // 4)
                cc = np.random.randint(cols // 4, 3 * cols // 4)
                size = np.random.randint(3, 8)
                for dr in range(-size, size+1):
                    for dc in range(-size, size+1):
                        r, c = cr + dr, cc + dc
                        if 0 <= r < rows and 0 <= c < cols:
                            dist = abs(dr) + abs(dc)
                            if dist < size // 2:
                                land_cover[r, c] = hash(LandCoverType.URBAN_HIGH) % 100
                            elif dist < size:
                                land_cover[r, c] = hash(LandCoverType.URBAN_LOW) % 100

        return land_cover

    def clear_cache(self):
        """Clear the terrain cache."""
        self._cache.clear()


# Convenience function for quick terrain setup
def create_terrain(location: str = "mojave_desert",
                   radius_m: float = 5000.0,
                   detail: str = "low") -> TerrainData:
    """Create terrain data for a preset location.

    Args:
        location: Preset location name or "lat,lon" string
        radius_m: Radius in meters
        detail: Detail level ("low", "medium", "high")

    Returns:
        TerrainData for the specified area
    """
    provider = TerrainProvider()

    # Parse location
    if location in provider.LOCATION_PRESETS:
        geo = provider.LOCATION_PRESETS[location]
    elif "," in location:
        parts = location.split(",")
        lat, lon = float(parts[0]), float(parts[1])
        geo = GeoLocation(lat, lon, f"Custom ({lat:.2f}, {lon:.2f})")
    else:
        raise ValueError(f"Unknown location: {location}. Use preset name or 'lat,lon' format.")

    # Parse detail level
    detail_map = {"low": DetailLevel.LOW, "medium": DetailLevel.MEDIUM, "high": DetailLevel.HIGH}
    detail_level = detail_map.get(detail.lower(), DetailLevel.LOW)

    config = TerrainConfig(center=geo, radius_m=radius_m, detail_level=detail_level)

    return provider.load_terrain(config)
