"""
Base classes for EOSIM Target Library.

Provides foundational classes for target modeling including geometry,
thermal properties, and rendering capabilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union, Callable
from enum import Enum
import numpy as np
from numpy.typing import NDArray


class MaterialType(Enum):
    """Common material types with thermal properties."""
    METAL_PAINTED = "metal_painted"
    METAL_BARE = "metal_bare"
    GLASS = "glass"
    RUBBER = "rubber"
    PLASTIC = "plastic"
    FABRIC = "fabric"
    CONCRETE = "concrete"
    ASPHALT = "asphalt"
    SKIN = "skin"
    VEGETATION = "vegetation"
    WATER = "water"
    SOIL = "soil"


@dataclass
class MaterialProperties:
    """Thermal and optical properties of a material.

    Attributes:
        emissivity: Thermal emissivity (0-1)
        reflectance: Solar reflectance (0-1)
        thermal_mass: J/(kg·K) - affects heating/cooling rate
        thermal_conductivity: W/(m·K)
        solar_absorptance: Solar absorption (0-1)
    """
    emissivity: float = 0.9
    reflectance: float = 0.3
    thermal_mass: float = 500.0
    thermal_conductivity: float = 1.0
    solar_absorptance: float = 0.7

    @classmethod
    def from_material_type(cls, mat_type: MaterialType) -> "MaterialProperties":
        """Get typical properties for a material type."""
        properties = {
            MaterialType.METAL_PAINTED: cls(0.85, 0.4, 450, 50, 0.6),
            MaterialType.METAL_BARE: cls(0.15, 0.7, 450, 200, 0.3),
            MaterialType.GLASS: cls(0.92, 0.08, 840, 1.0, 0.1),
            MaterialType.RUBBER: cls(0.95, 0.1, 2000, 0.15, 0.9),
            MaterialType.PLASTIC: cls(0.92, 0.3, 1500, 0.2, 0.7),
            MaterialType.FABRIC: cls(0.95, 0.4, 1300, 0.05, 0.6),
            MaterialType.CONCRETE: cls(0.92, 0.35, 880, 1.4, 0.65),
            MaterialType.ASPHALT: cls(0.93, 0.1, 920, 0.75, 0.9),
            MaterialType.SKIN: cls(0.98, 0.4, 3500, 0.5, 0.6),
            MaterialType.VEGETATION: cls(0.96, 0.25, 2500, 0.3, 0.75),
            MaterialType.WATER: cls(0.96, 0.06, 4186, 0.6, 0.94),
            MaterialType.SOIL: cls(0.92, 0.2, 800, 0.5, 0.8),
        }
        return properties.get(mat_type, cls())


@dataclass
class HotSpot:
    """A localized hot region within a target.

    Attributes:
        name: Identifier (e.g., "engine", "exhaust")
        relative_position: (y, x) normalized 0-1 within target
        relative_size: Size as fraction of target size
        temperature_delta_k: Temperature above base
        shape: "circle", "ellipse", or "rectangle"
        pulsing: Whether temperature varies periodically
        pulse_period_s: Period of temperature variation
        pulse_amplitude_k: Amplitude of temperature variation
    """
    name: str
    relative_position: tuple[float, float]
    relative_size: float
    temperature_delta_k: float
    shape: str = "circle"
    pulsing: bool = False
    pulse_period_s: float = 1.0
    pulse_amplitude_k: float = 0.0

    def get_temperature_delta(self, time_s: float = 0.0) -> float:
        """Get temperature delta at a given time."""
        if self.pulsing:
            phase = 2 * np.pi * time_s / self.pulse_period_s
            return self.temperature_delta_k + self.pulse_amplitude_k * np.sin(phase)
        return self.temperature_delta_k


@dataclass
class TargetGeometry:
    """Physical geometry of a target.

    Attributes:
        length_m: Length in meters
        width_m: Width in meters
        height_m: Height in meters
        shape: Basic shape ("box", "cylinder", "ellipsoid", "custom")
        outline_points: Custom outline as normalized (y, x) points
    """
    length_m: float
    width_m: float
    height_m: float
    shape: str = "box"
    outline_points: Optional[NDArray] = None

    @property
    def footprint_area_m2(self) -> float:
        """Ground footprint area."""
        if self.shape == "cylinder" or self.shape == "ellipsoid":
            return np.pi * self.length_m * self.width_m / 4
        return self.length_m * self.width_m

    @property
    def surface_area_m2(self) -> float:
        """Approximate total surface area."""
        if self.shape == "box":
            return 2 * (self.length_m * self.width_m +
                       self.length_m * self.height_m +
                       self.width_m * self.height_m)
        elif self.shape == "cylinder":
            r = min(self.length_m, self.width_m) / 2
            return 2 * np.pi * r * self.height_m + 2 * np.pi * r**2
        return self.length_m * self.width_m * 4  # Rough estimate


@dataclass
class TargetSignature:
    """Thermal/radiometric signature of a target.

    Attributes:
        temperature_map: 2D temperature distribution [K]
        emissivity_map: 2D emissivity distribution [0-1]
        geometry: Target geometry
        aspect_angle_deg: Viewing angle (0=front, 90=side, 180=rear)
        elevation_angle_deg: Viewing elevation
    """
    temperature_map: NDArray
    emissivity_map: NDArray
    geometry: TargetGeometry
    aspect_angle_deg: float = 0.0
    elevation_angle_deg: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.temperature_map.shape

    @property
    def mean_temperature(self) -> float:
        return float(np.mean(self.temperature_map))

    @property
    def max_temperature(self) -> float:
        return float(np.max(self.temperature_map))

    @property
    def thermal_contrast(self) -> float:
        """Temperature range within target."""
        return float(np.max(self.temperature_map) - np.min(self.temperature_map))


class Target(ABC):
    """Abstract base class for target models.

    Subclasses implement specific target types with appropriate
    thermal signatures, geometry, and time-varying behavior.
    """

    def __init__(
        self,
        name: str,
        geometry: TargetGeometry,
        base_temperature_k: float = 300.0,
        ambient_temperature_k: float = 290.0,
    ) -> None:
        """Initialize target.

        Args:
            name: Target identifier
            geometry: Physical geometry
            base_temperature_k: Base surface temperature
            ambient_temperature_k: Ambient environment temperature
        """
        self.name = name
        self.geometry = geometry
        self.base_temperature_k = base_temperature_k
        self.ambient_temperature_k = ambient_temperature_k
        self._hot_spots: list[HotSpot] = []
        self._materials: dict[str, MaterialProperties] = {}
        self._time_s = 0.0

    @abstractmethod
    def get_signature(
        self,
        resolution: tuple[int, int] = (64, 64),
        aspect_angle_deg: float = 0.0,
        elevation_angle_deg: float = 0.0,
    ) -> TargetSignature:
        """Generate thermal signature at given viewing angle.

        Args:
            resolution: Output resolution (height, width)
            aspect_angle_deg: Horizontal viewing angle (0=front)
            elevation_angle_deg: Vertical viewing angle (0=horizontal)

        Returns:
            TargetSignature with temperature and emissivity maps
        """
        pass

    def add_hot_spot(self, hot_spot: HotSpot) -> None:
        """Add a hot spot to the target."""
        self._hot_spots.append(hot_spot)

    def set_time(self, time_s: float) -> None:
        """Set simulation time for time-varying effects."""
        self._time_s = time_s

    def update(self, dt_s: float) -> None:
        """Update target state by time step."""
        self._time_s += dt_s

    def _apply_hot_spots(
        self,
        temperature_map: NDArray,
        base_temp: float,
    ) -> NDArray:
        """Apply hot spots to temperature map."""
        h, w = temperature_map.shape
        result = temperature_map.copy()

        for spot in self._hot_spots:
            delta_t = spot.get_temperature_delta(self._time_s)

            # Compute hot spot position and size
            cy = int(spot.relative_position[0] * h)
            cx = int(spot.relative_position[1] * w)
            radius = int(spot.relative_size * min(h, w) / 2)
            radius = max(1, radius)

            # Create mask based on shape
            yy, xx = np.ogrid[:h, :w]
            if spot.shape == "circle":
                mask = (yy - cy)**2 + (xx - cx)**2 <= radius**2
            elif spot.shape == "ellipse":
                mask = ((yy - cy) / radius)**2 + ((xx - cx) / (radius * 1.5))**2 <= 1
            else:  # rectangle
                mask = (np.abs(yy - cy) <= radius) & (np.abs(xx - cx) <= radius)

            # Apply temperature with gradient falloff
            dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
            falloff = np.clip(1 - dist / (radius * 1.5), 0, 1)
            result = np.where(mask, base_temp + delta_t * falloff[mask].reshape(-1, 1).max(), result)
            result[mask] = base_temp + delta_t

        return result

    def _create_base_shape(
        self,
        resolution: tuple[int, int],
    ) -> NDArray:
        """Create base shape mask."""
        h, w = resolution
        mask = np.zeros((h, w), dtype=bool)

        if self.geometry.shape == "box":
            # Fill rectangle
            margin_y = int(h * 0.1)
            margin_x = int(w * 0.1)
            mask[margin_y:h-margin_y, margin_x:w-margin_x] = True

        elif self.geometry.shape in ("cylinder", "ellipsoid"):
            # Fill ellipse
            cy, cx = h // 2, w // 2
            ry, rx = h // 2 - 2, w // 2 - 2
            yy, xx = np.ogrid[:h, :w]
            mask = ((yy - cy) / ry)**2 + ((xx - cx) / rx)**2 <= 1

        elif self.geometry.shape == "custom" and self.geometry.outline_points is not None:
            # Fill polygon from outline points
            try:
                import cv2
                points = (self.geometry.outline_points * [[h, w]]).astype(np.int32)
                cv2.fillPoly(mask.astype(np.uint8), [points], 1)
                mask = mask.astype(bool)
            except ImportError:
                mask[:, :] = True
        else:
            mask[:, :] = True

        return mask


class TargetRenderer:
    """Renders targets into scene temperature/emissivity maps.

    Handles placement, scaling, and composition of multiple targets
    into a scene.
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (480, 640),
        gsd_m: float = 0.5,
        background_temperature_k: float = 290.0,
        background_emissivity: float = 0.95,
    ) -> None:
        """Initialize renderer.

        Args:
            resolution: Output resolution (height, width)
            gsd_m: Ground sample distance in meters
            background_temperature_k: Background temperature
            background_emissivity: Background emissivity
        """
        self.resolution = resolution
        self.gsd_m = gsd_m
        self.background_temperature_k = background_temperature_k
        self.background_emissivity = background_emissivity
        self._rng = np.random.default_rng()

    def render(
        self,
        target: Target,
        position: tuple[int, int],
        aspect_angle_deg: float = 0.0,
        scale: float = 1.0,
    ) -> tuple[NDArray, NDArray]:
        """Render a single target into the scene.

        Args:
            target: Target to render
            position: (y, x) center position in pixels
            aspect_angle_deg: Viewing angle
            scale: Size scale factor

        Returns:
            Tuple of (temperature_map, emissivity_map)
        """
        h, w = self.resolution

        # Initialize background
        temp_map = np.full((h, w), self.background_temperature_k, dtype=np.float64)
        emis_map = np.full((h, w), self.background_emissivity, dtype=np.float64)

        # Get target signature
        target_size = self._compute_target_size(target, scale)
        signature = target.get_signature(
            resolution=target_size,
            aspect_angle_deg=aspect_angle_deg,
        )

        # Place target in scene
        self._place_target(temp_map, emis_map, signature, position)

        return temp_map, emis_map

    def render_multiple(
        self,
        targets: list[tuple[Target, tuple[int, int], float]],
    ) -> tuple[NDArray, NDArray]:
        """Render multiple targets into scene.

        Args:
            targets: List of (target, position, aspect_angle) tuples

        Returns:
            Tuple of (temperature_map, emissivity_map)
        """
        h, w = self.resolution
        temp_map = np.full((h, w), self.background_temperature_k, dtype=np.float64)
        emis_map = np.full((h, w), self.background_emissivity, dtype=np.float64)

        for target, position, aspect_angle in targets:
            target_size = self._compute_target_size(target, 1.0)
            signature = target.get_signature(
                resolution=target_size,
                aspect_angle_deg=aspect_angle,
            )
            self._place_target(temp_map, emis_map, signature, position)

        return temp_map, emis_map

    def _compute_target_size(
        self,
        target: Target,
        scale: float,
    ) -> tuple[int, int]:
        """Compute target size in pixels."""
        length_px = int(target.geometry.length_m / self.gsd_m * scale)
        width_px = int(target.geometry.width_m / self.gsd_m * scale)
        return (max(8, length_px), max(8, width_px))

    def _place_target(
        self,
        temp_map: NDArray,
        emis_map: NDArray,
        signature: TargetSignature,
        position: tuple[int, int],
    ) -> None:
        """Place target signature into scene maps."""
        h, w = temp_map.shape
        th, tw = signature.shape
        cy, cx = position

        # Compute placement bounds
        y0 = max(0, cy - th // 2)
        y1 = min(h, cy + th // 2)
        x0 = max(0, cx - tw // 2)
        x1 = min(w, cx + tw // 2)

        # Compute source bounds
        sy0 = max(0, th // 2 - cy)
        sy1 = sy0 + (y1 - y0)
        sx0 = max(0, tw // 2 - cx)
        sx1 = sx0 + (x1 - x0)

        # Place target (overwrite background)
        if y1 > y0 and x1 > x0 and sy1 > sy0 and sx1 > sx0:
            target_temp = signature.temperature_map[sy0:sy1, sx0:sx1]
            target_emis = signature.emissivity_map[sy0:sy1, sx0:sx1]

            # Only overwrite non-background pixels
            mask = target_temp != 0
            if mask.any():
                temp_map[y0:y1, x0:x1] = np.where(mask, target_temp, temp_map[y0:y1, x0:x1])
                emis_map[y0:y1, x0:x1] = np.where(mask, target_emis, emis_map[y0:y1, x0:x1])


class TargetGroup:
    """Collection of targets for group rendering.

    Manages multiple targets with relative positions and provides
    methods for randomization and animation.
    """

    def __init__(self, targets: Optional[list[Target]] = None) -> None:
        """Initialize target group.

        Args:
            targets: Initial list of targets
        """
        self.targets: list[Target] = targets or []
        self.positions: list[tuple[int, int]] = []
        self.aspects: list[float] = []
        self._rng = np.random.default_rng()

    def add(
        self,
        target: Target,
        position: tuple[int, int] = (0, 0),
        aspect_angle_deg: float = 0.0,
    ) -> None:
        """Add a target to the group."""
        self.targets.append(target)
        self.positions.append(position)
        self.aspects.append(aspect_angle_deg)

    def randomize_positions(
        self,
        area: tuple[int, int, int, int],
        min_separation: int = 20,
    ) -> None:
        """Randomize target positions within area.

        Args:
            area: (y_min, x_min, y_max, x_max) bounds
            min_separation: Minimum pixels between targets
        """
        y_min, x_min, y_max, x_max = area
        self.positions = []

        for i, target in enumerate(self.targets):
            # Try to find non-overlapping position
            for _ in range(100):
                y = self._rng.integers(y_min, y_max)
                x = self._rng.integers(x_min, x_max)

                # Check separation from existing positions
                valid = True
                for py, px in self.positions:
                    if np.sqrt((y - py)**2 + (x - px)**2) < min_separation:
                        valid = False
                        break

                if valid:
                    self.positions.append((y, x))
                    break
            else:
                # Fallback: just place it
                self.positions.append((y, x))

        # Randomize aspects
        self.aspects = [self._rng.uniform(0, 360) for _ in self.targets]

    def update(self, dt_s: float) -> None:
        """Update all targets."""
        for target in self.targets:
            target.update(dt_s)

    def get_render_list(self) -> list[tuple[Target, tuple[int, int], float]]:
        """Get list of (target, position, aspect) for rendering."""
        return list(zip(self.targets, self.positions, self.aspects))
