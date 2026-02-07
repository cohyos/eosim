"""
Scene management for EOSIM Studio.

A Scene is the 3D world containing objects (actors/props) that the camera
will capture. Objects can be static or animated along paths.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np


class ObjectMotion(Enum):
    """Motion type for scene objects."""
    STATIC = "static"
    LINEAR = "linear"  # Straight line motion
    PATH = "path"  # Follow a defined path
    ORBIT = "orbit"  # Circular orbit
    TRACKING = "tracking"  # Track another object


@dataclass
class Position3D:
    """3D position in scene coordinates (meters)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0  # altitude above ground

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_km(self) -> Tuple[float, float, float]:
        return (self.x / 1000, self.y / 1000, self.z / 1000)

    @classmethod
    def from_km(cls, x_km: float, y_km: float, z_km: float) -> "Position3D":
        return cls(x_km * 1000, y_km * 1000, z_km * 1000)

    def distance_to(self, other: "Position3D") -> float:
        """Calculate distance to another position."""
        return np.sqrt(
            (self.x - other.x)**2 +
            (self.y - other.y)**2 +
            (self.z - other.z)**2
        )


@dataclass
class Orientation3D:
    """3D orientation (degrees)."""
    heading: float = 0.0  # Yaw: 0=North, 90=East
    pitch: float = 0.0    # Nose up/down
    roll: float = 0.0     # Bank angle

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.heading, self.pitch, self.roll)


@dataclass
class MotionPath:
    """Defines a motion path for an object."""
    waypoints: List[Tuple[float, Position3D]] = field(default_factory=list)  # (time_sec, position)
    orientations: List[Tuple[float, Orientation3D]] = field(default_factory=list)  # (time_sec, orientation)
    speed_mps: float = 0.0  # For LINEAR motion
    interpolation: str = "linear"  # "linear", "smooth", "step"

    def add_waypoint(self, time_sec: float, position: Position3D,
                     orientation: Optional[Orientation3D] = None):
        """Add a waypoint to the path."""
        self.waypoints.append((time_sec, position))
        if orientation:
            self.orientations.append((time_sec, orientation))

    def get_position_at(self, time_sec: float) -> Position3D:
        """Get interpolated position at given time."""
        if not self.waypoints:
            return Position3D()

        if len(self.waypoints) == 1:
            return self.waypoints[0][1]

        # Find surrounding waypoints
        prev_wp = self.waypoints[0]
        next_wp = self.waypoints[-1]

        for i, (t, pos) in enumerate(self.waypoints):
            if t >= time_sec:
                next_wp = (t, pos)
                if i > 0:
                    prev_wp = self.waypoints[i - 1]
                break
            prev_wp = (t, pos)

        # Clamp to endpoints
        if time_sec <= self.waypoints[0][0]:
            return self.waypoints[0][1]
        if time_sec >= self.waypoints[-1][0]:
            return self.waypoints[-1][1]

        # Interpolate
        t0, p0 = prev_wp
        t1, p1 = next_wp

        if t1 == t0:
            return p0

        alpha = (time_sec - t0) / (t1 - t0)

        if self.interpolation == "smooth":
            # Smoothstep interpolation
            alpha = alpha * alpha * (3 - 2 * alpha)
        elif self.interpolation == "step":
            alpha = 0.0 if alpha < 0.5 else 1.0

        return Position3D(
            x=p0.x + alpha * (p1.x - p0.x),
            y=p0.y + alpha * (p1.y - p0.y),
            z=p0.z + alpha * (p1.z - p0.z)
        )

    def get_orientation_at(self, time_sec: float) -> Orientation3D:
        """Get interpolated orientation at given time."""
        if not self.orientations:
            # Auto-calculate from path direction
            return self._auto_orientation(time_sec)

        if len(self.orientations) == 1:
            return self.orientations[0][1]

        # Similar interpolation as position
        prev = self.orientations[0]
        next_o = self.orientations[-1]

        for i, (t, ori) in enumerate(self.orientations):
            if t >= time_sec:
                next_o = (t, ori)
                if i > 0:
                    prev = self.orientations[i - 1]
                break
            prev = (t, ori)

        if time_sec <= self.orientations[0][0]:
            return self.orientations[0][1]
        if time_sec >= self.orientations[-1][0]:
            return self.orientations[-1][1]

        t0, o0 = prev
        t1, o1 = next_o

        if t1 == t0:
            return o0

        alpha = (time_sec - t0) / (t1 - t0)

        return Orientation3D(
            heading=o0.heading + alpha * (o1.heading - o0.heading),
            pitch=o0.pitch + alpha * (o1.pitch - o0.pitch),
            roll=o0.roll + alpha * (o1.roll - o0.roll)
        )

    def _auto_orientation(self, time_sec: float) -> Orientation3D:
        """Auto-calculate orientation from motion direction."""
        dt = 0.1
        p0 = self.get_position_at(time_sec)
        p1 = self.get_position_at(time_sec + dt)

        dx = p1.x - p0.x
        dy = p1.y - p0.y
        dz = p1.z - p0.z

        # Calculate heading from dx, dy
        if abs(dx) < 0.001 and abs(dy) < 0.001:
            heading = 0.0
        else:
            heading = np.degrees(np.arctan2(dx, dy)) % 360

        # Calculate pitch from dz and horizontal distance
        horizontal = np.sqrt(dx**2 + dy**2)
        if horizontal < 0.001:
            pitch = 0.0
        else:
            pitch = np.degrees(np.arctan2(dz, horizontal))

        return Orientation3D(heading=heading, pitch=pitch, roll=0.0)


@dataclass
class SceneObject:
    """An object in the scene (actor/prop).

    Attributes:
        id: Unique identifier for this object instance
        object_type: Type from object library (e.g., "f16", "m1_abrams")
        name: User-friendly display name
        position: Current or initial position
        orientation: Current or initial orientation
        motion: Motion type
        motion_path: Path definition for animated objects
        visible: Whether object is visible
        metadata: Additional properties
    """
    id: str
    object_type: str  # From object library
    name: str = ""
    position: Position3D = field(default_factory=Position3D)
    orientation: Orientation3D = field(default_factory=Orientation3D)
    motion: ObjectMotion = ObjectMotion.STATIC
    motion_path: Optional[MotionPath] = None
    visible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.object_type}_{self.id[:4]}"

    def get_position_at(self, time_sec: float) -> Position3D:
        """Get object position at given time."""
        if self.motion == ObjectMotion.STATIC:
            return self.position
        elif self.motion_path:
            return self.motion_path.get_position_at(time_sec)
        else:
            return self.position

    def get_orientation_at(self, time_sec: float) -> Orientation3D:
        """Get object orientation at given time."""
        if self.motion == ObjectMotion.STATIC:
            return self.orientation
        elif self.motion_path:
            return self.motion_path.get_orientation_at(time_sec)
        else:
            return self.orientation

    def set_linear_motion(self, end_position: Position3D,
                          start_time: float = 0.0, end_time: float = 10.0):
        """Set up linear motion from current position to end position."""
        self.motion = ObjectMotion.LINEAR
        self.motion_path = MotionPath(interpolation="linear")
        self.motion_path.add_waypoint(start_time, self.position)
        self.motion_path.add_waypoint(end_time, end_position)


class Scene:
    """A scene containing objects for the camera to capture.

    The scene is the 3D world where objects exist and move. It provides:
    - Object management (add, remove, find)
    - Terrain configuration (geographic location, elevation, land cover)
    - Time-based state queries
    - Scene bounds and statistics

    Example:
        >>> scene = Scene("Combat Zone")
        >>> scene.set_terrain("mojave_desert", radius_m=5000, detail="low")
        >>> scene.add_object("tank1", "m1_abrams", Position3D(1000, 500, 0))
        >>> scene.add_object("jet1", "f16", Position3D(0, 0, 5000))
        >>> scene.get_objects_at(5.0)  # Get all object states at t=5s
    """

    def __init__(self, name: str = "Untitled Scene"):
        self.name = name
        self.description: str = ""
        self.objects: Dict[str, SceneObject] = {}
        self.duration_sec: float = 30.0  # Default scene duration
        self.ground_level: float = 0.0
        self.ambient_temperature_k: float = 290.0
        self._next_id = 1

        # Terrain configuration
        self._terrain_data = None
        self._terrain_provider = None

    def set_terrain(self, location: str = "mojave_desert",
                    radius_m: float = 5000.0,
                    detail: str = "low",
                    time_of_day: float = 12.0):
        """Configure terrain for the scene.

        Args:
            location: Preset name or "lat,lon" string. Available presets:
                - mojave_desert, persian_gulf, central_europe, korean_peninsula
                - sahara, arctic, pacific_islands, amazon, himalaya, great_plains
            radius_m: Terrain radius in meters
            detail: Detail level - "low" (100m), "medium" (30m), "high" (10m)
            time_of_day: Hour of day (0-24) affects thermal properties

        Example:
            >>> scene.set_terrain("persian_gulf", radius_m=10000, detail="medium")
            >>> scene.set_terrain("35.0,-116.0", radius_m=5000)  # Custom lat/lon
        """
        # Lazy import to avoid circular dependencies
        from eosim.studio.terrain import TerrainProvider, TerrainConfig, GeoLocation, DetailLevel

        if self._terrain_provider is None:
            self._terrain_provider = TerrainProvider()

        # Parse location
        if location in self._terrain_provider.LOCATION_PRESETS:
            geo = self._terrain_provider.LOCATION_PRESETS[location]
        elif "," in location:
            parts = location.split(",")
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
            geo = GeoLocation(lat, lon, f"Custom ({lat:.2f}, {lon:.2f})")
        else:
            raise ValueError(f"Unknown location: {location}")

        # Parse detail
        detail_map = {"low": DetailLevel.LOW, "medium": DetailLevel.MEDIUM, "high": DetailLevel.HIGH}
        detail_level = detail_map.get(detail.lower(), DetailLevel.LOW)

        # Create config and load terrain
        config = TerrainConfig(
            center=geo,
            radius_m=radius_m,
            detail_level=detail_level,
            time_of_day=time_of_day
        )

        self._terrain_data = self._terrain_provider.load_terrain(config)

        # Update scene properties based on terrain
        self.ambient_temperature_k = 290.0  # Will be overridden by terrain thermal map
        self.ground_level = self._terrain_data.min_elevation

    def get_terrain(self):
        """Get the terrain data, or None if not configured."""
        return self._terrain_data

    def get_terrain_presets(self) -> List[str]:
        """Get list of available terrain preset names."""
        from eosim.studio.terrain import TerrainProvider
        if self._terrain_provider is None:
            self._terrain_provider = TerrainProvider()
        return list(self._terrain_provider.LOCATION_PRESETS.keys())

    def get_elevation_at(self, x: float, y: float) -> float:
        """Get terrain elevation at local coordinates.

        Returns ground_level if no terrain is configured.
        """
        if self._terrain_data is not None:
            return self._terrain_data.get_elevation_at(x, y)
        return self.ground_level

    def add_object(self, object_type: str,
                   position: Optional[Position3D] = None,
                   name: Optional[str] = None,
                   orientation: Optional[Orientation3D] = None) -> SceneObject:
        """Add an object to the scene.

        Args:
            object_type: Type from library (e.g., "f16", "m1_abrams")
            position: Initial position (default: origin)
            name: Display name (default: auto-generated)
            orientation: Initial orientation (default: facing north)

        Returns:
            The created SceneObject
        """
        obj_id = f"obj_{self._next_id:04d}"
        self._next_id += 1

        obj = SceneObject(
            id=obj_id,
            object_type=object_type,
            name=name or f"{object_type}_{obj_id[-4:]}",
            position=position or Position3D(),
            orientation=orientation or Orientation3D()
        )

        self.objects[obj_id] = obj
        return obj

    def remove_object(self, obj_id: str) -> bool:
        """Remove an object from the scene."""
        if obj_id in self.objects:
            del self.objects[obj_id]
            return True
        return False

    def get_object(self, obj_id: str) -> Optional[SceneObject]:
        """Get an object by ID."""
        return self.objects.get(obj_id)

    def find_objects(self, object_type: Optional[str] = None,
                     name_contains: Optional[str] = None) -> List[SceneObject]:
        """Find objects matching criteria."""
        results = []
        for obj in self.objects.values():
            if object_type and obj.object_type != object_type:
                continue
            if name_contains and name_contains.lower() not in obj.name.lower():
                continue
            results.append(obj)
        return results

    def get_object_states_at(self, time_sec: float) -> List[Dict[str, Any]]:
        """Get all object states at a given time.

        Returns list of dicts with position, orientation, visibility for each object.
        """
        states = []
        for obj in self.objects.values():
            if not obj.visible:
                continue
            states.append({
                "id": obj.id,
                "name": obj.name,
                "type": obj.object_type,
                "position": obj.get_position_at(time_sec),
                "orientation": obj.get_orientation_at(time_sec),
            })
        return states

    def get_bounds(self) -> Tuple[Position3D, Position3D]:
        """Get scene bounding box (min, max corners)."""
        if not self.objects:
            return Position3D(-1000, -1000, 0), Position3D(1000, 1000, 1000)

        positions = [obj.position for obj in self.objects.values()]

        min_pos = Position3D(
            x=min(p.x for p in positions) - 100,
            y=min(p.y for p in positions) - 100,
            z=min(p.z for p in positions)
        )
        max_pos = Position3D(
            x=max(p.x for p in positions) + 100,
            y=max(p.y for p in positions) + 100,
            z=max(p.z for p in positions) + 100
        )

        return min_pos, max_pos

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scene to dictionary."""
        return {
            "name": self.name,
            "duration_sec": self.duration_sec,
            "ground_level": self.ground_level,
            "ambient_temperature_k": self.ambient_temperature_k,
            "objects": [
                {
                    "id": obj.id,
                    "object_type": obj.object_type,
                    "name": obj.name,
                    "position": obj.position.to_tuple(),
                    "orientation": obj.orientation.to_tuple(),
                    "motion": obj.motion.value,
                    "visible": obj.visible,
                }
                for obj in self.objects.values()
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scene":
        """Deserialize scene from dictionary."""
        scene = cls(data.get("name", "Untitled"))
        scene.duration_sec = data.get("duration_sec", 30.0)
        scene.ground_level = data.get("ground_level", 0.0)
        scene.ambient_temperature_k = data.get("ambient_temperature_k", 290.0)

        for obj_data in data.get("objects", []):
            pos = obj_data.get("position", (0, 0, 0))
            ori = obj_data.get("orientation", (0, 0, 0))

            obj = SceneObject(
                id=obj_data["id"],
                object_type=obj_data["object_type"],
                name=obj_data.get("name", ""),
                position=Position3D(*pos),
                orientation=Orientation3D(*ori),
                motion=ObjectMotion(obj_data.get("motion", "static")),
                visible=obj_data.get("visible", True)
            )
            scene.objects[obj.id] = obj
            scene._next_id = max(scene._next_id, int(obj.id.split("_")[1]) + 1)

        return scene
