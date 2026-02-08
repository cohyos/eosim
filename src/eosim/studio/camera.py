"""
Camera system for EOSIM Studio.

The Camera is the user-facing abstraction for the sensor. Instead of
technical parameters like FOV and NETD, users work with intuitive
concepts like "lens type" and "sensitivity".

The camera can:
- Be positioned in the scene
- Follow a path (dolly/crane motion)
- Track objects (follow mode)
- Have various "lens" settings
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np

from eosim.studio.scene import Position3D, Orientation3D, MotionPath


class LensType(Enum):
    """Camera lens presets (maps to FOV)."""
    ULTRA_WIDE = "ultra_wide"  # 90+ degrees
    WIDE = "wide"              # 60-90 degrees
    NORMAL = "normal"          # 30-60 degrees
    TELEPHOTO = "telephoto"    # 10-30 degrees
    SUPER_TELEPHOTO = "super_telephoto"  # < 10 degrees


class SpectrumMode(Enum):
    """Imaging spectrum mode."""
    THERMAL_LWIR = "thermal_lwir"  # Long-wave IR (8-14 um)
    THERMAL_MWIR = "thermal_mwir"  # Mid-wave IR (3-5 um)
    NIGHT_VISION = "night_vision"  # Near IR / image intensified
    VISIBLE = "visible"            # Visible light
    MULTI_SPECTRAL = "multi"       # Combined


class SensitivityLevel(Enum):
    """Camera sensitivity presets (maps to NETD/noise)."""
    LOW = "low"        # High noise, fast
    MEDIUM = "medium"  # Balanced
    HIGH = "high"      # Low noise, good detail
    ULTRA = "ultra"    # Very low noise, max detail


class CameraMode(Enum):
    """Camera operation mode."""
    FIXED = "fixed"          # Static position and orientation
    PATH = "path"            # Follow a defined path
    TRACKING = "tracking"    # Track a target object
    ORBIT = "orbit"          # Orbit around a point


@dataclass
class CameraPreset:
    """Predefined camera configuration."""
    name: str
    lens: LensType = LensType.NORMAL
    spectrum: SpectrumMode = SpectrumMode.THERMAL_LWIR
    sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM
    resolution: Tuple[int, int] = (640, 480)
    description: str = ""

    # Technical overrides (for power users)
    fov_deg: Optional[float] = None
    netd_mk: Optional[float] = None

    def get_fov(self) -> float:
        """Get actual FOV in degrees."""
        if self.fov_deg is not None:
            return self.fov_deg
        # Map lens type to FOV
        lens_fov = {
            LensType.ULTRA_WIDE: 100.0,
            LensType.WIDE: 70.0,
            LensType.NORMAL: 45.0,
            LensType.TELEPHOTO: 15.0,
            LensType.SUPER_TELEPHOTO: 5.0,
        }
        return lens_fov.get(self.lens, 45.0)

    def get_netd(self) -> float:
        """Get actual NETD in millikelvin."""
        if self.netd_mk is not None:
            return self.netd_mk
        # Map sensitivity to NETD
        sens_netd = {
            SensitivityLevel.LOW: 100.0,
            SensitivityLevel.MEDIUM: 50.0,
            SensitivityLevel.HIGH: 25.0,
            SensitivityLevel.ULTRA: 10.0,
        }
        return sens_netd.get(self.sensitivity, 50.0)


# Built-in presets
CAMERA_PRESETS = {
    "surveillance": CameraPreset(
        name="Surveillance Camera",
        lens=LensType.NORMAL,
        spectrum=SpectrumMode.THERMAL_LWIR,
        sensitivity=SensitivityLevel.HIGH,
        resolution=(640, 480),
        description="General surveillance thermal camera"
    ),
    "targeting": CameraPreset(
        name="Targeting Pod",
        lens=LensType.TELEPHOTO,
        spectrum=SpectrumMode.THERMAL_MWIR,
        sensitivity=SensitivityLevel.ULTRA,
        resolution=(1024, 768),
        description="High-resolution targeting system"
    ),
    "wide_area": CameraPreset(
        name="Wide Area Search",
        lens=LensType.WIDE,
        spectrum=SpectrumMode.THERMAL_LWIR,
        sensitivity=SensitivityLevel.MEDIUM,
        resolution=(1920, 1080),
        description="Wide area surveillance"
    ),
    "night_vision": CameraPreset(
        name="Night Vision",
        lens=LensType.NORMAL,
        spectrum=SpectrumMode.NIGHT_VISION,
        sensitivity=SensitivityLevel.HIGH,
        resolution=(1280, 720),
        description="Image intensified night vision"
    ),
    "hd_thermal": CameraPreset(
        name="HD Thermal",
        lens=LensType.NORMAL,
        spectrum=SpectrumMode.THERMAL_LWIR,
        sensitivity=SensitivityLevel.HIGH,
        resolution=(1920, 1080),
        description="High-definition thermal imaging"
    ),
}


@dataclass
class CameraPath:
    """Camera motion path (like a dolly or crane)."""
    path: MotionPath = field(default_factory=MotionPath)
    look_at_target: Optional[str] = None  # Object ID to look at
    smooth_motion: bool = True
    smooth_factor: float = 0.8  # 0 = no smoothing, 1 = max smoothing

    def add_keyframe(self, time_sec: float, position: Position3D,
                     orientation: Optional[Orientation3D] = None):
        """Add a camera keyframe."""
        self.path.add_waypoint(time_sec, position, orientation)

    def get_position_at(self, time_sec: float) -> Position3D:
        """Get camera position at time."""
        return self.path.get_position_at(time_sec)

    def get_orientation_at(self, time_sec: float,
                          scene_objects: Optional[Dict] = None) -> Orientation3D:
        """Get camera orientation at time."""
        if self.look_at_target and scene_objects:
            # Calculate look-at orientation
            target = scene_objects.get(self.look_at_target)
            if target:
                cam_pos = self.get_position_at(time_sec)
                target_pos = target.get_position_at(time_sec)
                return self._calculate_look_at(cam_pos, target_pos)

        return self.path.get_orientation_at(time_sec)

    def _calculate_look_at(self, from_pos: Position3D,
                          to_pos: Position3D) -> Orientation3D:
        """Calculate orientation to look at target."""
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        dz = to_pos.z - from_pos.z

        # Heading (azimuth)
        if abs(dx) < 0.001 and abs(dy) < 0.001:
            heading = 0.0
        else:
            heading = np.degrees(np.arctan2(dx, dy)) % 360

        # Pitch (elevation)
        horizontal = np.sqrt(dx**2 + dy**2)
        if horizontal < 0.001:
            pitch = -90.0 if dz < 0 else 90.0
        else:
            pitch = np.degrees(np.arctan2(-dz, horizontal))

        return Orientation3D(heading=heading, pitch=pitch, roll=0.0)


class Camera:
    """The virtual camera (sensor abstraction).

    The Camera provides a user-friendly interface to the underlying
    sensor simulation. Users think in terms of:
    - Lens (wide/telephoto) instead of FOV
    - Sensitivity (low/high) instead of NETD
    - Position and look-at instead of gimbal angles

    Example:
        >>> camera = Camera("Main Camera")
        >>> camera.set_preset("targeting")
        >>> camera.set_position(Position3D(0, 0, 5000))
        >>> camera.look_at(Position3D(1000, 500, 0))
    """

    def __init__(self, name: str = "Camera"):
        self.name = name

        # Current settings
        self.lens: LensType = LensType.NORMAL
        self.spectrum: SpectrumMode = SpectrumMode.THERMAL_LWIR
        self.sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM
        self.resolution: Tuple[int, int] = (640, 480)
        self.stabilization: bool = True

        # Position and orientation
        self.position: Position3D = Position3D(0, 0, 1000)
        self.orientation: Orientation3D = Orientation3D(0, -45, 0)

        # Motion mode
        self.mode: CameraMode = CameraMode.FIXED
        self.camera_path: Optional[CameraPath] = None
        self.tracking_target: Optional[str] = None  # Object ID

        # Technical overrides (for advanced users)
        self._fov_override: Optional[float] = None
        self._netd_override: Optional[float] = None

        # Colormap for display
        self.colormap: str = "iron"  # iron, rainbow, grayscale, etc.

    def set_preset(self, preset_name: str):
        """Apply a camera preset."""
        preset = CAMERA_PRESETS.get(preset_name)
        if preset:
            self.lens = preset.lens
            self.spectrum = preset.spectrum
            self.sensitivity = preset.sensitivity
            self.resolution = preset.resolution
            self._fov_override = preset.fov_deg
            self._netd_override = preset.netd_mk

    def set_position(self, position: Position3D):
        """Set camera position."""
        self.position = position
        if self.mode == CameraMode.FIXED:
            pass  # Just update position
        elif self.camera_path:
            # Add to path at current time
            pass

    def look_at(self, target: Position3D):
        """Point camera at a position."""
        dx = target.x - self.position.x
        dy = target.y - self.position.y
        dz = target.z - self.position.z

        # Calculate heading
        if abs(dx) < 0.001 and abs(dy) < 0.001:
            heading = self.orientation.heading
        else:
            heading = np.degrees(np.arctan2(dx, dy)) % 360

        # Calculate pitch
        horizontal = np.sqrt(dx**2 + dy**2)
        if horizontal < 0.001:
            pitch = -90.0 if dz < 0 else 90.0
        else:
            pitch = np.degrees(np.arctan2(-dz, horizontal))

        self.orientation = Orientation3D(heading=heading, pitch=pitch, roll=0.0)

    def track_object(self, object_id: str):
        """Set camera to track an object."""
        self.mode = CameraMode.TRACKING
        self.tracking_target = object_id

    def set_path(self, path: CameraPath):
        """Set camera motion path."""
        self.mode = CameraMode.PATH
        self.camera_path = path

    def get_position_at(self, time_sec: float) -> Position3D:
        """Get camera position at given time."""
        if self.mode == CameraMode.FIXED:
            return self.position
        elif self.mode == CameraMode.PATH and self.camera_path:
            return self.camera_path.get_position_at(time_sec)
        else:
            return self.position

    def get_orientation_at(self, time_sec: float,
                          scene_objects: Optional[Dict] = None) -> Orientation3D:
        """Get camera orientation at given time."""
        if self.mode == CameraMode.TRACKING and self.tracking_target and scene_objects:
            target = scene_objects.get(self.tracking_target)
            if target:
                cam_pos = self.get_position_at(time_sec)
                target_pos = target.get_position_at(time_sec)
                return self._calculate_look_at(cam_pos, target_pos)

        if self.mode == CameraMode.PATH and self.camera_path:
            return self.camera_path.get_orientation_at(time_sec, scene_objects)

        return self.orientation

    def _calculate_look_at(self, from_pos: Position3D,
                          to_pos: Position3D) -> Orientation3D:
        """Calculate orientation to look at a point."""
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        dz = to_pos.z - from_pos.z

        if abs(dx) < 0.001 and abs(dy) < 0.001:
            heading = 0.0
        else:
            heading = np.degrees(np.arctan2(dx, dy)) % 360

        horizontal = np.sqrt(dx**2 + dy**2)
        if horizontal < 0.001:
            pitch = -90.0 if dz < 0 else 90.0
        else:
            pitch = np.degrees(np.arctan2(-dz, horizontal))

        return Orientation3D(heading=heading, pitch=pitch, roll=0.0)

    def get_fov(self) -> float:
        """Get field of view in degrees."""
        if self._fov_override is not None:
            return self._fov_override
        lens_fov = {
            LensType.ULTRA_WIDE: 100.0,
            LensType.WIDE: 70.0,
            LensType.NORMAL: 45.0,
            LensType.TELEPHOTO: 15.0,
            LensType.SUPER_TELEPHOTO: 5.0,
        }
        return lens_fov.get(self.lens, 45.0)

    def get_netd(self) -> float:
        """Get NETD in millikelvin."""
        if self._netd_override is not None:
            return self._netd_override
        sens_netd = {
            SensitivityLevel.LOW: 100.0,
            SensitivityLevel.MEDIUM: 50.0,
            SensitivityLevel.HIGH: 25.0,
            SensitivityLevel.ULTRA: 10.0,
        }
        return sens_netd.get(self.sensitivity, 50.0)

    def set_fov(self, fov_deg: float):
        """Set FOV directly (advanced)."""
        self._fov_override = fov_deg
        # Update lens type to match
        if fov_deg >= 90:
            self.lens = LensType.ULTRA_WIDE
        elif fov_deg >= 60:
            self.lens = LensType.WIDE
        elif fov_deg >= 30:
            self.lens = LensType.NORMAL
        elif fov_deg >= 10:
            self.lens = LensType.TELEPHOTO
        else:
            self.lens = LensType.SUPER_TELEPHOTO

    def set_netd(self, netd_mk: float):
        """Set NETD directly (advanced)."""
        self._netd_override = netd_mk
        # Update sensitivity level to match
        if netd_mk >= 75:
            self.sensitivity = SensitivityLevel.LOW
        elif netd_mk >= 35:
            self.sensitivity = SensitivityLevel.MEDIUM
        elif netd_mk >= 15:
            self.sensitivity = SensitivityLevel.HIGH
        else:
            self.sensitivity = SensitivityLevel.ULTRA

    def to_dict(self) -> Dict[str, Any]:
        """Serialize camera settings."""
        return {
            "name": self.name,
            "lens": self.lens.value,
            "spectrum": self.spectrum.value,
            "sensitivity": self.sensitivity.value,
            "resolution": self.resolution,
            "stabilization": self.stabilization,
            "position": self.position.to_tuple(),
            "orientation": self.orientation.to_tuple(),
            "mode": self.mode.value,
            "tracking_target": self.tracking_target,
            "fov_override": self._fov_override,
            "netd_override": self._netd_override,
            "colormap": self.colormap,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Camera":
        """Deserialize camera from dictionary."""
        camera = cls(data.get("name", "Camera"))
        camera.lens = LensType(data.get("lens", "normal"))
        camera.spectrum = SpectrumMode(data.get("spectrum", "thermal_lwir"))
        camera.sensitivity = SensitivityLevel(data.get("sensitivity", "medium"))
        camera.resolution = tuple(data.get("resolution", (640, 480)))
        camera.stabilization = data.get("stabilization", True)

        pos = data.get("position", (0, 0, 1000))
        camera.position = Position3D(*pos)

        ori = data.get("orientation", (0, -45, 0))
        camera.orientation = Orientation3D(*ori)

        camera.mode = CameraMode(data.get("mode", "fixed"))
        camera.tracking_target = data.get("tracking_target")
        camera._fov_override = data.get("fov_override")
        camera._netd_override = data.get("netd_override")
        camera.colormap = data.get("colormap", "iron")

        return camera
