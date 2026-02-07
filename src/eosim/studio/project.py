"""
Project management for EOSIM Studio.

A Project is like a video project file that contains:
- Scene (the 3D world with objects)
- Camera settings
- Timeline (animation keyframes)
- Render settings

Projects can be saved/loaded as JSON files.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from eosim.studio.scene import Scene, Position3D
from eosim.studio.camera import Camera
from eosim.studio.timeline import Timeline


@dataclass
class RenderSettings:
    """Settings for video export."""
    output_format: str = "mp4"  # mp4, avi, frames
    resolution: tuple = (1920, 1080)
    fps: float = 30.0
    quality: str = "high"  # low, medium, high, lossless
    colormap: str = "iron"
    include_overlay: bool = True  # Timestamp, camera info
    include_crosshair: bool = False
    output_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_format": self.output_format,
            "resolution": self.resolution,
            "fps": self.fps,
            "quality": self.quality,
            "colormap": self.colormap,
            "include_overlay": self.include_overlay,
            "include_crosshair": self.include_crosshair,
            "output_path": self.output_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RenderSettings":
        return cls(
            output_format=data.get("output_format", "mp4"),
            resolution=tuple(data.get("resolution", (1920, 1080))),
            fps=data.get("fps", 30.0),
            quality=data.get("quality", "high"),
            colormap=data.get("colormap", "iron"),
            include_overlay=data.get("include_overlay", True),
            include_crosshair=data.get("include_crosshair", False),
            output_path=data.get("output_path", ""),
        )


class Project:
    """EOSIM Studio project.

    A project contains all the elements needed to create a sensor simulation video:
    - Scene with objects (targets)
    - Camera (sensor) configuration
    - Timeline with animation
    - Render settings

    Example:
        >>> project = Project("Mission Demo")
        >>> project.scene.add_object("f16", Position3D(5000, 0, 3000))
        >>> project.camera.set_preset("targeting")
        >>> project.timeline.duration_sec = 60.0
        >>> project.save("mission_demo.eosim")
    """

    FILE_EXTENSION = ".eosim"
    FILE_VERSION = "1.0"

    def __init__(self, name: str = "Untitled Project"):
        self.name = name
        self.created_at = datetime.now()
        self.modified_at = datetime.now()
        self.file_path: Optional[str] = None

        # Core components
        self.scene = Scene(f"{name} Scene")
        self.camera = Camera("Main Camera")
        self.timeline = Timeline(duration_sec=30.0, fps=30.0)
        self.render_settings = RenderSettings()

        # Project metadata
        self.description: str = ""
        self.author: str = ""
        self.tags: List[str] = []

    @property
    def duration(self) -> float:
        """Project duration in seconds."""
        return self.timeline.duration_sec

    @duration.setter
    def duration(self, value: float):
        """Set project duration."""
        self.timeline.duration_sec = value
        self.scene.duration_sec = value

    @property
    def fps(self) -> float:
        """Project frame rate."""
        return self.timeline.fps

    @fps.setter
    def fps(self, value: float):
        """Set project frame rate."""
        self.timeline.fps = value
        self.render_settings.fps = value

    def new_scene(self, name: Optional[str] = None):
        """Create a new empty scene."""
        self.scene = Scene(name or f"{self.name} Scene")
        self.timeline = Timeline(duration_sec=30.0, fps=self.fps)
        self.modified_at = datetime.now()

    def add_object(self, object_type: str,
                   position: Optional[Position3D] = None,
                   name: Optional[str] = None):
        """Shortcut to add an object to the scene."""
        obj = self.scene.add_object(object_type, position, name)
        self.modified_at = datetime.now()
        return obj

    def set_camera_position(self, x: float, y: float, z: float):
        """Set camera position (in meters)."""
        self.camera.position = Position3D(x, y, z)
        self.modified_at = datetime.now()

    def point_camera_at(self, x: float, y: float, z: float):
        """Point camera at a location."""
        self.camera.look_at(Position3D(x, y, z))
        self.modified_at = datetime.now()

    def save(self, file_path: Optional[str] = None) -> str:
        """Save project to file.

        Args:
            file_path: Path to save to. If None, uses existing path or creates new.

        Returns:
            Path where file was saved.
        """
        if file_path:
            self.file_path = file_path
        elif not self.file_path:
            # Generate default filename
            safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in self.name)
            self.file_path = f"{safe_name}{self.FILE_EXTENSION}"

        self.modified_at = datetime.now()

        data = self.to_dict()

        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return self.file_path

    @classmethod
    def load(cls, file_path: str) -> "Project":
        """Load project from file.

        Args:
            file_path: Path to the project file.

        Returns:
            Loaded Project instance.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        project = cls.from_dict(data)
        project.file_path = file_path

        return project

    def to_dict(self) -> Dict[str, Any]:
        """Serialize project to dictionary."""
        return {
            "version": self.FILE_VERSION,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "description": self.description,
            "author": self.author,
            "tags": self.tags,
            "scene": self.scene.to_dict(),
            "camera": self.camera.to_dict(),
            "timeline": self.timeline.to_dict(),
            "render_settings": self.render_settings.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        """Deserialize project from dictionary."""
        project = cls(data.get("name", "Untitled"))

        # Parse dates
        if "created_at" in data:
            try:
                project.created_at = datetime.fromisoformat(data["created_at"])
            except (ValueError, TypeError):
                pass

        if "modified_at" in data:
            try:
                project.modified_at = datetime.fromisoformat(data["modified_at"])
            except (ValueError, TypeError):
                pass

        project.description = data.get("description", "")
        project.author = data.get("author", "")
        project.tags = data.get("tags", [])

        # Load components
        if "scene" in data:
            project.scene = Scene.from_dict(data["scene"])
        if "camera" in data:
            project.camera = Camera.from_dict(data["camera"])
        if "timeline" in data:
            project.timeline = Timeline.from_dict(data["timeline"])
        if "render_settings" in data:
            project.render_settings = RenderSettings.from_dict(data["render_settings"])

        return project

    def create_demo_scene(self) -> "Project":
        """Populate project with demo content.

        Creates a sample scene with objects, camera path, and animation
        to demonstrate EOSIM Studio capabilities.
        """
        self.name = "Demo Scene"
        self.description = "Demonstration of EOSIM Studio features"
        self.duration = 20.0  # 20 second demo

        # Clear existing
        self.scene = Scene("Demo Combat Zone")
        self.scene.duration_sec = self.duration

        # Add ground targets
        tank = self.scene.add_object("m1_abrams", Position3D(2000, 1000, 0), "Tank Alpha")
        humvee = self.scene.add_object("humvee", Position3D(2200, 900, 0), "Support Vehicle")

        # Add aircraft
        jet = self.scene.add_object("f16", Position3D(-5000, 0, 3000), "Strike Fighter")

        # Animate the jet to fly over the scene
        from eosim.studio.scene import MotionPath, ObjectMotion
        jet.motion = ObjectMotion.PATH
        jet.motion_path = MotionPath(interpolation="smooth")
        jet.motion_path.add_waypoint(0.0, Position3D(-5000, 0, 3000))
        jet.motion_path.add_waypoint(10.0, Position3D(2000, 1000, 3000))
        jet.motion_path.add_waypoint(20.0, Position3D(8000, 2000, 3500))

        # Set up camera
        self.camera.set_preset("targeting")
        self.camera.position = Position3D(0, -500, 4000)
        self.camera.track_object(tank.id)  # Track the tank

        # Add timeline markers
        self.timeline.add_marker(0.0, "Start", "#00ff00")
        self.timeline.add_marker(10.0, "Flyover", "#ffcc00")
        self.timeline.add_marker(20.0, "End", "#ff0000")

        self.modified_at = datetime.now()
        return self

    def __repr__(self) -> str:
        return (
            f"Project(name='{self.name}', "
            f"objects={len(self.scene.objects)}, "
            f"duration={self.duration:.1f}s)"
        )


def create_new_project(name: str = "Untitled") -> Project:
    """Create a new empty project."""
    return Project(name)


def create_demo_project() -> Project:
    """Create a demo project with sample content."""
    return Project("Demo").create_demo_scene()
