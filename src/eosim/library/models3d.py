"""
3D Model Library for EOSIM.

Downloads and manages open source 3D models for visualization.
Supports OBJ, STL, and PLY formats from various repositories.

Usage:
------
>>> from eosim.library.models3d import ModelLibrary
>>> library = ModelLibrary()
>>> library.download_model("f16")  # Downloads F-16 model
>>> mesh = library.load_model("f16")
>>> mesh.render_to_image((256, 256), azimuth=45)
"""

import os
import json
import urllib.request
import zipfile
import tempfile
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import numpy as np
from numpy.typing import NDArray


# Model sources - URLs to free/open source 3D models
# These are placeholder URLs - actual models would come from sites like:
# - Sketchfab (CC licensed models)
# - GrabCAD (CAD models)
# - Clara.io
# - Free3D
# - TurboSquid free section

MODEL_SOURCES = {
    # Aircraft
    "f16": {
        "name": "F-16 Fighting Falcon",
        "category": "aircraft",
        "format": "obj",
        "dimensions": {"length": 15.0, "width": 9.5, "height": 5.1},
        "source": "procedural",
        "source_url": "https://www.turbosquid.com/Search/3D-Models/free/f-16",
        "license": "CC0",
        "siso_id": "1.2.225.1.1.3",
    },
    "f35": {
        "name": "F-35 Lightning II",
        "category": "aircraft",
        "format": "obj",
        "dimensions": {"length": 15.7, "width": 10.7, "height": 4.4},
        "source": "procedural",
        "source_url": "https://free3d.com/3d-model/f-35-lightning-ii-5462.html",
        "license": "CC0",
        "siso_id": "1.2.225.1.1.6",
    },
    "su27": {
        "name": "Su-27 Flanker",
        "category": "aircraft",
        "format": "obj",
        "dimensions": {"length": 21.9, "width": 14.7, "height": 5.9},
        "source": "procedural",
        "source_url": None,
        "license": "CC0",
        "siso_id": "1.2.222.1.2.1",
    },
    "mig29": {
        "name": "MiG-29 Fulcrum",
        "category": "aircraft",
        "format": "obj",
        "dimensions": {"length": 17.3, "width": 11.4, "height": 4.7},
        "source": "procedural",
        "source_url": "https://opengameart.org/content/mig",
        "license": "CC0",
        "siso_id": "1.2.222.1.2.2",
    },
    "rafale": {
        "name": "Dassault Rafale",
        "category": "aircraft",
        "format": "obj",
        "dimensions": {"length": 15.3, "width": 10.9, "height": 5.3},
        "source": "procedural",
        "source_url": None,
        "license": "CC0",
        "siso_id": "1.2.71.1.1.1",
    },
    "b2": {
        "name": "B-2 Spirit Stealth Bomber",
        "category": "aircraft",
        "format": "obj",
        "dimensions": {"length": 21.0, "width": 52.4, "height": 5.2},
        "source": "procedural",
        "source_url": None,
        "license": "CC0",
        "siso_id": "1.2.225.2.2.1",
    },
    "apache": {
        "name": "AH-64 Apache Helicopter",
        "category": "helicopter",
        "format": "obj",
        "dimensions": {"length": 17.7, "width": 14.6, "height": 4.0},
        "source": "procedural",
        "source_url": "https://free3d.com/3d-model/ah-64d-apache-longbow-2432.html",
        "license": "CC0",
        "siso_id": "1.2.225.20.1.1",
    },
    "blackhawk": {
        "name": "UH-60 Black Hawk",
        "category": "helicopter",
        "format": "obj",
        "dimensions": {"length": 19.8, "width": 16.4, "height": 5.1},
        "source": "procedural",
        "source_url": "https://sketchfab.com/nebulousflynn/collections/cc0-9e9b8c5442ab4b59ba16b6fa5e43b8da",
        "license": "CC0",
        "siso_id": "1.2.225.21.1.1",
    },
    "predator": {
        "name": "MQ-1 Predator UAV",
        "category": "uav",
        "format": "obj",
        "dimensions": {"length": 8.2, "width": 14.8, "height": 2.1},
        "source": "procedural",
        "source_url": None,
        "license": "CC0",
        "siso_id": "1.2.225.50.1.1",
    },

    # Vehicles
    "m1_abrams": {
        "name": "M1 Abrams Tank",
        "category": "vehicle",
        "format": "obj",
        "dimensions": {"length": 9.8, "width": 3.7, "height": 2.4},
        "source": "procedural",
        "source_url": "https://opengameart.org/content/abrams-tank",
        "license": "CC0",
        "siso_id": "1.1.225.1.1.1",
    },
    "t90": {
        "name": "T-90 Battle Tank",
        "category": "vehicle",
        "format": "obj",
        "dimensions": {"length": 9.5, "width": 3.8, "height": 2.2},
        "source": "procedural",
        "source_url": "https://free3d.com/3d-model/t-90-ms-tagil-2545.html",
        "license": "CC0",
        "siso_id": "1.1.222.1.1.4",
    },
    "humvee": {
        "name": "HMMWV Humvee",
        "category": "vehicle",
        "format": "obj",
        "dimensions": {"length": 4.6, "width": 2.2, "height": 1.8},
        "source": "procedural",
        "source_url": "https://opengameart.org/content/cc0-3d-vehicles-and-cars",
        "license": "CC0",
        "siso_id": "1.1.225.6.1.1",
    },
    "pickup_truck": {
        "name": "Pickup Truck",
        "category": "vehicle",
        "format": "obj",
        "dimensions": {"length": 5.4, "width": 2.0, "height": 1.8},
        "source": "procedural",
        "source_url": None,
        "license": "CC0",
        "siso_id": "1.1.225.27.1.0",
    },
    "sedan": {
        "name": "Sedan Car",
        "category": "vehicle",
        "format": "obj",
        "dimensions": {"length": 4.5, "width": 1.8, "height": 1.4},
        "source": "procedural",
        "source_url": None,
        "license": "CC0",
        "siso_id": "1.1.225.80.1.0",
    },
    "suv": {
        "name": "SUV",
        "category": "vehicle",
        "format": "obj",
        "dimensions": {"length": 4.8, "width": 2.0, "height": 1.8},
        "source": "procedural",
        "source_url": None,
        "license": "CC0",
        "siso_id": "1.1.225.81.1.0",
    },

    # Ships
    "destroyer": {
        "name": "Naval Destroyer",
        "category": "ship",
        "format": "obj",
        "dimensions": {"length": 155.0, "width": 20.0, "height": 45.0},
        "source": "procedural",
        "source_url": None,
        "license": "CC0",
        "siso_id": "1.3.225.4.1.1",
    },
    "carrier": {
        "name": "Aircraft Carrier",
        "category": "ship",
        "format": "obj",
        "dimensions": {"length": 333.0, "width": 77.0, "height": 75.0},
        "source": "procedural",
        "source_url": None,
        "license": "CC0",
        "siso_id": "1.3.225.12.1.1",
    },
    "patrol_boat": {
        "name": "Patrol Boat",
        "category": "ship",
        "format": "obj",
        "dimensions": {"length": 25.0, "width": 6.0, "height": 8.0},
        "source": "procedural",
        "source_url": "https://opengameart.org/content/ballistic-missile-submarine",
        "license": "CC0",
        "siso_id": "1.3.225.7.1.1",
    },

    # Missiles & Launchers
    "aim120": {
        "name": "AIM-120 AMRAAM",
        "category": "missile",
        "format": "obj",
        "dimensions": {"length": 3.7, "width": 0.18, "height": 0.18},
        "source": "procedural",
        "source_url": "https://free3d.com/3d-models/missile",
        "license": "CC0",
        "siso_id": "2.1.225.1.1.3",
    },
    "agm114": {
        "name": "AGM-114 Hellfire",
        "category": "missile",
        "format": "obj",
        "dimensions": {"length": 1.6, "width": 0.18, "height": 0.18},
        "source": "procedural",
        "source_url": None,
        "license": "CC0",
        "siso_id": "2.1.225.1.3.4",
    },
    "tomahawk": {
        "name": "Tomahawk Cruise Missile",
        "category": "missile",
        "format": "obj",
        "dimensions": {"length": 6.25, "width": 0.52, "height": 0.52},
        "source": "procedural",
        "source_url": "https://free3d.com/3d-models/missile",
        "license": "CC0",
        "siso_id": "2.2.225.1.2.1",
    },
    "patriot_launcher": {
        "name": "M901 Patriot Launcher",
        "category": "launcher",
        "format": "obj",
        "dimensions": {"length": 10.0, "width": 2.5, "height": 3.5},
        "source": "procedural",
        "source_url": "https://free3d.com/3d-models/missile-launcher",
        "license": "CC0",
        "siso_id": "1.1.225.28.1.1",
    },
    "s400_launcher": {
        "name": "S-400 Launcher",
        "category": "launcher",
        "format": "obj",
        "dimensions": {"length": 12.0, "width": 3.0, "height": 3.8},
        "source": "procedural",
        "source_url": None,
        "license": "CC0",
        "siso_id": "1.1.222.28.2.3",
    },

    # People
    "soldier_standing": {
        "name": "Soldier Standing",
        "category": "person",
        "format": "obj",
        "dimensions": {"length": 0.4, "width": 0.5, "height": 1.8},
        "source": "procedural",
        "source_url": "https://www.mixamo.com/",
        "license": "Royalty Free",
        "siso_id": "3.1.225.1.1.1",
    },
    "soldier_prone": {
        "name": "Soldier Prone",
        "category": "person",
        "format": "obj",
        "dimensions": {"length": 1.8, "width": 0.5, "height": 0.3},
        "source": "procedural",
        "source_url": None,
        "license": "CC0",
        "siso_id": "3.1.225.1.1.1",
    },
    "civilian": {
        "name": "Civilian Person",
        "category": "person",
        "format": "obj",
        "dimensions": {"length": 0.4, "width": 0.5, "height": 1.75},
        "source": "procedural",
        "source_url": "https://readyplayer.me/",
        "license": "CC BY-NC",
        "siso_id": "3.1.225.11.1.1",
    },
}


# Check for trimesh availability
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


@dataclass
class Mesh3D:
    """3D mesh with vertices and faces.

    Attributes:
        vertices: Nx3 array of (x, y, z) vertex positions
        faces: List of face vertex indices
        normals: Optional vertex normals
        thermal_zones: Dict mapping face indices to zone names
        name: Model name
    """
    vertices: NDArray
    faces: List[List[int]]
    normals: Optional[NDArray] = None
    thermal_zones: Dict[int, str] = field(default_factory=dict)
    name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.normals is None and len(self.vertices) > 0:
            self.normals = self._compute_normals()

    def _compute_normals(self) -> NDArray:
        """Compute face normals."""
        normals = []
        for face in self.faces:
            if len(face) >= 3:
                v0 = self.vertices[face[0]]
                v1 = self.vertices[face[1]]
                v2 = self.vertices[face[2]]
                normal = np.cross(v1 - v0, v2 - v0)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                normals.append(normal)
            else:
                normals.append([0, 0, 1])
        return np.array(normals)

    def get_bounds(self) -> Tuple[NDArray, NDArray]:
        """Get bounding box min and max."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def get_dimensions(self) -> Tuple[float, float, float]:
        """Get (length, width, height) dimensions."""
        vmin, vmax = self.get_bounds()
        dims = vmax - vmin
        return float(dims[0]), float(dims[1]), float(dims[2])

    def render_silhouette(
        self,
        resolution: Tuple[int, int] = (64, 64),
        azimuth_deg: float = 0,
        elevation_deg: float = 0,
    ) -> NDArray:
        """Render 2D silhouette from viewing angle.

        Args:
            resolution: Output image size (height, width)
            azimuth_deg: Horizontal viewing angle
            elevation_deg: Vertical viewing angle

        Returns:
            2D binary mask of object silhouette
        """
        h, w = resolution

        # Create rotation matrices
        az = np.radians(azimuth_deg)
        el = np.radians(elevation_deg)

        Ry = np.array([
            [np.cos(az), 0, np.sin(az)],
            [0, 1, 0],
            [-np.sin(az), 0, np.cos(az)]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(el), -np.sin(el)],
            [0, np.sin(el), np.cos(el)]
        ])

        R = Rx @ Ry
        rotated = self.vertices @ R.T

        # Project to 2D (orthographic)
        x_min, x_max = rotated[:, 0].min(), rotated[:, 0].max()
        y_min, y_max = rotated[:, 1].min(), rotated[:, 1].max()

        margin = 0.1
        x_range = max(x_max - x_min, 0.001)
        y_range = max(y_max - y_min, 0.001)
        scale = min((1 - 2*margin) * w / x_range, (1 - 2*margin) * h / y_range)

        cx = (x_max + x_min) / 2
        cy = (y_max + y_min) / 2

        # Create silhouette
        mask = np.zeros((h, w), dtype=np.uint8)

        for face in self.faces:
            if len(face) < 3:
                continue

            pts = []
            for vi in face[:4]:  # Max 4 vertices per face
                vx = int((rotated[vi, 0] - cx) * scale + w / 2)
                vy = int((rotated[vi, 1] - cy) * scale + h / 2)
                pts.append((vx, vy))

            # Fill polygon
            self._fill_polygon(mask, pts)

        return mask

    def _fill_polygon(self, mask: NDArray, pts: List[Tuple[int, int]]):
        """Fill a polygon in the mask using scanline."""
        if len(pts) < 3:
            return

        h, w = mask.shape

        # Get bounding box
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_min, x_max = max(0, min(xs)), min(w-1, max(xs))
        y_min, y_max = max(0, min(ys)), min(h-1, max(ys))

        # Simple scanline fill
        for y in range(y_min, y_max + 1):
            intersections = []
            n = len(pts)
            for i in range(n):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % n]

                if y1 == y2:
                    continue
                if y < min(y1, y2) or y > max(y1, y2):
                    continue

                x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                intersections.append(int(x))

            intersections.sort()

            for i in range(0, len(intersections) - 1, 2):
                x_start = max(0, intersections[i])
                x_end = min(w - 1, intersections[i + 1])
                mask[y, x_start:x_end + 1] = 255


class ModelLibrary:
    """Manager for 3D model library."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize model library.

        Args:
            cache_dir: Directory to cache downloaded models
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".eosim", "models")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, Dict[str, Any]] = MODEL_SOURCES.copy()
        self._loaded_models: Dict[str, Mesh3D] = {}

    def list_models(self, category: Optional[str] = None) -> List[str]:
        """List available model IDs.

        Args:
            category: Optional category filter

        Returns:
            List of model IDs
        """
        if category:
            return [k for k, v in self.models.items() if v.get("category") == category]
        return list(self.models.keys())

    def list_categories(self) -> List[str]:
        """List available categories."""
        return list(set(v.get("category", "other") for v in self.models.values()))

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata."""
        return self.models.get(model_id)

    def load_model(self, model_id: str) -> Optional[Mesh3D]:
        """Load a 3D model.

        Args:
            model_id: Model identifier

        Returns:
            Mesh3D or None if not found
        """
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        info = self.models.get(model_id)
        if info is None:
            return None

        # Check for local file override in cache dir (supports OBJ, GLB, STL via trimesh if available)
        for ext in [".obj", ".glb", ".stl", ".ply"]:
            local_path = self.cache_dir / f"{model_id}{ext}"
            if local_path.exists():
                print(f"Loading local override: {local_path}")
                if HAS_TRIMESH and ext != ".obj":
                    mesh = self._load_with_trimesh(local_path)
                else:
                    mesh = self._load_obj(local_path)

                if mesh:
                    mesh.name = info.get("name", model_id)
                    mesh.metadata = info
                    self._loaded_models[model_id] = mesh
                    return mesh

        # Try embedded detailed model first
        try:
            from eosim.library.embedded_models import get_embedded_model
            model_data = get_embedded_model(model_id)
            mesh = Mesh3D(
                vertices=model_data["vertices"],
                faces=model_data["faces"],
                thermal_zones=model_data.get("thermal_zones", {}),
                name=model_data.get("name", model_id),
                metadata=info
            )
            self._loaded_models[model_id] = mesh
            return mesh
        except (ImportError, KeyError):
            pass  # Fall back to procedural

        # Generate procedural model
        if info.get("source") == "procedural":
            mesh = self._generate_procedural(model_id, info)
            if mesh:
                mesh.name = info.get("name", model_id)
                mesh.metadata = info
                self._loaded_models[model_id] = mesh
                # Save to cache as OBJ for inspection
                self._save_obj(mesh, self.cache_dir / f"{model_id}.obj")
                return mesh

        return None

    def _load_with_trimesh(self, path: Path) -> Optional[Mesh3D]:
        """Load model using trimesh library."""
        try:
            import trimesh
            scene = trimesh.load(path)
            
            # If scene, dump to single mesh
            if isinstance(scene, trimesh.Scene):
                mesh_data = scene.dump(concatenate=True)
            else:
                mesh_data = scene

            if isinstance(mesh_data, list):
                mesh_data = mesh_data[0]  # Take first mesh if list

            return Mesh3D(
                vertices=np.array(mesh_data.vertices),
                faces=mesh_data.faces.tolist(),
                normals=np.array(mesh_data.vertex_normals) if len(mesh_data.vertex_normals) > 0 else None,
            )
        except Exception as e:
            print(f"Error loading with trimesh {path}: {e}")
            return None

    def _load_obj(self, path: Path) -> Optional[Mesh3D]:
        """Load OBJ file."""
        try:
            vertices = []
            faces = []

            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    if parts[0] == "v":
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif parts[0] == "f":
                        face = []
                        for p in parts[1:]:
                            # Handle v/vt/vn format
                            vi = int(p.split("/")[0]) - 1  # OBJ is 1-indexed
                            face.append(vi)
                        faces.append(face)

            if vertices and faces:
                return Mesh3D(
                    vertices=np.array(vertices),
                    faces=faces,
                )
        except Exception as e:
            print(f"Error loading {path}: {e}")

        return None

    def _save_obj(self, mesh: Mesh3D, path: Path):
        """Save mesh to OBJ file."""
        try:
            with open(path, "w") as f:
                f.write(f"# EOSIM generated model: {mesh.name}\n")
                f.write(f"# Vertices: {len(mesh.vertices)}\n")
                f.write(f"# Faces: {len(mesh.faces)}\n\n")

                for v in mesh.vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

                f.write("\n")

                for face in mesh.faces:
                    f.write("f " + " ".join(str(vi + 1) for vi in face) + "\n")
        except Exception as e:
            print(f"Error saving {path}: {e}")

    def _generate_procedural(self, model_id: str, info: Dict[str, Any]) -> Optional[Mesh3D]:
        """Generate procedural model based on category."""
        category = info.get("category", "")
        dims = info.get("dimensions", {})
        length = dims.get("length", 10.0)
        width = dims.get("width", 5.0)
        height = dims.get("height", 3.0)

        if category == "aircraft":
            return self._create_aircraft(length, width, height, model_id)
        elif category == "vehicle":
            return self._create_vehicle(length, width, height, model_id)
        elif category == "ship":
            return self._create_ship(length, width, height, model_id)
        elif category == "person":
            return self._create_person(length, width, height, model_id)
        elif category == "helicopter":
            return self._create_helicopter(length, width, height, model_id)
        elif category == "uav":
            return self._create_aircraft(length, width, height, model_id)
        elif category == "missile":
            return self._create_missile(length, width, height, model_id)
        elif category == "launcher":
            return self._create_launcher(length, width, height, model_id)
        else:
            return self._create_box(length, width, height)

    def _create_box(self, l: float, w: float, h: float) -> Mesh3D:
        """Create simple box mesh."""
        hl, hw, hh = l/2, w/2, h/2
        vertices = np.array([
            [-hl, -hw, 0], [hl, -hw, 0], [hl, hw, 0], [-hl, hw, 0],
            [-hl, -hw, h], [hl, -hw, h], [hl, hw, h], [-hl, hw, h],
        ])
        faces = [
            [0, 1, 2, 3], [4, 7, 6, 5],
            [0, 4, 5, 1], [2, 6, 7, 3],
            [0, 3, 7, 4], [1, 5, 6, 2],
        ]
        return Mesh3D(vertices=vertices, faces=faces)

    def _create_aircraft(self, length: float, wingspan: float, height: float,
                        model_id: str) -> Mesh3D:
        """Create detailed aircraft mesh."""
        verts = []
        faces = []
        zones = {}

        l = length / 2
        w = wingspan / 2
        h = height / 2

        # Fuselage - tapered cylinder
        fuse_sections = 12
        fuse_pts = 8
        fuse_radius = h * 0.6

        # Nose point
        verts.append([l, 0, 0])

        # Fuselage sections
        for s in range(fuse_sections):
            x = l - (s + 1) * length / (fuse_sections + 1)

            # Taper at ends
            if s < 2:
                r = fuse_radius * (0.3 + s * 0.35)
            elif s > fuse_sections - 3:
                r = fuse_radius * (1.0 - (s - fuse_sections + 3) * 0.2)
            else:
                r = fuse_radius

            for p in range(fuse_pts):
                angle = 2 * np.pi * p / fuse_pts
                y = r * np.cos(angle)
                z = r * np.sin(angle) + h * 0.2
                verts.append([x, y, z])

        # Tail point
        tail_idx = len(verts)
        verts.append([-l, 0, h * 0.2])

        # Nose faces
        for p in range(fuse_pts):
            next_p = (p + 1) % fuse_pts
            faces.append([0, 1 + p, 1 + next_p])
            zones[len(faces) - 1] = "fuselage"

        # Fuselage side faces
        for s in range(fuse_sections - 1):
            base1 = 1 + s * fuse_pts
            base2 = 1 + (s + 1) * fuse_pts
            for p in range(fuse_pts):
                next_p = (p + 1) % fuse_pts
                faces.append([base1 + p, base2 + p, base2 + next_p, base1 + next_p])
                zones[len(faces) - 1] = "fuselage"

        # Tail faces
        last_section = 1 + (fuse_sections - 1) * fuse_pts
        for p in range(fuse_pts):
            next_p = (p + 1) % fuse_pts
            faces.append([last_section + p, tail_idx, last_section + next_p])
            zones[len(faces) - 1] = "exhaust"

        # Main wings
        wing_root_x = l * 0.1
        wing_tip_x = -l * 0.3
        wing_thickness = h * 0.1

        for side in [-1, 1]:
            wing_start = len(verts)
            y_root = side * fuse_radius
            y_tip = side * w

            # Wing quad (simplified)
            verts.extend([
                [wing_root_x + l*0.2, y_root, h*0.2 - wing_thickness],
                [wing_tip_x + l*0.1, y_tip, h*0.2],
                [wing_tip_x - l*0.1, y_tip, h*0.2],
                [wing_root_x - l*0.2, y_root, h*0.2 - wing_thickness],
            ])

            if side > 0:
                faces.append([wing_start, wing_start+1, wing_start+2, wing_start+3])
            else:
                faces.append([wing_start, wing_start+3, wing_start+2, wing_start+1])
            zones[len(faces) - 1] = "wings"

        # Vertical stabilizer
        vs_start = len(verts)
        verts.extend([
            [-l*0.5, 0, h*0.5],
            [-l*0.7, 0, h*1.5],
            [-l*0.9, 0, h*1.3],
            [-l*0.8, 0, h*0.5],
        ])
        faces.append([vs_start, vs_start+1, vs_start+2, vs_start+3])
        zones[len(faces) - 1] = "fuselage"

        # Horizontal stabilizers
        for side in [-1, 1]:
            hs_start = len(verts)
            verts.extend([
                [-l*0.6, side * fuse_radius, h*0.3],
                [-l*0.7, side * w*0.4, h*0.3],
                [-l*0.85, side * w*0.35, h*0.3],
                [-l*0.75, side * fuse_radius, h*0.3],
            ])
            if side > 0:
                faces.append([hs_start, hs_start+1, hs_start+2, hs_start+3])
            else:
                faces.append([hs_start, hs_start+3, hs_start+2, hs_start+1])
            zones[len(faces) - 1] = "wings"

        # Cockpit canopy
        ck_start = len(verts)
        verts.extend([
            [l*0.3, -fuse_radius*0.4, h*0.5],
            [l*0.3, fuse_radius*0.4, h*0.5],
            [-l*0.1, fuse_radius*0.5, h*0.8],
            [-l*0.1, -fuse_radius*0.5, h*0.8],
        ])
        faces.append([ck_start, ck_start+1, ck_start+2, ck_start+3])
        zones[len(faces) - 1] = "cockpit"

        # Engine nozzles
        nozzle_r = fuse_radius * 0.35

        if "twin" in model_id.lower() or any(x in model_id for x in ["f15", "f18", "su", "mig"]):
            nozzle_positions = [fuse_radius * 0.5, -fuse_radius * 0.5]
        else:
            nozzle_positions = [0]

        for ny in nozzle_positions:
            nz_start = len(verts)
            nz_x = -l + 0.3
            nz_pts = 8

            for p in range(nz_pts):
                angle = 2 * np.pi * p / nz_pts
                verts.append([nz_x, ny + nozzle_r * np.cos(angle),
                             h*0.2 + nozzle_r * np.sin(angle)])

            # Nozzle back center
            nz_back = len(verts)
            verts.append([nz_x - 0.8, ny, h*0.2])

            for p in range(nz_pts):
                next_p = (p + 1) % nz_pts
                faces.append([nz_start + p, nz_back, nz_start + next_p])
                zones[len(faces) - 1] = "nozzle"

        return Mesh3D(
            vertices=np.array(verts),
            faces=faces,
            thermal_zones=zones
        )

    def _create_vehicle(self, length: float, width: float, height: float,
                       model_id: str) -> Mesh3D:
        """Create vehicle mesh."""
        verts = []
        faces = []
        zones = {}

        l, w, h = length/2, width/2, height

        is_tank = "tank" in model_id.lower() or "abrams" in model_id.lower() or "t90" in model_id.lower()
        is_truck = "truck" in model_id.lower() or "humvee" in model_id.lower()

        if is_tank:
            # Tank hull
            hull_h = h * 0.5
            verts.extend([
                [-l, -w, 0], [l*0.9, -w, 0], [l, -w*0.8, 0], [l, w*0.8, 0],
                [l*0.9, w, 0], [-l, w, 0],
                [-l*0.9, -w*0.9, hull_h], [l*0.7, -w*0.9, hull_h],
                [l*0.8, -w*0.7, hull_h], [l*0.8, w*0.7, hull_h],
                [l*0.7, w*0.9, hull_h], [-l*0.9, w*0.9, hull_h],
            ])

            # Hull faces
            faces.extend([
                [0, 1, 7, 6], [1, 2, 8, 7], [2, 3, 9, 8],
                [3, 4, 10, 9], [4, 5, 11, 10], [5, 0, 6, 11],
                [6, 7, 8, 9, 10, 11],  # Top
            ])
            for i in range(7):
                zones[i] = "body"

            # Turret
            turret_start = len(verts)
            turret_r = w * 0.5
            turret_h = h * 0.35
            turret_pts = 12

            for p in range(turret_pts):
                angle = 2 * np.pi * p / turret_pts
                verts.append([turret_r * np.cos(angle) - l*0.1,
                             turret_r * np.sin(angle), hull_h])
                verts.append([turret_r * 0.9 * np.cos(angle) - l*0.1,
                             turret_r * 0.9 * np.sin(angle), hull_h + turret_h])

            # Turret top
            top_center = len(verts)
            verts.append([-l*0.1, 0, hull_h + turret_h])

            for p in range(turret_pts):
                next_p = (p + 1) % turret_pts
                # Side
                faces.append([turret_start + p*2, turret_start + next_p*2,
                             turret_start + next_p*2 + 1, turret_start + p*2 + 1])
                zones[len(faces) - 1] = "engine"
                # Top
                faces.append([turret_start + p*2 + 1, turret_start + next_p*2 + 1, top_center])
                zones[len(faces) - 1] = "engine"

            # Gun barrel
            barrel_start = len(verts)
            barrel_l = l * 1.2
            barrel_r = 0.12
            barrel_z = hull_h + turret_h * 0.7

            for p in range(8):
                angle = 2 * np.pi * p / 8
                verts.append([0, barrel_r * np.cos(angle), barrel_z + barrel_r * np.sin(angle)])
                verts.append([barrel_l, barrel_r * 0.8 * np.cos(angle),
                             barrel_z + barrel_r * 0.8 * np.sin(angle)])

            for p in range(8):
                next_p = (p + 1) % 8
                faces.append([barrel_start + p*2, barrel_start + next_p*2,
                             barrel_start + next_p*2 + 1, barrel_start + p*2 + 1])
                zones[len(faces) - 1] = "body"

        else:
            # Regular vehicle
            body_h = h * 0.5
            verts.extend([
                [-l, -w, 0], [l, -w, 0], [l, w, 0], [-l, w, 0],
                [-l*0.95, -w, body_h], [l*0.95, -w, body_h],
                [l*0.95, w, body_h], [-l*0.95, w, body_h],
            ])

            faces.extend([
                [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
                [4, 5, 6, 7],
            ])
            for i in range(5):
                zones[i] = "body"

            # Cabin
            cabin_start = len(verts)
            cabin_h = h * 0.95 if not is_truck else h * 1.1
            cabin_front = l * 0.3 if not is_truck else l * 0.6
            cabin_rear = -l * 0.7 if not is_truck else l * 0.1

            verts.extend([
                [cabin_rear, -w*0.9, body_h],
                [cabin_front, -w*0.9, body_h],
                [cabin_front, w*0.9, body_h],
                [cabin_rear, w*0.9, body_h],
                [cabin_rear + l*0.1, -w*0.8, cabin_h],
                [cabin_front - l*0.1, -w*0.8, cabin_h],
                [cabin_front - l*0.1, w*0.8, cabin_h],
                [cabin_rear + l*0.1, w*0.8, cabin_h],
            ])

            faces.extend([
                [cabin_start, cabin_start+1, cabin_start+5, cabin_start+4],
                [cabin_start+1, cabin_start+2, cabin_start+6, cabin_start+5],
                [cabin_start+2, cabin_start+3, cabin_start+7, cabin_start+6],
                [cabin_start+3, cabin_start, cabin_start+4, cabin_start+7],
                [cabin_start+4, cabin_start+5, cabin_start+6, cabin_start+7],
            ])
            for i in range(5):
                zones[len(faces) - 5 + i] = "cabin"

            # Engine hood
            hood_start = len(verts)
            verts.extend([
                [cabin_front, -w*0.9, body_h],
                [l*0.9, -w*0.85, body_h*1.05],
                [l*0.9, w*0.85, body_h*1.05],
                [cabin_front, w*0.9, body_h],
            ])
            faces.append([hood_start, hood_start+1, hood_start+2, hood_start+3])
            zones[len(faces) - 1] = "engine"

            # Wheels
            wheel_r = h * 0.2
            wheel_positions = [
                (l*0.7, w*1.05), (l*0.7, -w*1.05),
                (-l*0.6, w*1.05), (-l*0.6, -w*1.05),
            ]

            for wx, wy in wheel_positions:
                ws = len(verts)
                for p in range(8):
                    angle = 2 * np.pi * p / 8
                    verts.append([wx + wheel_r * np.cos(angle),
                                 wy, wheel_r + wheel_r * np.sin(angle)])

                # Wheel face
                faces.append(list(range(ws, ws + 8)))
                zones[len(faces) - 1] = "wheels"

        return Mesh3D(
            vertices=np.array(verts),
            faces=faces,
            thermal_zones=zones
        )

    def _create_ship(self, length: float, width: float, height: float,
                    model_id: str) -> Mesh3D:
        """Create ship mesh."""
        verts = []
        faces = []
        zones = {}

        l, w, h = length/2, width/2, height

        # Hull
        hull_h = h * 0.3
        verts.extend([
            # Bow point
            [l, 0, hull_h * 0.5],
            # Hull cross-section forward
            [l*0.7, -w*0.6, 0], [l*0.7, -w*0.6, hull_h],
            [l*0.7, w*0.6, hull_h], [l*0.7, w*0.6, 0],
            # Hull cross-section mid
            [0, -w, 0], [0, -w, hull_h], [0, w, hull_h], [0, w, 0],
            # Hull cross-section aft
            [-l*0.8, -w*0.9, 0], [-l*0.8, -w*0.9, hull_h],
            [-l*0.8, w*0.9, hull_h], [-l*0.8, w*0.9, 0],
            # Stern
            [-l, -w*0.7, hull_h*0.5], [-l, w*0.7, hull_h*0.5],
        ])

        # Hull faces
        faces.extend([
            [0, 1, 2], [0, 2, 3], [0, 3, 4],  # Bow
            [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 8, 4],  # Forward
            [5, 9, 10, 6], [6, 10, 11, 7], [7, 11, 12, 8],  # Mid
            [9, 13, 10], [10, 13, 14, 11], [11, 14, 12],  # Aft
        ])
        for i in range(len(faces)):
            zones[i] = "hull"

        # Deck
        deck_start = len(verts)
        verts.extend([
            [l*0.5, -w*0.5, hull_h], [l*0.5, w*0.5, hull_h],
            [-l*0.7, w*0.8, hull_h], [-l*0.7, -w*0.8, hull_h],
        ])
        faces.append([deck_start, deck_start+1, deck_start+2, deck_start+3])
        zones[len(faces) - 1] = "deck"

        # Superstructure
        ss_start = len(verts)
        ss_h = h * 0.4
        verts.extend([
            [l*0.2, -w*0.3, hull_h], [l*0.2, w*0.3, hull_h],
            [-l*0.3, w*0.3, hull_h], [-l*0.3, -w*0.3, hull_h],
            [l*0.15, -w*0.25, hull_h + ss_h],
            [l*0.15, w*0.25, hull_h + ss_h],
            [-l*0.25, w*0.25, hull_h + ss_h],
            [-l*0.25, -w*0.25, hull_h + ss_h],
        ])

        faces.extend([
            [ss_start, ss_start+1, ss_start+5, ss_start+4],
            [ss_start+1, ss_start+2, ss_start+6, ss_start+5],
            [ss_start+2, ss_start+3, ss_start+7, ss_start+6],
            [ss_start+3, ss_start, ss_start+4, ss_start+7],
            [ss_start+4, ss_start+5, ss_start+6, ss_start+7],
        ])
        for i in range(5):
            zones[len(faces) - 5 + i] = "superstructure"

        return Mesh3D(
            vertices=np.array(verts),
            faces=faces,
            thermal_zones=zones
        )

    def _create_person(self, length: float, width: float, height: float,
                      model_id: str) -> Mesh3D:
        """Create humanoid mesh."""
        verts = []
        faces = []
        zones = {}

        h = height
        is_prone = "prone" in model_id.lower()

        if is_prone:
            # Lying down - elongated shape
            verts.extend([
                [-h/2, -width/4, 0], [h/2, -width/4, 0],
                [h/2, width/4, 0], [-h/2, width/4, 0],
                [-h/2, -width/4, height*0.15], [h/2, -width/4, height*0.15],
                [h/2, width/4, height*0.15], [-h/2, width/4, height*0.15],
            ])

            faces.extend([
                [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
                [4, 5, 6, 7],
            ])
            for i in range(4):
                zones[i] = "torso"
            zones[4] = "torso"

        else:
            # Standing person
            # Head
            head_r = h * 0.06
            head_z = h * 0.92
            head_start = len(verts)

            for p in range(8):
                angle = 2 * np.pi * p / 8
                verts.append([head_r * np.cos(angle), head_r * np.sin(angle), head_z])
                verts.append([head_r * np.cos(angle), head_r * np.sin(angle), h])

            head_top = len(verts)
            verts.append([0, 0, h])

            for p in range(8):
                next_p = (p + 1) % 8
                faces.append([head_start + p*2, head_start + next_p*2,
                             head_start + next_p*2 + 1, head_start + p*2 + 1])
                zones[len(faces) - 1] = "head"
                faces.append([head_start + p*2 + 1, head_start + next_p*2 + 1, head_top])
                zones[len(faces) - 1] = "head"

            # Torso
            torso_w = h * 0.12
            torso_d = h * 0.08
            torso_top = h * 0.88
            torso_bot = h * 0.45
            torso_start = len(verts)

            verts.extend([
                [torso_d, -torso_w, torso_bot], [torso_d, torso_w, torso_bot],
                [-torso_d, torso_w, torso_bot], [-torso_d, -torso_w, torso_bot],
                [torso_d*0.9, -torso_w*0.8, torso_top],
                [torso_d*0.9, torso_w*0.8, torso_top],
                [-torso_d*0.9, torso_w*0.8, torso_top],
                [-torso_d*0.9, -torso_w*0.8, torso_top],
            ])

            faces.extend([
                [torso_start, torso_start+1, torso_start+5, torso_start+4],
                [torso_start+1, torso_start+2, torso_start+6, torso_start+5],
                [torso_start+2, torso_start+3, torso_start+7, torso_start+6],
                [torso_start+3, torso_start, torso_start+4, torso_start+7],
            ])
            for i in range(4):
                zones[len(faces) - 4 + i] = "torso"

            # Legs
            leg_w = h * 0.05
            leg_sep = h * 0.04

            for side in [-1, 1]:
                leg_start = len(verts)
                ly = side * (leg_sep + leg_w)
                verts.extend([
                    [leg_w, ly - leg_w, 0], [leg_w, ly + leg_w, 0],
                    [-leg_w, ly + leg_w, 0], [-leg_w, ly - leg_w, 0],
                    [leg_w, ly - leg_w, torso_bot], [leg_w, ly + leg_w, torso_bot],
                    [-leg_w, ly + leg_w, torso_bot], [-leg_w, ly - leg_w, torso_bot],
                ])

                faces.extend([
                    [leg_start, leg_start+1, leg_start+5, leg_start+4],
                    [leg_start+1, leg_start+2, leg_start+6, leg_start+5],
                    [leg_start+2, leg_start+3, leg_start+7, leg_start+6],
                    [leg_start+3, leg_start, leg_start+4, leg_start+7],
                ])
                for i in range(4):
                    zones[len(faces) - 4 + i] = "legs"

            # Arms
            arm_w = h * 0.03
            arm_top = h * 0.85
            arm_bot = h * 0.45

            for side in [-1, 1]:
                arm_start = len(verts)
                ay = side * (torso_w + arm_w)
                verts.extend([
                    [arm_w, ay - arm_w, arm_bot - h*0.1],
                    [arm_w, ay + arm_w, arm_bot - h*0.1],
                    [-arm_w, ay + arm_w, arm_bot - h*0.1],
                    [-arm_w, ay - arm_w, arm_bot - h*0.1],
                    [arm_w, ay - arm_w, arm_top], [arm_w, ay + arm_w, arm_top],
                    [-arm_w, ay + arm_w, arm_top], [-arm_w, ay - arm_w, arm_top],
                ])

                faces.extend([
                    [arm_start, arm_start+1, arm_start+5, arm_start+4],
                    [arm_start+1, arm_start+2, arm_start+6, arm_start+5],
                    [arm_start+2, arm_start+3, arm_start+7, arm_start+6],
                    [arm_start+3, arm_start, arm_start+4, arm_start+7],
                ])
                for i in range(4):
                    zones[len(faces) - 4 + i] = "torso"

            # Hands
            hand_r = h * 0.025
            for side in [-1, 1]:
                hand_start = len(verts)
                hy = side * (torso_w + arm_w)
                hz = arm_bot - h*0.12
                verts.extend([
                    [hand_r, hy - hand_r, hz - hand_r*2],
                    [hand_r, hy + hand_r, hz - hand_r*2],
                    [-hand_r, hy + hand_r, hz - hand_r*2],
                    [-hand_r, hy - hand_r, hz - hand_r*2],
                    [hand_r, hy - hand_r, hz], [hand_r, hy + hand_r, hz],
                    [-hand_r, hy + hand_r, hz], [-hand_r, hy - hand_r, hz],
                ])

                faces.extend([
                    [hand_start, hand_start+1, hand_start+5, hand_start+4],
                    [hand_start+1, hand_start+2, hand_start+6, hand_start+5],
                ])
                for i in range(2):
                    zones[len(faces) - 2 + i] = "hands"

        return Mesh3D(
            vertices=np.array(verts),
            faces=faces,
            thermal_zones=zones
        )

    def _create_helicopter(self, length: float, width: float, height: float,
                          model_id: str) -> Mesh3D:
        """Create helicopter mesh."""
        verts = []
        faces = []
        zones = {}

        l, w, h = length/2, width/2, height

        # Main body (fuselage)
        body_h = h * 0.4
        body_w = w * 0.3
        body_start = len(verts)
        verts.extend([
            [-l*0.4, -body_w, 0], [l*0.3, -body_w, 0],
            [l*0.3, body_w, 0], [-l*0.4, body_w, 0],
            [-l*0.4, -body_w, body_h], [l*0.3, -body_w, body_h],
            [l*0.3, body_w, body_h], [-l*0.4, body_w, body_h],
        ])
        faces.extend([
            [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
            [4, 5, 6, 7],
        ])
        for i in range(5):
            zones[i] = "fuselage"

        # Cockpit (front)
        ck_start = len(verts)
        verts.extend([
            [l*0.3, -body_w*0.8, body_h*0.2],
            [l*0.6, -body_w*0.5, body_h*0.3],
            [l*0.6, body_w*0.5, body_h*0.3],
            [l*0.3, body_w*0.8, body_h*0.2],
            [l*0.3, -body_w*0.8, body_h],
            [l*0.5, -body_w*0.5, body_h*0.8],
            [l*0.5, body_w*0.5, body_h*0.8],
            [l*0.3, body_w*0.8, body_h],
        ])
        faces.extend([
            [ck_start, ck_start+1, ck_start+5, ck_start+4],
            [ck_start+1, ck_start+2, ck_start+6, ck_start+5],
            [ck_start+2, ck_start+3, ck_start+7, ck_start+6],
        ])
        for i in range(3):
            zones[len(faces) - 3 + i] = "cockpit"

        # Tail boom
        tail_start = len(verts)
        tail_w = body_w * 0.3
        verts.extend([
            [-l*0.4, -tail_w, body_h*0.3],
            [-l*0.4, tail_w, body_h*0.3],
            [-l*0.4, tail_w, body_h*0.7],
            [-l*0.4, -tail_w, body_h*0.7],
            [-l, -tail_w*0.5, body_h*0.4],
            [-l, tail_w*0.5, body_h*0.4],
            [-l, tail_w*0.5, body_h*0.6],
            [-l, -tail_w*0.5, body_h*0.6],
        ])
        faces.extend([
            [tail_start, tail_start+1, tail_start+5, tail_start+4],
            [tail_start+1, tail_start+2, tail_start+6, tail_start+5],
            [tail_start+2, tail_start+3, tail_start+7, tail_start+6],
            [tail_start+3, tail_start, tail_start+4, tail_start+7],
        ])
        for i in range(4):
            zones[len(faces) - 4 + i] = "fuselage"

        # Main rotor disk (simplified)
        rotor_start = len(verts)
        rotor_r = w
        rotor_z = body_h + h * 0.1
        for i in range(8):
            angle = 2 * np.pi * i / 8
            verts.append([rotor_r * np.cos(angle), rotor_r * np.sin(angle), rotor_z])
        faces.append(list(range(rotor_start, rotor_start + 8)))
        zones[len(faces) - 1] = "wings"

        # Tail rotor
        tr_start = len(verts)
        tr_r = h * 0.25
        tr_x = -l + 0.1
        for i in range(6):
            angle = 2 * np.pi * i / 6
            verts.append([tr_x, w*0.15 + tr_r * np.cos(angle), body_h*0.5 + tr_r * np.sin(angle)])
        faces.append(list(range(tr_start, tr_start + 6)))
        zones[len(faces) - 1] = "wings"

        return Mesh3D(
            vertices=np.array(verts),
            faces=faces,
            thermal_zones=zones
        )

    def _create_missile(self, length: float, diameter: float, height: float,
                       model_id: str) -> Mesh3D:
        """Create missile mesh (cylinder + fins)."""
        verts = []
        faces = []
        zones = {}

        radius = diameter / 2 if diameter > 0.01 else 0.09
        segments = 12

        # Nose tip
        verts.append([length/2, 0, 0])
        nose_idx = 0

        # Body rings
        rings = [length/2 - length*0.2, -length/2 + length*0.1, -length/2]

        for x in rings:
            for i in range(segments):
                angle = 2 * np.pi * i / segments
                verts.append([x, radius * np.cos(angle), radius * np.sin(angle)])

        # Nose faces
        for i in range(segments):
            faces.append([nose_idx, 1 + i, 1 + (i+1)%segments])
            zones[len(faces)-1] = "body"

        # Body faces
        for r in range(len(rings)-1):
            base1 = 1 + r * segments
            base2 = 1 + (r+1) * segments
            for i in range(segments):
                next_i = (i+1)%segments
                faces.append([base1 + i, base2 + i, base2 + next_i, base1 + next_i])
                zones[len(faces)-1] = "body"

        # Fins
        fin_span = max(radius * 3, 0.15)
        fin_root_x = -length/2 + length*0.15

        fin_verts_start = len(verts)
        verts.extend([
            [fin_root_x + 0.1, 0, radius],
            [fin_root_x, 0, fin_span],
            [fin_root_x - 0.1, 0, fin_span],
            [fin_root_x - 0.1, 0, radius],
        ])

        # 4 fins at 90 degree intervals
        for i in range(4):
            angle = np.pi/2 * i
            rot = np.array([[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])

            base_v = len(verts)
            for j in range(4):
                v = np.array(verts[fin_verts_start + j])
                verts.append(list(rot @ v))

            faces.append([base_v, base_v+1, base_v+2, base_v+3])
            zones[len(faces)-1] = "fins"

        return Mesh3D(vertices=np.array(verts), faces=faces, thermal_zones=zones)

    def _create_launcher(self, length: float, width: float, height: float,
                        model_id: str) -> Mesh3D:
        """Create missile launcher mesh."""
        # Truck base
        base = self._create_vehicle(length, width, height*0.6, "truck")

        # Launcher box/tubes
        box_l, box_w, box_h = length*0.7, width*0.8, height*0.5
        box = self._create_box(box_l, box_w, box_h)

        # Rotate box up (elevated launch angle)
        angle = np.radians(30)
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        box.vertices = box.vertices @ rot.T

        # Position box on truck bed
        box.vertices += np.array([-length*0.2, 0, height * 0.7])

        # Combine meshes
        base_v_count = len(base.vertices)
        vertices = np.vstack([base.vertices, box.vertices])

        faces = list(base.faces)
        for f in box.faces:
            faces.append([i + base_v_count for i in f])

        zones = dict(base.thermal_zones)
        for i in range(len(base.faces), len(faces)):
            zones[i] = "launcher"

        return Mesh3D(vertices=vertices, faces=faces, thermal_zones=zones)


# Global library instance
_library: Optional[ModelLibrary] = None


def get_library() -> ModelLibrary:
    """Get the global model library instance."""
    global _library
    if _library is None:
        _library = ModelLibrary()
    return _library


def list_models(category: Optional[str] = None) -> List[str]:
    """List available models."""
    return get_library().list_models(category)


def load_model(model_id: str) -> Optional[Mesh3D]:
    """Load a 3D model."""
    return get_library().load_model(model_id)


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Get model information."""
    return get_library().get_model_info(model_id)
