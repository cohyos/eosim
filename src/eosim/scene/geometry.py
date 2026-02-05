"""
Scene geometry handling for EOSIM.

Provides classes for representing and loading 3D scene geometry
including meshes, terrain, and basic primitives.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""

    min_point: NDArray[np.floating]  # (3,) xyz minimum
    max_point: NDArray[np.floating]  # (3,) xyz maximum

    def __post_init__(self) -> None:
        self.min_point = np.asarray(self.min_point, dtype=np.float64)
        self.max_point = np.asarray(self.max_point, dtype=np.float64)

    @property
    def center(self) -> NDArray[np.floating]:
        """Center point of bounding box."""
        return (self.min_point + self.max_point) / 2

    @property
    def size(self) -> NDArray[np.floating]:
        """Size (extent) in each dimension."""
        return self.max_point - self.min_point

    @property
    def diagonal(self) -> float:
        """Diagonal length of bounding box."""
        return float(np.linalg.norm(self.size))

    def contains(self, point: NDArray[np.floating]) -> bool:
        """Check if point is inside bounding box."""
        return bool(
            np.all(point >= self.min_point) and np.all(point <= self.max_point)
        )

    def intersects(self, other: "BoundingBox") -> bool:
        """Check if this box intersects another."""
        return bool(
            np.all(self.min_point <= other.max_point)
            and np.all(self.max_point >= other.min_point)
        )

    @classmethod
    def from_points(cls, points: NDArray[np.floating]) -> "BoundingBox":
        """Create bounding box from point cloud."""
        return cls(
            min_point=np.min(points, axis=0),
            max_point=np.max(points, axis=0),
        )


@dataclass
class Transform:
    """3D transformation (translation, rotation, scale)."""

    translation: NDArray[np.floating] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    rotation: NDArray[np.floating] = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )  # 3x3 rotation matrix
    scale: NDArray[np.floating] = field(
        default_factory=lambda: np.ones(3, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        self.translation = np.asarray(self.translation, dtype=np.float64)
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
        self.scale = np.asarray(self.scale, dtype=np.float64)

    @property
    def matrix(self) -> NDArray[np.floating]:
        """4x4 transformation matrix."""
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = self.rotation * self.scale
        M[:3, 3] = self.translation
        return M

    def apply(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply transformation to points (N, 3)."""
        scaled = points * self.scale
        rotated = scaled @ self.rotation.T
        return rotated + self.translation

    def apply_normal(self, normals: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply transformation to normals (rotation only)."""
        return normals @ self.rotation.T

    @classmethod
    def from_euler_xyz(
        cls,
        translation: NDArray[np.floating],
        angles_rad: NDArray[np.floating],
        scale: Optional[NDArray[np.floating]] = None,
    ) -> "Transform":
        """Create transform from Euler angles (XYZ order)."""
        rx, ry, rz = angles_rad
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        rotation = Rz @ Ry @ Rx

        return cls(
            translation=np.asarray(translation),
            rotation=rotation,
            scale=np.ones(3) if scale is None else np.asarray(scale),
        )


class Geometry(ABC):
    """Abstract base class for geometry types."""

    @property
    @abstractmethod
    def bounds(self) -> BoundingBox:
        """Get axis-aligned bounding box."""
        pass

    @abstractmethod
    def sample_surface(self, n_points: int) -> NDArray[np.floating]:
        """Sample random points on surface."""
        pass


@dataclass
class TriangleMesh(Geometry):
    """Triangle mesh geometry.

    Attributes:
        vertices: Vertex positions (N, 3)
        faces: Triangle face indices (M, 3)
        normals: Vertex normals (N, 3) or face normals (M, 3)
        uvs: Texture coordinates (N, 2), optional
        material_ids: Per-face material IDs (M,), optional
    """

    vertices: NDArray[np.floating]
    faces: NDArray[np.integer]
    normals: Optional[NDArray[np.floating]] = None
    uvs: Optional[NDArray[np.floating]] = None
    material_ids: Optional[NDArray[np.integer]] = None
    _bounds: Optional[BoundingBox] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int32)

        if self.normals is None:
            self.normals = self._compute_vertex_normals()
        else:
            self.normals = np.asarray(self.normals, dtype=np.float64)

    @property
    def n_vertices(self) -> int:
        """Number of vertices."""
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        """Number of triangular faces."""
        return len(self.faces)

    @property
    def bounds(self) -> BoundingBox:
        """Compute or return cached bounding box."""
        if self._bounds is None:
            self._bounds = BoundingBox.from_points(self.vertices)
        return self._bounds

    def _compute_vertex_normals(self) -> NDArray[np.floating]:
        """Compute vertex normals from face normals."""
        face_normals = self._compute_face_normals()
        vertex_normals = np.zeros_like(self.vertices)

        # Accumulate face normals to vertices
        for i, face in enumerate(self.faces):
            for vi in face:
                vertex_normals[vi] += face_normals[i]

        # Normalize
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return vertex_normals / norms

    def _compute_face_normals(self) -> NDArray[np.floating]:
        """Compute face normals."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        edges1 = v1 - v0
        edges2 = v2 - v0
        normals = np.cross(edges1, edges2)

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return normals / norms

    def face_areas(self) -> NDArray[np.floating]:
        """Compute area of each face."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        edges1 = v1 - v0
        edges2 = v2 - v0
        crosses = np.cross(edges1, edges2)
        return 0.5 * np.linalg.norm(crosses, axis=1)

    def total_area(self) -> float:
        """Total surface area."""
        return float(np.sum(self.face_areas()))

    def sample_surface(self, n_points: int) -> NDArray[np.floating]:
        """Sample random points uniformly on mesh surface."""
        areas = self.face_areas()
        probs = areas / areas.sum()

        # Select faces weighted by area
        face_indices = np.random.choice(len(self.faces), size=n_points, p=probs)

        # Random barycentric coordinates
        r1 = np.random.random(n_points)
        r2 = np.random.random(n_points)
        sqrt_r1 = np.sqrt(r1)

        u = 1 - sqrt_r1
        v = sqrt_r1 * (1 - r2)
        w = sqrt_r1 * r2

        # Get triangle vertices
        v0 = self.vertices[self.faces[face_indices, 0]]
        v1 = self.vertices[self.faces[face_indices, 1]]
        v2 = self.vertices[self.faces[face_indices, 2]]

        # Interpolate
        points = u[:, np.newaxis] * v0 + v[:, np.newaxis] * v1 + w[:, np.newaxis] * v2
        return points

    def transform(self, xform: Transform) -> "TriangleMesh":
        """Return transformed copy of mesh."""
        new_vertices = xform.apply(self.vertices)
        new_normals = xform.apply_normal(self.normals) if self.normals is not None else None

        return TriangleMesh(
            vertices=new_vertices,
            faces=self.faces.copy(),
            normals=new_normals,
            uvs=self.uvs.copy() if self.uvs is not None else None,
            material_ids=self.material_ids.copy() if self.material_ids is not None else None,
        )


@dataclass
class HeightField(Geometry):
    """Height field terrain geometry.

    Attributes:
        heights: 2D array of height values (rows, cols)
        x_range: (min, max) extent in X direction
        y_range: (min, max) extent in Y direction
        resolution: Grid cell size (assumed uniform)
    """

    heights: NDArray[np.floating]
    x_range: tuple[float, float] = (0.0, 1.0)
    y_range: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        self.heights = np.asarray(self.heights, dtype=np.float64)

    @property
    def shape(self) -> tuple[int, int]:
        """Grid dimensions (rows, cols)."""
        return self.heights.shape

    @property
    def resolution_x(self) -> float:
        """Grid resolution in X direction."""
        return (self.x_range[1] - self.x_range[0]) / (self.heights.shape[1] - 1)

    @property
    def resolution_y(self) -> float:
        """Grid resolution in Y direction."""
        return (self.y_range[1] - self.y_range[0]) / (self.heights.shape[0] - 1)

    @property
    def bounds(self) -> BoundingBox:
        """Compute bounding box."""
        return BoundingBox(
            min_point=np.array([self.x_range[0], self.y_range[0], float(np.min(self.heights))]),
            max_point=np.array([self.x_range[1], self.y_range[1], float(np.max(self.heights))]),
        )

    def height_at(self, x: float, y: float) -> float:
        """Interpolate height at (x, y) coordinate."""
        # Convert to grid coordinates
        col = (x - self.x_range[0]) / self.resolution_x
        row = (y - self.y_range[0]) / self.resolution_y

        # Bilinear interpolation
        # Clip to valid range, using epsilon to avoid index out of bounds
        row = np.clip(row, 0, self.heights.shape[0] - 1 - 1e-10)
        col = np.clip(col, 0, self.heights.shape[1] - 1 - 1e-10)

        r0, c0 = int(row), int(col)
        r1, c1 = min(r0 + 1, self.heights.shape[0] - 1), min(c0 + 1, self.heights.shape[1] - 1)
        fr, fc = row - r0, col - c0

        h00 = self.heights[r0, c0]
        h01 = self.heights[r0, c1]
        h10 = self.heights[r1, c0]
        h11 = self.heights[r1, c1]

        return float(
            h00 * (1 - fr) * (1 - fc)
            + h01 * (1 - fr) * fc
            + h10 * fr * (1 - fc)
            + h11 * fr * fc
        )

    def normal_at(self, x: float, y: float) -> NDArray[np.floating]:
        """Compute surface normal at (x, y) using finite differences."""
        eps = min(self.resolution_x, self.resolution_y) * 0.5
        hx_plus = self.height_at(x + eps, y)
        hx_minus = self.height_at(x - eps, y)
        hy_plus = self.height_at(x, y + eps)
        hy_minus = self.height_at(x, y - eps)

        dz_dx = (hx_plus - hx_minus) / (2 * eps)
        dz_dy = (hy_plus - hy_minus) / (2 * eps)

        normal = np.array([-dz_dx, -dz_dy, 1.0])
        return normal / np.linalg.norm(normal)

    def sample_surface(self, n_points: int) -> NDArray[np.floating]:
        """Sample random points on terrain surface."""
        x = np.random.uniform(self.x_range[0], self.x_range[1], n_points)
        y = np.random.uniform(self.y_range[0], self.y_range[1], n_points)
        z = np.array([self.height_at(xi, yi) for xi, yi in zip(x, y)])
        return np.column_stack([x, y, z])

    def to_mesh(self) -> TriangleMesh:
        """Convert height field to triangle mesh."""
        rows, cols = self.heights.shape

        # Create vertices
        x = np.linspace(self.x_range[0], self.x_range[1], cols)
        y = np.linspace(self.y_range[0], self.y_range[1], rows)
        xx, yy = np.meshgrid(x, y)

        vertices = np.column_stack([xx.ravel(), yy.ravel(), self.heights.ravel()])

        # Create faces (two triangles per grid cell)
        faces = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                v00 = i * cols + j
                v01 = i * cols + (j + 1)
                v10 = (i + 1) * cols + j
                v11 = (i + 1) * cols + (j + 1)

                faces.append([v00, v10, v01])
                faces.append([v01, v10, v11])

        return TriangleMesh(vertices=vertices, faces=np.array(faces))


def create_plane(
    width: float = 1.0,
    height: float = 1.0,
    center: Optional[NDArray[np.floating]] = None,
) -> TriangleMesh:
    """Create a simple plane (ground) mesh."""
    if center is None:
        center = np.zeros(3)

    hw, hh = width / 2, height / 2
    vertices = np.array([
        [center[0] - hw, center[1] - hh, center[2]],
        [center[0] + hw, center[1] - hh, center[2]],
        [center[0] + hw, center[1] + hh, center[2]],
        [center[0] - hw, center[1] + hh, center[2]],
    ])

    faces = np.array([[0, 1, 2], [0, 2, 3]])
    normals = np.array([[0, 0, 1]] * 4, dtype=np.float64)

    return TriangleMesh(vertices=vertices, faces=faces, normals=normals)


def create_box(
    size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    center: Optional[NDArray[np.floating]] = None,
) -> TriangleMesh:
    """Create a simple box mesh."""
    if center is None:
        center = np.zeros(3)

    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    cx, cy, cz = center

    vertices = np.array([
        [cx - hx, cy - hy, cz - hz],  # 0
        [cx + hx, cy - hy, cz - hz],  # 1
        [cx + hx, cy + hy, cz - hz],  # 2
        [cx - hx, cy + hy, cz - hz],  # 3
        [cx - hx, cy - hy, cz + hz],  # 4
        [cx + hx, cy - hy, cz + hz],  # 5
        [cx + hx, cy + hy, cz + hz],  # 6
        [cx - hx, cy + hy, cz + hz],  # 7
    ])

    # 12 triangles (2 per face)
    faces = np.array([
        # Bottom
        [0, 2, 1], [0, 3, 2],
        # Top
        [4, 5, 6], [4, 6, 7],
        # Front
        [0, 1, 5], [0, 5, 4],
        # Back
        [2, 3, 7], [2, 7, 6],
        # Left
        [0, 4, 7], [0, 7, 3],
        # Right
        [1, 2, 6], [1, 6, 5],
    ])

    return TriangleMesh(vertices=vertices, faces=faces)


def load_obj(filepath: Union[str, Path]) -> TriangleMesh:
    """Load mesh from OBJ file (basic loader).

    Args:
        filepath: Path to OBJ file

    Returns:
        TriangleMesh loaded from file
    """
    vertices = []
    normals = []
    faces = []
    normal_indices = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if parts[0] == "v":
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == "vn":
                normals.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                face_verts = []
                face_norms = []
                for p in parts[1:]:
                    indices = p.split("/")
                    face_verts.append(int(indices[0]) - 1)  # OBJ is 1-indexed
                    if len(indices) > 2 and indices[2]:
                        face_norms.append(int(indices[2]) - 1)
                # Triangulate if needed (simple fan triangulation)
                for i in range(1, len(face_verts) - 1):
                    faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])
                    if face_norms:
                        normal_indices.append([face_norms[0], face_norms[i], face_norms[i + 1]])

    mesh_normals = None
    if normals and normal_indices:
        # Convert to vertex normals (simplified)
        mesh_normals = np.array(normals, dtype=np.float64)

    return TriangleMesh(
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int32),
        normals=mesh_normals,
    )
