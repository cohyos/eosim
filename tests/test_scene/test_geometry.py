"""Tests for scene.geometry module."""

import pytest
import numpy as np

from eosim.scene.geometry import (
    BoundingBox,
    Transform,
    TriangleMesh,
    HeightField,
    create_plane,
    create_box,
)


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_create_from_points(self) -> None:
        """Create bounding box from point cloud."""
        points = np.array([
            [0, 0, 0],
            [1, 2, 3],
            [-1, 0, 1],
        ])
        bbox = BoundingBox.from_points(points)
        assert np.allclose(bbox.min_point, [-1, 0, 0])
        assert np.allclose(bbox.max_point, [1, 2, 3])

    def test_center(self) -> None:
        """Test center computation."""
        bbox = BoundingBox(
            min_point=np.array([0, 0, 0]),
            max_point=np.array([2, 4, 6]),
        )
        assert np.allclose(bbox.center, [1, 2, 3])

    def test_size(self) -> None:
        """Test size computation."""
        bbox = BoundingBox(
            min_point=np.array([0, 0, 0]),
            max_point=np.array([2, 4, 6]),
        )
        assert np.allclose(bbox.size, [2, 4, 6])

    def test_contains_point(self) -> None:
        """Test point containment."""
        bbox = BoundingBox(
            min_point=np.array([0, 0, 0]),
            max_point=np.array([1, 1, 1]),
        )
        assert bbox.contains(np.array([0.5, 0.5, 0.5]))
        assert not bbox.contains(np.array([2, 0.5, 0.5]))

    def test_intersects(self) -> None:
        """Test box intersection."""
        bbox1 = BoundingBox(
            min_point=np.array([0, 0, 0]),
            max_point=np.array([2, 2, 2]),
        )
        bbox2 = BoundingBox(
            min_point=np.array([1, 1, 1]),
            max_point=np.array([3, 3, 3]),
        )
        bbox3 = BoundingBox(
            min_point=np.array([5, 5, 5]),
            max_point=np.array([6, 6, 6]),
        )
        assert bbox1.intersects(bbox2)
        assert not bbox1.intersects(bbox3)


class TestTransform:
    """Tests for Transform class."""

    def test_identity(self) -> None:
        """Identity transform should not change points."""
        transform = Transform()
        points = np.array([[1, 2, 3], [4, 5, 6]])
        transformed = transform.apply(points)
        assert np.allclose(transformed, points)

    def test_translation(self) -> None:
        """Test translation transform."""
        transform = Transform(translation=np.array([1, 2, 3]))
        point = np.array([[0, 0, 0]])
        transformed = transform.apply(point)
        assert np.allclose(transformed, [[1, 2, 3]])

    def test_scale(self) -> None:
        """Test scale transform."""
        transform = Transform(scale=np.array([2, 2, 2]))
        point = np.array([[1, 1, 1]])
        transformed = transform.apply(point)
        assert np.allclose(transformed, [[2, 2, 2]])

    def test_from_euler_xyz(self) -> None:
        """Test creating transform from Euler angles."""
        # 90 degree rotation around Z
        angles = np.array([0, 0, np.pi / 2])
        transform = Transform.from_euler_xyz(
            translation=np.zeros(3),
            angles_rad=angles,
        )
        point = np.array([[1, 0, 0]])
        transformed = transform.apply(point)
        # X axis should become Y axis
        assert np.allclose(transformed, [[0, 1, 0]], atol=1e-10)


class TestTriangleMesh:
    """Tests for TriangleMesh class."""

    def test_create_mesh(self) -> None:
        """Create simple triangle mesh."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ])
        faces = np.array([[0, 1, 2]])
        mesh = TriangleMesh(vertices=vertices, faces=faces)

        assert mesh.n_vertices == 3
        assert mesh.n_faces == 1

    def test_normals_computed(self) -> None:
        """Normals should be computed if not provided."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ])
        faces = np.array([[0, 1, 2]])
        mesh = TriangleMesh(vertices=vertices, faces=faces)

        # Normal should point in +Z for XY plane triangle
        assert mesh.normals is not None
        # Vertex normals should point roughly in +Z
        assert np.all(mesh.normals[:, 2] > 0)

    def test_face_areas(self) -> None:
        """Test face area computation."""
        # Unit right triangle: area = 0.5
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ])
        faces = np.array([[0, 1, 2]])
        mesh = TriangleMesh(vertices=vertices, faces=faces)
        areas = mesh.face_areas()
        assert areas[0] == pytest.approx(0.5)

    def test_bounds(self) -> None:
        """Test bounding box computation."""
        vertices = np.array([
            [0, 0, 0],
            [2, 0, 0],
            [1, 3, 4],
        ])
        faces = np.array([[0, 1, 2]])
        mesh = TriangleMesh(vertices=vertices, faces=faces)

        bounds = mesh.bounds
        assert np.allclose(bounds.min_point, [0, 0, 0])
        assert np.allclose(bounds.max_point, [2, 3, 4])

    def test_sample_surface(self) -> None:
        """Test surface sampling."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ])
        faces = np.array([[0, 1, 2]])
        mesh = TriangleMesh(vertices=vertices, faces=faces)

        samples = mesh.sample_surface(100)
        assert samples.shape == (100, 3)
        # All points should be in XY plane (z=0)
        assert np.allclose(samples[:, 2], 0)


class TestHeightField:
    """Tests for HeightField class."""

    def test_create_heightfield(self) -> None:
        """Create simple height field."""
        heights = np.zeros((10, 10))
        hf = HeightField(heights=heights)
        assert hf.shape == (10, 10)

    def test_height_at(self) -> None:
        """Test height interpolation."""
        heights = np.array([
            [0, 1],
            [0, 1],
        ])
        hf = HeightField(heights=heights, x_range=(0, 1), y_range=(0, 1))

        # At corners
        assert hf.height_at(0, 0) == pytest.approx(0)
        assert hf.height_at(1, 0) == pytest.approx(1)

        # Interpolated
        assert hf.height_at(0.5, 0) == pytest.approx(0.5)

    def test_bounds(self) -> None:
        """Test bounding box."""
        heights = np.array([[0, 5], [10, 15]])
        hf = HeightField(heights=heights, x_range=(0, 100), y_range=(0, 100))
        bounds = hf.bounds
        assert bounds.min_point[2] == 0
        assert bounds.max_point[2] == 15

    def test_to_mesh(self) -> None:
        """Test conversion to mesh."""
        heights = np.zeros((3, 3))
        hf = HeightField(heights=heights)
        mesh = hf.to_mesh()

        assert mesh.n_vertices == 9  # 3x3 grid
        assert mesh.n_faces == 8  # 2x2 cells, 2 triangles each


class TestPrimitives:
    """Tests for primitive creation functions."""

    def test_create_plane(self) -> None:
        """Test plane creation."""
        plane = create_plane(width=2.0, height=3.0)
        assert plane.n_vertices == 4
        assert plane.n_faces == 2
        assert plane.total_area() == pytest.approx(6.0)

    def test_create_box(self) -> None:
        """Test box creation."""
        box = create_box(size=(1, 2, 3))
        assert box.n_vertices == 8
        assert box.n_faces == 12  # 6 faces × 2 triangles
        # Surface area = 2(1×2 + 1×3 + 2×3) = 2(2+3+6) = 22
        assert box.total_area() == pytest.approx(22.0)
