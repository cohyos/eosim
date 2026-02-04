# Object Management Architecture

## 1. Introduction

This document describes how EOSIM manages arena objects from definition through rendering:

1. **Input**: How objects are defined and loaded
2. **Storage**: Internal representation and scene database
3. **Spatial Organization**: Efficient access for rendering
4. **Property Query**: How renderer accesses physical properties
5. **Visibility**: Sensor-object intersection and occlusion
6. **Rendering Interface**: What the sensor "sees"

---

## 2. Object Lifecycle

### 2.1 Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OBJECT LIFECYCLE PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  DEFINITION │───▶│   LOADING   │───▶│  REGISTRY   │───▶│    SCENE    │
  │             │    │             │    │             │    │    GRAPH    │
  │ YAML/JSON   │    │ Parse +     │    │ Object DB   │    │ Spatial     │
  │ + 3D Models │    │ Validate    │    │ + Materials │    │ Hierarchy   │
  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                  │
        ┌─────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │   SPATIAL   │───▶│  THERMAL    │───▶│ VISIBILITY  │───▶│  RENDERING  │
  │   INDEX     │    │   STATE     │    │   QUERY     │    │             │
  │             │    │             │    │             │    │ Per-pixel   │
  │ BVH/Octree  │    │ T(x,y,z,t)  │    │ Ray-object  │    │ radiance    │
  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                  │
                                                                  ▼
                                                          ┌─────────────┐
                                                          │   SENSOR    │
                                                          │   OUTPUT    │
                                                          │             │
                                                          │ DN Image    │
                                                          └─────────────┘
```

---

## 3. Object Definition (Input)

### 3.1 Definition File Structure

```yaml
# objects/vehicles/tank_t72.yaml
# Complete object definition

object:
  # === IDENTITY ===
  id: "tank_t72"
  name: "T-72 Main Battle Tank"
  category: "ground_vehicle"
  subcategory: "main_battle_tank"
  version: "1.0"

  # === GEOMETRY ===
  geometry:
    source:
      format: "gltf"
      file: "models/vehicles/t72/t72_main.glb"

    transform:
      scale: 1.0
      up_axis: "Z"
      forward_axis: "Y"
      origin: "center_bottom"  # Where (0,0,0) is in model

    lod_levels:
      - level: 0
        file: "models/vehicles/t72/t72_lod0.glb"
        max_distance_m: 200
        triangle_count: 45000
      - level: 1
        file: "models/vehicles/t72/t72_lod1.glb"
        max_distance_m: 1000
        triangle_count: 8000
      - level: 2
        file: "models/vehicles/t72/t72_lod2.glb"
        max_distance_m: 5000
        triangle_count: 1500
      - level: 3
        type: "billboard"
        max_distance_m: .inf

    bounding_box:
      length_m: 9.53
      width_m: 3.59
      height_m: 2.23

  # === MATERIAL ZONES ===
  material_zones:
    # Each zone maps mesh parts to material + thermal properties

    - zone_id: "hull_upper"
      description: "Upper hull painted surfaces"
      mesh_groups: ["Hull_Upper", "Turret_Ring"]

      material:
        base: "olive_drab_paint"     # From material library
        overrides:
          emissivity_lwir: 0.91
          solar_absorptivity: 0.87

      thermal:
        model: "surface_balance"
        thickness_m: 0.05
        thermal_mass_factor: 1.0
        internal_coupling: "engine_bay"  # Receives heat from engine
        coupling_coefficient: 0.05

    - zone_id: "hull_lower"
      mesh_groups: ["Hull_Lower", "Skirts"]
      material:
        base: "olive_drab_paint"
      thermal:
        model: "surface_balance"
        ground_contact: true         # Different boundary condition

    - zone_id: "turret"
      mesh_groups: ["Turret_Body", "Mantlet", "Commander_Cupola"]
      material:
        base: "olive_drab_paint"
      thermal:
        model: "surface_balance"

    - zone_id: "engine_bay"
      mesh_groups: ["Engine_Deck", "Engine_Grilles", "Exhaust_Grilles"]
      material:
        base: "steel_painted"
        overrides:
          emissivity_lwir: 0.85
      thermal:
        model: "internal_source"
        heat_source:
          type: "state_dependent"
          parameter: "engine_power_fraction"
          max_power_W: 25000         # Waste heat when running

    - zone_id: "exhaust_stack"
      mesh_groups: ["Exhaust_Pipe_L", "Exhaust_Pipe_R"]
      material:
        base: "bare_steel"
        overrides:
          emissivity_lwir: 0.40
      thermal:
        model: "internal_source"
        heat_source:
          type: "state_dependent"
          parameter: "engine_power_fraction"
          max_temperature_K: 573     # 300°C max

    - zone_id: "tracks"
      mesh_groups: ["Track_L", "Track_R", "RoadWheels_*", "Sprockets"]
      material:
        base: "track_composite"
      thermal:
        model: "friction_heating"
        friction_coefficient: 0.015
        reference_speed_mps: 15
        max_delta_T_K: 35

    - zone_id: "gun_barrel"
      mesh_groups: ["MainGun_Barrel", "MainGun_Muzzle"]
      material:
        base: "bare_steel"
        overrides:
          emissivity_lwir: 0.25
      thermal:
        model: "surface_balance"
        special_events:
          - event: "gun_fired"
            delta_T_K: 150
            decay_time_s: 300

    - zone_id: "optics"
      mesh_groups: ["Gunner_Sight", "Commander_Sight", "Driver_Periscopes"]
      material:
        base: "ir_glass"
        overrides:
          emissivity_lwir: 0.10      # Highly reflective
          reflectivity_lwir: 0.90

  # === ARTICULATION ===
  articulation:
    joints:
      - name: "turret_traverse"
        type: "revolute"
        parent: null                 # Attached to hull
        child_meshes: ["Turret_*", "MainGun_*", "Commander_*", "Loader_Hatch"]
        axis: [0, 0, 1]             # Z-axis rotation
        limits_deg: [-180, 180]
        max_rate_deg_s: 24

      - name: "gun_elevation"
        type: "revolute"
        parent: "turret_traverse"
        child_meshes: ["MainGun_*"]
        axis: [1, 0, 0]             # X-axis rotation
        limits_deg: [-6, 14]
        max_rate_deg_s: 4

      - name: "commander_hatch"
        type: "revolute"
        parent: "turret_traverse"
        child_meshes: ["Commander_Hatch"]
        axis: [1, 0, 0]
        limits_deg: [0, 90]

      - name: "loader_hatch"
        type: "revolute"
        parent: "turret_traverse"
        child_meshes: ["Loader_Hatch"]
        axis: [-1, 0, 0]
        limits_deg: [0, 90]

  # === STATE VARIABLES ===
  state_variables:
    - name: "engine_running"
      type: "boolean"
      default: false

    - name: "engine_power_fraction"
      type: "float"
      range: [0, 1]
      default: 0
      depends_on: "engine_running"

    - name: "speed_mps"
      type: "float"
      range: [0, 18]               # Max 65 km/h
      default: 0

    - name: "gun_fired_time"
      type: "timestamp"
      default: null
      description: "Time of last main gun firing"

  # === THERMAL MODEL PARAMETERS ===
  thermal_config:
    warmup_time_constant_s: 300     # Engine warmup τ
    cooldown_time_constant_s: 600   # Engine cooldown τ
    hull_engine_coupling: 0.15      # Fraction of engine heat to hull
    ambient_reference: "air"        # or "ground" for buried objects

  # === SIGNATURE METADATA ===
  signatures:
    ir_signature_class: "MBT"
    radar_rcs_m2: 15.0
    visual_camouflage: "woodland"
```

### 3.2 Object Loading Pipeline

```python
# src/eosim/scene/object_loader.py

from pathlib import Path
from typing import Dict, Optional
import yaml

class ObjectLoader:
    """
    Loads object definitions from YAML files and 3D geometry.
    """

    def __init__(
        self,
        object_library_path: Path,
        material_library: 'MaterialLibrary',
        geometry_cache: Optional['GeometryCache'] = None
    ):
        self.library_path = object_library_path
        self.materials = material_library
        self.geo_cache = geometry_cache or GeometryCache()

    def load_object(self, object_id: str) -> 'ArenaObject':
        """
        Load complete object from definition file.

        Args:
            object_id: Object identifier (e.g., "tank_t72")

        Returns:
            Fully initialized ArenaObject
        """
        # Find definition file
        def_file = self._find_definition(object_id)

        # Parse YAML
        with open(def_file) as f:
            definition = yaml.safe_load(f)['object']

        # Validate schema
        self._validate_definition(definition)

        # Load geometry (with LOD)
        geometry = self._load_geometry(definition['geometry'])

        # Build material zones
        material_zones = self._build_material_zones(
            definition['material_zones'],
            geometry
        )

        # Build articulation system
        articulation = self._build_articulation(
            definition.get('articulation'),
            geometry
        )

        # Build thermal model
        thermal_model = self._build_thermal_model(
            definition['material_zones'],
            definition.get('thermal_config', {})
        )

        # Build state machine
        state = self._build_state(definition.get('state_variables', []))

        # Create object
        obj = ArenaObject(
            id=definition['id'],
            name=definition['name'],
            category=definition['category'],
            geometry=geometry,
            material_zones=material_zones,
            articulation=articulation,
            thermal_model=thermal_model,
            state=state,
            metadata=definition
        )

        return obj

    def _load_geometry(self, geo_config: dict) -> 'ObjectGeometry':
        """Load 3D geometry with all LOD levels."""

        lod_levels = []
        for lod in geo_config.get('lod_levels', []):
            if lod.get('type') == 'billboard':
                geo = BillboardGeometry()
            else:
                geo = self.geo_cache.load(
                    lod['file'],
                    format=geo_config['source']['format']
                )
            lod_levels.append(LODLevel(
                level=lod['level'],
                geometry=geo,
                switch_distance=lod['max_distance_m']
            ))

        return MultiLODGeometry(
            lod_levels=lod_levels,
            transform=self._parse_transform(geo_config['transform']),
            bounding_box=self._parse_bbox(geo_config['bounding_box'])
        )

    def _build_material_zones(
        self,
        zone_configs: list,
        geometry: 'ObjectGeometry'
    ) -> Dict[str, 'MaterialZone']:
        """Build material zones from config."""

        zones = {}
        for zc in zone_configs:
            # Get base material from library
            base_mat = self.materials.get(zc['material']['base'])

            # Apply overrides
            if 'overrides' in zc['material']:
                mat = base_mat.with_overrides(zc['material']['overrides'])
            else:
                mat = base_mat

            # Find mesh faces for this zone
            face_indices = geometry.get_faces_by_groups(zc['mesh_groups'])

            zones[zc['zone_id']] = MaterialZone(
                zone_id=zc['zone_id'],
                material=mat,
                face_indices=face_indices,
                thermal_config=zc.get('thermal', {})
            )

        return zones
```

---

## 4. Object Registry (Storage)

### 4.1 Scene Database

```python
# src/eosim/scene/registry.py

from typing import Dict, List, Optional, Iterator
import numpy as np

class ObjectRegistry:
    """
    Central registry of all objects in the scene.
    Provides efficient lookup and query capabilities.
    """

    def __init__(self):
        # Primary storage
        self._objects: Dict[str, ArenaObject] = {}

        # Indices for fast lookup
        self._by_category: Dict[str, List[str]] = {}
        self._by_type: Dict[str, List[str]] = {}

        # Spatial index (built on demand)
        self._spatial_index: Optional[SpatialIndex] = None
        self._spatial_dirty = True

    def register(self, obj: 'ArenaObject', instance_id: str = None) -> str:
        """
        Register an object in the scene.

        Args:
            obj: Object to register
            instance_id: Optional unique instance ID

        Returns:
            Instance ID for this object
        """
        # Generate instance ID if not provided
        if instance_id is None:
            instance_id = f"{obj.id}_{len(self._objects):04d}"

        # Store object
        self._objects[instance_id] = obj

        # Update indices
        cat = obj.category
        if cat not in self._by_category:
            self._by_category[cat] = []
        self._by_category[cat].append(instance_id)

        # Mark spatial index as needing rebuild
        self._spatial_dirty = True

        return instance_id

    def get(self, instance_id: str) -> Optional['ArenaObject']:
        """Get object by instance ID."""
        return self._objects.get(instance_id)

    def get_by_category(self, category: str) -> List['ArenaObject']:
        """Get all objects of a category."""
        ids = self._by_category.get(category, [])
        return [self._objects[id] for id in ids]

    def query_region(
        self,
        bounds: 'BoundingBox'
    ) -> List['ArenaObject']:
        """
        Query objects intersecting a bounding box.
        Uses spatial index for efficiency.
        """
        self._ensure_spatial_index()
        return self._spatial_index.query_box(bounds)

    def query_frustum(
        self,
        frustum: 'ViewFrustum'
    ) -> List['ArenaObject']:
        """
        Query objects visible within camera frustum.
        """
        self._ensure_spatial_index()
        return self._spatial_index.query_frustum(frustum)

    def iter_all(self) -> Iterator['ArenaObject']:
        """Iterate over all registered objects."""
        yield from self._objects.values()

    def _ensure_spatial_index(self):
        """Rebuild spatial index if needed."""
        if self._spatial_dirty:
            self._spatial_index = SpatialIndex()
            for id, obj in self._objects.items():
                self._spatial_index.insert(id, obj.world_bounds)
            self._spatial_dirty = False
```

### 4.2 Object Instance in Scene

```python
# src/eosim/scene/object_instance.py

@dataclass
class ObjectInstance:
    """
    Instance of an object placed in the scene.
    Combines object template with instance-specific state.
    """

    # Identity
    instance_id: str
    template: 'ArenaObject'          # Reference to object definition

    # Placement
    position: np.ndarray             # World position (x, y, z)
    rotation: np.ndarray             # Euler angles (roll, pitch, yaw) or quaternion
    scale: float = 1.0

    # Instance state (overrides template defaults)
    state: 'ObjectState'

    # Trajectory (for dynamic objects)
    trajectory: Optional['Trajectory'] = None

    # Cached transforms
    _world_transform: np.ndarray = None
    _world_bounds: 'BoundingBox' = None

    @property
    def world_transform(self) -> np.ndarray:
        """4x4 world transformation matrix."""
        if self._world_transform is None:
            self._world_transform = compute_transform(
                self.position, self.rotation, self.scale
            )
        return self._world_transform

    @property
    def world_bounds(self) -> 'BoundingBox':
        """Axis-aligned bounding box in world coordinates."""
        if self._world_bounds is None:
            local_bounds = self.template.geometry.bounding_box
            self._world_bounds = local_bounds.transform(self.world_transform)
        return self._world_bounds

    def update(self, time_s: float):
        """
        Update instance state for given simulation time.
        """
        # Update position/rotation from trajectory
        if self.trajectory is not None:
            pos, vel, att = self.trajectory.get_state(time_s)
            self.position = pos
            self.rotation = att
            self._world_transform = None  # Invalidate cache
            self._world_bounds = None

            # Update velocity-dependent state
            speed = np.linalg.norm(vel)
            self.state.set('speed_mps', speed)

        # Update articulation
        self.template.articulation.update(time_s)

    def invalidate_cache(self):
        """Invalidate cached transforms."""
        self._world_transform = None
        self._world_bounds = None
```

---

## 5. Spatial Organization

### 5.1 Bounding Volume Hierarchy (BVH)

```python
# src/eosim/scene/spatial_index.py

class BVHNode:
    """Node in Bounding Volume Hierarchy."""

    def __init__(self):
        self.bounds: BoundingBox = None
        self.left: Optional[BVHNode] = None
        self.right: Optional[BVHNode] = None
        self.objects: List[str] = []  # Leaf node: object IDs

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class SpatialIndex:
    """
    Spatial acceleration structure for fast object queries.
    Uses BVH (Bounding Volume Hierarchy).
    """

    def __init__(self, max_leaf_size: int = 4):
        self.root: BVHNode = None
        self.max_leaf_size = max_leaf_size
        self._objects: Dict[str, BoundingBox] = {}

    def insert(self, object_id: str, bounds: BoundingBox):
        """Add object to index."""
        self._objects[object_id] = bounds

    def build(self):
        """Build BVH from inserted objects."""
        if not self._objects:
            return

        items = list(self._objects.items())
        self.root = self._build_recursive(items, 0)

    def _build_recursive(
        self,
        items: List[tuple],
        depth: int
    ) -> BVHNode:
        """Recursively build BVH."""
        node = BVHNode()

        # Compute bounding box for all items
        node.bounds = BoundingBox.union([b for _, b in items])

        # Leaf node if few items
        if len(items) <= self.max_leaf_size:
            node.objects = [id for id, _ in items]
            return node

        # Choose split axis (cycle through x, y, z)
        axis = depth % 3

        # Sort by centroid along axis
        items.sort(key=lambda x: x[1].center[axis])

        # Split in middle
        mid = len(items) // 2
        node.left = self._build_recursive(items[:mid], depth + 1)
        node.right = self._build_recursive(items[mid:], depth + 1)

        return node

    def query_box(self, query: BoundingBox) -> List[str]:
        """Find all objects intersecting query box."""
        result = []
        self._query_box_recursive(self.root, query, result)
        return result

    def _query_box_recursive(
        self,
        node: BVHNode,
        query: BoundingBox,
        result: List[str]
    ):
        if node is None:
            return

        if not node.bounds.intersects(query):
            return

        if node.is_leaf():
            # Check individual objects
            for obj_id in node.objects:
                if self._objects[obj_id].intersects(query):
                    result.append(obj_id)
        else:
            self._query_box_recursive(node.left, query, result)
            self._query_box_recursive(node.right, query, result)

    def query_frustum(self, frustum: 'ViewFrustum') -> List[str]:
        """Find all objects inside view frustum."""
        result = []
        self._query_frustum_recursive(self.root, frustum, result)
        return result

    def query_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_distance: float = float('inf')
    ) -> List[tuple]:
        """
        Find all objects intersected by ray.

        Returns:
            List of (object_id, t_near, t_far) sorted by distance
        """
        result = []
        self._query_ray_recursive(self.root, origin, direction, max_distance, result)
        result.sort(key=lambda x: x[1])
        return result
```

---

## 6. Property Query System

### 6.1 Property Interface

```python
# src/eosim/scene/property_query.py

class PropertyQueryInterface:
    """
    Interface for querying object properties at any point.
    Used by renderer to get material/thermal properties.
    """

    def __init__(self, registry: ObjectRegistry, thermal_state: 'ThermalState'):
        self.registry = registry
        self.thermal = thermal_state

    def query_point(
        self,
        world_point: np.ndarray,
        wavelength_um: float,
        query_type: str = "full"
    ) -> 'SurfaceProperties':
        """
        Query surface properties at a world point.

        Args:
            world_point: (x, y, z) in world coordinates
            wavelength_um: Wavelength for spectral properties
            query_type: "full", "radiometric", "thermal", "geometric"

        Returns:
            SurfaceProperties at that point
        """
        # Find which object/face contains this point
        hit = self._find_surface(world_point)

        if hit is None:
            return None

        obj = hit.object
        face_idx = hit.face_index
        zone = obj.get_zone_for_face(face_idx)

        # Build surface properties
        props = SurfaceProperties()

        # Geometric properties
        props.position = world_point
        props.normal = hit.normal
        props.uv = hit.uv

        # Material properties (spectral)
        mat = zone.material
        props.emissivity = mat.get_emissivity(wavelength_um)
        props.reflectivity = mat.get_reflectivity(wavelength_um)
        props.brdf = mat.brdf

        # Thermal properties
        props.temperature_K = self.thermal.get_temperature(
            obj.instance_id,
            zone.zone_id,
            world_point
        )

        return props

    def query_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        wavelength_um: float,
        max_distance: float = float('inf')
    ) -> Optional['RayHitResult']:
        """
        Cast ray and get properties at hit point.

        Primary interface for ray-traced rendering.
        """
        # Find nearest intersection
        hit = self._trace_ray(origin, direction, max_distance)

        if hit is None:
            return None

        # Query properties at hit point
        props = self.query_point(hit.position, wavelength_um)

        return RayHitResult(
            hit_distance=hit.distance,
            position=hit.position,
            normal=hit.normal,
            properties=props
        )


@dataclass
class SurfaceProperties:
    """Properties at a surface point."""

    # Geometry
    position: np.ndarray
    normal: np.ndarray
    uv: Optional[np.ndarray] = None

    # Radiometric
    emissivity: float = 0.9
    reflectivity: float = 0.1
    transmissivity: float = 0.0
    brdf: Optional['BRDF'] = None

    # Thermal
    temperature_K: float = 300.0

    # Object reference
    object_id: str = None
    zone_id: str = None


@dataclass
class RayHitResult:
    """Result of ray-scene intersection."""

    hit_distance: float
    position: np.ndarray
    normal: np.ndarray
    properties: SurfaceProperties

    # For transparency/multiple hits
    next_hit: Optional['RayHitResult'] = None
```

### 6.2 Per-Pixel Property Map

```python
class PropertyMapGenerator:
    """
    Generate per-pixel property maps for the sensor view.
    These maps accelerate rendering by pre-computing lookups.
    """

    def __init__(
        self,
        registry: ObjectRegistry,
        thermal_state: 'ThermalState'
    ):
        self.registry = registry
        self.thermal = thermal_state

    def generate(
        self,
        sensor_geometry: 'SensorGeometry',
        wavelength_um: float
    ) -> 'PropertyMaps':
        """
        Generate property maps for sensor view.

        Returns maps of:
        - Object ID per pixel
        - Zone ID per pixel
        - Surface normal per pixel
        - Temperature per pixel
        - Emissivity per pixel
        - Distance per pixel
        """
        H, W = sensor_geometry.resolution

        maps = PropertyMaps(
            object_id=np.zeros((H, W), dtype=np.int32),
            zone_id=np.zeros((H, W), dtype=np.int32),
            normal=np.zeros((H, W, 3), dtype=np.float32),
            temperature=np.zeros((H, W), dtype=np.float32),
            emissivity=np.zeros((H, W), dtype=np.float32),
            distance=np.zeros((H, W), dtype=np.float32),
        )

        # For each pixel, cast ray and populate maps
        for i in range(H):
            for j in range(W):
                ray = sensor_geometry.get_ray(i, j)
                hit = self._trace_and_query(ray, wavelength_um)

                if hit is not None:
                    maps.object_id[i, j] = hit.object_id_int
                    maps.zone_id[i, j] = hit.zone_id_int
                    maps.normal[i, j] = hit.normal
                    maps.temperature[i, j] = hit.temperature_K
                    maps.emissivity[i, j] = hit.emissivity
                    maps.distance[i, j] = hit.distance
                else:
                    # Sky/background
                    maps.distance[i, j] = float('inf')

        return maps
```

---

## 7. Visibility and Sensor View

### 7.1 View Frustum Culling

```python
# src/eosim/scene/visibility.py

class ViewFrustum:
    """
    Camera view frustum for visibility culling.
    """

    def __init__(
        self,
        position: np.ndarray,
        look_at: np.ndarray,
        up: np.ndarray,
        fov_h_deg: float,
        fov_v_deg: float,
        near: float,
        far: float
    ):
        self.position = position
        self.look_at = look_at
        self.up = up
        self.fov_h = np.radians(fov_h_deg)
        self.fov_v = np.radians(fov_v_deg)
        self.near = near
        self.far = far

        # Compute frustum planes
        self._compute_planes()

    def _compute_planes(self):
        """Compute 6 frustum planes (near, far, left, right, top, bottom)."""
        # Forward, right, up vectors
        forward = (self.look_at - self.position)
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, self.up)
        right /= np.linalg.norm(right)

        up = np.cross(right, forward)

        # Plane normals and distances
        self.planes = []

        # Near plane
        self.planes.append(Plane(forward, self.position + forward * self.near))

        # Far plane
        self.planes.append(Plane(-forward, self.position + forward * self.far))

        # Side planes (pointing inward)
        half_h = np.tan(self.fov_h / 2)
        half_v = np.tan(self.fov_v / 2)

        # Left
        left_normal = forward + right * half_h
        left_normal /= np.linalg.norm(left_normal)
        self.planes.append(Plane(left_normal, self.position))

        # Right
        right_normal = forward - right * half_h
        right_normal /= np.linalg.norm(right_normal)
        self.planes.append(Plane(right_normal, self.position))

        # Top
        top_normal = forward - up * half_v
        top_normal /= np.linalg.norm(top_normal)
        self.planes.append(Plane(top_normal, self.position))

        # Bottom
        bottom_normal = forward + up * half_v
        bottom_normal /= np.linalg.norm(bottom_normal)
        self.planes.append(Plane(bottom_normal, self.position))

    def contains_point(self, point: np.ndarray) -> bool:
        """Test if point is inside frustum."""
        for plane in self.planes:
            if plane.signed_distance(point) < 0:
                return False
        return True

    def intersects_box(self, box: BoundingBox) -> bool:
        """Test if box intersects frustum."""
        for plane in self.planes:
            # Check if all corners are outside this plane
            corners = box.corners
            all_outside = all(
                plane.signed_distance(c) < 0 for c in corners
            )
            if all_outside:
                return False
        return True


class VisibilityResolver:
    """
    Determine which objects are visible to the sensor.
    """

    def __init__(self, registry: ObjectRegistry, spatial_index: SpatialIndex):
        self.registry = registry
        self.spatial = spatial_index

    def get_visible_objects(
        self,
        sensor_geometry: 'SensorGeometry'
    ) -> List['ObjectInstance']:
        """
        Get all objects potentially visible to sensor.

        Steps:
        1. Frustum culling (coarse)
        2. Distance culling
        3. Occlusion culling (optional)
        """
        frustum = sensor_geometry.view_frustum

        # Frustum query
        candidates = self.spatial.query_frustum(frustum)

        # Distance culling
        max_range = sensor_geometry.max_range_m
        visible = []
        for obj_id in candidates:
            obj = self.registry.get(obj_id)
            dist = np.linalg.norm(obj.position - sensor_geometry.position)
            if dist <= max_range:
                visible.append(obj)

        # Sort by distance (for potential occlusion culling)
        visible.sort(key=lambda o: np.linalg.norm(o.position - sensor_geometry.position))

        return visible
```

### 7.2 Ray-Object Intersection

```python
# src/eosim/scene/intersection.py

class RayIntersector:
    """
    Ray-mesh intersection testing.
    """

    def __init__(self, use_embree: bool = True):
        """
        Args:
            use_embree: Use Intel Embree for hardware-accelerated ray tracing
        """
        self.use_embree = use_embree
        if use_embree:
            try:
                import embree
                self.embree_available = True
            except ImportError:
                self.embree_available = False
                self.use_embree = False

    def intersect(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        geometry: 'ObjectGeometry',
        transform: np.ndarray
    ) -> Optional['IntersectionResult']:
        """
        Find intersection of ray with object geometry.

        Returns:
            IntersectionResult or None if no hit
        """
        if self.use_embree and self.embree_available:
            return self._intersect_embree(ray_origin, ray_direction, geometry, transform)
        else:
            return self._intersect_naive(ray_origin, ray_direction, geometry, transform)

    def _intersect_naive(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        geometry: 'ObjectGeometry',
        transform: np.ndarray
    ) -> Optional['IntersectionResult']:
        """
        Naive ray-triangle intersection (Möller-Trumbore algorithm).
        """
        # Transform ray to object space
        inv_transform = np.linalg.inv(transform)
        local_origin = (inv_transform @ np.append(ray_origin, 1))[:3]
        local_dir = (inv_transform[:3, :3] @ ray_direction)
        local_dir /= np.linalg.norm(local_dir)

        nearest_t = float('inf')
        nearest_face = -1
        nearest_uv = None

        # Test all triangles
        vertices = geometry.vertices
        faces = geometry.faces

        for face_idx, face in enumerate(faces):
            v0, v1, v2 = vertices[face]

            # Möller-Trumbore
            edge1 = v1 - v0
            edge2 = v2 - v0
            pvec = np.cross(local_dir, edge2)
            det = np.dot(edge1, pvec)

            if abs(det) < 1e-8:
                continue

            inv_det = 1.0 / det
            tvec = local_origin - v0
            u = np.dot(tvec, pvec) * inv_det

            if u < 0 or u > 1:
                continue

            qvec = np.cross(tvec, edge1)
            v = np.dot(local_dir, qvec) * inv_det

            if v < 0 or u + v > 1:
                continue

            t = np.dot(edge2, qvec) * inv_det

            if t > 0 and t < nearest_t:
                nearest_t = t
                nearest_face = face_idx
                nearest_uv = (u, v)

        if nearest_face < 0:
            return None

        # Compute hit point and normal
        hit_local = local_origin + local_dir * nearest_t
        hit_world = (transform @ np.append(hit_local, 1))[:3]

        # Face normal
        face = faces[nearest_face]
        v0, v1, v2 = vertices[face]
        normal_local = np.cross(v1 - v0, v2 - v0)
        normal_local /= np.linalg.norm(normal_local)
        normal_world = transform[:3, :3] @ normal_local
        normal_world /= np.linalg.norm(normal_world)

        return IntersectionResult(
            distance=nearest_t,
            position=hit_world,
            normal=normal_world,
            face_index=nearest_face,
            barycentric=nearest_uv
        )


@dataclass
class IntersectionResult:
    """Result of ray-geometry intersection."""
    distance: float
    position: np.ndarray
    normal: np.ndarray
    face_index: int
    barycentric: tuple  # (u, v) coordinates on triangle
```

---

## 8. Rendering Interface

### 8.1 What the Sensor "Sees"

```python
# src/eosim/render/sensor_view.py

class SensorView:
    """
    Represents what the sensor sees - the interface between
    scene management and rendering.
    """

    def __init__(
        self,
        registry: ObjectRegistry,
        thermal_state: 'ThermalState',
        property_query: PropertyQueryInterface
    ):
        self.registry = registry
        self.thermal = thermal_state
        self.query = property_query

    def render_radiance(
        self,
        sensor_geometry: 'SensorGeometry',
        wavelength_um: float,
        atmosphere: 'AtmosphereModel'
    ) -> np.ndarray:
        """
        Render radiance image as seen by sensor.

        For each pixel:
        1. Cast ray into scene
        2. Find surface intersection
        3. Get surface properties (T, ε, ρ)
        4. Compute surface leaving radiance
        5. Apply atmospheric effects
        6. Return at-sensor radiance
        """
        H, W = sensor_geometry.resolution
        radiance = np.zeros((H, W), dtype=np.float32)

        # Get visible objects
        visible_objects = VisibilityResolver(
            self.registry, self.registry._spatial_index
        ).get_visible_objects(sensor_geometry)

        # For each pixel
        for i in range(H):
            for j in range(W):
                # Get ray for this pixel
                ray_origin = sensor_geometry.position
                ray_dir = sensor_geometry.get_ray_direction(i, j)

                # Query scene
                hit = self.query.query_ray(
                    ray_origin, ray_dir, wavelength_um
                )

                if hit is None:
                    # Sky/background
                    L_surface = self._get_sky_radiance(ray_dir, wavelength_um)
                    radiance[i, j] = L_surface
                    continue

                # Surface leaving radiance
                props = hit.properties
                L_emitted = props.emissivity * planck_radiance(wavelength_um, props.temperature_K)

                # Reflected radiance (simplified - just downwelling sky)
                L_sky_down = atmosphere.get_downwelling(wavelength_um, props.position)
                L_reflected = props.reflectivity * L_sky_down

                L_surface = L_emitted + L_reflected

                # Atmospheric propagation
                path_length = hit.hit_distance
                atm_result = atmosphere.compute_path(
                    wavelength_um,
                    sensor_geometry.position,
                    props.position
                )

                L_sensor = L_surface * atm_result.transmission + atm_result.path_radiance

                radiance[i, j] = L_sensor

        return radiance

    def get_diagnostic_images(
        self,
        sensor_geometry: 'SensorGeometry',
        wavelength_um: float
    ) -> Dict[str, np.ndarray]:
        """
        Generate diagnostic images for visualization.
        """
        maps = PropertyMapGenerator(
            self.registry, self.thermal
        ).generate(sensor_geometry, wavelength_um)

        return {
            'object_id': maps.object_id,
            'temperature': maps.temperature,
            'emissivity': maps.emissivity,
            'distance': maps.distance,
            'normal_x': maps.normal[:, :, 0],
            'normal_y': maps.normal[:, :, 1],
            'normal_z': maps.normal[:, :, 2],
        }
```

### 8.2 Complete Rendering Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SENSOR VIEW RENDERING FLOW                               │
└─────────────────────────────────────────────────────────────────────────────┘

    SENSOR GEOMETRY                  SCENE QUERY                 OUTPUT
    ───────────────                  ───────────                 ──────

    ┌─────────────┐
    │ Pixel (i,j) │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  Get Ray    │────▶│ Frustum     │────▶│ Visible     │
    │  Direction  │     │ Culling     │     │ Objects     │
    └─────────────┘     └─────────────┘     └──────┬──────┘
                                                    │
                                                    ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │ Spatial     │◀────│  BVH        │◀────│ Ray-Box    │
    │ Index       │     │  Traversal  │     │ Test       │
    └─────────────┘     └─────────────┘     └──────┬──────┘
                                                    │
                                                    ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │ Ray-Mesh    │────▶│   Find      │────▶│ Get Face   │
    │ Intersect   │     │ Nearest Hit │     │ Index      │
    └─────────────┘     └─────────────┘     └──────┬──────┘
                                                    │
                                                    ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │ Material    │◀────│ Zone        │◀────│ Face →     │
    │ Properties  │     │ Lookup      │     │ Zone Map   │
    └──────┬──────┘     └─────────────┘     └─────────────┘
           │
           ▼
    ┌─────────────┐     ┌─────────────┐
    │ Thermal     │────▶│ Surface     │
    │ State       │     │ Temperature │
    └─────────────┘     └──────┬──────┘
                               │
                               ▼
                        ┌─────────────┐
                        │ L_surface = │
                        │ ε×B(T) +    │
                        │ ρ×L_refl    │
                        └──────┬──────┘
                               │
                               ▼
                        ┌─────────────┐
                        │ Atmosphere  │
                        │ τ, L_path   │
                        └──────┬──────┘
                               │
                               ▼
                        ┌─────────────┐
                        │ L_sensor =  │
                        │ L_surf×τ +  │
                        │ L_path      │
                        └──────┬──────┘
                               │
                               ▼
                        ┌─────────────┐
                        │ Radiance    │
                        │ Image[i,j]  │
                        └─────────────┘
```

---

## 9. Summary

### Data Flow

| Stage | Data Structure | Purpose |
|-------|---------------|---------|
| **Definition** | YAML files | Human-readable object specs |
| **Loading** | ObjectLoader | Parse, validate, build internal |
| **Registry** | ObjectRegistry | Central object database |
| **Spatial** | BVH/Octree | Fast spatial queries |
| **Instance** | ObjectInstance | Placed object with state |
| **Query** | PropertyQueryInterface | Get properties at any point |
| **Visibility** | ViewFrustum + BVH | What's in sensor view |
| **Intersection** | RayIntersector | Ray-object hit testing |
| **Rendering** | SensorView | Radiance computation |

### Key Interfaces

1. **Object Definition** → YAML with geometry, materials, thermal, articulation
2. **Registry Query** → Get objects by ID, category, or region
3. **Property Query** → Get T, ε, ρ at any world point
4. **Visibility Query** → What objects are in sensor frustum
5. **Ray Query** → What surface does ray hit, with full properties

### Implementation Priority

1. **Object loader + registry** - Core infrastructure
2. **Basic spatial index** - Required for any non-trivial scene
3. **Property query system** - Required for rendering
4. **Ray intersection** - Required for ray tracing
5. **Embree integration** - Performance optimization

---

## 10. References

1. **Pharr et al.** (2016) - "Physically Based Rendering" - BVH, ray intersection
2. **Intel Embree** - High-performance ray tracing kernels
3. **glTF 2.0 Specification** - Geometry format
4. **DIRSIG** - Scene database architecture
