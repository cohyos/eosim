"""
Frame renderer for EOSIM Studio.

The Renderer generates camera frames by:
1. Getting scene state at current time
2. Loading 3D models from the library
3. Projecting objects into camera view using proper 3D transforms
4. Rendering thermal/visible imagery with thermal zones
5. Applying colormaps and overlays
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from eosim.studio.scene import Scene, Position3D, Orientation3D
from eosim.studio.camera import Camera, SpectrumMode


def _load_model_library():
    """Lazy load the model library to avoid circular imports."""
    try:
        from eosim.library.models3d import ModelLibrary
        return ModelLibrary()
    except ImportError:
        return None


def _load_embedded_model(model_id: str) -> Optional[Dict[str, Any]]:
    """Try to load an embedded 3D model."""
    try:
        from eosim.library.embedded_models import get_embedded_model, list_embedded_models
        if model_id in list_embedded_models():
            return get_embedded_model(model_id)
    except ImportError:
        pass
    return None


@dataclass
class RenderedFrame:
    """A rendered frame from the camera."""
    image: NDArray  # RGB image (H, W, 3) uint8
    temperature_map: Optional[NDArray] = None  # Raw temperature data
    time_sec: float = 0.0
    frame_number: int = 0
    camera_position: Optional[Position3D] = None
    camera_orientation: Optional[Orientation3D] = None


class Renderer:
    """Renders camera frames from scene state.

    The renderer takes the current scene state and camera configuration
    and produces visual output (like a video frame).

    Example:
        >>> renderer = Renderer(scene, camera)
        >>> frame = renderer.render_frame(5.0)  # Render at t=5 seconds
        >>> frame.image  # RGB numpy array
    """

    # Colormaps for thermal visualization
    COLORMAPS = {
        "iron": [  # Classic thermal iron colormap
            (0.0, (0, 0, 0)),
            (0.2, (128, 0, 128)),
            (0.4, (255, 0, 0)),
            (0.6, (255, 128, 0)),
            (0.8, (255, 255, 0)),
            (1.0, (255, 255, 255)),
        ],
        "rainbow": [
            (0.0, (0, 0, 128)),
            (0.25, (0, 128, 255)),
            (0.5, (0, 255, 0)),
            (0.75, (255, 255, 0)),
            (1.0, (255, 0, 0)),
        ],
        "grayscale": [
            (0.0, (0, 0, 0)),
            (1.0, (255, 255, 255)),
        ],
        "hot": [
            (0.0, (0, 0, 0)),
            (0.33, (128, 0, 0)),
            (0.66, (255, 128, 0)),
            (1.0, (255, 255, 255)),
        ],
        "green": [  # Night vision style
            (0.0, (0, 16, 0)),
            (0.5, (0, 128, 0)),
            (1.0, (128, 255, 128)),
        ],
    }

    # Visible light colors for different object types (RGB)
    VISIBLE_COLORS = {
        # Aircraft - gray/silver with darker areas
        "f16": {"body": (140, 140, 150), "cockpit": (40, 60, 80), "exhaust": (60, 60, 70), "wings": (130, 130, 140)},
        "f22": {"body": (130, 130, 140), "cockpit": (30, 50, 70), "exhaust": (50, 50, 60), "wings": (120, 120, 130)},
        "apache": {"body": (90, 100, 90), "cockpit": (50, 70, 90), "exhaust": (70, 70, 80), "rotor_hub": (60, 60, 60)},
        # Tanks - tan/olive drab
        "m1_abrams": {"body": (140, 130, 100), "turret": (130, 120, 90), "tracks": (50, 45, 40), "gun": (80, 75, 70)},
        "t90": {"body": (100, 110, 90), "turret": (90, 100, 80), "tracks": (45, 40, 35), "gun": (70, 70, 65)},
        # Vehicles
        "humvee": {"body": (140, 130, 100), "cabin": (60, 80, 100), "wheels": (40, 40, 40)},
        # People - skin tones and clothing
        "soldier_standing": {"head": (200, 160, 130), "torso": (90, 100, 80), "arms": (90, 100, 80), "legs": (90, 100, 80)},
        "civilian": {"head": (210, 170, 140), "torso": (100, 100, 150), "arms": (210, 170, 140), "legs": (50, 50, 80)},
        # Ships
        "destroyer": {"hull": (120, 120, 130), "superstructure": (140, 140, 150), "funnel": (60, 60, 70)},
    }

    # Default visible colors by zone type
    VISIBLE_ZONE_COLORS = {
        "body": (140, 140, 145),
        "fuselage": (140, 140, 150),
        "wings": (130, 130, 140),
        "tail": (135, 135, 145),
        "cockpit": (40, 60, 80),
        "exhaust": (60, 60, 70),
        "engine": (80, 80, 85),
        "turret": (130, 120, 90),
        "tracks": (50, 45, 40),
        "gun": (80, 75, 70),
        "wheels": (40, 40, 40),
        "tires": (35, 35, 35),
        "cabin": (60, 80, 100),
        "head": (200, 160, 130),
        "torso": (90, 100, 80),
        "arms": (180, 150, 120),
        "legs": (70, 80, 70),
        "hull": (120, 120, 130),
        "superstructure": (140, 140, 150),
        "funnel": (60, 60, 70),
        "rotor_hub": (60, 60, 60),
    }

    def __init__(self, scene: Scene, camera: Camera):
        self.scene = scene
        self.camera = camera

        # Rendering settings
        self.background_temp_k = 290.0  # Ambient temperature
        self.atmosphere_attenuation = 0.0001  # Per meter

        # Visible mode settings
        self.sky_color = (135, 180, 220)  # Light blue sky
        self.ground_color = (120, 130, 100)  # Greenish ground

        # Cache for 3D models
        self._model_cache: Dict[str, Any] = {}

        # Try to load model library
        self._model_library = _load_model_library()

        # Use 3D models when available
        self.use_3d_models = True

    def _is_visible_mode(self) -> bool:
        """Check if camera is in visible light mode."""
        return self.camera.spectrum == SpectrumMode.VISIBLE

    def render_frame(self, time_sec: float, frame_number: int = 0) -> RenderedFrame:
        """Render a frame at the given time.

        Args:
            time_sec: Time in seconds
            frame_number: Frame number for metadata

        Returns:
            RenderedFrame with image data
        """
        # Get camera state
        cam_pos = self.camera.get_position_at(time_sec)
        cam_ori = self.camera.get_orientation_at(time_sec, self.scene.objects)

        # Get resolution
        width, height = self.camera.resolution

        # Get object states
        object_states = self.scene.get_object_states_at(time_sec)

        # Check rendering mode
        if self._is_visible_mode():
            # Visible light rendering - direct to RGB
            image = self._render_visible_frame(
                width, height, object_states, cam_pos, cam_ori, self.camera.get_fov()
            )
            temp_map = None  # No temperature data in visible mode
        else:
            # Thermal rendering
            # Create temperature map
            temp_map = np.full((height, width), self.background_temp_k, dtype=np.float32)

            for obj_state in object_states:
                self._render_object_to_map(
                    temp_map, obj_state, cam_pos, cam_ori, self.camera.get_fov()
                )

            # Apply noise based on sensitivity
            netd = self.camera.get_netd() / 1000.0  # Convert mK to K
            noise = np.random.normal(0, netd, temp_map.shape)
            temp_map = temp_map + noise

            # Convert to RGB using colormap
            image = self._apply_colormap(temp_map, self.camera.colormap)

        return RenderedFrame(
            image=image,
            temperature_map=temp_map,
            time_sec=time_sec,
            frame_number=frame_number,
            camera_position=cam_pos,
            camera_orientation=cam_ori
        )

    def _render_object_to_map(self, temp_map: NDArray,
                              obj_state: Dict[str, Any],
                              cam_pos: Position3D,
                              cam_ori: Orientation3D,
                              fov_deg: float):
        """Render a single object onto the temperature map.

        Uses 3D models from the library when available, falls back to
        simplified rectangle rendering otherwise.
        """
        obj_pos: Position3D = obj_state["position"]
        obj_ori: Orientation3D = obj_state["orientation"]
        obj_type: str = obj_state["type"]

        # Try to use 3D model if enabled
        if self.use_3d_models:
            model = self._get_3d_model(obj_type)
            if model is not None:
                self._render_3d_model(temp_map, model, obj_state, cam_pos, cam_ori, fov_deg)
                return

        # Fallback to simple rectangle rendering
        # Calculate relative position
        dx = obj_pos.x - cam_pos.x
        dy = obj_pos.y - cam_pos.y
        dz = obj_pos.z - cam_pos.z

        # Distance
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        if distance < 1:
            return

        # Get object thermal signature
        obj_temp = self._get_object_temperature(obj_type)
        obj_size = self._get_object_size(obj_type)

        # Calculate angular size
        angular_size_rad = np.arctan2(obj_size, distance)
        angular_size_deg = np.degrees(angular_size_rad)

        # Check if in field of view
        # Calculate bearing from camera
        bearing = np.degrees(np.arctan2(dx, dy)) % 360
        elevation = np.degrees(np.arctan2(-dz, np.sqrt(dx**2 + dy**2)))

        # Relative to camera orientation
        rel_bearing = (bearing - cam_ori.heading + 180) % 360 - 180
        rel_elevation = elevation - cam_ori.pitch

        # Check if in FOV
        half_fov = fov_deg / 2
        if abs(rel_bearing) > half_fov or abs(rel_elevation) > half_fov:
            return

        # Calculate pixel position
        h, w = temp_map.shape
        px = int(w / 2 + (rel_bearing / half_fov) * (w / 2))
        py = int(h / 2 - (rel_elevation / half_fov) * (h / 2))

        # Calculate pixel size
        pixel_size = max(1, int((angular_size_deg / fov_deg) * min(w, h)))

        # Apply atmospheric attenuation
        attenuation = np.exp(-self.atmosphere_attenuation * distance)
        apparent_temp = self.background_temp_k + (obj_temp - self.background_temp_k) * attenuation

        # Render object as a filled region
        half_size = pixel_size // 2
        y_min = max(0, py - half_size)
        y_max = min(h, py + half_size + 1)
        x_min = max(0, px - half_size)
        x_max = min(w, px + half_size + 1)

        if y_max > y_min and x_max > x_min:
            # Create object shape (simplified rectangle for now)
            temp_map[y_min:y_max, x_min:x_max] = apparent_temp

            # Add hot spots for vehicles/aircraft
            if obj_type in ["f16", "f22", "su27", "apache", "mig29"]:
                # Engine exhaust (rear)
                if apparent_temp > 350:
                    exhaust_y = max(y_min, min(y_max-1, py))
                    exhaust_x = max(x_min, min(x_max-1, px + half_size - 1))
                    temp_map[exhaust_y-1:exhaust_y+2, exhaust_x-1:exhaust_x+2] = apparent_temp + 150

            elif obj_type in ["m1_abrams", "t90", "humvee", "pickup_truck"]:
                # Engine hot spot
                engine_y = max(y_min, min(y_max-1, py))
                engine_x = max(x_min, min(x_max-1, px - half_size + 1))
                hs = max(1, half_size // 3)
                temp_map[engine_y-hs:engine_y+hs+1, engine_x:engine_x+hs+1] = apparent_temp + 70

    def _get_object_temperature(self, obj_type: str) -> float:
        """Get characteristic temperature for object type."""
        # Default temperatures by category
        temps = {
            # Aircraft (hot engines)
            "f16": 380, "f22": 400, "f35": 420, "su27": 390, "mig29": 385,
            "apache": 360, "blackhawk": 350, "predator": 320,

            # Vehicles
            "m1_abrams": 340, "t90": 345, "humvee": 320, "pickup_truck": 315,
            "sedan": 310, "suv": 312,

            # Ships
            "destroyer": 310, "carrier": 315, "frigate": 308,

            # People
            "soldier_standing": 310, "soldier_prone": 305, "civilian": 308,

            # Missiles/Launchers
            "aim120": 295, "s400_launcher": 305,
        }
        return temps.get(obj_type, 300.0)

    def _get_object_size(self, obj_type: str) -> float:
        """Get characteristic size (meters) for object type."""
        sizes = {
            # Aircraft
            "f16": 15, "f22": 19, "f35": 16, "su27": 22, "mig29": 17,
            "apache": 15, "blackhawk": 18, "predator": 8,

            # Vehicles
            "m1_abrams": 10, "t90": 9, "humvee": 5, "pickup_truck": 5,
            "sedan": 4, "suv": 5,

            # Ships
            "destroyer": 150, "carrier": 300, "frigate": 130,

            # People
            "soldier_standing": 1.8, "soldier_prone": 1.8, "civilian": 1.7,

            # Missiles
            "aim120": 3.5, "s400_launcher": 12,
        }
        return sizes.get(obj_type, 5.0)

    def _get_3d_model(self, obj_type: str) -> Optional[Dict[str, Any]]:
        """Get 3D model for object type from cache or library.

        Args:
            obj_type: Object type identifier

        Returns:
            Model dict with vertices, faces, thermal_zones or None
        """
        # Check cache first
        if obj_type in self._model_cache:
            return self._model_cache[obj_type]

        model = None

        # Try embedded models first (they have best detail)
        model = _load_embedded_model(obj_type)

        # Try model library if no embedded model
        if model is None and self._model_library is not None:
            try:
                mesh = self._model_library.get_model(obj_type)
                if mesh is not None:
                    model = {
                        "vertices": mesh.vertices,
                        "faces": mesh.faces,
                        "thermal_zones": mesh.thermal_zones or {},
                        "name": mesh.name,
                    }
            except Exception:
                pass

        # Cache result (even if None)
        self._model_cache[obj_type] = model
        return model

    def _transform_vertices(self, vertices: List[Tuple[float, float, float]],
                           obj_pos: Position3D, obj_ori: Orientation3D,
                           scale: float = 1.0) -> List[Tuple[float, float, float]]:
        """Transform model vertices to world coordinates.

        Args:
            vertices: Model vertices in local coordinates
            obj_pos: Object world position
            obj_ori: Object world orientation
            scale: Scale factor

        Returns:
            Transformed vertices in world coordinates
        """
        # Convert orientation to radians
        heading_rad = np.radians(obj_ori.heading)
        pitch_rad = np.radians(obj_ori.pitch)
        roll_rad = np.radians(obj_ori.roll)

        # Rotation matrices
        cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
        cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
        cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)

        # Combined rotation (ZYX order: heading, pitch, roll)
        R = np.array([
            [cos_h * cos_p, cos_h * sin_p * sin_r - sin_h * cos_r, cos_h * sin_p * cos_r + sin_h * sin_r],
            [sin_h * cos_p, sin_h * sin_p * sin_r + cos_h * cos_r, sin_h * sin_p * cos_r - cos_h * sin_r],
            [-sin_p, cos_p * sin_r, cos_p * cos_r]
        ])

        transformed = []
        for v in vertices:
            # Scale and rotate
            local = np.array([v[0] * scale, v[1] * scale, v[2] * scale])
            rotated = R @ local

            # Translate
            world = (
                rotated[0] + obj_pos.x,
                rotated[1] + obj_pos.y,
                rotated[2] + obj_pos.z
            )
            transformed.append(world)

        return transformed

    def _project_to_camera(self, world_vertices: List[Tuple[float, float, float]],
                          cam_pos: Position3D, cam_ori: Orientation3D,
                          fov_deg: float, width: int, height: int
                          ) -> List[Optional[Tuple[int, int, float]]]:
        """Project world vertices to screen coordinates.

        Args:
            world_vertices: Vertices in world coordinates
            cam_pos: Camera position
            cam_ori: Camera orientation
            fov_deg: Camera field of view
            width: Screen width
            height: Screen height

        Returns:
            List of (px, py, depth) or None if behind camera
        """
        # Camera rotation matrix (inverse of camera orientation)
        heading_rad = np.radians(-cam_ori.heading)
        pitch_rad = np.radians(-cam_ori.pitch)
        roll_rad = np.radians(-cam_ori.roll)

        cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
        cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
        cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)

        # Combined inverse rotation
        R_inv = np.array([
            [cos_h * cos_p, sin_h * cos_p, -sin_p],
            [cos_h * sin_p * sin_r - sin_h * cos_r, sin_h * sin_p * sin_r + cos_h * cos_r, cos_p * sin_r],
            [cos_h * sin_p * cos_r + sin_h * sin_r, sin_h * sin_p * cos_r - cos_h * sin_r, cos_p * cos_r]
        ])

        half_fov = np.radians(fov_deg / 2)
        focal_length = 1.0 / np.tan(half_fov)

        projected = []
        for v in world_vertices:
            # Translate to camera space
            dx = v[0] - cam_pos.x
            dy = v[1] - cam_pos.y
            dz = v[2] - cam_pos.z

            # Rotate to camera space
            cam_space = R_inv @ np.array([dx, dy, dz])

            # In our coordinate system, Y is forward
            depth = cam_space[1]

            if depth <= 0.1:  # Behind camera
                projected.append(None)
                continue

            # Perspective projection
            x_proj = cam_space[0] / depth * focal_length
            z_proj = -cam_space[2] / depth * focal_length

            # Convert to screen coordinates
            px = int(width / 2 + x_proj * width / 2)
            py = int(height / 2 + z_proj * height / 2)

            projected.append((px, py, depth))

        return projected

    def _render_3d_model(self, temp_map: NDArray,
                        model: Dict[str, Any],
                        obj_state: Dict[str, Any],
                        cam_pos: Position3D,
                        cam_ori: Orientation3D,
                        fov_deg: float):
        """Render a 3D model onto the temperature map.

        Args:
            temp_map: Temperature map to render onto
            model: 3D model dict with vertices, faces, thermal_zones
            obj_state: Object state dict
            cam_pos: Camera position
            cam_ori: Camera orientation
            fov_deg: Field of view
        """
        obj_pos: Position3D = obj_state["position"]
        obj_ori: Orientation3D = obj_state["orientation"]
        obj_type: str = obj_state["type"]

        # Get model scale (models are normalized, scale to actual size)
        actual_size = self._get_object_size(obj_type)
        model_vertices_raw = model.get("vertices", [])
        if len(model_vertices_raw) == 0:
            return

        # Convert to numpy array for calculations, list for iteration
        verts_array = np.array(model_vertices_raw)
        model_vertices = verts_array.tolist()
        model_size = np.max(verts_array.max(axis=0) - verts_array.min(axis=0))
        scale = actual_size / max(model_size, 0.1)

        # Transform vertices to world coordinates
        world_verts = self._transform_vertices(model_vertices, obj_pos, obj_ori, scale)

        # Calculate distance for atmospheric attenuation
        dx = obj_pos.x - cam_pos.x
        dy = obj_pos.y - cam_pos.y
        dz = obj_pos.z - cam_pos.z
        distance = np.sqrt(dx**2 + dy**2 + dz**2)

        if distance < 1:
            return

        attenuation = np.exp(-self.atmosphere_attenuation * distance)

        # Project to screen
        height, width = temp_map.shape
        projected = self._project_to_camera(world_verts, cam_pos, cam_ori, fov_deg, width, height)

        # Get base temperature
        base_temp = self._get_object_temperature(obj_type)
        thermal_zones = model.get("thermal_zones", {})

        # Temperature offsets by zone name (relative to base temp)
        zone_temp_offsets = {
            # Hot zones
            "exhaust": 150,      # Jet exhaust
            "engine": 80,        # Engine compartment
            "rotor_hub": 40,     # Helicopter rotor hub
            "weapons": 20,       # Missile/weapon bays
            # Warm zones
            "cockpit": 30,       # Cockpit/cabin area
            "cabin": 25,         # Vehicle cabin
            "wheels": 40,        # Hot wheels from friction
            "tires": 40,
            # Neutral zones
            "fuselage": 0,       # Aircraft body
            "body": 0,           # Vehicle body
            "wings": -5,         # Wings (cooler due to airflow)
            "tail": 0,           # Tail section
            "turret": 10,        # Tank turret
            "tracks": 30,        # Tank tracks (friction)
            "gun": 5,            # Gun barrel
            # Human zones
            "head": 3,           # Head
            "torso": 0,          # Torso
            "arms": -2,          # Arms
            "legs": -3,          # Legs
            # Ship zones
            "hull": 0,           # Ship hull
            "superstructure": 15,  # Superstructure
            "funnel": 100,       # Exhaust funnel
        }

        # Render each face
        faces = model.get("faces", [])
        for face_idx, face in enumerate(faces):
            # Get projected vertices for this face
            face_points = []
            behind_camera = False

            for vi in face:
                if vi >= len(projected) or projected[vi] is None:
                    behind_camera = True
                    break
                face_points.append(projected[vi])

            if behind_camera or len(face_points) < 3:
                continue

            # Determine temperature for this face based on thermal zones
            face_temp = base_temp

            # Look up zone name for this face
            zone_name = thermal_zones.get(face_idx, "body")
            if isinstance(zone_name, str):
                # Apply temperature offset based on zone
                offset = zone_temp_offsets.get(zone_name.lower(), 0)
                face_temp = base_temp + offset

            # Apply atmospheric attenuation
            apparent_temp = self.background_temp_k + (face_temp - self.background_temp_k) * attenuation

            # Rasterize the face
            self._fill_polygon(temp_map, face_points, apparent_temp)

    def _fill_polygon(self, temp_map: NDArray,
                     points: List[Tuple[int, int, float]],
                     temperature: float):
        """Fill a polygon on the temperature map using scanline algorithm.

        Args:
            temp_map: Temperature map to render onto
            points: List of (px, py, depth) screen coordinates
            temperature: Temperature value to fill with
        """
        if len(points) < 3:
            return

        height, width = temp_map.shape

        # Extract 2D points
        pts = [(p[0], p[1]) for p in points]

        # Get bounding box
        min_x = max(0, min(p[0] for p in pts))
        max_x = min(width - 1, max(p[0] for p in pts))
        min_y = max(0, min(p[1] for p in pts))
        max_y = min(height - 1, max(p[1] for p in pts))

        if min_x >= max_x or min_y >= max_y:
            return

        # Simple scanline fill
        for y in range(min_y, max_y + 1):
            # Find intersections with polygon edges
            intersections = []
            n = len(pts)
            for i in range(n):
                p1 = pts[i]
                p2 = pts[(i + 1) % n]

                if (p1[1] <= y < p2[1]) or (p2[1] <= y < p1[1]):
                    if p2[1] != p1[1]:
                        x = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                        intersections.append(x)

            if len(intersections) < 2:
                continue

            intersections.sort()

            # Fill between pairs of intersections
            for i in range(0, len(intersections) - 1, 2):
                x1 = max(0, int(intersections[i]))
                x2 = min(width - 1, int(intersections[i + 1]))
                if x1 <= x2:
                    temp_map[y, x1:x2 + 1] = temperature

    def _render_visible_frame(self, width: int, height: int,
                              object_states: List[Dict[str, Any]],
                              cam_pos: Position3D, cam_ori: Orientation3D,
                              fov_deg: float) -> NDArray:
        """Render a frame in visible light mode with realistic colors.

        Args:
            width: Image width
            height: Image height
            object_states: List of object state dicts
            cam_pos: Camera position
            cam_ori: Camera orientation
            fov_deg: Field of view

        Returns:
            RGB image as (H, W, 3) uint8 array
        """
        # Create image with sky background
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Sky gradient (lighter at horizon)
        for y in range(height):
            # Blend from deep blue (top) to lighter blue (bottom)
            blend = y / height
            sky_r = int(self.sky_color[0] * (0.6 + 0.4 * blend))
            sky_g = int(self.sky_color[1] * (0.7 + 0.3 * blend))
            sky_b = int(self.sky_color[2] * (0.8 + 0.2 * blend))
            image[y, :] = (sky_r, sky_g, sky_b)

        # Render each object
        for obj_state in object_states:
            self._render_visible_object(image, obj_state, cam_pos, cam_ori, fov_deg)

        return image

    def _render_visible_object(self, image: NDArray,
                               obj_state: Dict[str, Any],
                               cam_pos: Position3D,
                               cam_ori: Orientation3D,
                               fov_deg: float):
        """Render a single object in visible light mode.

        Args:
            image: RGB image to render onto
            obj_state: Object state dict
            cam_pos: Camera position
            cam_ori: Camera orientation
            fov_deg: Field of view
        """
        obj_type: str = obj_state["type"]

        # Try to use 3D model if available
        if self.use_3d_models:
            model = self._get_3d_model(obj_type)
            if model is not None:
                self._render_visible_3d_model(image, model, obj_state, cam_pos, cam_ori, fov_deg)
                return

        # Fallback: simple colored rectangle
        obj_pos: Position3D = obj_state["position"]
        obj_ori: Orientation3D = obj_state["orientation"]

        dx = obj_pos.x - cam_pos.x
        dy = obj_pos.y - cam_pos.y
        dz = obj_pos.z - cam_pos.z
        distance = np.sqrt(dx**2 + dy**2 + dz**2)

        if distance < 1:
            return

        # Get default color for this object type
        color = self.VISIBLE_COLORS.get(obj_type, {}).get("body", (140, 140, 140))

        # Calculate screen position (simplified)
        bearing = np.degrees(np.arctan2(dx, dy)) % 360
        elevation = np.degrees(np.arctan2(-dz, np.sqrt(dx**2 + dy**2)))
        rel_bearing = (bearing - cam_ori.heading + 180) % 360 - 180
        rel_elevation = elevation - cam_ori.pitch

        half_fov = fov_deg / 2
        if abs(rel_bearing) > half_fov or abs(rel_elevation) > half_fov:
            return

        h, w = image.shape[:2]
        px = int(w / 2 + (rel_bearing / half_fov) * (w / 2))
        py = int(h / 2 - (rel_elevation / half_fov) * (h / 2))

        obj_size = self._get_object_size(obj_type)
        angular_size = np.degrees(np.arctan2(obj_size, distance))
        pixel_size = max(2, int((angular_size / fov_deg) * min(w, h)))

        half = pixel_size // 2
        y1, y2 = max(0, py - half), min(h, py + half)
        x1, x2 = max(0, px - half), min(w, px + half)

        if y2 > y1 and x2 > x1:
            image[y1:y2, x1:x2] = color

    def _render_visible_3d_model(self, image: NDArray,
                                  model: Dict[str, Any],
                                  obj_state: Dict[str, Any],
                                  cam_pos: Position3D,
                                  cam_ori: Orientation3D,
                                  fov_deg: float):
        """Render a 3D model in visible light mode with realistic colors.

        Args:
            image: RGB image to render onto
            model: 3D model dict
            obj_state: Object state dict
            cam_pos: Camera position
            cam_ori: Camera orientation
            fov_deg: Field of view
        """
        obj_pos: Position3D = obj_state["position"]
        obj_ori: Orientation3D = obj_state["orientation"]
        obj_type: str = obj_state["type"]

        actual_size = self._get_object_size(obj_type)
        model_vertices_raw = model.get("vertices", [])
        if len(model_vertices_raw) == 0:
            return

        verts_array = np.array(model_vertices_raw)
        model_vertices = verts_array.tolist()
        model_size = np.max(verts_array.max(axis=0) - verts_array.min(axis=0))
        scale = actual_size / max(model_size, 0.1)

        # Transform and project vertices
        world_verts = self._transform_vertices(model_vertices, obj_pos, obj_ori, scale)
        height, width = image.shape[:2]
        projected = self._project_to_camera(world_verts, cam_pos, cam_ori, fov_deg, width, height)

        # Get distance for atmospheric effects
        dx = obj_pos.x - cam_pos.x
        dy = obj_pos.y - cam_pos.y
        dz = obj_pos.z - cam_pos.z
        distance = np.sqrt(dx**2 + dy**2 + dz**2)

        if distance < 1:
            return

        # Atmospheric haze factor (objects fade to sky color with distance)
        haze_factor = 1.0 - np.exp(-distance * 0.0003)

        # Get object-specific colors or use defaults
        obj_colors = self.VISIBLE_COLORS.get(obj_type, {})
        thermal_zones = model.get("thermal_zones", {})

        # Render each face
        faces = model.get("faces", [])
        for face_idx, face in enumerate(faces):
            face_points = []
            behind_camera = False

            for vi in face:
                if vi >= len(projected) or projected[vi] is None:
                    behind_camera = True
                    break
                face_points.append(projected[vi])

            if behind_camera or len(face_points) < 3:
                continue

            # Get zone name and color
            zone_name = thermal_zones.get(face_idx, "body")
            if isinstance(zone_name, str):
                zone_key = zone_name.lower()
                # Try object-specific color, then default zone color
                if zone_key in obj_colors:
                    base_color = obj_colors[zone_key]
                elif zone_key in self.VISIBLE_ZONE_COLORS:
                    base_color = self.VISIBLE_ZONE_COLORS[zone_key]
                else:
                    base_color = obj_colors.get("body", (140, 140, 140))
            else:
                base_color = obj_colors.get("body", (140, 140, 140))

            # Apply atmospheric haze (blend toward sky color)
            final_color = tuple(
                int(base_color[i] * (1 - haze_factor) + self.sky_color[i] * haze_factor)
                for i in range(3)
            )

            # Fill the polygon with this color
            self._fill_visible_polygon(image, face_points, final_color)

    def _fill_visible_polygon(self, image: NDArray,
                              points: List[Tuple[int, int, float]],
                              color: Tuple[int, int, int]):
        """Fill a polygon on the RGB image using scanline algorithm.

        Args:
            image: RGB image to render onto (H, W, 3)
            points: List of (px, py, depth) screen coordinates
            color: RGB color tuple
        """
        if len(points) < 3:
            return

        height, width = image.shape[:2]
        pts = [(p[0], p[1]) for p in points]

        min_x = max(0, min(p[0] for p in pts))
        max_x = min(width - 1, max(p[0] for p in pts))
        min_y = max(0, min(p[1] for p in pts))
        max_y = min(height - 1, max(p[1] for p in pts))

        if min_x >= max_x or min_y >= max_y:
            return

        for y in range(min_y, max_y + 1):
            intersections = []
            n = len(pts)
            for i in range(n):
                p1 = pts[i]
                p2 = pts[(i + 1) % n]

                if (p1[1] <= y < p2[1]) or (p2[1] <= y < p1[1]):
                    if p2[1] != p1[1]:
                        x = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                        intersections.append(x)

            if len(intersections) < 2:
                continue

            intersections.sort()

            for i in range(0, len(intersections) - 1, 2):
                x1 = max(0, int(intersections[i]))
                x2 = min(width - 1, int(intersections[i + 1]))
                if x1 <= x2:
                    image[y, x1:x2 + 1] = color

    def _apply_colormap(self, temp_map: NDArray, colormap_name: str) -> NDArray:
        """Apply colormap to temperature map.

        Args:
            temp_map: 2D array of temperatures
            colormap_name: Name of colormap to use

        Returns:
            RGB image as (H, W, 3) uint8 array
        """
        colormap = self.COLORMAPS.get(colormap_name, self.COLORMAPS["iron"])

        # Normalize to 0-1
        t_min, t_max = temp_map.min(), temp_map.max()
        if t_max > t_min:
            normalized = (temp_map - t_min) / (t_max - t_min)
        else:
            normalized = np.zeros_like(temp_map)

        # Apply colormap
        h, w = temp_map.shape
        image = np.zeros((h, w, 3), dtype=np.uint8)

        for i in range(len(colormap) - 1):
            t0, c0 = colormap[i]
            t1, c1 = colormap[i + 1]

            mask = (normalized >= t0) & (normalized < t1)
            if not np.any(mask):
                continue

            # Interpolate within this segment
            alpha = (normalized[mask] - t0) / (t1 - t0)
            alpha = alpha[:, np.newaxis]

            c0 = np.array(c0)
            c1 = np.array(c1)
            colors = c0 + alpha * (c1 - c0)

            image[mask] = colors.astype(np.uint8)

        # Handle values at t=1.0
        mask = normalized >= colormap[-1][0]
        if np.any(mask):
            image[mask] = colormap[-1][1]

        return image

    def render_with_overlay(self, frame: RenderedFrame,
                           show_info: bool = True,
                           show_crosshair: bool = False) -> NDArray:
        """Add overlay information to a rendered frame.

        Args:
            frame: The rendered frame
            show_info: Show timestamp and camera info
            show_crosshair: Show center crosshair

        Returns:
            Image with overlay as (H, W, 3) uint8 array
        """
        image = frame.image.copy()
        h, w = image.shape[:2]

        if show_crosshair:
            # Draw crosshair at center
            cx, cy = w // 2, h // 2
            color = (0, 255, 0)  # Green
            # Horizontal line
            image[cy, cx-20:cx+21] = color
            # Vertical line
            image[cy-20:cy+21, cx] = color

        # Note: Text overlay would require PIL or similar
        # For now, we'll add a simple indicator in corner

        if show_info:
            # Add colored box in corner as indicator
            # Time indicator (brightness corresponds to time)
            indicator_size = 10
            brightness = int(255 * (frame.time_sec % 1.0))
            image[5:5+indicator_size, 5:5+indicator_size] = (brightness, brightness, 255)

        return image


class RenderQueue:
    """Queue for rendering multiple frames (for video export)."""

    def __init__(self, renderer: Renderer, timeline_duration: float, fps: float):
        self.renderer = renderer
        self.duration = timeline_duration
        self.fps = fps
        self.total_frames = int(timeline_duration * fps)

        self.current_frame = 0
        self.frames: List[RenderedFrame] = []
        self.is_rendering = False
        self.progress = 0.0

    def render_all(self, callback=None) -> List[RenderedFrame]:
        """Render all frames.

        Args:
            callback: Optional callback(progress, frame_num, total_frames)

        Returns:
            List of all rendered frames
        """
        self.is_rendering = True
        self.frames = []

        for i in range(self.total_frames):
            time_sec = i / self.fps
            frame = self.renderer.render_frame(time_sec, i)
            self.frames.append(frame)

            self.current_frame = i
            self.progress = (i + 1) / self.total_frames

            if callback:
                callback(self.progress, i, self.total_frames)

        self.is_rendering = False
        return self.frames

    def render_frame(self, frame_num: int) -> RenderedFrame:
        """Render a single frame."""
        time_sec = frame_num / self.fps
        return self.renderer.render_frame(time_sec, frame_num)
