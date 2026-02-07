"""
Frame renderer for EOSIM Studio.

The Renderer generates camera frames by:
1. Getting scene state at current time
2. Projecting objects into camera view
3. Rendering thermal/visible imagery
4. Applying colormaps and overlays
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from eosim.studio.scene import Scene, Position3D, Orientation3D
from eosim.studio.camera import Camera, SpectrumMode


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

    def __init__(self, scene: Scene, camera: Camera):
        self.scene = scene
        self.camera = camera

        # Rendering settings
        self.background_temp_k = 290.0  # Ambient temperature
        self.atmosphere_attenuation = 0.0001  # Per meter

        # Cache for object renderers
        self._object_cache: Dict[str, Any] = {}

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

        # Create temperature map
        temp_map = np.full((height, width), self.background_temp_k, dtype=np.float32)

        # Get object states and render each
        object_states = self.scene.get_object_states_at(time_sec)

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
        """Render a single object onto the temperature map."""
        obj_pos: Position3D = obj_state["position"]
        obj_ori: Orientation3D = obj_state["orientation"]
        obj_type: str = obj_state["type"]

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
