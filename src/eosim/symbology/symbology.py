"""
HUD/display symbology overlay for electro-optical sensor imagery.

Provides MIL-STD-style heads-up display overlays including:
- Targeting reticles (crosshair, pipper, CCIP, CCRP)
- Track gates and correlation windows
- Compass rose and heading tape
- Pitch ladder
- Status displays (range, bearing, altitude)
- Threat classification markers
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List

import numpy as np
from numpy.typing import NDArray


class ReticleType(Enum):
    """Targeting reticle styles."""

    CROSSHAIR = "crosshair"
    PIPPER = "pipper"
    CCIP = "ccip"  # Continuously Computed Impact Point
    CCRP = "ccrp"  # Continuously Computed Release Point
    DIAMOND = "diamond"
    CIRCLE = "circle"


class ThreatLevel(Enum):
    """Threat classification levels."""

    UNKNOWN = "unknown"
    FRIENDLY = "friendly"
    NEUTRAL = "neutral"
    HOSTILE = "hostile"
    SUSPECT = "suspect"


@dataclass
class SymbologyConfig:
    """Configuration for symbology overlay rendering.

    Attributes:
        color: Drawing color as (R, G, B) tuple (0-255)
        line_width: Line width in pixels
        font_scale: Font size scale factor
        opacity: Overlay opacity (0-1)
        show_reticle: Show targeting reticle
        show_compass: Show compass rose / heading tape
        show_pitch_ladder: Show pitch ladder
        show_status: Show status text panel
        show_track_gates: Show track gates on tracked targets
    """

    color: Tuple[int, int, int] = (0, 255, 0)  # Green
    line_width: int = 1
    font_scale: float = 0.5
    opacity: float = 1.0
    show_reticle: bool = True
    show_compass: bool = True
    show_pitch_ladder: bool = True
    show_status: bool = True
    show_track_gates: bool = True


@dataclass
class TrackInfo:
    """Information about a tracked target for display.

    Attributes:
        x: Track gate center x (pixels)
        y: Track gate center y (pixels)
        width: Track gate width (pixels)
        height: Track gate height (pixels)
        track_id: Track identifier
        threat: Threat classification
        range_m: Range to target (meters)
        velocity_mps: Target velocity (m/s)
        bearing_deg: Bearing to target (degrees)
    """

    x: float = 0.0
    y: float = 0.0
    width: float = 32.0
    height: float = 32.0
    track_id: int = 0
    threat: ThreatLevel = ThreatLevel.UNKNOWN
    range_m: float = 0.0
    velocity_mps: float = 0.0
    bearing_deg: float = 0.0


@dataclass
class PlatformState:
    """Platform state for HUD display.

    Attributes:
        heading_deg: Platform heading (0-360)
        pitch_deg: Platform pitch angle
        roll_deg: Platform roll angle
        altitude_m: Altitude (meters)
        airspeed_mps: Airspeed (m/s)
        ground_speed_mps: Ground speed (m/s)
        latitude_deg: Latitude (degrees)
        longitude_deg: Longitude (degrees)
    """

    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    altitude_m: float = 0.0
    airspeed_mps: float = 0.0
    ground_speed_mps: float = 0.0
    latitude_deg: float = 0.0
    longitude_deg: float = 0.0


@dataclass
class SensorStatus:
    """Sensor operating status for display.

    Attributes:
        mode: Sensor operating mode string
        fov_deg: Current field of view (degrees)
        range_m: Selected/measured range (meters)
        polarity: Current display polarity
        agc_mode: Current AGC mode string
        laser_armed: Whether laser is armed
        laser_firing: Whether laser is firing
        recording: Whether recording is active
    """

    mode: str = "WHOT"
    fov_deg: float = 3.0
    range_m: float = 0.0
    polarity: str = "WHOT"
    agc_mode: str = "AUTO"
    laser_armed: bool = False
    laser_firing: bool = False
    recording: bool = False


class SymbologyRenderer:
    """Renders HUD symbology overlays onto sensor imagery.

    Draws symbology elements directly onto image arrays using
    NumPy operations (no OpenCV dependency).

    Example:
        >>> from eosim.symbology import SymbologyRenderer, SymbologyConfig
        >>> import numpy as np
        >>> image = np.zeros((480, 640), dtype=np.uint8)
        >>> renderer = SymbologyRenderer()
        >>> overlay = renderer.render(image, heading_deg=270, pitch_deg=5)
    """

    def __init__(
        self,
        config: Optional[SymbologyConfig] = None,
    ) -> None:
        """Initialize symbology renderer.

        Args:
            config: Symbology display configuration
        """
        self.config = config or SymbologyConfig()

    def render(
        self,
        image: NDArray,
        platform: Optional[PlatformState] = None,
        sensor: Optional[SensorStatus] = None,
        tracks: Optional[List[TrackInfo]] = None,
        reticle_type: ReticleType = ReticleType.CROSSHAIR,
    ) -> NDArray:
        """Render symbology overlay onto image.

        Args:
            image: Input image (2D grayscale or 3D RGB)
            platform: Platform state data
            sensor: Sensor status data
            tracks: List of tracked targets
            reticle_type: Type of targeting reticle

        Returns:
            Image with symbology overlay (3-channel uint8)
        """
        # Convert to RGB if grayscale
        if image.ndim == 2:
            rgb = np.stack([image, image, image], axis=-1).astype(np.uint8)
        else:
            rgb = image.copy().astype(np.uint8)

        h, w = rgb.shape[:2]

        if platform is None:
            platform = PlatformState()
        if sensor is None:
            sensor = SensorStatus()
        if tracks is None:
            tracks = []

        # Draw each symbology element
        if self.config.show_reticle:
            self._draw_reticle(rgb, reticle_type)

        if self.config.show_compass:
            self._draw_compass(rgb, platform.heading_deg)

        if self.config.show_pitch_ladder:
            self._draw_pitch_ladder(rgb, platform.pitch_deg, platform.roll_deg)

        if self.config.show_status:
            self._draw_status(rgb, platform, sensor)

        if self.config.show_track_gates:
            for track in tracks:
                self._draw_track_gate(rgb, track)

        return rgb

    def _draw_reticle(
        self,
        image: NDArray,
        reticle_type: ReticleType,
    ) -> None:
        """Draw targeting reticle at image center.

        Args:
            image: RGB image to draw on
            reticle_type: Style of reticle
        """
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        color = self.config.color
        lw = self.config.line_width

        if reticle_type == ReticleType.CROSSHAIR:
            # Standard crosshair with gap
            gap = 15
            length = 40
            _draw_line(image, cx - gap - length, cy, cx - gap, cy, color, lw)
            _draw_line(image, cx + gap, cy, cx + gap + length, cy, color, lw)
            _draw_line(image, cx, cy - gap - length, cx, cy - gap, color, lw)
            _draw_line(image, cx, cy + gap, cx, cy + gap + length, color, lw)

        elif reticle_type == ReticleType.PIPPER:
            # Pipper: circle with dot
            _draw_circle(image, cx, cy, 20, color, lw)
            _draw_circle(image, cx, cy, 2, color, -1)  # filled dot

        elif reticle_type == ReticleType.CCIP:
            # CCIP: crosshair + range-dependent pipper below
            gap = 10
            length = 30
            _draw_line(image, cx - gap - length, cy, cx - gap, cy, color, lw)
            _draw_line(image, cx + gap, cy, cx + gap + length, cy, color, lw)
            _draw_line(image, cx, cy - gap - length, cx, cy - gap, color, lw)
            # Pipper below center (simulates ballistic drop)
            pipper_y = cy + 60
            _draw_circle(image, cx, pipper_y, 8, color, lw)
            _draw_line(image, cx, cy + gap, cx, pipper_y - 10, color, lw)

        elif reticle_type == ReticleType.CCRP:
            # CCRP: azimuth steering line + release cue
            _draw_line(image, cx, 0, cx, h - 1, color, lw)
            # Steering circle
            _draw_circle(image, cx, cy + 80, 15, color, lw)

        elif reticle_type == ReticleType.DIAMOND:
            # Diamond reticle
            size = 20
            _draw_line(image, cx, cy - size, cx + size, cy, color, lw)
            _draw_line(image, cx + size, cy, cx, cy + size, color, lw)
            _draw_line(image, cx, cy + size, cx - size, cy, color, lw)
            _draw_line(image, cx - size, cy, cx, cy - size, color, lw)

        elif reticle_type == ReticleType.CIRCLE:
            # Simple circle reticle
            _draw_circle(image, cx, cy, 25, color, lw)
            # Tick marks at cardinal points
            tick = 8
            _draw_line(image, cx, cy - 25 - tick, cx, cy - 25, color, lw)
            _draw_line(image, cx, cy + 25, cx, cy + 25 + tick, color, lw)
            _draw_line(image, cx - 25 - tick, cy, cx - 25, cy, color, lw)
            _draw_line(image, cx + 25, cy, cx + 25 + tick, cy, color, lw)

    def _draw_compass(
        self,
        image: NDArray,
        heading_deg: float,
    ) -> None:
        """Draw compass rose / heading tape at top of image.

        Args:
            image: RGB image to draw on
            heading_deg: Current heading (0-360)
        """
        h, w = image.shape[:2]
        color = self.config.color
        lw = self.config.line_width
        tape_y = 25
        tape_width = min(w - 40, 300)
        tape_x0 = (w - tape_width) // 2
        tape_x1 = tape_x0 + tape_width

        # Draw tape outline
        _draw_line(image, tape_x0, tape_y, tape_x1, tape_y, color, lw)

        # Draw heading ticks
        deg_per_pixel = 60.0 / tape_width  # 60 degrees visible
        center_heading = heading_deg

        for bearing in range(0, 360, 5):
            delta = ((bearing - center_heading + 180) % 360) - 180
            pixel_offset = delta / deg_per_pixel
            x = w // 2 + int(pixel_offset)

            if tape_x0 <= x <= tape_x1:
                if bearing % 30 == 0:
                    tick_len = 12
                    _draw_line(image, x, tape_y, x, tape_y + tick_len, color, lw)
                    # Draw cardinal labels
                    label = _heading_label(bearing)
                    if label:
                        _draw_text(
                            image, label, x - 4, tape_y + tick_len + 12, color
                        )
                elif bearing % 10 == 0:
                    tick_len = 8
                    _draw_line(image, x, tape_y, x, tape_y + tick_len, color, lw)
                else:
                    tick_len = 4
                    _draw_line(image, x, tape_y, x, tape_y + tick_len, color, lw)

        # Center marker (triangle)
        cx = w // 2
        _draw_line(image, cx - 5, tape_y - 5, cx, tape_y, color, lw)
        _draw_line(image, cx + 5, tape_y - 5, cx, tape_y, color, lw)

        # Display heading value
        heading_str = f"{int(heading_deg) % 360:03d}"
        _draw_text(image, heading_str, cx - 10, tape_y - 8, color)

    def _draw_pitch_ladder(
        self,
        image: NDArray,
        pitch_deg: float,
        roll_deg: float,
    ) -> None:
        """Draw pitch ladder.

        Args:
            image: RGB image to draw on
            pitch_deg: Platform pitch (degrees)
            roll_deg: Platform roll (degrees)
        """
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        color = self.config.color
        lw = self.config.line_width

        # Pixels per degree (approximate)
        ppd = h / 30.0

        # Draw pitch lines at 5-degree increments
        for pitch in range(-30, 35, 5):
            if pitch == 0:
                continue  # Horizon drawn separately

            delta = pitch - pitch_deg
            y = cy - int(delta * ppd)

            if 50 < y < h - 50:
                half_width = 40 if abs(pitch) % 10 == 0 else 20

                if pitch > 0:
                    # Above horizon: solid lines
                    _draw_line(
                        image, cx - half_width, y, cx + half_width, y, color, lw
                    )
                else:
                    # Below horizon: dashed (draw segments)
                    dash_len = 8
                    for dx in range(-half_width, half_width, dash_len * 2):
                        x0 = cx + dx
                        x1 = min(cx + dx + dash_len, cx + half_width)
                        _draw_line(image, x0, y, x1, y, color, lw)

                # Pitch value labels
                if abs(pitch) % 10 == 0:
                    label = f"{abs(pitch)}"
                    _draw_text(
                        image, label, cx - half_width - 25, y + 3, color
                    )
                    _draw_text(
                        image, label, cx + half_width + 5, y + 3, color
                    )

        # Horizon line
        horizon_y = cy + int(pitch_deg * ppd)
        if 0 <= horizon_y < h:
            _draw_line(image, cx - 80, horizon_y, cx + 80, horizon_y, color, lw)

    def _draw_status(
        self,
        image: NDArray,
        platform: PlatformState,
        sensor: SensorStatus,
    ) -> None:
        """Draw status text panels.

        Args:
            image: RGB image to draw on
            platform: Platform state
            sensor: Sensor status
        """
        h, w = image.shape[:2]
        color = self.config.color
        line_h = 14

        # Left panel: sensor info
        left_x = 10
        y = h - 90

        _draw_text(image, f"MODE: {sensor.mode}", left_x, y, color)
        y += line_h
        _draw_text(image, f"FOV:  {sensor.fov_deg:.1f}°", left_x, y, color)
        y += line_h
        _draw_text(image, f"AGC:  {sensor.agc_mode}", left_x, y, color)
        y += line_h
        if sensor.range_m > 0:
            if sensor.range_m >= 1000:
                _draw_text(
                    image,
                    f"RNG:  {sensor.range_m / 1000:.1f}km",
                    left_x,
                    y,
                    color,
                )
            else:
                _draw_text(
                    image,
                    f"RNG:  {sensor.range_m:.0f}m",
                    left_x,
                    y,
                    color,
                )
        y += line_h

        # Laser indicator
        if sensor.laser_armed:
            laser_text = "LASER: FIRING" if sensor.laser_firing else "LASER: ARM"
            _draw_text(image, laser_text, left_x, y, color)

        # Right panel: platform info
        right_x = w - 120
        y = h - 90

        _draw_text(
            image,
            f"ALT: {platform.altitude_m:.0f}m",
            right_x,
            y,
            color,
        )
        y += line_h
        _draw_text(
            image,
            f"SPD: {platform.airspeed_mps:.0f}m/s",
            right_x,
            y,
            color,
        )
        y += line_h
        _draw_text(
            image,
            f"HDG: {platform.heading_deg:.0f}°",
            right_x,
            y,
            color,
        )
        y += line_h

        # Recording indicator
        if sensor.recording:
            _draw_text(image, "REC", w - 40, 15, (255, 0, 0))
            _draw_circle(image, w - 48, 11, 4, (255, 0, 0), -1)

    def _draw_track_gate(
        self,
        image: NDArray,
        track: TrackInfo,
    ) -> None:
        """Draw track gate around tracked target.

        Args:
            image: RGB image to draw on
            track: Track information
        """
        h, w = image.shape[:2]
        lw = self.config.line_width

        # Color by threat level
        color = _threat_color(track.threat)

        x0 = int(track.x - track.width / 2)
        y0 = int(track.y - track.height / 2)
        x1 = int(track.x + track.width / 2)
        y1 = int(track.y + track.height / 2)

        # Clamp to image bounds
        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))

        # Draw corner brackets (L-shaped at each corner)
        bracket = max(int(track.width * 0.3), 5)

        # Top-left
        _draw_line(image, x0, y0, x0 + bracket, y0, color, lw)
        _draw_line(image, x0, y0, x0, y0 + bracket, color, lw)
        # Top-right
        _draw_line(image, x1 - bracket, y0, x1, y0, color, lw)
        _draw_line(image, x1, y0, x1, y0 + bracket, color, lw)
        # Bottom-left
        _draw_line(image, x0, y1 - bracket, x0, y1, color, lw)
        _draw_line(image, x0, y1, x0 + bracket, y1, color, lw)
        # Bottom-right
        _draw_line(image, x1 - bracket, y1, x1, y1, color, lw)
        _draw_line(image, x1, y1 - bracket, x1, y1, color, lw)

        # Track ID label
        _draw_text(image, f"T{track.track_id}", x0, y0 - 5, color)

        # Range below gate
        if track.range_m > 0:
            if track.range_m >= 1000:
                range_str = f"{track.range_m / 1000:.1f}km"
            else:
                range_str = f"{track.range_m:.0f}m"
            _draw_text(image, range_str, x0, y1 + 12, color)

        # Threat indicator
        if track.threat != ThreatLevel.UNKNOWN:
            threat_sym = _threat_symbol(track.threat)
            _draw_text(image, threat_sym, x1 + 5, y0 + 5, color)


# --- Low-level drawing primitives (NumPy-based, no OpenCV) ---


def _draw_line(
    image: NDArray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Tuple[int, int, int],
    width: int = 1,
) -> None:
    """Draw a line on an RGB image using Bresenham's algorithm.

    Args:
        image: RGB image (H, W, 3)
        x0, y0: Start point
        x1, y1: End point
        color: RGB color tuple
        width: Line width in pixels
    """
    h, w = image.shape[:2]

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    half_w = max(width // 2, 0)

    while True:
        # Draw pixel with width
        for wy in range(-half_w, half_w + 1):
            for wx in range(-half_w, half_w + 1):
                px, py = x0 + wx, y0 + wy
                if 0 <= px < w and 0 <= py < h:
                    image[py, px] = color

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _draw_circle(
    image: NDArray,
    cx: int,
    cy: int,
    radius: int,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw a circle on an RGB image.

    Args:
        image: RGB image (H, W, 3)
        cx, cy: Center point
        radius: Circle radius
        color: RGB color tuple
        thickness: Line thickness (-1 for filled)
    """
    h, w = image.shape[:2]

    if thickness == -1:
        # Filled circle
        for y in range(max(0, cy - radius), min(h, cy + radius + 1)):
            for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                    image[y, x] = color
    else:
        # Circle outline using midpoint algorithm
        n_points = max(int(2 * np.pi * radius), 36)
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle))

            half_t = max(thickness // 2, 0)
            for dy in range(-half_t, half_t + 1):
                for dx in range(-half_t, half_t + 1):
                    px, py = x + dx, y + dy
                    if 0 <= px < w and 0 <= py < h:
                        image[py, px] = color


def _draw_text(
    image: NDArray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw text on an RGB image using a simple bitmap font.

    Uses a minimal 5x7 bitmap font for HUD-style text rendering
    without external dependencies.

    Args:
        image: RGB image (H, W, 3)
        text: String to draw
        x: Left position
        y: Top position
        color: RGB color tuple
    """
    h, w = image.shape[:2]
    char_w = 6  # 5 pixels + 1 spacing
    char_h = 8  # 7 pixels + 1 spacing

    for i, ch in enumerate(text):
        cx = x + i * char_w
        if cx + char_w < 0 or cx >= w:
            continue

        bitmap = _get_char_bitmap(ch)
        if bitmap is None:
            continue

        for row in range(7):
            for col in range(5):
                if bitmap[row] & (1 << (4 - col)):
                    px = cx + col
                    py = y + row
                    if 0 <= px < w and 0 <= py < h:
                        image[py, px] = color


def _get_char_bitmap(ch: str) -> Optional[List[int]]:
    """Get 5x7 bitmap for a character.

    Returns 7 rows of 5-bit bitmaps (MSB = leftmost pixel).

    Args:
        ch: Character to look up

    Returns:
        List of 7 integers representing rows, or None if unsupported
    """
    _FONT = {
        "0": [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        "1": [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        "2": [0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F],
        "3": [0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E],
        "4": [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        "5": [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        "6": [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        "7": [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        "8": [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        "9": [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        "A": [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        "B": [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        "C": [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        "D": [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
        "E": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        "F": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        "G": [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E],
        "H": [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        "I": [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        "J": [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        "K": [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        "L": [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        "M": [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        "N": [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        "O": [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        "P": [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        "Q": [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        "R": [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        "S": [0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E],
        "T": [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        "U": [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        "V": [0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04, 0x04],
        "W": [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
        "X": [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        "Y": [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        "Z": [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        " ": [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        ".": [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04],
        ":": [0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00],
        "-": [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        "/": [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10],
        "+": [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00],
    }

    ch_upper = ch.upper()
    return _FONT.get(ch_upper)


def _heading_label(bearing: int) -> str:
    """Get heading tape label for bearing.

    Args:
        bearing: Bearing in degrees

    Returns:
        Label string (cardinal direction or bearing/10)
    """
    labels = {0: "N", 90: "E", 180: "S", 270: "W"}
    if bearing in labels:
        return labels[bearing]
    if bearing % 30 == 0:
        return f"{bearing // 10}"
    return ""


def _threat_color(threat: ThreatLevel) -> Tuple[int, int, int]:
    """Get display color for threat level.

    Args:
        threat: Threat classification

    Returns:
        RGB color tuple
    """
    colors = {
        ThreatLevel.UNKNOWN: (255, 255, 0),    # Yellow
        ThreatLevel.FRIENDLY: (0, 255, 0),      # Green
        ThreatLevel.NEUTRAL: (0, 255, 255),     # Cyan
        ThreatLevel.HOSTILE: (255, 0, 0),        # Red
        ThreatLevel.SUSPECT: (255, 165, 0),      # Orange
    }
    return colors.get(threat, (255, 255, 255))


def _threat_symbol(threat: ThreatLevel) -> str:
    """Get display symbol for threat level.

    Args:
        threat: Threat classification

    Returns:
        Symbol string
    """
    symbols = {
        ThreatLevel.UNKNOWN: "?",
        ThreatLevel.FRIENDLY: "F",
        ThreatLevel.NEUTRAL: "N",
        ThreatLevel.HOSTILE: "H",
        ThreatLevel.SUSPECT: "S",
    }
    return symbols.get(threat, "?")


def create_symbology_renderer(
    color: Tuple[int, int, int] = (0, 255, 0),
    show_reticle: bool = True,
    show_compass: bool = True,
    show_pitch_ladder: bool = True,
    show_status: bool = True,
) -> SymbologyRenderer:
    """Factory function for symbology renderer.

    Args:
        color: Overlay color (R, G, B)
        show_reticle: Show targeting reticle
        show_compass: Show compass heading tape
        show_pitch_ladder: Show pitch ladder
        show_status: Show status text panels

    Returns:
        Configured SymbologyRenderer
    """
    config = SymbologyConfig(
        color=color,
        show_reticle=show_reticle,
        show_compass=show_compass,
        show_pitch_ladder=show_pitch_ladder,
        show_status=show_status,
    )
    return SymbologyRenderer(config)
