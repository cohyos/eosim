"""
Tactical symbology and HUD overlays for electro-optical displays.

This module provides MIL-STD compatible symbology including:
- Targeting reticles and crosshairs
- Track gates and designation boxes
- Range/bearing displays
- Compass and heading indicators
- Attitude indicators
- Status readouts
- MIL-STD-1787/2525 compliant symbols
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class SymbolColor(Enum):
    """Standard symbology colors."""
    GREEN = (0, 255, 0)        # Primary
    AMBER = (255, 191, 0)      # Caution/secondary
    RED = (255, 0, 0)          # Warning/hostile
    CYAN = (0, 255, 255)       # Friendly
    WHITE = (255, 255, 255)    # General
    YELLOW = (255, 255, 0)     # Highlight
    MAGENTA = (255, 0, 255)    # Unknown


class ReticleType(Enum):
    """Targeting reticle types."""
    CROSSHAIR = "crosshair"
    CIRCLE = "circle"
    CROSS_CIRCLE = "cross_circle"
    DIAMOND = "diamond"
    PLUS = "plus"
    AIMPOINT = "aimpoint"
    PIPPER = "pipper"
    CCIP = "ccip"  # Continuously Computed Impact Point
    CCRP = "ccrp"  # Continuously Computed Release Point


class TrackGateType(Enum):
    """Track gate types."""
    SQUARE = "square"
    CORNER = "corner"
    DIAMOND = "diamond"
    CIRCLE = "circle"
    ACQUISITION = "acquisition"
    TRACK = "track"


class ThreatLevel(Enum):
    """Threat classification levels."""
    UNKNOWN = "unknown"
    FRIENDLY = "friendly"
    NEUTRAL = "neutral"
    HOSTILE = "hostile"
    SUSPECT = "suspect"


@dataclass
class ScreenPosition:
    """Position on screen in normalized coordinates (0-1)."""
    x: float = 0.5
    y: float = 0.5

    def to_pixels(self, width: int, height: int) -> Tuple[int, int]:
        return (int(self.x * width), int(self.y * height))


@dataclass
class TargetDesignation:
    """Target designation information."""
    position: ScreenPosition
    range_m: float = 0.0
    bearing_deg: float = 0.0
    elevation_deg: float = 0.0
    velocity_m_s: float = 0.0
    heading_deg: float = 0.0
    threat_level: ThreatLevel = ThreatLevel.UNKNOWN
    track_id: str = ""
    is_locked: bool = False
    is_designated: bool = False
    time_to_impact_s: Optional[float] = None


@dataclass
class SensorStatus:
    """Sensor system status information."""
    mode: str = "STBY"              # Operating mode
    fov_deg: float = 10.0           # Field of view
    zoom: float = 1.0               # Zoom level
    polarity: str = "WHOT"          # White/Black hot
    agc_mode: str = "AUTO"          # AGC mode
    laser_armed: bool = False       # Laser rangefinder armed
    laser_firing: bool = False      # Laser firing
    designator_code: str = "1111"   # Laser code
    track_mode: str = "AREA"        # Track mode


@dataclass
class PlatformStatus:
    """Platform status information."""
    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    altitude_m: float = 0.0
    airspeed_m_s: float = 0.0
    ground_speed_m_s: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    mach: float = 0.0
    g_load: float = 1.0


@dataclass
class GimbalStatus:
    """Gimbal status for display."""
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    az_rate_deg_s: float = 0.0
    el_rate_deg_s: float = 0.0
    at_limit: bool = False
    mode: str = "MANUAL"


class SymbologyRenderer:
    """
    Renders tactical symbology overlays on sensor imagery.

    Implements MIL-STD style symbology with configurable appearance.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        color: SymbolColor = SymbolColor.GREEN,
        line_width: int = 1
    ):
        self.width = width
        self.height = height
        self.primary_color = color.value
        self.line_width = line_width

        # Try to load a monospace font, fall back to default
        self.font = None
        self.font_small = None
        self._load_fonts()

    def _load_fonts(self):
        """Load fonts for text rendering."""
        try:
            # Try to load a standard monospace font
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10)
        except (OSError, IOError):
            try:
                self.font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSansMono.ttf", 14)
                self.font_small = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSansMono.ttf", 10)
            except (OSError, IOError):
                # Fall back to default font
                self.font = ImageFont.load_default()
                self.font_small = self.font

    def create_overlay(self) -> Image.Image:
        """Create a transparent overlay image."""
        return Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))

    def draw_reticle(
        self,
        draw: ImageDraw.ImageDraw,
        reticle_type: ReticleType = ReticleType.CROSSHAIR,
        position: Optional[ScreenPosition] = None,
        size: int = 40,
        color: Optional[Tuple[int, int, int]] = None
    ):
        """Draw a targeting reticle."""
        if position is None:
            position = ScreenPosition(0.5, 0.5)
        if color is None:
            color = self.primary_color

        cx, cy = position.to_pixels(self.width, self.height)
        r = size // 2

        if reticle_type == ReticleType.CROSSHAIR:
            # Simple crosshair with gap
            gap = 5
            draw.line([(cx - r, cy), (cx - gap, cy)], fill=color, width=self.line_width)
            draw.line([(cx + gap, cy), (cx + r, cy)], fill=color, width=self.line_width)
            draw.line([(cx, cy - r), (cx, cy - gap)], fill=color, width=self.line_width)
            draw.line([(cx, cy + gap), (cx, cy + r)], fill=color, width=self.line_width)

        elif reticle_type == ReticleType.CIRCLE:
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)],
                        outline=color, width=self.line_width)

        elif reticle_type == ReticleType.CROSS_CIRCLE:
            # Circle with crosshair
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)],
                        outline=color, width=self.line_width)
            gap = r + 5
            draw.line([(cx - gap - 10, cy), (cx - gap, cy)], fill=color, width=self.line_width)
            draw.line([(cx + gap, cy), (cx + gap + 10, cy)], fill=color, width=self.line_width)
            draw.line([(cx, cy - gap - 10), (cx, cy - gap)], fill=color, width=self.line_width)
            draw.line([(cx, cy + gap), (cx, cy + gap + 10)], fill=color, width=self.line_width)

        elif reticle_type == ReticleType.DIAMOND:
            points = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
            draw.polygon(points, outline=color)

        elif reticle_type == ReticleType.PLUS:
            draw.line([(cx - r, cy), (cx + r, cy)], fill=color, width=self.line_width)
            draw.line([(cx, cy - r), (cx, cy + r)], fill=color, width=self.line_width)

        elif reticle_type == ReticleType.AIMPOINT:
            # Dot with circle
            draw.ellipse([(cx - 2, cy - 2), (cx + 2, cy + 2)], fill=color)
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)],
                        outline=color, width=self.line_width)

        elif reticle_type == ReticleType.PIPPER:
            # Traditional pipper (circle with center dot)
            draw.ellipse([(cx - 1, cy - 1), (cx + 1, cy + 1)], fill=color)
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)],
                        outline=color, width=self.line_width)
            # Tick marks
            for angle in [0, 90, 180, 270]:
                rad = math.radians(angle)
                x1 = cx + int((r - 5) * math.cos(rad))
                y1 = cy - int((r - 5) * math.sin(rad))
                x2 = cx + int((r + 5) * math.cos(rad))
                y2 = cy - int((r + 5) * math.sin(rad))
                draw.line([(x1, y1), (x2, y2)], fill=color, width=self.line_width)

    def draw_track_gate(
        self,
        draw: ImageDraw.ImageDraw,
        position: ScreenPosition,
        size: int = 30,
        gate_type: TrackGateType = TrackGateType.CORNER,
        color: Optional[Tuple[int, int, int]] = None,
        is_locked: bool = False
    ):
        """Draw a track gate around a target."""
        if color is None:
            color = self.primary_color

        cx, cy = position.to_pixels(self.width, self.height)
        r = size // 2
        corner_len = r // 2

        if gate_type == TrackGateType.SQUARE:
            draw.rectangle([(cx - r, cy - r), (cx + r, cy + r)],
                          outline=color, width=self.line_width)

        elif gate_type == TrackGateType.CORNER:
            # Corner brackets only
            # Top-left
            draw.line([(cx - r, cy - r), (cx - r, cy - r + corner_len)],
                     fill=color, width=self.line_width)
            draw.line([(cx - r, cy - r), (cx - r + corner_len, cy - r)],
                     fill=color, width=self.line_width)
            # Top-right
            draw.line([(cx + r, cy - r), (cx + r, cy - r + corner_len)],
                     fill=color, width=self.line_width)
            draw.line([(cx + r, cy - r), (cx + r - corner_len, cy - r)],
                     fill=color, width=self.line_width)
            # Bottom-left
            draw.line([(cx - r, cy + r), (cx - r, cy + r - corner_len)],
                     fill=color, width=self.line_width)
            draw.line([(cx - r, cy + r), (cx - r + corner_len, cy + r)],
                     fill=color, width=self.line_width)
            # Bottom-right
            draw.line([(cx + r, cy + r), (cx + r, cy + r - corner_len)],
                     fill=color, width=self.line_width)
            draw.line([(cx + r, cy + r), (cx + r - corner_len, cy + r)],
                     fill=color, width=self.line_width)

        elif gate_type == TrackGateType.DIAMOND:
            points = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
            draw.polygon(points, outline=color)

        elif gate_type == TrackGateType.CIRCLE:
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)],
                        outline=color, width=self.line_width)

        elif gate_type == TrackGateType.ACQUISITION:
            # Large dashed box for acquisition
            dash_len = 5
            for i in range(0, r * 2, dash_len * 2):
                # Top
                draw.line([(cx - r + i, cy - r), (cx - r + i + dash_len, cy - r)],
                         fill=color, width=self.line_width)
                # Bottom
                draw.line([(cx - r + i, cy + r), (cx - r + i + dash_len, cy + r)],
                         fill=color, width=self.line_width)
                # Left
                draw.line([(cx - r, cy - r + i), (cx - r, cy - r + i + dash_len)],
                         fill=color, width=self.line_width)
                # Right
                draw.line([(cx + r, cy - r + i), (cx + r, cy - r + i + dash_len)],
                         fill=color, width=self.line_width)

        # Add lock indicator if locked
        if is_locked:
            lock_color = SymbolColor.RED.value
            draw.ellipse([(cx - 3, cy - 3), (cx + 3, cy + 3)], fill=lock_color)

    def draw_compass_rose(
        self,
        draw: ImageDraw.ImageDraw,
        heading_deg: float,
        position: Optional[ScreenPosition] = None,
        size: int = 60,
        color: Optional[Tuple[int, int, int]] = None
    ):
        """Draw a compass rose with heading."""
        if position is None:
            position = ScreenPosition(0.5, 0.08)
        if color is None:
            color = self.primary_color

        cx, cy = position.to_pixels(self.width, self.height)
        r = size // 2

        # Draw compass arc
        draw.arc([(cx - r, cy - r // 2), (cx + r, cy + r // 2)],
                start=180, end=360, fill=color, width=self.line_width)

        # Draw tick marks and labels
        for deg in range(0, 360, 30):
            rad = math.radians(deg - heading_deg)
            # Only draw if in view (top arc)
            if -90 <= (deg - heading_deg) <= 90 or (deg - heading_deg) > 270 or (deg - heading_deg) < -270:
                tick_x = cx + int(r * 0.9 * math.sin(rad))
                tick_y = cy - int(r * 0.4 * math.cos(rad))

                if deg % 90 == 0:
                    # Cardinal direction
                    labels = {0: 'N', 90: 'E', 180: 'S', 270: 'W'}
                    draw.text((tick_x - 5, tick_y - 10), labels.get(deg, ''),
                             fill=color, font=self.font_small)
                else:
                    # Tick mark
                    draw.ellipse([(tick_x - 1, tick_y - 1), (tick_x + 1, tick_y + 1)],
                                fill=color)

        # Heading readout
        heading_text = f"{int(heading_deg):03d}°"
        draw.text((cx - 15, cy + 5), heading_text, fill=color, font=self.font)

        # Heading caret (center mark)
        draw.polygon([(cx, cy - r // 2 - 5), (cx - 5, cy - r // 2),
                     (cx + 5, cy - r // 2)], fill=color)

    def draw_pitch_ladder(
        self,
        draw: ImageDraw.ImageDraw,
        pitch_deg: float,
        roll_deg: float = 0.0,
        color: Optional[Tuple[int, int, int]] = None
    ):
        """Draw a pitch ladder (artificial horizon)."""
        if color is None:
            color = self.primary_color

        cx = self.width // 2
        cy = self.height // 2

        # Pixels per degree of pitch
        ppd = 8

        # Draw pitch lines
        for p in range(-30, 35, 5):
            if p == 0:
                continue  # Skip horizon

            y_offset = int((p - pitch_deg) * ppd)

            # Apply roll rotation
            roll_rad = math.radians(roll_deg)
            line_half_width = 40 if p % 10 == 0 else 20

            x1 = cx - line_half_width
            x2 = cx + line_half_width
            y1 = cy - y_offset
            y2 = cy - y_offset

            # Rotate around center
            x1r = cx + int((x1 - cx) * math.cos(roll_rad) - (y1 - cy) * math.sin(roll_rad))
            y1r = cy + int((x1 - cx) * math.sin(roll_rad) + (y1 - cy) * math.cos(roll_rad))
            x2r = cx + int((x2 - cx) * math.cos(roll_rad) - (y2 - cy) * math.sin(roll_rad))
            y2r = cy + int((x2 - cx) * math.sin(roll_rad) + (y2 - cy) * math.cos(roll_rad))

            if p > 0:
                # Above horizon - solid line
                draw.line([(x1r, y1r), (x2r, y2r)], fill=color, width=self.line_width)
            else:
                # Below horizon - dashed
                draw.line([(x1r, y1r), (x2r, y2r)], fill=color, width=self.line_width)

            # Pitch value labels (every 10°)
            if p % 10 == 0 and abs(p - pitch_deg) < 25:
                label = f"{abs(p)}"
                draw.text((x2r + 5, y2r - 5), label, fill=color, font=self.font_small)

        # Draw horizon line
        h_offset = int(-pitch_deg * ppd)
        x1 = cx - 80
        x2 = cx + 80
        y1 = cy - h_offset
        roll_rad = math.radians(roll_deg)
        x1r = cx + int((x1 - cx) * math.cos(roll_rad))
        y1r = cy - h_offset + int((x1 - cx) * math.sin(roll_rad))
        x2r = cx + int((x2 - cx) * math.cos(roll_rad))
        y2r = cy - h_offset + int((x2 - cx) * math.sin(roll_rad))

        draw.line([(x1r, y1r), (x2r, y2r)], fill=color, width=2)

    def draw_status_box(
        self,
        draw: ImageDraw.ImageDraw,
        sensor_status: SensorStatus,
        position: str = "top_left",
        color: Optional[Tuple[int, int, int]] = None
    ):
        """Draw sensor status information box."""
        if color is None:
            color = self.primary_color

        # Position
        if position == "top_left":
            x, y = 10, 10
        elif position == "top_right":
            x, y = self.width - 100, 10
        elif position == "bottom_left":
            x, y = 10, self.height - 80
        else:
            x, y = self.width - 100, self.height - 80

        # Status lines
        lines = [
            f"MODE: {sensor_status.mode}",
            f"FOV:  {sensor_status.fov_deg:.1f}°",
            f"POL:  {sensor_status.polarity}",
            f"AGC:  {sensor_status.agc_mode}",
        ]

        if sensor_status.laser_armed:
            lines.append(f"LRF:  {sensor_status.designator_code}")

        for i, line in enumerate(lines):
            draw.text((x, y + i * 14), line, fill=color, font=self.font_small)

    def draw_range_display(
        self,
        draw: ImageDraw.ImageDraw,
        range_m: float,
        position: str = "bottom_center",
        color: Optional[Tuple[int, int, int]] = None
    ):
        """Draw range to target display."""
        if color is None:
            color = self.primary_color

        if position == "bottom_center":
            x = self.width // 2 - 40
            y = self.height - 30
        else:
            x = self.width // 2 - 40
            y = 30

        if range_m >= 1000:
            range_text = f"R: {range_m / 1000:.2f} km"
        else:
            range_text = f"R: {range_m:.0f} m"

        draw.text((x, y), range_text, fill=color, font=self.font)

    def draw_gimbal_position(
        self,
        draw: ImageDraw.ImageDraw,
        gimbal_status: GimbalStatus,
        position: str = "bottom_right",
        color: Optional[Tuple[int, int, int]] = None
    ):
        """Draw gimbal position indicator."""
        if color is None:
            color = self.primary_color

        if position == "bottom_right":
            x = self.width - 100
            y = self.height - 60
        else:
            x = 10
            y = self.height - 60

        lines = [
            f"AZ:  {gimbal_status.azimuth_deg:+6.1f}°",
            f"EL:  {gimbal_status.elevation_deg:+6.1f}°",
            f"MODE: {gimbal_status.mode}",
        ]

        for i, line in enumerate(lines):
            draw.text((x, y + i * 14), line, fill=color, font=self.font_small)

        # Limit warning
        if gimbal_status.at_limit:
            draw.text((x, y + 42), "AT LIMIT", fill=SymbolColor.AMBER.value,
                     font=self.font_small)

    def draw_target_info(
        self,
        draw: ImageDraw.ImageDraw,
        target: TargetDesignation,
        color: Optional[Tuple[int, int, int]] = None
    ):
        """Draw target information near track gate."""
        if color is None:
            # Color based on threat level
            threat_colors = {
                ThreatLevel.UNKNOWN: SymbolColor.WHITE.value,
                ThreatLevel.FRIENDLY: SymbolColor.CYAN.value,
                ThreatLevel.NEUTRAL: SymbolColor.GREEN.value,
                ThreatLevel.HOSTILE: SymbolColor.RED.value,
                ThreatLevel.SUSPECT: SymbolColor.AMBER.value,
            }
            color = threat_colors.get(target.threat_level, self.primary_color)

        px, py = target.position.to_pixels(self.width, self.height)

        # Target ID
        if target.track_id:
            draw.text((px + 20, py - 20), target.track_id, fill=color, font=self.font_small)

        # Range
        if target.range_m > 0:
            if target.range_m >= 1000:
                range_text = f"{target.range_m / 1000:.1f}k"
            else:
                range_text = f"{target.range_m:.0f}m"
            draw.text((px + 20, py - 5), range_text, fill=color, font=self.font_small)

        # Velocity
        if target.velocity_m_s > 0:
            vel_text = f"{target.velocity_m_s:.0f}m/s"
            draw.text((px + 20, py + 10), vel_text, fill=color, font=self.font_small)

        # Time to impact
        if target.time_to_impact_s is not None:
            tti_text = f"TTI:{target.time_to_impact_s:.1f}s"
            draw.text((px + 20, py + 25), tti_text, fill=SymbolColor.RED.value,
                     font=self.font_small)

    def draw_mil_reticle(
        self,
        draw: ImageDraw.ImageDraw,
        mil_spacing: float = 1.0,
        color: Optional[Tuple[int, int, int]] = None
    ):
        """Draw mil-dot reticle for range estimation."""
        if color is None:
            color = self.primary_color

        cx = self.width // 2
        cy = self.height // 2

        # Pixels per mil (assuming 10° FOV = ~175 mils)
        pixels_per_mil = self.width / 175.0 * mil_spacing

        # Draw mil dots
        for i in range(-10, 11):
            if i == 0:
                continue
            # Horizontal
            x = cx + int(i * pixels_per_mil)
            if i % 5 == 0:
                draw.ellipse([(x - 2, cy - 2), (x + 2, cy + 2)], fill=color)
            else:
                draw.ellipse([(x - 1, cy - 1), (x + 1, cy + 1)], fill=color)

            # Vertical
            y = cy + int(i * pixels_per_mil)
            if i % 5 == 0:
                draw.ellipse([(cx - 2, y - 2), (cx + 2, y + 2)], fill=color)
            else:
                draw.ellipse([(cx - 1, y - 1), (cx + 1, y + 1)], fill=color)

        # Center crosshair
        draw.line([(cx - 20, cy), (cx - 5, cy)], fill=color, width=self.line_width)
        draw.line([(cx + 5, cy), (cx + 20, cy)], fill=color, width=self.line_width)
        draw.line([(cx, cy - 20), (cx, cy - 5)], fill=color, width=self.line_width)
        draw.line([(cx, cy + 5), (cx, cy + 20)], fill=color, width=self.line_width)

    def draw_flir_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        sensor_status: SensorStatus,
        gimbal_status: GimbalStatus,
        target: Optional[TargetDesignation] = None,
        heading_deg: float = 0.0
    ):
        """Draw complete FLIR display overlay."""
        # Reticle
        self.draw_reticle(draw, ReticleType.CROSS_CIRCLE)

        # Status box
        self.draw_status_box(draw, sensor_status, "top_left")

        # Gimbal position
        self.draw_gimbal_position(draw, gimbal_status, "bottom_right")

        # Compass
        self.draw_compass_rose(draw, heading_deg)

        # Target designation
        if target is not None:
            gate_type = TrackGateType.TRACK if target.is_locked else TrackGateType.ACQUISITION
            self.draw_track_gate(draw, target.position, gate_type=gate_type,
                               is_locked=target.is_locked)
            self.draw_target_info(draw, target)

            if target.range_m > 0:
                self.draw_range_display(draw, target.range_m)

    def render(
        self,
        base_image: np.ndarray,
        sensor_status: Optional[SensorStatus] = None,
        gimbal_status: Optional[GimbalStatus] = None,
        targets: Optional[List[TargetDesignation]] = None,
        platform_status: Optional[PlatformStatus] = None,
        show_reticle: bool = True,
        show_compass: bool = True,
        show_status: bool = True
    ) -> np.ndarray:
        """
        Render complete symbology overlay on image.

        Args:
            base_image: Input image (numpy array, RGB or grayscale)
            sensor_status: Sensor status information
            gimbal_status: Gimbal status information
            targets: List of target designations
            platform_status: Platform status for heading, attitude
            show_reticle: Show center reticle
            show_compass: Show compass rose
            show_status: Show status boxes

        Returns:
            Image with symbology overlay (numpy array)
        """
        # Convert numpy to PIL
        if len(base_image.shape) == 2:
            # Grayscale - convert to RGB
            base_pil = Image.fromarray(base_image).convert('RGB')
        else:
            base_pil = Image.fromarray(base_image)

        # Update dimensions
        self.width, self.height = base_pil.size

        # Create overlay
        overlay = self.create_overlay()
        draw = ImageDraw.Draw(overlay)

        # Draw elements
        if show_reticle:
            self.draw_reticle(draw, ReticleType.CROSS_CIRCLE)

        if show_compass and platform_status is not None:
            self.draw_compass_rose(draw, platform_status.heading_deg)

        if show_status:
            if sensor_status is not None:
                self.draw_status_box(draw, sensor_status, "top_left")
            if gimbal_status is not None:
                self.draw_gimbal_position(draw, gimbal_status, "bottom_right")

        # Draw targets
        if targets is not None:
            for target in targets:
                gate_type = TrackGateType.TRACK if target.is_locked else TrackGateType.CORNER
                self.draw_track_gate(draw, target.position, gate_type=gate_type,
                                   is_locked=target.is_locked)
                self.draw_target_info(draw, target)

                if target.is_designated and target.range_m > 0:
                    self.draw_range_display(draw, target.range_m)

        # Composite overlay on base image
        base_pil.paste(overlay, (0, 0), overlay)

        return np.array(base_pil)


# Convenience function
def create_default_symbology() -> SymbologyRenderer:
    """Create a symbology renderer with default settings."""
    return SymbologyRenderer(
        width=640,
        height=480,
        color=SymbolColor.GREEN,
        line_width=1
    )
