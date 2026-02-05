"""
Configuration classes for video-to-simulation processing.

Defines data structures for all three processing modes:
- Template: Use video as geometry/motion template with assigned thermal properties
- Radiance Map: Direct mapping of video intensity to scene radiance
- Thermal Estimate: Estimate thermal scene from visible video using segmentation
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Literal, Callable, Union
import numpy as np
from numpy.typing import NDArray


class VideoProcessingMode(Enum):
    """Video processing mode selection."""
    TEMPLATE = "template"           # Option 1: Geometry/motion template
    RADIANCE_MAP = "radiance_map"   # Option 2: Direct radiance mapping
    THERMAL_ESTIMATE = "thermal_estimate"  # Option 3: Thermal from visible


class SegmentationMethod(Enum):
    """Segmentation method for object detection."""
    THRESHOLD = "threshold"         # Simple intensity thresholding
    EDGE = "edge"                   # Edge-based segmentation
    COLOR = "color"                 # Color-based clustering
    CONTOUR = "contour"             # Contour detection
    ML_MODEL = "ml_model"           # Machine learning model


class ChannelMode(Enum):
    """How to handle color video channels."""
    LUMINANCE = "luminance"         # Convert to luminance (0.299R + 0.587G + 0.114B)
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    AVERAGE = "average"             # Simple average of channels
    MAX = "max"                     # Maximum across channels


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class VideoSource:
    """Metadata about a video source.

    Attributes:
        path: Path to video file
        fps: Frames per second
        frame_count: Total number of frames
        resolution: (height, width) in pixels
        duration_s: Duration in seconds
        codec: Video codec name
    """
    path: Path
    fps: float
    frame_count: int
    resolution: tuple[int, int]
    duration_s: float
    codec: str = ""

    @property
    def width(self) -> int:
        return self.resolution[1]

    @property
    def height(self) -> int:
        return self.resolution[0]


@dataclass
class FrameData:
    """Data extracted from a single video frame.

    Attributes:
        frame_index: Frame number (0-indexed)
        timestamp_s: Time position in seconds
        image: Raw pixel data (H, W, C) or (H, W)
        segmentation_mask: Object class per pixel (optional)
        motion_vectors: Optical flow field (optional)
        objects: Detected object instances (optional)
    """
    frame_index: int
    timestamp_s: float
    image: NDArray
    segmentation_mask: Optional[NDArray] = None
    motion_vectors: Optional[NDArray] = None
    objects: Optional[list["ObjectInstance"]] = None


@dataclass
class ObjectInstance:
    """A detected object in a frame.

    Attributes:
        object_id: Unique identifier for tracking across frames
        class_name: Object class (person, vehicle, building, etc.)
        class_id: Numeric class identifier
        bounding_box: (y0, x0, y1, x1) pixel coordinates
        mask: Boolean mask of object pixels
        centroid: (y, x) center position
        area_pixels: Number of pixels in object
        confidence: Detection confidence (0-1)
        velocity: (vy, vx) in pixels/frame if tracked
    """
    object_id: int
    class_name: str
    class_id: int = 0
    bounding_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    mask: Optional[NDArray] = None
    centroid: tuple[float, float] = (0.0, 0.0)
    area_pixels: int = 0
    confidence: float = 1.0
    velocity: Optional[tuple[float, float]] = None


# =============================================================================
# Thermal Property Configurations
# =============================================================================

@dataclass
class HotSpotConfig:
    """Configuration for a thermal hot spot within an object.

    Attributes:
        relative_position: (y, x) position normalized 0-1 within object bounds
        relative_size: Size as fraction of object size
        temperature_delta_k: Temperature added to base object temperature
        shape: Hot spot shape
    """
    relative_position: tuple[float, float] = (0.5, 0.5)
    relative_size: float = 0.2
    temperature_delta_k: float = 20.0
    shape: Literal["circle", "rectangle", "ellipse"] = "circle"


@dataclass
class ObjectThermalProperties:
    """Thermal/radiometric properties for an object class.

    Attributes:
        temperature_k: Base temperature in Kelvin
        temperature_std_k: Random temperature variation (std dev)
        emissivity: Surface emissivity (0-1)
        emissivity_std: Emissivity variation
        hot_spots: List of internal hot spots (e.g., engine)
    """
    temperature_k: float
    temperature_std_k: float = 2.0
    emissivity: float = 0.95
    emissivity_std: float = 0.02
    hot_spots: list[HotSpotConfig] = field(default_factory=list)


# Default thermal properties for common object classes
DEFAULT_THERMAL_PROPERTIES: dict[str, ObjectThermalProperties] = {
    "person": ObjectThermalProperties(
        temperature_k=305.0,
        temperature_std_k=2.0,
        emissivity=0.98,
    ),
    "vehicle": ObjectThermalProperties(
        temperature_k=320.0,
        temperature_std_k=5.0,
        emissivity=0.85,
        hot_spots=[
            HotSpotConfig(
                relative_position=(0.5, 0.3),  # Engine area
                relative_size=0.25,
                temperature_delta_k=30.0,
            ),
            HotSpotConfig(
                relative_position=(0.5, 0.9),  # Exhaust
                relative_size=0.1,
                temperature_delta_k=50.0,
            ),
        ],
    ),
    "car": ObjectThermalProperties(
        temperature_k=315.0,
        temperature_std_k=5.0,
        emissivity=0.85,
        hot_spots=[
            HotSpotConfig(relative_position=(0.5, 0.2), relative_size=0.2, temperature_delta_k=25.0),
        ],
    ),
    "truck": ObjectThermalProperties(
        temperature_k=325.0,
        temperature_std_k=8.0,
        emissivity=0.82,
        hot_spots=[
            HotSpotConfig(relative_position=(0.4, 0.15), relative_size=0.15, temperature_delta_k=40.0),
        ],
    ),
    "building": ObjectThermalProperties(
        temperature_k=295.0,
        temperature_std_k=3.0,
        emissivity=0.92,
    ),
    "vegetation": ObjectThermalProperties(
        temperature_k=290.0,
        temperature_std_k=3.0,
        emissivity=0.96,
    ),
    "road": ObjectThermalProperties(
        temperature_k=305.0,
        temperature_std_k=5.0,
        emissivity=0.93,
    ),
    "water": ObjectThermalProperties(
        temperature_k=288.0,
        temperature_std_k=2.0,
        emissivity=0.96,
    ),
    "sky": ObjectThermalProperties(
        temperature_k=250.0,
        temperature_std_k=10.0,
        emissivity=1.0,
    ),
    "aircraft": ObjectThermalProperties(
        temperature_k=280.0,
        temperature_std_k=5.0,
        emissivity=0.85,
        hot_spots=[
            HotSpotConfig(relative_position=(0.5, 0.9), relative_size=0.15, temperature_delta_k=200.0),
        ],
    ),
    "animal": ObjectThermalProperties(
        temperature_k=308.0,
        temperature_std_k=3.0,
        emissivity=0.95,
    ),
    "background": ObjectThermalProperties(
        temperature_k=290.0,
        temperature_std_k=5.0,
        emissivity=0.95,
    ),
}


# =============================================================================
# Mode-Specific Configurations
# =============================================================================

@dataclass
class TemplateConfig:
    """Configuration for Template mode (Option 1).

    Uses video as geometry/motion template with assigned thermal properties.

    Attributes:
        segmentation_method: How to segment objects
        class_properties: Mapping of class names to thermal properties
        background_temperature_k: Temperature for unclassified regions
        background_emissivity: Emissivity for unclassified regions
        use_optical_flow: Compute motion vectors for motion blur
        motion_blur_scale: Scale factor for motion blur amount
        threshold_value: Threshold for simple segmentation (0-255)
        min_object_size: Minimum object size in pixels
    """
    segmentation_method: SegmentationMethod = SegmentationMethod.THRESHOLD
    class_properties: dict[str, ObjectThermalProperties] = field(
        default_factory=lambda: DEFAULT_THERMAL_PROPERTIES.copy()
    )
    background_temperature_k: float = 290.0
    background_emissivity: float = 0.95
    use_optical_flow: bool = True
    motion_blur_scale: float = 1.0
    threshold_value: int = 128
    min_object_size: int = 100

    # For ML-based segmentation
    model_path: Optional[Path] = None
    model_type: str = "yolo"  # "yolo", "sam", "custom"


@dataclass
class RadianceMapConfig:
    """Configuration for Radiance Map mode (Option 2).

    Direct mapping of video intensity to scene radiance.

    Attributes:
        mapping_type: Type of intensity-to-radiance mapping
        input_range: Expected input intensity range
        output_radiance_range: Output radiance range in W/(m²·sr)
        gamma: Gamma correction factor
        channel_mode: How to handle color channels
        normalization: Per-frame or global normalization
        invert: Invert mapping (bright→cold or bright→hot)
        reference_temperature_k: Reference temperature for thermal context
    """
    mapping_type: Literal["linear", "gamma", "logarithmic", "sigmoid"] = "linear"
    input_range: tuple[float, float] = (0.0, 255.0)
    output_radiance_range: tuple[float, float] = (0.001, 0.1)  # W/(m²·sr)
    gamma: float = 1.0
    channel_mode: ChannelMode = ChannelMode.LUMINANCE
    normalization: Literal["none", "per_frame", "global"] = "none"
    invert: bool = False
    reference_temperature_k: float = 300.0

    # For lookup table mapping
    lut: Optional[NDArray] = None


@dataclass
class ThermalEstimateConfig:
    """Configuration for Thermal Estimate mode (Option 3).

    Estimate thermal scene from visible video using segmentation and
    class-based temperature assignment.

    Attributes:
        segmentation_method: Method for object segmentation
        class_temperatures: Mapping of class names to thermal properties
        ambient_temperature_k: Default ambient temperature
        estimation_method: How to estimate temperatures
        solar_loading: Account for solar heating effects
        time_of_day: Time context for temperature estimation
        smooth_boundaries: Apply smoothing at object boundaries
        boundary_blur_pixels: Size of boundary smoothing kernel
    """
    segmentation_method: SegmentationMethod = SegmentationMethod.THRESHOLD
    class_temperatures: dict[str, ObjectThermalProperties] = field(
        default_factory=lambda: DEFAULT_THERMAL_PROPERTIES.copy()
    )
    ambient_temperature_k: float = 290.0
    estimation_method: Literal["class_lookup", "brightness_correlation", "hybrid"] = "class_lookup"
    solar_loading: bool = False
    time_of_day: Literal["day", "night", "dawn", "dusk"] = "day"
    smooth_boundaries: bool = True
    boundary_blur_pixels: int = 3

    # For ML-based segmentation
    model_path: Optional[Path] = None
    model_type: str = "yolo"


# =============================================================================
# Main Video Simulation Configuration
# =============================================================================

@dataclass
class VideoSimulationConfig:
    """Complete configuration for video simulation.

    Attributes:
        mode: Processing mode (template, radiance_map, thermal_estimate)
        sensor_type: Target sensor type
        template_config: Configuration for template mode
        radiance_config: Configuration for radiance mapping mode
        thermal_config: Configuration for thermal estimation mode
        output_fps: Output frame rate (None = match input)
        frame_range: (start, end) frame indices to process
        output_resolution: Output resolution (None = match input)
        apply_motion_blur: Apply motion blur effects
        apply_noise: Apply sensor noise
        temporal_noise_correlation: Noise correlation between frames (0-1)
        output_path: Path for output video
        output_format: Output video format
        random_seed: Seed for reproducibility
    """
    mode: VideoProcessingMode = VideoProcessingMode.RADIANCE_MAP
    sensor_type: str = "lwir"

    # Mode-specific configs
    template_config: Optional[TemplateConfig] = None
    radiance_config: Optional[RadianceMapConfig] = None
    thermal_config: Optional[ThermalEstimateConfig] = None

    # Temporal settings
    output_fps: Optional[float] = None
    frame_range: Optional[tuple[int, int]] = None
    output_resolution: Optional[tuple[int, int]] = None

    # Effects
    apply_motion_blur: bool = False
    apply_noise: bool = True
    temporal_noise_correlation: float = 0.3

    # Output
    output_path: Optional[Path] = None
    output_format: str = "mp4"
    output_codec: str = "mp4v"

    # Simulation
    range_m: float = 500.0
    random_seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize mode-specific config if not provided."""
        if self.mode == VideoProcessingMode.TEMPLATE and self.template_config is None:
            self.template_config = TemplateConfig()
        elif self.mode == VideoProcessingMode.RADIANCE_MAP and self.radiance_config is None:
            self.radiance_config = RadianceMapConfig()
        elif self.mode == VideoProcessingMode.THERMAL_ESTIMATE and self.thermal_config is None:
            self.thermal_config = ThermalEstimateConfig()

    def get_active_config(self) -> Union[TemplateConfig, RadianceMapConfig, ThermalEstimateConfig]:
        """Get the configuration for the active mode."""
        if self.mode == VideoProcessingMode.TEMPLATE:
            return self.template_config or TemplateConfig()
        elif self.mode == VideoProcessingMode.RADIANCE_MAP:
            return self.radiance_config or RadianceMapConfig()
        else:
            return self.thermal_config or ThermalEstimateConfig()


# =============================================================================
# Result Structures
# =============================================================================

@dataclass
class FrameResult:
    """Result from processing a single frame.

    Attributes:
        frame_index: Frame number
        timestamp_s: Time position
        digital_image: Simulated sensor output
        dn_min: Minimum DN value
        dn_max: Maximum DN value
        dn_mean: Mean DN value
        objects_detected: Number of objects found
    """
    frame_index: int
    timestamp_s: float
    digital_image: NDArray
    dn_min: int = 0
    dn_max: int = 0
    dn_mean: float = 0.0
    objects_detected: int = 0


@dataclass
class VideoSimulationResult:
    """Result from video simulation.

    Attributes:
        output_path: Path to output video
        frames_processed: Number of frames processed
        total_frames: Total frames in input
        processing_time_s: Total processing time
        fps_achieved: Achieved processing FPS
        config: Configuration used
        frame_results: Per-frame statistics
    """
    output_path: Path
    frames_processed: int
    total_frames: int
    processing_time_s: float
    fps_achieved: float
    config: VideoSimulationConfig
    frame_results: list[FrameResult] = field(default_factory=list)

    @property
    def dn_range(self) -> tuple[int, int]:
        """Overall DN range across all frames."""
        if not self.frame_results:
            return (0, 0)
        min_dn = min(r.dn_min for r in self.frame_results)
        max_dn = max(r.dn_max for r in self.frame_results)
        return (min_dn, max_dn)
