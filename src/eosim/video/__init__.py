"""
EOSIM Video Processing Module.

Provides video-to-simulation conversion capabilities with three modes:

1. **Template Mode** (Option 1): Use video as geometry/motion template
   with assigned thermal properties. Good for creating realistic thermal
   videos when you want precise control over object temperatures.

2. **Radiance Map Mode** (Option 2): Direct mapping of video intensity
   to scene radiance. Simplest mode, applies sensor effects to video.

3. **Thermal Estimate Mode** (Option 3): Estimates thermal scene from
   visible video using segmentation and class-based temperature assignment.
   Best for automatically generating realistic thermal videos.

Quick Start:
    >>> from eosim.video import process_video, VideoSimulationConfig, VideoProcessingMode
    >>>
    >>> # Simple radiance mapping
    >>> config = VideoSimulationConfig(
    ...     mode=VideoProcessingMode.RADIANCE_MAP,
    ...     sensor_type="lwir",
    ... )
    >>> result = process_video("input.mp4", config, "output.mp4")
    >>>
    >>> # Or use convenience functions:
    >>> from eosim.video import simulate_video_thermal
    >>> output_path = simulate_video_thermal("input.mp4", sensor_type="lwir")

Classes:
    VideoProcessor: Main video processing pipeline
    VideoReader: Video file reading
    VideoWriter: Video file writing
    RadianceMapper: Intensity-to-radiance mapping
    ThermalEstimator: Thermal scene estimation
    TemplateMapper: Template-based scene generation
    Segmenter: Object segmentation

Configuration:
    VideoSimulationConfig: Main configuration
    VideoProcessingMode: Processing mode selection
    RadianceMapConfig: Radiance mapping settings
    ThermalEstimateConfig: Thermal estimation settings
    TemplateConfig: Template mode settings
"""

from eosim.video.config import (
    # Enums
    VideoProcessingMode,
    SegmentationMethod,
    ChannelMode,
    # Data structures
    VideoSource,
    FrameData,
    ObjectInstance,
    # Thermal properties
    ObjectThermalProperties,
    HotSpotConfig,
    DEFAULT_THERMAL_PROPERTIES,
    # Mode configs
    TemplateConfig,
    RadianceMapConfig,
    ThermalEstimateConfig,
    # Main config
    VideoSimulationConfig,
    # Results
    FrameResult,
    VideoSimulationResult,
)

from eosim.video.reader import (
    VideoReader,
    to_grayscale,
    compute_optical_flow,
    extract_frames,
)

from eosim.video.writer import (
    VideoWriter,
    frames_to_video,
    dn_to_display,
    create_comparison_video,
)

from eosim.video.radiance_mapper import (
    RadianceMapper,
    AdaptiveRadianceMapper,
    intensity_to_radiance,
    estimate_radiance_range_for_sensor,
)

from eosim.video.segmentation import (
    Segmenter,
    ObjectClassifier,
    ObjectTracker,
    segment_frame,
)

from eosim.video.thermal_estimator import (
    ThermalEstimator,
    TemplateMapper,
    estimate_thermal_from_visible,
)

from eosim.video.processor import (
    VideoProcessor,
    process_video,
    simulate_video_thermal,
    simulate_video_sensor,
    simulate_video_template,
)


__all__ = [
    # Enums
    "VideoProcessingMode",
    "SegmentationMethod",
    "ChannelMode",
    # Data structures
    "VideoSource",
    "FrameData",
    "ObjectInstance",
    # Thermal properties
    "ObjectThermalProperties",
    "HotSpotConfig",
    "DEFAULT_THERMAL_PROPERTIES",
    # Configurations
    "TemplateConfig",
    "RadianceMapConfig",
    "ThermalEstimateConfig",
    "VideoSimulationConfig",
    # Results
    "FrameResult",
    "VideoSimulationResult",
    # Reader/Writer
    "VideoReader",
    "VideoWriter",
    "to_grayscale",
    "compute_optical_flow",
    "extract_frames",
    "frames_to_video",
    "dn_to_display",
    "create_comparison_video",
    # Mappers
    "RadianceMapper",
    "AdaptiveRadianceMapper",
    "intensity_to_radiance",
    "estimate_radiance_range_for_sensor",
    # Segmentation
    "Segmenter",
    "ObjectClassifier",
    "ObjectTracker",
    "segment_frame",
    # Thermal estimation
    "ThermalEstimator",
    "TemplateMapper",
    "estimate_thermal_from_visible",
    # Processor
    "VideoProcessor",
    "process_video",
    "simulate_video_thermal",
    "simulate_video_sensor",
    "simulate_video_template",
]
