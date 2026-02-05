"""
Main video processing pipeline for EOSIM.

Integrates video reading, scene conversion, and simulation into
a complete video-to-simulation pipeline.
"""

import time
from pathlib import Path
from typing import Optional, Union, Callable
import numpy as np
from numpy.typing import NDArray

from eosim.video.config import (
    VideoSimulationConfig,
    VideoProcessingMode,
    VideoSimulationResult,
    FrameResult,
    FrameData,
)
from eosim.video.reader import VideoReader, compute_optical_flow
from eosim.video.writer import VideoWriter, dn_to_display
from eosim.video.radiance_mapper import RadianceMapper
from eosim.video.thermal_estimator import ThermalEstimator, TemplateMapper
from eosim.pipeline.simulation import SimulationPipeline, SimulationConfig, SceneInput, create_pipeline


class VideoProcessor:
    """Main video-to-simulation processor.

    Orchestrates the complete pipeline from input video to simulated
    sensor output video.

    Supports three processing modes:
    - TEMPLATE: Video as geometry/motion template with assigned properties
    - RADIANCE_MAP: Direct intensity-to-radiance mapping
    - THERMAL_ESTIMATE: Estimate thermal scene from visible video

    Example:
        >>> config = VideoSimulationConfig(
        ...     mode=VideoProcessingMode.RADIANCE_MAP,
        ...     sensor_type="lwir",
        ...     output_path=Path("output.mp4"),
        ... )
        >>> processor = VideoProcessor(config)
        >>> result = processor.process("input.mp4")
        >>> print(f"Processed {result.frames_processed} frames")
    """

    def __init__(
        self,
        config: Optional[VideoSimulationConfig] = None,
    ) -> None:
        """Initialize video processor.

        Args:
            config: Video simulation configuration
        """
        self.config = config or VideoSimulationConfig()

        # Initialize components based on mode
        self._radiance_mapper: Optional[RadianceMapper] = None
        self._thermal_estimator: Optional[ThermalEstimator] = None
        self._template_mapper: Optional[TemplateMapper] = None

        if self.config.mode == VideoProcessingMode.RADIANCE_MAP:
            self._radiance_mapper = RadianceMapper(self.config.radiance_config)
        elif self.config.mode == VideoProcessingMode.THERMAL_ESTIMATE:
            self._thermal_estimator = ThermalEstimator(self.config.thermal_config)
        else:  # TEMPLATE
            self._template_mapper = TemplateMapper(self.config.template_config)

        # Initialize simulation pipeline
        self._pipeline = create_pipeline(
            sensor_type=self.config.sensor_type,
            seed=self.config.random_seed,
        )

        # Random generator
        self._rng = np.random.default_rng(self.config.random_seed)

        # Temporal noise state
        self._noise_state: Optional[NDArray] = None

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> VideoSimulationResult:
        """Process video through simulation pipeline.

        Args:
            input_path: Path to input video
            output_path: Path for output video (overrides config)
            progress_callback: Callback for progress updates (frame, total)

        Returns:
            VideoSimulationResult with processing statistics
        """
        input_path = Path(input_path)
        output_path = Path(output_path) if output_path else self.config.output_path

        if output_path is None:
            output_path = input_path.with_stem(f"{input_path.stem}_simulated")

        start_time = time.time()
        frame_results = []

        # Open video reader
        start_frame = self.config.frame_range[0] if self.config.frame_range else 0
        end_frame = self.config.frame_range[1] if self.config.frame_range else None

        with VideoReader(input_path, start_frame=start_frame, end_frame=end_frame) as reader:
            source = reader.source
            total_frames = len(reader)

            # Determine output parameters
            output_fps = self.config.output_fps or source.fps
            output_resolution = self.config.output_resolution or source.resolution

            # For global normalization in radiance mode, pre-scan video
            if (self.config.mode == VideoProcessingMode.RADIANCE_MAP and
                self._radiance_mapper and
                self._radiance_mapper.config.normalization == "global"):
                self._compute_global_range(reader)
                reader.seek(start_frame)

            # Open video writer
            with VideoWriter(output_path, fps=output_fps, resolution=output_resolution) as writer:
                previous_frame: Optional[NDArray] = None

                for frame_data in reader:
                    # Process frame
                    result = self._process_frame(frame_data, previous_frame)

                    # Convert to displayable format
                    display_frame = dn_to_display(
                        result.digital_image,
                        colormap="inferno" if self.config.sensor_type in ("lwir", "mwir") else None,
                    )

                    # Write frame
                    writer.write_frame(display_frame)

                    # Store result
                    frame_results.append(result)

                    # Progress callback
                    if progress_callback:
                        progress_callback(len(frame_results), total_frames)

                    # Update previous frame for motion estimation
                    previous_frame = frame_data.image

        processing_time = time.time() - start_time

        return VideoSimulationResult(
            output_path=output_path,
            frames_processed=len(frame_results),
            total_frames=total_frames,
            processing_time_s=processing_time,
            fps_achieved=len(frame_results) / processing_time if processing_time > 0 else 0,
            config=self.config,
            frame_results=frame_results,
        )

    def _process_frame(
        self,
        frame_data: FrameData,
        previous_frame: Optional[NDArray],
    ) -> FrameResult:
        """Process a single frame through the simulation pipeline.

        Args:
            frame_data: Current frame data
            previous_frame: Previous frame for motion estimation

        Returns:
            FrameResult with simulated output
        """
        # Convert frame to scene input based on mode
        if self.config.mode == VideoProcessingMode.RADIANCE_MAP:
            scene = self._radiance_mapper.to_scene_input(frame_data.image)
        elif self.config.mode == VideoProcessingMode.THERMAL_ESTIMATE:
            scene = self._thermal_estimator.to_scene_input(frame_data.image)
        else:  # TEMPLATE
            scene = self._template_mapper.to_scene_input(
                frame_data.image, previous_frame
            )

        # Run simulation pipeline
        pipeline_result = self._pipeline.run(
            scene,
            range_m=self.config.range_m,
        )

        # Get digital image
        digital_image = pipeline_result.digital_image

        # Apply temporal noise correlation
        if self.config.apply_noise and self.config.temporal_noise_correlation > 0:
            digital_image = self._apply_temporal_noise(digital_image)

        # Apply motion blur if requested
        if self.config.apply_motion_blur and previous_frame is not None:
            digital_image = self._apply_motion_blur(
                digital_image, frame_data.image, previous_frame
            )

        return FrameResult(
            frame_index=frame_data.frame_index,
            timestamp_s=frame_data.timestamp_s,
            digital_image=digital_image,
            dn_min=int(digital_image.min()),
            dn_max=int(digital_image.max()),
            dn_mean=float(digital_image.mean()),
        )

    def _compute_global_range(self, reader: VideoReader) -> None:
        """Pre-scan video to compute global intensity range."""
        frames = []
        for frame_data in reader:
            frames.append(frame_data.image)
            if len(frames) >= 100:  # Sample up to 100 frames
                break

        if self._radiance_mapper:
            self._radiance_mapper.compute_global_range(frames)

    def _apply_temporal_noise(self, digital_image: NDArray) -> NDArray:
        """Apply temporally correlated noise.

        Creates more realistic noise that persists somewhat across frames.
        """
        correlation = self.config.temporal_noise_correlation

        if self._noise_state is None:
            # Initialize noise state
            self._noise_state = self._rng.normal(0, 1, digital_image.shape)

        # Update noise with correlation
        new_noise = self._rng.normal(0, 1, digital_image.shape)
        self._noise_state = (
            correlation * self._noise_state +
            np.sqrt(1 - correlation**2) * new_noise
        )

        # Scale noise based on signal level (simple approximation)
        noise_scale = 50  # DN units
        noisy = digital_image.astype(np.float64) + self._noise_state * noise_scale

        return np.clip(noisy, 0, 16383).astype(digital_image.dtype)

    def _apply_motion_blur(
        self,
        digital_image: NDArray,
        current_frame: NDArray,
        previous_frame: NDArray,
    ) -> NDArray:
        """Apply motion blur based on optical flow."""
        try:
            flow = compute_optical_flow(previous_frame, current_frame)

            # Compute motion magnitude
            flow_mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)

            # Simple motion blur: blend based on motion
            # More sophisticated would use flow-based warping
            blur_strength = np.clip(flow_mag / 20.0, 0, 1)

            from scipy.ndimage import gaussian_filter
            blurred = gaussian_filter(digital_image.astype(np.float64), sigma=2)

            # Blend based on motion
            result = (
                (1 - blur_strength) * digital_image +
                blur_strength * blurred
            )

            return result.astype(digital_image.dtype)

        except Exception:
            return digital_image

    def process_frame_by_frame(
        self,
        input_path: Union[str, Path],
    ):
        """Generator for frame-by-frame processing.

        Yields processed frames one at a time for custom handling.

        Args:
            input_path: Path to input video

        Yields:
            Tuple of (FrameResult, display_frame)
        """
        input_path = Path(input_path)

        start_frame = self.config.frame_range[0] if self.config.frame_range else 0
        end_frame = self.config.frame_range[1] if self.config.frame_range else None

        with VideoReader(input_path, start_frame=start_frame, end_frame=end_frame) as reader:
            previous_frame = None

            for frame_data in reader:
                result = self._process_frame(frame_data, previous_frame)

                display_frame = dn_to_display(
                    result.digital_image,
                    colormap="inferno" if self.config.sensor_type in ("lwir", "mwir") else None,
                )

                yield result, display_frame

                previous_frame = frame_data.image


def process_video(
    input_path: Union[str, Path],
    config: Optional[VideoSimulationConfig] = None,
    output_path: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> VideoSimulationResult:
    """Process video through simulation pipeline.

    Convenience function for video processing.

    Args:
        input_path: Path to input video
        config: Simulation configuration
        output_path: Path for output video
        progress_callback: Progress callback

    Returns:
        VideoSimulationResult
    """
    processor = VideoProcessor(config)
    return processor.process(input_path, output_path, progress_callback)


def simulate_video_thermal(
    input_path: Union[str, Path],
    sensor_type: str = "lwir",
    output_path: Optional[Union[str, Path]] = None,
    ambient_temperature_k: float = 290.0,
) -> Path:
    """Simulate thermal view of visible video.

    Quick convenience function for thermal simulation.

    Args:
        input_path: Input video path
        sensor_type: Target sensor type
        output_path: Output path (auto-generated if None)
        ambient_temperature_k: Ambient temperature

    Returns:
        Path to output video
    """
    from eosim.video.config import ThermalEstimateConfig

    config = VideoSimulationConfig(
        mode=VideoProcessingMode.THERMAL_ESTIMATE,
        sensor_type=sensor_type,
        thermal_config=ThermalEstimateConfig(
            ambient_temperature_k=ambient_temperature_k,
        ),
        output_path=Path(output_path) if output_path else None,
    )

    processor = VideoProcessor(config)
    result = processor.process(input_path, output_path)

    return result.output_path


def simulate_video_sensor(
    input_path: Union[str, Path],
    sensor_type: str = "lwir",
    output_path: Optional[Union[str, Path]] = None,
    radiance_scale: float = 1.0,
) -> Path:
    """Apply sensor model to video.

    Quick convenience function for radiance mapping.

    Args:
        input_path: Input video path
        sensor_type: Target sensor type
        output_path: Output path
        radiance_scale: Scale factor for radiance

    Returns:
        Path to output video
    """
    from eosim.video.config import RadianceMapConfig
    from eosim.video.radiance_mapper import estimate_radiance_range_for_sensor

    # Estimate appropriate radiance range
    rad_range = estimate_radiance_range_for_sensor(sensor_type)
    rad_range = (rad_range[0] * radiance_scale, rad_range[1] * radiance_scale)

    config = VideoSimulationConfig(
        mode=VideoProcessingMode.RADIANCE_MAP,
        sensor_type=sensor_type,
        radiance_config=RadianceMapConfig(
            output_radiance_range=rad_range,
        ),
        output_path=Path(output_path) if output_path else None,
    )

    processor = VideoProcessor(config)
    result = processor.process(input_path, output_path)

    return result.output_path


def simulate_video_template(
    input_path: Union[str, Path],
    sensor_type: str = "lwir",
    output_path: Optional[Union[str, Path]] = None,
    background_temperature_k: float = 290.0,
) -> Path:
    """Simulate video using template mode.

    Quick convenience function for template-based simulation.

    Args:
        input_path: Input video path
        sensor_type: Target sensor type
        output_path: Output path
        background_temperature_k: Background temperature

    Returns:
        Path to output video
    """
    from eosim.video.config import TemplateConfig

    config = VideoSimulationConfig(
        mode=VideoProcessingMode.TEMPLATE,
        sensor_type=sensor_type,
        template_config=TemplateConfig(
            background_temperature_k=background_temperature_k,
        ),
        output_path=Path(output_path) if output_path else None,
    )

    processor = VideoProcessor(config)
    result = processor.process(input_path, output_path)

    return result.output_path
