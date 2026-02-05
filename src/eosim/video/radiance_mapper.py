"""
Radiance mapping for video-to-simulation conversion (Option 2).

Maps video frame intensities directly to scene radiance values,
allowing sensor simulation to be applied to video content.
"""

from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from eosim.video.config import RadianceMapConfig, ChannelMode
from eosim.video.reader import to_grayscale
from eosim.pipeline.simulation import SceneInput


class RadianceMapper:
    """Maps video frame intensities to scene radiance.

    This is the simplest conversion mode (Option 2), directly translating
    pixel intensities to radiance values that can be processed through
    the sensor simulation pipeline.

    Supports various mapping functions:
    - Linear: Simple linear scaling
    - Gamma: Gamma-corrected scaling
    - Logarithmic: Log-space mapping for high dynamic range
    - Sigmoid: S-curve mapping for contrast enhancement

    Example:
        >>> config = RadianceMapConfig(
        ...     output_radiance_range=(0.001, 0.1),
        ...     gamma=2.2,
        ... )
        >>> mapper = RadianceMapper(config)
        >>> radiance_map = mapper.map_frame(video_frame)
        >>> scene = mapper.to_scene_input(video_frame)
    """

    def __init__(self, config: Optional[RadianceMapConfig] = None) -> None:
        """Initialize radiance mapper.

        Args:
            config: Radiance mapping configuration
        """
        self.config = config or RadianceMapConfig()
        self._global_min: Optional[float] = None
        self._global_max: Optional[float] = None

    def map_frame(
        self,
        frame: NDArray,
        normalize_range: Optional[tuple[float, float]] = None,
    ) -> NDArray:
        """Map frame intensities to radiance values.

        Args:
            frame: Input frame (H, W, C) or (H, W)
            normalize_range: Override normalization range (min, max)

        Returns:
            Radiance map in W/(m²·sr)
        """
        # Convert to grayscale if needed
        if frame.ndim == 3:
            gray = to_grayscale(frame, self.config.channel_mode)
        else:
            gray = frame.copy()

        # Convert to float
        intensity = gray.astype(np.float64)

        # Determine input range
        if normalize_range is not None:
            in_min, in_max = normalize_range
        elif self.config.normalization == "per_frame":
            in_min, in_max = intensity.min(), intensity.max()
        elif self.config.normalization == "global" and self._global_min is not None:
            in_min, in_max = self._global_min, self._global_max
        else:
            in_min, in_max = self.config.input_range

        # Normalize to 0-1
        if in_max > in_min:
            normalized = (intensity - in_min) / (in_max - in_min)
        else:
            normalized = np.zeros_like(intensity)

        normalized = np.clip(normalized, 0, 1)

        # Invert if requested
        if self.config.invert:
            normalized = 1.0 - normalized

        # Apply mapping function
        mapped = self._apply_mapping(normalized)

        # Scale to output radiance range
        out_min, out_max = self.config.output_radiance_range
        radiance = out_min + mapped * (out_max - out_min)

        return radiance

    def _apply_mapping(self, normalized: NDArray) -> NDArray:
        """Apply the configured mapping function.

        Args:
            normalized: Normalized intensity (0-1)

        Returns:
            Mapped values (0-1)
        """
        mapping_type = self.config.mapping_type

        if mapping_type == "linear":
            return normalized

        elif mapping_type == "gamma":
            # Gamma correction: output = input^gamma
            gamma = self.config.gamma
            return np.power(normalized, gamma)

        elif mapping_type == "logarithmic":
            # Logarithmic mapping for HDR-like effect
            # output = log(1 + k*input) / log(1 + k)
            k = 10.0  # Compression factor
            return np.log1p(k * normalized) / np.log1p(k)

        elif mapping_type == "sigmoid":
            # Sigmoid mapping for contrast enhancement
            # output = 1 / (1 + exp(-k*(input - 0.5)))
            k = 10.0  # Steepness
            return 1.0 / (1.0 + np.exp(-k * (normalized - 0.5)))

        else:
            raise ValueError(f"Unknown mapping type: {mapping_type}")

    def set_global_range(self, min_val: float, max_val: float) -> None:
        """Set global normalization range.

        Used when normalization="global" to use consistent scaling
        across all frames.

        Args:
            min_val: Minimum intensity value
            max_val: Maximum intensity value
        """
        self._global_min = min_val
        self._global_max = max_val

    def compute_global_range(self, frames: list[NDArray]) -> tuple[float, float]:
        """Compute global intensity range from frame list.

        Args:
            frames: List of video frames

        Returns:
            (min, max) intensity values
        """
        all_min = float('inf')
        all_max = float('-inf')

        for frame in frames:
            if frame.ndim == 3:
                gray = to_grayscale(frame, self.config.channel_mode)
            else:
                gray = frame

            all_min = min(all_min, gray.min())
            all_max = max(all_max, gray.max())

        self._global_min = all_min
        self._global_max = all_max

        return (all_min, all_max)

    def to_scene_input(
        self,
        frame: NDArray,
        normalize_range: Optional[tuple[float, float]] = None,
    ) -> SceneInput:
        """Convert frame to SceneInput for simulation pipeline.

        Args:
            frame: Input video frame
            normalize_range: Override normalization range

        Returns:
            SceneInput with radiance_map set
        """
        radiance_map = self.map_frame(frame, normalize_range)

        return SceneInput(radiance_map=radiance_map)


def intensity_to_radiance(
    intensity: NDArray,
    output_range: tuple[float, float] = (0.001, 0.1),
    input_range: tuple[float, float] = (0, 255),
    gamma: float = 1.0,
) -> NDArray:
    """Simple intensity to radiance conversion.

    Convenience function for quick conversion without full config.

    Args:
        intensity: Input intensity array
        output_range: Output radiance range in W/(m²·sr)
        input_range: Expected input range
        gamma: Gamma correction factor

    Returns:
        Radiance map in W/(m²·sr)
    """
    # Normalize to 0-1
    in_min, in_max = input_range
    if in_max > in_min:
        normalized = (intensity.astype(np.float64) - in_min) / (in_max - in_min)
    else:
        normalized = np.zeros_like(intensity, dtype=np.float64)

    normalized = np.clip(normalized, 0, 1)

    # Apply gamma
    if gamma != 1.0:
        normalized = np.power(normalized, gamma)

    # Scale to output range
    out_min, out_max = output_range
    radiance = out_min + normalized * (out_max - out_min)

    return radiance


def estimate_radiance_range_for_sensor(
    sensor_type: str,
    scene_temperature_k: float = 300.0,
) -> tuple[float, float]:
    """Estimate appropriate radiance range for a sensor type.

    Provides reasonable default radiance ranges based on typical
    scene conditions for different sensor types.

    Args:
        sensor_type: Sensor type ("lwir", "mwir", "swir", "visible")
        scene_temperature_k: Reference scene temperature

    Returns:
        (min, max) radiance range in W/(m²·sr)
    """
    sensor_type = sensor_type.lower()

    if sensor_type == "lwir":
        # LWIR 8-12 μm: ~20-60 W/(m²·sr) for 280-320K scenes
        from eosim.radiance.planck import planck_radiance_integrated
        L_low = planck_radiance_integrated(scene_temperature_k - 20, 8.0, 12.0)
        L_high = planck_radiance_integrated(scene_temperature_k + 30, 8.0, 12.0)
        return (float(L_low) * 0.8, float(L_high) * 1.2)

    elif sensor_type == "mwir":
        # MWIR 3-5 μm: ~0.5-5 W/(m²·sr) for typical scenes
        from eosim.radiance.planck import planck_radiance_integrated
        L_low = planck_radiance_integrated(scene_temperature_k - 20, 3.0, 5.0)
        L_high = planck_radiance_integrated(scene_temperature_k + 30, 3.0, 5.0)
        return (float(L_low) * 0.8, float(L_high) * 1.2)

    elif sensor_type == "swir":
        # SWIR 0.9-1.7 μm: Reflected light, typical 0.001-0.1 W/(m²·sr)
        return (0.001, 0.1)

    elif sensor_type == "visible":
        # Visible 0.4-0.7 μm: Wide range depending on lighting
        return (0.01, 100.0)

    else:
        # Default range
        return (0.001, 1.0)


class AdaptiveRadianceMapper(RadianceMapper):
    """Radiance mapper with adaptive histogram equalization.

    Provides better contrast in challenging scenes by adapting
    the mapping based on local image statistics.
    """

    def __init__(
        self,
        config: Optional[RadianceMapConfig] = None,
        clip_limit: float = 2.0,
        tile_size: tuple[int, int] = (8, 8),
    ) -> None:
        """Initialize adaptive mapper.

        Args:
            config: Base radiance mapping configuration
            clip_limit: CLAHE clip limit
            tile_size: CLAHE tile grid size
        """
        super().__init__(config)
        self.clip_limit = clip_limit
        self.tile_size = tile_size

    def map_frame(
        self,
        frame: NDArray,
        normalize_range: Optional[tuple[float, float]] = None,
    ) -> NDArray:
        """Map frame with adaptive histogram equalization.

        Args:
            frame: Input frame
            normalize_range: Override normalization range

        Returns:
            Radiance map with enhanced contrast
        """
        # Convert to grayscale
        if frame.ndim == 3:
            gray = to_grayscale(frame, self.config.channel_mode)
        else:
            gray = frame.copy()

        # Apply CLAHE for adaptive contrast
        try:
            import cv2

            # Ensure uint8 for CLAHE
            if gray.dtype != np.uint8:
                if gray.max() <= 1.0:
                    gray_u8 = (gray * 255).astype(np.uint8)
                else:
                    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
            else:
                gray_u8 = gray

            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_size,
            )
            enhanced = clahe.apply(gray_u8).astype(np.float64)

        except ImportError:
            # Fallback to simple histogram equalization
            enhanced = self._simple_histeq(gray.astype(np.float64))

        # Now apply standard mapping
        intensity = enhanced

        # Determine input range (after enhancement, typically 0-255)
        if normalize_range is not None:
            in_min, in_max = normalize_range
        else:
            in_min, in_max = 0, 255

        # Normalize to 0-1
        if in_max > in_min:
            normalized = (intensity - in_min) / (in_max - in_min)
        else:
            normalized = np.zeros_like(intensity)

        normalized = np.clip(normalized, 0, 1)

        if self.config.invert:
            normalized = 1.0 - normalized

        # Apply mapping function
        mapped = self._apply_mapping(normalized)

        # Scale to output radiance range
        out_min, out_max = self.config.output_radiance_range
        radiance = out_min + mapped * (out_max - out_min)

        return radiance

    def _simple_histeq(self, image: NDArray) -> NDArray:
        """Simple histogram equalization fallback."""
        # Flatten and compute histogram
        flat = image.flatten()
        hist, bins = np.histogram(flat, bins=256, range=(0, 256))

        # Compute CDF
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]

        # Apply equalization
        equalized = np.interp(flat, bins[:-1], cdf_normalized)

        return equalized.reshape(image.shape)
