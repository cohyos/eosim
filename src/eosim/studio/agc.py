"""
Automatic Gain Control and Dynamic Range Processing.

This module provides image enhancement algorithms for thermal imagery:
- Automatic Gain Control (AGC)
- Histogram equalization
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Digital Detail Enhancement (DDE)
- Plateau histogram equalization
- Local area processing
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import ndimage


class AGCMode(Enum):
    """Automatic Gain Control modes."""
    LINEAR = "linear"           # Linear stretch min-max
    HISTOGRAM_EQ = "histogram"  # Global histogram equalization
    CLAHE = "clahe"            # Contrast Limited Adaptive HE
    PLATEAU = "plateau"         # Plateau histogram equalization
    MANUAL = "manual"          # Manual gain/level
    ONCE = "once"              # AGC once, then hold
    AUTO_BRIGHT = "auto_bright" # Automatic brightness tracking
    DDE = "dde"                # Digital Detail Enhancement
    ICE = "ice"                # Image Contrast Enhancement
    LOG = "log"                # Logarithmic compression


class PolarityMode(Enum):
    """Image polarity/colorization modes."""
    WHITE_HOT = "white_hot"
    BLACK_HOT = "black_hot"
    COLOR_IRONBOW = "ironbow"
    COLOR_RAINBOW = "rainbow"
    COLOR_SEPIA = "sepia"
    COLOR_ARCTIC = "arctic"
    COLOR_LAVA = "lava"
    COLOR_ISOTHERM = "isotherm"


@dataclass
class AGCParameters:
    """Parameters for AGC processing."""
    mode: AGCMode = AGCMode.LINEAR
    polarity: PolarityMode = PolarityMode.WHITE_HOT

    # Manual gain/level (for MANUAL mode)
    gain: float = 1.0          # Contrast multiplier
    level: float = 0.5         # Brightness offset (0-1)
    brightness: float = 0.0    # Additional brightness adjustment

    # Linear stretch parameters
    low_percentile: float = 1.0   # Low clip percentile
    high_percentile: float = 99.0  # High clip percentile

    # CLAHE parameters
    clip_limit: float = 2.0     # Contrast limit (1.0 = no limit)
    tile_size: int = 8          # Tile grid size

    # DDE parameters
    dde_gain: float = 1.5       # Detail enhancement gain
    dde_threshold: float = 0.1  # Noise threshold

    # Plateau equalization
    plateau_value: float = 0.95  # Max histogram bin value

    # Temporal filtering
    temporal_filter: float = 0.0  # 0-1, temporal smoothing coefficient

    # Region of interest for AGC calculation
    roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)


@dataclass
class AGCState:
    """Internal state for temporal AGC filtering."""
    min_val: float = 0.0
    max_val: float = 1.0
    mean_val: float = 0.5
    histogram: Optional[np.ndarray] = None
    lut: Optional[np.ndarray] = None


class AGCProcessor:
    """
    Automatic Gain Control processor for thermal imagery.

    Implements various algorithms for dynamic range compression
    and image enhancement.
    """

    def __init__(self, params: Optional[AGCParameters] = None):
        self.params = params or AGCParameters()
        self.state = AGCState()
        self._colormap_cache: Dict[str, np.ndarray] = {}
        self._init_colormaps()

    def _init_colormaps(self):
        """Initialize colormap lookup tables."""
        # Create 256-entry colormaps
        indices = np.linspace(0, 1, 256)

        # Ironbow (classic thermal)
        ironbow = np.zeros((256, 3), dtype=np.uint8)
        ironbow[:, 0] = np.clip(255 * (2 * indices - 0.5), 0, 255)  # R
        ironbow[:, 1] = np.clip(255 * (2 * indices - 1.0), 0, 255)  # G
        ironbow[:, 2] = np.clip(255 * (3 * indices), 0, 255)  # B
        ironbow[:64, 2] = np.linspace(0, 200, 64)  # Blue at cold
        self._colormap_cache['ironbow'] = ironbow

        # Rainbow
        rainbow = np.zeros((256, 3), dtype=np.uint8)
        for i, v in enumerate(indices):
            if v < 0.2:
                rainbow[i] = [0, 0, int(255 * v / 0.2)]
            elif v < 0.4:
                rainbow[i] = [0, int(255 * (v - 0.2) / 0.2), 255]
            elif v < 0.6:
                rainbow[i] = [0, 255, int(255 * (0.6 - v) / 0.2)]
            elif v < 0.8:
                rainbow[i] = [int(255 * (v - 0.6) / 0.2), 255, 0]
            else:
                rainbow[i] = [255, int(255 * (1.0 - v) / 0.2), 0]
        self._colormap_cache['rainbow'] = rainbow

        # Sepia (warm tones)
        sepia = np.zeros((256, 3), dtype=np.uint8)
        sepia[:, 0] = np.clip(indices * 255 * 1.2, 0, 255)
        sepia[:, 1] = np.clip(indices * 255 * 0.9, 0, 255)
        sepia[:, 2] = np.clip(indices * 255 * 0.6, 0, 255)
        self._colormap_cache['sepia'] = sepia

        # Arctic (cold tones)
        arctic = np.zeros((256, 3), dtype=np.uint8)
        arctic[:, 0] = np.clip(indices * 255 * 0.8, 0, 255)
        arctic[:, 1] = np.clip(indices * 255 * 0.95, 0, 255)
        arctic[:, 2] = np.clip(255 * (0.3 + indices * 0.7), 0, 255)
        self._colormap_cache['arctic'] = arctic

        # Lava (hot)
        lava = np.zeros((256, 3), dtype=np.uint8)
        lava[:, 0] = np.clip(indices * 255 * 1.5, 0, 255)
        lava[:, 1] = np.clip((indices - 0.3) * 255 * 1.5, 0, 255)
        lava[:, 2] = np.clip((indices - 0.7) * 255 * 2, 0, 255)
        self._colormap_cache['lava'] = lava

    def process(
        self,
        image: np.ndarray,
        params: Optional[AGCParameters] = None
    ) -> np.ndarray:
        """
        Apply AGC processing to an image.

        Args:
            image: Input image (grayscale, any dtype)
            params: Optional override parameters

        Returns:
            Processed 8-bit image (grayscale or color depending on polarity)
        """
        if params is not None:
            self.params = params

        # Convert to float for processing
        img_float = image.astype(np.float32)

        # Extract ROI for statistics if specified
        if self.params.roi is not None:
            x, y, w, h = self.params.roi
            roi_img = img_float[y:y+h, x:x+w]
        else:
            roi_img = img_float

        # Apply selected AGC mode
        if self.params.mode == AGCMode.LINEAR:
            result = self._linear_stretch(img_float, roi_img)
        elif self.params.mode == AGCMode.HISTOGRAM_EQ:
            result = self._histogram_equalization(img_float)
        elif self.params.mode == AGCMode.CLAHE:
            result = self._clahe(img_float)
        elif self.params.mode == AGCMode.PLATEAU:
            result = self._plateau_equalization(img_float)
        elif self.params.mode == AGCMode.MANUAL:
            result = self._manual_agc(img_float)
        elif self.params.mode == AGCMode.DDE:
            result = self._digital_detail_enhancement(img_float, roi_img)
        elif self.params.mode == AGCMode.ICE:
            result = self._ice(img_float, roi_img)
        elif self.params.mode == AGCMode.LOG:
            result = self._log_compression(img_float, roi_img)
        elif self.params.mode == AGCMode.ONCE:
            result = self._agc_once(img_float, roi_img)
        elif self.params.mode == AGCMode.AUTO_BRIGHT:
            result = self._auto_brightness(img_float, roi_img)
        else:
            result = self._linear_stretch(img_float, roi_img)

        # Apply brightness adjustment
        if self.params.brightness != 0:
            result = np.clip(result + self.params.brightness, 0, 1)

        # Convert to 8-bit
        result_8bit = (result * 255).astype(np.uint8)

        # Apply polarity/colorization
        result_8bit = self._apply_polarity(result_8bit)

        return result_8bit

    def _apply_temporal_filter(
        self,
        new_min: float,
        new_max: float
    ) -> Tuple[float, float]:
        """Apply temporal smoothing to AGC parameters."""
        alpha = self.params.temporal_filter
        if alpha > 0:
            self.state.min_val = alpha * self.state.min_val + (1 - alpha) * new_min
            self.state.max_val = alpha * self.state.max_val + (1 - alpha) * new_max
            return self.state.min_val, self.state.max_val
        return new_min, new_max

    def _linear_stretch(
        self,
        image: np.ndarray,
        roi_image: np.ndarray
    ) -> np.ndarray:
        """Linear percentile stretch."""
        low = np.percentile(roi_image, self.params.low_percentile)
        high = np.percentile(roi_image, self.params.high_percentile)

        low, high = self._apply_temporal_filter(low, high)

        if high <= low:
            high = low + 1

        result = (image - low) / (high - low)
        return np.clip(result, 0, 1)

    def _histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Global histogram equalization."""
        # Normalize to 0-255 for histogram
        img_min, img_max = image.min(), image.max()
        if img_max <= img_min:
            return np.zeros_like(image)

        normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)

        # Calculate histogram
        hist, bins = np.histogram(normalized.flatten(), 256, [0, 256])

        # Calculate CDF
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]

        # Apply equalization
        result = cdf_normalized[normalized]

        return result

    def _clahe(self, image: np.ndarray) -> np.ndarray:
        """Contrast Limited Adaptive Histogram Equalization."""
        # Normalize to 0-255
        img_min, img_max = image.min(), image.max()
        if img_max <= img_min:
            return np.zeros_like(image)

        normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)

        # Tile dimensions
        tile_size = self.params.tile_size
        h, w = image.shape[:2]
        tile_h = max(h // tile_size, 1)
        tile_w = max(w // tile_size, 1)

        # Process each tile
        result = np.zeros_like(image, dtype=np.float32)

        for ty in range(tile_size):
            for tx in range(tile_size):
                # Tile boundaries
                y1 = ty * tile_h
                y2 = min((ty + 1) * tile_h, h) if ty < tile_size - 1 else h
                x1 = tx * tile_w
                x2 = min((tx + 1) * tile_w, w) if tx < tile_size - 1 else w

                tile = normalized[y1:y2, x1:x2]

                # Calculate histogram for tile
                hist, _ = np.histogram(tile.flatten(), 256, [0, 256])

                # Clip histogram
                clip_limit = int(self.params.clip_limit *
                                tile.size / 256)
                excess = 0
                for i in range(256):
                    if hist[i] > clip_limit:
                        excess += hist[i] - clip_limit
                        hist[i] = clip_limit

                # Redistribute excess
                redistrib = excess // 256
                hist += redistrib

                # Calculate CDF
                cdf = hist.cumsum()
                if cdf[-1] > 0:
                    cdf_norm = cdf / cdf[-1]
                else:
                    cdf_norm = np.linspace(0, 1, 256)

                # Apply to tile
                result[y1:y2, x1:x2] = cdf_norm[tile]

        return result

    def _plateau_equalization(self, image: np.ndarray) -> np.ndarray:
        """Plateau histogram equalization."""
        # Normalize to 0-255
        img_min, img_max = image.min(), image.max()
        if img_max <= img_min:
            return np.zeros_like(image)

        normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)

        # Calculate histogram
        hist, _ = np.histogram(normalized.flatten(), 256, [0, 256])

        # Apply plateau
        plateau_count = int(self.params.plateau_value * hist.max())
        hist = np.minimum(hist, plateau_count)

        # Calculate CDF
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]

        # Apply
        result = cdf_normalized[normalized]

        return result

    def _manual_agc(self, image: np.ndarray) -> np.ndarray:
        """Manual gain and level control."""
        # Normalize
        img_min, img_max = image.min(), image.max()
        if img_max <= img_min:
            return np.full_like(image, self.params.level)

        normalized = (image - img_min) / (img_max - img_min)

        # Apply gain and level
        # level = 0.5 means centered, gain = 1.0 means no change
        result = (normalized - 0.5) * self.params.gain + self.params.level

        return np.clip(result, 0, 1)

    def _digital_detail_enhancement(
        self,
        image: np.ndarray,
        roi_image: np.ndarray
    ) -> np.ndarray:
        """
        Digital Detail Enhancement (DDE).

        Enhances local contrast while preserving global dynamic range.
        """
        # First apply linear stretch
        base = self._linear_stretch(image, roi_image)

        # Calculate local mean (low-frequency component)
        kernel_size = max(image.shape) // 20
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, min(kernel_size, 31))

        local_mean = ndimage.uniform_filter(base, size=kernel_size)

        # High-frequency component (details)
        details = base - local_mean

        # Apply noise threshold
        threshold = self.params.dde_threshold
        mask = np.abs(details) > threshold
        details_enhanced = np.where(mask,
                                    details * self.params.dde_gain,
                                    details)

        # Reconstruct
        result = local_mean + details_enhanced

        return np.clip(result, 0, 1)

    def _ice(
        self,
        image: np.ndarray,
        roi_image: np.ndarray
    ) -> np.ndarray:
        """
        Image Contrast Enhancement (ICE).

        Combines adaptive histogram equalization with edge preservation.
        """
        # CLAHE as base
        clahe_result = self._clahe(image)

        # Edge detection for preservation
        sobel_x = ndimage.sobel(image, axis=1)
        sobel_y = ndimage.sobel(image, axis=0)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = edges / (edges.max() + 1e-10)

        # Linear stretch for comparison
        linear_result = self._linear_stretch(image, roi_image)

        # Blend based on edges (preserve edges from linear)
        edge_weight = np.clip(edges * 2, 0, 1)
        result = clahe_result * (1 - edge_weight) + linear_result * edge_weight

        return np.clip(result, 0, 1)

    def _log_compression(
        self,
        image: np.ndarray,
        roi_image: np.ndarray
    ) -> np.ndarray:
        """Logarithmic dynamic range compression."""
        # Shift to positive
        img_min = image.min()
        shifted = image - img_min + 1

        # Log compression
        log_img = np.log(shifted)

        # Normalize
        log_min, log_max = log_img.min(), log_img.max()
        if log_max <= log_min:
            return np.zeros_like(image)

        result = (log_img - log_min) / (log_max - log_min)

        return result

    def _agc_once(
        self,
        image: np.ndarray,
        roi_image: np.ndarray
    ) -> np.ndarray:
        """AGC once and hold."""
        if self.state.lut is None:
            # First frame: calculate LUT
            result = self._linear_stretch(image, roi_image)
            # Store the mapping parameters
            low = np.percentile(roi_image, self.params.low_percentile)
            high = np.percentile(roi_image, self.params.high_percentile)
            self.state.min_val = low
            self.state.max_val = high
            return result
        else:
            # Subsequent frames: use stored parameters
            if self.state.max_val <= self.state.min_val:
                return np.zeros_like(image)
            result = (image - self.state.min_val) / (self.state.max_val - self.state.min_val)
            return np.clip(result, 0, 1)

    def _auto_brightness(
        self,
        image: np.ndarray,
        roi_image: np.ndarray
    ) -> np.ndarray:
        """Auto-brightness with scene tracking."""
        # Target mean brightness
        target_mean = 0.4

        # Current scene mean
        current_mean = roi_image.mean()

        # Linear stretch first
        result = self._linear_stretch(image, roi_image)

        # Adjust to target brightness
        result_mean = result.mean()
        if result_mean > 0:
            adjustment = target_mean / result_mean
            adjustment = np.clip(adjustment, 0.5, 2.0)
            result = result * adjustment

        return np.clip(result, 0, 1)

    def _apply_polarity(self, image: np.ndarray) -> np.ndarray:
        """Apply polarity and colorization."""
        polarity = self.params.polarity

        if polarity == PolarityMode.WHITE_HOT:
            return image

        elif polarity == PolarityMode.BLACK_HOT:
            return 255 - image

        elif polarity == PolarityMode.COLOR_IRONBOW:
            return self._colormap_cache['ironbow'][image]

        elif polarity == PolarityMode.COLOR_RAINBOW:
            return self._colormap_cache['rainbow'][image]

        elif polarity == PolarityMode.COLOR_SEPIA:
            return self._colormap_cache['sepia'][image]

        elif polarity == PolarityMode.COLOR_ARCTIC:
            return self._colormap_cache['arctic'][image]

        elif polarity == PolarityMode.COLOR_LAVA:
            return self._colormap_cache['lava'][image]

        elif polarity == PolarityMode.COLOR_ISOTHERM:
            return self._apply_isotherm(image)

        return image

    def _apply_isotherm(
        self,
        image: np.ndarray,
        hot_threshold: int = 200,
        cold_threshold: int = 50
    ) -> np.ndarray:
        """Apply isotherm coloring for hot/cold regions."""
        # Start with grayscale RGB
        result = np.stack([image, image, image], axis=-1)

        # Color hot regions red
        hot_mask = image > hot_threshold
        result[hot_mask, 0] = 255
        result[hot_mask, 1] = 0
        result[hot_mask, 2] = 0

        # Color cold regions blue
        cold_mask = image < cold_threshold
        result[cold_mask, 0] = 0
        result[cold_mask, 1] = 0
        result[cold_mask, 2] = 255

        return result

    def reset(self):
        """Reset AGC state."""
        self.state = AGCState()


class HistogramAnalyzer:
    """Histogram analysis utilities for AGC optimization."""

    @staticmethod
    def calculate_entropy(image: np.ndarray) -> float:
        """Calculate image entropy (information content)."""
        hist, _ = np.histogram(image.flatten(), 256, [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        return entropy

    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        """Calculate RMS contrast."""
        return float(np.std(image) / (np.mean(image) + 1e-10))

    @staticmethod
    def calculate_dynamic_range(image: np.ndarray, percentile: float = 1.0) -> float:
        """Calculate used dynamic range."""
        low = np.percentile(image, percentile)
        high = np.percentile(image, 100 - percentile)
        return float(high - low)

    @staticmethod
    def suggest_agc_mode(image: np.ndarray) -> AGCMode:
        """Suggest optimal AGC mode based on image characteristics."""
        # Analyze histogram
        hist, _ = np.histogram(image.flatten(), 256)

        # Check for bimodal distribution
        peaks = []
        for i in range(1, 255):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append((i, hist[i]))

        # High contrast scene with few dominant values
        if len(peaks) <= 3 and max(hist) > image.size * 0.1:
            return AGCMode.HISTOGRAM_EQ

        # Low contrast scene
        dynamic_range = HistogramAnalyzer.calculate_dynamic_range(image)
        if dynamic_range < 50:
            return AGCMode.CLAHE

        # High dynamic range scene
        if dynamic_range > 200:
            return AGCMode.DDE

        # Default
        return AGCMode.LINEAR


class NUCProcessor:
    """
    Non-Uniformity Correction processor.

    Corrects for detector non-uniformities using various methods.
    """

    def __init__(
        self,
        gain_map: Optional[np.ndarray] = None,
        offset_map: Optional[np.ndarray] = None
    ):
        self.gain_map = gain_map
        self.offset_map = offset_map

    def apply_two_point(self, image: np.ndarray) -> np.ndarray:
        """Apply two-point non-uniformity correction."""
        result = image.astype(np.float32)

        if self.offset_map is not None:
            result = result - self.offset_map

        if self.gain_map is not None:
            result = result * self.gain_map

        return result

    def calibrate_two_point(
        self,
        cold_frame: np.ndarray,
        hot_frame: np.ndarray,
        cold_temp: float,
        hot_temp: float
    ):
        """
        Calibrate two-point NUC from uniform temperature sources.

        Args:
            cold_frame: Frame from cold blackbody
            hot_frame: Frame from hot blackbody
            cold_temp: Cold source temperature
            hot_temp: Hot source temperature
        """
        # Calculate gain (responsivity)
        delta_signal = hot_frame - cold_frame
        delta_temp = hot_temp - cold_temp

        mean_delta = delta_signal.mean()
        if mean_delta != 0:
            self.gain_map = mean_delta / delta_signal
        else:
            self.gain_map = np.ones_like(cold_frame)

        # Calculate offset
        # After gain correction, offset should make cold_frame uniform
        corrected_cold = cold_frame * self.gain_map
        self.offset_map = corrected_cold - corrected_cold.mean()

    def apply_scene_based_nuc(
        self,
        image: np.ndarray,
        alpha: float = 0.01
    ) -> np.ndarray:
        """
        Apply scene-based NUC (SBNUC).

        Uses temporal filtering to estimate and remove fixed pattern noise.

        Args:
            image: Current frame
            alpha: Learning rate for offset update

        Returns:
            Corrected frame
        """
        if self.offset_map is None:
            self.offset_map = np.zeros_like(image, dtype=np.float32)

        # Spatial high-pass to estimate offset
        smoothed = ndimage.uniform_filter(image.astype(np.float32), size=5)
        residual = image - smoothed

        # Update offset map with temporal averaging
        self.offset_map = (1 - alpha) * self.offset_map + alpha * residual

        # Apply correction
        return image - self.offset_map


class BadPixelCorrector:
    """Bad pixel detection and replacement."""

    def __init__(self, bad_pixel_map: Optional[np.ndarray] = None):
        self.bad_pixel_map = bad_pixel_map

    def detect_bad_pixels(
        self,
        frames: List[np.ndarray],
        threshold_sigma: float = 5.0
    ) -> np.ndarray:
        """
        Detect bad pixels from multiple frames.

        Args:
            frames: List of frames for analysis
            threshold_sigma: Detection threshold in standard deviations

        Returns:
            Boolean bad pixel map
        """
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames for bad pixel detection")

        # Stack frames
        stack = np.stack(frames, axis=0)

        # Calculate temporal statistics
        temporal_mean = stack.mean(axis=0)
        temporal_std = stack.std(axis=0)

        # Spatial statistics for comparison
        spatial_mean = temporal_mean.mean()
        spatial_std = temporal_mean.std()

        # Detect dead pixels (always dark)
        dead = temporal_mean < (spatial_mean - threshold_sigma * spatial_std)

        # Detect hot pixels (always bright)
        hot = temporal_mean > (spatial_mean + threshold_sigma * spatial_std)

        # Detect noisy pixels (high temporal variance)
        noisy = temporal_std > threshold_sigma * temporal_std.mean()

        self.bad_pixel_map = dead | hot | noisy
        return self.bad_pixel_map

    def replace_bad_pixels(
        self,
        image: np.ndarray,
        method: str = "median"
    ) -> np.ndarray:
        """
        Replace bad pixels with interpolated values.

        Args:
            image: Input image
            method: Replacement method ("median", "mean", "bilinear")

        Returns:
            Corrected image
        """
        if self.bad_pixel_map is None:
            return image

        result = image.copy()

        if method == "median":
            # Median filter for replacement values
            filtered = ndimage.median_filter(image, size=3)
            result[self.bad_pixel_map] = filtered[self.bad_pixel_map]

        elif method == "mean":
            # Mean filter
            filtered = ndimage.uniform_filter(image.astype(np.float32), size=3)
            result[self.bad_pixel_map] = filtered[self.bad_pixel_map]

        elif method == "bilinear":
            # Bilinear interpolation from neighbors
            from scipy.interpolate import griddata
            h, w = image.shape[:2]
            y, x = np.mgrid[:h, :w]

            good_mask = ~self.bad_pixel_map
            good_points = np.column_stack((y[good_mask], x[good_mask]))
            good_values = image[good_mask]

            bad_points = np.column_stack((y[self.bad_pixel_map],
                                         x[self.bad_pixel_map]))

            if len(bad_points) > 0 and len(good_points) > 0:
                interpolated = griddata(good_points, good_values,
                                        bad_points, method='linear')
                result[self.bad_pixel_map] = interpolated

        return result
