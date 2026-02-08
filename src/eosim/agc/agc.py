"""
Automatic Gain Control (AGC) for electro-optical sensor imagery.

Provides multiple AGC algorithms for optimizing image display contrast
and dynamic range compression, including:
- Linear stretch
- Histogram equalization
- Contrast-Limited Adaptive Histogram Equalization (CLAHE)
- Plateau equalization
- Detail/Digital Detail Enhancement (DDE)
- Information Content Enhancement (ICE)
- Logarithmic compression

Also provides polarity modes (white-hot, black-hot, color palettes)
for thermal imagery display.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


class AGCMode(Enum):
    """Available AGC algorithm modes."""

    LINEAR = "linear"
    HISTOGRAM_EQ = "histogram_eq"
    CLAHE = "clahe"
    PLATEAU = "plateau"
    DDE = "dde"
    ICE = "ice"
    LOGARITHMIC = "logarithmic"
    MANUAL = "manual"


class Polarity(Enum):
    """Image polarity / display palette modes."""

    WHITE_HOT = "white_hot"
    BLACK_HOT = "black_hot"
    IRONBOW = "ironbow"
    RAINBOW = "rainbow"
    LAVA = "lava"
    ARCTIC = "arctic"
    ISOTHERM = "isotherm"
    SEPIA = "sepia"


@dataclass
class AGCParameters:
    """Configuration parameters for AGC processing.

    Attributes:
        mode: AGC algorithm to use
        linear_percent: Percentage of pixels to clip for linear stretch (each tail)
        clahe_clip_limit: Contrast limit for CLAHE (higher = more contrast)
        clahe_grid_size: Tile grid size for CLAHE (rows, cols)
        plateau_level: Histogram plateau level for plateau equalization (0-1)
        dde_strength: DDE detail enhancement strength (0-1)
        dde_sigma_spatial: Spatial sigma for DDE bilateral filter
        ice_iterations: Number of ICE iterations
        ice_alpha: ICE blending factor (0-1)
        output_bits: Output bit depth
        manual_min: Manual mode minimum value
        manual_max: Manual mode maximum value
    """

    mode: AGCMode = AGCMode.HISTOGRAM_EQ
    linear_percent: float = 1.0
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    plateau_level: float = 0.01
    dde_strength: float = 0.5
    dde_sigma_spatial: float = 3.0
    ice_iterations: int = 3
    ice_alpha: float = 0.3
    output_bits: int = 8
    manual_min: float = 0.0
    manual_max: float = 1.0

    def __post_init__(self) -> None:
        if self.linear_percent < 0 or self.linear_percent > 50:
            raise ValueError("linear_percent must be in [0, 50]")
        if self.clahe_clip_limit < 1.0:
            raise ValueError("clahe_clip_limit must be >= 1.0")
        if self.plateau_level < 0 or self.plateau_level > 1:
            raise ValueError("plateau_level must be in [0, 1]")
        if self.dde_strength < 0 or self.dde_strength > 1:
            raise ValueError("dde_strength must be in [0, 1]")
        if self.output_bits < 1 or self.output_bits > 16:
            raise ValueError("output_bits must be in [1, 16]")

    @property
    def output_max(self) -> int:
        """Maximum output digital number."""
        return (1 << self.output_bits) - 1


@dataclass
class AGCState:
    """Tracking state for temporal AGC filtering.

    Attributes:
        filtered_min: Temporally filtered minimum value
        filtered_max: Temporally filtered maximum value
        histogram: Running histogram estimate
        frame_count: Number of frames processed
    """

    filtered_min: float = 0.0
    filtered_max: float = 1.0
    histogram: Optional[NDArray] = None
    frame_count: int = 0


class AGCProcessor:
    """Automatic Gain Control processor for sensor imagery.

    Applies configurable AGC algorithms to convert raw sensor data
    (typically 14-bit) to display-ready imagery (typically 8-bit).

    Example:
        >>> from eosim.agc import AGCProcessor, AGCParameters, AGCMode
        >>> params = AGCParameters(mode=AGCMode.CLAHE, clahe_clip_limit=3.0)
        >>> agc = AGCProcessor(params)
        >>> display_image = agc.process(raw_image)
    """

    def __init__(
        self,
        params: Optional[AGCParameters] = None,
        temporal_filter: float = 0.0,
    ) -> None:
        """Initialize AGC processor.

        Args:
            params: AGC configuration parameters
            temporal_filter: Temporal smoothing factor (0 = no smoothing, 0.99 = heavy)
        """
        self.params = params or AGCParameters()
        self.temporal_filter = temporal_filter
        self._state = AGCState()

    @property
    def state(self) -> AGCState:
        """Current AGC tracking state."""
        return self._state

    def process(self, image: NDArray) -> NDArray:
        """Apply AGC to image.

        Args:
            image: Input image (any numeric type, 2D)

        Returns:
            AGC-processed image scaled to output range [0, output_max]
        """
        img = np.asarray(image, dtype=np.float64)

        if img.ndim != 2:
            raise ValueError(f"Expected 2D image, got {img.ndim}D")

        mode = self.params.mode

        if mode == AGCMode.LINEAR:
            result = self._linear_stretch(img)
        elif mode == AGCMode.HISTOGRAM_EQ:
            result = self._histogram_equalization(img)
        elif mode == AGCMode.CLAHE:
            result = self._clahe(img)
        elif mode == AGCMode.PLATEAU:
            result = self._plateau_equalization(img)
        elif mode == AGCMode.DDE:
            result = self._dde(img)
        elif mode == AGCMode.ICE:
            result = self._ice(img)
        elif mode == AGCMode.LOGARITHMIC:
            result = self._logarithmic(img)
        elif mode == AGCMode.MANUAL:
            result = self._manual(img)
        else:
            raise ValueError(f"Unknown AGC mode: {mode}")

        self._state.frame_count += 1

        # Scale to output bit depth
        out_max = self.params.output_max
        result = np.clip(result * out_max, 0, out_max)

        return result.astype(np.float64)

    def _linear_stretch(self, img: NDArray) -> NDArray:
        """Linear contrast stretch with percentile clipping.

        Args:
            img: Input image

        Returns:
            Normalized image in [0, 1]
        """
        pct = self.params.linear_percent
        low = np.percentile(img, pct)
        high = np.percentile(img, 100 - pct)

        # Apply temporal filtering
        low, high = self._temporal_filter_bounds(low, high)

        if high <= low:
            return np.zeros_like(img)

        result = (img - low) / (high - low)
        return np.clip(result, 0, 1)

    def _histogram_equalization(self, img: NDArray) -> NDArray:
        """Global histogram equalization.

        Args:
            img: Input image

        Returns:
            Equalized image in [0, 1]
        """
        # Normalize to [0, 1] range first
        imin, imax = img.min(), img.max()
        if imax <= imin:
            return np.zeros_like(img)

        normalized = (img - imin) / (imax - imin)

        # Compute histogram and CDF
        nbins = 256
        hist, bin_edges = np.histogram(normalized, bins=nbins, range=(0, 1))
        cdf = hist.cumsum().astype(np.float64)
        cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
        total = cdf[-1]

        if total <= cdf_min:
            return normalized

        cdf_normalized = (cdf - cdf_min) / (total - cdf_min)

        # Map pixels through CDF
        indices = np.clip(
            (normalized * (nbins - 1)).astype(int), 0, nbins - 1
        )
        result = cdf_normalized[indices]

        return result

    def _clahe(self, img: NDArray) -> NDArray:
        """Contrast-Limited Adaptive Histogram Equalization.

        Divides image into tiles and applies histogram equalization
        with clipping to limit contrast amplification.

        Args:
            img: Input image

        Returns:
            CLAHE-processed image in [0, 1]
        """
        h, w = img.shape
        grid_h, grid_w = self.params.clahe_grid_size
        clip_limit = self.params.clahe_clip_limit

        # Normalize input
        imin, imax = img.min(), img.max()
        if imax <= imin:
            return np.zeros_like(img)

        normalized = (img - imin) / (imax - imin)

        nbins = 256
        tile_h = max(h // grid_h, 1)
        tile_w = max(w // grid_w, 1)

        # Compute mapping for each tile
        mappings = np.zeros((grid_h, grid_w, nbins))

        for i in range(grid_h):
            for j in range(grid_w):
                r0 = i * tile_h
                r1 = min((i + 1) * tile_h, h)
                c0 = j * tile_w
                c1 = min((j + 1) * tile_w, w)

                tile = normalized[r0:r1, c0:c1]
                n_pixels = tile.size

                if n_pixels == 0:
                    mappings[i, j] = np.linspace(0, 1, nbins)
                    continue

                # Compute tile histogram
                hist, _ = np.histogram(tile, bins=nbins, range=(0, 1))
                hist = hist.astype(np.float64)

                # Clip histogram
                clip_threshold = max(clip_limit * n_pixels / nbins, 1)
                excess = np.sum(np.maximum(hist - clip_threshold, 0))
                hist = np.minimum(hist, clip_threshold)
                hist += excess / nbins  # redistribute clipped counts

                # Compute CDF
                cdf = hist.cumsum()
                cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
                cdf_max = cdf[-1]

                if cdf_max <= cdf_min:
                    mappings[i, j] = np.linspace(0, 1, nbins)
                else:
                    mappings[i, j] = (cdf - cdf_min) / (cdf_max - cdf_min)

        # Bilinear interpolation between tile mappings
        result = np.zeros_like(normalized)

        for r in range(h):
            for c in range(w):
                # Find surrounding tiles
                ti = min(r / tile_h - 0.5, grid_h - 1.001)
                tj = min(c / tile_w - 0.5, grid_w - 1.001)
                ti = max(ti, 0)
                tj = max(tj, 0)

                i0 = int(ti)
                j0 = int(tj)
                i1 = min(i0 + 1, grid_h - 1)
                j1 = min(j0 + 1, grid_w - 1)

                alpha = ti - i0
                beta = tj - j0

                # Bin index for this pixel
                bin_idx = min(int(normalized[r, c] * (nbins - 1)), nbins - 1)
                bin_idx = max(bin_idx, 0)

                # Bilinear interpolation
                val = (
                    (1 - alpha) * (1 - beta) * mappings[i0, j0, bin_idx]
                    + (1 - alpha) * beta * mappings[i0, j1, bin_idx]
                    + alpha * (1 - beta) * mappings[i1, j0, bin_idx]
                    + alpha * beta * mappings[i1, j1, bin_idx]
                )

                result[r, c] = val

        return np.clip(result, 0, 1)

    def _plateau_equalization(self, img: NDArray) -> NDArray:
        """Plateau histogram equalization.

        Clips the histogram at a plateau level before equalization
        to prevent over-enhancement of dominant gray levels.

        Args:
            img: Input image

        Returns:
            Plateau-equalized image in [0, 1]
        """
        imin, imax = img.min(), img.max()
        if imax <= imin:
            return np.zeros_like(img)

        normalized = (img - imin) / (imax - imin)

        nbins = 256
        hist, bin_edges = np.histogram(normalized, bins=nbins, range=(0, 1))
        hist = hist.astype(np.float64)

        # Apply plateau clipping
        plateau = self.params.plateau_level * img.size / nbins
        excess = np.sum(np.maximum(hist - plateau, 0))
        hist = np.minimum(hist, plateau)
        hist += excess / nbins

        # Compute CDF
        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
        cdf_max = cdf[-1]

        if cdf_max <= cdf_min:
            return normalized

        cdf_normalized = (cdf - cdf_min) / (cdf_max - cdf_min)

        indices = np.clip(
            (normalized * (nbins - 1)).astype(int), 0, nbins - 1
        )
        return cdf_normalized[indices]

    def _dde(self, img: NDArray) -> NDArray:
        """Digital Detail Enhancement.

        Separates image into low-frequency base and high-frequency detail,
        applies AGC to base and enhances detail.

        Args:
            img: Input image

        Returns:
            DDE-processed image in [0, 1]
        """
        from scipy.ndimage import gaussian_filter

        sigma = self.params.dde_sigma_spatial
        strength = self.params.dde_strength

        # Separate base and detail layers
        base = gaussian_filter(img, sigma=sigma)
        detail = img - base

        # AGC on base layer (linear stretch)
        bmin, bmax = base.min(), base.max()
        if bmax > bmin:
            base_normalized = (base - bmin) / (bmax - bmin)
        else:
            base_normalized = np.zeros_like(base)

        # Normalize detail
        detail_rms = np.std(detail)
        if detail_rms > 0:
            detail_normalized = detail / (6.0 * detail_rms)
        else:
            detail_normalized = np.zeros_like(detail)

        # Combine with strength control
        result = base_normalized + strength * detail_normalized

        return np.clip(result, 0, 1)

    def _ice(self, img: NDArray) -> NDArray:
        """Information Content Enhancement.

        Iteratively refines contrast to maximize information content
        using local mean and variance adaptation.

        Args:
            img: Input image

        Returns:
            ICE-processed image in [0, 1]
        """
        from scipy.ndimage import uniform_filter

        imin, imax = img.min(), img.max()
        if imax <= imin:
            return np.zeros_like(img)

        result = (img - imin) / (imax - imin)
        alpha = self.params.ice_alpha
        kernel_size = 31

        for _ in range(self.params.ice_iterations):
            local_mean = uniform_filter(result, size=kernel_size)
            local_var = uniform_filter(result**2, size=kernel_size) - local_mean**2
            local_var = np.maximum(local_var, 1e-10)
            local_std = np.sqrt(local_var)

            # Adaptive contrast: boost low-variance regions
            global_std = np.std(result)
            if global_std > 0:
                gain = global_std / local_std
                gain = np.clip(gain, 0.5, 3.0)
            else:
                gain = np.ones_like(local_std)

            enhanced = local_mean + gain * (result - local_mean)
            result = (1 - alpha) * result + alpha * enhanced
            result = np.clip(result, 0, 1)

        return result

    def _logarithmic(self, img: NDArray) -> NDArray:
        """Logarithmic dynamic range compression.

        Args:
            img: Input image

        Returns:
            Log-compressed image in [0, 1]
        """
        imin = img.min()
        shifted = img - imin + 1.0

        result = np.log(shifted)
        rmax = result.max()
        if rmax > 0:
            result = result / rmax
        return result

    def _manual(self, img: NDArray) -> NDArray:
        """Manual gain/level control.

        Args:
            img: Input image

        Returns:
            Manually scaled image in [0, 1]
        """
        low = self.params.manual_min
        high = self.params.manual_max

        if high <= low:
            return np.zeros_like(img)

        result = (img - low) / (high - low)
        return np.clip(result, 0, 1)

    def _temporal_filter_bounds(
        self, low: float, high: float
    ) -> Tuple[float, float]:
        """Apply temporal filtering to min/max bounds.

        Args:
            low: Current frame minimum
            high: Current frame maximum

        Returns:
            Filtered (low, high) bounds
        """
        alpha = self.temporal_filter
        if alpha > 0 and self._state.frame_count > 0:
            low = alpha * self._state.filtered_min + (1 - alpha) * low
            high = alpha * self._state.filtered_max + (1 - alpha) * high

        self._state.filtered_min = low
        self._state.filtered_max = high

        return low, high

    def reset(self) -> None:
        """Reset AGC tracking state."""
        self._state = AGCState()


class PolarityMapper:
    """Maps grayscale imagery to display palettes.

    Applies polarity and color mapping to AGC-processed imagery.

    Example:
        >>> from eosim.agc import PolarityMapper, Polarity
        >>> mapper = PolarityMapper(Polarity.IRONBOW)
        >>> rgb_image = mapper.apply(grayscale_image)
    """

    # Color palette lookup tables (256 entries, RGB)
    _PALETTES = {
        Polarity.IRONBOW: None,  # Generated in __init__
        Polarity.RAINBOW: None,
        Polarity.LAVA: None,
        Polarity.ARCTIC: None,
        Polarity.SEPIA: None,
    }

    def __init__(self, polarity: Polarity = Polarity.WHITE_HOT) -> None:
        """Initialize polarity mapper.

        Args:
            polarity: Display polarity / palette mode
        """
        self.polarity = polarity

    def apply(
        self,
        image: NDArray,
        output_bits: int = 8,
    ) -> NDArray:
        """Apply polarity mapping to grayscale image.

        Args:
            image: Input grayscale image (2D, values in [0, max_val])
            output_bits: Output bit depth for normalization

        Returns:
            RGB image (H, W, 3) as uint8, or grayscale (H, W) for
            white-hot/black-hot
        """
        img = np.asarray(image, dtype=np.float64)
        max_val = (1 << output_bits) - 1

        # Normalize to [0, 1]
        if max_val > 0:
            normalized = np.clip(img / max_val, 0, 1)
        else:
            normalized = np.zeros_like(img)

        if self.polarity == Polarity.WHITE_HOT:
            return (normalized * 255).astype(np.uint8)

        elif self.polarity == Polarity.BLACK_HOT:
            return ((1.0 - normalized) * 255).astype(np.uint8)

        elif self.polarity == Polarity.ISOTHERM:
            return self._apply_isotherm(normalized)

        else:
            lut = self._get_palette_lut(self.polarity)
            indices = np.clip((normalized * 255).astype(int), 0, 255)
            return lut[indices]

    def _apply_isotherm(self, normalized: NDArray) -> NDArray:
        """Apply isotherm coloring (highlight specific temperature bands).

        Args:
            normalized: Normalized [0, 1] grayscale image

        Returns:
            RGB image with isotherm bands highlighted
        """
        h, w = normalized.shape
        rgb = np.stack([normalized, normalized, normalized], axis=-1)
        rgb = (rgb * 255).astype(np.uint8)

        # Define isotherm bands
        bands = [
            (0.2, 0.3, [0, 0, 255]),    # Cold: blue
            (0.45, 0.55, [0, 255, 0]),   # Mid: green
            (0.7, 0.8, [255, 255, 0]),   # Warm: yellow
            (0.9, 1.0, [255, 0, 0]),     # Hot: red
        ]

        for low, high, color in bands:
            mask = (normalized >= low) & (normalized < high)
            for c in range(3):
                rgb[..., c][mask] = color[c]

        return rgb

    @staticmethod
    def _get_palette_lut(polarity: Polarity) -> NDArray:
        """Generate color lookup table for palette.

        Args:
            polarity: Palette type

        Returns:
            LUT array (256, 3) as uint8
        """
        x = np.linspace(0, 1, 256)
        lut = np.zeros((256, 3), dtype=np.uint8)

        if polarity == Polarity.IRONBOW:
            # Ironbow: black -> blue -> red -> yellow -> white
            lut[:, 0] = np.clip(255 * (3 * x - 1), 0, 255).astype(np.uint8)
            lut[:, 1] = np.clip(255 * (3 * x - 2), 0, 255).astype(np.uint8)
            lut[:, 2] = np.clip(
                255 * np.where(x < 0.5, 2 * x, 2 * (1 - x)), 0, 255
            ).astype(np.uint8)

        elif polarity == Polarity.RAINBOW:
            # HSV-like rainbow mapping
            for i, xi in enumerate(x):
                h = xi * 300  # Hue 0-300 (red to blue)
                s = 1.0
                v = 1.0
                r, g, b = _hsv_to_rgb(h, s, v)
                lut[i] = [int(r * 255), int(g * 255), int(b * 255)]

        elif polarity == Polarity.LAVA:
            # Black -> red -> orange -> yellow -> white
            lut[:, 0] = np.clip(255 * (2 * x), 0, 255).astype(np.uint8)
            lut[:, 1] = np.clip(255 * (2 * x - 0.5), 0, 255).astype(np.uint8)
            lut[:, 2] = np.clip(255 * (3 * x - 2), 0, 255).astype(np.uint8)

        elif polarity == Polarity.ARCTIC:
            # Black -> blue -> cyan -> white
            lut[:, 0] = np.clip(255 * (2 * x - 1), 0, 255).astype(np.uint8)
            lut[:, 1] = np.clip(255 * (2 * x - 0.5), 0, 255).astype(np.uint8)
            lut[:, 2] = np.clip(255 * (1.5 * x), 0, 255).astype(np.uint8)

        elif polarity == Polarity.SEPIA:
            # Warm brown tones
            lut[:, 0] = np.clip(255 * x * 1.1, 0, 255).astype(np.uint8)
            lut[:, 1] = np.clip(255 * x * 0.85, 0, 255).astype(np.uint8)
            lut[:, 2] = np.clip(255 * x * 0.65, 0, 255).astype(np.uint8)

        return lut


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """Convert HSV to RGB.

    Args:
        h: Hue in degrees (0-360)
        s: Saturation (0-1)
        v: Value (0-1)

    Returns:
        (r, g, b) tuple in [0, 1]
    """
    h = h % 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    return r + m, g + m, b + m


def apply_agc(
    image: NDArray,
    mode: AGCMode = AGCMode.HISTOGRAM_EQ,
    **kwargs,
) -> NDArray:
    """Convenience function to apply AGC to an image.

    Args:
        image: Input image (2D)
        mode: AGC algorithm to use
        **kwargs: Additional AGCParameters fields

    Returns:
        AGC-processed image
    """
    params = AGCParameters(mode=mode, **kwargs)
    processor = AGCProcessor(params)
    return processor.process(image)


def apply_polarity(
    image: NDArray,
    polarity: Polarity = Polarity.WHITE_HOT,
    input_bits: int = 8,
) -> NDArray:
    """Convenience function to apply polarity mapping.

    Args:
        image: Input grayscale image
        polarity: Display polarity mode
        input_bits: Input bit depth

    Returns:
        Polarity-mapped image
    """
    mapper = PolarityMapper(polarity)
    return mapper.apply(image, output_bits=input_bits)


def create_agc_processor(
    mode: Union[str, AGCMode] = AGCMode.HISTOGRAM_EQ,
    temporal_filter: float = 0.0,
    **kwargs,
) -> AGCProcessor:
    """Factory function for AGC processor.

    Args:
        mode: AGC mode (string or enum)
        temporal_filter: Temporal smoothing factor
        **kwargs: Additional AGCParameters fields

    Returns:
        Configured AGCProcessor
    """
    if isinstance(mode, str):
        mode = AGCMode(mode)

    params = AGCParameters(mode=mode, **kwargs)
    return AGCProcessor(params, temporal_filter=temporal_filter)
