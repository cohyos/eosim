"""
Thermal Image Rendering for EOSIM.

Provides colormaps and rendering tools for thermal/IR imagery visualization.

Example 1: Basic thermal rendering
    >>> from eosim.visualization import ThermalRenderer, ThermalColormap
    >>> renderer = ThermalRenderer(ThermalColormap.IRONBOW)
    >>> rgb = renderer.render(temperature_image, vmin=280, vmax=320)
    >>> renderer.save(rgb, "output.png")

Example 2: Apply colormap directly
    >>> from eosim.visualization import apply_colormap
    >>> rgb = apply_colormap(data, 'plasma', vmin=0, vmax=100)

Example 3: Convert temperatures to RGB with custom range
    >>> from eosim.visualization import temperature_to_rgb
    >>> rgb = temperature_to_rgb(temps, temp_min=250, temp_max=350, colormap='jet')
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray


class ThermalColormap(Enum):
    """Standard colormaps for thermal imagery.

    Attributes:
        GRAYSCALE: Linear grayscale (hot=white)
        INVERTED: Inverted grayscale (hot=black)
        IRONBOW: Classic thermal imaging colormap
        RAINBOW: Rainbow colormap
        PLASMA: Plasma colormap (perceptually uniform)
        INFERNO: Inferno colormap (perceptually uniform)
        VIRIDIS: Viridis colormap (perceptually uniform)
        HOT: Hot metal colormap
        JET: Jet colormap (legacy, not perceptually uniform)
        COOL_WARM: Diverging cool-to-warm
    """

    GRAYSCALE = "grayscale"
    INVERTED = "inverted"
    IRONBOW = "ironbow"
    RAINBOW = "rainbow"
    PLASMA = "plasma"
    INFERNO = "inferno"
    VIRIDIS = "viridis"
    HOT = "hot"
    JET = "jet"
    COOL_WARM = "cool_warm"


@dataclass
class ColorbarConfig:
    """Configuration for colorbar display.

    Attributes:
        show: Whether to show colorbar
        label: Colorbar label
        orientation: 'vertical' or 'horizontal'
        shrink: Fraction of original axes
        aspect: Aspect ratio of colorbar
    """

    show: bool = True
    label: str = "Temperature [K]"
    orientation: str = "vertical"
    shrink: float = 0.8
    aspect: int = 20


def _create_ironbow_lut() -> NDArray:
    """Create ironbow colormap lookup table.

    Returns:
        256x3 RGB lookup table
    """
    # Ironbow colormap control points (approximate)
    # Black -> Blue -> Magenta -> Red -> Yellow -> White
    control_points = np.array(
        [
            [0, 0, 0],  # 0: Black
            [0, 0, 128],  # 32: Dark blue
            [128, 0, 128],  # 64: Purple
            [128, 0, 0],  # 96: Dark red
            [255, 0, 0],  # 128: Red
            [255, 128, 0],  # 160: Orange
            [255, 255, 0],  # 192: Yellow
            [255, 255, 128],  # 224: Light yellow
            [255, 255, 255],  # 255: White
        ],
        dtype=np.float32,
    )

    positions = np.array([0, 32, 64, 96, 128, 160, 192, 224, 255])

    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(3):
        lut[:, i] = np.interp(np.arange(256), positions, control_points[:, i])

    return lut


def _create_rainbow_lut() -> NDArray:
    """Create rainbow colormap lookup table.

    Returns:
        256x3 RGB lookup table
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        if i < 51:  # Violet to Blue
            lut[i] = [128 - int(128 * i / 51), 0, 255]
        elif i < 102:  # Blue to Cyan
            lut[i] = [0, int(255 * (i - 51) / 51), 255]
        elif i < 153:  # Cyan to Green
            lut[i] = [0, 255, 255 - int(255 * (i - 102) / 51)]
        elif i < 204:  # Green to Yellow
            lut[i] = [int(255 * (i - 153) / 51), 255, 0]
        else:  # Yellow to Red
            lut[i] = [255, 255 - int(255 * (i - 204) / 51), 0]

    return lut


def _create_hot_lut() -> NDArray:
    """Create hot metal colormap lookup table.

    Returns:
        256x3 RGB lookup table
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        if i < 85:
            lut[i] = [int(255 * i / 85), 0, 0]
        elif i < 170:
            lut[i] = [255, int(255 * (i - 85) / 85), 0]
        else:
            lut[i] = [255, 255, int(255 * (i - 170) / 85)]

    return lut


def _create_cool_warm_lut() -> NDArray:
    """Create diverging cool-to-warm colormap.

    Returns:
        256x3 RGB lookup table
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.5:
            # Cool (blue) to white
            s = t * 2
            lut[i] = [int(59 + 196 * s), int(76 + 179 * s), int(192 + 63 * s)]
        else:
            # White to warm (red)
            s = (t - 0.5) * 2
            lut[i] = [255, int(255 - 179 * s), int(255 - 196 * s)]

    return lut


def _get_colormap_lut(colormap: Union[ThermalColormap, str]) -> NDArray:
    """Get lookup table for colormap.

    Args:
        colormap: Colormap enum or string name

    Returns:
        256x3 RGB lookup table
    """
    if isinstance(colormap, str):
        colormap = ThermalColormap(colormap.lower())

    if colormap == ThermalColormap.GRAYSCALE:
        return np.stack([np.arange(256)] * 3, axis=1).astype(np.uint8)

    elif colormap == ThermalColormap.INVERTED:
        return np.stack([255 - np.arange(256)] * 3, axis=1).astype(np.uint8)

    elif colormap == ThermalColormap.IRONBOW:
        return _create_ironbow_lut()

    elif colormap == ThermalColormap.RAINBOW:
        return _create_rainbow_lut()

    elif colormap == ThermalColormap.HOT:
        return _create_hot_lut()

    elif colormap == ThermalColormap.COOL_WARM:
        return _create_cool_warm_lut()

    else:
        # For matplotlib-based colormaps, try to import
        try:
            import matplotlib.pyplot as plt

            cmap = plt.get_cmap(colormap.value)
            lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
            return lut
        except ImportError:
            # Fallback to grayscale
            return np.stack([np.arange(256)] * 3, axis=1).astype(np.uint8)


def apply_colormap(
    data: NDArray,
    colormap: Union[ThermalColormap, str] = ThermalColormap.IRONBOW,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> NDArray:
    """Apply colormap to 2D data array.

    Args:
        data: 2D array of values
        colormap: Colormap to apply
        vmin: Minimum value for scaling (default: data min)
        vmax: Maximum value for scaling (default: data max)

    Returns:
        HxWx3 RGB array (uint8)
    """
    data = np.asarray(data, dtype=np.float64)

    if vmin is None:
        vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = float(np.nanmax(data))

    # Normalize to 0-255
    if vmax > vmin:
        normalized = (data - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(data)

    normalized = np.clip(normalized, 0, 1)
    indices = (normalized * 255).astype(np.uint8)

    # Apply LUT
    lut = _get_colormap_lut(colormap)
    rgb = lut[indices]

    return rgb


def temperature_to_rgb(
    temperature_k: NDArray,
    temp_min: float = 250.0,
    temp_max: float = 350.0,
    colormap: Union[ThermalColormap, str] = ThermalColormap.IRONBOW,
) -> NDArray:
    """Convert temperature map to RGB image.

    Args:
        temperature_k: Temperature array in Kelvin
        temp_min: Minimum temperature for colormap
        temp_max: Maximum temperature for colormap
        colormap: Colormap to use

    Returns:
        HxWx3 RGB array (uint8)
    """
    return apply_colormap(temperature_k, colormap, temp_min, temp_max)


class ThermalRenderer:
    """Renderer for thermal imagery with colormap support.

    Provides high-level interface for converting thermal data to
    displayable RGB images with optional annotations.
    """

    def __init__(
        self,
        colormap: Union[ThermalColormap, str] = ThermalColormap.IRONBOW,
        colorbar: Optional[ColorbarConfig] = None,
    ) -> None:
        """Initialize thermal renderer.

        Args:
            colormap: Colormap for rendering
            colorbar: Colorbar configuration (None to disable)
        """
        self._colormap = colormap
        self._colorbar = colorbar
        self._lut = _get_colormap_lut(colormap)

    @property
    def colormap(self) -> Union[ThermalColormap, str]:
        """Get current colormap."""
        return self._colormap

    @colormap.setter
    def colormap(self, value: Union[ThermalColormap, str]) -> None:
        """Set colormap."""
        self._colormap = value
        self._lut = _get_colormap_lut(value)

    def render(
        self,
        data: NDArray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> NDArray:
        """Render thermal data to RGB image.

        Args:
            data: 2D thermal data array
            vmin: Minimum value for scaling
            vmax: Maximum value for scaling

        Returns:
            HxWx3 RGB array (uint8)
        """
        return apply_colormap(data, self._colormap, vmin, vmax)

    def render_with_overlay(
        self,
        data: NDArray,
        mask: Optional[NDArray] = None,
        overlay_color: Tuple[int, int, int] = (255, 0, 0),
        overlay_alpha: float = 0.5,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> NDArray:
        """Render thermal data with optional mask overlay.

        Args:
            data: 2D thermal data array
            mask: Boolean mask for overlay (True = overlay)
            overlay_color: RGB color for overlay
            overlay_alpha: Transparency of overlay (0-1)
            vmin: Minimum value for scaling
            vmax: Maximum value for scaling

        Returns:
            HxWx3 RGB array (uint8)
        """
        rgb = self.render(data, vmin, vmax)

        if mask is not None and mask.any():
            overlay = np.array(overlay_color, dtype=np.float32)
            rgb_float = rgb.astype(np.float32)

            # Blend overlay where mask is True
            rgb_float[mask] = (
                (1 - overlay_alpha) * rgb_float[mask] + overlay_alpha * overlay
            )
            rgb = np.clip(rgb_float, 0, 255).astype(np.uint8)

        return rgb

    def render_difference(
        self,
        data1: NDArray,
        data2: NDArray,
        symmetric: bool = True,
    ) -> NDArray:
        """Render difference between two thermal images.

        Args:
            data1: First thermal image
            data2: Second thermal image
            symmetric: Use symmetric colormap centered at zero

        Returns:
            HxWx3 RGB array (uint8)
        """
        diff = data1.astype(np.float64) - data2.astype(np.float64)

        if symmetric:
            max_abs = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
            vmin, vmax = -max_abs, max_abs
            return apply_colormap(diff, ThermalColormap.COOL_WARM, vmin, vmax)
        else:
            return apply_colormap(diff, self._colormap)

    def create_colorbar_image(
        self,
        height: int = 256,
        width: int = 32,
        orientation: str = "vertical",
    ) -> NDArray:
        """Create standalone colorbar image.

        Args:
            height: Height in pixels
            width: Width in pixels
            orientation: 'vertical' or 'horizontal'

        Returns:
            HxWx3 RGB array (uint8)
        """
        if orientation == "vertical":
            gradient = np.linspace(1, 0, height).reshape(-1, 1)
            gradient = np.tile(gradient, (1, width))
        else:
            gradient = np.linspace(0, 1, width).reshape(1, -1)
            gradient = np.tile(gradient, (height, 1))

        indices = (gradient * 255).astype(np.uint8)
        return self._lut[indices]

    def save(
        self,
        rgb: NDArray,
        filepath: Union[str, Path],
        format: Optional[str] = None,
    ) -> None:
        """Save RGB image to file.

        Args:
            rgb: HxWx3 RGB array
            filepath: Output file path
            format: Image format (auto-detected if None)
        """
        filepath = Path(filepath)

        try:
            from PIL import Image

            img = Image.fromarray(rgb)
            img.save(filepath, format=format)
        except ImportError:
            # Fallback to matplotlib
            import matplotlib.pyplot as plt

            plt.imsave(str(filepath), rgb, format=format)

    def to_figure(
        self,
        data: NDArray,
        title: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        figsize: Tuple[float, float] = (10, 8),
    ):
        """Create matplotlib figure with thermal image and colorbar.

        Args:
            data: 2D thermal data array
            title: Figure title
            vmin: Minimum value for scaling
            vmax: Maximum value for scaling
            figsize: Figure size in inches

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        # Create custom colormap from LUT
        colors = self._lut / 255.0
        cmap = LinearSegmentedColormap.from_list("thermal", colors, N=256)

        fig, ax = plt.subplots(figsize=figsize)

        if vmin is None:
            vmin = float(np.nanmin(data))
        if vmax is None:
            vmax = float(np.nanmax(data))

        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

        if title:
            ax.set_title(title)

        if self._colorbar:
            cbar = fig.colorbar(
                im,
                ax=ax,
                orientation=self._colorbar.orientation,
                shrink=self._colorbar.shrink,
                aspect=self._colorbar.aspect,
            )
            cbar.set_label(self._colorbar.label)

        ax.set_xlabel("X [pixels]")
        ax.set_ylabel("Y [pixels]")

        plt.tight_layout()
        return fig
