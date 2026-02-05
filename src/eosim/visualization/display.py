"""
Multi-Panel Display and Comparison Tools for EOSIM.

Provides tools for creating multi-panel displays, profile plots,
and image comparisons for thermal imagery analysis.

Example 1: Create multi-panel display
    >>> from eosim.visualization import MultiPanelDisplay
    >>> display = MultiPanelDisplay(rows=2, cols=2, figsize=(12, 10))
    >>> display.add_thermal(image1, 0, 0, title="Original")
    >>> display.add_thermal(image2, 0, 1, title="Processed")
    >>> display.add_histogram(image1, 1, 0, title="Original Histogram")
    >>> display.add_profile(image1, 1, 1, axis=0, title="Horizontal Profile")
    >>> display.save("multi_panel.png")

Example 2: Profile plotting
    >>> from eosim.visualization import ProfilePlotter
    >>> plotter = ProfilePlotter()
    >>> fig = plotter.horizontal_profile(image, row=240)
    >>> fig = plotter.vertical_profile(image, col=320)
    >>> fig = plotter.cross_section(image, (100, 100), (400, 400))

Example 3: Quick comparison
    >>> from eosim.visualization import create_comparison
    >>> fig = create_comparison(original, processed, titles=["Before", "After"])
    >>> fig.savefig("comparison.png")
"""

from pathlib import Path
from typing import Optional, Tuple, List, Union
import numpy as np
from numpy.typing import NDArray

from .thermal import ThermalColormap, apply_colormap, ThermalRenderer
from .analysis import compute_statistics, compute_histogram


def create_colorbar(
    vmin: float,
    vmax: float,
    colormap: Union[ThermalColormap, str] = ThermalColormap.IRONBOW,
    orientation: str = "vertical",
    label: str = "Value",
    figsize: Tuple[float, float] = (1, 6),
):
    """Create standalone colorbar figure.

    Args:
        vmin: Minimum value
        vmax: Maximum value
        colormap: Colormap to use
        orientation: 'vertical' or 'horizontal'
        label: Colorbar label
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # Create colormap from LUT
    renderer = ThermalRenderer(colormap)
    colors = renderer._lut / 255.0

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("thermal", colors, N=256)

    fig, ax = plt.subplots(figsize=figsize)

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    if orientation == "vertical":
        cb = fig.colorbar(sm, cax=ax, orientation="vertical")
    else:
        cb = fig.colorbar(sm, cax=ax, orientation="horizontal")

    cb.set_label(label)
    plt.tight_layout()

    return fig


def create_comparison(
    image1: NDArray,
    image2: NDArray,
    titles: Optional[List[str]] = None,
    colormap: Union[ThermalColormap, str] = ThermalColormap.IRONBOW,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_difference: bool = True,
    figsize: Tuple[float, float] = (15, 5),
):
    """Create side-by-side comparison figure.

    Args:
        image1: First image
        image2: Second image
        titles: List of titles [img1, img2, diff]
        colormap: Colormap for thermal images
        vmin: Minimum value for scaling
        vmax: Maximum value for scaling
        show_difference: Show difference panel
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt

    if titles is None:
        titles = ["Image 1", "Image 2", "Difference"]

    n_panels = 3 if show_difference else 2

    # Determine common range
    if vmin is None:
        vmin = min(float(np.nanmin(image1)), float(np.nanmin(image2)))
    if vmax is None:
        vmax = max(float(np.nanmax(image1)), float(np.nanmax(image2)))

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    renderer = ThermalRenderer(colormap)

    # Image 1
    rgb1 = renderer.render(image1, vmin, vmax)
    axes[0].imshow(rgb1)
    axes[0].set_title(titles[0])
    axes[0].axis("off")

    # Image 2
    rgb2 = renderer.render(image2, vmin, vmax)
    axes[1].imshow(rgb2)
    axes[1].set_title(titles[1])
    axes[1].axis("off")

    # Difference
    if show_difference:
        diff = image1.astype(np.float64) - image2.astype(np.float64)
        rgb_diff = renderer.render_difference(image1, image2, symmetric=True)
        axes[2].imshow(rgb_diff)
        axes[2].set_title(f"{titles[2]} (max: {np.abs(diff).max():.2f})")
        axes[2].axis("off")

    plt.tight_layout()
    return fig


class MultiPanelDisplay:
    """Multi-panel display for thermal imagery analysis.

    Provides a flexible grid-based layout for combining different
    visualizations including thermal images, histograms, profiles,
    and statistics.
    """

    def __init__(
        self,
        rows: int = 2,
        cols: int = 2,
        figsize: Optional[Tuple[float, float]] = None,
        colormap: Union[ThermalColormap, str] = ThermalColormap.IRONBOW,
    ) -> None:
        """Initialize multi-panel display.

        Args:
            rows: Number of rows
            cols: Number of columns
            figsize: Figure size (auto-calculated if None)
            colormap: Default colormap for thermal images
        """
        import matplotlib.pyplot as plt

        self._rows = rows
        self._cols = cols
        self._colormap = colormap
        self._renderer = ThermalRenderer(colormap)

        if figsize is None:
            figsize = (5 * cols, 4 * rows)

        self._fig, self._axes = plt.subplots(rows, cols, figsize=figsize)

        # Ensure axes is always 2D array
        if rows == 1 and cols == 1:
            self._axes = np.array([[self._axes]])
        elif rows == 1:
            self._axes = self._axes.reshape(1, -1)
        elif cols == 1:
            self._axes = self._axes.reshape(-1, 1)

    @property
    def figure(self):
        """Get matplotlib figure."""
        return self._fig

    def _get_ax(self, row: int, col: int):
        """Get axis at position."""
        return self._axes[row, col]

    def add_thermal(
        self,
        data: NDArray,
        row: int,
        col: int,
        title: str = "",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = True,
        colorbar_label: str = "Value",
    ) -> None:
        """Add thermal image panel.

        Args:
            data: 2D image array
            row: Row position
            col: Column position
            title: Panel title
            vmin: Minimum value for scaling
            vmax: Maximum value for scaling
            colorbar: Show colorbar
            colorbar_label: Colorbar label
        """
        ax = self._get_ax(row, col)

        if vmin is None:
            vmin = float(np.nanmin(data))
        if vmax is None:
            vmax = float(np.nanmax(data))

        # Create colormap
        from matplotlib.colors import LinearSegmentedColormap

        colors = self._renderer._lut / 255.0
        cmap = LinearSegmentedColormap.from_list("thermal", colors, N=256)

        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

        if title:
            ax.set_title(title)

        if colorbar:
            self._fig.colorbar(im, ax=ax, label=colorbar_label, shrink=0.8)

    def add_rgb(
        self,
        rgb: NDArray,
        row: int,
        col: int,
        title: str = "",
    ) -> None:
        """Add pre-rendered RGB image panel.

        Args:
            rgb: HxWx3 RGB array
            row: Row position
            col: Column position
            title: Panel title
        """
        ax = self._get_ax(row, col)
        ax.imshow(rgb)
        if title:
            ax.set_title(title)
        ax.axis("off")

    def add_histogram(
        self,
        data: NDArray,
        row: int,
        col: int,
        title: str = "Histogram",
        bins: int = 100,
        color: str = "steelblue",
        show_stats: bool = True,
    ) -> None:
        """Add histogram panel.

        Args:
            data: Data array
            row: Row position
            col: Column position
            title: Panel title
            bins: Number of bins
            color: Histogram color
            show_stats: Show statistics box
        """
        ax = self._get_ax(row, col)

        counts, bin_edges = compute_histogram(data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], color=color)
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

        if show_stats:
            stats = compute_statistics(data)
            stats_text = f"μ={stats.mean:.2f}\nσ={stats.std:.2f}"
            ax.text(
                0.98,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    def add_profile(
        self,
        data: NDArray,
        row: int,
        col: int,
        axis: int = 0,
        position: Optional[int] = None,
        title: str = "Profile",
        color: str = "black",
    ) -> None:
        """Add profile plot panel.

        Args:
            data: 2D image array
            row: Row position
            col: Column position
            axis: Profile axis (0=horizontal, 1=vertical)
            position: Profile position (default: center)
            title: Panel title
            color: Line color
        """
        ax = self._get_ax(row, col)

        if position is None:
            position = data.shape[1 - axis] // 2

        if axis == 0:
            profile = data[position, :]
            xlabel = "Column"
            direction = "Row"
        else:
            profile = data[:, position]
            xlabel = "Row"
            direction = "Column"

        ax.plot(profile, color=color)
        ax.set_title(f"{title} ({direction} {position})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    def add_text(
        self,
        text: str,
        row: int,
        col: int,
        fontsize: int = 12,
    ) -> None:
        """Add text panel.

        Args:
            text: Text content
            row: Row position
            col: Column position
            fontsize: Font size
        """
        ax = self._get_ax(row, col)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            text,
            transform=ax.transAxes,
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=fontsize,
            fontfamily="monospace",
        )

    def add_statistics(
        self,
        data: NDArray,
        row: int,
        col: int,
        title: str = "Statistics",
    ) -> None:
        """Add statistics panel.

        Args:
            data: Data array
            row: Row position
            col: Column position
            title: Panel title
        """
        stats = compute_statistics(data)

        text = (
            f"{title}\n"
            f"{'='*20}\n"
            f"Mean: {stats.mean:.4f}\n"
            f"Std: {stats.std:.4f}\n"
            f"Min: {stats.min:.4f}\n"
            f"Max: {stats.max:.4f}\n"
            f"Median: {stats.median:.4f}\n"
            f"Dynamic Range: {stats.dynamic_range:.4f}\n"
            f"SNR: {stats.snr:.2f}"
        )

        self.add_text(text, row, col)

    def add_difference(
        self,
        data1: NDArray,
        data2: NDArray,
        row: int,
        col: int,
        title: str = "Difference",
        symmetric: bool = True,
        colorbar: bool = True,
    ) -> None:
        """Add difference image panel.

        Args:
            data1: First image
            data2: Second image
            row: Row position
            col: Column position
            title: Panel title
            symmetric: Use symmetric colormap
            colorbar: Show colorbar
        """
        ax = self._get_ax(row, col)

        diff = data1.astype(np.float64) - data2.astype(np.float64)

        if symmetric:
            max_abs = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
            vmin, vmax = -max_abs, max_abs
            cmap = "coolwarm"
        else:
            vmin, vmax = float(np.nanmin(diff)), float(np.nanmax(diff))
            cmap = "viridis"

        im = ax.imshow(diff, cmap=cmap, vmin=vmin, vmax=vmax)

        if title:
            ax.set_title(title)

        if colorbar:
            self._fig.colorbar(im, ax=ax, shrink=0.8)

    def tight_layout(self) -> None:
        """Adjust layout."""
        import matplotlib.pyplot as plt

        plt.tight_layout()

    def save(
        self,
        filepath: Union[str, Path],
        dpi: int = 150,
        format: Optional[str] = None,
    ) -> None:
        """Save figure to file.

        Args:
            filepath: Output file path
            dpi: Resolution
            format: Image format
        """
        self.tight_layout()
        self._fig.savefig(filepath, dpi=dpi, format=format, bbox_inches="tight")

    def show(self) -> None:
        """Display figure."""
        import matplotlib.pyplot as plt

        self.tight_layout()
        plt.show()


class ProfilePlotter:
    """Plot profiles and cross-sections of images.

    Provides methods for extracting and visualizing line profiles,
    cross-sections, and multi-line comparisons.
    """

    def __init__(
        self,
        figsize: Tuple[float, float] = (10, 4),
    ) -> None:
        """Initialize profile plotter.

        Args:
            figsize: Default figure size
        """
        self._figsize = figsize

    def horizontal_profile(
        self,
        data: NDArray,
        row: int,
        title: str = "Horizontal Profile",
        ylabel: str = "Value",
    ):
        """Plot horizontal profile at given row.

        Args:
            data: 2D image array
            row: Row index
            title: Figure title
            ylabel: Y-axis label

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self._figsize)

        profile = data[row, :]
        ax.plot(profile)
        ax.set_title(f"{title} (Row {row})")
        ax.set_xlabel("Column")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def vertical_profile(
        self,
        data: NDArray,
        col: int,
        title: str = "Vertical Profile",
        ylabel: str = "Value",
    ):
        """Plot vertical profile at given column.

        Args:
            data: 2D image array
            col: Column index
            title: Figure title
            ylabel: Y-axis label

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self._figsize)

        profile = data[:, col]
        ax.plot(profile)
        ax.set_title(f"{title} (Column {col})")
        ax.set_xlabel("Row")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def cross_section(
        self,
        data: NDArray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        n_points: Optional[int] = None,
        title: str = "Cross-Section",
        ylabel: str = "Value",
    ):
        """Plot cross-section along arbitrary line.

        Args:
            data: 2D image array
            start: Start point (row, col)
            end: End point (row, col)
            n_points: Number of sample points (default: line length)
            title: Figure title
            ylabel: Y-axis label

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        from scipy import ndimage

        # Calculate line length
        length = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

        if n_points is None:
            n_points = int(length)

        # Sample along line
        rows = np.linspace(start[0], end[0], n_points)
        cols = np.linspace(start[1], end[1], n_points)

        # Interpolate values
        profile = ndimage.map_coordinates(data, [rows, cols], order=1)

        # Distance along profile
        distances = np.linspace(0, length, n_points)

        fig, ax = plt.subplots(figsize=self._figsize)
        ax.plot(distances, profile)
        ax.set_title(f"{title} ({start} → {end})")
        ax.set_xlabel("Distance [pixels]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def multi_profile(
        self,
        data: NDArray,
        positions: List[int],
        axis: int = 0,
        title: str = "Multi-Profile Comparison",
        ylabel: str = "Value",
    ):
        """Plot multiple profiles on same axis.

        Args:
            data: 2D image array
            positions: List of row/column positions
            axis: 0 for horizontal profiles, 1 for vertical
            title: Figure title
            ylabel: Y-axis label

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self._figsize)

        direction = "Row" if axis == 0 else "Column"

        for pos in positions:
            if axis == 0:
                profile = data[pos, :]
            else:
                profile = data[:, pos]

            ax.plot(profile, label=f"{direction} {pos}")

        ax.set_title(title)
        ax.set_xlabel("Column" if axis == 0 else "Row")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def profile_with_image(
        self,
        data: NDArray,
        position: int,
        axis: int = 0,
        colormap: Union[ThermalColormap, str] = ThermalColormap.IRONBOW,
        figsize: Tuple[float, float] = (12, 5),
    ):
        """Plot profile alongside image with line indicator.

        Args:
            data: 2D image array
            position: Row/column position
            axis: 0 for horizontal profile, 1 for vertical
            colormap: Colormap for image
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Render image
        rgb = apply_colormap(data, colormap)
        axes[0].imshow(rgb)

        # Draw profile line
        if axis == 0:
            axes[0].axhline(position, color="white", linestyle="--", linewidth=2)
            profile = data[position, :]
            xlabel = "Column"
            axes[0].set_title(f"Image (Row {position})")
        else:
            axes[0].axvline(position, color="white", linestyle="--", linewidth=2)
            profile = data[:, position]
            xlabel = "Row"
            axes[0].set_title(f"Image (Column {position})")

        axes[0].axis("off")

        # Plot profile
        axes[1].plot(profile)
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel("Value")
        axes[1].set_title("Profile")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
