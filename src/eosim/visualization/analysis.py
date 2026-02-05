"""
Scene Analysis and Statistics for EOSIM.

Provides tools for analyzing thermal scenes including histograms,
statistics, contrast metrics, and spatial analysis.

Example 1: Compute scene statistics
    >>> from eosim.visualization import compute_statistics
    >>> stats = compute_statistics(thermal_image)
    >>> print(f"Mean: {stats.mean:.2f}K, Std: {stats.std:.2f}K")

Example 2: Generate histogram figure
    >>> from eosim.visualization import SceneAnalyzer
    >>> analyzer = SceneAnalyzer()
    >>> fig = analyzer.histogram(image, bins=100, title="Temperature Distribution")
    >>> fig.savefig("histogram.png")

Example 3: Compute contrast metrics
    >>> from eosim.visualization import ContrastMetrics
    >>> metrics = ContrastMetrics.from_regions(target_region, background_region)
    >>> print(f"SCR: {metrics.scr:.2f}, Contrast: {metrics.contrast:.4f}")
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import numpy as np
from numpy.typing import NDArray


@dataclass
class ImageStatistics:
    """Statistical summary of an image.

    Attributes:
        mean: Mean value
        std: Standard deviation
        min: Minimum value
        max: Maximum value
        median: Median value
        p01: 1st percentile
        p99: 99th percentile
        dynamic_range: Difference between p99 and p01
        snr: Signal-to-noise ratio (mean/std)
    """

    mean: float
    std: float
    min: float
    max: float
    median: float
    p01: float
    p99: float
    dynamic_range: float
    snr: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "median": self.median,
            "p01": self.p01,
            "p99": self.p99,
            "dynamic_range": self.dynamic_range,
            "snr": self.snr,
        }


@dataclass
class ContrastMetrics:
    """Contrast metrics between target and background.

    Attributes:
        target_mean: Mean target intensity
        background_mean: Mean background intensity
        target_std: Target standard deviation
        background_std: Background standard deviation
        contrast: Weber contrast (target-bg)/bg
        michelson_contrast: Michelson contrast (max-min)/(max+min)
        scr: Signal-to-clutter ratio (target-bg)/bg_std
        cnr: Contrast-to-noise ratio
    """

    target_mean: float
    background_mean: float
    target_std: float
    background_std: float
    contrast: float
    michelson_contrast: float
    scr: float
    cnr: float

    @classmethod
    def from_regions(
        cls,
        target_region: NDArray,
        background_region: NDArray,
    ) -> "ContrastMetrics":
        """Compute contrast metrics from target and background regions.

        Args:
            target_region: Array of target pixel values
            background_region: Array of background pixel values

        Returns:
            ContrastMetrics instance
        """
        target = np.asarray(target_region).flatten()
        background = np.asarray(background_region).flatten()

        target_mean = float(np.nanmean(target))
        background_mean = float(np.nanmean(background))
        target_std = float(np.nanstd(target))
        background_std = float(np.nanstd(background))

        # Weber contrast
        if background_mean != 0:
            contrast = (target_mean - background_mean) / background_mean
        else:
            contrast = float("inf") if target_mean > 0 else 0.0

        # Michelson contrast
        max_val = max(target_mean, background_mean)
        min_val = min(target_mean, background_mean)
        if max_val + min_val != 0:
            michelson_contrast = (max_val - min_val) / (max_val + min_val)
        else:
            michelson_contrast = 0.0

        # Signal-to-clutter ratio
        if background_std != 0:
            scr = (target_mean - background_mean) / background_std
        else:
            scr = float("inf") if target_mean != background_mean else 0.0

        # Contrast-to-noise ratio
        combined_std = np.sqrt(target_std**2 + background_std**2)
        if combined_std != 0:
            cnr = abs(target_mean - background_mean) / combined_std
        else:
            cnr = float("inf") if target_mean != background_mean else 0.0

        return cls(
            target_mean=target_mean,
            background_mean=background_mean,
            target_std=target_std,
            background_std=background_std,
            contrast=contrast,
            michelson_contrast=michelson_contrast,
            scr=scr,
            cnr=cnr,
        )

    @classmethod
    def from_mask(
        cls,
        image: NDArray,
        target_mask: NDArray,
    ) -> "ContrastMetrics":
        """Compute contrast metrics using a target mask.

        Args:
            image: 2D image array
            target_mask: Boolean mask (True = target)

        Returns:
            ContrastMetrics instance
        """
        target_region = image[target_mask]
        background_region = image[~target_mask]
        return cls.from_regions(target_region, background_region)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "target_mean": self.target_mean,
            "background_mean": self.background_mean,
            "target_std": self.target_std,
            "background_std": self.background_std,
            "contrast": self.contrast,
            "michelson_contrast": self.michelson_contrast,
            "scr": self.scr,
            "cnr": self.cnr,
        }


def compute_histogram(
    data: NDArray,
    bins: int = 256,
    range: Optional[Tuple[float, float]] = None,
    density: bool = False,
) -> Tuple[NDArray, NDArray]:
    """Compute histogram of data.

    Args:
        data: Input array
        bins: Number of bins
        range: (min, max) range for histogram
        density: If True, normalize to probability density

    Returns:
        Tuple of (counts, bin_edges)
    """
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]

    if range is None:
        range = (float(data.min()), float(data.max()))

    counts, bin_edges = np.histogram(data, bins=bins, range=range, density=density)

    return counts, bin_edges


def compute_statistics(
    data: NDArray,
    mask: Optional[NDArray] = None,
) -> ImageStatistics:
    """Compute comprehensive statistics of image data.

    Args:
        data: Input array
        mask: Optional mask (True = include)

    Returns:
        ImageStatistics instance
    """
    data = np.asarray(data)

    if mask is not None:
        data = data[mask]

    data = data.flatten()
    data = data[~np.isnan(data)]

    if len(data) == 0:
        return ImageStatistics(
            mean=0, std=0, min=0, max=0, median=0, p01=0, p99=0, dynamic_range=0, snr=0
        )

    mean = float(np.mean(data))
    std = float(np.std(data))
    min_val = float(np.min(data))
    max_val = float(np.max(data))
    median = float(np.median(data))
    p01 = float(np.percentile(data, 1))
    p99 = float(np.percentile(data, 99))
    dynamic_range = p99 - p01
    snr = mean / std if std > 0 else float("inf")

    return ImageStatistics(
        mean=mean,
        std=std,
        min=min_val,
        max=max_val,
        median=median,
        p01=p01,
        p99=p99,
        dynamic_range=dynamic_range,
        snr=snr,
    )


class SceneAnalyzer:
    """Comprehensive scene analysis with visualization.

    Provides methods for analyzing thermal scenes including
    histograms, statistics, profiles, and spatial analysis.
    """

    def __init__(self, default_bins: int = 256) -> None:
        """Initialize scene analyzer.

        Args:
            default_bins: Default number of histogram bins
        """
        self._default_bins = default_bins

    def statistics(
        self,
        data: NDArray,
        mask: Optional[NDArray] = None,
    ) -> dict:
        """Compute scene statistics.

        Args:
            data: 2D image array
            mask: Optional mask (True = include)

        Returns:
            Dictionary of statistics
        """
        stats = compute_statistics(data, mask)
        return stats.to_dict()

    def histogram(
        self,
        data: NDArray,
        bins: Optional[int] = None,
        title: str = "Histogram",
        xlabel: str = "Value",
        ylabel: str = "Count",
        log_scale: bool = False,
        figsize: Tuple[float, float] = (10, 6),
        color: str = "steelblue",
        show_stats: bool = True,
    ):
        """Create histogram figure.

        Args:
            data: Input array
            bins: Number of bins (default: default_bins)
            title: Figure title
            xlabel: X-axis label
            ylabel: Y-axis label
            log_scale: Use logarithmic y-axis
            figsize: Figure size in inches
            color: Histogram color
            show_stats: Show statistics box

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        if bins is None:
            bins = self._default_bins

        counts, bin_edges = compute_histogram(data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], color=color)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if log_scale:
            ax.set_yscale("log")

        if show_stats:
            stats = compute_statistics(data)
            stats_text = (
                f"Mean: {stats.mean:.2f}\n"
                f"Std: {stats.std:.2f}\n"
                f"Min: {stats.min:.2f}\n"
                f"Max: {stats.max:.2f}"
            )
            ax.text(
                0.98,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=9,
            )

        plt.tight_layout()
        return fig

    def compare_histograms(
        self,
        data_list: List[NDArray],
        labels: List[str],
        bins: Optional[int] = None,
        title: str = "Histogram Comparison",
        xlabel: str = "Value",
        figsize: Tuple[float, float] = (10, 6),
        alpha: float = 0.6,
    ):
        """Compare histograms of multiple datasets.

        Args:
            data_list: List of data arrays
            labels: Labels for each dataset
            bins: Number of bins
            title: Figure title
            xlabel: X-axis label
            figsize: Figure size
            alpha: Histogram transparency

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        if bins is None:
            bins = self._default_bins

        fig, ax = plt.subplots(figsize=figsize)

        # Find common range
        all_data = np.concatenate([d.flatten() for d in data_list])
        range_min, range_max = float(np.nanmin(all_data)), float(np.nanmax(all_data))

        for data, label in zip(data_list, labels):
            counts, bin_edges = compute_histogram(
                data, bins=bins, range=(range_min, range_max)
            )
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.step(bin_centers, counts, label=label, alpha=alpha, where="mid")

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend()

        plt.tight_layout()
        return fig

    def spatial_profile(
        self,
        data: NDArray,
        axis: int = 0,
        position: Optional[int] = None,
        title: str = "Spatial Profile",
        xlabel: str = "Position [pixels]",
        ylabel: str = "Value",
        figsize: Tuple[float, float] = (10, 4),
    ):
        """Plot spatial profile along an axis.

        Args:
            data: 2D image array
            axis: Axis along which to profile (0=row, 1=column)
            position: Position of profile line (default: center)
            title: Figure title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        data = np.asarray(data)

        if position is None:
            position = data.shape[1 - axis] // 2

        if axis == 0:
            profile = data[position, :]
            x_label = "Column"
        else:
            profile = data[:, position]
            x_label = "Row"

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(profile)
        ax.set_title(f"{title} (at {x_label[:-1]} {position})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.tight_layout()
        return fig

    def contrast_analysis(
        self,
        image: NDArray,
        target_mask: NDArray,
        title: str = "Contrast Analysis",
        figsize: Tuple[float, float] = (12, 5),
    ):
        """Create contrast analysis figure with histograms and metrics.

        Args:
            image: 2D image array
            target_mask: Boolean target mask
            title: Figure title
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        # Compute metrics
        metrics = ContrastMetrics.from_mask(image, target_mask)

        # Extract regions
        target = image[target_mask].flatten()
        background = image[~target_mask].flatten()

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Histogram comparison
        ax1 = axes[0]
        all_data = np.concatenate([target, background])
        range_min, range_max = float(np.nanmin(all_data)), float(np.nanmax(all_data))

        ax1.hist(
            background,
            bins=50,
            range=(range_min, range_max),
            alpha=0.6,
            label="Background",
            color="blue",
        )
        ax1.hist(
            target,
            bins=50,
            range=(range_min, range_max),
            alpha=0.6,
            label="Target",
            color="red",
        )
        ax1.axvline(metrics.background_mean, color="blue", linestyle="--", linewidth=2)
        ax1.axvline(metrics.target_mean, color="red", linestyle="--", linewidth=2)
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Count")
        ax1.set_title("Distribution Comparison")
        ax1.legend()

        # Metrics display
        ax2 = axes[1]
        ax2.axis("off")

        metrics_text = (
            f"Contrast Metrics\n"
            f"{'='*30}\n"
            f"Target Mean: {metrics.target_mean:.2f}\n"
            f"Background Mean: {metrics.background_mean:.2f}\n"
            f"Target Std: {metrics.target_std:.2f}\n"
            f"Background Std: {metrics.background_std:.2f}\n"
            f"\n"
            f"Weber Contrast: {metrics.contrast:.4f}\n"
            f"Michelson Contrast: {metrics.michelson_contrast:.4f}\n"
            f"SCR: {metrics.scr:.2f}\n"
            f"CNR: {metrics.cnr:.2f}"
        )

        ax2.text(
            0.5,
            0.5,
            metrics_text,
            transform=ax2.transAxes,
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=12,
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        fig.suptitle(title)
        plt.tight_layout()
        return fig

    def power_spectrum(
        self,
        data: NDArray,
        title: str = "Power Spectrum",
        figsize: Tuple[float, float] = (8, 6),
        log_scale: bool = True,
    ):
        """Compute and display 2D power spectrum.

        Args:
            data: 2D image array
            title: Figure title
            figsize: Figure size
            log_scale: Use logarithmic color scale

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        data = np.asarray(data)

        # Compute 2D FFT and power spectrum
        fft = np.fft.fft2(data)
        fft_shifted = np.fft.fftshift(fft)
        power = np.abs(fft_shifted) ** 2

        if log_scale:
            power = np.log10(power + 1)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(power, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Frequency X")
        ax.set_ylabel("Frequency Y")
        plt.colorbar(im, ax=ax, label="Log Power" if log_scale else "Power")

        plt.tight_layout()
        return fig

    def radial_average(
        self,
        data: NDArray,
        title: str = "Radial Average",
        figsize: Tuple[float, float] = (8, 5),
    ):
        """Compute and plot radially averaged power spectrum.

        Args:
            data: 2D image array
            title: Figure title
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        data = np.asarray(data)
        h, w = data.shape

        # Compute power spectrum
        fft = np.fft.fft2(data)
        fft_shifted = np.fft.fftshift(fft)
        power = np.abs(fft_shifted) ** 2

        # Create radial coordinates
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        r = r.astype(int)

        # Compute radial average
        max_r = min(center_x, center_y)
        radial_mean = np.zeros(max_r)
        for i in range(max_r):
            mask = r == i
            if mask.any():
                radial_mean[i] = power[mask].mean()

        fig, ax = plt.subplots(figsize=figsize)
        ax.semilogy(radial_mean)
        ax.set_title(title)
        ax.set_xlabel("Spatial Frequency [cycles/image]")
        ax.set_ylabel("Power")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
