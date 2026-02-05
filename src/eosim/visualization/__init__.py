"""
EOSIM Visualization Module (Stage E).

Provides comprehensive visualization tools for thermal imagery analysis,
including colormap rendering, histogram analysis, and multi-panel displays.

Example 1: Render thermal image with colormap
    >>> from eosim.visualization import ThermalRenderer, ThermalColormap
    >>> import numpy as np
    >>> # Create thermal scene
    >>> scene = np.random.uniform(280, 320, (480, 640))
    >>> renderer = ThermalRenderer(colormap=ThermalColormap.IRONBOW)
    >>> rgb_image = renderer.render(scene)
    >>> renderer.save(rgb_image, "thermal_scene.png")

Example 2: Analyze scene with histogram
    >>> from eosim.visualization import SceneAnalyzer
    >>> analyzer = SceneAnalyzer()
    >>> fig = analyzer.histogram(scene, title="Temperature Distribution")
    >>> fig.savefig("histogram.png")
    >>> stats = analyzer.statistics(scene)
    >>> print(f"Mean: {stats['mean']:.1f}K, Contrast: {stats['contrast']:.2f}")

Example 3: Compare multiple images in multi-panel display
    >>> from eosim.visualization import MultiPanelDisplay
    >>> display = MultiPanelDisplay(rows=1, cols=3, figsize=(15, 5))
    >>> display.add_thermal(original, 0, 0, title="Original")
    >>> display.add_thermal(processed, 0, 1, title="Processed")
    >>> display.add_histogram(difference, 0, 2, title="Difference")
    >>> display.save("comparison.png")
"""

from eosim.visualization.thermal import (
    ThermalColormap,
    ThermalRenderer,
    apply_colormap,
    temperature_to_rgb,
)
from eosim.visualization.analysis import (
    SceneAnalyzer,
    ImageStatistics,
    ContrastMetrics,
    compute_histogram,
    compute_statistics,
)
from eosim.visualization.display import (
    MultiPanelDisplay,
    ProfilePlotter,
    create_comparison,
    create_colorbar,
)

__all__ = [
    # Thermal rendering
    "ThermalColormap",
    "ThermalRenderer",
    "apply_colormap",
    "temperature_to_rgb",
    # Analysis
    "SceneAnalyzer",
    "ImageStatistics",
    "ContrastMetrics",
    "compute_histogram",
    "compute_statistics",
    # Display
    "MultiPanelDisplay",
    "ProfilePlotter",
    "create_comparison",
    "create_colorbar",
]
