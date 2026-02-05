"""
Tests for EOSIM Visualization Module (Stage E).

Tests thermal rendering, scene analysis, and multi-panel display functionality.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from eosim.visualization import (
    ThermalColormap,
    ThermalRenderer,
    apply_colormap,
    temperature_to_rgb,
    SceneAnalyzer,
    ImageStatistics,
    ContrastMetrics,
    compute_histogram,
    compute_statistics,
    MultiPanelDisplay,
    ProfilePlotter,
    create_comparison,
    create_colorbar,
)


# =============================================================================
# Thermal Rendering Tests
# =============================================================================


class TestThermalColormap:
    """Test thermal colormap enumeration."""

    def test_colormap_values(self):
        """All colormaps have valid values."""
        for cmap in ThermalColormap:
            assert isinstance(cmap.value, str)

    def test_grayscale_value(self):
        """Grayscale has correct value."""
        assert ThermalColormap.GRAYSCALE.value == "grayscale"

    def test_ironbow_value(self):
        """Ironbow has correct value."""
        assert ThermalColormap.IRONBOW.value == "ironbow"


class TestApplyColormap:
    """Test colormap application function."""

    def test_apply_colormap_shape(self):
        """Output has correct shape."""
        data = np.random.randn(100, 100)
        rgb = apply_colormap(data)
        assert rgb.shape == (100, 100, 3)

    def test_apply_colormap_dtype(self):
        """Output has correct dtype."""
        data = np.random.randn(50, 50)
        rgb = apply_colormap(data)
        assert rgb.dtype == np.uint8

    def test_apply_colormap_range(self):
        """Output values are in valid range."""
        data = np.random.randn(50, 50)
        rgb = apply_colormap(data)
        assert rgb.min() >= 0
        assert rgb.max() <= 255

    def test_apply_colormap_grayscale(self):
        """Grayscale colormap works correctly."""
        data = np.array([[0, 0.5, 1.0]])
        rgb = apply_colormap(data, ThermalColormap.GRAYSCALE, vmin=0, vmax=1)

        # Check gradient from black to white
        assert rgb[0, 0, 0] == 0  # Black
        assert rgb[0, 2, 0] == 255  # White

    def test_apply_colormap_vmin_vmax(self):
        """Custom vmin/vmax work correctly."""
        data = np.array([[0, 50, 100]])
        rgb1 = apply_colormap(data, ThermalColormap.GRAYSCALE, vmin=0, vmax=100)
        rgb2 = apply_colormap(data, ThermalColormap.GRAYSCALE, vmin=0, vmax=200)

        # rgb2 should be darker at max value
        assert rgb2[0, 2, 0] < rgb1[0, 2, 0]

    def test_apply_colormap_constant_data(self):
        """Constant data doesn't cause division by zero."""
        data = np.ones((10, 10)) * 5
        rgb = apply_colormap(data)
        assert rgb.shape == (10, 10, 3)


class TestTemperatureToRGB:
    """Test temperature to RGB conversion."""

    def test_temperature_to_rgb_shape(self):
        """Output has correct shape."""
        temps = np.random.uniform(280, 320, (100, 100))
        rgb = temperature_to_rgb(temps)
        assert rgb.shape == (100, 100, 3)

    def test_temperature_to_rgb_defaults(self):
        """Default temperature range works."""
        temps = np.array([[250, 300, 350]])
        rgb = temperature_to_rgb(temps, temp_min=250, temp_max=350)
        assert rgb.shape == (1, 3, 3)


class TestThermalRenderer:
    """Test ThermalRenderer class."""

    @pytest.fixture
    def renderer(self):
        """Create renderer fixture."""
        return ThermalRenderer(ThermalColormap.IRONBOW)

    @pytest.fixture
    def sample_data(self):
        """Create sample thermal data."""
        return np.random.uniform(280, 320, (100, 100))

    def test_renderer_creation(self, renderer):
        """Renderer can be created."""
        assert renderer is not None
        assert renderer.colormap == ThermalColormap.IRONBOW

    def test_renderer_colormap_setter(self, renderer):
        """Colormap can be changed."""
        renderer.colormap = ThermalColormap.GRAYSCALE
        assert renderer.colormap == ThermalColormap.GRAYSCALE

    def test_render(self, renderer, sample_data):
        """Basic rendering works."""
        rgb = renderer.render(sample_data)
        assert rgb.shape == (100, 100, 3)
        assert rgb.dtype == np.uint8

    def test_render_with_vmin_vmax(self, renderer, sample_data):
        """Rendering with custom range works."""
        rgb = renderer.render(sample_data, vmin=270, vmax=330)
        assert rgb.shape == (100, 100, 3)

    def test_render_with_overlay(self, renderer, sample_data):
        """Overlay rendering works."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True

        rgb = renderer.render_with_overlay(sample_data, mask, overlay_color=(255, 0, 0))
        assert rgb.shape == (100, 100, 3)

        # Check that overlay region is reddish
        overlay_red = rgb[50, 50, 0]
        assert overlay_red > rgb[10, 10, 0]  # Overlay area has more red

    def test_render_difference(self, renderer):
        """Difference rendering works."""
        data1 = np.random.uniform(290, 310, (50, 50))
        data2 = data1 + np.random.randn(50, 50) * 5

        rgb = renderer.render_difference(data1, data2)
        assert rgb.shape == (50, 50, 3)

    def test_create_colorbar_image(self, renderer):
        """Colorbar image creation works."""
        cb = renderer.create_colorbar_image(height=256, width=32)
        assert cb.shape == (256, 32, 3)

    def test_create_colorbar_horizontal(self, renderer):
        """Horizontal colorbar works."""
        cb = renderer.create_colorbar_image(height=32, width=256, orientation="horizontal")
        assert cb.shape == (32, 256, 3)

    def test_to_figure(self, renderer, sample_data):
        """Figure creation works."""
        fig = renderer.to_figure(sample_data, title="Test")
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Analysis Tests
# =============================================================================


class TestImageStatistics:
    """Test ImageStatistics dataclass."""

    def test_statistics_creation(self):
        """Statistics can be created."""
        stats = ImageStatistics(
            mean=100,
            std=10,
            min=50,
            max=150,
            median=100,
            p01=60,
            p99=140,
            dynamic_range=80,
            snr=10,
        )
        assert stats.mean == 100

    def test_to_dict(self):
        """to_dict returns proper dictionary."""
        stats = ImageStatistics(
            mean=100, std=10, min=50, max=150, median=100, p01=60, p99=140, dynamic_range=80, snr=10
        )
        d = stats.to_dict()
        assert isinstance(d, dict)
        assert d["mean"] == 100


class TestComputeStatistics:
    """Test compute_statistics function."""

    def test_basic_statistics(self):
        """Basic statistics computation works."""
        data = np.array([1, 2, 3, 4, 5])
        stats = compute_statistics(data)

        assert stats.mean == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.median == 3.0

    def test_2d_data(self):
        """2D data is handled correctly."""
        data = np.random.randn(100, 100)
        stats = compute_statistics(data)

        assert_allclose(stats.mean, data.mean(), rtol=1e-10)
        assert_allclose(stats.std, data.std(), rtol=1e-10)

    def test_with_mask(self):
        """Mask is applied correctly."""
        data = np.array([[1, 2], [100, 200]])
        mask = np.array([[True, True], [False, False]])

        stats = compute_statistics(data, mask)
        assert stats.mean == 1.5  # Only 1 and 2 are included

    def test_snr_computation(self):
        """SNR is computed correctly."""
        data = np.ones((100,)) * 10 + np.random.randn(100) * 0.1
        stats = compute_statistics(data)

        # SNR should be approximately 100 (mean/std ≈ 10/0.1)
        assert stats.snr > 50


class TestComputeHistogram:
    """Test compute_histogram function."""

    def test_histogram_shape(self):
        """Histogram has correct shape."""
        data = np.random.randn(1000)
        counts, edges = compute_histogram(data, bins=100)

        assert len(counts) == 100
        assert len(edges) == 101

    def test_histogram_sum(self):
        """Histogram sums to total count."""
        data = np.random.randn(1000)
        counts, _ = compute_histogram(data, bins=50)

        assert counts.sum() == 1000

    def test_histogram_range(self):
        """Custom range works."""
        data = np.random.randn(1000)
        counts, edges = compute_histogram(data, bins=50, range=(-2, 2))

        assert edges[0] == -2
        assert edges[-1] == 2


class TestContrastMetrics:
    """Test ContrastMetrics class."""

    def test_from_regions(self):
        """from_regions creates valid metrics."""
        target = np.ones((10, 10)) * 100
        background = np.ones((50, 50)) * 50 + np.random.randn(50, 50) * 5

        metrics = ContrastMetrics.from_regions(target, background)

        assert metrics.target_mean == 100
        assert_allclose(metrics.background_mean, 50, rtol=0.1)
        assert metrics.contrast > 0  # Target is brighter

    def test_from_mask(self):
        """from_mask creates valid metrics."""
        image = np.ones((100, 100)) * 50
        image[40:60, 40:60] = 100

        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True

        metrics = ContrastMetrics.from_mask(image, mask)

        assert metrics.target_mean == 100
        assert metrics.background_mean == 50

    def test_scr_computation(self):
        """SCR is computed correctly."""
        target = np.ones((10,)) * 100
        background = np.ones((100,)) * 50 + np.random.randn(100) * 10

        metrics = ContrastMetrics.from_regions(target, background)

        # SCR = (target - bg) / bg_std ≈ 50 / 10 = 5
        assert 3 < metrics.scr < 7

    def test_to_dict(self):
        """to_dict returns proper dictionary."""
        metrics = ContrastMetrics.from_regions(np.array([100]), np.array([50]))
        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert "scr" in d
        assert "contrast" in d


class TestSceneAnalyzer:
    """Test SceneAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return SceneAnalyzer()

    @pytest.fixture
    def sample_image(self):
        """Create sample image."""
        np.random.seed(42)
        return np.random.uniform(280, 320, (100, 100))

    def test_statistics(self, analyzer, sample_image):
        """statistics method works."""
        stats = analyzer.statistics(sample_image)
        assert isinstance(stats, dict)
        assert "mean" in stats

    def test_histogram(self, analyzer, sample_image):
        """histogram creates figure."""
        fig = analyzer.histogram(sample_image, title="Test")
        assert fig is not None
        plt.close(fig)

    def test_compare_histograms(self, analyzer, sample_image):
        """compare_histograms creates figure."""
        data2 = sample_image + 10
        fig = analyzer.compare_histograms([sample_image, data2], ["Image 1", "Image 2"])
        assert fig is not None
        plt.close(fig)

    def test_spatial_profile(self, analyzer, sample_image):
        """spatial_profile creates figure."""
        fig = analyzer.spatial_profile(sample_image, axis=0)
        assert fig is not None
        plt.close(fig)

    def test_contrast_analysis(self, analyzer, sample_image):
        """contrast_analysis creates figure."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True

        fig = analyzer.contrast_analysis(sample_image, mask)
        assert fig is not None
        plt.close(fig)

    def test_power_spectrum(self, analyzer, sample_image):
        """power_spectrum creates figure."""
        fig = analyzer.power_spectrum(sample_image)
        assert fig is not None
        plt.close(fig)

    def test_radial_average(self, analyzer, sample_image):
        """radial_average creates figure."""
        fig = analyzer.radial_average(sample_image)
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Display Tests
# =============================================================================


class TestCreateColorbar:
    """Test create_colorbar function."""

    def test_colorbar_creation(self):
        """Colorbar figure can be created."""
        fig = create_colorbar(0, 100)
        assert fig is not None
        plt.close(fig)

    def test_colorbar_horizontal(self):
        """Horizontal colorbar works."""
        fig = create_colorbar(0, 100, orientation="horizontal")
        assert fig is not None
        plt.close(fig)


class TestCreateComparison:
    """Test create_comparison function."""

    def test_comparison_creation(self):
        """Comparison figure can be created."""
        image1 = np.random.randn(50, 50)
        image2 = image1 + np.random.randn(50, 50) * 0.1

        fig = create_comparison(image1, image2)
        assert fig is not None
        plt.close(fig)

    def test_comparison_without_difference(self):
        """Comparison without difference works."""
        image1 = np.random.randn(50, 50)
        image2 = image1 + 1

        fig = create_comparison(image1, image2, show_difference=False)
        assert fig is not None
        plt.close(fig)

    def test_comparison_custom_titles(self):
        """Custom titles work."""
        image1 = np.random.randn(50, 50)
        image2 = image1 + 1

        fig = create_comparison(image1, image2, titles=["A", "B", "Diff"])
        assert fig is not None
        plt.close(fig)


class TestMultiPanelDisplay:
    """Test MultiPanelDisplay class."""

    @pytest.fixture
    def display(self):
        """Create display fixture."""
        return MultiPanelDisplay(rows=2, cols=2)

    @pytest.fixture
    def sample_image(self):
        """Create sample image."""
        return np.random.uniform(280, 320, (50, 50))

    def test_display_creation(self, display):
        """Display can be created."""
        assert display is not None
        assert display.figure is not None

    def test_add_thermal(self, display, sample_image):
        """add_thermal works."""
        display.add_thermal(sample_image, 0, 0, title="Test")
        # No assertion needed - just check it doesn't raise
        plt.close(display.figure)

    def test_add_rgb(self, display):
        """add_rgb works."""
        rgb = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        display.add_rgb(rgb, 0, 0, title="RGB")
        plt.close(display.figure)

    def test_add_histogram(self, display, sample_image):
        """add_histogram works."""
        display.add_histogram(sample_image, 0, 0, title="Hist")
        plt.close(display.figure)

    def test_add_profile(self, display, sample_image):
        """add_profile works."""
        display.add_profile(sample_image, 0, 0, axis=0)
        plt.close(display.figure)

    def test_add_text(self, display):
        """add_text works."""
        display.add_text("Hello World", 0, 0)
        plt.close(display.figure)

    def test_add_statistics(self, display, sample_image):
        """add_statistics works."""
        display.add_statistics(sample_image, 0, 0)
        plt.close(display.figure)

    def test_add_difference(self, display, sample_image):
        """add_difference works."""
        image2 = sample_image + np.random.randn(50, 50) * 5
        display.add_difference(sample_image, image2, 0, 0)
        plt.close(display.figure)

    def test_full_panel(self):
        """Full multi-panel display works."""
        display = MultiPanelDisplay(rows=2, cols=2)
        image = np.random.uniform(280, 320, (50, 50))

        display.add_thermal(image, 0, 0, title="Thermal")
        display.add_histogram(image, 0, 1, title="Histogram")
        display.add_profile(image, 1, 0, axis=0, title="Profile")
        display.add_statistics(image, 1, 1)

        display.tight_layout()
        plt.close(display.figure)


class TestProfilePlotter:
    """Test ProfilePlotter class."""

    @pytest.fixture
    def plotter(self):
        """Create plotter fixture."""
        return ProfilePlotter()

    @pytest.fixture
    def sample_image(self):
        """Create sample image."""
        return np.random.randn(100, 100)

    def test_horizontal_profile(self, plotter, sample_image):
        """horizontal_profile creates figure."""
        fig = plotter.horizontal_profile(sample_image, row=50)
        assert fig is not None
        plt.close(fig)

    def test_vertical_profile(self, plotter, sample_image):
        """vertical_profile creates figure."""
        fig = plotter.vertical_profile(sample_image, col=50)
        assert fig is not None
        plt.close(fig)

    def test_cross_section(self, plotter, sample_image):
        """cross_section creates figure."""
        fig = plotter.cross_section(sample_image, (10, 10), (90, 90))
        assert fig is not None
        plt.close(fig)

    def test_multi_profile(self, plotter, sample_image):
        """multi_profile creates figure."""
        fig = plotter.multi_profile(sample_image, positions=[25, 50, 75])
        assert fig is not None
        plt.close(fig)

    def test_profile_with_image(self, plotter, sample_image):
        """profile_with_image creates figure."""
        fig = plotter.profile_with_image(sample_image, position=50, axis=0)
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Integration Tests
# =============================================================================


class TestVisualizationIntegration:
    """Integration tests for visualization module."""

    def test_full_analysis_workflow(self):
        """Complete analysis workflow works."""
        # Create synthetic thermal scene
        np.random.seed(42)
        background = 290 + np.random.randn(200, 200) * 2
        background[80:120, 80:120] = 310 + np.random.randn(40, 40) * 3

        # Render
        renderer = ThermalRenderer(ThermalColormap.IRONBOW)
        rgb = renderer.render(background, vmin=280, vmax=320)
        assert rgb.shape == (200, 200, 3)

        # Analyze
        analyzer = SceneAnalyzer()
        stats = analyzer.statistics(background)
        assert 290 < stats["mean"] < 300

        # Contrast analysis
        mask = np.zeros((200, 200), dtype=bool)
        mask[80:120, 80:120] = True
        metrics = ContrastMetrics.from_mask(background, mask)
        assert metrics.scr > 5  # Good contrast

        plt.close("all")

    def test_colormap_consistency(self):
        """Different colormaps produce different outputs."""
        data = np.linspace(0, 1, 100).reshape(10, 10)

        rgb_iron = apply_colormap(data, ThermalColormap.IRONBOW)
        rgb_gray = apply_colormap(data, ThermalColormap.GRAYSCALE)
        rgb_hot = apply_colormap(data, ThermalColormap.HOT)

        # All should be different
        assert not np.array_equal(rgb_iron, rgb_gray)
        assert not np.array_equal(rgb_iron, rgb_hot)
        assert not np.array_equal(rgb_gray, rgb_hot)

    def test_display_single_panel(self):
        """Single panel display works."""
        display = MultiPanelDisplay(rows=1, cols=1)
        data = np.random.randn(50, 50)
        display.add_thermal(data, 0, 0)
        plt.close(display.figure)

    def test_display_single_row(self):
        """Single row display works."""
        display = MultiPanelDisplay(rows=1, cols=3)
        data = np.random.randn(50, 50)
        display.add_thermal(data, 0, 0)
        display.add_histogram(data, 0, 1)
        display.add_profile(data, 0, 2)
        plt.close(display.figure)

    def test_display_single_column(self):
        """Single column display works."""
        display = MultiPanelDisplay(rows=3, cols=1)
        data = np.random.randn(50, 50)
        display.add_thermal(data, 0, 0)
        display.add_histogram(data, 1, 0)
        display.add_profile(data, 2, 0)
        plt.close(display.figure)
