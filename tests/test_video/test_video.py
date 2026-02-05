"""Tests for EOSIM video processing module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from eosim.video.config import (
    VideoSimulationConfig,
    VideoProcessingMode,
    RadianceMapConfig,
    ThermalEstimateConfig,
    TemplateConfig,
    ObjectThermalProperties,
    ChannelMode,
    SegmentationMethod,
)
from eosim.video.radiance_mapper import (
    RadianceMapper,
    intensity_to_radiance,
    estimate_radiance_range_for_sensor,
)
from eosim.video.segmentation import (
    Segmenter,
    ObjectClassifier,
    segment_frame,
)
from eosim.video.thermal_estimator import (
    ThermalEstimator,
    TemplateMapper,
    estimate_thermal_from_visible,
)
from eosim.video.reader import to_grayscale
from eosim.video.writer import dn_to_display


class TestRadianceMapper:
    """Tests for RadianceMapper."""

    def test_linear_mapping(self):
        """Test linear intensity to radiance mapping."""
        config = RadianceMapConfig(
            mapping_type="linear",
            input_range=(0, 255),
            output_radiance_range=(0.0, 1.0),
        )
        mapper = RadianceMapper(config)

        # Test with uniform intensity
        frame = np.full((100, 100), 128, dtype=np.uint8)
        radiance = mapper.map_frame(frame)

        assert radiance.shape == (100, 100)
        assert 0.4 < radiance.mean() < 0.6  # Should be around 0.5

    def test_gamma_mapping(self):
        """Test gamma-corrected mapping."""
        config = RadianceMapConfig(
            mapping_type="gamma",
            gamma=2.2,
            output_radiance_range=(0.0, 1.0),
        )
        mapper = RadianceMapper(config)

        frame = np.full((100, 100), 128, dtype=np.uint8)
        radiance = mapper.map_frame(frame)

        # Gamma correction should reduce mid-tones
        assert radiance.mean() < 0.5

    def test_invert_mapping(self):
        """Test inverted mapping."""
        config = RadianceMapConfig(
            invert=True,
            output_radiance_range=(0.0, 1.0),
        )
        mapper = RadianceMapper(config)

        # Black should map to max radiance
        black = np.zeros((10, 10), dtype=np.uint8)
        white = np.full((10, 10), 255, dtype=np.uint8)

        rad_black = mapper.map_frame(black)
        rad_white = mapper.map_frame(white)

        assert rad_black.mean() > rad_white.mean()

    def test_color_frame_luminance(self):
        """Test mapping of color frame using luminance."""
        config = RadianceMapConfig(
            channel_mode=ChannelMode.LUMINANCE,
            output_radiance_range=(0.0, 1.0),
        )
        mapper = RadianceMapper(config)

        # Create color frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 1] = 255  # Green channel

        radiance = mapper.map_frame(frame)
        assert radiance.shape == (100, 100)
        # Green has weight 0.587 in luminance
        assert 0.5 < radiance.mean() < 0.7

    def test_to_scene_input(self):
        """Test conversion to SceneInput."""
        mapper = RadianceMapper()
        frame = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        scene = mapper.to_scene_input(frame)

        assert scene.radiance_map is not None
        assert scene.radiance_map.shape == (100, 100)


class TestIntensityToRadiance:
    """Tests for intensity_to_radiance function."""

    def test_basic_conversion(self):
        """Test basic intensity to radiance conversion."""
        intensity = np.array([[0, 128, 255]], dtype=np.uint8)
        radiance = intensity_to_radiance(
            intensity,
            output_range=(0.0, 1.0),
            input_range=(0, 255),
        )

        assert radiance[0, 0] == pytest.approx(0.0, abs=0.01)
        assert radiance[0, 1] == pytest.approx(0.5, abs=0.01)
        assert radiance[0, 2] == pytest.approx(1.0, abs=0.01)

    def test_with_gamma(self):
        """Test conversion with gamma correction."""
        intensity = np.array([[128]], dtype=np.uint8)

        radiance_linear = intensity_to_radiance(intensity, gamma=1.0)
        radiance_gamma = intensity_to_radiance(intensity, gamma=2.2)

        # Gamma > 1 should reduce mid-tones
        assert radiance_gamma < radiance_linear


class TestEstimateRadianceRange:
    """Tests for estimate_radiance_range_for_sensor."""

    def test_lwir_range(self):
        """Test LWIR radiance range estimation."""
        min_rad, max_rad = estimate_radiance_range_for_sensor("lwir")
        assert min_rad > 0
        assert max_rad > min_rad
        # LWIR at 300K should be ~20-60 W/(m²·sr)
        assert 10 < min_rad < 50
        assert 40 < max_rad < 100

    def test_swir_range(self):
        """Test SWIR radiance range estimation."""
        min_rad, max_rad = estimate_radiance_range_for_sensor("swir")
        assert min_rad > 0
        assert max_rad > min_rad
        # SWIR is reflected, lower radiance
        assert max_rad < 1.0


class TestSegmenter:
    """Tests for Segmenter class."""

    def test_threshold_segmentation(self):
        """Test threshold-based segmentation."""
        segmenter = Segmenter(method=SegmentationMethod.THRESHOLD)

        # Create simple test image with bright object
        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[30:70, 30:70] = 200  # Bright square

        mask, objects = segmenter.segment(frame)

        assert mask.shape == (100, 100)
        # Should detect at least one object
        assert len(objects) > 0 or mask.max() > 0

    def test_color_segmentation(self):
        """Test color-based segmentation."""
        segmenter = Segmenter(
            method=SegmentationMethod.COLOR,
            min_object_size=50,
        )

        # Create color image with distinct regions
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:40, 20:40] = [255, 0, 0]  # Red square
        frame[60:80, 60:80] = [0, 255, 0]  # Green square

        mask, objects = segmenter.segment(frame)

        assert mask.shape == (100, 100)


class TestObjectClassifier:
    """Tests for ObjectClassifier."""

    def test_rule_classification(self):
        """Test rule-based classification."""
        from eosim.video.config import ObjectInstance

        classifier = ObjectClassifier()

        # Person-like object (tall, narrow)
        person = ObjectInstance(
            object_id=1,
            class_name="unknown",
            bounding_box=(0, 45, 100, 55),  # Tall rectangle
            area_pixels=1000,
        )

        classified = classifier.classify([person])
        # Should classify based on aspect ratio
        assert len(classified) == 1


class TestThermalEstimator:
    """Tests for ThermalEstimator."""

    def test_estimate_frame(self):
        """Test thermal estimation from frame."""
        config = ThermalEstimateConfig(
            ambient_temperature_k=290.0,
        )
        estimator = ThermalEstimator(config)

        # Simple test frame
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        temp_map, emis_map = estimator.estimate_frame(frame)

        assert temp_map.shape == (100, 100)
        assert emis_map.shape == (100, 100)
        assert temp_map.min() > 0  # Positive temperatures
        assert 0 < emis_map.min() <= emis_map.max() <= 1.0

    def test_to_scene_input(self):
        """Test conversion to SceneInput."""
        estimator = ThermalEstimator()
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        scene = estimator.to_scene_input(frame)

        assert scene.temperature_map is not None
        assert scene.emissivity_map is not None


class TestTemplateMapper:
    """Tests for TemplateMapper."""

    def test_map_frame(self):
        """Test template-based mapping."""
        config = TemplateConfig(
            background_temperature_k=290.0,
        )
        mapper = TemplateMapper(config)

        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        temp_map, emis_map, motion = mapper.map_frame(frame)

        assert temp_map.shape == (100, 100)
        assert emis_map.shape == (100, 100)

    def test_to_scene_input(self):
        """Test conversion to SceneInput."""
        mapper = TemplateMapper()
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        scene = mapper.to_scene_input(frame)

        assert scene.temperature_map is not None
        assert scene.emissivity_map is not None


class TestToGrayscale:
    """Tests for to_grayscale function."""

    def test_luminance_conversion(self):
        """Test luminance conversion."""
        # Create pure green image
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:, :, 1] = 255  # Green channel

        gray = to_grayscale(frame, ChannelMode.LUMINANCE)

        # Green has weight 0.587
        expected = int(255 * 0.587)
        assert abs(gray[0, 0] - expected) < 2

    def test_single_channel(self):
        """Test single channel extraction."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:, :, 0] = 100  # Red
        frame[:, :, 1] = 150  # Green
        frame[:, :, 2] = 200  # Blue

        red = to_grayscale(frame, ChannelMode.RED)
        green = to_grayscale(frame, ChannelMode.GREEN)
        blue = to_grayscale(frame, ChannelMode.BLUE)

        assert red[0, 0] == 100
        assert green[0, 0] == 150
        assert blue[0, 0] == 200


class TestDnToDisplay:
    """Tests for dn_to_display function."""

    def test_basic_conversion(self):
        """Test basic DN to display conversion."""
        # 14-bit DN image
        dn_image = np.array([[0, 8192, 16383]], dtype=np.uint16)

        display = dn_to_display(dn_image)

        assert display.dtype == np.uint8
        assert display.shape == (1, 3)

    def test_percentile_clipping(self):
        """Test percentile-based contrast stretch."""
        # Image with outliers
        dn_image = np.full((100, 100), 8000, dtype=np.uint16)
        dn_image[0, 0] = 0      # Low outlier
        dn_image[99, 99] = 16383  # High outlier

        display = dn_to_display(dn_image, percentile_clip=(5, 95))

        # Middle values should be near middle of output range
        assert display.dtype == np.uint8


class TestVideoSimulationConfig:
    """Tests for VideoSimulationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = VideoSimulationConfig()

        assert config.mode == VideoProcessingMode.RADIANCE_MAP
        assert config.radiance_config is not None

    def test_template_mode(self):
        """Test template mode initialization."""
        config = VideoSimulationConfig(mode=VideoProcessingMode.TEMPLATE)

        assert config.template_config is not None

    def test_thermal_estimate_mode(self):
        """Test thermal estimate mode initialization."""
        config = VideoSimulationConfig(mode=VideoProcessingMode.THERMAL_ESTIMATE)

        assert config.thermal_config is not None


class TestEstimateThermalFromVisible:
    """Tests for estimate_thermal_from_visible convenience function."""

    def test_basic_estimation(self):
        """Test basic thermal estimation."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        temp_map, emis_map = estimate_thermal_from_visible(
            frame,
            ambient_temperature_k=295.0,
        )

        assert temp_map.shape == (100, 100)
        assert emis_map.shape == (100, 100)
        assert temp_map.min() > 200  # Reasonable temperature range
        assert temp_map.max() < 500


class TestSegmentFrame:
    """Tests for segment_frame convenience function."""

    def test_basic_segmentation(self):
        """Test basic frame segmentation."""
        # Create frame with distinct object
        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[30:70, 30:70] = 200

        mask, objects = segment_frame(
            frame,
            method=SegmentationMethod.THRESHOLD,
            min_object_size=100,
        )

        assert mask.shape == (100, 100)
