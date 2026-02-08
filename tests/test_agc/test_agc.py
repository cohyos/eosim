"""Tests for the AGC (Automatic Gain Control) module."""

import numpy as np
import pytest

from eosim.agc import (
    AGCMode,
    Polarity,
    AGCParameters,
    AGCProcessor,
    PolarityMapper,
    apply_agc,
    apply_polarity,
    create_agc_processor,
)


class TestAGCParameters:
    """Tests for AGCParameters dataclass."""

    def test_default_parameters(self):
        params = AGCParameters()
        assert params.mode == AGCMode.HISTOGRAM_EQ
        assert params.output_bits == 8
        assert params.output_max == 255

    def test_custom_parameters(self):
        params = AGCParameters(
            mode=AGCMode.LINEAR,
            linear_percent=2.0,
            output_bits=14,
        )
        assert params.mode == AGCMode.LINEAR
        assert params.linear_percent == 2.0
        assert params.output_max == 16383

    def test_invalid_linear_percent(self):
        with pytest.raises(ValueError):
            AGCParameters(linear_percent=-1.0)
        with pytest.raises(ValueError):
            AGCParameters(linear_percent=51.0)

    def test_invalid_clip_limit(self):
        with pytest.raises(ValueError):
            AGCParameters(clahe_clip_limit=0.5)

    def test_invalid_output_bits(self):
        with pytest.raises(ValueError):
            AGCParameters(output_bits=0)
        with pytest.raises(ValueError):
            AGCParameters(output_bits=17)


class TestAGCProcessor:
    """Tests for AGCProcessor."""

    def setup_method(self):
        """Create test images."""
        self.rng = np.random.default_rng(42)
        # Simulate 14-bit sensor data with bimodal distribution
        self.raw_image = np.zeros((120, 160), dtype=np.float64)
        self.raw_image[:60, :] = 8000 + self.rng.normal(0, 200, (60, 160))
        self.raw_image[60:, :] = 10000 + self.rng.normal(0, 300, (60, 160))

    def test_linear_stretch(self):
        params = AGCParameters(mode=AGCMode.LINEAR, linear_percent=1.0)
        agc = AGCProcessor(params)
        result = agc.process(self.raw_image)

        assert result.shape == self.raw_image.shape
        assert result.min() >= 0
        assert result.max() <= 255
        # Should use most of the output range
        assert result.max() > 200

    def test_histogram_equalization(self):
        params = AGCParameters(mode=AGCMode.HISTOGRAM_EQ)
        agc = AGCProcessor(params)
        result = agc.process(self.raw_image)

        assert result.shape == self.raw_image.shape
        assert result.min() >= 0
        assert result.max() <= 255

    def test_clahe(self):
        params = AGCParameters(
            mode=AGCMode.CLAHE,
            clahe_clip_limit=2.0,
            clahe_grid_size=(4, 4),
        )
        agc = AGCProcessor(params)
        result = agc.process(self.raw_image)

        assert result.shape == self.raw_image.shape
        assert result.min() >= 0
        assert result.max() <= 255

    def test_plateau_equalization(self):
        params = AGCParameters(mode=AGCMode.PLATEAU, plateau_level=0.01)
        agc = AGCProcessor(params)
        result = agc.process(self.raw_image)

        assert result.shape == self.raw_image.shape
        assert result.min() >= 0
        assert result.max() <= 255

    def test_dde(self):
        params = AGCParameters(mode=AGCMode.DDE, dde_strength=0.5)
        agc = AGCProcessor(params)
        result = agc.process(self.raw_image)

        assert result.shape == self.raw_image.shape
        assert result.min() >= 0
        assert result.max() <= 255

    def test_ice(self):
        params = AGCParameters(mode=AGCMode.ICE, ice_iterations=2)
        agc = AGCProcessor(params)
        result = agc.process(self.raw_image)

        assert result.shape == self.raw_image.shape
        assert result.min() >= 0
        assert result.max() <= 255

    def test_logarithmic(self):
        params = AGCParameters(mode=AGCMode.LOGARITHMIC)
        agc = AGCProcessor(params)
        result = agc.process(self.raw_image)

        assert result.shape == self.raw_image.shape
        assert result.min() >= 0
        assert result.max() <= 255

    def test_manual(self):
        params = AGCParameters(
            mode=AGCMode.MANUAL,
            manual_min=8000,
            manual_max=10000,
        )
        agc = AGCProcessor(params)
        result = agc.process(self.raw_image)

        assert result.shape == self.raw_image.shape
        assert result.min() >= 0
        assert result.max() <= 255

    def test_constant_image(self):
        """AGC should handle uniform images gracefully."""
        constant = np.full((64, 64), 5000.0)
        params = AGCParameters(mode=AGCMode.HISTOGRAM_EQ)
        agc = AGCProcessor(params)
        result = agc.process(constant)

        assert result.shape == constant.shape
        assert np.isfinite(result).all()

    def test_3d_image_raises(self):
        """Should reject non-2D input."""
        rgb = np.zeros((64, 64, 3))
        agc = AGCProcessor()
        with pytest.raises(ValueError, match="2D"):
            agc.process(rgb)

    def test_temporal_filtering(self):
        """Test temporal AGC smoothing across frames."""
        agc = AGCProcessor(
            AGCParameters(mode=AGCMode.LINEAR),
            temporal_filter=0.5,
        )

        # Process two frames
        result1 = agc.process(self.raw_image)
        assert agc.state.frame_count == 1

        result2 = agc.process(self.raw_image * 1.1)
        assert agc.state.frame_count == 2

    def test_reset(self):
        agc = AGCProcessor()
        agc.process(self.raw_image)
        assert agc.state.frame_count == 1

        agc.reset()
        assert agc.state.frame_count == 0

    def test_output_14bit(self):
        params = AGCParameters(mode=AGCMode.LINEAR, output_bits=14)
        agc = AGCProcessor(params)
        result = agc.process(self.raw_image)

        assert result.max() <= 16383


class TestPolarityMapper:
    """Tests for PolarityMapper."""

    def setup_method(self):
        self.gray_image = np.linspace(0, 255, 64 * 64).reshape(64, 64)

    def test_white_hot(self):
        mapper = PolarityMapper(Polarity.WHITE_HOT)
        result = mapper.apply(self.gray_image)

        assert result.ndim == 2
        assert result.dtype == np.uint8

    def test_black_hot(self):
        mapper = PolarityMapper(Polarity.BLACK_HOT)
        result = mapper.apply(self.gray_image)

        assert result.ndim == 2
        assert result.dtype == np.uint8
        # Black hot should invert the image
        assert result[0, 0] > result[-1, -1]

    def test_ironbow(self):
        mapper = PolarityMapper(Polarity.IRONBOW)
        result = mapper.apply(self.gray_image)

        assert result.ndim == 3
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_rainbow(self):
        mapper = PolarityMapper(Polarity.RAINBOW)
        result = mapper.apply(self.gray_image)

        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_isotherm(self):
        mapper = PolarityMapper(Polarity.ISOTHERM)
        result = mapper.apply(self.gray_image)

        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_all_polarities(self):
        """All polarity modes should produce valid output."""
        for pol in Polarity:
            mapper = PolarityMapper(pol)
            result = mapper.apply(self.gray_image)
            assert np.isfinite(result).all(), f"Non-finite values for {pol}"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_apply_agc(self):
        img = np.random.default_rng(0).uniform(5000, 15000, (64, 64))
        result = apply_agc(img, mode=AGCMode.LINEAR, linear_percent=2.0)

        assert result.shape == img.shape
        assert result.min() >= 0
        assert result.max() <= 255

    def test_apply_polarity(self):
        img = np.linspace(0, 255, 64 * 64).reshape(64, 64)
        result = apply_polarity(img, polarity=Polarity.IRONBOW, input_bits=8)

        assert result.shape == (64, 64, 3)

    def test_create_agc_processor_string_mode(self):
        agc = create_agc_processor(mode="linear")
        assert agc.params.mode == AGCMode.LINEAR

    def test_create_agc_processor_enum_mode(self):
        agc = create_agc_processor(mode=AGCMode.CLAHE, clahe_clip_limit=4.0)
        assert agc.params.clahe_clip_limit == 4.0
