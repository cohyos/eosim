"""Tests for sensor.adc module."""

import pytest
import numpy as np

from eosim.sensor.adc import (
    ADCType,
    ADCParameters,
    ADCResult,
    ADCModel,
    LogarithmicADC,
    compute_quantization_noise,
    required_bit_depth,
    create_adc,
)


class TestADCParameters:
    """Tests for ADCParameters dataclass."""

    def test_default_creation(self) -> None:
        """Should create with default parameters."""
        params = ADCParameters()
        assert params.bit_depth == 14
        assert params.full_well_electrons == 100000.0

    def test_max_dn(self) -> None:
        """Should compute max DN from bit depth."""
        params = ADCParameters(bit_depth=12)
        assert params.max_dn == 4095

        params = ADCParameters(bit_depth=14)
        assert params.max_dn == 16383

        params = ADCParameters(bit_depth=16)
        assert params.max_dn == 65535

    def test_auto_gain(self) -> None:
        """Should compute gain to map full well to max DN."""
        params = ADCParameters(bit_depth=14, full_well_electrons=100000)
        expected_gain = 100000 / 16383
        assert params.gain_electrons_per_dn == pytest.approx(expected_gain)

    def test_lsb_electrons(self) -> None:
        """LSB should equal gain."""
        params = ADCParameters(bit_depth=14, full_well_electrons=100000)
        assert params.lsb_electrons == params.gain_electrons_per_dn

    def test_dynamic_range_db(self) -> None:
        """Should compute dynamic range in dB."""
        params = ADCParameters(bit_depth=14)
        expected = 20 * np.log10(16383)
        assert params.dynamic_range_db == pytest.approx(expected)

    def test_quantization_noise(self) -> None:
        """Should compute quantization noise."""
        params = ADCParameters(bit_depth=14, full_well_electrons=100000)
        lsb = 100000 / 16383
        expected = lsb / np.sqrt(12)
        assert params.quantization_noise_electrons == pytest.approx(expected)

    def test_validation_errors(self) -> None:
        """Should reject invalid parameters."""
        with pytest.raises(ValueError):
            ADCParameters(bit_depth=0)
        with pytest.raises(ValueError):
            ADCParameters(bit_depth=64)
        with pytest.raises(ValueError):
            ADCParameters(full_well_electrons=0)


class TestADCModel:
    """Tests for ADCModel class."""

    @pytest.fixture
    def adc(self) -> ADCModel:
        """Create default ADC for testing."""
        params = ADCParameters(
            bit_depth=14,
            full_well_electrons=100000,
            offset_dn=100,
        )
        return ADCModel(params)

    def test_linear_conversion(self, adc: ADCModel) -> None:
        """Should perform linear conversion."""
        electrons = np.full((10, 10), 50000.0)
        result = adc.convert(electrons)
        # Half full well -> half max DN + offset
        expected_dn = 50000 / adc.params.gain_electrons_per_dn + 100
        assert np.all(np.abs(result.dn.astype(float) - expected_dn) <= 1)

    def test_saturation_detection(self, adc: ADCModel) -> None:
        """Should detect saturated pixels."""
        electrons = np.array([[50000, 200000], [200000, 50000]])
        result = adc.convert(electrons)
        assert result.saturated_pixels == 2
        assert result.saturation_fraction == pytest.approx(0.5)

    def test_saturation_clipping(self, adc: ADCModel) -> None:
        """Should clip to max DN."""
        electrons = np.full((10, 10), 200000.0)
        result = adc.convert(electrons)
        assert np.all(result.dn == adc.params.max_dn)

    def test_negative_clipping(self) -> None:
        """Should clip negative values to zero."""
        # Use ADC with no offset to test negative clipping cleanly
        params = ADCParameters(bit_depth=14, full_well_electrons=100000, offset_dn=0)
        adc = ADCModel(params)
        electrons = np.array([[-100, 1000]])
        result = adc.convert(electrons)
        assert result.dn[0, 0] == 0
        assert result.clipped_low == 1

    def test_output_dtype(self, adc: ADCModel) -> None:
        """Should use appropriate integer type."""
        electrons = np.full((10, 10), 50000.0)
        result = adc.convert(electrons)
        # 14-bit -> uint16
        assert result.dn.dtype == np.uint16

    def test_8bit_output(self) -> None:
        """8-bit ADC should output uint8."""
        params = ADCParameters(bit_depth=8, full_well_electrons=25500)
        adc = ADCModel(params)
        result = adc.convert(np.full((10, 10), 10000.0))
        assert result.dn.dtype == np.uint8

    def test_inverse_conversion(self, adc: ADCModel) -> None:
        """Should invert conversion (approximately)."""
        original = np.full((10, 10), 50000.0)
        result = adc.convert(original)
        recovered = adc.inverse(result.dn)
        # Allow quantization error
        assert np.all(np.abs(recovered - original) < adc.params.lsb_electrons * 2)

    def test_electrons_to_dn_scalar(self, adc: ADCModel) -> None:
        """Should convert single value to DN."""
        dn = adc.electrons_to_dn(50000)
        expected = 50000 / adc.params.gain_electrons_per_dn + 100
        assert dn == pytest.approx(expected)

    def test_dn_to_electrons_scalar(self, adc: ADCModel) -> None:
        """Should convert DN to electrons."""
        electrons = adc.dn_to_electrons(1000)
        expected = (1000 - 100) * adc.params.gain_electrons_per_dn
        assert electrons == pytest.approx(expected)


class TestADCNonlinearity:
    """Tests for ADC nonlinearity."""

    def test_nonlinearity_effect(self) -> None:
        """Nonlinearity should alter output."""
        params_linear = ADCParameters(
            bit_depth=14,
            full_well_electrons=100000,
            nonlinearity_percent=0,
        )
        params_nonlinear = ADCParameters(
            bit_depth=14,
            full_well_electrons=100000,
            nonlinearity_percent=1.0,
        )
        adc_linear = ADCModel(params_linear)
        adc_nonlinear = ADCModel(params_nonlinear)

        electrons = np.linspace(0, 80000, 100)
        result_linear = adc_linear.convert(electrons)
        result_nonlinear = adc_nonlinear.convert(electrons)

        # Results should differ
        assert not np.allclose(result_linear.dn, result_nonlinear.dn)


class TestLogarithmicADC:
    """Tests for LogarithmicADC class."""

    @pytest.fixture
    def log_adc(self) -> LogarithmicADC:
        """Create logarithmic ADC for testing."""
        params = ADCParameters(bit_depth=14, full_well_electrons=100000)
        return LogarithmicADC(params, compression_factor=10.0)

    def test_logarithmic_compression(self, log_adc: LogarithmicADC) -> None:
        """Should compress dynamic range."""
        electrons = np.array([1000, 10000, 100000])
        result = log_adc.convert(electrons)
        # Check that ratios are compressed
        linear_ratio = electrons[-1] / electrons[0]  # 100x
        dn_ratio = float(result.dn[-1]) / float(result.dn[0])
        # DN ratio should be much less than 100 (logarithmic compression)
        assert dn_ratio < linear_ratio / 3  # At least 3x compression

    def test_inverse_recovery(self, log_adc: LogarithmicADC) -> None:
        """Should approximately recover original values."""
        original = np.array([1000.0, 10000.0, 50000.0])
        result = log_adc.convert(original)
        recovered = log_adc.inverse(result.dn)
        # Allow for quantization errors
        assert np.allclose(recovered, original, rtol=0.05)


class TestComputeQuantizationNoise:
    """Tests for compute_quantization_noise function."""

    def test_formula(self) -> None:
        """Should follow LSB/sqrt(12) formula."""
        noise = compute_quantization_noise(bit_depth=14, full_well=100000)
        lsb = 100000 / (2**14 - 1)
        expected = lsb / np.sqrt(12)
        assert noise == pytest.approx(expected)


class TestRequiredBitDepth:
    """Tests for required_bit_depth function."""

    def test_high_read_noise(self) -> None:
        """High read noise should require fewer bits."""
        bits = required_bit_depth(full_well=100000, read_noise=100)
        # Quant noise should be less than 0.5 × read noise
        max_dn = 2**bits
        quant_noise = (100000 / max_dn) / np.sqrt(12)
        assert quant_noise < 0.5 * 100

    def test_low_read_noise(self) -> None:
        """Low read noise should require more bits."""
        bits_low = required_bit_depth(full_well=100000, read_noise=10)
        bits_high = required_bit_depth(full_well=100000, read_noise=100)
        assert bits_low > bits_high


class TestCreateADC:
    """Tests for create_adc factory function."""

    def test_linear_creation(self) -> None:
        """Should create linear ADC."""
        adc = create_adc(bit_depth=12, full_well_electrons=50000)
        assert isinstance(adc, ADCModel)
        assert adc.params.bit_depth == 12
        assert adc.params.full_well_electrons == 50000

    def test_logarithmic_creation(self) -> None:
        """Should create logarithmic ADC."""
        adc = create_adc(bit_depth=14, adc_type="logarithmic")
        assert isinstance(adc, LogarithmicADC)
