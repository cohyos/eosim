"""Tests for sensor.base module."""

import pytest
import numpy as np

from eosim.sensor.base import (
    SensorParameters,
    SensorResult,
    SensorModel,
    SimpleSensorModel,
    create_sensor_model,
)
from eosim.sensor.fpa import FPAGeometry, DetectorProperties, FPAConfig, DetectorType


class TestSensorParameters:
    """Tests for SensorParameters dataclass."""

    @pytest.fixture
    def fpa_config(self) -> FPAConfig:
        """Create FPA config for testing."""
        geom = FPAGeometry(640, 480, 15.0)
        detector = DetectorProperties.from_detector_type(DetectorType.HGCDTE_MWIR)
        return FPAConfig(geometry=geom, detector=detector)

    def test_creation(self, fpa_config: FPAConfig) -> None:
        """Should create sensor parameters."""
        params = SensorParameters(
            fpa=fpa_config,
            integration_time_s=0.01,
            optics_f_number=2.0,
        )
        assert params.integration_time_s == 0.01
        assert params.optics_f_number == 2.0

    def test_pixel_area(self, fpa_config: FPAConfig) -> None:
        """Should expose pixel area from FPA."""
        params = SensorParameters(fpa=fpa_config)
        expected = (15e-6) ** 2
        assert params.pixel_area_m2 == pytest.approx(expected)

    def test_center_wavelength(self, fpa_config: FPAConfig) -> None:
        """Should compute center wavelength."""
        params = SensorParameters(
            fpa=fpa_config,
            spectral_band_um=(8.0, 12.0),
        )
        assert params.center_wavelength_um == pytest.approx(10.0)

    def test_bandwidth(self, fpa_config: FPAConfig) -> None:
        """Should compute bandwidth."""
        params = SensorParameters(
            fpa=fpa_config,
            spectral_band_um=(8.0, 12.0),
        )
        assert params.bandwidth_um == pytest.approx(4.0)

    def test_omega_pixel(self, fpa_config: FPAConfig) -> None:
        """Should compute pixel solid angle."""
        params = SensorParameters(fpa=fpa_config, optics_f_number=2.0)
        expected = np.pi / (4 * 4)  # π / (4 × F#²)
        assert params.omega_pixel == pytest.approx(expected)

    def test_validation_errors(self, fpa_config: FPAConfig) -> None:
        """Should reject invalid parameters."""
        with pytest.raises(ValueError, match="Integration time"):
            SensorParameters(fpa=fpa_config, integration_time_s=-0.01)
        with pytest.raises(ValueError, match="F-number"):
            SensorParameters(fpa=fpa_config, optics_f_number=-2.0)
        with pytest.raises(ValueError, match="Transmission"):
            SensorParameters(fpa=fpa_config, optics_transmission=1.5)
        with pytest.raises(ValueError, match="Band minimum"):
            SensorParameters(fpa=fpa_config, spectral_band_um=(12.0, 8.0))


class TestSimpleSensorModel:
    """Tests for SimpleSensorModel class."""

    @pytest.fixture
    def sensor(self) -> SimpleSensorModel:
        """Create sensor model for testing."""
        return create_sensor_model(
            detector_type=DetectorType.HGCDTE_MWIR,
            pixel_pitch_um=15.0,
            resolution=(64, 64),
            integration_time_s=0.01,
            f_number=2.0,
            spectral_band_um=(3.0, 5.0),
            seed=42,
        )

    def test_creation(self, sensor: SimpleSensorModel) -> None:
        """Should create sensor model."""
        assert sensor.params.integration_time_s == 0.01
        assert sensor.params.optics_f_number == 2.0

    def test_effective_qe(self, sensor: SimpleSensorModel) -> None:
        """Should have reasonable effective QE."""
        qe = sensor.effective_qe
        assert 0 < qe < 1

    def test_irradiance_to_electrons(self, sensor: SimpleSensorModel) -> None:
        """Should convert irradiance to electrons."""
        irradiance = np.full((64, 64), 1e-3)  # 1 mW/m²
        electrons = sensor.irradiance_to_electrons(irradiance)
        assert electrons.shape == (64, 64)
        assert np.all(electrons > 0)

    def test_higher_irradiance_more_electrons(self, sensor: SimpleSensorModel) -> None:
        """More irradiance should give more electrons."""
        irr_low = np.full((64, 64), 1e-4)
        irr_high = np.full((64, 64), 1e-3)
        e_low = sensor.irradiance_to_electrons(irr_low)
        e_high = sensor.irradiance_to_electrons(irr_high)
        assert np.all(e_high > e_low)

    def test_radiance_to_electrons(self, sensor: SimpleSensorModel) -> None:
        """Should convert radiance to electrons."""
        radiance = np.full((64, 64), 1.0)  # 1 W/(m²·sr)
        electrons = sensor.radiance_to_electrons(radiance)
        assert electrons.shape == (64, 64)
        assert np.all(electrons > 0)

    def test_apply_noise(self, sensor: SimpleSensorModel) -> None:
        """Should apply noise to signal."""
        electrons = np.full((64, 64), 10000.0)
        noisy = sensor.apply_noise(electrons)
        assert noisy.shape == (64, 64)
        # Should differ from input due to noise
        assert not np.allclose(electrons, noisy)

    def test_digitize(self, sensor: SimpleSensorModel) -> None:
        """Should convert electrons to DN."""
        electrons = np.full((64, 64), 50000.0)
        result = sensor.digitize(electrons)
        assert result.dn.shape == (64, 64)
        assert result.dn.dtype in [np.uint8, np.uint16, np.uint32]

    def test_apply_full_chain(self, sensor: SimpleSensorModel) -> None:
        """Should apply complete sensor chain."""
        irradiance = np.full((64, 64), 1e-3)
        result = sensor.apply(irradiance)
        assert isinstance(result, SensorResult)
        assert result.dn.shape == (64, 64)
        assert result.electrons is not None
        assert result.noisy_electrons is not None

    def test_apply_from_radiance(self, sensor: SimpleSensorModel) -> None:
        """Should apply chain starting from radiance."""
        radiance = np.full((64, 64), 1.0)
        result = sensor.apply_from_radiance(radiance)
        assert isinstance(result, SensorResult)
        assert result.dn.shape == (64, 64)

    def test_reproducibility(self) -> None:
        """Same seed should give same results."""
        sensor1 = create_sensor_model(seed=42, resolution=(32, 32))
        sensor2 = create_sensor_model(seed=42, resolution=(32, 32))
        irr = np.full((32, 32), 1e-3)
        r1 = sensor1.apply(irr)
        r2 = sensor2.apply(irr)
        assert np.allclose(r1.dn, r2.dn)

    def test_noise_contributions(self, sensor: SimpleSensorModel) -> None:
        """Should provide noise breakdown."""
        irradiance = np.full((64, 64), 1e-3)
        result = sensor.apply(irradiance)
        assert result.noise_contributions is not None
        assert result.noise_contributions.shot_variance > 0


class TestSensorMetrics:
    """Tests for sensor performance metrics computation."""

    @pytest.fixture
    def sensor(self) -> SimpleSensorModel:
        """Create sensor for testing."""
        return create_sensor_model(
            detector_type=DetectorType.HGCDTE_LWIR,
            resolution=(64, 64),
            spectral_band_um=(8.0, 12.0),
            seed=42,
        )

    def test_compute_metrics(self, sensor: SimpleSensorModel) -> None:
        """Should compute performance metrics."""
        metrics = sensor.compute_metrics(scene_temperature_k=300)
        assert metrics.nedt_k > 0
        assert metrics.snr > 0
        assert metrics.dynamic_range_db > 0
        # Note: well_fill > 1 indicates saturation conditions
        assert metrics.well_fill_fraction > 0

    def test_nedt_reasonable_range(self, sensor: SimpleSensorModel) -> None:
        """NEDT should be in reasonable range for LWIR."""
        metrics = sensor.compute_metrics(scene_temperature_k=300)
        # Typical LWIR sensors have NEDT 20-100 mK
        assert 0.001 < metrics.nedt_k < 1.0  # 1 mK to 1 K


class TestCreateSensorModel:
    """Tests for create_sensor_model factory function."""

    def test_mwir_creation(self) -> None:
        """Should create MWIR sensor."""
        sensor = create_sensor_model(
            detector_type=DetectorType.HGCDTE_MWIR,
            spectral_band_um=(3.0, 5.0),
        )
        assert sensor.params.spectral_band_um == (3.0, 5.0)

    def test_lwir_creation(self) -> None:
        """Should create LWIR sensor."""
        sensor = create_sensor_model(
            detector_type=DetectorType.HGCDTE_LWIR,
        )
        # Should use default LWIR band
        assert sensor.params.spectral_band_um[0] >= 8.0

    def test_visible_creation(self) -> None:
        """Should create visible sensor."""
        sensor = create_sensor_model(
            detector_type=DetectorType.SI_CCD,
        )
        # Should use default visible band
        assert sensor.params.spectral_band_um[0] < 1.0

    def test_string_detector_type(self) -> None:
        """Should accept string detector type."""
        sensor = create_sensor_model(detector_type="hgcdte_mwir")
        assert sensor.params.fpa.detector.detector_type == DetectorType.HGCDTE_MWIR

    def test_custom_resolution(self) -> None:
        """Should accept custom resolution."""
        sensor = create_sensor_model(resolution=(256, 320))
        assert sensor.params.fpa.geometry.resolution == (256, 320)

    def test_custom_integration_time(self) -> None:
        """Should accept custom integration time."""
        sensor = create_sensor_model(integration_time_s=0.005)
        assert sensor.params.integration_time_s == 0.005


class TestSensorResult:
    """Tests for SensorResult dataclass."""

    def test_creation(self) -> None:
        """Should create result object."""
        dn = np.zeros((64, 64), dtype=np.uint16)
        electrons = np.zeros((64, 64))
        noisy = np.zeros((64, 64))
        result = SensorResult(
            dn=dn,
            electrons=electrons,
            noisy_electrons=noisy,
        )
        assert result.dn.shape == (64, 64)
