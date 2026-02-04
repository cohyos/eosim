"""Tests for sensor.fpa module."""

import pytest
import numpy as np

from eosim.sensor.fpa import (
    FPAGeometry,
    DetectorType,
    DetectorProperties,
    FPAConfig,
    SpectralResponse,
    compute_ifov,
    compute_gsd,
    compute_fov,
    compute_nyquist_frequency,
)


class TestFPAGeometry:
    """Tests for FPAGeometry dataclass."""

    def test_basic_creation(self) -> None:
        """Should create FPA geometry with valid parameters."""
        geom = FPAGeometry(
            width_pixels=640,
            height_pixels=480,
            pixel_pitch_um=15.0,
        )
        assert geom.width_pixels == 640
        assert geom.height_pixels == 480
        assert geom.pixel_pitch_um == 15.0

    def test_resolution_property(self) -> None:
        """Resolution should be (height, width)."""
        geom = FPAGeometry(640, 480, 15.0)
        assert geom.resolution == (480, 640)

    def test_pixel_count(self) -> None:
        """Should compute total pixel count."""
        geom = FPAGeometry(640, 480, 15.0)
        assert geom.pixel_count == 640 * 480

    def test_pixel_area_m2(self) -> None:
        """Should compute pixel area in square meters."""
        geom = FPAGeometry(640, 480, 15.0)
        expected = (15e-6) ** 2
        assert geom.pixel_area_m2 == pytest.approx(expected)

    def test_array_dimensions_mm(self) -> None:
        """Should compute array dimensions in mm."""
        geom = FPAGeometry(640, 480, 15.0)
        assert geom.array_width_mm == pytest.approx(640 * 0.015)
        assert geom.array_height_mm == pytest.approx(480 * 0.015)

    def test_diagonal_mm(self) -> None:
        """Should compute array diagonal."""
        geom = FPAGeometry(640, 480, 15.0)
        width_mm = 640 * 0.015
        height_mm = 480 * 0.015
        expected = np.sqrt(width_mm**2 + height_mm**2)
        assert geom.diagonal_mm == pytest.approx(expected)

    def test_aspect_ratio(self) -> None:
        """Should compute aspect ratio."""
        geom = FPAGeometry(640, 480, 15.0)
        assert geom.aspect_ratio == pytest.approx(640 / 480)

    def test_fill_factor_default(self) -> None:
        """Default fill factor should be 1.0."""
        geom = FPAGeometry(640, 480, 15.0)
        assert geom.fill_factor == 1.0

    def test_fill_factor_custom(self) -> None:
        """Should accept custom fill factor."""
        geom = FPAGeometry(640, 480, 15.0, fill_factor=0.8)
        assert geom.fill_factor == 0.8

    def test_invalid_fill_factor(self) -> None:
        """Should reject invalid fill factor."""
        with pytest.raises(ValueError):
            FPAGeometry(640, 480, 15.0, fill_factor=1.5)
        with pytest.raises(ValueError):
            FPAGeometry(640, 480, 15.0, fill_factor=0.0)


class TestDetectorType:
    """Tests for DetectorType enum."""

    def test_detector_types(self) -> None:
        """Should have all expected detector types."""
        assert DetectorType.SI_CCD.value == "si_ccd"
        assert DetectorType.SI_CMOS.value == "si_cmos"
        assert DetectorType.INGAAS.value == "ingaas"
        assert DetectorType.INSB.value == "insb"
        assert DetectorType.HGCDTE_MWIR.value == "hgcdte_mwir"
        assert DetectorType.HGCDTE_LWIR.value == "hgcdte_lwir"
        assert DetectorType.QWIP.value == "qwip"
        assert DetectorType.MICROBOLOMETER.value == "microbolometer"


class TestDetectorProperties:
    """Tests for DetectorProperties dataclass."""

    def test_from_detector_type_mwir(self) -> None:
        """Should create MWIR detector with defaults."""
        props = DetectorProperties.from_detector_type(DetectorType.HGCDTE_MWIR)
        assert props.detector_type == DetectorType.HGCDTE_MWIR
        assert props.operating_temp_k < 100  # Cooled detector
        assert props.full_well_electrons > 0
        assert props.read_noise_electrons > 0

    def test_from_detector_type_lwir(self) -> None:
        """Should create LWIR detector with defaults."""
        props = DetectorProperties.from_detector_type(DetectorType.HGCDTE_LWIR)
        assert props.detector_type == DetectorType.HGCDTE_LWIR

    def test_from_detector_type_bolometer(self) -> None:
        """Should create microbolometer with defaults."""
        props = DetectorProperties.from_detector_type(DetectorType.MICROBOLOMETER)
        assert props.detector_type == DetectorType.MICROBOLOMETER
        # Bolometers operate at higher temps
        assert props.operating_temp_k > 200

    def test_custom_pixel_pitch(self) -> None:
        """Should accept custom pixel pitch."""
        props = DetectorProperties.from_detector_type(
            DetectorType.HGCDTE_MWIR,
            pixel_pitch_um=20.0,
        )
        assert props.pixel_pitch_um == 20.0

    def test_validation_errors(self) -> None:
        """Should validate parameter ranges."""
        with pytest.raises(ValueError):
            DetectorProperties(
                detector_type=DetectorType.HGCDTE_MWIR,
                full_well_electrons=-100,
            )


class TestFPAConfig:
    """Tests for FPAConfig dataclass."""

    def test_creation(self) -> None:
        """Should create FPA config from components."""
        geom = FPAGeometry(640, 480, 15.0)
        detector = DetectorProperties.from_detector_type(DetectorType.HGCDTE_MWIR)
        config = FPAConfig(geometry=geom, detector=detector)
        assert config.geometry.width_pixels == 640
        assert config.detector.detector_type == DetectorType.HGCDTE_MWIR


class TestSpectralResponse:
    """Tests for SpectralResponse dataclass."""

    def test_creation(self) -> None:
        """Should create spectral response from data."""
        wavelengths = np.array([3.0, 4.0, 5.0])
        response = np.array([0.5, 0.8, 0.5])
        sr = SpectralResponse(wavelengths_um=wavelengths, response=response)
        assert len(sr.wavelengths_um) == 3
        assert len(sr.response) == 3

    def test_interpolation(self) -> None:
        """Should interpolate response at arbitrary wavelengths."""
        wavelengths = np.array([3.0, 4.0, 5.0])
        response = np.array([0.0, 1.0, 0.0])
        sr = SpectralResponse(wavelengths_um=wavelengths, response=response)
        # Peak at 4.0 μm
        assert sr(4.0) == pytest.approx(1.0)
        # Half at 3.5 μm
        assert sr(3.5) == pytest.approx(0.5)

    def test_outside_range(self) -> None:
        """Should return zero outside defined range."""
        wavelengths = np.array([3.0, 4.0, 5.0])
        response = np.array([0.5, 1.0, 0.5])
        sr = SpectralResponse(wavelengths_um=wavelengths, response=response)
        assert sr(2.0) == 0.0
        assert sr(6.0) == 0.0

    def test_peak_wavelength(self) -> None:
        """Should find peak wavelength."""
        wavelengths = np.array([3.0, 4.0, 5.0])
        response = np.array([0.5, 1.0, 0.5])
        sr = SpectralResponse(wavelengths_um=wavelengths, response=response)
        assert sr.peak_wavelength_um == pytest.approx(4.0)


class TestComputeFunctions:
    """Tests for utility compute functions."""

    def test_compute_ifov(self) -> None:
        """Should compute IFOV correctly."""
        # IFOV = pixel_pitch / focal_length
        ifov = compute_ifov(15.0, 100.0)
        expected = (15e-6) / (100e-3)  # 0.15 mrad
        assert ifov == pytest.approx(expected)

    def test_compute_gsd(self) -> None:
        """Should compute GSD correctly."""
        # GSD = IFOV × altitude
        gsd = compute_gsd(15.0, 100.0, 10000.0)
        ifov = (15e-6) / (100e-3)
        expected = ifov * 10000.0
        assert gsd == pytest.approx(expected)

    def test_compute_fov(self) -> None:
        """Should compute FOV correctly."""
        # FOV = 2 × atan(array_size / (2 × focal_length))
        fov = compute_fov(640, 15.0, 100.0)
        array_size_m = 640 * 15e-6
        expected = 2 * np.arctan(array_size_m / (2 * 100e-3))
        assert fov == pytest.approx(expected)

    def test_compute_nyquist_frequency(self) -> None:
        """Should compute Nyquist frequency correctly."""
        # Nyquist = 1 / (2 × pixel_pitch)
        nyq = compute_nyquist_frequency(15.0)
        expected = 1 / (2 * 15e-6)  # cycles/m
        assert nyq == pytest.approx(expected)
