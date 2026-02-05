"""
Tests for EOSIM Calibration Module (Stage F).

Tests radiometric calibration, NUC, and pixel corrections.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import tempfile
import os

from eosim.calibration import (
    RadiometricCalibrator,
    CalibrationCurve,
    dn_to_radiance,
    radiance_to_temperature,
    temperature_to_radiance,
    TwoPointNUC,
    MultiPointNUC,
    NUCCoefficients,
    apply_nuc,
    compute_nuc_coefficients,
    BadPixelCorrector,
    FlatFieldCorrector,
    BadPixelMap,
    detect_bad_pixels,
    correct_bad_pixels,
    apply_flat_field,
)


# =============================================================================
# Radiometric Calibration Tests
# =============================================================================


class TestPlanckFunctions:
    """Test Planck function implementations."""

    def test_temperature_to_radiance_scalar(self):
        """Temperature to radiance works for scalar."""
        L = temperature_to_radiance(300, 10.0)
        assert L > 0
        # Typical LWIR value
        assert 5 < L < 15

    def test_temperature_to_radiance_array(self):
        """Temperature to radiance works for array."""
        temps = np.array([280, 300, 320])
        L = temperature_to_radiance(temps, 10.0)
        assert L.shape == (3,)
        assert all(L > 0)
        # Should increase with temperature
        assert L[0] < L[1] < L[2]

    def test_radiance_to_temperature_scalar(self):
        """Radiance to temperature works for scalar."""
        # First convert T to L, then back
        T_original = 300.0
        L = temperature_to_radiance(T_original, 10.0)
        T_recovered = radiance_to_temperature(L, 10.0)
        assert_allclose(T_recovered, T_original, rtol=1e-6)

    def test_radiance_temperature_roundtrip(self):
        """Roundtrip conversion preserves values."""
        temps = np.array([280, 290, 300, 310, 320])
        L = temperature_to_radiance(temps, 10.0)
        T_recovered = radiance_to_temperature(L, 10.0)
        assert_allclose(T_recovered, temps, rtol=1e-6)

    def test_emissivity_scaling(self):
        """Emissivity properly scales radiance."""
        L_full = temperature_to_radiance(300, 10.0, emissivity=1.0)
        L_half = temperature_to_radiance(300, 10.0, emissivity=0.5)
        assert_allclose(L_half, L_full * 0.5)

    def test_dn_to_radiance(self):
        """DN to radiance linear conversion."""
        L = dn_to_radiance(1000, gain=0.01, offset=0.5)
        assert L == 0.01 * 1000 + 0.5


class TestCalibrationCurve:
    """Test CalibrationCurve class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample calibration data."""
        temps = np.array([280, 290, 300, 310, 320])
        # Simulate linear response
        dn = temps * 100 - 20000
        return dn.tolist(), temps.tolist()

    def test_from_blackbody_polynomial(self, sample_data):
        """Polynomial calibration curve creation."""
        dn, temps = sample_data
        curve = CalibrationCurve.from_blackbody_data(dn, temps, degree=1)

        assert curve is not None
        assert curve.method == "polynomial"

    def test_from_blackbody_interpolation(self, sample_data):
        """Interpolation calibration curve creation."""
        dn, temps = sample_data
        curve = CalibrationCurve.from_blackbody_data(dn, temps, method="interpolation")

        assert curve is not None
        assert curve.method == "interpolation"

    def test_dn_to_temperature(self, sample_data):
        """DN to temperature conversion."""
        dn, temps = sample_data
        curve = CalibrationCurve.from_blackbody_data(dn, temps, degree=1)

        # Test at calibration points
        for d, t in zip(dn, temps):
            t_predicted = curve.dn_to_temperature(d)
            assert_allclose(t_predicted, t, rtol=0.01)

    def test_temperature_to_dn(self, sample_data):
        """Temperature to DN conversion."""
        dn, temps = sample_data
        curve = CalibrationCurve.from_blackbody_data(dn, temps, degree=1, method="polynomial")

        # Test roundtrip
        for d, t in zip(dn, temps):
            d_recovered = curve.temperature_to_dn(t)
            assert_allclose(d_recovered, d, rtol=0.05)

    def test_residual_error(self, sample_data):
        """Residual error computation."""
        dn, temps = sample_data
        curve = CalibrationCurve.from_blackbody_data(dn, temps, degree=1)

        error = curve.residual_error(dn, temps)
        # Linear fit to linear data should have very small error
        assert error < 1.0


class TestRadiometricCalibrator:
    """Test RadiometricCalibrator class."""

    @pytest.fixture
    def calibrator(self):
        """Create calibrator fixture."""
        return RadiometricCalibrator(wavelength_um=10.0)

    @pytest.fixture
    def calibration_data(self):
        """Generate calibration data."""
        temps = [280, 290, 300, 310, 320]
        dn = [1000, 3000, 5000, 7000, 9000]
        return dn, temps

    def test_calibrator_creation(self, calibrator):
        """Calibrator can be created."""
        assert calibrator is not None
        assert calibrator.wavelength_um == 10.0
        assert not calibrator.is_calibrated

    def test_fit(self, calibrator, calibration_data):
        """Calibrator can be fitted."""
        dn, temps = calibration_data
        error = calibrator.fit(dn, temps)

        assert calibrator.is_calibrated
        assert error < 5.0  # Less than 5K error

    def test_dn_to_temperature(self, calibrator, calibration_data):
        """DN to temperature after calibration."""
        dn, temps = calibration_data
        calibrator.fit(dn, temps)

        # Test at calibration point
        t = calibrator.dn_to_temperature(5000)
        assert_allclose(t, 300, rtol=0.05)

    def test_dn_to_temperature_array(self, calibrator, calibration_data):
        """DN to temperature with array input."""
        dn, temps = calibration_data
        calibrator.fit(dn, temps)

        dn_array = np.array([[1000, 5000], [5000, 9000]])
        t_array = calibrator.dn_to_temperature(dn_array)

        assert t_array.shape == (2, 2)

    def test_temperature_to_dn(self, calibrator, calibration_data):
        """Temperature to DN conversion."""
        dn, temps = calibration_data
        calibrator.fit(dn, temps)

        d = calibrator.temperature_to_dn(300)
        assert_allclose(d, 5000, rtol=0.1)

    def test_two_point_fit(self, calibrator):
        """Two-point calibration."""
        calibrator.fit_two_point(1000, 9000, 280, 320)

        assert calibrator.is_calibrated

    def test_dn_to_radiance(self, calibrator, calibration_data):
        """DN to radiance conversion."""
        dn, temps = calibration_data
        calibrator.fit(dn, temps)

        L = calibrator.dn_to_radiance(5000)
        assert L > 0

    def test_uncalibrated_error(self, calibrator):
        """Uncalibrated calibrator raises error."""
        with pytest.raises(ValueError):
            calibrator.dn_to_temperature(1000)

    def test_calibration_info(self, calibrator, calibration_data):
        """Calibration info is correct."""
        dn, temps = calibration_data
        calibrator.fit(dn, temps)

        info = calibrator.get_calibration_info()
        assert info["calibrated"] is True
        assert info["wavelength_um"] == 10.0


# =============================================================================
# NUC Tests
# =============================================================================


class TestNUCCoefficients:
    """Test NUCCoefficients dataclass."""

    def test_coefficients_creation(self):
        """Coefficients can be created."""
        gain = np.ones((10, 10))
        offset = np.zeros((10, 10))
        coeffs = NUCCoefficients(gain=gain, offset=offset)

        assert coeffs.shape == (10, 10)

    def test_save_load(self):
        """Coefficients can be saved and loaded."""
        gain = np.random.randn(10, 10)
        offset = np.random.randn(10, 10)
        coeffs = NUCCoefficients(gain=gain, offset=offset, reference_temperature=300.0)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            filepath = f.name

        try:
            coeffs.save(filepath)
            loaded = NUCCoefficients.load(filepath)

            assert_allclose(loaded.gain, gain)
            assert_allclose(loaded.offset, offset)
            assert loaded.reference_temperature == 300.0
        finally:
            os.unlink(filepath)


class TestApplyNUC:
    """Test apply_nuc function."""

    def test_apply_nuc_basic(self):
        """Basic NUC application."""
        raw = np.array([[100, 200], [150, 250]])
        gain = np.ones((2, 2))
        offset = np.ones((2, 2)) * 50
        coeffs = NUCCoefficients(gain=gain, offset=offset)

        corrected = apply_nuc(raw, coeffs)

        # corrected = gain * (raw - offset)
        expected = np.array([[50, 150], [100, 200]])
        assert_allclose(corrected, expected)

    def test_apply_nuc_with_gain(self):
        """NUC with gain scaling."""
        raw = np.array([[100, 200]])
        gain = np.array([[2.0, 0.5]])
        offset = np.zeros((1, 2))
        coeffs = NUCCoefficients(gain=gain, offset=offset)

        corrected = apply_nuc(raw, coeffs)
        expected = np.array([[200, 100]])
        assert_allclose(corrected, expected)


class TestComputeNUCCoefficients:
    """Test compute_nuc_coefficients function."""

    def test_compute_coefficients(self):
        """Coefficient computation from two points."""
        cold = np.array([[100, 200], [150, 250]])
        hot = np.array([[200, 400], [300, 500]])

        coeffs = compute_nuc_coefficients(cold, hot, expected_cold=150, expected_hot=350)

        # Apply coefficients to cold frame
        corrected_cold = apply_nuc(cold, coeffs)
        # All pixels should map to expected_cold
        assert_allclose(corrected_cold, 150, rtol=0.01)


class TestTwoPointNUC:
    """Test TwoPointNUC class."""

    @pytest.fixture
    def nuc(self):
        """Create NUC fixture."""
        return TwoPointNUC()

    @pytest.fixture
    def nonuniform_data(self):
        """Generate non-uniform detector data."""
        np.random.seed(42)
        # Create pixel-dependent gain and offset
        shape = (50, 50)
        pixel_gain = 1 + np.random.randn(*shape) * 0.1
        pixel_offset = np.random.randn(*shape) * 100

        # Simulate raw measurements
        T_cold, T_hot = 293, 323
        cold_signal = T_cold * pixel_gain + pixel_offset
        hot_signal = T_hot * pixel_gain + pixel_offset

        return cold_signal, hot_signal, T_cold, T_hot

    def test_nuc_creation(self, nuc):
        """NUC can be created."""
        assert nuc is not None
        assert not nuc.is_calibrated

    def test_calibrate(self, nuc, nonuniform_data):
        """NUC can be calibrated."""
        cold, hot, T_cold, T_hot = nonuniform_data
        gain, offset = nuc.calibrate(cold, hot, T_cold, T_hot)

        assert nuc.is_calibrated
        assert gain.shape == cold.shape
        assert offset.shape == cold.shape

    def test_apply(self, nuc, nonuniform_data):
        """NUC correction reduces non-uniformity."""
        cold, hot, T_cold, T_hot = nonuniform_data
        nuc.calibrate(cold, hot, T_cold, T_hot)

        # Apply to cold frame
        corrected = nuc.apply(cold)

        # Should be more uniform than original
        original_std = np.std(cold)
        corrected_std = np.std(corrected)
        assert corrected_std < original_std

    def test_residual_nonuniformity(self, nuc, nonuniform_data):
        """Residual non-uniformity is reduced."""
        cold, hot, T_cold, T_hot = nonuniform_data
        nuc.calibrate(cold, hot, T_cold, T_hot)

        corrected = nuc.apply(cold)
        rnu = nuc.compute_residual_nonuniformity(corrected)

        # RNU should be very small after correction
        assert rnu < 1.0  # Less than 1%

    def test_multiple_frames(self, nuc):
        """NUC handles multiple frame averaging."""
        np.random.seed(42)
        shape = (3, 20, 20)  # 3 frames
        cold_frames = np.random.randn(*shape) + 1000
        hot_frames = np.random.randn(*shape) + 2000

        gain, offset = nuc.calibrate(cold_frames, hot_frames)
        assert gain.shape == (20, 20)


class TestMultiPointNUC:
    """Test MultiPointNUC class."""

    @pytest.fixture
    def nuc(self):
        """Create multi-point NUC fixture."""
        return MultiPointNUC(polynomial_order=2)

    def test_multipoint_calibration(self, nuc):
        """Multi-point NUC calibration."""
        np.random.seed(42)
        shape = (20, 20)

        # Create non-linear pixel response
        temps = [280, 290, 300, 310, 320]
        frames = []
        for T in temps:
            # Quadratic response
            response = T + 0.01 * T**2 + np.random.randn(*shape) * 10
            frames.append(response)

        nuc.calibrate(frames, temps)
        assert nuc.is_calibrated

    def test_multipoint_correction(self, nuc):
        """Multi-point NUC reduces non-uniformity."""
        np.random.seed(42)
        shape = (20, 20)

        temps = [280, 290, 300, 310, 320]
        frames = []
        for T in temps:
            response = T + np.random.randn(*shape) * 50
            frames.append(response)

        nuc.calibrate(frames, temps)
        corrected = nuc.apply(frames[2])

        # Should be more uniform
        assert np.std(corrected) < np.std(frames[2])


# =============================================================================
# Corrections Tests
# =============================================================================


class TestBadPixelMap:
    """Test BadPixelMap dataclass."""

    def test_badpixelmap_creation(self):
        """BadPixelMap can be created."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        bad_map = BadPixelMap(mask=mask)

        assert bad_map.n_bad == 1
        assert bad_map.bad_pixel_fraction == 0.01

    def test_save_load(self):
        """BadPixelMap can be saved and loaded."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        bad_map = BadPixelMap(mask=mask)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            filepath = f.name

        try:
            bad_map.save(filepath)
            loaded = BadPixelMap.load(filepath)

            assert_array_equal(loaded.mask, mask)
        finally:
            os.unlink(filepath)


class TestDetectBadPixels:
    """Test detect_bad_pixels function."""

    def test_detect_sigma_method(self):
        """Sigma-based detection works."""
        image = np.ones((50, 50)) * 100
        # Add hot pixel
        image[25, 25] = 500
        # Add dead pixel
        image[10, 10] = 0

        bad_map = detect_bad_pixels(image, method="sigma", threshold_sigma=3.0)

        assert bad_map.mask[25, 25]  # Hot pixel detected
        assert bad_map.mask[10, 10]  # Dead pixel detected

    def test_detect_median_method(self):
        """Median-based detection works."""
        image = np.ones((50, 50)) * 100
        image[25, 25] = 500

        bad_map = detect_bad_pixels(image, method="median", threshold_sigma=3.0)

        assert bad_map.mask[25, 25]

    def test_detect_absolute_method(self):
        """Absolute threshold detection works."""
        image = np.ones((50, 50)) * 100
        image[25, 25] = 500

        bad_map = detect_bad_pixels(
            image, method="absolute", threshold_absolute=(50, 200)
        )

        assert bad_map.mask[25, 25]


class TestCorrectBadPixels:
    """Test correct_bad_pixels function."""

    def test_median_correction(self):
        """Median correction works."""
        image = np.ones((50, 50)) * 100.0
        image[25, 25] = 500.0

        mask = np.zeros((50, 50), dtype=bool)
        mask[25, 25] = True

        corrected = correct_bad_pixels(image, mask, method="median")

        # Bad pixel should now be close to 100
        assert_allclose(corrected[25, 25], 100, rtol=0.01)

    def test_mean_correction(self):
        """Mean correction works."""
        image = np.ones((50, 50)) * 100.0
        image[25, 25] = 500.0

        mask = np.zeros((50, 50), dtype=bool)
        mask[25, 25] = True

        corrected = correct_bad_pixels(image, mask, method="mean")

        assert_allclose(corrected[25, 25], 100, rtol=0.1)

    def test_bilinear_correction(self):
        """Bilinear correction works."""
        image = np.ones((50, 50)) * 100.0
        image[25, 25] = 500.0

        mask = np.zeros((50, 50), dtype=bool)
        mask[25, 25] = True

        corrected = correct_bad_pixels(image, mask, method="bilinear")

        assert_allclose(corrected[25, 25], 100, rtol=0.1)

    def test_no_bad_pixels(self):
        """No change when no bad pixels."""
        image = np.ones((50, 50)) * 100.0
        mask = np.zeros((50, 50), dtype=bool)

        corrected = correct_bad_pixels(image, mask)

        assert_allclose(corrected, image)


class TestBadPixelCorrector:
    """Test BadPixelCorrector class."""

    @pytest.fixture
    def corrector(self):
        """Create corrector fixture."""
        return BadPixelCorrector()

    def test_corrector_creation(self, corrector):
        """Corrector can be created."""
        assert corrector is not None

    def test_detect_and_correct(self, corrector):
        """Full detect and correct workflow."""
        image = np.ones((50, 50)) * 100.0
        image[25, 25] = 500.0

        bad_map = corrector.detect_bad_pixels(image, threshold_sigma=3.0)
        corrected = corrector.correct(image)

        assert bad_map.n_bad >= 1
        assert_allclose(corrected[25, 25], 100, rtol=0.1)

    def test_detect_from_noise(self, corrector):
        """Noise-based detection works."""
        np.random.seed(42)
        frames = [np.random.randn(50, 50) for _ in range(10)]
        # Add noisy pixel
        for f in frames:
            f[25, 25] = np.random.randn() * 100

        bad_map = corrector.detect_from_noise(frames, threshold_sigma=3.0)

        assert bad_map.mask[25, 25]


class TestApplyFlatField:
    """Test apply_flat_field function."""

    def test_flat_field_gain_only(self):
        """Flat field with gain only."""
        image = np.array([[100, 200], [150, 250]])
        gain = np.array([[1.0, 0.5], [2.0, 1.0]])

        corrected = apply_flat_field(image, gain)

        expected = np.array([[100, 100], [300, 250]])
        assert_allclose(corrected, expected)

    def test_flat_field_with_offset(self):
        """Flat field with gain and offset."""
        image = np.array([[110, 210], [160, 260]])
        gain = np.ones((2, 2))
        offset = np.ones((2, 2)) * 10

        corrected = apply_flat_field(image, gain, offset)

        expected = np.array([[100, 200], [150, 250]])
        assert_allclose(corrected, expected)


class TestFlatFieldCorrector:
    """Test FlatFieldCorrector class."""

    @pytest.fixture
    def corrector(self):
        """Create corrector fixture."""
        return FlatFieldCorrector()

    @pytest.fixture
    def nonuniform_response(self):
        """Create non-uniform flat field."""
        np.random.seed(42)
        shape = (50, 50)
        # Create non-uniform response
        pixel_response = 1 + np.random.randn(*shape) * 0.1
        # Simulate flat field at signal level 1000
        flat = 1000 * pixel_response
        return flat, pixel_response

    def test_corrector_creation(self, corrector):
        """Corrector can be created."""
        assert corrector is not None
        assert not corrector.is_calibrated

    def test_calibrate(self, corrector, nonuniform_response):
        """Calibration computes gain map."""
        flat, _ = nonuniform_response
        gain = corrector.calibrate(flat)

        assert corrector.is_calibrated
        assert gain.shape == flat.shape

    def test_apply(self, corrector, nonuniform_response):
        """Correction reduces non-uniformity."""
        flat, pixel_response = nonuniform_response
        corrector.calibrate(flat)

        # Create a scene with the same non-uniformity
        scene = 500 * pixel_response
        corrected = corrector.apply(scene)

        # Should be more uniform
        assert np.std(corrected) < np.std(scene)

    def test_residual_nonuniformity(self, corrector, nonuniform_response):
        """Residual non-uniformity is measured."""
        flat, _ = nonuniform_response
        corrector.calibrate(flat)

        corrected_flat = corrector.apply(flat)
        rnu = corrector.compute_residual_nonuniformity(corrected_flat)

        # Should be very small
        assert rnu < 1.0  # Less than 1%

    def test_multiple_frames(self, corrector):
        """Calibration with multiple frames."""
        np.random.seed(42)
        frames = [np.random.randn(30, 30) + 1000 for _ in range(5)]

        gain = corrector.calibrate(frames)
        assert gain.shape == (30, 30)

    def test_save_load(self, corrector, nonuniform_response):
        """Calibration can be saved and loaded."""
        flat, _ = nonuniform_response
        corrector.calibrate(flat)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            filepath = f.name

        try:
            corrector.save(filepath)

            new_corrector = FlatFieldCorrector()
            new_corrector.load(filepath)

            assert new_corrector.is_calibrated
            assert_allclose(new_corrector.gain_map, corrector.gain_map)
        finally:
            os.unlink(filepath)


# =============================================================================
# Integration Tests
# =============================================================================


class TestCalibrationIntegration:
    """Integration tests for calibration module."""

    def test_full_calibration_pipeline(self):
        """Full calibration workflow."""
        np.random.seed(42)
        shape = (100, 100)

        # Create detector non-uniformity
        pixel_gain = 1 + np.random.randn(*shape) * 0.05
        pixel_offset = np.random.randn(*shape) * 100

        # Add bad pixels
        bad_pixels = [(10, 10), (50, 50), (90, 90)]
        for y, x in bad_pixels:
            pixel_gain[y, x] = 0  # Dead pixel

        # Simulate calibration data
        T_cold, T_hot = 293, 323
        cold_frame = T_cold * pixel_gain + pixel_offset + np.random.randn(*shape) * 5
        hot_frame = T_hot * pixel_gain + pixel_offset + np.random.randn(*shape) * 5

        # 1. Detect bad pixels
        bpc = BadPixelCorrector()
        bad_map = bpc.detect_bad_pixels(cold_frame, threshold_sigma=3.0)
        assert bad_map.n_bad >= len(bad_pixels)

        # 2. Correct bad pixels
        cold_corrected = bpc.correct(cold_frame)
        hot_corrected = bpc.correct(hot_frame)

        # 3. Compute NUC
        nuc = TwoPointNUC()
        gain, offset = nuc.calibrate(cold_corrected, hot_corrected, T_cold, T_hot)

        # 4. Apply NUC to a test frame
        T_test = 310
        test_frame = T_test * pixel_gain + pixel_offset + np.random.randn(*shape) * 5
        test_corrected = bpc.correct(test_frame)
        test_nuc = nuc.apply(test_corrected)

        # 5. Check residual non-uniformity
        rnu = nuc.compute_residual_nonuniformity(test_nuc)
        assert rnu < 5.0  # Less than 5%

    def test_radiometric_calibration_with_nuc(self):
        """Radiometric calibration combined with NUC."""
        np.random.seed(42)
        shape = (50, 50)

        # Create detector response
        temps = [280, 290, 300, 310, 320]
        dn_means = [1000, 3000, 5000, 7000, 9000]

        # Simulate detector with NUC already applied
        frames = []
        for T, dn_mean in zip(temps, dn_means):
            frame = np.ones(shape) * dn_mean + np.random.randn(*shape) * 50
            frames.append(frame)

        # Compute mean DN at each temperature
        mean_dns = [np.mean(f) for f in frames]

        # Fit radiometric calibration
        cal = RadiometricCalibrator(wavelength_um=10.0)
        error = cal.fit(mean_dns, temps)

        # Convert a test frame to temperature
        test_frame = frames[2]  # At 300K
        temp_map = cal.dn_to_temperature(test_frame)

        # Should be close to 300K
        assert_allclose(np.mean(temp_map), 300, rtol=0.05)
