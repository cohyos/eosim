"""Tests for sensor.quantum_efficiency module."""

import pytest
import numpy as np

from eosim.sensor.quantum_efficiency import (
    QEModel,
    SpectrallyWeightedQE,
    create_qe_model,
    create_flat_qe,
    create_gaussian_qe,
)
from eosim.sensor.fpa import DetectorType


class TestQEModel:
    """Tests for QEModel dataclass."""

    def test_basic_creation(self) -> None:
        """Should create QE model from wavelength/QE arrays."""
        wavelengths = np.array([3.0, 4.0, 5.0])
        qe = np.array([0.5, 0.8, 0.5])
        model = QEModel(wavelengths, qe)
        assert len(model.wavelengths_um) == 3
        assert len(model.qe_values) == 3

    def test_interpolation(self) -> None:
        """Should interpolate QE at arbitrary wavelengths."""
        wavelengths = np.array([3.0, 4.0, 5.0])
        qe = np.array([0.0, 1.0, 0.0])
        model = QEModel(wavelengths, qe)
        # Peak at 4.0 μm
        assert model(4.0) == pytest.approx(1.0)
        # Half at 3.5 μm
        assert model(3.5) == pytest.approx(0.5)

    def test_array_input(self) -> None:
        """Should handle array input."""
        wavelengths = np.array([3.0, 4.0, 5.0])
        qe = np.array([0.5, 0.8, 0.5])
        model = QEModel(wavelengths, qe)
        result = model(np.array([3.0, 4.0, 5.0]))
        assert np.allclose(result, [0.5, 0.8, 0.5])

    def test_outside_range(self) -> None:
        """Should return zero outside defined range."""
        wavelengths = np.array([3.0, 4.0, 5.0])
        qe = np.array([0.5, 0.8, 0.5])
        model = QEModel(wavelengths, qe)
        assert model(2.0) == 0.0
        assert model(6.0) == 0.0

    def test_peak_qe(self) -> None:
        """Should find peak QE."""
        wavelengths = np.array([3.0, 4.0, 5.0])
        qe = np.array([0.5, 0.8, 0.5])
        model = QEModel(wavelengths, qe)
        assert model.peak_qe == pytest.approx(0.8)

    def test_peak_wavelength(self) -> None:
        """Should find peak wavelength."""
        wavelengths = np.array([3.0, 4.0, 5.0])
        qe = np.array([0.5, 0.8, 0.5])
        model = QEModel(wavelengths, qe)
        assert model.peak_wavelength_um == pytest.approx(4.0)

    def test_wavelength_range(self) -> None:
        """Should compute wavelength range where QE > 0."""
        wavelengths = np.array([3.0, 4.0, 5.0])
        qe = np.array([0.5, 0.8, 0.5])
        model = QEModel(wavelengths, qe)
        wl_range = model.wavelength_range
        assert wl_range[0] == pytest.approx(3.0)
        assert wl_range[1] == pytest.approx(5.0)

    def test_integrate(self) -> None:
        """Should integrate QE over wavelength range."""
        wavelengths = np.array([3.0, 4.0, 5.0])
        qe = np.array([0.5, 0.5, 0.5])  # Flat QE
        model = QEModel(wavelengths, qe)
        avg_qe = model.integrate(3.0, 5.0)
        assert avg_qe == pytest.approx(0.5, rel=0.01)

    def test_validation_qe_bounds(self) -> None:
        """Should reject QE values outside 0-1."""
        wavelengths = np.array([3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="QE values must be between"):
            QEModel(wavelengths, np.array([0.5, 1.5, 0.5]))
        with pytest.raises(ValueError, match="QE values must be between"):
            QEModel(wavelengths, np.array([0.5, -0.1, 0.5]))

    def test_validation_length_mismatch(self) -> None:
        """Should reject mismatched array lengths."""
        with pytest.raises(ValueError, match="same length"):
            QEModel(np.array([3.0, 4.0]), np.array([0.5, 0.8, 0.5]))


class TestCreateQEModel:
    """Tests for create_qe_model factory function."""

    def test_mwir_detector(self) -> None:
        """Should create MWIR QE model."""
        model = create_qe_model(DetectorType.HGCDTE_MWIR)
        assert model.detector_type == "hgcdte_mwir"
        # Should have reasonable QE in MWIR band
        assert model(4.0) > 0.5

    def test_lwir_detector(self) -> None:
        """Should create LWIR QE model."""
        model = create_qe_model(DetectorType.HGCDTE_LWIR)
        assert model.detector_type == "hgcdte_lwir"
        # Should have reasonable QE in LWIR band
        assert model(10.0) > 0.5

    def test_visible_detector(self) -> None:
        """Should create visible CCD QE model."""
        model = create_qe_model(DetectorType.SI_CCD)
        assert model.detector_type == "si_ccd"
        # Should have reasonable QE in visible band
        assert model(0.55) > 0.5

    def test_string_input(self) -> None:
        """Should accept string detector type."""
        model = create_qe_model("hgcdte_mwir")
        assert model.detector_type == "hgcdte_mwir"

    def test_unknown_detector(self) -> None:
        """Should raise error for unknown detector."""
        with pytest.raises(ValueError):
            create_qe_model("unknown_type")


class TestCreateFlatQE:
    """Tests for create_flat_qe function."""

    def test_flat_qe(self) -> None:
        """Should create constant QE model."""
        model = create_flat_qe(0.8, 3.0, 5.0)
        assert model(4.0) == pytest.approx(0.8)
        assert model(3.5) == pytest.approx(0.8)

    def test_flat_qe_outside_range(self) -> None:
        """Should be zero outside specified range."""
        model = create_flat_qe(0.8, 3.0, 5.0)
        assert model(2.5) == pytest.approx(0.0)
        assert model(5.5) == pytest.approx(0.0)


class TestCreateGaussianQE:
    """Tests for create_gaussian_qe function."""

    def test_gaussian_peak(self) -> None:
        """Should have peak at specified wavelength."""
        model = create_gaussian_qe(4.0, 0.8, 0.5)
        assert model(4.0) == pytest.approx(0.8, rel=0.01)

    def test_gaussian_fwhm(self) -> None:
        """Should have correct FWHM."""
        peak_qe = 0.8
        fwhm = 0.5
        model = create_gaussian_qe(4.0, peak_qe, fwhm)
        # At FWHM/2 from peak, QE should be ~half of peak
        assert model(4.0 + fwhm / 2) == pytest.approx(peak_qe / 2, rel=0.05)

    def test_gaussian_decay(self) -> None:
        """Should decay to near zero at edges."""
        model = create_gaussian_qe(4.0, 0.8, 0.5)
        # Should be very low at 3σ away
        assert model(4.0 + 1.5) < 0.1


class TestSpectrallyWeightedQE:
    """Tests for SpectrallyWeightedQE class."""

    def test_compute_signal(self) -> None:
        """Should compute signal from spectral irradiance."""
        qe = create_flat_qe(0.8, 3.0, 5.0)
        weighted = SpectrallyWeightedQE(qe, n_spectral_samples=50)

        # Constant irradiance
        def irradiance(w):
            return 1.0

        signal = weighted.compute_signal(
            irradiance,
            wavelength_min_um=3.0,
            wavelength_max_um=5.0,
            pixel_area_m2=1e-8,
            integration_time_s=0.01,
        )
        assert signal > 0

    def test_higher_irradiance_more_signal(self) -> None:
        """Higher irradiance should give more signal."""
        qe = create_flat_qe(0.8, 3.0, 5.0)
        weighted = SpectrallyWeightedQE(qe)

        signal_low = weighted.compute_signal(
            lambda w: 1.0,
            3.0, 5.0, 1e-8, 0.01,
        )
        signal_high = weighted.compute_signal(
            lambda w: 10.0,
            3.0, 5.0, 1e-8, 0.01,
        )
        assert signal_high > signal_low
