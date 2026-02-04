"""Tests for optics.mtf module."""

import pytest
import numpy as np

from eosim.optics.mtf import (
    MTFResult,
    compute_mtf_from_psf,
    diffraction_mtf,
    detector_mtf,
    motion_mtf,
    jitter_mtf,
    atmospheric_mtf,
    system_mtf,
    compute_system_mtf,
    mtf_frequency_at_threshold,
    mtf50,
    mtf10,
    area_under_mtf,
)
from eosim.optics.psf import GaussianPSF, AiryPSF


class TestMTFFromPSF:
    """Tests for MTF computed from PSF."""

    def test_mtf_at_zero_frequency(self) -> None:
        """MTF at zero frequency should be 1."""
        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 31)
        mtf = compute_mtf_from_psf(psf.kernel)
        assert mtf.mtf_radial[0] == pytest.approx(1.0, rel=0.01)

    def test_mtf_decreasing(self) -> None:
        """MTF should generally decrease with frequency."""
        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 31)
        mtf = compute_mtf_from_psf(psf.kernel)
        # MTF should decrease from 1 at f=0
        assert mtf.mtf_radial[0] > mtf.mtf_radial[-1]

    def test_mtf_positive(self) -> None:
        """MTF values should be positive."""
        psf = AiryPSF().compute(10.0, 2.0, 15.0, 31)
        mtf = compute_mtf_from_psf(psf.kernel)
        assert np.all(mtf.mtf_radial >= 0)

    def test_mtf_bounded(self) -> None:
        """MTF should be bounded between 0 and 1."""
        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 31)
        mtf = compute_mtf_from_psf(psf.kernel)
        assert np.all(mtf.mtf_radial >= 0)
        assert np.all(mtf.mtf_radial <= 1.01)  # Allow small numerical error

    def test_frequency_range(self) -> None:
        """Frequency should range from 0 to Nyquist."""
        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 31)
        mtf = compute_mtf_from_psf(psf.kernel)
        assert mtf.frequencies[0] == 0.0
        assert mtf.frequencies[-1] == pytest.approx(0.5)  # Nyquist

    def test_mtf_at_nyquist_stored(self) -> None:
        """MTF at Nyquist should be computed and stored."""
        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 31)
        mtf = compute_mtf_from_psf(psf.kernel)
        assert 0 <= mtf.mtf_at_nyquist <= 1


class TestDiffractionMTF:
    """Tests for analytical diffraction MTF."""

    def test_mtf_at_zero(self) -> None:
        """MTF at zero frequency should be 1."""
        frequencies = np.array([0.0])
        mtf = diffraction_mtf(frequencies, 10.0, 2.0, 15.0)
        assert mtf[0] == pytest.approx(1.0)

    def test_mtf_at_cutoff(self) -> None:
        """MTF should be zero at cutoff frequency."""
        # Cutoff fc = 1 / (λ × F/#) in cycles/μm
        wavelength = 10.0
        f_number = 2.0
        pixel_pitch = 15.0
        fc_um = 1 / (wavelength * f_number)
        fc_pixel = fc_um * pixel_pitch

        frequencies = np.array([fc_pixel])
        mtf = diffraction_mtf(frequencies, wavelength, f_number, pixel_pitch)
        assert mtf[0] == pytest.approx(0.0)

    def test_mtf_decreases_monotonically(self) -> None:
        """Diffraction MTF should decrease monotonically."""
        frequencies = np.linspace(0, 0.5, 50)
        mtf = diffraction_mtf(frequencies, 10.0, 2.0, 15.0)
        # Should be monotonically decreasing
        diffs = np.diff(mtf)
        assert np.all(diffs <= 0.001)  # Allow small numerical errors


class TestDetectorMTF:
    """Tests for detector sampling MTF."""

    def test_unity_fill_factor(self) -> None:
        """Full fill factor should give sinc response."""
        frequencies = np.linspace(0, 0.5, 50)
        mtf = detector_mtf(frequencies, fill_factor=1.0)
        # sinc(0) = 1
        assert mtf[0] == pytest.approx(1.0)
        # sinc(0.5) = sinc(pi/2)/pi/2 ≈ 0.636
        assert mtf[-1] == pytest.approx(np.abs(np.sinc(0.5)), rel=0.01)

    def test_zero_at_nyquist_frequencies(self) -> None:
        """Detector MTF should have zeros at integer frequencies."""
        # With fill factor 1, MTF = sinc(f), which is zero at f=1, 2, ...
        frequencies = np.array([1.0, 2.0])
        mtf = detector_mtf(frequencies, fill_factor=1.0)
        assert np.allclose(mtf, 0, atol=1e-10)

    def test_reduced_fill_factor(self) -> None:
        """Reduced fill factor should give wider MTF."""
        frequencies = np.linspace(0, 0.5, 50)
        mtf_full = detector_mtf(frequencies, fill_factor=1.0)
        mtf_half = detector_mtf(frequencies, fill_factor=0.25)
        # Smaller effective pixel = wider MTF
        # At high frequencies, should be higher with lower fill factor
        assert mtf_half[-1] > mtf_full[-1]


class TestMotionMTF:
    """Tests for motion blur MTF."""

    def test_zero_motion(self) -> None:
        """Zero motion should give unity MTF."""
        frequencies = np.linspace(0, 0.5, 50)
        mtf = motion_mtf(frequencies, motion_pixels=0.0)
        assert np.allclose(mtf, 1.0)

    def test_motion_reduces_mtf(self) -> None:
        """Motion should reduce MTF at high frequencies."""
        frequencies = np.linspace(0, 0.5, 50)
        mtf_no_motion = motion_mtf(frequencies, 0.0)
        mtf_with_motion = motion_mtf(frequencies, 1.0)
        # MTF at f=0 should still be 1
        assert mtf_with_motion[0] == pytest.approx(1.0)
        # MTF at high frequencies should be lower
        assert np.all(mtf_with_motion[1:] <= mtf_no_motion[1:])


class TestJitterMTF:
    """Tests for jitter MTF."""

    def test_zero_jitter(self) -> None:
        """Zero jitter should give unity MTF."""
        frequencies = np.linspace(0, 0.5, 50)
        mtf = jitter_mtf(frequencies, jitter_sigma_pixels=0.0)
        assert np.allclose(mtf, 1.0)

    def test_jitter_gaussian_decay(self) -> None:
        """Jitter MTF should decay as Gaussian."""
        frequencies = np.linspace(0, 0.5, 50)
        sigma = 0.5
        mtf = jitter_mtf(frequencies, sigma)
        # Check Gaussian form: exp(-2 × (π×f×σ)²)
        expected = np.exp(-2 * (np.pi * frequencies * sigma) ** 2)
        assert np.allclose(mtf, expected)


class TestAtmosphericMTF:
    """Tests for atmospheric turbulence MTF."""

    def test_infinite_r0(self) -> None:
        """Infinite r0 (no turbulence) should give unity MTF."""
        frequencies = np.linspace(0, 0.5, 50)
        mtf = atmospheric_mtf(frequencies, r0_pixels=float('inf'))
        assert np.allclose(mtf, 1.0)

    def test_turbulence_reduces_mtf(self) -> None:
        """Turbulence should reduce MTF."""
        frequencies = np.linspace(0, 0.5, 50)
        mtf = atmospheric_mtf(frequencies, r0_pixels=10.0)
        assert mtf[0] == pytest.approx(1.0)
        assert mtf[-1] < 1.0


class TestSystemMTF:
    """Tests for combined system MTF."""

    def test_product_of_components(self) -> None:
        """System MTF should be product of components."""
        frequencies = np.linspace(0, 0.5, 50)
        mtf_opt = diffraction_mtf(frequencies, 10.0, 2.0, 15.0)
        mtf_det = detector_mtf(frequencies, 1.0)
        mtf_sys = system_mtf(mtf_opt, mtf_det)
        expected = mtf_opt * mtf_det
        assert np.allclose(mtf_sys, expected)

    def test_system_lower_than_components(self) -> None:
        """System MTF should be lower than individual components."""
        frequencies = np.linspace(0, 0.5, 50)
        mtf_opt = diffraction_mtf(frequencies, 10.0, 2.0, 15.0)
        mtf_det = detector_mtf(frequencies, 1.0)
        mtf_sys = system_mtf(mtf_opt, mtf_det)
        # System MTF <= min(individual MTFs) for each frequency
        assert np.all(mtf_sys <= mtf_opt + 1e-10)
        assert np.all(mtf_sys <= mtf_det + 1e-10)

    def test_with_optional_components(self) -> None:
        """Should handle optional motion and jitter."""
        frequencies = np.linspace(0, 0.5, 50)
        mtf_opt = diffraction_mtf(frequencies, 10.0, 2.0, 15.0)
        mtf_det = detector_mtf(frequencies, 1.0)
        mtf_mot = motion_mtf(frequencies, 0.5)
        mtf_jit = jitter_mtf(frequencies, 0.2)

        mtf_sys = system_mtf(mtf_opt, mtf_det, mtf_mot, mtf_jit)
        expected = mtf_opt * mtf_det * mtf_mot * mtf_jit
        assert np.allclose(mtf_sys, expected)


class TestComputeSystemMTF:
    """Tests for convenience compute_system_mtf function."""

    def test_returns_mtf_result(self) -> None:
        """Should return MTFResult with all fields."""
        mtf = compute_system_mtf(
            wavelength_um=10.0,
            f_number=2.0,
            pixel_pitch_um=15.0,
        )
        assert isinstance(mtf, MTFResult)
        assert mtf.frequencies is not None
        assert mtf.mtf_radial is not None
        assert mtf.cutoff_frequency is not None

    def test_includes_motion(self) -> None:
        """Motion should reduce system MTF."""
        mtf_no_motion = compute_system_mtf(10.0, 2.0, 15.0, motion_pixels=0)
        mtf_with_motion = compute_system_mtf(10.0, 2.0, 15.0, motion_pixels=1.0)
        # MTF at Nyquist should be lower with motion
        assert mtf_with_motion.mtf_at_nyquist < mtf_no_motion.mtf_at_nyquist


class TestMTFMetrics:
    """Tests for MTF analysis functions."""

    def test_mtf50(self) -> None:
        """MTF50 should return frequency at 50% MTF."""
        frequencies = np.linspace(0, 0.5, 100)
        # Create simple MTF that crosses 0.5 at known point
        mtf = 1 - 2 * frequencies  # Crosses 0.5 at f=0.25
        f50 = mtf50(frequencies, mtf)
        assert f50 == pytest.approx(0.25, abs=0.01)

    def test_mtf10(self) -> None:
        """MTF10 should return frequency at 10% MTF."""
        frequencies = np.linspace(0, 0.5, 100)
        mtf = 1 - 2 * frequencies  # Crosses 0.1 at f=0.45
        f10 = mtf10(frequencies, mtf)
        assert f10 == pytest.approx(0.45, abs=0.01)

    def test_mtf_frequency_at_threshold(self) -> None:
        """Should interpolate threshold crossing."""
        frequencies = np.linspace(0, 0.5, 100)
        mtf = np.exp(-frequencies * 5)  # Exponential decay
        f30 = mtf_frequency_at_threshold(frequencies, mtf, 0.3)
        # Verify by checking MTF at returned frequency
        idx = np.argmin(np.abs(frequencies - f30))
        assert mtf[idx] == pytest.approx(0.3, abs=0.05)

    def test_area_under_mtf(self) -> None:
        """Area under MTF should be positive and bounded."""
        frequencies = np.linspace(0, 0.5, 100)
        mtf = diffraction_mtf(frequencies, 10.0, 2.0, 15.0)
        area = area_under_mtf(frequencies, mtf)
        # Area should be between 0 and 0.5 (max possible with Nyquist = 0.5)
        assert 0 < area < 0.5

    def test_higher_area_means_sharper(self) -> None:
        """Larger f-number (lower cutoff) should have less area."""
        frequencies = np.linspace(0, 0.5, 100)
        mtf_f2 = diffraction_mtf(frequencies, 10.0, 2.0, 15.0)
        mtf_f4 = diffraction_mtf(frequencies, 10.0, 4.0, 15.0)
        area_f2 = area_under_mtf(frequencies, mtf_f2)
        area_f4 = area_under_mtf(frequencies, mtf_f4)
        # Faster optics (lower f/#) has higher cutoff, more area
        assert area_f2 > area_f4
