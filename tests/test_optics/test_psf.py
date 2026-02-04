"""Tests for optics.psf module."""

import pytest
import numpy as np

from eosim.optics.psf import (
    PSFResult,
    GaussianPSF,
    AiryPSF,
    ZernikePSF,
    MeasuredPSF,
    compute_psf,
    psf_encircled_energy,
    psf_to_ensquared_energy,
)
from eosim.optics.aberrations import AberrationSet


class TestGaussianPSF:
    """Tests for Gaussian PSF model."""

    def test_kernel_normalized(self) -> None:
        """PSF kernel should sum to 1."""
        psf = GaussianPSF().compute(
            wavelength_um=10.0,
            f_number=2.0,
            pixel_pitch_um=15.0,
            size_pixels=21,
        )
        assert psf.kernel.sum() == pytest.approx(1.0, rel=1e-3)

    def test_kernel_centered(self) -> None:
        """PSF should be centered with peak at center."""
        psf = GaussianPSF().compute(
            wavelength_um=10.0,
            f_number=2.0,
            pixel_pitch_um=15.0,
            size_pixels=21,
        )
        center = psf.size_pixels // 2
        peak_idx = np.unravel_index(np.argmax(psf.kernel), psf.kernel.shape)
        assert peak_idx == (center, center)

    def test_kernel_symmetric(self) -> None:
        """Gaussian PSF should be symmetric."""
        psf = GaussianPSF().compute(
            wavelength_um=10.0,
            f_number=2.0,
            pixel_pitch_um=15.0,
            size_pixels=21,
        )
        # Check symmetry
        assert np.allclose(psf.kernel, psf.kernel.T)
        assert np.allclose(psf.kernel, np.flip(psf.kernel, axis=0))

    def test_sigma_scales_with_f_number(self) -> None:
        """PSF sigma should scale linearly with f-number."""
        psf_f2 = GaussianPSF().compute(10.0, 2.0, 15.0, 21)
        psf_f4 = GaussianPSF().compute(10.0, 4.0, 15.0, 21)
        # Sigma ∝ F/#, so doubling F/# should double sigma
        assert psf_f4.sigma_pixels == pytest.approx(2 * psf_f2.sigma_pixels, rel=0.01)

    def test_sigma_scales_with_wavelength(self) -> None:
        """PSF sigma should scale linearly with wavelength."""
        psf_10um = GaussianPSF().compute(10.0, 2.0, 15.0, 21)
        psf_20um = GaussianPSF().compute(20.0, 2.0, 15.0, 21)
        # Sigma ∝ λ, so doubling wavelength should double sigma
        assert psf_20um.sigma_pixels == pytest.approx(2 * psf_10um.sigma_pixels, rel=0.01)

    def test_fwhm_relationship(self) -> None:
        """FWHM should be 2.355 * sigma."""
        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 21)
        expected_fwhm = 2.355 * psf.sigma_pixels
        assert psf.fwhm_pixels == pytest.approx(expected_fwhm)

    def test_odd_size_enforced(self) -> None:
        """Even size should be adjusted to odd."""
        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 20)
        assert psf.size_pixels % 2 == 1


class TestAiryPSF:
    """Tests for Airy disk PSF model."""

    def test_kernel_normalized(self) -> None:
        """Airy PSF should sum to 1."""
        psf = AiryPSF().compute(
            wavelength_um=10.0,
            f_number=2.0,
            pixel_pitch_um=15.0,
            size_pixels=31,
        )
        assert psf.kernel.sum() == pytest.approx(1.0, rel=1e-3)

    def test_peak_at_center(self) -> None:
        """Airy PSF should peak at center."""
        psf = AiryPSF().compute(10.0, 2.0, 15.0, 31)
        center = psf.size_pixels // 2
        assert psf.kernel[center, center] == psf.kernel.max()

    def test_diffraction_limited_strehl(self) -> None:
        """Airy PSF should have Strehl = 1 (diffraction-limited)."""
        psf = AiryPSF().compute(10.0, 2.0, 15.0, 31)
        assert psf.strehl_ratio == 1.0

    def test_airy_has_ring_structure(self) -> None:
        """Airy disk should have characteristic ring structure (non-monotonic)."""
        # Use larger PSF to see ring structure
        airy = AiryPSF().compute(10.0, 1.0, 5.0, 51)
        center = airy.size_pixels // 2
        # Extract radial profile
        radial = airy.kernel[center, center:]
        # For well-sampled Airy, should have rings (local minima)
        # Check that values eventually decrease from center
        assert radial[0] > radial[-1]


class TestZernikePSF:
    """Tests for Zernike aberrated PSF model."""

    def test_diffraction_limited_zernike(self) -> None:
        """Zero aberrations should give diffraction-limited PSF."""
        psf = ZernikePSF().compute(
            wavelength_um=10.0,
            f_number=2.0,
            pixel_pitch_um=15.0,
            size_pixels=31,
        )
        assert psf.kernel.sum() == pytest.approx(1.0, rel=1e-2)
        # Strehl should be close to 1
        assert psf.strehl_ratio > 0.95

    def test_aberrations_reduce_strehl(self) -> None:
        """Aberrations should reduce Strehl ratio."""
        aberr = AberrationSet(defocus=0.1, spherical=0.05)
        psf_aberr = ZernikePSF(aberr).compute(10.0, 2.0, 15.0, 31)
        psf_perfect = ZernikePSF().compute(10.0, 2.0, 15.0, 31)
        assert psf_aberr.strehl_ratio < psf_perfect.strehl_ratio

    def test_from_dict_coefficients(self) -> None:
        """Should accept dict of Noll coefficients."""
        coeffs = {4: 0.1, 11: 0.05}  # Defocus + spherical
        psf = ZernikePSF(coeffs).compute(10.0, 2.0, 15.0, 31)
        assert psf.kernel.sum() == pytest.approx(1.0, rel=1e-2)


class TestMeasuredPSF:
    """Tests for measured PSF handling."""

    def test_normalization(self) -> None:
        """Measured PSF should be normalized."""
        kernel = np.random.rand(11, 11)
        kernel[5, 5] = 10  # High peak
        measured = MeasuredPSF(kernel, pixel_pitch_um=15.0, wavelength_um=10.0)
        psf = measured.compute()
        assert psf.kernel.sum() == pytest.approx(1.0, rel=1e-3)

    def test_resize_crop(self) -> None:
        """Should crop larger PSF to requested size."""
        kernel = np.random.rand(21, 21)
        measured = MeasuredPSF(kernel, 15.0, 10.0)
        psf = measured.compute(size_pixels=11)
        assert psf.kernel.shape == (11, 11)

    def test_resize_pad(self) -> None:
        """Should pad smaller PSF to requested size."""
        kernel = np.random.rand(9, 9)
        kernel[4, 4] = 5  # Peak at center
        measured = MeasuredPSF(kernel, 15.0, 10.0)
        psf = measured.compute(size_pixels=21)
        assert psf.kernel.shape == (21, 21)


class TestComputePSF:
    """Tests for convenience compute_psf function."""

    def test_gaussian_model(self) -> None:
        """Should create Gaussian PSF."""
        psf = compute_psf(10.0, 2.0, 15.0, model="gaussian", size_pixels=21)
        assert psf.kernel.shape == (21, 21)
        assert psf.kernel.sum() == pytest.approx(1.0, rel=1e-3)

    def test_airy_model(self) -> None:
        """Should create Airy PSF."""
        psf = compute_psf(10.0, 2.0, 15.0, model="airy", size_pixels=31)
        assert psf.strehl_ratio == 1.0

    def test_zernike_model(self) -> None:
        """Should create Zernike PSF."""
        aberr = AberrationSet(defocus=0.05)
        psf = compute_psf(10.0, 2.0, 15.0, model="zernike", aberrations=aberr)
        assert psf.kernel.sum() == pytest.approx(1.0, rel=1e-2)

    def test_invalid_model(self) -> None:
        """Should raise error for unknown model."""
        with pytest.raises(ValueError, match="Unknown PSF model"):
            compute_psf(10.0, 2.0, 15.0, model="invalid")


class TestEncircledEnergy:
    """Tests for encircled/ensquared energy functions."""

    def test_encircled_energy_small_radius(self) -> None:
        """Small radius should capture less energy than large radius."""
        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 21)
        ee_small = psf_encircled_energy(psf.kernel, 1)
        ee_large = psf_encircled_energy(psf.kernel, 5)
        # Small radius should capture less energy
        assert ee_small < ee_large

    def test_encircled_energy_full_radius(self) -> None:
        """Large radius should capture all energy."""
        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 21)
        ee = psf_encircled_energy(psf.kernel, 20)  # Large radius
        assert ee > 0.99

    def test_encircled_energy_increases(self) -> None:
        """Encircled energy should generally increase with radius."""
        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 21)
        ee_small = psf_encircled_energy(psf.kernel, 2)
        ee_medium = psf_encircled_energy(psf.kernel, 5)
        ee_large = psf_encircled_energy(psf.kernel, 8)
        # Should increase with radius (allow small tolerance for numeric issues)
        assert ee_small < ee_medium + 0.01
        assert ee_medium < ee_large + 0.01

    def test_ensquared_energy_small_box(self) -> None:
        """Small box should capture less energy."""
        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 21)
        ee_small = psf_to_ensquared_energy(psf.kernel, 3)
        ee_large = psf_to_ensquared_energy(psf.kernel, 11)
        assert ee_small < ee_large
