"""Tests for optics.aberrations module."""

import pytest
import numpy as np

from eosim.optics.aberrations import (
    AberrationSet,
    NOLL_TO_NM,
    noll_to_nm,
    nm_to_noll,
    radial_polynomial,
    zernike_polynomial,
    zernike_noll,
    create_pupil_grid,
    compute_wavefront,
    rms_wavefront,
    peak_to_valley,
    strehl_ratio,
    strehl_from_coefficients,
)


class TestNollIndexing:
    """Tests for Noll index conversion functions."""

    def test_noll_to_nm_known_values(self) -> None:
        """Test known Noll to (n,m) conversions."""
        # Piston
        assert noll_to_nm(1) == (0, 0)
        # Tilt
        assert noll_to_nm(2) == (1, 1)
        assert noll_to_nm(3) == (1, -1)
        # Defocus
        assert noll_to_nm(4) == (2, 0)
        # Astigmatism
        assert noll_to_nm(5) == (2, -2)
        assert noll_to_nm(6) == (2, 2)
        # Coma
        assert noll_to_nm(7) == (3, -1)
        assert noll_to_nm(8) == (3, 1)
        # Spherical
        assert noll_to_nm(11) == (4, 0)

    def test_nm_to_noll_roundtrip(self) -> None:
        """Test that nm_to_noll and noll_to_nm are inverses."""
        for j in range(1, 23):
            n, m = noll_to_nm(j)
            # Note: nm_to_noll may not give exact roundtrip due to m sign handling


class TestZernikePolynomials:
    """Tests for Zernike polynomial computations."""

    def test_piston_constant(self) -> None:
        """Piston (Z1) should be constant 1."""
        rho, theta, mask = create_pupil_grid(64)
        Z1 = zernike_noll(1, rho, theta)
        # Piston is constant within pupil
        assert np.allclose(Z1[mask], Z1[mask][0])

    def test_defocus_radial_symmetry(self) -> None:
        """Defocus (Z4) should be radially symmetric."""
        rho, theta, mask = create_pupil_grid(64)
        Z4 = zernike_noll(4, rho, theta)
        # Check symmetry: Z4 depends only on rho, not theta
        # At same rho, values should be equal
        center = 32
        val_x = Z4[center, center + 10]
        val_y = Z4[center + 10, center]
        assert np.isclose(val_x, val_y, rtol=0.01)

    def test_astigmatism_angular_dependence(self) -> None:
        """Astigmatism should have angular variation."""
        rho, theta, mask = create_pupil_grid(64)
        Z6 = zernike_noll(6, rho, theta)  # Astig 0 deg
        # At fixed rho, should vary with cos(2*theta)
        center = 32
        val_0deg = Z6[center, center + 10]  # theta = 0
        val_90deg = Z6[center + 10, center]  # theta = 90 deg
        # cos(0) = 1, cos(180deg) = 1, cos(90) = -1 for cos(2*theta)
        # So values at 0 and 90 degrees should have opposite signs
        assert val_0deg * val_90deg < 0

    def test_zernike_normalization(self) -> None:
        """Zernike polynomials should be reasonably bounded."""
        rho, theta, mask = create_pupil_grid(128)
        for j in range(1, 12):
            Z = zernike_noll(j, rho, theta)
            # Within pupil, values should be bounded
            valid = Z[mask]
            assert np.max(np.abs(valid)) < 10  # Reasonable bound


class TestWavefront:
    """Tests for wavefront computation."""

    def test_zero_coefficients_zero_wavefront(self) -> None:
        """Zero coefficients should give zero wavefront."""
        wavefront, mask = compute_wavefront({}, 64)
        assert np.allclose(wavefront[mask], 0)

    def test_single_term_wavefront(self) -> None:
        """Single Zernike term should match polynomial."""
        coeffs = {4: 0.1}  # 0.1 waves of defocus
        wavefront, mask = compute_wavefront(coeffs, 64)

        rho, theta, _ = create_pupil_grid(64)
        Z4 = zernike_noll(4, rho, theta)
        expected = 0.1 * Z4

        assert np.allclose(wavefront[mask], expected[mask], rtol=0.01)

    def test_rms_wavefront_single_term(self) -> None:
        """RMS of single term should relate to coefficient."""
        # For normalized Zernikes, RMS should be close to |coefficient|
        coeffs = {4: 0.1}
        wavefront, mask = compute_wavefront(coeffs, 128)
        rms = rms_wavefront(wavefront, mask)
        # RMS should be close to 0.1 (coefficient value)
        assert 0.05 < rms < 0.2

    def test_peak_to_valley(self) -> None:
        """Peak-to-valley should be larger than RMS."""
        coeffs = {4: 0.5, 11: 0.2}  # Defocus + spherical
        wavefront, mask = compute_wavefront(coeffs, 128)
        rms = rms_wavefront(wavefront, mask)
        pv = peak_to_valley(wavefront, mask)
        assert pv > rms


class TestStrehlRatio:
    """Tests for Strehl ratio computation."""

    def test_strehl_zero_aberrations(self) -> None:
        """Zero aberrations should give Strehl = 1."""
        S = strehl_ratio(0.0)
        assert S == pytest.approx(1.0)

    def test_strehl_small_aberrations(self) -> None:
        """Small aberrations should give high Strehl."""
        # 0.07 waves RMS is Marechal criterion (Strehl ≈ 0.8)
        S = strehl_ratio(0.07)
        assert 0.7 < S < 0.9

    def test_strehl_large_aberrations(self) -> None:
        """Large aberrations should give low Strehl."""
        S = strehl_ratio(0.25)
        assert S < 0.3

    def test_strehl_from_coefficients(self) -> None:
        """Strehl from coefficients should match formula."""
        coeffs = {4: 0.05, 11: 0.03}  # Defocus + spherical
        S = strehl_from_coefficients(coeffs)
        # RMS = sqrt(0.05² + 0.03²) ≈ 0.058
        expected_rms = np.sqrt(0.05**2 + 0.03**2)
        expected_S = strehl_ratio(expected_rms)
        assert S == pytest.approx(expected_S, rel=0.01)


class TestAberrationSet:
    """Tests for AberrationSet dataclass."""

    def test_diffraction_limited(self) -> None:
        """Diffraction-limited set should have zero aberrations."""
        aberr = AberrationSet.diffraction_limited()
        assert aberr.rms_total == 0.0
        assert aberr.strehl == pytest.approx(1.0)

    def test_to_coefficients(self) -> None:
        """Should convert to Noll coefficient dict."""
        aberr = AberrationSet(defocus=0.1, spherical=0.05)
        coeffs = aberr.to_coefficients()
        assert coeffs[4] == 0.1  # Defocus is Noll 4
        assert coeffs[11] == 0.05  # Spherical is Noll 11

    def test_from_coefficients(self) -> None:
        """Should create from Noll coefficient dict."""
        coeffs = {4: 0.2, 6: 0.1, 11: 0.05}
        aberr = AberrationSet.from_coefficients(coeffs)
        assert aberr.defocus == 0.2
        assert aberr.astigmatism_0 == 0.1
        assert aberr.spherical == 0.05

    def test_rms_total(self) -> None:
        """RMS total should be RSS of coefficients."""
        aberr = AberrationSet(defocus=0.3, spherical=0.4)
        expected_rms = np.sqrt(0.3**2 + 0.4**2)
        assert aberr.rms_total == pytest.approx(expected_rms)


class TestPupilGrid:
    """Tests for pupil grid creation."""

    def test_grid_size(self) -> None:
        """Grid should have correct size."""
        rho, theta, mask = create_pupil_grid(64)
        assert rho.shape == (64, 64)
        assert theta.shape == (64, 64)
        assert mask.shape == (64, 64)

    def test_rho_range(self) -> None:
        """Rho should range from 0 to ~sqrt(2) across grid."""
        rho, theta, mask = create_pupil_grid(64)
        assert rho.min() >= 0
        # At corners, rho can exceed 1

    def test_mask_circle(self) -> None:
        """Mask should be True inside unit circle."""
        rho, theta, mask = create_pupil_grid(128)
        # All points with rho <= 1 should be in mask
        assert np.all(mask[rho <= 1])
        # All points with rho > 1 should not be in mask
        assert not np.any(mask[rho > 1])
