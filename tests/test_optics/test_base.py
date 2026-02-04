"""Tests for optics.base module."""

import pytest
import numpy as np

from eosim.optics.base import (
    PSFModel,
    OpticsParameters,
    OpticsResult,
    SimpleOpticsModel,
    OpticsQualityMetrics,
    compute_optics_metrics,
    create_optics_model,
)
from eosim.optics.aberrations import AberrationSet


class TestOpticsParameters:
    """Tests for OpticsParameters dataclass."""

    def test_f_number_computation(self) -> None:
        """F-number should be focal_length / aperture."""
        params = OpticsParameters(
            focal_length_mm=100,
            aperture_diameter_mm=50,
            pixel_pitch_um=15.0,
        )
        assert params.f_number == pytest.approx(2.0)

    def test_ifov_computation(self) -> None:
        """IFOV should be pixel_pitch / focal_length."""
        params = OpticsParameters(
            focal_length_mm=100,
            aperture_diameter_mm=50,
            pixel_pitch_um=15.0,
        )
        expected = (15.0 / 1000) / 100  # 0.15 mrad
        assert params.fov_per_pixel_rad == pytest.approx(expected)
        assert params.fov_per_pixel_mrad == pytest.approx(0.15)

    def test_airy_radius(self) -> None:
        """Should compute Airy disk radius in pixels."""
        params = OpticsParameters(
            focal_length_mm=100,
            aperture_diameter_mm=50,  # f/2
            pixel_pitch_um=15.0,
            wavelength_um=10.0,
        )
        # Airy radius = 1.22 × λ × F/# = 1.22 × 10 × 2 = 24.4 μm
        # In pixels: 24.4 / 15 = 1.63
        assert params.airy_radius_pixels == pytest.approx(1.63, rel=0.01)

    def test_diffraction_cutoff(self) -> None:
        """Should compute diffraction cutoff frequency."""
        params = OpticsParameters(
            focal_length_mm=100,
            aperture_diameter_mm=50,  # f/2
            pixel_pitch_um=15.0,
            wavelength_um=10.0,
        )
        # fc = 1 / (λ × F/#) = 1 / (10 × 2) = 0.05 cy/μm
        # In cy/pixel: 0.05 × 15 = 0.75 cy/pixel
        assert params.diffraction_cutoff_cy_per_pixel == pytest.approx(0.75)

    def test_diffraction_limited_check(self) -> None:
        """Should identify diffraction-limited system."""
        params_no_aberr = OpticsParameters(
            focal_length_mm=100,
            aperture_diameter_mm=50,
            pixel_pitch_um=15.0,
        )
        assert params_no_aberr.is_diffraction_limited

        params_aberr = OpticsParameters(
            focal_length_mm=100,
            aperture_diameter_mm=50,
            pixel_pitch_um=15.0,
            aberrations=AberrationSet(defocus=0.2),
        )
        assert not params_aberr.is_diffraction_limited

    def test_validation_errors(self) -> None:
        """Should raise errors for invalid parameters."""
        with pytest.raises(ValueError, match="Focal length must be positive"):
            OpticsParameters(
                focal_length_mm=-100,
                aperture_diameter_mm=50,
                pixel_pitch_um=15.0,
            )

        with pytest.raises(ValueError, match="Transmittance must be between"):
            OpticsParameters(
                focal_length_mm=100,
                aperture_diameter_mm=50,
                pixel_pitch_um=15.0,
                transmittance=1.5,
            )


class TestSimpleOpticsModel:
    """Tests for SimpleOpticsModel class."""

    @pytest.fixture
    def optics(self) -> SimpleOpticsModel:
        """Create default optics model for testing."""
        params = OpticsParameters(
            focal_length_mm=100,
            aperture_diameter_mm=50,
            pixel_pitch_um=15.0,
            wavelength_um=10.0,
        )
        return SimpleOpticsModel(params)

    def test_compute_psf(self, optics: SimpleOpticsModel) -> None:
        """Should compute PSF."""
        psf = optics.compute_psf()
        assert psf.kernel is not None
        assert psf.kernel.sum() == pytest.approx(1.0, rel=1e-3)

    def test_compute_mtf(self, optics: SimpleOpticsModel) -> None:
        """Should compute MTF."""
        mtf = optics.compute_mtf()
        assert mtf.frequencies is not None
        assert mtf.mtf_radial[0] == pytest.approx(1.0, rel=0.01)

    def test_radiance_to_irradiance(self, optics: SimpleOpticsModel) -> None:
        """Should convert radiance to irradiance."""
        L = 1.0  # W/m²/sr/μm
        E = optics.radiance_to_irradiance(L)
        # E = π × L × τ / (4 × F#²)
        # With τ=0.9, F#=2: E = π × 1 × 0.9 / (4 × 4) = 0.177
        expected = np.pi * 1.0 * 0.9 / (4 * 4)
        assert E == pytest.approx(expected)

    def test_radiance_to_irradiance_array(self, optics: SimpleOpticsModel) -> None:
        """Should handle array input."""
        L = np.array([[1.0, 2.0], [3.0, 4.0]])
        E = optics.radiance_to_irradiance(L)
        assert E.shape == L.shape
        assert np.all(E > 0)

    def test_apply_full_chain(self, optics: SimpleOpticsModel) -> None:
        """Should apply complete optical chain."""
        radiance = np.ones((64, 64))
        radiance[30:34, 30:34] = 2.0  # Hot spot

        result = optics.apply(radiance)

        assert isinstance(result, OpticsResult)
        assert result.image.shape == (64, 64)
        assert result.psf is not None
        assert result.mtf is not None

    def test_blur_effect(self, optics: SimpleOpticsModel) -> None:
        """Output should be blurred version of input."""
        radiance = np.zeros((64, 64))
        radiance[32, 32] = 1.0  # Point source

        result = optics.apply(radiance)

        # Point should spread out
        assert result.image[32, 32] < radiance[32, 32] * optics.params.transmittance
        # Energy should spread to neighbors
        assert result.image[31, 32] > 0
        assert result.image[33, 32] > 0

    def test_psf_caching(self, optics: SimpleOpticsModel) -> None:
        """PSF should be cached after first computation."""
        psf1 = optics.compute_psf()
        psf2 = optics.compute_psf()
        assert psf1 is psf2  # Same object (cached)

    def test_cache_invalidation(self, optics: SimpleOpticsModel) -> None:
        """Should clear cache on invalidate."""
        psf1 = optics.compute_psf()
        optics.invalidate_cache()
        psf2 = optics.compute_psf()
        assert psf1 is not psf2  # Different object


class TestVignetting:
    """Tests for vignetting model."""

    def test_no_vignetting(self) -> None:
        """Unity vignetting factor should not change image."""
        params = OpticsParameters(
            focal_length_mm=100,
            aperture_diameter_mm=50,
            pixel_pitch_um=15.0,
            vignetting_factor=1.0,
        )
        optics = SimpleOpticsModel(params)
        image = np.ones((64, 64))
        vignetted = optics.apply_vignetting(image)
        assert np.allclose(vignetted, image)

    def test_vignetting_reduces_edges(self) -> None:
        """Vignetting should reduce edge values."""
        params = OpticsParameters(
            focal_length_mm=100,
            aperture_diameter_mm=50,
            pixel_pitch_um=15.0,
            vignetting_factor=0.5,
        )
        optics = SimpleOpticsModel(params)
        image = np.ones((64, 64))
        vignetted = optics.apply_vignetting(image)

        # Center should be unchanged (or close to it)
        assert vignetted[32, 32] == pytest.approx(1.0, rel=0.1)
        # Corners should be reduced
        assert vignetted[0, 0] < 1.0
        assert vignetted[0, 0] >= 0.5  # Minimum is vignetting_factor


class TestPSFModels:
    """Tests for different PSF model selections."""

    def test_gaussian_model(self) -> None:
        """Should use Gaussian PSF when specified."""
        params = OpticsParameters(
            focal_length_mm=100,
            aperture_diameter_mm=50,
            pixel_pitch_um=15.0,
            psf_model=PSFModel.GAUSSIAN,
        )
        optics = SimpleOpticsModel(params)
        psf = optics.compute_psf()
        # Gaussian should have Strehl = 1 (by definition of this simple model)
        assert psf.strehl_ratio == 1.0

    def test_airy_model(self) -> None:
        """Should use Airy PSF when specified."""
        params = OpticsParameters(
            focal_length_mm=100,
            aperture_diameter_mm=50,
            pixel_pitch_um=15.0,
            psf_model=PSFModel.AIRY,
        )
        optics = SimpleOpticsModel(params)
        psf = optics.compute_psf()
        assert psf.strehl_ratio == 1.0

    def test_zernike_model(self) -> None:
        """Should use Zernike PSF when specified."""
        params = OpticsParameters(
            focal_length_mm=100,
            aperture_diameter_mm=50,
            pixel_pitch_um=15.0,
            psf_model=PSFModel.ZERNIKE,
            aberrations=AberrationSet(defocus=0.05),
        )
        optics = SimpleOpticsModel(params)
        psf = optics.compute_psf()
        # Should have reduced Strehl due to aberrations
        assert psf.strehl_ratio < 1.0


class TestOpticsQualityMetrics:
    """Tests for optical quality metrics computation."""

    def test_compute_metrics(self) -> None:
        """Should compute all quality metrics."""
        optics = create_optics_model(
            focal_length_mm=100,
            aperture_diameter_mm=50,
            pixel_pitch_um=15.0,
        )
        metrics = compute_optics_metrics(optics)

        assert isinstance(metrics, OpticsQualityMetrics)
        assert 0 <= metrics.strehl_ratio <= 1
        assert 0 <= metrics.mtf_at_nyquist <= 1
        assert metrics.mtf50 > 0
        assert metrics.mtf10 > metrics.mtf50  # MTF10 at higher frequency
        assert metrics.psf_fwhm > 0

    def test_diffraction_limited_metrics(self) -> None:
        """Diffraction-limited system should have high Strehl."""
        optics = create_optics_model(
            focal_length_mm=100,
            aperture_diameter_mm=50,
            pixel_pitch_um=15.0,
            model="airy",
        )
        metrics = compute_optics_metrics(optics)
        assert metrics.strehl_ratio == pytest.approx(1.0)

    def test_aberrated_metrics(self) -> None:
        """Aberrated system should have reduced metrics."""
        aberr = AberrationSet(defocus=0.15, spherical=0.1)
        optics = create_optics_model(
            focal_length_mm=100,
            aperture_diameter_mm=50,
            pixel_pitch_um=15.0,
            model="zernike",
            aberrations=aberr,
        )
        metrics = compute_optics_metrics(optics)
        assert metrics.strehl_ratio < 0.8


class TestCreateOpticsModel:
    """Tests for factory function."""

    def test_basic_creation(self) -> None:
        """Should create optics model with defaults."""
        optics = create_optics_model(
            focal_length_mm=100,
            aperture_diameter_mm=50,
            pixel_pitch_um=15.0,
        )
        assert isinstance(optics, SimpleOpticsModel)
        assert optics.params.f_number == 2.0

    def test_with_aberrations(self) -> None:
        """Should create model with aberrations."""
        aberr = AberrationSet(spherical=0.1)
        optics = create_optics_model(
            focal_length_mm=100,
            aperture_diameter_mm=50,
            pixel_pitch_um=15.0,
            aberrations=aberr,
        )
        assert optics.params.aberrations is not None

    def test_model_selection(self) -> None:
        """Should select correct PSF model."""
        optics_gauss = create_optics_model(100, 50, 15.0, model="gaussian")
        optics_airy = create_optics_model(100, 50, 15.0, model="airy")

        assert optics_gauss.params.psf_model == PSFModel.GAUSSIAN
        assert optics_airy.params.psf_model == PSFModel.AIRY
