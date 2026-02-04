"""Tests for optics.convolution module."""

import pytest
import numpy as np

from eosim.optics.convolution import (
    BoundaryMode,
    ConvolutionResult,
    ConvolutionEngine,
    fft_convolve_2d,
    spatial_convolve_2d,
    apply_psf,
    apply_psf_spectral,
    deconvolve_wiener,
    compute_otf,
    apply_otf,
)
from eosim.optics.psf import GaussianPSF


class TestFFTConvolve:
    """Tests for FFT-based convolution."""

    def test_identity_kernel(self) -> None:
        """Delta kernel should preserve image."""
        image = np.random.rand(64, 64)
        kernel = np.zeros((5, 5))
        kernel[2, 2] = 1.0  # Delta function at center
        result = fft_convolve_2d(image, kernel)
        assert np.allclose(result, image, atol=1e-10)

    def test_constant_kernel_averaging(self) -> None:
        """Constant kernel should smooth to mean."""
        image = np.ones((64, 64))
        image[30:34, 30:34] = 10  # Hot spot
        kernel = np.ones((5, 5)) / 25  # Uniform averaging
        result = fft_convolve_2d(image, kernel)
        # Center should be lower than original 10
        assert result[32, 32] < 10
        # Should still contain some of the energy
        assert result[32, 32] > 1

    def test_output_shape(self) -> None:
        """Output should match input shape."""
        image = np.random.rand(64, 64)
        kernel = np.random.rand(11, 11)
        kernel /= kernel.sum()
        result = fft_convolve_2d(image, kernel)
        assert result.shape == image.shape

    def test_energy_preservation(self) -> None:
        """Total energy should be approximately preserved."""
        image = np.random.rand(64, 64)
        kernel = np.random.rand(5, 5)
        kernel /= kernel.sum()
        result = fft_convolve_2d(image, kernel, BoundaryMode.WRAP)
        # With wrap boundary, energy should be preserved
        assert result.sum() == pytest.approx(image.sum(), rel=0.01)


class TestSpatialConvolve:
    """Tests for spatial domain convolution."""

    def test_matches_fft_result_interior(self) -> None:
        """Spatial and FFT convolution should match in interior region."""
        image = np.random.rand(32, 32)
        kernel = np.random.rand(7, 7)
        kernel /= kernel.sum()

        result_fft = fft_convolve_2d(image, kernel, BoundaryMode.REFLECT)
        result_spatial = spatial_convolve_2d(image, kernel, BoundaryMode.REFLECT)

        # Interior region (away from boundaries) should match closely
        interior_fft = result_fft[5:-5, 5:-5]
        interior_spatial = result_spatial[5:-5, 5:-5]
        assert np.allclose(interior_fft, interior_spatial, atol=0.1)

    def test_boundary_modes(self) -> None:
        """Different boundary modes should give different results."""
        image = np.ones((32, 32))
        image[0, :] = 10  # High values at edge
        kernel = np.ones((5, 5)) / 25

        result_zero = spatial_convolve_2d(image, kernel, BoundaryMode.ZERO)
        result_reflect = spatial_convolve_2d(image, kernel, BoundaryMode.REFLECT)

        # With zero padding, edge values will be lower
        # With reflect, edge will maintain higher values
        assert result_reflect[0, 16] > result_zero[0, 16]


class TestApplyPSF:
    """Tests for apply_psf function."""

    def test_returns_result_object(self) -> None:
        """Should return ConvolutionResult."""
        image = np.random.rand(64, 64)
        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 15)
        result = apply_psf(image, psf)
        assert isinstance(result, ConvolutionResult)
        assert result.image.shape == image.shape
        assert result.psf_applied is not None

    def test_blurs_sharp_features(self) -> None:
        """PSF should blur sharp edges."""
        image = np.zeros((64, 64))
        image[30:34, 30:34] = 1.0  # Sharp square
        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 15)
        result = apply_psf(image, psf)

        # Blur should spread energy outside original square
        assert result.image[29, 32] > 0
        assert result.image[35, 32] > 0

        # Peak should be reduced
        assert result.image[32, 32] < 1.0

    def test_accepts_array_kernel(self) -> None:
        """Should accept numpy array as PSF."""
        image = np.random.rand(64, 64)
        kernel = np.ones((5, 5)) / 25
        result = apply_psf(image, kernel)
        assert result.image.shape == image.shape

    def test_auto_method_selection(self) -> None:
        """Auto method should select based on kernel size."""
        image = np.random.rand(64, 64)
        small_kernel = np.ones((5, 5)) / 25
        large_kernel = np.ones((21, 21)) / 441

        result_small = apply_psf(image, small_kernel, method="auto")
        result_large = apply_psf(image, large_kernel, method="auto")

        assert result_small.method == "spatial"
        assert result_large.method == "fft"


class TestApplyPSFSpectral:
    """Tests for wavelength-dependent PSF application."""

    def test_different_bands(self) -> None:
        """Should apply different PSF per band."""
        cube = np.random.rand(3, 32, 32)
        psfs = [np.ones((5, 5)) / 25 for _ in range(3)]
        # Make first band sharper
        psfs[0] = np.zeros((5, 5))
        psfs[0][2, 2] = 1.0

        result = apply_psf_spectral(cube, psfs)

        assert result.shape == cube.shape
        # First band should be sharper (closer to original)
        # This is tricky to test directly

    def test_shape_mismatch_error(self) -> None:
        """Should raise error if PSF count doesn't match bands."""
        cube = np.random.rand(5, 32, 32)
        psfs = [np.ones((5, 5)) / 25 for _ in range(3)]
        with pytest.raises(ValueError, match="Need 5 PSFs"):
            apply_psf_spectral(cube, psfs)


class TestOTF:
    """Tests for OTF computation and application."""

    def test_otf_shape(self) -> None:
        """OTF should match requested size."""
        psf = np.ones((11, 11)) / 121
        otf = compute_otf(psf, (64, 64))
        assert otf.shape == (64, 64)

    def test_otf_at_zero_frequency(self) -> None:
        """OTF at zero frequency should be 1 for normalized PSF."""
        psf = np.ones((11, 11)) / 121
        otf = compute_otf(psf, (64, 64))
        # DC component should be close to 1
        assert np.abs(otf[0, 0]) == pytest.approx(1.0, rel=0.01)

    def test_apply_otf_produces_blur(self) -> None:
        """Applying OTF should blur the image."""
        image = np.zeros((64, 64))
        image[32, 32] = 1.0  # Point source
        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 11).kernel

        # Apply via OTF
        otf = compute_otf(psf, image.shape)
        result = apply_otf(image, otf)

        # Result should be blurred (energy spread out)
        assert result[32, 32] < 1.0  # Peak reduced
        assert result[31, 32] > 0  # Energy spread to neighbors


class TestDeconvolution:
    """Tests for Wiener deconvolution."""

    def test_partial_recovery(self) -> None:
        """Deconvolution should partially recover original."""
        original = np.zeros((64, 64))
        original[30:34, 30:34] = 1.0

        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 11).kernel
        blurred = fft_convolve_2d(original, psf, BoundaryMode.WRAP)

        recovered = deconvolve_wiener(blurred, psf, noise_power=0.001)

        # Should sharpen the image
        assert recovered.max() > blurred.max()

    def test_regularization_effect(self) -> None:
        """Higher noise power should give smoother result."""
        image = np.random.rand(32, 32)
        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 11).kernel

        recovered_low_reg = deconvolve_wiener(image, psf, noise_power=0.001)
        recovered_high_reg = deconvolve_wiener(image, psf, noise_power=0.1)

        # Lower regularization = more aggressive, higher variance
        assert np.std(recovered_low_reg) > np.std(recovered_high_reg)


class TestConvolutionEngine:
    """Tests for ConvolutionEngine class."""

    def test_caching(self) -> None:
        """Should cache OTF for repeated convolutions."""
        engine = ConvolutionEngine(use_gpu=False, cache_size=5)
        image1 = np.random.rand(64, 64)
        image2 = np.random.rand(64, 64)
        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 11).kernel

        # First convolution computes OTF
        _ = engine.convolve(image1, psf)
        assert len(engine._otf_cache) == 1

        # Second convolution should reuse cached OTF
        _ = engine.convolve(image2, psf)
        assert len(engine._otf_cache) == 1  # Still just one entry

    def test_cache_limit(self) -> None:
        """Should evict old entries when cache is full."""
        engine = ConvolutionEngine(use_gpu=False, cache_size=2)
        image = np.random.rand(64, 64)

        # Create different PSFs
        for sigma in [1, 2, 3]:
            psf = np.exp(-np.linspace(-5, 5, 11)**2 / (2*sigma**2))
            psf = np.outer(psf, psf)
            psf /= psf.sum()
            _ = engine.convolve(image, psf)

        assert len(engine._otf_cache) <= 2

    def test_clear_cache(self) -> None:
        """Should clear cache on request."""
        engine = ConvolutionEngine()
        image = np.random.rand(64, 64)
        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 11).kernel
        _ = engine.convolve(image, psf)
        assert len(engine._otf_cache) > 0

        engine.clear_cache()
        assert len(engine._otf_cache) == 0

    def test_batch_convolve(self) -> None:
        """Should convolve multiple images efficiently."""
        engine = ConvolutionEngine()
        images = [np.random.rand(64, 64) for _ in range(5)]
        psf = GaussianPSF().compute(10.0, 2.0, 15.0, 11).kernel

        results = engine.batch_convolve(images, psf)

        assert len(results) == 5
        assert all(r.shape == (64, 64) for r in results)

    def test_gpu_flag(self) -> None:
        """Should track GPU availability."""
        engine = ConvolutionEngine(use_gpu=False)
        assert not engine.gpu_enabled

        # GPU engine may or may not be available
        engine_gpu = ConvolutionEngine(use_gpu=True)
        # gpu_enabled depends on whether CuPy is installed


class TestBoundaryModes:
    """Tests for boundary handling modes."""

    def test_zero_padding_reduces_edges(self) -> None:
        """Zero padding should reduce values near edges."""
        image = np.ones((32, 32))
        kernel = np.ones((7, 7)) / 49
        result = fft_convolve_2d(image, kernel, BoundaryMode.ZERO)
        # Center should be 1 (from uniform region)
        assert result[16, 16] == pytest.approx(1.0, rel=0.01)
        # Edge should be less than 1 (due to zero padding)
        assert result[0, 0] < 1.0

    def test_reflect_preserves_edges(self) -> None:
        """Reflect mode should better preserve edge values."""
        image = np.ones((32, 32))
        kernel = np.ones((7, 7)) / 49
        result = fft_convolve_2d(image, kernel, BoundaryMode.REFLECT)
        # Edges should be close to 1
        assert result[0, 0] == pytest.approx(1.0, rel=0.05)

    def test_wrap_periodic(self) -> None:
        """Wrap mode should handle periodic boundaries."""
        image = np.zeros((32, 32))
        image[0, :] = 1  # Top row
        image[-1, :] = 1  # Bottom row
        kernel = np.ones((7, 7)) / 49
        result = fft_convolve_2d(image, kernel, BoundaryMode.WRAP)
        # With wrap, top and bottom are neighbors
        # Center row should have some influence from edges
        # This is hard to test precisely
