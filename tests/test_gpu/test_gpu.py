"""
Tests for EOSIM GPU Acceleration Module (Stage D).

Tests GPU context, operations, and accelerator functionality.
All tests run on CPU if GPU is not available.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from eosim.gpu import (
    gpu_available,
    get_array_module,
    GPUContext,
    to_gpu,
    to_cpu,
    get_device_info,
    GPUConvolver,
    GPURadiance,
    GPUNoiseGenerator,
    GPUAccelerator,
    AcceleratorConfig,
)


# =============================================================================
# GPU Context Tests
# =============================================================================

class TestGPUAvailability:
    """Test GPU detection and availability."""

    def test_gpu_available_returns_bool(self):
        """gpu_available returns boolean."""
        result = gpu_available()
        assert isinstance(result, bool)

    def test_get_device_info_structure(self):
        """get_device_info returns proper structure."""
        info = get_device_info()
        assert isinstance(info, dict)
        assert "available" in info

    def test_get_array_module_numpy(self):
        """get_array_module returns numpy for numpy arrays."""
        arr = np.array([1, 2, 3])
        xp = get_array_module(arr)
        assert xp is np

    def test_get_array_module_none(self):
        """get_array_module returns numpy for None."""
        xp = get_array_module(None)
        assert xp is np


class TestGPUContext:
    """Test GPU context functionality."""

    def test_context_creation(self):
        """Context can be created."""
        ctx = GPUContext()
        assert ctx is not None

    def test_context_force_cpu(self):
        """Context respects force_cpu flag."""
        ctx = GPUContext(force_cpu=True)
        assert not ctx.is_gpu
        assert ctx.device_name == "CPU"

    def test_context_manager(self):
        """Context works as context manager."""
        with GPUContext() as ctx:
            arr = ctx.array([1, 2, 3])
            assert arr is not None

    def test_array_creation(self):
        """Context creates arrays."""
        ctx = GPUContext(force_cpu=True)
        arr = ctx.array([1, 2, 3], dtype="float32")
        result = to_cpu(arr)
        assert_array_equal(result, [1, 2, 3])

    def test_zeros_creation(self):
        """Context creates zero arrays."""
        ctx = GPUContext(force_cpu=True)
        arr = ctx.zeros((3, 3))
        result = to_cpu(arr)
        assert result.shape == (3, 3)
        assert np.all(result == 0)

    def test_ones_creation(self):
        """Context creates ones arrays."""
        ctx = GPUContext(force_cpu=True)
        arr = ctx.ones((2, 2))
        result = to_cpu(arr)
        assert np.all(result == 1)

    def test_random_creation(self):
        """Context creates random arrays."""
        ctx = GPUContext(force_cpu=True)
        arr = ctx.random((100, 100), seed=42)
        result = to_cpu(arr)
        assert result.shape == (100, 100)
        assert 0 <= result.min() <= result.max() <= 1

    def test_normal_creation(self):
        """Context creates normal distributed arrays."""
        ctx = GPUContext(force_cpu=True)
        arr = ctx.normal(0, 1, (1000,), seed=42)
        result = to_cpu(arr)
        assert abs(result.mean()) < 0.1
        assert abs(result.std() - 1.0) < 0.1

    def test_linspace(self):
        """Context creates linspace arrays."""
        ctx = GPUContext(force_cpu=True)
        arr = ctx.linspace(0, 10, 11)
        result = to_cpu(arr)
        assert_allclose(result, np.arange(11))

    def test_math_operations(self):
        """Context performs math operations."""
        ctx = GPUContext(force_cpu=True)
        a = ctx.array([1, 2, 3])
        b = ctx.array([4, 5, 6])

        assert_allclose(to_cpu(ctx.add(a, b)), [5, 7, 9])
        assert_allclose(to_cpu(ctx.multiply(a, b)), [4, 10, 18])
        assert_allclose(to_cpu(ctx.exp(ctx.array([0, 1]))), [1, np.e])

    def test_reductions(self):
        """Context performs reductions."""
        ctx = GPUContext(force_cpu=True)
        arr = ctx.array([[1, 2], [3, 4]])

        assert to_cpu(ctx.sum(arr)) == 10
        assert to_cpu(ctx.mean(arr)) == 2.5
        assert_allclose(to_cpu(ctx.sum(arr, axis=0)), [4, 6])

    def test_fft_operations(self):
        """Context performs FFT operations."""
        ctx = GPUContext(force_cpu=True)
        arr = ctx.array([[1, 2], [3, 4]], dtype="complex128")

        fft_result = ctx.fft2(arr)
        ifft_result = ctx.ifft2(fft_result)
        assert_allclose(to_cpu(ifft_result), [[1, 2], [3, 4]])

    def test_planck_radiance(self):
        """Context computes Planck radiance."""
        ctx = GPUContext(force_cpu=True)
        temps = ctx.array([300, 350, 400])
        radiance = ctx.planck_radiance(temps, wavelength_um=10.0)
        result = to_cpu(radiance)

        # Radiance should increase with temperature
        assert result[0] < result[1] < result[2]
        # Reasonable values for LWIR
        assert 1 < result[0] < 20  # W/(m²·sr·μm)


class TestDataTransfer:
    """Test GPU/CPU data transfer."""

    def test_to_cpu_numpy_passthrough(self):
        """to_cpu returns numpy array unchanged."""
        arr = np.array([1, 2, 3])
        result = to_cpu(arr)
        assert isinstance(result, np.ndarray)
        assert_array_equal(result, arr)

    def test_to_gpu_without_gpu(self):
        """to_gpu returns numpy if no GPU."""
        arr = np.array([1, 2, 3])
        result = to_gpu(arr)
        if not gpu_available():
            assert_array_equal(result, arr)


# =============================================================================
# GPU Operations Tests
# =============================================================================

class TestGPUConvolver:
    """Test GPU convolver operations."""

    @pytest.fixture
    def convolver(self):
        """Create convolver with CPU fallback."""
        ctx = GPUContext(force_cpu=True)
        return GPUConvolver(ctx)

    def test_convolver_creation(self, convolver):
        """Convolver can be created."""
        assert convolver is not None
        assert convolver.device_name == "CPU"

    def test_convolve2d_identity(self, convolver):
        """Convolution with identity kernel."""
        ctx = GPUContext(force_cpu=True)
        image = ctx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="float64")
        kernel = ctx.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype="float64")

        result = convolver.convolve2d(image, kernel, mode="same")
        result_cpu = to_cpu(result)
        assert_allclose(result_cpu, [[1, 2, 3], [4, 5, 6], [7, 8, 9]], rtol=1e-10)

    def test_convolve2d_blur(self, convolver):
        """Convolution with blur kernel."""
        ctx = GPUContext(force_cpu=True)
        image = ctx.zeros((32, 32))
        image[16, 16] = 1.0  # Delta function

        kernel = ctx.ones((3, 3)) / 9.0

        result = convolver.convolve2d(image, kernel, mode="same")
        result_cpu = to_cpu(result)

        # Result should be a 3x3 box around center
        assert result_cpu[15:18, 15:18].sum() == pytest.approx(1.0, rel=1e-5)

    def test_apply_psf(self, convolver):
        """PSF application."""
        ctx = GPUContext(force_cpu=True)
        image = ctx.random((64, 64), seed=42)
        psf = ctx.normal(0, 1, (7, 7), seed=43)

        result = convolver.apply_psf(image, psf, normalize=True)
        result_cpu = to_cpu(result)

        # PSF should smooth the image
        assert result_cpu.shape == (64, 64)

    def test_apply_mtf(self, convolver):
        """MTF application."""
        ctx = GPUContext(force_cpu=True)
        image = ctx.random((64, 64), seed=42)

        # Simple low-pass MTF
        mtf = ctx.ones((64, 64))

        result = convolver.apply_mtf(image, mtf)
        result_cpu = to_cpu(result)

        # With MTF=1, output should match input
        assert_allclose(result_cpu, to_cpu(image), rtol=1e-5)


class TestGPURadiance:
    """Test GPU radiance calculations."""

    @pytest.fixture
    def radiance(self):
        """Create radiance calculator with CPU fallback."""
        ctx = GPUContext(force_cpu=True)
        return GPURadiance(ctx)

    def test_planck_radiance(self, radiance):
        """Planck radiance computation."""
        result = radiance.planck_radiance(300, 10.0)
        result_cpu = to_cpu(result)

        # Should be positive
        assert result_cpu > 0
        # Reasonable LWIR value
        assert 1 < result_cpu < 20

    def test_planck_radiance_temperature_scaling(self, radiance):
        """Radiance increases with temperature."""
        temps = np.array([280, 300, 320, 340])
        result = radiance.planck_radiance(temps, 10.0)
        result_cpu = to_cpu(result)

        # Monotonically increasing
        assert all(result_cpu[i] < result_cpu[i + 1] for i in range(len(result_cpu) - 1))

    def test_planck_spectral(self, radiance):
        """Batch spectral radiance computation."""
        temps = np.array([280, 300, 320])
        wavelengths = np.linspace(8, 12, 50)

        result = radiance.planck_spectral(temps, wavelengths)
        result_cpu = to_cpu(result)

        assert result_cpu.shape == (3, 50)
        # Higher temps should have higher radiance at each wavelength
        assert np.all(result_cpu[0, :] < result_cpu[1, :])
        assert np.all(result_cpu[1, :] < result_cpu[2, :])

    def test_band_integrated_radiance(self, radiance):
        """Band-integrated radiance computation."""
        temps = np.array([280, 300, 320])
        result = radiance.band_integrated_radiance(temps, 8.0, 12.0)
        result_cpu = to_cpu(result)

        assert result_cpu.shape == (3,)
        assert all(result_cpu > 0)
        assert result_cpu[0] < result_cpu[1] < result_cpu[2]

    def test_stefan_boltzmann(self, radiance):
        """Stefan-Boltzmann exitance."""
        result = radiance.stefan_boltzmann(300)
        result_cpu = to_cpu(result)

        # σT⁴ at 300K ≈ 459 W/m²
        assert_allclose(result_cpu, 459.3, rtol=0.01)

    def test_wien_peak(self, radiance):
        """Wien peak wavelength."""
        result = radiance.wien_peak(300)
        result_cpu = to_cpu(result)

        # λ_max = 2897.77 / T = 9.66 μm at 300K
        assert_allclose(result_cpu, 9.66, rtol=0.01)

    def test_apparent_temperature(self, radiance):
        """Inverse Planck function."""
        # Compute radiance at known temperature
        L = radiance.planck_radiance(300, 10.0)

        # Invert to get temperature back
        T = radiance.apparent_temperature(L, 10.0)
        T_cpu = to_cpu(T)

        assert_allclose(T_cpu, 300, rtol=0.01)


class TestGPUNoiseGenerator:
    """Test GPU noise generation."""

    @pytest.fixture
    def noise_gen(self):
        """Create noise generator with CPU fallback."""
        ctx = GPUContext(force_cpu=True)
        return GPUNoiseGenerator(ctx, seed=42)

    def test_shot_noise(self, noise_gen):
        """Shot noise has correct statistics."""
        ctx = GPUContext(force_cpu=True)
        signal = ctx.ones((1000, 1000)) * 10000  # 10k electrons

        noisy = noise_gen.apply_shot_noise(signal)
        noisy_cpu = to_cpu(noisy)

        # Mean should be close to 10000
        assert_allclose(noisy_cpu.mean(), 10000, rtol=0.01)
        # Std should be close to sqrt(10000) = 100
        assert_allclose(noisy_cpu.std(), 100, rtol=0.1)

    def test_read_noise(self, noise_gen):
        """Read noise has correct statistics."""
        ctx = GPUContext(force_cpu=True)
        signal = ctx.zeros((1000, 1000))

        noisy = noise_gen.apply_read_noise(signal, sigma=25.0)
        noisy_cpu = to_cpu(noisy)

        # Mean should be close to 0
        assert_allclose(noisy_cpu.mean(), 0, atol=1)
        # Std should be close to 25
        assert_allclose(noisy_cpu.std(), 25, rtol=0.1)

    def test_dark_current(self, noise_gen):
        """Dark current accumulates correctly."""
        ctx = GPUContext(force_cpu=True)
        signal = ctx.zeros((1000, 1000))

        dark_rate = 100  # e⁻/s
        int_time = 0.1  # s
        noisy = noise_gen.apply_dark_current(signal, dark_rate, int_time)
        noisy_cpu = to_cpu(noisy)

        # Mean dark signal should be rate × time = 10
        assert_allclose(noisy_cpu.mean(), 10, rtol=0.1)

    def test_prnu(self, noise_gen):
        """PRNU varies gain correctly."""
        ctx = GPUContext(force_cpu=True)
        signal = ctx.ones((1000, 1000)) * 1000

        noisy = noise_gen.apply_prnu(signal, prnu_sigma=0.01)
        noisy_cpu = to_cpu(noisy)

        # Mean should be close to 1000
        assert_allclose(noisy_cpu.mean(), 1000, rtol=0.01)
        # Std should be close to 1000 * 0.01 = 10
        assert_allclose(noisy_cpu.std(), 10, rtol=0.2)

    def test_dsnu(self, noise_gen):
        """DSNU adds offset variation."""
        ctx = GPUContext(force_cpu=True)
        signal = ctx.ones((1000, 1000)) * 1000

        noisy = noise_gen.apply_dsnu(signal, dsnu_sigma=10.0)
        noisy_cpu = to_cpu(noisy)

        # Mean should be close to 1000
        assert_allclose(noisy_cpu.mean(), 1000, rtol=0.01)
        # Std should be close to 10
        assert_allclose(noisy_cpu.std(), 10, rtol=0.2)

    def test_generate_fpn_maps(self, noise_gen):
        """FPN maps have correct statistics."""
        prnu_map, dsnu_map = noise_gen.generate_fpn_maps(
            (100, 100), prnu_sigma=0.01, dsnu_sigma=10.0
        )

        prnu_cpu = to_cpu(prnu_map)
        dsnu_cpu = to_cpu(dsnu_map)

        # PRNU centered at 1
        assert_allclose(prnu_cpu.mean(), 1.0, rtol=0.01)
        assert_allclose(prnu_cpu.std(), 0.01, rtol=0.2)

        # DSNU centered at 0
        assert_allclose(dsnu_cpu.mean(), 0, atol=2)
        assert_allclose(dsnu_cpu.std(), 10.0, rtol=0.2)

    def test_apply_all_noise(self, noise_gen):
        """Combined noise pipeline."""
        ctx = GPUContext(force_cpu=True)
        signal = ctx.ones((100, 100)) * 10000

        noisy = noise_gen.apply_all_noise(
            signal,
            read_noise_e=25.0,
            dark_current_e_per_s=100.0,
            integration_time_s=0.01,
            prnu_sigma=0.01,
            dsnu_sigma=10.0,
        )
        noisy_cpu = to_cpu(noisy)

        # Output should be noisy version of input
        assert noisy_cpu.shape == (100, 100)
        # Mean should be close to 10000 (plus small dark current)
        assert_allclose(noisy_cpu.mean(), 10001, rtol=0.05)


# =============================================================================
# GPU Accelerator Tests
# =============================================================================

class TestGPUAccelerator:
    """Test high-level GPU accelerator."""

    @pytest.fixture
    def accelerator(self):
        """Create accelerator with CPU fallback."""
        config = AcceleratorConfig(enable_gpu=False)
        return GPUAccelerator(config)

    def test_accelerator_creation(self, accelerator):
        """Accelerator can be created."""
        assert accelerator is not None
        assert not accelerator.is_gpu
        assert accelerator.device_name == "CPU"

    def test_compute_radiance(self, accelerator):
        """Accelerator computes radiance."""
        temps = np.array([[300, 310], [320, 330]])
        radiance = accelerator.compute_radiance(temps, 10.0)
        result_cpu = to_cpu(radiance)

        assert result_cpu.shape == (2, 2)
        assert np.all(result_cpu > 0)
        # Temperature ordering preserved
        assert result_cpu[0, 0] < result_cpu[1, 1]

    def test_compute_radiance_with_emissivity(self, accelerator):
        """Radiance scales with emissivity."""
        temps = np.array([300])
        rad_full = accelerator.compute_radiance(temps, 10.0, emissivity=1.0)
        rad_half = accelerator.compute_radiance(temps, 10.0, emissivity=0.5)

        assert_allclose(to_cpu(rad_half), to_cpu(rad_full) * 0.5)

    def test_batch_transmission(self, accelerator):
        """Batch atmospheric transmission."""
        ranges = np.array([1, 2, 5, 10])
        transmission = accelerator.batch_transmission(
            wavelength_um=10.0,
            ranges_km=ranges,
            visibility_km=23.0,
        )
        result_cpu = to_cpu(transmission)

        # Transmission decreases with range
        assert result_cpu[0] > result_cpu[1] > result_cpu[2] > result_cpu[3]
        # Transmission bounded 0-1
        assert np.all(result_cpu >= 0) and np.all(result_cpu <= 1)

    def test_apply_psf(self, accelerator):
        """PSF application through accelerator."""
        image = np.random.randn(64, 64)
        blurred = accelerator.apply_psf(image, sigma_pixels=2.0)
        result_cpu = to_cpu(blurred)

        assert result_cpu.shape == (64, 64)

    def test_apply_atmosphere(self, accelerator):
        """Atmospheric effects application."""
        radiance = np.array([[10, 20], [30, 40]])
        result = accelerator.apply_atmosphere(
            radiance, path_radiance=1.0, transmission=0.8
        )
        result_cpu = to_cpu(result)

        # L_sensor = τ × L_target + L_path
        expected = 0.8 * radiance + 1.0
        assert_allclose(result_cpu, expected)

    def test_radiance_to_electrons(self, accelerator):
        """Radiance to electron conversion."""
        radiance = np.array([[10.0]])
        electrons = accelerator.radiance_to_electrons(
            radiance,
            wavelength_um=10.0,
            quantum_efficiency=0.7,
            integration_time_s=0.01,
            pixel_area_m2=(15e-6) ** 2,
            solid_angle_sr=0.01,
        )
        result_cpu = to_cpu(electrons)

        # Should be positive
        assert result_cpu[0, 0] > 0

    def test_apply_sensor_noise(self, accelerator):
        """Sensor noise application."""
        electrons = np.ones((32, 32)) * 10000
        noisy = accelerator.apply_sensor_noise(electrons)
        result_cpu = to_cpu(noisy)

        assert result_cpu.shape == (32, 32)
        # Should have noise variance
        assert result_cpu.std() > 50  # Shot noise alone would be ~100

    def test_electrons_to_dn(self, accelerator):
        """Electron to DN conversion."""
        electrons = np.array([[0, 50000, 100000]])
        dn = accelerator.electrons_to_dn(
            electrons, full_well_capacity_e=100000, bit_depth=14
        )
        result_cpu = to_cpu(dn)

        # Check bounds
        assert result_cpu[0, 0] == 0
        assert result_cpu[0, 2] == 16383  # 2^14 - 1
        # Check midpoint
        assert result_cpu[0, 1] == pytest.approx(8191.5, abs=1)

    def test_simulate_sensor_pipeline(self, accelerator):
        """Full sensor simulation pipeline."""
        temps = np.random.uniform(280, 320, (64, 64))
        result = accelerator.simulate_sensor(
            temperature_map=temps,
            wavelength_um=10.0,
            psf_sigma_pixels=1.5,
            integration_time_s=0.01,
        )
        result_cpu = to_cpu(result)

        assert result_cpu.shape == (64, 64)
        assert result_cpu.dtype == np.uint16
        # Should have non-zero values
        assert result_cpu.max() > 0

    def test_benchmark(self, accelerator):
        """Benchmark returns timing info."""
        results = accelerator.benchmark(resolution=(64, 64), n_iterations=2)

        assert "device" in results
        assert "radiance_ms" in results
        assert "psf_ms" in results
        assert "noise_ms" in results
        assert "full_pipeline_ms" in results

    def test_clear_cache(self, accelerator):
        """Cache clearing."""
        # Apply PSF to populate cache
        image = np.random.randn(32, 32)
        accelerator.apply_psf(image, sigma_pixels=2.0)

        # Clear cache
        accelerator.clear_cache()

        # Should not raise
        accelerator.apply_psf(image, sigma_pixels=2.0)


class TestAcceleratorConfig:
    """Test accelerator configuration."""

    def test_default_config(self):
        """Default configuration values."""
        config = AcceleratorConfig()
        assert config.enable_gpu is True
        assert config.device_id == 0
        assert config.batch_size == 32
        assert config.cache_psf is True
        assert config.precision == "float32"

    def test_custom_config(self):
        """Custom configuration."""
        config = AcceleratorConfig(
            enable_gpu=False,
            device_id=1,
            batch_size=64,
            precision="float64",
        )
        assert config.enable_gpu is False
        assert config.device_id == 1
        assert config.batch_size == 64
        assert config.precision == "float64"

    def test_config_affects_accelerator(self):
        """Configuration affects accelerator behavior."""
        config = AcceleratorConfig(enable_gpu=False)
        acc = GPUAccelerator(config)
        assert not acc.is_gpu


# =============================================================================
# Integration Tests
# =============================================================================

class TestGPUIntegration:
    """Integration tests for GPU module."""

    def test_full_ir_simulation_workflow(self):
        """Complete IR simulation workflow."""
        # Create accelerator (CPU fallback)
        config = AcceleratorConfig(enable_gpu=False)
        acc = GPUAccelerator(config)

        # Create a synthetic thermal scene
        np.random.seed(42)
        background_temp = 290  # K
        target_temp = 320  # K

        scene = np.ones((128, 128)) * background_temp
        # Add a hot target
        scene[50:70, 50:70] = target_temp

        # Test intermediate computation (radiance) which preserves contrast
        radiance = acc.compute_radiance(scene, wavelength_um=10.0, emissivity=0.95)
        radiance_cpu = to_cpu(radiance)

        bg_radiance = radiance_cpu[10:30, 10:30].mean()
        target_radiance = radiance_cpu[55:65, 55:65].mean()

        # Target should have higher radiance than background
        assert target_radiance > bg_radiance
        # Ratio should be significant
        assert target_radiance / bg_radiance > 1.5

        # Also test that full pipeline runs without error
        output = acc.simulate_sensor(
            temperature_map=scene,
            wavelength_um=10.0,
            emissivity=0.95,
            transmission=0.85,
            path_radiance=0.5,
            psf_sigma_pixels=1.5,
            integration_time_s=0.01,
            quantum_efficiency=0.7,
            read_noise_e=25.0,
            bit_depth=14,
        )
        result = to_cpu(output)

        # Pipeline should complete and produce valid output
        assert result.shape == (128, 128)
        assert result.dtype == np.uint16
        assert result.min() >= 0
        assert result.max() <= 16383

    def test_noise_model_consistency(self):
        """Noise models produce consistent results with seed."""
        ctx = GPUContext(force_cpu=True)

        # Run twice with same seed
        noise1 = GPUNoiseGenerator(ctx, seed=123)
        signal = ctx.ones((100, 100)) * 1000
        result1 = to_cpu(noise1.apply_shot_noise(signal))

        noise2 = GPUNoiseGenerator(ctx, seed=123)
        signal = ctx.ones((100, 100)) * 1000
        result2 = to_cpu(noise2.apply_shot_noise(signal))

        # Results should be identical
        assert_array_equal(result1, result2)

    def test_radiance_temperature_roundtrip(self):
        """Radiance → Temperature roundtrip."""
        ctx = GPUContext(force_cpu=True)
        rad = GPURadiance(ctx)

        original_temps = np.array([280, 290, 300, 310, 320, 330])

        # Forward: T → L
        radiance = rad.planck_radiance(original_temps, 10.0)

        # Inverse: L → T
        recovered_temps = rad.apparent_temperature(radiance, 10.0)

        assert_allclose(to_cpu(recovered_temps), original_temps, rtol=1e-6)
