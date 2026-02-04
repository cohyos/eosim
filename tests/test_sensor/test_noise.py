"""Tests for sensor.noise module."""

import pytest
import numpy as np

from eosim.sensor.noise import (
    NoiseType,
    NoiseParameters,
    NoiseContributions,
    NoiseModel,
    TemporalNoiseModel,
    SpatialNoiseModel,
    shot_noise,
    read_noise,
    dark_current_electrons,
    prnu_map,
    dsnu_map,
    total_noise_variance,
    snr_electrons,
    create_noise_model,
)


class TestNoiseParameters:
    """Tests for NoiseParameters dataclass."""

    def test_default_creation(self) -> None:
        """Should create with default parameters."""
        params = NoiseParameters()
        assert params.read_noise_electrons > 0
        assert params.dark_current_e_per_s > 0
        assert params.prnu_percent > 0
        assert params.dsnu_electrons > 0

    def test_prnu_factor(self) -> None:
        """Should convert PRNU percent to factor."""
        params = NoiseParameters(prnu_percent=1.5)
        assert params.prnu_factor == pytest.approx(0.015)

    def test_dark_current_at_temp(self) -> None:
        """Should scale dark current with temperature."""
        params = NoiseParameters(
            dark_current_e_per_s=1000,
            temperature_k=77,
            dark_current_doubling_temp_k=7,
        )
        # At reference temp
        assert params.dark_current_at_temp(77) == pytest.approx(1000)
        # 7K higher should double
        assert params.dark_current_at_temp(84) == pytest.approx(2000)
        # 7K lower should halve
        assert params.dark_current_at_temp(70) == pytest.approx(500)

    def test_validation_errors(self) -> None:
        """Should reject invalid parameters."""
        with pytest.raises(ValueError):
            NoiseParameters(read_noise_electrons=-10)
        with pytest.raises(ValueError):
            NoiseParameters(dark_current_e_per_s=-100)
        with pytest.raises(ValueError):
            NoiseParameters(prnu_percent=-1)
        with pytest.raises(ValueError):
            NoiseParameters(temperature_k=0)


class TestNoiseContributions:
    """Tests for NoiseContributions dataclass."""

    def test_total_variance(self) -> None:
        """Should sum all variances."""
        contrib = NoiseContributions(
            shot_variance=100,
            dark_variance=50,
            read_variance=25,
            prnu_variance=16,
            dsnu_variance=9,
        )
        expected = 100 + 50 + 25 + 16 + 9
        assert contrib.total_variance == pytest.approx(expected)

    def test_total_noise_electrons(self) -> None:
        """Should compute RMS noise."""
        contrib = NoiseContributions(
            shot_variance=100,
            read_variance=100,
        )
        assert contrib.total_noise_electrons == pytest.approx(np.sqrt(200))


class TestNoiseModel:
    """Tests for NoiseModel class."""

    @pytest.fixture
    def noise_model(self) -> NoiseModel:
        """Create default noise model for testing."""
        params = NoiseParameters(
            read_noise_electrons=30,
            dark_current_e_per_s=1000,
            prnu_percent=1.0,
            dsnu_electrons=50,
        )
        return NoiseModel(params, (64, 64), seed=42)

    def test_prnu_map_shape(self, noise_model: NoiseModel) -> None:
        """PRNU map should match resolution."""
        assert noise_model.prnu_map.shape == (64, 64)

    def test_prnu_map_centered(self, noise_model: NoiseModel) -> None:
        """PRNU map should be centered at 1.0."""
        prnu = noise_model.prnu_map
        assert np.abs(prnu.mean() - 1.0) < 0.1

    def test_dsnu_map_shape(self, noise_model: NoiseModel) -> None:
        """DSNU map should match resolution."""
        assert noise_model.dsnu_map.shape == (64, 64)

    def test_dsnu_map_centered(self, noise_model: NoiseModel) -> None:
        """DSNU map should be centered at 0."""
        dsnu = noise_model.dsnu_map
        assert np.abs(dsnu.mean()) < 10

    def test_apply_adds_noise(self, noise_model: NoiseModel) -> None:
        """Applied signal should differ from input."""
        signal = np.full((64, 64), 10000.0)
        noisy = noise_model.apply(signal, integration_time_s=0.01)
        # Should be different due to noise
        assert not np.allclose(signal, noisy)

    def test_apply_preserves_mean_approximately(self, noise_model: NoiseModel) -> None:
        """Mean should be approximately preserved."""
        signal = np.full((64, 64), 10000.0)
        noisy = noise_model.apply(signal, integration_time_s=0.01)
        # Add dark current to expected mean
        dark = noise_model.params.dark_current_e_per_s * 0.01
        expected_mean = 10000 + dark
        assert np.abs(noisy.mean() - expected_mean) / expected_mean < 0.1

    def test_reproducibility_with_seed(self) -> None:
        """Same seed should give same noise."""
        params = NoiseParameters()
        model1 = NoiseModel(params, (64, 64), seed=42)
        model2 = NoiseModel(params, (64, 64), seed=42)
        signal = np.full((64, 64), 10000.0)
        noisy1 = model1.apply(signal, 0.01)
        noisy2 = model2.apply(signal, 0.01)
        assert np.allclose(noisy1, noisy2)

    def test_compute_contributions(self, noise_model: NoiseModel) -> None:
        """Should compute noise breakdown."""
        contrib = noise_model.compute_contributions(10000, 0.01)
        assert contrib.shot_variance > 0
        assert contrib.read_variance > 0
        assert contrib.prnu_variance > 0
        assert contrib.dsnu_variance > 0

    def test_disable_noise_sources(self, noise_model: NoiseModel) -> None:
        """Should be able to disable individual noise sources."""
        signal = np.full((64, 64), 10000.0)
        # Disable all noise
        noisy = noise_model.apply(
            signal, 0.01,
            include_shot=False,
            include_dark=False,
            include_read=False,
            include_prnu=False,
            include_dsnu=False,
        )
        # Should be unchanged (except floating point)
        assert np.allclose(signal, noisy, rtol=1e-10)


class TestShotNoise:
    """Tests for shot_noise function."""

    def test_poisson_statistics(self) -> None:
        """Shot noise should follow Poisson statistics."""
        rng = np.random.default_rng(42)
        n_samples = 10000
        signal = 1000.0
        samples = np.array([shot_noise(signal, rng) for _ in range(n_samples)])
        # Mean should be close to signal
        assert np.abs(samples.mean() - signal) / signal < 0.05
        # Variance should be close to signal
        assert np.abs(samples.var() - signal) / signal < 0.1

    def test_array_input(self) -> None:
        """Should handle array input."""
        signal = np.full((10, 10), 1000.0)
        noisy = shot_noise(signal)
        assert noisy.shape == (10, 10)


class TestReadNoise:
    """Tests for read_noise function."""

    def test_gaussian_statistics(self) -> None:
        """Read noise should be Gaussian."""
        rng = np.random.default_rng(42)
        noise = read_noise((10000,), sigma_electrons=30, rng=rng)
        # Mean should be zero
        assert np.abs(noise.mean()) < 2
        # Std should be close to sigma
        assert np.abs(noise.std() - 30) < 3


class TestDarkCurrentElectrons:
    """Tests for dark_current_electrons function."""

    def test_mean_value(self) -> None:
        """Mean dark current should match rate × time."""
        rng = np.random.default_rng(42)
        dark = dark_current_electrons(
            dark_current_e_per_s=1000,
            integration_time_s=0.01,
            shape=(1000,),
            rng=rng,
            include_shot_noise=False,
        )
        expected = 1000 * 0.01
        assert np.abs(dark.mean() - expected) < 1

    def test_with_shot_noise(self) -> None:
        """With shot noise, variance should equal mean."""
        rng = np.random.default_rng(42)
        dark = dark_current_electrons(
            dark_current_e_per_s=10000,
            integration_time_s=0.01,
            shape=(10000,),
            rng=rng,
            include_shot_noise=True,
        )
        # Variance should be close to mean (Poisson)
        assert np.abs(dark.var() - dark.mean()) / dark.mean() < 0.1


class TestPRNUMap:
    """Tests for prnu_map function."""

    def test_centered_at_unity(self) -> None:
        """PRNU map should be centered at 1."""
        pmap = prnu_map((100, 100), prnu_percent=1.0)
        assert np.abs(pmap.mean() - 1.0) < 0.05

    def test_correct_spread(self) -> None:
        """PRNU map should have correct std."""
        pmap = prnu_map((1000, 1000), prnu_percent=2.0)
        # Std should be 2% = 0.02
        assert np.abs(pmap.std() - 0.02) < 0.002


class TestDSNUMap:
    """Tests for dsnu_map function."""

    def test_centered_at_zero(self) -> None:
        """DSNU map should be centered at 0."""
        dmap = dsnu_map((100, 100), dsnu_electrons=50)
        assert np.abs(dmap.mean()) < 5

    def test_correct_spread(self) -> None:
        """DSNU map should have correct std."""
        dmap = dsnu_map((1000, 1000), dsnu_electrons=50)
        assert np.abs(dmap.std() - 50) < 5


class TestTotalNoiseVariance:
    """Tests for total_noise_variance function."""

    def test_formula(self) -> None:
        """Should follow noise variance formula."""
        variance = total_noise_variance(
            signal_electrons=10000,
            dark_electrons=100,
            read_noise_electrons=30,
            prnu_percent=1.0,
            dsnu_electrons=50,
        )
        expected = (
            10000 + 100  # shot (signal + dark)
            + 30**2  # read
            + (0.01 * 10000)**2  # PRNU
            + 50**2  # DSNU
        )
        assert variance == pytest.approx(expected)


class TestSNRElectrons:
    """Tests for snr_electrons function."""

    def test_snr_calculation(self) -> None:
        """Should compute SNR correctly."""
        snr = snr_electrons(
            signal_electrons=10000,
            dark_electrons=100,
            read_noise_electrons=30,
        )
        noise = np.sqrt(10000 + 100 + 30**2)
        expected = 10000 / noise
        assert snr == pytest.approx(expected)

    def test_shot_limited(self) -> None:
        """At high signal, should approach sqrt(N)."""
        signal = 1e6
        snr = snr_electrons(signal, 0, 1)  # Minimal other noise
        assert snr == pytest.approx(np.sqrt(signal), rel=0.01)


class TestTemporalNoiseModel:
    """Tests for TemporalNoiseModel class."""

    def test_temporal_variance(self) -> None:
        """Should compute temporal (frame-to-frame) variance."""
        model = TemporalNoiseModel(read_noise_electrons=30, dark_current_e_per_s=1000)
        variance = model.temporal_noise_variance(10000, 0.01)
        # Shot + dark shot + read
        expected = 10000 + 10 + 30**2
        assert variance == pytest.approx(expected)


class TestSpatialNoiseModel:
    """Tests for SpatialNoiseModel class."""

    def test_spatial_variance(self) -> None:
        """Should compute spatial (FPN) variance."""
        model = SpatialNoiseModel(prnu_percent=1.0, dsnu_electrons=50)
        variance = model.spatial_noise_variance(10000)
        expected = (0.01 * 10000)**2 + 50**2
        assert variance == pytest.approx(expected)

    def test_prnu_limited_snr(self) -> None:
        """Should compute PRNU-limited SNR."""
        model = SpatialNoiseModel(prnu_percent=1.0)
        snr = model.prnu_limited_snr(10000)
        # SNR_max = 1 / PRNU_factor = 100
        assert snr == pytest.approx(100)


class TestCreateNoiseModel:
    """Tests for create_noise_model factory function."""

    def test_factory_creation(self) -> None:
        """Should create noise model via factory."""
        model = create_noise_model(
            resolution=(64, 64),
            read_noise_electrons=25,
            dark_current_e_per_s=500,
        )
        assert model.params.read_noise_electrons == 25
        assert model.params.dark_current_e_per_s == 500
