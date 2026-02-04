"""Tests for sensor.metrics module."""

import pytest
import numpy as np

from eosim.sensor.metrics import (
    SensorPerformanceMetrics,
    NEDTBreakdown,
    compute_nedt,
    compute_nedt_from_noise_components,
    compute_nedt_breakdown,
    snr_from_electrons,
    snr_shot_limited,
    snr_read_limited,
    snr_prnu_limited,
    dynamic_range_electrons,
    dynamic_range_temperature,
    detectivity_star,
    blip_detectivity,
    responsivity,
    noise_equivalent_power,
    minimum_resolvable_temperature_difference,
    compute_dL_dT_planck,
    compute_contrast,
    contrast_threshold,
)


class TestSensorPerformanceMetrics:
    """Tests for SensorPerformanceMetrics dataclass."""

    def test_creation(self) -> None:
        """Should create metrics object."""
        metrics = SensorPerformanceMetrics(
            nedt_k=0.05,
            snr=100,
            dynamic_range_db=80,
            well_fill_fraction=0.5,
        )
        assert metrics.nedt_k == 0.05
        assert metrics.snr == 100

    def test_string_representation(self) -> None:
        """Should have readable string format."""
        metrics = SensorPerformanceMetrics(
            nedt_k=0.05,
            snr=100,
            dynamic_range_db=80,
            well_fill_fraction=0.5,
        )
        s = str(metrics)
        assert "NEΔT" in s
        assert "SNR" in s


class TestComputeNEDT:
    """Tests for compute_nedt function."""

    def test_basic_calculation(self) -> None:
        """Should compute NEDT correctly."""
        # NEDT = (noise/signal) × (L/dL_dT)
        signal = 10000
        noise = 100
        L = 1.0
        dL_dT = 0.1
        nedt = compute_nedt(signal, noise, dL_dT, L)
        expected = (noise / signal) * (L / dL_dT)
        assert nedt == pytest.approx(expected)

    def test_higher_noise_higher_nedt(self) -> None:
        """More noise should give higher NEDT."""
        nedt_low = compute_nedt(10000, 50, 0.1, 1.0)
        nedt_high = compute_nedt(10000, 100, 0.1, 1.0)
        assert nedt_high > nedt_low

    def test_higher_signal_lower_nedt(self) -> None:
        """More signal should give lower NEDT."""
        nedt_low = compute_nedt(20000, 100, 0.1, 1.0)
        nedt_high = compute_nedt(10000, 100, 0.1, 1.0)
        assert nedt_low < nedt_high

    def test_zero_signal_returns_inf(self) -> None:
        """Zero signal should return infinity."""
        nedt = compute_nedt(0, 100, 0.1, 1.0)
        assert nedt == float('inf')


class TestComputeNEDTFromNoiseComponents:
    """Tests for compute_nedt_from_noise_components function."""

    def test_combines_noise_sources(self) -> None:
        """Should combine all noise sources."""
        nedt = compute_nedt_from_noise_components(
            signal_electrons=10000,
            shot_noise_electrons=100,
            dark_noise_electrons=10,
            read_noise_electrons=30,
            prnu_noise_electrons=50,
            dsnu_noise_electrons=20,
            dL_dT=0.1,
            L=1.0,
        )
        total_noise = np.sqrt(100**2 + 10**2 + 30**2 + 50**2 + 20**2)
        expected = compute_nedt(10000, total_noise, 0.1, 1.0)
        assert nedt == pytest.approx(expected)


class TestNEDTBreakdown:
    """Tests for NEDTBreakdown and compute_nedt_breakdown."""

    def test_breakdown_creation(self) -> None:
        """Should create breakdown object."""
        breakdown = compute_nedt_breakdown(
            signal_electrons=10000,
            shot_noise_electrons=100,
            dark_noise_electrons=10,
            read_noise_electrons=30,
            prnu_noise_electrons=50,
            dsnu_noise_electrons=20,
            dL_dT=0.1,
            L=1.0,
        )
        assert breakdown.nedt_shot > 0
        assert breakdown.nedt_read > 0
        assert breakdown.nedt_total > 0

    def test_dominant_source(self) -> None:
        """Should identify dominant noise source."""
        breakdown = NEDTBreakdown(
            nedt_shot=0.01,
            nedt_dark=0.005,
            nedt_read=0.05,  # Dominant
            nedt_prnu=0.01,
            nedt_dsnu=0.005,
            nedt_total=0.06,
        )
        assert breakdown.dominant_source() == "read"


class TestSNRFunctions:
    """Tests for SNR calculation functions."""

    def test_snr_from_electrons(self) -> None:
        """Should compute SNR = signal/noise."""
        snr = snr_from_electrons(10000, 100)
        assert snr == pytest.approx(100)

    def test_snr_shot_limited(self) -> None:
        """Should compute sqrt(N) for shot-limited case."""
        snr = snr_shot_limited(10000)
        assert snr == pytest.approx(100)

    def test_snr_read_limited(self) -> None:
        """Should compute N/σ_read for read-limited case."""
        snr = snr_read_limited(1000, 50)
        assert snr == pytest.approx(20)

    def test_snr_prnu_limited(self) -> None:
        """Should compute 100/PRNU% for PRNU-limited case."""
        snr = snr_prnu_limited(1.0)  # 1% PRNU
        assert snr == pytest.approx(100)


class TestDynamicRange:
    """Tests for dynamic range functions."""

    def test_dynamic_range_electrons(self) -> None:
        """Should compute DR in dB."""
        dr = dynamic_range_electrons(100000, 10)
        expected = 20 * np.log10(100000 / 10)
        assert dr == pytest.approx(expected)

    def test_dynamic_range_temperature(self) -> None:
        """Should compute temperature range."""
        dr = dynamic_range_temperature(200, 500)
        assert dr == pytest.approx(300)


class TestDetectivity:
    """Tests for detectivity functions."""

    def test_detectivity_star(self) -> None:
        """Should compute D* correctly."""
        d_star = detectivity_star(
            responsivity_a_per_w=1.0,
            noise_current_a=1e-12,
            detector_area_cm2=1e-4,
            bandwidth_hz=1e6,
        )
        expected = 1.0 * np.sqrt(1e-4 * 1e6) / 1e-12
        assert d_star == pytest.approx(expected)

    def test_blip_detectivity(self) -> None:
        """Should compute BLIP D*."""
        d_star = blip_detectivity(
            wavelength_um=10.0,
            quantum_efficiency=0.7,
            background_photon_flux=1e17,
        )
        # Should be a reasonable value
        assert d_star > 1e8


class TestResponsivity:
    """Tests for responsivity function."""

    def test_responsivity_formula(self) -> None:
        """Should follow R = ηλ/hc formula."""
        from eosim.core.constants import PLANCK_H, SPEED_OF_LIGHT
        r = responsivity(wavelength_um=10.0, quantum_efficiency=0.7)
        # R = η × λ × q / (h × c)
        electron_charge = 1.602176634e-19
        expected = 0.7 * 10e-6 * electron_charge / (PLANCK_H * SPEED_OF_LIGHT)
        assert r == pytest.approx(expected)


class TestNEP:
    """Tests for noise_equivalent_power function."""

    def test_nep_formula(self) -> None:
        """Should compute NEP = sqrt(A×Δf)/D*."""
        nep = noise_equivalent_power(
            detectivity_star=1e11,
            detector_area_cm2=1e-4,
            bandwidth_hz=1e6,
        )
        expected = np.sqrt(1e-4 * 1e6) / 1e11
        assert nep == pytest.approx(expected)


class TestMRTD:
    """Tests for minimum_resolvable_temperature_difference function."""

    def test_mrtd_formula(self) -> None:
        """Should compute MRTD = NEDT/MTF."""
        mrtd = minimum_resolvable_temperature_difference(0.05, 0.5)
        assert mrtd == pytest.approx(0.1)


class TestDLDT:
    """Tests for compute_dL_dT_planck function."""

    def test_positive_value(self) -> None:
        """Should return positive dL/dT."""
        dL_dT = compute_dL_dT_planck(
            temperature_k=300,
            wavelength_um=10.0,
            bandwidth_um=2.0,
        )
        assert dL_dT > 0

    def test_higher_temp_higher_derivative(self) -> None:
        """Higher temperature should have higher derivative (in LWIR)."""
        dL_dT_cold = compute_dL_dT_planck(250, 10.0, 2.0)
        dL_dT_warm = compute_dL_dT_planck(350, 10.0, 2.0)
        # Generally dL/dT increases with T in thermal IR
        # (not always true for all wavelengths, but for LWIR at Earth temps)
        assert dL_dT_warm > dL_dT_cold


class TestContrast:
    """Tests for contrast functions."""

    def test_compute_contrast(self) -> None:
        """Should compute (target-bg)/bg."""
        contrast = compute_contrast(150, 100)
        assert contrast == pytest.approx(0.5)

    def test_contrast_threshold(self) -> None:
        """Should compute minimum detectable contrast."""
        threshold = contrast_threshold(snr=100, confidence=0.95)
        # k/SNR where k ≈ 1.65 for 95%
        from scipy.stats import norm
        k = norm.ppf(0.95)
        assert threshold == pytest.approx(k / 100)
