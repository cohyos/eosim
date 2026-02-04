"""Tests for radiance.planck module."""

import pytest
import numpy as np

from eosim.radiance.planck import (
    spectral_radiance,
    spectral_exitance,
    total_radiance,
    total_exitance,
    band_radiance,
    band_exitance,
    wien_peak_wavelength,
    temperature_from_peak,
    contrast_temperature,
    radiance_difference,
)
from eosim.core.spectral import SpectralBand, BAND_LWIR, BAND_MWIR


class TestSpectralRadiance:
    """Tests for spectral_radiance function."""

    def test_positive_radiance(self) -> None:
        """Radiance should be positive for positive temperature."""
        L = spectral_radiance(10.0, 300.0)
        assert L > 0

    def test_higher_temperature_higher_radiance(self) -> None:
        """Higher temperature gives higher radiance at same wavelength."""
        L_300 = spectral_radiance(10.0, 300.0)
        L_400 = spectral_radiance(10.0, 400.0)
        assert L_400 > L_300

    def test_array_input(self) -> None:
        """Should handle array wavelength input."""
        wavelengths = np.array([8.0, 10.0, 12.0])
        L = spectral_radiance(wavelengths, 300.0)
        assert len(L) == 3
        assert np.all(L > 0)

    def test_peak_in_lwir_at_room_temp(self) -> None:
        """Room temperature peak should be in LWIR."""
        wavelengths = np.linspace(5.0, 20.0, 100)
        L = spectral_radiance(wavelengths, 300.0)
        peak_wavelength = wavelengths[np.argmax(L)]
        # Wien's law: peak at ~9.66 μm for 300K
        assert 8.0 < peak_wavelength < 12.0


class TestTotalRadiance:
    """Tests for total radiance functions."""

    def test_total_radiance_scaling(self) -> None:
        """Total radiance scales as T^4."""
        L1 = total_radiance(300.0)
        L2 = total_radiance(600.0)
        # 600/300 = 2, so ratio should be 2^4 = 16
        assert L2 / L1 == pytest.approx(16.0, rel=1e-6)

    def test_stefan_boltzmann(self) -> None:
        """Total exitance should match Stefan-Boltzmann law."""
        M = total_exitance(300.0)
        # M = σT^4, known value ~459 W/m²
        assert M == pytest.approx(459.3, rel=0.01)


class TestBandRadiance:
    """Tests for band-integrated radiance."""

    def test_band_radiance_positive(self) -> None:
        """Band radiance should be positive."""
        L = band_radiance(BAND_LWIR, 300.0)
        assert L > 0

    def test_hotter_higher_band_radiance(self) -> None:
        """Higher temperature gives more band-integrated radiance."""
        L_300 = band_radiance(BAND_LWIR, 300.0)
        L_350 = band_radiance(BAND_LWIR, 350.0)
        assert L_350 > L_300

    def test_mwir_vs_lwir_ratio(self) -> None:
        """At room temperature, LWIR should have more radiance than MWIR."""
        L_mwir = band_radiance(BAND_MWIR, 300.0)
        L_lwir = band_radiance(BAND_LWIR, 300.0)
        # Wien's peak at 300K is ~9.66 μm (LWIR), so LWIR > MWIR
        assert L_lwir > L_mwir


class TestWienDisplacement:
    """Tests for Wien's displacement law functions."""

    def test_wien_room_temperature(self) -> None:
        """Peak wavelength at room temperature."""
        peak = wien_peak_wavelength(300.0)
        # 2898 / 300 ≈ 9.66 μm
        assert peak == pytest.approx(9.66, rel=0.01)

    def test_wien_inverse(self) -> None:
        """Temperature from peak wavelength is inverse."""
        T_original = 350.0
        peak = wien_peak_wavelength(T_original)
        T_computed = temperature_from_peak(peak)
        assert T_computed == pytest.approx(T_original)


class TestContrastTemperature:
    """Tests for contrast temperature function."""

    def test_contrast_temperature_recovery(self) -> None:
        """Should recover known temperature from its radiance."""
        T_true = 310.0
        L = band_radiance(BAND_LWIR, T_true)
        T_recovered = contrast_temperature(L, BAND_LWIR, reference_T=300.0)
        assert T_recovered == pytest.approx(T_true, rel=0.01)


class TestRadianceDifference:
    """Tests for radiance difference function."""

    def test_positive_difference(self) -> None:
        """Target hotter than background gives positive difference."""
        dL = radiance_difference(320.0, 300.0, BAND_LWIR)
        assert dL > 0

    def test_negative_difference(self) -> None:
        """Target cooler than background gives negative difference."""
        dL = radiance_difference(280.0, 300.0, BAND_LWIR)
        assert dL < 0

    def test_zero_difference(self) -> None:
        """Same temperature gives zero difference."""
        dL = radiance_difference(300.0, 300.0, BAND_LWIR)
        assert dL == pytest.approx(0.0, abs=1e-10)
