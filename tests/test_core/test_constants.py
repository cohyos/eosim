"""Tests for core.constants module."""

import pytest
import numpy as np

from eosim.core.constants import (
    CONSTANTS,
    PhysicalConstants,
    BAND_LIMITS,
    MATERIAL_EMISSIVITY,
    ATMOSPHERIC_WINDOWS,
)


class TestPhysicalConstants:
    """Tests for PhysicalConstants."""

    def test_speed_of_light(self) -> None:
        """Test speed of light value."""
        assert CONSTANTS.c == pytest.approx(299792458.0)

    def test_planck_constant(self) -> None:
        """Test Planck constant value."""
        assert CONSTANTS.h == pytest.approx(6.62607015e-34)

    def test_boltzmann_constant(self) -> None:
        """Test Boltzmann constant value."""
        assert CONSTANTS.k_B == pytest.approx(1.380649e-23)

    def test_stefan_boltzmann_constant(self) -> None:
        """Test Stefan-Boltzmann constant is approximately correct."""
        # Known value: 5.670374419e-8 W/(m²·K⁴)
        assert CONSTANTS.sigma == pytest.approx(5.670374419e-8, rel=1e-6)

    def test_first_radiation_constant(self) -> None:
        """Test c1 is correctly computed."""
        expected = 2 * np.pi * CONSTANTS.h * CONSTANTS.c**2
        assert CONSTANTS.c1 == pytest.approx(expected)

    def test_second_radiation_constant(self) -> None:
        """Test c2 is correctly computed."""
        expected = CONSTANTS.h * CONSTANTS.c / CONSTANTS.k_B
        assert CONSTANTS.c2 == pytest.approx(expected)
        # Known value: ~0.01439 m·K
        assert CONSTANTS.c2 == pytest.approx(0.01439, rel=1e-3)

    def test_constants_are_frozen(self) -> None:
        """Test that constants dataclass is immutable."""
        with pytest.raises(Exception):  # FrozenInstanceError
            CONSTANTS.c = 3e8


class TestSpectralBandLimits:
    """Tests for SpectralBandLimits."""

    def test_visible_band(self) -> None:
        """Test visible band limits."""
        assert BAND_LIMITS.VIS_MIN == 0.38
        assert BAND_LIMITS.VIS_MAX == 0.70

    def test_mwir_band(self) -> None:
        """Test MWIR band limits."""
        assert BAND_LIMITS.MWIR_MIN == 3.0
        assert BAND_LIMITS.MWIR_MAX == 5.0

    def test_lwir_band(self) -> None:
        """Test LWIR band limits."""
        assert BAND_LIMITS.LWIR_MIN == 8.0
        assert BAND_LIMITS.LWIR_MAX == 14.0

    def test_band_order(self) -> None:
        """Test that bands are in spectral order."""
        assert BAND_LIMITS.VIS_MAX <= BAND_LIMITS.NIR_MIN
        assert BAND_LIMITS.NIR_MAX <= BAND_LIMITS.SWIR_MIN
        assert BAND_LIMITS.SWIR_MAX <= BAND_LIMITS.MWIR_MIN
        assert BAND_LIMITS.MWIR_MAX <= BAND_LIMITS.LWIR_MIN


class TestMaterialEmissivity:
    """Tests for material emissivity values."""

    def test_blackbody_emissivity(self) -> None:
        """Test blackbody has emissivity of 1."""
        assert MATERIAL_EMISSIVITY["blackbody"] == 1.0

    def test_polished_metal_low_emissivity(self) -> None:
        """Test polished metals have low emissivity."""
        assert MATERIAL_EMISSIVITY["aluminum_polished"] < 0.1
        assert MATERIAL_EMISSIVITY["steel_polished"] < 0.1

    def test_vegetation_high_emissivity(self) -> None:
        """Test vegetation has high emissivity."""
        assert MATERIAL_EMISSIVITY["vegetation"] > 0.9

    def test_emissivity_in_valid_range(self) -> None:
        """Test all emissivities are in [0, 1]."""
        for name, emissivity in MATERIAL_EMISSIVITY.items():
            assert 0 <= emissivity <= 1, f"Invalid emissivity for {name}"


class TestAtmosphericWindows:
    """Tests for atmospheric window definitions."""

    def test_mwir_window(self) -> None:
        """Test MWIR atmospheric window."""
        assert "mwir" in ATMOSPHERIC_WINDOWS
        lmin, lmax, trans = ATMOSPHERIC_WINDOWS["mwir"]
        assert lmin == 3.0
        assert lmax == 5.0
        assert 0 < trans <= 1

    def test_lwir_window(self) -> None:
        """Test LWIR atmospheric window."""
        assert "lwir" in ATMOSPHERIC_WINDOWS
        lmin, lmax, trans = ATMOSPHERIC_WINDOWS["lwir"]
        assert lmin == 8.0
        assert lmax == 14.0
        assert 0 < trans <= 1
