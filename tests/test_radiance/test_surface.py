"""Tests for radiance.surface module."""

import pytest
import numpy as np

from eosim.radiance.surface import (
    SurfaceProperties,
    IncidentRadiation,
    surface_spectral_radiance,
    surface_band_radiance,
    apparent_temperature,
    emissivity_from_radiance,
    radiance_contrast,
    temperature_contrast,
    SurfaceRadianceModel,
)
from eosim.radiance.planck import spectral_radiance, band_radiance
from eosim.core.spectral import BAND_LWIR


class TestSurfaceProperties:
    """Tests for SurfaceProperties dataclass."""

    def test_reflectance_from_emissivity(self) -> None:
        """Reflectance should be computed from emissivity."""
        surface = SurfaceProperties(emissivity=0.9, temperature_K=300.0)
        assert surface.reflectance == pytest.approx(0.1)

    def test_explicit_reflectance(self) -> None:
        """Explicit reflectance overrides computed value."""
        surface = SurfaceProperties(
            emissivity=0.9, temperature_K=300.0, reflectance=0.05
        )
        assert surface.reflectance == 0.05


class TestSurfaceSpectralRadiance:
    """Tests for surface_spectral_radiance function."""

    def test_blackbody_equals_planck(self) -> None:
        """Blackbody surface should equal Planck radiance."""
        surface = SurfaceProperties(emissivity=1.0, temperature_K=300.0)
        L_surface = surface_spectral_radiance(10.0, surface)
        L_planck = spectral_radiance(10.0, 300.0)
        assert L_surface == pytest.approx(L_planck)

    def test_graybody_less_than_blackbody(self) -> None:
        """Graybody emits less than blackbody at same temperature."""
        blackbody = SurfaceProperties(emissivity=1.0, temperature_K=300.0)
        graybody = SurfaceProperties(emissivity=0.5, temperature_K=300.0)
        L_bb = surface_spectral_radiance(10.0, blackbody)
        L_gb = surface_spectral_radiance(10.0, graybody)
        assert L_gb < L_bb

    def test_reflected_component(self) -> None:
        """Low emissivity surface should reflect incident radiation."""
        surface = SurfaceProperties(emissivity=0.1, temperature_K=200.0)
        incident = IncidentRadiation(downwelling_radiance=10.0)

        L_no_incident = surface_spectral_radiance(10.0, surface)
        L_with_incident = surface_spectral_radiance(10.0, surface, incident)

        # With incident radiation, leaving radiance should be higher
        assert L_with_incident > L_no_incident


class TestSurfaceBandRadiance:
    """Tests for surface_band_radiance function."""

    def test_band_radiance_positive(self) -> None:
        """Band radiance should be positive."""
        surface = SurfaceProperties(emissivity=0.9, temperature_K=300.0)
        L = surface_band_radiance(BAND_LWIR, surface)
        assert L > 0

    def test_blackbody_band_radiance(self) -> None:
        """Blackbody band radiance should match Planck band radiance."""
        surface = SurfaceProperties(emissivity=1.0, temperature_K=300.0)
        L_surface = surface_band_radiance(BAND_LWIR, surface, n_samples=50)
        L_planck = band_radiance(BAND_LWIR, 300.0, n_samples=50)
        assert L_surface == pytest.approx(L_planck, rel=0.01)


class TestApparentTemperature:
    """Tests for apparent_temperature function."""

    def test_blackbody_recovery(self) -> None:
        """Blackbody apparent temperature should equal true temperature."""
        T_true = 310.0
        L = band_radiance(BAND_LWIR, T_true)
        T_app = apparent_temperature(L, 1.0, BAND_LWIR)
        assert T_app == pytest.approx(T_true, rel=0.01)

    def test_graybody_correction(self) -> None:
        """Correcting for emissivity should recover true temperature."""
        T_true = 320.0
        emissivity = 0.8
        background_T = 280.0

        # Measured radiance from graybody
        L_surface = emissivity * band_radiance(BAND_LWIR, T_true)
        L_background = (1 - emissivity) * band_radiance(BAND_LWIR, background_T)
        L_measured = L_surface + L_background

        T_corrected = apparent_temperature(
            L_measured, emissivity, BAND_LWIR, background_T
        )
        assert T_corrected == pytest.approx(T_true, rel=0.02)


class TestRadianceContrast:
    """Tests for radiance_contrast function."""

    def test_positive_contrast(self) -> None:
        """Hotter target should give positive contrast."""
        target = SurfaceProperties(emissivity=0.9, temperature_K=320.0)
        background = SurfaceProperties(emissivity=0.9, temperature_K=300.0)
        contrast = radiance_contrast(target, background, BAND_LWIR)
        assert contrast > 0

    def test_zero_contrast(self) -> None:
        """Same temperature should give zero contrast."""
        target = SurfaceProperties(emissivity=0.9, temperature_K=300.0)
        background = SurfaceProperties(emissivity=0.9, temperature_K=300.0)
        contrast = radiance_contrast(target, background, BAND_LWIR)
        assert contrast == pytest.approx(0.0, abs=1e-10)


class TestTemperatureContrast:
    """Tests for temperature_contrast function."""

    def test_delta_t_sign(self) -> None:
        """Temperature contrast should have correct sign."""
        hot_target = SurfaceProperties(emissivity=0.9, temperature_K=320.0)
        cold_target = SurfaceProperties(emissivity=0.9, temperature_K=280.0)
        background = SurfaceProperties(emissivity=0.9, temperature_K=300.0)

        dT_hot = temperature_contrast(hot_target, background, BAND_LWIR)
        dT_cold = temperature_contrast(cold_target, background, BAND_LWIR)

        assert dT_hot > 0
        assert dT_cold < 0

    def test_delta_t_magnitude(self) -> None:
        """Temperature contrast should approximate true ΔT for blackbodies."""
        target = SurfaceProperties(emissivity=1.0, temperature_K=310.0)
        background = SurfaceProperties(emissivity=1.0, temperature_K=300.0)
        dT = temperature_contrast(target, background, BAND_LWIR)
        # For blackbodies, should be close to actual 10K difference
        assert dT == pytest.approx(10.0, rel=0.1)


class TestSurfaceRadianceModel:
    """Tests for SurfaceRadianceModel class."""

    def test_compute_radiance_image(self) -> None:
        """Should compute radiance for 2D temperature array."""
        model = SurfaceRadianceModel(BAND_LWIR, n_spectral_samples=20)

        temp_map = np.full((10, 10), 300.0)
        temp_map[4:6, 4:6] = 320.0  # Hot spot

        radiance_image = model.compute_radiance_image(temp_map, emissivity=0.9)

        assert radiance_image.shape == (10, 10)
        # Hot spot should have higher radiance
        assert radiance_image[5, 5] > radiance_image[0, 0]
