"""Tests for scene.materials module."""

import pytest
import numpy as np

from eosim.scene.materials import (
    SpectralProperty,
    ThermalProperties,
    LambertianBRDF,
    OrenNayarBRDF,
    Material,
    MaterialLibrary,
    material_library,
    create_graybody,
    create_blackbody,
)
from eosim.core.spectral import SpectralBand, BAND_LWIR


class TestSpectralProperty:
    """Tests for SpectralProperty class."""

    def test_constant_property(self) -> None:
        """Test constant (non-spectral) property."""
        prop = SpectralProperty.constant(0.9)
        assert not prop.is_spectral
        assert prop.at_wavelength(10.0) == pytest.approx(0.9)

    def test_spectral_property(self) -> None:
        """Test wavelength-dependent property."""
        wavelengths = np.array([8.0, 10.0, 12.0])
        values = np.array([0.8, 0.9, 0.85])
        prop = SpectralProperty.from_spectrum(wavelengths, values)

        assert prop.is_spectral
        assert prop.at_wavelength(10.0) == pytest.approx(0.9)
        assert prop.at_wavelength(9.0) == pytest.approx(0.85)  # Interpolated

    def test_band_average(self) -> None:
        """Test band averaging."""
        prop = SpectralProperty.constant(0.9)
        avg = prop.band_average(BAND_LWIR)
        assert avg == pytest.approx(0.9)


class TestThermalProperties:
    """Tests for ThermalProperties class."""

    def test_thermal_diffusivity(self) -> None:
        """Test thermal diffusivity calculation."""
        thermal = ThermalProperties(
            thermal_conductivity=1.0,
            specific_heat=1000.0,
            density=2000.0,
        )
        # α = k/(ρ·c) = 1.0/(2000×1000) = 5e-7 m²/s
        assert thermal.thermal_diffusivity == pytest.approx(5e-7)

    def test_thermal_inertia(self) -> None:
        """Test thermal inertia calculation."""
        thermal = ThermalProperties(
            thermal_conductivity=1.0,
            specific_heat=1000.0,
            density=2000.0,
        )
        # P = √(k·ρ·c) = √(1×2000×1000) ≈ 1414
        expected = np.sqrt(1.0 * 2000.0 * 1000.0)
        assert thermal.thermal_inertia == pytest.approx(expected)


class TestLambertianBRDF:
    """Tests for LambertianBRDF class."""

    def test_evaluate(self) -> None:
        """Test BRDF evaluation."""
        brdf = LambertianBRDF(reflectance=0.5)

        wi = np.array([0, 0, 1])  # Incident from above
        wo = np.array([0, 0, 1])  # Outgoing upward
        n = np.array([0, 0, 1])   # Normal pointing up

        # f_r = ρ/π = 0.5/π
        expected = 0.5 / np.pi
        assert brdf.evaluate(wi, wo, n) == pytest.approx(expected)

    def test_sample(self) -> None:
        """Test direction sampling."""
        brdf = LambertianBRDF(reflectance=0.5)

        wi = np.array([0, 0, -1])  # Incident from above
        n = np.array([0, 0, 1])    # Normal pointing up

        wo, pdf = brdf.sample(wi, n)

        # Sampled direction should be in upper hemisphere
        assert np.dot(wo, n) > 0
        assert pdf > 0


class TestOrenNayarBRDF:
    """Tests for OrenNayarBRDF class."""

    def test_reduces_to_lambertian(self) -> None:
        """With zero roughness, should be close to Lambertian."""
        oren_nayar = OrenNayarBRDF(reflectance=0.5, roughness=0.0)
        lambertian = LambertianBRDF(reflectance=0.5)

        wi = np.array([0, 0, 1])
        wo = np.array([0, 0, 1])
        n = np.array([0, 0, 1])

        # With zero roughness, A=1, B=0, so should match Lambertian
        assert oren_nayar.evaluate(wi, wo, n) == pytest.approx(
            lambertian.evaluate(wi, wo, n), rel=0.01
        )


class TestMaterial:
    """Tests for Material class."""

    def test_create_material(self) -> None:
        """Test material creation."""
        mat = Material(
            name="test",
            emissivity=SpectralProperty.constant(0.9),
        )
        assert mat.name == "test"
        assert mat.get_emissivity() == pytest.approx(0.9)

    def test_reflectance_from_emissivity(self) -> None:
        """Reflectance should be computed from emissivity."""
        mat = Material(
            name="test",
            emissivity=SpectralProperty.constant(0.8),
        )
        assert mat.get_reflectance() == pytest.approx(0.2)

    def test_from_preset(self) -> None:
        """Test creating material from preset."""
        mat = Material.from_preset("concrete")
        assert mat.name == "concrete"
        assert mat.get_emissivity() == pytest.approx(0.92)

    def test_preset_unknown_error(self) -> None:
        """Unknown preset should raise error."""
        with pytest.raises(ValueError, match="Unknown material"):
            Material.from_preset("unknown_material_xyz")


class TestMaterialLibrary:
    """Tests for MaterialLibrary class."""

    def test_add_and_get(self) -> None:
        """Test adding and retrieving materials."""
        lib = MaterialLibrary()
        mat = Material(name="custom", emissivity=SpectralProperty.constant(0.85))
        lib.add(mat)

        retrieved = lib.get("custom")
        assert retrieved.name == "custom"

    def test_get_from_preset(self) -> None:
        """Library should auto-create from presets."""
        lib = MaterialLibrary()
        mat = lib.get("water")
        assert mat.get_emissivity() == pytest.approx(0.96)

    def test_contains(self) -> None:
        """Test contains check."""
        lib = MaterialLibrary()
        assert "water" in lib  # In presets
        assert "nonexistent_xyz" not in lib


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_graybody(self) -> None:
        """Test graybody creation."""
        mat = create_graybody(0.85, name="gray")
        assert mat.name == "gray"
        assert mat.get_emissivity() == pytest.approx(0.85)

    def test_create_blackbody(self) -> None:
        """Test blackbody creation."""
        mat = create_blackbody()
        assert mat.get_emissivity() == pytest.approx(1.0)
