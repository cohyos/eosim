"""Tests for core.config module."""

import pytest
import tempfile
from pathlib import Path

from eosim.core.config import (
    FidelityLevel,
    SpectralMode,
    Position3D,
    Orientation,
    SpectralBandConfig,
    SpectralConfig,
    FPAConfig,
    OpticsConfig,
    DetectorConfig,
    SensorConfig,
    PlatformConfig,
    EnvironmentConfig,
    SceneConfig,
    MaterialConfig,
    ObjectConfig,
    OutputConfig,
    FidelityConfig,
    SimulationConfig,
)


class TestPosition3D:
    """Tests for Position3D."""

    def test_default_values(self) -> None:
        """Test default position is origin."""
        pos = Position3D()
        assert pos.x == 0.0
        assert pos.y == 0.0
        assert pos.z == 0.0

    def test_as_tuple(self) -> None:
        """Test conversion to tuple."""
        pos = Position3D(x=1.0, y=2.0, z=3.0)
        assert pos.as_tuple() == (1.0, 2.0, 3.0)


class TestOrientation:
    """Tests for Orientation."""

    def test_default_values(self) -> None:
        """Test default orientation is level."""
        ori = Orientation()
        assert ori.roll == 0.0
        assert ori.pitch == 0.0
        assert ori.yaw == 0.0

    def test_validation_roll_range(self) -> None:
        """Test roll angle validation."""
        with pytest.raises(ValueError):
            Orientation(roll=200.0)

    def test_validation_pitch_range(self) -> None:
        """Test pitch angle validation."""
        with pytest.raises(ValueError):
            Orientation(pitch=100.0)


class TestSpectralBandConfig:
    """Tests for SpectralBandConfig."""

    def test_valid_band(self) -> None:
        """Test creating valid band config."""
        band = SpectralBandConfig(
            name="LWIR", lambda_min_um=8.0, lambda_max_um=14.0, n_samples=20
        )
        assert band.name == "LWIR"
        assert band.lambda_min_um == 8.0

    def test_invalid_wavelengths(self) -> None:
        """Test validation of wavelength order."""
        with pytest.raises(ValueError, match="must be less than"):
            SpectralBandConfig(name="bad", lambda_min_um=14.0, lambda_max_um=8.0)


class TestSpectralConfig:
    """Tests for SpectralConfig."""

    def test_lwir_default(self) -> None:
        """Test LWIR default configuration."""
        config = SpectralConfig.lwir_default()
        assert config.mode == SpectralMode.BROADBAND
        assert len(config.bands) == 1
        assert config.bands[0].name == "LWIR"

    def test_mwir_default(self) -> None:
        """Test MWIR default configuration."""
        config = SpectralConfig.mwir_default()
        assert config.bands[0].lambda_min_um == 3.0

    def test_visible_default(self) -> None:
        """Test visible default configuration."""
        config = SpectralConfig.visible_default()
        assert config.bands[0].lambda_min_um == 0.4


class TestFPAConfig:
    """Tests for FPAConfig."""

    def test_create_fpa(self) -> None:
        """Test creating FPA configuration."""
        fpa = FPAConfig(width_pixels=640, height_pixels=480, pixel_pitch_um=17.0)
        assert fpa.width_pixels == 640
        assert fpa.height_pixels == 480

    def test_dimensions_mm(self) -> None:
        """Test FPA dimension calculations."""
        fpa = FPAConfig(width_pixels=640, height_pixels=480, pixel_pitch_um=17.0)
        assert fpa.width_mm == pytest.approx(10.88)
        assert fpa.height_mm == pytest.approx(8.16)

    def test_validation_positive_pixels(self) -> None:
        """Test validation of positive pixel count."""
        with pytest.raises(ValueError):
            FPAConfig(width_pixels=0, height_pixels=480, pixel_pitch_um=17.0)


class TestOpticsConfig:
    """Tests for OpticsConfig."""

    def test_aperture_calculation(self) -> None:
        """Test aperture diameter calculation."""
        optics = OpticsConfig(focal_length_mm=50.0, f_number=2.0)
        assert optics.aperture_mm == pytest.approx(25.0)


class TestDetectorConfig:
    """Tests for DetectorConfig."""

    def test_default_values(self) -> None:
        """Test detector default values."""
        det = DetectorConfig()
        assert det.quantum_efficiency == 0.7
        assert det.bit_depth == 14


class TestSensorConfig:
    """Tests for SensorConfig."""

    def test_ifov_calculation(self) -> None:
        """Test IFOV calculation."""
        sensor = SensorConfig(
            fpa=FPAConfig(width_pixels=640, height_pixels=480, pixel_pitch_um=17.0),
            optics=OpticsConfig(focal_length_mm=50.0, f_number=1.4),
            integration_time_ms=16.67,
        )
        # IFOV = pixel_pitch / focal_length = 17 μm / 50 mm = 0.34 mrad
        assert sensor.ifov_mrad == pytest.approx(0.34)

    def test_fov_calculation(self) -> None:
        """Test FOV calculation."""
        sensor = SensorConfig(
            fpa=FPAConfig(width_pixels=640, height_pixels=480, pixel_pitch_um=17.0),
            optics=OpticsConfig(focal_length_mm=50.0, f_number=1.4),
            integration_time_ms=16.67,
        )
        h_fov, v_fov = sensor.fov_deg
        assert h_fov > v_fov  # Wider than tall


class TestMaterialConfig:
    """Tests for MaterialConfig."""

    def test_reflectance_computed(self) -> None:
        """Test reflectance is computed from emissivity."""
        mat = MaterialConfig(name="test", emissivity=0.9)
        assert mat.reflectance == pytest.approx(0.1)

    def test_explicit_reflectance(self) -> None:
        """Test explicit reflectance is preserved."""
        mat = MaterialConfig(name="test", emissivity=0.9, reflectance=0.05)
        assert mat.reflectance == 0.05


class TestFidelityConfig:
    """Tests for FidelityConfig."""

    def test_preview_preset(self) -> None:
        """Test preview fidelity preset."""
        config = FidelityConfig.from_level(FidelityLevel.PREVIEW)
        assert config.psf_enabled is False
        assert config.thermal_solver_enabled is False

    def test_standard_preset(self) -> None:
        """Test standard fidelity preset."""
        config = FidelityConfig.from_level(FidelityLevel.STANDARD)
        assert config.psf_enabled is True
        assert config.thermal_solver_enabled is True

    def test_high_preset(self) -> None:
        """Test high fidelity preset."""
        config = FidelityConfig.from_level(FidelityLevel.HIGH)
        assert config.level == FidelityLevel.HIGH


class TestSimulationConfig:
    """Tests for SimulationConfig."""

    def test_minimal_example(self) -> None:
        """Test creating minimal example configuration."""
        config = SimulationConfig.minimal_example()
        assert config.name == "minimal_example"
        assert config.sensor.fpa.width_pixels == 640

    def test_yaml_round_trip(self) -> None:
        """Test YAML save and load round trip."""
        config = SimulationConfig.minimal_example()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config.to_yaml(f.name)
            loaded = SimulationConfig.from_yaml(f.name)

        assert loaded.name == config.name
        assert loaded.sensor.fpa.width_pixels == config.sensor.fpa.width_pixels
        assert loaded.sensor.optics.focal_length_mm == config.sensor.optics.focal_length_mm

    def test_default_components(self) -> None:
        """Test default components are created."""
        config = SimulationConfig(
            sensor=SensorConfig(
                fpa=FPAConfig(width_pixels=320, height_pixels=240, pixel_pitch_um=30.0),
                optics=OpticsConfig(focal_length_mm=100.0, f_number=2.0),
                integration_time_ms=10.0,
            )
        )
        # Check defaults are populated
        assert config.platform is not None
        assert config.environment is not None
        assert config.output is not None
        assert config.fidelity is not None
        assert config.n_frames == 1
