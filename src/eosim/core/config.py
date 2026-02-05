"""
Configuration models for EOSIM using Pydantic.

Defines type-safe configuration structures for simulation setup,
sensor parameters, scene definition, and output settings.
"""

from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml


# =============================================================================
# Enumerations
# =============================================================================


class FidelityLevel(str, Enum):
    """System-wide fidelity level for simulation."""

    PREVIEW = "preview"  # Fast preview, minimal physics
    DRAFT = "draft"  # Quick iteration, simplified physics
    STANDARD = "standard"  # Production use, balanced
    HIGH = "high"  # High accuracy, slower
    REFERENCE = "reference"  # Maximum fidelity, validation


class SpectralMode(str, Enum):
    """Spectral simulation mode."""

    BROADBAND = "broadband"  # Single integrated band
    MULTISPECTRAL = "multispectral"  # Multiple discrete bands
    HYPERSPECTRAL = "hyperspectral"  # Many narrow bands


class RenderBackend(str, Enum):
    """Rendering backend selection."""

    RAYCAST = "raycast"  # Simple ray casting
    MITSUBA = "mitsuba"  # Mitsuba 3 physically-based renderer
    RASTERIZER = "rasterizer"  # GPU rasterization


class NoiseModel(str, Enum):
    """Noise model complexity."""

    NONE = "none"  # No noise
    SIMPLE = "simple"  # Gaussian only
    STANDARD = "standard"  # Shot + read + dark
    FULL = "full"  # Full model with FPN


class AtmosphereModel(str, Enum):
    """Atmosphere model selection."""

    NONE = "none"  # No atmosphere
    BEER_LAMBERT = "beer_lambert"  # Simple extinction
    LUT = "lut"  # Look-up table based
    RAF_TRAN = "raf_tran"  # RAF-tran radiative transfer


# =============================================================================
# Geometry and Position
# =============================================================================


class Position3D(BaseModel):
    """3D position in meters."""

    x: float = Field(default=0.0, description="X coordinate [m]")
    y: float = Field(default=0.0, description="Y coordinate [m]")
    z: float = Field(default=0.0, description="Z coordinate [m]")

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


class Orientation(BaseModel):
    """Orientation in Euler angles (degrees)."""

    roll: float = Field(default=0.0, ge=-180, le=180, description="Roll angle [deg]")
    pitch: float = Field(default=0.0, ge=-90, le=90, description="Pitch angle [deg]")
    yaw: float = Field(default=0.0, ge=-180, le=180, description="Yaw/heading [deg]")


class GeodeticPosition(BaseModel):
    """Geodetic position (WGS84)."""

    latitude_deg: float = Field(ge=-90, le=90, description="Latitude [deg]")
    longitude_deg: float = Field(ge=-180, le=180, description="Longitude [deg]")
    altitude_m: float = Field(default=0.0, description="Altitude above MSL [m]")


# =============================================================================
# Spectral Configuration
# =============================================================================


class SpectralBandConfig(BaseModel):
    """Configuration for a spectral band."""

    name: str = Field(description="Band identifier")
    lambda_min_um: float = Field(gt=0, description="Minimum wavelength [μm]")
    lambda_max_um: float = Field(gt=0, description="Maximum wavelength [μm]")
    n_samples: int = Field(default=10, ge=1, description="Spectral samples")

    @model_validator(mode="after")
    def validate_wavelengths(self) -> "SpectralBandConfig":
        if self.lambda_min_um >= self.lambda_max_um:
            raise ValueError("lambda_min must be less than lambda_max")
        return self


class SpectralConfig(BaseModel):
    """Spectral simulation configuration."""

    mode: SpectralMode = Field(default=SpectralMode.BROADBAND)
    bands: list[SpectralBandConfig] = Field(default_factory=list)

    @classmethod
    def lwir_default(cls) -> "SpectralConfig":
        """Create default LWIR configuration."""
        return cls(
            mode=SpectralMode.BROADBAND,
            bands=[
                SpectralBandConfig(
                    name="LWIR", lambda_min_um=8.0, lambda_max_um=14.0, n_samples=20
                )
            ],
        )

    @classmethod
    def mwir_default(cls) -> "SpectralConfig":
        """Create default MWIR configuration."""
        return cls(
            mode=SpectralMode.BROADBAND,
            bands=[
                SpectralBandConfig(
                    name="MWIR", lambda_min_um=3.0, lambda_max_um=5.0, n_samples=20
                )
            ],
        )

    @classmethod
    def visible_default(cls) -> "SpectralConfig":
        """Create default visible configuration."""
        return cls(
            mode=SpectralMode.BROADBAND,
            bands=[
                SpectralBandConfig(
                    name="VIS", lambda_min_um=0.4, lambda_max_um=0.7, n_samples=30
                )
            ],
        )


# =============================================================================
# Sensor Configuration
# =============================================================================


class FPAConfig(BaseModel):
    """Focal Plane Array configuration."""

    width_pixels: int = Field(gt=0, description="Array width [pixels]")
    height_pixels: int = Field(gt=0, description="Array height [pixels]")
    pixel_pitch_um: float = Field(gt=0, description="Pixel pitch [μm]")
    fill_factor: float = Field(default=1.0, ge=0, le=1, description="Fill factor")

    @property
    def width_mm(self) -> float:
        """FPA width in mm."""
        return self.width_pixels * self.pixel_pitch_um / 1000

    @property
    def height_mm(self) -> float:
        """FPA height in mm."""
        return self.height_pixels * self.pixel_pitch_um / 1000


class OpticsConfig(BaseModel):
    """Optical system configuration."""

    focal_length_mm: float = Field(gt=0, description="Focal length [mm]")
    f_number: float = Field(gt=0, description="F-number")
    transmission: float = Field(default=0.9, ge=0, le=1, description="Transmission")

    @property
    def aperture_mm(self) -> float:
        """Aperture diameter [mm]."""
        return self.focal_length_mm / self.f_number


class DetectorConfig(BaseModel):
    """Detector characteristics configuration."""

    quantum_efficiency: float = Field(
        default=0.7, ge=0, le=1, description="Peak quantum efficiency"
    )
    dark_current_e_per_s: float = Field(
        default=1000.0, ge=0, description="Dark current [e⁻/s]"
    )
    read_noise_e: float = Field(default=50.0, ge=0, description="Read noise [e⁻ rms]")
    full_well_e: int = Field(default=100000, gt=0, description="Full well capacity [e⁻]")
    bit_depth: int = Field(default=14, ge=8, le=16, description="ADC bit depth")


class SensorConfig(BaseModel):
    """Complete sensor system configuration."""

    name: str = Field(default="sensor", description="Sensor identifier")
    fpa: FPAConfig
    optics: OpticsConfig
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    integration_time_ms: float = Field(gt=0, description="Integration time [ms]")
    frame_rate_hz: float = Field(default=30.0, gt=0, description="Frame rate [Hz]")
    spectral: SpectralConfig = Field(default_factory=SpectralConfig.lwir_default)

    @property
    def ifov_mrad(self) -> float:
        """Instantaneous field of view [mrad]."""
        return self.fpa.pixel_pitch_um / self.optics.focal_length_mm

    @property
    def fov_deg(self) -> tuple[float, float]:
        """Field of view (horizontal, vertical) [deg]."""
        import numpy as np

        h_fov = 2 * np.degrees(
            np.arctan(self.fpa.width_mm / (2 * self.optics.focal_length_mm))
        )
        v_fov = 2 * np.degrees(
            np.arctan(self.fpa.height_mm / (2 * self.optics.focal_length_mm))
        )
        return (h_fov, v_fov)


# =============================================================================
# Platform and Dynamics
# =============================================================================


class GimbalConfig(BaseModel):
    """Gimbal/pointing configuration."""

    azimuth_deg: float = Field(default=0.0, ge=-180, le=180)
    elevation_deg: float = Field(default=0.0, ge=-90, le=90)
    azimuth_rate_deg_s: float = Field(default=0.0, description="Azimuth rate [deg/s]")
    elevation_rate_deg_s: float = Field(default=0.0, description="Elevation rate [deg/s]")


class PlatformConfig(BaseModel):
    """Sensor platform configuration."""

    name: str = Field(default="platform", description="Platform identifier")
    position: Position3D = Field(default_factory=Position3D)
    orientation: Orientation = Field(default_factory=Orientation)
    velocity_m_s: Position3D = Field(
        default_factory=Position3D, description="Velocity vector [m/s]"
    )
    gimbal: GimbalConfig = Field(default_factory=GimbalConfig)


# =============================================================================
# Environment Configuration
# =============================================================================


class TimeConfig(BaseModel):
    """Simulation time configuration."""

    year: int = Field(default=2024, ge=1900, le=2100)
    month: int = Field(default=6, ge=1, le=12)
    day: int = Field(default=21, ge=1, le=31)
    hour: float = Field(default=12.0, ge=0, lt=24, description="Local time [hours]")
    utc_offset: float = Field(default=0.0, ge=-12, le=14, description="UTC offset [hours]")


class AtmosphereConfig(BaseModel):
    """Atmosphere configuration."""

    model: AtmosphereModel = Field(default=AtmosphereModel.BEER_LAMBERT)
    visibility_km: float = Field(default=23.0, gt=0, description="Visibility [km]")
    temperature_K: float = Field(default=288.15, gt=0, description="Ground temp [K]")
    relative_humidity: float = Field(default=0.5, ge=0, le=1)
    pressure_hPa: float = Field(default=1013.25, gt=0, description="Pressure [hPa]")


class EnvironmentConfig(BaseModel):
    """Environment and atmospheric configuration."""

    time: TimeConfig = Field(default_factory=TimeConfig)
    atmosphere: AtmosphereConfig = Field(default_factory=AtmosphereConfig)
    ground_temperature_K: float = Field(default=300.0, gt=0)
    ambient_temperature_K: float = Field(default=288.15, gt=0)
    wind_speed_m_s: float = Field(default=0.0, ge=0)
    cloud_cover: float = Field(default=0.0, ge=0, le=1)


# =============================================================================
# Scene Objects
# =============================================================================


class MaterialConfig(BaseModel):
    """Material properties configuration."""

    name: str = Field(description="Material identifier")
    emissivity: float = Field(ge=0, le=1, description="Thermal emissivity")
    reflectance: Optional[float] = Field(
        default=None, ge=0, le=1, description="Reflectance (computed if not set)"
    )
    temperature_K: Optional[float] = Field(
        default=None, gt=0, description="Fixed temperature [K]"
    )

    @model_validator(mode="after")
    def compute_reflectance(self) -> "MaterialConfig":
        if self.reflectance is None:
            # Kirchhoff's law for opaque surfaces: ε + ρ ≈ 1
            self.reflectance = 1.0 - self.emissivity
        return self


class ObjectConfig(BaseModel):
    """Scene object configuration."""

    name: str = Field(description="Object identifier")
    type: str = Field(description="Object type (vehicle, building, terrain, etc.)")
    position: Position3D = Field(default_factory=Position3D)
    orientation: Orientation = Field(default_factory=Orientation)
    geometry_file: Optional[str] = Field(
        default=None, description="Path to geometry file"
    )
    material: Optional[MaterialConfig] = Field(default=None)
    temperature_K: Optional[float] = Field(default=None, gt=0)
    enabled: bool = Field(default=True)


class SceneConfig(BaseModel):
    """Scene configuration."""

    name: str = Field(default="scene", description="Scene identifier")
    terrain_file: Optional[str] = Field(default=None, description="Terrain DEM file")
    background_temperature_K: float = Field(default=300.0, gt=0)
    objects: list[ObjectConfig] = Field(default_factory=list)


# =============================================================================
# Output Configuration
# =============================================================================


class OutputFormat(str, Enum):
    """Output file format."""

    TIFF = "tiff"
    PNG = "png"
    NUMPY = "numpy"
    HDF5 = "hdf5"
    ENVI = "envi"


class OutputConfig(BaseModel):
    """Output configuration."""

    directory: Path = Field(default=Path("output"))
    format: OutputFormat = Field(default=OutputFormat.TIFF)
    save_intermediate: bool = Field(default=False, description="Save pipeline stages")
    save_metadata: bool = Field(default=True, description="Save YAML metadata")
    filename_prefix: str = Field(default="frame")


# =============================================================================
# Fidelity Configuration
# =============================================================================


class FidelityConfig(BaseModel):
    """Fidelity settings for simulation."""

    level: FidelityLevel = Field(default=FidelityLevel.STANDARD)
    render_backend: RenderBackend = Field(default=RenderBackend.RAYCAST)
    noise_model: NoiseModel = Field(default=NoiseModel.STANDARD)
    atmosphere_model: AtmosphereModel = Field(default=AtmosphereModel.BEER_LAMBERT)
    psf_enabled: bool = Field(default=True)
    thermal_solver_enabled: bool = Field(default=True)

    @classmethod
    def from_level(cls, level: FidelityLevel) -> "FidelityConfig":
        """Create fidelity config from preset level."""
        presets = {
            FidelityLevel.PREVIEW: cls(
                level=level,
                render_backend=RenderBackend.RASTERIZER,
                noise_model=NoiseModel.NONE,
                atmosphere_model=AtmosphereModel.NONE,
                psf_enabled=False,
                thermal_solver_enabled=False,
            ),
            FidelityLevel.DRAFT: cls(
                level=level,
                render_backend=RenderBackend.RAYCAST,
                noise_model=NoiseModel.SIMPLE,
                atmosphere_model=AtmosphereModel.BEER_LAMBERT,
                psf_enabled=False,
                thermal_solver_enabled=False,
            ),
            FidelityLevel.STANDARD: cls(
                level=level,
                render_backend=RenderBackend.RAYCAST,
                noise_model=NoiseModel.STANDARD,
                atmosphere_model=AtmosphereModel.BEER_LAMBERT,
                psf_enabled=True,
                thermal_solver_enabled=True,
            ),
            FidelityLevel.HIGH: cls(
                level=level,
                render_backend=RenderBackend.MITSUBA,
                noise_model=NoiseModel.FULL,
                atmosphere_model=AtmosphereModel.RAF_TRAN,
                psf_enabled=True,
                thermal_solver_enabled=True,
            ),
            FidelityLevel.REFERENCE: cls(
                level=level,
                render_backend=RenderBackend.MITSUBA,
                noise_model=NoiseModel.FULL,
                atmosphere_model=AtmosphereModel.RAF_TRAN,
                psf_enabled=True,
                thermal_solver_enabled=True,
            ),
        }
        return presets.get(level, presets[FidelityLevel.STANDARD])


# =============================================================================
# Main Simulation Configuration
# =============================================================================


class SimulationConfig(BaseModel):
    """Complete simulation configuration."""

    name: str = Field(default="simulation", description="Simulation name")
    description: str = Field(default="", description="Description")
    sensor: SensorConfig
    platform: PlatformConfig = Field(default_factory=PlatformConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    scene: SceneConfig = Field(default_factory=SceneConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    fidelity: FidelityConfig = Field(default_factory=FidelityConfig)

    # Temporal settings
    n_frames: int = Field(default=1, ge=1, description="Number of frames")
    start_time_s: float = Field(default=0.0, ge=0, description="Start time [s]")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SimulationConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            # Use mode='json' to ensure Path objects are serialized as strings
            yaml.dump(self.model_dump(mode='json'), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def minimal_example(cls) -> "SimulationConfig":
        """Create a minimal example configuration."""
        return cls(
            name="minimal_example",
            sensor=SensorConfig(
                fpa=FPAConfig(width_pixels=640, height_pixels=480, pixel_pitch_um=17.0),
                optics=OpticsConfig(focal_length_mm=50.0, f_number=1.4),
                integration_time_ms=16.67,
            ),
        )
