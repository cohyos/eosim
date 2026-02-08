"""
EOSIM Sensor Library.

Provides pre-defined specifications for real-world EO/IR sensors including:
- L3Harris MX-series (MX-10, MX-15, MX-20, MX-25)
- Rafael TopLite, Toplite III
- FLIR Systems Star SAFIRE, Ultra
- Wescam MX-10D, MX-15D
- Thales Catherine, Sophie
- Various targeting pods (Sniper, Litening, LANTIRN)
- Ground-based systems (AN/AAQ series)

Each sensor specification includes:
- Optical parameters (focal length, aperture, FOV)
- Detector characteristics (type, resolution, pitch)
- Performance metrics (NEDT, detection ranges)
- Supported spectral bands
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any, Union
import numpy as np

from eosim.sensor import (
    FPAGeometry,
    DetectorType,
    DetectorProperties,
    FPAConfig,
    SpectralResponse,
    NoiseParameters,
    ADCParameters,
    ADCType,
    SensorModel,
    create_sensor_model,
)
from eosim.optics import PSFModel, create_optics_model


class SensorType(Enum):
    """Sensor system types."""
    TARGETING_POD = "targeting_pod"
    GIMBAL_EO_IR = "gimbal_eo_ir"
    GROUND_SYSTEM = "ground_system"
    HANDHELD = "handheld"
    THERMAL_IMAGER = "thermal_imager"
    FLIR = "flir"
    MWIR_CAMERA = "mwir_camera"
    LWIR_CAMERA = "lwir_camera"
    SWIR_CAMERA = "swir_camera"
    VISIBLE_CAMERA = "visible_camera"
    MULTISPECTRAL = "multispectral"


class MountType(Enum):
    """Sensor mount types."""
    AIRCRAFT_GIMBAL = "aircraft_gimbal"
    HELICOPTER_TURRET = "helicopter_turret"
    GROUND_VEHICLE = "ground_vehicle"
    TRIPOD = "tripod"
    HANDHELD = "handheld"
    NAVAL = "naval"
    FIXED = "fixed"


@dataclass
class OpticalSpec:
    """Optical system specifications.

    Attributes:
        focal_length_mm: Effective focal length
        aperture_mm: Clear aperture diameter
        fov_narrow_deg: Narrow field of view (degrees)
        fov_wide_deg: Wide field of view (degrees)
        zoom_range: Optical zoom range (min, max)
        f_number: F/# (calculated if not provided)
    """
    focal_length_mm: float
    aperture_mm: float
    fov_narrow_deg: float = 0.5
    fov_wide_deg: float = 20.0
    zoom_range: Tuple[float, float] = (1.0, 1.0)
    f_number: Optional[float] = None

    def __post_init__(self):
        if self.f_number is None:
            self.f_number = self.focal_length_mm / self.aperture_mm

    @property
    def ifov_narrow_urad(self) -> float:
        """Instantaneous FOV at narrow zoom (microradians)."""
        return np.deg2rad(self.fov_narrow_deg) * 1e6 / 640  # Approximate

    @property
    def ifov_wide_urad(self) -> float:
        """Instantaneous FOV at wide zoom (microradians)."""
        return np.deg2rad(self.fov_wide_deg) * 1e6 / 640


@dataclass
class DetectorSpec:
    """Detector specifications.

    Attributes:
        detector_type: Detector technology
        width_pixels: Horizontal resolution
        height_pixels: Vertical resolution
        pixel_pitch_um: Pixel pitch in micrometers
        operating_temp_k: Detector operating temperature
        spectral_band_um: Wavelength range (min, max)
        nedt_mk: Noise equivalent delta temperature (millikelvin)
        full_well_e: Full well capacity
        integration_time_ms: Typical integration time
    """
    detector_type: DetectorType
    width_pixels: int
    height_pixels: int
    pixel_pitch_um: float
    operating_temp_k: float = 77.0
    spectral_band_um: Tuple[float, float] = (8.0, 12.0)
    nedt_mk: float = 25.0
    full_well_e: float = 1e6
    integration_time_ms: float = 10.0
    bit_depth: int = 14


@dataclass
class PerformanceSpec:
    """Sensor performance specifications.

    Attributes:
        detection_range_vehicle_km: Detection range for vehicle-sized target
        recognition_range_vehicle_km: Recognition range
        identification_range_vehicle_km: Identification range
        detection_range_person_km: Detection range for human target
        mtf50_cy_mm: MTF50 in cycles/mm
        dri_criteria: Johnson DRI criteria (detection, recognition, identification)
    """
    detection_range_vehicle_km: float = 10.0
    recognition_range_vehicle_km: float = 5.0
    identification_range_vehicle_km: float = 2.5
    detection_range_person_km: float = 3.0
    recognition_range_person_km: float = 1.5
    identification_range_person_km: float = 0.75
    mtf50_cy_mm: float = 30.0
    dri_criteria: Tuple[float, float, float] = (1.0, 4.0, 8.0)  # Cycles across target


@dataclass
class GimbalSpec:
    """Gimbal/turret specifications.

    Attributes:
        azimuth_range_deg: Azimuth rotation range (min, max)
        elevation_range_deg: Elevation range (min, max)
        slew_rate_deg_s: Maximum slew rate
        jitter_urad: Line-of-sight jitter RMS
        stabilization: Stabilization accuracy
        weight_kg: System weight
        power_w: Power consumption
    """
    azimuth_range_deg: Tuple[float, float] = (-180.0, 180.0)
    elevation_range_deg: Tuple[float, float] = (-120.0, 30.0)
    slew_rate_deg_s: float = 60.0
    jitter_urad: float = 10.0
    stabilization_urad: float = 50.0
    weight_kg: float = 50.0
    power_w: float = 500.0


@dataclass
class SensorSpec:
    """Complete sensor system specification.

    Combines all sensor parameters into a single specification that can
    be used to create simulation models.
    """
    # Required fields (no defaults)
    id: str
    name: str
    manufacturer: str
    sensor_type: SensorType
    mount_type: MountType
    optics_ir: OpticalSpec
    detector_ir: DetectorSpec

    # Optional fields (with defaults)
    optics_visible: Optional[OpticalSpec] = None
    detector_visible: Optional[DetectorSpec] = None
    performance: PerformanceSpec = field(default_factory=PerformanceSpec)
    gimbal: Optional[GimbalSpec] = None

    # Additional features
    has_laser_designator: bool = False
    has_laser_rangefinder: bool = False
    has_laser_illuminator: bool = False
    has_spotter_scope: bool = False
    has_image_fusion: bool = False
    has_tracking: bool = True

    # Metadata
    description: str = ""
    country: str = "USA"
    year_introduced: int = 2000

    def to_sensor_model(
        self,
        channel: str = "ir",
        seed: Optional[int] = None,
    ) -> SensorModel:
        """Create a SensorModel from this specification.

        Args:
            channel: Which channel to create ("ir" or "visible")
            seed: Random seed for noise generation

        Returns:
            Configured SensorModel
        """
        if channel == "ir":
            det = self.detector_ir
            opt = self.optics_ir
        else:
            if self.detector_visible is None:
                raise ValueError("No visible channel on this sensor")
            det = self.detector_visible
            opt = self.optics_visible or self.optics_ir

        return create_sensor_model(
            detector_type=det.detector_type,
            resolution=(det.height_pixels, det.width_pixels),
            pixel_pitch_um=det.pixel_pitch_um,
            spectral_band_um=det.spectral_band_um,
            focal_length_mm=opt.focal_length_mm,
            f_number=opt.f_number,
            full_well=det.full_well_e,
            seed=seed,
        )


# =============================================================================
# Sensor Database
# =============================================================================

SENSORS: Dict[str, SensorSpec] = {}

# -----------------------------------------------------------------------------
# L3Harris MX-Series Airborne Turrets
# -----------------------------------------------------------------------------

SENSORS["mx10"] = SensorSpec(
    id="mx10",
    name="MX-10",
    manufacturer="L3Harris",
    sensor_type=SensorType.GIMBAL_EO_IR,
    mount_type=MountType.AIRCRAFT_GIMBAL,
    optics_ir=OpticalSpec(
        focal_length_mm=100.0,
        aperture_mm=50.0,
        fov_narrow_deg=1.8,
        fov_wide_deg=18.0,
        zoom_range=(1.0, 10.0),
    ),
    optics_visible=OpticalSpec(
        focal_length_mm=80.0,
        aperture_mm=40.0,
        fov_narrow_deg=0.8,
        fov_wide_deg=28.0,
    ),
    detector_ir=DetectorSpec(
        detector_type=DetectorType.MICROBOLOMETER,
        width_pixels=640,
        height_pixels=480,
        pixel_pitch_um=17.0,
        operating_temp_k=300.0,  # Uncooled
        spectral_band_um=(7.5, 13.5),
        nedt_mk=50.0,
    ),
    detector_visible=DetectorSpec(
        detector_type=DetectorType.SI_CMOS,
        width_pixels=1920,
        height_pixels=1080,
        pixel_pitch_um=5.0,
        operating_temp_k=300.0,
        spectral_band_um=(0.4, 0.9),
        nedt_mk=0,  # Not applicable
    ),
    performance=PerformanceSpec(
        detection_range_vehicle_km=5.0,
        recognition_range_vehicle_km=2.5,
        identification_range_vehicle_km=1.2,
        detection_range_person_km=1.5,
    ),
    gimbal=GimbalSpec(
        azimuth_range_deg=(-180, 180),
        elevation_range_deg=(-110, 30),
        slew_rate_deg_s=60,
        weight_kg=7.5,
        power_w=75,
    ),
    has_laser_designator=False,
    has_laser_rangefinder=True,
    has_tracking=True,
    description="Compact lightweight airborne gimbal for small UAS",
    country="USA",
    year_introduced=2010,
)

SENSORS["mx15"] = SensorSpec(
    id="mx15",
    name="MX-15",
    manufacturer="L3Harris",
    sensor_type=SensorType.GIMBAL_EO_IR,
    mount_type=MountType.AIRCRAFT_GIMBAL,
    optics_ir=OpticalSpec(
        focal_length_mm=350.0,
        aperture_mm=150.0,
        fov_narrow_deg=0.5,
        fov_wide_deg=25.0,
        zoom_range=(1.0, 50.0),
    ),
    optics_visible=OpticalSpec(
        focal_length_mm=450.0,
        aperture_mm=80.0,
        fov_narrow_deg=0.35,
        fov_wide_deg=30.0,
    ),
    detector_ir=DetectorSpec(
        detector_type=DetectorType.HGCDTE_MWIR,
        width_pixels=1280,
        height_pixels=1024,
        pixel_pitch_um=15.0,
        operating_temp_k=77.0,
        spectral_band_um=(3.0, 5.0),
        nedt_mk=20.0,
        full_well_e=2e6,
    ),
    detector_visible=DetectorSpec(
        detector_type=DetectorType.SI_CMOS,
        width_pixels=1920,
        height_pixels=1080,
        pixel_pitch_um=5.5,
        operating_temp_k=300.0,
        spectral_band_um=(0.4, 0.9),
    ),
    performance=PerformanceSpec(
        detection_range_vehicle_km=25.0,
        recognition_range_vehicle_km=12.0,
        identification_range_vehicle_km=6.0,
        detection_range_person_km=8.0,
    ),
    gimbal=GimbalSpec(
        azimuth_range_deg=(-180, 180),
        elevation_range_deg=(-120, 30),
        slew_rate_deg_s=80,
        weight_kg=42.0,
        power_w=300,
    ),
    has_laser_designator=True,
    has_laser_rangefinder=True,
    has_laser_illuminator=True,
    has_tracking=True,
    description="Multi-sensor airborne turret for ISR and targeting",
    country="USA",
    year_introduced=2005,
)

SENSORS["mx20"] = SensorSpec(
    id="mx20",
    name="MX-20",
    manufacturer="L3Harris",
    sensor_type=SensorType.GIMBAL_EO_IR,
    mount_type=MountType.AIRCRAFT_GIMBAL,
    optics_ir=OpticalSpec(
        focal_length_mm=500.0,
        aperture_mm=200.0,
        fov_narrow_deg=0.35,
        fov_wide_deg=22.0,
        zoom_range=(1.0, 65.0),
    ),
    optics_visible=OpticalSpec(
        focal_length_mm=600.0,
        aperture_mm=100.0,
        fov_narrow_deg=0.25,
        fov_wide_deg=28.0,
    ),
    detector_ir=DetectorSpec(
        detector_type=DetectorType.HGCDTE_MWIR,
        width_pixels=1280,
        height_pixels=1024,
        pixel_pitch_um=15.0,
        operating_temp_k=77.0,
        spectral_band_um=(3.0, 5.0),
        nedt_mk=18.0,
        full_well_e=3e6,
    ),
    detector_visible=DetectorSpec(
        detector_type=DetectorType.SI_CMOS,
        width_pixels=1920,
        height_pixels=1080,
        pixel_pitch_um=5.0,
        operating_temp_k=300.0,
        spectral_band_um=(0.4, 0.9),
    ),
    performance=PerformanceSpec(
        detection_range_vehicle_km=35.0,
        recognition_range_vehicle_km=18.0,
        identification_range_vehicle_km=9.0,
        detection_range_person_km=12.0,
    ),
    gimbal=GimbalSpec(
        azimuth_range_deg=(-180, 180),
        elevation_range_deg=(-120, 30),
        slew_rate_deg_s=90,
        weight_kg=54.0,
        power_w=400,
    ),
    has_laser_designator=True,
    has_laser_rangefinder=True,
    has_laser_illuminator=True,
    has_spotter_scope=True,
    has_tracking=True,
    description="High-performance multi-sensor turret for ISR/targeting",
    country="USA",
    year_introduced=2008,
)

SENSORS["mx25"] = SensorSpec(
    id="mx25",
    name="MX-25",
    manufacturer="L3Harris",
    sensor_type=SensorType.GIMBAL_EO_IR,
    mount_type=MountType.AIRCRAFT_GIMBAL,
    optics_ir=OpticalSpec(
        focal_length_mm=700.0,
        aperture_mm=250.0,
        fov_narrow_deg=0.25,
        fov_wide_deg=20.0,
        zoom_range=(1.0, 80.0),
    ),
    optics_visible=OpticalSpec(
        focal_length_mm=800.0,
        aperture_mm=130.0,
        fov_narrow_deg=0.18,
        fov_wide_deg=25.0,
    ),
    detector_ir=DetectorSpec(
        detector_type=DetectorType.HGCDTE_MWIR,
        width_pixels=1920,
        height_pixels=1536,
        pixel_pitch_um=12.0,
        operating_temp_k=77.0,
        spectral_band_um=(3.0, 5.0),
        nedt_mk=15.0,
        full_well_e=5e6,
    ),
    detector_visible=DetectorSpec(
        detector_type=DetectorType.SI_CMOS,
        width_pixels=2560,
        height_pixels=2048,
        pixel_pitch_um=4.5,
        operating_temp_k=300.0,
        spectral_band_um=(0.4, 0.9),
    ),
    performance=PerformanceSpec(
        detection_range_vehicle_km=50.0,
        recognition_range_vehicle_km=25.0,
        identification_range_vehicle_km=12.0,
        detection_range_person_km=18.0,
    ),
    gimbal=GimbalSpec(
        azimuth_range_deg=(-180, 180),
        elevation_range_deg=(-120, 30),
        slew_rate_deg_s=100,
        weight_kg=75.0,
        power_w=600,
    ),
    has_laser_designator=True,
    has_laser_rangefinder=True,
    has_laser_illuminator=True,
    has_spotter_scope=True,
    has_image_fusion=True,
    has_tracking=True,
    description="Premium long-range multi-sensor targeting system",
    country="USA",
    year_introduced=2012,
)

# -----------------------------------------------------------------------------
# Rafael TopLite Series
# -----------------------------------------------------------------------------

SENSORS["toplite_iii"] = SensorSpec(
    id="toplite_iii",
    name="TopLite III",
    manufacturer="Rafael",
    sensor_type=SensorType.GIMBAL_EO_IR,
    mount_type=MountType.HELICOPTER_TURRET,
    optics_ir=OpticalSpec(
        focal_length_mm=400.0,
        aperture_mm=160.0,
        fov_narrow_deg=0.6,
        fov_wide_deg=15.0,
        zoom_range=(1.0, 25.0),
    ),
    optics_visible=OpticalSpec(
        focal_length_mm=350.0,
        aperture_mm=70.0,
        fov_narrow_deg=0.5,
        fov_wide_deg=18.0,
    ),
    detector_ir=DetectorSpec(
        detector_type=DetectorType.INSB,
        width_pixels=640,
        height_pixels=512,
        pixel_pitch_um=15.0,
        operating_temp_k=77.0,
        spectral_band_um=(3.0, 5.0),
        nedt_mk=22.0,
    ),
    detector_visible=DetectorSpec(
        detector_type=DetectorType.SI_CCD,
        width_pixels=1024,
        height_pixels=768,
        pixel_pitch_um=8.0,
        operating_temp_k=280.0,
        spectral_band_um=(0.4, 0.9),
    ),
    performance=PerformanceSpec(
        detection_range_vehicle_km=20.0,
        recognition_range_vehicle_km=10.0,
        identification_range_vehicle_km=5.0,
    ),
    gimbal=GimbalSpec(
        azimuth_range_deg=(-180, 180),
        elevation_range_deg=(-110, 30),
        slew_rate_deg_s=70,
        weight_kg=38.0,
        power_w=280,
    ),
    has_laser_designator=True,
    has_laser_rangefinder=True,
    has_tracking=True,
    description="Helicopter-mounted targeting turret",
    country="Israel",
    year_introduced=2003,
)

SENSORS["reccelite"] = SensorSpec(
    id="reccelite",
    name="RecceLite",
    manufacturer="Rafael",
    sensor_type=SensorType.TARGETING_POD,
    mount_type=MountType.AIRCRAFT_GIMBAL,
    optics_ir=OpticalSpec(
        focal_length_mm=450.0,
        aperture_mm=180.0,
        fov_narrow_deg=0.4,
        fov_wide_deg=10.0,
        zoom_range=(1.0, 25.0),
    ),
    optics_visible=OpticalSpec(
        focal_length_mm=500.0,
        aperture_mm=90.0,
        fov_narrow_deg=0.3,
        fov_wide_deg=12.0,
    ),
    detector_ir=DetectorSpec(
        detector_type=DetectorType.INSB,
        width_pixels=1024,
        height_pixels=1024,
        pixel_pitch_um=15.0,
        operating_temp_k=77.0,
        spectral_band_um=(3.0, 5.0),
        nedt_mk=20.0,
    ),
    detector_visible=DetectorSpec(
        detector_type=DetectorType.SI_CCD,
        width_pixels=2048,
        height_pixels=2048,
        pixel_pitch_um=7.0,
        operating_temp_k=280.0,
        spectral_band_um=(0.4, 0.9),
    ),
    performance=PerformanceSpec(
        detection_range_vehicle_km=28.0,
        recognition_range_vehicle_km=14.0,
        identification_range_vehicle_km=7.0,
    ),
    gimbal=GimbalSpec(
        azimuth_range_deg=(-180, 180),
        elevation_range_deg=(-150, 15),
        slew_rate_deg_s=60,
        weight_kg=95.0,
        power_w=400,
    ),
    has_laser_designator=False,
    has_laser_rangefinder=True,
    has_tracking=True,
    description="Reconnaissance pod for tactical aircraft",
    country="Israel",
    year_introduced=2005,
)

SENSORS["litening"] = SensorSpec(
    id="litening",
    name="LITENING G4",
    manufacturer="Rafael/Northrop Grumman",
    sensor_type=SensorType.TARGETING_POD,
    mount_type=MountType.AIRCRAFT_GIMBAL,
    optics_ir=OpticalSpec(
        focal_length_mm=550.0,
        aperture_mm=200.0,
        fov_narrow_deg=0.35,
        fov_wide_deg=12.0,
        zoom_range=(1.0, 35.0),
    ),
    optics_visible=OpticalSpec(
        focal_length_mm=600.0,
        aperture_mm=100.0,
        fov_narrow_deg=0.3,
        fov_wide_deg=14.0,
    ),
    detector_ir=DetectorSpec(
        detector_type=DetectorType.INSB,
        width_pixels=1280,
        height_pixels=1024,
        pixel_pitch_um=15.0,
        operating_temp_k=77.0,
        spectral_band_um=(3.0, 5.0),
        nedt_mk=18.0,
        full_well_e=3e6,
    ),
    detector_visible=DetectorSpec(
        detector_type=DetectorType.SI_CCD,
        width_pixels=1920,
        height_pixels=1080,
        pixel_pitch_um=5.5,
        operating_temp_k=280.0,
        spectral_band_um=(0.4, 0.9),
    ),
    performance=PerformanceSpec(
        detection_range_vehicle_km=45.0,
        recognition_range_vehicle_km=22.0,
        identification_range_vehicle_km=11.0,
        detection_range_person_km=15.0,
    ),
    gimbal=GimbalSpec(
        azimuth_range_deg=(-180, 180),
        elevation_range_deg=(-155, 15),
        slew_rate_deg_s=80,
        weight_kg=200.0,
        power_w=800,
    ),
    has_laser_designator=True,
    has_laser_rangefinder=True,
    has_laser_illuminator=True,
    has_tracking=True,
    description="Advanced targeting pod for tactical aircraft",
    country="Israel/USA",
    year_introduced=2010,
)

# -----------------------------------------------------------------------------
# Lockheed Martin Sniper
# -----------------------------------------------------------------------------

SENSORS["sniper_atp"] = SensorSpec(
    id="sniper_atp",
    name="Sniper Advanced Targeting Pod",
    manufacturer="Lockheed Martin",
    sensor_type=SensorType.TARGETING_POD,
    mount_type=MountType.AIRCRAFT_GIMBAL,
    optics_ir=OpticalSpec(
        focal_length_mm=600.0,
        aperture_mm=220.0,
        fov_narrow_deg=0.3,
        fov_wide_deg=10.0,
        zoom_range=(1.0, 35.0),
    ),
    optics_visible=OpticalSpec(
        focal_length_mm=650.0,
        aperture_mm=110.0,
        fov_narrow_deg=0.25,
        fov_wide_deg=12.0,
    ),
    detector_ir=DetectorSpec(
        detector_type=DetectorType.HGCDTE_MWIR,
        width_pixels=1280,
        height_pixels=1024,
        pixel_pitch_um=15.0,
        operating_temp_k=77.0,
        spectral_band_um=(3.0, 5.0),
        nedt_mk=17.0,
        full_well_e=4e6,
    ),
    detector_visible=DetectorSpec(
        detector_type=DetectorType.SI_CCD,
        width_pixels=2048,
        height_pixels=1536,
        pixel_pitch_um=5.0,
        operating_temp_k=280.0,
        spectral_band_um=(0.4, 0.9),
    ),
    performance=PerformanceSpec(
        detection_range_vehicle_km=50.0,
        recognition_range_vehicle_km=25.0,
        identification_range_vehicle_km=12.0,
        detection_range_person_km=18.0,
    ),
    gimbal=GimbalSpec(
        azimuth_range_deg=(-180, 180),
        elevation_range_deg=(-150, 15),
        slew_rate_deg_s=90,
        weight_kg=200.0,
        power_w=900,
    ),
    has_laser_designator=True,
    has_laser_rangefinder=True,
    has_laser_illuminator=True,
    has_tracking=True,
    description="High-performance targeting pod for F-15, F-16, F-18, etc.",
    country="USA",
    year_introduced=2007,
)

# -----------------------------------------------------------------------------
# FLIR Systems
# -----------------------------------------------------------------------------

SENSORS["star_safire_380hd"] = SensorSpec(
    id="star_safire_380hd",
    name="Star SAFIRE 380-HD",
    manufacturer="FLIR Systems",
    sensor_type=SensorType.GIMBAL_EO_IR,
    mount_type=MountType.AIRCRAFT_GIMBAL,
    optics_ir=OpticalSpec(
        focal_length_mm=450.0,
        aperture_mm=180.0,
        fov_narrow_deg=0.4,
        fov_wide_deg=18.0,
        zoom_range=(1.0, 45.0),
    ),
    optics_visible=OpticalSpec(
        focal_length_mm=400.0,
        aperture_mm=80.0,
        fov_narrow_deg=0.35,
        fov_wide_deg=22.0,
    ),
    detector_ir=DetectorSpec(
        detector_type=DetectorType.HGCDTE_MWIR,
        width_pixels=1280,
        height_pixels=1024,
        pixel_pitch_um=15.0,
        operating_temp_k=77.0,
        spectral_band_um=(3.0, 5.0),
        nedt_mk=20.0,
    ),
    detector_visible=DetectorSpec(
        detector_type=DetectorType.SI_CMOS,
        width_pixels=1920,
        height_pixels=1080,
        pixel_pitch_um=5.0,
        operating_temp_k=300.0,
        spectral_band_um=(0.4, 0.9),
    ),
    performance=PerformanceSpec(
        detection_range_vehicle_km=30.0,
        recognition_range_vehicle_km=15.0,
        identification_range_vehicle_km=7.5,
        detection_range_person_km=10.0,
    ),
    gimbal=GimbalSpec(
        azimuth_range_deg=(-180, 180),
        elevation_range_deg=(-120, 30),
        slew_rate_deg_s=75,
        weight_kg=48.0,
        power_w=350,
    ),
    has_laser_designator=True,
    has_laser_rangefinder=True,
    has_tracking=True,
    description="Multi-sensor airborne surveillance system",
    country="USA",
    year_introduced=2012,
)

SENSORS["ultra_8500"] = SensorSpec(
    id="ultra_8500",
    name="FLIR Ultra 8500",
    manufacturer="FLIR Systems",
    sensor_type=SensorType.GIMBAL_EO_IR,
    mount_type=MountType.AIRCRAFT_GIMBAL,
    optics_ir=OpticalSpec(
        focal_length_mm=300.0,
        aperture_mm=120.0,
        fov_narrow_deg=0.6,
        fov_wide_deg=20.0,
        zoom_range=(1.0, 30.0),
    ),
    optics_visible=OpticalSpec(
        focal_length_mm=250.0,
        aperture_mm=60.0,
        fov_narrow_deg=0.5,
        fov_wide_deg=25.0,
    ),
    detector_ir=DetectorSpec(
        detector_type=DetectorType.INSB,
        width_pixels=640,
        height_pixels=512,
        pixel_pitch_um=15.0,
        operating_temp_k=77.0,
        spectral_band_um=(3.0, 5.0),
        nedt_mk=25.0,
    ),
    detector_visible=DetectorSpec(
        detector_type=DetectorType.SI_CMOS,
        width_pixels=1920,
        height_pixels=1080,
        pixel_pitch_um=5.0,
        operating_temp_k=300.0,
        spectral_band_um=(0.4, 0.9),
    ),
    performance=PerformanceSpec(
        detection_range_vehicle_km=18.0,
        recognition_range_vehicle_km=9.0,
        identification_range_vehicle_km=4.5,
    ),
    gimbal=GimbalSpec(
        azimuth_range_deg=(-180, 180),
        elevation_range_deg=(-110, 30),
        slew_rate_deg_s=60,
        weight_kg=34.0,
        power_w=200,
    ),
    has_laser_designator=False,
    has_laser_rangefinder=True,
    has_tracking=True,
    description="Compact HD airborne surveillance gimbal",
    country="USA",
    year_introduced=2010,
)

# -----------------------------------------------------------------------------
# Thales Systems
# -----------------------------------------------------------------------------

SENSORS["catherine_xp"] = SensorSpec(
    id="catherine_xp",
    name="Catherine XP",
    manufacturer="Thales",
    sensor_type=SensorType.THERMAL_IMAGER,
    mount_type=MountType.GROUND_VEHICLE,
    optics_ir=OpticalSpec(
        focal_length_mm=100.0,
        aperture_mm=70.0,
        fov_narrow_deg=2.0,
        fov_wide_deg=8.0,
        zoom_range=(1.0, 4.0),
    ),
    detector_ir=DetectorSpec(
        detector_type=DetectorType.HGCDTE_LWIR,
        width_pixels=640,
        height_pixels=480,
        pixel_pitch_um=25.0,
        operating_temp_k=77.0,
        spectral_band_um=(8.0, 12.0),
        nedt_mk=30.0,
    ),
    performance=PerformanceSpec(
        detection_range_vehicle_km=8.0,
        recognition_range_vehicle_km=4.0,
        identification_range_vehicle_km=2.0,
    ),
    has_tracking=False,
    description="Thermal sight for ground vehicles and weapon systems",
    country="France",
    year_introduced=2008,
)

SENSORS["sophie_mf"] = SensorSpec(
    id="sophie_mf",
    name="Sophie MF",
    manufacturer="Thales",
    sensor_type=SensorType.HANDHELD,
    mount_type=MountType.HANDHELD,
    optics_ir=OpticalSpec(
        focal_length_mm=80.0,
        aperture_mm=50.0,
        fov_narrow_deg=2.5,
        fov_wide_deg=10.0,
        zoom_range=(1.0, 4.0),
    ),
    detector_ir=DetectorSpec(
        detector_type=DetectorType.MICROBOLOMETER,
        width_pixels=640,
        height_pixels=480,
        pixel_pitch_um=17.0,
        operating_temp_k=300.0,
        spectral_band_um=(8.0, 14.0),
        nedt_mk=50.0,
    ),
    performance=PerformanceSpec(
        detection_range_vehicle_km=4.0,
        recognition_range_vehicle_km=2.0,
        identification_range_vehicle_km=1.0,
        detection_range_person_km=1.5,
    ),
    has_laser_rangefinder=True,
    has_tracking=False,
    description="Handheld multifunction thermal/day camera",
    country="France",
    year_introduced=2012,
)

# -----------------------------------------------------------------------------
# Generic/Reference Sensors
# -----------------------------------------------------------------------------

SENSORS["generic_mwir_hd"] = SensorSpec(
    id="generic_mwir_hd",
    name="Generic HD MWIR Camera",
    manufacturer="Generic",
    sensor_type=SensorType.MWIR_CAMERA,
    mount_type=MountType.FIXED,
    optics_ir=OpticalSpec(
        focal_length_mm=100.0,
        aperture_mm=50.0,
        fov_narrow_deg=5.0,
        fov_wide_deg=20.0,
    ),
    detector_ir=DetectorSpec(
        detector_type=DetectorType.HGCDTE_MWIR,
        width_pixels=1280,
        height_pixels=1024,
        pixel_pitch_um=15.0,
        operating_temp_k=77.0,
        spectral_band_um=(3.0, 5.0),
        nedt_mk=25.0,
        full_well_e=2e6,
    ),
    performance=PerformanceSpec(
        detection_range_vehicle_km=10.0,
        recognition_range_vehicle_km=5.0,
        identification_range_vehicle_km=2.5,
    ),
    description="Reference HD MWIR camera for benchmarking",
    country="Generic",
    year_introduced=2015,
)

SENSORS["generic_lwir_hd"] = SensorSpec(
    id="generic_lwir_hd",
    name="Generic HD LWIR Camera",
    manufacturer="Generic",
    sensor_type=SensorType.LWIR_CAMERA,
    mount_type=MountType.FIXED,
    optics_ir=OpticalSpec(
        focal_length_mm=75.0,
        aperture_mm=50.0,
        fov_narrow_deg=6.0,
        fov_wide_deg=24.0,
    ),
    detector_ir=DetectorSpec(
        detector_type=DetectorType.HGCDTE_LWIR,
        width_pixels=1280,
        height_pixels=1024,
        pixel_pitch_um=12.0,
        operating_temp_k=77.0,
        spectral_band_um=(8.0, 12.0),
        nedt_mk=22.0,
        full_well_e=5e6,
    ),
    performance=PerformanceSpec(
        detection_range_vehicle_km=8.0,
        recognition_range_vehicle_km=4.0,
        identification_range_vehicle_km=2.0,
    ),
    description="Reference HD LWIR camera for benchmarking",
    country="Generic",
    year_introduced=2015,
)

SENSORS["generic_uncooled"] = SensorSpec(
    id="generic_uncooled",
    name="Generic Uncooled Microbolometer",
    manufacturer="Generic",
    sensor_type=SensorType.THERMAL_IMAGER,
    mount_type=MountType.FIXED,
    optics_ir=OpticalSpec(
        focal_length_mm=50.0,
        aperture_mm=25.0,
        fov_narrow_deg=12.0,
        fov_wide_deg=36.0,
    ),
    detector_ir=DetectorSpec(
        detector_type=DetectorType.MICROBOLOMETER,
        width_pixels=640,
        height_pixels=480,
        pixel_pitch_um=17.0,
        operating_temp_k=300.0,
        spectral_band_um=(8.0, 14.0),
        nedt_mk=50.0,
        full_well_e=1e7,
    ),
    performance=PerformanceSpec(
        detection_range_vehicle_km=3.0,
        recognition_range_vehicle_km=1.5,
        identification_range_vehicle_km=0.75,
    ),
    description="Reference uncooled thermal camera",
    country="Generic",
    year_introduced=2010,
)

SENSORS["generic_swir"] = SensorSpec(
    id="generic_swir",
    name="Generic SWIR Camera",
    manufacturer="Generic",
    sensor_type=SensorType.SWIR_CAMERA,
    mount_type=MountType.FIXED,
    optics_ir=OpticalSpec(
        focal_length_mm=50.0,
        aperture_mm=35.0,
        fov_narrow_deg=8.0,
        fov_wide_deg=30.0,
    ),
    detector_ir=DetectorSpec(
        detector_type=DetectorType.INGAAS,
        width_pixels=640,
        height_pixels=512,
        pixel_pitch_um=20.0,
        operating_temp_k=253.0,  # TE cooled
        spectral_band_um=(0.9, 1.7),
        nedt_mk=0,  # Not applicable for SWIR
        full_well_e=500000,
    ),
    performance=PerformanceSpec(
        detection_range_vehicle_km=5.0,
        recognition_range_vehicle_km=2.5,
        identification_range_vehicle_km=1.25,
    ),
    description="Reference SWIR camera for low-light imaging",
    country="Generic",
    year_introduced=2012,
)

SENSORS["generic_visible_hd"] = SensorSpec(
    id="generic_visible_hd",
    name="Generic HD Visible Camera",
    manufacturer="Generic",
    sensor_type=SensorType.VISIBLE_CAMERA,
    mount_type=MountType.FIXED,
    optics_ir=OpticalSpec(  # Using same field for visible
        focal_length_mm=100.0,
        aperture_mm=50.0,
        fov_narrow_deg=4.0,
        fov_wide_deg=20.0,
    ),
    optics_visible=OpticalSpec(
        focal_length_mm=100.0,
        aperture_mm=50.0,
        fov_narrow_deg=4.0,
        fov_wide_deg=20.0,
    ),
    detector_ir=DetectorSpec(  # Placeholder for visible
        detector_type=DetectorType.SI_CMOS,
        width_pixels=1920,
        height_pixels=1080,
        pixel_pitch_um=5.0,
        operating_temp_k=300.0,
        spectral_band_um=(0.4, 0.7),
    ),
    detector_visible=DetectorSpec(
        detector_type=DetectorType.SI_CMOS,
        width_pixels=1920,
        height_pixels=1080,
        pixel_pitch_um=5.0,
        operating_temp_k=300.0,
        spectral_band_um=(0.4, 0.7),
    ),
    performance=PerformanceSpec(
        detection_range_vehicle_km=8.0,
        recognition_range_vehicle_km=4.0,
        identification_range_vehicle_km=2.0,
    ),
    description="Reference HD visible camera",
    country="Generic",
    year_introduced=2015,
)


# =============================================================================
# Sensor Library Interface
# =============================================================================

class SensorLibrary:
    """Central registry for sensor specifications."""

    @classmethod
    def get(cls, sensor_id: str) -> SensorSpec:
        """Get sensor specification by ID."""
        if sensor_id not in SENSORS:
            raise ValueError(f"Unknown sensor: {sensor_id}. Available: {list(SENSORS.keys())}")
        return SENSORS[sensor_id]

    @classmethod
    def list_all(cls) -> List[str]:
        """List all available sensor IDs."""
        return list(SENSORS.keys())

    @classmethod
    def list_by_type(cls, sensor_type: SensorType) -> List[str]:
        """List sensors by type."""
        return [
            sid for sid, spec in SENSORS.items()
            if spec.sensor_type == sensor_type
        ]

    @classmethod
    def list_by_manufacturer(cls, manufacturer: str) -> List[str]:
        """List sensors by manufacturer."""
        manufacturer_lower = manufacturer.lower()
        return [
            sid for sid, spec in SENSORS.items()
            if manufacturer_lower in spec.manufacturer.lower()
        ]

    @classmethod
    def search(cls, query: str) -> List[str]:
        """Search sensors by name or description."""
        query_lower = query.lower()
        return [
            sid for sid, spec in SENSORS.items()
            if (query_lower in spec.name.lower() or
                query_lower in spec.description.lower() or
                query_lower in spec.manufacturer.lower())
        ]

    @classmethod
    def get_info(cls, sensor_id: str) -> Dict[str, Any]:
        """Get detailed info about a sensor."""
        spec = cls.get(sensor_id)
        return {
            "id": spec.id,
            "name": spec.name,
            "manufacturer": spec.manufacturer,
            "type": spec.sensor_type.value,
            "mount": spec.mount_type.value,
            "detector_ir": {
                "type": spec.detector_ir.detector_type.value,
                "resolution": f"{spec.detector_ir.width_pixels}x{spec.detector_ir.height_pixels}",
                "pitch_um": spec.detector_ir.pixel_pitch_um,
                "band_um": spec.detector_ir.spectral_band_um,
                "nedt_mk": spec.detector_ir.nedt_mk,
            },
            "optics_ir": {
                "focal_mm": spec.optics_ir.focal_length_mm,
                "aperture_mm": spec.optics_ir.aperture_mm,
                "fov_deg": (spec.optics_ir.fov_narrow_deg, spec.optics_ir.fov_wide_deg),
            },
            "performance": {
                "detection_km": spec.performance.detection_range_vehicle_km,
                "recognition_km": spec.performance.recognition_range_vehicle_km,
                "identification_km": spec.performance.identification_range_vehicle_km,
            },
            "features": {
                "laser_designator": spec.has_laser_designator,
                "laser_rangefinder": spec.has_laser_rangefinder,
                "tracking": spec.has_tracking,
            },
            "description": spec.description,
        }

    @classmethod
    def compare(cls, sensor_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Compare multiple sensors."""
        return {sid: cls.get_info(sid) for sid in sensor_ids}


# Convenience functions
def get_sensor(sensor_id: str) -> SensorSpec:
    """Get sensor specification by ID."""
    return SensorLibrary.get(sensor_id)


def list_sensors(sensor_type: Optional[SensorType] = None) -> List[str]:
    """List available sensors, optionally filtered by type."""
    if sensor_type is not None:
        return SensorLibrary.list_by_type(sensor_type)
    return SensorLibrary.list_all()


def create_sensor_from_spec(
    sensor_id: str,
    channel: str = "ir",
    seed: Optional[int] = None,
) -> SensorModel:
    """Create a SensorModel from a library specification.

    Args:
        sensor_id: Sensor ID from library
        channel: Which channel ("ir" or "visible")
        seed: Random seed

    Returns:
        Configured SensorModel
    """
    spec = get_sensor(sensor_id)
    return spec.to_sensor_model(channel=channel, seed=seed)
