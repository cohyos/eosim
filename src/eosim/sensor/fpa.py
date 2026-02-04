"""
Focal Plane Array (FPA) geometry and pixel properties.

Defines FPA geometry, pixel layout, and physical properties for
various detector types used in EO/IR imaging systems.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray


class DetectorType(Enum):
    """Common FPA detector technologies."""

    SI_CCD = "si_ccd"           # Silicon CCD (visible)
    SI_CMOS = "si_cmos"         # Silicon CMOS (visible)
    INGAAS = "ingaas"           # InGaAs (SWIR)
    INSB = "insb"               # Indium Antimonide (MWIR)
    HGCDTE_MWIR = "hgcdte_mwir" # Mercury Cadmium Telluride (MWIR)
    HGCDTE_LWIR = "hgcdte_lwir" # Mercury Cadmium Telluride (LWIR)
    QWIP = "qwip"               # Quantum Well (LWIR)
    MICROBOLOMETER = "microbolometer"  # Uncooled thermal


@dataclass
class FPAGeometry:
    """FPA geometric properties.

    Attributes:
        width_pixels: Number of columns
        height_pixels: Number of rows
        pixel_pitch_um: Pixel pitch in micrometers
        fill_factor: Active area fraction (0-1)
        active_area_offset: Offset of active area from sensor edge (x, y) in pixels
    """

    width_pixels: int
    height_pixels: int
    pixel_pitch_um: float
    fill_factor: float = 1.0
    active_area_offset: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        """Validate geometry parameters."""
        if self.width_pixels <= 0 or self.height_pixels <= 0:
            raise ValueError("Pixel dimensions must be positive")
        if self.pixel_pitch_um <= 0:
            raise ValueError("Pixel pitch must be positive")
        if not 0 < self.fill_factor <= 1:
            raise ValueError("Fill factor must be between 0 and 1")

    @property
    def resolution(self) -> tuple[int, int]:
        """Return (height, width) for array shape."""
        return (self.height_pixels, self.width_pixels)

    @property
    def n_pixels(self) -> int:
        """Total number of pixels."""
        return self.width_pixels * self.height_pixels

    @property
    def pixel_count(self) -> int:
        """Total number of pixels (alias for n_pixels)."""
        return self.n_pixels

    @property
    def pixel_area_um2(self) -> float:
        """Physical pixel area in square micrometers."""
        return self.pixel_pitch_um ** 2

    @property
    def pixel_area_m2(self) -> float:
        """Physical pixel area in square meters."""
        return (self.pixel_pitch_um * 1e-6) ** 2

    @property
    def active_pixel_area_m2(self) -> float:
        """Active (light-sensitive) pixel area in square meters."""
        return self.pixel_area_m2 * self.fill_factor

    @property
    def sensor_width_mm(self) -> float:
        """Total sensor width in millimeters."""
        return self.width_pixels * self.pixel_pitch_um / 1000

    @property
    def sensor_height_mm(self) -> float:
        """Total sensor height in millimeters."""
        return self.height_pixels * self.pixel_pitch_um / 1000

    @property
    def sensor_diagonal_mm(self) -> float:
        """Sensor diagonal in millimeters."""
        return np.sqrt(self.sensor_width_mm**2 + self.sensor_height_mm**2)

    @property
    def array_width_mm(self) -> float:
        """Array width in millimeters (alias for sensor_width_mm)."""
        return self.sensor_width_mm

    @property
    def array_height_mm(self) -> float:
        """Array height in millimeters (alias for sensor_height_mm)."""
        return self.sensor_height_mm

    @property
    def diagonal_mm(self) -> float:
        """Diagonal in millimeters (alias for sensor_diagonal_mm)."""
        return self.sensor_diagonal_mm

    @property
    def aspect_ratio(self) -> float:
        """Width to height ratio."""
        return self.width_pixels / self.height_pixels

    def pixel_center_coords(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Get pixel center coordinates in micrometers from sensor center.

        Returns:
            Tuple of (x_coords, y_coords) arrays, each shape (height, width)
        """
        # Pixel indices
        x_idx = np.arange(self.width_pixels)
        y_idx = np.arange(self.height_pixels)

        # Convert to physical coordinates (centered on sensor)
        x_um = (x_idx - self.width_pixels / 2 + 0.5) * self.pixel_pitch_um
        y_um = (y_idx - self.height_pixels / 2 + 0.5) * self.pixel_pitch_um

        # Create 2D grids
        xx, yy = np.meshgrid(x_um, y_um)

        return xx, yy

    def pixel_corners(
        self,
        pixel_x: int,
        pixel_y: int,
    ) -> tuple[float, float, float, float]:
        """Get corner coordinates of a specific pixel in micrometers.

        Args:
            pixel_x: Pixel column index
            pixel_y: Pixel row index

        Returns:
            Tuple of (x_min, y_min, x_max, y_max) in micrometers from center
        """
        cx = (pixel_x - self.width_pixels / 2 + 0.5) * self.pixel_pitch_um
        cy = (pixel_y - self.height_pixels / 2 + 0.5) * self.pixel_pitch_um
        half = self.pixel_pitch_um / 2

        return (cx - half, cy - half, cx + half, cy + half)


@dataclass
class DetectorProperties:
    """Physical and electrical detector properties.

    Attributes:
        detector_type: Type of detector technology
        operating_temp_k: Detector operating temperature
        full_well_electrons: Maximum electrons per pixel
        read_noise_electrons: Read noise in electrons RMS
        dark_current_e_per_s: Dark current in electrons per second
        pixel_pitch_um: Pixel pitch in micrometers
        bit_depth: ADC bit depth
        conversion_gain_uv_e: Conversion gain (microvolts per electron)
        saturation_voltage_v: Output saturation voltage
    """

    detector_type: DetectorType
    operating_temp_k: float = 77.0
    full_well_electrons: float = 100000.0
    read_noise_electrons: float = 50.0
    dark_current_e_per_s: float = 1000.0
    pixel_pitch_um: float = 15.0
    bit_depth: int = 14
    conversion_gain_uv_e: float = 10.0
    saturation_voltage_v: float = 1.0

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.full_well_electrons <= 0:
            raise ValueError("Full well capacity must be positive")
        if self.read_noise_electrons < 0:
            raise ValueError("Read noise must be non-negative")
        if self.dark_current_e_per_s < 0:
            raise ValueError("Dark current must be non-negative")
        if self.operating_temp_k <= 0:
            raise ValueError("Operating temperature must be positive")

    @property
    def dynamic_range_db(self) -> float:
        """Dynamic range in decibels."""
        if self.read_noise_electrons <= 0:
            return float('inf')
        return 20 * np.log10(self.full_well_electrons / self.read_noise_electrons)

    @property
    def is_cooled(self) -> bool:
        """Whether detector requires cooling."""
        return self.operating_temp_k < 200

    @classmethod
    def from_detector_type(
        cls,
        detector_type: Union[DetectorType, str],
        pixel_pitch_um: float = 15.0,
    ) -> "DetectorProperties":
        """Create properties for common detector types.

        Args:
            detector_type: Detector type enum or string
            pixel_pitch_um: Pixel pitch in micrometers

        Returns:
            DetectorProperties with typical values for that detector type
        """
        if isinstance(detector_type, str):
            detector_type = DetectorType(detector_type)

        # Typical values for different detector types
        params = {
            DetectorType.SI_CCD: {
                "operating_temp_k": 253.0,  # -20C
                "full_well_electrons": 100000.0,
                "read_noise_electrons": 5.0,
                "dark_current_e_per_s": 10.0,
                "bit_depth": 16,
            },
            DetectorType.SI_CMOS: {
                "operating_temp_k": 300.0,
                "full_well_electrons": 50000.0,
                "read_noise_electrons": 3.0,
                "dark_current_e_per_s": 50.0,
                "bit_depth": 12,
            },
            DetectorType.INGAAS: {
                "operating_temp_k": 253.0,  # TE-cooled
                "full_well_electrons": 500000.0,
                "read_noise_electrons": 30.0,
                "dark_current_e_per_s": 5000.0,
                "bit_depth": 14,
            },
            DetectorType.INSB: {
                "operating_temp_k": 77.0,
                "full_well_electrons": 2000000.0,
                "read_noise_electrons": 50.0,
                "dark_current_e_per_s": 500.0,
                "bit_depth": 14,
            },
            DetectorType.HGCDTE_MWIR: {
                "operating_temp_k": 77.0,
                "full_well_electrons": 5000000.0,  # 5M e- typical for MWIR
                "read_noise_electrons": 40.0,
                "dark_current_e_per_s": 100.0,
                "bit_depth": 14,
            },
            DetectorType.HGCDTE_LWIR: {
                "operating_temp_k": 77.0,
                "full_well_electrons": 10000000.0,  # 10M e- typical for LWIR
                "read_noise_electrons": 100.0,
                "dark_current_e_per_s": 10000.0,
                "bit_depth": 14,
            },
            DetectorType.QWIP: {
                "operating_temp_k": 70.0,
                "full_well_electrons": 1000000.0,
                "read_noise_electrons": 200.0,
                "dark_current_e_per_s": 50000.0,
                "bit_depth": 14,
            },
            DetectorType.MICROBOLOMETER: {
                "operating_temp_k": 300.0,  # Uncooled
                "full_well_electrons": 10000000.0,  # 10M (scaled for photon model)
                "read_noise_electrons": 500.0,  # Higher noise (NETD ~50mK equiv)
                "dark_current_e_per_s": 0.0,
                "bit_depth": 14,
            },
        }

        if detector_type not in params:
            raise ValueError(f"Unknown detector type: {detector_type}")

        return cls(
            detector_type=detector_type,
            pixel_pitch_um=pixel_pitch_um,
            **params[detector_type],
        )


@dataclass
class SpectralResponse:
    """Detector spectral response characteristics.

    Can be used with wavelength/response arrays for interpolation,
    or with just min/max wavelengths for simple band definition.

    Attributes:
        wavelengths_um: Wavelength sample points (optional)
        response: Response values at each wavelength (optional)
        wavelength_min_um: Minimum wavelength sensitivity
        wavelength_max_um: Maximum wavelength sensitivity
        peak_wavelength_um: Wavelength of peak response
        cutoff_wavelength_um: Long-wave cutoff (50% response)
    """

    wavelengths_um: Optional[NDArray[np.floating]] = None
    response: Optional[NDArray[np.floating]] = None
    wavelength_min_um: Optional[float] = None
    wavelength_max_um: Optional[float] = None
    peak_wavelength_um: Optional[float] = None
    cutoff_wavelength_um: Optional[float] = None
    _interp: Optional[object] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Set defaults and create interpolator if needed."""
        from scipy.interpolate import interp1d

        if self.wavelengths_um is not None and self.response is not None:
            self.wavelengths_um = np.asarray(self.wavelengths_um)
            self.response = np.asarray(self.response)
            if self.wavelength_min_um is None:
                self.wavelength_min_um = float(self.wavelengths_um.min())
            if self.wavelength_max_um is None:
                self.wavelength_max_um = float(self.wavelengths_um.max())
            if self.peak_wavelength_um is None:
                idx = np.argmax(self.response)
                self.peak_wavelength_um = float(self.wavelengths_um[idx])
            # Create interpolator
            self._interp = interp1d(
                self.wavelengths_um,
                self.response,
                kind='linear',
                bounds_error=False,
                fill_value=0.0,
            )
        else:
            if self.peak_wavelength_um is None and self.wavelength_min_um is not None and self.wavelength_max_um is not None:
                self.peak_wavelength_um = (self.wavelength_min_um + self.wavelength_max_um) / 2
            if self.cutoff_wavelength_um is None and self.wavelength_max_um is not None:
                self.cutoff_wavelength_um = self.wavelength_max_um

    def __call__(self, wavelength_um: Union[float, NDArray]) -> Union[float, NDArray]:
        """Get response at specified wavelength(s).

        Args:
            wavelength_um: Wavelength(s) in micrometers

        Returns:
            Response value(s)
        """
        if self._interp is not None:
            result = self._interp(wavelength_um)
            if np.isscalar(wavelength_um):
                return float(result)
            return result
        # If no interpolator, return 1.0 in band, 0.0 outside
        if np.isscalar(wavelength_um):
            if self.wavelength_min_um <= wavelength_um <= self.wavelength_max_um:
                return 1.0
            return 0.0
        wl = np.asarray(wavelength_um)
        result = np.where(
            (wl >= self.wavelength_min_um) & (wl <= self.wavelength_max_um),
            1.0,
            0.0,
        )
        return result

    @property
    def bandwidth_um(self) -> float:
        """Spectral bandwidth in micrometers."""
        return self.wavelength_max_um - self.wavelength_min_um

    @property
    def center_wavelength_um(self) -> float:
        """Center wavelength."""
        return (self.wavelength_min_um + self.wavelength_max_um) / 2

    @classmethod
    def from_detector_type(
        cls,
        detector_type: Union[DetectorType, str],
    ) -> "SpectralResponse":
        """Create spectral response for common detector types."""
        if isinstance(detector_type, str):
            detector_type = DetectorType(detector_type)

        # Typical spectral ranges
        ranges = {
            DetectorType.SI_CCD: (0.35, 1.0, 0.55, 1.0),
            DetectorType.SI_CMOS: (0.35, 1.0, 0.55, 1.0),
            DetectorType.INGAAS: (0.9, 1.7, 1.3, 1.7),
            DetectorType.INSB: (1.0, 5.5, 4.5, 5.5),
            DetectorType.HGCDTE_MWIR: (1.0, 5.0, 4.0, 5.0),
            DetectorType.HGCDTE_LWIR: (7.5, 12.0, 10.0, 12.0),
            DetectorType.QWIP: (8.0, 10.0, 9.0, 10.0),
            DetectorType.MICROBOLOMETER: (7.5, 14.0, 10.0, 14.0),
        }

        if detector_type not in ranges:
            raise ValueError(f"Unknown detector type: {detector_type}")

        wl_min, wl_max, wl_peak, wl_cutoff = ranges[detector_type]
        return cls(wl_min, wl_max, wl_peak, wl_cutoff)


@dataclass
class FPAConfig:
    """Complete FPA configuration combining all properties.

    Attributes:
        geometry: FPA geometric properties
        detector: Detector physical properties
        spectral: Spectral response characteristics (optional)
        name: Optional descriptive name
    """

    geometry: FPAGeometry
    detector: DetectorProperties
    spectral: Optional[SpectralResponse] = None
    name: str = "FPA"

    @classmethod
    def create_standard(
        cls,
        format_name: str,
        detector_type: Union[DetectorType, str] = DetectorType.INSB,
    ) -> "FPAConfig":
        """Create FPA with standard format.

        Args:
            format_name: Standard format (e.g., "640x512", "1024x1024", "HD")
            detector_type: Detector technology

        Returns:
            FPAConfig for specified format
        """
        # Standard FPA formats
        formats = {
            "320x256": (320, 256, 30.0),
            "640x480": (640, 480, 25.0),
            "640x512": (640, 512, 15.0),
            "1024x768": (1024, 768, 15.0),
            "1024x1024": (1024, 1024, 15.0),
            "1280x1024": (1280, 1024, 12.0),
            "1920x1080": (1920, 1080, 10.0),
            "HD": (1920, 1080, 10.0),
            "2K": (2048, 2048, 10.0),
            "4K": (4096, 4096, 5.0),
        }

        if format_name not in formats:
            raise ValueError(f"Unknown format: {format_name}. Available: {list(formats.keys())}")

        width, height, pitch = formats[format_name]

        geometry = FPAGeometry(
            width_pixels=width,
            height_pixels=height,
            pixel_pitch_um=pitch,
        )

        detector = DetectorProperties.from_detector_type(detector_type)
        spectral = SpectralResponse.from_detector_type(detector_type)

        return cls(
            geometry=geometry,
            detector=detector,
            spectral=spectral,
            name=f"{format_name}_{detector_type.value if isinstance(detector_type, DetectorType) else detector_type}",
        )


def compute_pixel_solid_angle(
    pixel_pitch_um: float,
    focal_length_mm: float,
) -> float:
    """Compute solid angle subtended by a pixel.

    Args:
        pixel_pitch_um: Pixel pitch in micrometers
        focal_length_mm: Lens focal length in millimeters

    Returns:
        Solid angle in steradians
    """
    # Pixel pitch in meters
    d = pixel_pitch_um * 1e-6
    # Focal length in meters
    f = focal_length_mm * 1e-3

    # Small angle approximation: Ω ≈ (d/f)²
    return (d / f) ** 2


def compute_ifov(
    pixel_pitch_um: float,
    focal_length_mm: float,
) -> float:
    """Compute instantaneous field of view (IFOV) per pixel.

    Args:
        pixel_pitch_um: Pixel pitch in micrometers
        focal_length_mm: Lens focal length in millimeters

    Returns:
        IFOV in radians
    """
    return (pixel_pitch_um / 1000) / focal_length_mm


def compute_gsd(
    pixel_pitch_um: float,
    focal_length_mm: float,
    altitude_m: float,
) -> float:
    """Compute ground sample distance (GSD).

    Args:
        pixel_pitch_um: Pixel pitch in micrometers
        focal_length_mm: Lens focal length in millimeters
        altitude_m: Sensor altitude above ground in meters

    Returns:
        GSD in meters per pixel
    """
    ifov = compute_ifov(pixel_pitch_um, focal_length_mm)
    return ifov * altitude_m


def compute_fov(
    n_pixels: int,
    pixel_pitch_um: float,
    focal_length_mm: float,
) -> float:
    """Compute total field of view for one dimension.

    Args:
        n_pixels: Number of pixels in dimension
        pixel_pitch_um: Pixel pitch in micrometers
        focal_length_mm: Lens focal length in millimeters

    Returns:
        FOV in radians
    """
    # Sensor dimension in mm
    sensor_size_mm = n_pixels * pixel_pitch_um / 1000

    # FOV = 2 × arctan(sensor_size / (2 × focal_length))
    return 2 * np.arctan(sensor_size_mm / (2 * focal_length_mm))


def compute_fov_2d(
    geometry: FPAGeometry,
    focal_length_mm: float,
) -> tuple[float, float]:
    """Compute total field of view for both dimensions.

    Args:
        geometry: FPA geometry
        focal_length_mm: Lens focal length in millimeters

    Returns:
        Tuple of (horizontal_fov, vertical_fov) in radians
    """
    # Sensor dimensions in mm
    w_mm = geometry.sensor_width_mm
    h_mm = geometry.sensor_height_mm

    # FOV = 2 × arctan(sensor_size / (2 × focal_length))
    fov_h = 2 * np.arctan(w_mm / (2 * focal_length_mm))
    fov_v = 2 * np.arctan(h_mm / (2 * focal_length_mm))

    return (fov_h, fov_v)


def compute_nyquist_frequency(pixel_pitch_um: float) -> float:
    """Compute Nyquist spatial frequency.

    Args:
        pixel_pitch_um: Pixel pitch in micrometers

    Returns:
        Nyquist frequency in cycles per meter
    """
    pixel_pitch_m = pixel_pitch_um * 1e-6
    return 1 / (2 * pixel_pitch_m)
