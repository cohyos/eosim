"""
Focal Plane Array (FPA) detector physics simulation.

This module provides physically-based detector modeling including:
- Detector material properties (InSb, MCT, microbolometer)
- Quantum efficiency and responsivity
- Integration time effects
- Well capacity and saturation
- Readout electronics modeling
- Temperature-dependent performance
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

from eosim.studio.radiometry import (
    SpectralBand, BAND_WAVELENGTHS,
    planck_radiance_array, integrate_band_radiance,
    PLANCK_H, SPEED_OF_LIGHT, BOLTZMANN_K
)


class DetectorMaterial(Enum):
    """Common infrared detector materials."""
    INSB = "insb"               # Indium Antimonide (MWIR, cooled)
    MCT = "mct"                 # Mercury Cadmium Telluride (MWIR/LWIR, cooled)
    QWIP = "qwip"               # Quantum Well IR Photodetector
    TYPE2_SL = "type2_sl"       # Type-II Superlattice
    MICROBOLOMETER = "bolometer" # Uncooled microbolometer
    PbSe = "pbse"               # Lead Selenide (MWIR)
    InGaAs = "ingaas"           # Indium Gallium Arsenide (SWIR)


class ReadoutType(Enum):
    """Readout Integrated Circuit (ROIC) types."""
    CTIA = "ctia"               # Capacitive Trans-Impedance Amplifier
    DI = "di"                   # Direct Injection
    SFD = "sfd"                 # Source Follower per Detector
    BDI = "bdi"                 # Buffered Direct Injection


@dataclass
class DetectorMaterialProperties:
    """Physical properties for a detector material."""
    name: str
    cutoff_wavelength_um: float  # Spectral cutoff
    operating_temp_k: float  # Nominal operating temperature
    quantum_efficiency: float  # Peak QE (0-1)
    dark_current_density: float  # A/cm² at operating temp
    detectivity_star: float  # D* in cm√Hz/W
    thermal_coefficient: float  # QE temp coefficient (%/K)

    # For photovoltaic detectors
    is_photovoltaic: bool = True
    bandgap_ev: float = 0.1  # Bandgap energy


# Standard detector material properties
DETECTOR_MATERIALS: Dict[DetectorMaterial, DetectorMaterialProperties] = {
    DetectorMaterial.INSB: DetectorMaterialProperties(
        name="Indium Antimonide",
        cutoff_wavelength_um=5.5,
        operating_temp_k=77.0,
        quantum_efficiency=0.85,
        dark_current_density=1e-7,
        detectivity_star=1e11,
        thermal_coefficient=-0.3,
        bandgap_ev=0.23
    ),
    DetectorMaterial.MCT: DetectorMaterialProperties(
        name="Mercury Cadmium Telluride",
        cutoff_wavelength_um=12.0,  # Tunable
        operating_temp_k=77.0,
        quantum_efficiency=0.70,
        dark_current_density=1e-5,
        detectivity_star=5e10,
        thermal_coefficient=-0.5,
        bandgap_ev=0.1
    ),
    DetectorMaterial.MICROBOLOMETER: DetectorMaterialProperties(
        name="Vanadium Oxide Microbolometer",
        cutoff_wavelength_um=14.0,
        operating_temp_k=300.0,
        quantum_efficiency=0.80,  # Absorption efficiency
        dark_current_density=0.0,  # Not applicable
        detectivity_star=1e9,
        thermal_coefficient=-2.0,
        is_photovoltaic=False,
        bandgap_ev=0.0
    ),
    DetectorMaterial.QWIP: DetectorMaterialProperties(
        name="Quantum Well IR Photodetector",
        cutoff_wavelength_um=9.0,
        operating_temp_k=70.0,
        quantum_efficiency=0.30,
        dark_current_density=1e-4,
        detectivity_star=1e10,
        thermal_coefficient=-0.8,
        bandgap_ev=0.14
    ),
    DetectorMaterial.InGaAs: DetectorMaterialProperties(
        name="Indium Gallium Arsenide",
        cutoff_wavelength_um=1.7,
        operating_temp_k=300.0,  # Can operate at room temp
        quantum_efficiency=0.90,
        dark_current_density=1e-9,
        detectivity_star=1e12,
        thermal_coefficient=-0.1,
        bandgap_ev=0.75
    ),
}


@dataclass
class FPAConfiguration:
    """Focal Plane Array configuration parameters."""
    # Array dimensions
    width_pixels: int = 640
    height_pixels: int = 512
    pixel_pitch_um: float = 15.0

    # Detector properties
    material: DetectorMaterial = DetectorMaterial.MCT
    fill_factor: float = 0.90  # Active area fraction
    operability: float = 0.995  # Fraction of working pixels

    # Operating parameters
    integration_time_us: float = 10000.0  # 10 ms default
    frame_rate_hz: float = 30.0

    # Well capacity
    well_capacity_electrons: int = 10_000_000  # 10 Me-

    # Readout
    readout_type: ReadoutType = ReadoutType.CTIA
    readout_noise_electrons: float = 100.0  # RMS

    # Gain stages
    preamp_gain: float = 1.0
    adc_bits: int = 14
    adc_gain: float = 1.0  # DN per electron

    # Operating temperature
    detector_temp_k: Optional[float] = None  # Override material default

    @property
    def array_size(self) -> Tuple[int, int]:
        return (self.height_pixels, self.width_pixels)

    @property
    def pixel_area_cm2(self) -> float:
        """Pixel area in cm²."""
        pitch_cm = self.pixel_pitch_um * 1e-4
        return pitch_cm**2 * self.fill_factor

    @property
    def integration_time_s(self) -> float:
        return self.integration_time_us * 1e-6

    @property
    def max_dn(self) -> int:
        return 2**self.adc_bits - 1


@dataclass
class DetectorResponse:
    """Detector response calculation results."""
    signal_electrons: np.ndarray  # Photo-generated electrons
    dark_electrons: np.ndarray  # Dark current electrons
    total_electrons: np.ndarray  # Total electrons
    digital_counts: np.ndarray  # ADC output
    saturation_map: np.ndarray  # Boolean saturation mask
    snr: np.ndarray  # Signal-to-noise ratio


class FPADetector:
    """
    Focal Plane Array detector physics simulation.

    Models the complete signal chain from incident radiance
    to digital output.
    """

    def __init__(self, config: FPAConfiguration):
        self.config = config
        self.material_props = DETECTOR_MATERIALS[config.material]

        # Initialize detector state
        self._init_nonuniformity()
        self._init_bad_pixels()

        # Precompute responsivity
        self._calculate_responsivity()

    def _init_nonuniformity(self, prnu_percent: float = 2.0):
        """Initialize Photo-Response Non-Uniformity (PRNU)."""
        # Gain variation (multiplicative)
        self.gain_map = 1.0 + (np.random.randn(*self.config.array_size) *
                               prnu_percent / 100.0)

        # Offset variation (additive) - Dark Signal Non-Uniformity (DSNU)
        dsnu_electrons = self.config.readout_noise_electrons * 0.5
        self.offset_map = np.random.randn(*self.config.array_size) * dsnu_electrons

    def _init_bad_pixels(self):
        """Initialize bad pixel map."""
        n_pixels = self.config.width_pixels * self.config.height_pixels
        n_bad = int(n_pixels * (1 - self.config.operability))

        self.bad_pixel_map = np.zeros(self.config.array_size, dtype=bool)
        if n_bad > 0:
            bad_indices = np.random.choice(n_pixels, n_bad, replace=False)
            bad_y = bad_indices // self.config.width_pixels
            bad_x = bad_indices % self.config.width_pixels
            self.bad_pixel_map[bad_y, bad_x] = True

    def _calculate_responsivity(self):
        """Calculate detector responsivity in A/W."""
        props = self.material_props

        # For photovoltaic detectors:
        # R = (η × q × λ) / (h × c)
        # where η = quantum efficiency, q = electron charge,
        # λ = wavelength, h = Planck's constant, c = speed of light

        if props.is_photovoltaic:
            # Use center wavelength of operating band
            if self.config.material == DetectorMaterial.INSB:
                center_wavelength_um = 4.0  # MWIR center
            elif self.config.material == DetectorMaterial.MCT:
                center_wavelength_um = 10.0  # LWIR center
            elif self.config.material == DetectorMaterial.InGaAs:
                center_wavelength_um = 1.5  # SWIR
            else:
                center_wavelength_um = props.cutoff_wavelength_um * 0.7

            wavelength_m = center_wavelength_um * 1e-6
            q = 1.602e-19  # Electron charge

            self.responsivity = (props.quantum_efficiency * q * wavelength_m /
                                (PLANCK_H * SPEED_OF_LIGHT))  # A/W
        else:
            # Microbolometer: thermal detector
            # Responsivity in V/W, simplified model
            self.responsivity = 1e4  # Typical value

    def calculate_dark_current(self, temp_k: Optional[float] = None) -> float:
        """
        Calculate dark current density at given temperature.

        Uses Arrhenius relationship for temperature dependence.

        Args:
            temp_k: Detector temperature (uses config default if None)

        Returns:
            Dark current density in A/cm²
        """
        if temp_k is None:
            temp_k = self.config.detector_temp_k or self.material_props.operating_temp_k

        nominal_temp = self.material_props.operating_temp_k
        j0 = self.material_props.dark_current_density

        if self.material_props.is_photovoltaic:
            # Arrhenius: J_dark ∝ exp(-Eg/(2kT))
            eg = self.material_props.bandgap_ev * 1.602e-19  # Convert to Joules
            factor = math.exp((eg / (2 * BOLTZMANN_K)) *
                             (1/nominal_temp - 1/temp_k))
            return j0 * factor
        else:
            # Bolometer: less temperature dependent
            return j0 * (temp_k / nominal_temp)**2

    def radiance_to_electrons(
        self,
        radiance: np.ndarray,
        band: SpectralBand = SpectralBand.LWIR
    ) -> np.ndarray:
        """
        Convert incident radiance to photo-generated electrons.

        Args:
            radiance: Incident radiance in W/(m²·sr)
            band: Spectral band

        Returns:
            Electron count per pixel
        """
        # Solid angle subtended by each pixel (assuming f/2 optics)
        # Ω = π / (4 × F/#²) for on-axis
        f_number = 2.0  # Assumed
        solid_angle = math.pi / (4 * f_number**2)

        # Power per pixel: P = L × A_pixel × Ω
        pixel_area_m2 = self.config.pixel_area_cm2 * 1e-4
        power_per_pixel = radiance * pixel_area_m2 * solid_angle  # W

        # Current: I = P × R (responsivity)
        current = power_per_pixel * self.responsivity  # A

        # Electrons: N = I × t_int / q
        q = 1.602e-19
        electrons = current * self.config.integration_time_s / q

        # Apply quantum efficiency wavelength dependence
        electrons *= self.material_props.quantum_efficiency

        return electrons.astype(np.float32)

    def simulate_frame(
        self,
        radiance_map: np.ndarray,
        include_dark: bool = True,
        include_noise: bool = True,
        include_nonuniformity: bool = True
    ) -> DetectorResponse:
        """
        Simulate complete detector response to incident radiance.

        Args:
            radiance_map: Incident radiance in W/(m²·sr), shape matching FPA
            include_dark: Include dark current
            include_noise: Include all noise sources
            include_nonuniformity: Include PRNU/DSNU

        Returns:
            DetectorResponse with all signal components
        """
        # Resize radiance to match FPA if needed
        if radiance_map.shape != self.config.array_size:
            from scipy.ndimage import zoom
            zoom_factors = (self.config.array_size[0] / radiance_map.shape[0],
                           self.config.array_size[1] / radiance_map.shape[1])
            radiance_map = zoom(radiance_map, zoom_factors, order=1)

        # 1. Photo-generated signal
        signal_electrons = self.radiance_to_electrons(radiance_map)

        # 2. Dark current
        if include_dark:
            dark_current = self.calculate_dark_current()
            # Dark electrons = J_dark × A_pixel × t_int / q
            q = 1.602e-19
            dark_e_per_pixel = (dark_current * self.config.pixel_area_cm2 *
                               self.config.integration_time_s / q)
            dark_electrons = np.full(self.config.array_size, dark_e_per_pixel,
                                    dtype=np.float32)
        else:
            dark_electrons = np.zeros(self.config.array_size, dtype=np.float32)

        # 3. Apply non-uniformity
        if include_nonuniformity:
            signal_electrons = signal_electrons * self.gain_map
            dark_electrons = dark_electrons + self.offset_map

        # 4. Total electrons
        total_electrons = signal_electrons + dark_electrons

        # 5. Check saturation
        saturation_map = total_electrons > self.config.well_capacity_electrons
        total_electrons = np.clip(total_electrons, 0,
                                 self.config.well_capacity_electrons)

        # 6. Add noise
        if include_noise:
            total_electrons = self._add_noise(total_electrons, signal_electrons)

        # 7. Convert to digital counts
        # DN = electrons × gain / well_capacity × max_DN
        gain = self.config.preamp_gain * self.config.adc_gain
        digital_counts = (total_electrons * gain /
                         self.config.well_capacity_electrons *
                         self.config.max_dn)
        digital_counts = np.clip(digital_counts, 0, self.config.max_dn)
        digital_counts = digital_counts.astype(np.uint16)

        # 8. Handle bad pixels
        digital_counts[self.bad_pixel_map] = 0

        # 9. Calculate SNR
        snr = self._calculate_snr(signal_electrons, total_electrons)

        return DetectorResponse(
            signal_electrons=signal_electrons,
            dark_electrons=dark_electrons,
            total_electrons=total_electrons,
            digital_counts=digital_counts,
            saturation_map=saturation_map,
            snr=snr
        )

    def _add_noise(
        self,
        electrons: np.ndarray,
        signal_electrons: np.ndarray
    ) -> np.ndarray:
        """Add all noise sources to electron count."""
        result = electrons.copy()

        # Shot noise (Poisson, approximated as Gaussian for large counts)
        shot_noise_sigma = np.sqrt(np.maximum(electrons, 0))
        result += np.random.randn(*electrons.shape) * shot_noise_sigma

        # Readout noise (Gaussian)
        readout_noise = self.config.readout_noise_electrons
        result += np.random.randn(*electrons.shape) * readout_noise

        # kTC noise (reset noise) for capacitive readouts
        if self.config.readout_type == ReadoutType.CTIA:
            # kTC noise: σ² = kT/C, typically ~100-500 e- RMS
            ktc_noise = 200.0  # Typical value
            result += np.random.randn(*electrons.shape) * ktc_noise

        return np.maximum(result, 0)

    def _calculate_snr(
        self,
        signal_electrons: np.ndarray,
        total_electrons: np.ndarray
    ) -> np.ndarray:
        """Calculate signal-to-noise ratio."""
        # Noise components
        shot_noise_var = np.maximum(total_electrons, 1)  # Poisson
        readout_noise_var = self.config.readout_noise_electrons**2
        ktc_noise_var = 200.0**2 if self.config.readout_type == ReadoutType.CTIA else 0

        total_noise_var = shot_noise_var + readout_noise_var + ktc_noise_var
        total_noise = np.sqrt(total_noise_var)

        snr = np.where(total_noise > 0, signal_electrons / total_noise, 0)
        return snr

    def calculate_netd(
        self,
        scene_temp_k: float = 300.0,
        band: SpectralBand = SpectralBand.LWIR
    ) -> float:
        """
        Calculate Noise Equivalent Temperature Difference.

        NETD = ΔT where SNR = 1

        Args:
            scene_temp_k: Scene temperature for calculation
            band: Spectral band

        Returns:
            NETD in Kelvin
        """
        # Calculate signal at scene temperature
        radiance = integrate_band_radiance(scene_temp_k, band)
        signal_e = float(np.mean(self.radiance_to_electrons(
            np.array([[radiance]]), band)))

        # Calculate signal at scene_temp + 1K
        radiance_plus = integrate_band_radiance(scene_temp_k + 1, band)
        signal_e_plus = float(np.mean(self.radiance_to_electrons(
            np.array([[radiance_plus]]), band)))

        delta_signal = signal_e_plus - signal_e

        if delta_signal <= 0:
            return float('inf')

        # Noise
        total_noise_var = (signal_e +  # Shot noise
                          self.config.readout_noise_electrons**2)
        noise = math.sqrt(total_noise_var)

        # NETD = noise / (dS/dT)
        netd = noise / delta_signal

        return netd

    def get_performance_summary(self) -> Dict:
        """Get summary of detector performance metrics."""
        netd = self.calculate_netd()

        return {
            'material': self.material_props.name,
            'array_size': f"{self.config.width_pixels}×{self.config.height_pixels}",
            'pixel_pitch_um': self.config.pixel_pitch_um,
            'integration_time_ms': self.config.integration_time_us / 1000,
            'well_capacity_Me': self.config.well_capacity_electrons / 1e6,
            'readout_noise_e': self.config.readout_noise_electrons,
            'quantum_efficiency': self.material_props.quantum_efficiency,
            'dark_current_nA': self.calculate_dark_current() * self.config.pixel_area_cm2 * 1e9,
            'netd_mk': netd * 1000,  # mK
            'operability': self.config.operability,
            'adc_bits': self.config.adc_bits,
        }


@dataclass
class TDIConfiguration:
    """Time Delay Integration configuration."""
    tdi_stages: int = 8  # Number of TDI stages
    scan_direction: str = "vertical"  # "vertical" or "horizontal"
    line_rate_hz: float = 1000.0  # Scan line rate


class TDIDetector(FPADetector):
    """
    Time Delay Integration detector for scanning sensors.

    TDI increases SNR by √N where N is the number of TDI stages.
    """

    def __init__(self, config: FPAConfiguration, tdi_config: TDIConfiguration):
        super().__init__(config)
        self.tdi_config = tdi_config

    def simulate_tdi_frame(
        self,
        radiance_lines: List[np.ndarray],
        include_noise: bool = True
    ) -> np.ndarray:
        """
        Simulate TDI scanning acquisition.

        Args:
            radiance_lines: List of radiance line arrays (one per scan line)
            include_noise: Include noise sources

        Returns:
            Accumulated TDI image
        """
        n_stages = self.tdi_config.tdi_stages

        if len(radiance_lines) < n_stages:
            raise ValueError(f"Need at least {n_stages} lines for TDI")

        # Integration time per stage
        stage_int_time = 1.0 / self.tdi_config.line_rate_hz
        original_int_time = self.config.integration_time_us
        self.config.integration_time_us = stage_int_time * 1e6

        # Accumulate TDI stages
        height = len(radiance_lines) - n_stages + 1
        width = len(radiance_lines[0])
        accumulated = np.zeros((height, width), dtype=np.float32)

        for i in range(height):
            # Sum n_stages consecutive lines
            for j in range(n_stages):
                line_radiance = radiance_lines[i + j]
                if line_radiance.ndim == 1:
                    line_radiance = line_radiance.reshape(1, -1)

                electrons = self.radiance_to_electrons(line_radiance)
                accumulated[i, :] += electrons.flatten()

        # Restore integration time
        self.config.integration_time_us = original_int_time

        # Add noise (reduced by √N due to averaging)
        if include_noise:
            noise_factor = 1.0 / math.sqrt(n_stages)
            noise = np.random.randn(*accumulated.shape) * (
                self.config.readout_noise_electrons * noise_factor)
            accumulated += noise

        # Convert to digital counts
        digital = (accumulated * self.config.preamp_gain /
                  self.config.well_capacity_electrons *
                  self.config.max_dn)
        digital = np.clip(digital, 0, self.config.max_dn).astype(np.uint16)

        return digital

    @property
    def effective_integration_time_us(self) -> float:
        """Effective integration time with TDI."""
        return self.tdi_config.tdi_stages / self.tdi_config.line_rate_hz * 1e6

    @property
    def snr_improvement(self) -> float:
        """SNR improvement factor from TDI."""
        return math.sqrt(self.tdi_config.tdi_stages)


class MicrobolometerDetector(FPADetector):
    """
    Uncooled microbolometer detector with thermal model.

    Microbolometers detect temperature changes in the sensing element
    rather than directly counting photons.
    """

    def __init__(self, config: FPAConfiguration):
        # Force microbolometer material
        config.material = DetectorMaterial.MICROBOLOMETER
        super().__init__(config)

        # Thermal parameters
        self.thermal_mass = 1e-9  # J/K, typical MEMS bolometer
        self.thermal_conductance = 1e-7  # W/K
        self.temperature_coefficient = 0.02  # 2%/K for VOx
        self.bias_voltage = 2.5  # V
        self.pixel_resistance = 100e3  # Ohms at operating point

    def calculate_thermal_time_constant(self) -> float:
        """Calculate thermal time constant τ = C/G."""
        return self.thermal_mass / self.thermal_conductance

    def radiance_to_signal(
        self,
        radiance: np.ndarray,
        ambient_temp_k: float = 300.0
    ) -> np.ndarray:
        """
        Convert radiance to voltage signal for microbolometer.

        Args:
            radiance: Incident radiance W/(m²·sr)
            ambient_temp_k: Ambient/substrate temperature

        Returns:
            Voltage change per pixel
        """
        # Absorbed power
        absorption = self.material_props.quantum_efficiency  # ~0.8 for VOx
        pixel_area_m2 = self.config.pixel_area_cm2 * 1e-4

        # Assuming f/1 optics for uncooled camera
        solid_angle = math.pi / 4  # f/1

        power_absorbed = radiance * pixel_area_m2 * solid_angle * absorption

        # Temperature rise: ΔT = P / G
        delta_t = power_absorbed / self.thermal_conductance

        # Resistance change: ΔR = R × α × ΔT
        delta_r = self.pixel_resistance * self.temperature_coefficient * delta_t

        # Voltage signal: ΔV = I × ΔR = (V_bias/R) × ΔR
        current = self.bias_voltage / self.pixel_resistance
        delta_v = current * delta_r

        return delta_v

    def calculate_netd(self, scene_temp_k: float = 300.0, **kwargs) -> float:
        """Calculate NETD for microbolometer."""
        # Simplified NETD calculation for uncooled detector
        # Typical values: 30-50 mK
        thermal_noise_v = 1e-6  # Johnson noise approximation

        # Scene radiance derivative
        radiance = integrate_band_radiance(scene_temp_k, SpectralBand.LWIR)
        radiance_plus = integrate_band_radiance(scene_temp_k + 1, SpectralBand.LWIR)

        signal = float(np.mean(self.radiance_to_signal(np.array([[radiance]]))))
        signal_plus = float(np.mean(self.radiance_to_signal(np.array([[radiance_plus]]))))

        responsivity = signal_plus - signal  # V/K

        if responsivity > 0:
            netd = thermal_noise_v / responsivity
        else:
            netd = 0.050  # Default 50 mK

        return netd


# Factory functions for common detector configurations
def create_hd_cooled_mwir() -> FPADetector:
    """Create HD 1280×1024 cooled MWIR InSb detector."""
    config = FPAConfiguration(
        width_pixels=1280,
        height_pixels=1024,
        pixel_pitch_um=15.0,
        material=DetectorMaterial.INSB,
        integration_time_us=5000,
        frame_rate_hz=60,
        well_capacity_electrons=8_000_000,
        readout_noise_electrons=50,
        adc_bits=14
    )
    return FPADetector(config)


def create_hd_cooled_lwir() -> FPADetector:
    """Create HD 1280×1024 cooled LWIR MCT detector."""
    config = FPAConfiguration(
        width_pixels=1280,
        height_pixels=1024,
        pixel_pitch_um=15.0,
        material=DetectorMaterial.MCT,
        integration_time_us=10000,
        frame_rate_hz=30,
        well_capacity_electrons=10_000_000,
        readout_noise_electrons=100,
        adc_bits=14
    )
    return FPADetector(config)


def create_vga_uncooled() -> MicrobolometerDetector:
    """Create VGA 640×480 uncooled microbolometer."""
    config = FPAConfiguration(
        width_pixels=640,
        height_pixels=480,
        pixel_pitch_um=17.0,
        material=DetectorMaterial.MICROBOLOMETER,
        integration_time_us=16000,  # ~60 Hz
        frame_rate_hz=60,
        well_capacity_electrons=0,  # Not applicable
        readout_noise_electrons=0,  # Different noise model
        adc_bits=14
    )
    return MicrobolometerDetector(config)


def create_hd_uncooled() -> MicrobolometerDetector:
    """Create HD 1920×1080 uncooled microbolometer."""
    config = FPAConfiguration(
        width_pixels=1920,
        height_pixels=1080,
        pixel_pitch_um=12.0,
        material=DetectorMaterial.MICROBOLOMETER,
        integration_time_us=16000,
        frame_rate_hz=30,
        well_capacity_electrons=0,
        readout_noise_electrons=0,
        adc_bits=14
    )
    return MicrobolometerDetector(config)
