"""
Base sensor model integrating all sensor components.

Provides the SensorModel ABC and SimpleSensorModel implementation
that combines FPA, QE, noise, and ADC into a complete sensor simulation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from eosim.core.constants import PLANCK_H, SPEED_OF_LIGHT
from eosim.sensor.fpa import FPAConfig, FPAGeometry, DetectorProperties, DetectorType
from eosim.sensor.quantum_efficiency import QEModel, create_qe_model
from eosim.sensor.noise import NoiseModel, NoiseParameters, NoiseContributions
from eosim.sensor.adc import ADCModel, ADCParameters, ADCResult
from eosim.sensor.metrics import (
    SensorPerformanceMetrics,
    compute_nedt,
    snr_from_electrons,
    dynamic_range_electrons,
    compute_dL_dT_planck,
)


@dataclass
class SensorParameters:
    """Complete sensor configuration.

    Attributes:
        fpa: FPA configuration (geometry + detector)
        integration_time_s: Integration time in seconds
        optics_f_number: Optics f-number for irradiance calculation
        optics_transmission: Optics transmission factor
        cold_shield_efficiency: Cold shield efficiency (0-1)
        spectral_band_um: Spectral band (min, max) in micrometers
    """
    fpa: FPAConfig
    integration_time_s: float = 0.0002  # 200 μs default (suitable for 300K LWIR)
    optics_f_number: float = 2.0
    optics_transmission: float = 0.9
    cold_shield_efficiency: float = 1.0
    spectral_band_um: tuple[float, float] = (8.0, 12.0)

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.integration_time_s <= 0:
            raise ValueError("Integration time must be positive")
        if self.optics_f_number <= 0:
            raise ValueError("F-number must be positive")
        if not 0 < self.optics_transmission <= 1:
            raise ValueError("Transmission must be between 0 and 1")
        if not 0 <= self.cold_shield_efficiency <= 1:
            raise ValueError("Cold shield efficiency must be between 0 and 1")
        if self.spectral_band_um[0] >= self.spectral_band_um[1]:
            raise ValueError("Band minimum must be less than maximum")

    @property
    def pixel_area_m2(self) -> float:
        """Pixel area in square meters."""
        return self.fpa.geometry.pixel_area_m2

    @property
    def center_wavelength_um(self) -> float:
        """Center wavelength of spectral band."""
        return (self.spectral_band_um[0] + self.spectral_band_um[1]) / 2

    @property
    def bandwidth_um(self) -> float:
        """Spectral bandwidth in micrometers."""
        return self.spectral_band_um[1] - self.spectral_band_um[0]

    @property
    def omega_pixel(self) -> float:
        """Pixel solid angle in steradians.

        ω = A_pixel / f² = π / (4 × F#²) for pixel-sized aperture
        """
        return np.pi / (4 * self.optics_f_number**2)


@dataclass
class SensorResult:
    """Result of sensor simulation.

    Attributes:
        dn: Digital number output image
        electrons: Signal in electrons (before ADC)
        noisy_electrons: Signal with noise (before ADC)
        noise_contributions: Breakdown of noise sources
        adc_result: ADC conversion result
        metrics: Performance metrics (if computed)
    """
    dn: NDArray[np.integer]
    electrons: NDArray[np.floating]
    noisy_electrons: NDArray[np.floating]
    noise_contributions: Optional[NoiseContributions] = None
    adc_result: Optional[ADCResult] = None
    metrics: Optional[SensorPerformanceMetrics] = None


class SensorModel(ABC):
    """Abstract base class for sensor models.

    Defines the interface for sensor simulation:
    irradiance -> electrons -> noise -> DN
    """

    @abstractmethod
    def irradiance_to_electrons(
        self,
        irradiance: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Convert focal plane irradiance to electrons.

        Args:
            irradiance: Irradiance in W/m² (integrated over band)

        Returns:
            Signal in electrons
        """
        pass

    @abstractmethod
    def apply_noise(
        self,
        electrons: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Apply noise model to signal.

        Args:
            electrons: Signal in electrons

        Returns:
            Noisy signal in electrons
        """
        pass

    @abstractmethod
    def digitize(
        self,
        electrons: NDArray[np.floating],
    ) -> ADCResult:
        """Convert electrons to digital numbers.

        Args:
            electrons: Signal in electrons

        Returns:
            ADCResult with digital output
        """
        pass

    @abstractmethod
    def apply(
        self,
        irradiance: NDArray[np.floating],
    ) -> SensorResult:
        """Apply complete sensor model.

        Args:
            irradiance: Focal plane irradiance in W/m²

        Returns:
            SensorResult with DN image and diagnostics
        """
        pass

    @property
    @abstractmethod
    def params(self) -> SensorParameters:
        """Get sensor parameters."""
        pass


class SimpleSensorModel(SensorModel):
    """Simple sensor model implementation.

    Integrates FPA, QE, noise model, and ADC into a complete
    sensor simulation chain.
    """

    def __init__(
        self,
        params: SensorParameters,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize sensor model.

        Args:
            params: Sensor parameters
            seed: Random seed for noise generation
        """
        self._params = params
        self._seed = seed

        # Initialize QE model
        self._qe_model = create_qe_model(params.fpa.detector.detector_type)

        # Initialize noise model
        noise_params = NoiseParameters(
            read_noise_electrons=params.fpa.detector.read_noise_electrons,
            dark_current_e_per_s=params.fpa.detector.dark_current_e_per_s,
            prnu_percent=1.0,  # Default 1% PRNU
            dsnu_electrons=params.fpa.detector.read_noise_electrons * 0.5,
            temperature_k=params.fpa.detector.operating_temp_k,
        )
        self._noise_model = NoiseModel(
            noise_params,
            params.fpa.geometry.resolution,
            seed=seed,
        )

        # Initialize ADC
        adc_params = ADCParameters(
            bit_depth=params.fpa.detector.bit_depth,
            full_well_electrons=params.fpa.detector.full_well_electrons,
        )
        self._adc = ADCModel(adc_params)

        # Cache for effective QE
        self._effective_qe: Optional[float] = None

    @property
    def params(self) -> SensorParameters:
        """Get sensor parameters."""
        return self._params

    @property
    def qe_model(self) -> QEModel:
        """Get QE model."""
        return self._qe_model

    @property
    def noise_model(self) -> NoiseModel:
        """Get noise model."""
        return self._noise_model

    @property
    def adc(self) -> ADCModel:
        """Get ADC model."""
        return self._adc

    @property
    def effective_qe(self) -> float:
        """Get band-averaged QE."""
        if self._effective_qe is None:
            self._effective_qe = self._qe_model.integrate(
                self._params.spectral_band_um[0],
                self._params.spectral_band_um[1],
            )
        return self._effective_qe

    def irradiance_to_electrons(
        self,
        irradiance: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Convert focal plane irradiance to electrons.

        Signal chain:
        E [W/m²] → photons → electrons (via QE)

        electrons = E × A × t × η × λ / (h × c)

        Args:
            irradiance: Irradiance in W/m² (integrated over band)

        Returns:
            Signal in electrons
        """
        # Average photon energy in band
        center_wavelength_m = self._params.center_wavelength_um * 1e-6
        photon_energy = PLANCK_H * SPEED_OF_LIGHT / center_wavelength_m

        # Convert irradiance to photon flux (photons/m²/s)
        photon_flux = irradiance / photon_energy

        # Collect photons over pixel area and integration time
        photons = (
            photon_flux
            * self._params.pixel_area_m2
            * self._params.integration_time_s
        )

        # Convert to electrons via QE
        electrons = photons * self.effective_qe

        return electrons

    def radiance_to_electrons(
        self,
        radiance: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Convert scene radiance to electrons.

        Uses optics parameters to convert radiance to irradiance:
        E = π × L × τ / (4 × F#²)

        Args:
            radiance: Scene radiance in W/(m²·sr) integrated over band

        Returns:
            Signal in electrons
        """
        # Radiance to focal plane irradiance
        irradiance = (
            np.pi
            * radiance
            * self._params.optics_transmission
            * self._params.cold_shield_efficiency
            / (4 * self._params.optics_f_number**2)
        )

        return self.irradiance_to_electrons(irradiance)

    def apply_noise(
        self,
        electrons: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Apply noise model to signal.

        Args:
            electrons: Signal in electrons

        Returns:
            Noisy signal in electrons
        """
        return self._noise_model.apply(
            electrons,
            self._params.integration_time_s,
        )

    def digitize(
        self,
        electrons: NDArray[np.floating],
    ) -> ADCResult:
        """Convert electrons to digital numbers.

        Args:
            electrons: Signal in electrons

        Returns:
            ADCResult with digital output
        """
        return self._adc.convert(electrons)

    def apply(
        self,
        irradiance: NDArray[np.floating],
    ) -> SensorResult:
        """Apply complete sensor model.

        Args:
            irradiance: Focal plane irradiance in W/m²

        Returns:
            SensorResult with DN image and diagnostics
        """
        # Signal conversion
        electrons = self.irradiance_to_electrons(irradiance)

        # Compute noise contributions for middle of frame
        mid_signal = float(np.median(electrons))
        noise_contrib = self._noise_model.compute_contributions(
            mid_signal,
            self._params.integration_time_s,
        )

        # Apply noise
        noisy_electrons = self.apply_noise(electrons)

        # Digitize
        adc_result = self.digitize(noisy_electrons)

        return SensorResult(
            dn=adc_result.dn,
            electrons=electrons,
            noisy_electrons=noisy_electrons,
            noise_contributions=noise_contrib,
            adc_result=adc_result,
        )

    def apply_from_radiance(
        self,
        radiance: NDArray[np.floating],
    ) -> SensorResult:
        """Apply complete sensor model starting from scene radiance.

        Args:
            radiance: Scene radiance in W/(m²·sr)

        Returns:
            SensorResult with DN image and diagnostics
        """
        # Convert radiance to focal plane irradiance
        irradiance = (
            np.pi
            * radiance
            * self._params.optics_transmission
            * self._params.cold_shield_efficiency
            / (4 * self._params.optics_f_number**2)
        )

        return self.apply(irradiance)

    def compute_metrics(
        self,
        scene_temperature_k: float = 300.0,
    ) -> SensorPerformanceMetrics:
        """Compute sensor performance metrics.

        Args:
            scene_temperature_k: Reference scene temperature

        Returns:
            SensorPerformanceMetrics
        """
        # Compute reference signal (blackbody at scene temperature)
        from eosim.radiance.planck import planck_radiance_integrated

        L = planck_radiance_integrated(
            scene_temperature_k,
            self._params.spectral_band_um[0],
            self._params.spectral_band_um[1],
        )

        # Convert to electrons (use radiance_to_electrons formula)
        irradiance = (
            np.pi
            * L
            * self._params.optics_transmission
            * self._params.cold_shield_efficiency
            / (4 * self._params.optics_f_number**2)
        )

        center_wavelength_m = self._params.center_wavelength_um * 1e-6
        photon_energy = PLANCK_H * SPEED_OF_LIGHT / center_wavelength_m
        photons = (
            irradiance
            * self._params.pixel_area_m2
            * self._params.integration_time_s
            / photon_energy
        )
        signal_electrons = photons * self.effective_qe

        # Compute noise
        noise_contrib = self._noise_model.compute_contributions(
            signal_electrons,
            self._params.integration_time_s,
        )
        total_noise = noise_contrib.total_noise_electrons

        # Compute dL/dT
        dL_dT = compute_dL_dT_planck(
            scene_temperature_k,
            self._params.center_wavelength_um,
            self._params.bandwidth_um,
        )

        # NEDT
        nedt = compute_nedt(signal_electrons, total_noise, dL_dT, L)

        # SNR
        snr = snr_from_electrons(signal_electrons, total_noise)

        # Dynamic range
        dr_db = dynamic_range_electrons(
            self._params.fpa.detector.full_well_electrons,
            self._params.fpa.detector.read_noise_electrons,
        )

        # Well fill fraction
        well_fill = signal_electrons / self._params.fpa.detector.full_well_electrons

        return SensorPerformanceMetrics(
            nedt_k=nedt,
            snr=snr,
            dynamic_range_db=dr_db,
            well_fill_fraction=well_fill,
        )

    def reset_noise_seed(self, seed: int) -> None:
        """Reset random seed for noise generation."""
        self._noise_model.reset_seed(seed)


def create_sensor_model(
    detector_type: Union[DetectorType, str] = DetectorType.HGCDTE_MWIR,
    pixel_pitch_um: float = 15.0,
    resolution: tuple[int, int] = (480, 640),
    integration_time_s: Optional[float] = None,
    f_number: float = 2.0,
    spectral_band_um: Optional[tuple[float, float]] = None,
    seed: Optional[int] = None,
) -> SimpleSensorModel:
    """Factory function to create a sensor model.

    Args:
        detector_type: Detector type
        pixel_pitch_um: Pixel pitch in micrometers
        resolution: Array resolution (height, width)
        integration_time_s: Integration time (auto-selected if None)
        f_number: Optics f-number
        spectral_band_um: Spectral band (uses default for detector if None)
        seed: Random seed

    Returns:
        Configured SimpleSensorModel
    """
    if isinstance(detector_type, str):
        detector_type = DetectorType(detector_type)

    # Default spectral bands by detector type
    default_bands = {
        DetectorType.SI_CCD: (0.4, 0.7),
        DetectorType.SI_CMOS: (0.4, 0.9),
        DetectorType.INGAAS: (0.9, 1.7),
        DetectorType.INSB: (3.0, 5.0),
        DetectorType.HGCDTE_MWIR: (3.0, 5.0),
        DetectorType.HGCDTE_LWIR: (8.0, 12.0),
        DetectorType.QWIP: (8.0, 10.0),
        DetectorType.MICROBOLOMETER: (8.0, 14.0),
    }

    # Default integration times by detector type
    # Thermal IR detectors need shorter integration for 300K scenes
    default_integration_times = {
        DetectorType.SI_CCD: 0.033,        # 33 ms (video rate visible)
        DetectorType.SI_CMOS: 0.016,       # 16 ms (60 fps visible)
        DetectorType.INGAAS: 0.010,        # 10 ms (SWIR)
        DetectorType.INSB: 0.001,          # 1 ms (MWIR, cooled)
        DetectorType.HGCDTE_MWIR: 0.001,   # 1 ms (MWIR, cooled)
        DetectorType.HGCDTE_LWIR: 0.0002,  # 200 μs (LWIR, high photon flux)
        DetectorType.QWIP: 0.0005,         # 500 μs (LWIR)
        DetectorType.MICROBOLOMETER: 0.0001,  # 100 μs (scaled for photon model)
    }

    if spectral_band_um is None:
        spectral_band_um = default_bands.get(detector_type, (3.0, 5.0))

    if integration_time_s is None:
        integration_time_s = default_integration_times.get(detector_type, 0.001)

    # Create FPA config
    geometry = FPAGeometry(
        width_pixels=resolution[1],
        height_pixels=resolution[0],
        pixel_pitch_um=pixel_pitch_um,
    )

    detector = DetectorProperties.from_detector_type(
        detector_type,
        pixel_pitch_um=pixel_pitch_um,
    )

    fpa = FPAConfig(geometry=geometry, detector=detector)

    # Create sensor parameters
    params = SensorParameters(
        fpa=fpa,
        integration_time_s=integration_time_s,
        optics_f_number=f_number,
        spectral_band_um=spectral_band_um,
    )

    return SimpleSensorModel(params, seed=seed)
