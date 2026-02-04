"""
Image Generation Pipeline for EOSIM.

Integrates scene, radiance, atmosphere, optics, and sensor models
into a complete end-to-end simulation pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Union, Any
from enum import Enum
import numpy as np
from numpy.typing import NDArray


class SimulationMode(Enum):
    """Simulation mode/fidelity levels."""
    FAST = "fast"           # Simplified models, no spectral integration
    STANDARD = "standard"   # Full models, moderate spectral sampling
    HIGH_FIDELITY = "high_fidelity"  # Full models, dense spectral sampling


@dataclass
class SimulationConfig:
    """Configuration for simulation pipeline.

    Attributes:
        mode: Simulation fidelity mode
        spectral_samples: Number of spectral samples for integration
        include_atmosphere: Include atmospheric transmission
        include_optics_blur: Apply optical PSF
        include_detector_noise: Apply detector noise model
        include_motion_blur: Apply motion blur (if velocity specified)
        random_seed: Seed for reproducibility
    """
    mode: SimulationMode = SimulationMode.STANDARD
    spectral_samples: int = 20
    include_atmosphere: bool = True
    include_optics_blur: bool = True
    include_detector_noise: bool = True
    include_motion_blur: bool = False
    random_seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Adjust parameters based on mode."""
        if self.mode == SimulationMode.FAST:
            self.spectral_samples = min(self.spectral_samples, 5)
        elif self.mode == SimulationMode.HIGH_FIDELITY:
            self.spectral_samples = max(self.spectral_samples, 50)


@dataclass
class SceneInput:
    """Input scene specification.

    Can be temperature map, radiance map, or a scene model.

    Attributes:
        temperature_map: 2D temperature array [K]
        emissivity_map: 2D emissivity array (0-1)
        radiance_map: Pre-computed radiance [W/(m²·sr)]
        reflectance_map: Surface reflectance (0-1)
        background_temperature: Background/sky temperature [K]
        ambient_irradiance: Ambient irradiance [W/m²]
    """
    temperature_map: Optional[NDArray[np.floating]] = None
    emissivity_map: Optional[NDArray[np.floating]] = None
    radiance_map: Optional[NDArray[np.floating]] = None
    reflectance_map: Optional[NDArray[np.floating]] = None
    background_temperature: float = 290.0
    ambient_irradiance: float = 0.0

    def __post_init__(self) -> None:
        """Validate inputs."""
        if self.temperature_map is None and self.radiance_map is None:
            raise ValueError("Must provide either temperature_map or radiance_map")

        if self.temperature_map is not None and self.emissivity_map is None:
            # Default to blackbody
            self.emissivity_map = np.ones_like(self.temperature_map)

    @property
    def shape(self) -> tuple[int, int]:
        """Get scene shape."""
        if self.temperature_map is not None:
            return self.temperature_map.shape
        return self.radiance_map.shape


@dataclass
class PipelineResult:
    """Result from simulation pipeline.

    Attributes:
        digital_image: Final digital image [DN]
        radiance_image: Scene radiance at sensor [W/(m²·sr)]
        irradiance_image: Focal plane irradiance [W/m²]
        electrons_image: Signal in electrons
        intermediate_images: Dict of intermediate results
        metadata: Simulation metadata
    """
    digital_image: NDArray[np.integer]
    radiance_image: Optional[NDArray[np.floating]] = None
    irradiance_image: Optional[NDArray[np.floating]] = None
    electrons_image: Optional[NDArray[np.floating]] = None
    intermediate_images: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class SimulationPipeline:
    """Main simulation pipeline.

    Orchestrates the complete simulation chain:
    Scene → Radiance → Atmosphere → Optics → Sensor → Digital Image
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
    ) -> None:
        """Initialize pipeline.

        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

        # Component models (set via properties)
        self._atmosphere_model = None
        self._optics_model = None
        self._sensor_model = None

    @property
    def atmosphere_model(self):
        """Get atmosphere model."""
        return self._atmosphere_model

    @atmosphere_model.setter
    def atmosphere_model(self, model) -> None:
        """Set atmosphere model."""
        self._atmosphere_model = model

    @property
    def optics_model(self):
        """Get optics model."""
        return self._optics_model

    @optics_model.setter
    def optics_model(self, model) -> None:
        """Set optics model."""
        self._optics_model = model

    @property
    def sensor_model(self):
        """Get sensor model."""
        return self._sensor_model

    @sensor_model.setter
    def sensor_model(self, model) -> None:
        """Set sensor model."""
        self._sensor_model = model

    def compute_scene_radiance(
        self,
        scene: SceneInput,
        wavelength_um: Optional[float] = None,
    ) -> NDArray[np.floating]:
        """Compute scene radiance from temperature/emissivity.

        Args:
            scene: Scene input specification
            wavelength_um: Center wavelength for band (uses sensor band if None)

        Returns:
            Radiance image [W/(m²·sr)]
        """
        if scene.radiance_map is not None:
            return scene.radiance_map.copy()

        from eosim.radiance.planck import planck_radiance_integrated

        # Get spectral band from sensor if available
        if wavelength_um is None and self._sensor_model is not None:
            band = self._sensor_model.params.spectral_band_um
        else:
            band = (8.0, 12.0)  # Default LWIR

        # Compute blackbody radiance
        L_bb = planck_radiance_integrated(
            scene.temperature_map,
            band[0], band[1],
            n_samples=self.config.spectral_samples,
        )

        # Apply emissivity
        radiance = scene.emissivity_map * L_bb

        # Add reflected ambient (simplified)
        if scene.reflectance_map is not None:
            L_ambient = planck_radiance_integrated(
                scene.background_temperature,
                band[0], band[1],
            )
            radiance += scene.reflectance_map * L_ambient

        return radiance

    def apply_atmosphere(
        self,
        radiance: NDArray[np.floating],
        range_m: float = 1000.0,
    ) -> NDArray[np.floating]:
        """Apply atmospheric effects.

        Args:
            radiance: Input radiance [W/(m²·sr)]
            range_m: Range to target [m]

        Returns:
            Attenuated radiance with path radiance
        """
        if not self.config.include_atmosphere:
            return radiance

        if self._atmosphere_model is None:
            # Simple Beer-Lambert attenuation
            # Typical extinction ~0.1/km in LWIR
            extinction_per_m = 0.0001
            tau = np.exp(-extinction_per_m * range_m)
            return radiance * tau

        return self._atmosphere_model.apply(radiance, range_m)

    def apply_optics(
        self,
        radiance: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Apply optical system effects (blur, transmission).

        Args:
            radiance: Input radiance [W/(m²·sr)]

        Returns:
            Blurred radiance
        """
        if not self.config.include_optics_blur:
            return radiance

        if self._optics_model is not None:
            return self._optics_model.apply(radiance)

        # Default Gaussian blur
        from scipy.ndimage import gaussian_filter
        sigma = 0.8  # Moderate blur
        return gaussian_filter(radiance, sigma=sigma)

    def radiance_to_irradiance(
        self,
        radiance: NDArray[np.floating],
        f_number: float = 2.0,
        transmission: float = 0.9,
    ) -> NDArray[np.floating]:
        """Convert scene radiance to focal plane irradiance.

        E = π × L × τ / (4 × F#²)

        Args:
            radiance: Scene radiance [W/(m²·sr)]
            f_number: Optical f-number
            transmission: Optical transmission

        Returns:
            Focal plane irradiance [W/m²]
        """
        irradiance = np.pi * radiance * transmission / (4 * f_number**2)
        return irradiance

    def apply_sensor(
        self,
        irradiance: NDArray[np.floating],
    ) -> tuple[NDArray[np.integer], NDArray[np.floating]]:
        """Apply sensor model to convert irradiance to DN.

        Args:
            irradiance: Focal plane irradiance [W/m²]

        Returns:
            Tuple of (digital_image, electrons)
        """
        if self._sensor_model is not None:
            result = self._sensor_model.apply(irradiance)
            return result.dn, result.electrons

        # Simple default sensor model for LWIR
        # Parameters typical for HgCdTe LWIR detector
        from eosim.core.constants import PLANCK_H, SPEED_OF_LIGHT

        # Get band from config or use default LWIR
        band = self.config.spectral_band_um or (8.0, 12.0)
        center_wavelength_m = (band[0] + band[1]) / 2 * 1e-6

        photon_energy = PLANCK_H * SPEED_OF_LIGHT / center_wavelength_m
        pixel_area = (15e-6)**2
        integration_time = 0.001  # 1 ms for LWIR (shorter to avoid saturation)
        qe = 0.7

        photons = irradiance * pixel_area * integration_time / photon_energy
        electrons = photons * qe

        # Add noise if enabled
        if self.config.include_detector_noise:
            # Shot noise
            electrons = self._rng.poisson(np.maximum(electrons, 0)).astype(float)
            # Read noise
            electrons += self._rng.normal(0, 50, electrons.shape)

        # ADC (14-bit, 20M full well typical for LWIR HgCdTe)
        full_well = 20e6
        gain = full_well / 16383
        dn = np.clip(electrons / gain, 0, 16383).astype(np.uint16)

        return dn, electrons

    def run(
        self,
        scene: SceneInput,
        range_m: float = 1000.0,
        store_intermediates: bool = False,
    ) -> PipelineResult:
        """Run complete simulation pipeline.

        Args:
            scene: Input scene specification
            range_m: Range to target [m]
            store_intermediates: Store intermediate images

        Returns:
            PipelineResult with digital image and metadata
        """
        intermediates = {}

        # 1. Compute scene radiance
        radiance = self.compute_scene_radiance(scene)
        if store_intermediates:
            intermediates['scene_radiance'] = radiance.copy()

        # 2. Apply atmosphere
        radiance = self.apply_atmosphere(radiance, range_m)
        if store_intermediates:
            intermediates['at_sensor_radiance'] = radiance.copy()

        # 3. Apply optics (blur)
        radiance = self.apply_optics(radiance)
        if store_intermediates:
            intermediates['blurred_radiance'] = radiance.copy()

        # 4. Convert to irradiance
        f_number = 2.0
        transmission = 0.9
        if self._sensor_model is not None:
            f_number = self._sensor_model.params.optics_f_number
            transmission = self._sensor_model.params.optics_transmission

        irradiance = self.radiance_to_irradiance(radiance, f_number, transmission)
        if store_intermediates:
            intermediates['focal_plane_irradiance'] = irradiance.copy()

        # 5. Apply sensor model
        digital_image, electrons = self.apply_sensor(irradiance)

        # Build result
        metadata = {
            'range_m': range_m,
            'config': self.config,
            'scene_shape': scene.shape,
        }

        return PipelineResult(
            digital_image=digital_image,
            radiance_image=radiance,
            irradiance_image=irradiance,
            electrons_image=electrons,
            intermediate_images=intermediates,
            metadata=metadata,
        )


def create_pipeline(
    sensor_type: str = "lwir",
    mode: SimulationMode = SimulationMode.STANDARD,
    seed: Optional[int] = None,
) -> SimulationPipeline:
    """Factory function to create a configured pipeline.

    Args:
        sensor_type: Sensor type ("lwir", "mwir", "swir", "visible")
        mode: Simulation mode
        seed: Random seed

    Returns:
        Configured SimulationPipeline
    """
    from eosim.sensor.base import create_sensor_model
    from eosim.sensor.fpa import DetectorType

    config = SimulationConfig(mode=mode, random_seed=seed)
    pipeline = SimulationPipeline(config)

    # Create appropriate sensor
    sensor_map = {
        "lwir": DetectorType.HGCDTE_LWIR,
        "mwir": DetectorType.HGCDTE_MWIR,
        "swir": DetectorType.INGAAS,
        "visible": DetectorType.SI_CMOS,
    }

    detector_type = sensor_map.get(sensor_type.lower(), DetectorType.HGCDTE_LWIR)
    pipeline.sensor_model = create_sensor_model(
        detector_type=detector_type,
        seed=seed,
    )

    return pipeline


def quick_simulation(
    temperature_map: NDArray[np.floating],
    emissivity: Union[float, NDArray] = 0.95,
    sensor_type: str = "lwir",
    range_m: float = 1000.0,
    seed: Optional[int] = None,
) -> NDArray[np.integer]:
    """Quick simulation from temperature map to digital image.

    Args:
        temperature_map: 2D temperature array [K]
        emissivity: Emissivity (scalar or array)
        sensor_type: Sensor type
        range_m: Range to target [m]
        seed: Random seed

    Returns:
        Digital image [DN]
    """
    if np.isscalar(emissivity):
        emissivity = np.full_like(temperature_map, emissivity)

    scene = SceneInput(
        temperature_map=temperature_map,
        emissivity_map=emissivity,
    )

    pipeline = create_pipeline(sensor_type=sensor_type, seed=seed)
    result = pipeline.run(scene, range_m=range_m)

    return result.digital_image
