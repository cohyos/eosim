"""
GPU Pipeline Accelerator for EOSIM.

Provides high-level GPU acceleration for the complete simulation pipeline
including scene rendering, atmospheric propagation, and sensor modeling.

Example 1: Accelerate scene rendering
    >>> from eosim.gpu import GPUAccelerator, AcceleratorConfig
    >>> config = AcceleratorConfig(
    ...     enable_gpu=True,
    ...     batch_size=16,
    ...     cache_psf=True,
    ... )
    >>> accelerator = GPUAccelerator(config)
    >>> # Accelerate radiance computation
    >>> temperatures = np.random.uniform(280, 320, (480, 640))
    >>> radiance = accelerator.compute_radiance(temperatures, wavelength_um=10.0)
    >>> print(f"Computed on: {accelerator.device_name}")

Example 2: Batch atmospheric transmission
    >>> from eosim.gpu import GPUAccelerator
    >>> acc = GPUAccelerator()
    >>> # Compute transmission for multiple paths
    >>> ranges_km = np.array([1, 2, 5, 10, 20])
    >>> transmissions = acc.batch_transmission(
    ...     wavelength_um=10.0,
    ...     ranges_km=ranges_km,
    ...     visibility_km=23.0,
    ... )
    >>> for r, t in zip(ranges_km, transmissions):
    ...     print(f"Range {r} km: τ = {t:.3f}")

Example 3: Full sensor simulation pipeline
    >>> from eosim.gpu import GPUAccelerator, AcceleratorConfig
    >>> config = AcceleratorConfig(enable_gpu=True)
    >>> acc = GPUAccelerator(config)
    >>> # Full pipeline: radiance → optics → sensor
    >>> scene_temp = np.random.uniform(290, 310, (480, 640))
    >>> result = acc.simulate_sensor(
    ...     temperature_map=scene_temp,
    ...     wavelength_um=10.0,
    ...     psf_sigma_pixels=1.5,
    ...     integration_time_s=0.01,
    ...     quantum_efficiency=0.7,
    ... )
    >>> print(f"Output DN range: {result.min():.0f} - {result.max():.0f}")
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np
from numpy.typing import NDArray

from .context import GPUContext, to_cpu, to_gpu, gpu_available
from .operations import GPUConvolver, GPURadiance, GPUNoiseGenerator


@dataclass
class AcceleratorConfig:
    """Configuration for GPU accelerator.

    Attributes:
        enable_gpu: Enable GPU acceleration if available
        device_id: CUDA device ID (default 0)
        batch_size: Batch size for parallel operations
        cache_psf: Cache PSF kernels for reuse
        cache_luts: Cache atmosphere LUTs
        precision: Floating point precision ('float32' or 'float64')
        memory_limit_gb: GPU memory limit (None = no limit)
    """

    enable_gpu: bool = True
    device_id: int = 0
    batch_size: int = 32
    cache_psf: bool = True
    cache_luts: bool = True
    precision: str = "float32"
    memory_limit_gb: Optional[float] = None


@dataclass
class SensorConfig:
    """Sensor configuration for simulation.

    Attributes:
        resolution: Image resolution (height, width)
        pixel_pitch_um: Pixel pitch in micrometers
        quantum_efficiency: Quantum efficiency (0-1)
        read_noise_e: Read noise in electrons
        dark_current_e_per_s: Dark current rate
        full_well_capacity_e: Full well capacity
        bit_depth: ADC bit depth
        integration_time_s: Integration time in seconds
    """

    resolution: Tuple[int, int] = (480, 640)
    pixel_pitch_um: float = 15.0
    quantum_efficiency: float = 0.7
    read_noise_e: float = 25.0
    dark_current_e_per_s: float = 100.0
    full_well_capacity_e: float = 100000
    bit_depth: int = 14
    integration_time_s: float = 0.01


class GPUAccelerator:
    """High-level GPU accelerator for EO/IR simulation pipeline.

    Provides accelerated implementations of:
    - Planck radiance computation
    - Atmospheric transmission
    - PSF/MTF convolution
    - Sensor noise modeling
    - Complete pipeline integration
    """

    def __init__(
        self,
        config: Optional[AcceleratorConfig] = None,
    ) -> None:
        """Initialize GPU accelerator.

        Args:
            config: Accelerator configuration
        """
        self._config = config or AcceleratorConfig()

        # Initialize GPU context
        force_cpu = not self._config.enable_gpu
        self._ctx = GPUContext(
            device_id=self._config.device_id,
            force_cpu=force_cpu,
        )

        # Initialize operation modules
        self._convolver = GPUConvolver(self._ctx)
        self._radiance = GPURadiance(self._ctx)
        self._noise_gen = GPUNoiseGenerator(self._ctx)

        # Caches
        self._psf_cache: Dict[str, Any] = {}
        self._lut_cache: Dict[str, Any] = {}

    @property
    def is_gpu(self) -> bool:
        """Check if using GPU."""
        return self._ctx.is_gpu

    @property
    def device_name(self) -> str:
        """Get device name."""
        return self._ctx.device_name

    @property
    def context(self) -> GPUContext:
        """Get GPU context."""
        return self._ctx

    def compute_radiance(
        self,
        temperature_k,
        wavelength_um: float,
        emissivity: float = 1.0,
    ):
        """Compute spectral radiance from temperature map.

        Args:
            temperature_k: Temperature array [K]
            wavelength_um: Wavelength [μm]
            emissivity: Surface emissivity (0-1)

        Returns:
            Spectral radiance array [W/(m²·sr·μm)]
        """
        radiance = self._radiance.planck_radiance(temperature_k, wavelength_um)
        if emissivity != 1.0:
            radiance = radiance * emissivity
        return radiance

    def compute_band_radiance(
        self,
        temperature_k,
        wavelength_min_um: float,
        wavelength_max_um: float,
        emissivity: float = 1.0,
        n_points: int = 50,
    ):
        """Compute band-integrated radiance.

        Args:
            temperature_k: Temperature array [K]
            wavelength_min_um: Minimum wavelength [μm]
            wavelength_max_um: Maximum wavelength [μm]
            emissivity: Surface emissivity (0-1)
            n_points: Integration points

        Returns:
            Band-integrated radiance [W/(m²·sr)]
        """
        radiance = self._radiance.band_integrated_radiance(
            temperature_k,
            wavelength_min_um,
            wavelength_max_um,
            n_points,
        )
        if emissivity != 1.0:
            radiance = radiance * emissivity
        return radiance

    def batch_transmission(
        self,
        wavelength_um: float,
        ranges_km,
        visibility_km: float = 23.0,
        altitude_km: float = 0.0,
    ):
        """Compute atmospheric transmission for multiple ranges.

        Uses simplified Beer-Lambert model with visibility-based extinction.

        Args:
            wavelength_um: Wavelength [μm]
            ranges_km: Array of slant ranges [km]
            visibility_km: Meteorological visibility [km]
            altitude_km: Mean path altitude [km]

        Returns:
            Transmission array (0-1)
        """
        xp = self._ctx.xp
        ranges = xp.asarray(ranges_km)

        # Extinction coefficient from visibility
        # Beer-Lambert: τ = exp(-β × R)
        # At visibility range, contrast drops to 2%
        # τ(V) = 0.02 → β = -ln(0.02) / V ≈ 3.912 / V
        beta_vis = 3.912 / visibility_km  # km⁻¹

        # Wavelength dependence (Angstrom exponent approximation)
        # For IR, scattering is less wavelength-dependent
        if wavelength_um > 3.0:
            wavelength_factor = 1.0
        else:
            # Visible/NIR: β ∝ λ^(-α), α ≈ 1.3
            wavelength_factor = (0.55 / wavelength_um) ** 1.3

        # Altitude dependence (exponential atmosphere)
        h_scale = 8.5  # Scale height [km]
        altitude_factor = np.exp(-altitude_km / h_scale)

        beta = beta_vis * wavelength_factor * altitude_factor

        # Transmission
        transmission = xp.exp(-beta * ranges)

        return transmission

    def apply_psf(
        self,
        image,
        sigma_pixels: float = 1.0,
        wavelength_um: Optional[float] = None,
        f_number: Optional[float] = None,
        pixel_pitch_um: Optional[float] = None,
    ):
        """Apply point spread function to image.

        Args:
            image: Input image
            sigma_pixels: PSF sigma in pixels (Gaussian model)
            wavelength_um: Wavelength for diffraction-limited PSF [μm]
            f_number: F-number for diffraction-limited PSF
            pixel_pitch_um: Pixel pitch for diffraction-limited PSF [μm]

        Returns:
            PSF-blurred image
        """
        xp = self._ctx.xp

        # Check cache
        cache_key = f"psf_{sigma_pixels}_{wavelength_um}_{f_number}"
        if self._config.cache_psf and cache_key in self._psf_cache:
            psf = self._psf_cache[cache_key]
        else:
            # Generate PSF kernel
            if wavelength_um is not None and f_number is not None:
                # Airy disk approximation
                if pixel_pitch_um is None:
                    pixel_pitch_um = 15.0  # Default
                airy_radius_um = 1.22 * wavelength_um * f_number
                sigma_pixels = airy_radius_um / pixel_pitch_um / 2.35

            # Gaussian PSF
            size = int(6 * sigma_pixels) | 1  # Ensure odd
            size = max(size, 3)
            center = size // 2

            y, x = xp.ogrid[:size, :size]
            psf = xp.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma_pixels**2))
            psf = psf / xp.sum(psf)

            if self._config.cache_psf:
                self._psf_cache[cache_key] = psf

        return self._convolver.apply_psf(image, psf)

    def apply_atmosphere(
        self,
        at_target_radiance,
        path_radiance: float,
        transmission: float,
    ):
        """Apply atmospheric effects to radiance.

        L_sensor = τ × L_target + L_path

        Args:
            at_target_radiance: At-target radiance
            path_radiance: Path radiance contribution
            transmission: Atmospheric transmission (0-1)

        Returns:
            At-sensor radiance
        """
        xp = self._ctx.xp
        radiance = xp.asarray(at_target_radiance)
        return transmission * radiance + path_radiance

    def radiance_to_electrons(
        self,
        radiance,
        wavelength_um: float,
        quantum_efficiency: float,
        integration_time_s: float,
        pixel_area_m2: float,
        solid_angle_sr: float,
        delta_wavelength_um: float = 1.0,
    ):
        """Convert radiance to photoelectrons.

        Args:
            radiance: Spectral radiance [W/(m²·sr·μm)]
            wavelength_um: Center wavelength [μm]
            quantum_efficiency: Quantum efficiency (0-1)
            integration_time_s: Integration time [s]
            pixel_area_m2: Pixel area [m²]
            solid_angle_sr: Solid angle per pixel [sr]
            delta_wavelength_um: Bandwidth [μm]

        Returns:
            Photoelectrons array
        """
        xp = self._ctx.xp
        radiance = xp.asarray(radiance)

        # Photon energy
        h = 6.62607015e-34  # J·s
        c = 299792458.0  # m/s
        wavelength_m = wavelength_um * 1e-6
        photon_energy = h * c / wavelength_m  # J/photon

        # Power per pixel
        power = radiance * delta_wavelength_um * pixel_area_m2 * solid_angle_sr  # W

        # Photon rate
        photon_rate = power / photon_energy  # photons/s

        # Electrons
        electrons = photon_rate * integration_time_s * quantum_efficiency

        return electrons

    def apply_sensor_noise(
        self,
        electrons,
        read_noise_e: float = 25.0,
        dark_current_e_per_s: float = 100.0,
        integration_time_s: float = 0.01,
        prnu_sigma: float = 0.01,
        dsnu_sigma: float = 10.0,
    ):
        """Apply sensor noise to electron signal.

        Args:
            electrons: Signal in electrons
            read_noise_e: Read noise [e⁻]
            dark_current_e_per_s: Dark current [e⁻/pixel/s]
            integration_time_s: Integration time [s]
            prnu_sigma: PRNU standard deviation (fractional)
            dsnu_sigma: DSNU standard deviation [e⁻]

        Returns:
            Noisy signal in electrons
        """
        return self._noise_gen.apply_all_noise(
            electrons,
            read_noise_e=read_noise_e,
            dark_current_e_per_s=dark_current_e_per_s,
            integration_time_s=integration_time_s,
            prnu_sigma=prnu_sigma,
            dsnu_sigma=dsnu_sigma,
        )

    def electrons_to_dn(
        self,
        electrons,
        full_well_capacity_e: float,
        bit_depth: int,
        gain: float = 1.0,
        offset: float = 0.0,
    ):
        """Convert electrons to digital numbers.

        Args:
            electrons: Signal in electrons
            full_well_capacity_e: Full well capacity [e⁻]
            bit_depth: ADC bit depth
            gain: System gain [DN/e⁻]
            offset: DN offset

        Returns:
            Digital numbers (integer array)
        """
        xp = self._ctx.xp
        electrons = xp.asarray(electrons)

        # Clip to full well
        electrons = xp.clip(electrons, 0, full_well_capacity_e)

        # Compute DN
        max_dn = (1 << bit_depth) - 1
        if gain == 1.0 and offset == 0.0:
            # Default: linear mapping from 0-FWC to 0-max_dn
            dn = (electrons / full_well_capacity_e) * max_dn
        else:
            dn = electrons * gain + offset

        # Clip and quantize
        dn = xp.clip(dn, 0, max_dn)
        dn = xp.round(dn).astype(xp.uint16)

        return dn

    def simulate_sensor(
        self,
        temperature_map,
        wavelength_um: float = 10.0,
        emissivity: float = 0.95,
        transmission: float = 0.8,
        path_radiance: float = 0.0,
        psf_sigma_pixels: float = 1.0,
        integration_time_s: float = 0.01,
        quantum_efficiency: float = 0.7,
        read_noise_e: float = 25.0,
        dark_current_e_per_s: float = 100.0,
        full_well_capacity_e: float = 100000,
        bit_depth: int = 14,
        pixel_pitch_um: float = 15.0,
        f_number: float = 2.0,
    ):
        """Run complete sensor simulation pipeline.

        Pipeline: Temp → Radiance → Atmosphere → PSF → Electrons → Noise → DN

        Args:
            temperature_map: Scene temperature [K]
            wavelength_um: Center wavelength [μm]
            emissivity: Scene emissivity
            transmission: Atmospheric transmission
            path_radiance: Path radiance [W/(m²·sr·μm)]
            psf_sigma_pixels: PSF blur [pixels]
            integration_time_s: Integration time [s]
            quantum_efficiency: Detector QE
            read_noise_e: Read noise [e⁻]
            dark_current_e_per_s: Dark current [e⁻/s]
            full_well_capacity_e: Full well [e⁻]
            bit_depth: ADC bits
            pixel_pitch_um: Pixel pitch [μm]
            f_number: F-number

        Returns:
            Digital number output image
        """
        xp = self._ctx.xp

        # 1. Temperature to radiance
        radiance = self.compute_radiance(temperature_map, wavelength_um, emissivity)

        # 2. Apply atmosphere
        radiance = self.apply_atmosphere(radiance, path_radiance, transmission)

        # 3. Apply PSF
        radiance = self.apply_psf(radiance, sigma_pixels=psf_sigma_pixels)

        # 4. Radiance to electrons
        pixel_area_m2 = (pixel_pitch_um * 1e-6) ** 2
        solid_angle_sr = pixel_area_m2 / (f_number * pixel_pitch_um * 1e-6 * 4) ** 2
        solid_angle_sr = 1.0 / (4 * f_number**2)  # Simplified

        electrons = self.radiance_to_electrons(
            radiance,
            wavelength_um,
            quantum_efficiency,
            integration_time_s,
            pixel_area_m2,
            solid_angle_sr,
        )

        # 5. Apply noise
        electrons = self.apply_sensor_noise(
            electrons,
            read_noise_e=read_noise_e,
            dark_current_e_per_s=dark_current_e_per_s,
            integration_time_s=integration_time_s,
        )

        # 6. Convert to DN
        dn = self.electrons_to_dn(electrons, full_well_capacity_e, bit_depth)

        return dn

    def benchmark(
        self,
        resolution: Tuple[int, int] = (1024, 1024),
        n_iterations: int = 10,
    ) -> Dict[str, float]:
        """Benchmark GPU acceleration performance.

        Args:
            resolution: Test image resolution
            n_iterations: Number of iterations

        Returns:
            Dictionary of operation timings [ms]
        """
        import time

        xp = self._ctx.xp
        results = {}

        # Create test data
        temp_map = xp.random.uniform(280, 320, resolution).astype(xp.float32)
        image = xp.random.uniform(0, 100, resolution).astype(xp.float32)

        # Benchmark radiance computation
        self._ctx.synchronize()
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = self.compute_radiance(temp_map, 10.0)
        self._ctx.synchronize()
        results["radiance_ms"] = (time.perf_counter() - start) * 1000 / n_iterations

        # Benchmark PSF convolution
        self._ctx.synchronize()
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = self.apply_psf(image, sigma_pixels=2.0)
        self._ctx.synchronize()
        results["psf_ms"] = (time.perf_counter() - start) * 1000 / n_iterations

        # Benchmark noise application
        electrons = xp.ones(resolution) * 10000
        self._ctx.synchronize()
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = self.apply_sensor_noise(electrons)
        self._ctx.synchronize()
        results["noise_ms"] = (time.perf_counter() - start) * 1000 / n_iterations

        # Benchmark full pipeline
        self._ctx.synchronize()
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = self.simulate_sensor(temp_map)
        self._ctx.synchronize()
        results["full_pipeline_ms"] = (
            (time.perf_counter() - start) * 1000 / n_iterations
        )

        results["device"] = self.device_name
        results["resolution"] = resolution

        return results

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._psf_cache.clear()
        self._lut_cache.clear()

    def memory_info(self) -> Optional[Dict[str, float]]:
        """Get GPU memory information.

        Returns:
            Dictionary with memory info, or None if using CPU
        """
        info = self._ctx.memory_info()
        if info is None:
            return None
        return {
            "total_gb": info.total_gb,
            "free_gb": info.free_gb,
            "used_gb": info.used_gb,
        }
