"""
EOSIM Adjacency Effects Model (Stage B Enhancement).

Models atmospheric adjacency effect where scattered light from
surrounding terrain contributes to the at-sensor radiance.

Example 1: Compute adjacency contribution for a uniform background
    >>> from eosim.atmosphere import AdjacencyModel, PathGeometry, AtmosphereConditions
    >>> model = AdjacencyModel()
    >>> path = PathGeometry(ground_range_m=3000, altitude_end_m=500)
    >>> conditions = AtmosphereConditions(visibility_km=10.0)
    >>> target_radiance = 30.0  # W/(m²·sr·μm)
    >>> background_radiance = 25.0  # W/(m²·sr·μm)
    >>> adj = model.compute_adjacency(10.0, path, conditions, target_radiance, background_radiance)
    >>> print(f"Adjacency contribution: {adj:.2f} W/(m²·sr·μm)")

Example 2: Apply adjacency correction to an image
    >>> import numpy as np
    >>> radiance_image = np.random.randn(100, 100) * 5 + 30  # Scene radiance
    >>> corrected = model.apply_adjacency_to_image(radiance_image, 10.0, path, conditions)

Example 3: Analyze adjacency PSF
    >>> psf = model.adjacency_psf(radius_m=500, n_points=50)
    >>> effective_radius = model.effective_adjacency_radius(10.0, path, conditions)
    >>> print(f"Effective radius: {effective_radius:.0f} m")
"""

from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve

from eosim.atmosphere.base import PathGeometry, AtmosphereConditions


@dataclass
class AdjacencyResult:
    """Result of adjacency effect calculation.

    Attributes:
        adjacency_radiance: Scattered radiance from surroundings [W/(m²·sr·μm)]
        effective_reflectance: Effective background reflectance seen by sensor
        spherical_albedo: Atmospheric spherical albedo
        effective_radius_m: Effective radius of adjacency effect [m]
    """
    adjacency_radiance: float
    effective_reflectance: float
    spherical_albedo: float
    effective_radius_m: float


class AdjacencyModel:
    """Atmospheric adjacency effect model.

    The adjacency effect occurs when photons from the surrounding
    terrain are scattered into the sensor's field of view. This
    reduces contrast and affects spectral signatures.

    The model uses a simplified treatment based on:
    - Atmospheric scattering function
    - Background reflectance
    - Target-background contrast
    """

    def __init__(
        self,
        kernel_size: int = 51,
        max_radius_m: float = 1000.0,
    ) -> None:
        """Initialize adjacency model.

        Args:
            kernel_size: Size of adjacency kernel (pixels)
            max_radius_m: Maximum radius for adjacency effect [m]
        """
        self.kernel_size = kernel_size
        self.max_radius_m = max_radius_m

    def compute_adjacency(
        self,
        wavelength_um: float,
        path: PathGeometry,
        conditions: AtmosphereConditions,
        target_radiance: float,
        background_radiance: float,
        target_fraction: float = 0.01,
    ) -> AdjacencyResult:
        """Compute adjacency contribution for a point target.

        Args:
            wavelength_um: Wavelength [um]
            path: Path geometry
            conditions: Atmospheric conditions
            target_radiance: Target surface radiance [W/(m²·sr·μm)]
            background_radiance: Background radiance [W/(m²·sr·μm)]
            target_fraction: Fraction of FOV covered by target

        Returns:
            AdjacencyResult with adjacency contribution
        """
        # Estimate atmospheric scattering properties
        s_albedo = self._spherical_albedo(wavelength_um, conditions)
        tau = self._path_transmission(wavelength_um, path, conditions)

        # Effective radius scales with visibility
        r_eff = self._effective_radius(wavelength_um, path, conditions)

        # Adjacency contribution
        # L_adj = s × (1 - τ) × ρ_bg × E_sun / π
        # Simplified: scales with background radiance and scattering
        adjacency_factor = s_albedo * (1 - tau) * (1 - target_fraction)
        L_adj = adjacency_factor * background_radiance

        # Effective reflectance seen by sensor
        eff_refl = (
            target_fraction * target_radiance +
            (1 - target_fraction) * background_radiance +
            L_adj
        ) / background_radiance if background_radiance > 0 else 1.0

        return AdjacencyResult(
            adjacency_radiance=L_adj,
            effective_reflectance=eff_refl,
            spherical_albedo=s_albedo,
            effective_radius_m=r_eff,
        )

    def apply_adjacency_to_image(
        self,
        radiance_image: NDArray,
        wavelength_um: float,
        path: PathGeometry,
        conditions: AtmosphereConditions,
        gsd_m: float = 1.0,
    ) -> NDArray:
        """Apply adjacency effect to a radiance image.

        Args:
            radiance_image: 2D radiance image [W/(m²·sr·μm)]
            wavelength_um: Wavelength [um]
            path: Path geometry
            conditions: Atmospheric conditions
            gsd_m: Ground sample distance [m]

        Returns:
            Image with adjacency effect applied
        """
        # Get adjacency parameters
        s_albedo = self._spherical_albedo(wavelength_um, conditions)
        tau = self._path_transmission(wavelength_um, path, conditions)
        r_eff = self._effective_radius(wavelength_um, path, conditions)

        # Create adjacency kernel
        kernel = self._create_adjacency_kernel(r_eff, gsd_m)

        # Blur image with adjacency kernel
        blurred = convolve(radiance_image, kernel, mode='reflect')

        # Combine direct and scattered components
        # L_sensor = τ × L_target + (1-τ) × s × L_blurred
        adjacency_strength = (1 - tau) * s_albedo
        result = tau * radiance_image + adjacency_strength * blurred

        return result

    def _create_adjacency_kernel(
        self,
        effective_radius_m: float,
        gsd_m: float,
    ) -> NDArray:
        """Create adjacency convolution kernel.

        Uses exponential decay with distance, normalized to sum=1.
        """
        sigma_pixels = effective_radius_m / gsd_m
        size = max(3, min(self.kernel_size, int(6 * sigma_pixels) | 1))

        # Create distance array
        half = size // 2
        y, x = np.ogrid[-half:half+1, -half:half+1]
        r = np.sqrt(x**2 + y**2)

        # Exponential decay kernel (excluding center)
        kernel = np.exp(-r / sigma_pixels)
        kernel[half, half] = 0  # Remove center pixel

        # Normalize
        kernel /= kernel.sum()

        return kernel

    def adjacency_psf(
        self,
        radius_m: float = 500.0,
        n_points: int = 50,
    ) -> tuple[NDArray, NDArray]:
        """Get adjacency PSF as function of distance.

        Args:
            radius_m: Maximum radius [m]
            n_points: Number of points

        Returns:
            Tuple of (distances [m], psf values)
        """
        distances = np.linspace(0, radius_m, n_points)
        # Exponential decay PSF
        psf = np.exp(-distances / (radius_m / 3))
        psf /= psf.sum()
        return distances, psf

    def effective_adjacency_radius(
        self,
        wavelength_um: float,
        path: PathGeometry,
        conditions: AtmosphereConditions,
    ) -> float:
        """Get effective adjacency radius for given conditions.

        Args:
            wavelength_um: Wavelength [um]
            path: Path geometry
            conditions: Atmospheric conditions

        Returns:
            Effective radius [m]
        """
        return self._effective_radius(wavelength_um, path, conditions)

    def _spherical_albedo(
        self,
        wavelength_um: float,
        conditions: AtmosphereConditions,
    ) -> float:
        """Estimate atmospheric spherical albedo."""
        # Simplified estimate based on visibility
        # Higher visibility = less scattering = lower albedo
        base_albedo = 0.15  # Typical value

        # Scale with visibility (inverse relationship)
        vis_factor = 23.0 / conditions.visibility_km
        albedo = base_albedo * np.sqrt(vis_factor)

        # Wavelength dependence (more scattering at shorter wavelengths)
        if wavelength_um < 1.0:
            wave_factor = (0.55 / wavelength_um) ** 2
            albedo *= np.clip(wave_factor, 0.5, 2.0)
        elif wavelength_um > 3.0:
            albedo *= 0.3  # Less scattering in thermal IR

        return float(np.clip(albedo, 0.01, 0.5))

    def _path_transmission(
        self,
        wavelength_um: float,
        path: PathGeometry,
        conditions: AtmosphereConditions,
    ) -> float:
        """Estimate path transmission."""
        # Simple Beer-Lambert estimate
        beta = 3.912 / conditions.visibility_km  # Koschmieder
        R_km = path.slant_range_m / 1000.0

        # Wavelength scaling
        if wavelength_um < 1.0:
            beta *= (0.55 / wavelength_um) ** 1.5
        elif wavelength_um > 3.0:
            beta *= 0.5  # Less extinction in IR

        return float(np.exp(-beta * R_km))

    def _effective_radius(
        self,
        wavelength_um: float,
        path: PathGeometry,
        conditions: AtmosphereConditions,
    ) -> float:
        """Estimate effective radius of adjacency effect.

        Scales with altitude and visibility.
        """
        # Base radius scales with sensor altitude
        base_radius = path.altitude_end_m * 0.1

        # Visibility scaling (lower visibility = larger blur)
        vis_factor = np.sqrt(23.0 / conditions.visibility_km)

        # Wavelength scaling
        if wavelength_um < 1.0:
            wave_factor = (0.55 / wavelength_um)
        else:
            wave_factor = 1.0

        radius = base_radius * vis_factor * wave_factor

        return float(np.clip(radius, 10.0, self.max_radius_m))


class ContrastTransmission:
    """Contrast transmission model for atmospheric degradation.

    Models how atmospheric scattering reduces image contrast
    as a function of distance and visibility.
    """

    def __init__(self) -> None:
        """Initialize contrast transmission model."""
        pass

    def contrast_transmission(
        self,
        range_km: float,
        visibility_km: float,
        wavelength_um: float = 0.55,
    ) -> float:
        """Compute contrast transmission.

        C_r = C_0 × exp(-β × R)

        where C_r/C_0 is the contrast transmission.

        Args:
            range_km: Range [km]
            visibility_km: Visibility [km]
            wavelength_um: Wavelength [um]

        Returns:
            Contrast transmission (0-1)
        """
        # Extinction coefficient
        beta = 3.912 / visibility_km

        # Wavelength scaling
        beta *= (0.55 / wavelength_um) ** 1.3

        return float(np.exp(-beta * range_km))

    def maximum_detection_range(
        self,
        inherent_contrast: float,
        threshold_contrast: float,
        visibility_km: float,
        wavelength_um: float = 0.55,
    ) -> float:
        """Compute maximum range at which contrast is detectable.

        Args:
            inherent_contrast: Inherent target-background contrast
            threshold_contrast: Minimum detectable contrast
            visibility_km: Visibility [km]
            wavelength_um: Wavelength [um]

        Returns:
            Maximum detection range [km]
        """
        if inherent_contrast <= threshold_contrast:
            return 0.0

        beta = 3.912 / visibility_km
        beta *= (0.55 / wavelength_um) ** 1.3

        R_max = -np.log(threshold_contrast / inherent_contrast) / beta

        return float(max(0, R_max))

    def apparent_contrast(
        self,
        inherent_contrast: float,
        range_km: float,
        visibility_km: float,
    ) -> float:
        """Compute apparent contrast at range.

        Args:
            inherent_contrast: Inherent contrast at zero range
            range_km: Range [km]
            visibility_km: Visibility [km]

        Returns:
            Apparent contrast at range
        """
        ct = self.contrast_transmission(range_km, visibility_km)
        return inherent_contrast * ct
