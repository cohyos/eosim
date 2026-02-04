"""
Material definitions for EOSIM.

Provides classes for representing surface material properties including
emissivity, reflectance, BRDF models, and thermophysical properties.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from eosim.core.constants import MATERIAL_EMISSIVITY
from eosim.core.spectral import SpectralBand, WavelengthGrid


class BRDFType(Enum):
    """BRDF model types."""

    LAMBERTIAN = "lambertian"
    OREN_NAYAR = "oren_nayar"
    COOK_TORRANCE = "cook_torrance"
    PHONG = "phong"


@dataclass
class SpectralProperty:
    """A property that may vary with wavelength.

    Can represent constant values or wavelength-dependent spectra.
    """

    values: Union[float, NDArray[np.floating]]
    wavelengths_um: Optional[NDArray[np.floating]] = None

    def __post_init__(self) -> None:
        self.values = np.atleast_1d(np.asarray(self.values, dtype=np.float64))
        if self.wavelengths_um is not None:
            self.wavelengths_um = np.asarray(self.wavelengths_um, dtype=np.float64)

    @property
    def is_spectral(self) -> bool:
        """Check if this is a wavelength-dependent property."""
        return self.wavelengths_um is not None and len(self.values) > 1

    def at_wavelength(self, wavelength_um: Union[float, NDArray[np.floating]]) -> Union[float, NDArray[np.floating]]:
        """Get value at specific wavelength(s)."""
        if not self.is_spectral:
            return float(self.values[0])

        return np.interp(wavelength_um, self.wavelengths_um, self.values)

    def band_average(self, band: SpectralBand, n_samples: int = 50) -> float:
        """Compute average value over a spectral band."""
        if not self.is_spectral:
            return float(self.values[0])

        wavelengths = np.linspace(band.lambda_min_um, band.lambda_max_um, n_samples)
        values = self.at_wavelength(wavelengths)
        return float(np.mean(values))

    @classmethod
    def constant(cls, value: float) -> "SpectralProperty":
        """Create a constant (wavelength-independent) property."""
        return cls(values=value)

    @classmethod
    def from_spectrum(
        cls,
        wavelengths_um: NDArray[np.floating],
        values: NDArray[np.floating],
    ) -> "SpectralProperty":
        """Create a spectral property from wavelength-value pairs."""
        return cls(values=values, wavelengths_um=wavelengths_um)


@dataclass
class ThermalProperties:
    """Thermophysical properties of a material.

    Used for thermal balance calculations.
    """

    thermal_conductivity: float = 1.0  # W/(m·K)
    specific_heat: float = 1000.0  # J/(kg·K)
    density: float = 2000.0  # kg/m³
    thickness: float = 0.1  # m (for 1D thermal modeling)

    @property
    def thermal_diffusivity(self) -> float:
        """Thermal diffusivity α = k/(ρ·c) [m²/s]."""
        return self.thermal_conductivity / (self.density * self.specific_heat)

    @property
    def thermal_inertia(self) -> float:
        """Thermal inertia √(k·ρ·c) [J/(m²·K·s^0.5)]."""
        return np.sqrt(
            self.thermal_conductivity * self.density * self.specific_heat
        )

    @property
    def thermal_mass(self) -> float:
        """Thermal mass per unit area ρ·c·d [J/(m²·K)]."""
        return self.density * self.specific_heat * self.thickness


class BRDF(ABC):
    """Abstract base class for BRDF models."""

    @abstractmethod
    def evaluate(
        self,
        wi: NDArray[np.floating],
        wo: NDArray[np.floating],
        n: NDArray[np.floating],
    ) -> float:
        """Evaluate BRDF for given incident and outgoing directions.

        Args:
            wi: Incident direction (towards surface)
            wo: Outgoing direction (towards sensor)
            n: Surface normal

        Returns:
            BRDF value [1/sr]
        """
        pass

    @abstractmethod
    def sample(
        self,
        wi: NDArray[np.floating],
        n: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], float]:
        """Sample an outgoing direction and return (direction, pdf).

        Args:
            wi: Incident direction
            n: Surface normal

        Returns:
            Tuple of (outgoing direction, probability density)
        """
        pass


@dataclass
class LambertianBRDF(BRDF):
    """Lambertian (perfectly diffuse) BRDF.

    f_r = ρ/π where ρ is the reflectance.
    """

    reflectance: float = 0.5

    def evaluate(
        self,
        wi: NDArray[np.floating],
        wo: NDArray[np.floating],
        n: NDArray[np.floating],
    ) -> float:
        """Evaluate Lambertian BRDF (constant)."""
        return self.reflectance / np.pi

    def sample(
        self,
        wi: NDArray[np.floating],
        n: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], float]:
        """Sample cosine-weighted hemisphere."""
        # Generate random direction in hemisphere
        u1, u2 = np.random.random(2)
        r = np.sqrt(u1)
        theta = 2 * np.pi * u2

        # Local coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.sqrt(1 - u1)

        # Transform to world coordinates
        tangent, bitangent = self._create_basis(n)
        wo = x * tangent + y * bitangent + z * n
        wo = wo / np.linalg.norm(wo)

        # PDF for cosine-weighted sampling
        cos_theta = np.dot(wo, n)
        pdf = cos_theta / np.pi

        return wo, pdf

    def _create_basis(self, n: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Create orthonormal basis from normal."""
        if abs(n[0]) > 0.9:
            tangent = np.cross(np.array([0, 1, 0]), n)
        else:
            tangent = np.cross(np.array([1, 0, 0]), n)
        tangent = tangent / np.linalg.norm(tangent)
        bitangent = np.cross(n, tangent)
        return tangent, bitangent


@dataclass
class OrenNayarBRDF(BRDF):
    """Oren-Nayar BRDF for rough diffuse surfaces.

    Models microfacet scattering with roughness parameter.
    """

    reflectance: float = 0.5
    roughness: float = 0.3  # σ in radians

    def __post_init__(self) -> None:
        # Precompute coefficients
        sigma2 = self.roughness ** 2
        self.A = 1 - 0.5 * sigma2 / (sigma2 + 0.33)
        self.B = 0.45 * sigma2 / (sigma2 + 0.09)

    def evaluate(
        self,
        wi: NDArray[np.floating],
        wo: NDArray[np.floating],
        n: NDArray[np.floating],
    ) -> float:
        """Evaluate Oren-Nayar BRDF."""
        cos_theta_i = np.dot(wi, n)
        cos_theta_o = np.dot(wo, n)

        if cos_theta_i <= 0 or cos_theta_o <= 0:
            return 0.0

        theta_i = np.arccos(np.clip(cos_theta_i, -1, 1))
        theta_o = np.arccos(np.clip(cos_theta_o, -1, 1))

        alpha = max(theta_i, theta_o)
        beta = min(theta_i, theta_o)

        # Project directions onto plane perpendicular to normal
        wi_proj = wi - cos_theta_i * n
        wo_proj = wo - cos_theta_o * n

        wi_norm = np.linalg.norm(wi_proj)
        wo_norm = np.linalg.norm(wo_proj)

        if wi_norm > 1e-10 and wo_norm > 1e-10:
            cos_phi_diff = np.dot(wi_proj / wi_norm, wo_proj / wo_norm)
        else:
            cos_phi_diff = 0

        fr = (
            self.reflectance
            / np.pi
            * (self.A + self.B * max(0, cos_phi_diff) * np.sin(alpha) * np.tan(beta))
        )
        return fr

    def sample(
        self,
        wi: NDArray[np.floating],
        n: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], float]:
        """Sample using cosine-weighted distribution (approximation)."""
        # Use Lambertian sampling as approximation
        lamb = LambertianBRDF(self.reflectance)
        return lamb.sample(wi, n)


@dataclass
class Material:
    """Complete material definition.

    Combines radiometric, optical, and thermal properties.
    """

    name: str
    emissivity: SpectralProperty = field(default_factory=lambda: SpectralProperty.constant(0.9))
    reflectance: Optional[SpectralProperty] = None  # Computed from emissivity if None
    brdf: BRDF = field(default_factory=LambertianBRDF)
    thermal: ThermalProperties = field(default_factory=ThermalProperties)

    def __post_init__(self) -> None:
        # Compute reflectance from Kirchhoff's law if not provided
        if self.reflectance is None:
            if self.emissivity.is_spectral:
                self.reflectance = SpectralProperty(
                    values=1.0 - self.emissivity.values,
                    wavelengths_um=self.emissivity.wavelengths_um,
                )
            else:
                self.reflectance = SpectralProperty.constant(1.0 - float(self.emissivity.values[0]))

    def get_emissivity(self, wavelength_um: Optional[float] = None, band: Optional[SpectralBand] = None) -> float:
        """Get emissivity at wavelength or averaged over band."""
        if band is not None:
            return self.emissivity.band_average(band)
        elif wavelength_um is not None:
            return float(self.emissivity.at_wavelength(wavelength_um))
        else:
            return float(self.emissivity.values[0])

    def get_reflectance(self, wavelength_um: Optional[float] = None, band: Optional[SpectralBand] = None) -> float:
        """Get reflectance at wavelength or averaged over band."""
        if self.reflectance is None:
            return 1.0 - self.get_emissivity(wavelength_um, band)

        if band is not None:
            return self.reflectance.band_average(band)
        elif wavelength_um is not None:
            return float(self.reflectance.at_wavelength(wavelength_um))
        else:
            return float(self.reflectance.values[0])

    @classmethod
    def from_preset(cls, name: str) -> "Material":
        """Create material from preset name using default emissivities."""
        if name not in MATERIAL_EMISSIVITY:
            raise ValueError(f"Unknown material preset: {name}")

        emissivity = MATERIAL_EMISSIVITY[name]

        # Set typical thermal properties based on material type
        thermal = ThermalProperties()
        if "metal" in name or "steel" in name or "aluminum" in name:
            thermal = ThermalProperties(
                thermal_conductivity=50.0,
                specific_heat=500.0,
                density=7800.0,
            )
        elif "concrete" in name:
            thermal = ThermalProperties(
                thermal_conductivity=1.7,
                specific_heat=880.0,
                density=2400.0,
            )
        elif "water" in name:
            thermal = ThermalProperties(
                thermal_conductivity=0.6,
                specific_heat=4186.0,
                density=1000.0,
            )
        elif "vegetation" in name or "grass" in name or "forest" in name:
            thermal = ThermalProperties(
                thermal_conductivity=0.5,
                specific_heat=2000.0,
                density=800.0,
            )
        elif "soil" in name or "sand" in name:
            thermal = ThermalProperties(
                thermal_conductivity=0.3,
                specific_heat=800.0,
                density=1500.0,
            )

        return cls(
            name=name,
            emissivity=SpectralProperty.constant(emissivity),
            thermal=thermal,
        )


class MaterialLibrary:
    """Collection of materials with lookup by name."""

    def __init__(self) -> None:
        self._materials: dict[str, Material] = {}

    def add(self, material: Material) -> None:
        """Add a material to the library."""
        self._materials[material.name] = material

    def get(self, name: str) -> Material:
        """Get material by name."""
        if name not in self._materials:
            # Try to create from preset
            if name in MATERIAL_EMISSIVITY:
                mat = Material.from_preset(name)
                self._materials[name] = mat
                return mat
            raise KeyError(f"Material not found: {name}")
        return self._materials[name]

    def __contains__(self, name: str) -> bool:
        return name in self._materials or name in MATERIAL_EMISSIVITY

    def list_materials(self) -> list[str]:
        """List all available material names."""
        return list(set(list(self._materials.keys()) + list(MATERIAL_EMISSIVITY.keys())))


# Global material library instance
material_library = MaterialLibrary()


def create_graybody(emissivity: float, name: str = "graybody") -> Material:
    """Create a simple graybody material with constant emissivity."""
    return Material(
        name=name,
        emissivity=SpectralProperty.constant(emissivity),
    )


def create_blackbody(name: str = "blackbody") -> Material:
    """Create a perfect blackbody material."""
    return Material(
        name=name,
        emissivity=SpectralProperty.constant(1.0),
    )
