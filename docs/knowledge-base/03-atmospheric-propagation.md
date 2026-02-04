# Atmospheric Propagation

## 1. Introduction

This document covers atmospheric effects on electromagnetic radiation as it propagates from scene surfaces to the sensor. The atmosphere attenuates the signal (transmission loss) and adds unwanted signal (path radiance), fundamentally affecting what the sensor observes.

**Primary Backend**: RAF-tran (in-house, https://github.com/cohyos/RAF-tran)

---

## 2. Atmospheric Effects Overview

### 2.1 Key Phenomena

```
┌─────────────────────────────────────────────────────────────────┐
│                  ATMOSPHERIC EFFECTS ON RADIATION               │
└─────────────────────────────────────────────────────────────────┘

                         Sensor
                           ▲
                           │ L_sensor = L_surface × τ + L_path
                           │
    ┌──────────────────────┼──────────────────────┐
    │                ATMOSPHERE                    │
    │                                              │
    │  ┌─────────────┐    ┌─────────────┐         │
    │  │  ABSORPTION │    │ SCATTERING  │         │
    │  │             │    │             │         │
    │  │ H₂O, CO₂,   │    │ Rayleigh    │         │
    │  │ O₃, CH₄,... │    │ (molecules) │         │
    │  │             │    │             │         │
    │  │ Removes     │    │ Mie         │         │
    │  │ photons     │    │ (aerosols)  │         │
    │  └─────────────┘    └─────────────┘         │
    │         │                  │                 │
    │         └────────┬─────────┘                 │
    │                  ▼                           │
    │         Transmission τ(λ)                    │
    │         Path radiance L_path(λ)              │
    │                                              │
    └──────────────────────────────────────────────┘
                           │
                           │ L_surface
                           │
                       Surface
```

### 2.2 Radiative Transfer Equation

The fundamental equation for a path through the atmosphere:

```
L_sensor(λ) = L_surface(λ) × τ(λ) + L_path(λ)

Where:
  L_sensor  = Radiance reaching sensor [W/(m²·sr·μm)]
  L_surface = Radiance leaving target surface
  τ(λ)      = Atmospheric transmission (0-1)
  L_path    = Path radiance from atmosphere itself
```

**For thermal bands (MWIR/LWIR)**, expanded:
```
L_sensor = L_surface × τ + L_atm_emission × (1 - τ) + L_scattered

Where L_atm_emission is thermal emission from atmospheric gases
```

---

## 3. Molecular Absorption

### 3.1 Key Absorbing Species

| Species | Bands Affected | Notes |
|---------|---------------|-------|
| H₂O | VIS-LWIR, especially 2.7, 6.3 μm | Most variable, weather-dependent |
| CO₂ | 2.7, 4.3, 15 μm | Well-mixed, ~420 ppm |
| O₃ | UV, 9.6 μm | Stratospheric, UV shield |
| N₂O | 4.5, 7.8 μm | Trace gas |
| CH₄ | 3.3, 7.7 μm | Greenhouse gas |
| O₂ | 0.76 μm, 1.27 μm | Well-mixed |

### 3.2 Transmission Windows

The atmosphere has high transmission in specific "windows":

| Window | Range | Primary Use |
|--------|-------|-------------|
| Visible | 0.4-0.7 μm | Daylight imaging |
| NIR | 0.7-1.0 μm | Night vision |
| SWIR-1 | 1.0-1.35 μm | Low-light imaging |
| SWIR-2 | 1.5-1.8 μm | See through haze |
| SWIR-3 | 2.0-2.5 μm | Mineralogy |
| MWIR | 3.0-5.0 μm | Thermal, hot targets |
| LWIR | 8.0-14.0 μm | Thermal, ambient |

### 3.3 Beer-Lambert Law

For a uniform absorbing medium:

```
τ(λ) = exp(-∫ α(λ,s) ds) = exp(-τ_optical)

Where:
  α(λ,s) = Absorption coefficient at wavelength λ, position s
  τ_optical = Optical depth (dimensionless)
```

For discrete absorbers:
```
τ(λ) = exp(-Σ σᵢ(λ) × Nᵢ × L)

Where:
  σᵢ(λ) = Absorption cross-section of species i [cm²]
  Nᵢ    = Number density of species i [molecules/cm³]
  L     = Path length [cm]
```

---

## 4. Scattering

### 4.1 Rayleigh Scattering (Molecular)

Scattering by particles much smaller than wavelength (molecules):

```
σ_Rayleigh ∝ 1/λ⁴

Scattering coefficient:
  β_R(λ) = (8π³/3) × (n² - 1)² / (N × λ⁴) × F(δ)

Where:
  n = Refractive index of air
  N = Number density
  F(δ) = King factor (~1.05 for air)
```

**Effects**:
- Strong at short wavelengths (blue sky)
- Negligible in thermal IR
- Creates diffuse skylight

### 4.2 Mie Scattering (Aerosol)

Scattering by particles comparable to wavelength (aerosols, dust, haze):

```
Mie scattering depends on:
- Particle size distribution
- Refractive index (real and imaginary)
- Wavelength

Size parameter: x = 2πr/λ

Regimes:
  x << 1: Rayleigh regime (∝ λ⁻⁴)
  x ~ 1:  Mie regime (complex angular dependence)
  x >> 1: Geometric optics regime
```

### 4.3 Aerosol Models

Standard aerosol types (from MODTRAN):

| Model | Description | Visibility |
|-------|-------------|------------|
| Rural | Continental, low pollution | 23 km |
| Urban | Industrial pollution | 5-10 km |
| Maritime | Sea salt aerosols | 23 km |
| Desert | Mineral dust | Variable |
| Tropospheric | Background | >50 km |

Visibility relationship:
```
Visibility = 3.912 / β_ext(550nm)

Where β_ext is extinction coefficient at 550 nm
```

---

## 5. Standard Atmosphere Profiles

### 5.1 US Standard Atmosphere 1976

| Altitude [km] | Temperature [K] | Pressure [hPa] | Density [kg/m³] |
|---------------|-----------------|----------------|-----------------|
| 0 | 288.15 | 1013.25 | 1.225 |
| 5 | 255.65 | 540.48 | 0.736 |
| 10 | 223.15 | 264.99 | 0.413 |
| 15 | 216.65 | 121.11 | 0.195 |
| 20 | 216.65 | 55.29 | 0.089 |

### 5.2 MODTRAN Model Atmospheres

| Model | Description | Typical Use |
|-------|-------------|-------------|
| Tropical | Hot, humid equatorial | 15°N-15°S |
| Midlatitude Summer | Warm, moderate humidity | 30-60°N/S summer |
| Midlatitude Winter | Cold, dry | 30-60°N/S winter |
| Subarctic Summer | Cool, moderate | >60°N/S summer |
| Subarctic Winter | Very cold, dry | >60°N/S winter |
| US Standard 1976 | Average conditions | Default/reference |

---

## 6. Path Geometry

### 6.1 Slant Path vs Vertical

```
                    Sensor (altitude h_s)
                       ╲
                        ╲  Slant path
                         ╲ length = L
                          ╲
    Zenith angle θ ────────╲
                            ╲
    ─────────────────────────●──────────────── Ground (h_t)
                           Target

Slant path length:
  L = (h_s - h_t) / cos(θ)    [for flat Earth approximation]

Air mass factor:
  m = 1 / cos(θ)              [ratio to vertical path]

For large zenith angles (θ > 80°), use spherical geometry:
  m = [cos(θ) + 0.15 × (93.885 - θ)^(-1.253)]^(-1)   [Kasten-Young formula]
```

### 6.2 Path Radiance Sources

```
L_path = L_scattered + L_emitted

Scattered (VIS/NIR/SWIR):
  - Solar radiation scattered into line of sight
  - Stronger at shorter wavelengths
  - Reduces contrast

Emitted (MWIR/LWIR):
  - Thermal emission from atmospheric gases
  - Temperature-dependent
  - Significant in absorption bands
```

---

## 7. RAF-tran Integration

### 7.1 RAF-tran Capabilities Mapping

| EOSIM Need | RAF-tran Module | Notes |
|------------|-----------------|-------|
| Atmosphere profile | `atmosphere/` | US Std, MODTRAN variants |
| Gas absorption | `gas_optics/` | Correlated-k method |
| Molecular scattering | `scattering/rayleigh.py` | Wavelength-dependent |
| Aerosol scattering | `scattering/mie.py` | Size distribution support |
| RTE solution | `rte_solver/` | Two-stream, discrete ordinates |

### 7.2 EOSIM Wrapper Interface

```python
# src/eosim/atmosphere/raf_tran_backend.py

from dataclasses import dataclass
from typing import Optional
import numpy as np

# RAF-tran imports (via submodule)
from vendor.raf_tran.atmosphere import StandardAtmosphere
from vendor.raf_tran.gas_optics import CKDGasOptics
from vendor.raf_tran.scattering import RayleighScattering, MieScattering
from vendor.raf_tran.rte_solver import DiscreteOrdinatesSolver


@dataclass
class SlantPath:
    """Sensor-to-target atmospheric path geometry."""
    sensor_altitude_m: float
    target_altitude_m: float
    zenith_angle_deg: float
    azimuth_angle_deg: float

    @property
    def path_length_m(self) -> float:
        """Geometric path length."""
        dh = self.sensor_altitude_m - self.target_altitude_m
        return abs(dh) / np.cos(np.radians(self.zenith_angle_deg))

    @property
    def air_mass_factor(self) -> float:
        """Relative air mass compared to vertical path."""
        theta = np.radians(self.zenith_angle_deg)
        if self.zenith_angle_deg < 80:
            return 1.0 / np.cos(theta)
        else:
            # Kasten-Young formula for large angles
            return 1.0 / (np.cos(theta) + 0.15 * (93.885 - self.zenith_angle_deg)**(-1.253))


@dataclass
class AtmosphereResult:
    """Results from atmospheric calculation."""
    wavelengths_um: np.ndarray
    transmission: np.ndarray          # τ(λ), 0-1
    path_radiance: np.ndarray         # L_path(λ) [W/(m²·sr·μm)]
    downwelling_radiance: np.ndarray  # L_sky(λ) at target [W/(m²·sr·μm)]


class RAFTranAtmosphere:
    """
    EOSIM atmosphere module using RAF-tran backend.
    """

    def __init__(
        self,
        profile: str = "midlatitude_summer",
        aerosol_model: str = "rural",
        visibility_km: float = 23.0,
        h2o_column_cm: Optional[float] = None,
        solver: str = "discrete_ordinates",
        n_streams: int = 8
    ):
        """
        Initialize RAF-tran atmosphere model.

        Args:
            profile: Atmosphere profile name
            aerosol_model: Aerosol type (rural, urban, maritime, desert)
            visibility_km: Meteorological visibility
            h2o_column_cm: Water vapor column (None = use profile default)
            solver: RTE solver type (two_stream, discrete_ordinates)
            n_streams: Number of streams for discrete ordinates
        """
        self.atm_profile = StandardAtmosphere(profile)
        self.gas_optics = CKDGasOptics()
        self.rayleigh = RayleighScattering()
        self.mie = MieScattering(
            model=aerosol_model,
            visibility_km=visibility_km
        )

        if h2o_column_cm is not None:
            self.atm_profile.scale_h2o(h2o_column_cm)

        if solver == "two_stream":
            self.solver = TwoStreamSolver()
        else:
            self.solver = DiscreteOrdinatesSolver(n_streams=n_streams)

    def compute(
        self,
        wavelengths_um: np.ndarray,
        path: SlantPath,
        sun_zenith_deg: Optional[float] = None
    ) -> AtmosphereResult:
        """
        Compute atmospheric transmission and path radiance.

        Args:
            wavelengths_um: Wavelength array [μm]
            path: Slant path geometry
            sun_zenith_deg: Solar zenith angle (for scattered path radiance)

        Returns:
            AtmosphereResult with transmission and radiances
        """
        # Build layered atmosphere along path
        layers = self._discretize_path(path)

        # Compute optical depths
        tau_gas = self.gas_optics.optical_depth(wavelengths_um, layers)
        tau_rayleigh = self.rayleigh.optical_depth(wavelengths_um, layers)
        tau_mie = self.mie.optical_depth(wavelengths_um, layers)

        tau_total = tau_gas + tau_rayleigh + tau_mie

        # Direct beam transmission
        transmission = np.exp(-tau_total * path.air_mass_factor)

        # Solve RTE for path radiance
        path_radiance = self.solver.compute_path_radiance(
            wavelengths_um,
            tau_total,
            layers,
            path.zenith_angle_deg,
            sun_zenith_deg
        )

        # Downwelling sky radiance at target level
        downwelling = self.solver.compute_downwelling(
            wavelengths_um,
            tau_total,
            layers,
            path.target_altitude_m
        )

        return AtmosphereResult(
            wavelengths_um=wavelengths_um,
            transmission=transmission,
            path_radiance=path_radiance,
            downwelling_radiance=downwelling
        )

    def _discretize_path(self, path: SlantPath) -> list:
        """Discretize atmosphere into layers along path."""
        # Implementation: interpolate atmosphere profile along slant path
        pass
```

---

## 8. Lookup Table Approach

### 8.1 LUT Generation

For real-time performance, pre-compute atmosphere over parameter space:

```python
def generate_atmosphere_lut(
    atmosphere: RAFTranAtmosphere,
    wavelengths_um: np.ndarray,
    altitudes_m: np.ndarray,
    zenith_angles_deg: np.ndarray,
    output_file: str
) -> xr.Dataset:
    """
    Pre-compute atmospheric LUT.

    Returns xarray Dataset with dimensions:
      (wavelength, altitude, zenith)

    Variables:
      transmission, path_radiance, sky_radiance
    """
    import xarray as xr

    n_wl = len(wavelengths_um)
    n_alt = len(altitudes_m)
    n_zen = len(zenith_angles_deg)

    # Allocate arrays
    transmission = np.zeros((n_wl, n_alt, n_zen))
    path_radiance = np.zeros((n_wl, n_alt, n_zen))

    # Compute for each combination
    for i, alt in enumerate(altitudes_m):
        for j, zen in enumerate(zenith_angles_deg):
            path = SlantPath(
                sensor_altitude_m=alt,
                target_altitude_m=0,
                zenith_angle_deg=zen,
                azimuth_angle_deg=0
            )
            result = atmosphere.compute(wavelengths_um, path)

            transmission[:, i, j] = result.transmission
            path_radiance[:, i, j] = result.path_radiance

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "transmission": (["wavelength", "altitude", "zenith"], transmission),
            "path_radiance": (["wavelength", "altitude", "zenith"], path_radiance),
        },
        coords={
            "wavelength": wavelengths_um,
            "altitude": altitudes_m,
            "zenith": zenith_angles_deg,
        },
        attrs={
            "atmosphere_profile": atmosphere.atm_profile.name,
            "visibility_km": atmosphere.mie.visibility_km,
        }
    )

    ds.to_netcdf(output_file)
    return ds
```

### 8.2 LUT Interpolation

```python
class LUTAtmosphere:
    """Fast atmosphere using pre-computed LUT."""

    def __init__(self, lut_file: str):
        self.lut = xr.open_dataset(lut_file)

    def compute(
        self,
        wavelengths_um: np.ndarray,
        path: SlantPath
    ) -> AtmosphereResult:
        """Interpolate from LUT."""

        # Trilinear interpolation
        transmission = self.lut["transmission"].interp(
            wavelength=wavelengths_um,
            altitude=path.sensor_altitude_m,
            zenith=path.zenith_angle_deg,
            method="linear"
        ).values

        path_radiance = self.lut["path_radiance"].interp(
            wavelength=wavelengths_um,
            altitude=path.sensor_altitude_m,
            zenith=path.zenith_angle_deg,
            method="linear"
        ).values

        return AtmosphereResult(
            wavelengths_um=wavelengths_um,
            transmission=transmission,
            path_radiance=path_radiance,
            downwelling_radiance=np.zeros_like(transmission)  # Simplified
        )
```

---

## 9. Adjacency Effects

### 9.1 The Problem

Radiation from pixels adjacent to target scatters into the sensor's line of sight:

```
                     Sensor
                       │
                       │ Receives scattered
                       │ light from A and B
                       ▼
    ┌─────────────┬─────────────┬─────────────┐
    │   Pixel A   │   Target    │   Pixel B   │
    │   (bright)  │   (dark)    │   (bright)  │
    └─────────────┴─────────────┴─────────────┘

Effect: Dark target appears brighter due to
       scattered light from surroundings
```

### 9.2 Adjacency Modeling (Qiu Approach)

```
L_sensor(x,y) = L_direct(x,y) × τ_direct + ∫∫ L_surface(x',y') × K(x-x', y-y') dx' dy'

Where:
  K(Δx, Δy) = Adjacency kernel (PSF-like, but atmospheric)

The kernel depends on:
- Scattering optical depth
- Surface-sensor geometry
- Wavelength
```

### 9.3 Implementation

```python
def apply_adjacency_effect(
    radiance_map: np.ndarray,
    adjacency_kernel: np.ndarray,
    direct_transmission: float
) -> np.ndarray:
    """
    Apply adjacency effect via convolution.

    Args:
        radiance_map: Surface leaving radiance [H, W]
        adjacency_kernel: Pre-computed kernel [K, K]
        direct_transmission: τ_direct

    Returns:
        At-sensor radiance with adjacency effect
    """
    from scipy.ndimage import convolve

    # Direct component
    L_direct = radiance_map * direct_transmission

    # Scattered component (convolution)
    L_scattered = convolve(radiance_map, adjacency_kernel, mode='reflect')

    return L_direct + L_scattered
```

---

## 10. Turbulence and Scintillation

### 10.1 Atmospheric Turbulence

For long slant paths, turbulence causes:
- **Scintillation**: Intensity fluctuations
- **Beam spreading**: Effective blur
- **Image dancing**: Angular wander

Characterized by refractive index structure parameter C_n²:

```
Typical values:
  Strong turbulence: C_n² ~ 10⁻¹³ m^(-2/3)
  Moderate:          C_n² ~ 10⁻¹⁴ m^(-2/3)
  Weak:              C_n² ~ 10⁻¹⁵ m^(-2/3)
```

### 10.2 Fried Parameter

Atmospheric coherence length:

```
r₀ = [0.423 × k² × sec(θ) × ∫ C_n²(h) dh]^(-3/5)

Where:
  k = 2π/λ
  θ = Zenith angle

For r₀ < D_aperture: Resolution is seeing-limited
```

### 10.3 Simple Turbulence Model

```python
def apply_turbulence_blur(
    image: np.ndarray,
    cn2_path_integral: float,
    wavelength_um: float,
    pixel_ifov_rad: float
) -> np.ndarray:
    """
    Apply turbulence-induced blur.

    Simple model: Gaussian blur with seeing-limited width.
    """
    # Fried parameter
    k = 2 * np.pi / (wavelength_um * 1e-6)
    r0 = (0.423 * k**2 * cn2_path_integral)**(-3./5.)

    # Seeing angle
    theta_seeing = 0.98 * wavelength_um * 1e-6 / r0  # radians

    # Convert to pixels
    sigma_pixels = theta_seeing / pixel_ifov_rad / 2.355  # FWHM to sigma

    # Apply Gaussian blur
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(image, sigma=sigma_pixels)
```

---

## 11. Multi-Fidelity Approaches

### 11.1 Fidelity Levels

| Level | Approach | Speed | Accuracy | Use Case |
|-------|----------|-------|----------|----------|
| Simple | Beer-Lambert, constant extinction | Very fast | Low | Quick previews |
| LUT | Pre-computed RAF-tran tables | Fast | Medium | Real-time simulation |
| Full RT | RAF-tran per-pixel | Slow | High | Validation, final render |
| 3D MC | Monte Carlo (MCScene-like) | Very slow | Highest | Complex terrain, adjacency |

### 11.2 Selection Logic

```python
def create_atmosphere(
    fidelity: str,
    profile: str = "midlatitude_summer",
    **kwargs
) -> AtmosphereModel:
    """
    Factory function for atmosphere models.
    """
    if fidelity == "simple":
        return SimpleAtmosphere(visibility_km=kwargs.get("visibility_km", 23))

    elif fidelity == "lut":
        lut_file = kwargs.get("lut_file", "data/atmosphere/default.nc")
        return LUTAtmosphere(lut_file)

    elif fidelity == "full":
        return RAFTranAtmosphere(
            profile=profile,
            solver="discrete_ordinates",
            **kwargs
        )

    else:
        raise ValueError(f"Unknown fidelity: {fidelity}")
```

---

## 12. Summary

### Key Equations

1. **Radiative Transfer**:
   ```
   L_sensor = L_surface × τ + L_path
   ```

2. **Transmission** (Beer-Lambert):
   ```
   τ = exp(-τ_optical × air_mass)
   ```

3. **Slant Path**:
   ```
   L_path = (h_s - h_t) / cos(θ)
   air_mass = 1 / cos(θ)
   ```

### RAF-tran Integration

- Use RAF-tran as primary atmospheric backend
- Wrapper provides EOSIM-specific slant path geometry
- LUT generation for real-time performance
- Full RT mode for validation

### Implementation Priority

1. Simple Beer-Lambert for initial testing
2. RAF-tran wrapper with slant path support
3. LUT generation and interpolation
4. Adjacency effects for high-fidelity mode
5. Turbulence modeling for long-range scenarios

---

## 13. References

1. **MODTRAN Documentation** - Atmospheric radiative transfer
2. **RAF-tran** - https://github.com/cohyos/RAF-tran
3. **Bodhaine et al.** (1999) - Rayleigh scattering calculations
4. **Bohren & Huffman** (1983) - Absorption and Scattering of Light
5. **Qiu et al.** - Terrain-atmosphere coupling and adjacency
6. **Andrews & Phillips** (2005) - Laser Beam Propagation through Turbulence
