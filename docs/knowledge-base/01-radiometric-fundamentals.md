# Radiometric Fundamentals for EO/IR Simulation

## 1. Introduction

This document establishes the radiometric foundation for EOSIM. All image synthesis in electro-optical simulation ultimately derives from the physics of electromagnetic radiation—how it is emitted, reflected, transmitted, and detected.

---

## 2. Electromagnetic Spectrum for EO/IR

### 2.1 Wavelength Bands

| Band | Wavelength Range | Primary Source | Typical Applications |
|------|------------------|----------------|---------------------|
| **Visible (VIS)** | 0.4 – 0.7 μm | Reflected solar | Daylight imaging, color cameras |
| **Near-IR (NIR)** | 0.7 – 1.0 μm | Reflected solar | Night vision (with illumination), vegetation |
| **Short-Wave IR (SWIR)** | 1.0 – 2.5 μm | Reflected solar + thermal | Low-light, see through haze |
| **Mid-Wave IR (MWIR)** | 3.0 – 5.0 μm | Thermal emission | Hot targets, aircraft, missiles |
| **Long-Wave IR (LWIR)** | 8.0 – 14.0 μm | Thermal emission | Ambient temp objects, uncooled sensors |

### 2.2 Atmospheric Windows

The atmosphere is largely opaque to IR radiation except in specific transmission windows:

```
Transmission
    1.0 ┤                    ████████             ████████████████
        │                    ██    ██             ██            ██
    0.5 ┤████████████████████      ██             ██            ██
        │                          ██████████████████
    0.0 ┼────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────
         0.4  1    2    3    4    5    6    7    8    9   10   12   14  λ(μm)
              VIS  NIR  SWIR      MWIR           LWIR
                              ↑         ↑              ↑
                           H₂O/CO₂    CO₂           H₂O
                           absorption
```

---

## 3. Blackbody Radiation

### 3.1 Planck's Law

The spectral radiance of a perfect blackbody at temperature T:

```
                    2hc²           1
L_λ(T) = ────────── × ─────────────────────
                λ⁵     exp(hc/λkT) - 1

Where:
  L_λ  = Spectral radiance [W·m⁻²·sr⁻¹·μm⁻¹]
  h    = Planck's constant = 6.626 × 10⁻³⁴ J·s
  c    = Speed of light = 2.998 × 10⁸ m/s
  k    = Boltzmann constant = 1.381 × 10⁻²³ J/K
  λ    = Wavelength [m]
  T    = Absolute temperature [K]
```

### 3.2 Simplified Form (First and Second Radiation Constants)

```
              C₁
L_λ(T) = ─────────────────────
         λ⁵[exp(C₂/λT) - 1]

Where:
  C₁ = 2hc² = 1.191 × 10⁸ W·μm⁴·m⁻²·sr⁻¹
  C₂ = hc/k = 1.439 × 10⁴ μm·K
```

### 3.3 Wien's Displacement Law

The wavelength of peak emission:

```
λ_max = b/T

Where:
  b = Wien's displacement constant = 2897.8 μm·K

Examples:
  Sun (5778 K):     λ_max ≈ 0.50 μm (visible green)
  Human (310 K):    λ_max ≈ 9.35 μm (LWIR)
  Hot engine (400 K): λ_max ≈ 7.24 μm (LWIR)
  Jet exhaust (800 K): λ_max ≈ 3.62 μm (MWIR)
```

### 3.4 Stefan-Boltzmann Law

Total radiant exitance (integrated over all wavelengths):

```
M = σT⁴

Where:
  M = Total radiant exitance [W/m²]
  σ = Stefan-Boltzmann constant = 5.670 × 10⁻⁸ W·m⁻²·K⁻⁴
```

### 3.5 Implementation

```python
import numpy as np

# Physical constants
C1 = 1.191042e8   # W·μm⁴·m⁻²·sr⁻¹
C2 = 1.4387752e4  # μm·K

def planck_radiance(wavelength_um: np.ndarray, temperature_K: float) -> np.ndarray:
    """
    Compute spectral radiance using Planck's Law.

    Args:
        wavelength_um: Wavelength(s) in micrometers
        temperature_K: Temperature in Kelvin

    Returns:
        Spectral radiance in W/(m²·sr·μm)
    """
    λ = wavelength_um
    T = temperature_K

    # Avoid overflow in exponential
    x = C2 / (λ * T)

    # Use different formulations depending on x to maintain precision
    radiance = np.where(
        x < 100,
        C1 / (λ**5 * (np.exp(x) - 1)),
        C1 / (λ**5) * np.exp(-x)  # Wien approximation for large x
    )

    return radiance


def band_integrated_radiance(
    wavelength_min_um: float,
    wavelength_max_um: float,
    temperature_K: float,
    n_samples: int = 100
) -> float:
    """
    Compute band-integrated radiance.

    Returns:
        In-band radiance in W/(m²·sr)
    """
    wavelengths = np.linspace(wavelength_min_um, wavelength_max_um, n_samples)
    spectral_radiance = planck_radiance(wavelengths, temperature_K)

    # Trapezoidal integration
    return np.trapz(spectral_radiance, wavelengths)
```

---

## 4. Real Surface Radiation

### 4.1 Emissivity

Real surfaces are not perfect blackbodies. Emissivity ε(λ) characterizes how efficiently a surface emits compared to a blackbody:

```
L_emitted(λ, T) = ε(λ) × L_blackbody(λ, T)

Where:
  ε(λ) = Spectral emissivity, 0 ≤ ε ≤ 1
```

### 4.2 Kirchhoff's Law

For opaque surfaces in thermal equilibrium:

```
ε(λ) + ρ(λ) = 1

Where:
  ε(λ) = Emissivity (absorbed/emitted fraction)
  ρ(λ) = Reflectivity (reflected fraction)
```

For partially transparent materials (e.g., thin films, gases):

```
ε(λ) + ρ(λ) + τ(λ) = 1

Where:
  τ(λ) = Transmissivity
```

### 4.3 Typical Emissivity Values

| Material | MWIR (3-5 μm) | LWIR (8-14 μm) |
|----------|---------------|----------------|
| Water | 0.96 | 0.98 |
| Vegetation | 0.94 | 0.96 |
| Concrete | 0.90 | 0.92 |
| Asphalt | 0.92 | 0.93 |
| Bare soil | 0.90 | 0.92 |
| Painted metal | 0.80-0.90 | 0.85-0.95 |
| Polished metal | 0.05-0.20 | 0.02-0.10 |
| Human skin | 0.97 | 0.98 |
| Glass | 0.85 | 0.90 |

---

## 5. Surface Radiance Equation

### 5.1 Total Leaving Radiance

The radiance leaving a surface toward the sensor combines emission and reflection:

```
L_surface(λ) = ε(λ) × L_BB(λ, T_surface) + ρ(λ) × L_incident(λ)

Where:
  L_surface   = Total radiance leaving surface toward sensor
  L_BB        = Blackbody radiance at surface temperature
  L_incident  = Incident radiance from environment (sun, sky, surroundings)
```

### 5.2 Incident Radiance Components

```
L_incident = L_solar + L_sky_downwelling + L_terrain_reflected

Where:
  L_solar           = Direct solar irradiance × cos(θ_sun) / π  [for diffuse]
  L_sky_downwelling = Atmospheric thermal emission downward
  L_terrain_reflected = Radiance from surrounding terrain
```

### 5.3 Day vs Night Scenarios

**Daytime (VIS/NIR/SWIR)**:
- Dominated by reflected solar radiation
- L_surface ≈ ρ(λ) × E_sun × cos(θ_sun) / π

**Nighttime or Thermal (MWIR/LWIR)**:
- Dominated by thermal emission
- L_surface ≈ ε(λ) × L_BB(λ, T_surface)

**Transition (dawn/dusk, SWIR/MWIR crossover)**:
- Both terms significant
- Full equation required

---

## 6. Radiometric Quantities and Units

### 6.1 Fundamental Quantities

| Quantity | Symbol | Units | Description |
|----------|--------|-------|-------------|
| Radiant energy | Q | J | Total electromagnetic energy |
| Radiant flux (power) | Φ | W | Energy per unit time |
| Radiant intensity | I | W/sr | Power per solid angle |
| Irradiance | E | W/m² | Power per unit area (incident) |
| Radiant exitance | M | W/m² | Power per unit area (emitted) |
| Radiance | L | W/(m²·sr) | Power per area per solid angle |
| Spectral radiance | L_λ | W/(m²·sr·μm) | Radiance per wavelength |

### 6.2 Relationship Diagram

```
                    ┌──────────────┐
                    │ Radiant Flux │
                    │    Φ [W]     │
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  Intensity  │ │ Irradiance  │ │  Exitance   │
    │  I [W/sr]   │ │  E [W/m²]   │ │  M [W/m²]   │
    └─────────────┘ └──────┬──────┘ └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Radiance   │
                    │ L [W/m²/sr] │
                    └─────────────┘
```

### 6.3 Conversion Relationships

```
Irradiance from point source:
  E = I / d²

Radiance from Lambertian surface:
  L = M / π = E × ρ / π

Irradiance from extended source:
  E = ∫ L × cos(θ) dΩ
```

---

## 7. At-Sensor Radiance

### 7.1 Radiative Transfer Equation

The radiance reaching the sensor after atmospheric propagation:

```
L_sensor(λ) = L_surface(λ) × τ_atm(λ) + L_path(λ)

Where:
  L_sensor  = Radiance at sensor aperture
  L_surface = Radiance leaving target surface
  τ_atm     = Atmospheric transmission along path
  L_path    = Path radiance (atmospheric scattering into LOS)
```

### 7.2 Expanded Form

```
L_sensor = [ε × L_BB(T) + ρ × L_incident] × τ_atm + L_path

         = ε × L_BB(T) × τ_atm           ← Target thermal emission
         + ρ × L_incident × τ_atm        ← Reflected environment
         + L_path                        ← Atmospheric path radiance
```

### 7.3 Signal Chain to Sensor

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AT-SENSOR RADIANCE COMPUTATION                   │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Surface    │    │  Atmosphere  │    │   Sensor     │    │   Detector   │
│              │    │              │    │   Optics     │    │              │
│ T_surface    │───▶│ τ(λ), L_path │───▶│ τ_optics     │───▶│ Responsivity │
│ ε(λ), ρ(λ)  │    │              │    │ PSF          │    │ QE(λ)        │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
   L_surface          L_at_sensor         L_focal_plane       Signal [e⁻]
```

---

## 8. Photon Flux and Detector Signal

### 8.1 Photon Energy

```
E_photon = hc/λ = hν

Where:
  E_photon = Energy of single photon [J]
  h = Planck's constant
  c = Speed of light
  λ = Wavelength
  ν = Frequency
```

### 8.2 Photon Flux from Radiance

```
Φ_photon = L × A_pixel × Ω × τ_optics × Δλ / E_photon
         = L × A_pixel × Ω × τ_optics × Δλ × λ / (hc)

Where:
  Φ_photon  = Photon flux [photons/s]
  L         = Spectral radiance [W/(m²·sr·μm)]
  A_pixel   = Pixel area [m²]
  Ω         = Solid angle subtended by optics [sr]
  τ_optics  = Optical transmission
  Δλ        = Bandwidth [μm]
```

### 8.3 Solid Angle of Optics

```
Ω = π × (D/2)² / f² = π / (4 × F#²)

Where:
  D  = Aperture diameter
  f  = Focal length
  F# = f-number = f/D
```

### 8.4 Electrons Generated

```
N_electrons = Φ_photon × QE(λ) × t_int

Where:
  QE(λ)  = Quantum efficiency at wavelength λ
  t_int  = Integration time [s]
```

---

## 9. Contrast and Detection

### 9.1 Thermal Contrast

```
ΔL = L_target - L_background

Contrast ratio:
  C = L_target / L_background

Contrast temperature (for thermal):
  ΔT = T_target - T_background
```

### 9.2 NEΔT (Noise Equivalent Temperature Difference)

The minimum detectable temperature difference:

```
NEΔT = noise_electrons / (∂N/∂T)

Where:
  ∂N/∂T = Rate of change of signal electrons with temperature
```

### 9.3 Apparent Temperature

The temperature a blackbody would need to produce the same radiance:

```
T_apparent such that: L_BB(T_apparent) = L_measured / ε

For graybody:
  T_apparent ≠ T_actual (unless ε = 1)
```

---

## 10. Band Integration

### 10.1 Effective In-Band Radiance

```
L_band = ∫[λ₁ to λ₂] L(λ) × R(λ) dλ / ∫[λ₁ to λ₂] R(λ) dλ

Where:
  R(λ) = Relative spectral response of sensor
```

### 10.2 Implementation

```python
def compute_inband_radiance(
    wavelengths_um: np.ndarray,
    spectral_radiance: np.ndarray,
    spectral_response: np.ndarray
) -> float:
    """
    Compute effective in-band radiance weighted by sensor response.

    Args:
        wavelengths_um: Wavelength array [μm]
        spectral_radiance: L(λ) [W/(m²·sr·μm)]
        spectral_response: R(λ) [dimensionless, 0-1]

    Returns:
        Effective in-band radiance [W/(m²·sr)]
    """
    numerator = np.trapz(spectral_radiance * spectral_response, wavelengths_um)
    denominator = np.trapz(spectral_response, wavelengths_um)

    return numerator / denominator
```

---

## 11. Summary: Key Equations for EOSIM

### 11.1 Core Radiometric Chain

1. **Source Radiance** (Planck + emissivity):
   ```
   L_source(λ) = ε(λ) × C₁ / [λ⁵(exp(C₂/λT) - 1)]
   ```

2. **Surface Leaving Radiance** (emission + reflection):
   ```
   L_surface(λ) = ε(λ) × L_BB(T) + ρ(λ) × L_incident(λ)
   ```

3. **At-Sensor Radiance** (atmospheric propagation):
   ```
   L_sensor(λ) = L_surface(λ) × τ_atm(λ) + L_path(λ)
   ```

4. **Focal Plane Irradiance** (optics):
   ```
   E_fp(λ) = π × L_sensor(λ) × τ_optics / (4 × F#²)
   ```

5. **Detector Signal** (photon conversion):
   ```
   N_e = ∫ E_fp(λ) × A_pixel × QE(λ) × λ/(hc) × t_int dλ
   ```

### 11.2 Constants Reference

| Constant | Symbol | Value | Units |
|----------|--------|-------|-------|
| Speed of light | c | 2.998 × 10⁸ | m/s |
| Planck constant | h | 6.626 × 10⁻³⁴ | J·s |
| Boltzmann constant | k | 1.381 × 10⁻²³ | J/K |
| Stefan-Boltzmann | σ | 5.670 × 10⁻⁸ | W/(m²·K⁴) |
| Wien displacement | b | 2897.8 | μm·K |
| First radiation const | C₁ | 1.191 × 10⁸ | W·μm⁴/(m²·sr) |
| Second radiation const | C₂ | 1.439 × 10⁴ | μm·K |

---

## 12. References

1. **Planck, M.** (1901). "On the Law of Distribution of Energy in the Normal Spectrum"
2. **Schott, J.R.** (2007). "Remote Sensing: The Image Chain Approach" - Chapter 4: Radiometry
3. **DIRSIG Documentation** - Radiometric fundamentals
4. **Dereniak & Boreman** (1996). "Infrared Detectors and Systems" - Chapters 1-3
5. **Hudson, R.D.** (1969). "Infrared System Engineering" - Classic reference
