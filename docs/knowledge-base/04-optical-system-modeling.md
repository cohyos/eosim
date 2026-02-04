# Optical System Modeling

## 1. Introduction

This document covers the modeling of optical systems in EOSIM, including lens parameters, point spread function (PSF), modulation transfer function (MTF), and optical aberrations. The optics module transforms the at-sensor radiance field into the irradiance distribution on the focal plane.

---

## 2. Optical System Overview

### 2.1 Image Formation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OPTICAL SYSTEM CHAIN                                │
└─────────────────────────────────────────────────────────────────────────────┘

   Scene                 Entrance       Optical          Focal Plane
  Radiance               Pupil         Elements           Irradiance
     │                     │              │                   │
     │                     │              │                   │
     ▼                     ▼              ▼                   ▼
┌─────────┐          ┌─────────┐    ┌─────────┐         ┌─────────┐
│ L(θ,φ)  │─────────▶│ Aperture│───▶│ Lenses  │────────▶│ E(x,y)  │
│ [W/m²sr]│          │ D, f/#  │    │ PSF,MTF │         │ [W/m²]  │
└─────────┘          └─────────┘    └─────────┘         └─────────┘
                           │              │
                           │              │
                      Collecting     Spreading &
                        Area        Aberrations
```

### 2.2 Key Parameters

| Parameter | Symbol | Units | Description |
|-----------|--------|-------|-------------|
| Focal length | f | mm | Distance from lens to focal plane |
| Aperture diameter | D | mm | Clear aperture size |
| f-number | F/# = f/D | - | Ratio of focal length to aperture |
| Field of view | FOV | degrees | Angular extent of scene imaged |
| Instantaneous FOV | IFOV | mrad | Angular extent of single pixel |
| Optical transmission | τ_opt | - | Fraction of light transmitted |

---

## 3. Fundamental Relationships

### 3.1 Image Scale and Field of View

```
Pixel IFOV:
  IFOV = d_pixel / f

Where:
  d_pixel = Pixel pitch [m]
  f = Focal length [m]
  IFOV in radians

Full FOV:
  FOV = 2 × arctan(W_sensor / (2 × f))

Where:
  W_sensor = Sensor width (N_pixels × d_pixel)
```

### 3.2 Ground Sample Distance (GSD)

```
GSD = IFOV × R = (d_pixel / f) × R

Where:
  R = Slant range to target [m]
  GSD = Ground resolution [m/pixel]

Example:
  d_pixel = 15 μm, f = 100 mm, R = 1000 m
  GSD = (15×10⁻⁶ / 0.1) × 1000 = 0.15 m = 15 cm
```

### 3.3 Focal Plane Irradiance

From a Lambertian source with radiance L:

```
E_fp = (π × L × τ_opt) / (4 × F/#²)

For extended source filling the pixel:
  Power on pixel = E_fp × A_pixel

Where:
  E_fp = Focal plane irradiance [W/m²]
  τ_opt = Optical transmission
  F/# = f-number
  A_pixel = Pixel area [m²]
```

### 3.4 Solid Angle Relationships

```
Pixel solid angle (in object space):
  Ω_pixel = IFOV² = (d_pixel / f)²  [sr, small angle]

Aperture solid angle (from focal plane):
  Ω_aperture = π × (D/2)² / f² = π / (4 × F/#²)  [sr]
```

---

## 4. Point Spread Function (PSF)

### 4.1 Definition

The PSF describes how a point source is imaged:

```
E_fp(x,y) = L_point × PSF(x,y)

Properties:
- ∫∫ PSF(x,y) dx dy = 1 (normalized)
- PSF is the impulse response of the optical system
- Real images = Ideal image ⊗ PSF (convolution)
```

### 4.2 Diffraction-Limited PSF (Airy Pattern)

For a circular aperture with no aberrations:

```
PSF_Airy(r) = [2 × J₁(πr/λF/#) / (πr/λF/#)]²

Where:
  J₁ = Bessel function of first kind, order 1
  r = Radial distance from center [m]
  λ = Wavelength [m]
  F/# = f-number

Airy disk radius (first zero):
  r_Airy = 1.22 × λ × F/#

FWHM of Airy disk:
  FWHM ≈ 1.03 × λ × F/#
```

### 4.3 Gaussian PSF Approximation

For many practical purposes:

```
PSF_Gaussian(r) = (1 / (2πσ²)) × exp(-r² / (2σ²))

Where σ is related to FWHM:
  FWHM = 2.355 × σ

Diffraction-limited σ:
  σ_diff ≈ 0.44 × λ × F/#
```

### 4.4 Implementation

```python
import numpy as np
from scipy.special import j1

def airy_psf(
    size_pixels: int,
    pixel_pitch_m: float,
    wavelength_m: float,
    f_number: float
) -> np.ndarray:
    """
    Generate 2D Airy disk PSF.

    Args:
        size_pixels: Output array size (odd recommended)
        pixel_pitch_m: Pixel pitch in meters
        wavelength_m: Wavelength in meters
        f_number: f-number of optical system

    Returns:
        Normalized 2D PSF array
    """
    center = size_pixels // 2
    y, x = np.ogrid[-center:size_pixels-center, -center:size_pixels-center]
    r = np.sqrt(x**2 + y**2) * pixel_pitch_m

    # Airy pattern argument
    arg = np.pi * r / (wavelength_m * f_number)

    # Handle r=0 case (limit is 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        psf = np.where(arg == 0, 1.0, (2 * j1(arg) / arg)**2)

    # Normalize
    psf /= psf.sum()

    return psf


def gaussian_psf(
    size_pixels: int,
    sigma_pixels: float
) -> np.ndarray:
    """
    Generate 2D Gaussian PSF.

    Args:
        size_pixels: Output array size
        sigma_pixels: Standard deviation in pixels

    Returns:
        Normalized 2D PSF array
    """
    center = size_pixels // 2
    y, x = np.ogrid[-center:size_pixels-center, -center:size_pixels-center]
    r2 = x**2 + y**2

    psf = np.exp(-r2 / (2 * sigma_pixels**2))
    psf /= psf.sum()

    return psf
```

---

## 5. Modulation Transfer Function (MTF)

### 5.1 Definition

The MTF is the magnitude of the optical transfer function (OTF):

```
OTF(f_x, f_y) = F{PSF(x, y)}   (Fourier transform)
MTF(f_x, f_y) = |OTF(f_x, f_y)|

Where:
  f_x, f_y = Spatial frequencies [cycles/m or cycles/pixel]
```

### 5.2 System MTF

The total system MTF is the product of component MTFs:

```
MTF_system = MTF_optics × MTF_detector × MTF_motion × MTF_atmosphere × MTF_electronics

Each component degrades image sharpness independently.
```

### 5.3 Component MTFs

**Diffraction-limited optics** (circular aperture):
```
MTF_diff(f) = (2/π) × [arccos(f/f_c) - (f/f_c)×√(1-(f/f_c)²)]  for f ≤ f_c
            = 0                                                   for f > f_c

Cutoff frequency:
  f_c = D / (λ × f) = 1 / (λ × F/#)  [cycles/m at focal plane]
```

**Detector sampling** (pixel footprint):
```
MTF_detector(f) = sinc(π × f × d_pixel)

Where:
  sinc(x) = sin(x)/x
  d_pixel = Pixel pitch
```

**Motion blur**:
```
MTF_motion(f) = sinc(π × f × Δx)

Where:
  Δx = Image motion during integration [m]
```

### 5.4 Nyquist Frequency

```
f_Nyquist = 1 / (2 × d_pixel)  [cycles/m]

Or in angular terms:
  f_Nyquist = 1 / (2 × IFOV)  [cycles/rad]

Aliasing occurs when scene has frequencies > f_Nyquist
```

### 5.5 MTF Computation

```python
def compute_mtf_from_psf(psf: np.ndarray) -> np.ndarray:
    """
    Compute MTF from PSF via FFT.

    Args:
        psf: 2D point spread function (normalized)

    Returns:
        2D MTF array (shifted so DC is at center)
    """
    # Fourier transform of PSF
    otf = np.fft.fft2(psf)

    # MTF is magnitude of OTF
    mtf = np.abs(otf)

    # Shift so DC is at center
    mtf = np.fft.fftshift(mtf)

    # Normalize to 1 at DC
    mtf /= mtf.max()

    return mtf


def diffraction_mtf(
    frequencies: np.ndarray,
    cutoff_frequency: float
) -> np.ndarray:
    """
    Compute diffraction-limited MTF.

    Args:
        frequencies: Spatial frequency array [cycles/unit]
        cutoff_frequency: Diffraction cutoff [same units]

    Returns:
        MTF values
    """
    f_norm = frequencies / cutoff_frequency

    mtf = np.zeros_like(f_norm)
    valid = f_norm <= 1

    mtf[valid] = (2/np.pi) * (
        np.arccos(f_norm[valid]) -
        f_norm[valid] * np.sqrt(1 - f_norm[valid]**2)
    )

    return mtf
```

---

## 6. Optical Aberrations

### 6.1 Wavefront Aberrations

Aberrations are deviations from ideal spherical wavefront:

```
W(ρ, θ) = Σ Zₙ × Pₙ(ρ, θ)

Where:
  W = Wavefront error
  Zₙ = Zernike coefficients
  Pₙ = Zernike polynomials
  ρ = Normalized pupil radius (0-1)
  θ = Pupil angle
```

### 6.2 Common Aberrations

| Aberration | Zernike | Effect on PSF |
|------------|---------|---------------|
| Piston | Z₀ | No effect (constant phase) |
| Tilt | Z₁, Z₂ | Image shift |
| Defocus | Z₃ | Symmetric blur |
| Astigmatism | Z₄, Z₅ | Directional blur |
| Coma | Z₆, Z₇ | Comet-like tail |
| Spherical | Z₈ | Symmetric halo |
| Trefoil | Z₉, Z₁₀ | Three-fold pattern |

### 6.3 Strehl Ratio

Metric of optical quality:

```
Strehl = Peak(PSF_actual) / Peak(PSF_diffraction_limited)

For small aberrations:
  Strehl ≈ exp(-(2π × σ_W / λ)²)

Where:
  σ_W = RMS wavefront error

"Diffraction-limited" criterion:
  Strehl ≥ 0.8 (Maréchal criterion)
  Corresponds to σ_W ≤ λ/14
```

### 6.4 Aberrated PSF Computation

```python
def compute_aberrated_psf(
    size_pixels: int,
    pixel_pitch_m: float,
    wavelength_m: float,
    f_number: float,
    zernike_coeffs: dict
) -> np.ndarray:
    """
    Compute PSF with wavefront aberrations.

    Args:
        size_pixels: Output size
        pixel_pitch_m: Pixel pitch
        wavelength_m: Wavelength
        f_number: f-number
        zernike_coeffs: Dict of Zernike coefficients {index: value_waves}

    Returns:
        Aberrated PSF
    """
    # Create pupil grid
    pupil_size = size_pixels
    center = pupil_size // 2
    y, x = np.ogrid[-center:pupil_size-center, -center:pupil_size-center]
    rho = np.sqrt(x**2 + y**2) / center  # Normalized radius
    theta = np.arctan2(y, x)

    # Pupil mask (circular aperture)
    pupil_mask = rho <= 1.0

    # Compute wavefront from Zernike polynomials
    wavefront = np.zeros((pupil_size, pupil_size))
    for idx, coeff in zernike_coeffs.items():
        wavefront += coeff * zernike_polynomial(idx, rho, theta)

    # Complex pupil function
    phase = 2 * np.pi * wavefront  # waves to radians
    pupil = pupil_mask * np.exp(1j * phase)

    # PSF = |FFT(pupil)|²
    psf = np.abs(np.fft.fftshift(np.fft.fft2(pupil)))**2

    # Normalize
    psf /= psf.sum()

    return psf
```

---

## 7. Field-Dependent Effects

### 7.1 Vignetting

Reduction in illumination toward field edges:

```
Vignetting factor:
  V(θ) = cos⁴(θ)  (natural vignetting)

Where θ = Field angle

Additional mechanical vignetting may occur from lens hoods, baffles.
```

### 7.2 Distortion

Geometric mapping errors:

**Radial distortion** (Brown-Conrady model):
```
r_distorted = r × (1 + k₁r² + k₂r⁴ + k₃r⁶ + ...)

Where:
  r = Radial distance from optical axis
  k₁, k₂, k₃ = Radial distortion coefficients

k₁ > 0: Pincushion distortion
k₁ < 0: Barrel distortion
```

**Tangential distortion**:
```
Δx = 2p₁xy + p₂(r² + 2x²)
Δy = p₁(r² + 2y²) + 2p₂xy

Where p₁, p₂ = Tangential coefficients
```

### 7.3 Field-Dependent PSF

PSF varies across the field:
- Diffraction remains constant
- Aberrations increase toward edges
- Particularly coma and astigmatism

```python
def get_psf_at_field_position(
    field_angle_deg: float,
    optics: OpticsModel
) -> np.ndarray:
    """
    Get PSF at specified field position.

    For sophisticated models, PSF varies with field angle.
    Simple approach: Interpolate from sampled field positions.
    """
    # Find bracketing PSF samples
    # Interpolate between them
    pass
```

---

## 8. Spectral Effects

### 8.1 Chromatic Aberration

Refractive optics have wavelength-dependent focus:

**Longitudinal chromatic aberration**:
```
Δf = f × (n_blue - n_red) / (n_d - 1) / V

Where:
  V = Abbe number (dispersion measure)
  n = Refractive index at different wavelengths
```

**Lateral chromatic aberration**:
Color-dependent magnification

### 8.2 Spectral Transmission

Optical transmission varies with wavelength:

```python
def optical_transmission(
    wavelength_um: np.ndarray,
    n_elements: int = 6,
    coating: str = "ar"
) -> np.ndarray:
    """
    Estimate optical transmission vs wavelength.

    Args:
        wavelength_um: Wavelength array
        n_elements: Number of optical elements
        coating: Coating type (ar, standard)

    Returns:
        Transmission array (0-1)
    """
    # Per-surface transmission
    if coating == "ar":
        T_surface = 0.995  # AR coated
    else:
        T_surface = 0.96   # Uncoated glass-air

    # Total transmission (2 surfaces per element)
    T_total = T_surface ** (2 * n_elements)

    # Spectral variation (simplified)
    # Real systems need measured data
    return np.full_like(wavelength_um, T_total)
```

---

## 9. PSF Convolution

### 9.1 Image Formation

```
Image = IdealImage ⊗ PSF

E_focal_plane(x,y) = ∫∫ L(x',y') × PSF(x-x', y-y') dx' dy'
```

### 9.2 Efficient Convolution

```python
def apply_psf(
    image: np.ndarray,
    psf: np.ndarray,
    method: str = "fft"
) -> np.ndarray:
    """
    Apply PSF blur to image.

    Args:
        image: Input image [H, W]
        psf: Point spread function (must sum to 1)
        method: "fft" for large PSFs, "direct" for small

    Returns:
        Blurred image
    """
    if method == "fft":
        # FFT convolution (fast for large PSFs)
        from scipy.signal import fftconvolve
        return fftconvolve(image, psf, mode='same')

    else:
        # Direct convolution (accurate for small PSFs)
        from scipy.ndimage import convolve
        return convolve(image, psf, mode='reflect')
```

### 9.3 GPU-Accelerated Convolution

```python
def apply_psf_gpu(
    image: np.ndarray,
    psf: np.ndarray
) -> np.ndarray:
    """
    GPU-accelerated PSF convolution using CuPy.
    """
    import cupy as cp
    from cupyx.scipy.signal import fftconvolve

    # Transfer to GPU
    image_gpu = cp.asarray(image)
    psf_gpu = cp.asarray(psf)

    # Convolve on GPU
    result_gpu = fftconvolve(image_gpu, psf_gpu, mode='same')

    # Transfer back
    return cp.asnumpy(result_gpu)
```

---

## 10. Optics Module Implementation

### 10.1 Class Structure

```python
# src/eosim/optics/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class OpticsConfig:
    """Optical system configuration."""
    focal_length_mm: float
    f_number: float
    transmission: float = 0.85

    # Detector coupling
    pixel_pitch_um: float = 15.0
    resolution: tuple = (640, 512)

    # Aberrations (Zernike coefficients in waves)
    defocus_waves: float = 0.0
    astigmatism_waves: float = 0.0
    coma_waves: float = 0.0
    spherical_waves: float = 0.0

    # Distortion
    k1: float = 0.0  # Radial distortion
    k2: float = 0.0

    @property
    def ifov_mrad(self) -> float:
        """Instantaneous field of view in milliradians."""
        return self.pixel_pitch_um / self.focal_length_mm

    @property
    def fov_deg(self) -> tuple:
        """Full field of view (H, V) in degrees."""
        h = 2 * np.degrees(np.arctan(
            self.resolution[0] * self.pixel_pitch_um * 1e-3 /
            (2 * self.focal_length_mm)
        ))
        v = 2 * np.degrees(np.arctan(
            self.resolution[1] * self.pixel_pitch_um * 1e-3 /
            (2 * self.focal_length_mm)
        ))
        return (h, v)


class OpticsModel(ABC):
    """Abstract base class for optics models."""

    @abstractmethod
    def get_psf(self, wavelength_um: float, field_position: tuple = (0, 0)) -> np.ndarray:
        """Get PSF at specified wavelength and field position."""
        pass

    @abstractmethod
    def apply(self, radiance_image: np.ndarray, wavelength_um: float) -> np.ndarray:
        """Apply optical effects to radiance image."""
        pass

    @abstractmethod
    def get_transmission(self, wavelength_um: float) -> float:
        """Get optical transmission at wavelength."""
        pass


class SimpleOptics(OpticsModel):
    """Simple Gaussian PSF optics model."""

    def __init__(self, config: OpticsConfig):
        self.config = config

    def get_psf(self, wavelength_um: float, field_position: tuple = (0, 0)) -> np.ndarray:
        # Diffraction-limited FWHM
        fwhm_m = 1.03 * wavelength_um * 1e-6 * self.config.f_number
        fwhm_pixels = fwhm_m / (self.config.pixel_pitch_um * 1e-6)

        # Add aberration contribution (simplified)
        aberration_blur = np.sqrt(
            self.config.defocus_waves**2 +
            self.config.astigmatism_waves**2 +
            self.config.coma_waves**2
        ) * 2  # Approximate pixel blur from waves

        sigma = np.sqrt((fwhm_pixels/2.355)**2 + aberration_blur**2)

        return gaussian_psf(15, sigma)

    def apply(self, radiance_image: np.ndarray, wavelength_um: float) -> np.ndarray:
        psf = self.get_psf(wavelength_um)
        blurred = apply_psf(radiance_image, psf)
        return blurred * self.config.transmission

    def get_transmission(self, wavelength_um: float) -> float:
        return self.config.transmission


class PhysicalOptics(OpticsModel):
    """Physical optics model with full aberration support."""

    def __init__(self, config: OpticsConfig):
        self.config = config
        self._psf_cache = {}

    def get_psf(self, wavelength_um: float, field_position: tuple = (0, 0)) -> np.ndarray:
        # Check cache
        cache_key = (wavelength_um, field_position)
        if cache_key in self._psf_cache:
            return self._psf_cache[cache_key]

        # Compute aberrated PSF
        zernike = {
            4: self.config.defocus_waves,
            5: self.config.astigmatism_waves,
            7: self.config.coma_waves,
            11: self.config.spherical_waves,
        }

        psf = compute_aberrated_psf(
            size_pixels=31,
            pixel_pitch_m=self.config.pixel_pitch_um * 1e-6,
            wavelength_m=wavelength_um * 1e-6,
            f_number=self.config.f_number,
            zernike_coeffs=zernike
        )

        self._psf_cache[cache_key] = psf
        return psf

    def apply(self, radiance_image: np.ndarray, wavelength_um: float) -> np.ndarray:
        psf = self.get_psf(wavelength_um)
        blurred = apply_psf(radiance_image, psf)
        # Apply vignetting
        blurred = self._apply_vignetting(blurred)
        return blurred * self.config.transmission

    def _apply_vignetting(self, image: np.ndarray) -> np.ndarray:
        """Apply cos⁴ vignetting."""
        h, w = image.shape
        y, x = np.ogrid[:h, :w]
        cy, cx = h/2, w/2

        # Field angle per pixel
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r_max = np.sqrt(cx**2 + cy**2)
        theta = np.arctan(r / r_max * np.tan(np.radians(self.config.fov_deg[0]/2)))

        vignette = np.cos(theta)**4
        return image * vignette

    def get_transmission(self, wavelength_um: float) -> float:
        return self.config.transmission
```

---

## 11. Summary

### Key Equations

1. **IFOV**: `IFOV = d_pixel / f`
2. **GSD**: `GSD = IFOV × Range`
3. **Focal plane irradiance**: `E = π × L × τ / (4 × F/#²)`
4. **Airy radius**: `r_Airy = 1.22 × λ × F/#`
5. **System MTF**: `MTF_sys = MTF_optics × MTF_detector × ...`
6. **Nyquist**: `f_Nyquist = 1 / (2 × d_pixel)`

### Implementation Priority

1. **Gaussian PSF** - Fastest, adequate for many cases
2. **Diffraction-limited (Airy)** - For well-corrected optics
3. **MTF-based degradation** - For system analysis
4. **Full Zernike aberrations** - High-fidelity mode
5. **Field-dependent effects** - Vignetting, distortion

### Key Considerations

- PSF convolution is computationally expensive → use FFT
- GPU acceleration provides significant speedup
- Cache PSFs when wavelength/field don't change
- For broadband sensors, may need wavelength-integrated PSF

---

## 12. References

1. **Goodman** (2005) - "Introduction to Fourier Optics"
2. **Born & Wolf** (2013) - "Principles of Optics"
3. **Holst & Lomheim** (2011) - "CMOS/CCD Sensors and Camera Systems"
4. **Shannon** (1997) - "The Art and Science of Optical Design"
5. **ISETCam Documentation** - Optics modeling approach
