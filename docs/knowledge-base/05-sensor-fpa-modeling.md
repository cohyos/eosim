# Sensor and Focal Plane Array Modeling

## 1. Introduction

This document covers the modeling of focal plane array (FPA) detectors in EOSIM. The sensor module converts optical irradiance at the focal plane into digital output values, accounting for quantum efficiency, noise sources, and signal processing.

---

## 2. Detector Types

### 2.1 Common FPA Technologies

| Detector | Spectral Range | Operating Temp | Typical NEΔT | Notes |
|----------|---------------|----------------|--------------|-------|
| **Si CCD/CMOS** | 0.4-1.0 μm | Ambient | N/A | Visible cameras |
| **InGaAs** | 0.9-1.7 μm | TE-cooled | N/A | SWIR imaging |
| **InSb** | 1.0-5.5 μm | 77 K | 10-25 mK | MWIR, cooled |
| **HgCdTe (MCT)** | 1-12+ μm | 77 K | 15-30 mK | Tunable bandgap |
| **QWIP** | 8-10 μm | 70 K | 20-35 mK | Quantum well |
| **Microbolometer** | 8-14 μm | Ambient | 30-100 mK | Uncooled LWIR |

### 2.2 Detector Comparison

```
┌────────────────────────────────────────────────────────────────────┐
│                    DETECTOR TECHNOLOGY COMPARISON                  │
└────────────────────────────────────────────────────────────────────┘

                    PHOTON DETECTORS              THERMAL DETECTORS
                    ────────────────              ─────────────────
                    InSb, HgCdTe, QWIP           Microbolometer

Mechanism:          Photon → electron-hole       Photon → heat → ΔR
Response time:      ~μs (fast)                   ~ms (slow)
Cooling:            Required (77K)               Not required
Sensitivity:        High (NEΔT ~20mK)           Moderate (NEΔT ~50mK)
Cost:               High                         Low
Spectral:           Band-specific               Broadband thermal
```

---

## 3. Signal Chain

### 3.1 Complete Signal Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DETECTOR SIGNAL CHAIN                             │
└─────────────────────────────────────────────────────────────────────────────┘

  Photons   QE      Electrons    +Dark      +Noise     Gain    ADC      DN
    │       │          │          │           │         │       │        │
    ▼       ▼          ▼          ▼           ▼         ▼       ▼        ▼
┌──────┐ ┌──────┐ ┌──────────┐ ┌──────┐ ┌─────────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ Φ_ph │→│ η(λ) │→│ N_signal │→│ +N_d │→│ +noise  │→│ ×G   │→│ ADC  │→│ DN   │
│      │ │      │ │          │ │      │ │         │ │      │ │      │ │      │
└──────┘ └──────┘ └──────────┘ └──────┘ └─────────┘ └──────┘ └──────┘ └──────┘
                                           │
                               ┌───────────┼───────────┐
                               ▼           ▼           ▼
                            Shot       Read        Dark
                            noise      noise       noise
```

### 3.2 Mathematical Model

```
Signal electrons:
  N_signal = Φ_photon × η(λ) × t_int

Dark electrons:
  N_dark = I_dark × t_int

Total electrons before noise:
  N_total = N_signal + N_dark

After noise (stochastic):
  N_noisy = Poisson(N_total) + Gaussian(0, σ_read)

Voltage:
  V = N_noisy × G_conversion  [V/e⁻]

Digital number:
  DN = clip(round((V - V_offset) / V_step), 0, 2^bits - 1)
```

---

## 4. Quantum Efficiency

### 4.1 Definition

```
η(λ) = Number of electrons generated / Number of incident photons

Factors affecting QE:
- Absorption depth vs detector thickness
- Surface reflection losses
- Recombination losses
- Fill factor
```

### 4.2 Typical QE Curves

```
QE (%)
100├─────────────────────────────────────────────
   │            ╭───────╮
 80│           ╱         ╲      HgCdTe (MWIR)
   │          ╱           ╲
 60│         ╱             ╲
   │   Si   ╱               ╲
 40│  ╱────╲                 ╲
   │ ╱      ╲                 ╲
 20│╱        ╲                 ╲
   │          ╲─────────────────╲───
  0├────┬────┬────┬────┬────┬────┬────┬────
   0.4  0.6  0.8  1.0  2.0  3.0  4.0  5.0  λ(μm)
```

### 4.3 Implementation

```python
import numpy as np
from scipy.interpolate import interp1d

class QuantumEfficiency:
    """Spectral quantum efficiency model."""

    def __init__(
        self,
        wavelengths_um: np.ndarray,
        qe_values: np.ndarray
    ):
        """
        Initialize from measured or modeled QE curve.

        Args:
            wavelengths_um: Wavelength sample points [μm]
            qe_values: QE values at each wavelength (0-1)
        """
        self._interp = interp1d(
            wavelengths_um,
            qe_values,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )

    def __call__(self, wavelength_um: float) -> float:
        """Get QE at specified wavelength."""
        return float(self._interp(wavelength_um))

    @classmethod
    def from_detector_type(cls, detector_type: str) -> 'QuantumEfficiency':
        """Create QE model for common detector types."""
        if detector_type == "InSb":
            wl = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0])
            qe = np.array([0.5, 0.7, 0.75, 0.70, 0.60, 0.30, 0.0])
        elif detector_type == "HgCdTe_MWIR":
            wl = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.5])
            qe = np.array([0.6, 0.7, 0.75, 0.72, 0.65, 0.50])
        elif detector_type == "HgCdTe_LWIR":
            wl = np.array([6.0, 8.0, 10.0, 12.0, 14.0])
            qe = np.array([0.5, 0.65, 0.60, 0.50, 0.30])
        elif detector_type == "microbolometer":
            wl = np.array([7.0, 8.0, 10.0, 12.0, 14.0])
            qe = np.array([0.8, 0.85, 0.80, 0.75, 0.70])  # Absorption, not QE
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

        return cls(wl, qe)
```

---

## 5. Noise Sources

### 5.1 Noise Components

| Noise Source | Distribution | Variance | Notes |
|--------------|--------------|----------|-------|
| **Shot noise** | Poisson | σ² = N | Fundamental photon statistics |
| **Dark current** | Poisson | σ² = I_d × t | Thermal generation |
| **Read noise** | Gaussian | σ² = σ_read² | Amplifier noise |
| **PRNU** | Multiplicative | σ² = (k×N)² | Pixel response variation |
| **DSNU** | Additive | σ² = σ_dsnu² | Dark signal variation |
| **Quantization** | Uniform | σ² = Δ²/12 | ADC discretization |

### 5.2 Shot Noise

Fundamental quantum noise from discrete photon arrival:

```
σ_shot = √N_electrons

For thermal detector signal:
  N = ∫ L(λ) × A × Ω × τ_opt × η(λ) × λ/(hc) × t_int dλ

Signal-to-noise from shot noise alone:
  SNR_shot = N / √N = √N
```

### 5.3 Dark Current

Thermally generated carriers (temperature-dependent):

```
I_dark = I_0 × exp(-E_g / (2kT))

Where:
  I_0 = Material constant [e⁻/s]
  E_g = Bandgap energy [eV]
  k = Boltzmann constant
  T = Operating temperature [K]

Typical values:
  InSb @ 77K: ~100-1000 e⁻/s
  HgCdTe @ 77K: ~10-100 e⁻/s
  Microbolometer @ 300K: (different mechanism)
```

### 5.4 Read Noise

Noise from readout amplifier:

```
σ_read = constant [e⁻ rms]

Typical values:
  Scientific CCD: 2-10 e⁻
  Consumer CMOS: 10-50 e⁻
  IR FPA (ROIC): 20-200 e⁻
```

### 5.5 Fixed Pattern Noise

**Photo-Response Non-Uniformity (PRNU)**:
```
N_pixel = N_ideal × (1 + PRNU_pixel)

Where PRNU_pixel ~ N(0, σ_prnu)
σ_prnu typically 0.5-2% of signal
```

**Dark Signal Non-Uniformity (DSNU)**:
```
N_dark_pixel = N_dark_mean + DSNU_pixel

Where DSNU_pixel is fixed offset per pixel
```

### 5.6 Total Noise

```
σ_total² = σ_shot² + σ_dark² + σ_read² + σ_prnu² + σ_dsnu² + σ_quant²

Total noise (electrons):
σ_total = √(N_signal + N_dark + σ_read² + (PRNU×N_signal)² + σ_dsnu²)
```

---

## 6. Noise Implementation

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class NoiseModel:
    """FPA noise parameters."""
    read_noise_e: float = 50.0           # Read noise [e⁻ rms]
    dark_current_e_s: float = 1000.0     # Dark current [e⁻/s]
    prnu_percent: float = 1.0            # PRNU [% of signal]
    dsnu_e: float = 20.0                 # DSNU [e⁻ rms]
    quantization_bits: int = 14          # ADC bits


class SensorNoise:
    """
    Sensor noise injection model following EMVA 1288 standard.
    """

    def __init__(
        self,
        noise_model: NoiseModel,
        resolution: tuple,
        seed: Optional[int] = None
    ):
        self.params = noise_model
        self.resolution = resolution
        self.rng = np.random.default_rng(seed)

        # Generate fixed pattern maps (constant per sensor instance)
        self._prnu_map = 1.0 + self.rng.normal(
            0, noise_model.prnu_percent / 100,
            resolution
        )
        self._dsnu_map = self.rng.normal(
            0, noise_model.dsnu_e,
            resolution
        )

    def apply(
        self,
        signal_electrons: np.ndarray,
        integration_time_s: float
    ) -> np.ndarray:
        """
        Apply all noise sources to ideal signal.

        Args:
            signal_electrons: Ideal signal in electrons [H, W]
            integration_time_s: Integration time

        Returns:
            Noisy signal in electrons
        """
        # Apply PRNU (multiplicative)
        signal = signal_electrons * self._prnu_map

        # Add dark current
        dark = self.params.dark_current_e_s * integration_time_s
        signal = signal + dark

        # Apply shot noise (Poisson)
        # For large counts, use Gaussian approximation
        signal = np.where(
            signal > 100,
            signal + self.rng.normal(0, np.sqrt(np.maximum(signal, 0)), signal.shape),
            self.rng.poisson(np.maximum(signal, 0))
        )

        # Add DSNU (additive fixed pattern)
        signal = signal + self._dsnu_map

        # Add read noise (Gaussian)
        signal = signal + self.rng.normal(0, self.params.read_noise_e, signal.shape)

        return signal

    def get_noise_components(
        self,
        signal_electrons: float,
        integration_time_s: float
    ) -> dict:
        """
        Calculate individual noise contributions.

        Returns dict with each noise component in electrons RMS.
        """
        dark = self.params.dark_current_e_s * integration_time_s

        return {
            'shot': np.sqrt(signal_electrons + dark),
            'dark': np.sqrt(dark),
            'read': self.params.read_noise_e,
            'prnu': signal_electrons * self.params.prnu_percent / 100,
            'dsnu': self.params.dsnu_e,
            'total': np.sqrt(
                signal_electrons + dark +
                self.params.read_noise_e**2 +
                (signal_electrons * self.params.prnu_percent / 100)**2 +
                self.params.dsnu_e**2
            )
        }
```

---

## 7. Analog-to-Digital Conversion

### 7.1 ADC Model

```
DN = clip(floor((V - V_offset) / V_LSB + 0.5), 0, 2^bits - 1)

Where:
  V = Input voltage
  V_offset = Zero offset
  V_LSB = Voltage per least significant bit
  bits = ADC resolution (typically 12-16)
```

### 7.2 Dynamic Range

```
Dynamic range = Full_well_capacity / Read_noise

In dB: DR = 20 × log10(FWC / σ_read)

Example:
  FWC = 100,000 e⁻
  σ_read = 50 e⁻
  DR = 20 × log10(2000) = 66 dB
```

### 7.3 Implementation

```python
@dataclass
class ADCModel:
    """Analog-to-digital converter model."""
    bit_depth: int = 14
    full_well_capacity_e: float = 100000
    gain_e_per_dn: float = 10.0  # Electrons per DN
    offset_dn: int = 100          # Black level offset

    @property
    def max_dn(self) -> int:
        return 2**self.bit_depth - 1

    def convert(self, electrons: np.ndarray) -> np.ndarray:
        """
        Convert electrons to digital numbers.

        Args:
            electrons: Signal in electrons

        Returns:
            Digital numbers (uint16 or similar)
        """
        # Clip to full well capacity
        electrons = np.clip(electrons, 0, self.full_well_capacity_e)

        # Convert to DN
        dn = electrons / self.gain_e_per_dn + self.offset_dn

        # Quantize and clip
        dn = np.clip(np.round(dn), 0, self.max_dn)

        return dn.astype(np.uint16)
```

---

## 8. Performance Metrics

### 8.1 NEΔT (Noise Equivalent Temperature Difference)

Minimum detectable temperature difference:

```
NEΔT = σ_total / (dN/dT)

Where:
  σ_total = Total noise in electrons
  dN/dT = Rate of signal change with temperature

Derivation:
  dN/dT = A × Ω × τ × η × t_int × ∫ (dL/dT) × λ/(hc) dλ

  dL/dT from Planck: dL/dT = L × C₂/(λT²) × exp(C₂/λT)/(exp(C₂/λT)-1)
```

### 8.2 SNR (Signal-to-Noise Ratio)

```
SNR = N_signal / σ_total

In dB: SNR_dB = 20 × log10(SNR)

Johnson criterion for detection:
  SNR ≥ 2.8 for 50% detection probability
```

### 8.3 Detectivity (D*)

Normalized sensitivity metric:

```
D* = √(A × Δf) / NEP  [cm·√Hz/W]

Where:
  A = Detector area
  Δf = Noise bandwidth
  NEP = Noise equivalent power

Higher D* = better sensitivity
```

### 8.4 Implementation

```python
def compute_nedt(
    signal_electrons: float,
    noise_electrons: float,
    temperature_K: float,
    wavelength_um: float,
    bandwidth_um: float = 1.0
) -> float:
    """
    Compute NEΔT for a thermal detector.

    Args:
        signal_electrons: Signal level [e⁻]
        noise_electrons: Total noise [e⁻ rms]
        temperature_K: Scene temperature [K]
        wavelength_um: Center wavelength [μm]
        bandwidth_um: Spectral bandwidth [μm]

    Returns:
        NEΔT in Kelvin
    """
    # Planck contrast (dL/dT normalized by L)
    C2 = 14387.752  # μm·K
    x = C2 / (wavelength_um * temperature_K)
    contrast = x * np.exp(x) / (np.exp(x) - 1) / temperature_K

    # dN/dT
    dN_dT = signal_electrons * contrast

    # NEΔT
    return noise_electrons / dN_dT


def compute_snr(
    signal_electrons: float,
    noise_model: NoiseModel,
    integration_time_s: float
) -> float:
    """Compute signal-to-noise ratio."""
    dark = noise_model.dark_current_e_s * integration_time_s

    noise_total = np.sqrt(
        signal_electrons + dark +
        noise_model.read_noise_e**2 +
        (signal_electrons * noise_model.prnu_percent / 100)**2 +
        noise_model.dsnu_e**2
    )

    return signal_electrons / noise_total
```

---

## 9. Temporal Effects

### 9.1 Integration Modes

**Snapshot (global shutter)**:
- All pixels integrate simultaneously
- No motion artifacts
- Common in scientific sensors

**Rolling shutter**:
- Rows integrate sequentially
- Motion causes skew/wobble
- Common in CMOS sensors

**Integrate-while-read (IWR)**:
- Overlapped integration and readout
- Higher frame rates possible

### 9.2 Frame Timing

```
┌─────────────────────────────────────────────────────────────────┐
│                      FRAME TIMING DIAGRAM                       │
└─────────────────────────────────────────────────────────────────┘

Frame N                              Frame N+1
├──────────────────────────────────┤├──────────────────
│                                  │
├─────────────────┬────────────────┤
│   Integration   │    Readout     │
│     (t_int)     │    (t_read)    │
└─────────────────┴────────────────┘

Frame period: T_frame = t_int + t_read (non-overlapping)
           or T_frame = max(t_int, t_read) (IWR)

Frame rate: f = 1 / T_frame
```

### 9.3 Motion Blur

```python
def compute_motion_blur(
    ground_velocity_mps: float,
    gsd_m: float,
    integration_time_s: float
) -> float:
    """
    Compute motion blur in pixels.

    Args:
        ground_velocity_mps: Relative ground motion [m/s]
        gsd_m: Ground sample distance [m/pixel]
        integration_time_s: Integration time [s]

    Returns:
        Blur extent in pixels
    """
    motion_m = ground_velocity_mps * integration_time_s
    blur_pixels = motion_m / gsd_m
    return blur_pixels
```

---

## 10. Sensor Module Implementation

### 10.1 Complete Sensor Model

```python
# src/eosim/sensor/fpa.py

from dataclasses import dataclass
import numpy as np

@dataclass
class SensorConfig:
    """FPA sensor configuration."""
    # Geometry
    resolution: tuple = (640, 512)     # (width, height)
    pixel_pitch_um: float = 15.0

    # Detector
    detector_type: str = "InSb"
    operating_temperature_K: float = 77.0

    # Timing
    integration_time_ms: float = 10.0
    frame_rate_hz: float = 30.0

    # Noise (from datasheet or measurement)
    read_noise_e: float = 50.0
    dark_current_e_s: float = 1000.0
    prnu_percent: float = 1.0
    dsnu_e: float = 20.0

    # Well and ADC
    full_well_capacity_e: float = 2.0e6
    adc_bits: int = 14
    gain_e_per_dn: float = 100.0
    offset_dn: int = 100

    # Spectral
    wavelength_min_um: float = 3.0
    wavelength_max_um: float = 5.0


class FPASensor:
    """
    Complete focal plane array sensor model.
    """

    def __init__(self, config: SensorConfig, seed: int = None):
        self.config = config

        # Initialize sub-models
        self.qe = QuantumEfficiency.from_detector_type(config.detector_type)
        self.noise = SensorNoise(
            NoiseModel(
                read_noise_e=config.read_noise_e,
                dark_current_e_s=config.dark_current_e_s,
                prnu_percent=config.prnu_percent,
                dsnu_e=config.dsnu_e,
                quantization_bits=config.adc_bits
            ),
            config.resolution,
            seed=seed
        )
        self.adc = ADCModel(
            bit_depth=config.adc_bits,
            full_well_capacity_e=config.full_well_capacity_e,
            gain_e_per_dn=config.gain_e_per_dn,
            offset_dn=config.offset_dn
        )

    def process(
        self,
        irradiance_wm2: np.ndarray,
        wavelength_um: float
    ) -> np.ndarray:
        """
        Process focal plane irradiance to digital output.

        Args:
            irradiance_wm2: Focal plane irradiance [W/m²]
            wavelength_um: Effective wavelength [μm]

        Returns:
            Digital output image [DN]
        """
        # Convert irradiance to photons
        pixel_area = (self.config.pixel_pitch_um * 1e-6)**2
        photon_energy = 6.626e-34 * 3e8 / (wavelength_um * 1e-6)
        photon_flux = irradiance_wm2 * pixel_area / photon_energy

        # Apply quantum efficiency
        electron_flux = photon_flux * self.qe(wavelength_um)

        # Integrate
        t_int = self.config.integration_time_ms * 1e-3
        electrons = electron_flux * t_int

        # Apply noise
        electrons_noisy = self.noise.apply(electrons, t_int)

        # ADC conversion
        dn = self.adc.convert(electrons_noisy)

        return dn

    def get_performance_metrics(
        self,
        scene_temperature_K: float,
        wavelength_um: float
    ) -> dict:
        """
        Calculate sensor performance metrics for given scene.
        """
        # Estimate signal level
        from ..radiance import planck_radiance
        L = planck_radiance(wavelength_um, scene_temperature_K)

        # Simplified focal plane irradiance
        # E = π × L × τ / (4 × F#²) - need F# from optics
        # For now, use placeholder
        E = L * 0.01  # Rough approximation

        t_int = self.config.integration_time_ms * 1e-3
        pixel_area = (self.config.pixel_pitch_um * 1e-6)**2
        photon_energy = 6.626e-34 * 3e8 / (wavelength_um * 1e-6)
        signal_e = E * pixel_area * self.qe(wavelength_um) * t_int / photon_energy

        # Noise
        noise_info = self.noise.get_noise_components(signal_e, t_int)

        # NEΔT
        nedt = compute_nedt(
            signal_e, noise_info['total'],
            scene_temperature_K, wavelength_um
        )

        return {
            'signal_electrons': signal_e,
            'noise_electrons': noise_info['total'],
            'snr': signal_e / noise_info['total'],
            'nedt_mK': nedt * 1000,
            'noise_breakdown': noise_info
        }
```

---

## 11. Non-Uniformity Correction (NUC)

### 11.1 Two-Point Correction

```
DN_corrected = (DN_raw - offset) × gain

Where offset and gain are per-pixel calibration coefficients
from viewing two known uniform sources (blackbodies).
```

### 11.2 NUC Implementation

```python
class NonUniformityCorrection:
    """Two-point NUC calibration."""

    def __init__(self, resolution: tuple):
        self.resolution = resolution
        self.gain = np.ones(resolution)
        self.offset = np.zeros(resolution)

    def calibrate(
        self,
        image_cold: np.ndarray,
        image_hot: np.ndarray,
        temp_cold_K: float,
        temp_hot_K: float
    ):
        """
        Compute NUC coefficients from two calibration images.
        """
        # Mean response at each temperature
        mean_cold = image_cold.mean()
        mean_hot = image_hot.mean()

        # Gain: normalize to mean response
        self.gain = (mean_hot - mean_cold) / (image_hot - image_cold + 1e-10)

        # Offset: adjust so corrected cold = mean cold
        self.offset = image_cold - mean_cold / self.gain

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply NUC correction."""
        return (image - self.offset) * self.gain
```

---

## 12. Summary

### Key Equations

1. **Signal**: `N = Φ_photon × η × t_int`
2. **Shot noise**: `σ_shot = √N`
3. **Total noise**: `σ = √(N + N_dark + σ_read² + (PRNU×N)² + σ_dsnu²)`
4. **SNR**: `SNR = N_signal / σ_total`
5. **NEΔT**: `NEΔT = σ_total / (dN/dT)`

### Implementation Priority

1. **Basic signal chain** - Irradiance → electrons → DN
2. **Shot + read noise** - Fundamental noise sources
3. **Dark current** - Temperature-dependent
4. **Fixed pattern noise** - PRNU/DSNU maps
5. **Full EMVA 1288 model** - Complete characterization

### Key Considerations

- Noise model should match sensor datasheet specifications
- FPN maps should persist across frames (fixed per sensor instance)
- Consider well capacity saturation
- NUC important for quantitative radiometry

---

## 13. References

1. **EMVA 1288** - Standard for Measurement of Image Sensor Characteristics
2. **Holst** (2008) - "Electro-Optical Imaging System Performance"
3. **Rogalski** (2010) - "Infrared Detectors"
4. **Dereniak & Boreman** (1996) - "Infrared Detectors and Systems"
5. **ISETCam** - Sensor noise modeling implementation
6. **Pyxel (ESA)** - Detector simulation framework
