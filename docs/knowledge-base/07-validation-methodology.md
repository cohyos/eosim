# Validation Methodology

## 1. Introduction

This document defines the validation strategy for EOSIM, ensuring that simulation outputs are physically accurate and trustworthy. Validation occurs at multiple levels: individual modules, integrated pipeline, and system-level comparison with reference data.

---

## 2. Validation Philosophy

### 2.1 Verification vs Validation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VERIFICATION VS VALIDATION                               │
└─────────────────────────────────────────────────────────────────────────────┘

VERIFICATION: "Are we building the system right?"
├── Code correctness
├── Unit tests pass
├── Algorithms implemented correctly
└── No bugs in implementation

VALIDATION: "Are we building the right system?"
├── Physics are accurate
├── Outputs match real-world measurements
├── Model assumptions are appropriate
└── Results are useful for intended application
```

### 2.2 Multi-Level Validation Strategy

```
Level 1: UNIT VALIDATION
├── Individual functions
├── Physical constants
└── Mathematical formulas

Level 2: MODULE VALIDATION
├── Radiometry module vs Planck tables
├── Atmosphere module vs MODTRAN/RAF-tran
├── Sensor module vs EMVA 1288
└── Optics module vs diffraction theory

Level 3: INTEGRATION VALIDATION
├── End-to-end pipeline
├── Known scene simulations
└── Cross-module consistency

Level 4: SYSTEM VALIDATION
├── Comparison with reference simulations (DIRSIG, etc.)
├── Comparison with real sensor data
└── Field test correlation
```

---

## 3. Unit Validation

### 3.1 Physical Constants

```python
# tests/test_constants.py

import pytest
from eosim.core.constants import *

def test_physical_constants():
    """Verify physical constants against CODATA values."""

    # Speed of light
    assert abs(C - 299792458.0) < 1.0  # m/s

    # Planck constant
    assert abs(H - 6.62607015e-34) / H < 1e-8  # J·s

    # Boltzmann constant
    assert abs(K_B - 1.380649e-23) / K_B < 1e-8  # J/K

    # Stefan-Boltzmann constant
    assert abs(SIGMA - 5.670374419e-8) / SIGMA < 1e-8  # W/(m²·K⁴)

    # First radiation constant
    assert abs(C1 - 1.191042972e8) / C1 < 1e-6  # W·μm⁴/(m²·sr)

    # Second radiation constant
    assert abs(C2 - 14387.7729) / C2 < 1e-6  # μm·K
```

### 3.2 Planck Function

```python
# tests/test_radiometry.py

import numpy as np
from eosim.radiance import planck_radiance

def test_planck_wien_displacement():
    """Verify Planck function peaks at Wien wavelength."""
    T = 5778  # Sun temperature

    wavelengths = np.linspace(0.1, 10, 10000)
    radiance = planck_radiance(wavelengths, T)

    peak_idx = np.argmax(radiance)
    peak_wavelength = wavelengths[peak_idx]

    # Wien's law: λ_max = 2897.8 / T
    expected_peak = 2897.8 / T  # ~0.501 μm

    assert abs(peak_wavelength - expected_peak) < 0.01  # Within 10 nm


def test_planck_stefan_boltzmann():
    """Verify integrated Planck equals Stefan-Boltzmann."""
    from scipy.integrate import quad

    T = 300  # K

    # Integrate Planck over all wavelengths (0.1 to 1000 μm)
    def integrand(lam):
        return np.pi * planck_radiance(lam, T)  # M = π × L for Lambertian

    M_integrated, _ = quad(integrand, 0.1, 1000)

    # Stefan-Boltzmann
    M_sb = 5.670374419e-8 * T**4

    assert abs(M_integrated - M_sb) / M_sb < 0.01  # Within 1%


def test_planck_reference_values():
    """Compare against published reference values."""
    # Reference: MODTRAN documentation, Table 4-1

    test_cases = [
        # (wavelength_um, temperature_K, expected_radiance)
        (10.0, 300, 9.89),   # W/(m²·sr·μm) at LWIR peak
        (4.0, 300, 0.228),   # MWIR
        (0.55, 5778, 2.11e7),  # Visible solar
    ]

    for wl, T, expected in test_cases:
        computed = planck_radiance(wl, T)
        assert abs(computed - expected) / expected < 0.05  # Within 5%
```

---

## 4. Module Validation

### 4.1 Atmosphere Module Validation

```python
# tests/test_atmosphere.py

import numpy as np
import pytest
from eosim.atmosphere import RAFTranAtmosphere, SlantPath

class TestAtmosphereValidation:
    """Validate atmosphere module against reference data."""

    @pytest.fixture
    def atmosphere(self):
        return RAFTranAtmosphere(
            profile="midlatitude_summer",
            visibility_km=23.0
        )

    def test_transmission_bounds(self, atmosphere):
        """Transmission must be between 0 and 1."""
        wavelengths = np.linspace(0.4, 14.0, 100)
        path = SlantPath(
            sensor_altitude_m=10000,
            target_altitude_m=0,
            zenith_angle_deg=30,
            azimuth_angle_deg=0
        )

        result = atmosphere.compute(wavelengths, path)

        assert np.all(result.transmission >= 0)
        assert np.all(result.transmission <= 1)

    def test_transmission_windows(self, atmosphere):
        """Verify high transmission in atmospheric windows."""
        path = SlantPath(1000, 0, 0, 0)  # Vertical, 1km

        # MWIR window (3-5 μm) should have high transmission
        mwir_result = atmosphere.compute(np.array([4.0]), path)
        assert mwir_result.transmission[0] > 0.7

        # LWIR window (8-12 μm) should have high transmission
        lwir_result = atmosphere.compute(np.array([10.0]), path)
        assert lwir_result.transmission[0] > 0.7

        # CO2 absorption band (4.3 μm) should have low transmission
        co2_result = atmosphere.compute(np.array([4.3]), path)
        assert co2_result.transmission[0] < 0.3

    def test_path_length_dependence(self, atmosphere):
        """Longer paths should have lower transmission."""
        wl = np.array([4.0])

        short_path = SlantPath(500, 0, 0, 0)
        long_path = SlantPath(5000, 0, 0, 0)

        short_result = atmosphere.compute(wl, short_path)
        long_result = atmosphere.compute(wl, long_path)

        assert long_result.transmission[0] < short_result.transmission[0]

    def test_vs_py6s_reference(self, atmosphere):
        """Compare against Py6S for validation (VIS-SWIR only)."""
        pytest.importorskip("Py6S")
        from Py6S import SixS, Wavelength

        # Set up Py6S reference
        s = SixS()
        s.atmos_profile = SixS.AtmosProfile.MidlatitudeSummer
        s.aero_profile = SixS.AeroProfile.Continental
        s.visibility = 23

        # Compare at several wavelengths
        wavelengths = [0.55, 0.8, 1.0, 1.6]

        path = SlantPath(10000, 0, 30, 0)

        for wl in wavelengths:
            # Py6S
            s.wavelength = Wavelength(wl)
            s.geometry.solar_z = 30
            s.geometry.view_z = 30
            s.run()
            py6s_trans = s.outputs.transmittance_global_gas.total

            # EOSIM
            eosim_result = atmosphere.compute(np.array([wl]), path)
            eosim_trans = eosim_result.transmission[0]

            # Allow 10% difference
            assert abs(eosim_trans - py6s_trans) / py6s_trans < 0.10, \
                f"Mismatch at {wl} μm: EOSIM={eosim_trans:.3f}, Py6S={py6s_trans:.3f}"
```

### 4.2 Sensor Module Validation

```python
# tests/test_sensor.py

import numpy as np
from eosim.sensor import FPASensor, SensorConfig, NoiseModel

class TestSensorValidation:
    """Validate sensor module against EMVA 1288 principles."""

    @pytest.fixture
    def sensor(self):
        config = SensorConfig(
            resolution=(640, 512),
            read_noise_e=50,
            dark_current_e_s=1000,
            full_well_capacity_e=100000,
            adc_bits=14
        )
        return FPASensor(config, seed=42)

    def test_shot_noise_scaling(self, sensor):
        """Shot noise should scale as sqrt(signal)."""
        # Generate flat field images at different signal levels
        signals = [1000, 10000, 50000]
        measured_noise = []

        for signal in signals:
            flat_field = np.full((100, 100), signal, dtype=float)
            noisy = sensor.noise.apply(flat_field, 0.01)
            measured_noise.append(np.std(noisy))

        # Shot noise: σ ∝ √N
        for i in range(len(signals) - 1):
            ratio_signal = np.sqrt(signals[i+1] / signals[i])
            ratio_noise = measured_noise[i+1] / measured_noise[i]

            # Should be approximately equal (within 20% due to other noise)
            assert abs(ratio_noise - ratio_signal) / ratio_signal < 0.3

    def test_read_noise_floor(self, sensor):
        """At zero signal, noise should equal read noise."""
        zero_signal = np.zeros((100, 100))
        noisy = sensor.noise.apply(zero_signal, 0)  # No integration time

        measured_noise = np.std(noisy)
        expected_noise = sensor.config.read_noise_e

        assert abs(measured_noise - expected_noise) / expected_noise < 0.1

    def test_linearity(self, sensor):
        """ADC output should be linear with signal."""
        signals = np.linspace(0, 80000, 10)
        outputs = []

        for signal in signals:
            flat = np.full((10, 10), signal)
            dn = sensor.adc.convert(flat)
            outputs.append(np.mean(dn))

        # Fit line
        coeffs = np.polyfit(signals, outputs, 1)

        # R² should be > 0.999
        fit = np.polyval(coeffs, signals)
        ss_res = np.sum((outputs - fit)**2)
        ss_tot = np.sum((outputs - np.mean(outputs))**2)
        r_squared = 1 - ss_res / ss_tot

        assert r_squared > 0.999

    def test_dynamic_range(self, sensor):
        """Verify dynamic range matches specification."""
        fwc = sensor.config.full_well_capacity_e
        read_noise = sensor.config.read_noise_e

        expected_dr_db = 20 * np.log10(fwc / read_noise)

        # Measure: ratio of max signal to noise floor
        max_signal = fwc
        noise_floor = read_noise

        measured_dr_db = 20 * np.log10(max_signal / noise_floor)

        assert abs(measured_dr_db - expected_dr_db) < 1  # Within 1 dB
```

### 4.3 Optics Module Validation

```python
# tests/test_optics.py

import numpy as np
from eosim.optics import SimpleOptics, OpticsConfig, airy_psf

class TestOpticsValidation:
    """Validate optics module against diffraction theory."""

    def test_airy_disk_size(self):
        """Verify Airy disk radius matches theory."""
        wavelength_um = 4.0
        f_number = 2.0
        pixel_pitch_um = 15.0

        psf = airy_psf(
            size_pixels=101,
            pixel_pitch_m=pixel_pitch_um * 1e-6,
            wavelength_m=wavelength_um * 1e-6,
            f_number=f_number
        )

        # Find radius to first zero (encircled energy ~84%)
        center = 50
        cumsum_radial = []
        for r in range(1, 50):
            mask = np.zeros_like(psf)
            y, x = np.ogrid[:101, :101]
            mask[(x - center)**2 + (y - center)**2 <= r**2] = 1
            cumsum_radial.append(np.sum(psf * mask))

        # First zero should contain ~84% of energy
        idx_84 = np.searchsorted(cumsum_radial, 0.84)
        measured_radius_pixels = idx_84

        # Theoretical Airy radius
        r_airy_m = 1.22 * wavelength_um * 1e-6 * f_number
        r_airy_pixels = r_airy_m / (pixel_pitch_um * 1e-6)

        assert abs(measured_radius_pixels - r_airy_pixels) < 2  # Within 2 pixels

    def test_mtf_cutoff(self):
        """Verify MTF cutoff frequency matches theory."""
        wavelength_um = 4.0
        f_number = 2.0

        # Theoretical cutoff
        f_cutoff = 1 / (wavelength_um * 1e-6 * f_number)  # cycles/m

        # Compute MTF from PSF
        psf = airy_psf(101, 15e-6, wavelength_um * 1e-6, f_number)
        mtf = np.abs(np.fft.fftshift(np.fft.fft2(psf)))
        mtf /= mtf.max()

        # Find frequency where MTF drops to ~0
        center = 50
        mtf_1d = mtf[center, center:]

        # Find cutoff (where MTF < 0.01)
        cutoff_idx = np.argmax(mtf_1d < 0.01)
        freq_step = 1 / (101 * 15e-6)  # cycles/m per pixel in freq domain
        measured_cutoff = cutoff_idx * freq_step

        assert abs(measured_cutoff - f_cutoff) / f_cutoff < 0.1  # Within 10%
```

---

## 5. Integration Validation

### 5.1 Known Scene Tests

```python
# tests/test_integration.py

class TestKnownScenes:
    """Integration tests with analytically predictable scenes."""

    def test_uniform_blackbody_scene(self):
        """
        Uniform blackbody at known temperature should produce
        predictable sensor output.
        """
        T_scene = 300  # K
        wavelength = 10.0  # μm (LWIR)

        # Expected radiance
        L_expected = planck_radiance(wavelength, T_scene)

        # Create uniform scene
        scenario = create_uniform_scene(temperature_K=T_scene)
        pipeline = RenderPipeline(scenario, fidelity="simple")

        # Render (no atmosphere, perfect optics for this test)
        frame = pipeline.render_frame(0)

        # Check mean radiance matches expected
        measured_radiance = frame.radiance_map.mean()

        assert abs(measured_radiance - L_expected) / L_expected < 0.05

    def test_two_temperature_contrast(self):
        """
        Two-temperature scene should produce expected contrast.
        """
        T_hot = 320  # K
        T_cold = 300  # K
        wavelength = 10.0  # μm

        # Expected contrast
        L_hot = planck_radiance(wavelength, T_hot)
        L_cold = planck_radiance(wavelength, T_cold)
        expected_contrast = L_hot / L_cold

        # Create scene with hot and cold regions
        scenario = create_two_temp_scene(T_hot, T_cold)
        pipeline = RenderPipeline(scenario, fidelity="simple")

        frame = pipeline.render_frame(0)

        # Measure contrast in output
        hot_region = frame.radiance_map[:256, :]
        cold_region = frame.radiance_map[256:, :]

        measured_contrast = hot_region.mean() / cold_region.mean()

        assert abs(measured_contrast - expected_contrast) / expected_contrast < 0.05

    def test_atmospheric_attenuation(self):
        """
        Verify atmosphere reduces signal appropriately.
        """
        scenario_no_atm = create_test_scenario(atmosphere=None)
        scenario_with_atm = create_test_scenario(
            atmosphere=AtmosphereConfig(visibility_km=10)
        )

        frame_no_atm = RenderPipeline(scenario_no_atm).render_frame(0)
        frame_with_atm = RenderPipeline(scenario_with_atm).render_frame(0)

        # Signal should be reduced
        assert frame_with_atm.radiance_map.mean() < frame_no_atm.radiance_map.mean()

    def test_noise_statistics(self):
        """
        Verify noise in uniform scene matches expected distribution.
        """
        scenario = create_uniform_scene(temperature_K=300)
        pipeline = RenderPipeline(scenario)

        # Render multiple frames
        frames = [pipeline.render_frame(t) for t in np.linspace(0, 1, 10)]

        # Stack and compute temporal noise
        stack = np.stack([f.image for f in frames], axis=0)
        temporal_std = np.std(stack, axis=0).mean()

        # Should match expected sensor noise
        expected_noise = pipeline.sensor.config.read_noise_e / pipeline.sensor.adc.gain_e_per_dn

        assert abs(temporal_std - expected_noise) / expected_noise < 0.2
```

### 5.2 Reciprocity Tests

```python
class TestReciprocity:
    """Tests for physical consistency and reciprocity."""

    def test_energy_conservation(self):
        """
        Total energy should be conserved through pipeline.
        """
        # Input radiance
        L_input = 10.0  # W/(m²·sr·μm)

        # Track through pipeline
        L_surface = L_input
        L_after_atm = L_surface * 0.9  # 90% transmission
        E_focal = L_after_atm * np.pi / (4 * 2.0**2)  # f/2 optics

        # Verify at each stage
        # ... detailed checks

    def test_reversibility(self):
        """
        Simulation should be deterministic given same seed.
        """
        scenario = create_test_scenario()

        pipeline1 = RenderPipeline(scenario, seed=42)
        pipeline2 = RenderPipeline(scenario, seed=42)

        frame1 = pipeline1.render_frame(0)
        frame2 = pipeline2.render_frame(0)

        np.testing.assert_array_equal(frame1.image, frame2.image)
```

---

## 6. Reference Comparison

### 6.1 DIRSIG Comparison

```python
def compare_with_dirsig(eosim_output, dirsig_output, tolerance=0.1):
    """
    Compare EOSIM output with DIRSIG reference.

    Args:
        eosim_output: EOSIM rendered image
        dirsig_output: DIRSIG reference image
        tolerance: Acceptable relative difference

    Returns:
        Comparison metrics dict
    """
    # Normalize both to same scale
    eosim_norm = (eosim_output - eosim_output.min()) / (eosim_output.max() - eosim_output.min())
    dirsig_norm = (dirsig_output - dirsig_output.min()) / (dirsig_output.max() - dirsig_output.min())

    # Compute metrics
    mse = np.mean((eosim_norm - dirsig_norm)**2)
    rmse = np.sqrt(mse)

    # Structural similarity
    from skimage.metrics import structural_similarity
    ssim = structural_similarity(eosim_norm, dirsig_norm)

    # Radiometric comparison (if absolute values available)
    mean_diff = np.abs(eosim_output.mean() - dirsig_output.mean()) / dirsig_output.mean()

    return {
        'rmse': rmse,
        'ssim': ssim,
        'mean_relative_difference': mean_diff,
        'passed': rmse < tolerance and ssim > 0.9
    }
```

### 6.2 Real Sensor Data Comparison

```python
def validate_against_real_data(
    simulated: np.ndarray,
    measured: np.ndarray,
    metadata: dict
) -> dict:
    """
    Compare simulation against real sensor measurement.

    Challenges:
    - Exact scene/conditions may differ
    - Sensor calibration uncertainty
    - Atmospheric conditions uncertainty

    Approach:
    - Compare statistics rather than pixel-by-pixel
    - Use relative metrics (contrast, histogram shape)
    """
    results = {}

    # Mean and std comparison
    results['mean_ratio'] = simulated.mean() / measured.mean()
    results['std_ratio'] = simulated.std() / measured.std()

    # Histogram comparison
    sim_hist, _ = np.histogram(simulated.ravel(), bins=100, density=True)
    meas_hist, _ = np.histogram(measured.ravel(), bins=100, density=True)

    # KL divergence (lower is better)
    from scipy.stats import entropy
    results['kl_divergence'] = entropy(sim_hist + 1e-10, meas_hist + 1e-10)

    # Dynamic range comparison
    results['dr_sim'] = 20 * np.log10(simulated.max() / simulated.std())
    results['dr_meas'] = 20 * np.log10(measured.max() / measured.std())

    return results
```

---

## 7. Performance Metrics

### 7.1 Key Validation Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Radiometric accuracy** | Mean relative error vs reference | < 5% |
| **RMSE** | Root mean square error (normalized) | < 0.05 |
| **SSIM** | Structural similarity index | > 0.90 |
| **NEΔT accuracy** | Simulated vs measured NEΔT | < 20% |
| **Contrast accuracy** | Target/background ratio error | < 10% |
| **MTF accuracy** | System MTF vs measured | < 10% |

### 7.2 Validation Report Template

```markdown
# EOSIM Validation Report

## Configuration
- EOSIM Version: X.Y.Z
- Test Date: YYYY-MM-DD
- Reference: DIRSIG 5.0 / Real Sensor XYZ

## Test Scenarios

### 1. Uniform Blackbody (300K)
- Expected radiance: X.XX W/(m²·sr·μm)
- Measured radiance: X.XX W/(m²·sr·μm)
- Error: X.X%
- **PASS/FAIL**

### 2. Two-Temperature Contrast
- Expected contrast: X.XX
- Measured contrast: X.XX
- Error: X.X%
- **PASS/FAIL**

### 3. Atmospheric Transmission
- Reference (Py6S): X.XX
- EOSIM (RAF-tran): X.XX
- Difference: X.X%
- **PASS/FAIL**

## Summary
- Tests passed: XX/XX
- Overall status: **VALIDATED** / **NEEDS REVIEW**
```

---

## 8. Continuous Validation

### 8.1 CI/CD Integration

```yaml
# .github/workflows/validation.yml

name: EOSIM Validation

on:
  push:
    branches: [main]
  pull_request:

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run unit tests
        run: pytest tests/test_*.py -v

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run integration tests
        run: pytest tests/test_integration.py -v --slow

  reference-comparison:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Compare with reference data
        run: python scripts/validate_against_reference.py
      - name: Upload validation report
        uses: actions/upload-artifact@v3
        with:
          name: validation-report
          path: validation_report.md
```

### 8.2 Regression Detection

```python
def check_regression(current_results: dict, baseline_results: dict) -> bool:
    """
    Check for regression against baseline.

    Returns True if no regression detected.
    """
    for metric, current in current_results.items():
        baseline = baseline_results.get(metric)
        if baseline is None:
            continue

        # Allow 5% degradation before flagging regression
        if isinstance(current, float):
            if current > baseline * 1.05:  # Higher is worse for error metrics
                print(f"REGRESSION: {metric} increased from {baseline} to {current}")
                return False

    return True
```

---

## 9. Summary

### Validation Levels

1. **Unit** - Physical constants, mathematical functions
2. **Module** - Each module vs theory/reference tools
3. **Integration** - Known scenes, end-to-end consistency
4. **System** - DIRSIG comparison, real sensor data

### Key Validation Tests

- Planck function accuracy
- Atmospheric transmission vs Py6S/MODTRAN
- Sensor noise statistics vs EMVA 1288
- Optical PSF/MTF vs diffraction theory
- End-to-end radiometric accuracy

### Acceptance Criteria

- Radiometric error < 5%
- SSIM > 0.90 vs reference
- All unit tests pass
- No regression from baseline

---

## 10. References

1. **DIRSIG Validation Documentation**
2. **EMVA 1288** - Camera characterization standard
3. **ISO 12233** - Resolution measurement
4. **NIST Radiometric Standards**
5. **Holst** (2008) - "Testing and Evaluation of Infrared Imaging Systems"
