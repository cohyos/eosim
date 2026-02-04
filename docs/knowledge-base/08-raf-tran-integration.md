# RAF-tran Integration Analysis for EOSIM

## Executive Summary

**RAF-tran** (https://github.com/cohyos/RAF-tran) is an in-house Python atmospheric radiative transfer library that should serve as the **primary atmosphere backend** for EOSIM. This provides significant advantages over external dependencies like Py6S.

---

## RAF-tran Capabilities

### Core Features

| Capability | Description | EOSIM Relevance |
|------------|-------------|-----------------|
| **Atmosphere Profiles** | US Standard 1976, MODTRAN variants | ✅ Direct use for atmospheric state |
| **Gas Absorption** | Correlated-k distribution | ✅ Required for spectral transmission |
| **Rayleigh Scattering** | Wavelength-dependent molecular | ✅ VIS/NIR atmospheric effects |
| **Mie Scattering** | Full theory for aerosols | ✅ Aerosol/haze modeling |
| **RTE Solvers** | Two-stream, discrete ordinates | ✅ Path radiance computation |
| **Spectral Range** | Shortwave (solar) + Longwave (thermal) | ✅ Covers VIS through LWIR |

### Technical Alignment

```
RAF-tran Architecture          EOSIM Atmosphere Module Needs
─────────────────────────────  ─────────────────────────────
atmosphere/                 →  Standard atmosphere profiles
gas_optics/                 →  Spectral transmission τ(λ)
scattering/                 →  Path radiance L_path
rte_solver/                 →  Multi-stream radiative transfer
utils/                      →  Physical constants, spectral utils
```

---

## Integration Strategy

### Option A: Direct Integration (Recommended)

Embed RAF-tran as the **native atmosphere engine** within EOSIM:

```
eosim/
├── src/eosim/
│   ├── atmosphere/
│   │   ├── base.py              # Abstract interface
│   │   ├── raf_tran_backend.py  # Primary: RAF-tran wrapper
│   │   ├── simple.py            # Fallback: Beer-Lambert
│   │   └── py6s_backend.py      # Optional: External validation
│   └── ...
└── vendor/
    └── raf_tran/                # Vendored or submodule
```

**Advantages**:
- Full control over atmospheric calculations
- No external service dependencies
- Consistent Python ecosystem
- JAX GPU acceleration available
- Can extend/customize for EO/IR specific needs

### Option B: Submodule Integration

Add RAF-tran as a git submodule:

```bash
git submodule add https://github.com/cohyos/RAF-tran vendor/raf_tran
```

### Option C: Package Dependency

If RAF-tran is published to PyPI:

```toml
[project]
dependencies = [
    "raf-tran>=1.0",
]
```

---

## Required Extensions for EOSIM

RAF-tran provides the foundation, but EOSIM may need these extensions:

### 1. Slant Path Geometry

RAF-tran likely assumes vertical columns. EOSIM needs arbitrary sensor-to-target paths:

```python
# Extension needed
def compute_slant_path(
    sensor_altitude_m: float,
    target_altitude_m: float,
    zenith_angle_deg: float,
    azimuth_angle_deg: float
) -> AtmosphereResult:
    """
    Integrate atmospheric effects along arbitrary slant path.

    Returns:
        transmission: τ(λ) along path
        path_radiance: L_path(λ) integrated along path
        sky_radiance: L_sky(λ) downwelling at target
    """
```

### 2. Band-Integrated Quantities

EOSIM sensors often work in broad bands, not monochromatic:

```python
# Extension needed
def band_integrate(
    spectral_quantity: np.ndarray,  # e.g., τ(λ)
    wavelengths_um: np.ndarray,
    sensor_response: np.ndarray     # Relative spectral response
) -> float:
    """Integrate spectral quantity over sensor band."""
```

### 3. Thermal Emission (MWIR/LWIR)

Ensure RAF-tran's longwave capabilities include:
- Atmospheric self-emission along path
- Downwelling sky radiance for target reflection
- Temperature-dependent emission profiles

### 4. Lookup Table Generation

For real-time performance, pre-compute LUTs:

```python
# Extension needed
def generate_atmosphere_lut(
    atmosphere_profile: str,
    wavelength_range_um: tuple,
    altitude_range_m: tuple,
    zenith_range_deg: tuple
) -> xr.Dataset:
    """
    Pre-compute transmission/radiance LUT.

    Dimensions: (wavelength, altitude, zenith)
    Variables: transmission, path_radiance, sky_radiance
    """
```

---

## EOSIM Atmosphere Module Design with RAF-tran

```python
# src/eosim/atmosphere/raf_tran_backend.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

# Import RAF-tran components
from raf_tran.atmosphere import StandardAtmosphere
from raf_tran.gas_optics import CKDGasOptics
from raf_tran.scattering import RayleighScattering, MieScattering
from raf_tran.rte_solver import TwoStreamSolver, DiscreteOrdinatesSolver


@dataclass
class PathGeometry:
    """Sensor-to-target atmospheric path."""
    sensor_altitude_m: float
    target_altitude_m: float
    zenith_angle_deg: float
    path_length_m: float  # Computed from above


@dataclass
class AtmosphereResult:
    """Atmospheric transmission and radiance."""
    wavelengths_um: np.ndarray
    transmission: np.ndarray      # τ(λ), dimensionless
    path_radiance: np.ndarray     # L_path(λ), W/m²/sr/μm
    sky_radiance: np.ndarray      # L_sky(λ), W/m²/sr/μm


class RAFTranAtmosphere:
    """
    EOSIM atmosphere module using RAF-tran backend.

    Provides transmission and path radiance for arbitrary
    sensor-target geometries across VIS-LWIR spectrum.
    """

    def __init__(
        self,
        profile: str = "us_standard_1976",
        aerosol_model: str = "rural",
        visibility_km: float = 23.0,
        use_gpu: bool = False
    ):
        self.atmosphere = StandardAtmosphere(profile)
        self.gas_optics = CKDGasOptics()
        self.rayleigh = RayleighScattering()
        self.mie = MieScattering(model=aerosol_model, visibility=visibility_km)

        # Select solver based on accuracy needs
        self.solver = TwoStreamSolver()  # Fast
        # self.solver = DiscreteOrdinatesSolver(n_streams=8)  # Accurate

        self._use_gpu = use_gpu

    def query(
        self,
        wavelengths_um: np.ndarray,
        path: PathGeometry
    ) -> AtmosphereResult:
        """
        Compute atmospheric effects for given path geometry.

        Args:
            wavelengths_um: Wavelength array in micrometers
            path: Sensor-to-target path geometry

        Returns:
            AtmosphereResult with transmission and radiances
        """
        # 1. Get atmospheric profile along path
        profile = self._interpolate_path(path)

        # 2. Compute gas absorption optical depth
        tau_gas = self.gas_optics.optical_depth(
            wavelengths_um, profile
        )

        # 3. Compute scattering optical depths
        tau_rayleigh = self.rayleigh.optical_depth(
            wavelengths_um, profile
        )
        tau_mie = self.mie.optical_depth(
            wavelengths_um, profile
        )

        # 4. Total optical depth
        tau_total = tau_gas + tau_rayleigh + tau_mie

        # 5. Solve radiative transfer for path radiance
        path_radiance, sky_radiance = self.solver.solve(
            wavelengths_um,
            tau_total,
            profile,
            path.zenith_angle_deg
        )

        # 6. Transmission
        transmission = np.exp(-tau_total)

        return AtmosphereResult(
            wavelengths_um=wavelengths_um,
            transmission=transmission,
            path_radiance=path_radiance,
            sky_radiance=sky_radiance
        )

    def _interpolate_path(self, path: PathGeometry):
        """Interpolate atmosphere profile along slant path."""
        # Implementation details...
        pass


# Factory function for multi-fidelity selection
def create_atmosphere(
    backend: str = "raf_tran",
    fidelity: str = "medium",
    **kwargs
) -> AtmosphereModel:
    """
    Create atmosphere model with specified backend and fidelity.

    Args:
        backend: "raf_tran", "py6s", "simple"
        fidelity: "simple", "medium", "high"
    """
    if backend == "raf_tran":
        if fidelity == "simple":
            return RAFTranAtmosphere(solver="two_stream", **kwargs)
        elif fidelity == "medium":
            return RAFTranAtmosphere(solver="dom_4", **kwargs)
        else:  # high
            return RAFTranAtmosphere(solver="dom_16", **kwargs)
    elif backend == "py6s":
        return Py6SAtmosphere(**kwargs)
    else:
        return SimpleAtmosphere(**kwargs)
```

---

## Validation Strategy

### Cross-Validation with Py6S/MODTRAN

```python
def validate_raf_tran():
    """Compare RAF-tran outputs against Py6S reference."""

    # Test case: Nadir view, sea level to 10km
    wavelengths = np.linspace(0.4, 14.0, 1000)  # VIS to LWIR

    # RAF-tran result
    raf = RAFTranAtmosphere(profile="midlatitude_summer")
    raf_result = raf.query(wavelengths, PathGeometry(...))

    # Py6S reference (where applicable, VIS-SWIR only)
    py6s_result = run_py6s_reference(wavelengths[:500], ...)

    # Compare transmission in overlapping bands
    assert np.allclose(
        raf_result.transmission[:500],
        py6s_result.transmission,
        rtol=0.05  # 5% tolerance
    )
```

---

## Benefits Summary

| Aspect | With RAF-tran | Without (Py6S only) |
|--------|---------------|---------------------|
| **Spectral Range** | VIS → LWIR (0.4-14+ μm) | VIS → SWIR only |
| **Thermal Emission** | ✅ Native support | ❌ Not supported |
| **GPU Acceleration** | ✅ JAX backend | ❌ Not available |
| **Customization** | ✅ Full source access | ❌ Black box |
| **Dependencies** | Python-native | Requires Fortran 6S |
| **Ownership** | In-house control | External dependency |

---

## Recommendation

**Use RAF-tran as the primary atmosphere backend for EOSIM.**

1. Add as git submodule or vendor the code
2. Create `RAFTranAtmosphere` wrapper class implementing EOSIM interface
3. Extend with slant-path geometry for arbitrary sensor views
4. Add LUT generation for real-time performance
5. Keep Py6S as optional validation reference

This approach gives EOSIM a **complete, VIS-to-LWIR atmospheric capability** with full control and no external Fortran dependencies.

---

## Implementation Priority

1. **Immediate**: Add RAF-tran as submodule
2. **Phase 1**: Create basic wrapper with transmission query
3. **Phase 2**: Add slant-path geometry support
4. **Phase 3**: Implement LUT generation for performance
5. **Phase 4**: Enable JAX GPU acceleration path
