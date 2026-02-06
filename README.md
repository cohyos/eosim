# EOSIM - Electro-Optical/Infrared Simulation Framework

A PC-based, open-source electro-optical/infrared simulation framework that generates synthetic image/video output from configurable FPA-based passive sensor models across visible and infrared bands.

## Features

- **Multi-spectral Support**: Visible, SWIR, MWIR, and LWIR bands
- **Realistic Sensor Models**: Complete signal chain from photons to digital numbers
- **Optical Effects**: PSF, MTF, aberrations, blur
- **Noise Models**: Shot noise, read noise, dark current, PRNU, DSNU
- **Atmosphere**: Transmission and path radiance effects
- **Post-processing**: Motion blur, jitter, vignetting, blooming

## Quick Start

```python
from eosim.examples import example_vehicle_on_road

# Run a pre-configured example
result = example_vehicle_on_road()
print(f"Generated {result.sensor_type} image")

# Or create custom simulations
from eosim.pipeline import quick_simulation
import numpy as np

# Create temperature scene
temps = 300 + 10 * np.random.randn(480, 640)

# Generate thermal image
image = quick_simulation(temps, sensor_type="lwir")
```

## Example Scenarios

EOSIM includes 10 comprehensive example scenarios:

1. **Vehicle on Road** (LWIR) - Hot vehicle on cool road background
2. **Person in Forest** (MWIR) - Human target with vegetation clutter
3. **Aircraft Against Sky** (MWIR) - Fast-moving aircraft with motion blur
4. **Ship on Ocean** (LWIR) - Maritime surveillance scenario
5. **Building Thermal** (LWIR) - Structural thermal inspection
6. **Wildlife Tracking** (MWIR) - Animal detection in natural habitat
7. **Industrial Monitoring** (LWIR) - Hot machinery detection
8. **Night Vision** (SWIR) - Low-light surveillance
9. **Solar Panel Inspection** (LWIR) - Defect detection
10. **Urban Surveillance** (Visible) - Multi-spectral urban scene

## Installation

```bash
pip install -e .
```

## Modules

- `eosim.core` - Physical constants, spectral utilities
- `eosim.sensor` - FPA geometry, noise models, ADC, metrics
- `eosim.optics` - PSF, MTF, Zernike aberrations
- `eosim.radiance` - Planck functions, surface radiance
- `eosim.atmosphere` - Atmospheric transmission
- `eosim.pipeline` - Simulation engine and effects
- `eosim.library` - **NEW**: 3D Object Library & Viewer ([Guide](docs/3D_MODELS_GUIDE.md))
- `eosim.output` - File I/O (NumPy, PNG, TIFF, ENVI)
- `eosim.examples` - Example scenarios

## Sensor Types

- **LWIR** (8-12 μm): HgCdTe, Microbolometer
- **MWIR** (3-5 μm): HgCdTe, InSb
- **SWIR** (0.9-1.7 μm): InGaAs
- **Visible** (0.4-0.7 μm): Si CCD/CMOS

## License

See LICENSE file for details.
