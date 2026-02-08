# EOSIM - Electro-Optical/Infrared Simulation Framework

A PC-based, open-source electro-optical/infrared simulation framework that generates synthetic image/video output from configurable FPA-based passive sensor models across visible and infrared bands.

## Features

- **Multi-spectral Support**: Visible, SWIR, MWIR, and LWIR bands
- **Realistic Sensor Models**: Complete signal chain from photons to digital numbers
- **Optical Effects**: PSF, MTF, aberrations, blur
- **Noise Models**: Shot noise, read noise, dark current, PRNU, DSNU
- **Atmosphere**: Transmission and path radiance effects
- **Post-processing**: Motion blur, jitter, vignetting, blooming
- **EOSIM Studio**: Video-production-style GUI for creating sensor simulation videos
- **3D Object Library**: 50+ military/civilian objects with thermal signatures
- **Enhanced Thermal Shapes**: Realistic thermal zones for vehicles, people, aircraft

## EOSIM Studio - Video Creator Interface

EOSIM Studio provides a video-production-style interface for sensor simulation. Think of it as a virtual camera system:

| Video Concept | EOSIM Implementation |
|---------------|----------------------|
| Camera | Sensor with lens presets |
| Scene | 3D world with objects |
| Actors | Targets (vehicles, aircraft, people) |
| Timeline | Scrub, play, keyframe animation |
| Render | Export to MP4/AVI or images |

### Launch Studio

```python
from eosim.studio import launch
launch()
```

### Studio Features

- **Scene View**: 3D overhead view with objects and camera FOV cone
- **Camera View**: Live preview of thermal imagery
- **Timeline**: Video-editor style with play/pause, scrub, keyframes
- **Camera Presets**: Surveillance, Targeting, Wide Area, Night Vision
- **Object Library**: Drag-and-drop objects into scene
- **Video Export**: MP4, AVI, or PNG sequence

### Keyboard Shortcuts

- `Space` - Play/Pause
- `←/→` - Step frame backward/forward
- `Home/End` - Go to start/end

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

## 3D Object Library

Browse and preview 50+ objects with the 3D Object Viewer:

```python
from eosim.library.object_viewer import launch_object_viewer
launch_object_viewer()
```

Or on Windows: run `run_browser.bat`

### Object Categories

| Category | Examples |
|----------|----------|
| Aircraft | F-16, F-22, Su-27, Apache, Predator UAV |
| Vehicles | M1 Abrams, T-90, Humvee, Pickup Truck |
| Ships | Destroyer, Aircraft Carrier, Frigate |
| People | Soldier (standing/prone), Civilian |
| Missiles | AIM-120, S-400 Launcher |

Each object includes:
- Accurate dimensions (SISO-REF-010 compliant)
- Thermal signatures with hot spots
- 3D mesh for visualization
- Metadata (source, license)

See [3D Models Guide](docs/3D_MODELS_GUIDE.md) for details.

## Enhanced Thermal Shapes

Realistic thermal zone modeling for targets:

```python
from eosim.targets import VehicleTarget, AircraftTarget

# Vehicle with thermal zones
vehicle = VehicleTarget.sedan(engine_state='running')
sig = vehicle.get_signature(use_enhanced_shape=True)
# Engine: 370K, Cabin: 320K, Wheels: 340K, Exhaust: 400K

# Aircraft with afterburner exhaust plume
jet = AircraftTarget.fighter_jet(afterburner=True)
sig = jet.get_signature(use_enhanced_shape=True, aspect_angle_deg=180)
# Exhaust plume gradient: 500K at nozzle → 300K
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

Optional dependencies:
```bash
pip install opencv-python  # For video export
pip install trimesh        # For advanced 3D mesh operations
```

## Modules

- `eosim.core` - Physical constants, spectral utilities
- `eosim.sensor` - FPA geometry, noise models, ADC, metrics
- `eosim.optics` - PSF, MTF, Zernike aberrations
- `eosim.radiance` - Planck functions, surface radiance
- `eosim.atmosphere` - Atmospheric transmission
- `eosim.pipeline` - Simulation engine and effects
- `eosim.library` - 3D Object Library & Viewer
- `eosim.targets` - Target models with thermal signatures
- `eosim.studio` - **NEW**: Video-production GUI
- `eosim.output` - File I/O (NumPy, PNG, TIFF, ENVI)
- `eosim.examples` - Example scenarios

## Sensor Types

- **LWIR** (8-12 μm): HgCdTe, Microbolometer
- **MWIR** (3-5 μm): HgCdTe, InSb
- **SWIR** (0.9-1.7 μm): InGaAs
- **Visible** (0.4-0.7 μm): Si CCD/CMOS

## License

See LICENSE file for details.
