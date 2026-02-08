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
- **Automatic Gain Control**: 8 AGC modes (histogram EQ, CLAHE, DDE, ICE, logarithmic, etc.) with polarity palettes (white-hot, black-hot, ironbow, rainbow)
- **Gimbal Simulation**: 2-axis servo dynamics with PID control, scan patterns (raster, spiral, rosette, sector), target tracking, LOS jitter
- **HUD Symbology**: MIL-STD overlays with targeting reticles, track gates, compass heading tape, pitch ladder, status displays

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

## Automatic Gain Control (AGC)

Convert raw sensor data to display-ready imagery with multiple AGC algorithms:

```python
from eosim.agc import AGCProcessor, AGCParameters, AGCMode, PolarityMapper, Polarity

# Apply CLAHE for local contrast enhancement
params = AGCParameters(mode=AGCMode.CLAHE, clahe_clip_limit=3.0, output_bits=8)
agc = AGCProcessor(params)
display = agc.process(raw_14bit_image)

# Apply ironbow color palette
mapper = PolarityMapper(Polarity.IRONBOW)
rgb = mapper.apply(display)
```

Available AGC modes: `LINEAR`, `HISTOGRAM_EQ`, `CLAHE`, `PLATEAU`, `DDE`, `ICE`, `LOGARITHMIC`, `MANUAL`

Available palettes: `WHITE_HOT`, `BLACK_HOT`, `IRONBOW`, `RAINBOW`, `LAVA`, `ARCTIC`, `ISOTHERM`, `SEPIA`

## Gimbal Simulation

Simulate 2-axis gimbal servo dynamics with PID control, scan patterns, and target tracking:

```python
from eosim.gimbal import create_flir_turret, ScanPatternGenerator, ScanPattern, TargetTracker

# Create a FLIR turret gimbal
gimbal = create_flir_turret()
gimbal.command_position(30.0, 15.0)  # Az, El in degrees
states = gimbal.simulate(5.0, dt=0.01)

# Generate a raster scan pattern
gen = ScanPatternGenerator()
az, el, t = gen.generate(ScanPattern.RASTER, duration_s=10.0, sample_rate_hz=100,
                         fov_width_deg=10.0, fov_height_deg=8.0)

# Track a moving target
tracker = TargetTracker(gimbal)
tracker.track(target_az_deg=45.0, target_el_deg=10.0, dt=0.01)
```

Scan patterns: `RASTER`, `SPIRAL`, `ROSETTE`, `SECTOR`, `STARE`

## HUD Symbology

Render MIL-STD-style HUD overlays onto sensor imagery:

```python
from eosim.symbology import SymbologyRenderer, PlatformState, SensorStatus, TrackInfo, ThreatLevel

renderer = SymbologyRenderer()
platform = PlatformState(heading_deg=270, pitch_deg=5, altitude_m=5000, airspeed_mps=200)
sensor = SensorStatus(mode="WHOT", fov_deg=3.0, range_m=4500)
tracks = [
    TrackInfo(x=300, y=200, track_id=1, threat=ThreatLevel.HOSTILE, range_m=3500),
]
overlay = renderer.render(sensor_image, platform=platform, sensor=sensor, tracks=tracks)
```

Reticle types: `CROSSHAIR`, `PIPPER`, `CCIP`, `CCRP`, `DIAMOND`, `CIRCLE`

## Example Scenarios

EOSIM includes 15+ example scenarios:

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
- `eosim.agc` - Automatic gain control and display palettes
- `eosim.gimbal` - Gimbal servo dynamics and scan patterns
- `eosim.symbology` - HUD symbology overlays
- `eosim.studio` - Video-production GUI with sensor physics
- `eosim.output` - File I/O (NumPy, PNG, TIFF, ENVI)
- `eosim.examples` - Example scenarios

## Sensor Types

- **LWIR** (8-12 μm): HgCdTe, Microbolometer
- **MWIR** (3-5 μm): HgCdTe, InSb
- **SWIR** (0.9-1.7 μm): InGaAs
- **Visible** (0.4-0.7 μm): Si CCD/CMOS

## License

See LICENSE file for details.
