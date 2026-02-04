# EOSIM User Interface and Workflow Design

## Overview

This document specifies how users interact with EOSIM: configuration, simulation setup, execution, and output visualization. The design supports three usage modes:

1. **YAML Configuration Files** - Declarative scenario definition
2. **Python API** - Programmatic control for advanced users
3. **CLI Interface** - Command-line execution and batch processing

Future consideration: Web-based GUI for interactive setup.

---

## 1. Simulation Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EOSIM SIMULATION WORKFLOW                         │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │   SCENARIO   │────▶│   VALIDATE   │────▶│   EXECUTE    │
    │   DEFINITION │     │   & PREVIEW  │     │   SIMULATION │
    └──────────────┘     └──────────────┘     └──────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │ • Arena/Scene│     │ • 3D Preview │     │ • Render     │
    │ • Sensor     │     │ • Path Check │     │ • Apply FX   │
    │ • Platform   │     │ • Validate   │     │ • Export     │
    │ • Atmosphere │     │   Config     │     │              │
    │ • Time/Date  │     │              │     │              │
    └──────────────┘     └──────────────┘     └──────────────┘
                                                    │
                                                    ▼
                                            ┌──────────────┐
                                            │   OUTPUT     │
                                            │ • Frames     │
                                            │ • Video      │
                                            │ • Metadata   │
                                            │ • Diagnostics│
                                            └──────────────┘
```

---

## 2. Configuration File Structure

### 2.1 Master Scenario File (`scenario.yaml`)

```yaml
# scenario.yaml - Master simulation scenario definition
# EOSIM Electro-Optical Simulation Scenario

metadata:
  name: "Urban Surveillance MWIR"
  description: "MWIR sensor on UAV surveilling urban area"
  author: "Operator Name"
  created: "2024-01-15"
  version: "1.0"

# Time and date settings
time:
  start: "2024-06-15T14:30:00Z"      # ISO 8601 UTC
  duration_seconds: 120               # Simulation duration
  time_step_seconds: 0.033            # ~30 fps output

# Reference to sub-configuration files (modular)
includes:
  arena: "arena/urban_city.yaml"
  sensor: "sensors/mwir_640x512.yaml"
  platform: "platforms/uav_fixed_wing.yaml"
  atmosphere: "atmosphere/summer_hazy.yaml"

# Or inline definitions (for simple scenarios)
# arena:
#   terrain: ...
#   targets: ...

# Output configuration
output:
  directory: "./output/urban_mwir_001"
  formats:
    - type: "video"
      codec: "h264"
      filename: "simulation.mp4"
      fps: 30
    - type: "frames"
      format: "tiff"
      bit_depth: 16
      prefix: "frame_"
    - type: "metadata"
      format: "json"
      include_per_frame: true

# Simulation quality settings
quality:
  fidelity: "medium"                  # simple | medium | high
  atmosphere_model: "raf_tran"        # raf_tran | simple | lut
  render_samples: 64                  # For ray tracing
  enable_noise: true
  enable_motion_blur: false

# Diagnostics and intermediate outputs
diagnostics:
  enabled: true
  save_radiance_maps: false           # Pre-atmosphere
  save_at_sensor_radiance: true       # Post-atmosphere
  save_pre_noise: true                # Before sensor noise
  export_sensor_metadata: true
```

---

### 2.2 Arena/Scene Configuration (`arena/*.yaml`)

```yaml
# arena/urban_city.yaml - Scene/Environment Definition

arena:
  name: "Urban City Block"

  # Coordinate system
  coordinate_system:
    type: "local_enu"                 # East-North-Up local tangent
    origin:
      latitude: 32.0853
      longitude: 34.7818
      altitude_m: 0.0

  # Terrain definition
  terrain:
    type: "dem"                       # dem | flat | procedural
    source: "data/terrain/tel_aviv_dem.tif"
    texture: "data/terrain/tel_aviv_ortho.tif"
    material:
      type: "mixed_urban"
      emissivity: 0.92
      thermal_inertia: 1200           # J/(m²·K·s^0.5)

  # Static objects (buildings, infrastructure)
  static_objects:
    - name: "building_001"
      geometry: "data/models/office_building.obj"
      position: [100.0, 50.0, 0.0]    # ENU meters
      rotation: [0, 0, 45]            # Euler angles (degrees)
      scale: 1.0
      material:
        type: "concrete"
        emissivity: 0.90
        temperature_offset_K: 5.0     # Above ambient

    - name: "building_002"
      geometry: "data/models/residential.obj"
      position: [200.0, 100.0, 0.0]
      rotation: [0, 0, 0]
      material:
        type: "brick"
        emissivity: 0.93

  # Dynamic targets (vehicles, people)
  targets:
    - name: "vehicle_001"
      type: "ground_vehicle"
      geometry: "data/models/sedan.obj"
      material:
        body:
          type: "metal_painted"
          emissivity: 0.85
        engine:
          type: "hot_metal"
          temperature_K: 380          # Running engine
        exhaust:
          type: "hot_gas"
          temperature_K: 450
      trajectory:
        type: "waypoints"
        waypoints:
          - time: 0.0
            position: [50.0, 20.0, 0.0]
            velocity: [5.0, 0.0, 0.0]  # m/s
          - time: 10.0
            position: [100.0, 20.0, 0.0]
            velocity: [5.0, 0.0, 0.0]
          - time: 20.0
            position: [100.0, 70.0, 0.0]
            velocity: [0.0, 5.0, 0.0]
        interpolation: "cubic_spline"

    - name: "person_001"
      type: "human"
      geometry: "data/models/human_standing.obj"
      material:
        skin:
          emissivity: 0.98
          temperature_K: 306          # ~33°C skin
        clothing:
          emissivity: 0.95
          temperature_K: 300
      trajectory:
        type: "stationary"
        position: [80.0, 40.0, 0.0]
        orientation: [0, 0, 90]

  # Environment lighting
  sun:
    mode: "ephemeris"                 # Computed from time/location
    # Or override:
    # azimuth_deg: 180
    # elevation_deg: 45

  # Sky/background
  sky:
    type: "clear"                     # clear | cloudy | overcast
    cloud_cover: 0.1                  # 0-1 fraction

  # Ambient conditions
  ambient:
    temperature_K: 303                # ~30°C
    wind_speed_mps: 3.0
    wind_direction_deg: 270           # From west
    relative_humidity: 0.60
```

---

### 2.3 Sensor Configuration (`sensors/*.yaml`)

```yaml
# sensors/mwir_640x512.yaml - Sensor/Camera Definition

sensor:
  name: "MWIR Cooled InSb"
  type: "thermal_infrared"

  # Spectral band
  spectral:
    band: "MWIR"
    wavelength_min_um: 3.0
    wavelength_max_um: 5.0
    spectral_response: "data/sensors/insb_qe.csv"  # Optional measured QE

  # Optics
  optics:
    focal_length_mm: 100
    f_number: 2.0
    transmission: 0.85                # Optical transmission

    # PSF/blur model
    psf:
      type: "gaussian"                # gaussian | airy | measured
      fwhm_pixels: 1.2
      # Or for measured PSF:
      # type: "measured"
      # file: "data/sensors/psf_measured.npy"

    # Distortion model (optional)
    distortion:
      type: "brown_conrady"
      k1: -0.1
      k2: 0.01
      p1: 0.0
      p2: 0.0

  # Detector/FPA
  detector:
    type: "InSb"
    resolution: [640, 512]            # Width x Height pixels
    pixel_pitch_um: 15.0

    # Quantum efficiency
    quantum_efficiency: 0.70          # At peak wavelength

    # Integration
    integration_time_ms: 10.0
    frame_rate_hz: 30

    # Well capacity and gain
    full_well_capacity_e: 2.0e6
    gain_e_per_dn: 100

    # Noise parameters
    noise:
      read_noise_e: 50                # Electrons RMS
      dark_current_e_per_s: 1.0e4     # At operating temp
      operating_temperature_K: 77     # Cooled detector

      # Fixed pattern noise
      prnu_percent: 1.0               # Photo-response non-uniformity
      dsnu_electrons: 20              # Dark signal non-uniformity

    # ADC
    adc:
      bit_depth: 14
      offset_dn: 100

  # Performance metrics (for validation)
  specified_performance:
    nedt_mk: 25                       # Noise equivalent delta T
    mtf_at_nyquist: 0.3
```

---

### 2.4 Platform Configuration (`platforms/*.yaml`)

```yaml
# platforms/uav_fixed_wing.yaml - Sensor Platform Definition

platform:
  name: "Fixed Wing UAV"
  type: "aerial"

  # Platform dynamics
  dynamics:
    type: "6dof"                      # 6dof | 3dof | static

    # Motion source
    motion:
      type: "trajectory_file"         # trajectory_file | waypoints | orbit | manual

      # For trajectory file (e.g., from flight planner)
      file: "data/trajectories/uav_mission_001.csv"
      format: "csv"
      columns:
        time: "time_s"
        x: "pos_east_m"
        y: "pos_north_m"
        z: "pos_up_m"
        roll: "roll_deg"
        pitch: "pitch_deg"
        yaw: "yaw_deg"
      interpolation: "cubic"

      # Or for simple waypoint definition:
      # type: "waypoints"
      # waypoints:
      #   - time: 0.0
      #     position: [0, 0, 500]       # ENU meters
      #     attitude: [0, -10, 90]      # Roll, pitch, yaw degrees
      #     velocity: [50, 0, 0]        # m/s

      # Or for circular orbit:
      # type: "orbit"
      # center: [100, 100, 0]           # Ground point ENU
      # radius_m: 500
      # altitude_m: 1000
      # angular_rate_deg_s: 5.0
      # start_angle_deg: 0
      # look_at_center: true

  # Sensor gimbal/mount
  gimbal:
    type: "2axis_stabilized"          # fixed | 2axis | 3axis

    # Gimbal pointing
    pointing:
      mode: "track_point"             # fixed | track_point | lead_target | stare

      # For track_point mode
      track_point: [100, 50, 0]       # ENU coordinates to track

      # Or for fixed pointing (relative to platform)
      # mode: "fixed"
      # azimuth_deg: 0                 # Forward
      # elevation_deg: -45             # 45° down

      # Or track a moving target
      # mode: "lead_target"
      # target: "vehicle_001"
      # lead_time_s: 0.5

    # Gimbal limits
    limits:
      azimuth_min_deg: -180
      azimuth_max_deg: 180
      elevation_min_deg: -90
      elevation_max_deg: 30

    # Gimbal dynamics (for realism)
    dynamics:
      max_rate_deg_s: 60
      acceleration_deg_s2: 120

    # Stabilization performance
    stabilization:
      jitter_rms_urad: 50             # Line-of-sight jitter

  # Platform vibration (affects image quality)
  vibration:
    enabled: true
    psd_file: "data/platforms/uav_vibration_psd.csv"
    # Or simple model:
    # rms_angular_mrad: 0.5
    # frequency_hz: 20
```

---

### 2.5 Atmosphere Configuration (`atmosphere/*.yaml`)

```yaml
# atmosphere/summer_hazy.yaml - Atmospheric Conditions

atmosphere:
  name: "Summer Hazy"

  # Atmospheric profile
  profile:
    type: "standard"                  # standard | custom | radiosonde
    model: "midlatitude_summer"       # us_standard_1976 | midlatitude_summer | tropical | etc.

  # Aerosols
  aerosols:
    model: "rural"                    # rural | urban | maritime | desert
    visibility_km: 10.0               # Meteorological visibility

  # Additional gases (optional overrides)
  gases:
    h2o_column_cm: 2.0                # Precipitable water vapor
    o3_column_du: 300                 # Ozone in Dobson units
    co2_ppm: 420                      # CO2 concentration

  # Atmospheric turbulence (for long paths)
  turbulence:
    enabled: true
    cn2_model: "hufnagel_valley"      # Refractive index structure
    # cn2_at_ground: 1.0e-14          # Or specify directly

  # Weather effects
  weather:
    rain: false
    fog: false
    snow: false

  # Backend selection
  backend:
    type: "raf_tran"                  # raf_tran | simple | lut
    solver: "discrete_ordinates"      # two_stream | discrete_ordinates
    n_streams: 8                      # For discrete ordinates
```

---

## 3. Python API Usage

### 3.1 Basic Simulation Setup

```python
#!/usr/bin/env python3
"""Example: Basic EOSIM simulation setup and execution."""

from eosim import Scenario, Arena, Sensor, Platform, Atmosphere
from eosim.pipeline import Pipeline
from eosim.output import VideoWriter, FrameExporter

# Load scenario from YAML
scenario = Scenario.from_yaml("scenarios/urban_mwir.yaml")

# Or build programmatically
scenario = Scenario(
    name="Urban MWIR Demo",
    start_time="2024-06-15T14:30:00Z",
    duration=60.0,  # seconds
    frame_rate=30.0
)

# Configure arena
arena = Arena()
arena.set_terrain(
    dem_file="data/terrain/dem.tif",
    texture_file="data/terrain/ortho.tif"
)
arena.add_target(
    name="vehicle_001",
    geometry="data/models/sedan.obj",
    position=(100, 50, 0),
    temperature=350  # Kelvin
)
arena.set_ambient_temperature(303)  # 30°C
scenario.arena = arena

# Configure sensor
sensor = Sensor.from_preset("mwir_640x512")
# Or customize:
sensor = Sensor(
    band="MWIR",
    resolution=(640, 512),
    focal_length_mm=100,
    f_number=2.0,
    integration_time_ms=10,
    noise_model="realistic"
)
scenario.sensor = sensor

# Configure platform trajectory
platform = Platform(type="aerial")
platform.set_trajectory_from_waypoints([
    {"time": 0, "position": (0, 0, 500), "attitude": (0, -30, 90)},
    {"time": 30, "position": (500, 0, 500), "attitude": (0, -30, 90)},
    {"time": 60, "position": (500, 500, 500), "attitude": (0, -30, 180)},
])
platform.gimbal.track_point((100, 50, 0))
scenario.platform = platform

# Configure atmosphere
atmosphere = Atmosphere(
    profile="midlatitude_summer",
    visibility_km=10,
    backend="raf_tran"
)
scenario.atmosphere = atmosphere

# Validate configuration
errors = scenario.validate()
if errors:
    for err in errors:
        print(f"Config error: {err}")
    exit(1)

# Create pipeline
pipeline = Pipeline(scenario, fidelity="medium")

# Execute simulation
print("Running simulation...")
for frame in pipeline.run():
    print(f"Frame {frame.index}: t={frame.time:.3f}s")

    # Access frame data
    image = frame.image              # numpy array (H, W) uint16
    radiance = frame.radiance_map    # (H, W) float32 W/m²/sr/μm
    metadata = frame.metadata        # dict with sensor state, etc.

# Export results
print("Exporting outputs...")
pipeline.export_video("output/simulation.mp4", codec="h264", fps=30)
pipeline.export_frames("output/frames/", format="tiff", bit_depth=16)
pipeline.export_metadata("output/metadata.json")

print("Done!")
```

### 3.2 Interactive/Real-time Preview

```python
"""Example: Real-time preview with interactive controls."""

from eosim import Scenario
from eosim.preview import InteractiveViewer

# Load scenario
scenario = Scenario.from_yaml("scenarios/urban_mwir.yaml")

# Launch interactive viewer
viewer = InteractiveViewer(scenario)

# Viewer provides:
# - 3D scene preview (wireframe/textured)
# - Platform path visualization
# - Sensor footprint overlay
# - Time scrubbing
# - Real-time parameter adjustment
# - Single frame render preview

viewer.run()  # Blocks until window closed

# After viewer closes, access any modifications
modified_scenario = viewer.get_scenario()
modified_scenario.save("scenarios/urban_mwir_modified.yaml")
```

### 3.3 Batch Processing

```python
"""Example: Batch processing multiple scenarios."""

from eosim import Scenario, Pipeline
from pathlib import Path
import concurrent.futures

def process_scenario(yaml_path: Path) -> dict:
    """Process a single scenario and return summary."""
    scenario = Scenario.from_yaml(yaml_path)
    pipeline = Pipeline(scenario)

    output_dir = Path("output") / yaml_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run and export
    pipeline.run_all()
    pipeline.export_video(output_dir / "video.mp4")

    return {
        "scenario": yaml_path.name,
        "frames": pipeline.frame_count,
        "output": str(output_dir)
    }

# Process all scenarios in directory
scenario_files = list(Path("scenarios/batch/").glob("*.yaml"))

# Parallel processing
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_scenario, scenario_files))

for r in results:
    print(f"Completed: {r['scenario']} -> {r['output']}")
```

---

## 4. Command-Line Interface (CLI)

### 4.1 CLI Commands

```bash
# Validate scenario configuration
eosim validate scenario.yaml

# Preview scenario (opens 3D viewer)
eosim preview scenario.yaml

# Run simulation
eosim run scenario.yaml --output ./output/

# Run with options
eosim run scenario.yaml \
    --output ./output/ \
    --fidelity high \
    --format video \
    --fps 30 \
    --verbose

# Render single frame at specific time
eosim render-frame scenario.yaml --time 10.5 --output frame_10s.tiff

# Export configuration template
eosim template --type full > my_scenario.yaml
eosim template --type sensor > my_sensor.yaml
eosim template --type platform > my_platform.yaml

# List available presets
eosim presets --sensors
eosim presets --atmospheres
eosim presets --platforms

# Convert between formats
eosim convert trajectory.csv --to trajectory.yaml

# Batch processing
eosim batch scenarios/*.yaml --output ./batch_output/ --parallel 4
```

### 4.2 CLI Output Example

```
$ eosim run urban_mwir.yaml --output ./output/ --verbose

EOSIM Electro-Optical Simulation v0.1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Scenario: Urban Surveillance MWIR
  Duration: 120.0s @ 30 fps (3600 frames)
  Sensor: MWIR 640x512 InSb
  Platform: Fixed Wing UAV
  Atmosphere: RAF-tran (midlatitude_summer, vis=10km)

Validating configuration... ✓
Loading scene geometry... ✓
  - Terrain: 2048x2048 DEM loaded
  - Objects: 15 static, 3 dynamic
Initializing atmosphere... ✓
  - RAF-tran backend ready
  - LUT precomputation: 2.3s
Initializing sensor model... ✓
  - NEΔT estimate: 28 mK

Rendering frames:
  [████████████████████████████████████████] 3600/3600 (100%)
  Elapsed: 5m 23s | Rate: 11.2 fps | ETA: 0s

Exporting outputs:
  → output/simulation.mp4 (H.264, 1920x1080, 30fps)
  → output/frames/ (3600 TIFF files, 16-bit)
  → output/metadata.json

Summary:
  Total time: 5m 45s
  Output size: 2.3 GB
  Peak memory: 4.2 GB

Done! Output saved to: ./output/
```

---

## 5. Output Formats and Visualization

### 5.1 Output File Structure

```
output/
├── simulation.mp4              # Rendered video (display-ready)
├── simulation_raw.mp4          # Raw sensor output video
├── frames/
│   ├── frame_00000.tiff        # 16-bit TIFF frames
│   ├── frame_00001.tiff
│   └── ...
├── radiance/                   # Optional: at-sensor radiance maps
│   ├── radiance_00000.npy
│   └── ...
├── metadata.json               # Simulation metadata
├── per_frame_metadata/         # Per-frame detailed metadata
│   ├── frame_00000.json
│   └── ...
├── diagnostics/                # Optional diagnostics
│   ├── atmosphere_transmission.png
│   ├── psf_applied.png
│   └── noise_analysis.png
└── scenario_snapshot.yaml      # Copy of input configuration
```

### 5.2 Metadata Format

```json
{
  "simulation": {
    "name": "Urban Surveillance MWIR",
    "eosim_version": "0.1.0",
    "timestamp": "2024-01-15T10:30:00Z",
    "duration_s": 120.0,
    "frame_count": 3600,
    "frame_rate_hz": 30.0
  },
  "sensor": {
    "name": "MWIR Cooled InSb",
    "band": "MWIR",
    "wavelength_range_um": [3.0, 5.0],
    "resolution": [640, 512],
    "pixel_pitch_um": 15.0,
    "focal_length_mm": 100,
    "fov_deg": [5.5, 4.4],
    "ifov_mrad": 0.15,
    "nedt_mk": 28
  },
  "atmosphere": {
    "model": "midlatitude_summer",
    "visibility_km": 10.0,
    "backend": "raf_tran"
  },
  "frames": [
    {
      "index": 0,
      "time_s": 0.0,
      "platform": {
        "position_enu_m": [0.0, 0.0, 500.0],
        "attitude_deg": [0.0, -30.0, 90.0],
        "velocity_mps": [50.0, 0.0, 0.0]
      },
      "sensor": {
        "los_azimuth_deg": 180.0,
        "los_elevation_deg": -30.0,
        "slant_range_m": 577.35,
        "gsd_m": 0.087
      },
      "scene": {
        "sun_azimuth_deg": 180.0,
        "sun_elevation_deg": 45.0,
        "ambient_temp_K": 303.0
      }
    }
  ]
}
```

### 5.3 Frame Metadata (Per-Frame JSON)

```json
{
  "frame_index": 100,
  "time_s": 3.333,
  "platform_state": {
    "position_enu_m": [166.5, 0.0, 500.0],
    "attitude_rpy_deg": [0.5, -29.8, 90.2],
    "velocity_enu_mps": [49.9, 0.1, 0.0],
    "acceleration_mps2": [0.0, 0.0, 0.0]
  },
  "gimbal_state": {
    "azimuth_deg": -10.5,
    "elevation_deg": -35.2,
    "los_vector_enu": [0.12, -0.58, -0.81]
  },
  "sensor_geometry": {
    "ground_footprint_corners_enu_m": [
      [85.2, 30.1, 0.0],
      [114.8, 30.1, 0.0],
      [114.8, 69.9, 0.0],
      [85.2, 69.9, 0.0]
    ],
    "center_ground_point_enu_m": [100.0, 50.0, 0.0],
    "slant_range_m": 582.4,
    "gsd_m": 0.087
  },
  "radiometry": {
    "mean_radiance_w_m2_sr_um": 2.45,
    "min_radiance": 1.82,
    "max_radiance": 8.91,
    "dynamic_range_db": 36.8
  },
  "targets_in_fov": [
    {
      "name": "vehicle_001",
      "pixel_location": [320, 280],
      "pixel_extent": [45, 22],
      "apparent_temperature_K": 352,
      "contrast_ratio": 1.16
    }
  ]
}
```

### 5.4 Video Output Options

```yaml
# Video export configuration options
output:
  formats:
    - type: "video"
      codec: "h264"                   # h264 | h265 | prores | raw
      filename: "simulation.mp4"
      fps: 30
      resolution: "native"            # native | 1080p | 4k | custom
      # custom_resolution: [1920, 1080]

      # Display mapping (how to visualize thermal data)
      display:
        colormap: "iron"              # iron | grayscale | rainbow | plasma
        temperature_range_K: [280, 350]  # Auto if not specified
        histogram_equalization: false
        gamma: 1.0

      # Overlays (optional)
      overlays:
        timestamp: true
        sensor_info: true
        crosshair: false
        scale_bar: true
        colorbar: true
        target_boxes: false           # Bounding boxes around targets

      # Compression
      quality: "high"                 # low | medium | high | lossless
      bitrate_mbps: 20                # For custom bitrate
```

---

## 6. Interactive Viewer Features

### 6.1 3D Preview Mode

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  EOSIM Preview - Urban Surveillance MWIR                          [─][□][×] │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────┐ ┌─────────────────────────────────┐ │
│ │                                     │ │  Sensor View (Simulated)        │ │
│ │                                     │ │  ┌───────────────────────────┐  │ │
│ │     3D Scene View                   │ │  │                           │  │ │
│ │                                     │ │  │    [Rendered Frame]       │  │ │
│ │   [Platform Path]     [Targets]     │ │  │                           │  │ │
│ │         ╲               ●           │ │  │                           │  │ │
│ │          ╲             /            │ │  └───────────────────────────┘  │ │
│ │           ◇──────────/              │ │  T_min: 285K  T_max: 355K       │ │
│ │          UAV    FOV cone            │ │  NEΔT: 28mK   SNR: 45dB         │ │
│ │                                     │ │                                 │ │
│ └─────────────────────────────────────┘ └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│ Timeline: [|━━━━━━━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━] │
│           0s              30s             60s             90s          120s │
├─────────────────────────────────────────────────────────────────────────────┤
│ [▶ Play] [⏸ Pause] [⏹ Stop] [⏮ Start] [⏭ End] | Speed: [1x ▼] | Frame: 900 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Controls: LMB=Rotate RMB=Pan Scroll=Zoom | [Render Frame] [Export Sequence] │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Viewer Controls

| Control | Action |
|---------|--------|
| Left Mouse + Drag | Rotate 3D view |
| Right Mouse + Drag | Pan view |
| Scroll Wheel | Zoom in/out |
| Space | Play/Pause timeline |
| Left/Right Arrow | Step frame |
| Home/End | Jump to start/end |
| R | Render current frame |
| F | Toggle fullscreen sensor view |
| G | Toggle ground footprint overlay |
| P | Toggle platform path |
| T | Toggle target labels |
| Ctrl+S | Save current configuration |

---

## 7. Data Input Requirements

### 7.1 Geometry Formats

| Type | Supported Formats | Notes |
|------|-------------------|-------|
| 3D Models | `.obj`, `.gltf`, `.fbx`, `.stl` | OBJ preferred for simplicity |
| Terrain DEM | `.tif` (GeoTIFF), `.hgt`, `.asc` | Must include CRS metadata |
| Textures | `.tif`, `.png`, `.jpg` | 16-bit TIFF for thermal |
| Trajectories | `.csv`, `.yaml`, `.json` | Time-stamped positions |

### 7.2 Material Libraries

```yaml
# materials/default_library.yaml
materials:
  concrete:
    emissivity: 0.92
    reflectivity: 0.08
    thermal_conductivity_w_mk: 1.0
    specific_heat_j_kgk: 880
    density_kg_m3: 2400

  asphalt:
    emissivity: 0.93
    reflectivity: 0.07
    thermal_conductivity_w_mk: 0.75
    specific_heat_j_kgk: 920
    density_kg_m3: 2100

  vegetation:
    emissivity: 0.96
    reflectivity: 0.04
    thermal_conductivity_w_mk: 0.3
    specific_heat_j_kgk: 1800
    density_kg_m3: 700
    transpiration_rate: 0.3  # Evaporative cooling factor

  metal_painted:
    emissivity: 0.85
    reflectivity: 0.15
    thermal_conductivity_w_mk: 50
    specific_heat_j_kgk: 500
    density_kg_m3: 7800

  glass:
    emissivity: 0.90
    reflectivity: 0.10
    thermal_conductivity_w_mk: 1.0
    specific_heat_j_kgk: 840
    density_kg_m3: 2500
    transmissivity_ir: 0.0  # Opaque in thermal IR

  water:
    emissivity: 0.98
    reflectivity: 0.02
    # Water uses special thermal model
    thermal_model: "water_body"
```

### 7.3 Sensor Presets

```bash
$ eosim presets --sensors

Available Sensor Presets:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Name                  Band    Resolution   FOV        NEΔT
────────────────────────────────────────────────────────────────────
mwir_640x512          MWIR    640×512      5.5°×4.4°  25mK
mwir_1280x1024        MWIR    1280×1024    5.5°×4.4°  20mK
lwir_640x480          LWIR    640×480      12°×9°     50mK
lwir_uncooled_vga     LWIR    640×512      24°×18°    80mK
swir_640x512          SWIR    640×512      8°×6.4°    N/A
visible_hd            VIS     1920×1080    30°×17°    N/A
visible_4k            VIS     3840×2160    40°×22°    N/A
multispectral_5band   MS      512×512      15°×15°    N/A

Use: sensor = Sensor.from_preset("mwir_640x512")
```

---

## 8. Summary

This design provides EOSIM with a flexible, professional interface supporting:

1. **Declarative YAML configs** for reproducible scenarios
2. **Python API** for programmatic control and integration
3. **CLI** for automation and batch processing
4. **Interactive preview** for scenario development
5. **Comprehensive outputs** with rich metadata
6. **Modular configuration** allowing reuse of components

The workflow emphasizes:
- Clear separation of scene, sensor, platform, and atmosphere
- Multi-fidelity options at every stage
- Extensive metadata for downstream analysis
- Professional output formats for visualization and archival
