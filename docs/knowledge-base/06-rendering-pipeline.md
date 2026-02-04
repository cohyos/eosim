# Rendering Pipeline

## 1. Introduction

This document describes the end-to-end rendering pipeline for EOSIM, from 3D scene definition to final sensor output. The pipeline integrates all previously documented modules (scene, thermal, atmosphere, optics, sensor) into a coherent image synthesis workflow.

---

## 2. Pipeline Overview

### 2.1 Complete Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EOSIM RENDERING PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
│  SCENE    │──▶│  THERMAL  │──▶│ RADIANCE  │──▶│   ATMOS   │──▶│  OPTICS   │
│           │   │           │   │           │   │           │   │           │
│ Geometry  │   │ T(x,y,z)  │   │ L_surface │   │ L_sensor  │   │ E_focal   │
│ Materials │   │ Solver    │   │ ε,ρ,BRDF  │   │ τ, L_path │   │ PSF,MTF   │
└───────────┘   └───────────┘   └───────────┘   └───────────┘   └───────────┘
                                                                      │
      ┌───────────────────────────────────────────────────────────────┘
      │
      ▼
┌───────────┐   ┌───────────┐   ┌───────────┐
│  SENSOR   │──▶│    ADC    │──▶│  OUTPUT   │
│           │   │           │   │           │
│ QE, Noise │   │ DN        │   │ Frame/    │
│ e⁻        │   │ Quantize  │   │ Video     │
└───────────┘   └───────────┘   └───────────┘
```

### 2.2 Data Flow

| Stage | Input | Output | Units |
|-------|-------|--------|-------|
| Scene | 3D geometry, materials | Visible surfaces, properties | - |
| Thermal | Surface properties, environment | Temperature map | K |
| Radiance | T, ε, ρ, illumination | Surface leaving radiance | W/(m²·sr·μm) |
| Atmosphere | L_surface, path geometry | At-sensor radiance | W/(m²·sr·μm) |
| Optics | L_sensor | Focal plane irradiance | W/m² |
| Sensor | E_focal | Digital output | DN (uint16) |

---

## 3. Rendering Modes

### 3.1 Ray Tracing (High Fidelity)

```python
def render_ray_traced(
    scene: Scene,
    sensor: Sensor,
    platform: Platform,
    atmosphere: Atmosphere,
    samples_per_pixel: int = 64
) -> np.ndarray:
    """
    Full ray-traced rendering with path tracing.

    For each pixel:
    1. Cast rays through pixel into scene
    2. Find surface intersection
    3. Compute surface radiance (emission + reflection)
    4. Propagate through atmosphere
    5. Apply optical effects
    6. Accumulate in sensor

    Suitable for:
    - Validation renders
    - Complex BRDF materials
    - Multiple scattering scenarios
    """
    pass
```

**Advantages**:
- Physically accurate
- Handles complex geometry and reflections
- Natural anti-aliasing via supersampling

**Disadvantages**:
- Computationally expensive
- Slower for real-time

### 3.2 Rasterization (Real-Time)

```python
def render_rasterized(
    scene: Scene,
    sensor: Sensor,
    platform: Platform,
    atmosphere: Atmosphere
) -> np.ndarray:
    """
    GPU-accelerated rasterization rendering.

    Pipeline:
    1. Project 3D geometry to sensor focal plane
    2. Rasterize triangles with depth test
    3. Fragment shader computes radiance
    4. Post-process: atmosphere, optics, noise

    Suitable for:
    - Real-time preview
    - Training data generation
    - Interactive scenarios
    """
    pass
```

**Advantages**:
- Fast (GPU-accelerated)
- Deterministic
- Good for simple scenes

**Disadvantages**:
- Limited reflection handling
- Approximations for complex physics

### 3.3 Hybrid Approach

Combine rasterization for geometry with ray tracing for effects:

```python
def render_hybrid(
    scene: Scene,
    sensor: Sensor,
    ...
) -> np.ndarray:
    """
    Hybrid rendering:
    1. Rasterize geometry for G-buffer (depth, normals, materials)
    2. Compute radiance using physics models
    3. Ray trace reflections if needed
    4. Post-process atmosphere and sensor effects
    """
    pass
```

---

## 4. Spectral Rendering

### 4.1 Monochromatic vs Spectral

**Monochromatic** (effective wavelength):
- Single representative wavelength per band
- Fast computation
- Adequate for narrowband sensors

**Spectral** (wavelength-sampled):
- Multiple wavelength samples integrated
- Required for accurate broadband simulation
- Higher computational cost

### 4.2 Spectral Integration

```python
def spectral_render(
    scene: Scene,
    wavelengths_um: np.ndarray,
    sensor_response: np.ndarray
) -> np.ndarray:
    """
    Spectrally-resolved rendering.

    For each wavelength:
    1. Compute spectral radiance L(λ)
    2. Apply spectral atmosphere τ(λ), L_path(λ)
    3. Apply spectral optics τ_opt(λ)
    4. Apply spectral QE η(λ)

    Integrate over sensor response:
    Signal = ∫ L(λ) × τ(λ) × τ_opt(λ) × η(λ) × R(λ) dλ
    """
    n_wavelengths = len(wavelengths_um)
    H, W = scene.resolution

    # Accumulator
    integrated_signal = np.zeros((H, W))

    for i, wl in enumerate(wavelengths_um):
        # Render at this wavelength
        L_surface = compute_surface_radiance(scene, wl)
        L_sensor = apply_atmosphere(L_surface, atmosphere, wl)
        E_focal = apply_optics(L_sensor, optics, wl)

        # Weight by sensor response
        weight = sensor_response[i]
        integrated_signal += E_focal * weight

    # Normalize by response integral
    integrated_signal /= np.trapz(sensor_response, wavelengths_um)

    return integrated_signal
```

### 4.3 Hero Wavelength Optimization

For efficiency, use a few "hero" wavelengths:

```python
HERO_WAVELENGTHS = {
    'MWIR': [3.5, 4.0, 4.5],  # 3 samples across 3-5 μm
    'LWIR': [8.5, 10.0, 12.0],  # 3 samples across 8-14 μm
    'VIS': [0.45, 0.55, 0.65],  # RGB approximation
}
```

---

## 5. Rendering Equation Implementation

### 5.1 Surface Radiance Computation

```python
def compute_surface_radiance(
    scene: Scene,
    pixel_coords: tuple,
    wavelength_um: float,
    view_direction: np.ndarray
) -> float:
    """
    Compute leaving radiance at a surface point.

    L_leaving = L_emitted + L_reflected

    L_emitted = ε(λ) × B(λ, T)

    L_reflected = ρ(λ) × ∫ BRDF × L_incident × cos(θ_i) dω_i
    """
    # Get surface properties at this point
    surface = scene.get_surface_at_pixel(pixel_coords)
    T = scene.thermal_state.get_temperature(surface.id)
    emissivity = surface.material.get_emissivity(wavelength_um)
    reflectivity = 1.0 - emissivity  # Kirchhoff

    # Emitted radiance (Planck)
    L_emitted = emissivity * planck_radiance(wavelength_um, T)

    # Reflected radiance
    L_reflected = 0.0

    if reflectivity > 0.01:  # Skip if nearly black
        # Sample incident directions
        for sample in sample_hemisphere(surface.normal):
            # Check visibility
            L_incident = compute_incident_radiance(
                surface.position, sample.direction, scene, wavelength_um
            )
            # BRDF
            brdf = surface.material.brdf(
                sample.direction, view_direction, surface.normal, wavelength_um
            )
            # Cosine term
            cos_theta = max(0, np.dot(sample.direction, surface.normal))

            L_reflected += brdf * L_incident * cos_theta * sample.weight

    return L_emitted + reflectivity * L_reflected


def compute_incident_radiance(
    position: np.ndarray,
    direction: np.ndarray,
    scene: Scene,
    wavelength_um: float
) -> float:
    """
    Compute radiance arriving from a direction.

    Sources:
    - Direct solar (if visible to sun)
    - Sky radiance (if direction hits sky)
    - Other surfaces (if direction hits scene)
    """
    # Cast ray in direction
    hit = scene.ray_intersect(position, direction)

    if hit is None:
        # Ray escapes to sky
        if direction[2] > 0:  # Upward
            return scene.sky.radiance(direction, wavelength_um)
        else:  # Downward (shouldn't happen for reflection)
            return 0.0

    # Ray hits another surface
    return compute_surface_radiance(
        scene, hit.pixel_coords, wavelength_um, -direction
    )
```

### 5.2 Direct Illumination

```python
def compute_direct_solar(
    surface: Surface,
    scene: Scene,
    wavelength_um: float
) -> float:
    """
    Compute direct solar contribution.
    """
    sun_dir = scene.sun.direction
    cos_theta = np.dot(surface.normal, sun_dir)

    if cos_theta <= 0:
        return 0.0  # Surface faces away from sun

    # Shadow test
    if scene.is_shadowed(surface.position, sun_dir):
        return 0.0

    # Solar irradiance at surface
    E_solar = scene.sun.irradiance(wavelength_um) * cos_theta

    # Convert to radiance via BRDF
    # For Lambertian: L = ρ × E / π
    reflectivity = surface.material.get_reflectivity(wavelength_um)

    if surface.material.brdf_type == "lambertian":
        return reflectivity * E_solar / np.pi
    else:
        # General BRDF evaluation
        view_dir = scene.sensor.get_view_direction(surface.position)
        brdf = surface.material.brdf(sun_dir, view_dir, surface.normal, wavelength_um)
        return brdf * E_solar
```

---

## 6. Pipeline Implementation

### 6.1 Pipeline Class

```python
# src/eosim/pipeline/chain.py

from dataclasses import dataclass
from typing import Iterator, Optional
import numpy as np

@dataclass
class FrameResult:
    """Result of rendering a single frame."""
    index: int
    time_s: float
    image: np.ndarray               # Final DN output [H, W]
    radiance_map: Optional[np.ndarray] = None  # Pre-sensor radiance
    metadata: dict = None


class RenderPipeline:
    """
    End-to-end EOSIM rendering pipeline.
    """

    def __init__(
        self,
        scenario: 'Scenario',
        fidelity: str = "medium"
    ):
        self.scenario = scenario
        self.fidelity = fidelity

        # Initialize modules based on fidelity
        self._init_modules()

    def _init_modules(self):
        """Initialize pipeline modules."""
        fidelity = self.fidelity

        # Thermal solver
        if fidelity == "simple":
            self.thermal = PrescribedThermal(self.scenario.arena)
        elif fidelity == "medium":
            self.thermal = SteadyStateThermal(self.scenario.arena)
        else:
            self.thermal = Transient1DThermal(self.scenario.arena)

        # Atmosphere
        if fidelity == "simple":
            self.atmosphere = SimpleAtmosphere(
                visibility_km=self.scenario.atmosphere.visibility_km
            )
        else:
            self.atmosphere = RAFTranAtmosphere(
                profile=self.scenario.atmosphere.profile,
                visibility_km=self.scenario.atmosphere.visibility_km
            )

        # Optics
        self.optics = SimpleOptics(self.scenario.sensor.optics_config)

        # Sensor
        self.sensor = FPASensor(self.scenario.sensor.config)

        # Renderer
        if fidelity == "high":
            self.renderer = RayTraceRenderer(samples=64)
        else:
            self.renderer = RasterRenderer()

    def render_frame(self, time_s: float) -> FrameResult:
        """
        Render a single frame at specified time.
        """
        # Update scene state for this time
        self.scenario.update_time(time_s)

        # Get sensor geometry
        platform_state = self.scenario.platform.get_state(time_s)
        sensor_geometry = self.scenario.sensor.get_geometry(platform_state)

        # Step 1: Thermal solution
        thermal_state = self.thermal.solve(
            self.scenario.arena,
            self.scenario.environment
        )

        # Step 2: Render surface radiance
        L_surface = self.renderer.render(
            self.scenario.arena,
            thermal_state,
            sensor_geometry,
            self.scenario.sensor.wavelength_um
        )

        # Step 3: Atmospheric propagation
        path = self._compute_atmosphere_path(sensor_geometry)
        atm_result = self.atmosphere.compute(
            np.array([self.scenario.sensor.wavelength_um]),
            path
        )

        L_sensor = L_surface * atm_result.transmission[0] + atm_result.path_radiance[0]

        # Step 4: Optical effects
        E_focal = self.optics.apply(L_sensor, self.scenario.sensor.wavelength_um)

        # Step 5: Sensor response
        dn_image = self.sensor.process(E_focal, self.scenario.sensor.wavelength_um)

        return FrameResult(
            index=int(time_s * self.scenario.frame_rate),
            time_s=time_s,
            image=dn_image,
            radiance_map=L_sensor if self.scenario.diagnostics.save_radiance else None,
            metadata=self._build_metadata(time_s, platform_state, sensor_geometry)
        )

    def run(self) -> Iterator[FrameResult]:
        """
        Generator that yields frames for the entire scenario.
        """
        t = 0.0
        dt = 1.0 / self.scenario.frame_rate

        while t <= self.scenario.duration:
            yield self.render_frame(t)
            t += dt

    def run_all(self) -> list:
        """Run complete simulation and return all frames."""
        return list(self.run())

    def _compute_atmosphere_path(self, sensor_geometry) -> SlantPath:
        """Compute atmospheric path from sensor to scene center."""
        return SlantPath(
            sensor_altitude_m=sensor_geometry.altitude_m,
            target_altitude_m=0,  # Ground level
            zenith_angle_deg=sensor_geometry.look_angle_deg,
            azimuth_angle_deg=sensor_geometry.azimuth_deg
        )

    def _build_metadata(self, time_s, platform_state, sensor_geometry) -> dict:
        """Build per-frame metadata."""
        return {
            'time_s': time_s,
            'platform': {
                'position': platform_state.position.tolist(),
                'attitude': platform_state.attitude.tolist(),
            },
            'sensor': {
                'look_angle_deg': sensor_geometry.look_angle_deg,
                'slant_range_m': sensor_geometry.slant_range_m,
                'gsd_m': sensor_geometry.gsd_m,
            }
        }
```

### 6.2 Renderer Base Class

```python
# src/eosim/render/base.py

from abc import ABC, abstractmethod
import numpy as np

class Renderer(ABC):
    """Abstract base class for scene renderers."""

    @abstractmethod
    def render(
        self,
        scene: Scene,
        thermal_state: ThermalState,
        sensor_geometry: SensorGeometry,
        wavelength_um: float
    ) -> np.ndarray:
        """
        Render scene to surface radiance image.

        Args:
            scene: Scene geometry and materials
            thermal_state: Surface temperatures
            sensor_geometry: Sensor position and orientation
            wavelength_um: Rendering wavelength

        Returns:
            Surface leaving radiance [H, W] in W/(m²·sr·μm)
        """
        pass


class RasterRenderer(Renderer):
    """Fast GPU-accelerated rasterization renderer."""

    def render(self, scene, thermal_state, sensor_geometry, wavelength_um):
        # Implementation using OpenGL/Vulkan or software rasterization
        H, W = scene.resolution

        # Initialize output
        radiance = np.zeros((H, W))

        # Rasterize each object
        for obj in scene.objects:
            # Project to image plane
            # Rasterize triangles
            # Compute radiance at each pixel
            pass

        return radiance


class RayTraceRenderer(Renderer):
    """Ray tracing renderer using Mitsuba 3 or custom implementation."""

    def __init__(self, samples: int = 64):
        self.samples = samples

    def render(self, scene, thermal_state, sensor_geometry, wavelength_um):
        # Use Mitsuba 3 for ray tracing
        # Or custom implementation
        pass
```

---

## 7. Mitsuba 3 Integration

### 7.1 Scene Conversion

```python
# src/eosim/render/mitsuba_backend.py

def scene_to_mitsuba(scene: Scene, thermal_state: ThermalState) -> dict:
    """
    Convert EOSIM scene to Mitsuba 3 scene dictionary.
    """
    mi_scene = {
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 8
        },
        'sensor': {
            'type': 'perspective',
            'fov': scene.sensor.fov_deg,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=scene.sensor.position,
                target=scene.sensor.look_at,
                up=[0, 0, 1]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': scene.sensor.resolution[0],
                'height': scene.sensor.resolution[1],
            }
        }
    }

    # Add geometry
    for i, obj in enumerate(scene.objects):
        T = thermal_state.get_temperature(obj.id)
        emissivity = obj.material.emissivity

        mi_scene[f'shape_{i}'] = {
            'type': 'obj',
            'filename': obj.geometry_file,
            'to_world': mi.ScalarTransform4f(obj.transform),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'spectrum',
                    'value': 1.0 - emissivity
                }
            },
            # Emission (thermal)
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'blackbody',
                    'temperature': T,
                    'scale': emissivity
                }
            }
        }

    return mi_scene


class MitsubaRenderer(Renderer):
    """High-fidelity renderer using Mitsuba 3."""

    def __init__(self, samples: int = 64, use_gpu: bool = True):
        import mitsuba as mi
        if use_gpu:
            mi.set_variant('cuda_ad_rgb')
        else:
            mi.set_variant('scalar_rgb')

        self.samples = samples

    def render(self, scene, thermal_state, sensor_geometry, wavelength_um):
        import mitsuba as mi

        # Convert scene
        mi_dict = scene_to_mitsuba(scene, thermal_state)
        mi_scene = mi.load_dict(mi_dict)

        # Render
        image = mi.render(mi_scene, spp=self.samples)

        # Convert to numpy
        return np.array(image)
```

---

## 8. Parallel and Batch Processing

### 8.1 Frame-Level Parallelism

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def render_frames_parallel(
    pipeline: RenderPipeline,
    times: list,
    max_workers: int = 4
) -> list:
    """
    Render multiple frames in parallel.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(pipeline.render_frame, times))
    return results
```

### 8.2 GPU Batching

```python
def render_batch_gpu(
    pipeline: RenderPipeline,
    times: list,
    batch_size: int = 8
) -> list:
    """
    Batch multiple frames for GPU efficiency.
    """
    results = []

    for i in range(0, len(times), batch_size):
        batch_times = times[i:i+batch_size]

        # Prepare batch on GPU
        # Render batch
        # Transfer results

        batch_results = [pipeline.render_frame(t) for t in batch_times]
        results.extend(batch_results)

    return results
```

---

## 9. Diagnostic Outputs

### 9.1 Intermediate Products

```python
@dataclass
class DiagnosticOutputs:
    """Intermediate outputs for debugging and analysis."""
    surface_temperature: np.ndarray    # T(x,y) [K]
    surface_radiance: np.ndarray       # L_surface [W/(m²·sr·μm)]
    at_sensor_radiance: np.ndarray     # L_sensor (post-atmosphere)
    focal_plane_irradiance: np.ndarray # E_focal (post-optics)
    pre_noise_signal: np.ndarray       # Electrons (pre-noise)
    final_image: np.ndarray            # DN output


def render_with_diagnostics(
    pipeline: RenderPipeline,
    time_s: float
) -> DiagnosticOutputs:
    """
    Render with all intermediate outputs saved.
    """
    # ... render each stage and capture intermediate results
    pass
```

### 9.2 Visualization

```python
def visualize_pipeline_stages(diagnostics: DiagnosticOutputs):
    """
    Create visualization of all pipeline stages.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Temperature map
    im0 = axes[0,0].imshow(diagnostics.surface_temperature, cmap='hot')
    axes[0,0].set_title('Surface Temperature [K]')
    plt.colorbar(im0, ax=axes[0,0])

    # Surface radiance
    im1 = axes[0,1].imshow(diagnostics.surface_radiance, cmap='inferno')
    axes[0,1].set_title('Surface Radiance [W/(m²·sr·μm)]')
    plt.colorbar(im1, ax=axes[0,1])

    # At-sensor radiance
    im2 = axes[0,2].imshow(diagnostics.at_sensor_radiance, cmap='inferno')
    axes[0,2].set_title('At-Sensor Radiance')
    plt.colorbar(im2, ax=axes[0,2])

    # Pre-noise signal
    im3 = axes[1,0].imshow(diagnostics.pre_noise_signal, cmap='gray')
    axes[1,0].set_title('Pre-Noise Signal [e⁻]')
    plt.colorbar(im3, ax=axes[1,0])

    # Final image
    im4 = axes[1,1].imshow(diagnostics.final_image, cmap='gray')
    axes[1,1].set_title('Final Output [DN]')
    plt.colorbar(im4, ax=axes[1,1])

    # Histogram
    axes[1,2].hist(diagnostics.final_image.ravel(), bins=100)
    axes[1,2].set_title('DN Histogram')
    axes[1,2].set_xlabel('DN')
    axes[1,2].set_ylabel('Count')

    plt.tight_layout()
    return fig
```

---

## 10. Performance Optimization

### 10.1 Caching Strategy

```python
class CachedPipeline(RenderPipeline):
    """Pipeline with intelligent caching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}

    def render_frame(self, time_s):
        # Cache thermal state if scene hasn't changed
        thermal_key = self._get_thermal_cache_key(time_s)
        if thermal_key not in self._cache:
            self._cache[thermal_key] = self.thermal.solve(...)

        # Cache atmosphere LUT
        if 'atmosphere_lut' not in self._cache:
            self._cache['atmosphere_lut'] = self.atmosphere.generate_lut(...)

        # ... rest of rendering with cached data
```

### 10.2 Level of Detail

```python
def adaptive_render(
    pipeline: RenderPipeline,
    target_fps: float = 30.0
) -> Iterator[FrameResult]:
    """
    Adaptive rendering that adjusts fidelity to meet frame rate.
    """
    import time

    target_frame_time = 1.0 / target_fps

    for t in pipeline.frame_times:
        start = time.time()

        # Render with current fidelity
        result = pipeline.render_frame(t)

        elapsed = time.time() - start

        # Adjust fidelity for next frame
        if elapsed > target_frame_time * 1.2:
            pipeline.decrease_fidelity()
        elif elapsed < target_frame_time * 0.5:
            pipeline.increase_fidelity()

        yield result
```

---

## 11. Summary

### Pipeline Stages

1. **Scene Setup** - Geometry, materials, environment
2. **Thermal Solver** - Surface temperatures
3. **Radiance Computation** - L = ε×B(T) + ρ×L_reflected
4. **Atmospheric Propagation** - L_sensor = L_surface×τ + L_path
5. **Optical Effects** - PSF convolution, vignetting
6. **Sensor Response** - QE, noise, ADC
7. **Output** - DN image/video

### Implementation Priority

1. **Basic rasterization** - Fast initial results
2. **Full spectral integration** - Accurate broadband
3. **Ray tracing option** - High-fidelity validation
4. **Diagnostic outputs** - Debug and analysis
5. **GPU acceleration** - Performance

### Key Considerations

- Modular design allows swapping renderers
- Diagnostic outputs essential for validation
- Caching critical for performance
- Spectral rendering for accurate radiometry

---

## 12. References

1. **DIRSIG Documentation** - Rendering architecture
2. **Pharr et al.** (2016) - "Physically Based Rendering" (PBRT)
3. **Mitsuba 3 Documentation** - Spectral rendering
4. **ISETCam** - Pipeline architecture
5. **SENSOR++** - Three-stage architecture
