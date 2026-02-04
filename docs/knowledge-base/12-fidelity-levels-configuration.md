# Fidelity Levels and Configuration

## 1. Introduction

EOSIM supports **multi-fidelity simulation** at two levels:

1. **Object Fidelity** - Detail level of individual objects (geometry, materials, thermal)
2. **System Fidelity** - Accuracy level of simulation modules (atmosphere, optics, sensor)

This allows users to balance **accuracy vs. performance** based on their needs.

---

## 2. Fidelity Level Definitions

### 2.1 Standard Fidelity Tiers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FIDELITY TIER OVERVIEW                              │
└─────────────────────────────────────────────────────────────────────────────┘

TIER          ACCURACY    SPEED       USE CASE
────          ────────    ─────       ────────
PREVIEW       ~60%        100+ fps    Real-time preview, scene setup
DRAFT         ~80%        10-30 fps   Quick iterations, debugging
STANDARD      ~90%        1-10 fps    Production simulation
HIGH          ~95%        0.1-1 fps   Validation, final renders
REFERENCE     ~99%        <0.1 fps    Ground truth, algorithm validation
```

### 2.2 Fidelity Dimensions

```
                         FIDELITY DIMENSIONS

        ┌─────────────────────────────────────────────────┐
        │                                                 │
        │   GEOMETRIC        RADIOMETRIC      TEMPORAL    │
        │   ──────────       ───────────      ────────    │
        │                                                 │
        │   • LOD level      • Spectral       • Frame     │
        │   • Mesh detail      sampling         rate      │
        │   • Texture res    • BRDF model    • Motion     │
        │   • Articulation   • Material        blur       │
        │                      zones         • Thermal    │
        │                                      dynamics   │
        │                                                 │
        │   ATMOSPHERIC      OPTICAL          SENSOR      │
        │   ───────────      ───────          ──────      │
        │                                                 │
        │   • RT solver      • PSF model     • Noise      │
        │   • Scattering     • Aberrations     model      │
        │   • Adjacency      • Field effects • FPN        │
        │   • Turbulence     • Distortion    • NUC        │
        │                                                 │
        └─────────────────────────────────────────────────┘
```

---

## 3. Object Fidelity

### 3.1 Geometry Fidelity

```python
@dataclass
class GeometryFidelity:
    """Geometry detail level for an object."""

    level: str                    # preview, draft, standard, high, reference

    # Mesh detail
    max_triangles: int            # Triangle budget
    lod_bias: float               # LOD selection bias (negative = higher detail)

    # Texture detail
    max_texture_size: int         # Maximum texture resolution
    use_normal_maps: bool
    use_displacement: bool

    # Articulation
    articulation_enabled: bool
    articulation_rate_hz: float   # Update rate for moving parts


# Presets
GEOMETRY_FIDELITY = {
    'preview': GeometryFidelity(
        level='preview',
        max_triangles=1000,
        lod_bias=2.0,             # Use coarser LODs
        max_texture_size=256,
        use_normal_maps=False,
        use_displacement=False,
        articulation_enabled=False,
        articulation_rate_hz=0
    ),
    'draft': GeometryFidelity(
        level='draft',
        max_triangles=5000,
        lod_bias=1.0,
        max_texture_size=512,
        use_normal_maps=False,
        use_displacement=False,
        articulation_enabled=True,
        articulation_rate_hz=10
    ),
    'standard': GeometryFidelity(
        level='standard',
        max_triangles=20000,
        lod_bias=0.0,
        max_texture_size=1024,
        use_normal_maps=True,
        use_displacement=False,
        articulation_enabled=True,
        articulation_rate_hz=30
    ),
    'high': GeometryFidelity(
        level='high',
        max_triangles=100000,
        lod_bias=-1.0,            # Use finer LODs
        max_texture_size=2048,
        use_normal_maps=True,
        use_displacement=True,
        articulation_enabled=True,
        articulation_rate_hz=60
    ),
    'reference': GeometryFidelity(
        level='reference',
        max_triangles=float('inf'),
        lod_bias=-2.0,            # Always highest LOD
        max_texture_size=4096,
        use_normal_maps=True,
        use_displacement=True,
        articulation_enabled=True,
        articulation_rate_hz=120
    ),
}
```

### 3.2 Material Fidelity

```python
@dataclass
class MaterialFidelity:
    """Material/radiometric detail level."""

    level: str

    # Spectral sampling
    spectral_mode: str            # monochromatic, 3band, 10band, hyperspectral
    wavelength_samples: int       # Number of wavelength samples

    # BRDF model
    brdf_model: str               # lambertian, phong, cook_torrance, measured
    brdf_samples: int             # For importance sampling

    # Material zones
    max_zones_per_object: int     # Zone consolidation
    zone_blending: bool           # Smooth transitions between zones

    # Emissivity/reflectivity
    spectral_properties: bool     # Use spectral ε(λ) vs. scalar ε
    temperature_dependent: bool   # ε(T) variation


MATERIAL_FIDELITY = {
    'preview': MaterialFidelity(
        level='preview',
        spectral_mode='monochromatic',
        wavelength_samples=1,
        brdf_model='lambertian',
        brdf_samples=1,
        max_zones_per_object=3,
        zone_blending=False,
        spectral_properties=False,
        temperature_dependent=False
    ),
    'draft': MaterialFidelity(
        level='draft',
        spectral_mode='monochromatic',
        wavelength_samples=1,
        brdf_model='lambertian',
        brdf_samples=1,
        max_zones_per_object=8,
        zone_blending=False,
        spectral_properties=False,
        temperature_dependent=False
    ),
    'standard': MaterialFidelity(
        level='standard',
        spectral_mode='3band',
        wavelength_samples=3,
        brdf_model='phong',
        brdf_samples=4,
        max_zones_per_object=20,
        zone_blending=True,
        spectral_properties=True,
        temperature_dependent=False
    ),
    'high': MaterialFidelity(
        level='high',
        spectral_mode='10band',
        wavelength_samples=10,
        brdf_model='cook_torrance',
        brdf_samples=16,
        max_zones_per_object=50,
        zone_blending=True,
        spectral_properties=True,
        temperature_dependent=True
    ),
    'reference': MaterialFidelity(
        level='reference',
        spectral_mode='hyperspectral',
        wavelength_samples=100,
        brdf_model='measured',
        brdf_samples=64,
        max_zones_per_object=float('inf'),
        zone_blending=True,
        spectral_properties=True,
        temperature_dependent=True
    ),
}
```

### 3.3 Thermal Fidelity

```python
@dataclass
class ThermalFidelity:
    """Thermal modeling detail level."""

    level: str

    # Solver type
    solver: str                   # prescribed, steady_state, transient_1d, transient_3d

    # Temporal resolution
    thermal_timestep_s: float     # Thermal solver timestep
    history_duration_s: float     # How much thermal history to track

    # Spatial resolution
    subsurface_layers: int        # For 1D transient
    mesh_resolution: str          # For 3D: coarse, medium, fine

    # Physics
    internal_sources: bool        # Engine heat, etc.
    mutual_radiation: bool        # Inter-surface radiation exchange
    ground_coupling: bool         # Conduction to ground
    convection_model: str         # simple, forced, natural_forced


THERMAL_FIDELITY = {
    'preview': ThermalFidelity(
        level='preview',
        solver='prescribed',       # User-specified temperatures
        thermal_timestep_s=float('inf'),
        history_duration_s=0,
        subsurface_layers=0,
        mesh_resolution='none',
        internal_sources=False,
        mutual_radiation=False,
        ground_coupling=False,
        convection_model='none'
    ),
    'draft': ThermalFidelity(
        level='draft',
        solver='steady_state',
        thermal_timestep_s=60,
        history_duration_s=0,
        subsurface_layers=0,
        mesh_resolution='none',
        internal_sources=True,
        mutual_radiation=False,
        ground_coupling=False,
        convection_model='simple'
    ),
    'standard': ThermalFidelity(
        level='standard',
        solver='transient_1d',
        thermal_timestep_s=10,
        history_duration_s=3600,   # 1 hour
        subsurface_layers=5,
        mesh_resolution='none',
        internal_sources=True,
        mutual_radiation=False,
        ground_coupling=True,
        convection_model='forced'
    ),
    'high': ThermalFidelity(
        level='high',
        solver='transient_1d',
        thermal_timestep_s=1,
        history_duration_s=86400,  # 24 hours
        subsurface_layers=20,
        mesh_resolution='none',
        internal_sources=True,
        mutual_radiation=True,
        ground_coupling=True,
        convection_model='natural_forced'
    ),
    'reference': ThermalFidelity(
        level='reference',
        solver='transient_3d',
        thermal_timestep_s=0.1,
        history_duration_s=86400,
        subsurface_layers=50,
        mesh_resolution='fine',
        internal_sources=True,
        mutual_radiation=True,
        ground_coupling=True,
        convection_model='natural_forced'
    ),
}
```

### 3.4 Combined Object Fidelity

```python
@dataclass
class ObjectFidelity:
    """Combined fidelity settings for an object."""

    geometry: GeometryFidelity
    material: MaterialFidelity
    thermal: ThermalFidelity

    @classmethod
    def from_preset(cls, level: str) -> 'ObjectFidelity':
        """Create from preset level."""
        return cls(
            geometry=GEOMETRY_FIDELITY[level],
            material=MATERIAL_FIDELITY[level],
            thermal=THERMAL_FIDELITY[level]
        )

    @classmethod
    def custom(
        cls,
        geometry: str = 'standard',
        material: str = 'standard',
        thermal: str = 'standard'
    ) -> 'ObjectFidelity':
        """Create custom combination."""
        return cls(
            geometry=GEOMETRY_FIDELITY[geometry],
            material=MATERIAL_FIDELITY[material],
            thermal=THERMAL_FIDELITY[thermal]
        )
```

---

## 4. System Fidelity

### 4.1 Atmosphere Fidelity

```python
@dataclass
class AtmosphereFidelity:
    """Atmospheric modeling fidelity."""

    level: str

    # Radiative transfer
    rt_solver: str                # beer_lambert, two_stream, dom_4, dom_16, monte_carlo
    backend: str                  # simple, raf_tran, py6s

    # Spectral
    spectral_resolution: str      # band_average, 10cm-1, 1cm-1, line_by_line
    wavelength_samples: int

    # Scattering
    rayleigh: bool
    mie: bool
    multiple_scattering: bool

    # Effects
    adjacency: bool               # Terrain adjacency effects
    turbulence: bool              # Atmospheric turbulence/scintillation
    refraction: bool              # Path bending

    # Performance
    use_lut: bool                 # Pre-computed lookup tables
    lut_resolution: str           # coarse, medium, fine


ATMOSPHERE_FIDELITY = {
    'preview': AtmosphereFidelity(
        level='preview',
        rt_solver='beer_lambert',
        backend='simple',
        spectral_resolution='band_average',
        wavelength_samples=1,
        rayleigh=False,
        mie=False,
        multiple_scattering=False,
        adjacency=False,
        turbulence=False,
        refraction=False,
        use_lut=True,
        lut_resolution='coarse'
    ),
    'draft': AtmosphereFidelity(
        level='draft',
        rt_solver='two_stream',
        backend='raf_tran',
        spectral_resolution='band_average',
        wavelength_samples=3,
        rayleigh=True,
        mie=True,
        multiple_scattering=False,
        adjacency=False,
        turbulence=False,
        refraction=False,
        use_lut=True,
        lut_resolution='medium'
    ),
    'standard': AtmosphereFidelity(
        level='standard',
        rt_solver='dom_4',
        backend='raf_tran',
        spectral_resolution='10cm-1',
        wavelength_samples=10,
        rayleigh=True,
        mie=True,
        multiple_scattering=True,
        adjacency=False,
        turbulence=False,
        refraction=False,
        use_lut=True,
        lut_resolution='fine'
    ),
    'high': AtmosphereFidelity(
        level='high',
        rt_solver='dom_16',
        backend='raf_tran',
        spectral_resolution='1cm-1',
        wavelength_samples=50,
        rayleigh=True,
        mie=True,
        multiple_scattering=True,
        adjacency=True,
        turbulence=True,
        refraction=True,
        use_lut=False,
        lut_resolution='none'
    ),
    'reference': AtmosphereFidelity(
        level='reference',
        rt_solver='monte_carlo',
        backend='raf_tran',
        spectral_resolution='line_by_line',
        wavelength_samples=1000,
        rayleigh=True,
        mie=True,
        multiple_scattering=True,
        adjacency=True,
        turbulence=True,
        refraction=True,
        use_lut=False,
        lut_resolution='none'
    ),
}
```

### 4.2 Optics Fidelity

```python
@dataclass
class OpticsFidelity:
    """Optical system modeling fidelity."""

    level: str

    # PSF model
    psf_model: str                # none, gaussian, airy, zernike, measured, wave_optics
    psf_samples: int              # PSF kernel size

    # Aberrations
    include_aberrations: bool
    aberration_order: int         # Zernike polynomial order

    # Field effects
    field_dependent_psf: bool
    vignetting: bool
    distortion: bool

    # Spectral
    chromatic_aberration: bool
    spectral_psf: bool            # Wavelength-dependent PSF

    # Stray light
    stray_light: bool
    ghost_reflections: bool


OPTICS_FIDELITY = {
    'preview': OpticsFidelity(
        level='preview',
        psf_model='none',
        psf_samples=1,
        include_aberrations=False,
        aberration_order=0,
        field_dependent_psf=False,
        vignetting=False,
        distortion=False,
        chromatic_aberration=False,
        spectral_psf=False,
        stray_light=False,
        ghost_reflections=False
    ),
    'draft': OpticsFidelity(
        level='draft',
        psf_model='gaussian',
        psf_samples=5,
        include_aberrations=False,
        aberration_order=0,
        field_dependent_psf=False,
        vignetting=False,
        distortion=False,
        chromatic_aberration=False,
        spectral_psf=False,
        stray_light=False,
        ghost_reflections=False
    ),
    'standard': OpticsFidelity(
        level='standard',
        psf_model='airy',
        psf_samples=15,
        include_aberrations=True,
        aberration_order=4,        # Up to spherical
        field_dependent_psf=False,
        vignetting=True,
        distortion=True,
        chromatic_aberration=False,
        spectral_psf=False,
        stray_light=False,
        ghost_reflections=False
    ),
    'high': OpticsFidelity(
        level='high',
        psf_model='zernike',
        psf_samples=31,
        include_aberrations=True,
        aberration_order=15,
        field_dependent_psf=True,
        vignetting=True,
        distortion=True,
        chromatic_aberration=True,
        spectral_psf=True,
        stray_light=True,
        ghost_reflections=False
    ),
    'reference': OpticsFidelity(
        level='reference',
        psf_model='wave_optics',
        psf_samples=65,
        include_aberrations=True,
        aberration_order=36,       # Full Zernike set
        field_dependent_psf=True,
        vignetting=True,
        distortion=True,
        chromatic_aberration=True,
        spectral_psf=True,
        stray_light=True,
        ghost_reflections=True
    ),
}
```

### 4.3 Sensor Fidelity

```python
@dataclass
class SensorFidelity:
    """Sensor/detector modeling fidelity."""

    level: str

    # Noise model
    shot_noise: bool
    read_noise: bool
    dark_current: bool
    prnu: bool                    # Photo-response non-uniformity
    dsnu: bool                    # Dark signal non-uniformity
    quantization: bool

    # Temporal
    integration_model: str        # instant, integrate, rolling_shutter
    motion_blur: bool
    frame_transfer: bool

    # Non-linearity
    linearity_correction: bool
    blooming: bool
    persistence: bool

    # Calibration
    nuc_applied: bool             # Non-uniformity correction


SENSOR_FIDELITY = {
    'preview': SensorFidelity(
        level='preview',
        shot_noise=False,
        read_noise=False,
        dark_current=False,
        prnu=False,
        dsnu=False,
        quantization=True,
        integration_model='instant',
        motion_blur=False,
        frame_transfer=False,
        linearity_correction=False,
        blooming=False,
        persistence=False,
        nuc_applied=True
    ),
    'draft': SensorFidelity(
        level='draft',
        shot_noise=True,
        read_noise=True,
        dark_current=False,
        prnu=False,
        dsnu=False,
        quantization=True,
        integration_model='instant',
        motion_blur=False,
        frame_transfer=False,
        linearity_correction=False,
        blooming=False,
        persistence=False,
        nuc_applied=True
    ),
    'standard': SensorFidelity(
        level='standard',
        shot_noise=True,
        read_noise=True,
        dark_current=True,
        prnu=True,
        dsnu=True,
        quantization=True,
        integration_model='integrate',
        motion_blur=True,
        frame_transfer=False,
        linearity_correction=False,
        blooming=False,
        persistence=False,
        nuc_applied=False
    ),
    'high': SensorFidelity(
        level='high',
        shot_noise=True,
        read_noise=True,
        dark_current=True,
        prnu=True,
        dsnu=True,
        quantization=True,
        integration_model='integrate',
        motion_blur=True,
        frame_transfer=True,
        linearity_correction=True,
        blooming=True,
        persistence=False,
        nuc_applied=False
    ),
    'reference': SensorFidelity(
        level='reference',
        shot_noise=True,
        read_noise=True,
        dark_current=True,
        prnu=True,
        dsnu=True,
        quantization=True,
        integration_model='rolling_shutter',
        motion_blur=True,
        frame_transfer=True,
        linearity_correction=True,
        blooming=True,
        persistence=True,
        nuc_applied=False
    ),
}
```

### 4.4 Rendering Fidelity

```python
@dataclass
class RenderingFidelity:
    """Rendering engine fidelity."""

    level: str

    # Engine
    renderer: str                 # simple, rasterizer, ray_tracer, path_tracer
    samples_per_pixel: int
    max_bounces: int

    # Anti-aliasing
    antialiasing: str             # none, msaa_2x, msaa_4x, supersampling

    # Shadows
    shadows: str                  # none, hard, soft
    shadow_samples: int

    # Reflections
    reflections: str              # none, planar, ray_traced
    reflection_bounces: int

    # Global illumination
    global_illumination: bool
    ambient_occlusion: bool


RENDERING_FIDELITY = {
    'preview': RenderingFidelity(
        level='preview',
        renderer='simple',
        samples_per_pixel=1,
        max_bounces=0,
        antialiasing='none',
        shadows='none',
        shadow_samples=0,
        reflections='none',
        reflection_bounces=0,
        global_illumination=False,
        ambient_occlusion=False
    ),
    'draft': RenderingFidelity(
        level='draft',
        renderer='rasterizer',
        samples_per_pixel=1,
        max_bounces=0,
        antialiasing='none',
        shadows='hard',
        shadow_samples=1,
        reflections='none',
        reflection_bounces=0,
        global_illumination=False,
        ambient_occlusion=False
    ),
    'standard': RenderingFidelity(
        level='standard',
        renderer='rasterizer',
        samples_per_pixel=1,
        max_bounces=1,
        antialiasing='msaa_2x',
        shadows='soft',
        shadow_samples=4,
        reflections='planar',
        reflection_bounces=1,
        global_illumination=False,
        ambient_occlusion=True
    ),
    'high': RenderingFidelity(
        level='high',
        renderer='ray_tracer',
        samples_per_pixel=16,
        max_bounces=4,
        antialiasing='msaa_4x',
        shadows='soft',
        shadow_samples=16,
        reflections='ray_traced',
        reflection_bounces=2,
        global_illumination=False,
        ambient_occlusion=True
    ),
    'reference': RenderingFidelity(
        level='reference',
        renderer='path_tracer',
        samples_per_pixel=256,
        max_bounces=16,
        antialiasing='supersampling',
        shadows='soft',
        shadow_samples=64,
        reflections='ray_traced',
        reflection_bounces=8,
        global_illumination=True,
        ambient_occlusion=True
    ),
}
```

---

## 5. System Fidelity Configuration

### 5.1 Combined System Fidelity

```python
@dataclass
class SystemFidelity:
    """Complete system fidelity configuration."""

    # Overall preset
    preset: str

    # Per-module settings
    atmosphere: AtmosphereFidelity
    optics: OpticsFidelity
    sensor: SensorFidelity
    rendering: RenderingFidelity

    # Default object fidelity for new objects
    default_object_fidelity: ObjectFidelity

    @classmethod
    def from_preset(cls, level: str) -> 'SystemFidelity':
        """Create system fidelity from preset."""
        return cls(
            preset=level,
            atmosphere=ATMOSPHERE_FIDELITY[level],
            optics=OPTICS_FIDELITY[level],
            sensor=SENSOR_FIDELITY[level],
            rendering=RENDERING_FIDELITY[level],
            default_object_fidelity=ObjectFidelity.from_preset(level)
        )

    @classmethod
    def custom(cls, **kwargs) -> 'SystemFidelity':
        """Create custom fidelity configuration."""
        return cls(
            preset='custom',
            atmosphere=ATMOSPHERE_FIDELITY[kwargs.get('atmosphere', 'standard')],
            optics=OPTICS_FIDELITY[kwargs.get('optics', 'standard')],
            sensor=SENSOR_FIDELITY[kwargs.get('sensor', 'standard')],
            rendering=RENDERING_FIDELITY[kwargs.get('rendering', 'standard')],
            default_object_fidelity=ObjectFidelity.from_preset(
                kwargs.get('objects', 'standard')
            )
        )
```

### 5.2 YAML Configuration

```yaml
# scenario.yaml

fidelity:
  # Option 1: Use preset for entire system
  preset: "standard"

  # Option 2: Custom per-module settings
  # preset: "custom"
  # atmosphere: "high"
  # optics: "standard"
  # sensor: "high"
  # rendering: "draft"
  # objects: "standard"

  # Option 3: Fine-grained control
  # custom:
  #   atmosphere:
  #     rt_solver: "dom_8"
  #     adjacency: true
  #     turbulence: false
  #   sensor:
  #     shot_noise: true
  #     prnu: true
  #     motion_blur: false

# Per-object fidelity overrides
objects:
  - id: "target_vehicle"
    fidelity: "high"              # This object at high fidelity

  - id: "background_building_*"
    fidelity: "draft"             # Background at lower fidelity

  - id: "primary_target"
    fidelity:
      geometry: "high"
      material: "reference"       # Extra accurate materials
      thermal: "high"
```

---

## 6. Fidelity Summary Table

### 6.1 Complete Comparison

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              FIDELITY LEVEL COMPARISON                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ COMPONENT        │ PREVIEW        │ DRAFT          │ STANDARD       │ HIGH           │ REFERENCE    │
├──────────────────┼────────────────┼────────────────┼────────────────┼────────────────┼──────────────┤
│ GEOMETRY                                                                                         │
│  Triangles       │ 1K             │ 5K             │ 20K            │ 100K           │ Unlimited    │
│  LOD bias        │ +2             │ +1             │ 0              │ -1             │ -2           │
│  Articulation    │ Off            │ On (10Hz)      │ On (30Hz)      │ On (60Hz)      │ On (120Hz)   │
├──────────────────┼────────────────┼────────────────┼────────────────┼────────────────┼──────────────┤
│ MATERIALS                                                                                        │
│  Spectral        │ Mono           │ Mono           │ 3-band         │ 10-band        │ Hyperspectral│
│  BRDF            │ Lambertian     │ Lambertian     │ Phong          │ Cook-Torrance  │ Measured     │
│  Zones           │ 3              │ 8              │ 20             │ 50             │ Unlimited    │
├──────────────────┼────────────────┼────────────────┼────────────────┼────────────────┼──────────────┤
│ THERMAL                                                                                          │
│  Solver          │ Prescribed     │ Steady-state   │ 1D Transient   │ 1D Transient   │ 3D Transient │
│  History         │ None           │ None           │ 1 hour         │ 24 hours       │ 24 hours     │
│  Internal src    │ No             │ Yes            │ Yes            │ Yes            │ Yes          │
│  Mutual rad      │ No             │ No             │ No             │ Yes            │ Yes          │
├──────────────────┼────────────────┼────────────────┼────────────────┼────────────────┼──────────────┤
│ ATMOSPHERE                                                                                       │
│  RT Solver       │ Beer-Lambert   │ Two-stream     │ DOM-4          │ DOM-16         │ Monte Carlo  │
│  Spectral        │ Band avg       │ Band avg       │ 10 cm⁻¹        │ 1 cm⁻¹         │ Line-by-line │
│  Scattering      │ None           │ Single         │ Multiple       │ Multiple       │ Multiple     │
│  Adjacency       │ No             │ No             │ No             │ Yes            │ Yes          │
├──────────────────┼────────────────┼────────────────┼────────────────┼────────────────┼──────────────┤
│ OPTICS                                                                                           │
│  PSF             │ None           │ Gaussian       │ Airy           │ Zernike        │ Wave optics  │
│  Aberrations     │ None           │ None           │ 4th order      │ 15th order     │ Full         │
│  Field effects   │ None           │ None           │ Vignette+dist  │ Field PSF      │ Full         │
├──────────────────┼────────────────┼────────────────┼────────────────┼────────────────┼──────────────┤
│ SENSOR                                                                                           │
│  Noise           │ None           │ Shot+Read      │ Full temporal  │ Full + FPN     │ Full + NL    │
│  Integration     │ Instant        │ Instant        │ Integrate      │ Integrate      │ Rolling      │
│  Motion blur     │ No             │ No             │ Yes            │ Yes            │ Yes          │
├──────────────────┼────────────────┼────────────────┼────────────────┼────────────────┼──────────────┤
│ RENDERING                                                                                        │
│  Engine          │ Simple         │ Rasterizer     │ Rasterizer     │ Ray tracer     │ Path tracer  │
│  Samples/pixel   │ 1              │ 1              │ 1              │ 16             │ 256          │
│  Shadows         │ None           │ Hard           │ Soft (4)       │ Soft (16)      │ Soft (64)    │
│  GI              │ No             │ No             │ No             │ No             │ Yes          │
├──────────────────┼────────────────┼────────────────┼────────────────┼────────────────┼──────────────┤
│ PERFORMANCE                                                                                      │
│  Est. speed      │ 100+ fps       │ 10-30 fps      │ 1-10 fps       │ 0.1-1 fps      │ <0.1 fps     │
│  Memory          │ Low            │ Low            │ Medium         │ High           │ Very High    │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Fidelity Selection Guidelines

### 7.1 Use Case Recommendations

| Use Case | Recommended Fidelity | Rationale |
|----------|---------------------|-----------|
| Scene setup / preview | PREVIEW | Fast iteration |
| Algorithm development | DRAFT | Balance of speed and realism |
| Training data generation | STANDARD | Good enough for ML |
| Production simulation | STANDARD / HIGH | Accurate results |
| Sensor performance analysis | HIGH | Need accurate sensor model |
| Algorithm validation | HIGH / REFERENCE | Compare against ground truth |
| Publication / final renders | REFERENCE | Maximum accuracy |

### 7.2 Per-Object Fidelity Strategy

```
Scene with primary target and background:

┌─────────────────────────────────────────────────────────────────┐
│                    FIDELITY ALLOCATION STRATEGY                 │
└─────────────────────────────────────────────────────────────────┘

                    PRIMARY TARGET
                    ┌───────────────┐
                    │   HIGH or     │  ← Maximum detail where it matters
                    │   REFERENCE   │
                    └───────────────┘

    SECONDARY TARGETS              BACKGROUND
    ┌───────────────┐             ┌───────────────┐
    │   STANDARD    │             │   DRAFT or    │
    │               │             │   PREVIEW     │
    └───────────────┘             └───────────────┘

    ↑ Moderate detail              ↑ Low detail OK
      (still need accurate           (far from camera,
       thermal signatures)            minimal impact)
```

### 7.3 Adaptive Fidelity

```python
class AdaptiveFidelityManager:
    """
    Dynamically adjust fidelity based on:
    - Distance from sensor
    - Object importance
    - Available compute budget
    """

    def __init__(self, base_fidelity: SystemFidelity, target_fps: float):
        self.base = base_fidelity
        self.target_fps = target_fps
        self.current_fps = target_fps

    def get_object_fidelity(
        self,
        obj: 'ArenaObject',
        distance_m: float,
        importance: float
    ) -> ObjectFidelity:
        """
        Determine fidelity for specific object.

        Args:
            obj: The object
            distance_m: Distance from sensor
            importance: 0-1 importance weight
        """
        # Base fidelity from object definition
        base_level = obj.fidelity_hint or self.base.default_object_fidelity

        # Distance-based adjustment
        if distance_m > 2000:
            distance_modifier = -2
        elif distance_m > 500:
            distance_modifier = -1
        elif distance_m < 100:
            distance_modifier = +1
        else:
            distance_modifier = 0

        # Importance-based adjustment
        importance_modifier = int((importance - 0.5) * 2)

        # Performance-based adjustment
        if self.current_fps < self.target_fps * 0.8:
            performance_modifier = -1
        elif self.current_fps > self.target_fps * 1.2:
            performance_modifier = +1
        else:
            performance_modifier = 0

        # Compute final level
        levels = ['preview', 'draft', 'standard', 'high', 'reference']
        base_idx = levels.index(base_level.geometry.level)
        final_idx = np.clip(
            base_idx + distance_modifier + importance_modifier + performance_modifier,
            0, len(levels) - 1
        )

        return ObjectFidelity.from_preset(levels[final_idx])
```

---

## 8. Implementation

### 8.1 Fidelity Manager

```python
# src/eosim/core/fidelity.py

class FidelityManager:
    """
    Central manager for fidelity settings.
    """

    def __init__(self, system_fidelity: SystemFidelity):
        self.system = system_fidelity
        self.object_overrides: Dict[str, ObjectFidelity] = {}

    def set_object_fidelity(self, object_id: str, fidelity: ObjectFidelity):
        """Override fidelity for specific object."""
        self.object_overrides[object_id] = fidelity

    def get_object_fidelity(self, object_id: str) -> ObjectFidelity:
        """Get fidelity for object (with overrides)."""
        if object_id in self.object_overrides:
            return self.object_overrides[object_id]
        return self.system.default_object_fidelity

    def get_module_config(self, module: str) -> dict:
        """Get configuration dict for a module."""
        if module == 'atmosphere':
            return asdict(self.system.atmosphere)
        elif module == 'optics':
            return asdict(self.system.optics)
        elif module == 'sensor':
            return asdict(self.system.sensor)
        elif module == 'rendering':
            return asdict(self.system.rendering)
        else:
            raise ValueError(f"Unknown module: {module}")
```

---

## 9. Summary

### Key Concepts

1. **Object Fidelity** = Geometry + Material + Thermal detail
2. **System Fidelity** = Atmosphere + Optics + Sensor + Rendering
3. **5 Standard Tiers**: Preview → Draft → Standard → High → Reference
4. **Mixed Fidelity**: Different levels for different scene elements
5. **Adaptive Fidelity**: Dynamic adjustment based on distance/importance/performance

### Configuration Flow

```
User specifies:
  scenario.fidelity.preset = "standard"

System applies:
  atmosphere → DOM-4, 10-band, single scattering
  optics     → Airy PSF, 4th order aberrations
  sensor     → Full noise, motion blur
  rendering  → Rasterizer, soft shadows
  objects    → 20K tris, 20 zones, 1D transient

Per-object overrides:
  target.fidelity = "high"  → 100K tris, 50 zones
```

---

## 10. References

1. **DIRSIG** - Multi-fidelity architecture
2. **Unreal Engine** - Scalability settings
3. **PBRT** - Rendering quality levels
4. **ISETCam** - Sensor model fidelity options
