# Scene and Thermal Modeling

## 1. Introduction

This document covers 3D scene representation and thermal physics modeling for EOSIM. Accurate prediction of surface temperatures is essential for infrared simulation, as thermal emission dominates the signal in MWIR and LWIR bands.

---

## 2. Scene Geometry Representation

### 2.1 Geometry Types

| Type | Description | Use Case |
|------|-------------|----------|
| **Triangle Mesh** | Explicit triangulated surfaces | Detailed objects (vehicles, buildings) |
| **Height Field (DEM)** | Regular grid of elevations | Terrain |
| **Voxel Grid** | 3D volumetric cells | Vegetation canopy, participating media |
| **Implicit Surfaces** | Mathematical definitions | Procedural geometry |
| **Point Clouds** | Discrete 3D points | LiDAR-derived scenes |

### 2.2 Coordinate Systems

```
┌─────────────────────────────────────────────────────────────────┐
│                    COORDINATE SYSTEM HIERARCHY                  │
└─────────────────────────────────────────────────────────────────┘

World (Geodetic)          Local ENU              Object Local
    WGS84               East-North-Up            Model Space
      │                      │                       │
      │    Transform         │     Transform         │
      ├──────────────────────┼─────────────────────────
      │                      │                       │
   (lat, lon, alt)      (x_e, x_n, x_u)         (x, y, z)
                              │
                        Scene Origin
                     (reference point)
```

**Local ENU (East-North-Up)**:
- Origin at scene reference point (lat, lon, alt)
- X-axis: East
- Y-axis: North
- Z-axis: Up (normal to ellipsoid)
- Units: meters

### 2.3 Scene Graph Structure

```python
class SceneNode:
    """Hierarchical scene representation."""

    def __init__(self):
        self.transform: np.ndarray  # 4x4 transformation matrix
        self.geometry: Optional[Geometry]
        self.material: Optional[Material]
        self.children: List[SceneNode]
        self.name: str

class Scene:
    """Complete scene container."""

    def __init__(self):
        self.root: SceneNode
        self.terrain: Terrain
        self.atmosphere: AtmosphereState
        self.sun: SunPosition
        self.time: datetime
        self.ambient_temperature_K: float
```

---

## 3. Material Properties

### 3.1 Radiometric Properties

```python
@dataclass
class SpectralMaterial:
    """Wavelength-dependent material properties."""

    # Spectral properties (can be scalar or array vs wavelength)
    emissivity: Union[float, np.ndarray]      # ε(λ)
    reflectivity: Union[float, np.ndarray]    # ρ(λ) = 1 - ε(λ) for opaque

    # BRDF model for non-Lambertian surfaces
    brdf_model: str = "lambertian"  # lambertian, phong, cook_torrance, measured
    brdf_params: dict = field(default_factory=dict)
```

### 3.2 Thermophysical Properties

```python
@dataclass
class ThermalMaterial:
    """Properties for thermal modeling."""

    # Conduction
    thermal_conductivity: float    # k [W/(m·K)]
    specific_heat: float           # c_p [J/(kg·K)]
    density: float                 # ρ [kg/m³]

    # Derived: thermal diffusivity
    @property
    def diffusivity(self) -> float:
        """α = k / (ρ × c_p) [m²/s]"""
        return self.thermal_conductivity / (self.density * self.specific_heat)

    # Derived: thermal inertia
    @property
    def thermal_inertia(self) -> float:
        """P = √(k × ρ × c_p) [J/(m²·K·s^0.5)]"""
        return np.sqrt(self.thermal_conductivity * self.density * self.specific_heat)
```

### 3.3 Common Material Database

| Material | k [W/m·K] | c_p [J/kg·K] | ρ [kg/m³] | P [J/m²Ks^0.5] | ε_LWIR |
|----------|-----------|--------------|-----------|----------------|--------|
| Concrete | 1.0 | 880 | 2400 | 1450 | 0.92 |
| Asphalt | 0.75 | 920 | 2100 | 1200 | 0.93 |
| Brick | 0.72 | 840 | 1800 | 1050 | 0.90 |
| Glass | 1.0 | 840 | 2500 | 1450 | 0.90 |
| Steel | 50 | 500 | 7800 | 14000 | 0.20* |
| Aluminum | 205 | 900 | 2700 | 22300 | 0.05* |
| Soil (dry) | 0.3 | 800 | 1500 | 600 | 0.92 |
| Soil (wet) | 1.5 | 2000 | 1800 | 2300 | 0.95 |
| Vegetation | 0.3 | 1800 | 700 | 620 | 0.96 |
| Water | 0.6 | 4186 | 1000 | 1580 | 0.98 |

*Bare metal; painted metal typically ε = 0.85-0.95

---

## 4. Thermal Balance Equation

### 4.1 Surface Energy Balance

The temperature of a surface element is determined by energy balance:

```
ρ × c_p × V × dT/dt = Q_solar + Q_LW_in - Q_LW_out - Q_convection - Q_conduction

Where:
  ρ × c_p × V × dT/dt  = Rate of thermal energy storage
  Q_solar              = Absorbed solar radiation
  Q_LW_in              = Absorbed longwave (IR) from sky/surroundings
  Q_LW_out             = Emitted longwave radiation
  Q_convection         = Convective heat transfer to air
  Q_conduction         = Conductive heat transfer to adjacent materials
```

### 4.2 Component Equations

**Solar absorption**:
```
Q_solar = α_solar × E_solar × cos(θ_sun) × A_projected × shadow_factor

Where:
  α_solar = Solar absorptivity (≈ 1 - albedo for diffuse)
  E_solar = Solar irradiance [W/m²] (≈1000 W/m² at noon)
  θ_sun   = Angle between surface normal and sun direction
  A_projected = Projected area facing sun
  shadow_factor = 0 if shadowed, 1 if illuminated
```

**Longwave radiation exchange**:
```
Q_LW_out = ε × σ × T⁴ × A

Q_LW_in = ε × A × [F_sky × σ × T_sky⁴ + F_ground × σ × T_ground⁴ + F_surround × σ × T_surround⁴]

Where:
  F_sky, F_ground, F_surround = View factors to sky, ground, surroundings
  T_sky ≈ T_air - 20K (clear sky approximation)
```

**Convection**:
```
Q_convection = h × A × (T_surface - T_air)

Free convection (low wind):
  h ≈ 5-10 W/(m²·K)

Forced convection:
  h ≈ 5.7 + 3.8 × v_wind  [W/(m²·K)]

Where v_wind is wind speed in m/s
```

**Conduction** (1D approximation):
```
Q_conduction = k × A × dT/dx

For surface layer:
  Q_conduction ≈ k × A × (T_surface - T_substrate) / Δx
```

### 4.3 Simplified Steady-State Solution

For quasi-steady conditions (slowly varying):

```
T_surface ≈ T_air + (Q_solar - Q_LW_net) / (h + h_rad)

Where:
  h_rad = 4 × ε × σ × T_air³  (linearized radiative coefficient)
```

---

## 5. Thermal Solver Implementation

### 5.1 Solver Hierarchy (Multi-Fidelity)

```
┌─────────────────────────────────────────────────────────────────┐
│                     THERMAL SOLVER FIDELITY                     │
└─────────────────────────────────────────────────────────────────┘

Level 1: Prescribed Temperature
├── User specifies T(x,y,t) directly
├── Fastest execution
└── Use: Quick tests, known thermal signatures

Level 2: Steady-State Balance
├── Solve energy balance at each timestep independently
├── No thermal mass effects
└── Use: Slowly varying scenes, daytime equilibrium

Level 3: 1D Transient
├── Solve 1D heat equation normal to each surface
├── Captures diurnal lag, thermal inertia
└── Use: Dawn/dusk, realistic dynamics

Level 4: 3D Transient (External)
├── Full 3D heat conduction (FEM/FDM)
├── Highest fidelity, slowest
└── Use: Complex structures, validation
```

### 5.2 1D Transient Thermal Model

Heat equation in 1D (depth z from surface):

```
∂T/∂t = α × ∂²T/∂z²

Boundary conditions:
  Surface (z=0): -k × ∂T/∂z = Q_net (net heat flux)
  Deep (z→∞):    T → T_deep (constant deep temperature)
```

**Numerical solution** (explicit finite difference):

```python
def solve_1d_thermal(
    T_initial: np.ndarray,     # Temperature profile [K]
    z_grid: np.ndarray,        # Depth grid [m]
    dt: float,                 # Time step [s]
    material: ThermalMaterial,
    Q_net_surface: float       # Net heat flux at surface [W/m²]
) -> np.ndarray:
    """
    Single timestep of 1D thermal diffusion.

    Returns:
        Updated temperature profile
    """
    T = T_initial.copy()
    dz = z_grid[1] - z_grid[0]
    alpha = material.diffusivity

    # Stability criterion: dt < dz² / (2α)
    assert dt < dz**2 / (2 * alpha), "Timestep too large for stability"

    # Interior points (explicit scheme)
    T[1:-1] = T[1:-1] + alpha * dt / dz**2 * (T[2:] - 2*T[1:-1] + T[:-2])

    # Surface boundary (flux condition)
    T[0] = T[1] + Q_net_surface * dz / material.thermal_conductivity

    # Deep boundary (fixed temperature)
    # T[-1] unchanged

    return T
```

### 5.3 Shadow Calculation

```python
def compute_shadow_factor(
    surface_point: np.ndarray,      # (x, y, z)
    surface_normal: np.ndarray,     # Unit normal
    sun_direction: np.ndarray,      # Unit vector toward sun
    scene_geometry: Scene
) -> float:
    """
    Compute shadow factor for a surface point.

    Returns:
        0.0 if fully shadowed
        1.0 if fully illuminated
        0.0-1.0 for partial shadow (soft shadows)
    """
    # Check if surface faces away from sun
    cos_theta = np.dot(surface_normal, sun_direction)
    if cos_theta <= 0:
        return 0.0  # Self-shadowed (facing away)

    # Ray trace toward sun to check for occluders
    ray_origin = surface_point + 0.001 * surface_normal  # Offset to avoid self-intersection
    hit = scene_geometry.ray_intersect(ray_origin, sun_direction)

    if hit is not None:
        return 0.0  # Blocked by geometry

    return 1.0  # Illuminated
```

---

## 6. Diurnal Temperature Cycle

### 6.1 Solar Position

```python
def compute_sun_position(
    latitude_deg: float,
    longitude_deg: float,
    datetime_utc: datetime
) -> Tuple[float, float]:
    """
    Compute sun azimuth and elevation.

    Returns:
        (azimuth_deg, elevation_deg)
        Azimuth: 0=North, 90=East, 180=South, 270=West
        Elevation: 0=horizon, 90=zenith
    """
    # Julian day calculation
    jd = compute_julian_day(datetime_utc)

    # Solar declination and hour angle
    declination = compute_declination(jd)
    hour_angle = compute_hour_angle(jd, longitude_deg)

    lat_rad = np.radians(latitude_deg)
    dec_rad = np.radians(declination)
    ha_rad = np.radians(hour_angle)

    # Elevation
    sin_elev = (np.sin(lat_rad) * np.sin(dec_rad) +
                np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad))
    elevation_deg = np.degrees(np.arcsin(sin_elev))

    # Azimuth
    cos_az = (np.sin(dec_rad) - np.sin(lat_rad) * sin_elev) / (np.cos(lat_rad) * np.cos(np.arcsin(sin_elev)))
    azimuth_deg = np.degrees(np.arccos(np.clip(cos_az, -1, 1)))
    if hour_angle > 0:
        azimuth_deg = 360 - azimuth_deg

    return azimuth_deg, elevation_deg
```

### 6.2 Typical Diurnal Patterns

```
Temperature
    │
    │                    ╭─────╮
 Tmax├──────────────────╱       ╲
    │                 ╱           ╲
    │               ╱               ╲
 Tavg├─────────────╱─────────────────╲─────────
    │           ╱                       ╲
    │         ╱                           ╲
 Tmin├───────╱                              ╲────
    │
    └────┬────┬────┬────┬────┬────┬────┬────┬────
         0    3    6    9   12   15   18   21   24  Hour
                   │              │
               Sunrise         Solar Noon

Key observations:
- T_min occurs just after sunrise (thermal lag)
- T_max occurs 2-3 hours after solar noon
- High thermal inertia = smaller ΔT, delayed peaks
- Low thermal inertia = larger ΔT, faster response
```

### 6.3 Thermal Crossover

Critical times when target-background contrast reverses:

```
Temperature
    │
    │     Target (metal)──────╮
    │    ╱                     ╲
    │   ╱    ╳ Crossover        ╲
    │  ╱    ╱ ╲                   ╲ ╳
    │ ╱   ╱     ╲     Background   ╲╱
    │╱  ╱         ─────(vegetation)──────
    ├─╱─────────────────────────────────────
    │
    └────┬────┬────┬────┬────┬────┬────┬────
         0    6   12   18   24  Hour
              │         │
          Morning    Evening
          crossover  crossover

Detection implications:
- Near crossover: Low contrast, difficult detection
- Away from crossover: Good thermal contrast
- MWIR vs LWIR may have different crossover times
```

---

## 7. Special Surface Types

### 7.1 Water Bodies

Water has unique thermal behavior:
- High thermal inertia → stable temperature
- Evaporative cooling
- Wind-driven mixing
- Specular reflection

```python
class WaterThermalModel:
    """Specialized thermal model for water surfaces."""

    def compute_temperature(
        self,
        depth_m: float,
        wind_speed_mps: float,
        air_temp_K: float,
        solar_irradiance_wm2: float
    ) -> float:
        # Simplified mixed-layer model
        # Surface temperature varies slowly, dominated by:
        # - Seasonal cycle (deep water)
        # - Diurnal cycle (shallow water)
        # - Wind mixing depth
        pass
```

### 7.2 Vegetation

Vegetation includes evapotranspiration:

```
Q_evap = L_v × E_t

Where:
  L_v = Latent heat of vaporization ≈ 2.45 × 10⁶ J/kg
  E_t = Evapotranspiration rate [kg/(m²·s)]

Effect: Vegetation is typically 5-15°C cooler than equivalent
non-transpiring surface during daytime.
```

### 7.3 Internal Heat Sources

For vehicles, buildings with internal loads:

```
Q_internal = P_internal  [W]

Examples:
- Running vehicle engine: 10-50 kW waste heat
- Building HVAC exhaust: 1-10 kW
- Human body: ~100 W
- Electronics: varies
```

---

## 8. View Factors and Radiative Exchange

### 8.1 View Factor Definition

The view factor F_ij is the fraction of radiation leaving surface i that reaches surface j:

```
F_ij = (1/A_i) × ∫∫ (cos θ_i × cos θ_j) / (π × r²) dA_i dA_j

Properties:
- Reciprocity: A_i × F_ij = A_j × F_ji
- Enclosure: Σ F_ij = 1 (for all j including self)
```

### 8.2 Common View Factors

**Horizontal surface to sky hemisphere**:
```
F_sky = (1 + cos β) / 2

Where β = surface tilt angle from horizontal
For horizontal surface: F_sky = 1.0
For vertical wall: F_sky = 0.5
```

**Surface to ground**:
```
F_ground = (1 - cos β) / 2
```

**Urban canyon** (infinite parallel walls):
```
F_sky = (1 - H/W) for H/W < 1
F_wall_opposite = H/W × [√(1 + (W/H)²) - 1]

Where H = building height, W = street width
```

### 8.3 Effective Sky Temperature

```
T_sky = ε_sky^(1/4) × T_air

Clear sky emissivity (Brunt equation):
  ε_sky ≈ 0.52 + 0.065 × √(e_a)

Where e_a = water vapor pressure [hPa]

Simplified:
  T_sky ≈ T_air - 20K (clear, dry)
  T_sky ≈ T_air - 10K (humid)
  T_sky ≈ T_air - 2K  (overcast)
```

---

## 9. Implementation Architecture

### 9.1 Thermal Module Structure

```python
# src/eosim/thermal/solver_base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ThermalState:
    """Thermal state of the scene at a given time."""
    time: datetime
    surface_temperatures: Dict[str, float]  # facet_id -> T_K
    subsurface_profiles: Dict[str, np.ndarray]  # facet_id -> T(z)

class ThermalSolver(ABC):
    """Abstract base class for thermal solvers."""

    @abstractmethod
    def initialize(self, scene: Scene, start_time: datetime) -> ThermalState:
        """Initialize thermal state."""
        pass

    @abstractmethod
    def step(self, state: ThermalState, dt: float, environment: Environment) -> ThermalState:
        """Advance thermal state by dt seconds."""
        pass

    @abstractmethod
    def get_surface_temperature(self, state: ThermalState, facet_id: str) -> float:
        """Get temperature of specific surface."""
        pass


class PrescribedThermal(ThermalSolver):
    """Level 1: User-prescribed temperatures."""
    pass

class SteadyStateThermal(ThermalSolver):
    """Level 2: Instantaneous energy balance."""
    pass

class Transient1DThermal(ThermalSolver):
    """Level 3: 1D transient with thermal mass."""
    pass

class ExternalThermal(ThermalSolver):
    """Level 4: Interface to external solver (e.g., FEM)."""
    pass
```

### 9.2 Environment State

```python
@dataclass
class Environment:
    """Environmental conditions affecting thermal balance."""

    # Solar
    sun_azimuth_deg: float
    sun_elevation_deg: float
    solar_irradiance_wm2: float  # Direct normal
    diffuse_fraction: float       # Fraction of irradiance that's diffuse

    # Atmospheric
    air_temperature_K: float
    sky_temperature_K: float
    relative_humidity: float

    # Wind
    wind_speed_mps: float
    wind_direction_deg: float

    # Derived
    @property
    def convection_coefficient(self) -> float:
        """Forced convection heat transfer coefficient [W/(m²·K)]."""
        return 5.7 + 3.8 * self.wind_speed_mps
```

---

## 10. Summary

### Key Concepts

1. **Geometry**: Triangle meshes for objects, DEMs for terrain, hierarchical scene graph
2. **Materials**: Dual properties - radiometric (ε, ρ) and thermophysical (k, c_p, ρ)
3. **Energy Balance**: Solar + LW_in = LW_out + convection + conduction + storage
4. **Thermal Solvers**: Multi-fidelity from prescribed to full 3D transient
5. **Diurnal Cycle**: Critical for realistic IR simulation, thermal crossover times
6. **View Factors**: Determine radiative exchange between surfaces and sky

### Implementation Priority

1. **Prescribed temperatures** - Fastest path to initial results
2. **Steady-state balance** - Reasonable daytime accuracy
3. **1D transient** - Captures thermal inertia, essential for dawn/dusk
4. **View factor calculation** - Important for urban canyons
5. **Special models** - Water, vegetation, internal sources

---

## 11. References

1. **DIRSIG Thermodynamics Documentation** - Thermal prediction methodology
2. **Kottler et al.** (2017) - Urban thermal modeling for DIRSIG
3. **Incropera & DeWitt** (2007) - "Fundamentals of Heat and Mass Transfer"
4. **Stull** (1988) - "An Introduction to Boundary Layer Meteorology"
5. **Oke** (1987) - "Boundary Layer Climates" - Urban thermal environment
