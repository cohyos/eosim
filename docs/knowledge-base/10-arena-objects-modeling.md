# Arena Objects Physical Modeling

## 1. Introduction

This document covers the physical modeling of objects within the EOSIM arena: vehicles, aircraft, buildings, people, vegetation, and other targets. Each object type has unique geometric, thermal, and radiometric characteristics that must be accurately modeled for realistic EO/IR simulation.

---

## 2. Object Taxonomy

### 2.1 Object Categories

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ARENA OBJECT TAXONOMY                               │
└─────────────────────────────────────────────────────────────────────────────┘

STATIC OBJECTS                          DYNAMIC OBJECTS
├── Terrain                             ├── Ground Vehicles
│   ├── Natural terrain (DEM)           │   ├── Wheeled (cars, trucks, APCs)
│   ├── Roads/Surfaces                  │   ├── Tracked (tanks, IFVs)
│   └── Water bodies                    │   └── Motorcycles
│                                       │
├── Buildings                           ├── Aircraft
│   ├── Residential                     │   ├── Fixed-wing
│   ├── Commercial                      │   ├── Rotary-wing
│   ├── Industrial                      │   └── UAVs
│   └── Military structures             │
│                                       ├── Watercraft
├── Vegetation                          │   ├── Ships
│   ├── Trees (deciduous, coniferous)   │   └── Small boats
│   ├── Shrubs                          │
│   └── Grass/Crops                     ├── People
│                                       │   ├── Standing/Walking
└── Infrastructure                      │   ├── Running
    ├── Bridges                         │   └── Prone/Crouching
    ├── Power lines                     │
    └── Fences                          └── Projectiles
                                            ├── Missiles
                                            └── Munitions
```

### 2.2 Object Properties Schema

```python
@dataclass
class ArenaObject:
    """Base class for all arena objects."""

    # Identity
    id: str
    name: str
    category: str                    # vehicle, building, person, etc.
    subcategory: str                 # tank, sedan, residential, etc.

    # Geometry
    geometry: ObjectGeometry
    bounding_box: BoundingBox
    level_of_detail: List[LODLevel]

    # Material/Radiometric
    materials: Dict[str, Material]   # Named material zones
    thermal_model: ThermalModel

    # State
    state: ObjectState               # Engine on/off, doors, etc.

    # Dynamics (for mobile objects)
    trajectory: Optional[Trajectory]
    articulation: Optional[Articulation]
```

---

## 3. Geometry Representation

### 3.1 Supported Formats

| Format | Extension | Use Case | Notes |
|--------|-----------|----------|-------|
| Wavefront OBJ | `.obj` | General meshes | Simple, widely supported |
| glTF/GLB | `.gltf`, `.glb` | Complex models | PBR materials, animations |
| FBX | `.fbx` | CAD-derived | Autodesk standard |
| STL | `.stl` | Simple shapes | No materials |
| OpenFlight | `.flt` | Simulation | Military/training standard |

### 3.2 Geometry Structure

```python
@dataclass
class ObjectGeometry:
    """3D geometry representation."""

    # Mesh data
    vertices: np.ndarray          # (N, 3) vertex positions
    faces: np.ndarray             # (M, 3) triangle indices
    normals: np.ndarray           # (N, 3) vertex normals
    uvs: Optional[np.ndarray]     # (N, 2) texture coordinates

    # Material assignment
    face_materials: np.ndarray    # (M,) material index per face
    material_names: List[str]     # Material name list

    # Coordinate system
    origin: np.ndarray            # Object origin in local coords
    up_axis: str = "Z"            # Z-up or Y-up
    units: str = "meters"


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    min_corner: np.ndarray        # (x_min, y_min, z_min)
    max_corner: np.ndarray        # (x_max, y_max, z_max)

    @property
    def dimensions(self) -> np.ndarray:
        return self.max_corner - self.min_corner

    @property
    def center(self) -> np.ndarray:
        return (self.min_corner + self.max_corner) / 2
```

### 3.3 Level of Detail (LOD)

```python
@dataclass
class LODLevel:
    """Single level of detail."""
    level: int                    # 0 = highest detail
    geometry: ObjectGeometry
    switch_distance_m: float      # Distance to switch to this LOD
    face_count: int

# Example LOD configuration for a vehicle
vehicle_lods = [
    LODLevel(0, high_detail_mesh, 100, 50000),    # < 100m: full detail
    LODLevel(1, medium_mesh, 500, 10000),         # 100-500m: medium
    LODLevel(2, low_mesh, 2000, 2000),            # 500-2000m: low
    LODLevel(3, billboard, float('inf'), 2),      # > 2000m: billboard
]
```

---

## 4. Material Zones

### 4.1 Material Zone Concept

Objects are divided into **material zones** with distinct radiometric and thermal properties:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VEHICLE MATERIAL ZONES (Example: Tank)                   │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │     Turret      │ ← Painted steel
                    │  ┌───────────┐  │
                    │  │   Gun     │  │ ← Bare metal (hot after firing)
                    │  └───────────┘  │
            ┌───────┴─────────────────┴───────┐
            │              Hull               │ ← Painted steel
            │  ┌─────┐              ┌─────┐  │
            │  │Exhaust│            │Engine│  │ ← HOT zones
            │  └─────┘              └─────┘  │
            ├─────────────────────────────────┤
            │           Tracks               │ ← Rubber/metal
            └─────────────────────────────────┘

Material zones for this tank:
  - hull_painted: ε=0.90, T=ambient+5K
  - turret_painted: ε=0.90, T=ambient+3K
  - engine_bay: ε=0.85, T=ambient+80K (running)
  - exhaust: ε=0.70, T=ambient+150K (running)
  - tracks_rubber: ε=0.95, T=ambient+20K (moving)
  - gun_barrel: ε=0.30 (bare metal), T=ambient+200K (after firing)
  - optics_glass: ε=0.10 (reflective)
```

### 4.2 Material Zone Definition

```python
@dataclass
class MaterialZone:
    """Material properties for an object zone."""

    name: str
    face_indices: np.ndarray      # Which faces belong to this zone

    # Radiometric properties
    emissivity: Union[float, SpectralCurve]
    reflectivity: Union[float, SpectralCurve]
    brdf_model: str = "lambertian"
    brdf_params: dict = field(default_factory=dict)

    # Thermal properties
    thermal_conductivity: float   # W/(m·K)
    specific_heat: float          # J/(kg·K)
    density: float                # kg/m³
    thickness: float              # m (for thin-shell model)

    # Temperature offset/source
    temperature_mode: str = "computed"  # computed, offset, fixed, internal_source
    temperature_offset_K: float = 0.0
    internal_heat_W: float = 0.0
```

### 4.3 Standard Material Library

```yaml
# materials/standard_library.yaml

materials:
  # Paints
  olive_drab_paint:
    emissivity: 0.90
    reflectivity: 0.10
    solar_absorptivity: 0.85
    thermal_conductivity: 0.5
    specific_heat: 500
    density: 1500

  desert_tan_paint:
    emissivity: 0.88
    reflectivity: 0.12
    solar_absorptivity: 0.70

  urban_gray_paint:
    emissivity: 0.90
    reflectivity: 0.10
    solar_absorptivity: 0.60

  # Metals
  steel_bare:
    emissivity: 0.20
    reflectivity: 0.80
    thermal_conductivity: 50
    specific_heat: 500
    density: 7800

  aluminum_bare:
    emissivity: 0.05
    reflectivity: 0.95
    thermal_conductivity: 205
    specific_heat: 900
    density: 2700

  # Rubber/Plastics
  tire_rubber:
    emissivity: 0.95
    reflectivity: 0.05
    thermal_conductivity: 0.15
    specific_heat: 2000
    density: 1100

  # Glass
  window_glass:
    emissivity_lwir: 0.90      # Opaque in LWIR
    emissivity_mwir: 0.85
    reflectivity_vis: 0.08
    transmissivity_vis: 0.90

  # Natural
  human_skin:
    emissivity: 0.98
    temperature_fixed_K: 306   # ~33°C skin temperature

  vegetation_leaf:
    emissivity: 0.96
    reflectivity_nir: 0.50     # High NIR reflectance
    transpiration_rate: 0.3
```

---

## 5. Ground Vehicle Modeling

### 5.1 Vehicle Thermal Signature Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   GROUND VEHICLE THERMAL SIGNATURE                          │
└─────────────────────────────────────────────────────────────────────────────┘

HEAT SOURCES:
  ├── Engine
  │   ├── Block temperature: 80-120°C (running)
  │   ├── Exhaust manifold: 200-400°C
  │   └── Radiator: 60-90°C
  │
  ├── Exhaust System
  │   ├── Exhaust pipe: 150-300°C
  │   ├── Muffler: 100-200°C
  │   └── Exhaust plume: 200-600°C (gas)
  │
  ├── Drivetrain
  │   ├── Transmission: 60-100°C
  │   ├── Differential: 50-80°C
  │   └── Brakes: 50-300°C (depending on use)
  │
  ├── Tracks/Wheels
  │   ├── Friction heating: +10-30°C above ambient
  │   └── Track pads (rubber): higher than metal
  │
  └── Crew/Passengers
      └── Body heat through hatches/windows

HEAT SINKS:
  ├── Painted surfaces (radiative cooling)
  ├── Airflow (convective cooling)
  └── Ground contact (conductive)
```

### 5.2 Vehicle State Model

```python
@dataclass
class VehicleState:
    """Dynamic state of a ground vehicle."""

    # Engine state
    engine_running: bool = False
    engine_rpm: float = 0.0
    engine_load_percent: float = 0.0
    time_since_engine_start_s: float = 0.0
    time_since_engine_stop_s: float = float('inf')

    # Motion state
    speed_mps: float = 0.0
    acceleration_mps2: float = 0.0

    # Component states
    brakes_applied: bool = False
    brake_temperature_K: float = 300.0
    headlights_on: bool = False

    # Articulation
    turret_azimuth_deg: float = 0.0
    gun_elevation_deg: float = 0.0
    hatches_open: List[str] = field(default_factory=list)


class VehicleThermalModel:
    """Thermal model for ground vehicles."""

    def __init__(self, vehicle_type: str, thermal_params: dict):
        self.vehicle_type = vehicle_type
        self.params = thermal_params

    def compute_zone_temperatures(
        self,
        state: VehicleState,
        ambient_temp_K: float,
        solar_load_wm2: float,
        wind_speed_mps: float,
        time_of_day: datetime
    ) -> Dict[str, float]:
        """
        Compute temperature for each material zone.

        Returns dict of zone_name -> temperature_K
        """
        temps = {}

        # Engine bay temperature
        if state.engine_running:
            # Warm-up curve
            warmup_factor = 1 - np.exp(-state.time_since_engine_start_s / 300)
            engine_delta = self.params['engine_delta_K'] * warmup_factor * (0.5 + 0.5 * state.engine_load_percent / 100)
            temps['engine_bay'] = ambient_temp_K + engine_delta
        else:
            # Cool-down curve
            cooldown_factor = np.exp(-state.time_since_engine_stop_s / 600)
            temps['engine_bay'] = ambient_temp_K + self.params['engine_delta_K'] * cooldown_factor

        # Exhaust temperature
        if state.engine_running:
            exhaust_delta = self.params['exhaust_delta_K'] * (0.3 + 0.7 * state.engine_load_percent / 100)
            temps['exhaust'] = ambient_temp_K + exhaust_delta
        else:
            temps['exhaust'] = ambient_temp_K

        # Track/wheel temperature (friction heating)
        friction_heat = 0.1 * state.speed_mps**2  # Simplified model
        temps['tracks'] = ambient_temp_K + min(friction_heat, 30)

        # Brake temperature
        if state.brakes_applied:
            temps['brakes'] = state.brake_temperature_K
        else:
            temps['brakes'] = ambient_temp_K + 10

        # Hull and turret (solar heated + engine conducted)
        hull_solar = solar_load_wm2 * self.params['solar_absorptivity'] / (5.7 + 3.8 * wind_speed_mps)
        engine_conducted = (temps.get('engine_bay', ambient_temp_K) - ambient_temp_K) * 0.1
        temps['hull'] = ambient_temp_K + hull_solar + engine_conducted
        temps['turret'] = temps['hull'] - 2  # Slightly cooler (less engine heat)

        return temps
```

### 5.3 Vehicle Database Entry

```yaml
# vehicles/m1_abrams.yaml

vehicle:
  name: "M1A2 Abrams"
  category: "ground_vehicle"
  subcategory: "main_battle_tank"

  geometry:
    model_file: "models/m1_abrams.glb"
    scale: 1.0
    origin_offset: [0, 0, 0]
    bounding_box:
      length_m: 9.83
      width_m: 3.66
      height_m: 2.44

  material_zones:
    - name: "hull"
      mesh_groups: ["Hull", "Fenders"]
      material: "olive_drab_paint"
      area_m2: 45.0

    - name: "turret"
      mesh_groups: ["Turret", "Mantlet"]
      material: "olive_drab_paint"
      area_m2: 20.0

    - name: "engine_bay"
      mesh_groups: ["EngineDeck", "Grilles"]
      material: "steel_painted"
      area_m2: 8.0
      internal_source: true

    - name: "exhaust"
      mesh_groups: ["Exhaust_L", "Exhaust_R"]
      material: "bare_steel"
      area_m2: 0.5
      internal_source: true

    - name: "tracks"
      mesh_groups: ["Track_L", "Track_R", "RoadWheels"]
      material: "track_rubber_steel"
      area_m2: 15.0

    - name: "gun"
      mesh_groups: ["MainGun"]
      material: "bare_steel"
      area_m2: 3.0

  thermal_model:
    type: "vehicle_parametric"
    params:
      engine_delta_K: 80           # Max engine bay temp rise
      exhaust_delta_K: 200         # Max exhaust temp rise
      engine_warmup_time_s: 300
      engine_cooldown_time_s: 600
      solar_absorptivity: 0.85
      mass_kg: 62000
      engine_power_kW: 1100

  articulation:
    - joint: "turret_rotation"
      axis: [0, 0, 1]
      range_deg: [-180, 180]
      rate_deg_s: 40

    - joint: "gun_elevation"
      axis: [1, 0, 0]
      parent: "turret_rotation"
      range_deg: [-10, 20]
      rate_deg_s: 20

    - joint: "commander_hatch"
      type: "hinge"
      range_deg: [0, 90]

  signatures:
    radar_cross_section_m2: 20.0
    acoustic_signature_db: 95
```

---

## 6. Aircraft Modeling

### 6.1 Aircraft Thermal Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AIRCRAFT THERMAL SIGNATURE                             │
└─────────────────────────────────────────────────────────────────────────────┘

PRIMARY HEAT SOURCES:
  ├── Engine(s)
  │   ├── Turbine: 400-700°C
  │   ├── Exhaust nozzle: 300-600°C
  │   └── Exhaust plume: 200-1000°C (afterburner: 1500°C+)
  │
  ├── Aerodynamic Heating
  │   ├── Leading edges: Function of Mach number
  │   ├── Stagnation points: T_stag = T_ambient × (1 + 0.2 × M²)
  │   └── Skin friction heating
  │
  └── Avionics/Systems
      ├── Radar: 50-100°C (aperture)
      └── EO/IR sensors: Cooled or ambient

COOLING EFFECTS:
  ├── Airflow (dominant at cruise)
  ├── Fuel cooling (internal)
  └── Radiative to sky
```

### 6.2 Aircraft Model

```python
@dataclass
class AircraftState:
    """Dynamic state of an aircraft."""

    # Flight state
    altitude_m: float
    airspeed_mps: float
    mach_number: float
    angle_of_attack_deg: float

    # Engine state
    engine_power_percent: float   # 0-100, >100 for afterburner
    afterburner_on: bool = False

    # Control surfaces
    flap_position_deg: float = 0
    gear_down: bool = False
    bay_doors_open: bool = False


class AircraftThermalModel:
    """Thermal model for aircraft."""

    def compute_zone_temperatures(
        self,
        state: AircraftState,
        ambient_temp_K: float
    ) -> Dict[str, float]:
        temps = {}

        # Aerodynamic heating (simplified)
        gamma = 1.4  # Air specific heat ratio
        recovery_factor = 0.9
        T_recovery = ambient_temp_K * (1 + recovery_factor * (gamma - 1) / 2 * state.mach_number**2)

        temps['skin'] = T_recovery
        temps['leading_edge'] = ambient_temp_K * (1 + (gamma - 1) / 2 * state.mach_number**2)

        # Engine/exhaust
        if state.afterburner_on:
            temps['exhaust_nozzle'] = 800 + 273  # ~800°C
            temps['exhaust_plume'] = 1500 + 273
        else:
            exhaust_base = 300 + state.engine_power_percent * 3  # 300-600°C
            temps['exhaust_nozzle'] = exhaust_base + 273
            temps['exhaust_plume'] = exhaust_base * 0.7 + 273

        return temps
```

---

## 7. Building Modeling

### 7.1 Building Thermal Zones

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BUILDING THERMAL MODELING                              │
└─────────────────────────────────────────────────────────────────────────────┘

EXTERIOR:
  ├── Roof
  │   ├── Material (concrete, metal, tile)
  │   ├── Solar loading (dominant daytime)
  │   └── Radiative cooling (nighttime)
  │
  ├── Walls
  │   ├── Sun-facing: Higher temperature
  │   ├── Shaded: Lower temperature
  │   └── Material thermal mass
  │
  └── Windows
      ├── LWIR: Opaque, reflects interior
      ├── MWIR: Partially transparent
      └── VIS: Transparent (interior visible)

INTERIOR EFFECTS:
  ├── HVAC
  │   ├── Heating: Interior warmer
  │   └── Cooling: Interior cooler
  │
  ├── Internal loads
  │   ├── Occupants
  │   ├── Equipment
  │   └── Lighting
  │
  └── Thermal bridges
      └── Heat leaks at junctions
```

### 7.2 Building Model

```python
@dataclass
class BuildingState:
    """State of a building."""

    # Occupancy
    occupied: bool = True
    occupant_count: int = 0

    # HVAC
    hvac_mode: str = "off"        # off, heating, cooling
    interior_setpoint_K: float = 294  # 21°C
    hvac_power_W: float = 0

    # Lighting
    lights_on: bool = False

    # Doors/windows
    doors_open: List[str] = field(default_factory=list)
    windows_open: List[str] = field(default_factory=list)


class BuildingThermalModel:
    """Thermal model for buildings."""

    def compute_surface_temperatures(
        self,
        state: BuildingState,
        ambient_temp_K: float,
        solar_irradiance: Dict[str, float],  # Per-surface solar load
        wind_speed_mps: float,
        sky_temp_K: float
    ) -> Dict[str, float]:
        """
        Compute exterior surface temperatures.
        """
        temps = {}

        for surface_name, surface in self.surfaces.items():
            # Solar heating
            Q_solar = solar_irradiance.get(surface_name, 0) * surface.solar_absorptivity

            # Convective cooling
            h_conv = 5.7 + 3.8 * wind_speed_mps

            # Radiative to sky (for roof/upper surfaces)
            if surface.view_factor_sky > 0.5:
                Q_rad_sky = surface.emissivity * 5.67e-8 * (ambient_temp_K**4 - sky_temp_K**4)
            else:
                Q_rad_sky = 0

            # Interior heat flow
            if state.hvac_mode == "heating":
                Q_interior = surface.u_value * (state.interior_setpoint_K - ambient_temp_K)
            elif state.hvac_mode == "cooling":
                Q_interior = surface.u_value * (state.interior_setpoint_K - ambient_temp_K)
            else:
                Q_interior = 0

            # Steady-state surface temperature
            T_surface = ambient_temp_K + (Q_solar - Q_rad_sky + Q_interior) / h_conv
            temps[surface_name] = T_surface

        return temps
```

---

## 8. Human/Personnel Modeling

### 8.1 Human Thermal Signature

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      HUMAN THERMAL SIGNATURE                                │
└─────────────────────────────────────────────────────────────────────────────┘

BODY REGIONS:
  ├── Head/Face
  │   ├── Exposed skin: 33-35°C
  │   ├── Hair-covered: 30-32°C
  │   └── Highest contrast region
  │
  ├── Torso
  │   ├── Clothed: 28-32°C (depends on clothing)
  │   └── Bare: 33-34°C
  │
  ├── Arms/Hands
  │   ├── Hands exposed: 30-34°C
  │   └── Arms clothed: 28-32°C
  │
  └── Legs/Feet
      ├── Feet in boots: 25-30°C
      └── Legs clothed: 28-32°C

ACTIVITY EFFECTS:
  ├── Resting: Baseline temperatures
  ├── Walking: +1-2°C
  ├── Running: +3-5°C, visible perspiration
  └── Physical exertion: +5-10°C
```

### 8.2 Human Model

```python
@dataclass
class HumanState:
    """State of a human target."""

    # Activity
    activity: str = "standing"    # standing, walking, running, prone, crouching
    activity_intensity: float = 0.5  # 0-1

    # Clothing
    clothing_type: str = "light"  # light, medium, heavy, tactical
    head_covered: bool = False
    hands_covered: bool = False

    # Carried equipment
    carrying_weapon: bool = False
    backpack: bool = False


@dataclass
class HumanThermalProfile:
    """Temperature profile for human body regions."""

    head_face_K: float = 307      # ~34°C
    head_hair_K: float = 304      # ~31°C
    neck_K: float = 306           # ~33°C
    torso_K: float = 305          # ~32°C (clothed)
    arms_K: float = 303           # ~30°C (clothed)
    hands_K: float = 305          # ~32°C
    legs_K: float = 302           # ~29°C (clothed)
    feet_K: float = 300           # ~27°C (in shoes)


class HumanThermalModel:
    """Thermal model for human targets."""

    CLOTHING_INSULATION = {
        'light': 0.3,      # T-shirt, shorts
        'medium': 0.5,     # Long sleeves, pants
        'heavy': 0.8,      # Jacket, winter clothing
        'tactical': 0.6,   # Military gear
    }

    def compute_apparent_temperatures(
        self,
        state: HumanState,
        ambient_temp_K: float,
        wind_speed_mps: float
    ) -> HumanThermalProfile:
        """Compute apparent temperatures for each body region."""

        # Core body temperature
        T_core = 310  # 37°C

        # Activity-induced heating
        activity_heat = {
            'standing': 0,
            'walking': 2,
            'running': 5,
            'prone': -1,      # Less activity
            'crouching': 1,
        }
        T_activity_offset = activity_heat.get(state.activity, 0) * state.activity_intensity

        # Clothing insulation effect
        insulation = self.CLOTHING_INSULATION.get(state.clothing_type, 0.5)

        # Wind chill effect on exposed skin
        wind_chill = 2 * np.sqrt(wind_speed_mps)

        profile = HumanThermalProfile()

        # Exposed regions (face, hands if uncovered)
        profile.head_face_K = T_core - 3 + T_activity_offset - wind_chill
        profile.hands_K = T_core - 5 + T_activity_offset - (0 if state.hands_covered else wind_chill)

        # Clothed regions
        clothing_reduction = insulation * (T_core - ambient_temp_K) * 0.3
        profile.torso_K = T_core - 5 - clothing_reduction + T_activity_offset
        profile.legs_K = T_core - 8 - clothing_reduction + T_activity_offset

        return profile
```

---

## 9. Vegetation Modeling

### 9.1 Vegetation Thermal Behavior

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VEGETATION THERMAL BEHAVIOR                            │
└─────────────────────────────────────────────────────────────────────────────┘

KEY CHARACTERISTICS:
  ├── Evapotranspiration
  │   ├── Dominant cooling mechanism
  │   ├── Rate depends on: solar load, humidity, wind, soil moisture
  │   └── Typically 5-15°C cooler than equivalent non-vegetated surface
  │
  ├── Low Thermal Inertia
  │   ├── Rapid response to solar changes
  │   ├── Quick cool-down at sunset
  │   └── Close to air temperature at night
  │
  ├── High Emissivity
  │   ├── ε ≈ 0.96-0.98 (LWIR)
  │   └── Near-blackbody behavior
  │
  └── Spectral Reflectance
      ├── VIS: Green peak ~550nm, low red
      ├── NIR: "Red edge" ~700nm, high reflectance >750nm
      └── SWIR: Water absorption features
```

### 9.2 Vegetation Model

```python
@dataclass
class VegetationState:
    """State of vegetation."""

    vegetation_type: str          # deciduous, coniferous, grass, crop
    health: float = 1.0           # 0-1, affects NIR reflectance
    moisture_content: float = 0.7 # 0-1

    # Seasonal
    leaf_area_index: float = 3.0
    canopy_closure: float = 0.8


class VegetationThermalModel:
    """Thermal model for vegetation."""

    def compute_canopy_temperature(
        self,
        state: VegetationState,
        ambient_temp_K: float,
        solar_irradiance_wm2: float,
        relative_humidity: float,
        wind_speed_mps: float,
        soil_moisture: float
    ) -> float:
        """
        Compute vegetation canopy temperature.

        Uses simplified Penman-Monteith evapotranspiration model.
        """
        # Available energy
        Q_net = solar_irradiance_wm2 * (1 - 0.2)  # 20% albedo

        # Stomatal conductance (simplified)
        g_s = 0.01 * state.moisture_content * state.health  # m/s

        # Aerodynamic conductance
        g_a = 0.01 * np.sqrt(wind_speed_mps)  # m/s

        # Vapor pressure deficit
        T_air_C = ambient_temp_K - 273.15
        e_sat = 0.611 * np.exp(17.27 * T_air_C / (T_air_C + 237.3))  # kPa
        vpd = e_sat * (1 - relative_humidity)  # kPa

        # Latent heat flux (evapotranspiration)
        lambda_v = 2.45e6  # J/kg latent heat
        rho_air = 1.2  # kg/m³
        cp_air = 1004  # J/(kg·K)
        gamma = 0.066  # kPa/K psychrometric constant

        # Penman-Monteith (simplified)
        delta = 4098 * e_sat / (T_air_C + 237.3)**2
        ET = (delta * Q_net + rho_air * cp_air * vpd * g_a) / (delta + gamma * (1 + g_a / g_s))
        ET = max(ET, 0)  # Can't be negative

        # Canopy temperature
        # T_canopy = T_air + (Q_net - ET) / (rho_air * cp_air * g_a)
        sensible_heat = Q_net - ET
        T_canopy = ambient_temp_K + sensible_heat / (rho_air * cp_air * g_a)

        return T_canopy
```

---

## 10. Object Dynamics

### 10.1 Trajectory Specification

```python
@dataclass
class Trajectory:
    """Object trajectory over time."""

    trajectory_type: str           # waypoints, spline, parametric, external

    # For waypoint-based
    waypoints: List[TrajectoryWaypoint] = None

    # For external file
    trajectory_file: str = None
    file_format: str = "csv"

    # Interpolation
    interpolation: str = "cubic"   # linear, cubic, hermite


@dataclass
class TrajectoryWaypoint:
    """Single waypoint in trajectory."""

    time_s: float
    position: np.ndarray           # (x, y, z) in arena coordinates
    velocity: Optional[np.ndarray] = None
    attitude: Optional[np.ndarray] = None  # (roll, pitch, yaw) degrees


class TrajectoryInterpolator:
    """Interpolate trajectory between waypoints."""

    def __init__(self, trajectory: Trajectory):
        self.trajectory = trajectory
        self._build_interpolators()

    def get_state(self, time_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get position, velocity, attitude at time.

        Returns:
            (position, velocity, attitude)
        """
        position = self._pos_interp(time_s)
        velocity = self._vel_interp(time_s)
        attitude = self._att_interp(time_s)

        return position, velocity, attitude
```

### 10.2 Articulation

```python
@dataclass
class ArticulationJoint:
    """Single articulation joint."""

    name: str
    joint_type: str               # revolute, prismatic
    axis: np.ndarray              # Rotation/translation axis
    parent: Optional[str] = None  # Parent joint name
    range_min: float = 0
    range_max: float = 0
    current_value: float = 0
    rate: float = 0               # Current rate of change


class ArticulatedObject:
    """Object with articulated parts."""

    def __init__(self, joints: List[ArticulationJoint]):
        self.joints = {j.name: j for j in joints}
        self._build_kinematic_chain()

    def set_joint_angle(self, joint_name: str, angle: float):
        """Set angle for a revolute joint."""
        joint = self.joints[joint_name]
        joint.current_value = np.clip(angle, joint.range_min, joint.range_max)

    def get_transform(self, part_name: str) -> np.ndarray:
        """Get 4x4 transform matrix for a part."""
        pass

    def animate_to(self, joint_name: str, target: float, duration_s: float):
        """Animate joint to target over duration."""
        pass
```

---

## 11. Summary

### Object Modeling Requirements

| Object Type | Geometry | Materials | Thermal | Dynamics |
|-------------|----------|-----------|---------|----------|
| Ground vehicles | Multi-LOD mesh | 5-10 zones | Engine, exhaust, tracks | Trajectory, articulation |
| Aircraft | Detailed mesh | 5-8 zones | Engine, aero heating | 6DOF trajectory |
| Buildings | Simple mesh | 3-5 zones | HVAC, solar | Static |
| People | Articulated mesh | 5-8 zones | Body regions | Walk/run paths |
| Vegetation | Billboards/mesh | 1-2 zones | Evapotranspiration | Wind sway |

### Implementation Priority

1. **Material zone system** - Foundation for all objects
2. **Ground vehicles** - Primary targets for many scenarios
3. **Buildings** - Static background
4. **People** - Small but important targets
5. **Aircraft** - Complex thermal signatures
6. **Vegetation** - Environmental realism

### Key Considerations

- Material zones drive thermal appearance
- State-dependent temperatures essential for realism
- LOD system needed for large scenes
- Database of validated signatures valuable

---

## 12. References

1. **PRISM** (1988) - US Army thermal signature prediction
2. **MuSES** - ThermoAnalytics vehicle thermal modeling
3. **DIRSIG** - Target signature documentation
4. **NVESD Sensors Model** - Target acquisition
5. **Holst** (2008) - IR target signatures
