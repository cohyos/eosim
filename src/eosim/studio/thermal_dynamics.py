"""
Thermal dynamics simulation for time-varying material temperatures.

This module provides physically-based thermal modeling including:
- Solar heating and radiative cooling
- Thermal mass and time constants
- Material thermal properties database
- Diurnal temperature cycles
- Engine/exhaust thermal signatures
- Shadowing effects
- Wind cooling (convection)
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

# Stefan-Boltzmann constant
STEFAN_BOLTZMANN = 5.670374419e-8  # W/m²/K⁴


class MaterialCategory(Enum):
    """Material categories for thermal properties."""
    METAL = "metal"
    CONCRETE = "concrete"
    ASPHALT = "asphalt"
    SOIL = "soil"
    VEGETATION = "vegetation"
    WATER = "water"
    GLASS = "glass"
    PLASTIC = "plastic"
    RUBBER = "rubber"
    FABRIC = "fabric"
    COMPOSITE = "composite"
    CERAMIC = "ceramic"
    PAINT = "paint"


@dataclass
class ThermalMaterialProperties:
    """Thermal properties for a material type."""
    name: str
    category: MaterialCategory

    # Optical properties
    emissivity: float = 0.9           # Thermal emissivity (0-1)
    solar_absorptivity: float = 0.5   # Solar absorption (0-1)
    reflectivity_lwir: float = 0.1    # LWIR reflectivity

    # Thermal properties
    thermal_conductivity: float = 1.0  # W/(m·K)
    specific_heat: float = 1000.0      # J/(kg·K)
    density: float = 2000.0            # kg/m³

    # Derived: thermal diffusivity = k / (ρ × c)
    @property
    def thermal_diffusivity(self) -> float:
        return self.thermal_conductivity / (self.density * self.specific_heat)

    # Effective thickness for surface temperature calculation
    effective_thickness_m: float = 0.05  # 5 cm default

    @property
    def thermal_mass(self) -> float:
        """Thermal mass per unit area (J/(m²·K))."""
        return self.density * self.specific_heat * self.effective_thickness_m

    @property
    def time_constant(self) -> float:
        """Thermal time constant (seconds)."""
        # τ = ρ × c × d / h, where h is heat transfer coefficient
        # Simplified: assuming h ≈ 10 W/(m²·K) for natural convection
        h = 10.0
        return self.thermal_mass / h


# Standard material properties database
MATERIAL_DATABASE: Dict[str, ThermalMaterialProperties] = {
    # Metals
    "steel_painted": ThermalMaterialProperties(
        name="Painted Steel",
        category=MaterialCategory.METAL,
        emissivity=0.85,
        solar_absorptivity=0.4,
        thermal_conductivity=50.0,
        specific_heat=500.0,
        density=7800.0,
        effective_thickness_m=0.005
    ),
    "steel_bare": ThermalMaterialProperties(
        name="Bare Steel",
        category=MaterialCategory.METAL,
        emissivity=0.25,
        solar_absorptivity=0.6,
        thermal_conductivity=50.0,
        specific_heat=500.0,
        density=7800.0,
        effective_thickness_m=0.005
    ),
    "aluminum": ThermalMaterialProperties(
        name="Aluminum",
        category=MaterialCategory.METAL,
        emissivity=0.15,
        solar_absorptivity=0.4,
        thermal_conductivity=200.0,
        specific_heat=900.0,
        density=2700.0,
        effective_thickness_m=0.003
    ),
    "aluminum_anodized": ThermalMaterialProperties(
        name="Anodized Aluminum",
        category=MaterialCategory.METAL,
        emissivity=0.77,
        solar_absorptivity=0.65,
        thermal_conductivity=200.0,
        specific_heat=900.0,
        density=2700.0,
        effective_thickness_m=0.003
    ),

    # Construction materials
    "concrete": ThermalMaterialProperties(
        name="Concrete",
        category=MaterialCategory.CONCRETE,
        emissivity=0.92,
        solar_absorptivity=0.6,
        thermal_conductivity=1.5,
        specific_heat=880.0,
        density=2300.0,
        effective_thickness_m=0.10
    ),
    "asphalt": ThermalMaterialProperties(
        name="Asphalt",
        category=MaterialCategory.ASPHALT,
        emissivity=0.95,
        solar_absorptivity=0.85,
        thermal_conductivity=0.75,
        specific_heat=920.0,
        density=2100.0,
        effective_thickness_m=0.05
    ),
    "brick": ThermalMaterialProperties(
        name="Brick",
        category=MaterialCategory.CERAMIC,
        emissivity=0.93,
        solar_absorptivity=0.7,
        thermal_conductivity=0.7,
        specific_heat=840.0,
        density=1800.0,
        effective_thickness_m=0.10
    ),

    # Natural materials
    "soil_dry": ThermalMaterialProperties(
        name="Dry Soil",
        category=MaterialCategory.SOIL,
        emissivity=0.92,
        solar_absorptivity=0.8,
        thermal_conductivity=0.3,
        specific_heat=800.0,
        density=1500.0,
        effective_thickness_m=0.15
    ),
    "soil_wet": ThermalMaterialProperties(
        name="Wet Soil",
        category=MaterialCategory.SOIL,
        emissivity=0.95,
        solar_absorptivity=0.9,
        thermal_conductivity=1.5,
        specific_heat=1500.0,
        density=1800.0,
        effective_thickness_m=0.15
    ),
    "sand": ThermalMaterialProperties(
        name="Sand",
        category=MaterialCategory.SOIL,
        emissivity=0.90,
        solar_absorptivity=0.75,
        thermal_conductivity=0.25,
        specific_heat=830.0,
        density=1600.0,
        effective_thickness_m=0.10
    ),
    "grass": ThermalMaterialProperties(
        name="Grass/Vegetation",
        category=MaterialCategory.VEGETATION,
        emissivity=0.96,
        solar_absorptivity=0.75,
        thermal_conductivity=0.5,
        specific_heat=2000.0,
        density=500.0,
        effective_thickness_m=0.05
    ),
    "forest_canopy": ThermalMaterialProperties(
        name="Forest Canopy",
        category=MaterialCategory.VEGETATION,
        emissivity=0.98,
        solar_absorptivity=0.80,
        thermal_conductivity=0.2,
        specific_heat=2500.0,
        density=300.0,
        effective_thickness_m=0.20
    ),
    "water": ThermalMaterialProperties(
        name="Water",
        category=MaterialCategory.WATER,
        emissivity=0.96,
        solar_absorptivity=0.94,
        thermal_conductivity=0.6,
        specific_heat=4186.0,
        density=1000.0,
        effective_thickness_m=1.0  # Deep water is thermally stable
    ),

    # Vehicle materials
    "tire_rubber": ThermalMaterialProperties(
        name="Tire Rubber",
        category=MaterialCategory.RUBBER,
        emissivity=0.95,
        solar_absorptivity=0.90,
        thermal_conductivity=0.15,
        specific_heat=1900.0,
        density=1200.0,
        effective_thickness_m=0.02
    ),
    "vehicle_paint_dark": ThermalMaterialProperties(
        name="Dark Vehicle Paint",
        category=MaterialCategory.PAINT,
        emissivity=0.90,
        solar_absorptivity=0.85,
        thermal_conductivity=0.5,
        specific_heat=1500.0,
        density=1500.0,
        effective_thickness_m=0.001
    ),
    "vehicle_paint_light": ThermalMaterialProperties(
        name="Light Vehicle Paint",
        category=MaterialCategory.PAINT,
        emissivity=0.90,
        solar_absorptivity=0.35,
        thermal_conductivity=0.5,
        specific_heat=1500.0,
        density=1500.0,
        effective_thickness_m=0.001
    ),
    "glass_window": ThermalMaterialProperties(
        name="Window Glass",
        category=MaterialCategory.GLASS,
        emissivity=0.92,
        solar_absorptivity=0.20,  # Most solar passes through
        thermal_conductivity=1.0,
        specific_heat=840.0,
        density=2500.0,
        effective_thickness_m=0.006
    ),
    "camouflage_net": ThermalMaterialProperties(
        name="Camouflage Netting",
        category=MaterialCategory.FABRIC,
        emissivity=0.90,
        solar_absorptivity=0.70,
        thermal_conductivity=0.05,
        specific_heat=1300.0,
        density=100.0,
        effective_thickness_m=0.005
    ),
}


@dataclass
class SolarPosition:
    """Sun position and irradiance."""
    elevation_deg: float = 45.0      # Solar elevation angle
    azimuth_deg: float = 180.0       # Solar azimuth (S=180)
    direct_irradiance: float = 800.0  # W/m² direct beam
    diffuse_irradiance: float = 100.0 # W/m² sky diffuse
    air_mass: float = 1.5            # Atmospheric path length

    @property
    def total_irradiance(self) -> float:
        return self.direct_irradiance + self.diffuse_irradiance

    @classmethod
    def from_time_location(
        cls,
        hour: float,
        day_of_year: int,
        latitude: float
    ) -> 'SolarPosition':
        """Calculate solar position from time and location."""
        # Simplified solar position calculation
        # Declination angle
        declination = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))

        # Hour angle (15° per hour from solar noon)
        hour_angle = 15 * (hour - 12)

        # Solar elevation
        lat_rad = math.radians(latitude)
        dec_rad = math.radians(declination)
        hour_rad = math.radians(hour_angle)

        sin_elev = (math.sin(lat_rad) * math.sin(dec_rad) +
                   math.cos(lat_rad) * math.cos(dec_rad) * math.cos(hour_rad))
        elevation = math.degrees(math.asin(max(-1, min(1, sin_elev))))

        # Solar azimuth
        if elevation > 0:
            cos_az = ((math.sin(dec_rad) - math.sin(lat_rad) * sin_elev) /
                     (math.cos(lat_rad) * math.cos(math.radians(elevation))))
            cos_az = max(-1, min(1, cos_az))
            azimuth = math.degrees(math.acos(cos_az))
            if hour > 12:
                azimuth = 360 - azimuth
        else:
            azimuth = 180

        # Air mass
        if elevation > 0:
            air_mass = 1.0 / math.sin(math.radians(max(elevation, 1)))
        else:
            air_mass = 38.0  # Horizon value

        # Direct irradiance (simplified clear sky model)
        if elevation > 0:
            # Extra-terrestrial irradiance ~1361 W/m²
            # Atmospheric transmission
            tau = 0.7 ** (air_mass ** 0.678)
            direct = 1361 * tau * math.sin(math.radians(elevation))
            diffuse = 0.1 * 1361 * math.sin(math.radians(max(elevation, 10)))
        else:
            direct = 0
            diffuse = 0

        return cls(
            elevation_deg=elevation,
            azimuth_deg=azimuth,
            direct_irradiance=direct,
            diffuse_irradiance=diffuse,
            air_mass=air_mass
        )


@dataclass
class AmbientConditions:
    """Ambient environmental conditions."""
    air_temperature_k: float = 293.0  # 20°C
    sky_temperature_k: float = 253.0  # Effective sky temperature
    ground_temperature_k: float = 290.0
    wind_speed_m_s: float = 2.0       # Wind speed
    relative_humidity: float = 0.5    # 0-1
    cloud_cover: float = 0.0          # 0-1


@dataclass
class ThermalState:
    """Thermal state of a surface element."""
    temperature_k: float = 293.0     # Current temperature
    time_s: float = 0.0              # Time since start
    internal_heat_w: float = 0.0     # Internal heat generation (e.g., engine)
    is_shadowed: bool = False        # In shadow
    material: str = "concrete"       # Material type


class ThermalSolver:
    """
    Solver for transient thermal problems.

    Uses simple explicit finite difference method for
    surface temperature evolution.
    """

    def __init__(self, material: ThermalMaterialProperties):
        self.material = material

    def calculate_heat_fluxes(
        self,
        state: ThermalState,
        solar: SolarPosition,
        ambient: AmbientConditions,
        surface_normal: Tuple[float, float, float] = (0, 0, 1)
    ) -> Dict[str, float]:
        """
        Calculate all heat fluxes on a surface.

        Args:
            state: Current thermal state
            solar: Solar position and irradiance
            ambient: Ambient conditions
            surface_normal: Surface normal vector (x, y, z pointing up)

        Returns:
            Dictionary of heat fluxes (W/m²)
        """
        fluxes = {}
        mat = self.material
        T = state.temperature_k

        # 1. Solar absorption (if not shadowed and sun is up)
        if not state.is_shadowed and solar.elevation_deg > 0:
            # Calculate cos of incidence angle
            sun_vec = (
                math.cos(math.radians(solar.elevation_deg)) *
                    math.sin(math.radians(solar.azimuth_deg)),
                math.cos(math.radians(solar.elevation_deg)) *
                    math.cos(math.radians(solar.azimuth_deg)),
                math.sin(math.radians(solar.elevation_deg))
            )
            cos_inc = max(0, sum(s * n for s, n in zip(sun_vec, surface_normal)))

            q_solar_direct = mat.solar_absorptivity * solar.direct_irradiance * cos_inc
            q_solar_diffuse = mat.solar_absorptivity * solar.diffuse_irradiance * 0.5
            fluxes['solar'] = q_solar_direct + q_solar_diffuse
        else:
            fluxes['solar'] = 0.0

        # 2. Thermal radiation emission
        fluxes['emission'] = -mat.emissivity * STEFAN_BOLTZMANN * T**4

        # 3. Atmospheric/sky radiation absorption
        # View factor to sky (horizontal surface = 0.5)
        vf_sky = 0.5 * (1 + surface_normal[2])  # Simplified
        q_sky = mat.emissivity * STEFAN_BOLTZMANN * ambient.sky_temperature_k**4 * vf_sky
        fluxes['sky_absorption'] = q_sky

        # 4. Ground radiation (for tilted surfaces)
        vf_ground = 0.5 * (1 - surface_normal[2])
        q_ground = mat.emissivity * STEFAN_BOLTZMANN * ambient.ground_temperature_k**4 * vf_ground
        fluxes['ground_absorption'] = q_ground

        # 5. Convection
        # Heat transfer coefficient (simplified)
        # Natural: h ≈ 5-10 W/(m²·K)
        # Forced: h ≈ 10.45 - v + 10√v (for v < 5 m/s)
        v = ambient.wind_speed_m_s
        if v < 0.1:
            h_conv = 5.0  # Natural convection
        else:
            h_conv = 10.45 - v + 10 * math.sqrt(v)
        h_conv = max(5.0, min(h_conv, 50.0))

        q_conv = h_conv * (ambient.air_temperature_k - T)
        fluxes['convection'] = q_conv

        # 6. Internal heat generation
        fluxes['internal'] = state.internal_heat_w

        # Total
        fluxes['total'] = sum(fluxes.values())

        return fluxes

    def step(
        self,
        state: ThermalState,
        solar: SolarPosition,
        ambient: AmbientConditions,
        dt: float,
        surface_normal: Tuple[float, float, float] = (0, 0, 1)
    ) -> ThermalState:
        """
        Advance thermal state by one time step.

        Args:
            state: Current thermal state
            solar: Solar conditions
            ambient: Ambient conditions
            dt: Time step in seconds
            surface_normal: Surface normal direction

        Returns:
            Updated thermal state
        """
        fluxes = self.calculate_heat_fluxes(state, solar, ambient, surface_normal)

        # Temperature change: dT = Q × dt / (m × c)
        q_net = fluxes['total']  # W/m²
        thermal_mass = self.material.thermal_mass  # J/(m²·K)

        dT = q_net * dt / thermal_mass

        # Limit temperature change for stability
        max_dT = 10.0  # Max 10K per step
        dT = max(-max_dT, min(dT, max_dT))

        new_state = ThermalState(
            temperature_k=state.temperature_k + dT,
            time_s=state.time_s + dt,
            internal_heat_w=state.internal_heat_w,
            is_shadowed=state.is_shadowed,
            material=state.material
        )

        return new_state


class DiurnalCycleSimulator:
    """
    Simulate daily temperature cycles for terrain and objects.
    """

    def __init__(
        self,
        latitude: float = 35.0,
        day_of_year: int = 180
    ):
        self.latitude = latitude
        self.day_of_year = day_of_year
        self.solvers: Dict[str, ThermalSolver] = {}

    def get_solver(self, material_name: str) -> ThermalSolver:
        """Get or create thermal solver for a material."""
        if material_name not in self.solvers:
            if material_name in MATERIAL_DATABASE:
                self.solvers[material_name] = ThermalSolver(
                    MATERIAL_DATABASE[material_name]
                )
            else:
                # Default to concrete
                self.solvers[material_name] = ThermalSolver(
                    MATERIAL_DATABASE["concrete"]
                )
        return self.solvers[material_name]

    def simulate_day(
        self,
        material_name: str,
        initial_temp_k: float = 290.0,
        ambient_temp_day_k: float = 298.0,
        ambient_temp_night_k: float = 283.0,
        wind_speed: float = 2.0,
        cloud_cover: float = 0.0,
        time_step_s: float = 60.0
    ) -> List[Tuple[float, float]]:
        """
        Simulate 24-hour temperature cycle.

        Args:
            material_name: Material type
            initial_temp_k: Starting temperature
            ambient_temp_day_k: Daytime air temperature
            ambient_temp_night_k: Nighttime air temperature
            wind_speed: Wind speed in m/s
            cloud_cover: Cloud cover fraction 0-1
            time_step_s: Simulation time step

        Returns:
            List of (hour, temperature_k) tuples
        """
        solver = self.get_solver(material_name)
        state = ThermalState(temperature_k=initial_temp_k, material=material_name)
        results = []

        for step in range(int(24 * 3600 / time_step_s)):
            hour = (step * time_step_s) / 3600.0

            # Get solar position
            solar = SolarPosition.from_time_location(
                hour, self.day_of_year, self.latitude
            )

            # Reduce solar for clouds
            if cloud_cover > 0:
                solar.direct_irradiance *= (1 - cloud_cover)
                solar.diffuse_irradiance *= (1 - 0.5 * cloud_cover)

            # Interpolate ambient temperature
            if 6 <= hour <= 18:
                # Daytime
                t = (hour - 6) / 12.0
                air_temp = ambient_temp_night_k + (ambient_temp_day_k - ambient_temp_night_k) * math.sin(t * math.pi)
            else:
                # Nighttime
                air_temp = ambient_temp_night_k

            ambient = AmbientConditions(
                air_temperature_k=air_temp,
                sky_temperature_k=air_temp - 20 - 10 * (1 - cloud_cover),
                ground_temperature_k=air_temp - 2,
                wind_speed_m_s=wind_speed,
                cloud_cover=cloud_cover
            )

            # Step simulation
            state = solver.step(state, solar, ambient, time_step_s)
            results.append((hour, state.temperature_k))

        return results

    def get_temperature_at_time(
        self,
        material_name: str,
        hour: float,
        ambient_temp_k: float = 293.0,
        cloud_cover: float = 0.0
    ) -> float:
        """
        Get approximate surface temperature at given time.

        Uses simplified steady-state approximation for quick queries.

        Args:
            material_name: Material type
            hour: Hour of day (0-24)
            ambient_temp_k: Ambient air temperature
            cloud_cover: Cloud cover fraction

        Returns:
            Surface temperature in Kelvin
        """
        mat = MATERIAL_DATABASE.get(material_name, MATERIAL_DATABASE["concrete"])
        solar = SolarPosition.from_time_location(hour, self.day_of_year, self.latitude)

        # Reduce solar for clouds
        solar_flux = solar.total_irradiance * (1 - 0.8 * cloud_cover)

        if solar.elevation_deg > 0:
            # Simplified equilibrium temperature
            # Solar heating increases temp above ambient
            absorbed = mat.solar_absorptivity * solar_flux * 0.5  # Avg incidence

            # Approximate temperature rise
            h_eff = 10.0 + 5 * 2.0  # Combined radiation + convection
            delta_t = absorbed / h_eff

            return ambient_temp_k + delta_t
        else:
            # Night: radiative cooling below ambient
            sky_temp = ambient_temp_k - 15
            delta_t = -mat.emissivity * (ambient_temp_k - sky_temp) * 0.2
            return ambient_temp_k + delta_t


@dataclass
class HeatSource:
    """A localized heat source (engine, exhaust, etc.)."""
    name: str
    power_watts: float = 1000.0       # Heat output
    area_m2: float = 0.5              # Emitting area
    temperature_k: float = 400.0      # Surface temperature

    # Time dynamics
    warmup_time_s: float = 300.0      # Time to reach operating temp
    cooldown_time_s: float = 600.0    # Time to cool after shutdown

    @property
    def heat_flux(self) -> float:
        """Heat flux in W/m²."""
        return self.power_watts / self.area_m2


class VehicleThermalModel:
    """
    Thermal model for vehicle signatures.

    Models engines, exhausts, tires, and surface heating.
    """

    def __init__(self):
        self.heat_sources: Dict[str, HeatSource] = {}
        self.surface_temps: Dict[str, float] = {}
        self.engine_running = False
        self.time_since_start_s = 0.0
        self.time_since_stop_s = float('inf')

    def add_heat_source(self, source: HeatSource):
        """Add a heat source to the vehicle."""
        self.heat_sources[source.name] = source

    def start_engine(self):
        """Start the engine."""
        self.engine_running = True
        self.time_since_start_s = 0.0
        self.time_since_stop_s = float('inf')

    def stop_engine(self):
        """Stop the engine."""
        self.engine_running = False
        self.time_since_stop_s = 0.0

    def update(self, dt: float, ambient_temp_k: float = 293.0):
        """
        Update vehicle thermal state.

        Args:
            dt: Time step in seconds
            ambient_temp_k: Ambient temperature
        """
        if self.engine_running:
            self.time_since_start_s += dt
        else:
            self.time_since_stop_s += dt

        # Update heat source temperatures
        for name, source in self.heat_sources.items():
            if self.engine_running:
                # Warmup dynamics
                warmup_factor = 1 - math.exp(-self.time_since_start_s /
                                             source.warmup_time_s)
                temp = (ambient_temp_k +
                       (source.temperature_k - ambient_temp_k) * warmup_factor)
            else:
                # Cooldown dynamics
                cooldown_factor = math.exp(-self.time_since_stop_s /
                                          source.cooldown_time_s)
                temp = (ambient_temp_k +
                       (source.temperature_k - ambient_temp_k) * cooldown_factor)

            self.surface_temps[name] = temp

    def get_thermal_signature(self) -> Dict[str, float]:
        """Get current thermal signature."""
        return self.surface_temps.copy()


# Pre-defined vehicle thermal models
def create_tank_thermal_model() -> VehicleThermalModel:
    """Create thermal model for a main battle tank."""
    model = VehicleThermalModel()

    model.add_heat_source(HeatSource(
        name="engine_deck",
        power_watts=50000,
        area_m2=4.0,
        temperature_k=380,
        warmup_time_s=600,
        cooldown_time_s=1800
    ))
    model.add_heat_source(HeatSource(
        name="exhaust",
        power_watts=30000,
        area_m2=0.3,
        temperature_k=600,
        warmup_time_s=120,
        cooldown_time_s=300
    ))
    model.add_heat_source(HeatSource(
        name="track_left",
        power_watts=5000,
        area_m2=3.0,
        temperature_k=330,
        warmup_time_s=300,
        cooldown_time_s=600
    ))
    model.add_heat_source(HeatSource(
        name="track_right",
        power_watts=5000,
        area_m2=3.0,
        temperature_k=330,
        warmup_time_s=300,
        cooldown_time_s=600
    ))
    model.add_heat_source(HeatSource(
        name="gun_barrel",
        power_watts=0,  # Only heats when fired
        area_m2=0.5,
        temperature_k=293,  # Ambient when cold
        warmup_time_s=1,
        cooldown_time_s=120
    ))

    return model


def create_truck_thermal_model() -> VehicleThermalModel:
    """Create thermal model for a military truck."""
    model = VehicleThermalModel()

    model.add_heat_source(HeatSource(
        name="engine_hood",
        power_watts=15000,
        area_m2=2.0,
        temperature_k=360,
        warmup_time_s=300,
        cooldown_time_s=900
    ))
    model.add_heat_source(HeatSource(
        name="exhaust_pipe",
        power_watts=8000,
        area_m2=0.2,
        temperature_k=550,
        warmup_time_s=60,
        cooldown_time_s=180
    ))
    model.add_heat_source(HeatSource(
        name="tires",
        power_watts=2000,
        area_m2=2.0,
        temperature_k=320,
        warmup_time_s=600,
        cooldown_time_s=900
    ))

    return model


def create_aircraft_thermal_model() -> VehicleThermalModel:
    """Create thermal model for a jet aircraft."""
    model = VehicleThermalModel()

    model.add_heat_source(HeatSource(
        name="exhaust_plume",
        power_watts=500000,
        area_m2=2.0,
        temperature_k=900,
        warmup_time_s=60,
        cooldown_time_s=120
    ))
    model.add_heat_source(HeatSource(
        name="engine_nacelle",
        power_watts=50000,
        area_m2=5.0,
        temperature_k=450,
        warmup_time_s=120,
        cooldown_time_s=300
    ))
    model.add_heat_source(HeatSource(
        name="leading_edge",
        power_watts=10000,
        area_m2=10.0,
        temperature_k=320,  # Aerodynamic heating
        warmup_time_s=60,
        cooldown_time_s=120
    ))

    return model
