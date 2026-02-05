"""
Vehicle target models for EOSIM.

Provides detailed thermal signature models for various vehicle types
including cars, trucks, tanks, and other ground vehicles.

Example Usage:
--------------
# Example 1: Create a running sedan and get its thermal signature
>>> from eosim.targets import VehicleTarget
>>> car = VehicleTarget.sedan(engine_state="running", speed_kmh=60)
>>> signature = car.get_signature(aspect_angle_deg=90)  # Side view
>>> print(f"Engine temp: {signature.max_temperature:.0f}K")

# Example 2: Model a parked truck cooling down over time
>>> truck = VehicleTarget.truck(engine_state="cooling", time_since_shutoff_s=300)
>>> for t in range(0, 600, 60):
...     truck.set_time(t)
...     sig = truck.get_signature()
...     print(f"t={t}s: max temp = {sig.max_temperature:.1f}K")

# Example 3: Create a military tank with active systems
>>> tank = TankTarget(
...     engine_power_kw=1100,
...     turret_active=True,
...     track_friction_heat=True,
... )
>>> signature = tank.get_signature(aspect_angle_deg=45)
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from eosim.targets.base import (
    Target,
    TargetSignature,
    TargetGeometry,
    HotSpot,
    MaterialProperties,
    MaterialType,
)


@dataclass
class EngineModel:
    """Engine thermal model.

    Attributes:
        power_kw: Engine power in kilowatts
        operating_temp_k: Normal operating temperature
        idle_temp_k: Idle temperature
        max_temp_k: Maximum temperature under load
        cooling_rate: Cooling rate constant (1/s)
        warmup_rate: Warmup rate constant (1/s)
    """
    power_kw: float = 100.0
    operating_temp_k: float = 360.0
    idle_temp_k: float = 330.0
    max_temp_k: float = 400.0
    cooling_rate: float = 0.005
    warmup_rate: float = 0.01

    def get_temperature(
        self,
        state: str,
        throttle: float = 0.0,
        time_s: float = 0.0,
        ambient_k: float = 290.0,
    ) -> float:
        """Get engine temperature based on state.

        Args:
            state: "off", "starting", "idle", "running", "cooling"
            throttle: Throttle position 0-1
            time_s: Time in current state
            ambient_k: Ambient temperature

        Returns:
            Engine temperature in Kelvin
        """
        if state == "off":
            return ambient_k

        elif state == "starting":
            # Rapid warmup from ambient
            target = self.idle_temp_k
            return ambient_k + (target - ambient_k) * (1 - np.exp(-self.warmup_rate * 2 * time_s))

        elif state == "idle":
            return self.idle_temp_k

        elif state == "running":
            # Temperature varies with throttle
            base = self.idle_temp_k
            delta = (self.operating_temp_k - self.idle_temp_k) * throttle
            return base + delta

        elif state == "cooling":
            # Exponential cooling from operating temp
            return ambient_k + (self.operating_temp_k - ambient_k) * np.exp(-self.cooling_rate * time_s)

        return ambient_k


class VehicleTarget(Target):
    """General vehicle target model.

    Models ground vehicles with engine, exhaust, tires, and body
    thermal characteristics.
    """

    def __init__(
        self,
        name: str = "vehicle",
        geometry: Optional[TargetGeometry] = None,
        base_temperature_k: float = 300.0,
        ambient_temperature_k: float = 290.0,
        engine_model: Optional[EngineModel] = None,
        engine_state: str = "off",
        speed_kmh: float = 0.0,
    ) -> None:
        """Initialize vehicle target.

        Args:
            name: Vehicle identifier
            geometry: Vehicle geometry (default sedan-sized)
            base_temperature_k: Base body temperature
            ambient_temperature_k: Ambient temperature
            engine_model: Engine thermal model
            engine_state: Current engine state
            speed_kmh: Current speed in km/h
        """
        if geometry is None:
            geometry = TargetGeometry(length_m=4.5, width_m=1.8, height_m=1.4)

        super().__init__(name, geometry, base_temperature_k, ambient_temperature_k)

        self.engine_model = engine_model or EngineModel()
        self.engine_state = engine_state
        self.speed_kmh = speed_kmh
        self._time_since_state_change = 0.0

        # Setup default materials
        self._materials = {
            "body": MaterialProperties.from_material_type(MaterialType.METAL_PAINTED),
            "glass": MaterialProperties.from_material_type(MaterialType.GLASS),
            "tires": MaterialProperties.from_material_type(MaterialType.RUBBER),
        }

        # Setup hot spots
        self._setup_hot_spots()

    def _setup_hot_spots(self) -> None:
        """Configure vehicle hot spots."""
        # Engine compartment (front)
        self._hot_spots.append(HotSpot(
            name="engine",
            relative_position=(0.3, 0.25),
            relative_size=0.25,
            temperature_delta_k=0,  # Updated dynamically
        ))

        # Exhaust (rear)
        self._hot_spots.append(HotSpot(
            name="exhaust",
            relative_position=(0.5, 0.9),
            relative_size=0.08,
            temperature_delta_k=0,  # Updated dynamically
        ))

        # Front tires
        self._hot_spots.append(HotSpot(
            name="tire_fl",
            relative_position=(0.15, 0.2),
            relative_size=0.1,
            temperature_delta_k=0,
        ))
        self._hot_spots.append(HotSpot(
            name="tire_fr",
            relative_position=(0.85, 0.2),
            relative_size=0.1,
            temperature_delta_k=0,
        ))

    def get_signature(
        self,
        resolution: tuple[int, int] = (64, 64),
        aspect_angle_deg: float = 0.0,
        elevation_angle_deg: float = 0.0,
    ) -> TargetSignature:
        """Generate vehicle thermal signature."""
        h, w = resolution

        # Create base shape
        mask = self._create_vehicle_shape(resolution, aspect_angle_deg)

        # Initialize temperature map
        temp_map = np.where(mask, self.base_temperature_k, 0.0)
        emis_map = np.where(mask, 0.9, 0.0)

        # Calculate engine temperature
        throttle = min(self.speed_kmh / 120.0, 1.0)
        engine_temp = self.engine_model.get_temperature(
            self.engine_state,
            throttle,
            self._time_since_state_change,
            self.ambient_temperature_k,
        )

        # Update hot spot temperatures
        for spot in self._hot_spots:
            if spot.name == "engine":
                spot.temperature_delta_k = engine_temp - self.base_temperature_k
            elif spot.name == "exhaust":
                if self.engine_state in ("idle", "running"):
                    spot.temperature_delta_k = (engine_temp - self.base_temperature_k) * 1.5
                else:
                    spot.temperature_delta_k = 0
            elif "tire" in spot.name:
                # Tire heating from speed
                spot.temperature_delta_k = self.speed_kmh * 0.15

        # Apply hot spots
        temp_map = self._apply_hot_spots(temp_map, self.base_temperature_k)

        # Add surface temperature variation
        noise = np.random.normal(0, 2, temp_map.shape)
        temp_map = np.where(mask, temp_map + noise, 0)

        return TargetSignature(
            temperature_map=temp_map,
            emissivity_map=emis_map,
            geometry=self.geometry,
            aspect_angle_deg=aspect_angle_deg,
            elevation_angle_deg=elevation_angle_deg,
        )

    def _create_vehicle_shape(
        self,
        resolution: tuple[int, int],
        aspect_angle_deg: float,
    ) -> NDArray:
        """Create vehicle silhouette based on view angle."""
        h, w = resolution
        mask = np.zeros((h, w), dtype=bool)

        # Simplified vehicle shape
        # Adjust proportions based on aspect angle
        angle_rad = np.radians(aspect_angle_deg % 360)

        # Body (main rectangle)
        body_h = int(h * 0.6)
        body_w = int(w * 0.8)
        y0 = (h - body_h) // 2
        x0 = (w - body_w) // 2
        mask[y0:y0+body_h, x0:x0+body_w] = True

        # Roof/cabin (smaller rectangle on top)
        cabin_h = int(h * 0.35)
        cabin_w = int(w * 0.5)
        cy0 = y0 - cabin_h // 3
        cx0 = (w - cabin_w) // 2
        if cy0 > 0:
            mask[cy0:cy0+cabin_h, cx0:cx0+cabin_w] = True

        return mask

    def set_engine_state(self, state: str) -> None:
        """Change engine state and reset timer."""
        self.engine_state = state
        self._time_since_state_change = 0.0

    def update(self, dt_s: float) -> None:
        """Update vehicle state."""
        super().update(dt_s)
        self._time_since_state_change += dt_s

    @classmethod
    def sedan(
        cls,
        engine_state: str = "off",
        speed_kmh: float = 0.0,
        color_dark: bool = False,
    ) -> "VehicleTarget":
        """Create a sedan car target."""
        geometry = TargetGeometry(length_m=4.5, width_m=1.8, height_m=1.4)
        engine = EngineModel(power_kw=120, operating_temp_k=355, idle_temp_k=325)
        base_temp = 295.0 if color_dark else 300.0

        return cls(
            name="sedan",
            geometry=geometry,
            base_temperature_k=base_temp,
            engine_model=engine,
            engine_state=engine_state,
            speed_kmh=speed_kmh,
        )

    @classmethod
    def suv(
        cls,
        engine_state: str = "off",
        speed_kmh: float = 0.0,
    ) -> "VehicleTarget":
        """Create an SUV target."""
        geometry = TargetGeometry(length_m=4.8, width_m=2.0, height_m=1.8)
        engine = EngineModel(power_kw=200, operating_temp_k=365, idle_temp_k=330)

        return cls(
            name="suv",
            geometry=geometry,
            engine_model=engine,
            engine_state=engine_state,
            speed_kmh=speed_kmh,
        )

    @classmethod
    def truck(
        cls,
        engine_state: str = "off",
        speed_kmh: float = 0.0,
        loaded: bool = False,
    ) -> "VehicleTarget":
        """Create a truck target."""
        geometry = TargetGeometry(length_m=12.0, width_m=2.5, height_m=4.0)
        engine = EngineModel(
            power_kw=400,
            operating_temp_k=380,
            idle_temp_k=340,
            max_temp_k=420,
        )

        truck = cls(
            name="truck",
            geometry=geometry,
            base_temperature_k=305.0 if loaded else 298.0,
            engine_model=engine,
            engine_state=engine_state,
            speed_kmh=speed_kmh,
        )

        # Add trailer heat signature
        truck._hot_spots.append(HotSpot(
            name="trailer_friction",
            relative_position=(0.5, 0.7),
            relative_size=0.15,
            temperature_delta_k=10 if loaded else 5,
        ))

        return truck


# Aliases for convenience
CarTarget = VehicleTarget


class TruckTarget(VehicleTarget):
    """Heavy truck target with enhanced thermal model."""

    def __init__(
        self,
        engine_state: str = "off",
        speed_kmh: float = 0.0,
        cargo_type: str = "general",
    ) -> None:
        geometry = TargetGeometry(length_m=16.0, width_m=2.5, height_m=4.2)
        engine = EngineModel(
            power_kw=500,
            operating_temp_k=385,
            idle_temp_k=345,
        )

        super().__init__(
            name="heavy_truck",
            geometry=geometry,
            engine_model=engine,
            engine_state=engine_state,
            speed_kmh=speed_kmh,
        )

        self.cargo_type = cargo_type

        # Add wheel heat
        for i, x_pos in enumerate([0.15, 0.4, 0.65, 0.85]):
            self._hot_spots.append(HotSpot(
                name=f"wheel_{i}",
                relative_position=(0.5, x_pos),
                relative_size=0.08,
                temperature_delta_k=speed_kmh * 0.2,
            ))


class TankTarget(Target):
    """Military tank/armored vehicle target.

    Models thermal signature including engine, tracks, turret,
    and active systems.
    """

    def __init__(
        self,
        engine_power_kw: float = 1100.0,
        turret_active: bool = False,
        track_friction_heat: bool = True,
        engine_state: str = "idle",
    ) -> None:
        geometry = TargetGeometry(length_m=9.5, width_m=3.7, height_m=2.4)

        super().__init__(
            name="tank",
            geometry=geometry,
            base_temperature_k=310.0,
            ambient_temperature_k=290.0,
        )

        self.engine_power_kw = engine_power_kw
        self.turret_active = turret_active
        self.track_friction_heat = track_friction_heat
        self.engine_state = engine_state

        self.engine_model = EngineModel(
            power_kw=engine_power_kw,
            operating_temp_k=420,
            idle_temp_k=370,
            max_temp_k=480,
        )

        self._setup_tank_hot_spots()

    def _setup_tank_hot_spots(self) -> None:
        """Configure tank-specific hot spots."""
        # Engine deck (rear)
        self._hot_spots.append(HotSpot(
            name="engine_deck",
            relative_position=(0.5, 0.85),
            relative_size=0.25,
            temperature_delta_k=80,
        ))

        # Exhaust grilles
        self._hot_spots.append(HotSpot(
            name="exhaust_left",
            relative_position=(0.2, 0.9),
            relative_size=0.1,
            temperature_delta_k=120,
        ))
        self._hot_spots.append(HotSpot(
            name="exhaust_right",
            relative_position=(0.8, 0.9),
            relative_size=0.1,
            temperature_delta_k=120,
        ))

        # Gun barrel (if recently fired - pulsing)
        self._hot_spots.append(HotSpot(
            name="gun_barrel",
            relative_position=(0.5, 0.1),
            relative_size=0.05,
            temperature_delta_k=50,
            pulsing=True,
            pulse_period_s=30.0,
            pulse_amplitude_k=30,
        ))

        # Tracks
        if self.track_friction_heat:
            for y_pos in [0.1, 0.9]:
                self._hot_spots.append(HotSpot(
                    name=f"track_{y_pos}",
                    relative_position=(y_pos, 0.5),
                    relative_size=0.08,
                    temperature_delta_k=25,
                ))

    def get_signature(
        self,
        resolution: tuple[int, int] = (64, 64),
        aspect_angle_deg: float = 0.0,
        elevation_angle_deg: float = 0.0,
    ) -> TargetSignature:
        """Generate tank thermal signature."""
        h, w = resolution

        # Create tank shape (rectangular with turret)
        mask = np.zeros((h, w), dtype=bool)

        # Hull
        hull_h = int(h * 0.7)
        hull_w = int(w * 0.9)
        y0 = (h - hull_h) // 2
        x0 = (w - hull_w) // 2
        mask[y0:y0+hull_h, x0:x0+hull_w] = True

        # Turret (circular on top)
        cy, cx = h // 2, int(w * 0.4)
        turret_r = int(min(h, w) * 0.2)
        yy, xx = np.ogrid[:h, :w]
        turret_mask = (yy - cy)**2 + (xx - cx)**2 <= turret_r**2
        mask |= turret_mask

        # Temperature map
        engine_temp = self.engine_model.get_temperature(
            self.engine_state,
            throttle=0.5,
            time_s=self._time_s,
            ambient_k=self.ambient_temperature_k,
        )

        # Update hot spots based on engine temp
        for spot in self._hot_spots:
            if "exhaust" in spot.name or "engine" in spot.name:
                spot.temperature_delta_k = engine_temp - self.base_temperature_k

        temp_map = np.where(mask, self.base_temperature_k, 0.0)
        temp_map = self._apply_hot_spots(temp_map, self.base_temperature_k)

        emis_map = np.where(mask, 0.85, 0.0)  # Military paint

        return TargetSignature(
            temperature_map=temp_map,
            emissivity_map=emis_map,
            geometry=self.geometry,
            aspect_angle_deg=aspect_angle_deg,
        )
