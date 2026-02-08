"""
Aircraft target models for EOSIM.

Provides thermal signature models for aircraft including jets,
helicopters, and drones.

Features realistic exhaust plume thermal gradients:
- Engine nozzle: 500K+ at core
- Plume gradient: 500K at nozzle falling to 300K (ambient)
- Plume expands and cools with distance
- Afterburner increases temperatures significantly

Example Usage:
--------------
# Example 1: Create a fighter jet at high throttle
>>> from eosim.targets import AircraftTarget
>>> jet = AircraftTarget.fighter_jet(throttle=0.9, altitude_m=8000)
>>> signature = jet.get_signature(aspect_angle_deg=180)  # Rear view
>>> print(f"Exhaust plume temp: {signature.max_temperature:.0f}K")

# Example 2: Model a helicopter with rotor downwash heat
>>> from eosim.targets import HelicopterTarget
>>> helo = HelicopterTarget(rotor_rpm=400, engine_power_percent=75)
>>> signature = helo.get_signature(elevation_angle_deg=-30)  # From below

# Example 3: Small surveillance drone
>>> from eosim.targets import DroneTarget
>>> drone = DroneTarget.quadcopter(battery_temp_k=310, motors_active=True)
>>> signature = drone.get_signature()

# Example 4: Use enhanced exhaust plume gradient
>>> jet = AircraftTarget.fighter_jet(throttle=0.9, afterburner=True)
>>> sig = jet.get_signature(use_enhanced_shape=True, aspect_angle_deg=180)
>>> print(f"Nozzle temp: {sig.max_temperature:.0f}K")  # ~900K with afterburner
"""

from dataclasses import dataclass
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
from eosim.targets.shapes import AircraftWithPlumeShape, ThermalZone, ThermalZoneConfig


@dataclass
class JetEngineModel:
    """Jet engine thermal model.

    Attributes:
        max_thrust_kn: Maximum thrust in kilonewtons
        bypass_ratio: Engine bypass ratio
        exhaust_temp_idle_k: Exhaust temperature at idle
        exhaust_temp_max_k: Maximum exhaust temperature
        nozzle_temp_k: Nozzle temperature
        plume_length_m: Exhaust plume length at max thrust
    """
    max_thrust_kn: float = 100.0
    bypass_ratio: float = 0.4
    exhaust_temp_idle_k: float = 600.0
    exhaust_temp_max_k: float = 1800.0
    nozzle_temp_k: float = 800.0
    plume_length_m: float = 5.0

    def get_exhaust_temperature(self, throttle: float) -> float:
        """Get exhaust temperature at given throttle setting."""
        return self.exhaust_temp_idle_k + (
            self.exhaust_temp_max_k - self.exhaust_temp_idle_k
        ) * throttle**2

    def get_plume_length(self, throttle: float, altitude_m: float) -> float:
        """Get exhaust plume length accounting for altitude."""
        # Plume extends more at altitude (lower pressure)
        altitude_factor = 1 + altitude_m / 20000
        return self.plume_length_m * throttle * altitude_factor


class AircraftTarget(Target):
    """Fixed-wing aircraft target model.

    Models thermal signature including fuselage, wings, engines,
    and exhaust plume.
    """

    def __init__(
        self,
        name: str = "aircraft",
        geometry: Optional[TargetGeometry] = None,
        engine_model: Optional[JetEngineModel] = None,
        num_engines: int = 2,
        throttle: float = 0.5,
        altitude_m: float = 5000.0,
        speed_mach: float = 0.8,
        afterburner: bool = False,
    ) -> None:
        """Initialize aircraft target.

        Args:
            name: Aircraft identifier
            geometry: Aircraft geometry
            engine_model: Jet engine model
            num_engines: Number of engines
            throttle: Throttle setting 0-1
            altitude_m: Altitude in meters
            speed_mach: Speed in Mach number
            afterburner: Afterburner engaged (fighter jets)
        """
        if geometry is None:
            geometry = TargetGeometry(length_m=15.0, width_m=10.0, height_m=4.0)

        # Base temperature depends on altitude (adiabatic cooling)
        base_temp = 288.15 - 0.0065 * altitude_m  # Standard atmosphere
        base_temp = max(base_temp, 220.0)  # Minimum at tropopause

        super().__init__(name, geometry, base_temp, base_temp)

        self.engine_model = engine_model or JetEngineModel()
        self.num_engines = num_engines
        self.throttle = throttle
        self.altitude_m = altitude_m
        self.speed_mach = speed_mach
        self.afterburner = afterburner

        # Aerodynamic heating at high speed
        self._aero_heating = self._compute_aero_heating()

        self._setup_aircraft_hot_spots()

    def _compute_aero_heating(self) -> float:
        """Compute aerodynamic heating from speed."""
        # Simplified adiabatic heating model
        # T_recovery = T_static * (1 + 0.2 * M^2) for recovery factor ~1
        recovery_factor = 0.9
        gamma = 1.4
        temp_ratio = 1 + recovery_factor * (gamma - 1) / 2 * self.speed_mach**2
        return self.base_temperature_k * (temp_ratio - 1)

    def _setup_aircraft_hot_spots(self) -> None:
        """Configure aircraft hot spots."""
        exhaust_temp = self.engine_model.get_exhaust_temperature(self.throttle)
        delta_t = exhaust_temp - self.base_temperature_k

        # Engine nozzles (rear)
        if self.num_engines == 1:
            self._hot_spots.append(HotSpot(
                name="engine_nozzle",
                relative_position=(0.5, 0.95),
                relative_size=0.12,
                temperature_delta_k=delta_t,
            ))
        elif self.num_engines == 2:
            for y_pos, name in [(0.3, "left"), (0.7, "right")]:
                self._hot_spots.append(HotSpot(
                    name=f"engine_{name}",
                    relative_position=(y_pos, 0.95),
                    relative_size=0.1,
                    temperature_delta_k=delta_t,
                ))
        elif self.num_engines == 4:
            for y_pos in [0.2, 0.4, 0.6, 0.8]:
                self._hot_spots.append(HotSpot(
                    name=f"engine_{y_pos}",
                    relative_position=(y_pos, 0.9),
                    relative_size=0.08,
                    temperature_delta_k=delta_t,
                ))

        # Leading edge heating (aerodynamic)
        if self.speed_mach > 0.5:
            self._hot_spots.append(HotSpot(
                name="nose",
                relative_position=(0.5, 0.05),
                relative_size=0.08,
                temperature_delta_k=self._aero_heating,
            ))
            # Wing leading edges
            self._hot_spots.append(HotSpot(
                name="wing_leading",
                relative_position=(0.5, 0.4),
                relative_size=0.15,
                temperature_delta_k=self._aero_heating * 0.7,
                shape="ellipse",
            ))

    def get_signature(
        self,
        resolution: tuple[int, int] = (64, 64),
        aspect_angle_deg: float = 0.0,
        elevation_angle_deg: float = 0.0,
        use_enhanced_shape: bool = True,
    ) -> TargetSignature:
        """Generate aircraft thermal signature.

        Args:
            resolution: Output resolution (height, width)
            aspect_angle_deg: Viewing angle (0=front, 90=side, 180=rear)
            elevation_angle_deg: Elevation viewing angle
            use_enhanced_shape: Use enhanced exhaust plume with thermal gradient

        Returns:
            TargetSignature with temperature and emissivity maps
        """
        h, w = resolution

        if use_enhanced_shape:
            # Use enhanced shape with exhaust plume gradient
            plume_shape = AircraftWithPlumeShape(
                length_m=self.geometry.length_m,
                wingspan_m=self.geometry.width_m,
                height_m=self.geometry.height_m,
                throttle=self.throttle,
                afterburner=self.afterburner,
                altitude_m=self.altitude_m,
                speed_mach=self.speed_mach,
                num_engines=self.num_engines,
            )

            temp_map, emis_map, mask = plume_shape.render(
                resolution, aspect_angle_deg, elevation_angle_deg
            )

            return TargetSignature(
                temperature_map=temp_map,
                emissivity_map=emis_map,
                geometry=self.geometry,
                aspect_angle_deg=aspect_angle_deg,
                elevation_angle_deg=elevation_angle_deg,
            )

        # Legacy rendering path
        # Create aircraft shape based on view angle
        mask = self._create_aircraft_shape(resolution, aspect_angle_deg)

        # Temperature map
        temp_map = np.where(mask, self.base_temperature_k + self._aero_heating * 0.3, 0.0)
        temp_map = self._apply_hot_spots(temp_map, self.base_temperature_k)

        # Emissivity (mostly painted metal)
        emis_map = np.where(mask, 0.85, 0.0)

        return TargetSignature(
            temperature_map=temp_map,
            emissivity_map=emis_map,
            geometry=self.geometry,
            aspect_angle_deg=aspect_angle_deg,
            elevation_angle_deg=elevation_angle_deg,
        )

    def _create_aircraft_shape(
        self,
        resolution: tuple[int, int],
        aspect_angle_deg: float,
    ) -> NDArray:
        """Create aircraft silhouette."""
        h, w = resolution
        mask = np.zeros((h, w), dtype=bool)

        # Fuselage (ellipse)
        cy, cx = h // 2, w // 2
        fuse_h = int(h * 0.25)
        fuse_w = int(w * 0.8)

        yy, xx = np.ogrid[:h, :w]
        fuselage = ((yy - cy) / fuse_h)**2 + ((xx - cx) / (fuse_w/2))**2 <= 1
        mask |= fuselage

        # Wings (depending on aspect angle)
        angle_factor = abs(np.cos(np.radians(aspect_angle_deg)))
        wing_span = int(h * 0.8 * angle_factor)
        wing_chord = int(w * 0.2)

        if wing_span > 5:
            # Left wing
            wing_y0 = cy - wing_span // 2
            wing_x0 = cx - wing_chord // 2
            mask[wing_y0:cy, wing_x0:wing_x0+wing_chord] = True
            # Right wing
            mask[cy:cy+wing_span//2, wing_x0:wing_x0+wing_chord] = True

        # Tail
        tail_h = int(h * 0.15)
        tail_w = int(w * 0.1)
        tail_x = int(w * 0.85)
        mask[cy-tail_h:cy+tail_h, tail_x:tail_x+tail_w] = True

        return mask

    def set_throttle(self, throttle: float) -> None:
        """Update throttle setting and recalculate hot spots."""
        self.throttle = np.clip(throttle, 0, 1)
        self._hot_spots.clear()
        self._setup_aircraft_hot_spots()

    @classmethod
    def fighter_jet(
        cls,
        throttle: float = 0.5,
        altitude_m: float = 8000.0,
        afterburner: bool = False,
    ) -> "AircraftTarget":
        """Create a fighter jet target.

        Args:
            throttle: Throttle setting 0-1
            altitude_m: Altitude in meters
            afterburner: Afterburner engaged (significantly increases exhaust temps)

        Returns:
            AircraftTarget configured as fighter jet
        """
        geometry = TargetGeometry(length_m=16.0, width_m=11.0, height_m=5.0)
        engine = JetEngineModel(
            max_thrust_kn=130,
            bypass_ratio=0.3,
            exhaust_temp_idle_k=700,
            exhaust_temp_max_k=2200 if afterburner else 1800,
            plume_length_m=8.0 if afterburner else 4.0,
        )

        effective_throttle = throttle * (1.5 if afterburner else 1.0)

        return cls(
            name="fighter_jet",
            geometry=geometry,
            engine_model=engine,
            num_engines=2,
            throttle=min(effective_throttle, 1.0),
            altitude_m=altitude_m,
            speed_mach=0.9 + throttle * 0.8,
            afterburner=afterburner,
        )

    @classmethod
    def commercial_airliner(
        cls,
        throttle: float = 0.6,
        altitude_m: float = 10000.0,
    ) -> "AircraftTarget":
        """Create a commercial airliner target."""
        geometry = TargetGeometry(length_m=60.0, width_m=50.0, height_m=15.0)
        engine = JetEngineModel(
            max_thrust_kn=250,
            bypass_ratio=5.0,
            exhaust_temp_idle_k=500,
            exhaust_temp_max_k=900,
            plume_length_m=3.0,
        )

        return cls(
            name="airliner",
            geometry=geometry,
            engine_model=engine,
            num_engines=2,
            throttle=throttle,
            altitude_m=altitude_m,
            speed_mach=0.82,
        )

    @classmethod
    def turboprop(
        cls,
        throttle: float = 0.7,
        altitude_m: float = 5000.0,
    ) -> "AircraftTarget":
        """Create a turboprop aircraft target."""
        geometry = TargetGeometry(length_m=20.0, width_m=25.0, height_m=6.0)
        engine = JetEngineModel(
            max_thrust_kn=30,
            bypass_ratio=50.0,  # High bypass for turboprop
            exhaust_temp_idle_k=400,
            exhaust_temp_max_k=600,
            plume_length_m=1.0,
        )

        return cls(
            name="turboprop",
            geometry=geometry,
            engine_model=engine,
            num_engines=2,
            throttle=throttle,
            altitude_m=altitude_m,
            speed_mach=0.45,
        )


class HelicopterTarget(Target):
    """Helicopter target model.

    Models thermal signature including engine, rotor, and exhaust.
    """

    def __init__(
        self,
        name: str = "helicopter",
        geometry: Optional[TargetGeometry] = None,
        rotor_rpm: float = 300.0,
        engine_power_percent: float = 50.0,
        altitude_m: float = 500.0,
    ) -> None:
        if geometry is None:
            geometry = TargetGeometry(length_m=15.0, width_m=3.0, height_m=4.5)

        super().__init__(name, geometry, 295.0, 290.0)

        self.rotor_rpm = rotor_rpm
        self.engine_power_percent = engine_power_percent
        self.altitude_m = altitude_m

        self._setup_helicopter_hot_spots()

    def _setup_helicopter_hot_spots(self) -> None:
        """Configure helicopter hot spots."""
        power_factor = self.engine_power_percent / 100.0

        # Engine/transmission (top, rear of cabin)
        self._hot_spots.append(HotSpot(
            name="engine",
            relative_position=(0.5, 0.3),
            relative_size=0.2,
            temperature_delta_k=80 * power_factor,
        ))

        # Exhaust
        self._hot_spots.append(HotSpot(
            name="exhaust",
            relative_position=(0.3, 0.35),
            relative_size=0.08,
            temperature_delta_k=200 * power_factor,
        ))

        # Main rotor hub (friction heating)
        self._hot_spots.append(HotSpot(
            name="rotor_hub",
            relative_position=(0.5, 0.25),
            relative_size=0.1,
            temperature_delta_k=20 * (self.rotor_rpm / 400),
        ))

        # Tail rotor gearbox
        self._hot_spots.append(HotSpot(
            name="tail_gearbox",
            relative_position=(0.5, 0.95),
            relative_size=0.06,
            temperature_delta_k=30 * power_factor,
        ))

    def get_signature(
        self,
        resolution: tuple[int, int] = (64, 64),
        aspect_angle_deg: float = 0.0,
        elevation_angle_deg: float = 0.0,
    ) -> TargetSignature:
        """Generate helicopter thermal signature."""
        h, w = resolution

        # Create helicopter shape
        mask = self._create_helicopter_shape(resolution, aspect_angle_deg)

        temp_map = np.where(mask, self.base_temperature_k, 0.0)
        temp_map = self._apply_hot_spots(temp_map, self.base_temperature_k)

        emis_map = np.where(mask, 0.88, 0.0)

        return TargetSignature(
            temperature_map=temp_map,
            emissivity_map=emis_map,
            geometry=self.geometry,
            aspect_angle_deg=aspect_angle_deg,
        )

    def _create_helicopter_shape(
        self,
        resolution: tuple[int, int],
        aspect_angle_deg: float,
    ) -> NDArray:
        """Create helicopter silhouette."""
        h, w = resolution
        mask = np.zeros((h, w), dtype=bool)

        # Fuselage (elongated ellipse)
        cy = h // 2
        fuse_h = int(h * 0.4)
        fuse_w = int(w * 0.5)
        fuse_cx = int(w * 0.35)

        yy, xx = np.ogrid[:h, :w]
        fuselage = ((yy - cy) / fuse_h)**2 + ((xx - fuse_cx) / (fuse_w/2))**2 <= 1
        mask |= fuselage

        # Tail boom
        tail_h = int(h * 0.15)
        tail_w = int(w * 0.5)
        tail_x0 = fuse_cx + fuse_w // 2 - 5
        mask[cy-tail_h:cy+tail_h, tail_x0:tail_x0+tail_w] = True

        # Tail rotor area
        tail_end = tail_x0 + tail_w
        tr_r = int(h * 0.12)
        tr_mask = (yy - cy)**2 + (xx - tail_end)**2 <= tr_r**2
        mask |= tr_mask

        return mask

    @classmethod
    def utility(cls, engine_power_percent: float = 60.0) -> "HelicopterTarget":
        """Create a utility helicopter."""
        return cls(
            name="utility_helo",
            geometry=TargetGeometry(length_m=13.0, width_m=2.5, height_m=4.0),
            rotor_rpm=320,
            engine_power_percent=engine_power_percent,
        )

    @classmethod
    def attack(cls, weapons_hot: bool = False) -> "HelicopterTarget":
        """Create an attack helicopter."""
        helo = cls(
            name="attack_helo",
            geometry=TargetGeometry(length_m=17.0, width_m=4.0, height_m=4.5),
            rotor_rpm=290,
            engine_power_percent=70.0,
        )

        if weapons_hot:
            # Add weapon system heat
            helo._hot_spots.append(HotSpot(
                name="targeting_system",
                relative_position=(0.5, 0.1),
                relative_size=0.05,
                temperature_delta_k=15,
            ))

        return helo


class DroneTarget(Target):
    """Drone/UAV target model.

    Models small to medium unmanned aircraft thermal signatures.
    """

    def __init__(
        self,
        name: str = "drone",
        geometry: Optional[TargetGeometry] = None,
        motor_count: int = 4,
        motors_active: bool = True,
        battery_temp_k: float = 305.0,
        payload_temp_k: float = 300.0,
    ) -> None:
        if geometry is None:
            geometry = TargetGeometry(length_m=0.5, width_m=0.5, height_m=0.15)

        super().__init__(name, geometry, 295.0, 290.0)

        self.motor_count = motor_count
        self.motors_active = motors_active
        self.battery_temp_k = battery_temp_k
        self.payload_temp_k = payload_temp_k

        self._setup_drone_hot_spots()

    def _setup_drone_hot_spots(self) -> None:
        """Configure drone hot spots."""
        # Battery (center)
        self._hot_spots.append(HotSpot(
            name="battery",
            relative_position=(0.5, 0.5),
            relative_size=0.2,
            temperature_delta_k=self.battery_temp_k - self.base_temperature_k,
        ))

        # Motors (corners for quadcopter)
        if self.motors_active and self.motor_count == 4:
            motor_positions = [(0.2, 0.2), (0.2, 0.8), (0.8, 0.2), (0.8, 0.8)]
            for i, (y, x) in enumerate(motor_positions):
                self._hot_spots.append(HotSpot(
                    name=f"motor_{i}",
                    relative_position=(y, x),
                    relative_size=0.1,
                    temperature_delta_k=25,
                ))

        # Payload/camera (front-bottom)
        self._hot_spots.append(HotSpot(
            name="payload",
            relative_position=(0.5, 0.3),
            relative_size=0.12,
            temperature_delta_k=self.payload_temp_k - self.base_temperature_k,
        ))

    def get_signature(
        self,
        resolution: tuple[int, int] = (32, 32),
        aspect_angle_deg: float = 0.0,
        elevation_angle_deg: float = 0.0,
    ) -> TargetSignature:
        """Generate drone thermal signature."""
        h, w = resolution

        # Simple X-shape for quadcopter
        mask = np.zeros((h, w), dtype=bool)

        # Center body
        cy, cx = h // 2, w // 2
        body_r = int(min(h, w) * 0.2)
        yy, xx = np.ogrid[:h, :w]
        mask |= (yy - cy)**2 + (xx - cx)**2 <= body_r**2

        # Arms (diagonal)
        if self.motor_count == 4:
            for dy, dx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                arm_len = int(min(h, w) * 0.35)
                for i in range(arm_len):
                    y = cy + dy * i
                    x = cx + dx * i
                    if 0 <= y < h and 0 <= x < w:
                        mask[y, x] = True
                        if y+1 < h:
                            mask[y+1, x] = True
                        if x+1 < w:
                            mask[y, x+1] = True

        temp_map = np.where(mask, self.base_temperature_k, 0.0)
        temp_map = self._apply_hot_spots(temp_map, self.base_temperature_k)

        emis_map = np.where(mask, 0.92, 0.0)  # Plastic body

        return TargetSignature(
            temperature_map=temp_map,
            emissivity_map=emis_map,
            geometry=self.geometry,
            aspect_angle_deg=aspect_angle_deg,
        )

    @classmethod
    def quadcopter(
        cls,
        battery_temp_k: float = 305.0,
        motors_active: bool = True,
    ) -> "DroneTarget":
        """Create a quadcopter drone."""
        return cls(
            name="quadcopter",
            geometry=TargetGeometry(length_m=0.4, width_m=0.4, height_m=0.1),
            motor_count=4,
            motors_active=motors_active,
            battery_temp_k=battery_temp_k,
        )

    @classmethod
    def fixed_wing_uav(cls, engine_running: bool = True) -> "DroneTarget":
        """Create a fixed-wing UAV."""
        drone = cls(
            name="fixed_wing_uav",
            geometry=TargetGeometry(length_m=2.0, width_m=3.0, height_m=0.4),
            motor_count=1,
            motors_active=engine_running,
            battery_temp_k=310.0,
        )

        if engine_running:
            drone._hot_spots.append(HotSpot(
                name="engine",
                relative_position=(0.5, 0.9),
                relative_size=0.1,
                temperature_delta_k=50,
            ))

        return drone
