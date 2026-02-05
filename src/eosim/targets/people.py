"""
Human/personnel target models for EOSIM.

Provides thermal signature models for people in various poses,
activities, and environmental conditions.

Example Usage:
--------------
# Example 1: Standing person at different activity levels
>>> from eosim.targets import PersonTarget
>>> person_idle = PersonTarget.standing(activity="idle")
>>> person_running = PersonTarget.standing(activity="running")
>>> print(f"Idle temp: {person_idle.base_temperature_k:.1f}K")
>>> print(f"Running temp: {person_running.base_temperature_k:.1f}K")

# Example 2: Generate a crowd with random positions
>>> from eosim.targets import CrowdGenerator
>>> crowd = CrowdGenerator(count=20, area=(100, 100, 400, 500))
>>> targets = crowd.generate(activity_mix={"walking": 0.6, "idle": 0.4})
>>> renderer.render_multiple(targets.get_render_list())

# Example 3: Person with equipment (backpack, weapon)
>>> from eosim.targets import PersonTarget
>>> soldier = PersonTarget.standing(
...     activity="walking",
...     clothing="military",
...     equipment=["backpack", "rifle"],
... )
>>> signature = soldier.get_signature(aspect_angle_deg=90)
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
    TargetGroup,
)


@dataclass
class ClothingProperties:
    """Thermal properties of clothing.

    Attributes:
        emissivity: Clothing emissivity
        insulation: Thermal insulation factor (1=normal, 2=heavy)
        coverage: Body coverage fraction (0-1)
        color_factor: Solar absorption (0=white, 1=black)
    """
    emissivity: float = 0.95
    insulation: float = 1.0
    coverage: float = 0.8
    color_factor: float = 0.5

    @classmethod
    def summer(cls) -> "ClothingProperties":
        """Light summer clothing."""
        return cls(emissivity=0.95, insulation=0.5, coverage=0.5, color_factor=0.3)

    @classmethod
    def winter(cls) -> "ClothingProperties":
        """Heavy winter clothing."""
        return cls(emissivity=0.92, insulation=2.0, coverage=0.95, color_factor=0.5)

    @classmethod
    def military(cls) -> "ClothingProperties":
        """Military uniform."""
        return cls(emissivity=0.90, insulation=1.2, coverage=0.9, color_factor=0.6)

    @classmethod
    def athletic(cls) -> "ClothingProperties":
        """Athletic wear."""
        return cls(emissivity=0.95, insulation=0.3, coverage=0.4, color_factor=0.4)


class PersonTarget(Target):
    """Human target model.

    Models thermal signature based on body temperature, activity level,
    clothing, and environmental conditions.
    """

    # Core body temperature
    CORE_TEMP_K = 310.15  # 37°C

    # Activity metabolic rates (W/m² of body surface)
    ACTIVITY_METABOLIC = {
        "sleeping": 40,
        "sitting": 60,
        "idle": 70,
        "standing": 80,
        "walking": 120,
        "walking_fast": 170,
        "running": 300,
        "sprinting": 500,
    }

    def __init__(
        self,
        name: str = "person",
        pose: str = "standing",
        activity: str = "idle",
        clothing: Optional[ClothingProperties] = None,
        ambient_temperature_k: float = 290.0,
        equipment: Optional[list[str]] = None,
    ) -> None:
        """Initialize person target.

        Args:
            name: Person identifier
            pose: Body pose ("standing", "sitting", "prone", "crouching")
            activity: Activity level
            clothing: Clothing properties
            ambient_temperature_k: Ambient temperature
            equipment: List of equipment items
        """
        # Geometry depends on pose
        geometry = self._get_pose_geometry(pose)

        # Surface temperature depends on activity and clothing
        self.pose = pose
        self.activity = activity
        self.clothing = clothing or ClothingProperties()
        self.equipment = equipment or []

        surface_temp = self._calculate_surface_temperature(ambient_temperature_k)

        super().__init__(name, geometry, surface_temp, ambient_temperature_k)

        self._setup_body_hot_spots()

    def _get_pose_geometry(self, pose: str) -> TargetGeometry:
        """Get geometry for body pose."""
        if pose == "standing":
            return TargetGeometry(length_m=1.75, width_m=0.45, height_m=0.25)
        elif pose == "sitting":
            return TargetGeometry(length_m=1.2, width_m=0.5, height_m=0.4)
        elif pose == "prone":
            return TargetGeometry(length_m=0.4, width_m=1.8, height_m=0.25)
        elif pose == "crouching":
            return TargetGeometry(length_m=1.0, width_m=0.5, height_m=0.5)
        else:
            return TargetGeometry(length_m=1.75, width_m=0.45, height_m=0.25)

    def _calculate_surface_temperature(self, ambient_k: float) -> float:
        """Calculate effective surface temperature."""
        # Get metabolic rate
        metabolic = self.ACTIVITY_METABOLIC.get(self.activity, 80)

        # Heat loss model (simplified)
        # Higher metabolic rate -> higher surface temp
        # More insulation -> higher surface temp
        # Lower ambient -> lower surface temp (more heat loss)

        # Base skin temperature without clothing
        skin_temp = self.CORE_TEMP_K - 4  # ~33°C

        # Modify for activity (more activity -> more blood flow to skin)
        activity_factor = metabolic / 80  # Normalized to standing
        skin_temp += (activity_factor - 1) * 2

        # Clothing effect
        exposed_fraction = 1 - self.clothing.coverage
        clothed_temp = skin_temp - (skin_temp - ambient_k) * 0.3 / self.clothing.insulation

        # Weighted average of exposed and clothed areas
        surface_temp = exposed_fraction * skin_temp + (1 - exposed_fraction) * clothed_temp

        return surface_temp

    def _setup_body_hot_spots(self) -> None:
        """Configure body region hot spots."""
        # Head (warmest exposed area)
        self._hot_spots.append(HotSpot(
            name="head",
            relative_position=(0.5, 0.1),
            relative_size=0.15,
            temperature_delta_k=3.0,
        ))

        # Neck
        self._hot_spots.append(HotSpot(
            name="neck",
            relative_position=(0.5, 0.18),
            relative_size=0.08,
            temperature_delta_k=2.5,
        ))

        # Hands (if exposed)
        if self.clothing.coverage < 0.9:
            for y_pos in [0.2, 0.8]:
                self._hot_spots.append(HotSpot(
                    name=f"hand_{y_pos}",
                    relative_position=(y_pos, 0.5),
                    relative_size=0.06,
                    temperature_delta_k=2.0,
                ))

        # Chest/torso (core)
        self._hot_spots.append(HotSpot(
            name="torso",
            relative_position=(0.5, 0.35),
            relative_size=0.25,
            temperature_delta_k=1.5 if self.clothing.insulation < 1.5 else 0.5,
        ))

        # Equipment hot spots
        for equip in self.equipment:
            if equip == "backpack":
                self._hot_spots.append(HotSpot(
                    name="backpack",
                    relative_position=(0.5, 0.4),
                    relative_size=0.15,
                    temperature_delta_k=-2.0,  # Cooler (insulation)
                ))
            elif equip == "rifle":
                self._hot_spots.append(HotSpot(
                    name="rifle",
                    relative_position=(0.3, 0.5),
                    relative_size=0.08,
                    temperature_delta_k=-5.0,  # Metal, cooler
                ))
            elif equip == "radio":
                self._hot_spots.append(HotSpot(
                    name="radio",
                    relative_position=(0.7, 0.3),
                    relative_size=0.05,
                    temperature_delta_k=5.0,  # Electronics heat
                ))

    def get_signature(
        self,
        resolution: tuple[int, int] = (48, 24),
        aspect_angle_deg: float = 0.0,
        elevation_angle_deg: float = 0.0,
    ) -> TargetSignature:
        """Generate person thermal signature."""
        h, w = resolution

        # Create body shape based on pose
        mask = self._create_body_shape(resolution, aspect_angle_deg)

        # Temperature map
        temp_map = np.where(mask, self.base_temperature_k, 0.0)
        temp_map = self._apply_hot_spots(temp_map, self.base_temperature_k)

        # Add small variations
        noise = np.random.normal(0, 0.5, temp_map.shape)
        temp_map = np.where(mask, temp_map + noise, 0)

        # Emissivity (clothing + skin)
        emis_map = np.where(mask, self.clothing.emissivity, 0.0)

        return TargetSignature(
            temperature_map=temp_map,
            emissivity_map=emis_map,
            geometry=self.geometry,
            aspect_angle_deg=aspect_angle_deg,
        )

    def _create_body_shape(
        self,
        resolution: tuple[int, int],
        aspect_angle_deg: float,
    ) -> NDArray:
        """Create body silhouette based on pose."""
        h, w = resolution
        mask = np.zeros((h, w), dtype=bool)

        if self.pose == "standing":
            # Head
            head_cy = int(h * 0.1)
            head_cx = w // 2
            head_r = int(min(h, w) * 0.12)
            yy, xx = np.ogrid[:h, :w]
            mask |= (yy - head_cy)**2 + (xx - head_cx)**2 <= head_r**2

            # Torso
            torso_y0 = int(h * 0.15)
            torso_y1 = int(h * 0.55)
            torso_x0 = int(w * 0.25)
            torso_x1 = int(w * 0.75)
            mask[torso_y0:torso_y1, torso_x0:torso_x1] = True

            # Legs
            leg_y0 = int(h * 0.55)
            leg_y1 = h
            leg_w = int(w * 0.18)
            # Left leg
            mask[leg_y0:leg_y1, torso_x0:torso_x0+leg_w] = True
            # Right leg
            mask[leg_y0:leg_y1, torso_x1-leg_w:torso_x1] = True

            # Arms (if side view)
            if 45 < (aspect_angle_deg % 180) < 135:
                arm_y0 = int(h * 0.18)
                arm_y1 = int(h * 0.5)
                arm_x = int(w * 0.15)
                mask[arm_y0:arm_y1, arm_x:arm_x+3] = True
                mask[arm_y0:arm_y1, w-arm_x-3:w-arm_x] = True

        elif self.pose == "prone":
            # Lying down - elliptical shape
            cy, cx = h // 2, w // 2
            mask = ((yy - cy) / (h * 0.3))**2 + ((xx - cx) / (w * 0.45))**2 <= 1

        elif self.pose == "sitting":
            # Seated - upper body
            upper_h = int(h * 0.6)
            mask[:upper_h, int(w*0.2):int(w*0.8)] = True

        elif self.pose == "crouching":
            # Crouched - compact
            cy, cx = h // 2, w // 2
            r = int(min(h, w) * 0.4)
            yy, xx = np.ogrid[:h, :w]
            mask = (yy - cy)**2 + (xx - cx)**2 <= r**2

        return mask

    def set_activity(self, activity: str) -> None:
        """Change activity level and recalculate temperature."""
        self.activity = activity
        self.base_temperature_k = self._calculate_surface_temperature(
            self.ambient_temperature_k
        )
        self._hot_spots.clear()
        self._setup_body_hot_spots()

    @classmethod
    def standing(
        cls,
        activity: str = "idle",
        clothing: str = "normal",
        equipment: Optional[list[str]] = None,
    ) -> "PersonTarget":
        """Create a standing person."""
        clothing_map = {
            "normal": ClothingProperties(),
            "summer": ClothingProperties.summer(),
            "winter": ClothingProperties.winter(),
            "military": ClothingProperties.military(),
            "athletic": ClothingProperties.athletic(),
        }

        return cls(
            name="standing_person",
            pose="standing",
            activity=activity,
            clothing=clothing_map.get(clothing, ClothingProperties()),
            equipment=equipment,
        )

    @classmethod
    def prone(cls, activity: str = "idle") -> "PersonTarget":
        """Create a prone person."""
        return cls(
            name="prone_person",
            pose="prone",
            activity=activity,
        )

    @classmethod
    def sitting(cls, activity: str = "sitting") -> "PersonTarget":
        """Create a sitting person."""
        return cls(
            name="sitting_person",
            pose="sitting",
            activity=activity,
        )


class CrowdGenerator:
    """Generates groups of people with various characteristics.

    Creates crowds with configurable size, activity distribution,
    and spatial arrangement.
    """

    def __init__(
        self,
        count: int = 10,
        area: tuple[int, int, int, int] = (0, 0, 480, 640),
        min_separation: int = 30,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize crowd generator.

        Args:
            count: Number of people to generate
            area: (y_min, x_min, y_max, x_max) placement area
            min_separation: Minimum pixels between people
            seed: Random seed for reproducibility
        """
        self.count = count
        self.area = area
        self.min_separation = min_separation
        self._rng = np.random.default_rng(seed)

    def generate(
        self,
        activity_mix: Optional[dict[str, float]] = None,
        clothing_mix: Optional[dict[str, float]] = None,
        pose_mix: Optional[dict[str, float]] = None,
    ) -> TargetGroup:
        """Generate a crowd of people.

        Args:
            activity_mix: Dict of activity -> probability
            clothing_mix: Dict of clothing -> probability
            pose_mix: Dict of pose -> probability

        Returns:
            TargetGroup with generated people
        """
        # Default distributions
        if activity_mix is None:
            activity_mix = {"idle": 0.4, "walking": 0.5, "standing": 0.1}
        if clothing_mix is None:
            clothing_mix = {"normal": 0.8, "athletic": 0.2}
        if pose_mix is None:
            pose_mix = {"standing": 1.0}

        group = TargetGroup()

        for i in range(self.count):
            # Sample characteristics
            activity = self._sample_from_mix(activity_mix)
            clothing = self._sample_from_mix(clothing_mix)
            pose = self._sample_from_mix(pose_mix)

            # Create person
            person = PersonTarget(
                name=f"person_{i}",
                pose=pose,
                activity=activity,
                clothing=getattr(ClothingProperties, clothing, ClothingProperties)()
                if hasattr(ClothingProperties, clothing)
                else ClothingProperties(),
            )

            group.add(person)

        # Randomize positions
        group.randomize_positions(self.area, self.min_separation)

        return group

    def _sample_from_mix(self, mix: dict[str, float]) -> str:
        """Sample from probability distribution."""
        items = list(mix.keys())
        probs = np.array(list(mix.values()))
        probs = probs / probs.sum()
        return self._rng.choice(items, p=probs)

    def generate_formation(
        self,
        formation: str = "line",
        spacing_m: float = 2.0,
        gsd_m: float = 0.5,
    ) -> TargetGroup:
        """Generate people in a specific formation.

        Args:
            formation: Formation type ("line", "grid", "circle", "random")
            spacing_m: Spacing between people in meters
            gsd_m: Ground sample distance for conversion

        Returns:
            TargetGroup in formation
        """
        spacing_px = int(spacing_m / gsd_m)
        y_min, x_min, y_max, x_max = self.area
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2

        group = TargetGroup()
        positions = []

        if formation == "line":
            # Horizontal line
            start_x = center_x - (self.count - 1) * spacing_px // 2
            for i in range(self.count):
                positions.append((center_y, start_x + i * spacing_px))

        elif formation == "grid":
            # Square grid
            cols = int(np.ceil(np.sqrt(self.count)))
            rows = int(np.ceil(self.count / cols))
            start_y = center_y - (rows - 1) * spacing_px // 2
            start_x = center_x - (cols - 1) * spacing_px // 2

            for i in range(self.count):
                row = i // cols
                col = i % cols
                positions.append((start_y + row * spacing_px, start_x + col * spacing_px))

        elif formation == "circle":
            # Circle formation
            radius = spacing_px * self.count / (2 * np.pi)
            for i in range(self.count):
                angle = 2 * np.pi * i / self.count
                y = int(center_y + radius * np.sin(angle))
                x = int(center_x + radius * np.cos(angle))
                positions.append((y, x))

        else:  # random
            return self.generate()

        # Create people and assign positions
        for i, pos in enumerate(positions):
            person = PersonTarget.standing(activity="idle")
            group.add(person, position=pos, aspect_angle_deg=self._rng.uniform(0, 360))

        group.positions = positions
        return group
