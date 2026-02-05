"""
EOSIM Scene Module - Stage C Enhanced

Provides geometry, materials, and scene generation capabilities.

Example Usage:
--------------
# Example 1: Create a simple scene with a vehicle
>>> from eosim.scene import SceneGenerator, SceneConfig
>>> from eosim.targets import VehicleTarget
>>> config = SceneConfig(resolution=(480, 640), gsd_m=0.5)
>>> gen = SceneGenerator(config)
>>> vehicle = VehicleTarget.sedan(engine_state="running")
>>> gen.add_target(vehicle, position=(240, 320))
>>> result = gen.generate()
>>> print(f"Scene shape: {result.temperature_map.shape}")

# Example 2: Use a pre-defined scenario template
>>> from eosim.scene import create_military_scenario
>>> scene = create_military_scenario(num_vehicles=3, num_people=5)
>>> result = scene.generate()

# Example 3: Generate batch of randomized scenes
>>> from eosim.scene import SceneTemplate
>>> template = SceneTemplate.desert_environment()
>>> scenes = SceneTemplate.generate_batch(template, count=5)
"""

from eosim.scene.geometry import (
    BoundingBox,
    Transform,
    Geometry,
    TriangleMesh,
    HeightField,
    create_plane,
    create_box,
    load_obj,
)
from eosim.scene.materials import (
    BRDFType,
    SpectralProperty,
    ThermalProperties,
    BRDF,
    LambertianBRDF,
    OrenNayarBRDF,
    Material,
    MaterialLibrary,
    material_library,
    create_graybody,
    create_blackbody,
)
from eosim.scene.generator import (
    TimeOfDay,
    TerrainType,
    WeatherCondition,
    EnvironmentConfig,
    BackgroundConfig,
    SceneConfig,
    SceneResult,
    SceneGenerator,
    RandomizationConfig,
    SceneTemplate,
    create_military_scenario,
    create_surveillance_scenario,
)

__all__ = [
    # Geometry
    "BoundingBox",
    "Transform",
    "Geometry",
    "TriangleMesh",
    "HeightField",
    "create_plane",
    "create_box",
    "load_obj",
    # Materials
    "BRDFType",
    "SpectralProperty",
    "ThermalProperties",
    "BRDF",
    "LambertianBRDF",
    "OrenNayarBRDF",
    "Material",
    "MaterialLibrary",
    "material_library",
    "create_graybody",
    "create_blackbody",
    # Scene Generator (Stage C)
    "TimeOfDay",
    "TerrainType",
    "WeatherCondition",
    "EnvironmentConfig",
    "BackgroundConfig",
    "SceneConfig",
    "SceneResult",
    "SceneGenerator",
    "RandomizationConfig",
    "SceneTemplate",
    "create_military_scenario",
    "create_surveillance_scenario",
]
