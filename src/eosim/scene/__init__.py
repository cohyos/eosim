"""Scene module: geometry, materials, targets, and environment."""

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
]
