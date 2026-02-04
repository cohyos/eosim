"""
Simulation pipeline module.

Provides the main simulation pipeline and effects processing.
"""

from eosim.pipeline.simulation import (
    SimulationMode,
    SimulationConfig,
    SceneInput,
    PipelineResult,
    SimulationPipeline,
    create_pipeline,
    quick_simulation,
)

from eosim.pipeline.effects import (
    MotionBlurParams,
    JitterParams,
    BloomParams,
    VignetteParams,
    BandingParams,
    DeadPixelParams,
    motion_blur_kernel,
    apply_motion_blur,
    apply_jitter,
    apply_blooming,
    apply_vignetting,
    apply_banding,
    apply_dead_pixels,
    EffectsChain,
)

__all__ = [
    # Simulation
    "SimulationMode",
    "SimulationConfig",
    "SceneInput",
    "PipelineResult",
    "SimulationPipeline",
    "create_pipeline",
    "quick_simulation",
    # Effects
    "MotionBlurParams",
    "JitterParams",
    "BloomParams",
    "VignetteParams",
    "BandingParams",
    "DeadPixelParams",
    "motion_blur_kernel",
    "apply_motion_blur",
    "apply_jitter",
    "apply_blooming",
    "apply_vignetting",
    "apply_banding",
    "apply_dead_pixels",
    "EffectsChain",
]
