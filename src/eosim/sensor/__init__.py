"""
Sensor module: FPA, noise models, ADC, and detector physics.

This module provides complete sensor simulation including:
- FPA geometry and detector properties
- Quantum efficiency models
- Noise models (shot, read, dark, PRNU, DSNU)
- ADC conversion
- Sensor performance metrics (NEDT, SNR)
"""

from eosim.sensor.fpa import (
    FPAGeometry,
    DetectorType,
    DetectorProperties,
    FPAConfig,
    SpectralResponse,
    compute_ifov,
    compute_gsd,
    compute_fov,
    compute_fov_2d,
    compute_nyquist_frequency,
    compute_pixel_solid_angle,
)

from eosim.sensor.quantum_efficiency import (
    QEModel,
    SpectrallyWeightedQE,
    create_qe_model,
    create_flat_qe,
    create_gaussian_qe,
)

from eosim.sensor.noise import (
    NoiseType,
    NoiseParameters,
    NoiseContributions,
    NoiseModel,
    TemporalNoiseModel,
    SpatialNoiseModel,
    shot_noise,
    read_noise,
    dark_current_electrons,
    prnu_map,
    dsnu_map,
    total_noise_variance,
    snr_electrons,
    create_noise_model,
)

from eosim.sensor.adc import (
    ADCType,
    ADCParameters,
    ADCResult,
    ADCModel,
    LogarithmicADC,
    compute_quantization_noise,
    required_bit_depth,
    create_adc,
)

from eosim.sensor.metrics import (
    SensorPerformanceMetrics,
    NEDTBreakdown,
    compute_nedt,
    compute_nedt_from_noise_components,
    compute_nedt_breakdown,
    snr_from_electrons,
    snr_shot_limited,
    snr_read_limited,
    snr_prnu_limited,
    dynamic_range_electrons,
    dynamic_range_temperature,
    detectivity_star,
    blip_detectivity,
    responsivity,
    noise_equivalent_power,
    minimum_resolvable_temperature_difference,
    compute_dL_dT_planck,
    compute_contrast,
    contrast_threshold,
)

from eosim.sensor.base import (
    SensorParameters,
    SensorResult,
    SensorModel,
    SimpleSensorModel,
    create_sensor_model,
)

__all__ = [
    # FPA
    "FPAGeometry",
    "DetectorType",
    "DetectorProperties",
    "FPAConfig",
    "SpectralResponse",
    "compute_ifov",
    "compute_gsd",
    "compute_fov",
    "compute_fov_2d",
    "compute_nyquist_frequency",
    "compute_pixel_solid_angle",
    # Quantum Efficiency
    "QEModel",
    "SpectrallyWeightedQE",
    "create_qe_model",
    "create_flat_qe",
    "create_gaussian_qe",
    # Noise
    "NoiseType",
    "NoiseParameters",
    "NoiseContributions",
    "NoiseModel",
    "TemporalNoiseModel",
    "SpatialNoiseModel",
    "shot_noise",
    "read_noise",
    "dark_current_electrons",
    "prnu_map",
    "dsnu_map",
    "total_noise_variance",
    "snr_electrons",
    "create_noise_model",
    # ADC
    "ADCType",
    "ADCParameters",
    "ADCResult",
    "ADCModel",
    "LogarithmicADC",
    "compute_quantization_noise",
    "required_bit_depth",
    "create_adc",
    # Metrics
    "SensorPerformanceMetrics",
    "NEDTBreakdown",
    "compute_nedt",
    "compute_nedt_from_noise_components",
    "compute_nedt_breakdown",
    "snr_from_electrons",
    "snr_shot_limited",
    "snr_read_limited",
    "snr_prnu_limited",
    "dynamic_range_electrons",
    "dynamic_range_temperature",
    "detectivity_star",
    "blip_detectivity",
    "responsivity",
    "noise_equivalent_power",
    "minimum_resolvable_temperature_difference",
    "compute_dL_dT_planck",
    "compute_contrast",
    "contrast_threshold",
    # Sensor Model
    "SensorParameters",
    "SensorResult",
    "SensorModel",
    "SimpleSensorModel",
    "create_sensor_model",
]
