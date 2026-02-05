"""
EOSIM Calibration Module (Stage F).

Provides radiometric calibration, non-uniformity correction,
bad pixel replacement, and temperature calibration tools.

Example 1: Non-Uniformity Correction (NUC)
    >>> from eosim.calibration import NUCCalibrator, TwoPointNUC
    >>> # Capture calibration frames at two temperatures
    >>> cold_frame = capture_frame(blackbody_temp=293)
    >>> hot_frame = capture_frame(blackbody_temp=323)
    >>> # Compute NUC coefficients
    >>> nuc = TwoPointNUC()
    >>> gain, offset = nuc.calibrate(cold_frame, hot_frame, T_cold=293, T_hot=323)
    >>> # Apply correction
    >>> corrected = nuc.apply(raw_frame, gain, offset)

Example 2: Radiometric calibration (DN to temperature)
    >>> from eosim.calibration import RadiometricCalibrator
    >>> cal = RadiometricCalibrator()
    >>> # Fit calibration curve from blackbody measurements
    >>> cal.fit(dn_values=[1000, 5000, 10000], temperatures=[280, 300, 320])
    >>> # Convert DN image to temperature
    >>> temp_image = cal.dn_to_temperature(raw_image)
    >>> # Or convert to radiance
    >>> radiance = cal.dn_to_radiance(raw_image, wavelength_um=10.0)

Example 3: Bad pixel correction
    >>> from eosim.calibration import BadPixelCorrector
    >>> corrector = BadPixelCorrector()
    >>> # Detect bad pixels from flat field
    >>> bad_map = corrector.detect_bad_pixels(flat_frame, threshold_sigma=3.0)
    >>> # Correct image using interpolation
    >>> corrected = corrector.correct(raw_image, bad_map)
    >>> print(f"Corrected {bad_map.sum()} bad pixels")
"""

from eosim.calibration.radiometric import (
    RadiometricCalibrator,
    CalibrationCurve,
    dn_to_radiance,
    radiance_to_temperature,
    temperature_to_radiance,
)
from eosim.calibration.nuc import (
    TwoPointNUC,
    MultiPointNUC,
    NUCCoefficients,
    apply_nuc,
    compute_nuc_coefficients,
)
from eosim.calibration.corrections import (
    BadPixelCorrector,
    FlatFieldCorrector,
    BadPixelMap,
    detect_bad_pixels,
    correct_bad_pixels,
    apply_flat_field,
)

__all__ = [
    # Radiometric
    "RadiometricCalibrator",
    "CalibrationCurve",
    "dn_to_radiance",
    "radiance_to_temperature",
    "temperature_to_radiance",
    # NUC
    "TwoPointNUC",
    "MultiPointNUC",
    "NUCCoefficients",
    "apply_nuc",
    "compute_nuc_coefficients",
    # Corrections
    "BadPixelCorrector",
    "FlatFieldCorrector",
    "BadPixelMap",
    "detect_bad_pixels",
    "correct_bad_pixels",
    "apply_flat_field",
]
