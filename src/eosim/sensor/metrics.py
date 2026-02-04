"""
Sensor performance metrics.

Provides calculations for key sensor performance metrics:
- NEΔT (Noise Equivalent Delta Temperature)
- SNR (Signal-to-Noise Ratio)
- Dynamic Range
- Detectivity (D*)
- Responsivity
"""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from eosim.core.constants import PLANCK_H, SPEED_OF_LIGHT, BOLTZMANN_K


@dataclass
class SensorPerformanceMetrics:
    """Collection of sensor performance metrics.

    Attributes:
        nedt_k: Noise Equivalent Delta Temperature in Kelvin
        snr: Signal-to-noise ratio at reference radiance
        dynamic_range_db: Dynamic range in dB
        well_fill_fraction: Fraction of full well used
        saturation_temperature_k: Temperature at which saturation occurs
        detectivity_star: Specific detectivity D* (cm·Hz^0.5/W)
    """
    nedt_k: float
    snr: float
    dynamic_range_db: float
    well_fill_fraction: float
    saturation_temperature_k: Optional[float] = None
    detectivity_star: Optional[float] = None

    def __str__(self) -> str:
        """String representation of metrics."""
        lines = [
            f"NEΔT: {self.nedt_k*1000:.2f} mK",
            f"SNR: {self.snr:.1f}",
            f"Dynamic Range: {self.dynamic_range_db:.1f} dB",
            f"Well Fill: {self.well_fill_fraction*100:.1f}%",
        ]
        if self.saturation_temperature_k is not None:
            lines.append(f"Saturation Temp: {self.saturation_temperature_k:.1f} K")
        if self.detectivity_star is not None:
            lines.append(f"D*: {self.detectivity_star:.2e} cm·Hz^0.5/W")
        return "\n".join(lines)


def compute_nedt(
    signal_electrons: float,
    noise_electrons: float,
    dL_dT: float,
    L: float,
) -> float:
    """Compute Noise Equivalent Delta Temperature.

    NEΔT = (σ_total / S) × (L / dL/dT) = σ_total / (dS/dT)

    Args:
        signal_electrons: Signal in electrons
        noise_electrons: Total noise in electrons (RMS)
        dL_dT: Derivative of radiance with respect to temperature
        L: Radiance at scene temperature

    Returns:
        NEΔT in Kelvin
    """
    if L <= 0 or dL_dT <= 0 or signal_electrons <= 0:
        return float('inf')

    # SNR
    snr = signal_electrons / noise_electrons

    # NEΔT = 1/SNR × L / (dL/dT)
    nedt = (1.0 / snr) * (L / dL_dT)

    return nedt


def compute_nedt_from_noise_components(
    signal_electrons: float,
    shot_noise_electrons: float,
    dark_noise_electrons: float,
    read_noise_electrons: float,
    prnu_noise_electrons: float,
    dsnu_noise_electrons: float,
    dL_dT: float,
    L: float,
) -> float:
    """Compute NEΔT from individual noise components.

    Args:
        signal_electrons: Signal in electrons
        shot_noise_electrons: Shot noise RMS
        dark_noise_electrons: Dark noise RMS
        read_noise_electrons: Read noise RMS
        prnu_noise_electrons: PRNU noise RMS
        dsnu_noise_electrons: DSNU noise RMS
        dL_dT: Derivative of radiance with respect to temperature
        L: Radiance at scene temperature

    Returns:
        NEΔT in Kelvin
    """
    # Total noise (root sum of squares)
    total_noise = np.sqrt(
        shot_noise_electrons**2
        + dark_noise_electrons**2
        + read_noise_electrons**2
        + prnu_noise_electrons**2
        + dsnu_noise_electrons**2
    )

    return compute_nedt(signal_electrons, total_noise, dL_dT, L)


def snr_from_electrons(
    signal_electrons: float,
    total_noise_electrons: float,
) -> float:
    """Compute signal-to-noise ratio.

    Args:
        signal_electrons: Signal in electrons
        total_noise_electrons: Total noise in electrons (RMS)

    Returns:
        SNR (dimensionless)
    """
    if total_noise_electrons <= 0:
        return float('inf') if signal_electrons > 0 else 0.0
    return signal_electrons / total_noise_electrons


def snr_shot_limited(signal_electrons: float) -> float:
    """Compute SNR in shot-noise limited regime.

    When shot noise dominates: SNR = sqrt(N)

    Args:
        signal_electrons: Signal in electrons

    Returns:
        Shot-noise limited SNR
    """
    if signal_electrons <= 0:
        return 0.0
    return np.sqrt(signal_electrons)


def snr_read_limited(
    signal_electrons: float,
    read_noise_electrons: float,
) -> float:
    """Compute SNR in read-noise limited regime.

    When read noise dominates: SNR = N / σ_read

    Args:
        signal_electrons: Signal in electrons
        read_noise_electrons: Read noise in electrons

    Returns:
        Read-noise limited SNR
    """
    if read_noise_electrons <= 0:
        return float('inf') if signal_electrons > 0 else 0.0
    return signal_electrons / read_noise_electrons


def snr_prnu_limited(prnu_percent: float) -> float:
    """Compute SNR in PRNU-limited regime.

    At high signal levels, SNR is limited by PRNU:
    SNR_max = 1 / PRNU_factor

    Args:
        prnu_percent: PRNU as percentage (e.g., 1.0 for 1%)

    Returns:
        PRNU-limited SNR
    """
    if prnu_percent <= 0:
        return float('inf')
    return 100.0 / prnu_percent


def dynamic_range_electrons(
    full_well_electrons: float,
    read_noise_electrons: float,
) -> float:
    """Compute dynamic range in dB.

    DR = 20 × log10(full_well / read_noise)

    Args:
        full_well_electrons: Full well capacity in electrons
        read_noise_electrons: Read noise in electrons

    Returns:
        Dynamic range in dB
    """
    if read_noise_electrons <= 0:
        return float('inf')
    return 20 * np.log10(full_well_electrons / read_noise_electrons)


def dynamic_range_temperature(
    min_detectable_temp_k: float,
    saturation_temp_k: float,
) -> float:
    """Compute temperature dynamic range.

    Args:
        min_detectable_temp_k: Minimum detectable temperature
        saturation_temp_k: Saturation temperature

    Returns:
        Temperature dynamic range in K
    """
    return saturation_temp_k - min_detectable_temp_k


def detectivity_star(
    responsivity_a_per_w: float,
    noise_current_a: float,
    detector_area_cm2: float,
    bandwidth_hz: float,
) -> float:
    """Compute specific detectivity D*.

    D* = R × sqrt(A × Δf) / i_n

    Args:
        responsivity_a_per_w: Responsivity in A/W
        noise_current_a: Noise current in A
        detector_area_cm2: Detector area in cm²
        bandwidth_hz: Noise bandwidth in Hz

    Returns:
        D* in cm·Hz^0.5/W
    """
    if noise_current_a <= 0:
        return float('inf')
    return responsivity_a_per_w * np.sqrt(detector_area_cm2 * bandwidth_hz) / noise_current_a


def blip_detectivity(
    wavelength_um: float,
    quantum_efficiency: float,
    background_photon_flux: float,
) -> float:
    """Compute background-limited D* (BLIP).

    D*_BLIP = (λ/hc) × sqrt(η / 2Q_B)

    Args:
        wavelength_um: Wavelength in micrometers
        quantum_efficiency: Quantum efficiency (0-1)
        background_photon_flux: Background photon flux (photons/s/cm²)

    Returns:
        BLIP D* in cm·Hz^0.5/W
    """
    wavelength_m = wavelength_um * 1e-6

    # Photon energy
    photon_energy = PLANCK_H * SPEED_OF_LIGHT / wavelength_m

    # D*_BLIP
    if background_photon_flux <= 0:
        return float('inf')

    d_star = (wavelength_m / (PLANCK_H * SPEED_OF_LIGHT)) * np.sqrt(
        quantum_efficiency / (2 * background_photon_flux)
    )

    # Convert to cm·Hz^0.5/W (multiply by 100 for cm)
    return d_star * 100


def responsivity(
    wavelength_um: float,
    quantum_efficiency: float,
    gain: float = 1.0,
) -> float:
    """Compute detector responsivity.

    R = (η × λ × G) / (h × c)

    Args:
        wavelength_um: Wavelength in micrometers
        quantum_efficiency: Quantum efficiency (0-1)
        gain: Internal gain factor (default 1)

    Returns:
        Responsivity in A/W
    """
    wavelength_m = wavelength_um * 1e-6

    # R = η × λ / (h × c) × q × G
    # Where q (electron charge) converts photons to current
    electron_charge = 1.602176634e-19

    r = (quantum_efficiency * wavelength_m * gain * electron_charge) / (
        PLANCK_H * SPEED_OF_LIGHT
    )

    return r


def noise_equivalent_power(
    detectivity_star: float,
    detector_area_cm2: float,
    bandwidth_hz: float,
) -> float:
    """Compute Noise Equivalent Power from D*.

    NEP = sqrt(A × Δf) / D*

    Args:
        detectivity_star: D* in cm·Hz^0.5/W
        detector_area_cm2: Detector area in cm²
        bandwidth_hz: Noise bandwidth in Hz

    Returns:
        NEP in W
    """
    if detectivity_star <= 0:
        return float('inf')
    return np.sqrt(detector_area_cm2 * bandwidth_hz) / detectivity_star


def minimum_resolvable_temperature_difference(
    nedt_k: float,
    mtf_at_frequency: float,
) -> float:
    """Compute Minimum Resolvable Temperature Difference (MRTD).

    MRTD ≈ NEΔT / MTF for a given spatial frequency.

    Args:
        nedt_k: NEΔT in Kelvin
        mtf_at_frequency: System MTF at the target spatial frequency

    Returns:
        MRTD in Kelvin
    """
    if mtf_at_frequency <= 0:
        return float('inf')
    return nedt_k / mtf_at_frequency


def compute_dL_dT_planck(
    temperature_k: float,
    wavelength_um: float,
    bandwidth_um: float = 1.0,
) -> float:
    """Compute dL/dT from Planck's law.

    Args:
        temperature_k: Temperature in Kelvin
        wavelength_um: Center wavelength in micrometers
        bandwidth_um: Spectral bandwidth in micrometers

    Returns:
        dL/dT in W/(m²·sr·K)
    """
    wavelength_m = wavelength_um * 1e-6
    T = temperature_k

    # Planck's law constants
    c1 = 2 * PLANCK_H * SPEED_OF_LIGHT**2
    c2 = PLANCK_H * SPEED_OF_LIGHT / BOLTZMANN_K

    # Planck radiance
    exp_term = np.exp(c2 / (wavelength_m * T))

    # dL/dT = L × (c2 / (λ × T²)) × exp(c2/(λT)) / (exp(c2/(λT)) - 1)
    L = c1 / (wavelength_m**5 * (exp_term - 1))
    dL_dT = L * (c2 / (wavelength_m * T**2)) * exp_term / (exp_term - 1)

    # Multiply by bandwidth (approximation for narrow band)
    dL_dT_band = dL_dT * (bandwidth_um * 1e-6)

    return dL_dT_band


def compute_contrast(
    target_signal: float,
    background_signal: float,
) -> float:
    """Compute target-background contrast.

    C = (S_target - S_background) / S_background

    Args:
        target_signal: Target signal level
        background_signal: Background signal level

    Returns:
        Contrast (dimensionless)
    """
    if background_signal <= 0:
        return float('inf') if target_signal > 0 else 0.0
    return (target_signal - background_signal) / background_signal


def contrast_threshold(snr: float, confidence: float = 0.95) -> float:
    """Compute minimum detectable contrast.

    C_min = k / SNR where k depends on confidence level.

    Args:
        snr: Signal-to-noise ratio
        confidence: Detection confidence (default 95%)

    Returns:
        Minimum detectable contrast
    """
    # k factor for different confidence levels
    # 95% -> k ≈ 1.65, 99% -> k ≈ 2.33
    from scipy.stats import norm
    k = norm.ppf(confidence)

    if snr <= 0:
        return float('inf')
    return k / snr


@dataclass
class NEDTBreakdown:
    """Breakdown of NEDT contributions from noise sources."""
    nedt_shot: float
    nedt_dark: float
    nedt_read: float
    nedt_prnu: float
    nedt_dsnu: float
    nedt_total: float

    def dominant_source(self) -> str:
        """Return the dominant noise source for NEDT."""
        sources = {
            'shot': self.nedt_shot,
            'dark': self.nedt_dark,
            'read': self.nedt_read,
            'prnu': self.nedt_prnu,
            'dsnu': self.nedt_dsnu,
        }
        return max(sources, key=sources.get)

    def as_dict(self) -> dict:
        """Return breakdown as dictionary."""
        return {
            'shot': self.nedt_shot,
            'dark': self.nedt_dark,
            'read': self.nedt_read,
            'prnu': self.nedt_prnu,
            'dsnu': self.nedt_dsnu,
            'total': self.nedt_total,
        }


def compute_nedt_breakdown(
    signal_electrons: float,
    shot_noise_electrons: float,
    dark_noise_electrons: float,
    read_noise_electrons: float,
    prnu_noise_electrons: float,
    dsnu_noise_electrons: float,
    dL_dT: float,
    L: float,
) -> NEDTBreakdown:
    """Compute NEDT breakdown by noise source.

    Args:
        signal_electrons: Signal in electrons
        shot_noise_electrons: Shot noise RMS
        dark_noise_electrons: Dark noise RMS
        read_noise_electrons: Read noise RMS
        prnu_noise_electrons: PRNU noise RMS
        dsnu_noise_electrons: DSNU noise RMS
        dL_dT: Derivative of radiance with respect to temperature
        L: Radiance at scene temperature

    Returns:
        NEDTBreakdown with individual contributions
    """
    def nedt_from_single_noise(noise: float) -> float:
        if signal_electrons <= 0 or L <= 0 or dL_dT <= 0:
            return float('inf')
        return (noise / signal_electrons) * (L / dL_dT)

    return NEDTBreakdown(
        nedt_shot=nedt_from_single_noise(shot_noise_electrons),
        nedt_dark=nedt_from_single_noise(dark_noise_electrons),
        nedt_read=nedt_from_single_noise(read_noise_electrons),
        nedt_prnu=nedt_from_single_noise(prnu_noise_electrons),
        nedt_dsnu=nedt_from_single_noise(dsnu_noise_electrons),
        nedt_total=compute_nedt_from_noise_components(
            signal_electrons,
            shot_noise_electrons,
            dark_noise_electrons,
            read_noise_electrons,
            prnu_noise_electrons,
            dsnu_noise_electrons,
            dL_dT,
            L,
        ),
    )
