"""
Noise models for FPA sensors.

Implements all major noise sources in the sensor signal chain:
- Shot noise (photon counting statistics)
- Dark current noise
- Read noise (amplifier noise)
- Fixed pattern noise (PRNU, DSNU)
- Quantization noise
"""

from dataclasses import dataclass, field
from typing import Optional, Union
from enum import Enum
import numpy as np
from numpy.typing import NDArray


class NoiseType(Enum):
    """Types of noise sources in FPA sensors."""
    SHOT = "shot"
    DARK = "dark"
    READ = "read"
    PRNU = "prnu"  # Photo Response Non-Uniformity
    DSNU = "dsnu"  # Dark Signal Non-Uniformity
    QUANTIZATION = "quantization"


@dataclass
class NoiseParameters:
    """Parameters defining sensor noise characteristics.

    Attributes:
        read_noise_electrons: RMS read noise in electrons
        dark_current_e_per_s: Dark current in electrons per second per pixel
        prnu_percent: PRNU as percentage (e.g., 1.0 for 1%)
        dsnu_electrons: DSNU standard deviation in electrons
        quantization_bits: ADC bit depth for quantization noise
        temperature_k: Detector temperature in Kelvin
        dark_current_doubling_temp_k: Temperature for dark current doubling
    """
    read_noise_electrons: float = 30.0
    dark_current_e_per_s: float = 1000.0
    prnu_percent: float = 1.0
    dsnu_electrons: float = 50.0
    quantization_bits: int = 14
    temperature_k: float = 77.0
    dark_current_doubling_temp_k: float = 7.0

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.read_noise_electrons < 0:
            raise ValueError("Read noise must be non-negative")
        if self.dark_current_e_per_s < 0:
            raise ValueError("Dark current must be non-negative")
        if self.prnu_percent < 0:
            raise ValueError("PRNU must be non-negative")
        if self.dsnu_electrons < 0:
            raise ValueError("DSNU must be non-negative")
        if self.quantization_bits < 1 or self.quantization_bits > 32:
            raise ValueError("Quantization bits must be between 1 and 32")
        if self.temperature_k <= 0:
            raise ValueError("Temperature must be positive")

    @property
    def prnu_factor(self) -> float:
        """PRNU as a fraction (0-1)."""
        return self.prnu_percent / 100.0

    def dark_current_at_temp(self, temp_k: float) -> float:
        """Compute dark current at a different temperature.

        Uses the rule of thumb that dark current doubles every ~7K.

        Args:
            temp_k: Target temperature in Kelvin

        Returns:
            Dark current at target temperature in e-/s/pixel
        """
        delta_t = temp_k - self.temperature_k
        doubling_factor = 2 ** (delta_t / self.dark_current_doubling_temp_k)
        return self.dark_current_e_per_s * doubling_factor


@dataclass
class NoiseContributions:
    """Breakdown of noise contributions.

    All values are variances (in electrons²) unless otherwise noted.
    """
    shot_variance: float = 0.0
    dark_variance: float = 0.0
    read_variance: float = 0.0
    prnu_variance: float = 0.0
    dsnu_variance: float = 0.0
    quantization_variance: float = 0.0

    @property
    def total_variance(self) -> float:
        """Total noise variance (sum of all components)."""
        return (
            self.shot_variance
            + self.dark_variance
            + self.read_variance
            + self.prnu_variance
            + self.dsnu_variance
            + self.quantization_variance
        )

    @property
    def total_noise_electrons(self) -> float:
        """Total noise in electrons (RMS)."""
        return np.sqrt(self.total_variance)

    def as_dict(self) -> dict:
        """Return contributions as dictionary."""
        return {
            "shot": self.shot_variance,
            "dark": self.dark_variance,
            "read": self.read_variance,
            "prnu": self.prnu_variance,
            "dsnu": self.dsnu_variance,
            "quantization": self.quantization_variance,
            "total": self.total_variance,
        }


class NoiseModel:
    """Complete noise model for FPA sensors.

    Applies the full noise chain to a signal in electrons:
    1. PRNU (multiplicative gain variation)
    2. Shot noise (Poisson statistics)
    3. Dark current addition
    4. DSNU (additive offset variation)
    5. Read noise (Gaussian amplifier noise)

    The noise variance follows:
    σ² = N_signal + N_dark + σ_read² + (PRNU × N)² + σ_dsnu²
    """

    def __init__(
        self,
        params: NoiseParameters,
        resolution: tuple[int, int],
        seed: Optional[int] = None,
    ) -> None:
        """Initialize noise model.

        Args:
            params: Noise parameters
            resolution: FPA resolution (height, width)
            seed: Random seed for reproducibility
        """
        self.params = params
        self.resolution = resolution
        self._rng = np.random.default_rng(seed)

        # Generate fixed pattern noise maps
        self._prnu_map = self._generate_prnu_map()
        self._dsnu_map = self._generate_dsnu_map()

    def _generate_prnu_map(self) -> NDArray[np.floating]:
        """Generate PRNU gain map.

        Returns:
            Multiplicative gain map centered at 1.0
        """
        prnu_sigma = self.params.prnu_factor
        return 1.0 + self._rng.normal(0, prnu_sigma, self.resolution)

    def _generate_dsnu_map(self) -> NDArray[np.floating]:
        """Generate DSNU offset map.

        Returns:
            Additive offset map in electrons
        """
        return self._rng.normal(0, self.params.dsnu_electrons, self.resolution)

    @property
    def prnu_map(self) -> NDArray[np.floating]:
        """Get the PRNU gain map."""
        return self._prnu_map.copy()

    @property
    def dsnu_map(self) -> NDArray[np.floating]:
        """Get the DSNU offset map."""
        return self._dsnu_map.copy()

    def apply(
        self,
        signal_electrons: NDArray[np.floating],
        integration_time_s: float,
        include_shot: bool = True,
        include_dark: bool = True,
        include_read: bool = True,
        include_prnu: bool = True,
        include_dsnu: bool = True,
    ) -> NDArray[np.floating]:
        """Apply noise model to signal.

        Args:
            signal_electrons: Input signal in electrons
            integration_time_s: Integration time in seconds
            include_shot: Include shot noise
            include_dark: Include dark current
            include_read: Include read noise
            include_prnu: Include PRNU
            include_dsnu: Include DSNU

        Returns:
            Noisy signal in electrons
        """
        signal = signal_electrons.copy().astype(np.float64)

        # 1. Apply PRNU (multiplicative)
        if include_prnu:
            signal = signal * self._prnu_map

        # 2. Add dark current
        dark_electrons = 0.0
        if include_dark:
            dark_electrons = self.params.dark_current_e_per_s * integration_time_s
            signal = signal + dark_electrons

        # 3. Apply shot noise (Poisson)
        if include_shot:
            # Clip to non-negative for Poisson
            signal_positive = np.maximum(signal, 0)
            signal = self._rng.poisson(signal_positive).astype(np.float64)

        # 4. Apply DSNU (additive offset)
        if include_dsnu:
            signal = signal + self._dsnu_map

        # 5. Apply read noise (Gaussian)
        if include_read:
            read_noise = self._rng.normal(
                0, self.params.read_noise_electrons, signal.shape
            )
            signal = signal + read_noise

        return signal

    def compute_contributions(
        self,
        signal_electrons: float,
        integration_time_s: float,
    ) -> NoiseContributions:
        """Compute individual noise contributions.

        Args:
            signal_electrons: Signal level in electrons
            integration_time_s: Integration time in seconds

        Returns:
            NoiseContributions with variance breakdown
        """
        # Shot noise variance = signal + dark
        dark_electrons = self.params.dark_current_e_per_s * integration_time_s
        shot_variance = signal_electrons + dark_electrons

        # Dark variance (already included in shot)
        dark_variance = dark_electrons

        # Read noise variance
        read_variance = self.params.read_noise_electrons ** 2

        # PRNU variance = (PRNU_factor × signal)²
        prnu_variance = (self.params.prnu_factor * signal_electrons) ** 2

        # DSNU variance
        dsnu_variance = self.params.dsnu_electrons ** 2

        # Quantization noise variance = LSB²/12
        # (computed but typically small)
        quantization_variance = 0.0  # Will be set by ADC model

        return NoiseContributions(
            shot_variance=shot_variance,
            dark_variance=dark_variance,
            read_variance=read_variance,
            prnu_variance=prnu_variance,
            dsnu_variance=dsnu_variance,
            quantization_variance=quantization_variance,
        )

    def reset_seed(self, seed: int) -> None:
        """Reset random number generator with new seed.

        Note: This does NOT regenerate FPN maps.
        """
        self._rng = np.random.default_rng(seed)

    def regenerate_fpn(self, seed: Optional[int] = None) -> None:
        """Regenerate fixed pattern noise maps.

        Args:
            seed: Optional new seed for FPN generation
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._prnu_map = self._generate_prnu_map()
        self._dsnu_map = self._generate_dsnu_map()


def shot_noise(
    signal_electrons: Union[float, NDArray],
    rng: Optional[np.random.Generator] = None,
) -> Union[float, NDArray]:
    """Apply shot noise to signal.

    Shot noise follows Poisson statistics where the variance
    equals the signal level.

    Args:
        signal_electrons: Input signal in electrons
        rng: Random number generator

    Returns:
        Signal with shot noise applied
    """
    if rng is None:
        rng = np.random.default_rng()

    signal = np.asarray(signal_electrons, dtype=np.float64)
    signal_positive = np.maximum(signal, 0)
    noisy = rng.poisson(signal_positive)

    if np.isscalar(signal_electrons):
        return float(noisy)
    return noisy.astype(np.float64)


def read_noise(
    shape: tuple[int, ...],
    sigma_electrons: float,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.floating]:
    """Generate read noise.

    Read noise is Gaussian-distributed with zero mean.

    Args:
        shape: Output shape
        sigma_electrons: RMS read noise in electrons
        rng: Random number generator

    Returns:
        Read noise array in electrons
    """
    if rng is None:
        rng = np.random.default_rng()

    return rng.normal(0, sigma_electrons, shape)


def dark_current_electrons(
    dark_current_e_per_s: float,
    integration_time_s: float,
    shape: tuple[int, ...],
    rng: Optional[np.random.Generator] = None,
    include_shot_noise: bool = True,
) -> NDArray[np.floating]:
    """Generate dark current signal.

    Args:
        dark_current_e_per_s: Dark current rate in e-/s/pixel
        integration_time_s: Integration time in seconds
        shape: Output shape
        rng: Random number generator
        include_shot_noise: Apply Poisson statistics to dark current

    Returns:
        Dark current signal in electrons
    """
    if rng is None:
        rng = np.random.default_rng()

    mean_dark = dark_current_e_per_s * integration_time_s

    if include_shot_noise:
        return rng.poisson(mean_dark, shape).astype(np.float64)
    else:
        return np.full(shape, mean_dark, dtype=np.float64)


def prnu_map(
    shape: tuple[int, int],
    prnu_percent: float,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.floating]:
    """Generate PRNU (Photo Response Non-Uniformity) map.

    PRNU represents pixel-to-pixel gain variation and is
    multiplicative (applied as signal × gain).

    Args:
        shape: FPA resolution (height, width)
        prnu_percent: PRNU as percentage (e.g., 1.0 for 1%)
        rng: Random number generator

    Returns:
        Multiplicative gain map centered at 1.0
    """
    if rng is None:
        rng = np.random.default_rng()

    prnu_sigma = prnu_percent / 100.0
    return 1.0 + rng.normal(0, prnu_sigma, shape)


def dsnu_map(
    shape: tuple[int, int],
    dsnu_electrons: float,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.floating]:
    """Generate DSNU (Dark Signal Non-Uniformity) map.

    DSNU represents pixel-to-pixel offset variation and is
    additive (applied as signal + offset).

    Args:
        shape: FPA resolution (height, width)
        dsnu_electrons: DSNU standard deviation in electrons
        rng: Random number generator

    Returns:
        Additive offset map in electrons
    """
    if rng is None:
        rng = np.random.default_rng()

    return rng.normal(0, dsnu_electrons, shape)


def total_noise_variance(
    signal_electrons: float,
    dark_electrons: float,
    read_noise_electrons: float,
    prnu_percent: float,
    dsnu_electrons: float,
) -> float:
    """Compute total noise variance analytically.

    σ² = N_signal + N_dark + σ_read² + (PRNU × N)² + σ_dsnu²

    Args:
        signal_electrons: Signal level in electrons
        dark_electrons: Dark current in electrons
        read_noise_electrons: Read noise RMS in electrons
        prnu_percent: PRNU as percentage
        dsnu_electrons: DSNU standard deviation in electrons

    Returns:
        Total noise variance in electrons²
    """
    prnu_factor = prnu_percent / 100.0

    shot_var = signal_electrons + dark_electrons
    read_var = read_noise_electrons ** 2
    prnu_var = (prnu_factor * signal_electrons) ** 2
    dsnu_var = dsnu_electrons ** 2

    return shot_var + read_var + prnu_var + dsnu_var


def snr_electrons(
    signal_electrons: float,
    dark_electrons: float,
    read_noise_electrons: float,
    prnu_percent: float = 0.0,
    dsnu_electrons: float = 0.0,
) -> float:
    """Compute signal-to-noise ratio.

    SNR = S / σ_total

    Args:
        signal_electrons: Signal level in electrons
        dark_electrons: Dark current in electrons
        read_noise_electrons: Read noise RMS in electrons
        prnu_percent: PRNU as percentage
        dsnu_electrons: DSNU standard deviation in electrons

    Returns:
        Signal-to-noise ratio (dimensionless)
    """
    variance = total_noise_variance(
        signal_electrons,
        dark_electrons,
        read_noise_electrons,
        prnu_percent,
        dsnu_electrons,
    )
    if variance <= 0:
        return float('inf') if signal_electrons > 0 else 0.0
    return signal_electrons / np.sqrt(variance)


@dataclass
class TemporalNoiseModel:
    """Model for temporal (frame-to-frame) noise analysis.

    Separates temporal noise (varies frame-to-frame) from
    fixed pattern noise (constant across frames).
    """

    read_noise_electrons: float = 30.0
    dark_current_e_per_s: float = 1000.0

    def temporal_noise_variance(
        self,
        signal_electrons: float,
        integration_time_s: float,
    ) -> float:
        """Compute temporal noise variance.

        Temporal noise includes shot noise, dark shot noise,
        and read noise (but NOT fixed pattern noise).

        Args:
            signal_electrons: Signal level in electrons
            integration_time_s: Integration time in seconds

        Returns:
            Temporal noise variance in electrons²
        """
        dark_electrons = self.dark_current_e_per_s * integration_time_s
        shot_var = signal_electrons + dark_electrons
        read_var = self.read_noise_electrons ** 2
        return shot_var + read_var

    def temporal_snr(
        self,
        signal_electrons: float,
        integration_time_s: float,
    ) -> float:
        """Compute temporal SNR.

        Args:
            signal_electrons: Signal level in electrons
            integration_time_s: Integration time in seconds

        Returns:
            Temporal signal-to-noise ratio
        """
        variance = self.temporal_noise_variance(signal_electrons, integration_time_s)
        if variance <= 0:
            return float('inf') if signal_electrons > 0 else 0.0
        return signal_electrons / np.sqrt(variance)


@dataclass
class SpatialNoiseModel:
    """Model for spatial (pixel-to-pixel) noise analysis.

    Focuses on fixed pattern noise characteristics.
    """

    prnu_percent: float = 1.0
    dsnu_electrons: float = 50.0

    def spatial_noise_variance(
        self,
        signal_electrons: float,
    ) -> float:
        """Compute spatial (fixed pattern) noise variance.

        Args:
            signal_electrons: Signal level in electrons

        Returns:
            Spatial noise variance in electrons²
        """
        prnu_factor = self.prnu_percent / 100.0
        prnu_var = (prnu_factor * signal_electrons) ** 2
        dsnu_var = self.dsnu_electrons ** 2
        return prnu_var + dsnu_var

    def prnu_limited_snr(
        self,
        signal_electrons: float,
    ) -> float:
        """Compute PRNU-limited SNR (high signal limit).

        At high signal levels, SNR becomes limited by PRNU:
        SNR_max ≈ 1 / PRNU_factor

        Args:
            signal_electrons: Signal level in electrons

        Returns:
            PRNU-limited SNR
        """
        if self.prnu_percent <= 0:
            return float('inf')
        return signal_electrons / (self.prnu_percent / 100.0 * signal_electrons)


def create_noise_model(
    resolution: tuple[int, int],
    read_noise_electrons: float = 30.0,
    dark_current_e_per_s: float = 1000.0,
    prnu_percent: float = 1.0,
    dsnu_electrons: float = 50.0,
    seed: Optional[int] = None,
) -> NoiseModel:
    """Factory function to create a NoiseModel.

    Args:
        resolution: FPA resolution (height, width)
        read_noise_electrons: RMS read noise
        dark_current_e_per_s: Dark current rate
        prnu_percent: PRNU percentage
        dsnu_electrons: DSNU standard deviation
        seed: Random seed

    Returns:
        Configured NoiseModel
    """
    params = NoiseParameters(
        read_noise_electrons=read_noise_electrons,
        dark_current_e_per_s=dark_current_e_per_s,
        prnu_percent=prnu_percent,
        dsnu_electrons=dsnu_electrons,
    )
    return NoiseModel(params, resolution, seed)
