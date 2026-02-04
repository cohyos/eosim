"""
Image effects and post-processing for EOSIM.

Provides realistic imaging effects including:
- Motion blur
- Jitter/vibration
- Blooming
- Banding/striping artifacts
- Vignetting
"""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve, gaussian_filter


@dataclass
class MotionBlurParams:
    """Parameters for motion blur effect.

    Attributes:
        velocity_pixels_per_frame: Motion velocity in pixels per frame
        angle_deg: Motion direction angle in degrees
        exposure_samples: Number of samples for blur integration
    """
    velocity_pixels_per_frame: float = 5.0
    angle_deg: float = 0.0
    exposure_samples: int = 10


def motion_blur_kernel(
    length: float,
    angle_deg: float = 0.0,
    size: Optional[int] = None,
) -> NDArray[np.floating]:
    """Generate motion blur kernel.

    Args:
        length: Blur length in pixels
        angle_deg: Motion angle (0 = horizontal, 90 = vertical)
        size: Kernel size (auto if None)

    Returns:
        Normalized motion blur kernel
    """
    if size is None:
        size = int(np.ceil(length * 2)) + 1
        if size % 2 == 0:
            size += 1

    kernel = np.zeros((size, size))
    center = size // 2

    angle_rad = np.deg2rad(angle_deg)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    # Draw line through center
    for t in np.linspace(-length / 2, length / 2, int(length * 3)):
        x = int(round(center + t * dx))
        y = int(round(center + t * dy))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1

    # Normalize
    if kernel.sum() > 0:
        kernel /= kernel.sum()
    else:
        kernel[center, center] = 1.0

    return kernel


def apply_motion_blur(
    image: NDArray,
    params: MotionBlurParams,
) -> NDArray:
    """Apply motion blur to image.

    Args:
        image: Input image
        params: Motion blur parameters

    Returns:
        Blurred image
    """
    kernel = motion_blur_kernel(
        params.velocity_pixels_per_frame,
        params.angle_deg,
    )
    return convolve(image.astype(float), kernel, mode='reflect')


@dataclass
class JitterParams:
    """Parameters for platform jitter/vibration.

    Attributes:
        amplitude_pixels: RMS jitter amplitude in pixels
        frequency_frames: Jitter frequency in cycles per frame
        n_samples: Number of integration samples
    """
    amplitude_pixels: float = 0.5
    frequency_frames: float = 10.0
    n_samples: int = 20


def apply_jitter(
    image: NDArray,
    params: JitterParams,
    rng: Optional[np.random.Generator] = None,
) -> NDArray:
    """Apply platform jitter effect.

    Simulates high-frequency vibration during exposure.

    Args:
        image: Input image
        params: Jitter parameters
        rng: Random number generator

    Returns:
        Jittered image
    """
    if rng is None:
        rng = np.random.default_rng()

    from scipy.ndimage import shift

    # Generate random jitter offsets
    result = np.zeros_like(image, dtype=float)

    for i in range(params.n_samples):
        # Sinusoidal + random jitter
        t = i / params.n_samples
        dx = params.amplitude_pixels * (
            np.sin(2 * np.pi * params.frequency_frames * t) +
            rng.normal(0, 0.3)
        )
        dy = params.amplitude_pixels * (
            np.cos(2 * np.pi * params.frequency_frames * t) +
            rng.normal(0, 0.3)
        )

        shifted = shift(image.astype(float), [dy, dx], mode='reflect')
        result += shifted

    return result / params.n_samples


@dataclass
class BloomParams:
    """Parameters for sensor blooming effect.

    Attributes:
        threshold_fraction: Fraction of full well for bloom onset
        bloom_sigma: Gaussian spread of bloom
        bloom_strength: Relative strength of bloom
    """
    threshold_fraction: float = 0.9
    bloom_sigma: float = 3.0
    bloom_strength: float = 0.2


def apply_blooming(
    image: NDArray,
    params: BloomParams,
    full_well: float = 16383,
) -> NDArray:
    """Apply sensor blooming effect.

    Simulates charge overflow from saturated pixels.

    Args:
        image: Input image (DN)
        params: Blooming parameters
        full_well: Full well value (max DN)

    Returns:
        Image with blooming
    """
    threshold = params.threshold_fraction * full_well

    # Find saturated/near-saturated pixels
    overflow = np.maximum(image.astype(float) - threshold, 0)

    # Spread overflow to neighbors
    bloom = gaussian_filter(overflow, sigma=params.bloom_sigma)
    bloom *= params.bloom_strength

    # Add bloom back
    result = image.astype(float) + bloom

    return np.clip(result, 0, full_well)


@dataclass
class VignetteParams:
    """Parameters for vignetting effect.

    Attributes:
        strength: Vignette strength (0 = none, 1 = black corners)
        radius_fraction: Radius at which vignetting reaches strength
    """
    strength: float = 0.3
    radius_fraction: float = 1.0


def apply_vignetting(
    image: NDArray,
    params: VignetteParams,
) -> NDArray:
    """Apply optical vignetting effect.

    Args:
        image: Input image
        params: Vignetting parameters

    Returns:
        Vignetted image
    """
    h, w = image.shape[:2]
    y, x = np.ogrid[:h, :w]

    # Distance from center, normalized
    cy, cx = h / 2, w / 2
    max_r = np.sqrt(cx**2 + cy**2)
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max_r

    # Cosine^4 vignetting model
    r_norm = r / params.radius_fraction
    vignette = 1 - params.strength * np.minimum(r_norm, 1)**4

    return image * vignette


@dataclass
class BandingParams:
    """Parameters for row/column banding artifacts.

    Attributes:
        row_noise_amplitude: Row-wise noise amplitude
        col_noise_amplitude: Column-wise noise amplitude
        pattern_frequency: Periodic pattern frequency (0 = random)
    """
    row_noise_amplitude: float = 5.0
    col_noise_amplitude: float = 3.0
    pattern_frequency: float = 0.0


def apply_banding(
    image: NDArray,
    params: BandingParams,
    rng: Optional[np.random.Generator] = None,
) -> NDArray:
    """Apply row/column banding artifacts.

    Args:
        image: Input image
        params: Banding parameters
        rng: Random number generator

    Returns:
        Image with banding
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = image.shape[:2]
    result = image.astype(float)

    # Row noise (horizontal stripes)
    if params.row_noise_amplitude > 0:
        if params.pattern_frequency > 0:
            row_pattern = params.row_noise_amplitude * np.sin(
                2 * np.pi * params.pattern_frequency * np.arange(h) / h
            )
        else:
            row_pattern = rng.normal(0, params.row_noise_amplitude, h)
        result += row_pattern[:, np.newaxis]

    # Column noise (vertical stripes)
    if params.col_noise_amplitude > 0:
        if params.pattern_frequency > 0:
            col_pattern = params.col_noise_amplitude * np.sin(
                2 * np.pi * params.pattern_frequency * np.arange(w) / w
            )
        else:
            col_pattern = rng.normal(0, params.col_noise_amplitude, w)
        result += col_pattern[np.newaxis, :]

    return result


@dataclass
class DeadPixelParams:
    """Parameters for dead/hot pixel defects.

    Attributes:
        dead_fraction: Fraction of dead (stuck low) pixels
        hot_fraction: Fraction of hot (stuck high) pixels
        cluster_probability: Probability of defect clusters
    """
    dead_fraction: float = 0.0001
    hot_fraction: float = 0.0001
    cluster_probability: float = 0.1


def apply_dead_pixels(
    image: NDArray,
    params: DeadPixelParams,
    full_well: float = 16383,
    rng: Optional[np.random.Generator] = None,
) -> NDArray:
    """Apply dead and hot pixel defects.

    Args:
        image: Input image
        params: Dead pixel parameters
        full_well: Full well value
        rng: Random number generator

    Returns:
        Image with pixel defects
    """
    if rng is None:
        rng = np.random.default_rng()

    result = image.astype(float)
    h, w = image.shape[:2]
    n_pixels = h * w

    # Dead pixels (stuck at low value)
    n_dead = int(n_pixels * params.dead_fraction)
    if n_dead > 0:
        dead_y = rng.integers(0, h, n_dead)
        dead_x = rng.integers(0, w, n_dead)
        result[dead_y, dead_x] = rng.uniform(0, 10, n_dead)

    # Hot pixels (stuck at high value)
    n_hot = int(n_pixels * params.hot_fraction)
    if n_hot > 0:
        hot_y = rng.integers(0, h, n_hot)
        hot_x = rng.integers(0, w, n_hot)
        result[hot_y, hot_x] = rng.uniform(full_well * 0.9, full_well, n_hot)

    return result


class EffectsChain:
    """Chain of effects to apply sequentially."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize effects chain.

        Args:
            seed: Random seed
        """
        self._effects: list[tuple[Callable, dict]] = []
        self._rng = np.random.default_rng(seed)

    def add_motion_blur(self, params: MotionBlurParams) -> "EffectsChain":
        """Add motion blur effect."""
        self._effects.append((apply_motion_blur, {'params': params}))
        return self

    def add_jitter(self, params: JitterParams) -> "EffectsChain":
        """Add jitter effect."""
        self._effects.append((apply_jitter, {'params': params, 'rng': self._rng}))
        return self

    def add_blooming(self, params: BloomParams, full_well: float = 16383) -> "EffectsChain":
        """Add blooming effect."""
        self._effects.append((apply_blooming, {'params': params, 'full_well': full_well}))
        return self

    def add_vignetting(self, params: VignetteParams) -> "EffectsChain":
        """Add vignetting effect."""
        self._effects.append((apply_vignetting, {'params': params}))
        return self

    def add_banding(self, params: BandingParams) -> "EffectsChain":
        """Add banding effect."""
        self._effects.append((apply_banding, {'params': params, 'rng': self._rng}))
        return self

    def add_dead_pixels(self, params: DeadPixelParams, full_well: float = 16383) -> "EffectsChain":
        """Add dead pixel effect."""
        self._effects.append((apply_dead_pixels, {'params': params, 'full_well': full_well, 'rng': self._rng}))
        return self

    def apply(self, image: NDArray) -> NDArray:
        """Apply all effects in chain.

        Args:
            image: Input image

        Returns:
            Processed image
        """
        result = image.astype(float)
        for effect_fn, kwargs in self._effects:
            result = effect_fn(result, **kwargs)
        return result
