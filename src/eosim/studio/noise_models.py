"""
Advanced noise models for electro-optical sensor simulation.

This module provides realistic noise modeling including:
- Shot noise (photon noise)
- Readout noise
- Dark current noise
- Fixed pattern noise (FPN)
- 1/f (flicker) noise
- Temporal random noise
- Quantization noise
- Electromagnetic interference (EMI)
- Microphonics
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import ndimage
from scipy import signal


class NoiseType(Enum):
    """Types of sensor noise."""
    SHOT = "shot"                   # Photon shot noise
    READOUT = "readout"             # Read noise (Gaussian)
    DARK = "dark"                   # Dark current shot noise
    FPN_OFFSET = "fpn_offset"       # Fixed pattern - offset (DSNU)
    FPN_GAIN = "fpn_gain"           # Fixed pattern - gain (PRNU)
    FLICKER = "flicker"             # 1/f noise
    TEMPORAL = "temporal"           # Temporal random noise
    QUANTIZATION = "quantization"   # ADC quantization noise
    EMI = "emi"                     # Electromagnetic interference
    MICROPHONICS = "microphonics"   # Mechanical vibration noise
    ROW_NOISE = "row_noise"         # Row-correlated noise
    COLUMN_NOISE = "column_noise"   # Column-correlated noise
    STRIPING = "striping"           # Periodic striping artifacts


@dataclass
class NoiseParameters:
    """Parameters for noise simulation."""
    # Enable/disable individual noise sources
    enable_shot_noise: bool = True
    enable_readout_noise: bool = True
    enable_dark_noise: bool = True
    enable_fpn: bool = True
    enable_flicker_noise: bool = False
    enable_temporal_noise: bool = True
    enable_quantization_noise: bool = True
    enable_emi: bool = False
    enable_microphonics: bool = False
    enable_row_noise: bool = False
    enable_column_noise: bool = False

    # Noise levels (in electrons or appropriate units)
    readout_noise_e: float = 50.0      # RMS read noise in electrons
    dark_current_e_s: float = 100.0    # Dark current in e-/s
    integration_time_s: float = 0.01   # Integration time

    # Fixed pattern noise (as fraction of signal)
    dsnu_percent: float = 1.0          # Dark signal non-uniformity
    prnu_percent: float = 2.0          # Photo-response non-uniformity

    # 1/f noise
    flicker_knee_hz: float = 100.0     # Corner frequency
    flicker_amplitude: float = 0.5     # Relative amplitude

    # ADC parameters
    adc_bits: int = 14
    well_capacity_e: int = 10_000_000

    # EMI parameters
    emi_frequency_hz: float = 60.0     # Power line frequency
    emi_amplitude: float = 0.1         # Relative amplitude

    # Microphonics
    microphonics_freq_hz: float = 30.0
    microphonics_amplitude: float = 0.05

    # Row/column noise
    row_noise_amplitude: float = 0.5   # In DN
    column_noise_amplitude: float = 0.5


@dataclass
class NoiseState:
    """Internal state for correlated/temporal noise."""
    fpn_offset_map: Optional[np.ndarray] = None
    fpn_gain_map: Optional[np.ndarray] = None
    flicker_phase: float = 0.0
    emi_phase: float = 0.0
    frame_count: int = 0
    temporal_buffer: Optional[np.ndarray] = None


class NoiseGenerator:
    """
    Comprehensive noise generator for sensor simulation.

    Generates physically realistic noise for infrared detector arrays.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        params: Optional[NoiseParameters] = None,
        seed: Optional[int] = None
    ):
        self.shape = shape
        self.params = params or NoiseParameters()
        self.state = NoiseState()

        if seed is not None:
            np.random.seed(seed)

        # Initialize fixed pattern maps
        self._init_fpn()

    def _init_fpn(self):
        """Initialize fixed pattern noise maps."""
        # DSNU - offset non-uniformity (additive)
        self.state.fpn_offset_map = np.random.randn(*self.shape).astype(np.float32)
        self.state.fpn_offset_map *= self.params.dsnu_percent / 100.0

        # PRNU - gain non-uniformity (multiplicative)
        self.state.fpn_gain_map = 1.0 + np.random.randn(*self.shape).astype(np.float32)
        self.state.fpn_gain_map *= self.params.prnu_percent / 100.0
        self.state.fpn_gain_map = np.clip(self.state.fpn_gain_map, 0.8, 1.2)

    def generate_shot_noise(self, signal_electrons: np.ndarray) -> np.ndarray:
        """
        Generate photon shot noise.

        Shot noise follows Poisson statistics, approximated as Gaussian
        for large electron counts.

        Args:
            signal_electrons: Signal level in electrons per pixel

        Returns:
            Shot noise in electrons
        """
        # Poisson noise: σ = √N
        sigma = np.sqrt(np.maximum(signal_electrons, 0))

        # For low counts, use true Poisson; for high counts, use Gaussian
        noise = np.zeros_like(signal_electrons)

        low_count_mask = signal_electrons < 100
        if np.any(low_count_mask):
            # True Poisson for low counts
            low_counts = signal_electrons[low_count_mask]
            noise[low_count_mask] = (np.random.poisson(np.maximum(low_counts, 0).astype(int))
                                     - low_counts)

        high_count_mask = ~low_count_mask
        if np.any(high_count_mask):
            # Gaussian approximation for high counts
            noise[high_count_mask] = np.random.randn(np.sum(high_count_mask)) * sigma[high_count_mask]

        return noise.astype(np.float32)

    def generate_readout_noise(self) -> np.ndarray:
        """
        Generate readout noise.

        Readout noise is Gaussian-distributed and independent per pixel per frame.

        Returns:
            Readout noise in electrons
        """
        return (np.random.randn(*self.shape) *
                self.params.readout_noise_e).astype(np.float32)

    def generate_dark_noise(self) -> np.ndarray:
        """
        Generate dark current noise.

        Dark current is a Poisson process, contributing both signal and noise.

        Returns:
            Dark noise in electrons
        """
        # Expected dark electrons
        dark_electrons = (self.params.dark_current_e_s *
                         self.params.integration_time_s)

        # Shot noise on dark current
        if dark_electrons > 0:
            sigma = math.sqrt(dark_electrons)
            return (np.random.randn(*self.shape) * sigma).astype(np.float32)
        return np.zeros(self.shape, dtype=np.float32)

    def generate_fpn_offset(self, scale: float = 1.0) -> np.ndarray:
        """
        Generate fixed pattern noise (offset/DSNU).

        This is constant per pixel across frames.

        Args:
            scale: Scaling factor for noise level

        Returns:
            FPN offset in electrons or DN
        """
        return self.state.fpn_offset_map * scale

    def generate_fpn_gain(self) -> np.ndarray:
        """
        Generate fixed pattern noise (gain/PRNU).

        This is a multiplicative factor per pixel.

        Returns:
            PRNU gain map (multiplicative)
        """
        return self.state.fpn_gain_map

    def generate_flicker_noise(self, frame_rate: float = 30.0) -> np.ndarray:
        """
        Generate 1/f (flicker) noise.

        1/f noise has a power spectrum proportional to 1/f.

        Args:
            frame_rate: Frame rate for temporal coherence

        Returns:
            Flicker noise contribution
        """
        # Generate 1/f spectrum in frequency domain
        h, w = self.shape
        fy = np.fft.fftfreq(h)
        fx = np.fft.fftfreq(w)
        fx_grid, fy_grid = np.meshgrid(fx, fy)
        f = np.sqrt(fx_grid**2 + fy_grid**2)
        f[0, 0] = 1  # Avoid division by zero

        # 1/f power spectrum (pink noise)
        knee = self.params.flicker_knee_hz / (frame_rate * max(h, w))
        spectrum = 1.0 / np.sqrt(f + knee)
        spectrum[0, 0] = 0  # Remove DC

        # Random phase
        phase = np.random.random(self.shape) * 2 * np.pi
        complex_noise = spectrum * np.exp(1j * phase)

        # Inverse FFT
        noise = np.real(np.fft.ifft2(complex_noise))

        # Normalize and scale
        noise = noise / np.std(noise) * self.params.flicker_amplitude

        return noise.astype(np.float32)

    def generate_temporal_noise(self) -> np.ndarray:
        """
        Generate temporally correlated noise.

        This simulates noise that changes slowly over time.

        Returns:
            Temporal noise contribution
        """
        self.state.frame_count += 1

        # Low-pass filtered random noise
        raw_noise = np.random.randn(*self.shape)
        filtered = ndimage.gaussian_filter(raw_noise, sigma=2)

        # Add temporal correlation via phase
        temporal_factor = 0.5 + 0.5 * np.sin(self.state.frame_count * 0.1)
        noise = filtered * temporal_factor

        return noise.astype(np.float32)

    def generate_quantization_noise(self, signal_dn: np.ndarray) -> np.ndarray:
        """
        Generate quantization noise from ADC.

        Quantization noise is uniform distribution with variance = (LSB²)/12.

        Args:
            signal_dn: Signal in digital numbers

        Returns:
            Quantized signal
        """
        max_dn = 2**self.params.adc_bits - 1

        # Quantize
        quantized = np.round(signal_dn)
        quantized = np.clip(quantized, 0, max_dn)

        return quantized.astype(np.float32)

    def generate_emi_noise(self, time_s: float = 0.0) -> np.ndarray:
        """
        Generate electromagnetic interference noise.

        Models power line interference and other periodic EMI sources.

        Args:
            time_s: Current time for phase coherence

        Returns:
            EMI noise pattern
        """
        freq = self.params.emi_frequency_hz
        amplitude = self.params.emi_amplitude

        # Horizontal banding from power line
        h, w = self.shape
        y_coords = np.arange(h) / h

        # Phase progression down the image (simulating rolling shutter)
        phase = 2 * np.pi * freq * time_s
        emi = amplitude * np.sin(2 * np.pi * y_coords * 2 + phase)

        # Expand to full image
        emi_2d = np.tile(emi[:, np.newaxis], (1, w))

        return emi_2d.astype(np.float32)

    def generate_microphonics(self, time_s: float = 0.0) -> np.ndarray:
        """
        Generate microphonic noise from mechanical vibration.

        Models cooler vibration and other mechanical noise sources.

        Args:
            time_s: Current time for temporal coherence

        Returns:
            Microphonic noise pattern
        """
        freq = self.params.microphonics_freq_hz
        amplitude = self.params.microphonics_amplitude

        # Circular pattern modulation (from cooler vibration)
        h, w = self.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r_norm = r / max(h, w)

        phase = 2 * np.pi * freq * time_s
        pattern = amplitude * np.sin(r_norm * 4 * np.pi + phase)

        return pattern.astype(np.float32)

    def generate_row_noise(self) -> np.ndarray:
        """
        Generate row-correlated noise.

        Common in rolling shutter sensors and certain readout architectures.

        Returns:
            Row noise pattern
        """
        h, w = self.shape
        row_values = np.random.randn(h) * self.params.row_noise_amplitude
        return np.tile(row_values[:, np.newaxis], (1, w)).astype(np.float32)

    def generate_column_noise(self) -> np.ndarray:
        """
        Generate column-correlated noise.

        Common in column-parallel ADC architectures.

        Returns:
            Column noise pattern
        """
        h, w = self.shape
        col_values = np.random.randn(w) * self.params.column_noise_amplitude
        return np.tile(col_values[np.newaxis, :], (h, 1)).astype(np.float32)

    def generate_striping(
        self,
        orientation: str = "horizontal",
        frequency: float = 0.1
    ) -> np.ndarray:
        """
        Generate periodic striping artifacts.

        Args:
            orientation: "horizontal" or "vertical"
            frequency: Stripe frequency (stripes per pixel)

        Returns:
            Striping pattern
        """
        h, w = self.shape

        if orientation == "horizontal":
            coords = np.arange(h) / h
            pattern = np.sin(2 * np.pi * coords * h * frequency)
            stripes = np.tile(pattern[:, np.newaxis], (1, w))
        else:
            coords = np.arange(w) / w
            pattern = np.sin(2 * np.pi * coords * w * frequency)
            stripes = np.tile(pattern[np.newaxis, :], (h, 1))

        return (stripes * 0.1).astype(np.float32)

    def add_all_noise(
        self,
        signal_electrons: np.ndarray,
        time_s: float = 0.0,
        params: Optional[NoiseParameters] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Add all enabled noise sources to a signal.

        Args:
            signal_electrons: Clean signal in electrons
            time_s: Current time for temporal effects
            params: Optional parameter override

        Returns:
            Tuple of (noisy signal, noise statistics dict)
        """
        if params is not None:
            self.params = params

        result = signal_electrons.copy().astype(np.float32)
        noise_stats = {}

        # Apply PRNU (multiplicative)
        if self.params.enable_fpn:
            prnu = self.generate_fpn_gain()
            result = result * prnu
            noise_stats['prnu_rms'] = float(np.std(prnu - 1))

        # Shot noise
        if self.params.enable_shot_noise:
            shot = self.generate_shot_noise(result)
            result += shot
            noise_stats['shot_rms'] = float(np.std(shot))

        # Dark current noise
        if self.params.enable_dark_noise:
            dark = self.generate_dark_noise()
            result += dark
            noise_stats['dark_rms'] = float(np.std(dark))

        # Readout noise
        if self.params.enable_readout_noise:
            readout = self.generate_readout_noise()
            result += readout
            noise_stats['readout_rms'] = float(np.std(readout))

        # FPN offset (DSNU)
        if self.params.enable_fpn:
            dsnu = self.generate_fpn_offset(scale=signal_electrons.mean() * 0.01)
            result += dsnu
            noise_stats['dsnu_rms'] = float(np.std(dsnu))

        # Flicker noise
        if self.params.enable_flicker_noise:
            flicker = self.generate_flicker_noise()
            flicker_scaled = flicker * signal_electrons.mean() * 0.01
            result += flicker_scaled
            noise_stats['flicker_rms'] = float(np.std(flicker_scaled))

        # Temporal noise
        if self.params.enable_temporal_noise:
            temporal = self.generate_temporal_noise()
            temporal_scaled = temporal * self.params.readout_noise_e * 0.5
            result += temporal_scaled
            noise_stats['temporal_rms'] = float(np.std(temporal_scaled))

        # EMI
        if self.params.enable_emi:
            emi = self.generate_emi_noise(time_s)
            emi_scaled = emi * signal_electrons.mean()
            result += emi_scaled
            noise_stats['emi_rms'] = float(np.std(emi_scaled))

        # Microphonics
        if self.params.enable_microphonics:
            micro = self.generate_microphonics(time_s)
            micro_scaled = micro * signal_electrons.mean()
            result += micro_scaled
            noise_stats['micro_rms'] = float(np.std(micro_scaled))

        # Row noise
        if self.params.enable_row_noise:
            row = self.generate_row_noise()
            result += row
            noise_stats['row_rms'] = float(np.std(row))

        # Column noise
        if self.params.enable_column_noise:
            col = self.generate_column_noise()
            result += col
            noise_stats['column_rms'] = float(np.std(col))

        # Clip to positive
        result = np.maximum(result, 0)

        # Convert to DN and apply quantization
        gain = self.params.well_capacity_e / (2**self.params.adc_bits - 1)
        signal_dn = result / gain

        if self.params.enable_quantization_noise:
            signal_dn = self.generate_quantization_noise(signal_dn)

        # Calculate total noise
        noise_stats['total_rms'] = float(np.std(signal_dn - signal_electrons / gain))

        return signal_dn, noise_stats


class SpatialNoiseFilter:
    """
    Spatial noise filtering algorithms.

    Provides various methods to reduce spatial noise while preserving detail.
    """

    @staticmethod
    def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
        """Apply median filter for salt-and-pepper noise."""
        return ndimage.median_filter(image, size=size)

    @staticmethod
    def gaussian_filter(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian smoothing."""
        return ndimage.gaussian_filter(image, sigma=sigma)

    @staticmethod
    def bilateral_filter(
        image: np.ndarray,
        spatial_sigma: float = 2.0,
        range_sigma: float = 0.1
    ) -> np.ndarray:
        """
        Apply bilateral filter (edge-preserving smoothing).

        Args:
            image: Input image (normalized 0-1)
            spatial_sigma: Spatial kernel sigma
            range_sigma: Range (intensity) sigma

        Returns:
            Filtered image
        """
        # Simplified bilateral implementation
        h, w = image.shape[:2]
        kernel_size = int(spatial_sigma * 3) * 2 + 1

        # Create spatial kernel
        y, x = np.ogrid[-kernel_size//2:kernel_size//2+1,
                        -kernel_size//2:kernel_size//2+1]
        spatial_kernel = np.exp(-(x**2 + y**2) / (2 * spatial_sigma**2))

        # Pad image
        pad = kernel_size // 2
        padded = np.pad(image, pad, mode='reflect')

        result = np.zeros_like(image)

        for i in range(h):
            for j in range(w):
                # Extract neighborhood
                neighborhood = padded[i:i+kernel_size, j:j+kernel_size]
                center_val = image[i, j]

                # Range kernel
                range_kernel = np.exp(-((neighborhood - center_val)**2) /
                                      (2 * range_sigma**2))

                # Combined kernel
                kernel = spatial_kernel * range_kernel
                kernel = kernel / kernel.sum()

                result[i, j] = np.sum(neighborhood * kernel)

        return result

    @staticmethod
    def wiener_filter(
        image: np.ndarray,
        noise_variance: float = 0.01
    ) -> np.ndarray:
        """
        Apply Wiener filter for noise reduction.

        Args:
            image: Input image
            noise_variance: Estimated noise variance

        Returns:
            Filtered image
        """
        # FFT of image
        f_image = np.fft.fft2(image)
        power = np.abs(f_image)**2

        # Wiener filter: H* / (|H|² + N/S)
        # Assuming H = 1 (no blur), this simplifies to S / (S + N)
        signal_power = power.mean()
        noise_power = noise_variance * image.size

        wiener = power / (power + noise_power / signal_power)
        filtered = np.fft.ifft2(f_image * wiener)

        return np.real(filtered)


class TemporalNoiseFilter:
    """
    Temporal noise filtering for video sequences.

    Reduces noise by averaging across frames while handling motion.
    """

    def __init__(self, buffer_size: int = 8):
        self.buffer_size = buffer_size
        self.frame_buffer: List[np.ndarray] = []

    def add_frame(self, frame: np.ndarray):
        """Add a frame to the buffer."""
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

    def get_filtered_frame(
        self,
        method: str = "mean"
    ) -> Optional[np.ndarray]:
        """
        Get temporally filtered frame.

        Args:
            method: Filtering method ("mean", "median", "recursive")

        Returns:
            Filtered frame or None if buffer not ready
        """
        if len(self.frame_buffer) < 2:
            return self.frame_buffer[-1] if self.frame_buffer else None

        stack = np.stack(self.frame_buffer, axis=0)

        if method == "mean":
            return np.mean(stack, axis=0)
        elif method == "median":
            return np.median(stack, axis=0)
        elif method == "recursive":
            # IIR filter: y[n] = α × x[n] + (1-α) × y[n-1]
            alpha = 2.0 / (len(self.frame_buffer) + 1)
            result = stack[0].copy()
            for frame in stack[1:]:
                result = alpha * frame + (1 - alpha) * result
            return result

        return self.frame_buffer[-1]

    def motion_compensated_filter(
        self,
        threshold: float = 10.0
    ) -> Optional[np.ndarray]:
        """
        Motion-compensated temporal filtering.

        Reduces filtering in areas with motion.

        Args:
            threshold: Motion detection threshold

        Returns:
            Filtered frame
        """
        if len(self.frame_buffer) < 3:
            return self.frame_buffer[-1] if self.frame_buffer else None

        current = self.frame_buffer[-1]
        previous = self.frame_buffer[-2]

        # Detect motion
        diff = np.abs(current - previous)
        motion_mask = diff > threshold

        # Temporal mean of static areas
        stack = np.stack(self.frame_buffer, axis=0)
        temporal_mean = np.mean(stack, axis=0)

        # Blend based on motion
        result = np.where(motion_mask, current, temporal_mean)

        return result

    def clear(self):
        """Clear the frame buffer."""
        self.frame_buffer.clear()


def calculate_snr(
    signal: np.ndarray,
    noise_rms: float
) -> np.ndarray:
    """Calculate signal-to-noise ratio."""
    return signal / (noise_rms + 1e-10)


def calculate_netd_from_noise(
    noise_electrons: float,
    responsivity_e_per_k: float
) -> float:
    """
    Calculate NETD from noise and responsivity.

    Args:
        noise_electrons: RMS noise in electrons
        responsivity_e_per_k: Signal electrons per Kelvin

    Returns:
        NETD in Kelvin
    """
    if responsivity_e_per_k <= 0:
        return float('inf')
    return noise_electrons / responsivity_e_per_k


def estimate_noise_from_image(
    image: np.ndarray,
    method: str = "mad"
) -> float:
    """
    Estimate noise level from a single image.

    Args:
        image: Input image
        method: Estimation method ("mad", "std", "laplacian")

    Returns:
        Estimated noise standard deviation
    """
    if method == "mad":
        # Median Absolute Deviation (robust to outliers)
        # For Gaussian noise, σ ≈ 1.4826 × MAD
        median = np.median(image)
        mad = np.median(np.abs(image - median))
        return float(1.4826 * mad)

    elif method == "std":
        # Simple standard deviation
        return float(np.std(image))

    elif method == "laplacian":
        # Laplacian-based noise estimation
        # Works on high-frequency content
        laplacian = ndimage.laplace(image)
        # For Gaussian noise, σ ≈ √(π/2) × mean(|Laplacian|) / 6
        return float(math.sqrt(math.pi / 2) * np.mean(np.abs(laplacian)) / 6)

    return float(np.std(image))
