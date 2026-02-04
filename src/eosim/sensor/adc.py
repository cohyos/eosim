"""
Analog-to-Digital Converter (ADC) models.

Handles conversion from analog signal (electrons or voltage) to
digital numbers (DN), including quantization effects.
"""

from dataclasses import dataclass
from typing import Optional, Union
from enum import Enum
import numpy as np
from numpy.typing import NDArray


class ADCType(Enum):
    """Types of ADC architectures."""
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    PIECEWISE_LINEAR = "piecewise_linear"


@dataclass
class ADCParameters:
    """Parameters for ADC conversion.

    Attributes:
        bit_depth: Number of bits (e.g., 12, 14, 16)
        full_well_electrons: Full well capacity in electrons
        gain_electrons_per_dn: Conversion gain (electrons per DN)
        offset_dn: Output offset (black level) in DN
        max_dn: Maximum output DN (2^bit_depth - 1)
        nonlinearity_percent: Integral nonlinearity as percentage
    """
    bit_depth: int = 14
    full_well_electrons: float = 100000.0
    gain_electrons_per_dn: Optional[float] = None
    offset_dn: float = 0.0
    nonlinearity_percent: float = 0.0

    def __post_init__(self) -> None:
        """Validate and compute derived parameters."""
        if self.bit_depth < 1 or self.bit_depth > 32:
            raise ValueError("Bit depth must be between 1 and 32")
        if self.full_well_electrons <= 0:
            raise ValueError("Full well capacity must be positive")
        if self.offset_dn < 0:
            raise ValueError("Offset must be non-negative")

        # Compute gain if not specified
        if self.gain_electrons_per_dn is None:
            # Default: map full well to max DN
            self.gain_electrons_per_dn = self.full_well_electrons / self.max_dn

    @property
    def max_dn(self) -> int:
        """Maximum DN value (2^bit_depth - 1)."""
        return (1 << self.bit_depth) - 1

    @property
    def lsb_electrons(self) -> float:
        """Electrons per least significant bit."""
        return self.gain_electrons_per_dn

    @property
    def dynamic_range_db(self) -> float:
        """Dynamic range in dB."""
        return 20 * np.log10(self.max_dn)

    @property
    def quantization_noise_electrons(self) -> float:
        """RMS quantization noise in electrons.

        Quantization noise variance = LSB² / 12
        """
        return self.lsb_electrons / np.sqrt(12)


@dataclass
class ADCResult:
    """Result of ADC conversion.

    Attributes:
        dn: Digital number output
        saturated_pixels: Count of saturated pixels
        saturation_fraction: Fraction of saturated pixels
        clipped_low: Count of pixels clipped to zero
    """
    dn: NDArray[np.integer]
    saturated_pixels: int = 0
    saturation_fraction: float = 0.0
    clipped_low: int = 0


class ADCModel:
    """Model for analog-to-digital conversion.

    Converts signal in electrons to digital numbers (DN) with:
    - Linear or nonlinear transfer function
    - Quantization
    - Saturation handling
    - Offset (black level)
    """

    def __init__(
        self,
        params: ADCParameters,
        adc_type: ADCType = ADCType.LINEAR,
    ) -> None:
        """Initialize ADC model.

        Args:
            params: ADC parameters
            adc_type: Type of ADC transfer function
        """
        self.params = params
        self.adc_type = adc_type

        # Precompute nonlinearity LUT if needed
        self._nonlinearity_lut: Optional[NDArray] = None
        if params.nonlinearity_percent > 0:
            self._build_nonlinearity_lut()

    def _build_nonlinearity_lut(self) -> None:
        """Build lookup table for integral nonlinearity."""
        # Simple sinusoidal nonlinearity model
        n_points = self.params.max_dn + 1
        x = np.linspace(0, 1, n_points)
        # INL as fraction of full scale
        inl_amplitude = self.params.nonlinearity_percent / 100.0
        # Sinusoidal INL (common in real ADCs)
        inl = inl_amplitude * np.sin(2 * np.pi * x)
        self._nonlinearity_lut = (x + inl) * self.params.max_dn

    def convert(
        self,
        electrons: NDArray[np.floating],
        add_quantization_noise: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> ADCResult:
        """Convert electrons to digital numbers.

        Args:
            electrons: Input signal in electrons
            add_quantization_noise: Add dither for quantization noise
            rng: Random number generator for dither

        Returns:
            ADCResult with DN values and statistics
        """
        if rng is None:
            rng = np.random.default_rng()

        # Convert to DN (floating point first)
        dn_float = electrons / self.params.gain_electrons_per_dn

        # Add offset
        dn_float = dn_float + self.params.offset_dn

        # Add quantization noise (dithering) if requested
        if add_quantization_noise:
            dither = rng.uniform(-0.5, 0.5, dn_float.shape)
            dn_float = dn_float + dither

        # Apply nonlinearity if present
        if self._nonlinearity_lut is not None:
            dn_float = self._apply_nonlinearity(dn_float)

        # Quantize (round to integer)
        dn_int = np.round(dn_float).astype(np.int32)

        # Track saturation before clipping
        saturated = dn_int > self.params.max_dn
        saturated_count = int(np.sum(saturated))
        clipped_low = int(np.sum(dn_int < 0))

        # Clip to valid range
        dn_int = np.clip(dn_int, 0, self.params.max_dn)

        # Convert to appropriate integer type
        if self.params.bit_depth <= 8:
            dn_out = dn_int.astype(np.uint8)
        elif self.params.bit_depth <= 16:
            dn_out = dn_int.astype(np.uint16)
        else:
            dn_out = dn_int.astype(np.uint32)

        return ADCResult(
            dn=dn_out,
            saturated_pixels=saturated_count,
            saturation_fraction=saturated_count / electrons.size,
            clipped_low=clipped_low,
        )

    def _apply_nonlinearity(self, dn_float: NDArray) -> NDArray:
        """Apply integral nonlinearity via LUT."""
        if self._nonlinearity_lut is None:
            return dn_float

        # Normalize to 0-1 range
        dn_norm = np.clip(dn_float / self.params.max_dn, 0, 1)
        # Interpolate from LUT
        indices = dn_norm * (len(self._nonlinearity_lut) - 1)
        return np.interp(indices, np.arange(len(self._nonlinearity_lut)),
                        self._nonlinearity_lut)

    def inverse(self, dn: NDArray[np.integer]) -> NDArray[np.floating]:
        """Convert DN back to electrons (inverse mapping).

        Args:
            dn: Digital number input

        Returns:
            Estimated electrons (without noise recovery)
        """
        dn_float = dn.astype(np.float64)

        # Remove offset
        dn_float = dn_float - self.params.offset_dn

        # Convert to electrons
        electrons = dn_float * self.params.gain_electrons_per_dn

        return electrons

    def electrons_to_dn(self, electrons: float) -> float:
        """Convert single electron value to DN (no quantization).

        Args:
            electrons: Signal in electrons

        Returns:
            DN value (float, not quantized)
        """
        return electrons / self.params.gain_electrons_per_dn + self.params.offset_dn

    def dn_to_electrons(self, dn: float) -> float:
        """Convert single DN value to electrons.

        Args:
            dn: Digital number

        Returns:
            Signal in electrons
        """
        return (dn - self.params.offset_dn) * self.params.gain_electrons_per_dn


class LogarithmicADC(ADCModel):
    """ADC with logarithmic transfer function.

    Used for high dynamic range applications where linear
    quantization would waste bits on bright regions.
    """

    def __init__(
        self,
        params: ADCParameters,
        compression_factor: float = 1.0,
    ) -> None:
        """Initialize logarithmic ADC.

        Args:
            params: ADC parameters
            compression_factor: Logarithmic compression strength
        """
        super().__init__(params, ADCType.LOGARITHMIC)
        self.compression_factor = compression_factor

    def convert(
        self,
        electrons: NDArray[np.floating],
        add_quantization_noise: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> ADCResult:
        """Convert electrons to DN with logarithmic compression.

        Args:
            electrons: Input signal in electrons
            add_quantization_noise: Add dither
            rng: Random number generator

        Returns:
            ADCResult with DN values
        """
        if rng is None:
            rng = np.random.default_rng()

        # Apply logarithmic compression
        # DN = max_dn × log(1 + k×e/FW) / log(1 + k)
        k = self.compression_factor
        fw = self.params.full_well_electrons
        max_dn = self.params.max_dn

        e_normalized = np.maximum(electrons, 0) / fw
        dn_float = max_dn * np.log1p(k * e_normalized) / np.log1p(k)

        # Add offset
        dn_float = dn_float + self.params.offset_dn

        # Add dither if requested
        if add_quantization_noise:
            dither = rng.uniform(-0.5, 0.5, dn_float.shape)
            dn_float = dn_float + dither

        # Quantize
        dn_int = np.round(dn_float).astype(np.int32)

        # Track saturation
        saturated = dn_int > max_dn
        saturated_count = int(np.sum(saturated))
        clipped_low = int(np.sum(dn_int < 0))

        # Clip
        dn_int = np.clip(dn_int, 0, max_dn)

        # Output type
        if self.params.bit_depth <= 8:
            dn_out = dn_int.astype(np.uint8)
        elif self.params.bit_depth <= 16:
            dn_out = dn_int.astype(np.uint16)
        else:
            dn_out = dn_int.astype(np.uint32)

        return ADCResult(
            dn=dn_out,
            saturated_pixels=saturated_count,
            saturation_fraction=saturated_count / electrons.size,
            clipped_low=clipped_low,
        )

    def inverse(self, dn: NDArray[np.integer]) -> NDArray[np.floating]:
        """Convert DN back to electrons (inverse log mapping)."""
        dn_float = dn.astype(np.float64)
        dn_float = dn_float - self.params.offset_dn

        k = self.compression_factor
        fw = self.params.full_well_electrons
        max_dn = self.params.max_dn

        # Inverse: e = FW × (exp(dn/max_dn × log(1+k)) - 1) / k
        dn_normalized = np.clip(dn_float / max_dn, 0, 1)
        electrons = fw * (np.expm1(dn_normalized * np.log1p(k))) / k

        return electrons


def compute_quantization_noise(bit_depth: int, full_well: float) -> float:
    """Compute RMS quantization noise in electrons.

    Args:
        bit_depth: ADC bit depth
        full_well: Full well capacity in electrons

    Returns:
        RMS quantization noise in electrons
    """
    max_dn = (1 << bit_depth) - 1
    lsb_electrons = full_well / max_dn
    return lsb_electrons / np.sqrt(12)


def required_bit_depth(
    full_well: float,
    read_noise: float,
    target_quantization_ratio: float = 0.5,
) -> int:
    """Compute required bit depth to not be quantization limited.

    The rule of thumb is that LSB should be less than read noise
    to avoid quantization-limited performance.

    Args:
        full_well: Full well capacity in electrons
        read_noise: Read noise in electrons
        target_quantization_ratio: Target ratio of quant noise to read noise

    Returns:
        Required bit depth (rounded up)
    """
    # Want: LSB / sqrt(12) < target_ratio × read_noise
    # LSB = FW / (2^n - 1) ≈ FW / 2^n
    # FW / 2^n / sqrt(12) < target_ratio × read_noise
    # 2^n > FW / (sqrt(12) × target_ratio × read_noise)

    required_levels = full_well / (np.sqrt(12) * target_quantization_ratio * read_noise)
    return int(np.ceil(np.log2(required_levels)))


def create_adc(
    bit_depth: int = 14,
    full_well_electrons: float = 100000.0,
    gain: Optional[float] = None,
    offset_dn: float = 0.0,
    adc_type: str = "linear",
) -> ADCModel:
    """Factory function to create an ADC model.

    Args:
        bit_depth: Number of bits
        full_well_electrons: Full well capacity
        gain: Electrons per DN (computed if None)
        offset_dn: Black level offset
        adc_type: "linear" or "logarithmic"

    Returns:
        Configured ADCModel
    """
    params = ADCParameters(
        bit_depth=bit_depth,
        full_well_electrons=full_well_electrons,
        gain_electrons_per_dn=gain,
        offset_dn=offset_dn,
    )

    if adc_type == "logarithmic":
        return LogarithmicADC(params)
    else:
        return ADCModel(params)
