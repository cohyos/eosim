"""
Non-Uniformity Correction (NUC) for EOSIM.

Provides two-point and multi-point NUC algorithms for correcting
fixed pattern noise in infrared detectors.

Example 1: Two-point NUC
    >>> from eosim.calibration import TwoPointNUC
    >>> nuc = TwoPointNUC()
    >>> # Calibrate from blackbody images
    >>> gain, offset = nuc.calibrate(cold_frame, hot_frame, T_cold=293, T_hot=323)
    >>> corrected = nuc.apply(raw_frame, gain, offset)

Example 2: Multi-point NUC for better linearity
    >>> from eosim.calibration import MultiPointNUC
    >>> nuc = MultiPointNUC()
    >>> frames = [frame1, frame2, frame3, frame4]  # At different temps
    >>> temps = [290, 300, 310, 320]
    >>> nuc.calibrate(frames, temps)
    >>> corrected = nuc.apply(raw_frame)

Example 3: Apply pre-computed NUC coefficients
    >>> from eosim.calibration import apply_nuc, NUCCoefficients
    >>> coeffs = NUCCoefficients(gain=gain_map, offset=offset_map)
    >>> corrected = apply_nuc(raw_frame, coeffs)
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class NUCCoefficients:
    """Non-uniformity correction coefficients.

    Attributes:
        gain: Per-pixel gain correction (multiply)
        offset: Per-pixel offset correction (subtract before gain)
        gain2: Second-order gain for polynomial NUC (optional)
        reference_temperature: Reference temperature for calibration [K]
    """

    gain: NDArray
    offset: NDArray
    gain2: Optional[NDArray] = None
    reference_temperature: float = 300.0

    @property
    def shape(self) -> Tuple[int, int]:
        """Get detector shape."""
        return self.gain.shape

    def save(self, filepath: str) -> None:
        """Save coefficients to file.

        Args:
            filepath: Output file path (.npz)
        """
        data = {
            "gain": self.gain,
            "offset": self.offset,
            "reference_temperature": self.reference_temperature,
        }
        if self.gain2 is not None:
            data["gain2"] = self.gain2

        np.savez(filepath, **data)

    @classmethod
    def load(cls, filepath: str) -> "NUCCoefficients":
        """Load coefficients from file.

        Args:
            filepath: Input file path (.npz)

        Returns:
            NUCCoefficients instance
        """
        data = np.load(filepath)
        return cls(
            gain=data["gain"],
            offset=data["offset"],
            gain2=data.get("gain2"),
            reference_temperature=float(data.get("reference_temperature", 300.0)),
        )


def apply_nuc(
    raw_image: NDArray,
    coefficients: NUCCoefficients,
) -> NDArray:
    """Apply NUC correction to raw image.

    corrected = gain × (raw - offset)

    For polynomial NUC:
    corrected = gain × (raw - offset) + gain2 × (raw - offset)²

    Args:
        raw_image: Raw detector image
        coefficients: NUC coefficients

    Returns:
        Corrected image
    """
    diff = raw_image.astype(np.float64) - coefficients.offset
    corrected = coefficients.gain * diff

    if coefficients.gain2 is not None:
        corrected += coefficients.gain2 * diff**2

    return corrected


def compute_nuc_coefficients(
    cold_frame: NDArray,
    hot_frame: NDArray,
    expected_cold: float,
    expected_hot: float,
) -> NUCCoefficients:
    """Compute two-point NUC coefficients.

    Args:
        cold_frame: Image at cold reference
        hot_frame: Image at hot reference
        expected_cold: Expected output at cold reference
        expected_hot: Expected output at hot reference

    Returns:
        NUCCoefficients instance
    """
    cold = cold_frame.astype(np.float64)
    hot = hot_frame.astype(np.float64)

    # Compute per-pixel gain and offset
    # corrected = gain × (raw - offset)
    # At cold: expected_cold = gain × (cold - offset)
    # At hot: expected_hot = gain × (hot - offset)

    # Solve: gain = (expected_hot - expected_cold) / (hot - cold)
    delta_raw = hot - cold
    delta_expected = expected_hot - expected_cold

    # Avoid division by zero
    delta_raw = np.where(np.abs(delta_raw) < 1e-10, 1e-10, delta_raw)

    gain = delta_expected / delta_raw

    # offset = cold - expected_cold / gain
    offset = cold - expected_cold / np.where(np.abs(gain) < 1e-10, 1e-10, gain)

    return NUCCoefficients(
        gain=gain,
        offset=offset,
        reference_temperature=(expected_cold + expected_hot) / 2,
    )


class TwoPointNUC:
    """Two-point non-uniformity correction.

    Uses measurements at two reference temperatures to compute
    per-pixel gain and offset corrections.
    """

    def __init__(self) -> None:
        """Initialize two-point NUC."""
        self._coefficients: Optional[NUCCoefficients] = None
        self._cold_mean: float = 0.0
        self._hot_mean: float = 0.0

    @property
    def is_calibrated(self) -> bool:
        """Check if NUC has been calibrated."""
        return self._coefficients is not None

    @property
    def coefficients(self) -> Optional[NUCCoefficients]:
        """Get NUC coefficients."""
        return self._coefficients

    def calibrate(
        self,
        cold_frame: NDArray,
        hot_frame: NDArray,
        T_cold: float = 293.0,
        T_hot: float = 323.0,
        average_frames: bool = False,
    ) -> Tuple[NDArray, NDArray]:
        """Calibrate NUC from blackbody measurements.

        Args:
            cold_frame: Image(s) at cold reference. If 3D, frames are averaged.
            hot_frame: Image(s) at hot reference. If 3D, frames are averaged.
            T_cold: Cold reference temperature [K]
            T_hot: Hot reference temperature [K]
            average_frames: If True, average multiple frames

        Returns:
            Tuple of (gain, offset) arrays
        """
        # Handle multiple frames
        if cold_frame.ndim == 3:
            cold = np.mean(cold_frame, axis=0)
        else:
            cold = cold_frame.astype(np.float64)

        if hot_frame.ndim == 3:
            hot = np.mean(hot_frame, axis=0)
        else:
            hot = hot_frame.astype(np.float64)

        # Compute expected values (use array means as reference)
        self._cold_mean = float(np.mean(cold))
        self._hot_mean = float(np.mean(hot))

        # Use mean values as expected outputs
        expected_cold = self._cold_mean
        expected_hot = self._hot_mean

        self._coefficients = compute_nuc_coefficients(
            cold, hot, expected_cold, expected_hot
        )
        self._coefficients.reference_temperature = (T_cold + T_hot) / 2

        return self._coefficients.gain, self._coefficients.offset

    def apply(
        self,
        raw_image: NDArray,
        gain: Optional[NDArray] = None,
        offset: Optional[NDArray] = None,
    ) -> NDArray:
        """Apply NUC correction.

        Args:
            raw_image: Raw detector image
            gain: Override gain (uses calibrated if None)
            offset: Override offset (uses calibrated if None)

        Returns:
            Corrected image
        """
        if gain is not None and offset is not None:
            coeffs = NUCCoefficients(gain=gain, offset=offset)
        elif self._coefficients is not None:
            coeffs = self._coefficients
        else:
            raise ValueError("NUC not calibrated. Call calibrate() first.")

        return apply_nuc(raw_image, coeffs)

    def compute_residual_nonuniformity(
        self,
        corrected_frame: NDArray,
    ) -> float:
        """Compute residual non-uniformity after correction.

        RNU = std(corrected) / mean(corrected) × 100%

        Args:
            corrected_frame: Corrected image of uniform source

        Returns:
            Residual non-uniformity [%]
        """
        mean_val = np.mean(corrected_frame)
        if mean_val == 0:
            return 0.0
        return float(np.std(corrected_frame) / mean_val * 100)


class MultiPointNUC:
    """Multi-point non-uniformity correction.

    Uses measurements at multiple reference temperatures for
    improved linearity correction.
    """

    def __init__(self, polynomial_order: int = 2) -> None:
        """Initialize multi-point NUC.

        Args:
            polynomial_order: Order of polynomial fit (1=linear, 2=quadratic)
        """
        self._order = polynomial_order
        self._poly_coeffs: Optional[NDArray] = None
        self._reference_values: Optional[NDArray] = None
        self._shape: Optional[Tuple[int, int]] = None

    @property
    def is_calibrated(self) -> bool:
        """Check if NUC has been calibrated."""
        return self._poly_coeffs is not None

    def calibrate(
        self,
        frames: List[NDArray],
        temperatures: List[float],
    ) -> None:
        """Calibrate multi-point NUC.

        Args:
            frames: List of images at each temperature
            temperatures: List of reference temperatures [K]
        """
        if len(frames) < self._order + 1:
            raise ValueError(
                f"Need at least {self._order + 1} frames for order-{self._order} fit"
            )

        n_frames = len(frames)
        self._shape = frames[0].shape
        h, w = self._shape

        # Stack frames and compute polynomial fit per pixel
        # We fit: expected = f(raw) where expected is the mean response
        raw_stack = np.array([f.astype(np.float64) for f in frames])  # (n, h, w)

        # Compute expected outputs (use global mean at each temp)
        expected = np.array([np.mean(f) for f in frames])

        # Fit polynomial for each pixel
        # poly_coeffs shape: (order+1, h, w)
        self._poly_coeffs = np.zeros((self._order + 1, h, w))
        self._reference_values = expected

        for i in range(h):
            for j in range(w):
                pixel_values = raw_stack[:, i, j]
                # Fit polynomial: expected = c0 + c1*raw + c2*raw² + ...
                coeffs = np.polyfit(pixel_values, expected, self._order)
                self._poly_coeffs[:, i, j] = coeffs

    def apply(
        self,
        raw_image: NDArray,
    ) -> NDArray:
        """Apply multi-point NUC correction.

        Args:
            raw_image: Raw detector image

        Returns:
            Corrected image
        """
        if not self.is_calibrated:
            raise ValueError("NUC not calibrated. Call calibrate() first.")

        raw = raw_image.astype(np.float64)
        corrected = np.zeros_like(raw)

        # Apply polynomial correction
        for i in range(self._order + 1):
            corrected += self._poly_coeffs[i] * raw ** (self._order - i)

        return corrected

    def to_two_point(
        self,
        cold_value: float,
        hot_value: float,
    ) -> NUCCoefficients:
        """Convert to equivalent two-point NUC at specified values.

        Args:
            cold_value: Cold reference DN value
            hot_value: Hot reference DN value

        Returns:
            NUCCoefficients for two-point correction
        """
        if not self.is_calibrated:
            raise ValueError("NUC not calibrated. Call calibrate() first.")

        h, w = self._shape

        # Evaluate polynomial at cold and hot values
        expected_cold = np.zeros((h, w))
        expected_hot = np.zeros((h, w))

        for i in range(self._order + 1):
            expected_cold += self._poly_coeffs[i] * cold_value ** (self._order - i)
            expected_hot += self._poly_coeffs[i] * hot_value ** (self._order - i)

        # Compute equivalent gain and offset
        delta = hot_value - cold_value
        delta = max(delta, 1e-10)

        gain = (expected_hot - expected_cold) / delta
        offset_raw = cold_value - expected_cold / np.where(np.abs(gain) < 1e-10, 1e-10, gain)

        return NUCCoefficients(
            gain=gain,
            offset=offset_raw,
        )

    def compute_linearity_error(
        self,
        frames: List[NDArray],
        temperatures: List[float],
    ) -> NDArray:
        """Compute per-pixel linearity error.

        Args:
            frames: Test frames at known temperatures
            temperatures: True temperatures

        Returns:
            Per-pixel RMS error
        """
        if not self.is_calibrated:
            raise ValueError("NUC not calibrated. Call calibrate() first.")

        errors = []
        for frame, temp in zip(frames, temperatures):
            corrected = self.apply(frame)
            # Expected is the mean at this temperature
            expected = np.mean(frame)
            error = (corrected - expected) ** 2
            errors.append(error)

        return np.sqrt(np.mean(errors, axis=0))
