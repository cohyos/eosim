"""
Pixel Correction Algorithms for EOSIM.

Provides bad pixel detection/correction and flat field correction.

Example 1: Bad pixel detection and correction
    >>> from eosim.calibration import BadPixelCorrector
    >>> corrector = BadPixelCorrector()
    >>> bad_map = corrector.detect_bad_pixels(flat_frame, threshold_sigma=3.0)
    >>> corrected = corrector.correct(raw_image, bad_map)

Example 2: Flat field correction
    >>> from eosim.calibration import FlatFieldCorrector
    >>> ffc = FlatFieldCorrector()
    >>> ffc.calibrate(flat_field_images)
    >>> corrected = ffc.apply(raw_image)

Example 3: Combined correction pipeline
    >>> from eosim.calibration import detect_bad_pixels, correct_bad_pixels, apply_flat_field
    >>> bad_map = detect_bad_pixels(dark_frame, method='threshold')
    >>> corrected = correct_bad_pixels(raw_image, bad_map)
    >>> corrected = apply_flat_field(corrected, flat_gain)
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage


@dataclass
class BadPixelMap:
    """Bad pixel map with metadata.

    Attributes:
        mask: Boolean mask (True = bad pixel)
        dead_pixels: Coordinates of dead (low response) pixels
        hot_pixels: Coordinates of hot (high response) pixels
        noisy_pixels: Coordinates of noisy pixels
    """

    mask: NDArray
    dead_pixels: Optional[NDArray] = None
    hot_pixels: Optional[NDArray] = None
    noisy_pixels: Optional[NDArray] = None

    @property
    def n_bad(self) -> int:
        """Total number of bad pixels."""
        return int(self.mask.sum())

    @property
    def bad_pixel_fraction(self) -> float:
        """Fraction of bad pixels."""
        return self.n_bad / self.mask.size

    def save(self, filepath: str) -> None:
        """Save bad pixel map to file.

        Args:
            filepath: Output file path (.npz)
        """
        data = {"mask": self.mask}
        if self.dead_pixels is not None:
            data["dead_pixels"] = self.dead_pixels
        if self.hot_pixels is not None:
            data["hot_pixels"] = self.hot_pixels
        if self.noisy_pixels is not None:
            data["noisy_pixels"] = self.noisy_pixels

        np.savez(filepath, **data)

    @classmethod
    def load(cls, filepath: str) -> "BadPixelMap":
        """Load bad pixel map from file.

        Args:
            filepath: Input file path (.npz)

        Returns:
            BadPixelMap instance
        """
        data = np.load(filepath)
        return cls(
            mask=data["mask"],
            dead_pixels=data.get("dead_pixels"),
            hot_pixels=data.get("hot_pixels"),
            noisy_pixels=data.get("noisy_pixels"),
        )


def detect_bad_pixels(
    image: NDArray,
    method: str = "sigma",
    threshold_sigma: float = 3.0,
    threshold_absolute: Optional[Tuple[float, float]] = None,
    neighborhood_size: int = 5,
) -> BadPixelMap:
    """Detect bad pixels in image.

    Args:
        image: Input image (flat field or dark frame)
        method: Detection method ('sigma', 'absolute', 'median')
        threshold_sigma: Sigma threshold for 'sigma' method
        threshold_absolute: (min, max) for 'absolute' method
        neighborhood_size: Size for 'median' method

    Returns:
        BadPixelMap with detected bad pixels
    """
    img = image.astype(np.float64)
    mask = np.zeros(img.shape, dtype=bool)

    if method == "sigma":
        # Sigma-based threshold
        mean = np.mean(img)
        std = np.std(img)

        dead = img < mean - threshold_sigma * std
        hot = img > mean + threshold_sigma * std

        mask = dead | hot

        dead_pixels = np.array(np.where(dead)).T
        hot_pixels = np.array(np.where(hot)).T

    elif method == "absolute":
        # Absolute threshold
        if threshold_absolute is None:
            threshold_absolute = (0, np.inf)

        dead = img < threshold_absolute[0]
        hot = img > threshold_absolute[1]

        mask = dead | hot

        dead_pixels = np.array(np.where(dead)).T
        hot_pixels = np.array(np.where(hot)).T

    elif method == "median":
        # Median filter comparison
        median_filtered = ndimage.median_filter(img, size=neighborhood_size)
        diff = np.abs(img - median_filtered)

        threshold = threshold_sigma * np.std(diff)
        mask = diff > threshold

        # Classify based on sign
        dead_pixels = np.array(np.where((mask) & (img < median_filtered))).T
        hot_pixels = np.array(np.where((mask) & (img >= median_filtered))).T

    else:
        raise ValueError(f"Unknown method: {method}")

    return BadPixelMap(
        mask=mask,
        dead_pixels=dead_pixels if len(dead_pixels) > 0 else None,
        hot_pixels=hot_pixels if len(hot_pixels) > 0 else None,
    )


def correct_bad_pixels(
    image: NDArray,
    bad_map: Union[BadPixelMap, NDArray],
    method: str = "median",
    kernel_size: int = 3,
) -> NDArray:
    """Correct bad pixels using interpolation.

    Args:
        image: Input image
        bad_map: Bad pixel map (BadPixelMap or boolean mask)
        method: Correction method ('median', 'mean', 'bilinear')
        kernel_size: Kernel size for median/mean methods

    Returns:
        Corrected image
    """
    if isinstance(bad_map, BadPixelMap):
        mask = bad_map.mask
    else:
        mask = bad_map

    corrected = image.astype(np.float64).copy()

    if not mask.any():
        return corrected

    if method == "median":
        # Replace bad pixels with local median
        filtered = ndimage.median_filter(corrected, size=kernel_size)
        corrected[mask] = filtered[mask]

    elif method == "mean":
        # Replace with local mean (excluding bad pixels)
        kernel = np.ones((kernel_size, kernel_size))
        good_mask = ~mask

        # Compute sum and count of good neighbors
        sum_img = ndimage.convolve(corrected * good_mask, kernel)
        count_img = ndimage.convolve(good_mask.astype(float), kernel)
        count_img = np.maximum(count_img, 1)  # Avoid division by zero

        mean_img = sum_img / count_img
        corrected[mask] = mean_img[mask]

    elif method == "bilinear":
        # Bilinear interpolation from nearest good neighbors
        good_mask = ~mask
        y, x = np.indices(image.shape)

        # Find nearest good pixels in each direction
        for i, j in zip(*np.where(mask)):
            neighbors = []
            weights = []

            # Search in 4 directions
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for dist in range(1, min(image.shape) // 2):
                    ni, nj = i + dy * dist, j + dx * dist
                    if 0 <= ni < image.shape[0] and 0 <= nj < image.shape[1]:
                        if good_mask[ni, nj]:
                            neighbors.append(corrected[ni, nj])
                            weights.append(1.0 / dist)
                            break

            if neighbors:
                weights = np.array(weights)
                corrected[i, j] = np.average(neighbors, weights=weights)

    else:
        raise ValueError(f"Unknown method: {method}")

    return corrected


def apply_flat_field(
    image: NDArray,
    flat_gain: NDArray,
    flat_offset: Optional[NDArray] = None,
) -> NDArray:
    """Apply flat field correction.

    corrected = (image - flat_offset) × flat_gain

    Args:
        image: Input image
        flat_gain: Flat field gain map
        flat_offset: Optional offset map (dark frame)

    Returns:
        Flat-field corrected image
    """
    corrected = image.astype(np.float64)

    if flat_offset is not None:
        corrected = corrected - flat_offset

    corrected = corrected * flat_gain

    return corrected


class BadPixelCorrector:
    """Bad pixel detection and correction.

    Provides methods for detecting bad pixels from calibration frames
    and correcting them in operational images.
    """

    def __init__(
        self,
        method: str = "median",
        kernel_size: int = 3,
    ) -> None:
        """Initialize bad pixel corrector.

        Args:
            method: Correction method ('median', 'mean', 'bilinear')
            kernel_size: Kernel size for correction
        """
        self._method = method
        self._kernel_size = kernel_size
        self._bad_map: Optional[BadPixelMap] = None

    @property
    def bad_pixel_map(self) -> Optional[BadPixelMap]:
        """Get current bad pixel map."""
        return self._bad_map

    def detect_bad_pixels(
        self,
        image: NDArray,
        threshold_sigma: float = 3.0,
        method: str = "sigma",
    ) -> BadPixelMap:
        """Detect bad pixels from calibration image.

        Args:
            image: Flat field or dark frame
            threshold_sigma: Detection threshold
            method: Detection method

        Returns:
            BadPixelMap
        """
        self._bad_map = detect_bad_pixels(
            image, method=method, threshold_sigma=threshold_sigma
        )
        return self._bad_map

    def detect_from_noise(
        self,
        frames: List[NDArray],
        threshold_sigma: float = 3.0,
    ) -> BadPixelMap:
        """Detect noisy pixels from temporal variation.

        Args:
            frames: List of frames (same scene, different times)
            threshold_sigma: Threshold for noisy pixel detection

        Returns:
            BadPixelMap with noisy pixels
        """
        stack = np.array([f.astype(np.float64) for f in frames])
        temporal_std = np.std(stack, axis=0)

        mean_std = np.mean(temporal_std)
        std_std = np.std(temporal_std)

        threshold = mean_std + threshold_sigma * std_std
        noisy_mask = temporal_std > threshold

        self._bad_map = BadPixelMap(
            mask=noisy_mask,
            noisy_pixels=np.array(np.where(noisy_mask)).T,
        )

        return self._bad_map

    def detect_dead_pixels(
        self,
        dark_frame: NDArray,
        flat_frame: NDArray,
        dark_threshold: float = 0.1,
        response_threshold: float = 0.5,
    ) -> BadPixelMap:
        """Detect dead pixels using dark and flat frames.

        Dead pixels have abnormal dark signal or low flat response.

        Args:
            dark_frame: Dark frame
            flat_frame: Flat field frame
            dark_threshold: Dark signal threshold (fraction of mean)
            response_threshold: Response threshold (fraction of mean)

        Returns:
            BadPixelMap
        """
        dark = dark_frame.astype(np.float64)
        flat = flat_frame.astype(np.float64)

        # Dead pixels: very high dark current
        dark_mean = np.mean(dark)
        dark_mask = dark > dark_mean * (1 + dark_threshold * 10)

        # Dead pixels: very low flat response
        flat_mean = np.mean(flat)
        flat_mask = flat < flat_mean * response_threshold

        combined_mask = dark_mask | flat_mask

        self._bad_map = BadPixelMap(
            mask=combined_mask,
            dead_pixels=np.array(np.where(combined_mask)).T,
        )

        return self._bad_map

    def set_bad_pixel_map(self, bad_map: Union[BadPixelMap, NDArray]) -> None:
        """Set bad pixel map directly.

        Args:
            bad_map: BadPixelMap or boolean mask
        """
        if isinstance(bad_map, np.ndarray):
            self._bad_map = BadPixelMap(mask=bad_map)
        else:
            self._bad_map = bad_map

    def correct(
        self,
        image: NDArray,
        bad_map: Optional[Union[BadPixelMap, NDArray]] = None,
    ) -> NDArray:
        """Correct bad pixels in image.

        Args:
            image: Input image
            bad_map: Override bad pixel map

        Returns:
            Corrected image
        """
        if bad_map is None:
            bad_map = self._bad_map

        if bad_map is None:
            raise ValueError("No bad pixel map. Call detect_bad_pixels() first.")

        return correct_bad_pixels(image, bad_map, self._method, self._kernel_size)


class FlatFieldCorrector:
    """Flat field correction.

    Computes and applies flat field gain correction to normalize
    pixel response.
    """

    def __init__(self) -> None:
        """Initialize flat field corrector."""
        self._gain_map: Optional[NDArray] = None
        self._offset_map: Optional[NDArray] = None
        self._reference_value: float = 1.0

    @property
    def is_calibrated(self) -> bool:
        """Check if corrector has been calibrated."""
        return self._gain_map is not None

    @property
    def gain_map(self) -> Optional[NDArray]:
        """Get flat field gain map."""
        return self._gain_map

    @property
    def offset_map(self) -> Optional[NDArray]:
        """Get offset map."""
        return self._offset_map

    def calibrate(
        self,
        flat_frames: Union[NDArray, List[NDArray]],
        dark_frames: Optional[Union[NDArray, List[NDArray]]] = None,
        normalize: bool = True,
    ) -> NDArray:
        """Calibrate flat field from uniform illumination images.

        Args:
            flat_frames: Flat field image(s). If list/3D, averaged.
            dark_frames: Optional dark frame(s) for offset correction.
            normalize: Normalize gain to mean of 1

        Returns:
            Flat field gain map
        """
        # Average multiple frames
        if isinstance(flat_frames, list):
            flat = np.mean([f.astype(np.float64) for f in flat_frames], axis=0)
        elif flat_frames.ndim == 3:
            flat = np.mean(flat_frames.astype(np.float64), axis=0)
        else:
            flat = flat_frames.astype(np.float64)

        # Handle dark frames
        if dark_frames is not None:
            if isinstance(dark_frames, list):
                dark = np.mean([d.astype(np.float64) for d in dark_frames], axis=0)
            elif dark_frames.ndim == 3:
                dark = np.mean(dark_frames.astype(np.float64), axis=0)
            else:
                dark = dark_frames.astype(np.float64)

            self._offset_map = dark
            flat = flat - dark

        # Compute gain map
        # gain = reference / flat_value
        # This normalizes all pixels to have the same response
        mean_flat = np.mean(flat)
        self._reference_value = mean_flat

        # Avoid division by zero
        flat = np.maximum(flat, mean_flat * 0.01)

        self._gain_map = mean_flat / flat

        if normalize:
            # Normalize so mean gain is 1
            self._gain_map = self._gain_map / np.mean(self._gain_map)

        return self._gain_map

    def apply(
        self,
        image: NDArray,
        gain_map: Optional[NDArray] = None,
        offset_map: Optional[NDArray] = None,
    ) -> NDArray:
        """Apply flat field correction.

        Args:
            image: Input image
            gain_map: Override gain map
            offset_map: Override offset map

        Returns:
            Corrected image
        """
        gain = gain_map if gain_map is not None else self._gain_map
        offset = offset_map if offset_map is not None else self._offset_map

        if gain is None:
            raise ValueError("Not calibrated. Call calibrate() first.")

        return apply_flat_field(image, gain, offset)

    def compute_residual_nonuniformity(
        self,
        corrected_flat: NDArray,
    ) -> float:
        """Compute residual non-uniformity after correction.

        Args:
            corrected_flat: Corrected flat field image

        Returns:
            Residual non-uniformity [%]
        """
        mean_val = np.mean(corrected_flat)
        if mean_val == 0:
            return 0.0
        return float(np.std(corrected_flat) / mean_val * 100)

    def save(self, filepath: str) -> None:
        """Save calibration to file.

        Args:
            filepath: Output file path (.npz)
        """
        data = {
            "gain_map": self._gain_map,
            "reference_value": self._reference_value,
        }
        if self._offset_map is not None:
            data["offset_map"] = self._offset_map

        np.savez(filepath, **data)

    def load(self, filepath: str) -> None:
        """Load calibration from file.

        Args:
            filepath: Input file path (.npz)
        """
        data = np.load(filepath)
        self._gain_map = data["gain_map"]
        self._reference_value = float(data["reference_value"])
        self._offset_map = data.get("offset_map")
