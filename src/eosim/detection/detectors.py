"""
Target Detection Algorithms for EOSIM.

Provides threshold-based, CFAR, and matched filter detection algorithms.

Example 1: Simple threshold detection
    >>> from eosim.detection import ThresholdDetector
    >>> detector = ThresholdDetector(threshold=5.0, min_area=4)
    >>> result = detector.detect(image - background)

Example 2: CFAR detection
    >>> from eosim.detection import CFARDetector
    >>> cfar = CFARDetector(guard_cells=2, reference_cells=8, pfa=1e-6)
    >>> result = cfar.detect(image)

Example 3: Matched filter detection
    >>> from eosim.detection import MatchedFilterDetector
    >>> mf = MatchedFilterDetector(target_template)
    >>> result = mf.detect(image, threshold=4.0)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.special import erfinv


@dataclass
class Detection:
    """Single target detection.

    Attributes:
        x: X coordinate (column)
        y: Y coordinate (row)
        snr: Signal-to-noise ratio
        intensity: Peak intensity
        area: Detection area in pixels
        bbox: Bounding box (y_min, x_min, y_max, x_max)
        confidence: Detection confidence score
    """

    x: float
    y: float
    snr: float = 0.0
    intensity: float = 0.0
    area: int = 1
    bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 1.0

    @property
    def position(self) -> Tuple[float, float]:
        """Get (x, y) position tuple."""
        return (self.x, self.y)

    def distance_to(self, other: "Detection") -> float:
        """Compute Euclidean distance to another detection."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class DetectionResult:
    """Result from detection algorithm.

    Attributes:
        detections: List of Detection objects
        threshold_map: Detection threshold at each pixel
        snr_map: SNR map if computed
        detection_mask: Binary detection mask
    """

    detections: List[Detection]
    threshold_map: Optional[NDArray] = None
    snr_map: Optional[NDArray] = None
    detection_mask: Optional[NDArray] = None

    @property
    def n_detections(self) -> int:
        """Number of detections."""
        return len(self.detections)

    def get_positions(self) -> NDArray:
        """Get detection positions as Nx2 array."""
        if not self.detections:
            return np.array([]).reshape(0, 2)
        return np.array([[d.x, d.y] for d in self.detections])


def _local_background_stats(
    image: NDArray,
    guard_size: int,
    reference_size: int,
) -> Tuple[NDArray, NDArray]:
    """Compute local background mean and standard deviation.

    Uses a ring-shaped reference region around each pixel.

    Args:
        image: Input image
        guard_size: Guard cell radius
        reference_size: Reference cell radius

    Returns:
        Tuple of (mean, std) arrays
    """
    # Create kernels for inner (guard) and outer (guard + reference) regions
    inner_radius = guard_size
    outer_radius = guard_size + reference_size

    y, x = np.ogrid[
        -outer_radius : outer_radius + 1, -outer_radius : outer_radius + 1
    ]
    r = np.sqrt(x**2 + y**2)

    inner_mask = r <= inner_radius
    outer_mask = r <= outer_radius
    ring_mask = outer_mask & ~inner_mask

    ring_kernel = ring_mask.astype(float)
    ring_count = ring_kernel.sum()

    if ring_count == 0:
        ring_count = 1

    ring_kernel /= ring_count

    # Compute local mean
    local_mean = ndimage.convolve(image.astype(np.float64), ring_kernel, mode="reflect")

    # Compute local variance: E[X²] - E[X]²
    local_sq_mean = ndimage.convolve(
        image.astype(np.float64) ** 2, ring_kernel, mode="reflect"
    )
    local_var = local_sq_mean - local_mean**2
    local_var = np.maximum(local_var, 0)  # Numerical stability

    local_std = np.sqrt(local_var)

    return local_mean, local_std


def _find_connected_components(
    mask: NDArray,
    min_area: int = 1,
) -> Tuple[List[Tuple[int, int]], List[int], NDArray]:
    """Find connected components in binary mask.

    Args:
        mask: Binary detection mask
        min_area: Minimum component area

    Returns:
        Tuple of (centroids, areas, labeled_mask)
    """
    labeled, n_features = ndimage.label(mask)

    centroids = []
    areas = []

    for i in range(1, n_features + 1):
        component_mask = labeled == i
        area = component_mask.sum()

        if area >= min_area:
            # Compute centroid
            y_coords, x_coords = np.where(component_mask)
            centroid = (float(x_coords.mean()), float(y_coords.mean()))
            centroids.append(centroid)
            areas.append(area)

    return centroids, areas, labeled


class ThresholdDetector:
    """Simple threshold-based detection.

    Detects targets as connected regions above a threshold.
    """

    def __init__(
        self,
        threshold: float = 3.0,
        min_area: int = 1,
        max_area: int = 10000,
        use_sigma: bool = True,
    ) -> None:
        """Initialize threshold detector.

        Args:
            threshold: Detection threshold (sigma or absolute)
            min_area: Minimum detection area in pixels
            max_area: Maximum detection area in pixels
            use_sigma: If True, threshold is in sigma units
        """
        self._threshold = threshold
        self._min_area = min_area
        self._max_area = max_area
        self._use_sigma = use_sigma

    def detect(
        self,
        image: NDArray,
        background: Optional[NDArray] = None,
    ) -> DetectionResult:
        """Detect targets in image.

        Args:
            image: Input image (or difference image)
            background: Optional background estimate

        Returns:
            DetectionResult with detections
        """
        if background is not None:
            diff = image.astype(np.float64) - background
        else:
            diff = image.astype(np.float64)

        if self._use_sigma:
            mean = np.mean(diff)
            std = np.std(diff)
            threshold = mean + self._threshold * std
        else:
            threshold = self._threshold

        # Create detection mask
        mask = diff > threshold

        # Find connected components
        centroids, areas, labeled = _find_connected_components(mask, self._min_area)

        # Create detections
        detections = []
        for (x, y), area in zip(centroids, areas):
            if area <= self._max_area:
                intensity = float(diff[int(y), int(x)])
                snr = intensity / std if self._use_sigma and std > 0 else intensity

                # Compute bounding box
                component_mask = labeled == (len(detections) + 1)
                y_coords, x_coords = np.where(component_mask)
                bbox = (
                    int(y_coords.min()),
                    int(x_coords.min()),
                    int(y_coords.max()),
                    int(x_coords.max()),
                )

                detections.append(
                    Detection(
                        x=x,
                        y=y,
                        snr=snr,
                        intensity=intensity,
                        area=area,
                        bbox=bbox,
                    )
                )

        return DetectionResult(
            detections=detections,
            detection_mask=mask,
        )


class CFARDetector:
    """Constant False Alarm Rate (CFAR) detector.

    Maintains constant Pfa by adapting threshold to local background.
    """

    def __init__(
        self,
        guard_cells: int = 2,
        reference_cells: int = 8,
        pfa: float = 1e-6,
        min_area: int = 1,
    ) -> None:
        """Initialize CFAR detector.

        Args:
            guard_cells: Guard cell radius
            reference_cells: Reference cell radius
            pfa: Probability of false alarm
            min_area: Minimum detection area
        """
        self._guard = guard_cells
        self._reference = reference_cells
        self._pfa = pfa
        self._min_area = min_area

        # Compute threshold multiplier from Pfa (Gaussian assumption)
        # Pfa = 1 - CDF(T) = 0.5 * erfc(T / sqrt(2))
        # T = sqrt(2) * erfinv(1 - 2*Pfa)
        self._threshold_multiplier = np.sqrt(2) * erfinv(1 - 2 * pfa)

    @property
    def threshold_multiplier(self) -> float:
        """Get CFAR threshold multiplier."""
        return self._threshold_multiplier

    def detect(
        self,
        image: NDArray,
    ) -> DetectionResult:
        """Detect targets using CFAR.

        Args:
            image: Input image

        Returns:
            DetectionResult with detections
        """
        img = image.astype(np.float64)

        # Compute local statistics
        local_mean, local_std = _local_background_stats(
            img, self._guard, self._reference
        )

        # Ensure non-zero std
        local_std = np.maximum(local_std, 1e-10)

        # Compute SNR map
        snr_map = (img - local_mean) / local_std

        # Compute adaptive threshold
        threshold_map = local_mean + self._threshold_multiplier * local_std

        # Create detection mask
        mask = img > threshold_map

        # Find connected components
        centroids, areas, labeled = _find_connected_components(mask, self._min_area)

        # Create detections
        detections = []
        for (x, y), area in zip(centroids, areas):
            ix, iy = int(x), int(y)
            ix = min(max(ix, 0), img.shape[1] - 1)
            iy = min(max(iy, 0), img.shape[0] - 1)

            snr = float(snr_map[iy, ix])
            intensity = float(img[iy, ix])

            detections.append(
                Detection(
                    x=x,
                    y=y,
                    snr=snr,
                    intensity=intensity,
                    area=area,
                    confidence=1 - self._pfa,
                )
            )

        return DetectionResult(
            detections=detections,
            threshold_map=threshold_map,
            snr_map=snr_map,
            detection_mask=mask,
        )


class MatchedFilterDetector:
    """Matched filter (template matching) detector.

    Optimally detects targets with known spatial signature.
    """

    def __init__(
        self,
        template: NDArray,
        normalize: bool = True,
    ) -> None:
        """Initialize matched filter detector.

        Args:
            template: Target template (should be zero-mean if normalize=True)
            normalize: Normalize template
        """
        self._template = template.astype(np.float64)

        if normalize:
            self._template = self._template - self._template.mean()
            norm = np.sqrt(np.sum(self._template**2))
            if norm > 0:
                self._template = self._template / norm

    @property
    def template(self) -> NDArray:
        """Get template."""
        return self._template

    def detect(
        self,
        image: NDArray,
        threshold: float = 4.0,
        min_separation: int = 5,
    ) -> DetectionResult:
        """Detect targets using matched filter.

        Args:
            image: Input image
            threshold: Detection threshold (SNR)
            min_separation: Minimum separation between detections

        Returns:
            DetectionResult with detections
        """
        img = image.astype(np.float64)
        img = img - img.mean()

        # Correlate with template
        correlation = ndimage.correlate(img, self._template, mode="reflect")

        # Estimate local noise
        # Use local std in a window
        window_size = max(self._template.shape)
        kernel = np.ones((window_size, window_size)) / (window_size**2)
        local_sq_mean = ndimage.convolve(img**2, kernel, mode="reflect")
        local_mean = ndimage.convolve(img, kernel, mode="reflect")
        local_var = local_sq_mean - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 1e-10))

        # SNR map
        snr_map = correlation / local_std

        # Find peaks above threshold
        mask = snr_map > threshold

        # Non-maximum suppression
        max_filtered = ndimage.maximum_filter(snr_map, size=min_separation)
        peaks = (snr_map == max_filtered) & mask

        # Extract detections
        y_coords, x_coords = np.where(peaks)
        detections = []

        for x, y in zip(x_coords, y_coords):
            snr = float(snr_map[y, x])
            intensity = float(correlation[y, x])

            detections.append(
                Detection(
                    x=float(x),
                    y=float(y),
                    snr=snr,
                    intensity=intensity,
                    area=1,
                )
            )

        # Sort by SNR
        detections.sort(key=lambda d: d.snr, reverse=True)

        return DetectionResult(
            detections=detections,
            snr_map=snr_map,
            detection_mask=mask,
        )

    @classmethod
    def from_gaussian(
        cls,
        sigma: float = 1.5,
        size: Optional[int] = None,
    ) -> "MatchedFilterDetector":
        """Create matched filter with Gaussian template.

        Args:
            sigma: Gaussian sigma in pixels
            size: Template size (default: 6*sigma)

        Returns:
            MatchedFilterDetector instance
        """
        if size is None:
            size = int(6 * sigma) | 1  # Ensure odd

        y, x = np.ogrid[:size, :size]
        center = size // 2
        template = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma**2))

        return cls(template, normalize=True)
