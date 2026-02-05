"""
Detection Performance Metrics for EOSIM.

Provides tools for evaluating detection and tracking performance
including ROC curves, Pd/Pfa, and confusion matrices.

Example 1: Compute Pd and Pfa
    >>> from eosim.detection import compute_pd_pfa
    >>> pd, pfa = compute_pd_pfa(detections, ground_truth, threshold=0.5)
    >>> print(f"Pd={pd:.3f}, Pfa={pfa:.3f}")

Example 2: Generate ROC curve
    >>> from eosim.detection import ROCCurve
    >>> roc = ROCCurve()
    >>> for threshold in thresholds:
    ...     pd, pfa = compute_pd_pfa(detector(image, threshold), truth)
    ...     roc.add_point(pfa, pd, threshold)
    >>> print(f"AUC: {roc.auc:.3f}")

Example 3: Comprehensive detection evaluation
    >>> from eosim.detection import evaluate_detection
    >>> metrics = evaluate_detection(
    ...     detections=detections,
    ...     ground_truth=ground_truth,
    ...     association_radius=5.0,
    ... )
    >>> print(f"Precision: {metrics.precision:.3f}, Recall: {metrics.recall:.3f}")
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Union
import numpy as np
from numpy.typing import NDArray

from .detectors import Detection, DetectionResult


@dataclass
class ConfusionMatrix:
    """Confusion matrix for binary detection.

    Attributes:
        true_positives: Number of true positives
        false_positives: Number of false positives
        false_negatives: Number of false negatives
        true_negatives: Number of true negatives
    """

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    @property
    def total_positives(self) -> int:
        """Total ground truth positives."""
        return self.true_positives + self.false_negatives

    @property
    def total_negatives(self) -> int:
        """Total ground truth negatives."""
        return self.false_positives + self.true_negatives

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)."""
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN) = Pd."""
        return self.true_positives / self.total_positives if self.total_positives > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 score = 2 × precision × recall / (precision + recall)."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Convert to dictionary."""
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
        }


@dataclass
class DetectionMetrics:
    """Comprehensive detection performance metrics.

    Attributes:
        pd: Probability of detection
        pfa: Probability of false alarm
        precision: Detection precision
        recall: Detection recall (same as Pd)
        f1_score: F1 score
        n_targets: Number of ground truth targets
        n_detections: Number of detections
        n_true_positives: Number of true positives
        n_false_positives: Number of false positives
        n_false_negatives: Number of false negatives
        mean_position_error: Mean localization error for true positives
    """

    pd: float = 0.0
    pfa: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    n_targets: int = 0
    n_detections: int = 0
    n_true_positives: int = 0
    n_false_positives: int = 0
    n_false_negatives: int = 0
    mean_position_error: float = 0.0

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Convert to dictionary."""
        return {
            "pd": self.pd,
            "pfa": self.pfa,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "n_targets": self.n_targets,
            "n_detections": self.n_detections,
            "n_true_positives": self.n_true_positives,
            "n_false_positives": self.n_false_positives,
            "n_false_negatives": self.n_false_negatives,
            "mean_position_error": self.mean_position_error,
        }


@dataclass
class ROCCurve:
    """Receiver Operating Characteristic curve.

    Stores (Pfa, Pd) points for different thresholds.
    """

    pfa_values: List[float] = field(default_factory=list)
    pd_values: List[float] = field(default_factory=list)
    thresholds: List[float] = field(default_factory=list)

    def add_point(
        self,
        pfa: float,
        pd: float,
        threshold: Optional[float] = None,
    ) -> None:
        """Add point to ROC curve.

        Args:
            pfa: Probability of false alarm
            pd: Probability of detection
            threshold: Detection threshold (optional)
        """
        self.pfa_values.append(pfa)
        self.pd_values.append(pd)
        if threshold is not None:
            self.thresholds.append(threshold)

    @property
    def auc(self) -> float:
        """Compute Area Under Curve using trapezoidal rule."""
        if len(self.pfa_values) < 2:
            return 0.0

        # Sort by Pfa
        indices = np.argsort(self.pfa_values)
        pfa = np.array(self.pfa_values)[indices]
        pd = np.array(self.pd_values)[indices]

        # Trapezoidal integration
        auc = 0.0
        for i in range(len(pfa) - 1):
            auc += (pfa[i + 1] - pfa[i]) * (pd[i] + pd[i + 1]) / 2

        return float(auc)

    def pd_at_pfa(
        self,
        target_pfa: float,
    ) -> float:
        """Interpolate Pd at target Pfa.

        Args:
            target_pfa: Target Pfa value

        Returns:
            Interpolated Pd
        """
        if len(self.pfa_values) == 0:
            return 0.0

        return float(np.interp(target_pfa, self.pfa_values, self.pd_values))

    def pfa_at_pd(
        self,
        target_pd: float,
    ) -> float:
        """Interpolate Pfa at target Pd.

        Args:
            target_pd: Target Pd value

        Returns:
            Interpolated Pfa
        """
        if len(self.pd_values) == 0:
            return 1.0

        # Sort by Pd for interpolation
        indices = np.argsort(self.pd_values)
        pd = np.array(self.pd_values)[indices]
        pfa = np.array(self.pfa_values)[indices]

        return float(np.interp(target_pd, pd, pfa))

    def threshold_at_pfa(
        self,
        target_pfa: float,
    ) -> Optional[float]:
        """Get threshold for target Pfa.

        Args:
            target_pfa: Target Pfa value

        Returns:
            Threshold or None if not available
        """
        if len(self.thresholds) != len(self.pfa_values):
            return None

        return float(np.interp(target_pfa, self.pfa_values, self.thresholds))

    @classmethod
    def from_results(
        cls,
        results: List[Tuple[float, float, float]],
    ) -> "ROCCurve":
        """Create ROC curve from list of (pfa, pd, threshold) tuples.

        Args:
            results: List of (pfa, pd, threshold) tuples

        Returns:
            ROCCurve instance
        """
        roc = cls()
        for pfa, pd, threshold in results:
            roc.add_point(pfa, pd, threshold)
        return roc

    def to_arrays(self) -> Tuple[NDArray, NDArray]:
        """Convert to numpy arrays.

        Returns:
            Tuple of (pfa_array, pd_array)
        """
        return np.array(self.pfa_values), np.array(self.pd_values)


def confusion_matrix(
    detections: List[Detection],
    ground_truth: List[Tuple[float, float]],
    association_radius: float = 5.0,
    image_size: Optional[Tuple[int, int]] = None,
) -> ConfusionMatrix:
    """Compute confusion matrix for detection results.

    Args:
        detections: List of Detection objects
        ground_truth: List of (x, y) target positions
        association_radius: Maximum distance for true positive
        image_size: Image size for TN computation (height, width)

    Returns:
        ConfusionMatrix
    """
    cm = ConfusionMatrix()
    gt = list(ground_truth)

    # Match detections to ground truth
    matched_gt = set()
    matched_det = set()

    for i, det in enumerate(detections):
        best_dist = float("inf")
        best_j = -1

        for j, (gx, gy) in enumerate(gt):
            if j in matched_gt:
                continue

            dist = np.sqrt((det.x - gx) ** 2 + (det.y - gy) ** 2)
            if dist < best_dist and dist <= association_radius:
                best_dist = dist
                best_j = j

        if best_j >= 0:
            matched_gt.add(best_j)
            matched_det.add(i)
            cm.true_positives += 1

    cm.false_positives = len(detections) - cm.true_positives
    cm.false_negatives = len(gt) - cm.true_positives

    # True negatives (if image size provided)
    if image_size is not None:
        total_pixels = image_size[0] * image_size[1]
        cm.true_negatives = total_pixels - len(gt) - cm.false_positives

    return cm


def compute_pd_pfa(
    detections: List[Detection],
    ground_truth: List[Tuple[float, float]],
    association_radius: float = 5.0,
    n_pixels: Optional[int] = None,
) -> Tuple[float, float]:
    """Compute Pd and Pfa from detection results.

    Args:
        detections: List of Detection objects
        ground_truth: List of (x, y) target positions
        association_radius: Maximum distance for true positive
        n_pixels: Total number of pixels for Pfa calculation

    Returns:
        Tuple of (Pd, Pfa)
    """
    cm = confusion_matrix(detections, ground_truth, association_radius)

    # Pd = TP / (TP + FN)
    pd = cm.recall

    # Pfa = FP / (FP + TN) ≈ FP / n_pixels for low Pfa
    if n_pixels is not None:
        pfa = cm.false_positives / n_pixels
    else:
        # Cannot compute Pfa without knowing total negatives
        pfa = 0.0

    return pd, pfa


def evaluate_detection(
    detections: Union[List[Detection], DetectionResult],
    ground_truth: List[Tuple[float, float]],
    association_radius: float = 5.0,
    image_size: Optional[Tuple[int, int]] = None,
) -> DetectionMetrics:
    """Comprehensive detection evaluation.

    Args:
        detections: Detection list or DetectionResult
        ground_truth: List of (x, y) target positions
        association_radius: Maximum distance for true positive
        image_size: Image size for Pfa computation

    Returns:
        DetectionMetrics
    """
    if isinstance(detections, DetectionResult):
        det_list = detections.detections
    else:
        det_list = detections

    gt = list(ground_truth)
    n_pixels = image_size[0] * image_size[1] if image_size else None

    # Compute confusion matrix
    cm = confusion_matrix(det_list, gt, association_radius, image_size)

    # Compute Pd and Pfa
    pd = cm.recall
    pfa = cm.false_positives / n_pixels if n_pixels else 0.0

    # Compute position errors for true positives
    position_errors = []
    matched_gt = set()

    for det in det_list:
        for j, (gx, gy) in enumerate(gt):
            if j in matched_gt:
                continue

            dist = np.sqrt((det.x - gx) ** 2 + (det.y - gy) ** 2)
            if dist <= association_radius:
                position_errors.append(dist)
                matched_gt.add(j)
                break

    mean_error = np.mean(position_errors) if position_errors else 0.0

    return DetectionMetrics(
        pd=pd,
        pfa=pfa,
        precision=cm.precision,
        recall=cm.recall,
        f1_score=cm.f1_score,
        n_targets=len(gt),
        n_detections=len(det_list),
        n_true_positives=cm.true_positives,
        n_false_positives=cm.false_positives,
        n_false_negatives=cm.false_negatives,
        mean_position_error=float(mean_error),
    )


def generate_roc_curve(
    detector,
    image: NDArray,
    ground_truth: List[Tuple[float, float]],
    thresholds: NDArray,
    association_radius: float = 5.0,
) -> ROCCurve:
    """Generate ROC curve by varying detection threshold.

    Args:
        detector: Detector with detect(image, threshold) method
        image: Input image
        ground_truth: List of (x, y) target positions
        thresholds: Array of thresholds to test
        association_radius: Association radius for scoring

    Returns:
        ROCCurve
    """
    roc = ROCCurve()
    n_pixels = image.shape[0] * image.shape[1]

    for threshold in thresholds:
        # Run detection
        if hasattr(detector, "_threshold"):
            detector._threshold = threshold
            result = detector.detect(image)
        else:
            result = detector.detect(image, threshold=threshold)

        detections = result.detections if hasattr(result, "detections") else result

        # Compute metrics
        pd, pfa = compute_pd_pfa(detections, ground_truth, association_radius, n_pixels)
        roc.add_point(pfa, pd, threshold)

    return roc
