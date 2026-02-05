"""
EOSIM Detection and Tracking Module (Stage G).

Provides target detection algorithms, tracking filters, and
performance metrics for thermal/IR imagery analysis.

Example 1: CFAR Detection
    >>> from eosim.detection import CFARDetector, DetectionResult
    >>> detector = CFARDetector(
    ...     guard_cells=2,
    ...     reference_cells=8,
    ...     pfa=1e-6,
    ... )
    >>> result = detector.detect(thermal_image)
    >>> print(f"Found {len(result.detections)} targets")
    >>> for det in result.detections:
    ...     print(f"  Target at ({det.x:.0f}, {det.y:.0f}), SNR={det.snr:.1f}")

Example 2: Multi-target tracking with Kalman filter
    >>> from eosim.detection import KalmanTracker, Track
    >>> tracker = KalmanTracker(
    ...     process_noise=0.1,
    ...     measurement_noise=1.0,
    ... )
    >>> for frame_idx, detections in enumerate(detection_sequence):
    ...     tracks = tracker.update(detections, frame_idx)
    ...     for track in tracks:
    ...         print(f"Track {track.id}: position={track.position}, velocity={track.velocity}")

Example 3: Detection performance evaluation
    >>> from eosim.detection import evaluate_detection, ROCCurve
    >>> # Run detector at multiple thresholds
    >>> results = evaluate_detection(detector, images, ground_truth)
    >>> roc = ROCCurve.from_results(results)
    >>> print(f"AUC: {roc.auc:.3f}")
    >>> print(f"Pd at Pfa=1e-6: {roc.pd_at_pfa(1e-6):.3f}")
"""

from eosim.detection.detectors import (
    ThresholdDetector,
    CFARDetector,
    MatchedFilterDetector,
    Detection,
    DetectionResult,
)
from eosim.detection.tracking import (
    KalmanTracker,
    Track,
    TrackState,
    MultiTargetTracker,
)
from eosim.detection.metrics import (
    DetectionMetrics,
    ROCCurve,
    evaluate_detection,
    compute_pd_pfa,
    confusion_matrix,
)

__all__ = [
    # Detectors
    "ThresholdDetector",
    "CFARDetector",
    "MatchedFilterDetector",
    "Detection",
    "DetectionResult",
    # Tracking
    "KalmanTracker",
    "Track",
    "TrackState",
    "MultiTargetTracker",
    # Metrics
    "DetectionMetrics",
    "ROCCurve",
    "evaluate_detection",
    "compute_pd_pfa",
    "confusion_matrix",
]
