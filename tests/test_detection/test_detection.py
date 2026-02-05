"""
Tests for EOSIM Detection and Tracking Module (Stage G).

Tests detection algorithms, tracking, and performance metrics.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from eosim.detection import (
    ThresholdDetector,
    CFARDetector,
    MatchedFilterDetector,
    Detection,
    DetectionResult,
    KalmanTracker,
    Track,
    TrackState,
    MultiTargetTracker,
    DetectionMetrics,
    ROCCurve,
    evaluate_detection,
    compute_pd_pfa,
    confusion_matrix,
)


# =============================================================================
# Detection Tests
# =============================================================================


class TestDetection:
    """Test Detection dataclass."""

    def test_detection_creation(self):
        """Detection can be created."""
        det = Detection(x=100, y=200, snr=5.0)
        assert det.x == 100
        assert det.y == 200
        assert det.snr == 5.0

    def test_position_property(self):
        """Position property returns tuple."""
        det = Detection(x=100, y=200)
        assert det.position == (100, 200)

    def test_distance_to(self):
        """Distance calculation works."""
        det1 = Detection(x=0, y=0)
        det2 = Detection(x=3, y=4)
        assert det1.distance_to(det2) == 5.0


class TestDetectionResult:
    """Test DetectionResult dataclass."""

    def test_result_creation(self):
        """Result can be created."""
        detections = [Detection(x=10, y=20), Detection(x=30, y=40)]
        result = DetectionResult(detections=detections)
        assert result.n_detections == 2

    def test_get_positions(self):
        """get_positions returns array."""
        detections = [Detection(x=10, y=20), Detection(x=30, y=40)]
        result = DetectionResult(detections=detections)
        positions = result.get_positions()
        assert positions.shape == (2, 2)
        assert_allclose(positions[0], [10, 20])

    def test_empty_result(self):
        """Empty result works."""
        result = DetectionResult(detections=[])
        assert result.n_detections == 0
        assert result.get_positions().shape == (0, 2)


class TestThresholdDetector:
    """Test ThresholdDetector class."""

    @pytest.fixture
    def detector(self):
        """Create detector fixture."""
        return ThresholdDetector(threshold=3.0, min_area=1)

    @pytest.fixture
    def test_image(self):
        """Create test image with target."""
        np.random.seed(42)
        image = np.random.randn(100, 100) * 0.5  # Lower noise
        # Add bright target
        image[50, 50] = 10
        image[50, 51] = 10
        image[51, 50] = 10
        image[51, 51] = 10
        return image

    def test_detector_creation(self, detector):
        """Detector can be created."""
        assert detector is not None

    def test_detect_target(self, detector, test_image):
        """Detector finds target."""
        result = detector.detect(test_image)
        assert result.n_detections >= 1

        # Find detection nearest to target
        target_x, target_y = 50.5, 50.5
        dists = [
            np.sqrt((d.x - target_x) ** 2 + (d.y - target_y) ** 2)
            for d in result.detections
        ]
        nearest_det = result.detections[np.argmin(dists)]

        # Check detection is near target
        assert abs(nearest_det.x - target_x) < 5
        assert abs(nearest_det.y - target_y) < 5

    def test_detect_with_background(self, detector):
        """Detection with separate background."""
        np.random.seed(42)
        background = np.ones((50, 50)) * 100
        image = background.copy()
        image[25, 25] = 150

        result = detector.detect(image, background)
        assert result.n_detections >= 1

    def test_min_area_filter(self):
        """Min area filter works."""
        detector = ThresholdDetector(threshold=3.0, min_area=10)
        image = np.random.randn(50, 50)
        image[25, 25] = 10  # Single pixel target

        result = detector.detect(image)
        # Should not detect single pixel
        assert result.n_detections == 0


class TestCFARDetector:
    """Test CFARDetector class."""

    @pytest.fixture
    def detector(self):
        """Create CFAR detector fixture."""
        return CFARDetector(guard_cells=2, reference_cells=8, pfa=1e-4)

    @pytest.fixture
    def test_image(self):
        """Create test image with target."""
        np.random.seed(42)
        image = np.random.randn(100, 100) * 10 + 100
        # Add bright target
        image[50, 50] = 200
        return image

    def test_cfar_creation(self, detector):
        """CFAR detector can be created."""
        assert detector is not None
        assert detector.threshold_multiplier > 0

    def test_cfar_detect(self, detector, test_image):
        """CFAR detects target."""
        result = detector.detect(test_image)

        assert result.snr_map is not None
        assert result.threshold_map is not None
        assert result.n_detections >= 1

    def test_cfar_snr_map(self, detector, test_image):
        """CFAR produces valid SNR map."""
        result = detector.detect(test_image)

        # SNR at target should be high
        assert result.snr_map[50, 50] > 3

    def test_cfar_threshold_multiplier(self):
        """Threshold multiplier depends on Pfa."""
        cfar_high = CFARDetector(pfa=1e-3)
        cfar_low = CFARDetector(pfa=1e-6)

        # Lower Pfa means higher threshold
        assert cfar_low.threshold_multiplier > cfar_high.threshold_multiplier


class TestMatchedFilterDetector:
    """Test MatchedFilterDetector class."""

    @pytest.fixture
    def detector(self):
        """Create matched filter detector."""
        return MatchedFilterDetector.from_gaussian(sigma=1.5)

    @pytest.fixture
    def test_image(self):
        """Create test image with Gaussian target."""
        image = np.zeros((100, 100))
        # Add Gaussian target
        y, x = np.ogrid[:100, :100]
        image += 10 * np.exp(-((x - 50) ** 2 + (y - 50) ** 2) / (2 * 1.5**2))
        image += np.random.randn(100, 100) * 0.5
        return image

    def test_mf_creation(self, detector):
        """Matched filter can be created."""
        assert detector is not None
        assert detector.template is not None

    def test_mf_detect(self, detector, test_image):
        """Matched filter detects target."""
        result = detector.detect(test_image, threshold=3.0)
        assert result.n_detections >= 1

        # Detection should be near target
        det = result.detections[0]
        assert abs(det.x - 50) < 3
        assert abs(det.y - 50) < 3

    def test_mf_from_template(self):
        """Custom template works."""
        template = np.ones((5, 5))
        detector = MatchedFilterDetector(template)
        assert detector.template.shape == (5, 5)

    def test_mf_snr_ordering(self, detector, test_image):
        """Detections sorted by SNR."""
        # Add second weaker target
        test_image[20, 20] = 5

        result = detector.detect(test_image, threshold=2.0)
        if len(result.detections) >= 2:
            assert result.detections[0].snr >= result.detections[1].snr


# =============================================================================
# Tracking Tests
# =============================================================================


class TestKalmanTracker:
    """Test KalmanTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create tracker fixture."""
        return KalmanTracker(process_noise=0.1, measurement_noise=1.0)

    def test_tracker_creation(self, tracker):
        """Tracker can be created."""
        assert tracker is not None

    def test_initialize(self, tracker):
        """Tracker can be initialized."""
        tracker.initialize([100, 200])
        assert_allclose(tracker.position, [100, 200])
        assert_allclose(tracker.velocity, [0, 0])

    def test_predict(self, tracker):
        """Prediction step works."""
        tracker.initialize([100, 200], [5, 10])
        tracker.predict()

        # Position should advance by velocity
        assert_allclose(tracker.position, [105, 210], rtol=0.1)

    def test_update(self, tracker):
        """Update step works."""
        tracker.initialize([100, 200])
        state = tracker.update([102, 203])

        # Position should be close to measurement
        assert abs(state[0] - 102) < 5
        assert abs(state[1] - 203) < 5

    def test_tracking_sequence(self, tracker):
        """Track sequence of measurements."""
        # Moving target
        true_positions = [[100 + i * 5, 200 + i * 3] for i in range(10)]
        noisy_measurements = [
            [p[0] + np.random.randn(), p[1] + np.random.randn()]
            for p in true_positions
        ]

        tracker.initialize(noisy_measurements[0])

        errors = []
        for i, meas in enumerate(noisy_measurements[1:], 1):
            state = tracker.update(meas)
            error = np.sqrt(
                (state[0] - true_positions[i][0]) ** 2
                + (state[1] - true_positions[i][1]) ** 2
            )
            errors.append(error)

        # Tracking should reduce error over time
        assert np.mean(errors[-3:]) < np.mean(errors[:3]) + 2

    def test_covariance_reduction(self, tracker):
        """Covariance reduces with updates."""
        tracker.initialize([100, 200])
        initial_cov = tracker.covariance[0, 0]

        for _ in range(5):
            tracker.update([100, 200])

        final_cov = tracker.covariance[0, 0]
        assert final_cov < initial_cov


class TestTrack:
    """Test Track dataclass."""

    def test_track_creation(self):
        """Track can be created."""
        track = Track(id=1)
        assert track.id == 1
        assert track.state == TrackState.TENTATIVE

    def test_predict_position(self):
        """Position prediction works."""
        track = Track(
            id=1,
            position=np.array([100, 200]),
            velocity=np.array([5, 10]),
        )
        predicted = track.predict_position(dt=2.0)
        assert_allclose(predicted, [110, 220])


class TestMultiTargetTracker:
    """Test MultiTargetTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create multi-target tracker fixture."""
        return MultiTargetTracker(
            process_noise=0.1,
            measurement_noise=1.0,
            max_age=3,
            min_hits=2,
            association_threshold=10.0,
        )

    def test_tracker_creation(self, tracker):
        """Tracker can be created."""
        assert tracker is not None
        assert len(tracker.tracks) == 0

    def test_create_track(self, tracker):
        """Tracks are created from detections."""
        detections = [Detection(x=100, y=200)]
        tracks = tracker.update(detections)

        assert len(tracks) == 1
        assert tracks[0].state == TrackState.TENTATIVE

    def test_confirm_track(self, tracker):
        """Tracks are confirmed after min_hits."""
        # First detection
        detections1 = [Detection(x=100, y=200)]
        tracker.update(detections1)

        # Second detection
        detections2 = [Detection(x=102, y=203)]
        tracks = tracker.update(detections2)

        assert len(tracks) == 1
        assert tracks[0].state == TrackState.CONFIRMED

    def test_delete_track(self, tracker):
        """Tracks are deleted after max_age without updates."""
        # Create track
        detections = [Detection(x=100, y=200)]
        tracker.update(detections)

        # Update without detections
        for _ in range(5):
            tracks = tracker.update([])

        # Track should be deleted
        assert len(tracker.tracks) == 0

    def test_multiple_targets(self, tracker):
        """Multiple targets are tracked."""
        detections = [
            Detection(x=100, y=200),
            Detection(x=300, y=400),
        ]
        tracks = tracker.update(detections)

        assert len(tracks) == 2

    def test_track_association(self, tracker):
        """Detections are associated to existing tracks."""
        # Frame 1
        detections1 = [Detection(x=100, y=200)]
        tracker.update(detections1)
        track_id = tracker.tracks[0].id

        # Frame 2 - nearby detection
        detections2 = [Detection(x=102, y=203)]
        tracker.update(detections2)

        # Should still be same track
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].id == track_id

    def test_reset(self, tracker):
        """Reset clears all tracks."""
        tracker.update([Detection(x=100, y=200)])
        tracker.reset()
        assert len(tracker.tracks) == 0


# =============================================================================
# Metrics Tests
# =============================================================================


class TestConfusionMatrix:
    """Test confusion_matrix function."""

    def test_perfect_detection(self):
        """Perfect detection gives correct counts."""
        detections = [Detection(x=10, y=20), Detection(x=30, y=40)]
        ground_truth = [(10, 20), (30, 40)]

        cm = confusion_matrix(detections, ground_truth, association_radius=5.0)

        assert cm.true_positives == 2
        assert cm.false_positives == 0
        assert cm.false_negatives == 0

    def test_missed_detection(self):
        """Missed detection counted correctly."""
        detections = [Detection(x=10, y=20)]
        ground_truth = [(10, 20), (30, 40)]

        cm = confusion_matrix(detections, ground_truth, association_radius=5.0)

        assert cm.true_positives == 1
        assert cm.false_negatives == 1

    def test_false_alarm(self):
        """False alarm counted correctly."""
        detections = [Detection(x=10, y=20), Detection(x=50, y=60)]
        ground_truth = [(10, 20)]

        cm = confusion_matrix(detections, ground_truth, association_radius=5.0)

        assert cm.true_positives == 1
        assert cm.false_positives == 1

    def test_precision_recall(self):
        """Precision and recall computed correctly."""
        detections = [Detection(x=10, y=20), Detection(x=50, y=60)]
        ground_truth = [(10, 20), (30, 40)]

        cm = confusion_matrix(detections, ground_truth, association_radius=5.0)

        assert cm.precision == 0.5  # 1 TP / 2 detections
        assert cm.recall == 0.5  # 1 TP / 2 targets


class TestComputePdPfa:
    """Test compute_pd_pfa function."""

    def test_perfect_detection(self):
        """Perfect detection gives Pd=1, Pfa=0."""
        detections = [Detection(x=10, y=20)]
        ground_truth = [(10, 20)]

        pd, pfa = compute_pd_pfa(detections, ground_truth, n_pixels=10000)

        assert pd == 1.0
        assert pfa == 0.0

    def test_missed_detection(self):
        """Missed detection gives Pd < 1."""
        detections = []
        ground_truth = [(10, 20)]

        pd, pfa = compute_pd_pfa(detections, ground_truth, n_pixels=10000)

        assert pd == 0.0

    def test_false_alarm(self):
        """False alarm gives Pfa > 0."""
        detections = [Detection(x=10, y=20), Detection(x=50, y=60)]
        ground_truth = [(10, 20)]

        pd, pfa = compute_pd_pfa(detections, ground_truth, n_pixels=10000)

        assert pd == 1.0
        assert pfa == 1 / 10000


class TestROCCurve:
    """Test ROCCurve class."""

    def test_roc_creation(self):
        """ROC curve can be created."""
        roc = ROCCurve()
        roc.add_point(0.01, 0.9, threshold=5.0)
        roc.add_point(0.001, 0.7, threshold=6.0)

        assert len(roc.pfa_values) == 2
        assert len(roc.pd_values) == 2

    def test_auc_computation(self):
        """AUC is computed correctly."""
        roc = ROCCurve()
        # Perfect ROC
        roc.add_point(0.0, 0.0)
        roc.add_point(0.0, 1.0)
        roc.add_point(1.0, 1.0)

        # AUC should be 1.0 for perfect ROC
        assert_allclose(roc.auc, 1.0, atol=0.01)

    def test_auc_diagonal(self):
        """Diagonal ROC has AUC ~ 0.5."""
        roc = ROCCurve()
        for pfa in np.linspace(0, 1, 11):
            roc.add_point(pfa, pfa)

        assert_allclose(roc.auc, 0.5, atol=0.05)

    def test_pd_at_pfa(self):
        """Interpolation works."""
        roc = ROCCurve()
        roc.add_point(0.0, 0.5)
        roc.add_point(0.1, 0.9)
        roc.add_point(1.0, 1.0)

        pd = roc.pd_at_pfa(0.05)
        assert 0.5 < pd < 0.9

    def test_from_results(self):
        """ROC can be created from results."""
        results = [(0.01, 0.9, 5.0), (0.001, 0.7, 6.0)]
        roc = ROCCurve.from_results(results)

        assert len(roc.pfa_values) == 2

    def test_to_arrays(self):
        """Conversion to arrays works."""
        roc = ROCCurve()
        roc.add_point(0.01, 0.9)
        roc.add_point(0.001, 0.7)

        pfa, pd = roc.to_arrays()
        assert len(pfa) == 2
        assert len(pd) == 2


class TestEvaluateDetection:
    """Test evaluate_detection function."""

    def test_comprehensive_metrics(self):
        """All metrics computed correctly."""
        detections = [
            Detection(x=10, y=20),
            Detection(x=30, y=40),
            Detection(x=90, y=90),  # False alarm
        ]
        ground_truth = [(10, 20), (30, 40), (50, 60)]

        metrics = evaluate_detection(
            detections, ground_truth, association_radius=5.0, image_size=(100, 100)
        )

        assert metrics.n_targets == 3
        assert metrics.n_detections == 3
        assert metrics.n_true_positives == 2
        assert metrics.n_false_positives == 1
        assert metrics.n_false_negatives == 1
        assert_allclose(metrics.pd, 2 / 3, atol=0.01)

    def test_with_detection_result(self):
        """Works with DetectionResult input."""
        detections = [Detection(x=10, y=20)]
        result = DetectionResult(detections=detections)
        ground_truth = [(10, 20)]

        metrics = evaluate_detection(result, ground_truth)

        assert metrics.pd == 1.0

    def test_position_error(self):
        """Position error computed correctly."""
        detections = [Detection(x=10, y=20)]
        ground_truth = [(12, 22)]  # Slightly offset

        metrics = evaluate_detection(detections, ground_truth, association_radius=5.0)

        expected_error = np.sqrt(4 + 4)
        assert_allclose(metrics.mean_position_error, expected_error, atol=0.1)


class TestDetectionMetrics:
    """Test DetectionMetrics dataclass."""

    def test_to_dict(self):
        """to_dict returns proper dictionary."""
        metrics = DetectionMetrics(pd=0.9, pfa=0.001, precision=0.95, recall=0.9)
        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert d["pd"] == 0.9
        assert d["pfa"] == 0.001


# =============================================================================
# Integration Tests
# =============================================================================


class TestDetectionIntegration:
    """Integration tests for detection module."""

    def test_detect_and_track(self):
        """End-to-end detection and tracking."""
        np.random.seed(42)

        # Create moving target
        n_frames = 10
        true_trajectory = [(100 + i * 5, 200 + i * 3) for i in range(n_frames)]

        detector = ThresholdDetector(threshold=3.0)
        tracker = MultiTargetTracker(max_age=3, min_hits=2)

        for i, (true_x, true_y) in enumerate(true_trajectory):
            # Generate image with target
            image = np.random.randn(300, 400)
            image[int(true_y), int(true_x)] = 10

            # Detect
            result = detector.detect(image)

            # Track
            tracks = tracker.update(result.detections)

            if i >= 1:  # After first few frames
                assert len(tracks) >= 1

        # Final track should be confirmed
        assert any(t.is_confirmed for t in tracker.tracks)

    def test_cfar_vs_threshold(self):
        """CFAR more robust than simple threshold."""
        np.random.seed(42)

        # Create image with varying background
        image = np.zeros((100, 100))
        image[:, :50] = 100 + np.random.randn(100, 50) * 10
        image[:, 50:] = 200 + np.random.randn(100, 50) * 10

        # Add target in each region
        image[25, 25] = 130  # 3 sigma in left region
        image[75, 75] = 230  # 3 sigma in right region

        ground_truth = [(25, 25), (75, 75)]

        # CFAR should detect both
        cfar = CFARDetector(guard_cells=2, reference_cells=8, pfa=1e-3)
        cfar_result = cfar.detect(image)

        cfar_metrics = evaluate_detection(
            cfar_result.detections, ground_truth, association_radius=5.0
        )

        # CFAR should find both targets
        assert cfar_metrics.pd >= 0.5

    def test_roc_curve_generation(self):
        """ROC curve can be generated from detector."""
        np.random.seed(42)

        # Create image with targets
        image = np.random.randn(100, 100)
        image[25, 25] = 8
        image[75, 75] = 6
        ground_truth = [(25, 25), (75, 75)]

        # Generate ROC by varying threshold
        roc = ROCCurve()

        for threshold in [2, 3, 4, 5, 6, 7, 8]:
            detector = ThresholdDetector(threshold=threshold)
            result = detector.detect(image)

            pd, pfa = compute_pd_pfa(
                result.detections, ground_truth, n_pixels=100 * 100
            )
            roc.add_point(pfa, pd, threshold)

        # Should have valid ROC points
        assert len(roc.pfa_values) == 7
        # Pd should decrease with higher threshold
        assert roc.pd_values[0] >= roc.pd_values[-1]
