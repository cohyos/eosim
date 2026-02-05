"""
Target Tracking Algorithms for EOSIM.

Provides Kalman filter-based tracking for single and multiple targets.

Example 1: Single target Kalman filter
    >>> from eosim.detection import KalmanTracker
    >>> tracker = KalmanTracker(process_noise=0.1, measurement_noise=1.0)
    >>> tracker.initialize([100, 200])  # Initial position
    >>> for measurement in measurements:
    ...     state = tracker.update(measurement)
    ...     print(f"Position: {state[:2]}, Velocity: {state[2:4]}")

Example 2: Multi-target tracking
    >>> from eosim.detection import MultiTargetTracker
    >>> tracker = MultiTargetTracker(max_age=5, min_hits=3)
    >>> for frame_idx, detections in enumerate(detection_sequence):
    ...     tracks = tracker.update(detections)
    ...     for track in tracks:
    ...         print(f"Track {track.id}: {track.position}")

Example 3: Track management
    >>> tracker = MultiTargetTracker()
    >>> tracks = tracker.update(detections)
    >>> confirmed = [t for t in tracks if t.is_confirmed]
    >>> print(f"Confirmed tracks: {len(confirmed)}")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict
import numpy as np
from numpy.typing import NDArray

from .detectors import Detection


class TrackState(Enum):
    """Track lifecycle state."""

    TENTATIVE = "tentative"  # Not yet confirmed
    CONFIRMED = "confirmed"  # Confirmed track
    COASTED = "coasted"  # No recent detections
    DELETED = "deleted"  # Marked for deletion


@dataclass
class Track:
    """Single target track.

    Attributes:
        id: Unique track identifier
        state: Track lifecycle state
        position: Current position estimate [x, y]
        velocity: Current velocity estimate [vx, vy]
        covariance: State covariance matrix
        hits: Number of detection associations
        age: Track age in frames
        time_since_update: Frames since last update
        history: Position history
    """

    id: int
    state: TrackState = TrackState.TENTATIVE
    position: NDArray = field(default_factory=lambda: np.zeros(2))
    velocity: NDArray = field(default_factory=lambda: np.zeros(2))
    covariance: NDArray = field(default_factory=lambda: np.eye(4))
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    history: List[NDArray] = field(default_factory=list)

    @property
    def is_confirmed(self) -> bool:
        """Check if track is confirmed."""
        return self.state == TrackState.CONFIRMED

    @property
    def state_vector(self) -> NDArray:
        """Get full state vector [x, y, vx, vy]."""
        return np.concatenate([self.position, self.velocity])

    def predict_position(self, dt: float = 1.0) -> NDArray:
        """Predict position at time dt in future.

        Args:
            dt: Time step

        Returns:
            Predicted [x, y] position
        """
        return self.position + self.velocity * dt


class KalmanTracker:
    """Kalman filter for single target tracking.

    Implements constant velocity motion model with position measurements.
    State: [x, y, vx, vy]
    Measurement: [x, y]
    """

    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 1.0,
        dt: float = 1.0,
    ) -> None:
        """Initialize Kalman tracker.

        Args:
            process_noise: Process noise standard deviation
            measurement_noise: Measurement noise standard deviation
            dt: Time step
        """
        self._dt = dt
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise

        # State transition matrix (constant velocity)
        self._F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        # Measurement matrix (observe position only)
        self._H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )

        # Process noise covariance
        q = process_noise**2
        self._Q = np.array(
            [
                [dt**4 / 4, 0, dt**3 / 2, 0],
                [0, dt**4 / 4, 0, dt**3 / 2],
                [dt**3 / 2, 0, dt**2, 0],
                [0, dt**3 / 2, 0, dt**2],
            ]
        ) * q

        # Measurement noise covariance
        r = measurement_noise**2
        self._R = np.eye(2) * r

        # State and covariance
        self._x = np.zeros(4)
        self._P = np.eye(4) * 100  # Large initial uncertainty

        self._initialized = False

    @property
    def state(self) -> NDArray:
        """Get current state estimate."""
        return self._x.copy()

    @property
    def position(self) -> NDArray:
        """Get position estimate."""
        return self._x[:2].copy()

    @property
    def velocity(self) -> NDArray:
        """Get velocity estimate."""
        return self._x[2:4].copy()

    @property
    def covariance(self) -> NDArray:
        """Get state covariance."""
        return self._P.copy()

    def initialize(
        self,
        position: NDArray,
        velocity: Optional[NDArray] = None,
    ) -> None:
        """Initialize filter with measurement.

        Args:
            position: Initial position [x, y]
            velocity: Initial velocity [vx, vy] (default: zero)
        """
        self._x[:2] = np.asarray(position)
        if velocity is not None:
            self._x[2:4] = np.asarray(velocity)
        else:
            self._x[2:4] = 0

        self._initialized = True

    def predict(self) -> NDArray:
        """Predict next state.

        Returns:
            Predicted state vector
        """
        # State prediction
        self._x = self._F @ self._x

        # Covariance prediction
        self._P = self._F @ self._P @ self._F.T + self._Q

        return self._x.copy()

    def update(
        self,
        measurement: Optional[NDArray],
    ) -> NDArray:
        """Update with measurement.

        Args:
            measurement: Position measurement [x, y], or None for prediction only

        Returns:
            Updated state vector
        """
        # Predict
        self.predict()

        if measurement is None:
            return self._x.copy()

        # Innovation
        z = np.asarray(measurement)
        y = z - self._H @ self._x

        # Innovation covariance
        S = self._H @ self._P @ self._H.T + self._R

        # Kalman gain
        K = self._P @ self._H.T @ np.linalg.inv(S)

        # State update
        self._x = self._x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(4) - K @ self._H
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T

        return self._x.copy()

    def likelihood(
        self,
        measurement: NDArray,
    ) -> float:
        """Compute measurement likelihood.

        Args:
            measurement: Position measurement [x, y]

        Returns:
            Log-likelihood of measurement
        """
        z = np.asarray(measurement)
        y = z - self._H @ self._x

        S = self._H @ self._P @ self._H.T + self._R
        S_inv = np.linalg.inv(S)

        mahalanobis_sq = y.T @ S_inv @ y
        log_det = np.log(np.linalg.det(S))

        return -0.5 * (mahalanobis_sq + log_det + 2 * np.log(2 * np.pi))


class MultiTargetTracker:
    """Multi-target tracker with track management.

    Uses global nearest neighbor (GNN) association and Kalman filtering.
    """

    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 1.0,
        max_age: int = 5,
        min_hits: int = 3,
        association_threshold: float = 50.0,
    ) -> None:
        """Initialize multi-target tracker.

        Args:
            process_noise: Kalman filter process noise
            measurement_noise: Kalman filter measurement noise
            max_age: Maximum frames without update before deletion
            min_hits: Minimum hits to confirm track
            association_threshold: Maximum distance for association
        """
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._max_age = max_age
        self._min_hits = min_hits
        self._association_threshold = association_threshold

        self._tracks: Dict[int, Tuple[Track, KalmanTracker]] = {}
        self._next_id = 0
        self._frame_count = 0

    @property
    def tracks(self) -> List[Track]:
        """Get all active tracks."""
        return [track for track, _ in self._tracks.values()]

    @property
    def confirmed_tracks(self) -> List[Track]:
        """Get confirmed tracks only."""
        return [t for t in self.tracks if t.is_confirmed]

    def _create_track(self, detection: Detection) -> int:
        """Create new track from detection.

        Args:
            detection: Detection to initialize track

        Returns:
            Track ID
        """
        track_id = self._next_id
        self._next_id += 1

        track = Track(
            id=track_id,
            state=TrackState.TENTATIVE,
            position=np.array([detection.x, detection.y]),
            velocity=np.zeros(2),
            hits=1,
            age=0,
        )
        track.history.append(track.position.copy())

        kf = KalmanTracker(
            process_noise=self._process_noise,
            measurement_noise=self._measurement_noise,
        )
        kf.initialize([detection.x, detection.y])

        self._tracks[track_id] = (track, kf)

        return track_id

    def _compute_cost_matrix(
        self,
        detections: List[Detection],
    ) -> NDArray:
        """Compute cost matrix for association.

        Args:
            detections: List of detections

        Returns:
            Cost matrix (n_tracks × n_detections)
        """
        track_list = list(self._tracks.values())
        n_tracks = len(track_list)
        n_dets = len(detections)

        if n_tracks == 0 or n_dets == 0:
            return np.array([]).reshape(n_tracks, n_dets)

        cost = np.zeros((n_tracks, n_dets))

        for i, (track, kf) in enumerate(track_list):
            predicted_pos = kf.position
            for j, det in enumerate(detections):
                dist = np.sqrt(
                    (predicted_pos[0] - det.x) ** 2 + (predicted_pos[1] - det.y) ** 2
                )
                cost[i, j] = dist

        return cost

    def _associate(
        self,
        cost: NDArray,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to tracks using Hungarian algorithm.

        Args:
            cost: Cost matrix

        Returns:
            Tuple of (matches, unmatched_tracks, unmatched_detections)
        """
        if cost.size == 0:
            return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

        n_tracks, n_dets = cost.shape

        # Simple greedy association (can be replaced with Hungarian)
        matches = []
        unmatched_tracks = list(range(n_tracks))
        unmatched_dets = list(range(n_dets))

        # Sort by cost
        indices = np.argsort(cost.ravel())

        for idx in indices:
            i, j = divmod(idx, n_dets)

            if cost[i, j] > self._association_threshold:
                break

            if i in unmatched_tracks and j in unmatched_dets:
                matches.append((i, j))
                unmatched_tracks.remove(i)
                unmatched_dets.remove(j)

        return matches, unmatched_tracks, unmatched_dets

    def update(
        self,
        detections: List[Detection],
    ) -> List[Track]:
        """Update tracker with new detections.

        Args:
            detections: List of detections in current frame

        Returns:
            List of active tracks
        """
        self._frame_count += 1

        # Predict all tracks
        for track, kf in self._tracks.values():
            kf.predict()
            track.age += 1

        # Compute cost and associate
        track_list = list(self._tracks.items())
        cost = self._compute_cost_matrix(detections)
        matches, unmatched_tracks, unmatched_dets = self._associate(cost)

        # Update matched tracks
        for track_idx, det_idx in matches:
            track_id = track_list[track_idx][0]
            track, kf = self._tracks[track_id]
            det = detections[det_idx]

            # Kalman update
            kf.update(np.array([det.x, det.y]))

            # Update track
            track.position = kf.position
            track.velocity = kf.velocity
            track.covariance = kf.covariance
            track.hits += 1
            track.time_since_update = 0
            track.history.append(track.position.copy())

            # Confirm track if enough hits
            if track.hits >= self._min_hits:
                track.state = TrackState.CONFIRMED

        # Handle unmatched tracks
        for track_idx in unmatched_tracks:
            track_id = track_list[track_idx][0]
            track, kf = self._tracks[track_id]

            track.time_since_update += 1
            track.position = kf.position
            track.velocity = kf.velocity

            if track.time_since_update > self._max_age:
                track.state = TrackState.DELETED
            elif track.state == TrackState.CONFIRMED:
                track.state = TrackState.COASTED

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx])

        # Remove deleted tracks
        self._tracks = {
            tid: (t, kf)
            for tid, (t, kf) in self._tracks.items()
            if t.state != TrackState.DELETED
        }

        return self.tracks

    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID.

        Args:
            track_id: Track identifier

        Returns:
            Track or None if not found
        """
        if track_id in self._tracks:
            return self._tracks[track_id][0]
        return None

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracks.clear()
        self._frame_count = 0
