"""
Timeline system for EOSIM Studio.

The Timeline provides video-editor style control over the simulation:
- Time scrubbing
- Keyframe animation
- Playback controls
- Frame-accurate seeking
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import time
import threading


class PlaybackState(Enum):
    """Timeline playback state."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


class KeyframeType(Enum):
    """Type of keyframe."""
    OBJECT_POSITION = "object_position"
    OBJECT_ORIENTATION = "object_orientation"
    OBJECT_VISIBILITY = "object_visibility"
    CAMERA_POSITION = "camera_position"
    CAMERA_ORIENTATION = "camera_orientation"
    CAMERA_ZOOM = "camera_zoom"
    CAMERA_SETTINGS = "camera_settings"
    MARKER = "marker"  # Just a label


@dataclass
class Keyframe:
    """A keyframe on the timeline.

    Keyframes mark specific times where properties are set.
    Between keyframes, values are interpolated.
    """
    time_sec: float
    keyframe_type: KeyframeType
    target_id: str  # Object ID or "camera"
    property_name: str
    value: Any
    interpolation: str = "linear"  # linear, smooth, step
    label: str = ""

    def __lt__(self, other):
        return self.time_sec < other.time_sec


@dataclass
class TimelineMarker:
    """A labeled marker on the timeline."""
    time_sec: float
    label: str
    color: str = "#ffcc00"


class Timeline:
    """Video-editor style timeline for animation control.

    The Timeline manages:
    - Current playback time
    - Keyframes for animation
    - Playback controls (play, pause, seek)
    - Frame callbacks for rendering

    Example:
        >>> timeline = Timeline(duration_sec=30.0, fps=30)
        >>> timeline.add_keyframe(0.0, "camera", "position", (0, 0, 1000))
        >>> timeline.add_keyframe(10.0, "camera", "position", (1000, 500, 1000))
        >>> timeline.on_frame(lambda t, f: render_frame(t))
        >>> timeline.play()
    """

    def __init__(self, duration_sec: float = 30.0, fps: float = 30.0):
        self.duration_sec = duration_sec
        self.fps = fps
        self.frame_duration = 1.0 / fps

        # Current state
        self.current_time: float = 0.0
        self.current_frame: int = 0
        self.state: PlaybackState = PlaybackState.STOPPED
        self.loop: bool = False

        # Keyframes organized by target
        self.keyframes: Dict[str, List[Keyframe]] = {}  # target_id -> list of keyframes
        self.markers: List[TimelineMarker] = []

        # Playback
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_playback = threading.Event()

        # Callbacks
        self._frame_callbacks: List[Callable[[float, int], None]] = []
        self._state_callbacks: List[Callable[[PlaybackState], None]] = []
        self._time_callbacks: List[Callable[[float], None]] = []

    @property
    def total_frames(self) -> int:
        """Total number of frames in timeline."""
        return int(self.duration_sec * self.fps)

    def time_to_frame(self, time_sec: float) -> int:
        """Convert time to frame number."""
        return int(time_sec * self.fps)

    def frame_to_time(self, frame: int) -> float:
        """Convert frame number to time."""
        return frame / self.fps

    # -------------------------------------------------------------------------
    # Keyframe Management
    # -------------------------------------------------------------------------

    def add_keyframe(self, time_sec: float, target_id: str,
                     property_name: str, value: Any,
                     keyframe_type: KeyframeType = KeyframeType.OBJECT_POSITION,
                     interpolation: str = "linear") -> Keyframe:
        """Add a keyframe to the timeline.

        Args:
            time_sec: Time position in seconds
            target_id: ID of object or "camera"
            property_name: Property being animated
            value: Value at this keyframe
            keyframe_type: Type of keyframe
            interpolation: Interpolation mode

        Returns:
            The created Keyframe
        """
        kf = Keyframe(
            time_sec=time_sec,
            keyframe_type=keyframe_type,
            target_id=target_id,
            property_name=property_name,
            value=value,
            interpolation=interpolation
        )

        if target_id not in self.keyframes:
            self.keyframes[target_id] = []

        self.keyframes[target_id].append(kf)
        self.keyframes[target_id].sort()  # Keep sorted by time

        return kf

    def remove_keyframe(self, keyframe: Keyframe):
        """Remove a keyframe."""
        if keyframe.target_id in self.keyframes:
            self.keyframes[keyframe.target_id] = [
                kf for kf in self.keyframes[keyframe.target_id]
                if kf != keyframe
            ]

    def get_keyframes_at(self, time_sec: float,
                         tolerance: float = 0.1) -> List[Keyframe]:
        """Get all keyframes near a specific time."""
        result = []
        for kf_list in self.keyframes.values():
            for kf in kf_list:
                if abs(kf.time_sec - time_sec) <= tolerance:
                    result.append(kf)
        return result

    def get_keyframes_for_target(self, target_id: str) -> List[Keyframe]:
        """Get all keyframes for a specific target."""
        return self.keyframes.get(target_id, [])

    def get_interpolated_value(self, target_id: str, property_name: str,
                               time_sec: float) -> Optional[Any]:
        """Get interpolated value at a specific time.

        Returns None if no keyframes exist for this property.
        """
        kf_list = self.keyframes.get(target_id, [])

        # Filter to relevant property
        relevant = [kf for kf in kf_list if kf.property_name == property_name]
        if not relevant:
            return None

        # Find surrounding keyframes
        prev_kf = None
        next_kf = None

        for kf in relevant:
            if kf.time_sec <= time_sec:
                prev_kf = kf
            elif next_kf is None:
                next_kf = kf
                break

        # Handle edge cases
        if prev_kf is None and next_kf is None:
            return None
        if prev_kf is None:
            return next_kf.value
        if next_kf is None:
            return prev_kf.value
        if prev_kf.time_sec == next_kf.time_sec:
            return prev_kf.value

        # Interpolate
        alpha = (time_sec - prev_kf.time_sec) / (next_kf.time_sec - prev_kf.time_sec)

        # Apply interpolation curve
        interp = prev_kf.interpolation
        if interp == "smooth":
            alpha = alpha * alpha * (3 - 2 * alpha)
        elif interp == "step":
            alpha = 0.0 if alpha < 1.0 else 1.0

        # Interpolate based on value type
        return self._interpolate_values(prev_kf.value, next_kf.value, alpha)

    def _interpolate_values(self, v0: Any, v1: Any, alpha: float) -> Any:
        """Interpolate between two values."""
        if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
            return v0 + alpha * (v1 - v0)

        if isinstance(v0, (list, tuple)) and isinstance(v1, (list, tuple)):
            return type(v0)(
                a + alpha * (b - a) for a, b in zip(v0, v1)
            )

        # For non-numeric types, use step interpolation
        return v0 if alpha < 0.5 else v1

    # -------------------------------------------------------------------------
    # Markers
    # -------------------------------------------------------------------------

    def add_marker(self, time_sec: float, label: str, color: str = "#ffcc00"):
        """Add a labeled marker to the timeline."""
        marker = TimelineMarker(time_sec=time_sec, label=label, color=color)
        self.markers.append(marker)
        self.markers.sort(key=lambda m: m.time_sec)

    def remove_marker(self, marker: TimelineMarker):
        """Remove a marker."""
        self.markers = [m for m in self.markers if m != marker]

    def get_next_marker(self, from_time: float) -> Optional[TimelineMarker]:
        """Get the next marker after a given time."""
        for marker in self.markers:
            if marker.time_sec > from_time:
                return marker
        return None

    def get_prev_marker(self, from_time: float) -> Optional[TimelineMarker]:
        """Get the previous marker before a given time."""
        prev = None
        for marker in self.markers:
            if marker.time_sec >= from_time:
                break
            prev = marker
        return prev

    # -------------------------------------------------------------------------
    # Playback Control
    # -------------------------------------------------------------------------

    def play(self):
        """Start playback."""
        if self.state == PlaybackState.PLAYING:
            return

        self.state = PlaybackState.PLAYING
        self._stop_playback.clear()
        self._notify_state_change()

        # Start playback thread
        self._playback_thread = threading.Thread(target=self._playback_loop)
        self._playback_thread.daemon = True
        self._playback_thread.start()

    def pause(self):
        """Pause playback."""
        if self.state == PlaybackState.PLAYING:
            self.state = PlaybackState.PAUSED
            self._stop_playback.set()
            self._notify_state_change()

    def stop(self):
        """Stop playback and return to start."""
        self._stop_playback.set()
        self.state = PlaybackState.STOPPED
        self.seek(0.0)
        self._notify_state_change()

    def toggle_playback(self):
        """Toggle between play and pause."""
        if self.state == PlaybackState.PLAYING:
            self.pause()
        else:
            self.play()

    def seek(self, time_sec: float):
        """Seek to a specific time."""
        self.current_time = max(0.0, min(time_sec, self.duration_sec))
        self.current_frame = self.time_to_frame(self.current_time)
        self._notify_time_change()
        self._notify_frame(self.current_time, self.current_frame)

    def seek_frame(self, frame: int):
        """Seek to a specific frame."""
        self.seek(self.frame_to_time(frame))

    def step_forward(self, frames: int = 1):
        """Step forward by N frames."""
        self.seek_frame(self.current_frame + frames)

    def step_backward(self, frames: int = 1):
        """Step backward by N frames."""
        self.seek_frame(max(0, self.current_frame - frames))

    def go_to_start(self):
        """Go to the beginning."""
        self.seek(0.0)

    def go_to_end(self):
        """Go to the end."""
        self.seek(self.duration_sec)

    def go_to_next_marker(self):
        """Jump to the next marker."""
        marker = self.get_next_marker(self.current_time)
        if marker:
            self.seek(marker.time_sec)

    def go_to_prev_marker(self):
        """Jump to the previous marker."""
        marker = self.get_prev_marker(self.current_time)
        if marker:
            self.seek(marker.time_sec)

    def _playback_loop(self):
        """Internal playback loop (runs in thread)."""
        last_time = time.time()

        while not self._stop_playback.is_set():
            # Calculate elapsed real time
            now = time.time()
            dt = now - last_time
            last_time = now

            # Advance timeline
            self.current_time += dt
            self.current_frame = self.time_to_frame(self.current_time)

            # Check for end
            if self.current_time >= self.duration_sec:
                if self.loop:
                    self.current_time = 0.0
                    self.current_frame = 0
                else:
                    self.current_time = self.duration_sec
                    self.state = PlaybackState.STOPPED
                    self._notify_state_change()
                    break

            # Notify callbacks
            self._notify_time_change()
            self._notify_frame(self.current_time, self.current_frame)

            # Sleep until next frame
            sleep_time = self.frame_duration - (time.time() - now)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_frame(self, callback: Callable[[float, int], None]):
        """Register callback for each frame.

        Callback receives (time_sec, frame_number).
        """
        self._frame_callbacks.append(callback)

    def on_state_change(self, callback: Callable[[PlaybackState], None]):
        """Register callback for playback state changes."""
        self._state_callbacks.append(callback)

    def on_time_change(self, callback: Callable[[float], None]):
        """Register callback for time changes (including scrubbing)."""
        self._time_callbacks.append(callback)

    def _notify_frame(self, time_sec: float, frame: int):
        """Notify frame callbacks."""
        for cb in self._frame_callbacks:
            try:
                cb(time_sec, frame)
            except Exception as e:
                print(f"Frame callback error: {e}")

    def _notify_state_change(self):
        """Notify state change callbacks."""
        for cb in self._state_callbacks:
            try:
                cb(self.state)
            except Exception as e:
                print(f"State callback error: {e}")

    def _notify_time_change(self):
        """Notify time change callbacks."""
        for cb in self._time_callbacks:
            try:
                cb(self.current_time)
            except Exception as e:
                print(f"Time callback error: {e}")

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize timeline to dictionary."""
        return {
            "duration_sec": self.duration_sec,
            "fps": self.fps,
            "loop": self.loop,
            "keyframes": [
                {
                    "time_sec": kf.time_sec,
                    "type": kf.keyframe_type.value,
                    "target_id": kf.target_id,
                    "property_name": kf.property_name,
                    "value": kf.value,
                    "interpolation": kf.interpolation,
                    "label": kf.label,
                }
                for kf_list in self.keyframes.values()
                for kf in kf_list
            ],
            "markers": [
                {
                    "time_sec": m.time_sec,
                    "label": m.label,
                    "color": m.color,
                }
                for m in self.markers
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Timeline":
        """Deserialize timeline from dictionary."""
        timeline = cls(
            duration_sec=data.get("duration_sec", 30.0),
            fps=data.get("fps", 30.0)
        )
        timeline.loop = data.get("loop", False)

        for kf_data in data.get("keyframes", []):
            timeline.add_keyframe(
                time_sec=kf_data["time_sec"],
                target_id=kf_data["target_id"],
                property_name=kf_data["property_name"],
                value=kf_data["value"],
                keyframe_type=KeyframeType(kf_data.get("type", "object_position")),
                interpolation=kf_data.get("interpolation", "linear")
            )

        for m_data in data.get("markers", []):
            timeline.add_marker(
                time_sec=m_data["time_sec"],
                label=m_data["label"],
                color=m_data.get("color", "#ffcc00")
            )

        return timeline
