"""
Video output handling for EOSIM.

Provides VideoWriter class for creating video files from simulated frames.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray


class VideoWriter:
    """Video file writer for simulation output.

    Creates video files from sequences of frames with configurable
    codec, resolution, and frame rate.

    Example:
        >>> writer = VideoWriter("output.mp4", fps=30, resolution=(480, 640))
        >>> for frame in simulated_frames:
        ...     writer.write_frame(frame)
        >>> writer.close()

        # Or use context manager:
        >>> with VideoWriter("output.mp4", fps=30) as writer:
        ...     for frame in simulated_frames:
        ...         writer.write_frame(frame)
    """

    def __init__(
        self,
        path: Union[str, Path],
        fps: float = 30.0,
        resolution: Optional[tuple[int, int]] = None,
        codec: str = "mp4v",
        is_color: bool = False,
    ) -> None:
        """Initialize video writer.

        Args:
            path: Output video file path
            fps: Output frame rate
            resolution: (height, width), auto-detected from first frame if None
            codec: Video codec (fourcc code)
            is_color: Whether to write color video
        """
        self._path = Path(path)
        self._fps = fps
        self._resolution = resolution
        self._codec = codec
        self._is_color = is_color

        self._writer = None
        self._frame_count = 0
        self._is_open = False

        # Create parent directory if needed
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _initialize_writer(self, frame: NDArray) -> None:
        """Initialize the OpenCV VideoWriter with frame dimensions."""
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV is required for video writing. "
                "Install with: pip install opencv-python"
            )

        # Determine resolution from frame if not specified
        if self._resolution is None:
            if frame.ndim == 2:
                self._resolution = frame.shape
            else:
                self._resolution = frame.shape[:2]

        # Determine if color from frame
        is_color = self._is_color or (frame.ndim == 3 and frame.shape[2] >= 3)

        # Get fourcc code
        fourcc = cv2.VideoWriter_fourcc(*self._codec)

        # OpenCV expects (width, height)
        frame_size = (self._resolution[1], self._resolution[0])

        self._writer = cv2.VideoWriter(
            str(self._path),
            fourcc,
            self._fps,
            frame_size,
            isColor=is_color,
        )

        if not self._writer.isOpened():
            raise IOError(f"Could not open video writer: {self._path}")

        self._is_open = True

    @property
    def path(self) -> Path:
        """Get output path."""
        return self._path

    @property
    def frame_count(self) -> int:
        """Get number of frames written."""
        return self._frame_count

    @property
    def is_open(self) -> bool:
        """Check if writer is open."""
        return self._is_open

    @property
    def resolution(self) -> Optional[tuple[int, int]]:
        """Get output resolution."""
        return self._resolution

    def write_frame(self, frame: NDArray) -> None:
        """Write a frame to the video.

        Args:
            frame: Frame data (H, W) for grayscale or (H, W, 3) for color.
                   Values should be in range appropriate for the data type
                   (0-255 for uint8, 0-65535 for uint16, 0-1 for float).
        """
        # Initialize writer on first frame
        if self._writer is None:
            self._initialize_writer(frame)

        import cv2

        # Convert frame to appropriate format
        output_frame = self._prepare_frame(frame)

        # Resize if needed
        if self._resolution is not None:
            if output_frame.shape[:2] != self._resolution:
                output_frame = cv2.resize(
                    output_frame,
                    (self._resolution[1], self._resolution[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

        self._writer.write(output_frame)
        self._frame_count += 1

    def _prepare_frame(self, frame: NDArray) -> NDArray:
        """Prepare frame for writing.

        Converts to uint8 BGR format expected by OpenCV.
        """
        import cv2

        # Handle different input types
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            # Assume 0-1 range for float
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        elif frame.dtype == np.uint16:
            # Scale 16-bit to 8-bit
            frame = (frame / 256).astype(np.uint8)
        elif frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Handle color conversion
        if frame.ndim == 2:
            # Grayscale - convert to BGR if writer expects color
            if self._is_color:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3:
            if frame.shape[2] == 3:
                # Assume RGB, convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif frame.shape[2] == 4:
                # RGBA to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        return frame

    def __enter__(self) -> "VideoWriter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def close(self) -> None:
        """Close video file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._is_open = False


def frames_to_video(
    frames: list[NDArray],
    output_path: Union[str, Path],
    fps: float = 30.0,
    codec: str = "mp4v",
) -> Path:
    """Convert a list of frames to a video file.

    Args:
        frames: List of frame arrays
        output_path: Output video path
        fps: Frame rate
        codec: Video codec

    Returns:
        Path to created video
    """
    if not frames:
        raise ValueError("No frames provided")

    output_path = Path(output_path)

    with VideoWriter(output_path, fps=fps, codec=codec) as writer:
        for frame in frames:
            writer.write_frame(frame)

    return output_path


def dn_to_display(
    dn_image: NDArray,
    bit_depth: int = 14,
    percentile_clip: tuple[float, float] = (1, 99),
    colormap: Optional[str] = None,
) -> NDArray:
    """Convert DN image to displayable format.

    Applies percentile clipping and optional colormap for visualization.

    Args:
        dn_image: Digital number image
        bit_depth: Bit depth of DN values
        percentile_clip: (low, high) percentiles for contrast stretch
        colormap: OpenCV colormap name (e.g., "jet", "hot", "inferno")

    Returns:
        8-bit image suitable for display/video
    """
    # Convert to float for processing
    img = dn_image.astype(np.float32)

    # Apply percentile clipping
    low = np.percentile(img, percentile_clip[0])
    high = np.percentile(img, percentile_clip[1])

    if high > low:
        img = (img - low) / (high - low)
    else:
        img = np.zeros_like(img)

    img = np.clip(img, 0, 1)

    # Convert to 8-bit
    img_8bit = (img * 255).astype(np.uint8)

    # Apply colormap if requested
    if colormap is not None:
        try:
            import cv2
            colormap_codes = {
                "jet": cv2.COLORMAP_JET,
                "hot": cv2.COLORMAP_HOT,
                "inferno": cv2.COLORMAP_INFERNO,
                "plasma": cv2.COLORMAP_PLASMA,
                "viridis": cv2.COLORMAP_VIRIDIS,
                "turbo": cv2.COLORMAP_TURBO,
                "rainbow": cv2.COLORMAP_RAINBOW,
                "bone": cv2.COLORMAP_BONE,
            }
            cmap = colormap_codes.get(colormap.lower(), cv2.COLORMAP_JET)
            img_8bit = cv2.applyColorMap(img_8bit, cmap)
            # Convert BGR to RGB
            img_8bit = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2RGB)
        except ImportError:
            pass  # Return grayscale if OpenCV not available

    return img_8bit


def create_comparison_video(
    original_frames: list[NDArray],
    simulated_frames: list[NDArray],
    output_path: Union[str, Path],
    fps: float = 30.0,
    layout: str = "side_by_side",
    labels: tuple[str, str] = ("Original", "Simulated"),
) -> Path:
    """Create side-by-side comparison video.

    Args:
        original_frames: Original video frames
        simulated_frames: Simulated output frames
        output_path: Output video path
        fps: Frame rate
        layout: "side_by_side" or "top_bottom"
        labels: Labels for each video

    Returns:
        Path to created video
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV required for comparison video")

    if len(original_frames) != len(simulated_frames):
        raise ValueError("Frame counts must match")

    output_path = Path(output_path)

    with VideoWriter(output_path, fps=fps, is_color=True) as writer:
        for orig, sim in zip(original_frames, simulated_frames):
            # Convert both to displayable format
            if orig.dtype != np.uint8:
                orig = dn_to_display(orig)
            if sim.dtype != np.uint8:
                sim = dn_to_display(sim, colormap="inferno")

            # Ensure same size
            h = max(orig.shape[0], sim.shape[0])
            w = max(orig.shape[1], sim.shape[1])

            # Resize if needed
            if orig.shape[:2] != (h, w):
                orig = cv2.resize(orig, (w, h))
            if sim.shape[:2] != (h, w):
                sim = cv2.resize(sim, (w, h))

            # Ensure 3 channels
            if orig.ndim == 2:
                orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
            if sim.ndim == 2:
                sim = cv2.cvtColor(sim, cv2.COLOR_GRAY2RGB)

            # Combine frames
            if layout == "side_by_side":
                combined = np.hstack([orig, sim])
            else:  # top_bottom
                combined = np.vstack([orig, sim])

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined, labels[0], (10, 30), font, 1, (255, 255, 255), 2)
            if layout == "side_by_side":
                cv2.putText(combined, labels[1], (w + 10, 30), font, 1, (255, 255, 255), 2)
            else:
                cv2.putText(combined, labels[1], (10, h + 30), font, 1, (255, 255, 255), 2)

            writer.write_frame(combined)

    return output_path
