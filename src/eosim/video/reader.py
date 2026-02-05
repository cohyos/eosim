"""
Video input handling for EOSIM.

Provides VideoReader class for extracting frames from video files
and converting them to formats suitable for simulation.
"""

from pathlib import Path
from typing import Optional, Iterator, Union
import numpy as np
from numpy.typing import NDArray

from eosim.video.config import VideoSource, FrameData, ChannelMode


class VideoReader:
    """Video file reader for frame extraction.

    Supports common video formats through OpenCV. Provides iteration
    over frames with optional frame skipping and range selection.

    Example:
        >>> reader = VideoReader("input.mp4")
        >>> print(f"Video: {reader.source.resolution}, {reader.source.fps} fps")
        >>> for frame in reader:
        ...     process_frame(frame.image)
        >>> reader.close()

        # Or use context manager:
        >>> with VideoReader("input.mp4") as reader:
        ...     for frame in reader:
        ...         process_frame(frame.image)
    """

    def __init__(
        self,
        path: Union[str, Path],
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        frame_step: int = 1,
        color_mode: str = "rgb",
    ) -> None:
        """Initialize video reader.

        Args:
            path: Path to video file
            start_frame: First frame to read (0-indexed)
            end_frame: Last frame to read (exclusive, None = all)
            frame_step: Read every Nth frame
            color_mode: Output color mode ("rgb", "bgr", "gray")
        """
        self._path = Path(path)
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._frame_step = frame_step
        self._color_mode = color_mode.lower()

        self._cap = None
        self._source: Optional[VideoSource] = None
        self._current_frame = 0
        self._is_open = False

        self._open()

    def _open(self) -> None:
        """Open video file and read metadata."""
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV is required for video processing. "
                "Install with: pip install opencv-python"
            )

        if not self._path.exists():
            raise FileNotFoundError(f"Video file not found: {self._path}")

        self._cap = cv2.VideoCapture(str(self._path))

        if not self._cap.isOpened():
            raise IOError(f"Could not open video file: {self._path}")

        # Read video properties
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))

        # Decode fourcc to string
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        self._source = VideoSource(
            path=self._path,
            fps=fps if fps > 0 else 30.0,
            frame_count=frame_count,
            resolution=(height, width),
            duration_s=frame_count / fps if fps > 0 else 0,
            codec=codec,
        )

        # Set end frame if not specified
        if self._end_frame is None:
            self._end_frame = frame_count

        # Seek to start frame
        if self._start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)

        self._current_frame = self._start_frame
        self._is_open = True

    @property
    def source(self) -> VideoSource:
        """Get video source metadata."""
        if self._source is None:
            raise RuntimeError("Video not opened")
        return self._source

    @property
    def is_open(self) -> bool:
        """Check if video is open."""
        return self._is_open

    @property
    def current_frame(self) -> int:
        """Get current frame position."""
        return self._current_frame

    @property
    def frames_remaining(self) -> int:
        """Get number of frames remaining."""
        if self._end_frame is None:
            return 0
        return max(0, (self._end_frame - self._current_frame) // self._frame_step)

    def seek(self, frame_index: int) -> bool:
        """Seek to a specific frame.

        Args:
            frame_index: Frame number to seek to

        Returns:
            True if seek successful
        """
        if self._cap is None:
            return False

        import cv2
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self._current_frame = frame_index
        return True

    def read_frame(self) -> Optional[FrameData]:
        """Read the next frame.

        Returns:
            FrameData or None if no more frames
        """
        if self._cap is None or not self._is_open:
            return None

        if self._end_frame is not None and self._current_frame >= self._end_frame:
            return None

        import cv2

        ret, frame = self._cap.read()

        if not ret or frame is None:
            return None

        # Convert color space
        if self._color_mode == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self._color_mode == "gray":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # else keep BGR

        timestamp = self._current_frame / self._source.fps if self._source else 0

        frame_data = FrameData(
            frame_index=self._current_frame,
            timestamp_s=timestamp,
            image=frame,
        )

        # Move to next frame (with step)
        self._current_frame += self._frame_step
        if self._frame_step > 1 and self._current_frame < (self._end_frame or float('inf')):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame)

        return frame_data

    def __iter__(self) -> Iterator[FrameData]:
        """Iterate over frames."""
        # Reset to start
        self.seek(self._start_frame)

        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame

    def __len__(self) -> int:
        """Get number of frames to be read."""
        if self._source is None or self._end_frame is None:
            return 0
        return (self._end_frame - self._start_frame) // self._frame_step

    def __enter__(self) -> "VideoReader":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def close(self) -> None:
        """Close video file."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False


def to_grayscale(
    image: NDArray,
    mode: ChannelMode = ChannelMode.LUMINANCE,
) -> NDArray:
    """Convert color image to grayscale.

    Args:
        image: Input image (H, W, C) or (H, W)
        mode: Conversion mode

    Returns:
        Grayscale image (H, W)
    """
    if image.ndim == 2:
        return image

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected color image with 3+ channels, got shape {image.shape}")

    if mode == ChannelMode.LUMINANCE:
        # BT.601 luminance
        return (
            0.299 * image[:, :, 0] +
            0.587 * image[:, :, 1] +
            0.114 * image[:, :, 2]
        ).astype(image.dtype)
    elif mode == ChannelMode.RED:
        return image[:, :, 0]
    elif mode == ChannelMode.GREEN:
        return image[:, :, 1]
    elif mode == ChannelMode.BLUE:
        return image[:, :, 2]
    elif mode == ChannelMode.AVERAGE:
        return np.mean(image[:, :, :3], axis=2).astype(image.dtype)
    elif mode == ChannelMode.MAX:
        return np.max(image[:, :, :3], axis=2).astype(image.dtype)
    else:
        raise ValueError(f"Unknown channel mode: {mode}")


def compute_optical_flow(
    frame1: NDArray,
    frame2: NDArray,
    method: str = "farneback",
) -> NDArray:
    """Compute optical flow between two frames.

    Args:
        frame1: First frame (grayscale)
        frame2: Second frame (grayscale)
        method: Flow computation method ("farneback", "lucas_kanade")

    Returns:
        Flow field (H, W, 2) with (dy, dx) per pixel
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV required for optical flow computation")

    # Ensure grayscale
    if frame1.ndim == 3:
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    if frame2.ndim == 3:
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

    # Ensure uint8
    if frame1.dtype != np.uint8:
        frame1 = (frame1 * 255).astype(np.uint8) if frame1.max() <= 1 else frame1.astype(np.uint8)
    if frame2.dtype != np.uint8:
        frame2 = (frame2 * 255).astype(np.uint8) if frame2.max() <= 1 else frame2.astype(np.uint8)

    if method == "farneback":
        flow = cv2.calcOpticalFlowFarneback(
            frame1, frame2, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
    else:
        raise ValueError(f"Unknown optical flow method: {method}")

    # OpenCV returns (dx, dy), convert to (dy, dx) for consistency with image coords
    return np.stack([flow[:, :, 1], flow[:, :, 0]], axis=-1)


def extract_frames(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    frame_indices: Optional[list[int]] = None,
    format: str = "png",
) -> list[Path]:
    """Extract specific frames from video to image files.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_indices: Specific frames to extract (None = all)
        format: Output image format

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    with VideoReader(video_path) as reader:
        for frame_data in reader:
            if frame_indices is not None and frame_data.frame_index not in frame_indices:
                continue

            try:
                import cv2
                output_path = output_dir / f"frame_{frame_data.frame_index:06d}.{format}"

                # Convert RGB to BGR for OpenCV
                if frame_data.image.ndim == 3:
                    image = cv2.cvtColor(frame_data.image, cv2.COLOR_RGB2BGR)
                else:
                    image = frame_data.image

                cv2.imwrite(str(output_path), image)
                saved.append(output_path)
            except ImportError:
                raise ImportError("OpenCV required for frame extraction")

    return saved
