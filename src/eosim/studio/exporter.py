"""
Video export for EOSIM Studio.

Exports rendered frames to video files (MP4, AVI) or image sequences.
Uses OpenCV if available, otherwise falls back to image sequence export.
"""

import os
from typing import Optional, Callable, List
from dataclasses import dataclass
import numpy as np

from eosim.studio.renderer import Renderer, RenderedFrame, RenderQueue
from eosim.studio.project import RenderSettings


# Check for OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


@dataclass
class ExportProgress:
    """Export progress information."""
    current_frame: int
    total_frames: int
    percent: float
    elapsed_sec: float
    estimated_remaining_sec: float
    status: str


class VideoExporter:
    """Exports rendered frames to video files.

    Supports:
    - MP4 (requires OpenCV with ffmpeg)
    - AVI (requires OpenCV)
    - Image sequence (PNG, always available)

    Example:
        >>> exporter = VideoExporter(renderer, settings)
        >>> exporter.export("output.mp4", progress_callback=print_progress)
    """

    # Video codec mappings
    CODECS = {
        "mp4": "mp4v",  # or 'avc1' for H.264
        "avi": "XVID",
        "mov": "mp4v",
    }

    def __init__(self, renderer: Renderer, settings: RenderSettings):
        self.renderer = renderer
        self.settings = settings

        self.is_exporting = False
        self.cancel_requested = False

    def can_export_video(self) -> bool:
        """Check if video export is available."""
        return HAS_OPENCV

    def export(self, output_path: str,
               duration_sec: Optional[float] = None,
               fps: Optional[float] = None,
               progress_callback: Optional[Callable[[ExportProgress], None]] = None) -> bool:
        """Export video to file.

        Args:
            output_path: Path to output file
            duration_sec: Duration to export (default: from settings)
            fps: Frame rate (default: from settings)
            progress_callback: Optional callback for progress updates

        Returns:
            True if export succeeded
        """
        import time

        duration = duration_sec or 30.0
        frame_rate = fps or self.settings.fps
        total_frames = int(duration * frame_rate)

        self.is_exporting = True
        self.cancel_requested = False
        start_time = time.time()

        # Determine export format
        _, ext = os.path.splitext(output_path)
        ext = ext.lower().lstrip(".")

        if ext in ["mp4", "avi", "mov"] and HAS_OPENCV:
            success = self._export_video_opencv(
                output_path, ext, total_frames, frame_rate,
                progress_callback, start_time
            )
        else:
            # Fall back to image sequence
            success = self._export_image_sequence(
                output_path, total_frames, frame_rate,
                progress_callback, start_time
            )

        self.is_exporting = False
        return success

    def _export_video_opencv(self, output_path: str, format: str,
                            total_frames: int, fps: float,
                            progress_callback, start_time) -> bool:
        """Export using OpenCV VideoWriter."""
        import time

        width, height = self.settings.resolution
        fourcc = cv2.VideoWriter_fourcc(*self.CODECS.get(format, "mp4v"))

        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"Failed to open video writer for {output_path}")
            return False

        try:
            for i in range(total_frames):
                if self.cancel_requested:
                    break

                # Render frame
                time_sec = i / fps
                frame = self.renderer.render_frame(time_sec, i)

                # Add overlay if requested
                if self.settings.include_overlay:
                    image = self.renderer.render_with_overlay(
                        frame,
                        show_info=True,
                        show_crosshair=self.settings.include_crosshair
                    )
                else:
                    image = frame.image

                # Resize if needed
                if image.shape[1] != width or image.shape[0] != height:
                    image = cv2.resize(image, (width, height))

                # Convert RGB to BGR for OpenCV
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                writer.write(bgr)

                # Progress callback
                if progress_callback:
                    elapsed = time.time() - start_time
                    if i > 0:
                        remaining = elapsed * (total_frames - i) / i
                    else:
                        remaining = 0

                    progress_callback(ExportProgress(
                        current_frame=i + 1,
                        total_frames=total_frames,
                        percent=(i + 1) / total_frames * 100,
                        elapsed_sec=elapsed,
                        estimated_remaining_sec=remaining,
                        status="Rendering"
                    ))

        finally:
            writer.release()

        return not self.cancel_requested

    def _export_image_sequence(self, output_path: str,
                               total_frames: int, fps: float,
                               progress_callback, start_time) -> bool:
        """Export as image sequence."""
        import time

        # Create output directory
        base_dir = os.path.splitext(output_path)[0] + "_frames"
        os.makedirs(base_dir, exist_ok=True)

        try:
            for i in range(total_frames):
                if self.cancel_requested:
                    break

                # Render frame
                time_sec = i / fps
                frame = self.renderer.render_frame(time_sec, i)

                # Add overlay if requested
                if self.settings.include_overlay:
                    image = self.renderer.render_with_overlay(
                        frame,
                        show_info=True,
                        show_crosshair=self.settings.include_crosshair
                    )
                else:
                    image = frame.image

                # Save as PNG
                frame_path = os.path.join(base_dir, f"frame_{i:06d}.png")
                self._save_png(frame_path, image)

                # Progress callback
                if progress_callback:
                    elapsed = time.time() - start_time
                    if i > 0:
                        remaining = elapsed * (total_frames - i) / i
                    else:
                        remaining = 0

                    progress_callback(ExportProgress(
                        current_frame=i + 1,
                        total_frames=total_frames,
                        percent=(i + 1) / total_frames * 100,
                        elapsed_sec=elapsed,
                        estimated_remaining_sec=remaining,
                        status="Saving frames"
                    ))

            # Write info file
            info_path = os.path.join(base_dir, "info.txt")
            with open(info_path, 'w') as f:
                f.write(f"EOSIM Studio Export\n")
                f.write(f"Frames: {total_frames}\n")
                f.write(f"FPS: {fps}\n")
                f.write(f"Duration: {total_frames/fps:.2f}s\n")
                f.write(f"\nTo convert to video with ffmpeg:\n")
                f.write(f"ffmpeg -framerate {fps} -i frame_%06d.png -c:v libx264 output.mp4\n")

        except Exception as e:
            print(f"Export error: {e}")
            return False

        return not self.cancel_requested

    def _save_png(self, path: str, image: np.ndarray):
        """Save image as PNG without external dependencies."""
        import struct
        import zlib

        height, width = image.shape[:2]

        def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
            chunk = chunk_type + data
            crc = zlib.crc32(chunk) & 0xffffffff
            return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc)

        # PNG signature
        signature = b'\x89PNG\r\n\x1a\n'

        # IHDR chunk
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        ihdr = make_chunk(b'IHDR', ihdr_data)

        # IDAT chunk (image data)
        raw_data = b''
        for y in range(height):
            raw_data += b'\x00'  # Filter type: None
            for x in range(width):
                r, g, b = image[y, x, :3]
                raw_data += bytes([r, g, b])

        compressed = zlib.compress(raw_data, 9)
        idat = make_chunk(b'IDAT', compressed)

        # IEND chunk
        iend = make_chunk(b'IEND', b'')

        with open(path, 'wb') as f:
            f.write(signature + ihdr + idat + iend)

    def cancel(self):
        """Request cancellation of export."""
        self.cancel_requested = True


def export_video(project, output_path: str,
                progress_callback: Optional[Callable[[ExportProgress], None]] = None) -> bool:
    """Convenience function to export a project to video.

    Args:
        project: EOSIM Studio project
        output_path: Path to output video file
        progress_callback: Optional progress callback

    Returns:
        True if export succeeded
    """
    from eosim.studio.renderer import Renderer

    renderer = Renderer(project.scene, project.camera)
    exporter = VideoExporter(renderer, project.render_settings)

    return exporter.export(
        output_path,
        duration_sec=project.duration,
        fps=project.fps,
        progress_callback=progress_callback
    )
