"""
Image segmentation for video-to-simulation conversion.

Provides various segmentation methods for identifying objects in video frames,
used by both Template mode (Option 1) and Thermal Estimate mode (Option 3).
"""

from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from eosim.video.config import (
    SegmentationMethod,
    ObjectInstance,
    FrameData,
)


class Segmenter:
    """Base class for image segmentation.

    Identifies objects in video frames and produces segmentation masks
    that can be used for thermal property assignment.
    """

    def __init__(
        self,
        method: SegmentationMethod = SegmentationMethod.THRESHOLD,
        min_object_size: int = 100,
    ) -> None:
        """Initialize segmenter.

        Args:
            method: Segmentation method to use
            min_object_size: Minimum object size in pixels
        """
        self.method = method
        self.min_object_size = min_object_size

    def segment(self, frame: NDArray) -> tuple[NDArray, list[ObjectInstance]]:
        """Segment frame into objects.

        Args:
            frame: Input frame (H, W, C) or (H, W)

        Returns:
            Tuple of (segmentation_mask, object_list)
            - segmentation_mask: (H, W) array with object IDs (0=background)
            - object_list: List of detected ObjectInstance
        """
        if self.method == SegmentationMethod.THRESHOLD:
            return self._threshold_segment(frame)
        elif self.method == SegmentationMethod.EDGE:
            return self._edge_segment(frame)
        elif self.method == SegmentationMethod.COLOR:
            return self._color_segment(frame)
        elif self.method == SegmentationMethod.CONTOUR:
            return self._contour_segment(frame)
        else:
            raise ValueError(f"Unsupported segmentation method: {self.method}")

    def _to_grayscale(self, frame: NDArray) -> NDArray:
        """Convert frame to grayscale."""
        if frame.ndim == 2:
            return frame
        # Luminance conversion
        return (
            0.299 * frame[:, :, 0] +
            0.587 * frame[:, :, 1] +
            0.114 * frame[:, :, 2]
        ).astype(np.uint8)

    def _threshold_segment(
        self,
        frame: NDArray,
        threshold: int = 128,
    ) -> tuple[NDArray, list[ObjectInstance]]:
        """Simple threshold-based segmentation.

        Objects are identified as connected regions above threshold.
        """
        gray = self._to_grayscale(frame)

        # Apply threshold
        binary = (gray > threshold).astype(np.uint8)

        # Label connected components
        mask, objects = self._label_components(binary, frame)

        return mask, objects

    def _edge_segment(self, frame: NDArray) -> tuple[NDArray, list[ObjectInstance]]:
        """Edge-based segmentation using Canny edges and contours."""
        try:
            import cv2
        except ImportError:
            # Fallback to threshold
            return self._threshold_segment(frame)

        gray = self._to_grayscale(frame)

        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Fill enclosed regions
        filled = self._flood_fill_edges(edges)

        # Label components
        mask, objects = self._label_components(filled, frame)

        return mask, objects

    def _color_segment(self, frame: NDArray) -> tuple[NDArray, list[ObjectInstance]]:
        """Color-based segmentation using K-means clustering."""
        try:
            import cv2
        except ImportError:
            return self._threshold_segment(frame)

        if frame.ndim == 2:
            return self._threshold_segment(frame)

        # Reshape for k-means
        h, w = frame.shape[:2]
        pixels = frame.reshape(-1, 3).astype(np.float32)

        # K-means clustering
        k = 5  # Number of clusters
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, _ = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Reshape labels to image
        labels_img = labels.reshape(h, w).astype(np.uint8)

        # Convert to binary (foreground vs background)
        # Assume brightest cluster is background (sky) for outdoor scenes
        cluster_brightness = [
            np.mean(frame[labels_img == i]) for i in range(k)
        ]
        background_cluster = np.argmax(cluster_brightness)
        binary = (labels_img != background_cluster).astype(np.uint8)

        mask, objects = self._label_components(binary, frame)

        return mask, objects

    def _contour_segment(self, frame: NDArray) -> tuple[NDArray, list[ObjectInstance]]:
        """Contour-based segmentation."""
        try:
            import cv2
        except ImportError:
            return self._threshold_segment(frame)

        gray = self._to_grayscale(frame)

        # Adaptive thresholding for better local contrast
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        mask, objects = self._label_components(binary, frame)

        return mask, objects

    def _label_components(
        self,
        binary: NDArray,
        original_frame: NDArray,
    ) -> tuple[NDArray, list[ObjectInstance]]:
        """Label connected components and create object instances."""
        try:
            import cv2
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
        except ImportError:
            # Simple fallback using scipy if available
            try:
                from scipy import ndimage
                labels, num_labels = ndimage.label(binary)
                # Compute stats manually
                objects = []
                for i in range(1, num_labels + 1):
                    mask = labels == i
                    if mask.sum() < self.min_object_size:
                        labels[mask] = 0
                        continue
                    ys, xs = np.where(mask)
                    obj = ObjectInstance(
                        object_id=i,
                        class_name="object",
                        class_id=1,
                        bounding_box=(ys.min(), xs.min(), ys.max(), xs.max()),
                        mask=mask,
                        centroid=(ys.mean(), xs.mean()),
                        area_pixels=mask.sum(),
                    )
                    objects.append(obj)
                return labels.astype(np.int32), objects
            except ImportError:
                # No scipy either, return simple threshold
                labels = binary.astype(np.int32)
                return labels, []

        objects = []
        # Start from 1 (0 is background)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            if area < self.min_object_size:
                # Remove small objects
                labels[labels == i] = 0
                continue

            cx, cy = centroids[i]
            mask = labels == i

            obj = ObjectInstance(
                object_id=i,
                class_name="object",  # Generic, can be classified later
                class_id=1,
                bounding_box=(y, x, y + h, x + w),
                mask=mask,
                centroid=(cy, cx),
                area_pixels=area,
            )
            objects.append(obj)

        return labels.astype(np.int32), objects

    def _flood_fill_edges(self, edges: NDArray) -> NDArray:
        """Fill enclosed regions from edge map."""
        try:
            import cv2
        except ImportError:
            return edges

        h, w = edges.shape

        # Create padded image for flood fill
        padded = np.zeros((h + 2, w + 2), np.uint8)
        padded[1:-1, 1:-1] = edges

        # Flood fill from corners (assumes corners are background)
        filled = padded.copy()
        cv2.floodFill(filled, None, (0, 0), 255)

        # Invert to get foreground
        filled = 255 - filled[1:-1, 1:-1]

        # Combine with original edges
        result = cv2.bitwise_or(filled, edges)

        return result


class ObjectClassifier:
    """Classifies detected objects into semantic categories.

    Uses simple heuristics or ML models to assign class labels
    to segmented objects.
    """

    # Default class mapping based on object properties
    DEFAULT_CLASSES = {
        "person": {"min_aspect": 1.5, "max_aspect": 4.0, "min_area": 500, "max_area": 50000},
        "vehicle": {"min_aspect": 0.3, "max_aspect": 2.0, "min_area": 2000, "max_area": 200000},
        "building": {"min_aspect": 0.2, "max_aspect": 5.0, "min_area": 10000},
        "small_object": {"max_area": 500},
    }

    def __init__(
        self,
        class_rules: Optional[dict] = None,
        use_ml: bool = False,
        model_path: Optional[str] = None,
    ) -> None:
        """Initialize classifier.

        Args:
            class_rules: Custom classification rules
            use_ml: Use ML model for classification
            model_path: Path to ML model
        """
        self.class_rules = class_rules or self.DEFAULT_CLASSES
        self.use_ml = use_ml
        self.model_path = model_path
        self._model = None

    def classify(
        self,
        objects: list[ObjectInstance],
        frame: Optional[NDArray] = None,
    ) -> list[ObjectInstance]:
        """Classify objects into semantic categories.

        Args:
            objects: List of detected objects
            frame: Original frame (for ML classification)

        Returns:
            Objects with updated class_name fields
        """
        if self.use_ml and self._model is not None:
            return self._ml_classify(objects, frame)

        # Rule-based classification
        for obj in objects:
            obj.class_name = self._rule_classify(obj)

        return objects

    def _rule_classify(self, obj: ObjectInstance) -> str:
        """Classify object using simple rules."""
        y0, x0, y1, x1 = obj.bounding_box
        h = y1 - y0
        w = x1 - x0

        if h == 0 or w == 0:
            return "unknown"

        aspect_ratio = h / w
        area = obj.area_pixels

        # Check each class rule
        for class_name, rules in self.class_rules.items():
            matches = True

            if "min_aspect" in rules and aspect_ratio < rules["min_aspect"]:
                matches = False
            if "max_aspect" in rules and aspect_ratio > rules["max_aspect"]:
                matches = False
            if "min_area" in rules and area < rules["min_area"]:
                matches = False
            if "max_area" in rules and area > rules["max_area"]:
                matches = False

            if matches:
                return class_name

        return "unknown"

    def _ml_classify(
        self,
        objects: list[ObjectInstance],
        frame: Optional[NDArray],
    ) -> list[ObjectInstance]:
        """Classify using ML model."""
        # Placeholder for ML classification
        # Would integrate with YOLO, etc.
        return self._rule_classify_all(objects)

    def _rule_classify_all(self, objects: list[ObjectInstance]) -> list[ObjectInstance]:
        """Apply rule classification to all objects."""
        for obj in objects:
            obj.class_name = self._rule_classify(obj)
        return objects


class ObjectTracker:
    """Tracks objects across frames for temporal consistency.

    Maintains object IDs across frames and computes velocities
    for motion blur computation.
    """

    def __init__(
        self,
        max_distance: float = 100.0,
        max_frames_missing: int = 5,
    ) -> None:
        """Initialize tracker.

        Args:
            max_distance: Maximum centroid distance for matching
            max_frames_missing: Frames before object is removed
        """
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing

        self._tracked_objects: dict[int, ObjectInstance] = {}
        self._last_positions: dict[int, tuple[float, float]] = {}
        self._frames_missing: dict[int, int] = {}
        self._next_id = 1

    def update(
        self,
        objects: list[ObjectInstance],
        frame_index: int,
    ) -> list[ObjectInstance]:
        """Update tracking with new detections.

        Args:
            objects: Detected objects in current frame
            frame_index: Current frame index

        Returns:
            Objects with consistent IDs and velocities
        """
        # Match new detections to tracked objects
        matched = self._match_objects(objects)

        # Update tracked objects
        new_tracked = {}
        for obj in matched:
            oid = obj.object_id

            # Compute velocity if we have previous position
            if oid in self._last_positions:
                last_y, last_x = self._last_positions[oid]
                curr_y, curr_x = obj.centroid
                obj.velocity = (curr_y - last_y, curr_x - last_x)

            new_tracked[oid] = obj
            self._last_positions[oid] = obj.centroid

        # Update missing counts
        for oid in list(self._frames_missing.keys()):
            if oid not in new_tracked:
                self._frames_missing[oid] += 1
                if self._frames_missing[oid] > self.max_frames_missing:
                    del self._frames_missing[oid]
                    if oid in self._last_positions:
                        del self._last_positions[oid]
            else:
                self._frames_missing[oid] = 0

        self._tracked_objects = new_tracked

        return list(new_tracked.values())

    def _match_objects(self, objects: list[ObjectInstance]) -> list[ObjectInstance]:
        """Match detections to tracked objects."""
        if not self._tracked_objects:
            # First frame - assign new IDs
            for obj in objects:
                obj.object_id = self._next_id
                self._frames_missing[self._next_id] = 0
                self._next_id += 1
            return objects

        # Compute distance matrix
        tracked_list = list(self._tracked_objects.values())
        distances = np.zeros((len(objects), len(tracked_list)))

        for i, obj in enumerate(objects):
            for j, tracked in enumerate(tracked_list):
                dy = obj.centroid[0] - tracked.centroid[0]
                dx = obj.centroid[1] - tracked.centroid[1]
                distances[i, j] = np.sqrt(dy**2 + dx**2)

        # Greedy matching
        matched_detections = set()
        matched_tracks = set()

        for _ in range(min(len(objects), len(tracked_list))):
            # Find minimum distance
            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            i, j = min_idx

            if i in matched_detections or j in matched_tracks:
                distances[i, j] = float('inf')
                continue

            if distances[i, j] > self.max_distance:
                break

            # Match found
            objects[i].object_id = tracked_list[j].object_id
            matched_detections.add(i)
            matched_tracks.add(j)
            distances[i, j] = float('inf')

        # Assign new IDs to unmatched detections
        for i, obj in enumerate(objects):
            if i not in matched_detections:
                obj.object_id = self._next_id
                self._frames_missing[self._next_id] = 0
                self._next_id += 1

        return objects

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracked_objects = {}
        self._last_positions = {}
        self._frames_missing = {}
        self._next_id = 1


def segment_frame(
    frame: NDArray,
    method: Union[SegmentationMethod, str] = SegmentationMethod.THRESHOLD,
    min_object_size: int = 100,
    classify: bool = True,
) -> tuple[NDArray, list[ObjectInstance]]:
    """Convenience function for single-frame segmentation.

    Args:
        frame: Input frame
        method: Segmentation method
        min_object_size: Minimum object size
        classify: Whether to classify objects

    Returns:
        (segmentation_mask, objects)
    """
    if isinstance(method, str):
        method = SegmentationMethod(method)

    segmenter = Segmenter(method=method, min_object_size=min_object_size)
    mask, objects = segmenter.segment(frame)

    if classify:
        classifier = ObjectClassifier()
        objects = classifier.classify(objects, frame)

    return mask, objects
