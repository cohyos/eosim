"""
Thermal scene estimation from visible video (Option 3).

Estimates thermal/temperature maps from visible video by:
1. Segmenting objects in the scene
2. Classifying objects into semantic categories
3. Assigning temperatures based on class and context
"""

from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from eosim.video.config import (
    ThermalEstimateConfig,
    TemplateConfig,
    ObjectThermalProperties,
    ObjectInstance,
    HotSpotConfig,
    DEFAULT_THERMAL_PROPERTIES,
    SegmentationMethod,
)
from eosim.video.segmentation import Segmenter, ObjectClassifier, segment_frame
from eosim.pipeline.simulation import SceneInput


class ThermalEstimator:
    """Estimates thermal scenes from visible video frames.

    Combines segmentation, classification, and temperature assignment
    to generate realistic thermal scenes from visible imagery.

    Example:
        >>> config = ThermalEstimateConfig(
        ...     ambient_temperature_k=295.0,
        ...     time_of_day="day",
        ... )
        >>> estimator = ThermalEstimator(config)
        >>> temp_map, emissivity_map = estimator.estimate_frame(visible_frame)
        >>> scene = estimator.to_scene_input(visible_frame)
    """

    def __init__(
        self,
        config: Optional[ThermalEstimateConfig] = None,
    ) -> None:
        """Initialize thermal estimator.

        Args:
            config: Thermal estimation configuration
        """
        self.config = config or ThermalEstimateConfig()

        # Initialize segmenter
        self._segmenter = Segmenter(
            method=self.config.segmentation_method,
            min_object_size=100,
        )

        # Initialize classifier
        self._classifier = ObjectClassifier()

        # Random generator for temperature variation
        self._rng = np.random.default_rng()

    def estimate_frame(
        self,
        frame: NDArray,
        segmentation_mask: Optional[NDArray] = None,
        objects: Optional[list[ObjectInstance]] = None,
    ) -> tuple[NDArray, NDArray]:
        """Estimate thermal scene from visible frame.

        Args:
            frame: Visible video frame (H, W, C) or (H, W)
            segmentation_mask: Pre-computed segmentation (optional)
            objects: Pre-detected objects (optional)

        Returns:
            Tuple of (temperature_map, emissivity_map)
            - temperature_map: Temperature in Kelvin (H, W)
            - emissivity_map: Emissivity values 0-1 (H, W)
        """
        h, w = frame.shape[:2]

        # Segment frame if not provided
        if segmentation_mask is None or objects is None:
            segmentation_mask, objects = self._segmenter.segment(frame)
            objects = self._classifier.classify(objects, frame)

        # Initialize with background
        temperature_map = np.full(
            (h, w),
            self.config.ambient_temperature_k,
            dtype=np.float64,
        )
        emissivity_map = np.full((h, w), 0.95, dtype=np.float64)

        # Apply class-based temperatures
        for obj in objects:
            self._apply_object_thermal(
                temperature_map,
                emissivity_map,
                obj,
                frame,
            )

        # Apply environmental adjustments
        if self.config.solar_loading:
            temperature_map = self._apply_solar_loading(temperature_map, frame)

        # Smooth boundaries if requested
        if self.config.smooth_boundaries:
            temperature_map = self._smooth_boundaries(
                temperature_map,
                segmentation_mask,
                self.config.boundary_blur_pixels,
            )

        return temperature_map, emissivity_map

    def _apply_object_thermal(
        self,
        temperature_map: NDArray,
        emissivity_map: NDArray,
        obj: ObjectInstance,
        frame: NDArray,
    ) -> None:
        """Apply thermal properties to an object region."""
        # Get thermal properties for this class
        class_name = obj.class_name.lower()
        if class_name in self.config.class_temperatures:
            props = self.config.class_temperatures[class_name]
        elif class_name in DEFAULT_THERMAL_PROPERTIES:
            props = DEFAULT_THERMAL_PROPERTIES[class_name]
        else:
            props = DEFAULT_THERMAL_PROPERTIES["background"]

        # Get object mask
        if obj.mask is not None:
            mask = obj.mask
        else:
            # Create mask from bounding box
            y0, x0, y1, x1 = obj.bounding_box
            mask = np.zeros(temperature_map.shape, dtype=bool)
            mask[y0:y1, x0:x1] = True

        # Base temperature with variation
        base_temp = props.temperature_k + self._rng.normal(0, props.temperature_std_k)

        # Apply base temperature
        temperature_map[mask] = base_temp
        emissivity_map[mask] = props.emissivity + self._rng.normal(0, props.emissivity_std)
        emissivity_map[mask] = np.clip(emissivity_map[mask], 0.01, 1.0)

        # Apply hot spots
        if props.hot_spots:
            self._apply_hot_spots(temperature_map, obj, props.hot_spots, base_temp)

        # Brightness correlation (optional)
        if self.config.estimation_method in ("brightness_correlation", "hybrid"):
            self._apply_brightness_correlation(
                temperature_map, frame, mask, base_temp
            )

    def _apply_hot_spots(
        self,
        temperature_map: NDArray,
        obj: ObjectInstance,
        hot_spots: list[HotSpotConfig],
        base_temp: float,
    ) -> None:
        """Apply hot spots within an object."""
        y0, x0, y1, x1 = obj.bounding_box
        obj_h = y1 - y0
        obj_w = x1 - x0

        for spot in hot_spots:
            # Compute hot spot position
            rel_y, rel_x = spot.relative_position
            spot_y = int(y0 + rel_y * obj_h)
            spot_x = int(x0 + rel_x * obj_w)

            # Compute hot spot size
            spot_radius = int(spot.relative_size * min(obj_h, obj_w) / 2)
            spot_radius = max(1, spot_radius)

            # Create hot spot mask
            h, w = temperature_map.shape
            yy, xx = np.ogrid[:h, :w]

            if spot.shape == "circle":
                spot_mask = (yy - spot_y)**2 + (xx - spot_x)**2 <= spot_radius**2
            elif spot.shape == "ellipse":
                spot_mask = (
                    ((yy - spot_y) / spot_radius)**2 +
                    ((xx - spot_x) / (spot_radius * 1.5))**2
                ) <= 1
            else:  # rectangle
                spot_mask = (
                    (np.abs(yy - spot_y) <= spot_radius) &
                    (np.abs(xx - spot_x) <= spot_radius)
                )

            # Apply hot spot temperature
            temperature_map[spot_mask] = base_temp + spot.temperature_delta_k

    def _apply_brightness_correlation(
        self,
        temperature_map: NDArray,
        frame: NDArray,
        mask: NDArray,
        base_temp: float,
    ) -> None:
        """Correlate temperature with visible brightness within object."""
        # Get brightness in object region
        if frame.ndim == 3:
            brightness = (
                0.299 * frame[:, :, 0] +
                0.587 * frame[:, :, 1] +
                0.114 * frame[:, :, 2]
            )
        else:
            brightness = frame.astype(np.float64)

        # Normalize brightness within object
        obj_brightness = brightness[mask]
        if len(obj_brightness) == 0:
            return

        b_min, b_max = obj_brightness.min(), obj_brightness.max()
        if b_max > b_min:
            normalized = (brightness - b_min) / (b_max - b_min)
        else:
            return

        # Map brightness to temperature variation (±5K)
        temp_variation = (normalized - 0.5) * 10.0

        # Apply only within object
        temperature_map[mask] += temp_variation[mask]

    def _apply_solar_loading(
        self,
        temperature_map: NDArray,
        frame: NDArray,
    ) -> NDArray:
        """Adjust temperatures for solar loading effects."""
        # Simple model: brighter visible = more solar heating
        if frame.ndim == 3:
            brightness = np.mean(frame, axis=2)
        else:
            brightness = frame.astype(np.float64)

        # Normalize brightness
        b_norm = brightness / max(brightness.max(), 1)

        # Time of day scaling
        time_scale = {
            "day": 1.0,
            "dawn": 0.3,
            "dusk": 0.3,
            "night": 0.0,
        }
        scale = time_scale.get(self.config.time_of_day, 0.5)

        # Add solar heating (up to 15K for bright sunlit surfaces)
        solar_heating = b_norm * 15.0 * scale

        return temperature_map + solar_heating

    def _smooth_boundaries(
        self,
        temperature_map: NDArray,
        segmentation_mask: NDArray,
        blur_size: int,
    ) -> NDArray:
        """Smooth temperature transitions at object boundaries."""
        try:
            from scipy.ndimage import gaussian_filter, binary_dilation

            # Find boundary pixels
            dilated = binary_dilation(segmentation_mask > 0, iterations=blur_size)
            eroded = binary_dilation(segmentation_mask == 0, iterations=blur_size)
            boundary = dilated & eroded

            # Apply local smoothing at boundaries
            smoothed = gaussian_filter(temperature_map, sigma=blur_size)

            # Blend at boundaries
            result = temperature_map.copy()
            result[boundary] = smoothed[boundary]

            return result

        except ImportError:
            return temperature_map

    def to_scene_input(
        self,
        frame: NDArray,
        segmentation_mask: Optional[NDArray] = None,
        objects: Optional[list[ObjectInstance]] = None,
    ) -> SceneInput:
        """Convert frame to SceneInput for simulation pipeline.

        Args:
            frame: Visible video frame
            segmentation_mask: Pre-computed segmentation (optional)
            objects: Pre-detected objects (optional)

        Returns:
            SceneInput with temperature and emissivity maps
        """
        temperature_map, emissivity_map = self.estimate_frame(
            frame, segmentation_mask, objects
        )

        return SceneInput(
            temperature_map=temperature_map,
            emissivity_map=emissivity_map,
            background_temperature=self.config.ambient_temperature_k - 20,
        )


class TemplateMapper:
    """Maps video frames to thermal scenes using template approach (Option 1).

    Uses video as geometry/motion reference with user-defined
    thermal property assignments.
    """

    def __init__(
        self,
        config: Optional[TemplateConfig] = None,
    ) -> None:
        """Initialize template mapper.

        Args:
            config: Template configuration
        """
        self.config = config or TemplateConfig()

        self._segmenter = Segmenter(
            method=self.config.segmentation_method,
            min_object_size=self.config.min_object_size,
        )
        self._classifier = ObjectClassifier()
        self._rng = np.random.default_rng()

    def map_frame(
        self,
        frame: NDArray,
        previous_frame: Optional[NDArray] = None,
    ) -> tuple[NDArray, NDArray, Optional[NDArray]]:
        """Map visible frame to thermal scene.

        Args:
            frame: Current visible frame
            previous_frame: Previous frame for motion estimation

        Returns:
            Tuple of (temperature_map, emissivity_map, motion_vectors)
        """
        h, w = frame.shape[:2]

        # Segment and classify
        mask, objects = self._segmenter.segment(frame)
        objects = self._classifier.classify(objects, frame)

        # Initialize maps
        temperature_map = np.full(
            (h, w),
            self.config.background_temperature_k,
            dtype=np.float64,
        )
        emissivity_map = np.full(
            (h, w),
            self.config.background_emissivity,
            dtype=np.float64,
        )

        # Apply object properties
        for obj in objects:
            class_name = obj.class_name.lower()
            if class_name in self.config.class_properties:
                props = self.config.class_properties[class_name]
            else:
                props = ObjectThermalProperties(
                    temperature_k=self.config.background_temperature_k + 10,
                    emissivity=0.9,
                )

            # Apply to object region
            if obj.mask is not None:
                obj_mask = obj.mask
            else:
                y0, x0, y1, x1 = obj.bounding_box
                obj_mask = np.zeros((h, w), dtype=bool)
                obj_mask[y0:y1, x0:x1] = True

            # Temperature with variation
            temp = props.temperature_k + self._rng.normal(0, props.temperature_std_k)
            temperature_map[obj_mask] = temp
            emissivity_map[obj_mask] = props.emissivity

            # Hot spots
            if props.hot_spots:
                for spot in props.hot_spots:
                    self._apply_hot_spot(temperature_map, obj, spot, temp)

        # Compute motion vectors if requested
        motion_vectors = None
        if self.config.use_optical_flow and previous_frame is not None:
            from eosim.video.reader import compute_optical_flow
            try:
                motion_vectors = compute_optical_flow(previous_frame, frame)
                motion_vectors = motion_vectors * self.config.motion_blur_scale
            except Exception:
                pass

        return temperature_map, emissivity_map, motion_vectors

    def _apply_hot_spot(
        self,
        temperature_map: NDArray,
        obj: ObjectInstance,
        spot: HotSpotConfig,
        base_temp: float,
    ) -> None:
        """Apply a hot spot to the temperature map."""
        y0, x0, y1, x1 = obj.bounding_box
        obj_h = y1 - y0
        obj_w = x1 - x0

        rel_y, rel_x = spot.relative_position
        spot_y = int(y0 + rel_y * obj_h)
        spot_x = int(x0 + rel_x * obj_w)
        spot_r = int(spot.relative_size * min(obj_h, obj_w) / 2)

        h, w = temperature_map.shape
        yy, xx = np.ogrid[:h, :w]
        spot_mask = (yy - spot_y)**2 + (xx - spot_x)**2 <= spot_r**2

        temperature_map[spot_mask] = base_temp + spot.temperature_delta_k

    def to_scene_input(
        self,
        frame: NDArray,
        previous_frame: Optional[NDArray] = None,
    ) -> SceneInput:
        """Convert frame to SceneInput.

        Args:
            frame: Current visible frame
            previous_frame: Previous frame for motion

        Returns:
            SceneInput for simulation
        """
        temp_map, emis_map, _ = self.map_frame(frame, previous_frame)

        return SceneInput(
            temperature_map=temp_map,
            emissivity_map=emis_map,
            background_temperature=self.config.background_temperature_k - 30,
        )


def estimate_thermal_from_visible(
    frame: NDArray,
    ambient_temperature_k: float = 290.0,
    class_properties: Optional[dict[str, ObjectThermalProperties]] = None,
) -> tuple[NDArray, NDArray]:
    """Convenience function for thermal estimation.

    Args:
        frame: Visible video frame
        ambient_temperature_k: Ambient temperature
        class_properties: Custom thermal properties

    Returns:
        (temperature_map, emissivity_map)
    """
    config = ThermalEstimateConfig(
        ambient_temperature_k=ambient_temperature_k,
    )
    if class_properties:
        config.class_temperatures.update(class_properties)

    estimator = ThermalEstimator(config)
    return estimator.estimate_frame(frame)
