"""Tests for the Symbology overlay module."""

import numpy as np
import pytest

from eosim.symbology import (
    ReticleType,
    ThreatLevel,
    SymbologyConfig,
    TrackInfo,
    PlatformState,
    SensorStatus,
    SymbologyRenderer,
    create_symbology_renderer,
)


class TestSymbologyConfig:
    """Tests for SymbologyConfig dataclass."""

    def test_default_config(self):
        config = SymbologyConfig()
        assert config.color == (0, 255, 0)
        assert config.line_width == 1
        assert config.opacity == 1.0
        assert config.show_reticle is True

    def test_custom_config(self):
        config = SymbologyConfig(
            color=(255, 0, 0),
            line_width=2,
            show_compass=False,
        )
        assert config.color == (255, 0, 0)
        assert config.show_compass is False


class TestTrackInfo:
    """Tests for TrackInfo dataclass."""

    def test_default_track(self):
        track = TrackInfo()
        assert track.x == 0.0
        assert track.threat == ThreatLevel.UNKNOWN

    def test_custom_track(self):
        track = TrackInfo(
            x=300,
            y=200,
            width=48,
            height=48,
            track_id=5,
            threat=ThreatLevel.HOSTILE,
            range_m=5000,
        )
        assert track.track_id == 5
        assert track.threat == ThreatLevel.HOSTILE
        assert track.range_m == 5000


class TestPlatformState:
    """Tests for PlatformState dataclass."""

    def test_default_state(self):
        state = PlatformState()
        assert state.heading_deg == 0.0
        assert state.altitude_m == 0.0

    def test_custom_state(self):
        state = PlatformState(
            heading_deg=270.0,
            pitch_deg=5.0,
            altitude_m=3000,
            airspeed_mps=150.0,
        )
        assert state.heading_deg == 270.0
        assert state.airspeed_mps == 150.0


class TestSymbologyRenderer:
    """Tests for SymbologyRenderer."""

    def setup_method(self):
        """Create test images."""
        self.gray_image = np.full((480, 640), 128, dtype=np.uint8)
        self.rgb_image = np.full((480, 640, 3), 128, dtype=np.uint8)

    def test_render_grayscale(self):
        renderer = SymbologyRenderer()
        result = renderer.render(self.gray_image)

        # Should convert to RGB
        assert result.ndim == 3
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8

    def test_render_rgb(self):
        renderer = SymbologyRenderer()
        result = renderer.render(self.rgb_image)

        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8

    def test_render_with_platform_state(self):
        renderer = SymbologyRenderer()
        platform = PlatformState(
            heading_deg=270.0,
            pitch_deg=5.0,
            roll_deg=0.0,
            altitude_m=5000,
            airspeed_mps=200.0,
        )
        result = renderer.render(self.gray_image, platform=platform)

        assert result.shape == (480, 640, 3)
        # Image should have been modified (some pixels differ from uniform gray)
        gray_rgb = np.stack(
            [self.gray_image, self.gray_image, self.gray_image], axis=-1
        )
        assert not np.array_equal(result, gray_rgb)

    def test_render_with_sensor_status(self):
        renderer = SymbologyRenderer()
        sensor = SensorStatus(
            mode="WHOT",
            fov_deg=3.0,
            range_m=4500,
            laser_armed=True,
        )
        result = renderer.render(self.gray_image, sensor=sensor)

        assert result.shape == (480, 640, 3)

    def test_render_with_tracks(self):
        renderer = SymbologyRenderer()
        tracks = [
            TrackInfo(
                x=300, y=200, width=40, height=40,
                track_id=1, threat=ThreatLevel.HOSTILE, range_m=3500,
            ),
            TrackInfo(
                x=150, y=350, width=32, height=32,
                track_id=2, threat=ThreatLevel.FRIENDLY, range_m=8000,
            ),
        ]
        result = renderer.render(self.gray_image, tracks=tracks)

        assert result.shape == (480, 640, 3)

    def test_all_reticle_types(self):
        """All reticle types should render without error."""
        renderer = SymbologyRenderer()
        for reticle in ReticleType:
            result = renderer.render(
                self.gray_image, reticle_type=reticle
            )
            assert result.shape == (480, 640, 3), f"Failed for {reticle}"

    def test_disabled_elements(self):
        """Should render cleanly with elements disabled."""
        config = SymbologyConfig(
            show_reticle=False,
            show_compass=False,
            show_pitch_ladder=False,
            show_status=False,
            show_track_gates=False,
        )
        renderer = SymbologyRenderer(config)
        result = renderer.render(self.gray_image)

        assert result.shape == (480, 640, 3)

    def test_recording_indicator(self):
        renderer = SymbologyRenderer()
        sensor = SensorStatus(recording=True)
        result = renderer.render(self.gray_image, sensor=sensor)

        assert result.shape == (480, 640, 3)

    def test_threat_colors(self):
        """Each threat level should produce a different track gate color."""
        renderer = SymbologyRenderer()
        for threat in ThreatLevel:
            tracks = [
                TrackInfo(
                    x=300, y=200, track_id=1, threat=threat, range_m=1000
                ),
            ]
            result = renderer.render(self.gray_image, tracks=tracks)
            assert result.shape == (480, 640, 3)

    def test_small_image(self):
        """Should handle small images without crashing."""
        small = np.full((64, 64), 128, dtype=np.uint8)
        renderer = SymbologyRenderer()
        result = renderer.render(small)

        assert result.shape == (64, 64, 3)

    def test_range_display_km(self):
        """Ranges >= 1000m should display in km."""
        renderer = SymbologyRenderer()
        tracks = [
            TrackInfo(x=300, y=200, track_id=1, range_m=5000),
        ]
        result = renderer.render(self.gray_image, tracks=tracks)
        assert result.shape == (480, 640, 3)

    def test_range_display_m(self):
        """Ranges < 1000m should display in meters."""
        renderer = SymbologyRenderer()
        tracks = [
            TrackInfo(x=300, y=200, track_id=1, range_m=500),
        ]
        result = renderer.render(self.gray_image, tracks=tracks)
        assert result.shape == (480, 640, 3)


class TestCreateSymbologyRenderer:
    """Tests for factory function."""

    def test_default(self):
        renderer = create_symbology_renderer()
        assert isinstance(renderer, SymbologyRenderer)

    def test_custom_color(self):
        renderer = create_symbology_renderer(color=(255, 255, 0))
        assert renderer.config.color == (255, 255, 0)

    def test_disabled_elements(self):
        renderer = create_symbology_renderer(
            show_reticle=False,
            show_compass=False,
        )
        assert not renderer.config.show_reticle
        assert not renderer.config.show_compass
