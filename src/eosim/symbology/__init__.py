"""
EOSIM Symbology Overlay Module.

Provides MIL-STD-style HUD symbology overlays for sensor imagery,
including targeting reticles, track gates, compass heading tape,
pitch ladder, and status displays.

Example 1: Add crosshair reticle to sensor image
    >>> from eosim.symbology import SymbologyRenderer
    >>> import numpy as np
    >>> image = np.zeros((480, 640), dtype=np.uint8)
    >>> renderer = SymbologyRenderer()
    >>> overlay = renderer.render(image)

Example 2: Full HUD with platform state
    >>> from eosim.symbology import (
    ...     SymbologyRenderer, PlatformState, SensorStatus
    ... )
    >>> platform = PlatformState(heading_deg=270, pitch_deg=5, altitude_m=5000)
    >>> sensor = SensorStatus(mode="WHOT", fov_deg=3.0, range_m=4500)
    >>> overlay = renderer.render(image, platform=platform, sensor=sensor)

Example 3: Display tracked targets with threat classification
    >>> from eosim.symbology import TrackInfo, ThreatLevel
    >>> tracks = [
    ...     TrackInfo(x=300, y=200, track_id=1, threat=ThreatLevel.HOSTILE,
    ...              range_m=3500),
    ...     TrackInfo(x=150, y=350, track_id=2, threat=ThreatLevel.FRIENDLY,
    ...              range_m=8000),
    ... ]
    >>> overlay = renderer.render(image, tracks=tracks)
"""

from eosim.symbology.symbology import (
    ReticleType,
    ThreatLevel,
    SymbologyConfig,
    TrackInfo,
    PlatformState,
    SensorStatus,
    SymbologyRenderer,
    create_symbology_renderer,
)

__all__ = [
    "ReticleType",
    "ThreatLevel",
    "SymbologyConfig",
    "TrackInfo",
    "PlatformState",
    "SensorStatus",
    "SymbologyRenderer",
    "create_symbology_renderer",
]
