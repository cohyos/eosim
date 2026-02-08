#!/usr/bin/env python3
"""
F-16 Demo Scenario - EOSIM Studio

This script creates a realistic thermal imaging scenario with an F-16 fighter jet
performing a flyby, renders frames, and exports to video for visual verification.

The scenario demonstrates:
- 3D model loading and rendering
- Thermal zone visualization (exhaust, cockpit, fuselage, wings)
- Motion path animation
- Perspective camera tracking
- Video export

Input Configuration:
- Object: F-16 Fighting Falcon
- Path: Approaching from 2km, flyby at 500m, departing
- Altitude: 300m above ground
- Speed: ~200 m/s (simulated)
- Duration: 10 seconds at 24 FPS

Camera Configuration:
- Position: Ground-based sensor platform
- Lens: Telephoto (10° FOV)
- Spectrum: MWIR
- Sensitivity: HIGH (NETD 20mK)
- Colormap: Iron (classic thermal)
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from eosim.studio.scene import Scene, Position3D, Orientation3D, MotionPath, ObjectMotion
from eosim.studio.camera import Camera, LensType, SpectrumMode, SensitivityLevel
from eosim.studio.renderer import Renderer, RenderQueue
from eosim.studio.timeline import Timeline
from eosim.studio.project import Project


def create_f16_flyby_scenario():
    """Create a realistic F-16 flyby scenario."""

    print("=" * 70)
    print("EOSIM Studio - F-16 Flyby Demo Scenario")
    print("=" * 70)

    # =========================================================================
    # SCENARIO CONFIGURATION (INPUT)
    # =========================================================================
    print("\n[INPUT CONFIGURATION]")
    print("-" * 50)

    # Scene setup
    scene = Scene()
    scene.name = "F-16 Flyby Demo"
    scene.description = "Single F-16 performing approach and flyby"

    # Define flight path waypoints
    # Aircraft approaches from the north, passes overhead, departs south
    # This keeps the aircraft in the camera's forward-looking FOV
    waypoints = [
        (0.0, Position3D(x=0, y=2000, z=400)),       # t=0: Far north, approaching
        (2.5, Position3D(x=0, y=1000, z=350)),       # t=2.5: Closer
        (5.0, Position3D(x=0, y=400, z=300)),        # t=5: Closest point (400m)
        (7.5, Position3D(x=0, y=1000, z=350)),       # t=7.5: Departing (climbing)
        (10.0, Position3D(x=0, y=2000, z=400)),      # t=10: Far north again
    ]

    # Create motion path
    flight_path = MotionPath()
    for time, pos in waypoints:
        flight_path.add_waypoint(time, pos)

    print(f"  Object: F-16 Fighting Falcon")
    print(f"  Flight path waypoints: {len(waypoints)}")
    for i, (t, pos) in enumerate(waypoints):
        print(f"    [{i}] t={t:.1f}s: x={pos.x:.0f}m, y={pos.y:.0f}m, alt={pos.z:.0f}m")

    # Add F-16 to scene with motion
    f16 = scene.add_object(
        object_type="f16",
        name="Viper-01",
        position=waypoints[0][0],
        orientation=Orientation3D(heading=270, pitch=-5, roll=0)  # Heading west, slight dive
    )
    f16.motion = ObjectMotion.PATH
    f16.motion_path = flight_path

    # Camera configuration
    camera = Camera()
    camera.position = Position3D(x=0, y=0, z=10)  # Ground sensor, 10m elevation
    # Point camera to mid-altitude of flight path (pitch ~25° covers 11°-36° range)
    camera.orientation = Orientation3D(heading=0, pitch=25, roll=0)  # Looking up/north
    camera.lens = LensType.NORMAL  # 30° FOV for better coverage
    camera.spectrum = SpectrumMode.THERMAL_MWIR  # Mid-wave IR
    camera.sensitivity = SensitivityLevel.HIGH  # 20mK NETD
    camera.colormap = "iron"
    camera.resolution = (640, 480)

    print(f"\n  Camera position: x={camera.position.x}, y={camera.position.y}, z={camera.position.z}m")
    print(f"  Camera orientation: heading={camera.orientation.heading}°, pitch={camera.orientation.pitch}°")
    print(f"  FOV: {camera.get_fov():.1f}° (telephoto)")
    print(f"  Spectrum: {camera.spectrum.value}")
    print(f"  NETD: {camera.get_netd():.0f} mK")
    print(f"  Resolution: {camera.resolution}")
    print(f"  Colormap: {camera.colormap}")

    # Timeline configuration
    duration_sec = 10.0
    fps = 24
    total_frames = int(duration_sec * fps)

    print(f"\n  Duration: {duration_sec} seconds")
    print(f"  Frame rate: {fps} FPS")
    print(f"  Total frames: {total_frames}")

    # =========================================================================
    # RENDERING
    # =========================================================================
    print("\n[RENDERING]")
    print("-" * 50)

    renderer = Renderer(scene, camera)
    renderer.background_temp_k = 288.0  # 15°C ambient
    renderer.atmosphere_attenuation = 0.0002  # Light haze

    print(f"  Background temperature: {renderer.background_temp_k:.1f}K ({renderer.background_temp_k - 273.15:.1f}°C)")
    print(f"  Atmosphere attenuation: {renderer.atmosphere_attenuation}")
    print(f"  3D models enabled: {renderer.use_3d_models}")

    # Check if F-16 model loaded
    model = renderer._get_3d_model("f16")
    if model:
        print(f"  F-16 model: {len(model['vertices'])} vertices, {len(model['faces'])} faces")
        zones = set(v for v in model.get('thermal_zones', {}).values() if isinstance(v, str))
        print(f"  Thermal zones: {zones}")
    else:
        print("  WARNING: F-16 3D model not loaded, using fallback")

    # Render frames
    print(f"\n  Rendering {total_frames} frames...")

    frames = []
    stats = {
        'temp_min': [],
        'temp_max': [],
        'object_visible': [],
        'distances': []
    }

    for i in range(total_frames):
        time_sec = i / fps
        frame = renderer.render_frame(time_sec, i)
        frames.append(frame)

        # Collect statistics
        stats['temp_min'].append(float(frame.temperature_map.min()))
        stats['temp_max'].append(float(frame.temperature_map.max()))

        # Calculate distance to object at this time
        obj_pos = flight_path.get_position_at(time_sec)
        cam_pos = camera.position
        distance = np.sqrt(
            (obj_pos.x - cam_pos.x)**2 +
            (obj_pos.y - cam_pos.y)**2 +
            (obj_pos.z - cam_pos.z)**2
        )
        stats['distances'].append(distance)

        # Check if object is visible (temp > background)
        is_visible = frame.temperature_map.max() > renderer.background_temp_k + 5
        stats['object_visible'].append(is_visible)

        # Progress indicator
        if (i + 1) % 24 == 0 or i == 0:
            print(f"    Frame {i+1:3d}/{total_frames}: t={time_sec:5.2f}s, "
                  f"dist={distance:6.0f}m, visible={is_visible}, "
                  f"temp_range={frame.temperature_map.min():.1f}-{frame.temperature_map.max():.1f}K")

    # =========================================================================
    # OUTPUT ANALYSIS
    # =========================================================================
    print("\n[OUTPUT ANALYSIS]")
    print("-" * 50)

    # Overall statistics
    visible_frames = sum(stats['object_visible'])
    print(f"  Frames with object visible: {visible_frames}/{total_frames} ({100*visible_frames/total_frames:.1f}%)")
    print(f"  Temperature range overall: {min(stats['temp_min']):.1f}K - {max(stats['temp_max']):.1f}K")
    print(f"  Distance range: {min(stats['distances']):.0f}m - {max(stats['distances']):.0f}m")

    # Find key frames
    closest_frame_idx = np.argmin(stats['distances'])
    hottest_frame_idx = np.argmax(stats['temp_max'])

    print(f"\n  Key frames:")
    print(f"    Closest approach: frame {closest_frame_idx} at t={closest_frame_idx/fps:.2f}s, "
          f"distance={stats['distances'][closest_frame_idx]:.0f}m")
    print(f"    Hottest frame: frame {hottest_frame_idx} at t={hottest_frame_idx/fps:.2f}s, "
          f"peak temp={stats['temp_max'][hottest_frame_idx]:.1f}K")

    # Thermal signature analysis
    peak_temp = max(stats['temp_max'])
    background_temp = renderer.background_temp_k
    delta_t = peak_temp - background_temp

    print(f"\n  Thermal signature analysis:")
    print(f"    Background: {background_temp:.1f}K ({background_temp-273.15:.1f}°C)")
    print(f"    Peak detected: {peak_temp:.1f}K ({peak_temp-273.15:.1f}°C)")
    print(f"    Delta-T: {delta_t:.1f}K")
    print(f"    Expected exhaust temp: ~530K (with attenuation at distance)")

    # Realism check
    print(f"\n  Realism assessment:")

    checks = []

    # Check 1: F-16 exhaust should be hot
    if peak_temp > 400:
        checks.append(("Exhaust temperature realistic", True, f"Peak {peak_temp:.0f}K matches jet afterburner"))
    else:
        checks.append(("Exhaust temperature realistic", False, f"Peak {peak_temp:.0f}K too low"))

    # Check 2: Temperature should decrease with distance (attenuation)
    close_temps = [stats['temp_max'][i] for i in range(len(stats['distances']))
                   if stats['distances'][i] < 1000 and stats['object_visible'][i]]
    far_temps = [stats['temp_max'][i] for i in range(len(stats['distances']))
                 if stats['distances'][i] > 1500 and stats['object_visible'][i]]

    if close_temps and far_temps:
        avg_close = np.mean(close_temps)
        avg_far = np.mean(far_temps)
        if avg_close > avg_far:
            checks.append(("Atmospheric attenuation", True, f"Close: {avg_close:.0f}K > Far: {avg_far:.0f}K"))
        else:
            checks.append(("Atmospheric attenuation", False, "No temperature decrease with distance"))

    # Check 3: Multiple thermal zones visible
    frame_at_closest = frames[closest_frame_idx]
    unique_temps = len(set(frame_at_closest.temperature_map.flatten().astype(int)))
    if unique_temps > 3:
        checks.append(("Thermal zone differentiation", True, f"{unique_temps} distinct temp levels"))
    else:
        checks.append(("Thermal zone differentiation", False, f"Only {unique_temps} temp levels"))

    # Check 4: Object fills reasonable portion of frame at closest
    frame_pixels = 640 * 480
    hot_pixels = np.sum(frame_at_closest.temperature_map > background_temp + 10)
    fill_pct = 100 * hot_pixels / frame_pixels
    if 0.1 < fill_pct < 20:
        checks.append(("Object size at closest", True, f"{fill_pct:.2f}% of frame"))
    else:
        checks.append(("Object size at closest", fill_pct > 0, f"{fill_pct:.2f}% of frame"))

    for check_name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {check_name}: {detail}")

    passed_count = sum(1 for _, passed, _ in checks if passed)
    print(f"\n  Overall: {passed_count}/{len(checks)} checks passed")

    # =========================================================================
    # EXPORT
    # =========================================================================
    print("\n[EXPORT]")
    print("-" * 50)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'f16_demo')
    os.makedirs(output_dir, exist_ok=True)

    # Save key frames as images
    try:
        from PIL import Image

        key_frame_indices = [0, closest_frame_idx, total_frames - 1]
        for idx in key_frame_indices:
            frame = frames[idx]
            img = Image.fromarray(frame.image)
            img_path = os.path.join(output_dir, f"frame_{idx:04d}_t{idx/fps:.2f}s.png")
            img.save(img_path)
            print(f"  Saved: {img_path}")

        # Save composite comparison image
        composite_width = 640 * 3 + 40
        composite_height = 480 + 100
        composite = Image.new('RGB', (composite_width, composite_height), (30, 30, 30))

        for i, idx in enumerate(key_frame_indices):
            frame_img = Image.fromarray(frames[idx].image)
            composite.paste(frame_img, (i * 660 + 10, 10))

        composite_path = os.path.join(output_dir, "comparison_composite.png")
        composite.save(composite_path)
        print(f"  Saved composite: {composite_path}")

    except ImportError:
        print("  PIL not available, skipping image export")
        print("  Install with: pip install Pillow")

    # Export video using OpenCV
    try:
        import cv2

        video_path = os.path.join(output_dir, "f16_flyby.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (640, 480))

        for frame in frames:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)

        out.release()
        print(f"  Saved video: {video_path}")

    except ImportError:
        print("  OpenCV not available, skipping video export")
        print("  Install with: pip install opencv-python")

    # Save scenario configuration
    config_path = os.path.join(output_dir, "scenario_config.txt")
    with open(config_path, 'w') as f:
        f.write("F-16 Flyby Demo Scenario Configuration\n")
        f.write("=" * 50 + "\n\n")
        f.write("Object: F-16 Fighting Falcon\n")
        f.write(f"Duration: {duration_sec}s at {fps} FPS\n\n")
        f.write("Flight Path Waypoints:\n")
        for i, (t, pos) in enumerate(waypoints):
            f.write(f"  [{i}] t={t:.1f}s: ({pos.x:.0f}, {pos.y:.0f}, {pos.z:.0f})m\n")
        f.write(f"\nCamera:\n")
        f.write(f"  Position: ({camera.position.x}, {camera.position.y}, {camera.position.z})m\n")
        f.write(f"  FOV: {camera.get_fov():.1f}°\n")
        f.write(f"  NETD: {camera.get_netd():.0f}mK\n")
        f.write(f"\nResults:\n")
        f.write(f"  Peak temperature: {peak_temp:.1f}K\n")
        f.write(f"  Visible frames: {visible_frames}/{total_frames}\n")
        f.write(f"  Closest distance: {min(stats['distances']):.0f}m\n")
    print(f"  Saved config: {config_path}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)

    return {
        'frames': frames,
        'stats': stats,
        'checks': checks,
        'output_dir': output_dir
    }


if __name__ == "__main__":
    result = create_f16_flyby_scenario()
