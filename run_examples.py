#!/usr/bin/env python3
"""
EOSIM Example Runner CLI

A command-line interface to run and visualize EOSIM simulation examples.

Usage:
    python run_examples.py                    # Interactive menu
    python run_examples.py --list             # List all examples
    python run_examples.py --run vehicle_on_road  # Run specific example
    python run_examples.py --run all          # Run all examples
    python run_examples.py --run all --no-display  # Run without display
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")

    try:
        import scipy
    except ImportError:
        missing.append("scipy")

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        sys.exit(1)


def list_examples():
    """List all available examples."""
    from eosim.examples.scenarios import EXAMPLES

    print("\n" + "=" * 70)
    print("EOSIM Available Examples")
    print("=" * 70)

    descriptions = {
        "vehicle_on_road": "Hot vehicle on road (LWIR at 500m)",
        "person_in_forest": "Human detection in vegetation (MWIR at 200m)",
        "aircraft_sky": "Aircraft with motion blur (MWIR at 5km)",
        "ship_ocean": "Maritime vessel on ocean (LWIR at 2km)",
        "building_thermal": "Building thermal inspection (LWIR at 50m)",
        "wildlife_tracking": "Wildlife detection in forest (MWIR at 300m)",
        "industrial_monitoring": "Industrial equipment monitoring (LWIR at 100m)",
        "night_vision_swir": "Night surveillance scene (SWIR at 100m)",
        "solar_panel_inspection": "Solar panel defect detection (LWIR at 30m)",
        "urban_surveillance": "Urban street scene (Visible at 200m)",
        # Realistic examples
        "f16_500m": "★ F-16 thermal signature at 500m (MWIR) - REALISTIC",
        "f16_visible": "★ F-16 visible/color rendering (RGB) - REALISTIC",
        "f16_video": "★ F-16 flyby VIDEO (30 frames) - REALISTIC",
        "realistic_vehicle": "★ Vehicle with thermal gradients (LWIR) - REALISTIC",
        "realistic_person": "★ Person with thermal features (MWIR) - REALISTIC",
    }

    print(f"\n{'#':<3} {'Name':<25} {'Description':<40}")
    print("-" * 70)

    for i, name in enumerate(EXAMPLES.keys(), 1):
        desc = descriptions.get(name, "")
        print(f"{i:<3} {name:<25} {desc:<40}")

    print("-" * 70)
    print(f"Total: {len(EXAMPLES)} examples\n")


def run_example(name: str, output_dir: Path, display: bool = True, seed: int = 42):
    """Run a single example and save/display results."""
    import numpy as np

    from eosim.examples.scenarios import get_example_by_name, list_examples as get_names

    available = get_names()
    if name not in available:
        print(f"Error: Unknown example '{name}'")
        print(f"Available: {', '.join(available)}")
        return None

    print(f"\nRunning example: {name}...")
    print("-" * 50)

    try:
        result = get_example_by_name(name, seed=seed)
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()
        return None

    print(f"  Name: {result.name}")
    print(f"  Description: {result.description}")
    print(f"  Sensor: {result.sensor_type}")
    print(f"  Image shape: {result.digital_image.shape}")
    print(f"  Temperature range: {result.temperature_map.min():.1f}K - {result.temperature_map.max():.1f}K")
    print(f"  Output range: {result.digital_image.min()} - {result.digital_image.max()} DN")

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw numpy arrays
    np.save(output_dir / f"{name}_digital.npy", result.digital_image)
    np.save(output_dir / f"{name}_temperature.npy", result.temperature_map)

    # Save as images
    try:
        from PIL import Image

        # Check for RGB image in metadata
        if 'rgb_image' in result.metadata:
            rgb = result.metadata['rgb_image']
            rgb_uint8 = (rgb * 255).astype(np.uint8)
            Image.fromarray(rgb_uint8).save(output_dir / f"{name}_rgb.png")
            print(f"  Saved RGB: {output_dir / name}_rgb.png")

        # Check for video frames
        if 'video_frames' in result.metadata:
            frames = result.metadata['video_frames']
            fps = result.metadata.get('fps', 15)
            print(f"  Video: {len(frames)} frames at {fps} fps")

            # Save individual frames
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            for i, frame in enumerate(frames):
                if frame.ndim == 3:
                    Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")
                else:
                    frame_norm = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
                    Image.fromarray(frame_norm).save(frames_dir / f"frame_{i:04d}.png")
            print(f"  Saved {len(frames)} frames to: {frames_dir}")

            # Try to create video with OpenCV
            try:
                import cv2
                video_path = output_dir / f"{name}.mp4"
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h), True)
                for frame in frames:
                    if frame.ndim == 3:
                        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        gray = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
                        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    out.write(bgr)
                out.release()
                print(f"  Saved video: {video_path}")
            except ImportError:
                print("  Note: Install opencv-python for video output: pip install opencv-python")

        # Normalize digital image to 8-bit for PNG
        img = result.digital_image.astype(np.float64)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img_normalized = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_normalized = np.zeros_like(img, dtype=np.uint8)

        Image.fromarray(img_normalized).save(output_dir / f"{name}_output.png")

        # Save temperature map as colored image (skip for visible/video)
        if result.metadata.get('mode') not in ['visible', 'video']:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            temp = result.temperature_map
            temp_min, temp_max = temp.min(), temp.max()
            if temp_max > temp_min:
                temp_normalized = (temp - temp_min) / (temp_max - temp_min)
            else:
                temp_normalized = np.zeros_like(temp)

            temp_colored = (cm.hot(temp_normalized)[:, :, :3] * 255).astype(np.uint8)
            Image.fromarray(temp_colored).save(output_dir / f"{name}_temperature.png")

        print(f"  Saved: {output_dir / name}_*.png/npy")

    except ImportError:
        print("  Note: PIL not available, saved .npy files only")

    # Display if requested
    if display:
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # Try interactive backend
        except:
            pass

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f"{result.name}: {result.description}", fontsize=12)

            # Temperature map
            im1 = axes[0].imshow(result.temperature_map, cmap='hot')
            axes[0].set_title('Temperature Map [K]')
            plt.colorbar(im1, ax=axes[0], label='Temperature [K]')
            axes[0].axis('off')

            # Digital output
            im2 = axes[1].imshow(result.digital_image, cmap='gray')
            axes[1].set_title(f'Sensor Output [{result.sensor_type}]')
            plt.colorbar(im2, ax=axes[1], label='Digital Number [DN]')
            axes[1].axis('off')

            plt.tight_layout()

            # Save figure
            fig.savefig(output_dir / f"{name}_combined.png", dpi=150, bbox_inches='tight')
            print(f"  Saved combined figure: {output_dir / name}_combined.png")

            plt.show(block=False)
            plt.pause(0.5)

        except Exception as e:
            print(f"  Display not available: {e}")

    return result


def run_all_examples(output_dir: Path, display: bool = True, seed: int = 42):
    """Run all examples."""
    from eosim.examples.scenarios import list_examples as get_names

    names = get_names()
    results = {}

    print(f"\nRunning all {len(names)} examples...")
    print("=" * 70)

    for i, name in enumerate(names, 1):
        print(f"\n[{i}/{len(names)}] ", end="")
        result = run_example(name, output_dir, display=False, seed=seed)
        if result:
            results[name] = result

    print("\n" + "=" * 70)
    print(f"Completed: {len(results)}/{len(names)} examples")
    print(f"Output saved to: {output_dir.absolute()}")

    # Create summary montage if display enabled
    if display and results:
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            n = len(results)
            cols = 4
            rows = (n + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
            axes = axes.flatten() if n > 1 else [axes]

            for ax, (name, result) in zip(axes, results.items()):
                ax.imshow(result.digital_image, cmap='gray')
                ax.set_title(name.replace('_', '\n'), fontsize=9)
                ax.axis('off')

            # Hide empty subplots
            for ax in axes[len(results):]:
                ax.axis('off')

            fig.suptitle('EOSIM Example Gallery', fontsize=14)
            plt.tight_layout()

            fig.savefig(output_dir / "gallery.png", dpi=150, bbox_inches='tight')
            print(f"Saved gallery: {output_dir / 'gallery.png'}")

            plt.show()

        except Exception as e:
            print(f"Could not create gallery: {e}")

    return results


def interactive_menu():
    """Run interactive menu."""
    from eosim.examples.scenarios import list_examples as get_names

    while True:
        print("\n" + "=" * 50)
        print("EOSIM Example Runner")
        print("=" * 50)
        print("\nOptions:")
        print("  1. List all examples")
        print("  2. Run a specific example")
        print("  3. Run all examples")
        print("  4. Run all examples (no display)")
        print("  q. Quit")

        choice = input("\nEnter choice: ").strip().lower()

        if choice == '1':
            list_examples()

        elif choice == '2':
            list_examples()
            names = get_names()
            print("\nEnter example name or number (or 'back' to return):")
            selection = input("> ").strip()

            if selection.lower() == 'back':
                continue

            # Handle numeric input
            if selection.isdigit():
                idx = int(selection) - 1
                if 0 <= idx < len(names):
                    selection = names[idx]
                else:
                    print(f"Invalid number. Enter 1-{len(names)}")
                    continue

            if selection in names:
                output_dir = Path("./output") / selection
                run_example(selection, output_dir, display=True)
                input("\nPress Enter to continue...")
            else:
                print(f"Unknown example: {selection}")

        elif choice == '3':
            output_dir = Path("./output")
            run_all_examples(output_dir, display=True)
            input("\nPress Enter to continue...")

        elif choice == '4':
            output_dir = Path("./output")
            run_all_examples(output_dir, display=False)
            input("\nPress Enter to continue...")

        elif choice in ('q', 'quit', 'exit'):
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Enter 1-4 or 'q' to quit.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EOSIM Example Runner - Run and visualize simulation examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_examples.py                         Interactive menu
  python run_examples.py --list                  List available examples
  python run_examples.py --run vehicle_on_road   Run specific example
  python run_examples.py --run all               Run all examples
  python run_examples.py --run all --no-display  Run without showing plots
  python run_examples.py --run 1                 Run example by number
        """
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available examples'
    )

    parser.add_argument(
        '--run', '-r',
        type=str,
        metavar='NAME',
        help='Run example by name, number, or "all"'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./output',
        help='Output directory (default: ./output)'
    )

    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display plots (just save files)'
    )

    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Check dependencies
    check_dependencies()

    output_dir = Path(args.output)
    display = not args.no_display

    if args.list:
        list_examples()

    elif args.run:
        from eosim.examples.scenarios import list_examples as get_names
        names = get_names()

        selection = args.run.strip()

        if selection.lower() == 'all':
            run_all_examples(output_dir, display=display, seed=args.seed)

        elif selection.isdigit():
            idx = int(selection) - 1
            if 0 <= idx < len(names):
                name = names[idx]
                run_example(name, output_dir / name, display=display, seed=args.seed)
            else:
                print(f"Invalid number. Enter 1-{len(names)}")
                sys.exit(1)

        elif selection in names:
            run_example(selection, output_dir / selection, display=display, seed=args.seed)

        else:
            print(f"Unknown example: {selection}")
            print(f"Available: {', '.join(names)}")
            sys.exit(1)

    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    main()
