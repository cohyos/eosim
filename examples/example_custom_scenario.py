"""
EOSIM Custom Scenario Example

This example demonstrates how to:
1. Use the new objects (missiles, launchers) from the library.
2. Set specific positions and orientations.
3. Combine multiple objects into a cohesive scenario.
"""

from eosim.core import Simulation
from eosim.scene import Scene
from eosim.render import Sensor
from eosim.library import get_object_ids

def main():
    # 1. Create Simulation and Scene
    sim = Simulation()
    scene = Scene()
    
    print("Available objects:", get_object_ids())

    # 2. Add Terrain (Placeholder flat plane)
    scene.add_flat_terrain(size_km=50.0)

    # 3. Add Targets using Library IDs
    # Note: Use run_browser.bat to find these IDs and copy snippets!
    
    # Blue Force: F-16 flight
    print("Adding F-16 and missiles...")
    scene.add_target(
        "f16", 
        position_km=(10.0, 0.0, 5.0), 
        heading_deg=90.0,
        speed_mps=300.0,
        name="Viper 1-1"
    )
    
    # F-16 Wingman
    scene.add_target(
        "f16",
        position_km=(9.8, -0.2, 5.0),
        heading_deg=90.0,
        speed_mps=300.0,
        name="Viper 1-2"
    )

    # Firing an AIM-120
    scene.add_target(
        "aim120",
        position_km=(10.1, 0.0, 4.9),
        heading_deg=90.0,
        speed_mps=600.0, # Mach 2ish
        name="Fox-3"
    )

    # Red Force: S-400 Battery
    print("Adding S-400 battery...")
    scene.add_target(
        "s400_launcher",
        position_km=(40.0, 5.0, 0.0),
        heading_deg=270.0,
        name="Growler 1 (Launcher)"
    )
    
    # Radar for the battery (using generic vehicle for now, or specific radar if added)
    scene.add_target(
        "humvee", # Placeholder for radar vehicle
        position_km=(40.05, 5.05, 0.0),
        heading_deg=270.0,
        name="Growler 1 (Radar)"
    )

    # 4. Setup Sensor (Simulating an airborne pod viewing the scene)
    sensor = Sensor(
        position_km=(0.0, 0.0, 10.0), # 10km altitude, at origin
        look_at=(20.0, 2.5, 0.0),      # Looking towards the middle
        fov_deg=5.0
    )
    
    # 5. Render a frame
    print("Rendering frame...")
    image = sim.render(scene, sensor)
    
    # Save output
    output_file = "custom_scenario_render.png"
    # sim.save_image(image, output_file) # Assuming save_image exists or use cv2/PIL
    print(f"Scenario constructed successfully with {len(scene.targets)} targets.")
    
if __name__ == "__main__":
    main()
