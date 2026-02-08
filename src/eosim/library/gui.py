"""
EOSIM Scenario Builder GUI.

Provides a Windows-like graphical user interface for building and running
EO/IR sensor simulation scenarios.

Usage:
------
>>> from eosim.library.gui import launch_scenario_builder
>>> launch_scenario_builder()

Or from command line:
$ python -m eosim.library.gui
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional, Dict, Any, List, Callable
import threading
import numpy as np
from dataclasses import dataclass, field


@dataclass
class GUIState:
    """Current state of the GUI."""
    sensor_id: str = "mx15"
    targets: List[Dict[str, Any]] = field(default_factory=list)
    platform_enabled: bool = False
    platform_type: str = "fixed_wing"
    environment: str = "clear_day"
    background: str = "terrain"
    seed: int = 42
    result_image: Optional[np.ndarray] = None


class ScenarioBuilderGUI:
    """Main GUI application for scenario building."""

    def __init__(self, root: tk.Tk):
        """Initialize the GUI.

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("EOSIM Scenario Builder")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # State
        self.state = GUIState()
        self._result = None
        self._scenario = None

        # Load library data
        self._load_library_data()

        # Setup UI
        self._setup_styles()
        self._create_menu()
        self._create_main_layout()

        # Bind events
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_library_data(self):
        """Load sensor and object library data."""
        try:
            from eosim.library import (
                list_sensors, list_objects, list_object_categories,
                EnvironmentType, BackgroundType,
            )
            self.sensors = list_sensors()
            self.objects = list_objects()
            self.categories = list_object_categories()
            self.environments = [e.value for e in EnvironmentType]
            self.backgrounds = [b.value for b in BackgroundType]
            self.platform_types = [
                "fixed_wing", "rotary_wing", "ground_vehicle",
                "naval_surface", "tripod", "handheld", "satellite"
            ]
        except ImportError as e:
            messagebox.showerror("Import Error", f"Failed to load library: {e}")
            self.sensors = []
            self.objects = []
            self.categories = []
            self.environments = ["clear_day", "clear_night"]
            self.backgrounds = ["terrain", "sky", "urban"]
            self.platform_types = ["fixed_wing", "rotary_wing"]

    def _setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')  # Use clam theme for better cross-platform look

        # Configure custom styles
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 9))
        style.configure("Run.TButton", font=("Segoe UI", 10, "bold"))

    def _create_menu(self):
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Scenario", command=self._new_scenario, accelerator="Ctrl+N")
        file_menu.add_command(label="Load Scenario...", command=self._load_scenario)
        file_menu.add_command(label="Save Scenario...", command=self._save_scenario)
        file_menu.add_separator()
        file_menu.add_command(label="Export Image...", command=self._export_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Clear Targets", command=self._clear_targets)

        # Run menu
        run_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Run", menu=run_menu)
        run_menu.add_command(label="Run Scenario", command=self._run_scenario, accelerator="F5")

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Object Viewer...", command=self._open_object_viewer)
        tools_menu.add_command(label="Sensor Browser...", command=self._open_sensor_browser)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

        # Bind keyboard shortcuts
        self.root.bind("<Control-n>", lambda e: self._new_scenario())
        self.root.bind("<F5>", lambda e: self._run_scenario())

    def _create_main_layout(self):
        """Create the main window layout."""
        # Main paned window (horizontal split)
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Configuration
        left_frame = ttk.Frame(main_paned, width=400)
        main_paned.add(left_frame, weight=1)

        # Right panel - Preview and Results
        right_frame = ttk.Frame(main_paned, width=600)
        main_paned.add(right_frame, weight=2)

        # Build left panel
        self._create_config_panel(left_frame)

        # Build right panel
        self._create_preview_panel(right_frame)

        # Status bar
        self._create_status_bar()

    def _create_config_panel(self, parent):
        """Create the configuration panel."""
        # Create notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Sensor Configuration
        sensor_tab = ttk.Frame(notebook, padding=10)
        notebook.add(sensor_tab, text="Sensor")
        self._create_sensor_tab(sensor_tab)

        # Tab 2: Target Configuration
        target_tab = ttk.Frame(notebook, padding=10)
        notebook.add(target_tab, text="Targets")
        self._create_target_tab(target_tab)

        # Tab 3: Platform Configuration (6DOF)
        platform_tab = ttk.Frame(notebook, padding=10)
        notebook.add(platform_tab, text="Platform (6DOF)")
        self._create_platform_tab(platform_tab)

        # Tab 4: Environment
        env_tab = ttk.Frame(notebook, padding=10)
        notebook.add(env_tab, text="Environment")
        self._create_environment_tab(env_tab)

        # Run button at bottom
        run_frame = ttk.Frame(parent, padding=5)
        run_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.run_btn = ttk.Button(
            run_frame, text="Run Scenario (F5)",
            command=self._run_scenario, style="Run.TButton"
        )
        self.run_btn.pack(fill=tk.X, pady=5)

    def _create_sensor_tab(self, parent):
        """Create sensor configuration tab."""
        # Sensor selection
        ttk.Label(parent, text="Sensor Selection", style="Header.TLabel").pack(anchor=tk.W)

        sensor_frame = ttk.LabelFrame(parent, text="Sensor", padding=10)
        sensor_frame.pack(fill=tk.X, pady=5)

        ttk.Label(sensor_frame, text="Sensor:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.sensor_var = tk.StringVar(value=self.state.sensor_id)
        self.sensor_combo = ttk.Combobox(
            sensor_frame, textvariable=self.sensor_var,
            values=self.sensors, state="readonly", width=30
        )
        self.sensor_combo.grid(row=0, column=1, sticky=tk.W, pady=2)
        self.sensor_combo.bind("<<ComboboxSelected>>", self._on_sensor_changed)

        # Sensor info display
        self.sensor_info = tk.Text(sensor_frame, height=8, width=40, state=tk.DISABLED)
        self.sensor_info.grid(row=1, column=0, columnspan=2, pady=5, sticky=tk.W+tk.E)

        # Sensor position
        pos_frame = ttk.LabelFrame(parent, text="Sensor Position", padding=10)
        pos_frame.pack(fill=tk.X, pady=5)

        ttk.Label(pos_frame, text="Altitude (m):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.altitude_var = tk.StringVar(value="3000")
        ttk.Entry(pos_frame, textvariable=self.altitude_var, width=15).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(pos_frame, text="X Position (m):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.sensor_x_var = tk.StringVar(value="0")
        ttk.Entry(pos_frame, textvariable=self.sensor_x_var, width=15).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(pos_frame, text="Y Position (m):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.sensor_y_var = tk.StringVar(value="0")
        ttk.Entry(pos_frame, textvariable=self.sensor_y_var, width=15).grid(row=2, column=1, sticky=tk.W)

        # Update sensor info
        self._update_sensor_info()

    def _create_target_tab(self, parent):
        """Create target configuration tab."""
        ttk.Label(parent, text="Target Configuration", style="Header.TLabel").pack(anchor=tk.W)

        # Add target section
        add_frame = ttk.LabelFrame(parent, text="Add Target", padding=10)
        add_frame.pack(fill=tk.X, pady=5)

        # Category filter
        ttk.Label(add_frame, text="Category:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.category_var = tk.StringVar(value="ALL")
        category_combo = ttk.Combobox(
            add_frame, textvariable=self.category_var,
            values=["ALL"] + self.categories, state="readonly", width=20
        )
        category_combo.grid(row=0, column=1, sticky=tk.W, pady=2)
        category_combo.bind("<<ComboboxSelected>>", self._on_category_changed)

        # Object selection
        ttk.Label(add_frame, text="Object:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.object_var = tk.StringVar()
        self.object_combo = ttk.Combobox(
            add_frame, textvariable=self.object_var,
            values=self.objects, state="readonly", width=30
        )
        self.object_combo.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=2)

        # Target parameters
        ttk.Label(add_frame, text="Range (km):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.range_var = tk.StringVar(value="5.0")
        ttk.Entry(add_frame, textvariable=self.range_var, width=10).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(add_frame, text="Aspect (deg):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.aspect_var = tk.StringVar(value="90")
        ttk.Entry(add_frame, textvariable=self.aspect_var, width=10).grid(row=3, column=1, sticky=tk.W)

        ttk.Label(add_frame, text="Heading (deg):").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.heading_var = tk.StringVar(value="0")
        ttk.Entry(add_frame, textvariable=self.heading_var, width=10).grid(row=4, column=1, sticky=tk.W)

        ttk.Label(add_frame, text="Name:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.target_name_var = tk.StringVar(value="")
        ttk.Entry(add_frame, textvariable=self.target_name_var, width=20).grid(row=5, column=1, sticky=tk.W)

        # Buttons row
        btn_frame = ttk.Frame(add_frame)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=10)

        preview_btn = ttk.Button(btn_frame, text="Preview 3D", command=self._preview_object)
        preview_btn.pack(side=tk.LEFT, padx=5)

        add_btn = ttk.Button(btn_frame, text="Add Target", command=self._add_target)
        add_btn.pack(side=tk.LEFT, padx=5)

        # Target list
        list_frame = ttk.LabelFrame(parent, text="Target List", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create treeview for targets
        columns = ("name", "object", "range", "aspect", "heading")
        self.target_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=8)

        self.target_tree.heading("name", text="Name")
        self.target_tree.heading("object", text="Object")
        self.target_tree.heading("range", text="Range (km)")
        self.target_tree.heading("aspect", text="Aspect")
        self.target_tree.heading("heading", text="Heading")

        self.target_tree.column("name", width=80)
        self.target_tree.column("object", width=100)
        self.target_tree.column("range", width=70)
        self.target_tree.column("aspect", width=60)
        self.target_tree.column("heading", width=60)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.target_tree.yview)
        self.target_tree.configure(yscrollcommand=scrollbar.set)

        self.target_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Remove button
        remove_btn = ttk.Button(parent, text="Remove Selected", command=self._remove_target)
        remove_btn.pack(pady=5)

    def _create_platform_tab(self, parent):
        """Create platform (6DOF) configuration tab."""
        ttk.Label(parent, text="6DOF Platform Configuration", style="Header.TLabel").pack(anchor=tk.W)

        # Enable platform checkbox
        self.platform_enabled_var = tk.BooleanVar(value=False)
        enable_check = ttk.Checkbutton(
            parent, text="Enable 6DOF Platform Motion",
            variable=self.platform_enabled_var,
            command=self._on_platform_toggle
        )
        enable_check.pack(anchor=tk.W, pady=5)

        # Platform settings frame
        self.platform_frame = ttk.LabelFrame(parent, text="Platform Settings", padding=10)
        self.platform_frame.pack(fill=tk.X, pady=5)

        # Platform type
        ttk.Label(self.platform_frame, text="Platform Type:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.platform_type_var = tk.StringVar(value="fixed_wing")
        platform_combo = ttk.Combobox(
            self.platform_frame, textvariable=self.platform_type_var,
            values=self.platform_types, state="readonly", width=20
        )
        platform_combo.grid(row=0, column=1, sticky=tk.W, pady=2)

        # Speed
        ttk.Label(self.platform_frame, text="Speed (m/s):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.platform_speed_var = tk.StringVar(value="100")
        ttk.Entry(self.platform_frame, textvariable=self.platform_speed_var, width=15).grid(row=1, column=1, sticky=tk.W)

        # Heading
        ttk.Label(self.platform_frame, text="Heading (deg):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.platform_heading_var = tk.StringVar(value="0")
        ttk.Entry(self.platform_frame, textvariable=self.platform_heading_var, width=15).grid(row=2, column=1, sticky=tk.W)

        # Orientation
        orient_frame = ttk.LabelFrame(parent, text="Platform Orientation", padding=10)
        orient_frame.pack(fill=tk.X, pady=5)

        ttk.Label(orient_frame, text="Roll (deg):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.roll_var = tk.StringVar(value="0")
        ttk.Entry(orient_frame, textvariable=self.roll_var, width=10).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(orient_frame, text="Pitch (deg):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.pitch_var = tk.StringVar(value="0")
        ttk.Entry(orient_frame, textvariable=self.pitch_var, width=10).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(orient_frame, text="Yaw (deg):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.yaw_var = tk.StringVar(value="0")
        ttk.Entry(orient_frame, textvariable=self.yaw_var, width=10).grid(row=2, column=1, sticky=tk.W)

        # Gimbal settings
        gimbal_frame = ttk.LabelFrame(parent, text="Gimbal Settings", padding=10)
        gimbal_frame.pack(fill=tk.X, pady=5)

        ttk.Label(gimbal_frame, text="Gimbal Mode:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.gimbal_mode_var = tk.StringVar(value="track_target")
        gimbal_combo = ttk.Combobox(
            gimbal_frame, textvariable=self.gimbal_mode_var,
            values=["stabilized", "track_target", "position", "scan"],
            state="readonly", width=15
        )
        gimbal_combo.grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(gimbal_frame, text="Track Target #:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.track_target_var = tk.StringVar(value="0")
        ttk.Entry(gimbal_frame, textvariable=self.track_target_var, width=10).grid(row=1, column=1, sticky=tk.W)

        # Orbit settings
        orbit_frame = ttk.LabelFrame(parent, text="Orbit Pattern", padding=10)
        orbit_frame.pack(fill=tk.X, pady=5)

        self.orbit_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(orbit_frame, text="Enable Orbit", variable=self.orbit_enabled_var).grid(row=0, column=0, columnspan=2, sticky=tk.W)

        ttk.Label(orbit_frame, text="Radius (m):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.orbit_radius_var = tk.StringVar(value="5000")
        ttk.Entry(orbit_frame, textvariable=self.orbit_radius_var, width=15).grid(row=1, column=1, sticky=tk.W)

        # Initially disable platform controls
        self._set_platform_controls_state(False)

    def _create_environment_tab(self, parent):
        """Create environment configuration tab."""
        ttk.Label(parent, text="Environment Configuration", style="Header.TLabel").pack(anchor=tk.W)

        # Environment preset
        env_frame = ttk.LabelFrame(parent, text="Environment", padding=10)
        env_frame.pack(fill=tk.X, pady=5)

        ttk.Label(env_frame, text="Preset:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.environment_var = tk.StringVar(value="clear_day")
        env_combo = ttk.Combobox(
            env_frame, textvariable=self.environment_var,
            values=self.environments, state="readonly", width=20
        )
        env_combo.grid(row=0, column=1, sticky=tk.W, pady=2)

        # Background
        bg_frame = ttk.LabelFrame(parent, text="Background", padding=10)
        bg_frame.pack(fill=tk.X, pady=5)

        ttk.Label(bg_frame, text="Type:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.background_var = tk.StringVar(value="terrain")
        bg_combo = ttk.Combobox(
            bg_frame, textvariable=self.background_var,
            values=self.backgrounds, state="readonly", width=20
        )
        bg_combo.grid(row=0, column=1, sticky=tk.W, pady=2)

        # Simulation settings
        sim_frame = ttk.LabelFrame(parent, text="Simulation", padding=10)
        sim_frame.pack(fill=tk.X, pady=5)

        ttk.Label(sim_frame, text="Random Seed:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.seed_var = tk.StringVar(value="42")
        ttk.Entry(sim_frame, textvariable=self.seed_var, width=15).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(sim_frame, text="Resolution:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.resolution_var = tk.StringVar(value="640x480")
        res_combo = ttk.Combobox(
            sim_frame, textvariable=self.resolution_var,
            values=["320x240", "640x480", "1280x720", "1920x1080"],
            state="readonly", width=15
        )
        res_combo.grid(row=1, column=1, sticky=tk.W, pady=2)

    def _create_preview_panel(self, parent):
        """Create the preview/results panel."""
        # Title
        ttk.Label(parent, text="Simulation Preview", style="Title.TLabel").pack(anchor=tk.W, pady=5)

        # Canvas for image display
        canvas_frame = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=2)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg="black", width=640, height=480)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Create placeholder text
        self.canvas.create_text(
            320, 240, text="Run scenario to see preview",
            fill="gray", font=("Segoe UI", 14)
        )

        # Results info
        results_frame = ttk.LabelFrame(parent, text="Results", padding=10)
        results_frame.pack(fill=tk.X, pady=5)

        self.results_text = tk.Text(results_frame, height=8, width=60, state=tk.DISABLED)
        self.results_text.pack(fill=tk.X)

    def _create_status_bar(self):
        """Create the status bar."""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = ttk.Label(
            self.status_frame, text="Ready", style="Status.TLabel"
        )
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)

        self.progress = ttk.Progressbar(
            self.status_frame, mode="indeterminate", length=100
        )
        self.progress.pack(side=tk.RIGHT, padx=5, pady=2)

    # Event handlers
    def _on_sensor_changed(self, event=None):
        """Handle sensor selection change."""
        self._update_sensor_info()

    def _on_category_changed(self, event=None):
        """Handle category filter change."""
        category = self.category_var.get()
        if category == "ALL":
            self.object_combo["values"] = self.objects
        else:
            try:
                from eosim.library import ObjectLibrary
                filtered = ObjectLibrary.get_by_category(category)
                self.object_combo["values"] = [obj.id for obj in filtered]
            except Exception:
                self.object_combo["values"] = self.objects

    def _on_platform_toggle(self):
        """Handle platform enable/disable toggle."""
        enabled = self.platform_enabled_var.get()
        self._set_platform_controls_state(enabled)

    def _set_platform_controls_state(self, enabled: bool):
        """Enable or disable platform controls."""
        state = tk.NORMAL if enabled else tk.DISABLED
        for child in self.platform_frame.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass

    def _update_sensor_info(self):
        """Update sensor information display."""
        sensor_id = self.sensor_var.get()
        try:
            from eosim.library import get_sensor
            spec = get_sensor(sensor_id)
            info = f"""Name: {spec.name}
Manufacturer: {spec.manufacturer}
Type: {spec.sensor_type.value}
Mount: {spec.mount_type.value}

IR Detector:
  Resolution: {spec.detector_ir.width_pixels}x{spec.detector_ir.height_pixels}
  Band: {spec.detector_ir.spectral_band_um[0]}-{spec.detector_ir.spectral_band_um[1]} um
  NETD: {spec.detector_ir.nedt_mk} mK

Optics:
  Focal Length: {spec.optics_ir.focal_length_mm} mm
  FOV: {spec.optics_ir.fov_narrow_deg}-{spec.optics_ir.fov_wide_deg} deg"""
        except Exception as e:
            info = f"Error loading sensor info: {e}"

        self.sensor_info.config(state=tk.NORMAL)
        self.sensor_info.delete(1.0, tk.END)
        self.sensor_info.insert(tk.END, info)
        self.sensor_info.config(state=tk.DISABLED)

    def _preview_object(self):
        """Preview the selected object in 3D viewer."""
        obj_id = self.object_var.get()
        if not obj_id:
            messagebox.showwarning("Warning", "Please select an object first")
            return

        try:
            from eosim.library.object_viewer import ObjectViewer3D
            viewer = ObjectViewer3D(self.root)
            viewer.show_object(obj_id)
            self._update_status(f"Previewing: {obj_id}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preview object: {e}")

    def _add_target(self):
        """Add a target to the list."""
        obj_id = self.object_var.get()
        if not obj_id:
            messagebox.showwarning("Warning", "Please select an object type")
            return

        try:
            range_km = float(self.range_var.get())
            aspect = float(self.aspect_var.get())
            heading = float(self.heading_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric values")
            return

        name = self.target_name_var.get() or f"Target_{len(self.state.targets) + 1}"

        target = {
            "name": name,
            "object_id": obj_id,
            "range_km": range_km,
            "aspect_deg": aspect,
            "heading_deg": heading,
        }
        self.state.targets.append(target)

        # Add to treeview
        self.target_tree.insert("", tk.END, values=(
            name, obj_id, f"{range_km:.1f}", f"{aspect:.0f}", f"{heading:.0f}"
        ))

        # Clear name field
        self.target_name_var.set("")
        self._update_status(f"Added target: {name}")

    def _remove_target(self):
        """Remove selected target from list."""
        selected = self.target_tree.selection()
        if not selected:
            return

        for item in selected:
            idx = self.target_tree.index(item)
            self.target_tree.delete(item)
            if idx < len(self.state.targets):
                del self.state.targets[idx]

    def _clear_targets(self):
        """Clear all targets."""
        self.state.targets.clear()
        for item in self.target_tree.get_children():
            self.target_tree.delete(item)
        self._update_status("Cleared all targets")

    def _run_scenario(self):
        """Build and run the scenario."""
        if not self.state.targets:
            messagebox.showwarning("Warning", "Please add at least one target")
            return

        self._update_status("Building scenario...")
        self.progress.start()
        self.run_btn.config(state=tk.DISABLED)

        # Run in separate thread to keep UI responsive
        thread = threading.Thread(target=self._run_scenario_thread)
        thread.start()

    def _run_scenario_thread(self):
        """Run scenario in background thread."""
        try:
            from eosim.library import (
                ScenarioBuilder, run_scenario,
                EnvironmentType, BackgroundType, Position3D,
            )

            # Build scenario
            builder = ScenarioBuilder()
            builder.set_name("GUI Scenario")

            # Set sensor
            altitude = float(self.altitude_var.get())
            sensor_x = float(self.sensor_x_var.get())
            sensor_y = float(self.sensor_y_var.get())
            builder.set_sensor(
                self.sensor_var.get(),
                position=Position3D(sensor_x, sensor_y, altitude, "m")
            )

            # Add targets
            for target in self.state.targets:
                builder.add_target(
                    target["object_id"],
                    range_km=target["range_km"],
                    aspect_deg=target["aspect_deg"],
                    heading_deg=target["heading_deg"],
                    name=target["name"],
                )

            # Set environment
            env_type = EnvironmentType(self.environment_var.get())
            bg_type = BackgroundType(self.background_var.get())
            builder.set_environment(env_type=env_type)
            builder.set_background(bg_type)

            # Set platform if enabled
            if self.platform_enabled_var.get():
                builder.set_platform(
                    platform_type=self.platform_type_var.get(),
                    speed_ms=float(self.platform_speed_var.get()),
                    heading_deg=float(self.platform_heading_var.get()),
                    orientation_deg=(
                        float(self.roll_var.get()),
                        float(self.pitch_var.get()),
                        float(self.yaw_var.get()),
                    ),
                )

                gimbal_mode = self.gimbal_mode_var.get()
                if gimbal_mode == "track_target":
                    builder.set_gimbal_track(int(self.track_target_var.get()))

                if self.orbit_enabled_var.get():
                    builder.set_orbit(
                        center=Position3D(0, 0, 0, "m"),
                        radius_m=float(self.orbit_radius_var.get()),
                    )

            # Set seed
            builder.set_seed(int(self.seed_var.get()))

            # Build and run
            scenario = builder.build()
            self._scenario = scenario

            result = run_scenario(scenario, verbose=False)
            self._result = result

            # Update UI in main thread
            self.root.after(0, self._update_results)

        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))

        finally:
            self.root.after(0, self._finish_run)

    def _update_results(self):
        """Update the results display."""
        if self._result is None:
            return

        # Update image
        self._display_image(self._result.digital_image)

        # Update results text
        metadata = self._result.metadata
        results_text = f"""Scenario: {metadata.get('scenario_name', 'N/A')}
Sensor: {metadata.get('sensor_name', 'N/A')}
Targets: {metadata.get('n_targets', 0)}
Range: {metadata.get('range_km', 0):.1f} km
Image Shape: {self._result.digital_image.shape}
"""
        if "platform" in metadata:
            platform = metadata["platform"]
            results_text += f"""
Platform: {platform.get('type', 'N/A')}
Gimbal Mode: {platform.get('gimbal_mode', 'N/A')}
Gimbal Angles: Az={platform['gimbal_angles'][0]:.1f}, El={platform['gimbal_angles'][1]:.1f}
"""

        if self._result.detection_metrics:
            dm = self._result.detection_metrics
            results_text += f"""
Detection Metrics:
  SNR: {dm.get('snr_db', 0):.1f} dB
  Contrast: {dm.get('contrast', 0):.3f}
"""

        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results_text)
        self.results_text.config(state=tk.DISABLED)

        self._update_status("Scenario completed successfully")

    def _display_image(self, image: np.ndarray):
        """Display image on canvas."""
        try:
            from PIL import Image, ImageTk

            # Normalize and convert to 8-bit
            img_normalized = image.astype(np.float64)
            img_min, img_max = img_normalized.min(), img_normalized.max()
            if img_max > img_min:
                img_normalized = (img_normalized - img_min) / (img_max - img_min) * 255
            img_8bit = img_normalized.astype(np.uint8)

            # Create PIL image
            pil_image = Image.fromarray(img_8bit)

            # Resize to fit canvas
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                pil_image = pil_image.resize((canvas_w, canvas_h), Image.Resampling.NEAREST)

            # Convert to PhotoImage
            self._photo = ImageTk.PhotoImage(pil_image)

            # Clear and display
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_w // 2, canvas_h // 2,
                image=self._photo, anchor=tk.CENTER
            )

        except ImportError:
            # Fallback if PIL not available
            self.canvas.delete("all")
            self.canvas.create_text(
                320, 240, text="PIL not available for image display",
                fill="red", font=("Segoe UI", 12)
            )

    def _finish_run(self):
        """Clean up after scenario run."""
        self.progress.stop()
        self.run_btn.config(state=tk.NORMAL)

    def _show_error(self, message: str):
        """Show error message."""
        messagebox.showerror("Error", f"Scenario failed:\n{message}")
        self._update_status("Error running scenario")

    def _update_status(self, message: str):
        """Update status bar message."""
        self.status_label.config(text=message)

    # File operations
    def _new_scenario(self):
        """Create a new scenario."""
        self._clear_targets()
        self.sensor_var.set("mx15")
        self.altitude_var.set("3000")
        self.platform_enabled_var.set(False)
        self._on_platform_toggle()
        self._result = None
        self._scenario = None
        self.canvas.delete("all")
        self.canvas.create_text(
            320, 240, text="Run scenario to see preview",
            fill="gray", font=("Segoe UI", 14)
        )
        self._update_status("New scenario created")

    def _load_scenario(self):
        """Load scenario from file."""
        messagebox.showinfo("Info", "Load scenario feature coming soon")

    def _save_scenario(self):
        """Save scenario to file."""
        messagebox.showinfo("Info", "Save scenario feature coming soon")

    def _export_image(self):
        """Export result image to file."""
        if self._result is None:
            messagebox.showwarning("Warning", "No result to export")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("TIFF files", "*.tiff"), ("All files", "*.*")]
        )

        if filename:
            try:
                from PIL import Image

                img = self._result.digital_image.astype(np.float64)
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min) * 255
                img_8bit = img.astype(np.uint8)

                pil_image = Image.fromarray(img_8bit)
                pil_image.save(filename)

                self._update_status(f"Exported to {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")

    def _open_object_viewer(self):
        """Open the 3D object viewer."""
        try:
            from eosim.library.object_viewer import ObjectViewer3D
            viewer = ObjectViewer3D(self.root)
            self._update_status("Object Viewer opened")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Object Viewer: {e}")

    def _open_sensor_browser(self):
        """Open the sensor browser."""
        # Create a simple sensor info dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Sensor Browser")
        dialog.geometry("600x500")

        ttk.Label(dialog, text="Sensor Library", font=("Segoe UI", 12, "bold")).pack(pady=10)

        # Sensor listbox
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        sensor_list = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=("Consolas", 10))
        sensor_list.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=sensor_list.yview)

        for sensor_id in self.sensors:
            sensor_list.insert(tk.END, sensor_id)

        # Info display
        info_text = tk.Text(dialog, height=12, state=tk.DISABLED, font=("Consolas", 9))
        info_text.pack(fill=tk.X, padx=10, pady=5)

        def on_select(event):
            selection = sensor_list.curselection()
            if not selection:
                return
            sensor_id = sensor_list.get(selection[0])
            try:
                from eosim.library import get_sensor
                spec = get_sensor(sensor_id)
                info = f"""Name: {spec.name}
Manufacturer: {spec.manufacturer}
Type: {spec.sensor_type.value}
Mount: {spec.mount_type.value}

IR Detector:
  Resolution: {spec.detector_ir.width_pixels}x{spec.detector_ir.height_pixels}
  Band: {spec.detector_ir.spectral_band_um[0]}-{spec.detector_ir.spectral_band_um[1]} um
  Pixel Pitch: {spec.detector_ir.pixel_pitch_um} um
  NETD: {spec.detector_ir.nedt_mk} mK

Optics:
  Focal Length: {spec.optics_ir.focal_length_mm} mm
  Aperture: {spec.optics_ir.aperture_mm} mm
  FOV: {spec.optics_ir.fov_narrow_deg}-{spec.optics_ir.fov_wide_deg} deg"""
                info_text.config(state=tk.NORMAL)
                info_text.delete(1.0, tk.END)
                info_text.insert(tk.END, info)
                info_text.config(state=tk.DISABLED)
            except Exception as e:
                info_text.config(state=tk.NORMAL)
                info_text.delete(1.0, tk.END)
                info_text.insert(tk.END, f"Error: {e}")
                info_text.config(state=tk.DISABLED)

        sensor_list.bind("<<ListboxSelect>>", on_select)

        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)

    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About EOSIM Scenario Builder",
            "EOSIM Scenario Builder\n\n"
            "A graphical interface for building and running\n"
            "EO/IR sensor simulation scenarios.\n\n"
            "Features:\n"
            "- Sensor library selection\n"
            "- Target object configuration\n"
            "- 6DOF platform motion\n"
            "- Environment settings\n"
            "- Real-time preview\n"
            "- 3D Object Viewer\n\n"
            "Version 1.1"
        )

    def _on_close(self):
        """Handle window close."""
        self.root.destroy()


def launch_scenario_builder():
    """Launch the scenario builder GUI.

    Example:
        >>> from eosim.library.gui import launch_scenario_builder
        >>> launch_scenario_builder()
    """
    root = tk.Tk()
    app = ScenarioBuilderGUI(root)
    root.mainloop()


# Allow running as script
if __name__ == "__main__":
    launch_scenario_builder()
