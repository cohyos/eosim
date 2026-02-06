"""
EOSIM 3D Object Viewer.

Provides visualization of 3D objects from the object library, showing:
- 3D wireframe/surface views
- Multiple viewing angles
- Thermal profile information
- Object dimensions

Usage:
------
>>> from eosim.library.object_viewer import launch_object_viewer
>>> launch_object_viewer()

Or view a specific object:
>>> from eosim.library.object_viewer import ObjectViewer3D
>>> viewer = ObjectViewer3D()
>>> viewer.show_object("f16")
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from numpy.typing import NDArray


class ObjectViewer3D:
    """3D object viewer with multiple visualization modes."""

    def __init__(self, root: Optional[tk.Tk] = None):
        """Initialize the object viewer.

        Args:
            root: Optional Tkinter root window. If None, creates new window.
        """
        self.standalone = root is None
        if self.standalone:
            self.root = tk.Tk()
            self.root.title("EOSIM 3D Object Viewer")
        else:
            self.root = tk.Toplevel(root)
            self.root.title("Object Viewer")

        self.root.geometry("1000x700")
        self.root.minsize(800, 600)

        # Current state
        self.current_object_id: Optional[str] = None
        self.current_object = None
        self.azimuth = 45.0
        self.elevation = 30.0

        # Load library
        self._load_library()

        # Build UI
        self._build_ui()

    def _load_library(self):
        """Load object library data."""
        try:
            from eosim.library import list_objects, list_categories, get_object
            self.objects = list_objects()
            self.categories = list_categories()
            self.get_object = get_object
        except ImportError:
            self.objects = []
            self.categories = []
            self.get_object = None

    def _build_ui(self):
        """Build the user interface."""
        # Main layout
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Object selection
        left_frame = ttk.Frame(main_paned, width=250)
        main_paned.add(left_frame, weight=1)

        # Right panel - Viewer
        right_frame = ttk.Frame(main_paned, width=700)
        main_paned.add(right_frame, weight=3)

        self._build_selection_panel(left_frame)
        self._build_viewer_panel(right_frame)

    def _build_selection_panel(self, parent):
        """Build the object selection panel."""
        # Title
        ttk.Label(parent, text="Object Library", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, pady=5)

        # Category filter
        filter_frame = ttk.LabelFrame(parent, text="Filter", padding=5)
        filter_frame.pack(fill=tk.X, pady=5)

        ttk.Label(filter_frame, text="Category:").pack(anchor=tk.W)
        self.category_var = tk.StringVar(value="ALL")
        category_combo = ttk.Combobox(
            filter_frame, textvariable=self.category_var,
            values=["ALL"] + self.categories, state="readonly", width=20
        )
        category_combo.pack(fill=tk.X, pady=2)
        category_combo.bind("<<ComboboxSelected>>", self._on_category_changed)

        # Object list
        list_frame = ttk.LabelFrame(parent, text="Objects", padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Listbox with scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.object_listbox = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set,
            font=("Consolas", 10), height=20
        )
        self.object_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.object_listbox.yview)

        self.object_listbox.bind("<<ListboxSelect>>", self._on_object_selected)

        # Populate list
        self._populate_object_list()

        # Object info
        info_frame = ttk.LabelFrame(parent, text="Object Info", padding=5)
        info_frame.pack(fill=tk.X, pady=5)

        self.info_text = tk.Text(info_frame, height=10, width=30, state=tk.DISABLED,
                                 font=("Consolas", 9))
        self.info_text.pack(fill=tk.X)

    def _build_viewer_panel(self, parent):
        """Build the 3D viewer panel."""
        # Title
        ttk.Label(parent, text="3D Preview", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, pady=5)

        # View controls
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, pady=5)

        ttk.Label(controls_frame, text="Azimuth:").pack(side=tk.LEFT, padx=5)
        self.azimuth_var = tk.DoubleVar(value=45.0)
        azimuth_scale = ttk.Scale(
            controls_frame, from_=0, to=360, variable=self.azimuth_var,
            orient=tk.HORIZONTAL, length=150, command=self._on_view_changed
        )
        azimuth_scale.pack(side=tk.LEFT, padx=5)

        ttk.Label(controls_frame, text="Elevation:").pack(side=tk.LEFT, padx=5)
        self.elevation_var = tk.DoubleVar(value=30.0)
        elevation_scale = ttk.Scale(
            controls_frame, from_=-90, to=90, variable=self.elevation_var,
            orient=tk.HORIZONTAL, length=150, command=self._on_view_changed
        )
        elevation_scale.pack(side=tk.LEFT, padx=5)

        # Preset views
        preset_frame = ttk.Frame(parent)
        preset_frame.pack(fill=tk.X, pady=5)

        ttk.Label(preset_frame, text="Views:").pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="Front", command=lambda: self._set_view(0, 0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Side", command=lambda: self._set_view(90, 0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Rear", command=lambda: self._set_view(180, 0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Top", command=lambda: self._set_view(0, 90)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="3/4", command=lambda: self._set_view(45, 30)).pack(side=tk.LEFT, padx=2)

        # Canvas for 3D view
        canvas_frame = ttk.LabelFrame(parent, text="3D View", padding=5)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg="black", width=600, height=400)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Placeholder text
        self.canvas.create_text(
            300, 200, text="Select an object to view",
            fill="gray", font=("Segoe UI", 14), tags="placeholder"
        )

        # Multi-view panel
        multiview_frame = ttk.LabelFrame(parent, text="Multi-Angle Views", padding=5)
        multiview_frame.pack(fill=tk.X, pady=5)

        self.multiview_canvas = tk.Canvas(multiview_frame, bg="black", height=120)
        self.multiview_canvas.pack(fill=tk.X)

        # Thermal info
        thermal_frame = ttk.LabelFrame(parent, text="Thermal Profile", padding=5)
        thermal_frame.pack(fill=tk.X, pady=5)

        self.thermal_canvas = tk.Canvas(thermal_frame, bg="#1a1a2e", height=80)
        self.thermal_canvas.pack(fill=tk.X)

    def _populate_object_list(self, category: str = "ALL"):
        """Populate the object listbox."""
        self.object_listbox.delete(0, tk.END)

        if category == "ALL":
            objects = self.objects
        else:
            try:
                from eosim.library.objects import ObjectLibrary
                filtered = ObjectLibrary.get_by_category(category)
                objects = [obj.id for obj in filtered]
            except Exception:
                objects = self.objects

        for obj_id in sorted(objects):
            self.object_listbox.insert(tk.END, obj_id)

    def _on_category_changed(self, event=None):
        """Handle category filter change."""
        category = self.category_var.get()
        self._populate_object_list(category)

    def _on_object_selected(self, event=None):
        """Handle object selection."""
        selection = self.object_listbox.curselection()
        if not selection:
            return

        obj_id = self.object_listbox.get(selection[0])
        self.show_object(obj_id)

    def _on_view_changed(self, event=None):
        """Handle view angle change."""
        self.azimuth = self.azimuth_var.get()
        self.elevation = self.elevation_var.get()
        self._update_view()

    def _set_view(self, azimuth: float, elevation: float):
        """Set view angles."""
        self.azimuth_var.set(azimuth)
        self.elevation_var.set(elevation)
        self.azimuth = azimuth
        self.elevation = elevation
        self._update_view()

    def show_object(self, object_id: str):
        """Show a specific object.

        Args:
            object_id: ID of the object to display
        """
        if self.get_object is None:
            return

        try:
            self.current_object = self.get_object(object_id)
            self.current_object_id = object_id
            self._update_info()
            self._update_view()
            self._update_multiview()
            self._update_thermal_profile()
        except Exception as e:
            self._show_error(f"Failed to load object: {e}")

    def _update_info(self):
        """Update object information display."""
        if self.current_object is None:
            return

        obj = self.current_object
        dims = obj.dimensions

        info = f"""ID: {obj.id}
Name: {obj.name}
Category: {obj.category.value}

Dimensions:
  Length: {dims.length_m:.1f} m
  Width:  {dims.width_m:.1f} m
  Height: {dims.height_m:.1f} m

Thermal:
  Base: {obj.thermal.base_temperature_k:.0f} K
  Engine: {obj.thermal.engine_temperature_k:.0f} K
  Plume: {obj.thermal.exhaust_temperature_k:.0f} K
  Emissivity: {obj.thermal.emissivity:.2f}

{obj.description[:100]}..."""

        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info)
        self.info_text.config(state=tk.DISABLED)

    def _update_view(self):
        """Update the main 3D view."""
        if self.current_object is None:
            return

        self.canvas.delete("all")

        # Get canvas dimensions
        self.canvas.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        if cw < 10 or ch < 10:
            cw, ch = 600, 400

        # Generate silhouette at current angle
        resolution = (ch - 40, cw - 40)
        try:
            temp_map, emis_map = self.current_object.get_signature(
                resolution=resolution,
                aspect_angle_deg=self.azimuth,
                elevation_angle_deg=self.elevation,
            )

            # Convert to display
            self._draw_thermal_image(self.canvas, temp_map, 20, 20, cw - 40, ch - 40)

        except Exception as e:
            self.canvas.create_text(
                cw // 2, ch // 2, text=f"Error: {e}",
                fill="red", font=("Segoe UI", 10)
            )

        # Draw info overlay
        self.canvas.create_text(
            10, 10, text=f"{self.current_object.name}", anchor=tk.NW,
            fill="white", font=("Segoe UI", 12, "bold")
        )
        self.canvas.create_text(
            10, ch - 10, text=f"Az: {self.azimuth:.0f}° El: {self.elevation:.0f}°",
            anchor=tk.SW, fill="gray", font=("Segoe UI", 9)
        )

        # Draw dimension lines
        dims = self.current_object.dimensions
        self.canvas.create_text(
            cw - 10, 10, text=f"L: {dims.length_m:.1f}m × W: {dims.width_m:.1f}m × H: {dims.height_m:.1f}m",
            anchor=tk.NE, fill="cyan", font=("Segoe UI", 9)
        )

    def _update_multiview(self):
        """Update the multi-angle view panel."""
        if self.current_object is None:
            return

        self.multiview_canvas.delete("all")

        # Get canvas dimensions
        self.multiview_canvas.update_idletasks()
        cw = self.multiview_canvas.winfo_width()
        ch = self.multiview_canvas.winfo_height()

        if cw < 10:
            cw = 600
        if ch < 10:
            ch = 120

        # Draw views at 0, 45, 90, 135, 180 degrees
        angles = [0, 45, 90, 135, 180]
        view_width = (cw - 20) // len(angles)
        view_height = ch - 20

        for i, angle in enumerate(angles):
            x_offset = 10 + i * view_width
            y_offset = 10

            try:
                resolution = (view_height - 20, view_width - 20)
                temp_map, _ = self.current_object.get_signature(
                    resolution=resolution,
                    aspect_angle_deg=angle,
                    elevation_angle_deg=0,
                )
                self._draw_thermal_image(
                    self.multiview_canvas, temp_map,
                    x_offset + 10, y_offset,
                    view_width - 20, view_height - 20,
                    simple=True
                )
            except Exception:
                pass

            # Label
            self.multiview_canvas.create_text(
                x_offset + view_width // 2, ch - 5,
                text=f"{angle}°", fill="gray", font=("Segoe UI", 8)
            )

    def _update_thermal_profile(self):
        """Update thermal profile display."""
        if self.current_object is None:
            return

        self.thermal_canvas.delete("all")

        cw = self.thermal_canvas.winfo_width()
        ch = self.thermal_canvas.winfo_height()

        if cw < 10:
            cw = 600
        if ch < 10:
            ch = 80

        thermal = self.current_object.thermal

        # Draw temperature bars
        temps = [
            ("Base", thermal.base_temperature_k, "#4a69bd"),
            ("Engine", thermal.engine_temperature_k, "#e55039"),
            ("Exhaust", thermal.exhaust_temperature_k, "#f39c12"),
        ]

        bar_width = 80
        bar_height = 40
        spacing = 20

        max_temp = max(t[1] for t in temps)
        start_x = 10

        for i, (label, temp, color) in enumerate(temps):
            x = start_x + i * (bar_width + spacing)

            # Background bar
            self.thermal_canvas.create_rectangle(
                x, 30, x + bar_width, 30 + bar_height,
                outline="gray", fill="#2d2d2d"
            )

            # Temperature bar
            fill_height = int(bar_height * temp / max_temp)
            self.thermal_canvas.create_rectangle(
                x, 30 + bar_height - fill_height,
                x + bar_width, 30 + bar_height,
                fill=color, outline=""
            )

            # Label
            self.thermal_canvas.create_text(
                x + bar_width // 2, 15, text=label,
                fill="white", font=("Segoe UI", 9)
            )
            self.thermal_canvas.create_text(
                x + bar_width // 2, 30 + bar_height + 15,
                text=f"{temp:.0f} K", fill="white", font=("Segoe UI", 8)
            )

        # Emissivity display
        x = start_x + 3 * (bar_width + spacing)
        self.thermal_canvas.create_text(
            x, 15, text="Emissivity", fill="white", font=("Segoe UI", 9), anchor=tk.W
        )
        self.thermal_canvas.create_text(
            x, 45, text=f"{thermal.emissivity:.2f}", fill="cyan",
            font=("Segoe UI", 14, "bold"), anchor=tk.W
        )

        # Hot spots count
        if thermal.hot_spots:
            x = start_x + 4 * (bar_width + spacing)
            self.thermal_canvas.create_text(
                x, 15, text="Hot Spots", fill="white", font=("Segoe UI", 9), anchor=tk.W
            )
            self.thermal_canvas.create_text(
                x, 45, text=f"{len(thermal.hot_spots)}", fill="orange",
                font=("Segoe UI", 14, "bold"), anchor=tk.W
            )

    def _draw_thermal_image(
        self,
        canvas: tk.Canvas,
        temp_map: NDArray,
        x: int, y: int,
        width: int, height: int,
        simple: bool = False
    ):
        """Draw thermal image on canvas using rectangles."""
        if temp_map is None or temp_map.size == 0:
            return

        h, w = temp_map.shape

        # Normalize temperature map
        t_min, t_max = temp_map.min(), temp_map.max()
        if t_max > t_min:
            normalized = (temp_map - t_min) / (t_max - t_min)
        else:
            normalized = np.zeros_like(temp_map)

        # Downsample for performance
        if simple:
            step = max(1, min(h, w) // 20)
        else:
            step = max(1, min(h, w) // 80)

        scale_x = width / w
        scale_y = height / h

        # Draw pixels as rectangles
        for iy in range(0, h, step):
            for ix in range(0, w, step):
                val = normalized[iy, ix]
                if val > 0.01:  # Only draw non-background
                    # Iron colormap approximation
                    r, g, b = self._iron_colormap(val)
                    color = f"#{r:02x}{g:02x}{b:02x}"

                    px = x + int(ix * scale_x)
                    py = y + int(iy * scale_y)
                    pw = max(1, int(step * scale_x))
                    ph = max(1, int(step * scale_y))

                    canvas.create_rectangle(
                        px, py, px + pw, py + ph,
                        fill=color, outline=""
                    )

    def _iron_colormap(self, value: float) -> Tuple[int, int, int]:
        """Convert value (0-1) to iron colormap RGB."""
        # Iron colormap: black -> purple -> red -> orange -> yellow -> white
        value = max(0, min(1, value))

        if value < 0.25:
            # Black to purple
            t = value / 0.25
            r = int(128 * t)
            g = 0
            b = int(128 * t)
        elif value < 0.5:
            # Purple to red
            t = (value - 0.25) / 0.25
            r = int(128 + 127 * t)
            g = 0
            b = int(128 * (1 - t))
        elif value < 0.75:
            # Red to orange/yellow
            t = (value - 0.5) / 0.25
            r = 255
            g = int(200 * t)
            b = 0
        else:
            # Yellow to white
            t = (value - 0.75) / 0.25
            r = 255
            g = int(200 + 55 * t)
            b = int(200 * t)

        return (r, g, b)

    def _show_error(self, message: str):
        """Show error on canvas."""
        self.canvas.delete("all")
        self.canvas.create_text(
            300, 200, text=message, fill="red", font=("Segoe UI", 12)
        )

    def run(self):
        """Run the viewer (standalone mode)."""
        if self.standalone:
            self.root.mainloop()


def launch_object_viewer():
    """Launch the standalone 3D object viewer.

    Example:
        >>> from eosim.library.object_viewer import launch_object_viewer
        >>> launch_object_viewer()
    """
    viewer = ObjectViewer3D()
    viewer.run()


# Allow running as script
if __name__ == "__main__":
    launch_object_viewer()
