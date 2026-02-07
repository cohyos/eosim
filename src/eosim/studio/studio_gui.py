"""
EOSIM Studio - Main GUI Application.

A video-production-style interface for creating sensor simulation videos.
The interface includes:
- Scene View: 3D overhead view of the scene
- Camera View: Live preview of what the camera sees
- Timeline: Video-editor style timeline with keyframes
- Properties: Camera and object settings
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Tuple, Dict, Any
import threading
import numpy as np

from eosim.studio.project import Project, create_demo_project
from eosim.studio.scene import Scene, SceneObject, Position3D, Orientation3D
from eosim.studio.camera import Camera, LensType, SpectrumMode, SensitivityLevel, CAMERA_PRESETS
from eosim.studio.timeline import Timeline, PlaybackState
from eosim.studio.renderer import Renderer, RenderedFrame


class TimelineWidget(ttk.Frame):
    """Video-editor style timeline widget."""

    def __init__(self, parent, timeline: Timeline, on_seek=None):
        super().__init__(parent)
        self.timeline = timeline
        self.on_seek = on_seek

        self._build_ui()
        self._bind_events()

    def _build_ui(self):
        # Transport controls
        transport = ttk.Frame(self)
        transport.pack(fill=tk.X, pady=2)

        self.btn_start = ttk.Button(transport, text="|<", width=3,
                                    command=self.timeline.go_to_start)
        self.btn_start.pack(side=tk.LEFT, padx=1)

        self.btn_prev = ttk.Button(transport, text="<", width=3,
                                   command=lambda: self.timeline.step_backward(10))
        self.btn_prev.pack(side=tk.LEFT, padx=1)

        self.btn_play = ttk.Button(transport, text="Play", width=6,
                                   command=self._toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=1)

        self.btn_next = ttk.Button(transport, text=">", width=3,
                                   command=lambda: self.timeline.step_forward(10))
        self.btn_next.pack(side=tk.LEFT, padx=1)

        self.btn_end = ttk.Button(transport, text=">|", width=3,
                                  command=self.timeline.go_to_end)
        self.btn_end.pack(side=tk.LEFT, padx=1)

        # Time display
        self.time_var = tk.StringVar(value="00:00.00")
        ttk.Label(transport, textvariable=self.time_var, width=10,
                 font=("Consolas", 11)).pack(side=tk.LEFT, padx=10)

        # Frame display
        self.frame_var = tk.StringVar(value="F: 0")
        ttk.Label(transport, textvariable=self.frame_var, width=10,
                 font=("Consolas", 9)).pack(side=tk.LEFT)

        # Loop toggle
        self.loop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(transport, text="Loop", variable=self.loop_var,
                       command=self._on_loop_changed).pack(side=tk.RIGHT, padx=5)

        # Timeline scrubber
        scrub_frame = ttk.Frame(self)
        scrub_frame.pack(fill=tk.X, pady=2)

        self.scrubber = ttk.Scale(scrub_frame, from_=0, to=self.timeline.duration_sec,
                                  orient=tk.HORIZONTAL, command=self._on_scrub)
        self.scrubber.pack(fill=tk.X, padx=5)

        # Timeline canvas (for keyframes/markers)
        self.canvas = tk.Canvas(self, height=40, bg="#2d2d2d")
        self.canvas.pack(fill=tk.X, padx=5, pady=2)

        # Duration labels
        dur_frame = ttk.Frame(self)
        dur_frame.pack(fill=tk.X, padx=5)
        ttk.Label(dur_frame, text="0:00").pack(side=tk.LEFT)
        dur_str = f"{int(self.timeline.duration_sec//60)}:{int(self.timeline.duration_sec%60):02d}"
        ttk.Label(dur_frame, text=dur_str).pack(side=tk.RIGHT)

    def _bind_events(self):
        self.timeline.on_time_change(self._on_time_changed)
        self.timeline.on_state_change(self._on_state_changed)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

    def _toggle_play(self):
        self.timeline.toggle_playback()

    def _on_loop_changed(self):
        self.timeline.loop = self.loop_var.get()

    def _on_scrub(self, value):
        time_sec = float(value)
        self.timeline.seek(time_sec)
        if self.on_seek:
            self.on_seek(time_sec)

    def _on_time_changed(self, time_sec: float):
        # Update time display
        mins = int(time_sec // 60)
        secs = time_sec % 60
        self.time_var.set(f"{mins:02d}:{secs:05.2f}")
        self.frame_var.set(f"F: {self.timeline.current_frame}")

        # Update scrubber (without triggering callback)
        self.scrubber.set(time_sec)

        # Update canvas playhead
        self._draw_timeline()

    def _on_state_changed(self, state: PlaybackState):
        if state == PlaybackState.PLAYING:
            self.btn_play.config(text="Pause")
        else:
            self.btn_play.config(text="Play")

    def _on_canvas_click(self, event):
        # Click to seek
        width = self.canvas.winfo_width()
        if width > 0:
            time_sec = (event.x / width) * self.timeline.duration_sec
            self.timeline.seek(time_sec)

    def _draw_timeline(self):
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        if width < 10:
            return

        # Draw background
        self.canvas.create_rectangle(0, 0, width, height, fill="#2d2d2d", outline="")

        # Draw tick marks
        num_ticks = 10
        for i in range(num_ticks + 1):
            x = int(i * width / num_ticks)
            tick_height = 10 if i % 5 == 0 else 5
            self.canvas.create_line(x, height, x, height - tick_height, fill="#666666")

        # Draw markers
        for marker in self.timeline.markers:
            x = int((marker.time_sec / self.timeline.duration_sec) * width)
            self.canvas.create_polygon(
                x-5, 5, x+5, 5, x, 15,
                fill=marker.color, outline=""
            )

        # Draw keyframes
        for target_id, kf_list in self.timeline.keyframes.items():
            for kf in kf_list:
                x = int((kf.time_sec / self.timeline.duration_sec) * width)
                self.canvas.create_oval(
                    x-3, height//2-3, x+3, height//2+3,
                    fill="#00aaff", outline="#ffffff"
                )

        # Draw playhead
        x = int((self.timeline.current_time / self.timeline.duration_sec) * width)
        self.canvas.create_line(x, 0, x, height, fill="#ff4444", width=2)

    def update_duration(self, duration_sec: float):
        """Update timeline duration."""
        self.timeline.duration_sec = duration_sec
        self.scrubber.config(to=duration_sec)


class CameraViewWidget(ttk.Frame):
    """Live camera preview widget."""

    def __init__(self, parent, renderer: Renderer):
        super().__init__(parent)
        self.renderer = renderer

        self._build_ui()
        self._current_frame: Optional[RenderedFrame] = None

    def _build_ui(self):
        # Header
        header = ttk.Frame(self)
        header.pack(fill=tk.X, pady=2)
        ttk.Label(header, text="Camera View", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)

        # Camera info
        self.info_var = tk.StringVar(value="")
        ttk.Label(header, textvariable=self.info_var,
                 font=("Consolas", 9)).pack(side=tk.RIGHT)

        # Canvas for rendering
        self.canvas = tk.Canvas(self, bg="black", width=640, height=480)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Colormap selector
        cmap_frame = ttk.Frame(self)
        cmap_frame.pack(fill=tk.X, pady=2)
        ttk.Label(cmap_frame, text="Display:").pack(side=tk.LEFT, padx=5)

        self.colormap_var = tk.StringVar(value="iron")
        for cmap in ["iron", "rainbow", "grayscale", "hot", "green"]:
            ttk.Radiobutton(cmap_frame, text=cmap.title(), value=cmap,
                           variable=self.colormap_var,
                           command=self._on_colormap_changed).pack(side=tk.LEFT)

    def _on_colormap_changed(self):
        self.renderer.camera.colormap = self.colormap_var.get()
        if self._current_frame:
            self.render_at(self._current_frame.time_sec)

    def render_at(self, time_sec: float):
        """Render frame at given time and display."""
        try:
            frame = self.renderer.render_frame(time_sec, int(time_sec * 30))
            self._current_frame = frame
            self._display_frame(frame)
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(
                320, 240, text=f"Render error: {e}",
                fill="red", font=("Segoe UI", 10)
            )

    def _display_frame(self, frame: RenderedFrame):
        """Display rendered frame on canvas."""
        self.canvas.delete("all")

        # Get canvas size
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 640, 480

        # Scale image to fit
        img = frame.image
        ih, iw = img.shape[:2]

        # Calculate scaling
        scale = min(cw / iw, ch / ih)
        new_w = int(iw * scale)
        new_h = int(ih * scale)

        # Draw pixels (simplified - in production use PIL)
        step = max(1, min(ih, iw) // 100)
        x_off = (cw - new_w) // 2
        y_off = (ch - new_h) // 2

        for y in range(0, ih, step):
            for x in range(0, iw, step):
                r, g, b = img[y, x]
                color = f"#{r:02x}{g:02x}{b:02x}"

                px = x_off + int(x * scale)
                py = y_off + int(y * scale)
                pw = max(1, int(step * scale))
                ph = max(1, int(step * scale))

                self.canvas.create_rectangle(
                    px, py, px + pw, py + ph,
                    fill=color, outline=""
                )

        # Draw crosshair
        cx = cw // 2
        cy = ch // 2
        self.canvas.create_line(cx - 20, cy, cx + 20, cy, fill="#00ff00", width=1)
        self.canvas.create_line(cx, cy - 20, cx, cy + 20, fill="#00ff00", width=1)

        # Update info
        if frame.camera_position:
            pos = frame.camera_position
            self.info_var.set(
                f"T: {frame.time_sec:.2f}s | "
                f"Pos: ({pos.x/1000:.1f}, {pos.y/1000:.1f}, {pos.z/1000:.1f})km"
            )


class SceneViewWidget(ttk.Frame):
    """3D overhead scene view."""

    def __init__(self, parent, scene: Scene, camera: Camera):
        super().__init__(parent)
        self.scene = scene
        self.camera = camera
        self.current_time = 0.0

        self._build_ui()

    def _build_ui(self):
        # Header
        header = ttk.Frame(self)
        header.pack(fill=tk.X, pady=2)
        ttk.Label(header, text="Scene View", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)

        # Canvas
        self.canvas = tk.Canvas(self, bg="#1a1a2e", width=400, height=300)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Bind events
        self.canvas.bind("<Configure>", lambda e: self.update_view())

    def update_view(self, time_sec: float = None):
        """Update the scene view."""
        if time_sec is not None:
            self.current_time = time_sec

        self.canvas.delete("all")

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        # Get scene bounds
        min_pos, max_pos = self.scene.get_bounds()

        # Calculate scale
        scene_w = max_pos.x - min_pos.x
        scene_h = max_pos.y - min_pos.y
        if scene_w < 1:
            scene_w = 2000
        if scene_h < 1:
            scene_h = 2000

        margin = 50
        scale = min((cw - 2*margin) / scene_w, (ch - 2*margin) / scene_h)

        def world_to_screen(pos: Position3D) -> Tuple[int, int]:
            x = margin + int((pos.x - min_pos.x) * scale)
            y = ch - margin - int((pos.y - min_pos.y) * scale)
            return (x, y)

        # Draw grid
        grid_step = 1000  # 1km grid
        for gx in range(int(min_pos.x // grid_step) * int(grid_step),
                       int(max_pos.x) + int(grid_step), int(grid_step)):
            x, _ = world_to_screen(Position3D(gx, 0, 0))
            self.canvas.create_line(x, margin, x, ch - margin, fill="#333344")

        for gy in range(int(min_pos.y // grid_step) * int(grid_step),
                       int(max_pos.y) + int(grid_step), int(grid_step)):
            _, y = world_to_screen(Position3D(0, gy, 0))
            self.canvas.create_line(margin, y, cw - margin, y, fill="#333344")

        # Draw objects
        for obj in self.scene.objects.values():
            if not obj.visible:
                continue

            pos = obj.get_position_at(self.current_time)
            ori = obj.get_orientation_at(self.current_time)

            sx, sy = world_to_screen(pos)

            # Color by type
            if "f16" in obj.object_type or "aircraft" in obj.object_type.lower():
                color = "#00aaff"
                size = 12
            elif "tank" in obj.object_type or "abrams" in obj.object_type:
                color = "#ff8800"
                size = 10
            elif "soldier" in obj.object_type or "person" in obj.object_type:
                color = "#00ff88"
                size = 6
            else:
                color = "#aaaaaa"
                size = 8

            # Draw object
            self.canvas.create_oval(
                sx - size, sy - size, sx + size, sy + size,
                fill=color, outline="white"
            )

            # Draw heading indicator
            angle_rad = np.radians(90 - ori.heading)  # Convert to math coords
            dx = np.cos(angle_rad) * size * 2
            dy = -np.sin(angle_rad) * size * 2
            self.canvas.create_line(sx, sy, sx + dx, sy + dy, fill="white", width=2)

            # Label
            self.canvas.create_text(sx, sy - size - 8, text=obj.name,
                                   fill="white", font=("Segoe UI", 8))

        # Draw camera
        cam_pos = self.camera.get_position_at(self.current_time)
        cam_ori = self.camera.get_orientation_at(self.current_time, self.scene.objects)

        cx, cy = world_to_screen(cam_pos)

        # Camera icon
        self.canvas.create_rectangle(cx - 8, cy - 6, cx + 8, cy + 6,
                                    fill="#ff4444", outline="white", width=2)

        # FOV cone
        fov = self.camera.get_fov()
        cone_length = 80
        half_fov = np.radians(fov / 2)
        cam_heading = np.radians(90 - cam_ori.heading)

        left_angle = cam_heading + half_fov
        right_angle = cam_heading - half_fov

        lx = cx + cone_length * np.cos(left_angle)
        ly = cy - cone_length * np.sin(left_angle)
        rx = cx + cone_length * np.cos(right_angle)
        ry = cy - cone_length * np.sin(right_angle)

        self.canvas.create_polygon(
            cx, cy, lx, ly, rx, ry,
            fill="#ff444433", outline="#ff4444"
        )


class CameraSettingsWidget(ttk.Frame):
    """Camera settings panel."""

    def __init__(self, parent, camera: Camera, on_change=None):
        super().__init__(parent)
        self.camera = camera
        self.on_change = on_change

        self._build_ui()

    def _build_ui(self):
        # Header
        ttk.Label(self, text="Camera Settings",
                 font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=5)

        # Presets
        preset_frame = ttk.LabelFrame(self, text="Presets", padding=5)
        preset_frame.pack(fill=tk.X, pady=5)

        self.preset_var = tk.StringVar(value="surveillance")
        for name, preset in CAMERA_PRESETS.items():
            ttk.Radiobutton(preset_frame, text=preset.name, value=name,
                           variable=self.preset_var,
                           command=self._on_preset_changed).pack(anchor=tk.W)

        # Basic settings
        basic_frame = ttk.LabelFrame(self, text="Basic", padding=5)
        basic_frame.pack(fill=tk.X, pady=5)

        # Lens
        ttk.Label(basic_frame, text="Lens:").pack(anchor=tk.W)
        self.lens_var = tk.StringVar(value=self.camera.lens.value)
        lens_combo = ttk.Combobox(basic_frame, textvariable=self.lens_var,
                                  values=[lt.value for lt in LensType],
                                  state="readonly", width=15)
        lens_combo.pack(fill=tk.X, pady=2)
        lens_combo.bind("<<ComboboxSelected>>", self._on_lens_changed)

        # Sensitivity
        ttk.Label(basic_frame, text="Sensitivity:").pack(anchor=tk.W, pady=(5, 0))
        self.sens_var = tk.StringVar(value=self.camera.sensitivity.value)
        sens_combo = ttk.Combobox(basic_frame, textvariable=self.sens_var,
                                  values=[s.value for s in SensitivityLevel],
                                  state="readonly", width=15)
        sens_combo.pack(fill=tk.X, pady=2)
        sens_combo.bind("<<ComboboxSelected>>", self._on_sens_changed)

        # Spectrum
        ttk.Label(basic_frame, text="Spectrum:").pack(anchor=tk.W, pady=(5, 0))
        self.spectrum_var = tk.StringVar(value=self.camera.spectrum.value)
        spectrum_combo = ttk.Combobox(basic_frame, textvariable=self.spectrum_var,
                                      values=[s.value for s in SpectrumMode],
                                      state="readonly", width=15)
        spectrum_combo.pack(fill=tk.X, pady=2)
        spectrum_combo.bind("<<ComboboxSelected>>", self._on_spectrum_changed)

        # Advanced settings (collapsible)
        self.advanced_visible = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Show Advanced",
                       variable=self.advanced_visible,
                       command=self._toggle_advanced).pack(anchor=tk.W, pady=5)

        self.advanced_frame = ttk.LabelFrame(self, text="Advanced", padding=5)

        # FOV slider
        ttk.Label(self.advanced_frame, text="FOV (degrees):").pack(anchor=tk.W)
        self.fov_var = tk.DoubleVar(value=self.camera.get_fov())
        self.fov_scale = ttk.Scale(self.advanced_frame, from_=5, to=120,
                                   variable=self.fov_var, orient=tk.HORIZONTAL,
                                   command=self._on_fov_changed)
        self.fov_scale.pack(fill=tk.X)
        self.fov_label = ttk.Label(self.advanced_frame, text=f"{self.camera.get_fov():.1f}°")
        self.fov_label.pack(anchor=tk.E)

        # NETD slider
        ttk.Label(self.advanced_frame, text="NETD (mK):").pack(anchor=tk.W, pady=(5, 0))
        self.netd_var = tk.DoubleVar(value=self.camera.get_netd())
        self.netd_scale = ttk.Scale(self.advanced_frame, from_=5, to=200,
                                    variable=self.netd_var, orient=tk.HORIZONTAL,
                                    command=self._on_netd_changed)
        self.netd_scale.pack(fill=tk.X)
        self.netd_label = ttk.Label(self.advanced_frame, text=f"{self.camera.get_netd():.0f} mK")
        self.netd_label.pack(anchor=tk.E)

    def _toggle_advanced(self):
        if self.advanced_visible.get():
            self.advanced_frame.pack(fill=tk.X, pady=5)
        else:
            self.advanced_frame.pack_forget()

    def _on_preset_changed(self):
        self.camera.set_preset(self.preset_var.get())
        self._update_ui()
        self._notify_change()

    def _on_lens_changed(self, event=None):
        self.camera.lens = LensType(self.lens_var.get())
        self.camera._fov_override = None
        self._update_ui()
        self._notify_change()

    def _on_sens_changed(self, event=None):
        self.camera.sensitivity = SensitivityLevel(self.sens_var.get())
        self.camera._netd_override = None
        self._update_ui()
        self._notify_change()

    def _on_spectrum_changed(self, event=None):
        self.camera.spectrum = SpectrumMode(self.spectrum_var.get())
        self._notify_change()

    def _on_fov_changed(self, value=None):
        self.camera.set_fov(self.fov_var.get())
        self.fov_label.config(text=f"{self.camera.get_fov():.1f}°")
        self._notify_change()

    def _on_netd_changed(self, value=None):
        self.camera.set_netd(self.netd_var.get())
        self.netd_label.config(text=f"{self.camera.get_netd():.0f} mK")
        self._notify_change()

    def _update_ui(self):
        self.lens_var.set(self.camera.lens.value)
        self.sens_var.set(self.camera.sensitivity.value)
        self.spectrum_var.set(self.camera.spectrum.value)
        self.fov_var.set(self.camera.get_fov())
        self.netd_var.set(self.camera.get_netd())
        self.fov_label.config(text=f"{self.camera.get_fov():.1f}°")
        self.netd_label.config(text=f"{self.camera.get_netd():.0f} mK")

    def _notify_change(self):
        if self.on_change:
            self.on_change()


class ObjectListWidget(ttk.Frame):
    """Scene object list."""

    def __init__(self, parent, scene: Scene, on_select=None):
        super().__init__(parent)
        self.scene = scene
        self.on_select = on_select

        self._build_ui()

    def _build_ui(self):
        # Header
        header = ttk.Frame(self)
        header.pack(fill=tk.X, pady=2)
        ttk.Label(header, text="Objects",
                 font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)

        ttk.Button(header, text="+", width=3,
                  command=self._add_object).pack(side=tk.RIGHT)

        # Listbox
        list_frame = ttk.Frame(self)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                  font=("Consolas", 10))
        self.listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)

        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        self.refresh()

    def refresh(self):
        """Refresh the object list."""
        self.listbox.delete(0, tk.END)
        for obj_id, obj in self.scene.objects.items():
            self.listbox.insert(tk.END, f"{obj.name} ({obj.object_type})")

    def _on_select(self, event):
        selection = self.listbox.curselection()
        if selection and self.on_select:
            idx = selection[0]
            obj_id = list(self.scene.objects.keys())[idx]
            self.on_select(obj_id)

    def _add_object(self):
        # Simple dialog to add object
        dialog = tk.Toplevel(self)
        dialog.title("Add Object")
        dialog.geometry("300x200")

        ttk.Label(dialog, text="Object Type:").pack(pady=5)
        type_var = tk.StringVar(value="f16")
        types = ["f16", "f22", "m1_abrams", "t90", "humvee", "soldier_standing",
                "apache", "destroyer"]
        ttk.Combobox(dialog, textvariable=type_var, values=types).pack(pady=5)

        ttk.Label(dialog, text="Name:").pack(pady=5)
        name_var = tk.StringVar(value="")
        ttk.Entry(dialog, textvariable=name_var).pack(pady=5)

        def add():
            self.scene.add_object(type_var.get(), name=name_var.get() or None)
            self.refresh()
            dialog.destroy()

        ttk.Button(dialog, text="Add", command=add).pack(pady=10)


class Studio:
    """Main EOSIM Studio application."""

    def __init__(self, project: Optional[Project] = None):
        self.project = project or Project("New Project")

        # Create renderer
        self.renderer = Renderer(self.project.scene, self.project.camera)

        # Create window
        self.root = tk.Tk()
        self.root.title(f"EOSIM Studio - {self.project.name}")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 700)

        self._build_ui()
        self._bind_events()

        # Initial render
        self.root.after(100, self._initial_render)

    def _build_ui(self):
        # Menu bar
        self._build_menu()

        # Main layout: left panel, center, right panel
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Objects
        left_frame = ttk.Frame(main, width=200)
        main.add(left_frame, weight=1)

        self.object_list = ObjectListWidget(left_frame, self.project.scene,
                                           on_select=self._on_object_selected)
        self.object_list.pack(fill=tk.BOTH, expand=True)

        # Center panel - Views
        center_frame = ttk.Frame(main, width=800)
        main.add(center_frame, weight=4)

        # Scene view (top)
        self.scene_view = SceneViewWidget(center_frame, self.project.scene,
                                         self.project.camera)
        self.scene_view.pack(fill=tk.BOTH, expand=True)

        # Camera view (middle)
        self.camera_view = CameraViewWidget(center_frame, self.renderer)
        self.camera_view.pack(fill=tk.BOTH, expand=True)

        # Timeline (bottom)
        timeline_frame = ttk.LabelFrame(center_frame, text="Timeline", padding=5)
        timeline_frame.pack(fill=tk.X, pady=5)

        self.timeline_widget = TimelineWidget(timeline_frame, self.project.timeline,
                                             on_seek=self._on_time_changed)
        self.timeline_widget.pack(fill=tk.X)

        # Right panel - Settings
        right_frame = ttk.Frame(main, width=250)
        main.add(right_frame, weight=1)

        self.camera_settings = CameraSettingsWidget(right_frame, self.project.camera,
                                                   on_change=self._on_camera_changed)
        self.camera_settings.pack(fill=tk.X, pady=5)

        # Render button
        ttk.Button(right_frame, text="Export Video...",
                  command=self._export_video).pack(fill=tk.X, pady=10)

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Project", command=self._new_project)
        file_menu.add_command(label="Open...", command=self._open_project)
        file_menu.add_command(label="Save", command=self._save_project)
        file_menu.add_command(label="Save As...", command=self._save_project_as)
        file_menu.add_separator()
        file_menu.add_command(label="Load Demo", command=self._load_demo)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Reset View", command=self._reset_view)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _bind_events(self):
        # Timeline playback callback
        self.project.timeline.on_frame(self._on_frame)

        # Keyboard shortcuts
        self.root.bind("<space>", lambda e: self.project.timeline.toggle_playback())
        self.root.bind("<Left>", lambda e: self.project.timeline.step_backward())
        self.root.bind("<Right>", lambda e: self.project.timeline.step_forward())
        self.root.bind("<Home>", lambda e: self.project.timeline.go_to_start())
        self.root.bind("<End>", lambda e: self.project.timeline.go_to_end())

    def _initial_render(self):
        """Initial render after window is shown."""
        self._on_time_changed(0.0)

    def _on_time_changed(self, time_sec: float):
        """Handle time change (scrubbing or playback)."""
        self.scene_view.update_view(time_sec)
        self.camera_view.render_at(time_sec)

    def _on_frame(self, time_sec: float, frame_num: int):
        """Handle frame during playback."""
        # Update on main thread
        self.root.after(0, lambda: self._on_time_changed(time_sec))

    def _on_camera_changed(self):
        """Handle camera settings change."""
        self._on_time_changed(self.project.timeline.current_time)

    def _on_object_selected(self, obj_id: str):
        """Handle object selection."""
        # Could show object properties panel
        pass

    def _new_project(self):
        self.project = Project("New Project")
        self.renderer = Renderer(self.project.scene, self.project.camera)
        self.camera_view.renderer = self.renderer
        self.scene_view.scene = self.project.scene
        self.scene_view.camera = self.project.camera
        self.object_list.scene = self.project.scene
        self.object_list.refresh()
        self.root.title(f"EOSIM Studio - {self.project.name}")
        self._on_time_changed(0.0)

    def _open_project(self):
        path = filedialog.askopenfilename(
            filetypes=[("EOSIM Project", "*.eosim"), ("All Files", "*.*")]
        )
        if path:
            try:
                self.project = Project.load(path)
                self.renderer = Renderer(self.project.scene, self.project.camera)
                self.camera_view.renderer = self.renderer
                self.scene_view.scene = self.project.scene
                self.scene_view.camera = self.project.camera
                self.object_list.scene = self.project.scene
                self.object_list.refresh()
                self.root.title(f"EOSIM Studio - {self.project.name}")
                self._on_time_changed(0.0)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open project: {e}")

    def _save_project(self):
        if self.project.file_path:
            self.project.save()
        else:
            self._save_project_as()

    def _save_project_as(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".eosim",
            filetypes=[("EOSIM Project", "*.eosim"), ("All Files", "*.*")]
        )
        if path:
            self.project.save(path)

    def _load_demo(self):
        self.project = create_demo_project()
        self.renderer = Renderer(self.project.scene, self.project.camera)
        self.camera_view.renderer = self.renderer
        self.scene_view.scene = self.project.scene
        self.scene_view.camera = self.project.camera
        self.timeline_widget.timeline = self.project.timeline
        self.timeline_widget.update_duration(self.project.duration)
        self.object_list.scene = self.project.scene
        self.object_list.refresh()
        self.root.title(f"EOSIM Studio - {self.project.name}")
        self._on_time_changed(0.0)

    def _export_video(self):
        """Export project to video file."""
        from eosim.studio.exporter import VideoExporter, HAS_OPENCV

        # Get output path
        if HAS_OPENCV:
            filetypes = [
                ("MP4 Video", "*.mp4"),
                ("AVI Video", "*.avi"),
                ("Image Sequence", "*.png"),
            ]
        else:
            filetypes = [("Image Sequence", "*.png")]
            messagebox.showinfo(
                "Note",
                "OpenCV not available. Will export as image sequence.\n"
                "Install opencv-python for video export."
            )

        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=filetypes
        )

        if not path:
            return

        # Create export dialog
        export_dialog = tk.Toplevel(self.root)
        export_dialog.title("Exporting Video...")
        export_dialog.geometry("400x150")
        export_dialog.transient(self.root)
        export_dialog.grab_set()

        ttk.Label(export_dialog, text="Exporting...",
                 font=("Segoe UI", 11)).pack(pady=10)

        progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(export_dialog, variable=progress_var,
                                       maximum=100, length=350)
        progress_bar.pack(pady=10)

        status_var = tk.StringVar(value="Starting export...")
        ttk.Label(export_dialog, textvariable=status_var).pack(pady=5)

        def update_progress(progress):
            progress_var.set(progress.percent)
            status_var.set(
                f"Frame {progress.current_frame}/{progress.total_frames} "
                f"({progress.percent:.1f}%)"
            )
            export_dialog.update()

        def run_export():
            exporter = VideoExporter(self.renderer, self.project.render_settings)
            success = exporter.export(
                path,
                duration_sec=self.project.duration,
                fps=self.project.fps,
                progress_callback=update_progress
            )

            export_dialog.destroy()
            if success:
                messagebox.showinfo("Export Complete", f"Video saved to:\n{path}")
            else:
                messagebox.showerror("Export Failed", "Export was cancelled or failed.")

        # Run export in thread to keep UI responsive
        self.root.after(100, run_export)

    def _reset_view(self):
        self._on_time_changed(0.0)

    def _show_about(self):
        messagebox.showinfo(
            "About EOSIM Studio",
            "EOSIM Studio v1.0\n\n"
            "Video-production-style interface for\n"
            "electro-optical sensor simulation.\n\n"
            "Create thermal imagery videos with\n"
            "ease using familiar video editing concepts."
        )

    def launch(self):
        """Launch the studio application."""
        self.root.mainloop()


def launch_studio(project: Optional[Project] = None):
    """Launch EOSIM Studio.

    Args:
        project: Optional project to open. If None, creates new project.

    Example:
        >>> from eosim.studio import launch_studio
        >>> launch_studio()
    """
    studio = Studio(project)
    studio.launch()


# Allow running as script
if __name__ == "__main__":
    launch_studio()
