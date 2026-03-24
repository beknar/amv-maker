"""
AMV Maker — gui.py
Main GUI application. Lets the user pick an image, audio file, and visualizer
type, then renders an AMV and plays it back in an embedded video player.

Usage:
    python gui.py
"""

import queue
import threading
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk
from pathlib import Path

from PIL import Image, ImageTk

from render import VISUALIZER_TYPES, render_video
from player import VideoPlayer


class AMVMakerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AMV Maker")
        self.geometry("750x1050")
        self.resizable(False, False)
        self.configure(bg="#1a1a2e")

        self._image_paths: list[str] = []
        self._audio_paths: list[str] = []
        default_output = str(Path.home() / "Videos" / "amv_output.mp4")
        self._output_path = tk.StringVar(value=default_output)
        self._vis_checks: dict[str, tk.BooleanVar] = {}
        self._vis_colors: dict[str, tuple[int, int, int]] = {}
        default_colors = {
            "Bar Graph": (200, 80, 200),
            "Oscilloscope": (0, 200, 255),
            "Radial": (255, 100, 180),
            "Particle": (220, 100, 220),
        }
        for vt in VISUALIZER_TYPES:
            self._vis_checks[vt] = tk.BooleanVar(value=(vt == "Bar Graph"))
            self._vis_colors[vt] = default_colors.get(vt, (200, 80, 200))
        self._vis_swatches: dict[str, tk.Label] = {}
        self._bar_count = tk.IntVar(value=40)
        self._petal_count = tk.IntVar(value=25)
        self._raindrop_count = tk.IntVar(value=0)
        self._lightning_intensity = tk.IntVar(value=0)
        self._heart_intensity = tk.IntVar(value=0)
        self._heart_color: tuple[int, int, int] = (255, 80, 150)
        self._duration = tk.StringVar(value="")
        self._status = tk.StringVar(value="Ready")
        self._rendering = False
        self._queue: queue.Queue = queue.Queue()

        self._build_ui()

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabelframe", background="#1a1a2e", foreground="#e0e0e0")
        style.configure("TLabelframe.Label", background="#1a1a2e", foreground="#c080c0")
        style.configure("TLabel", background="#1a1a2e", foreground="#e0e0e0")
        style.configure("TButton", background="#2a2a4e", foreground="#e0e0e0")
        style.configure("TEntry", fieldbackground="#2a2a4e", foreground="#e0e0e0")

        # ── tracks (image + audio paired) ──
        trk = ttk.LabelFrame(self, text="  Tracks (Image + Audio, in order)  ", padding=8)
        trk.pack(fill=tk.X, padx=10, pady=(10, 5))

        self._track_listbox = tk.Listbox(trk, height=5, bg="#2a2a4e", fg="#e0e0e0",
                                         selectmode=tk.SINGLE, activestyle="none")
        self._track_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        trk_btns = tk.Frame(trk, bg="#1a1a2e")
        trk_btns.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(trk_btns, text="Add Track", width=10, command=self._add_track).pack(pady=2)
        ttk.Button(trk_btns, text="Remove", width=10, command=self._remove_track).pack(pady=2)
        ttk.Button(trk_btns, text="Move Up", width=10, command=self._move_track_up).pack(pady=2)
        ttk.Button(trk_btns, text="Move Down", width=10, command=self._move_track_down).pack(pady=2)

        # ── settings (two-column: visualizers left, params right) ──
        cfg = ttk.LabelFrame(self, text="  Settings  ", padding=8)
        cfg.pack(fill=tk.X, padx=10, pady=5)

        # left: visualizer checkbuttons with color swatches
        left = tk.Frame(cfg, bg="#1a1a2e")
        left.pack(side=tk.LEFT, anchor=tk.NW, padx=(0, 15))
        ttk.Label(left, text="Visualizers:").pack(anchor=tk.W)
        for vt in VISUALIZER_TYPES:
            row_f = tk.Frame(left, bg="#1a1a2e")
            row_f.pack(anchor=tk.W)
            tk.Checkbutton(row_f, text=vt, variable=self._vis_checks[vt],
                           bg="#1a1a2e", fg="#e0e0e0", selectcolor="#2a2a4e",
                           activebackground="#1a1a2e", activeforeground="#e0e0e0",
                           width=14, anchor=tk.W
                           ).pack(side=tk.LEFT)
            hex_c = "#%02x%02x%02x" % self._vis_colors[vt]
            swatch = tk.Label(row_f, bg=hex_c, width=3, relief=tk.RAISED, cursor="hand2")
            swatch.pack(side=tk.LEFT, padx=4)
            self._vis_swatches[vt] = swatch
            swatch.bind("<Button-1>", lambda e, name=vt: self._pick_vis_color(name))

        # right: all other settings
        right = tk.Frame(cfg, bg="#1a1a2e")
        right.pack(side=tk.LEFT, anchor=tk.NW, fill=tk.X, expand=True)

        r = 0
        for label, var, extra in [
            ("Bars:", self._bar_count, None),
            ("Petals:", self._petal_count, None),
            ("Rainfall:", self._raindrop_count, "(0 = off)"),
            ("Lightning:", self._lightning_intensity, "(0-10)"),
        ]:
            ttk.Label(right, text=label).grid(row=r, column=0, sticky=tk.W, pady=1)
            ttk.Spinbox(right, from_=0, to=500, textvariable=var, width=8).grid(
                row=r, column=1, sticky=tk.W, padx=5)
            if extra:
                ttk.Label(right, text=extra).grid(row=r, column=2, sticky=tk.W)
            r += 1

        ttk.Label(right, text="Hearts:").grid(row=r, column=0, sticky=tk.W, pady=1)
        ttk.Spinbox(right, from_=0, to=20, textvariable=self._heart_intensity, width=8).grid(
            row=r, column=1, sticky=tk.W, padx=5)
        self._heart_swatch = tk.Label(
            right, bg="#ff5096", width=3, relief=tk.RAISED, cursor="hand2"
        )
        self._heart_swatch.grid(row=r, column=2, sticky=tk.W, padx=5)
        self._heart_swatch.bind("<Button-1>", lambda e: self._pick_heart_color())
        r += 1

        ttk.Label(right, text="Duration:").grid(row=r, column=0, sticky=tk.W, pady=1)
        ttk.Entry(right, textvariable=self._duration, width=8).grid(
            row=r, column=1, sticky=tk.W, padx=5)
        ttk.Label(right, text="(blank=full)").grid(row=r, column=2, sticky=tk.W)
        r += 1

        # output path — full width below
        out_frame = tk.Frame(self, bg="#1a1a2e")
        out_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        ttk.Label(out_frame, text="Output:").pack(side=tk.LEFT)
        ttk.Entry(out_frame, textvariable=self._output_path, width=50).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(out_frame, text="Browse…", command=self._browse_output).pack(
            side=tk.LEFT)

        # ── render ──
        ren = tk.Frame(self, bg="#1a1a2e")
        ren.pack(fill=tk.X, padx=10, pady=5)

        self.btn_render = ttk.Button(ren, text="Render", command=self._start_render)
        self.btn_render.pack(side=tk.LEFT, padx=5)

        self._progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(
            ren, mode="determinate", length=200,
            variable=self._progress_var, maximum=100
        )
        self.progress.pack(side=tk.LEFT, padx=10)

        self.lbl_status = ttk.Label(ren, textvariable=self._status)
        self.lbl_status.pack(side=tk.LEFT, padx=5)

        # ── player ──
        pf = ttk.LabelFrame(self, text="  Preview  ", padding=5)
        pf.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.player = VideoPlayer(pf)
        self.player.pack()

    # ── track management (paired image + audio) ────────────────────────────

    def _add_track(self):
        img = filedialog.askopenfilename(
            title="Select Image for Track",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.apng *.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")]
        )
        if not img:
            return
        aud = filedialog.askopenfilename(
            title="Select Audio for Track",
            filetypes=[("Audio", "*.mp3 *.wav *.aac *.ogg *.flac"), ("All", "*.*")]
        )
        if not aud:
            return
        self._image_paths.append(img)
        self._audio_paths.append(aud)
        label = f"{Path(aud).stem}  |  {Path(img).name}"
        self._track_listbox.insert(tk.END, label)

    def _remove_track(self):
        sel = self._track_listbox.curselection()
        if sel:
            idx = sel[0]
            self._image_paths.pop(idx)
            self._audio_paths.pop(idx)
            self._track_listbox.delete(idx)

    def _move_track_up(self):
        sel = self._track_listbox.curselection()
        if not sel or sel[0] == 0:
            return
        idx = sel[0]
        for lst in [self._image_paths, self._audio_paths]:
            lst[idx - 1], lst[idx] = lst[idx], lst[idx - 1]
        text = self._track_listbox.get(idx)
        self._track_listbox.delete(idx)
        self._track_listbox.insert(idx - 1, text)
        self._track_listbox.selection_set(idx - 1)

    def _move_track_down(self):
        sel = self._track_listbox.curselection()
        if not sel or sel[0] >= len(self._audio_paths) - 1:
            return
        idx = sel[0]
        for lst in [self._image_paths, self._audio_paths]:
            lst[idx], lst[idx + 1] = lst[idx + 1], lst[idx]
        text = self._track_listbox.get(idx)
        self._track_listbox.delete(idx)
        self._track_listbox.insert(idx + 1, text)
        self._track_listbox.selection_set(idx + 1)

    def _pick_vis_color(self, vis_name: str):
        initial = "#%02x%02x%02x" % self._vis_colors[vis_name]
        result = colorchooser.askcolor(color=initial, title=f"{vis_name} Color")
        if result and result[0]:
            rgb = tuple(int(c) for c in result[0])
            self._vis_colors[vis_name] = rgb
            self._vis_swatches[vis_name].configure(bg=result[1])

    def _pick_heart_color(self):
        initial = "#%02x%02x%02x" % self._heart_color
        result = colorchooser.askcolor(color=initial, title="Heart Color")
        if result and result[0]:
            rgb = tuple(int(c) for c in result[0])
            self._heart_color = rgb
            self._heart_swatch.configure(bg=result[1])

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save Output As",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("All", "*.*")]
        )
        if path:
            self._output_path.set(path)

    # ── rendering ───────────────────────────────────────────────────────────

    def _start_render(self):
        out = self._output_path.get()

        if not self._audio_paths or not self._image_paths:
            messagebox.showerror("Error", "Please add at least one track (image + audio).")
            return
        for img in self._image_paths:
            if not Path(img).exists():
                messagebox.showerror("Error", f"Image file not found:\n{img}")
                return
        for aud in self._audio_paths:
            if not Path(aud).exists():
                messagebox.showerror("Error", f"Audio file not found:\n{aud}")
                return
        selected_vis = [vt for vt, var in self._vis_checks.items() if var.get()]
        if not selected_vis:
            messagebox.showerror("Error", "Please select at least one visualizer.")
            return
        if not out:
            messagebox.showerror("Error", "Please specify an output file path.")
            return

        dur_text = self._duration.get().strip()
        duration = float(dur_text) if dur_text else None

        self._rendering = True
        self.btn_render.configure(state=tk.DISABLED)
        self._status.set("Rendering… 0%")
        self._progress_var.set(0)

        imgs = self._image_paths if len(self._image_paths) > 1 else self._image_paths[0]
        auds = self._audio_paths if len(self._audio_paths) > 1 else self._audio_paths[0]
        params = dict(
            image_path=imgs,
            audio_path=auds,
            output_path=out,
            bar_count=self._bar_count.get(),
            petal_count=self._petal_count.get(),
            raindrop_count=self._raindrop_count.get(),
            lightning_intensity=self._lightning_intensity.get(),
            heart_intensity=self._heart_intensity.get(),
            heart_color=self._heart_color,
            duration=duration,
            visualizer=[vt for vt, var in self._vis_checks.items() if var.get()],
            vis_colors={vt: self._vis_colors[vt] for vt in VISUALIZER_TYPES},
        )

        thread = threading.Thread(target=self._render_worker, args=(params,), daemon=True)
        thread.start()
        self.after(200, self._poll_render)

    def _render_worker(self, params: dict):
        try:
            def on_progress(pct):
                self._queue.put(("progress", pct))

            render_video(**params, progress_callback=on_progress)
            self._queue.put(("done", params["output_path"]))
        except Exception as e:
            self._queue.put(("error", str(e)))

    def _poll_render(self):
        try:
            while True:
                msg = self._queue.get_nowait()
                if msg[0] == "done":
                    self._on_render_done(msg[1])
                    return
                elif msg[0] == "error":
                    self._on_render_error(msg[1])
                    return
                elif msg[0] == "progress":
                    pct = msg[1]
                    self._progress_var.set(pct * 100)
                    self._status.set(f"Rendering… {int(pct * 100)}%")
        except queue.Empty:
            pass
        if self._rendering:
            self.after(100, self._poll_render)

    def _on_render_done(self, video_path: str):
        self._rendering = False
        self._progress_var.set(100)
        self.btn_render.configure(state=tk.NORMAL)
        self._status.set(f"Done! Saved to {video_path}")

        # auto-load into player — use the MP4 itself as audio source
        # (it has the concatenated audio embedded)
        try:
            self.player.load(video_path, video_path)
        except Exception as e:
            messagebox.showwarning("Playback", f"Video saved but player failed:\n{e}")

    def _on_render_error(self, error_msg: str):
        self._rendering = False
        self._progress_var.set(0)
        self.btn_render.configure(state=tk.NORMAL)
        self._status.set("Error")
        messagebox.showerror("Render Error", error_msg)


if __name__ == "__main__":
    app = AMVMakerApp()
    app.mainloop()
