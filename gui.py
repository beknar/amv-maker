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
        self.geometry("720x1020")
        self.resizable(False, False)
        self.configure(bg="#1a1a2e")

        self._image_path = tk.StringVar()
        self._audio_paths: list[str] = []
        default_output = str(Path.home() / "Videos" / "amv_output.mp4")
        self._output_path = tk.StringVar(value=default_output)
        self._visualizer = tk.StringVar(value=VISUALIZER_TYPES[0])
        self._bar_count = tk.IntVar(value=40)
        self._petal_count = tk.IntVar(value=25)
        self._raindrop_count = tk.IntVar(value=0)
        self._lightning_intensity = tk.IntVar(value=0)
        self._vis_color: tuple[int, int, int] = (200, 80, 200)
        self._duration = tk.StringVar(value="")
        self._status = tk.StringVar(value="Ready")
        self._rendering = False
        self._queue: queue.Queue = queue.Queue()

        self._thumb_photo: ImageTk.PhotoImage | None = None

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

        # ── inputs ──
        inp = ttk.LabelFrame(self, text="  Inputs  ", padding=8)
        inp.pack(fill=tk.X, padx=10, pady=(10, 5))

        self._file_row(inp, "Image:", self._image_path, self._browse_image,
                       [("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.apng *.mp4 *.avi *.mov *.mkv *.webm")])

        # thumbnail preview
        self._thumb_label = tk.Label(inp, bg="#1a1a2e", width=20, height=6)
        self._thumb_label.grid(row=1, column=0, columnspan=3, pady=(5, 0))

        # ── audio playlist ──
        aud = ttk.LabelFrame(self, text="  Audio Tracks (in order)  ", padding=8)
        aud.pack(fill=tk.X, padx=10, pady=(5, 5))

        self._audio_listbox = tk.Listbox(aud, height=4, bg="#2a2a4e", fg="#e0e0e0",
                                         selectmode=tk.SINGLE, activestyle="none")
        self._audio_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        aud_btns = tk.Frame(aud, bg="#1a1a2e")
        aud_btns.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(aud_btns, text="Add", width=8, command=self._add_audio).pack(pady=2)
        ttk.Button(aud_btns, text="Remove", width=8, command=self._remove_audio).pack(pady=2)
        ttk.Button(aud_btns, text="Move Up", width=8, command=self._move_audio_up).pack(pady=2)
        ttk.Button(aud_btns, text="Move Down", width=8, command=self._move_audio_down).pack(pady=2)

        # ── settings ──
        cfg = ttk.LabelFrame(self, text="  Settings  ", padding=8)
        cfg.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(cfg, text="Visualizer:").grid(row=0, column=0, sticky=tk.W, pady=2)
        vis_combo = ttk.Combobox(
            cfg, textvariable=self._visualizer,
            values=VISUALIZER_TYPES, state="readonly", width=20
        )
        vis_combo.grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(cfg, text="Color:").grid(row=0, column=2, sticky=tk.W, padx=(15, 0), pady=2)
        self._color_swatch = tk.Label(
            cfg, bg="#c850c8", width=3, relief=tk.RAISED, cursor="hand2"
        )
        self._color_swatch.grid(row=0, column=3, sticky=tk.W, padx=5)
        self._color_swatch.bind("<Button-1>", lambda e: self._pick_color())

        ttk.Label(cfg, text="Bars:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(cfg, from_=10, to=80, textvariable=self._bar_count, width=8).grid(
            row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(cfg, text="Petals:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(cfg, from_=0, to=100, textvariable=self._petal_count, width=8).grid(
            row=2, column=1, sticky=tk.W, padx=5)

        ttk.Label(cfg, text="Rainfall:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(cfg, from_=0, to=500, textvariable=self._raindrop_count, width=8).grid(
            row=3, column=1, sticky=tk.W, padx=5)
        ttk.Label(cfg, text="(0 = off)").grid(row=3, column=2, sticky=tk.W)

        ttk.Label(cfg, text="Lightning:").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(cfg, from_=0, to=10, textvariable=self._lightning_intensity, width=8).grid(
            row=4, column=1, sticky=tk.W, padx=5)
        ttk.Label(cfg, text="(0 = off, 1-10)").grid(row=4, column=2, sticky=tk.W)

        ttk.Label(cfg, text="Duration (s):").grid(row=5, column=0, sticky=tk.W, pady=2)
        ttk.Entry(cfg, textvariable=self._duration, width=10).grid(
            row=5, column=1, sticky=tk.W, padx=5)
        ttk.Label(cfg, text="(blank = full track)").grid(row=5, column=2, sticky=tk.W)

        ttk.Label(cfg, text="Output:").grid(row=6, column=0, sticky=tk.W, pady=2)
        ttk.Entry(cfg, textvariable=self._output_path, width=45).grid(
            row=6, column=1, columnspan=2, sticky=tk.W, padx=5)
        ttk.Button(cfg, text="Browse…", command=self._browse_output).grid(
            row=6, column=3, padx=5)

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

    def _file_row(self, parent, label, var, browse_cmd, filetypes):
        row = parent.grid_size()[1]
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=var, width=45).grid(
            row=row, column=1, padx=5, sticky=tk.W)
        ttk.Button(parent, text="Browse…", command=browse_cmd).grid(
            row=row, column=2, padx=5)

    # ── browse dialogs ──────────────────────────────────────────────────────

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.apng *.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")]
        )
        if path:
            self._image_path.set(path)
            self._show_thumbnail(path)

    def _add_audio(self):
        paths = filedialog.askopenfilenames(
            title="Select Audio Track(s)",
            filetypes=[("Audio", "*.mp3 *.wav *.aac *.ogg *.flac"), ("All", "*.*")]
        )
        for path in paths:
            self._audio_paths.append(path)
            self._audio_listbox.insert(tk.END, Path(path).name)

    def _remove_audio(self):
        sel = self._audio_listbox.curselection()
        if sel:
            idx = sel[0]
            self._audio_paths.pop(idx)
            self._audio_listbox.delete(idx)

    def _move_audio_up(self):
        sel = self._audio_listbox.curselection()
        if not sel or sel[0] == 0:
            return
        idx = sel[0]
        # swap in list
        self._audio_paths[idx - 1], self._audio_paths[idx] = \
            self._audio_paths[idx], self._audio_paths[idx - 1]
        # swap in listbox
        text = self._audio_listbox.get(idx)
        self._audio_listbox.delete(idx)
        self._audio_listbox.insert(idx - 1, text)
        self._audio_listbox.selection_set(idx - 1)

    def _move_audio_down(self):
        sel = self._audio_listbox.curselection()
        if not sel or sel[0] >= len(self._audio_paths) - 1:
            return
        idx = sel[0]
        self._audio_paths[idx], self._audio_paths[idx + 1] = \
            self._audio_paths[idx + 1], self._audio_paths[idx]
        text = self._audio_listbox.get(idx)
        self._audio_listbox.delete(idx)
        self._audio_listbox.insert(idx + 1, text)
        self._audio_listbox.selection_set(idx + 1)

    def _pick_color(self):
        initial = "#%02x%02x%02x" % self._vis_color
        result = colorchooser.askcolor(color=initial, title="Visualizer Color")
        if result and result[0]:
            rgb = tuple(int(c) for c in result[0])
            self._vis_color = rgb
            self._color_swatch.configure(bg=result[1])

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save Output As",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("All", "*.*")]
        )
        if path:
            self._output_path.set(path)

    def _show_thumbnail(self, path: str):
        try:
            from render import _is_video_file
            if _is_video_file(path):
                import cv2
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                else:
                    return
            else:
                img = Image.open(path)
            img.thumbnail((160, 100), Image.LANCZOS)
            self._thumb_photo = ImageTk.PhotoImage(img)
            self._thumb_label.configure(image=self._thumb_photo)
        except Exception:
            pass

    # ── rendering ───────────────────────────────────────────────────────────

    def _start_render(self):
        img = self._image_path.get()
        out = self._output_path.get()

        if not img or not Path(img).exists():
            messagebox.showerror("Error", "Please select a valid image file.")
            return
        if not self._audio_paths:
            messagebox.showerror("Error", "Please add at least one audio track.")
            return
        for aud in self._audio_paths:
            if not Path(aud).exists():
                messagebox.showerror("Error", f"Audio file not found:\n{aud}")
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

        aud = self._audio_paths if len(self._audio_paths) > 1 else self._audio_paths[0]
        params = dict(
            image_path=img,
            audio_path=aud,
            output_path=out,
            bar_count=self._bar_count.get(),
            petal_count=self._petal_count.get(),
            raindrop_count=self._raindrop_count.get(),
            lightning_intensity=self._lightning_intensity.get(),
            duration=duration,
            visualizer=self._visualizer.get(),
            vis_color=self._vis_color,
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
