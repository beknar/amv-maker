"""
AMV Maker — player.py
Embeddable tkinter video player using OpenCV for frame reading
and pygame.mixer for audio playback.
Optimized for 30+ FPS playback.
"""

import time
import tkinter as tk

import cv2
import numpy as np
import pygame
from PIL import Image, ImageTk


class VideoPlayer(tk.Frame):
    """A tkinter widget that plays an MP4 with synced audio."""

    DISPLAY_W = 640
    DISPLAY_H = 400

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self._cap: cv2.VideoCapture | None = None
        self._fps = 30.0
        self._total_frames = 0
        self._duration = 0.0
        self._playing = False
        self._paused = False
        self._audio_path: str | None = None
        self._after_id: str | None = None
        self._photo: ImageTk.PhotoImage | None = None

        # FPS tracking
        self._display_fps: float = 0.0
        self._frame_count: int = 0
        self._fps_update_time: float = 0.0
        self._tick_count: int = 0  # for throttling UI updates

        # persistent canvas item IDs (avoid delete/create each frame)
        self._canvas_img_id: int | None = None
        self._canvas_fps_id: int | None = None

        # ── canvas ──
        self.canvas = tk.Canvas(
            self, width=self.DISPLAY_W, height=self.DISPLAY_H, bg="#1a1a2e"
        )
        self.canvas.pack(side=tk.TOP, padx=5, pady=5)

        # ── transport controls ──
        ctrl = tk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=5)

        self.btn_play = tk.Button(ctrl, text="\u25B6 Play", width=8, command=self.play)
        self.btn_play.pack(side=tk.LEFT, padx=2)

        self.btn_pause = tk.Button(ctrl, text="\u23F8 Pause", width=8, command=self.pause)
        self.btn_pause.pack(side=tk.LEFT, padx=2)

        self.btn_stop = tk.Button(ctrl, text="\u23F9 Stop", width=8, command=self.stop)
        self.btn_stop.pack(side=tk.LEFT, padx=2)

        self.time_label = tk.Label(ctrl, text="0:00 / 0:00", width=14)
        self.time_label.pack(side=tk.RIGHT, padx=5)

        # ── seek slider ──
        self.seek_var = tk.DoubleVar(value=0)
        self.seek_slider = tk.Scale(
            self, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.seek_var, showvalue=False, command=self._on_seek
        )
        self.seek_slider.pack(side=tk.TOP, fill=tk.X, padx=5)

        # init pygame mixer (audio only, no display)
        if not pygame.mixer.get_init():
            pygame.mixer.init()

    def load(self, video_path: str, audio_path: str):
        """Load a video file for playback."""
        self.stop()

        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = self._total_frames / self._fps
        self._audio_path = audio_path

        self.seek_slider.configure(to=self._duration)

        # reset canvas items
        self.canvas.delete("all")
        self._canvas_img_id = None
        self._canvas_fps_id = None

        self._show_frame(0)
        self._update_time(0)

    def play(self):
        """Start or resume playback."""
        if self._cap is None:
            return

        if self._paused:
            self._paused = False
            pygame.mixer.music.unpause()
            self._schedule_next_frame()
            return

        if self._playing:
            return

        self._playing = True
        self._paused = False
        self._frame_count = 0
        self._tick_count = 0
        self._fps_update_time = time.perf_counter()
        self._display_fps = 0.0

        if self._audio_path:
            pygame.mixer.music.load(self._audio_path)
            pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES) / self._fps
            pygame.mixer.music.play(start=pos)

        self._schedule_next_frame()

    def pause(self):
        if not self._playing:
            return
        self._paused = True
        pygame.mixer.music.pause()
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None

    def stop(self):
        self._playing = False
        self._paused = False
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        if self._cap:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._show_frame(0)
        self._update_time(0)
        self.seek_var.set(0)
        # hide FPS when stopped
        if self._canvas_fps_id is not None:
            self.canvas.itemconfigure(self._canvas_fps_id, text="")

    def _schedule_next_frame(self):
        if not self._playing or self._paused:
            return
        # use 1ms delay — let the tick function manage frame pacing
        self._after_id = self.after(1, self._tick)

    def _tick(self):
        if not self._playing or self._paused or self._cap is None:
            return

        now = time.perf_counter()

        # FPS tracking
        self._frame_count += 1
        self._tick_count += 1
        elapsed = now - self._fps_update_time
        if elapsed >= 0.5:
            self._display_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_update_time = now

        # audio as master clock
        audio_ms = pygame.mixer.music.get_pos()
        if audio_ms < 0:
            self.stop()
            return

        audio_time = audio_ms / 1000.0
        target_frame = int(audio_time * self._fps)

        if target_frame >= self._total_frames:
            self.stop()
            return

        current_frame = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        # skip frames if behind, seek if way off
        if target_frame > current_frame + 3:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        elif target_frame <= current_frame:
            # already ahead of audio — wait for next tick
            self._schedule_next_frame()
            return

        ret, frame = self._cap.read()
        if not ret:
            self.stop()
            return

        self._display_cv_frame(frame)

        # throttle UI updates: time label + seek slider every 10 frames
        if self._tick_count % 10 == 0:
            self._update_time(audio_time)
            self.seek_var.set(audio_time)

        self._schedule_next_frame()

    def _show_frame(self, frame_num: int):
        """Display a specific frame by number."""
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self._cap.read()
        if ret:
            self._display_cv_frame(frame)

    def _display_cv_frame(self, frame: np.ndarray):
        """Convert a cv2 BGR frame and show it on the canvas."""
        # resize in cv2 (much faster than PIL LANCZOS)
        small = cv2.resize(frame, (self.DISPLAY_W, self.DISPLAY_H),
                           interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        # go straight from numpy to ImageTk via PIL (no expensive resize)
        self._photo = ImageTk.PhotoImage(
            image=Image.fromarray(rgb, "RGB")
        )

        if self._canvas_img_id is None:
            self._canvas_img_id = self.canvas.create_image(
                0, 0, anchor=tk.NW, image=self._photo
            )
            self._canvas_fps_id = self.canvas.create_text(
                8, 8, anchor=tk.NW, text="",
                fill="#00ff00", font=("Consolas", 11, "bold"),
            )
        else:
            self.canvas.itemconfigure(self._canvas_img_id, image=self._photo)

        # update FPS text (canvas text update is cheap)
        if self._playing and self._canvas_fps_id is not None:
            self.canvas.itemconfigure(
                self._canvas_fps_id,
                text=f"FPS: {self._display_fps:.1f}",
            )

    def _update_time(self, current: float):
        cur = f"{int(current)//60}:{int(current)%60:02d}"
        tot = f"{int(self._duration)//60}:{int(self._duration)%60:02d}"
        self.time_label.config(text=f"{cur} / {tot}")

    def _on_seek(self, value):
        """Handle seek slider drag."""
        if not self._playing or self._cap is None:
            return
        t = float(value)
        frame = int(t * self._fps)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        if self._audio_path:
            pygame.mixer.music.stop()
            pygame.mixer.music.play(start=t)
        self._update_time(t)

    def destroy(self):
        self.stop()
        if self._cap:
            self._cap.release()
            self._cap = None
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        super().destroy()
