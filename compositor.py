"""AMV Maker — compositor.py
Background loading, frame composition, and the build_renderer function
that produces the per-frame callback for MoviePy.
"""

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from constants import (
    WIDTH, HEIGHT, FPS, DEFAULT_VIS_COLOR, HEART_COLOR, CROSSFADE_SECONDS,
    BAR_SWEEP_SPEED,
)
from effects import Petal, Raindrop, Heart, Particle
from visualizers import (
    draw_bar_visualizer, draw_oscilloscope, draw_radial_visualizer, draw_lightning,
)


# ── video background loading ────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"}


def load_video_frames(video_path: str) -> tuple[list[Image.Image], float]:
    """Extract all frames from a video file as PIL RGBA images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video background: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_duration = 1.0 / fps
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).convert("RGBA").resize((WIDTH, HEIGHT), Image.LANCZOS)
        frames.append(img)

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames found in video: {video_path}")

    return frames, frame_duration


def _is_video_file(path: str) -> bool:
    """Check if a file path has a video extension."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


# ── background source loading ───────────────────────────────────────────────

def _load_bg_source(bg_image) -> tuple[list[Image.Image], float]:
    """Load a background source into a list of RGBA frames and frame duration."""
    frames = []
    frame_duration = 1.0 / FPS

    if isinstance(bg_image, (list, tuple)):
        frames, frame_duration = bg_image
    else:
        n_frames = getattr(bg_image, "n_frames", 1)
        if isinstance(n_frames, int) and n_frames > 1:
            for i in range(bg_image.n_frames):
                bg_image.seek(i)
                frame = bg_image.convert("RGBA").resize((WIDTH, HEIGHT), Image.LANCZOS)
                frames.append(frame)
            info = bg_image.info
            dur_ms = info.get("duration", 100)
            if dur_ms > 0:
                frame_duration = dur_ms / 1000.0
        else:
            frames.append(bg_image.convert("RGBA").resize((WIDTH, HEIGHT), Image.LANCZOS))

    return frames, frame_duration


# ── frame renderer ──────────────────────────────────────────────────────────

def build_renderer(bg_image, bar_data: np.ndarray,
                   petals: list[Petal], visualizer: str | list[str] = "Bar Graph",
                   waveforms: list | None = None, particles: list | None = None,
                   raindrops: list[Raindrop] | None = None,
                   vis_colors: dict[str, tuple[int, int, int]] | None = None,
                   lightning_intensity: int = 0,
                   beat_frames: np.ndarray | None = None,
                   hearts: list[Heart] | None = None,
                   heart_color: tuple[int, int, int] = HEART_COLOR,
                   track_backgrounds: list | None = None,
                   track_durations: list[float] | None = None,
                   bar_colors: list[tuple[int, int, int]] | None = None,
                   bar_sweep_speed: float = BAR_SWEEP_SPEED):
    """Return a function(t) -> numpy frame for VideoClip."""

    if track_backgrounds and track_durations:
        bg_segments = []
        for tb in track_backgrounds:
            f, fd = _load_bg_source(tb)
            bg_segments.append((f, fd, len(f) > 1))
    else:
        f, fd = _load_bg_source(bg_image)
        bg_segments = [(f, fd, len(f) > 1)]
        track_durations = None

    if beat_frames is None:
        beat_frames = np.array([], dtype=int)

    # pre-compute per-bar color index for each frame (sweeping transitions)
    n_bars = bar_data.shape[1]
    _bar_color_map = None  # will be a list of lists: _bar_color_map[frame][bar] = color_idx
    if bar_colors and len(bar_colors) > 1:
        total_frames = bar_data.shape[0]
        sweep_frames = max(1, int(FPS * bar_sweep_speed))

        # find beat frames for color switching
        if len(beat_frames) == 0:
            # fallback: energy peaks
            energy = bar_data.mean(axis=1)
            threshold = energy.mean() + energy.std() * 0.5
            beat_list = []
            cooldown = 0
            for f in range(total_frames):
                if cooldown > 0:
                    cooldown -= 1
                elif energy[f] > threshold:
                    beat_list.append(f)
                    cooldown = FPS
        else:
            beat_list = [int(b) for b in beat_frames]

        # for each beat, pick a new color and a sweep direction
        rng = random.Random(42)
        transitions = []  # (start_frame, new_color_idx, left_to_right)
        current_idx = 0
        for bf in beat_list:
            choices = [i for i in range(len(bar_colors)) if i != current_idx]
            new_idx = rng.choice(choices) if choices else 0
            left_to_right = rng.choice([True, False])
            transitions.append((bf, new_idx, left_to_right))
            current_idx = new_idx

        # build the per-bar color map
        # start with all bars at color 0
        bar_state = [0] * n_bars
        _bar_color_map = []
        trans_idx = 0
        active_sweep = None  # (start_frame, new_color, left_to_right)

        for f in range(total_frames):
            # check if a new transition starts
            if trans_idx < len(transitions) and f >= transitions[trans_idx][0]:
                active_sweep = transitions[trans_idx]
                trans_idx += 1

            if active_sweep is not None:
                sweep_start, new_color, ltr = active_sweep
                progress = f - sweep_start
                if progress < sweep_frames:
                    # how many bars have switched so far
                    bars_switched = int((progress + 1) / sweep_frames * n_bars)
                    bars_switched = min(bars_switched, n_bars)
                    if ltr:
                        for b in range(bars_switched):
                            bar_state[b] = new_color
                    else:
                        for b in range(n_bars - bars_switched, n_bars):
                            bar_state[b] = new_color
                elif progress == sweep_frames:
                    # finish sweep — all bars to new color
                    bar_state = [new_color] * n_bars
                    active_sweep = None

            _bar_color_map.append(list(bar_state))

    def make_frame(t: float) -> np.ndarray:
        frame_idx = int(t * FPS)
        frame_idx = min(frame_idx, bar_data.shape[0] - 1)
        amplitudes = bar_data[frame_idx]

        # select background segment
        seg_idx = 0
        seg_start = 0.0
        if track_durations:
            for i, end_t in enumerate(track_durations):
                if t < end_t:
                    seg_idx = i
                    break
                seg_start = end_t
            else:
                seg_idx = len(track_durations) - 1
                seg_start = track_durations[-2] if len(track_durations) > 1 else 0.0

        def _get_seg_frame(idx, time, start):
            sf, sfd, sa = bg_segments[min(idx, len(bg_segments) - 1)]
            if sa:
                bi = int((time - start) / sfd) % len(sf)
                return sf[bi].copy()
            return sf[0].copy()

        img = _get_seg_frame(seg_idx, t, seg_start)

        # video crossfade between tracks
        xfade_half = CROSSFADE_SECONDS / 2
        if track_durations and len(bg_segments) > 1:
            for i, end_t in enumerate(track_durations[:-1]):
                if end_t - xfade_half <= t < end_t + xfade_half:
                    blend = (t - (end_t - xfade_half)) / CROSSFADE_SECONDS
                    blend = max(0.0, min(1.0, blend))
                    prev_start = track_durations[i - 1] if i > 0 else 0.0
                    img_out = _get_seg_frame(i, t, prev_start)
                    img_in = _get_seg_frame(i + 1, t, end_t)
                    img = Image.blend(img_out.convert("RGBA"),
                                      img_in.convert("RGBA"), blend)
                    break

        overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # draw selected visualizers
        _colors = vis_colors or {}
        vis_list = visualizer if isinstance(visualizer, list) else [visualizer]
        for vis in vis_list:
            vc = _colors.get(vis, DEFAULT_VIS_COLOR)
            if vis == "Bar Graph":
                per_bar = None
                if _bar_color_map is not None:
                    fidx = min(frame_idx, len(_bar_color_map) - 1)
                    per_bar = _bar_color_map[fidx]
                draw_bar_visualizer(draw, amplitudes, vc,
                                    bar_colors=bar_colors, per_bar_color_indices=per_bar)
            elif vis == "Oscilloscope" and waveforms:
                wf_idx = min(frame_idx, len(waveforms) - 1)
                draw_oscilloscope(draw, waveforms[wf_idx], vc)
            elif vis == "Radial":
                draw_radial_visualizer(draw, amplitudes, vc)
            elif vis == "Particle" and particles is not None:
                _dt = 1.0 / FPS
                energy = float(amplitudes.mean())
                for p in particles:
                    if p.life <= 0 and energy > 0.3:
                        p.reset(energy)
                for p in particles:
                    p.update(_dt)
                    p.draw(overlay, vc)

        dt = 1.0 / FPS

        for p in petals:
            p.update(dt)
            p.draw(overlay)

        if raindrops:
            for r in raindrops:
                r.update(dt)
                r.draw(draw)

        if hearts:
            for h in hearts:
                h.update(dt)
                h.draw(draw, heart_color)

        flash_alpha = 0
        if lightning_intensity > 0:
            flash_alpha = draw_lightning(draw, lightning_intensity, frame_idx,
                                         beat_frames, FPS)

        if flash_alpha > 0:
            flash = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, flash_alpha))
            img = Image.alpha_composite(img, flash)

        img = Image.alpha_composite(img, overlay)
        return np.array(img.convert("RGB"))

    return make_frame
