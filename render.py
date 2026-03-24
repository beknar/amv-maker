"""
AMV Maker — render.py
Composites a static anime image with an audio visualizer and optional
cherry-blossom petal particles, then encodes to MP4.

Usage (CLI):
    python render.py --image inputs/image.png --audio inputs/track.mp3 -o output.mp4

Usage (programmatic):
    from render import render_video
    render_video("image.png", "track.mp3", output_path="out.mp4")

Requires: moviepy, Pillow, numpy, librosa
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Callable

import cv2
import librosa
import numpy as np
import proglog
from moviepy import AudioFileClip, VideoClip
from PIL import Image, ImageDraw

# ── defaults ────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 800
FPS = 30
BAR_COUNT = 40
BAR_BOX_H = 10
BAR_GAP = 4             # horizontal gap between bar columns
BOX_VGAP = 3            # vertical gap between stacked boxes
DEFAULT_VIS_COLOR = (200, 80, 200)  # RGB — alpha is always applied separately
VIS_ALPHA = 180
VIS_BOTTOM_Y = HEIGHT - 40
MAX_BOXES = 25

NUM_PETALS = 25
NUM_RAINDROPS = 0
LIGHTNING_INTENSITY = 0   # 0 = off, 1-10 scale
HEART_INTENSITY = 0       # 0 = off, 1-20 max hearts on screen
HEART_COLOR = (255, 80, 150)

VISUALIZER_TYPES = ["Bar Graph", "Oscilloscope", "Radial", "Particle"]


# ── petal particles ─────────────────────────────────────────────────────────

class Petal:
    """A single cherry-blossom petal that drifts across the frame."""

    def __init__(self):
        self.reset(initial=True)

    def reset(self, initial: bool = False):
        self.x = random.uniform(-50, WIDTH + 50)
        self.y = random.uniform(-HEIGHT, 0) if not initial else random.uniform(0, HEIGHT)
        self.size = random.uniform(12, 28)
        self.speed_y = random.uniform(30, 80)
        self.speed_x = random.uniform(-20, 20)
        self.rotation = random.uniform(0, 360)
        self.rot_speed = random.uniform(30, 120)
        self.alpha = random.randint(160, 240)

    def update(self, dt: float):
        self.y += self.speed_y * dt
        self.x += self.speed_x * dt
        self.rotation += self.rot_speed * dt
        if self.y > HEIGHT + 30:
            self.reset()

    def draw(self, overlay: Image.Image):
        draw = ImageDraw.Draw(overlay)
        r = max(1, int(self.size))
        rh = max(1, r // 2)
        cx, cy = int(self.x), int(self.y)
        color = (255, 140, 180, self.alpha)
        draw.ellipse([cx - r, cy - rh, cx + r, cy + rh], fill=color)


# ── rainfall particles ──────────────────────────────────────────────────────

class Raindrop:
    """A single raindrop that falls straight down with slight drift."""

    def __init__(self):
        self.reset(initial=True)

    def reset(self, initial: bool = False):
        self.x = random.uniform(0, WIDTH)
        self.y = random.uniform(-HEIGHT, 0) if not initial else random.uniform(0, HEIGHT)
        self.length = random.uniform(18, 50)
        self.speed = random.uniform(400, 800)
        self.drift = random.uniform(-10, 10)
        self.alpha = random.randint(140, 230)
        self.width = random.choice([1, 2, 2, 3])

    def update(self, dt: float):
        self.y += self.speed * dt
        self.x += self.drift * dt
        if self.y > HEIGHT + self.length:
            self.reset()

    def draw(self, draw: ImageDraw.ImageDraw):
        x = int(self.x)
        y1 = int(self.y)
        y0 = int(self.y - self.length)
        color = (180, 200, 255, self.alpha)
        draw.line([(x, y0), (x, y1)], fill=color, width=self.width)


# ── heart particles ────────────────────────────────────────────────────────

class Heart:
    """A hollow heart that fades in and out in the top-left area."""

    def __init__(self, max_hearts: int):
        self.max_hearts = max_hearts
        self.reset()

    def reset(self):
        # position in top-left quadrant with some randomness
        self.x = random.uniform(30, WIDTH * 0.3)
        self.y = random.uniform(30, HEIGHT * 0.3)
        self.size = random.uniform(12, 30)
        self.phase = 0.0  # 0→1 fade in, 1→2 hold, 2→3 fade out
        self.speed = random.uniform(0.4, 0.8)  # phase speed per second
        self.active = False
        self.wait = random.uniform(0.5, 3.0)  # seconds before appearing

    def update(self, dt: float):
        if not self.active:
            self.wait -= dt
            if self.wait <= 0:
                self.active = True
            return

        self.phase += self.speed * dt
        if self.phase >= 3.0:
            self.reset()

    def get_alpha(self) -> int:
        if not self.active:
            return 0
        if self.phase < 1.0:
            # fade in
            return int(220 * self.phase)
        elif self.phase < 2.0:
            # hold
            return 220
        else:
            # fade out
            return int(220 * (3.0 - self.phase))

    def draw(self, draw_ctx: ImageDraw.ImageDraw,
             color: tuple[int, int, int] = HEART_COLOR):
        alpha = self.get_alpha()
        if alpha <= 0:
            return
        cx, cy = int(self.x), int(self.y)
        s = self.size
        fill = (color[0], color[1], color[2], alpha)
        # draw hollow heart using two overlapping circles + triangle outline
        # left bump
        r = s * 0.5
        lx, ly = cx - r * 0.5, cy - r * 0.3
        rx, ry = cx + r * 0.5, cy - r * 0.3
        # draw as polygon outline for hollow effect
        pts = _heart_polygon(cx, cy, s)
        draw_ctx.polygon(pts, outline=fill)
        # draw slightly thicker for visibility
        for offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = [(p[0] + offset[0], p[1] + offset[1]) for p in pts]
            draw_ctx.polygon(shifted, outline=fill)


def _heart_polygon(cx: float, cy: float, size: float) -> list[tuple[int, int]]:
    """Generate heart shape as a polygon of points."""
    points = []
    steps = 30
    for i in range(steps):
        t = 2 * math.pi * i / steps
        # parametric heart curve
        x = size * 0.8 * (16 * math.sin(t) ** 3) / 16
        y = -size * 0.8 * (13 * math.cos(t) - 5 * math.cos(2 * t) -
                           2 * math.cos(3 * t) - math.cos(4 * t)) / 16
        points.append((int(cx + x), int(cy + y)))
    return points


# ── audio analysis ──────────────────────────────────────────────────────────

def analyse_audio(audio_path: str, n_bars: int, fps: int):
    """Return a (frames, n_bars) array of per-bar amplitudes in 0..1 range.

    Uses a mel-scaled spectrogram with dB scaling for perceptually even bars.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    hop = sr // fps

    # mel spectrogram gives perceptually spaced frequency bands
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_bars, hop_length=hop, n_fft=2048
    )
    # convert to dB scale for better dynamic range
    S_db = librosa.power_to_db(S, ref=np.max)
    # S_db is now in range [-80, 0] roughly; map to [0, 1]
    # clip at -60 dB floor (quiet threshold)
    S_db = np.clip(S_db, -60, 0)
    bars = ((S_db + 60) / 60).T.astype(np.float32)  # shape: (frames, n_bars)

    return bars, len(y) / sr, y, sr


def analyse_audio_waveform(audio_samples: np.ndarray, sr: int, fps: int):
    """Return per-frame waveform chunks for oscilloscope display."""
    hop = sr // fps
    frames = len(audio_samples) // hop
    waveforms = []
    for i in range(frames):
        start = i * hop
        end = start + hop
        chunk = audio_samples[start:end]
        # downsample to WIDTH points
        if len(chunk) > WIDTH:
            indices = np.linspace(0, len(chunk) - 1, WIDTH, dtype=int)
            chunk = chunk[indices]
        mx = np.abs(chunk).max()
        if mx > 0:
            chunk = chunk / mx
        waveforms.append(chunk)
    return waveforms


def detect_beats(audio_samples: np.ndarray, sr: int, fps: int, min_gap_s: float = 5.0):
    """Detect drum/percussive onsets and return frame indices of beats.

    Returns beat frame indices spaced at least min_gap_s seconds apart,
    prioritising the strongest onsets.
    """
    # onset strength focused on percussive content
    onset_env = librosa.onset.onset_strength(y=audio_samples, sr=sr, hop_length=sr // fps)
    # pick peaks in onset envelope
    peaks = librosa.util.peak_pick(
        onset_env, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.3, wait=int(min_gap_s * fps)
    )
    if len(peaks) == 0:
        return np.array([], dtype=int)
    # sort by strength descending, then enforce min_gap
    strengths = onset_env[peaks]
    order = np.argsort(-strengths)
    selected = []
    for idx in order:
        frame = peaks[idx]
        if all(abs(frame - s) >= min_gap_s * fps for s in selected):
            selected.append(frame)
    selected.sort()
    return np.array(selected, dtype=int)


# ── lightning effect ───────────────────────────────────────────────────────

def _draw_lightning_bolt(draw: ImageDraw.ImageDraw, x: int, y_start: int, y_end: int,
                         alpha: int, branch_chance: float = 0.3):
    """Draw a jagged lightning bolt from (x, y_start) down to y_end."""
    segments = random.randint(6, 12)
    step_y = (y_end - y_start) / segments
    points = [(x, y_start)]
    cx = x
    for i in range(1, segments):
        cx += random.randint(-40, 40)
        cy = int(y_start + step_y * i)
        points.append((cx, cy))
    points.append((cx + random.randint(-15, 15), y_end))

    # draw glow layers (thick to thin)
    for width, a_mult in [(7, 0.2), (4, 0.4), (2, 0.8), (1, 1.0)]:
        color = (220, 220, 255, int(alpha * a_mult))
        for j in range(len(points) - 1):
            draw.line([points[j], points[j + 1]], fill=color, width=width)

    # draw branches
    for i in range(1, len(points) - 1):
        if random.random() < branch_chance:
            bx, by = points[i]
            branch_len = random.randint(20, 60)
            bx_end = bx + random.choice([-1, 1]) * branch_len
            by_end = by + random.randint(15, 40)
            branch_color = (200, 200, 255, int(alpha * 0.5))
            draw.line([(bx, by), (bx_end, by_end)], fill=branch_color, width=1)


def draw_lightning(draw: ImageDraw.ImageDraw, intensity: int, frame_idx: int,
                   beat_frames: np.ndarray, fps: int):
    """Draw lightning bolts on beat frames. Returns flash_alpha for screen flash.

    intensity: 1-10 scale controlling brightness, bolt count, and flash strength.
    Returns flash alpha (0-255) for a whole-screen white flash overlay.
    Bolts persist for several frames with fading alpha; flash is kept subtle
    so it doesn't wash out the bolts.
    """
    if intensity <= 0 or len(beat_frames) == 0:
        return 0

    flash_alpha = 0
    bolt_duration = 6   # frames the bolt stays visible
    flash_duration = 4  # frames the flash decays over

    for beat in beat_frames:
        diff = frame_idx - beat
        if diff < 0 or diff >= max(bolt_duration, flash_duration):
            continue

        # screen flash — subtle so it doesn't cover the bolts
        if diff < flash_duration:
            t = 1.0 - diff / flash_duration
            base_alpha = int(15 + intensity * 8)  # 23-95 range (subtle)
            flash_alpha = max(flash_alpha, int(base_alpha * t))

        # draw bolts — persist with fading alpha
        if diff < bolt_duration:
            fade = 1.0 - diff / bolt_duration
            # seed random per beat so the same bolt shape persists across frames
            rng_state = random.getstate()
            random.seed(int(beat) * 31337)
            num_bolts = max(1, intensity // 3)
            for _ in range(num_bolts):
                bx = random.randint(int(WIDTH * 0.1), int(WIDTH * 0.9))
                bolt_alpha = int(min(255, 150 + intensity * 10) * fade)
                _draw_lightning_bolt(draw, bx, 0, random.randint(HEIGHT // 2, HEIGHT),
                                    bolt_alpha, branch_chance=0.2 + intensity * 0.05)
            random.setstate(rng_state)

    return flash_alpha


# ── visualizer renderers ───────────────────────────────────────────────────

def _vis_rgba(rgb: tuple[int, int, int], alpha: int = VIS_ALPHA) -> tuple[int, int, int, int]:
    """Combine an RGB color with an alpha value."""
    return (rgb[0], rgb[1], rgb[2], alpha)


def draw_bar_visualizer(draw: ImageDraw.ImageDraw, amplitudes: np.ndarray,
                        color: tuple[int, int, int] = DEFAULT_VIS_COLOR):
    """Stacked semi-transparent bar graph made of individual boxes."""
    fill = _vis_rgba(color)
    total_bar_width = WIDTH // max(1, len(amplitudes))
    box_w = max(1, total_bar_width - BAR_GAP)
    box_step = BAR_BOX_H + BOX_VGAP
    for i, amp in enumerate(amplitudes):
        num_boxes = int(amp * MAX_BOXES)
        x0 = i * total_bar_width
        for j in range(num_boxes):
            y_top = VIS_BOTTOM_Y - (j + 1) * box_step
            y_bot = y_top + BAR_BOX_H
            draw.rectangle([x0, y_top, x0 + box_w, y_bot], fill=fill)


def draw_oscilloscope(draw: ImageDraw.ImageDraw, waveform: np.ndarray,
                      color: tuple[int, int, int] = DEFAULT_VIS_COLOR):
    """Waveform trace across the bottom of the frame."""
    if len(waveform) < 2:
        return
    center_y = VIS_BOTTOM_Y - 60
    amplitude_h = 80
    points = []
    for i, sample in enumerate(waveform):
        x = int(i * WIDTH / len(waveform))
        y = int(center_y - sample * amplitude_h)
        points.append((x, y))
    for thickness, alpha in [(5, 40), (3, 80), (1, 200)]:
        fill = _vis_rgba(color, alpha)
        for j in range(len(points) - 1):
            draw.line([points[j], points[j + 1]], fill=fill, width=thickness)


def draw_radial_visualizer(draw: ImageDraw.ImageDraw, amplitudes: np.ndarray,
                           color: tuple[int, int, int] = DEFAULT_VIS_COLOR):
    """Bars arranged in a circle at the center of the screen, 50% of height."""
    fill = _vis_rgba(color)
    cx, cy = WIDTH // 2, HEIGHT // 2
    n = len(amplitudes)
    inner_r = HEIGHT // 2  # inner radius spans full screen height
    max_bar_len = int(inner_r * 0.5)  # bars extend beyond screen edge
    for i, amp in enumerate(amplitudes):
        angle = 2 * math.pi * i / n - math.pi / 2
        bar_len = inner_r + amp * max_bar_len
        x0 = cx + int(inner_r * math.cos(angle))
        y0 = cy + int(inner_r * math.sin(angle))
        x1 = cx + int(bar_len * math.cos(angle))
        y1 = cy + int(bar_len * math.sin(angle))
        draw.line([(x0, y0), (x1, y1)], fill=fill, width=3)


class Particle:
    """Energy-reactive particle for particle visualizer."""

    def __init__(self):
        self.reset(0)

    def reset(self, energy: float):
        self.x = random.uniform(100, WIDTH - 100)
        self.y = random.uniform(VIS_BOTTOM_Y - 200, VIS_BOTTOM_Y)
        speed = 30 + energy * 150
        angle = random.uniform(0, 2 * math.pi)
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle)
        self.life = 1.0
        self.decay = random.uniform(0.3, 0.8)
        self.size = random.uniform(2, 6)

    def update(self, dt: float):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 40 * dt  # gravity
        self.life -= self.decay * dt

    def draw(self, overlay: Image.Image, color: tuple[int, int, int] = DEFAULT_VIS_COLOR):
        if self.life <= 0:
            return
        draw = ImageDraw.Draw(overlay)
        alpha = max(0, min(255, int(self.life * 180)))
        r = max(1, int(self.size * self.life))
        cx, cy = int(self.x), int(self.y)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=_vis_rgba(color, alpha))


# ── frame renderer ──────────────────────────────────────────────────────────

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
                   track_durations: list[float] | None = None):
    """Return a function(t) -> numpy frame for VideoClip.

    Args:
        bg_image: single PIL image, tuple of (frames, duration), or ignored
                  if track_backgrounds is provided.
        track_backgrounds: list of bg sources (one per audio track).
        track_durations: cumulative end times for each track in seconds.
    """

    # multi-track backgrounds: each track has its own bg source
    if track_backgrounds and track_durations:
        bg_segments = []
        for tb in track_backgrounds:
            frames, fdur = _load_bg_source(tb)
            bg_segments.append((frames, fdur, len(frames) > 1))
    else:
        frames, fdur = _load_bg_source(bg_image)
        bg_segments = [(frames, fdur, len(frames) > 1)]
        track_durations = None

    if beat_frames is None:
        beat_frames = np.array([], dtype=int)

    def make_frame(t: float) -> np.ndarray:
        frame_idx = int(t * FPS)
        frame_idx = min(frame_idx, bar_data.shape[0] - 1)
        amplitudes = bar_data[frame_idx]

        # select background segment based on track timing, with crossfade
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

        def _get_seg_frame(idx: int, time: float, start: float) -> Image.Image:
            sf, sfd, sa = bg_segments[min(idx, len(bg_segments) - 1)]
            if sa:
                bi = int((time - start) / sfd) % len(sf)
                return sf[bi].copy()
            return sf[0].copy()

        img = _get_seg_frame(seg_idx, t, seg_start)

        # video crossfade between tracks
        xfade_half = CROSSFADE_SECONDS / 2
        if track_durations and len(bg_segments) > 1:
            # check if we're near a track boundary
            for i, end_t in enumerate(track_durations[:-1]):
                if end_t - xfade_half <= t < end_t + xfade_half:
                    # blend between track i and track i+1
                    blend = (t - (end_t - xfade_half)) / CROSSFADE_SECONDS
                    blend = max(0.0, min(1.0, blend))
                    prev_start = track_durations[i - 1] if i > 0 else 0.0
                    next_start = end_t
                    img_out = _get_seg_frame(i, t, prev_start)
                    img_in = _get_seg_frame(i + 1, t, next_start)
                    img = Image.blend(img_out.convert("RGBA"),
                                      img_in.convert("RGBA"), blend)
                    break
        overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # draw selected visualizers (one or more), each with its own color
        _colors = vis_colors or {}
        vis_list = visualizer if isinstance(visualizer, list) else [visualizer]
        for vis in vis_list:
            vc = _colors.get(vis, DEFAULT_VIS_COLOR)
            if vis == "Bar Graph":
                draw_bar_visualizer(draw, amplitudes, vc)
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

        # draw petals
        for p in petals:
            p.update(dt)
            p.draw(overlay)

        # draw rainfall
        if raindrops:
            for r in raindrops:
                r.update(dt)
                r.draw(draw)

        # draw hearts
        if hearts:
            for h in hearts:
                h.update(dt)
                h.draw(draw, heart_color)

        # draw lightning — get flash alpha first, then draw bolts on overlay
        flash_alpha = 0
        if lightning_intensity > 0:
            flash_alpha = draw_lightning(draw, lightning_intensity, frame_idx,
                                         beat_frames, FPS)

        # apply screen flash BEFORE overlay so bolts appear on top of flash
        if flash_alpha > 0:
            flash = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, flash_alpha))
            img = Image.alpha_composite(img, flash)

        img = Image.alpha_composite(img, overlay)

        return np.array(img.convert("RGB"))

    return make_frame


# ── video background loading ────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"}


def load_video_frames(video_path: str) -> tuple[list[Image.Image], float]:
    """Extract all frames from a video file as PIL RGBA images.

    Returns (frames_list, frame_duration_seconds).
    """
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


# ── audio concatenation ─────────────────────────────────────────────────────

CROSSFADE_SECONDS = 3.0  # duration of audio/video crossfade between tracks


def concatenate_audio_files(audio_paths: list[str], output_path: str,
                            crossfade_s: float = CROSSFADE_SECONDS) -> tuple[str, list[float]]:
    """Concatenate multiple audio files with crossfade into one WAV file.

    Returns (output_path, track_end_times) where track_end_times are cumulative
    seconds marking the midpoint of each crossfade (the switch point for video).
    """
    import soundfile as sf

    all_tracks = []
    target_sr = None

    for path in audio_paths:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        if target_sr is None:
            target_sr = sr
        all_tracks.append(y)

    if len(all_tracks) == 1:
        sf.write(output_path, all_tracks[0], target_sr)
        return output_path, [len(all_tracks[0]) / target_sr]

    # crossfade: overlap the end of track N with the start of track N+1
    xfade_samples = int(crossfade_s * target_sr)
    track_end_times = []
    combined = all_tracks[0].copy()

    for i in range(1, len(all_tracks)):
        xfade = min(xfade_samples, len(combined), len(all_tracks[i]))

        if xfade > 0:
            # fade out the tail of the current combined audio
            fade_out = np.linspace(1.0, 0.0, xfade, dtype=np.float32)
            # fade in the head of the next track
            fade_in = np.linspace(0.0, 1.0, xfade, dtype=np.float32)

            # the overlap region: blend both
            overlap = combined[-xfade:] * fade_out + all_tracks[i][:xfade] * fade_in
            combined[-xfade:] = overlap
            # append the rest of the next track (after the crossfade region)
            combined = np.concatenate([combined, all_tracks[i][xfade:]])
        else:
            combined = np.concatenate([combined, all_tracks[i]])

        # track switch point is at the midpoint of the crossfade
        track_end_times.append(len(combined) / target_sr -
                               len(all_tracks[i][xfade:]) / target_sr -
                               xfade / target_sr / 2)

    track_end_times.append(len(combined) / target_sr)

    sf.write(output_path, combined, target_sr)
    return output_path, track_end_times


# ── public API ──────────────────────────────────────────────────────────────

def render_video(
    image_path: str | list[str],
    audio_path: str | list[str],
    output_path: str = "output.mp4",
    bar_count: int = BAR_COUNT,
    petal_count: int = NUM_PETALS,
    raindrop_count: int = NUM_RAINDROPS,
    lightning_intensity: int = LIGHTNING_INTENSITY,
    heart_intensity: int = HEART_INTENSITY,
    heart_color: tuple[int, int, int] = HEART_COLOR,
    duration: float | None = None,
    visualizer: str | list[str] = "Bar Graph",
    vis_color: tuple[int, int, int] = DEFAULT_VIS_COLOR,
    vis_colors: dict[str, tuple[int, int, int]] | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> str:
    """Render an AMV and return the output file path.

    Args:
        image_path: single path or list of paths (one per audio track).
        audio_path: single path or list of paths to concatenate in order.
        progress_callback: called with a float 0.0–1.0 representing progress.
    """
    if progress_callback:
        progress_callback(0.0)

    # concatenate multiple audio files if needed
    import tempfile
    _temp_audio = None
    _original_audio_paths = None
    _crossfade_track_times = None
    if isinstance(audio_path, list):
        _original_audio_paths = list(audio_path)
        if len(audio_path) == 1:
            audio_path = audio_path[0]
        else:
            _temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            _temp_audio.close()
            _, _crossfade_track_times = concatenate_audio_files(
                audio_path, _temp_audio.name
            )
            audio_path = _temp_audio.name

    bar_data, audio_duration, y_samples, sr = analyse_audio(
        audio_path, bar_count, FPS
    )

    if progress_callback:
        progress_callback(0.1)

    dur = duration or audio_duration

    # load background(s)
    track_backgrounds = None
    track_durations_list = None
    bg = None

    image_paths = image_path if isinstance(image_path, list) else [image_path]
    original_audio_paths = _original_audio_paths if '_original_audio_paths' in dir() else None

    if len(image_paths) > 1 and _original_audio_paths is not None:
        # multiple images paired with audio tracks
        track_backgrounds = []
        for img_p in image_paths:
            if _is_video_file(img_p):
                f, fd = load_video_frames(img_p)
                track_backgrounds.append((f, fd))
            else:
                track_backgrounds.append(Image.open(img_p))
        # use crossfade-aware track times if available, else compute from files
        if _crossfade_track_times:
            track_durations_list = _crossfade_track_times
        else:
            track_durations_list = []
            cumulative = 0.0
            for i in range(len(_original_audio_paths)):
                y_trk, sr_trk = librosa.load(_original_audio_paths[i], sr=None, mono=True)
                cumulative += len(y_trk) / sr_trk
                track_durations_list.append(cumulative)
    else:
        p = image_paths[0]
        if _is_video_file(p):
            frames, frame_dur = load_video_frames(p)
            bg = (frames, frame_dur)
        else:
            bg = Image.open(p)

    petals = [Petal() for _ in range(petal_count)]

    vis_list = visualizer if isinstance(visualizer, list) else [visualizer]
    waveforms = None
    particles = None
    if "Oscilloscope" in vis_list:
        waveforms = analyse_audio_waveform(y_samples, sr, FPS)
    if "Particle" in vis_list:
        particles = [Particle() for _ in range(80)]

    raindrops = [Raindrop() for _ in range(raindrop_count)] if raindrop_count > 0 else None
    hearts_list = [Heart(heart_intensity) for _ in range(heart_intensity)] if heart_intensity > 0 else None

    beat_frames = None
    if lightning_intensity > 0:
        beat_frames = detect_beats(y_samples, sr, FPS, min_gap_s=5.0)

    # build per-visualizer color map
    _vis_colors = vis_colors or {}
    # fallback: if vis_colors not provided, use single vis_color for all
    if not _vis_colors:
        vl = visualizer if isinstance(visualizer, list) else [visualizer]
        _vis_colors = {v: vis_color for v in vl}

    make_frame = build_renderer(bg, bar_data, petals, visualizer, waveforms, particles,
                                raindrops, _vis_colors, lightning_intensity, beat_frames,
                                hearts_list, heart_color,
                                track_backgrounds, track_durations_list)

    clip = VideoClip(make_frame, duration=dur).with_fps(FPS)
    audio_clip = AudioFileClip(audio_path)
    if dur < audio_duration:
        audio_clip = audio_clip.subclipped(0, dur)
    clip = clip.with_audio(audio_clip)

    if progress_callback:
        progress_callback(0.15)

    # resolve output to absolute path
    output_path = str(Path(output_path).resolve())

    # custom logger that reports encoding progress back to the callback
    logger: str | proglog.ProgressBarLogger = "bar"
    if progress_callback:
        class _ProgressLogger(proglog.ProgressBarLogger):
            def bars_callback(self, bar, attr, value, old_value=None):
                if attr == "index" and bar in self.bars:
                    total = self.bars[bar].get("total")
                    if total and total > 0:
                        # map encoding progress to 0.15–1.0 range
                        pct = 0.15 + 0.85 * (value / total)
                        progress_callback(min(pct, 1.0))
        logger = _ProgressLogger()

    clip.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=4,
        logger=logger,
        ffmpeg_params=[
            "-movflags", "+faststart",   # moov atom at front for seeking
            "-g", str(FPS),              # keyframe every 1 second
            "-bf", "0",                  # no B-frames (cleaner seeking)
        ],
    )

    if progress_callback:
        progress_callback(1.0)

    # clean up temp concatenated audio file
    if _temp_audio is not None:
        try:
            os.remove(_temp_audio.name)
        except OSError:
            pass

    return output_path


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AMV Maker — render a music video")
    parser.add_argument("--image", required=True, help="Path to background image")
    parser.add_argument("--audio", required=True, nargs="+",
                        help="Path(s) to audio file(s), concatenated in order")
    parser.add_argument("-o", "--output", default="output.mp4", help="Output video path")
    parser.add_argument("--bars", type=int, default=BAR_COUNT)
    parser.add_argument("--petals", type=int, default=NUM_PETALS)
    parser.add_argument("--raindrops", type=int, default=NUM_RAINDROPS)
    parser.add_argument("--lightning", type=int, default=LIGHTNING_INTENSITY,
                        help="Lightning intensity 0-10 (0=off)")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--visualizer", choices=VISUALIZER_TYPES, nargs="+",
                        default=["Bar Graph"],
                        help="Visualizer(s) to show (1-4)")
    args = parser.parse_args()

    if not Path(args.image).exists():
        sys.exit(f"Image not found: {args.image}")
    for p in args.audio:
        if not Path(p).exists():
            sys.exit(f"Audio not found: {p}")

    audio = args.audio if len(args.audio) > 1 else args.audio[0]
    render_video(
        image_path=args.image,
        audio_path=audio,
        output_path=args.output,
        bar_count=args.bars,
        petal_count=args.petals,
        raindrop_count=args.raindrops,
        lightning_intensity=args.lightning,
        duration=args.duration,
        visualizer=args.visualizer,
    )
    print("Done!")


if __name__ == "__main__":
    main()
