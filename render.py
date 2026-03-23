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
    """
    if intensity <= 0 or len(beat_frames) == 0:
        return 0

    # check if we're near a beat (within a few frames for flash decay)
    flash_alpha = 0
    flash_duration = 4  # frames of flash decay
    for beat in beat_frames:
        diff = frame_idx - beat
        if 0 <= diff < flash_duration:
            # decay over flash_duration frames
            t = 1.0 - diff / flash_duration
            base_alpha = int(30 + intensity * 15)  # 45-180 range
            flash_alpha = max(flash_alpha, int(base_alpha * t))

            if diff == 0:
                # draw actual bolt(s) on the beat frame
                num_bolts = max(1, intensity // 3)
                for _ in range(num_bolts):
                    bx = random.randint(int(WIDTH * 0.1), int(WIDTH * 0.9))
                    bolt_alpha = min(255, 100 + intensity * 15)
                    _draw_lightning_bolt(draw, bx, 0, random.randint(HEIGHT // 2, HEIGHT),
                                        bolt_alpha, branch_chance=0.2 + intensity * 0.05)

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
    # inner radius + max bar length = 100% of screen height / 2
    total_radius = HEIGHT * 0.5
    inner_r = int(total_radius * 0.4)
    max_bar_len = int(total_radius * 0.6)
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

def build_renderer(bg_image: Image.Image, bar_data: np.ndarray,
                   petals: list[Petal], visualizer: str = "Bar Graph",
                   waveforms: list | None = None, particles: list | None = None,
                   raindrops: list[Raindrop] | None = None,
                   vis_color: tuple[int, int, int] = DEFAULT_VIS_COLOR,
                   lightning_intensity: int = 0,
                   beat_frames: np.ndarray | None = None):
    """Return a function(t) -> numpy frame for VideoClip."""

    # extract animated frames or use single static image
    bg_frames = []
    bg_frame_duration = 1.0 / FPS  # default: match video FPS

    if isinstance(bg_image, (list, tuple)):
        # pre-extracted frames (e.g. from MP4) — (frames_list, duration)
        bg_frames, bg_frame_duration = bg_image
    else:
        n_frames = getattr(bg_image, "n_frames", 1)
        if isinstance(n_frames, int) and n_frames > 1:
            # animated GIF/APNG — extract all frames
            for i in range(bg_image.n_frames):
                bg_image.seek(i)
                frame = bg_image.convert("RGBA").resize((WIDTH, HEIGHT), Image.LANCZOS)
                bg_frames.append(frame)
            info = bg_image.info
            dur_ms = info.get("duration", 100)
            if dur_ms > 0:
                bg_frame_duration = dur_ms / 1000.0
        else:
            bg_frames.append(bg_image.convert("RGBA").resize((WIDTH, HEIGHT), Image.LANCZOS))

    bg_is_animated = len(bg_frames) > 1

    if beat_frames is None:
        beat_frames = np.array([], dtype=int)

    def make_frame(t: float) -> np.ndarray:
        frame_idx = int(t * FPS)
        frame_idx = min(frame_idx, bar_data.shape[0] - 1)
        amplitudes = bar_data[frame_idx]

        # select background frame — cycle through GIF frames at their native rate
        if bg_is_animated:
            bg_idx = int(t / bg_frame_duration) % len(bg_frames)
            img = bg_frames[bg_idx].copy()
        else:
            img = bg_frames[0].copy()
        overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # draw chosen visualizer
        if visualizer == "Oscilloscope" and waveforms:
            wf_idx = min(frame_idx, len(waveforms) - 1)
            draw_oscilloscope(draw, waveforms[wf_idx], vis_color)
        elif visualizer == "Radial":
            draw_radial_visualizer(draw, amplitudes, vis_color)
        elif visualizer == "Particle" and particles is not None:
            dt = 1.0 / FPS
            energy = float(amplitudes.mean())
            for p in particles:
                if p.life <= 0 and energy > 0.3:
                    p.reset(energy)
            for p in particles:
                p.update(dt)
                p.draw(overlay, vis_color)
        else:
            draw_bar_visualizer(draw, amplitudes, vis_color)

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

        # draw lightning
        flash_alpha = 0
        if lightning_intensity > 0:
            flash_alpha = draw_lightning(draw, lightning_intensity, frame_idx,
                                         beat_frames, FPS)

        img = Image.alpha_composite(img, overlay)

        # apply screen flash for lightning
        if flash_alpha > 0:
            flash = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, flash_alpha))
            img = Image.alpha_composite(img, flash)

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


# ── public API ──────────────────────────────────────────────────────────────

def render_video(
    image_path: str,
    audio_path: str,
    output_path: str = "output.mp4",
    bar_count: int = BAR_COUNT,
    petal_count: int = NUM_PETALS,
    raindrop_count: int = NUM_RAINDROPS,
    lightning_intensity: int = LIGHTNING_INTENSITY,
    duration: float | None = None,
    visualizer: str = "Bar Graph",
    vis_color: tuple[int, int, int] = DEFAULT_VIS_COLOR,
    progress_callback: Callable[[float], None] | None = None,
) -> str:
    """Render an AMV and return the output file path.

    Args:
        progress_callback: called with a float 0.0–1.0 representing progress.
    """
    if progress_callback:
        progress_callback(0.0)

    bar_data, audio_duration, y_samples, sr = analyse_audio(
        audio_path, bar_count, FPS
    )

    if progress_callback:
        progress_callback(0.1)

    dur = duration or audio_duration

    if _is_video_file(image_path):
        frames, frame_dur = load_video_frames(image_path)
        bg = (frames, frame_dur)  # tuple signals pre-extracted video frames
    else:
        bg = Image.open(image_path)
    petals = [Petal() for _ in range(petal_count)]

    waveforms = None
    particles = None
    if visualizer == "Oscilloscope":
        waveforms = analyse_audio_waveform(y_samples, sr, FPS)
    elif visualizer == "Particle":
        particles = [Particle() for _ in range(80)]

    raindrops = [Raindrop() for _ in range(raindrop_count)] if raindrop_count > 0 else None

    beat_frames = None
    if lightning_intensity > 0:
        beat_frames = detect_beats(y_samples, sr, FPS, min_gap_s=5.0)

    make_frame = build_renderer(bg, bar_data, petals, visualizer, waveforms, particles,
                                raindrops, vis_color, lightning_intensity, beat_frames)

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

    return output_path


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AMV Maker — render a music video")
    parser.add_argument("--image", required=True, help="Path to background image")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("-o", "--output", default="output.mp4", help="Output video path")
    parser.add_argument("--bars", type=int, default=BAR_COUNT)
    parser.add_argument("--petals", type=int, default=NUM_PETALS)
    parser.add_argument("--raindrops", type=int, default=NUM_RAINDROPS)
    parser.add_argument("--lightning", type=int, default=LIGHTNING_INTENSITY,
                        help="Lightning intensity 0-10 (0=off)")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--visualizer", choices=VISUALIZER_TYPES, default="Bar Graph")
    args = parser.parse_args()

    for p, label in [(args.image, "Image"), (args.audio, "Audio")]:
        if not Path(p).exists():
            sys.exit(f"{label} not found: {p}")

    render_video(
        image_path=args.image,
        audio_path=args.audio,
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
