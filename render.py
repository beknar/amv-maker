"""
AMV Maker — render.py
Composites a static/animated image with audio visualizers and effects,
then encodes to MP4.

Usage (CLI):
    python render.py --image inputs/image.png --audio inputs/track.mp3 -o output.mp4

Usage (programmatic):
    from render import render_video
    render_video("image.png", "track.mp3", output_path="out.mp4")
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable

import cv2
import librosa
import numpy as np
import proglog
from moviepy import AudioFileClip, VideoClip
from PIL import Image, ImageDraw

from constants import (
    WIDTH, HEIGHT, FPS, BAR_COUNT, DEFAULT_VIS_COLOR,
    NUM_PETALS, NUM_RAINDROPS, LIGHTNING_INTENSITY,
    HEART_INTENSITY, HEART_COLOR, CROSSFADE_SECONDS,
    VISUALIZER_TYPES,
)
from effects import Petal, Raindrop, Heart, Particle
from visualizers import (
    draw_bar_visualizer, draw_oscilloscope, draw_radial_visualizer, draw_lightning,
)
from audio import (
    analyse_audio, analyse_audio_waveform, detect_beats, concatenate_audio_files,
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
                   track_durations: list[float] | None = None):
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
    """Render an AMV and return the output file path."""
    if progress_callback:
        progress_callback(0.0)

    # concatenate multiple audio files if needed
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

    if len(image_paths) > 1 and _original_audio_paths is not None:
        track_backgrounds = []
        for img_p in image_paths:
            if _is_video_file(img_p):
                f, fd = load_video_frames(img_p)
                track_backgrounds.append((f, fd))
            else:
                track_backgrounds.append(Image.open(img_p))
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

    _vis_colors = vis_colors or {}
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

    output_path = str(Path(output_path).resolve())

    logger: str | proglog.ProgressBarLogger = "bar"
    if progress_callback:
        class _ProgressLogger(proglog.ProgressBarLogger):
            def bars_callback(self, bar, attr, value, old_value=None):
                if attr == "index" and bar in self.bars:
                    total = self.bars[bar].get("total")
                    if total and total > 0:
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
            "-movflags", "+faststart",
            "-g", str(FPS),
            "-bf", "0",
        ],
    )

    if progress_callback:
        progress_callback(1.0)

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
