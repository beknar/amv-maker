"""
AMV Maker — render.py
Public API for rendering AMVs and CLI entry point.

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

import librosa
import numpy as np
import proglog
from moviepy import AudioFileClip, VideoClip
from PIL import Image

from constants import (
    WIDTH, HEIGHT, FPS, BAR_COUNT, DEFAULT_VIS_COLOR,
    NUM_PETALS, NUM_RAINDROPS, LIGHTNING_INTENSITY,
    HEART_INTENSITY, HEART_COLOR, BAR_SWEEP_SPEED, VISUALIZER_TYPES,
)
from effects import Petal, Raindrop, Heart, Particle
from audio import (
    analyse_audio, analyse_audio_waveform, detect_beats, concatenate_audio_files,
)
from compositor import build_renderer, load_video_frames, _is_video_file


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
    bar_colors: list[tuple[int, int, int]] | None = None,
    bar_sweep_speed: float = BAR_SWEEP_SPEED,
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
        _vis_colors = {v: vis_color for v in vis_list}

    # detect beats for bar color cycling even if lightning is off
    if bar_colors and len(bar_colors) > 1 and beat_frames is None:
        beat_frames = detect_beats(y_samples, sr, FPS, min_gap_s=1.0)

    make_frame = build_renderer(bg, bar_data, petals, visualizer, waveforms, particles,
                                raindrops, _vis_colors, lightning_intensity, beat_frames,
                                hearts_list, heart_color,
                                track_backgrounds, track_durations_list,
                                bar_colors=bar_colors, bar_sweep_speed=bar_sweep_speed)

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
