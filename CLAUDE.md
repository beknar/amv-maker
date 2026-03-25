# Anime Music Video Maker — CLAUDE.md

## Project Overview

AMV Maker generates music videos in the style of the Poisonpop Candy / Alya YouTube
channels: anime images (static, animated GIF/APNG, or video) composited with audio-reactive
visualizers, decorative effects, and encoded to MP4. Supports multi-track playlists with
per-track images and audio crossfading.

## Architecture

Six source modules, a GUI, and a video player:

| File | Purpose |
|------|---------|
| `constants.py` | Shared defaults (resolution, FPS, effect limits) |
| `effects.py` | Animation classes: Petal, Raindrop, Heart, Particle |
| `visualizers.py` | Draw functions: bar graph, oscilloscope, radial, lightning |
| `audio.py` | Audio analysis (mel spectrogram), beat detection, crossfade concatenation |
| `compositor.py` | `build_renderer()` — frame composition, background switching, crossfade |
| `render.py` | `render_video()` public API, CLI entry point, encoding pipeline |
| `gui.py` | tkinter GUI with track playlist, settings, color pickers, render controls |
| `player.py` | Embedded video player (OpenCV frames + pygame audio + tkinter canvas) |

## Key Design Decisions

- **PIL/Pillow** renders each frame as RGBA overlays composited onto the background.
  No GPU rendering — all CPU-based via numpy/PIL.
- **MoviePy 2.x** drives the encoding pipeline via `VideoClip(make_frame, duration)`.
- **librosa** provides mel spectrograms (dB-scaled) for perceptually balanced visualizers
  and onset detection for beat-synced lightning.
- **pygame.mixer** handles audio playback in the preview player (audio extracted from
  rendered MP4 to temp WAV via ffmpeg).
- **OpenCV** reads video frames for preview playback and extracts frames from MP4 backgrounds.
- Multiple visualizers can be active simultaneously, each with its own color.
- Multi-track: images paired with audio tracks; audio crossfades (3s) with video blending.

## Inputs

- Background: static image (PNG/JPG), animated GIF/APNG, or video (MP4/AVI/MOV/MKV/WebM)
- Audio: MP3, WAV, AAC, OGG, FLAC — single or multiple tracks
- Multiple images paired 1:1 with audio tracks for per-song backgrounds

## Outputs

- MP4 (H.264 + AAC) at 1280x800, 30 FPS
- faststart enabled, keyframe every 1s, no B-frames (clean seeking)

## Effects & Visualizers

- **Visualizers** (1-4 simultaneous): Bar Graph, Oscilloscope, Radial, Particle
- **Bar Graph** has 4 configurable colors that cycle on beats with sweeping L-to-R or R-to-L
  transitions (precomputed per-bar color map in `compositor.py`)
- **Effects**: Cherry blossom petals, rainfall, lightning bolts (beat-synced), hollow hearts
- All effects have intensity/count controls; visualizers and hearts have per-effect color pickers
- Beat detection (librosa onset strength) drives both lightning timing and bar color cycling

## Dependencies

Python 3.12+, moviepy, Pillow, numpy, soundfile, librosa, opencv-python, pygame.
FFmpeg required (bundled via imageio-ffmpeg).

## Testing

79 unit tests in `tests/` via pytest. Tests cover all effect classes, visualizer draw
functions, audio analysis, build_renderer, render_video (mocked), and the video player.

```bash
python -m pytest tests/ -v
```

## Running

```bash
python -m venv venv && venv\Scripts\activate.bat && pip install -r requirements.txt
python gui.py          # GUI
python render.py --help  # CLI
```
