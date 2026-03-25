# AMV Maker

Generate anime music videos in the style of Poisonpop Candy— anime images composited with audio-reactive visualizers, decorative effects, and rendered to MP4.

![app demo image](demo-image-1.PNG)

## Features

### Visualizers (1-4 simultaneous)

Select any combination of visualizers to render together, each with its own color:

- **Bar Graph** — Stacked semi-transparent boxes that build up and collapse with audio amplitude, with visible gaps between individual boxes. Supports 4 configurable colors that cycle on beats with sweeping left-to-right or right-to-left transitions, so 1-4 colors can be visible at once.
- **Oscilloscope** — Waveform trace with a multi-layer glow effect.
- **Radial** — Frequency bars arranged in a full-screen circle that pulse outward.
- **Particle** — Energy-reactive particles that spawn on beats with gravity and decay.

### Animation Effects

- **Cherry Blossom Petals** (0-100) — Drifting petal particles with randomized size, speed, and transparency.
- **Rainfall** (0-500) — Raindrops falling with varying speed and horizontal drift.
- **Lightning** (0-10) — Beat-synced forked lightning bolts with screen flash, detected from drum/percussive onsets.
- **Hearts** (0-20) — Hollow heart outlines that fade in and out in the top-left corner. Configurable color.

### Multi-Track Support

- Add multiple tracks, each pairing an image with an audio file.
- Images switch when songs change, with a 3-second video crossfade blend.
- Audio tracks crossfade (fade out / fade in) at transitions.
- Reorder tracks with Move Up / Move Down.

### Background Types

- Static images (PNG, JPG, BMP)
- Animated GIF / APNG (cycles at native frame rate)
- Video files (MP4, AVI, MOV, MKV, WebM) as animated backgrounds

### GUI Application

- **Track playlist** — Add/Remove/Reorder paired image+audio tracks
- **Visualizer checkbuttons** — Select 1-4 visualizers with per-effect color pickers
- **Bar graph 4-color swatches** — Four colors that sweep across bars on beats
- **Effect controls** — Spinboxes for bars, petals, rainfall, lightning, hearts, duration
- **Heart color picker** — Separate color swatch for heart outlines
- **Determinate progress bar** with percentage during rendering
- **Embedded video player** — Play/Pause/Stop, seek slider, time display, FPS counter
- **Background rendering** — GUI stays responsive while encoding

### Video Player

- Audio-video sync using audio as master clock
- Optimized for 30+ FPS (OpenCV resize, persistent canvas items, throttled UI updates)
- Audio extracted from rendered MP4 via ffmpeg for pygame playback

### Audio Analysis

- Mel-scaled spectrogram with dB scaling for perceptually balanced frequency bars
- Beat/onset detection via librosa for lightning sync
- Audio crossfade concatenation for multi-track playlists

## Requirements

- Python 3.12+
- FFmpeg (on PATH or bundled via imageio-ffmpeg)

### Python Dependencies

- moviepy
- Pillow
- numpy
- soundfile
- librosa
- opencv-python
- pygame

## Installation

```bash
git clone https://github.com/beknar/amv-maker.git
cd amv-maker
python -m venv venv

# Windows
venv\Scripts\activate.bat

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

### GUI

```bash
python gui.py
```

1. Click **Add Track** to select an image and audio file (repeat for multiple tracks).
2. Check which visualizers to enable and click their color swatches to customize.
3. Adjust effects (bars, petals, rainfall, lightning, hearts, duration).
4. Set the output file path.
5. Click **Render** and watch the progress bar.
6. When complete, the video auto-loads into the player — press Play to preview.

### Recommended Settings

For all effects running simultaneously with the Bar Graph visualizer:

| Setting | Value |
|---------|-------|
| Visualizer | Bar Graph |
| Bars | 100 |
| Petals | 50 |
| Rainfall | 50 |
| Lightning | 8 |
| Hearts | 10 |

### CLI

```bash
python render.py --image image.png --audio track.mp3 -o output.mp4
```

#### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | (required) | Path to background image/video |
| `--audio` | (required) | Path(s) to audio file(s), concatenated in order |
| `-o, --output` | `output.mp4` | Output video path |
| `--visualizer` | `Bar Graph` | Visualizer(s): `"Bar Graph"` `Oscilloscope` `Radial` `Particle` |
| `--bars` | `40` | Number of frequency bars |
| `--petals` | `25` | Petal particles (0 to disable) |
| `--raindrops` | `0` | Raindrops (0 to disable) |
| `--lightning` | `0` | Lightning intensity 0-10 |
| `--duration` | full track | Limit output to N seconds |

#### Examples

```bash
# Bar graph + oscilloscope with rainfall
python render.py --image image.png --audio track.mp3 --visualizer "Bar Graph" Oscilloscope --raindrops 150

# Multi-track audio
python render.py --image image.png --audio track1.mp3 track2.mp3 track3.mp3

# All visualizers with lightning
python render.py --image image.png --audio track.mp3 --visualizer "Bar Graph" Oscilloscope Radial Particle --lightning 7

# Radial only, no petals, 30 second clip
python render.py --image image.png --audio track.mp3 --visualizer Radial --petals 0 --duration 30
```

## Output

- Format: MP4 (H.264 video, AAC audio)
- Resolution: 1280x800
- Frame rate: 30 FPS
- Fast-start enabled for seeking

## Testing

```bash
python -m pytest tests/ -v
```

79 unit tests covering effects, visualizers, audio analysis, rendering, and the video player.

## Project Structure

```
amv-maker/
  gui.py             # tkinter GUI application
  player.py          # Embedded video player (OpenCV + pygame)
  render.py          # render_video() API and CLI entry point
  compositor.py      # Frame composition and build_renderer()
  visualizers.py     # Visualizer draw functions and lightning
  effects.py         # Petal, Raindrop, Heart, Particle classes
  audio.py           # Audio analysis, beat detection, concatenation
  constants.py       # Shared defaults and configuration
  requirements.txt   # Python dependencies
  tests/             # Unit tests (pytest)
  CLAUDE.md          # Project context for Claude Code
  PROPOSAL.md        # Original project proposal
  image.png          # Reference screenshot
```
