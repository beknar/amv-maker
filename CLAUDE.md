# Anime Music Video Maker — CLAUDE.md

Project summary for Claude Code / Claude CLI: a concise, machine-friendly description
of the AMV (Anime Music Video) Maker project in this directory. Use this file as the
primary context when asking Claude to analyze, modify, or generate code for the project.

## Project Overview

The AMV Maker produces short music videos in the style of the Poisonpop Candy / Alya
YouTube channels: a central anime image (static or lightly animated) with a waveform
visualizer and optional decorative particles or petals. The visualizer may be a stacked
transparent-bar graph, circular/radial plot, oscilloscope trace, or particle/object-based
visualization.

## Goals

- Take one or more user-supplied images (static or animated) and an audio track.
- Generate a rendered video (MP4 preferred) at typical resolutions (example: 1280×800).
- Support multiple visualizer styles and optional decorative animation layers.

## Inputs

- Static image (png/jpg) of an anime scene (typically a character/portrait).
- Optional animated image (APNG / animated sprites / short video clip).
- Music track (MP3/WAV/AAC) matching the style of Poisonpop Candy (kawaii phonk / candies).
- User choice of visualizer type and simple parameters (colors, amplitude thresholds).

## Outputs

- Final rendered video file (MP4 recommended, WMV optional) at resolutions such as 1280×800.

## Tech Stack / Dependencies

- Python 3.12 (or latest stable 3.x)
- moviepy / ffmpeg-python (rendering, composition, encoding; requires ffmpeg installed)
- FFmpeg binary available on PATH or specified to the toolchain
- Pillow (PIL) for image processing and sprite manipulation
- Optional: numpy, scipy, librosa, soundfile for audio analysis and DSP

Minimal install suggestion (developer environment):

```bash
python -m venv venv
venv\Scripts\activate.bat   # Windows
pip install moviepy pillow numpy soundfile
```

## Visualizer Types (examples)

- Bar graph (stacked semi-transparent boxes that add/remove with amplitude)
- Oscilloscope / waveform trace (line plot of the waveform)
- Circular / radial visualizer (bars or blobs arranged radially)
- Particle / object-based visualizers (spawned particles that react to energy)

## Example Workflow

1. User places media (image + audio) in an `inputs/` folder or provides paths.
2. Run a render script that: analyzes audio amplitude/FFT, produces per-frame visualizer
   overlays, composes overlays with base image(s), writes output frames or pipes to ffmpeg,
   and encodes final MP4.

## Notes for Claude Code / CLI

- This file is the canonical project summary to provide to Claude when requesting code
  changes, feature additions, or issue triage.
- For generation tasks, prefer prompts that specify: target resolution, visualizer type,
  desired duration, and whether the image is animated.

## Asset & Source Links

- Target reference channel: https://www.youtube.com/channel/UC8qGpw8GUxcQZftXQeFavXw

## Next Steps (developer)

- Add a `requirements.txt` listing chosen libs if not present.
- Add an `inputs/` folder with example image + audio for tests.
- Create a small `render.py` proof-of-concept using `moviepy` and `pydub`/`soundfile`.

---

If you want, I can now:
- create `requirements.txt` and a starter `render.py` script, or
- generate a Claude-ready prompt to implement a specific visualizer (bar, oscilloscope, radial).
