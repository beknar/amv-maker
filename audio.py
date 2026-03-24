"""AMV Maker — audio.py
Audio analysis, waveform extraction, beat detection, and concatenation.
"""

import librosa
import numpy as np

from constants import WIDTH, CROSSFADE_SECONDS


def analyse_audio(audio_path: str, n_bars: int, fps: int):
    """Return a (frames, n_bars) array of per-bar amplitudes in 0..1 range.

    Uses a mel-scaled spectrogram with dB scaling for perceptually even bars.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    hop = sr // fps

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_bars, hop_length=hop, n_fft=2048
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = np.clip(S_db, -60, 0)
    bars = ((S_db + 60) / 60).T.astype(np.float32)

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
    onset_env = librosa.onset.onset_strength(y=audio_samples, sr=sr, hop_length=sr // fps)
    peaks = librosa.util.peak_pick(
        onset_env, pre_max=3, post_max=3, pre_avg=5, post_avg=5,
        delta=0.3, wait=int(min_gap_s * fps)
    )
    if len(peaks) == 0:
        return np.array([], dtype=int)
    strengths = onset_env[peaks]
    order = np.argsort(-strengths)
    selected = []
    for idx in order:
        frame = peaks[idx]
        if all(abs(frame - s) >= min_gap_s * fps for s in selected):
            selected.append(frame)
    selected.sort()
    return np.array(selected, dtype=int)


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

    xfade_samples = int(crossfade_s * target_sr)
    track_end_times = []
    combined = all_tracks[0].copy()

    for i in range(1, len(all_tracks)):
        xfade = min(xfade_samples, len(combined), len(all_tracks[i]))

        if xfade > 0:
            fade_out = np.linspace(1.0, 0.0, xfade, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, xfade, dtype=np.float32)
            overlap = combined[-xfade:] * fade_out + all_tracks[i][:xfade] * fade_in
            combined[-xfade:] = overlap
            combined = np.concatenate([combined, all_tracks[i][xfade:]])
        else:
            combined = np.concatenate([combined, all_tracks[i]])

        track_end_times.append(len(combined) / target_sr -
                               len(all_tracks[i][xfade:]) / target_sr -
                               xfade / target_sr / 2)

    track_end_times.append(len(combined) / target_sr)

    sf.write(output_path, combined, target_sr)
    return output_path, track_end_times
