"""Unit tests for render.py — visualizers, particles, audio analysis, rendering."""

import math
import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image, ImageDraw

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from render import (
    WIDTH, HEIGHT, FPS, VIS_BOTTOM_Y, MAX_BOXES, BAR_BOX_H, BOX_VGAP,
    BAR_GAP, DEFAULT_VIS_COLOR, VIS_ALPHA,
    _vis_rgba,
    Petal, Raindrop, Particle,
    draw_bar_visualizer, draw_oscilloscope, draw_radial_visualizer,
    analyse_audio_waveform, build_renderer,
)


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def overlay():
    return Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))


@pytest.fixture
def draw_ctx(overlay):
    return ImageDraw.Draw(overlay)


@pytest.fixture
def synthetic_bar_data():
    random.seed(42)
    np.random.seed(42)
    return np.random.rand(100, 40).astype(np.float32)


@pytest.fixture
def synthetic_image():
    return Image.new("RGBA", (WIDTH, HEIGHT), (30, 30, 60, 255))


def overlay_has_content(overlay):
    """Check if any pixel in the overlay has non-zero alpha."""
    arr = np.array(overlay)
    return arr[:, :, 3].any()


# ── _vis_rgba ───────────────────────────────────────────────────────────────

class TestVisRgba:
    def test_default_alpha(self):
        assert _vis_rgba((200, 80, 200)) == (200, 80, 200, VIS_ALPHA)

    def test_custom_alpha(self):
        assert _vis_rgba((0, 0, 0), alpha=100) == (0, 0, 0, 100)

    def test_white_color(self):
        assert _vis_rgba((255, 255, 255), alpha=50) == (255, 255, 255, 50)


# ── Petal ───────────────────────────────────────────────────────────────────

class TestPetal:
    def test_initial_position_bounds(self):
        random.seed(0)
        for _ in range(50):
            p = Petal()
            assert -50 <= p.x <= WIDTH + 50
            assert 0 <= p.y <= HEIGHT  # initial=True

    def test_reset_non_initial_y_above_screen(self):
        random.seed(0)
        p = Petal()
        p.reset(initial=False)
        assert -HEIGHT <= p.y <= 0

    def test_update_moves_downward(self):
        random.seed(0)
        p = Petal()
        old_y = p.y
        p.update(1.0)
        assert p.y > old_y

    def test_update_wraps_at_bottom(self):
        random.seed(0)
        p = Petal()
        p.y = HEIGHT + 31
        p.update(0.01)
        # after reset, y should be above screen
        assert p.y < HEIGHT

    def test_draw_produces_pixels(self, overlay):
        random.seed(0)
        p = Petal()
        p.x = WIDTH // 2
        p.y = HEIGHT // 2
        p.draw(overlay)
        assert overlay_has_content(overlay)

    def test_draw_small_size_no_crash(self, overlay):
        random.seed(0)
        p = Petal()
        p.size = 0.5
        p.x = WIDTH // 2
        p.y = HEIGHT // 2
        p.draw(overlay)  # should not crash


# ── Raindrop ────────────────────────────────────────────────────────────────

class TestRaindrop:
    def test_initial_position_bounds(self):
        random.seed(0)
        for _ in range(50):
            r = Raindrop()
            assert 0 <= r.x <= WIDTH
            assert 0 <= r.y <= HEIGHT  # initial=True

    def test_reset_non_initial_y_above_screen(self):
        random.seed(0)
        r = Raindrop()
        r.reset(initial=False)
        assert -HEIGHT <= r.y <= 0

    def test_update_moves_downward(self):
        random.seed(0)
        r = Raindrop()
        old_y = r.y
        r.update(0.1)
        assert r.y > old_y

    def test_update_resets_past_bottom(self):
        random.seed(0)
        r = Raindrop()
        r.y = HEIGHT + r.length + 1
        r.update(0.01)
        assert r.y < HEIGHT

    def test_draw_produces_pixels(self, overlay):
        random.seed(0)
        r = Raindrop()
        r.x = WIDTH // 2
        r.y = HEIGHT // 2
        draw = ImageDraw.Draw(overlay)
        r.draw(draw)
        assert overlay_has_content(overlay)

    def test_properties_within_expected_ranges(self):
        random.seed(0)
        r = Raindrop()
        assert 18 <= r.length <= 50
        assert 400 <= r.speed <= 800
        assert 140 <= r.alpha <= 230
        assert r.width in [1, 2, 3]


# ── Particle ────────────────────────────────────────────────────────────────

class TestParticle:
    def test_initial_life(self):
        p = Particle()
        assert p.life == 1.0

    def test_reset_with_higher_energy_gives_more_speed(self):
        random.seed(0)
        p1 = Particle()
        p1.reset(0.0)
        speed_low = math.sqrt(p1.vx ** 2 + p1.vy ** 2)

        random.seed(0)
        p2 = Particle()
        p2.reset(1.0)
        speed_high = math.sqrt(p2.vx ** 2 + p2.vy ** 2)

        assert speed_high > speed_low

    def test_update_decreases_life(self):
        random.seed(0)
        p = Particle()
        p.update(0.5)
        assert p.life < 1.0

    def test_update_applies_gravity(self):
        random.seed(0)
        p = Particle()
        old_vy = p.vy
        p.update(1.0)
        assert p.vy == pytest.approx(old_vy + 40.0)

    def test_draw_does_nothing_when_dead(self, overlay):
        p = Particle()
        p.life = 0
        p.draw(overlay)
        assert not overlay_has_content(overlay)

    def test_draw_renders_when_alive(self, overlay):
        random.seed(0)
        p = Particle()
        p.life = 0.8
        p.x = WIDTH // 2
        p.y = HEIGHT // 2
        p.draw(overlay)
        assert overlay_has_content(overlay)

    def test_draw_with_custom_color(self, overlay):
        random.seed(0)
        p = Particle()
        p.life = 0.8
        p.x = WIDTH // 2
        p.y = HEIGHT // 2
        p.draw(overlay, color=(255, 0, 0))
        arr = np.array(overlay)
        # find pixels with non-zero alpha
        mask = arr[:, :, 3] > 0
        assert mask.any()
        # red channel should be dominant
        assert arr[mask, 0].mean() > arr[mask, 2].mean()


# ── draw_bar_visualizer ─────────────────────────────────────────────────────

class TestDrawBarVisualizer:
    def test_draws_for_nonzero_amplitudes(self, overlay, draw_ctx):
        amps = np.ones(10, dtype=np.float32)
        draw_bar_visualizer(draw_ctx, amps)
        assert overlay_has_content(overlay)

    def test_zero_amplitudes_draws_nothing(self, overlay, draw_ctx):
        amps = np.zeros(10, dtype=np.float32)
        draw_bar_visualizer(draw_ctx, amps)
        assert not overlay_has_content(overlay)

    def test_empty_amplitudes_no_crash(self, overlay, draw_ctx):
        amps = np.array([], dtype=np.float32)
        draw_bar_visualizer(draw_ctx, amps)
        assert not overlay_has_content(overlay)

    def test_single_bar(self, overlay, draw_ctx):
        amps = np.array([0.5], dtype=np.float32)
        draw_bar_visualizer(draw_ctx, amps)
        assert overlay_has_content(overlay)

    def test_custom_color(self, overlay, draw_ctx):
        amps = np.ones(5, dtype=np.float32)
        draw_bar_visualizer(draw_ctx, amps, color=(0, 255, 0))
        arr = np.array(overlay)
        mask = arr[:, :, 3] > 0
        assert mask.any()
        # green channel should be 0 or 255 for drawn pixels
        assert arr[mask, 1].max() == 255

    def test_full_amplitude_draws_max_boxes(self, overlay, draw_ctx):
        amps = np.array([1.0], dtype=np.float32)
        draw_bar_visualizer(draw_ctx, amps)
        arr = np.array(overlay)
        # check that content extends upward from VIS_BOTTOM_Y
        top_row_with_content = np.where(arr[:, :, 3].any(axis=1))[0].min()
        expected_top = VIS_BOTTOM_Y - MAX_BOXES * (BAR_BOX_H + BOX_VGAP)
        assert top_row_with_content <= expected_top + BAR_BOX_H


# ── draw_oscilloscope ───────────────────────────────────────────────────────

class TestDrawOscilloscope:
    def test_short_waveform_no_draw(self, overlay, draw_ctx):
        wf = np.array([0.5])
        draw_oscilloscope(draw_ctx, wf)
        assert not overlay_has_content(overlay)

    def test_empty_waveform_no_crash(self, overlay, draw_ctx):
        wf = np.array([], dtype=np.float32)
        draw_oscilloscope(draw_ctx, wf)
        assert not overlay_has_content(overlay)

    def test_valid_waveform_draws(self, overlay, draw_ctx):
        wf = np.sin(np.linspace(0, 2 * np.pi, 100))
        draw_oscilloscope(draw_ctx, wf)
        assert overlay_has_content(overlay)

    def test_custom_color(self, overlay, draw_ctx):
        wf = np.sin(np.linspace(0, 2 * np.pi, 100))
        draw_oscilloscope(draw_ctx, wf, color=(255, 0, 0))
        arr = np.array(overlay)
        mask = arr[:, :, 3] > 0
        assert mask.any()
        assert arr[mask, 0].mean() > arr[mask, 2].mean()


# ── draw_radial_visualizer ──────────────────────────────────────────────────

class TestDrawRadialVisualizer:
    def test_draws_for_nonzero_amplitudes(self, overlay, draw_ctx):
        amps = np.ones(8, dtype=np.float32)
        draw_radial_visualizer(draw_ctx, amps)
        assert overlay_has_content(overlay)

    def test_empty_amplitudes_no_crash(self, overlay, draw_ctx):
        amps = np.array([], dtype=np.float32)
        draw_radial_visualizer(draw_ctx, amps)

    def test_single_bar(self, overlay, draw_ctx):
        amps = np.array([1.0], dtype=np.float32)
        draw_radial_visualizer(draw_ctx, amps)
        assert overlay_has_content(overlay)

    def test_custom_color(self, overlay, draw_ctx):
        amps = np.ones(8, dtype=np.float32)
        draw_radial_visualizer(draw_ctx, amps, color=(0, 0, 255))
        arr = np.array(overlay)
        mask = arr[:, :, 3] > 0
        assert mask.any()
        assert arr[mask, 2].max() == 255


# ── analyse_audio_waveform ──────────────────────────────────────────────────

class TestAnalyseAudioWaveform:
    def test_correct_frame_count(self):
        sr = 44100
        fps = 30
        duration_s = 2.0
        samples = np.random.randn(int(sr * duration_s)).astype(np.float32)
        waveforms = analyse_audio_waveform(samples, sr, fps)
        expected = len(samples) // (sr // fps)
        assert len(waveforms) == expected

    def test_chunks_normalized(self):
        sr = 44100
        fps = 30
        samples = np.sin(np.linspace(0, 100, sr)).astype(np.float32)
        waveforms = analyse_audio_waveform(samples, sr, fps)
        for chunk in waveforms:
            assert np.abs(chunk).max() <= 1.0 + 1e-6

    def test_silent_audio_no_crash(self):
        sr = 44100
        fps = 30
        samples = np.zeros(sr, dtype=np.float32)
        waveforms = analyse_audio_waveform(samples, sr, fps)
        assert len(waveforms) > 0
        for chunk in waveforms:
            assert np.allclose(chunk, 0)

    def test_very_short_audio(self):
        sr = 44100
        fps = 30
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        waveforms = analyse_audio_waveform(samples, sr, fps)
        assert len(waveforms) == 0

    def test_downsamples_long_chunks(self):
        sr = 44100
        fps = 1  # hop = 44100, much larger than WIDTH
        samples = np.random.randn(sr * 2).astype(np.float32)
        waveforms = analyse_audio_waveform(samples, sr, fps)
        for chunk in waveforms:
            assert len(chunk) == WIDTH


# ── build_renderer ──────────────────────────────────────────────────────────

class TestBuildRenderer:
    def test_returns_callable(self, synthetic_image, synthetic_bar_data):
        make_frame = build_renderer(synthetic_image, synthetic_bar_data, [])
        assert callable(make_frame)

    def test_frame_shape_and_dtype(self, synthetic_image, synthetic_bar_data):
        make_frame = build_renderer(synthetic_image, synthetic_bar_data, [])
        frame = make_frame(0.0)
        assert frame.shape == (HEIGHT, WIDTH, 3)
        assert frame.dtype == np.uint8

    def test_frame_values_in_range(self, synthetic_image, synthetic_bar_data):
        make_frame = build_renderer(synthetic_image, synthetic_bar_data, [])
        frame = make_frame(0.5)
        assert frame.min() >= 0
        assert frame.max() <= 255

    def test_bar_graph_default(self, synthetic_image, synthetic_bar_data):
        make_frame = build_renderer(synthetic_image, synthetic_bar_data, [],
                                    visualizer="Bar Graph")
        frame = make_frame(0.5)
        assert frame.shape == (HEIGHT, WIDTH, 3)

    def test_oscilloscope(self, synthetic_image, synthetic_bar_data):
        sr = 44100
        samples = np.sin(np.linspace(0, 100, sr * 3)).astype(np.float32)
        waveforms = analyse_audio_waveform(samples, sr, FPS)
        make_frame = build_renderer(synthetic_image, synthetic_bar_data, [],
                                    visualizer="Oscilloscope", waveforms=waveforms)
        frame = make_frame(0.5)
        assert frame.shape == (HEIGHT, WIDTH, 3)

    def test_radial(self, synthetic_image, synthetic_bar_data):
        make_frame = build_renderer(synthetic_image, synthetic_bar_data, [],
                                    visualizer="Radial")
        frame = make_frame(0.5)
        assert frame.shape == (HEIGHT, WIDTH, 3)

    def test_particle(self, synthetic_image, synthetic_bar_data):
        particles = [Particle() for _ in range(10)]
        make_frame = build_renderer(synthetic_image, synthetic_bar_data, [],
                                    visualizer="Particle", particles=particles)
        frame = make_frame(0.5)
        assert frame.shape == (HEIGHT, WIDTH, 3)

    def test_zero_petals(self, synthetic_image, synthetic_bar_data):
        make_frame = build_renderer(synthetic_image, synthetic_bar_data, [])
        frame = make_frame(0.0)
        assert frame.shape == (HEIGHT, WIDTH, 3)

    def test_with_petals(self, synthetic_image, synthetic_bar_data):
        random.seed(0)
        petals = [Petal() for _ in range(10)]
        make_frame = build_renderer(synthetic_image, synthetic_bar_data, petals)
        frame = make_frame(0.5)
        assert frame.shape == (HEIGHT, WIDTH, 3)

    def test_with_raindrops(self, synthetic_image, synthetic_bar_data):
        random.seed(0)
        raindrops = [Raindrop() for _ in range(20)]
        make_frame = build_renderer(synthetic_image, synthetic_bar_data, [],
                                    raindrops=raindrops)
        frame_with = make_frame(0.5)
        assert frame_with.shape == (HEIGHT, WIDTH, 3)

    def test_custom_vis_color(self, synthetic_image, synthetic_bar_data):
        make_frame = build_renderer(synthetic_image, synthetic_bar_data, [],
                                    vis_colors={"Bar Graph": (0, 255, 0)})
        frame = make_frame(0.5)
        assert frame.shape == (HEIGHT, WIDTH, 3)

    def test_frame_index_clamped(self, synthetic_image, synthetic_bar_data):
        make_frame = build_renderer(synthetic_image, synthetic_bar_data, [])
        # t way beyond bar_data length — should clamp, not crash
        frame = make_frame(999.0)
        assert frame.shape == (HEIGHT, WIDTH, 3)


# ── analyse_audio (mocked) ──────────────────────────────────────────────────

class TestAnalyseAudio:
    @patch("render.librosa")
    def test_returns_correct_shape(self, mock_librosa):
        from render import analyse_audio

        sr = 44100
        n_samples = sr * 2
        mock_librosa.load.return_value = (np.random.randn(n_samples).astype(np.float32), sr)
        mock_librosa.feature.melspectrogram.return_value = np.random.rand(40, 60).astype(np.float32)
        mock_librosa.power_to_db.return_value = np.random.uniform(-60, 0, (40, 60)).astype(np.float32)

        bars, duration, y, returned_sr = analyse_audio("fake.mp3", 40, 30)
        assert bars.shape[1] == 40
        assert returned_sr == sr
        assert duration == pytest.approx(n_samples / sr)

    @patch("render.librosa")
    def test_amplitudes_in_range(self, mock_librosa):
        from render import analyse_audio

        sr = 44100
        mock_librosa.load.return_value = (np.random.randn(sr).astype(np.float32), sr)
        mock_librosa.feature.melspectrogram.return_value = np.random.rand(40, 30).astype(np.float32)
        mock_librosa.power_to_db.return_value = np.random.uniform(-60, 0, (40, 30)).astype(np.float32)

        bars, *_ = analyse_audio("fake.mp3", 40, 30)
        assert bars.min() >= 0.0
        assert bars.max() <= 1.0


# ── render_video (mocked) ──────────────────────────────────────────────────

class TestRenderVideo:
    @patch("render.AudioFileClip")
    @patch("render.VideoClip")
    @patch("render.Image")
    @patch("render.analyse_audio")
    def test_progress_callback_called(self, mock_analyse, mock_pil, mock_clip_cls, mock_audio_cls):
        from render import render_video

        mock_analyse.return_value = (
            np.random.rand(30, 40).astype(np.float32),
            1.0,
            np.zeros(44100, dtype=np.float32),
            44100,
        )
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_img.resize.return_value = mock_img
        mock_img.copy.return_value = mock_img
        mock_pil.open.return_value = mock_img
        mock_pil.new.return_value = mock_img
        mock_pil.alpha_composite.return_value = mock_img

        mock_clip = MagicMock()
        mock_clip.with_fps.return_value = mock_clip
        mock_clip.with_audio.return_value = mock_clip
        mock_clip_cls.return_value = mock_clip

        mock_audio = MagicMock()
        mock_audio.subclipped.return_value = mock_audio
        mock_audio_cls.return_value = mock_audio

        callback = MagicMock()
        render_video("img.png", "audio.mp3", progress_callback=callback)

        # should be called with 0.0 at start and 1.0 at end
        calls = [c[0][0] for c in callback.call_args_list]
        assert calls[0] == 0.0
        assert calls[-1] == 1.0

    @patch("render.AudioFileClip")
    @patch("render.VideoClip")
    @patch("render.Image")
    @patch("render.analyse_audio")
    def test_returns_absolute_path(self, mock_analyse, mock_pil, mock_clip_cls, mock_audio_cls):
        from render import render_video
        import os

        mock_analyse.return_value = (
            np.random.rand(30, 40).astype(np.float32),
            1.0,
            np.zeros(44100, dtype=np.float32),
            44100,
        )
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_img.resize.return_value = mock_img
        mock_img.copy.return_value = mock_img
        mock_pil.open.return_value = mock_img
        mock_pil.new.return_value = mock_img
        mock_pil.alpha_composite.return_value = mock_img

        mock_clip = MagicMock()
        mock_clip.with_fps.return_value = mock_clip
        mock_clip.with_audio.return_value = mock_clip
        mock_clip_cls.return_value = mock_clip

        mock_audio = MagicMock()
        mock_audio_cls.return_value = mock_audio

        result = render_video("img.png", "audio.mp3", output_path="out.mp4")
        assert os.path.isabs(result)

    @patch("render.AudioFileClip")
    @patch("render.VideoClip")
    @patch("render.Image")
    @patch("render.analyse_audio")
    def test_no_callback_no_crash(self, mock_analyse, mock_pil, mock_clip_cls, mock_audio_cls):
        from render import render_video

        mock_analyse.return_value = (
            np.random.rand(30, 40).astype(np.float32),
            1.0,
            np.zeros(44100, dtype=np.float32),
            44100,
        )
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_img.resize.return_value = mock_img
        mock_img.copy.return_value = mock_img
        mock_pil.open.return_value = mock_img
        mock_pil.new.return_value = mock_img
        mock_pil.alpha_composite.return_value = mock_img

        mock_clip = MagicMock()
        mock_clip.with_fps.return_value = mock_clip
        mock_clip.with_audio.return_value = mock_clip
        mock_clip_cls.return_value = mock_clip

        mock_audio = MagicMock()
        mock_audio_cls.return_value = mock_audio

        render_video("img.png", "audio.mp3", progress_callback=None)
