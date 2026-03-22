"""Unit tests for player.py — VideoPlayer widget."""

import tkinter as tk
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def tk_root():
    """Create a withdrawn tkinter root for testing widgets."""
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


def _make_mock_cap():
    """Create a mock cv2.VideoCapture that returns valid frames."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((800, 1280, 3), dtype=np.uint8))
    mock_cap.get.return_value = 0
    return mock_cap


@pytest.fixture
def player(tk_root):
    with patch("player.pygame") as mock_pg:
        mock_pg.mixer.get_init.return_value = True
        from player import VideoPlayer
        vp = VideoPlayer(tk_root)
        vp._pygame = mock_pg
        yield vp
        # clean up to prevent teardown errors from _show_frame
        vp._cap = None
        vp._after_id = None


# ── __init__ ────────────────────────────────────────────────────────────────

class TestPlayerInit:
    def test_initial_state(self, player):
        assert player._playing is False
        assert player._paused is False
        assert player._cap is None
        assert player._fps == 30.0
        assert player._total_frames == 0

    def test_widgets_exist(self, player):
        assert player.canvas is not None
        assert player.btn_play is not None
        assert player.btn_pause is not None
        assert player.btn_stop is not None
        assert player.time_label is not None
        assert player.seek_slider is not None


# ── load ────────────────────────────────────────────────────────────────────

class TestPlayerLoad:
    @patch("player.cv2")
    def test_raises_if_video_cannot_open(self, mock_cv2, player):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        with pytest.raises(RuntimeError, match="Cannot open video"):
            player.load("bad.mp4", "audio.mp3")

    @patch("player.cv2")
    def test_sets_properties_from_capture(self, mock_cv2, player):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FPS: 24.0,
            mock_cv2.CAP_PROP_FRAME_COUNT: 240,
            mock_cv2.CAP_PROP_POS_FRAMES: 0,
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((800, 1280, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.COLOR_BGR2RGB = 4
        mock_cv2.resize.return_value = np.zeros((400, 640, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((400, 640, 3), dtype=np.uint8)

        # Patch _display_cv_frame to avoid tkinter ImageTk issues in tests
        with patch.object(player, "_display_cv_frame"):
            player.load("test.mp4", "audio.mp3")

        assert player._fps == 24.0
        assert player._total_frames == 240
        assert player._duration == 10.0
        assert player._audio_path == "audio.mp3"


# ── play ────────────────────────────────────────────────────────────────────

class TestPlayerPlay:
    def test_returns_if_no_cap(self, player):
        player._cap = None
        player.play()
        assert player._playing is False

    def test_resumes_from_pause(self, player):
        player._cap = _make_mock_cap()
        player._playing = True
        player._paused = True
        player._audio_path = "audio.mp3"

        with patch("player.pygame") as mock_pg:
            player.play()

        assert player._paused is False

    def test_does_not_double_start(self, player):
        player._cap = _make_mock_cap()
        player._playing = True
        player._paused = False
        old_count = player._frame_count
        player.play()
        # frame_count should not have been reset
        assert player._frame_count == old_count


# ── pause ───────────────────────────────────────────────────────────────────

class TestPlayerPause:
    def test_does_nothing_if_not_playing(self, player):
        player._playing = False
        player.pause()
        assert player._paused is False

    def test_sets_paused_flag(self, player):
        player._playing = True
        player._after_id = None
        with patch("player.pygame"):
            player.pause()
        assert player._paused is True

    def test_cancels_after_callback(self, player):
        player._playing = True
        player._after_id = "fake_id"
        with patch.object(player, "after_cancel") as mock_cancel:
            with patch("player.pygame"):
                player.pause()
            mock_cancel.assert_called_once_with("fake_id")
        assert player._after_id is None


# ── stop ────────────────────────────────────────────────────────────────────

class TestPlayerStop:
    def test_resets_state(self, player):
        player._playing = True
        player._paused = True
        player._after_id = None
        player._cap = None

        with patch("player.pygame"):
            player.stop()

        assert player._playing is False
        assert player._paused is False

    def test_cancels_pending_callback(self, player):
        player._after_id = "some_id"
        with patch.object(player, "after_cancel") as mock_cancel:
            with patch("player.pygame"):
                player.stop()
            mock_cancel.assert_called_once_with("some_id")

    def test_handles_pygame_stop_exception(self, player):
        player._after_id = None
        player._cap = None
        with patch("player.pygame") as mock_pg:
            mock_pg.mixer.music.stop.side_effect = Exception("mixer error")
            player.stop()  # should not raise

    def test_resets_seek_var(self, player):
        player._after_id = None
        player._cap = None
        player.seek_var.set(50.0)
        with patch("player.pygame"):
            player.stop()
        assert player.seek_var.get() == 0


# ── _update_time ────────────────────────────────────────────────────────────

class TestUpdateTime:
    def test_formats_zero(self, player):
        player._duration = 60.0
        player._update_time(0)
        assert "0:00" in player.time_label.cget("text")

    def test_formats_90_seconds(self, player):
        player._duration = 120.0
        player._update_time(90)
        assert "1:30" in player.time_label.cget("text")

    def test_formats_over_10_minutes(self, player):
        player._duration = 700.0
        player._update_time(605)
        assert "10:05" in player.time_label.cget("text")


# ── _on_seek ────────────────────────────────────────────────────────────────

class TestOnSeek:
    def test_does_nothing_if_not_playing(self, player):
        player._playing = False
        player._cap = _make_mock_cap()
        player._on_seek("10.0")
        player._cap.set.assert_not_called()

    def test_does_nothing_if_no_cap(self, player):
        player._playing = True
        player._cap = None
        player._on_seek("10.0")  # should not crash


# ── destroy ─────────────────────────────────────────────────────────────────

class TestDestroy:
    def test_releases_capture(self, player):
        mock_cap = _make_mock_cap()
        player._cap = mock_cap
        player._after_id = None
        with patch("player.pygame"):
            with patch.object(player, "_show_frame"):
                player.destroy()
        mock_cap.release.assert_called_once()
        assert player._cap is None

    def test_handles_pygame_quit_exception(self, player):
        player._cap = None
        player._after_id = None
        with patch("player.pygame") as mock_pg:
            mock_pg.mixer.quit.side_effect = Exception("quit error")
            player.destroy()  # should not raise
