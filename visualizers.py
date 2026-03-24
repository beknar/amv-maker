"""AMV Maker — visualizers.py
Drawing functions for audio visualizers and lightning effects.
"""

import math
import random

import numpy as np
from PIL import ImageDraw

from constants import (
    WIDTH, HEIGHT, FPS, BAR_BOX_H, BAR_GAP, BOX_VGAP,
    DEFAULT_VIS_COLOR, VIS_ALPHA, VIS_BOTTOM_Y, MAX_BOXES,
)


def _vis_rgba(rgb: tuple[int, int, int], alpha: int = VIS_ALPHA) -> tuple[int, int, int, int]:
    """Combine an RGB color with an alpha value."""
    return (rgb[0], rgb[1], rgb[2], alpha)


# ── bar graph ───────────────────────────────────────────────────────────────

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


# ── oscilloscope ────────────────────────────────────────────────────────────

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


# ── radial ──────────────────────────────────────────────────────────────────

def draw_radial_visualizer(draw: ImageDraw.ImageDraw, amplitudes: np.ndarray,
                           color: tuple[int, int, int] = DEFAULT_VIS_COLOR):
    """Bars arranged in a circle at the center of the screen."""
    fill = _vis_rgba(color)
    cx, cy = WIDTH // 2, HEIGHT // 2
    n = len(amplitudes)
    inner_r = HEIGHT // 2
    max_bar_len = int(inner_r * 0.5)
    for i, amp in enumerate(amplitudes):
        angle = 2 * math.pi * i / n - math.pi / 2
        bar_len = inner_r + amp * max_bar_len
        x0 = cx + int(inner_r * math.cos(angle))
        y0 = cy + int(inner_r * math.sin(angle))
        x1 = cx + int(bar_len * math.cos(angle))
        y1 = cy + int(bar_len * math.sin(angle))
        draw.line([(x0, y0), (x1, y1)], fill=fill, width=3)


# ── lightning ───────────────────────────────────────────────────────────────

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

    for width, a_mult in [(7, 0.2), (4, 0.4), (2, 0.8), (1, 1.0)]:
        color = (220, 220, 255, int(alpha * a_mult))
        for j in range(len(points) - 1):
            draw.line([points[j], points[j + 1]], fill=color, width=width)

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
    """Draw lightning bolts on beat frames. Returns flash_alpha for screen flash."""
    if intensity <= 0 or len(beat_frames) == 0:
        return 0

    flash_alpha = 0
    bolt_duration = 6
    flash_duration = 4

    for beat in beat_frames:
        diff = frame_idx - beat
        if diff < 0 or diff >= max(bolt_duration, flash_duration):
            continue

        if diff < flash_duration:
            t = 1.0 - diff / flash_duration
            base_alpha = int(15 + intensity * 8)
            flash_alpha = max(flash_alpha, int(base_alpha * t))

        if diff < bolt_duration:
            fade = 1.0 - diff / bolt_duration
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
