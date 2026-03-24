"""AMV Maker — effects.py
Particle/animation effect classes: Petal, Raindrop, Heart, Particle.
"""

import math
import random

from PIL import Image, ImageDraw

from constants import (
    WIDTH, HEIGHT, DEFAULT_VIS_COLOR, VIS_ALPHA, VIS_BOTTOM_Y, HEART_COLOR,
)


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


# ── heart particles ─────────────────────────────────────────────────────────

def _heart_polygon(cx: float, cy: float, size: float) -> list[tuple[int, int]]:
    """Generate heart shape as a polygon of points."""
    points = []
    steps = 30
    for i in range(steps):
        t = 2 * math.pi * i / steps
        x = size * 0.8 * (16 * math.sin(t) ** 3) / 16
        y = -size * 0.8 * (13 * math.cos(t) - 5 * math.cos(2 * t) -
                           2 * math.cos(3 * t) - math.cos(4 * t)) / 16
        points.append((int(cx + x), int(cy + y)))
    return points


class Heart:
    """A hollow heart that fades in and out in the top-left area."""

    def __init__(self, max_hearts: int):
        self.max_hearts = max_hearts
        self.reset()

    def reset(self):
        self.x = random.uniform(30, WIDTH * 0.3)
        self.y = random.uniform(30, HEIGHT * 0.3)
        self.size = random.uniform(12, 30)
        self.phase = 0.0
        self.speed = random.uniform(0.4, 0.8)
        self.active = False
        self.wait = random.uniform(0.5, 3.0)

    def update(self, dt: float):
        if not self.active:
            self.wait -= dt
            if self.wait <= 0:
                self.active = True
            return
        self.phase += self.speed * dt
        if self.phase >= 3.0:
            self.reset()

    def get_alpha(self) -> int:
        if not self.active:
            return 0
        if self.phase < 1.0:
            return int(220 * self.phase)
        elif self.phase < 2.0:
            return 220
        else:
            return int(220 * (3.0 - self.phase))

    def draw(self, draw_ctx: ImageDraw.ImageDraw,
             color: tuple[int, int, int] = HEART_COLOR):
        alpha = self.get_alpha()
        if alpha <= 0:
            return
        cx, cy = int(self.x), int(self.y)
        fill = (color[0], color[1], color[2], alpha)
        pts = _heart_polygon(cx, cy, self.size)
        draw_ctx.polygon(pts, outline=fill)
        for offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = [(p[0] + offset[0], p[1] + offset[1]) for p in pts]
            draw_ctx.polygon(shifted, outline=fill)


# ── energy-reactive particles ───────────────────────────────────────────────

def _vis_rgba(rgb: tuple[int, int, int], alpha: int = VIS_ALPHA) -> tuple[int, int, int, int]:
    """Combine an RGB color with an alpha value."""
    return (rgb[0], rgb[1], rgb[2], alpha)


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
        self.vy += 40 * dt
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
