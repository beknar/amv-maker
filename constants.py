"""AMV Maker — constants.py
Shared constants and defaults used across all modules.
"""

WIDTH, HEIGHT = 1280, 800
FPS = 30
BAR_COUNT = 40
BAR_BOX_H = 10
BAR_GAP = 4
BOX_VGAP = 3
DEFAULT_VIS_COLOR = (200, 80, 200)
VIS_ALPHA = 180
VIS_BOTTOM_Y = HEIGHT - 40
MAX_BOXES = 25

NUM_PETALS = 25
NUM_RAINDROPS = 0
LIGHTNING_INTENSITY = 0
HEART_INTENSITY = 0
HEART_COLOR = (255, 80, 150)
CROSSFADE_SECONDS = 3.0
BAR_SWEEP_SPEED = 0.3    # seconds for color sweep across all bars

VISUALIZER_TYPES = ["Bar Graph", "Oscilloscope", "Radial", "Particle"]
