import pygame
import math
import os
from PIL import Image, ImageFilter

# ============================================================
# REALISTIC HEARTBEAT ANIMATION
# Put your heart image in the same folder as this file:
#     heart.png
#
# The supplied image can simply be renamed to "heart.png".
# ============================================================

pygame.init()

# ---------------- WINDOW ----------------
WIDTH, HEIGHT = 900, 700
FPS = 120

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Realistic Heart Beat ❤️")

clock = pygame.time.Clock()

# ---------------- COLORS ----------------
BG = (255, 255, 255)

# ---------------- LOAD HEART ----------------
IMAGE_FILE = "heart.png"

if not os.path.exists(IMAGE_FILE):
    raise FileNotFoundError(
        "heart.png nahi mila!\n\n"
        "Is Python file ke same folder me apni heart image ko "
        "'heart.png' naam se rakho."
    )

original = Image.open(IMAGE_FILE).convert("RGBA")

# Image ko square/canvas ke andar fit karna
MAX_SIZE = 430

scale = min(
    MAX_SIZE / original.width,
    MAX_SIZE / original.height
)

base_w = int(original.width * scale)
base_h = int(original.height * scale)

original = original.resize(
    (base_w, base_h),
    Image.Resampling.LANCZOS
)

# ------------------------------------------------------------
# White background ko transparent banana.
# Isse image ka white square nahi dikhega.
# Bahut halka shadow preserve kiya jayega.
# ------------------------------------------------------------
pixels = original.load()

for y in range(original.height):
    for x in range(original.width):

        r, g, b, a = pixels[x, y]

        # Pure/near white pixels transparent
        brightness = (r + g + b) / 3

        if r > 245 and g > 245 and b > 245:
            pixels[x, y] = (r, g, b, 0)

        # Light gray shadow ko halka transparent rakho
        elif r > 210 and g > 210 and b > 210:
            alpha = int(max(0, 255 - (brightness - 210) * 8))
            pixels[x, y] = (r, g, b, min(a, alpha))

heart_base = pygame.image.fromstring(
    original.tobytes(),
    original.size,
    original.mode
).convert_alpha()

# ---------------- SOFT GLOW ----------------
glow_source = original.copy()

# Heart ke around soft pink glow
glow = glow_source.filter(ImageFilter.GaussianBlur(18))

glow_surface = pygame.image.fromstring(
    glow.tobytes(),
    glow.size,
    glow.mode
).convert_alpha()

# Glow ko thoda transparent
glow_surface.set_alpha(45)

# ---------------- HEARTBEAT SETTINGS ----------------
# One complete heartbeat cycle.
# Values represent seconds.

CYCLE = 1.05

# Realistic "lub-dub":
#
# 0.00 -> normal
# 0.10 -> first beat
# 0.20 -> relax
# 0.31 -> stronger second beat
# 0.45 -> relax
# 1.05 -> rest

def smoothstep(t):
    """Very smooth 0..1 transition."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def pulse_amount(time):
    """
    Returns heartbeat intensity.
    Two pulses:
        small first beat
        stronger second beat
    """

    t = time % CYCLE

    # ---------------- FIRST BEAT ----------------
    if 0.045 <= t < 0.155:
        p = (t - 0.045) / 0.110

        # 0 -> 1 -> 0
        if p < 0.5:
            v = smoothstep(p * 2.0)
        else:
            v = 1.0 - smoothstep((p - 0.5) * 2.0)

        return v * 0.075

    # ---------------- SECOND BEAT ----------------
    if 0.235 <= t < 0.385:
        p = (t - 0.235) / 0.150

        if p < 0.5:
            v = smoothstep(p * 2.0)
        else:
            v = 1.0 - smoothstep((p - 0.5) * 2.0)

        return v * 0.145

    return 0.0


# ---------------- DRAW FUNCTION ----------------
def draw_heart(elapsed):

    pulse = pulse_amount(elapsed)

    # Subtle breathing even when not beating
    resting = 0.008 * math.sin(elapsed * 2.0 * math.pi / CYCLE)

    intensity = pulse + resting

    # Slightly different X/Y scaling makes it feel less robotic.
    scale_x = 1.0 + intensity
    scale_y = 1.0 + intensity * 0.94

    new_w = max(1, int(heart_base.get_width() * scale_x))
    new_h = max(1, int(heart_base.get_height() * scale_y))

    # Smooth high-quality scaling
    heart = pygame.transform.smoothscale(
        heart_base,
        (new_w, new_h)
    )

    # Glow also expands with the beat
    glow_x = max(1, int(glow_surface.get_width() * (1.0 + intensity * 1.25)))
    glow_y = max(1, int(glow_surface.get_height() * (1.0 + intensity * 1.25)))

    current_glow = pygame.transform.smoothscale(
        glow_surface,
        (glow_x, glow_y)
    )

    # Glow becomes brighter during heartbeat
    glow_alpha = int(28 + pulse * 500)
    glow_alpha = max(20, min(90, glow_alpha))
    current_glow.set_alpha(glow_alpha)

    # ---------------- POSITION ----------------
    center_x = WIDTH // 2
    center_y = HEIGHT // 2 - 10

    glow_rect = current_glow.get_rect(
        center=(center_x, center_y)
    )

    heart_rect = heart.get_rect(
        center=(center_x, center_y)
    )

    # ---------------- DRAW ----------------
    screen.blit(current_glow, glow_rect)
    screen.blit(heart, heart_rect)


# ---------------- MAIN LOOP ----------------
running = True
start_time = pygame.time.get_ticks() / 1000.0

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            # ESC = exit
            if event.key == pygame.K_ESCAPE:
                running = False

    elapsed = pygame.time.get_ticks() / 1000.0 - start_time

    # White background
    screen.fill(BG)

    # Heart
    draw_heart(elapsed)

    pygame.display.flip()

    clock.tick(FPS)


pygame.quit()
