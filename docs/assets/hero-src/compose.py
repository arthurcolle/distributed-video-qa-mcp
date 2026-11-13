"""Hero compositor. Everything drawn sits on one square grid: pitch P, margin M, 27 x 13 cells at 3584 x 1792."""
import sys, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ART, MARK, OUT = map(Path, sys.argv[1:4])   # e.g. hero-src/art.jpg ../distributed-wordmark.png out.png
DEBUG = "--debug" in sys.argv
W, H = 3584, 1792
P, M = 128, 64                      # (W - 2M) / P = 27, (H - 2M) / P = 13
COL = lambda k: M + k * P
ROW = lambda k: M + k * P
SCALE = 1.0                         # scene scale, anchored bottom-right
DISC = (2689, 719, 627)             # fitted to the disc's top arc (median residual 0.44 px)
COPPER = (226, 166, 116); CREAM = (244, 237, 226); GREY = (176, 168, 158); RULE = (74, 56, 44)

# ------------------------------------------------------------------ base: scaled scene + extended background
src = Image.open(ART).convert("RGB")
sw, sh = int(W * SCALE), int(H * SCALE)
scene = src.resize((sw, sh), Image.LANCZOS)
CONTENT_RIGHT = 3465                                                 # answer card's right edge, 55 px inside the frame
ox, oy = 0, 0                                                        # the recomposed art is used at full size
arr = np.asarray(scene).astype(np.float32)
canvas = np.pad(arr, ((oy, H - oy - sh), (ox, W - ox - sw), (0, 0)), mode="edge")   # extend the flat background on every side
blurred = np.asarray(Image.fromarray(canvas.astype(np.uint8)).filter(ImageFilter.GaussianBlur(40))).astype(np.float32)
yy, xx = np.mgrid[0:H, 0:W]
feather = 160.0
wx = np.ones((H, W)); wy = np.ones((H, W))
w = (wx * wy)[..., None]
base = canvas * w + blurred * (1 - w)
lum = base.mean(axis=2)
def S(x, y):  # map a source-art coordinate into the canvas
    return ox + x * SCALE, oy + y * SCALE

# ------------------------------------------------------------------ grid, drawn only over dark background (passes behind objects)
grid = Image.new("L", (W, H), 0); g = ImageDraw.Draw(grid)
for k in range(28): g.line([(COL(k), ROW(0)), (COL(k), ROW(13))], fill=255, width=2)
for k in range(14): g.line([(COL(0), ROW(k)), (COL(27), ROW(k))], fill=255, width=2)
behind = np.clip((34.0 - lum) / 18.0, 0, 1)                         # 1 on background, 0 on objects
ga = np.asarray(grid).astype(np.float32) / 255.0 * 0.075 * behind
base = base * (1 - ga[..., None]) + np.array(COPPER, np.float32) * ga[..., None]
img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))

over = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(over)

def cross(x, y, arm=30, width=3, alpha=230):
    d.line([(x - arm, y), (x + arm, y)], fill=(*COPPER, alpha), width=width)
    d.line([(x, y - arm), (x, y + arm)], fill=(*COPPER, alpha), width=width)

frame = Image.new("RGBA", (W, H), (0, 0, 0, 0)); fd = ImageDraw.Draw(frame)

def hline(y, x0, x1, alpha=255, color=RULE, width=2):
    fd.line([(x0, y), (x1, y)], fill=(*color, alpha), width=width)   # frame layer: masked behind objects

# frame: top and bottom rules between the corner marks, left rule down the margin
ARM = 30
hline(ROW(0), COL(0) + ARM + 14, COL(27) - ARM - 14, alpha=200)
# The plinth breaks the bottom of the frame, so the bottom line stops one cell before the first object on that row.
band = lum[ROW(13) - 10:ROW(13) + 10]
first_obj = int(np.argmax((band > 18).any(axis=0)))
BOTTOM_END = COL(int((first_obj - P - M) // P))
hline(ROW(13), COL(0) + ARM + 14, BOTTOM_END, alpha=200)
fd.line([(COL(0), ROW(0) + ARM + 14), (COL(0), ROW(13) - ARM - 14)], fill=(*RULE, 170), width=2)
for x, y in [(COL(0), ROW(0)), (COL(27), ROW(0)), (COL(0), ROW(13)), (COL(27), ROW(13))]:
    cross(x, y)

# ------------------------------------------------------------------ typography (cap height = 1 cell, line pitch = 1.5 cells)
AV = "/System/Library/Fonts/Avenir Next.ttc"
heavy = next(i for i in range(16) if ImageFont.truetype(AV, 10, index=i).getname()[1] == "Heavy")
def capheight(font):
    box = font.getbbox("H"); return box[3] - box[1]
size = 100
while capheight(ImageFont.truetype(AV, size + 1, index=heavy)) <= P: size += 1
head = ImageFont.truetype(AV, size, index=heavy)
lines = ["ANY VIDEO.", "EVERY FRAME.", "ANSWERED."]
X0 = COL(1)
baselines = [ROW(3), ROW(4) + P // 2, ROW(6)]                        # first and last on grid rows
txt = Image.new("RGBA", (W, H), (0, 0, 0, 0)); t = ImageDraw.Draw(txt)
glow = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
for line, base_y in zip(lines, baselines):
    left_bearing = head.getbbox(line)[0]
    t.text((X0 - left_bearing, base_y), line, font=head, fill=(*CREAM, 255), anchor="ls")
    gd.text((X0 - left_bearing, base_y), line, font=head, fill=(150, 90, 45, 120), anchor="ls")
text_right = max(X0 + head.getlength(l) for l in lines)

# rule under the headline (row 7), crosshair where it meets the left frame
hline(ROW(7), COL(0) + ARM + 14, COL(8), alpha=255)
cross(COL(0), ROW(7), arm=22, width=3, alpha=200)

# wordmark: 1.5 cells tall, top at row 7.5, bottom on row 9
mark = Image.open(MARK).convert("RGBA")
alpha = np.asarray(mark)[..., 3]
rows_on = np.nonzero(alpha.max(axis=1) > 40)[0]; cols_on = np.nonzero(alpha.max(axis=0) > 40)[0]
mark = mark.crop((int(cols_on.min()), int(rows_on.min()), int(cols_on.max()) + 1, int(rows_on.max()) + 1))
mh = P; mw = round(mark.width * mh / mark.height)
mark = mark.resize((mw, mh), Image.LANCZOS)
MARK_TOP = ROW(8)                   # visible letters span exactly row 8 to row 9

# mono sublines, letter-spaced, baselines on row 10 and 10.75
MONO = next(p for p in ["/Users/arthurcolle/Library/Fonts/JetBrainsMonoNerdFontMono-Regular.ttf",
                        "/Library/Fonts/JetBrainsMonoNerdFontMono-Regular.ttf"] if Path(p).exists())
def spaced(draw, xy, text, font, fill, tracking):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill, anchor="ls")
        x += font.getlength(ch) + tracking
    return x
m1 = ImageFont.truetype(MONO, 50); m2 = ImageFont.truetype(MONO, 30)
MONO1 = ROW(10) + P // 2            # row 10.5
MONO2 = ROW(11)                     # row 11
x_end1 = spaced(t, (X0, MONO1), "Distributed Video QA MCP", m1, (*CREAM, 255), 3)
x_end2 = spaced(t, (X0 + 1, MONO2), "DOWNLOAD / TRANSCRIBE / EMBED / ASK", m2, (*GREY, 255), 7)

# bottom rule (row 12) with crosshair on the frame
hline(ROW(12), COL(0) + ARM + 14, COL(8), alpha=255)
cross(COL(0), ROW(12), arm=22, width=3, alpha=200)

# ------------------------------------------------------------------ details: dashed arc concentric with the disc, tick strip
cx, cy = S(DISC[0], DISC[1]); r = DISC[2] * SCALE + 46
arc = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ad = ImageDraw.Draw(arc)
for a0 in np.arange(116.0, 206, 3.2):            # upper-left quadrant only; dashes 2 degrees long
    pts = [(cx + r * math.cos(math.radians(a)), cy - r * math.sin(math.radians(a))) for a in np.linspace(a0, a0 + 2.0, 6)]
    ad.line(pts, fill=(*COPPER, 130), width=3)
arc_np = np.asarray(arc).copy()
arc_np[..., 3] = (arc_np[..., 3].astype(np.float32) * behind).astype(np.uint8)   # passes behind objects
arc = Image.fromarray(arc_np)
tick_y0, tick_y1 = ROW(12) + P // 2, ROW(12) + 3 * P // 4
for i in range(13):                       # 13 ticks, 1/6 cell apart, from column 24 to column 26
    x = round(COL(24) + i * P / 6)
    d.line([(x, tick_y0), (x, tick_y1)], fill=(*COPPER, 230 if i % 6 == 0 else 140), width=3)

if DEBUG:
    for k in range(28): d.line([(COL(k), 0), (COL(k), H)], fill=(0, 255, 255, 60), width=1)
    for k in range(14): d.line([(0, ROW(k)), (W, ROW(k))], fill=(0, 255, 255, 60), width=1)

# ------------------------------------------------------------------ composite
img = img.convert("RGBA")
img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(26)))
img = Image.alpha_composite(img, arc)
frame_np = np.asarray(frame).copy()
frame_np[..., 3] = (frame_np[..., 3].astype(np.float32) * behind).astype(np.uint8)
img = Image.alpha_composite(img, Image.fromarray(frame_np))
img = Image.alpha_composite(img, over)
img = Image.alpha_composite(img, txt)
img.paste(mark, (X0, MARK_TOP), mark)
ink = np.asarray(txt)[..., 3] > 30
ink[MARK_TOP:MARK_TOP + mh, X0:X0 + mw] |= np.asarray(mark)[..., 3] > 30
ink_d = np.asarray(Image.fromarray((ink * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(97))) > 0  # 48 px halo
objects = lum > 26
clash = int((ink_d & objects).sum())
ys, xs = np.nonzero(ink_d & objects)
print(f"collision check: {clash} artwork pixels inside the 48 px halo of the type"
      + (f" (x {xs.min()}..{xs.max()}, y {ys.min()}..{ys.max()})" if clash else ""))
img = img.convert("RGB")
img.save(OUT.with_suffix(".full.png"))
img.resize((W // 2, H // 2), Image.LANCZOS).save(OUT, optimize=True)
print(f"bottom frame line ends at {BOTTOM_END} (first object on that row at x {first_obj})")
print(f"answer card right edge {CONTENT_RIGHT} (frame column {COL(27)}, clearance {COL(27) - CONTENT_RIGHT})")
cap1 = MONO1 - capheight(m1)
print(f"rhythm: rule7 {ROW(7)} -> mark top {MARK_TOP} (gap {MARK_TOP - ROW(7)}), mark bottom {MARK_TOP + mh}, "
      f"mono1 cap top {cap1} (gap {cap1 - MARK_TOP - mh}), mono2 baseline {MONO2} -> rule12 {ROW(12)} (gap {ROW(12) - MONO2})")
print(f"headline {size}px (cap {capheight(head)}), right edge {text_right:.0f}; mono ends {x_end1:.0f}/{x_end2:.0f}; "
      f"arc r {r:.0f} about ({cx:.0f},{cy:.0f})")
