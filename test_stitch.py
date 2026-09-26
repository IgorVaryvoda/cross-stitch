import tempfile

import numpy as np
from PIL import Image, ImageDraw

import stitch

dmc, rgb = stitch.load_dmc()
lab = stitch.to_lab(rgb)
code = {r["dmc"]: i for i, r in enumerate(dmc)}

# 20x20 grey background with a red square and a black square inside
a = np.full((20, 20, 3), (90, 90, 90), np.uint8)
a[5:10, 5:10] = rgb[code["321"]]
a[10:15, 10:15] = rgb[code["310"]]
img = Image.fromarray(a)

chart, fabric = stitch.make_chart(img, 20, 10, lab, fabric=12)
assert (chart[0] == stitch.EMPTY).all(), "border background must be fabric"
assert (chart[5:10, 5:10] == code["321"]).all() and (chart[10:15, 10:15] == code["310"]).all()
assert tuple(fabric) == (90, 90, 90)

chart, _ = stitch.make_chart(img, 20, 1, lab)
assert len(np.unique(chart)) == 1, "colour budget must hold"

lone = np.full((5, 5), code["310"])
lone[2, 2] = code["3371"]  # near-black speck gets absorbed, a red one stays
assert (stitch.clean(lone, lab, 15) == code["310"]).all()
lone[2, 2] = code["321"]
assert stitch.clean(lone, lab, 15)[2, 2] == code["321"]

# A rendered chart: 9 px cells with dark grid lines and pixel noise. The grid reader must return the cells exactly.
rng = np.random.default_rng(1)
pal = rgb[[code["310"], code["321"], code["3865"], code["797"], code["704"]]]
cells = rng.integers(0, len(pal), (30, 40))
r = np.repeat(np.repeat(pal[cells], 9, 0), 9, 1).astype(float)
r[::9], r[:, ::9] = 40, 40
r = np.clip(r + rng.normal(0, 6, r.shape), 0, 255).astype(np.uint8)
g = stitch.read_grid(Image.fromarray(r[4:, 3:]))  # shifted so the grid does not start at pixel 0
assert g is not None and g.shape[:2] == (29, 39), g if g is None else g.shape
assert (stitch.nearest(stitch.to_lab(g).reshape(-1, 3), stitch.to_lab(pal)).reshape(29, 39) == cells[1:, 1:]).all()
assert stitch.read_grid(img) is None, "a plain image is not a stitch render"

# A blend is the linear-light mix of its two threads and carries both DMC numbers.
bd, brgb = stitch.add_blends(dmc, rgb, 15)
i = next(i for i, r in enumerate(bd) if r["dmc"] == "3371+310")
assert len(bd) > len(dmc) and brgb[i].tolist() == [19, 9, 4], brgb[i]

# Dithering a flat mid-tone between two threads mixes both, with the right average; no dither gives one colour.
two = lab[[code["310"], code["B5200"]]]
mid = np.full((20, 20, 3), stitch.to_lab(np.array([[119, 119, 119]]))[0])
out = stitch.dither(mid, two, 1.0)
assert 0.3 < out.mean() < 0.7, out.mean()
assert len(np.unique(stitch.dither(mid, two, 0.0001))) == 1

# One bright pixel (a star) in a dark 60x60 image survives a 3x downscale only with highlights on.
sky = np.full((60, 60, 3), (10, 20, 50), np.uint8)
sky[31, 31] = 255
star = Image.fromarray(sky)
assert stitch.resample(star, 20)[..., 0].max() < 60  # averaged away
assert stitch.resample(star, 20, highlights=25).max() == 255

# Page borders avoid a busy band and fall back to the even split elsewhere.
busy = np.zeros((10, 100), int)
busy[:, 60:70] = np.add.outer(np.arange(10), np.arange(10)) % 3 + 1
xb = stitch.cuts(busy, 0, 47)
assert xb[0] == 0 and xb[-1] == 100 and all(0 < b - a <= 47 for a, b in zip(xb, xb[1:])), xb
assert not any(57 <= b <= 72 for b in xb[1:-1]) and abs(xb[1] - 33) <= 1, xb

# Automatic settings: a smooth night gradient with stars gets dither and highlights; a plain border gets fabric.
y = np.linspace(0, 1, 400)[:, None, None]
night = (np.array([5, 10, 40]) * (1 - y) + np.array([60, 90, 160]) * y) * np.ones((1, 600, 1))
for sy, sx in rng.integers(10, 180, (40, 2)) * [1, 3]:
    night[sy:sy + 2, sx:sx + 2] = 255
auto = stitch.auto_settings(Image.fromarray(night.astype(np.uint8)), 200, None, lab)
assert {"dither", "highlights"} <= set(auto) and "fabric" not in auto, auto
assert "fabric" in stitch.auto_settings(img, 20, None, lab)

# Backstitch: a thin diagonal line becomes one straight diagonal path of the line's colour.
line = Image.new("RGB", (200, 200), (230, 230, 230))
ImageDraw.Draw(line).line((20, 20, 180, 180), fill=(0, 0, 0), width=2)
holes = np.linspace(0, 200, 21)
segs = stitch.backstitch(line, holes, holes, 30, rgb)
assert len(segs) >= 14 and all(i2 - i == 1 and j2 - j == 1 for i, j, i2, j2, _ in segs), segs
assert {dmc[k]["dmc"] for *_, k in segs} == {"310"}

with tempfile.TemporaryDirectory() as d:
    stitch.write_pdf(f"{d}/t.pdf", "t", chart, dmc, rgb, lab, fabric, 14)
print("ok")
