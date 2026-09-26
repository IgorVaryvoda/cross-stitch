"""Score stitch.py against the human chart of Нічка (needs pdftocairo and the local reference files).

    python bench.py
"""
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import stitch

HUMAN_PDF = "donotopenlol/Нічка.pdf"
RENDER = "ref/nichka_key-000.jpg"  # the author's stitch preview, extracted from "Нічка - ключ.pdf" with pdfimages
# Where each of the 12 human chart pages sits on the grid that read_grid() finds in RENDER (found by alignment search)
OFFSETS = {1: (16, -13), 2: (63, -13), 3: (110, -13), 4: (16, 55), 5: (63, 55), 6: (110, 55),
           7: (16, 123), 8: (63, 123), 9: (110, 123), 10: (16, 191), 11: (63, 191), 12: (110, 191)}
CELL = re.compile(r'<path fill-rule="nonzero" fill="rgb\(([\d.]+)%, ([\d.]+)%, ([\d.]+)%\)" fill-opacity="1" '
                  r'd="M ([\d.]+) ([\d.]+) L ([\d.]+) [\d.]+ L [\d.]+ ([\d.]+) L [\d.]+ [\d.]+ Z M [\d.]+ [\d.]+ "/>')


def human_grid(shape):
    """Human chart as colour ids on the render grid; -1 = bare. Grey overlap cells are skipped."""
    grid, colours = np.full(shape, -1), {}
    with tempfile.TemporaryDirectory() as d:
        for page, (ox, oy) in OFFSETS.items():
            svg = Path(d) / f"{page}.svg"
            subprocess.run(["pdftocairo", "-svg", "-f", str(page), "-l", str(page), HUMAN_PDF, str(svg)],
                           check=True, stderr=subprocess.DEVNULL)
            for m in CELL.finditer(svg.read_text()):
                r, g, b, x0, y0, x1, _ = map(float, m.groups())
                if x1 - x0 > 10.5 or abs(r - 75.29) < 0.1 and abs(g - 75.29) < 0.1:
                    continue
                x, y = round((x0 - 69.52) / 9.9492) + ox, round((y0 - 70.68) / 9.9492) + oy
                if 0 <= y < shape[0] and 0 <= x < shape[1]:
                    grid[y, x] = colours.setdefault((r, g, b), len(colours))
    return grid


def agreement(a, b):
    """Share of cells whose colour is consistent with the majority mapping, in both directions."""
    m = (a >= 0) & (b >= 0)
    a, b = a[m], b[m]
    purity = lambda u, v: sum(np.bincount(v[u == k]).max() for k in np.unique(u)) / len(u)
    return purity(a, b), purity(b, a)


def noise(chart):
    """Lone-stitch share and mean number of colours per 10x10 block."""
    h, w = chart.shape
    p = np.pad(chart, 1, constant_values=-2)
    nb = np.stack([p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
    lone = ((nb != chart).all(0) & (chart >= 0)).sum() / (chart >= 0).sum()
    blocks = [len(np.unique(b[b >= 0])) for y in range(0, h - 9, 10) for x in range(0, w - 9, 10)
              for b in [chart[y:y + 10, x:x + 10]]]
    return lone, np.mean(blocks)


# Photo mode has no human chart; score what a viewer sees: colour error after a one-stitch blur, plus confetti.
PHOTOS = [  # (image, width, colours, blends, make_chart options, clean)
    ("sunset.png", 250, 60, 20, {}, 15),
    ("sunset.png", 250, 40, 0, {"dither_strength": 0.5}, 0),
    ("moon.png", 200, 40, 0, {"dither_strength": 0.5, "highlights": 25}, 0),
    ("moon.png", 200, 40, 0, {"highlights": 25}, 15),
    ("moon.png", 200, 50, 20, {"dither_strength": 0.3, "highlights": 25}, 0),
    ("nichka.jpg", 175, 43, 0, {}, 15),
]


def photo_scores(chart, target, pal_rgb):
    blur = lambda a: np.asarray(Image.fromarray(a.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1)), float)
    seen = np.linalg.norm(stitch.to_lab(blur(pal_rgb[chart])) - stitch.to_lab(blur(target)), axis=-1).mean()
    cell = np.linalg.norm(stitch.to_lab(pal_rgb[chart].astype(float)) - stitch.to_lab(target), axis=-1).mean()
    return seen, cell, noise(chart)[0]


def photo_bench(**overrides):
    dmc, rgb = stitch.load_dmc()
    for f, width, colours, blends, opts, cl in PHOTOS:
        d, r = stitch.add_blends(dmc, rgb, blends) if blends else (dmc, rgb)
        lab = stitch.to_lab(r)
        img = Image.open(f).convert("RGB")
        chart, _ = stitch.make_chart(img, width, colours, lab, **{**opts, **overrides})
        if cl:
            chart = stitch.clean(chart, lab, cl)
        target = stitch.resample(img, width)
        seen, cell, lone = photo_scores(chart, target, r)
        threads = {c for k in np.unique(chart) for c in d[k]["dmc"].split("+")}
        tag = f"{f} w{width} c{colours} b{blends} {opts}"
        print(f"{tag:62s} seen dE {seen:.2f}  cell dE {cell:.2f}  lone {lone:.3f}  threads {len(threads)}", flush=True)


if __name__ == "__main__":
    dmc, rgb = stitch.load_dmc()
    lab = stitch.to_lab(rgb)
    raw, _ = stitch.make_chart(Image.open(RENDER), None, 43, lab, fabric=0)
    human = human_grid(raw.shape)
    for name, chart in (("human chart", human), ("stitch.py, no clean", raw), ("stitch.py, clean 15", stitch.clean(raw, lab, 15))):
        lone, blk = noise(chart)
        hm, mh = agreement(human, chart)
        bare = ((human >= 0) == (chart >= 0)).mean()
        print(f"{name:22s} agree {hm:.3f}/{mh:.3f}  bare {bare:.4f}  lone {lone:.3f}  colours/10x10 {blk:.1f}")
    photo_bench()
