#!/usr/bin/env python3
"""Turn an image into a DMC cross-stitch pattern: PDF chart + PNG preview.

    python stitch.py photo.jpg --width 200                 # fabric, blends, dither, highlights are picked automatically
    python stitch.py drawing.png --width 150 --backstitch  # trace thin outlines as backstitch
    python stitch.py render.jpg                            # a render of stitches: the grid is read cell by cell

A stitch render needs about 4 px per stitch or more; below that, give --width and it is treated as a photo.
Any option you give yourself wins over the automatic choice; --no-auto turns the automatic choices off.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from scipy import ndimage

HERE = Path(__file__).parent
SYMBOLS = "ABCDEFGHJKLMNPQRTXYZ23456789#%&@+=?$<>abdefghknqrt"
SHAPES = "●■▲◆★♥✚✖○□△◇☆♡✕▼▽◐◑◒◓◧◨◩◪▣▤▥▦▧▨◈◉◎⊕⊗⊙⊞⊠♣♠♦✿❄✱✳✶⬟⬢≡≈±÷§¶#%&@$?=+<>♪☀☂☾✓✗"\
         "ABCDEFGHJKLMNPRSTUVWXYZ23456789"  # shapes first: the most used colours get them
FONTS = ["/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/Library/Fonts/DejaVuSans.ttf", "C:/Windows/Fonts/DejaVuSans.ttf"]
FONT = next((f for f in FONTS if Path(f).exists()), None)  # shape symbols need a font that has them
SYMS = SHAPES if FONT else SYMBOLS
STITCHES_PER_SKEIN_14CT = 1800  # ponytail: rough 2-strand full-cross estimate, tune from your own usage
EMPTY = -1
CELL = 3.4 * mm
COLS_PP, ROWS_PP = 47, 68
OVERLAP = 2  # rows/columns repeated from the next page, greyed out


def to_lab(rgb):
    c = np.asarray(rgb, float) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    f = xyz / [0.95047, 1.0, 1.08883]
    f = np.where(f > 0.008856, np.cbrt(f), 7.787 * f + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def nearest(lab, pal, chunk=2048):
    p2 = (pal**2).sum(1)
    return np.concatenate([(p2 - 2 * lab[i:i + chunk] @ pal.T).argmin(1) for i in range(0, len(lab), chunk)])


def load_dmc():
    with open(HERE / "dmc.csv") as f:
        rows = list(csv.DictReader(f))
    return rows, np.array([[int(r["r"]), int(r["g"]), int(r["b"])] for r in rows])


def add_blends(dmc, dmc_rgb, max_delta):
    """Add a blend (one strand of each colour in the needle) for every pair of colours closer than max_delta.

    Blends fill the gaps between DMC colours, which matters for soft gradients like skies and skin."""
    lab = to_lab(dmc_rgb)
    lin = np.where(dmc_rgb / 255 > 0.04045, ((dmc_rgb / 255 + 0.055) / 1.055) ** 2.4, dmc_rgb / 255 / 12.92)
    i, j = np.nonzero(np.triu(np.linalg.norm(lab[:, None] - lab[None], axis=-1) < max_delta, 1))
    mix = (lin[i] + lin[j]) / 2
    mix = 255 * np.where(mix > 0.0031308, 1.055 * mix ** (1 / 2.4) - 0.055, 12.92 * mix)
    rows = [{"dmc": f"{dmc[a]['dmc']}+{dmc[b]['dmc']}", "name": f"{dmc[a]['name']} + {dmc[b]['name']}"} for a, b in zip(i, j)]
    return dmc + rows, np.concatenate([dmc_rgb, mix.round().astype(int)])


def background_mask(lab, tol=None):
    """Stitches close to the border colour and connected to the border stay as bare fabric.

    Without a tolerance, use the colour spread of the border itself: tight for clean renders, looser for photos."""
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    fabric = np.median(border, 0)
    if tol is None:
        tol = max(2.5, 1.3 * np.percentile(np.linalg.norm(border - fabric, axis=1), 99))
    labels, _ = ndimage.label(np.linalg.norm(lab - fabric, axis=-1) < tol)
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    return np.isin(labels, edge[edge > 0])


def _edges(gray, axis):
    g = np.abs(np.diff(gray, axis=axis)).mean(axis=1 - axis)
    return g - np.convolve(g, np.ones(31) / 31, "same")


def _autocorr(g, n):
    ac = np.correlate(g, g, "full")[len(g) - 1:][:n]
    return ac / ac[0]


def find_grid(img, min_pitch=4, max_pitch=60, min_conf=0.8):
    """If the image is a render of stitches (a chart photo or preview), return the left/top pixel of every
    column/row of cells, else None."""
    if min(img.size) < 64:
        return None
    rgb = np.asarray(img.convert("RGB"))
    gray = rgb.mean(2)
    gx, gy = _edges(gray, 1), _edges(gray, 0)
    ac = (_autocorr(gx, max_pitch + 2) + _autocorr(gy, max_pitch + 2)) / 2
    peaks = [l for l in range(min_pitch, max_pitch) if ac[l] > ac[l - 1] and ac[l] >= ac[l + 1]]
    if not peaks or max(ac[peaks]) < min_conf:
        return None
    lag = min(l for l in peaks if ac[l] >= 0.8 * max(ac[peaks]))  # fundamental, not a harmonic
    grids = []
    for g in (gx, gy):
        # Fit a comb of grid lines: sub-pixel pitch and phase that land on the most edge energy.
        pos = np.arange(len(g))
        best = (-np.inf, 0, 0)
        for pitch in np.arange(lag - 0.6, lag + 0.6, 0.002):
            n = int((len(g) - 1) // pitch)
            for q in np.linspace(0, pitch, 32, endpoint=False):
                s = np.interp(q + pitch * np.arange(n), pos, g).sum()
                if s > best[0]:
                    best = (s, pitch, q)
        _, pitch, q = best
        grids.append(np.arange(q + 1, len(g) + 1 - 0.7 * pitch, pitch))  # +1: diff() shifts edges left
    return grids


def read_grid(img, grid=None):
    """One RGB value per cell of a stitch render, or None if the image is not one."""
    grid = grid or find_grid(img)
    if grid is None:
        return None
    rgb = np.asarray(img.convert("RGB"))
    xs, ys = grid
    p = xs[1] - xs[0]
    lo, hi = round(p * 0.3), round(p * 0.7)  # cell centre only, away from grid lines and cross texture
    return np.array([[np.median(rgb[round(y) + lo:round(y) + hi, round(x) + lo:round(x) + hi].reshape(-1, 3), 0)
                      for x in xs] for y in ys])


def _box(a, size):
    """Area-average a float image (H x W x C) down to size."""
    return np.stack([np.asarray(Image.fromarray(a[..., i].astype(np.float32)).resize(size, Image.Resampling.BOX))
                     for i in range(a.shape[2])], -1)


def resample(img, width, highlights=0, lowlights=0):
    """Average each stitch cell, but keep details thinner than a stitch that averaging would wash out.

    A top-hat filter finds features narrower than about 1.5 stitches that are highlights (L*) lighter than their
    surroundings (stars, sparkles) or lowlights darker (hair strands, ripples, ink lines). A cell that such a
    feature touches takes the feature's colour. Edges between large areas are not features, so shapes do not grow."""
    size = (width, round(width * img.height / img.width))
    small = np.asarray(img.resize(size, Image.Resampling.BOX)).astype(float)
    if not (highlights or lowlights):
        return small.round().astype(np.uint8)
    rgb = np.asarray(img, float)
    lum = to_lab(rgb)[..., 0]
    k = max(3, round(1.5 * img.width / width) | 1)
    for thresh, hat in ((lowlights, lambda: ndimage.grey_closing(lum, k) - lum),
                        (highlights, lambda: lum - ndimage.grey_opening(lum, k))):  # highlights last: stars win
        if not thresh:
            continue
        m = (hat() > thresh).astype(float)[..., None]
        cover = _box(m, size)
        colour = _box(rgb * m, size) / np.maximum(cover, 1e-9)
        small = np.where(cover > 0.02, colour, small)
    return small.round().astype(np.uint8)


def backstitch(img, hx, hy, thresh, dmc_rgb, max_threads=4, min_len=3, render=False):
    """Find lines thinner than a stitch (ink outlines, lashes, a thread) and route them from hole to hole.

    Line pixels (black top-hat > thresh) vote for their nearest hole; holes with enough votes are joined to their
    8 neighbours by a minimum spanning tree, so each line becomes one path with no loops, ladders or crossings.
    hx, hy: pixel position of every hole column/row. render: the image is a stitch render, so first remove the
    thread texture and seams that repeat in every cell. Returns [(row, col, row2, col2, dmc index)] in hole units."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

    rgb = np.asarray(img.convert("RGB"), float)
    lum = to_lab(rgb)[..., 0]
    p = (hx[-1] - hx[0]) / (len(hx) - 1)
    if render:
        # Position inside the cell, in whole pixels; average (lightness - cell median) there over all cells.
        yy, xx = np.mgrid[0:lum.shape[0], 0:lum.shape[1]]
        cy, cx = np.interp(yy, hy, np.arange(len(hy))), np.interp(xx, hx, np.arange(len(hx)))
        cell = np.floor(cy).astype(int) * len(hx) + np.floor(cx).astype(int)
        med = ndimage.median(lum, cell, np.arange(cell.max() + 1))
        pos = (np.floor((cy % 1) * p).astype(int) * 64 + np.floor((cx % 1) * p).astype(int))
        dev = lum - np.asarray(med)[cell]
        template = ndimage.median(dev, pos, np.arange(pos.max() + 1))  # median: sparse real lines do not count
        lum = lum - np.asarray(template)[pos]
    hat = ndimage.grey_closing(lum, max(3, round(0.6 * p) | 1)) - lum  # thinner than a stitch; wider is stitches
    ys, xs = np.nonzero(hat > thresh)
    H, W = len(hy), len(hx)
    hole = np.interp(ys, hy, np.arange(H)).round().astype(int) * W + np.interp(xs, hx, np.arange(W)).round().astype(int)
    votes = np.bincount(hole, minlength=H * W)
    colour = np.stack([np.bincount(hole, rgb[ys, xs, i], H * W) for i in range(3)], -1) / np.maximum(votes, 1)[:, None]
    on = votes >= max(2, 0.5 * p)

    rows, cols, wts = [], [], []
    ii, jj = np.divmod(np.nonzero(on)[0], W)
    for di, dj, wt in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.42), (1, -1, 1.42)):
        i2, j2 = ii + di, jj + dj
        ok = (i2 < H) & (j2 >= 0) & (j2 < W)
        ok[ok] &= on[i2[ok] * W + j2[ok]]
        rows += list(ii[ok] * W + jj[ok]); cols += list(i2[ok] * W + j2[ok]); wts += [wt] * ok.sum()
    if not rows:
        return []
    tree = minimum_spanning_tree(coo_matrix((wts, (rows, cols)), shape=(H * W, H * W))).tocoo()
    edges = set(zip(tree.row.tolist(), tree.col.tolist()))

    # Prune one-segment spurs that stick out of a junction, then drop fragments shorter than min_len.
    deg = np.bincount(np.array([n for e in edges for n in e]), minlength=H * W)
    edges = {(a, b) for a, b in edges if not (min(deg[a], deg[b]) == 1 and max(deg[a], deg[b]) >= 3)}
    if not edges:
        return []
    ea, eb = np.array([e[0] for e in edges]), np.array([e[1] for e in edges])
    _, comp = connected_components(coo_matrix((np.ones(len(ea)), (ea, eb)), shape=(H * W, H * W)), directed=False)
    size = np.bincount(comp[ea], minlength=comp.max() + 1)
    keep = size[comp[ea]] >= min_len
    ea, eb = ea[keep], eb[keep]
    if not len(ea):
        return []

    # Snap colours to a few threads: nearest DMC per segment, then fold rare ones into the common ones.
    pal = to_lab(dmc_rgb)
    seg = (colour[ea] * votes[ea, None] + colour[eb] * votes[eb, None]) / (votes[ea] + votes[eb])[:, None]
    k = nearest(to_lab(seg), pal)
    top = [v for v, _ in sorted(zip(*np.unique(k, return_counts=True)), key=lambda x: -x[1])[:max_threads]]
    k = np.array(top)[nearest(pal[k], pal[top])]
    return [(a // W, a % W, b // W, b % W, int(t)) for a, b, t in zip(ea.tolist(), eb.tolist(), k.tolist())]


def dither(lab, pal, strength):
    """Serpentine Floyd-Steinberg in Lab: pass part of each cell's colour error to its neighbours.

    Turns gradient bands into fine mixes of the two nearest colours, like a stitcher's tweeding."""
    h, w = lab.shape[:2]
    err = lab.astype(float).copy()
    out = np.empty((h, w), int)
    for y in range(h):
        xs, step = (range(w), 1) if y % 2 == 0 else (range(w - 1, -1, -1), -1)
        for x in xs:
            k = ((pal - err[y, x]) ** 2).sum(1).argmin()
            out[y, x] = k
            e = (err[y, x] - pal[k]) * strength
            if 0 <= x + step < w:
                err[y, x + step] += e * 7 / 16
            if y + 1 < h:
                if 0 <= x - step < w:
                    err[y + 1, x - step] += e * 3 / 16
                err[y + 1, x] += e * 5 / 16
                if 0 <= x + step < w:
                    err[y + 1, x + step] += e * 1 / 16
    return out


def make_chart(img, width, colors, dmc_lab, fabric=None, min_delta=0, highlights=0, dither_strength=0, lowlights=0, grid=None):
    """fabric: None = stitch everything, 0 = auto tolerance, >0 = tolerance. min_delta: also merge colours closer than this."""
    small = read_grid(img, grid) if width is None else None
    if small is None:
        width = width or 120
        small = resample(img.convert("RGB"), width, highlights, lowlights)
    height, width = small.shape[:2]
    lab = to_lab(small)
    bg = background_mask(lab, fabric or None) if fabric is not None else np.zeros((height, width), bool)
    fabric_rgb = np.median(small[bg], 0) if bg.any() else np.array([255, 255, 255])

    px = lab[~bg]
    idx = nearest(px, dmc_lab)
    used, counts = np.unique(idx, return_counts=True)
    d = np.linalg.norm(dmc_lab[used][:, None] - dmc_lab[used][None], axis=-1)
    np.fill_diagonal(d, np.inf)
    # Merge the colour whose removal adds the least total error, until the budget fits.
    while len(used) > 1:
        if len(used) <= colors and d.min() >= min_delta:
            break
        i = (np.sqrt(counts) * d.min(1)).argmin() if len(used) > colors else d.min(1).argmin()
        counts[d[i].argmin()] += counts[i]
        used, counts = np.delete(used, i), np.delete(counts, i)
        d = np.delete(np.delete(d, i, 0), i, 1)

    chart = np.full((height, width), EMPTY)
    if dither_strength:
        chart[:] = used[dither(lab, dmc_lab[used], dither_strength)]
        chart[bg] = EMPTY
    else:
        chart[~bg] = used[nearest(px, dmc_lab[used])]
    return chart, fabric_rgb


def clean(chart, dmc_lab, max_delta):
    """Lone stitches (no 8-neighbour of the same colour) take the most common neighbour colour that is close to them.

    Distinct accents (eye highlights, a red bead) have no close neighbour and stay."""
    h, w = chart.shape
    p = np.pad(chart, 1, mode="edge")
    nbrs = np.stack([p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
    out = chart.copy()
    for y, x in zip(*np.nonzero((nbrs != chart).all(0) & (chart != EMPTY))):
        vals, n = np.unique(nbrs[:, y, x], return_counts=True)
        for k in vals[np.argsort(-n)]:
            if k != EMPTY and np.linalg.norm(dmc_lab[k] - dmc_lab[chart[y, x]]) < max_delta:
                out[y, x] = k
                break
    return out


def render(chart, dmc_rgb, fabric_rgb, scale=8, bs=()):
    rgb = np.where((chart == EMPTY)[..., None], fabric_rgb, dmc_rgb[chart.clip(0)]).astype(np.uint8)
    im = Image.fromarray(rgb).resize((chart.shape[1] * scale, chart.shape[0] * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(im)
    for i, j, i2, j2, k in bs:
        draw.line((j * scale, i * scale, j2 * scale, i2 * scale), fill=tuple(int(v) for v in dmc_rgb[k]), width=max(1, scale // 4))
    return im


def cuts(chart, axis, max_len, slack=8):
    """Page borders along one axis: as many pages as an even split needs, each border placed where the fewest
    colour changes are nearby, so a page edge does not run through a busy area such as a face or the moon."""
    size = chart.shape[1 - axis]
    n = -(-size // max_len)
    a = chart if axis == 0 else chart.T  # borders fall between columns of a
    change = np.r_[0, (a[:, 1:] != a[:, :-1]).sum(0), 0].astype(float)
    cost = np.convolve(change, np.ones(5), "same")  # busyness within 2 cells of the border
    lo = max(1, size // n - slack)
    best = {0: (0.0, [0])}  # position -> (cost, borders so far), one page at a time
    for t in range(1, n):
        nxt = {}
        for pos, (c0, path) in best.items():
            for q in range(pos + lo, min(pos + max_len, size - lo) + 1):
                cq = c0 + cost[q] + 0.1 * abs(q - t * size / n)  # ties go to the even split
                if q not in nxt or cq < nxt[q][0]:
                    nxt[q] = (cq, path + [q])
        best = nxt
    finals = [(c0, path) for pos, (c0, path) in best.items() if size - pos <= max_len]
    return min(finals)[1] + [size]


def dmc_sort_key(num):
    return (0, int(num), "") if num.isdigit() else (1, 0, num)


def write_pdf(path, title, chart, dmc, dmc_rgb, dmc_lab, fabric_rgb, count, bs=()):
    h, w = chart.shape
    used, n = np.unique(chart[chart != EMPTY], return_counts=True)
    order = np.argsort(-n)
    if FONT:
        pdfmetrics.registerFont(TTFont("Sym", FONT))
    symfont = "Sym" if FONT else "Helvetica-Bold"
    sym = {int(used[i]): SYMS[j] for j, i in enumerate(order)}
    stitches = dict(zip(used.tolist(), n.tolist()))
    per_skein = STITCHES_PER_SKEIN_14CT * count / 14
    threads, names = {}, {}
    for k, cnt in stitches.items():
        parts = dmc[k]["dmc"].split("+")
        for code, name in zip(parts, dmc[k]["name"].split(" + ")):
            threads[code] = threads.get(code, 0) + cnt / len(parts)
            names[code] = name
    bs_cm = {}  # backstitch thread length; front and back of each segment, one strand
    for i, j, i2, j2, k in bs:
        code = dmc[k]["dmc"]
        bs_cm[code] = bs_cm.get(code, 0) + 2 * np.hypot(i2 - i, j2 - j) * 2.54 / count
        names[code] = dmc[k]["name"]
        threads.setdefault(code, 0)
    # ponytail: 8 m skein = 48 m of single strand, ~40 m usable after waste
    skeins = {code: int(np.ceil(v / per_skein + bs_cm.get(code, 0) / 4000)) for code, v in threads.items()}

    W, H = A4
    M = 12 * mm
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle(title)

    def fill(k):
        c.setFillColorRGB(*(dmc_rgb[k] / 255))

    def ink(k):
        c.setFillColorRGB(*((0, 0, 0) if dmc_lab[k][0] > 55 else (1, 1, 1)))

    # Cover: preview with page map + key facts
    c.setFont("Helvetica-Bold", 20)
    c.drawString(M, H - M - 14, title)
    c.setFont("Helvetica", 10)
    total = int(n.sum())
    lines = [
        f"{w} x {h} stitches, {len(used)} symbols from {len(threads)} DMC threads, {total:,} full cross stitches",
        "Finished size: " + ", ".join(f"{ct} ct {w / ct * 2.54:.1f} x {h / ct * 2.54:.1f} cm" for ct in (14, 16, 18)),
        f"About {sum(skeins.values())} skeins in total ({count} ct, 2 strands). "
        "Add 5 cm of fabric margin on every side.",
    ]
    if (chart == EMPTY).any():
        lines.append("Fabric colour #%02X%02X%02X: bare cells are left unstitched." % tuple(int(v) for v in fabric_rgb))
    for i, s in enumerate(lines):
        c.drawString(M, H - M - 34 - i * 14, s)
    box_w, box_h = W - 2 * M, H - 2 * M - 110
    s = min(box_w / w, box_h / h)
    ox, oy = M + (box_w - w * s) / 2, M
    c.drawImage(ImageReader(render(chart, dmc_rgb, fabric_rgb, 4, bs)), ox, oy, w * s, h * s)
    xb, yb = cuts(chart, 0, COLS_PP), cuts(chart, 1, ROWS_PP)
    grid_cols, grid_rows = len(xb) - 1, len(yb) - 1
    pages = [(yb[i], yb[i + 1], xb[j], xb[j + 1], i, j) for i in range(grid_rows) for j in range(grid_cols)]
    c.setStrokeColorRGB(1, 0, 0)
    c.setFont("Helvetica-Bold", 14)
    for pno, (r0, r1, c0, c1, _, _) in enumerate(pages, 1):
        pw, ph = (c1 - c0) * s, (r1 - r0) * s
        x, y = ox + c0 * s, oy + (h - r0) * s - ph
        c.rect(x, y, pw, ph)
        c.setFillColorRGB(1, 0, 0)
        c.drawCentredString(x + pw / 2, y + ph / 2 - 5, str(pno))
    c.showPage()

    # Key (every symbol) and shopping list (every real thread; a blend uses one strand of each part)
    key = sorted(used.tolist(), key=lambda k: dmc_sort_key(dmc[k]["dmc"].split("+")[0]))
    rows = [("head", "Colour key (DMC stranded cotton)")]
    rows += [("key", k) for k in key]
    bs_threads = sorted({k for *_, k in bs}, key=lambda k: dmc_sort_key(dmc[k]["dmc"]))
    if bs_threads:
        rows += [("gap", None), ("head", "Backstitch (1 strand)")] + [("bs", k) for k in bs_threads]
    rows += [("gap", None), ("head", f"Threads to buy ({count} ct)")]
    rows += [("buy", code) for code in sorted(threads, key=dmc_sort_key)]
    row_h = 6 * mm
    y = 0
    for kind, v in rows:
        if y < M + row_h:
            if y:
                c.showPage()
            y = H - M - 6 * mm
        if kind == "head":
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica-Bold", 13)
            c.drawString(M, y, v)
            y -= row_h
            c.setFont("Helvetica-Bold", 8)
            cols = {"C": ("Sym", "DMC", "Strands", "Name", "Stitches"), "B": ("Line", "DMC", "Strands", "Name", "Length"),
                    "T": ("", "DMC", "", "Name", "Skeins")}[v[0]]
            for x, t in zip((M, M + 12 * mm, M + 38 * mm, M + 52 * mm, M + 165 * mm), cols):
                c.drawString(x, y, t)
        elif kind == "key":
            fill(v)
            c.rect(M, y - 1.5 * mm, 8 * mm, 5 * mm, stroke=1, fill=1)
            ink(v)
            c.setFont(symfont, 9)
            c.drawCentredString(M + 4 * mm, y, sym[v])
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica", 8)
            c.drawString(M + 12 * mm, y, dmc[v]["dmc"])
            c.drawString(M + 38 * mm, y, "1 + 1" if "+" in dmc[v]["dmc"] else "2")
            c.drawString(M + 52 * mm, y, dmc[v]["name"][:70])
            c.drawRightString(M + 180 * mm, y, f"{stitches[v]:,}")
        elif kind == "bs":
            c.setStrokeColorRGB(*(dmc_rgb[v] / 255))
            c.setLineWidth(2)
            c.line(M, y + 1 * mm, M + 8 * mm, y + 1 * mm)
            c.setStrokeColorRGB(0, 0, 0)
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica", 8)
            c.drawString(M + 12 * mm, y, dmc[v]["dmc"])
            c.drawString(M + 38 * mm, y, "1")
            c.drawString(M + 52 * mm, y, dmc[v]["name"][:70])
            c.drawRightString(M + 180 * mm, y, f"{bs_cm[dmc[v]['dmc']] / 100:.1f} m")
        elif kind == "buy":
            c.setFont("Helvetica", 8)
            c.drawString(M + 12 * mm, y, v)
            c.drawString(M + 52 * mm, y, names[v])
            c.drawRightString(M + 180 * mm, y, str(skeins[v]))
        y -= row_h
    c.showPage()

    # Chart pages: each shows its own block plus OVERLAP greyed cells from the neighbours
    for pno, (r0, r1, c0, c1, gi, gj) in enumerate(pages, 1):
        v0, v1, u0, u1 = max(r0 - OVERLAP, 0), min(r1 + OVERLAP, h), max(c0 - OVERLAP, 0), min(c1 + OVERLAP, w)
        ox, top = M + 8 * mm, H - M - 14 * mm

        def cx(x):
            return ox + (x - u0) * CELL

        def cy(y):
            return top - (y - v0) * CELL

        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(M, H - M - 11, f"{title}  -  page {pno}/{len(pages)}  -  columns {c0 + 1}-{c1}, rows {r0 + 1}-{r1}")
        c.setFont(symfont, CELL * 0.7)
        for y in range(v0, v1):
            for x in range(u0, u1):
                k = chart[y, x]
                if k == EMPTY:
                    continue
                fill(k)
                c.rect(cx(x), cy(y + 1), CELL, CELL, stroke=0, fill=1)
                ink(k)
                c.drawCentredString(cx(x) + CELL / 2, cy(y + 1) + CELL * 0.25, sym[k])
        # Grey veil over the overlap from neighbouring pages
        c.setFillColorRGB(1, 1, 1)
        c.setFillAlpha(0.65)
        for x0, y0, x1, y1 in ((u0, v0, u1, r0), (u0, r1, u1, v1), (u0, r0, c0, r1), (c1, r0, u1, r1)):
            if x1 > x0 and y1 > y0:
                c.rect(cx(x0), cy(y1), (x1 - x0) * CELL, (y1 - y0) * CELL, stroke=0, fill=1)
        c.setFillAlpha(1)
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 6)
        for x in range(u0, u1 + 1):
            c.setLineWidth(1.0 if x % 10 == 0 else 0.2)
            c.line(cx(x), cy(v0), cx(x), cy(v1))
            if x % 10 == 0:
                c.drawCentredString(cx(x), cy(v0) + 1.5 * mm, str(x))
        for y in range(v0, v1 + 1):
            c.setLineWidth(1.0 if y % 10 == 0 else 0.2)
            c.line(cx(u0), cy(y), cx(u1), cy(y))
            if y % 10 == 0:
                c.drawRightString(cx(u0) - 1 * mm, cy(y) - 2, str(y))
        # Backstitch on top of everything in this window
        c.setLineCap(1)
        c.setLineWidth(1.8)
        for i, j, i2, j2, k in bs:
            if v0 <= min(i, i2) and max(i, i2) <= v1 and u0 <= min(j, j2) and max(j, j2) <= u1:
                c.setStrokeColorRGB(*(dmc_rgb[k] / 255))
                c.line(cx(j), cy(i), cx(j2), cy(i2))
        c.setStrokeColorRGB(0, 0, 0)
        # Page border and neighbour page numbers
        c.setLineWidth(1.5)
        c.rect(cx(c0), cy(r1), (c1 - c0) * CELL, (r1 - r0) * CELL)
        c.setFont("Helvetica-Bold", 9)
        for di, dj, x, y in ((-1, 0, (cx(c0) + cx(c1)) / 2, cy(v0) + 5 * mm), (1, 0, (cx(c0) + cx(c1)) / 2, cy(v1) - 5 * mm),
                             (0, -1, cx(u0) - 7 * mm, (cy(r0) + cy(r1)) / 2), (0, 1, cx(u1) + 4 * mm, (cy(r0) + cy(r1)) / 2)):
            if 0 <= gi + di < grid_rows and 0 <= gj + dj < grid_cols:
                c.drawCentredString(x, y, f"({(gi + di) * grid_cols + gj + dj + 1})")
        # Centre arrows
        c.setFillColorRGB(1, 0, 0)
        if c0 <= w // 2 < c1:
            X = cx(w // 2)
            p = c.beginPath()
            p.moveTo(X, cy(v0)), p.lineTo(X - 1.5 * mm, cy(v0) + 4 * mm), p.lineTo(X + 1.5 * mm, cy(v0) + 4 * mm), p.close()
            c.drawPath(p, stroke=0, fill=1)
        if r0 <= h // 2 < r1:
            Y = cy(h // 2)
            p = c.beginPath()
            p.moveTo(cx(u0), Y), p.lineTo(cx(u0) - 4 * mm, Y + 1.5 * mm), p.lineTo(cx(u0) - 4 * mm, Y - 1.5 * mm), p.close()
            c.drawPath(p, stroke=0, fill=1)
        c.showPage()
    c.save()


def auto_settings(img, width, grid, dmc_lab):
    """Pick fabric / blends / dither / highlights from the image. Thresholds were tuned on bench.py's images."""
    chosen = {}
    small = read_grid(img, grid) if grid else resample(img, width)
    L = to_lab(small)
    border = np.concatenate([L[0], L[-1], L[:, 0], L[:, -1]])
    if np.percentile(np.linalg.norm(border - np.median(border, 0), axis=1), 90) < 4:
        chosen["fabric"] = (0, "plain border")
    if grid:
        return chosen  # a stitch render already uses whole thread colours
    flat = L.reshape(-1, 3)
    if np.linalg.norm(dmc_lab[nearest(flat, dmc_lab)] - flat, axis=1).mean() > 5:
        chosen["blends"] = (20, "colours fall between DMC threads")
    step = lambda a: np.concatenate([np.linalg.norm(a[:, 1:] - a[:, :-1], axis=-1).ravel(),
                                     np.linalg.norm(a[1:] - a[:-1], axis=-1).ravel()])  # both directions
    smooth = step(L) < 2
    if smooth.mean() >= 0.8:
        trial, _ = make_chart(img, width, 50, dmc_lab)
        jump = step(dmc_lab[trial]) > 3
        if (jump & smooth).sum() / smooth.sum() >= 0.02:
            chosen["dither"] = (0.3, "smooth gradients would band")
    if (L[..., 0] < 30).mean() >= 0.4:
        lum = to_lab(np.asarray(img, float))[..., 0]
        cell = img.width / width
        base = ndimage.grey_opening(lum, max(3, round(1.5 * cell) | 1))
        lbl, _ = ndimage.label((lum - base > 25) & (base < 30))
        sizes = np.bincount(lbl.ravel())[1:]
        if ((sizes > 0) & (sizes <= (1.5 * cell) ** 2)).sum() >= 20:
            chosen["highlights"] = (25, "bright specks on a dark background")
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--width", type=int, help="pattern width in stitches (default: read the stitch grid if the image is a chart render, else 120)")
    ap.add_argument("--colors", type=int, default=min(100, len(SYMS)), help=f"max DMC colours or symbols (up to {len(SYMS)})")
    ap.add_argument("--fabric", nargs="?", type=float, const=0, default=None, metavar="TOL",
                    help="leave the border-connected background unstitched (colour tolerance, default: auto)")
    ap.add_argument("--clean", type=float, help="remove lone stitches closer than this colour distance (default 15, 0 with --dither)")
    ap.add_argument("--dither", nargs="?", type=float, const=0.5, metavar="STRENGTH",
                    help="mix neighbouring colours in gradients instead of bands (0..1, default 0.5); good for skies")
    ap.add_argument("--highlights", nargs="?", type=float, const=25, metavar="L",
                    help="keep small bright spots such as stars as single light stitches (lightness jump, default 25)")
    ap.add_argument("--lowlights", nargs="?", type=float, const=25, default=0, metavar="L",
                    help="keep thin dark details such as hair strands as stitches (darkness jump, default 25)")
    ap.add_argument("--backstitch", nargs="?", type=float, const=-1, default=0, metavar="L",
                    help="trace thin dark lines (outlines, lashes, threads) as backstitch (darkness jump, default 30; 40 on renders)")
    ap.add_argument("--count", type=int, default=14, help="Aida count for the skein estimate")
    ap.add_argument("--blends", nargs="?", type=float, const=15, default=None, metavar="MAXDE",
                    help="also allow blends of two close DMC colours (colour distance up to MAXDE, default 15); good for photos")
    ap.add_argument("--no-auto", action="store_true", help="do not pick fabric/blends/dither/highlights from the image")
    ap.add_argument("--palette", help="only use these DMC numbers, comma separated (e.g. the threads you own)")
    ap.add_argument("-o", "--out", help="output name without extension (default: image name)")
    a = ap.parse_args()

    dmc, dmc_rgb = load_dmc()
    if a.palette:
        want = {s.strip().upper() for s in a.palette.split(",") if s.strip()}
        keep = [i for i, r in enumerate(dmc) if r["dmc"].upper() in want]
        if unknown := want - {dmc[i]["dmc"].upper() for i in keep}:
            ap.error(f"unknown DMC numbers: {', '.join(sorted(unknown))}")
        dmc, dmc_rgb = [dmc[i] for i in keep], dmc_rgb[keep]
    singles = len(dmc)
    img = Image.open(a.image).convert("RGB")
    grid = find_grid(img) if a.width is None else None
    width = a.width or 120
    if not a.no_auto:
        for key, (value, why) in auto_settings(img, width, grid, to_lab(dmc_rgb)).items():
            if getattr(a, key) is None:
                setattr(a, key, value)
                print(f"auto: --{key} {value}  ({why})")
    a.dither, a.highlights = a.dither or 0, a.highlights or 0
    if not 1 <= a.colors <= len(SYMS):
        ap.error(f"--colors must be 1..{len(SYMS)}")
    if a.blends:
        dmc, dmc_rgb = add_blends(dmc, dmc_rgb, a.blends)
    dmc_lab = to_lab(dmc_rgb)
    chart, fabric_rgb = make_chart(img, a.width, a.colors, dmc_lab, a.fabric, highlights=a.highlights,
                                   dither_strength=a.dither, lowlights=a.lowlights, grid=grid)
    a.clean = (0 if a.dither else 15) if a.clean is None else a.clean  # cleaning would undo the dither
    if a.clean:
        chart = clean(chart, dmc_lab, a.clean)
    bs = []
    if a.backstitch == -1:
        a.backstitch = 40 if grid else 30  # render textures need a higher bar
    if a.backstitch:
        if grid:
            (xs, ys), p = grid, grid[0][1] - grid[0][0]
            hx, hy = np.r_[xs, xs[-1] + p], np.r_[ys, ys[-1] + p]
        else:
            hx, hy = np.linspace(0, img.width, chart.shape[1] + 1), np.linspace(0, img.height, chart.shape[0] + 1)
        bs = backstitch(img, hx, hy, a.backstitch, dmc_rgb[:singles], render=bool(grid))
    ys, xs = np.nonzero(chart != EMPTY)
    y0, x0 = ys.min(), xs.min()
    chart = chart[y0:ys.max() + 1, x0:xs.max() + 1]  # the fabric margin is advice, not chart
    h, w = chart.shape
    bs = [(i - y0, j - x0, i2 - y0, j2 - x0, k) for i, j, i2, j2, k in bs
          if all(0 <= r - y0 <= h for r in (i, i2)) and all(0 <= q - x0 <= w for q in (j, j2))]
    out = Path(a.out or Path(a.image).with_suffix(""))
    render(chart, dmc_rgb, fabric_rgb, bs=bs).save(f"{out}_preview.png")
    write_pdf(f"{out}.pdf", out.name, chart, dmc, dmc_rgb, dmc_lab, fabric_rgb, a.count, bs)
    print(f"{out}.pdf  {out}_preview.png  ({w}x{h}, {len(np.unique(chart[chart != EMPTY]))} colours, {len(bs)} backstitch segments)")


if __name__ == "__main__":
    main()
