# cross-stitch

Turn an image into a printable DMC cross-stitch pattern: a PDF with a cover, colour key, shopping list and
chart pages, plus a PNG preview of the finished piece.

```
pip install -r requirements.txt
python stitch.py photo.jpg --width 200
```

## What it does

- **Photos and paintings.** Resizes to the stitch grid, matches every cell to DMC stranded cotton in CIELAB,
  and merges the palette down to the colour budget (default 100) while keeping small distinct accents.
- **Stitch renders.** If the image already is a picture of stitches (a pattern preview, a chart photo), it finds
  the grid (sub-pixel pitch and phase) and reads each cell instead of resampling. Needs about 4 px per stitch.
- **Automatic settings**, printed with the reason, unless you give the option yourself or pass `--no-auto`:
  - `--fabric`: a plain border becomes bare fabric, not stitches.
  - `--blends`: blends (one strand each of two close colours) when the colours fall between DMC threads.
  - `--dither`: light error diffusion when smooth gradients would show bands.
  - `--highlights`: stars and sparkles on a dark background stay as single light stitches.
- **Opt-in detail tools**: `--lowlights` keeps thin dark details (hair, ripples) as stitches; `--backstitch`
  traces lines thinner than a stitch and routes them from hole to hole as backstitch.
- **Clean charts**: lone stitches merge into a close neighbour colour, but distinct accents stay.
- **Printable PDF**: A4 pages of 47 x 68 stitches, page borders placed away from busy areas, two greyed rows of
  overlap from each neighbouring page, neighbour page numbers, centre arrows, shape symbols (DejaVu Sans if
  installed), a key with strands per symbol, and skeins per thread.

## Options

```
python stitch.py IMAGE [--width N] [--colors N] [--count CT] [--palette 310,321,...]
                       [--fabric [TOL]] [--blends [MAXDE]] [--dither [S]] [--highlights [L]]
                       [--lowlights [L]] [--backstitch [L]] [--clean DE] [--no-auto] [-o NAME]
```

`--palette` limits the chart to threads you own. `--count` sets the fabric count for the skein estimate.
Run `python stitch.py -h` for details.

## Tests and benchmark

`python test_stitch.py` runs the self-checks.

`bench.py` scores the output against a designer's hand-made chart and on photos (colour error after a
one-stitch blur, lone-stitch share). Its reference charts and images are copyrighted, so they are not in the
repository; the script documents what it expects.

## Limits

- Skein counts are estimates (about 1,800 two-strand full crosses per skein on 14 ct).
- Backstitch detection finds lines darker than their surroundings; metallic thread in renders can give false lines.
- Metallic and specialty threads are not in the colour table.

## Credits

DMC colour table: [sharlagelfand/dmc](https://github.com/sharlagelfand/dmc) (MIT, see [LICENSE-dmc](LICENSE-dmc)). `BLANC` was added by hand.
DMC is a trademark of DMC; this project is not affiliated with DMC.

## License

MIT, see [LICENSE](LICENSE).
