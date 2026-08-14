#!/usr/bin/env python3
"""Draw every stencil shape E-022/E-023/E-026 has tried, as ASCII.

Chris, 2026-08-14: *"Please draw all your designs in the experiment log."*

**The drawings are generated from `build_stencil` itself, never typed.** A
hand-drawn diagram of a geometry is a second definition of that geometry, and
the second definition is the one that silently goes stale — the same argument
that made `rollout_spatial.py --export-mask` write the globe's AMOC mask
rather than `app.js` tracing a corridor by hand. So this script lays a
synthetic all-ocean grid, calls the real `build_stencil` with the real
latitude row, and recovers each neighbour's (dy, dx) from the indices the
model would actually gather. It therefore shows the integer rounding onto the
0.25 deg grid, the 1/cos(phi) zonal stretch, AND the half-sector rotation of
every second ring — three things a freehand circle would get wrong.

    python3 ml/draw_stencils.py            # all designs, to stdout
    python3 ml/draw_stencils.py --md       # fenced for EXPERIMENTS.md

Latitude matters and is stated on every drawing: at 40 N one cell is 27.8 km
north-south but only 21.3 km east-west, so a ring that is a circle on the
ground is an ellipse in cells. The drawings are in KILOMETRES (physical
truth); the dots sit where the ROUNDED cells put them.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal import KM_PER_DEG, build_stencil          # noqa: E402

LAT0 = 40.0            # Gulf Stream latitude — the corridor's own neighbourhood
DLAT = 0.25
CHAR_ASPECT = 0.5      # a monospace cell is about half as wide as it is tall

# (title, slots, ring_km, runs, note) — ring_km is exactly the CLI string, so
# a drawing and a dispatch cannot disagree about what shape was meant.
DESIGNS = [
    ("3x3 touching (E-022)", 9, 0.0, "#219-#221",
     "the first shape tried: eight cells that TOUCH. LOST by 6.3 seed sd."),
    ("13-point (E-022)", 13, 0.0, "#222-#224",
     "5x5 with the outer diagonals trimmed. LOST by 8.1 seed sd."),
    ("ring of 8 @ 222 km (E-023)", 9, "222", "e023r222",
     "WON: corridor AUC 0.6043, +4.4 seed sd. The reigning champion."),
    ("ring of 16 @ 222 km", 17, "222", "#234",
     "density at ONE radius. n=1, kept as a control, not an arm."),
    ("two rings, 8+8 @ 222/555 km", 17, "222,555", "#237-#239",
     "outer ring rotated half a sector: 16 bearings, not 8 bearings twice."),
    ("three rings, 8+8+8 @ 222/555/1000 km", 25, "222,555,1000", "#255-#257",
     "the widest shape yet: 24 points, but only 16 distinct bearings."),
    ("three rings, 4+4+4 @ 222/555/1000 km", 13, "222,555,1000", "#275/#259/#260",
     "same reach at half the width. 12 points on 8 bearings, 4 of them twice."),
    ("spiral of 13, 222 -> 1000 km", 14, "spiral:222-1000", "#261-#263",
     "the twin of the row above +1 slot: same reach, 13 bearings not 8."),
    ("spiral of 8, 111 -> 890 km", 9, "spiral:111-890", "#264-#266",
     "the champion's exact width, spent on eight radii instead of one."),
    ("spiral of 24, 111 -> 4444 km", 25, "spiral:111-4444", "#267-#269",
     "24 bearings AND 4444 km: the first shape that can hold a month of"
     " Gulf Stream."),
    ("spiral of 36, 111 -> 4444 km", 37, "spiral:111-4444", "#270-#272",
     "the same reach at 1.5x the density — does the extra angle pay?"),
]


def offsets_of(slots, ring_km):
    """(dy, dx) per slot, read out of the real `build_stencil`.

    A synthetic all-ocean grid big enough that no point falls off it, one
    centre pixel picked in the middle, and the neighbour indices decoded back
    into offsets. Nothing here re-derives the geometry.

    The grid is SIZED FROM THE DESIGN, not fixed. It was 121x161, which is
    ample for a 1000 km ring and silently truncates a 4444 km spiral: six of
    its twenty-four points fell off the synthetic edge, `build_stencil`
    correctly wrote -1, and the table reported an 18-point shape reaching
    1999 km. The drawing would have been of a design nobody dispatched."""
    r_max = max(nominal_radii(ring_km) or [60.0])
    half = int(r_max / (KM_PER_DEG * np.cos(np.radians(LAT0)) * DLAT)) + 4
    H, W = 2 * half + 1, 2 * half + 1
    y0, x0 = H // 2, W // 2
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    ys, xs = ys.ravel(), xs.ravel()
    lats = LAT0 + (np.arange(H) - y0) * DLAT
    nbr = build_stencil(H, W, ys, xs, slots, ring_km=ring_km, lats=lats)
    row = nbr[y0 * W + x0]
    out = []
    for j in row:
        if j < 0:
            out.append(None)
            continue
        out.append((int(ys[j]) - y0, int(xs[j]) - x0))
    return out


def to_km(dy, dx):
    """Grid offset -> kilometres on the ground at LAT0."""
    return (dx * DLAT * KM_PER_DEG * np.cos(np.radians(LAT0)),
            dy * DLAT * KM_PER_DEG)


def nominal_radii(ring_km):
    """The radii the DESIGN names, not the ones rounding lands on. Drawing the
    measured radii of a 3x3 would print `[21, 28, 35] km` — three "rings" that
    are one square — and drawing them for a 222 km ring prints [213, 223, 224],
    which is the grid's error bar dressed up as structure."""
    s = str(ring_km)
    if s.startswith("spiral:"):
        r0, r1 = (float(v) for v in s[len("spiral:"):].split("-"))
        return [r0, r1]                        # ends only; the rest is a ramp
    if s in ("0.0", "0", ""):
        return []                              # fixed table: cells, not circles
    return [float(r) for r in s.split(",")]


def draw(title, slots, ring_km, runs, note, width=63):
    offs = offsets_of(slots, ring_km)
    pts = [to_km(*o) for o in offs[1:] if o is not None]
    span = max(max(abs(x) for x, _ in pts), max(abs(y) for _, y in pts)) * 1.18
    height = int(round(width * CHAR_ASPECT)) | 1          # keep it odd
    cx, cy = width // 2, height // 2
    grid = [[" "] * width for _ in range(height)]

    def put(km_x, km_y, ch, over=False):
        c = cx + int(round(km_x / span * cx))
        r = cy - int(round(km_y / span * cy))
        if 0 <= r < height and 0 <= c < width:
            if over or grid[r][c] == " ":
                grid[r][c] = ch

    nom = nominal_radii(ring_km)
    if nom:
        # a dotted guide circle per named radius, so "eight bearings sampled
        # three times over" reads as such instead of as twenty-four points.
        # For a spiral only the OUTER circle is drawn: the inner one is a
        # 3-character blob around the centre that hides the near points.
        for rad in (nom[-1:] if str(ring_km).startswith("spiral:") else nom):
            for a in np.arange(0, 2 * np.pi, 0.035):
                put(rad * np.sin(a), rad * np.cos(a), ".")
    ALPH = "123456789abcdefghijklmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ"
    for k, o in enumerate(offs[1:]):
        if o is None:
            continue
        x, y = to_km(*o)
        # ring designs label by which RING a point is on; the spiral labels by
        # ORDER, because the order is the whole idea
        if str(ring_km).startswith("spiral:"):
            ch = ALPH[k]
        elif nom:
            ch = str(1 + min(range(len(nom)),
                             key=lambda i: abs(nom[i] - np.hypot(x, y))))
        else:
            ch = "o"
        put(x, y, ch, over=True)
    put(0, 0, "@", over=True)

    (nb, nraw), rose = bearings(pts)
    rad_s = (f"{min(np.hypot(*p) for p in pts):.0f}-"
             f"{max(np.hypot(*p) for p in pts):.0f} km (adjacent cells)"
             if not nom else
             "/".join(f"{r:.0f}" for r in nom) + " km"
             + (" (geometric ramp)" if str(ring_km).startswith("spiral:") else ""))
    head = (f"{title}   [{runs}]\n"
            f"  {slots} slots = centre + {len(pts)} neighbours  ·  {rad_s}  ·  "
            f"{nb} bearings >=10 deg apart"
            + (f" ({nraw} distinct to 1 deg)" if nraw != nb else "") + "\n"
            f"  {note}")
    body = "\n".join("".join(r).rstrip() for r in grid)
    foot = (scale_bar(span, width) + "\n" + rose + "\n"
            f"  @ = the pixel predicted  ·  lat {LAT0:.0f} N, 0.25 deg grid  ·  "
            f"THE NINE VIEWS ARE NOT TO A COMMON SCALE")
    return head + "\n\n" + body + "\n" + foot


def bearings(pts, tol=10.0):
    """How many DIRECTIONS this shape watches, and a 5-degree-per-character
    rose showing which — the quantity Chris's spiral is an argument about.

    TWO counts are reported and neither alone is honest. The 0.25 deg rounding
    perturbs bearings, so read literally off the rounded cells a three-ring
    shape that nominally samples sixteen directions looks like twenty — a
    fiction of the grid. Clustering at 10 deg removes that fiction, and then
    UNDERSTATES a dense spiral: at 36 points the golden angle puts some
    bearings 4.8 deg apart, which merge under a 10 deg rule even though at
    4000 km those two points sit 335 km apart and are in no sense the same
    place. So `n10` is "directions at least 10 deg apart" and `nraw` is
    "distinct to 1 deg"; for rings they agree, and where they diverge the gap
    is itself the fact worth reading."""
    b = sorted(np.degrees(np.arctan2(x, y)) % 360 for x, y in pts)
    keep = []
    for v in b:
        if not keep or min((v - keep[-1]) % 360, (keep[-1] - v) % 360) > tol:
            keep.append(v)
    if len(keep) > 1 and min((keep[0] - keep[-1]) % 360,
                             (keep[-1] - keep[0]) % 360) <= tol:
        keep.pop()
    raw = len({round(v) % 360 for v in b})
    bins = ["."] * 72
    for v in b:
        bins[int(v // 5) % 72] = "|"
    r = "".join(bins)
    return (len(keep), raw), ("  N" + r[:18] + "E" + r[18:36] + "S" + r[36:54]
                       + "W" + r[54:] + "N   <- bearings watched, 5 deg/char")


def gap_ratio(n):
    """max/min gap between the bearings of an n-point golden-angle spiral.

    This is not a curiosity, it is how the point count gets CHOSEN. Three
    lengths of arc exist at any n (the three-distance theorem), and the ratio
    of the longest to the shortest collapses to phi = 1.618 exactly when n is
    a FIBONACCI number and is phi^2 = 2.618 for every other n. So 8 and 13
    points are as evenly spread as a spiral gets, while 12 leaves a gap 2.6x
    its own smallest — a shape with a blind sector, which is the one thing
    the design is meant to avoid. Both dispatched arms use Fibonacci counts."""
    b = sorted((k * 137.50776405003785) % 360 for k in range(n))
    g = [(b[(i + 1) % n] - b[i]) % 360 for i in range(n)]
    return max(g) / min(g)


def scale_bar(span, width):
    """A ruler, because the nine drawings differ in width by a factor of 60.

    Without it the 3x3 and the 222 km ring are the SAME PICTURE — eight points
    around a centre — and the only thing distinguishing the shape that lost by
    6.3 seed sd from the shape that won by 4.4 is a number in a caption."""
    for nice in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
        if nice >= span * 0.55:
            break
    n = max(int(round(nice / span * (width // 2))), 4)
    return "  " + "|" + "-" * (n - 2) + "|" + f" {nice:g} km"


def table():
    """One row per design. `bearings/point` is the efficiency Chris's spiral
    argues for: a shape that spends two points on one direction has bought the
    second one nothing, if direction is what carries the signal."""
    rows = [("shape", "runs", "slots", "pts", "reach km", "bear>=10", "bear~1",
             "b/pt", "gap max/min")]
    for title, slots, ring_km, runs, _ in DESIGNS:
        offs = offsets_of(slots, ring_km)
        pts = [to_km(*o) for o in offs[1:] if o is not None]
        (nb, nraw), _ = bearings(pts)
        rows.append((title, runs, str(slots), str(len(pts)),
                     f"{min(np.hypot(*p) for p in pts):.0f}-"
                     f"{max(np.hypot(*p) for p in pts):.0f}",
                     str(nb), str(nraw), f"{nb / len(pts):.2f}",
                     f"{gap_ratio(len(pts)):.2f}"
                     if str(ring_km).startswith("spiral:") else "-"))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = ["| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(rows[0])) + " |",
           "|" + "|".join("-" * (w[i] + 2) for i in range(len(w))) + "|"]
    for r in rows[1:]:
        out.append("| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |")
    return "\n".join(out)


def svg(cols=4, cell=300, pad=62):
    """The same nine designs as one SVG sheet — for reading on a phone, where
    a 95-column ASCII canvas is a horizontal scrollbar.

    Panels share ONE scale (every design drawn against the widest, 1000 km),
    which the ASCII cannot do: at a common scale the 3x3 collapses to a dot,
    and here that dot is the point — it is why E-022 lost. Each panel keeps
    its own km label so the collapse is legible rather than mysterious."""
    S = max(max(np.hypot(*to_km(*o))
                for o in offsets_of(s, r)[1:] if o) for _, s, r, _, _ in DESIGNS)
    S *= 1.12
    rows = -(-len(DESIGNS) // cols)
    W, H = cols * cell, rows * (cell + pad) + 26
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="ui-monospace,monospace">',
           f'<rect width="{W}" height="{H}" fill="#0d1117"/>']
    for i, (title, slots, ring_km, runs, _) in enumerate(DESIGNS):
        ox, oy = (i % cols) * cell, (i // cols) * (cell + pad) + pad
        cx, cy, R = ox + cell / 2, oy + cell / 2 - 6, cell / 2 - 16

        def place(x, y):
            """km -> px on a SQUARE-ROOT radial axis, shared by every panel.

            A linear common scale stopped working the moment a 4444 km spiral
            joined a 222 km ring on the same sheet: at 9937 km across, the
            champion is 2% of its panel and the four E-022/E-023 shapes are
            indistinguishable dots. sqrt(r) keeps one scale — so the panels
            are still comparable, which is the whole reason for the sheet —
            while spending pixels where the designs actually differ. Bearings
            are untouched (this is a radial transform only), and each panel
            prints its true reach in km underneath, so nothing here has to be
            read off the picture."""
            r = np.hypot(x, y)
            f = 0.0 if r == 0 else (R * np.sqrt(r / S)) / r
            return cx + x * f, cy - y * f
        offs = offsets_of(slots, ring_km)
        pts = [to_km(*o) for o in offs[1:] if o is not None]
        (nb, nraw), _r = bearings(pts)
        out.append(f'<text x="{ox + 10}" y="{oy - 32}" fill="#e6edf3" '
                   f'font-size="13">{title}</text>')
        out.append(f'<text x="{ox + 10}" y="{oy - 16}" fill="#7d8590" '
                   f'font-size="11">{runs} · {slots} slots · {nraw} bearings'
                   f'</text>')
        for rad in nominal_radii(ring_km) or []:
            rr = R * np.sqrt(rad / S)
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rr:.1f}" '
                       f'fill="none" stroke="#30363d" stroke-dasharray="2 4"/>')
        for x, y in pts:
            px, py = place(x, y)
            out.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" '
                       f'x2="{px:.1f}" y2="{py:.1f}" '
                       f'stroke="#1f6feb" stroke-opacity="0.35"/>')
        for x, y in pts:
            px, py = place(x, y)
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" '
                       f'r="3.4" fill="#58a6ff"/>')
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.2" '
                   f'fill="#f85149"/>')
        out.append(f'<text x="{ox + 10}" y="{oy + cell - 14}" fill="#7d8590" '
                   f'font-size="10">reach '
                   f'{min(np.hypot(*p) for p in pts):.0f}–'
                   f'{max(np.hypot(*p) for p in pts):.0f} km</text>')
    out.append(f'<text x="10" y="{H - 26}" fill="#7d8590" font-size="11">'
               f'ONE SCALE FOR ALL {len(DESIGNS)} · panel radius {S:.0f} km on '
               f'a SQUARE-ROOT radial axis (bearings exact; true reach printed '
               f'per panel) · red = the pixel predicted</text>')
    out.append(f'<text x="10" y="{H - 9}" fill="#484f58" font-size="10">'
               f'lat {LAT0:.0f}N, 0.25° grid · generated from build_stencil by '
               f'ml/draw_stencils.py — the shapes cannot drift from the code'
               f'</text>')
    out.append("</svg>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="fence each drawing")
    ap.add_argument("--svg", metavar="PATH", help="write an SVG sheet instead")
    a = ap.parse_args()
    if a.svg:
        with open(a.svg, "w") as f:
            f.write(svg())
        print(f"wrote {a.svg}")
        return
    for d in DESIGNS:
        art = draw(*d)
        if a.md:
            title, _, rest = art.partition("\n")
            print(f"**{title}**\n\n```\n{rest.strip(chr(10))}\n```\n")
        else:
            print(art + "\n")
    print(table())


if __name__ == "__main__":
    main()
