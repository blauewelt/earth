#!/usr/bin/env python3
"""Write `data/family7_index.json` — the small in-repo index the globe reads
before it touches the Hugging Face Hub — plus the two static grids that go
with it.

WHAT FAMILY 7 IS, in one sentence: the first input tensor covering the whole
globe rather than the North Atlantic window — every 0.25° grid point from the
South Pole to the North Pole, one value per channel per five-day bin from 1982
to 2024 — built by `ml/build_family7.py` and published to
`chfrank/earth-tensors` under `tensors/family7_global025_pentad_l0/`
(recipe `f7l0`; see `ml/plans/E070_family7_build.md`).

The globe's "Global tensor (family 7)" layer paints ONE channel of ONE pentad
by a single HTTP range read of the group's `.npy`. To compute that read it
needs four numbers per file that are not in `manifest.json`: the `.npy`
header length, the array shape, the dtype and the byte size of one bin's
slab. This script parses them out of the real headers, verifies every file's
sha256 by downloading it back, measures the CORS headers a browser on our
origin would actually get, and writes all of it into one ~30 KB JSON.

It also writes the two STATICS as ordinary baked grids, because they do not
change with time and a 14.5 MB range read for a constant would be absurd:

    data/family7_sphere.json   classGrid, packed  (0 ocean · 1 land ·
                               2 ice sheet · 3 inland water)
    data/family7_elev.json     regular grid, int16 metres

THE CHECKS THIS WILL NOT SKIP (`ml/CLAUDE.md` §0.2 — a step that reports
success is not evidence it did anything):

  1. **The restore.** Every published file is downloaded back and its sha256
     compared with the manifest's. A manifest is a claim; the bytes are the
     fact. (`--trust-manifest` skips the 100 GB re-download when the build
     job has already restore-verified in the same session; it is recorded in
     the index so a reader knows which kind of index they are holding.)
  2. **CORS**, measured with `Origin: https://blauewelt.github.io` exactly as
     `ml/upload_cone_samples.py` does — CLAUDE.md §3 admits `huggingface.co`
     on measured properties, never assumed ones.
  3. **The header parse is checked against the file size**: header_len +
     prod(shape) * itemsize must equal the manifest's byte count, or the
     range arithmetic the browser is about to do is wrong.

FIXTURE MODE (`--fixture <smoke work dir>`) writes `data/family7/fixture/` —
the same schema over the T=5 tensor `python3 ml/build_family7.py --smoke`
produces, DECIMATED to a 5° grid (every 20th row and column of the 0.25°
grid, every 10th of the 1° grid, which lands exactly on point-aligned 5° and
10° grids that are still south-first and still wrap at the dateline). The
decimation is not cosmetic: one 0.25° bin is 14.5 MB and git is not where
that belongs. Everything the browser computes — the offset arithmetic, the
float16 decode, the un-z-scoring, the cell bounds, the dateline wrap — runs
identically on it, which is what makes it a fixture rather than a mock.

Run (after the build job has published):
    python3 ml/publish_family7_index.py
Fixture (no network):
    python3 ml/build_family7.py --smoke --smoke-dir /tmp/f7fix
    python3 ml/publish_family7_index.py --fixture /tmp/f7fix/work
"""
import argparse
import ast
import datetime as _dt
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

REPO_ID = "chfrank/earth-tensors"
PREFIX = "tensors/family7_global025_pentad_l0"
STEM = "family7_global025_pentad_l0"
GROUPS = ("g025", "g100", "rg100")
ORIGIN = "https://blauewelt.github.io"

INDEX = os.path.join(ROOT, "data", "family7_index.json")
SPHERE = os.path.join(ROOT, "data", "family7_sphere.json")
ELEV = os.path.join(ROOT, "data", "family7_elev.json")
FIXTURE_DIR = os.path.join(ROOT, "data", "family7", "fixture")

# The decimation the fixture uses. 721 rows step 20 → 37 rows at exactly 5°
# (−90 … 90); 1440 cols step 20 → 72 at 5° (−180 … 175); 181/360 step 10 → 19
# and 36 at 10°. All four land on whole degrees, so the fixture's grid is a
# real point-aligned global grid and not a ragged crop.
FIX_STRIDE = {"g025": 20, "g100": 10, "rg100": 10}

# ---------------------------------------------------------------- vocabulary
# Plain-English label + unit for every channel, from E-070 §2. This lives HERE
# rather than in src/app.js for the same reason a categorical grid ships its
# own palette (CLAUDE.md §2.3): the producer's vocabulary travels with the
# producer's bytes, so re-baking with a channel added or renamed cannot leave
# the page describing a channel that is no longer there.
#
# `sign` decides the legend: "div" for a channel whose zero is meaningful and
# whose sign is a direction (the current and wind components, sea-surface
# height, wind stress, the two turbulent fluxes) — those get the app's
# diverging blue = down/cool, red = up/warm ramp; "seq" for everything else.
CHANNELS = {
    # g025 — 0.25°, seven channels
    "cur_speed": ("Surface current speed", "m/s", "seq", "speed"),
    "log_mld":   ("Mixed-layer depth (log₁₀)", "log₁₀ m", "seq", "precip"),
    "ssh":       ("Sea surface height", "m", "div", "anom"),
    "cur_u":     ("Surface current, eastward", "m/s", "div", "anom"),
    "cur_v":     ("Surface current, northward", "m/s", "div", "anom"),
    "sst":       ("Sea-surface temperature (OISST, observed)", "°C", "seq", "sst"),
    "sea_ice":   ("Sea-ice concentration", "0–1", "seq", "precip"),
    # g100 — 1°, fifteen channels
    "tau_x":     ("Wind stress on the surface, eastward", "N/m²", "div", "anom"),
    "tau_y":     ("Wind stress on the surface, northward", "N/m²", "div", "anom"),
    "tau_x_std": ("Wind stress spread within the pentad, eastward", "N/m²", "seq", "precip"),
    "tau_y_std": ("Wind stress spread within the pentad, northward", "N/m²", "seq", "precip"),
    "t2m":       ("Air temperature at 2 m", "°C", "seq", "t2m"),
    "u10":       ("Wind at 10 m, eastward", "m/s", "div", "anom"),
    "v10":       ("Wind at 10 m, northward", "m/s", "div", "anom"),
    "sp":        ("Surface pressure", "hPa", "seq", "precip"),
    "log_prate": ("Precipitation rate (log1p)", "log1p mm/day", "seq", "rain"),
    "log_swe":   ("Snow water equivalent (log1p)", "log1p mm", "seq", "precip"),
    "soilw":     ("Soil moisture, 0–10 cm", "fraction", "seq", "precip"),
    "tsoil":     ("Soil temperature, 0–10 cm", "°C", "seq", "sst"),
    "lhtfl":     ("Latent heat flux (positive = up)", "W/m²", "div", "anom"),
    "shtfl":     ("Sensible heat flux (positive = up)", "W/m²", "div", "anom"),
    # The SHARED surface temperature: one instrument (the reanalysis) over
    # land, sea and ice alike. `sst` above is the OBSERVED field and is
    # missing wherever OISST does not look — E-071 §6.1's correction of
    # 4 Sep, a channel is shared only when the measurand AND the instrument
    # match on both sides.
    "skt":       ("Skin temperature (NCEP reanalysis, every surface)", "°C",
                  "seq", "sst"),
}

SPHERE_CLASSES = [
    {"code": 0, "label": "ocean", "rgb": [30, 72, 130]},
    {"code": 1, "label": "land", "rgb": [122, 106, 74]},
    {"code": 2, "label": "ice sheet or glacier", "rgb": [222, 232, 240]},
    {"code": 3, "label": "inland water", "rgb": [64, 148, 186]},
]


def rg_channel(name):
    """Label/unit for a Roemmich–Gilson depth channel (`rg_t10`, `rg_s1900`)."""
    kind = "Ocean temperature" if name.startswith("rg_t") else "Ocean salinity"
    unit = "°C" if name.startswith("rg_t") else "PSU"
    depth = name.split("_")[1][1:]
    ramp = "sst" if name.startswith("rg_t") else "precip"
    return (f"{kind} at {depth} dbar", unit, "seq", ramp)


def describe(name):
    if name in CHANNELS:
        return CHANNELS[name]
    if name.startswith("rg_"):
        return rg_channel(name)
    # An unknown channel is described as itself rather than guessed at — a
    # wrong unit on a read-out is worse than no unit.
    return (name, "", "seq", "precip")


# ------------------------------------------------------------- .npy headers
def parse_npy_header(head_bytes):
    """(header_len, shape, dtype_str, fortran_order) from the first bytes of a
    `.npy` file. Format 1.0/2.0/3.0: magic, version, then a 2- or 4-byte
    little-endian header length and that many bytes of a Python literal dict.
    Parsed with `ast.literal_eval`, never `eval` — the file is remote."""
    if head_bytes[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file (bad magic)")
    major = head_bytes[6]
    if major == 1:
        n = int.from_bytes(head_bytes[8:10], "little")
        off = 10
    else:
        n = int.from_bytes(head_bytes[8:12], "little")
        off = 12
    if len(head_bytes) < off + n:
        raise ValueError("header probe too short — read more bytes")
    d = ast.literal_eval(head_bytes[off:off + n].decode("latin1"))
    return off + n, list(d["shape"]), d["descr"], bool(d["fortran_order"])


def sha256_file(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(buf), b""):
            h.update(blk)
    return h.hexdigest()


def resolve_url(name):
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{PREFIX}/{name}"


CORS_PROBE_BYTES = 4096


def measure_cors(url, origin=ORIGIN, tries=4):
    """What a BROWSER on our origin would actually get back — an anonymous
    ranged GET carrying `Origin`, following the Hub's redirect to its CDN,
    reporting the headers on whatever finally answers. Anonymous because that
    is what the page is; a measurement made with a token measures something
    else. Identical in shape to `ml/upload_cone_samples.py::measure_cors`."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"Origin": origin,
                              "Range": f"bytes=0-{CORS_PROBE_BYTES - 1}"})
            with urllib.request.urlopen(req, timeout=90) as r:
                h = {k.lower(): v for k, v in r.headers.items()}
                body = r.read()
                return dict(
                    status=r.status, final_url=r.url, bytes=len(body),
                    access_control_allow_origin=h.get("access-control-allow-origin"),
                    access_control_expose_headers=h.get("access-control-expose-headers"),
                    accept_ranges=h.get("accept-ranges"),
                    content_range=h.get("content-range"),
                    head=body,
                )
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            wait = 30 * (attempt + 1)
            print(f"    HTTP 429 from the Hub — waiting {wait}s "
                  f"(attempt {attempt + 2}/{tries})", flush=True)
            time.sleep(wait)


# ----------------------------------------------------------- shared writers
def norm_rows(arr):
    return [[float(m), float(s)] for m, s in arr]


def group_block(name, url, nbytes, sha, header_len, shape, dtype, fortran,
                chans, norm, grid, extra=None):
    if fortran:
        raise SystemExit(f"{name}: Fortran order — the slab arithmetic assumes "
                         f"C order (bin-major), and this file is not")
    itemsize = int(dtype[-1])
    cells = 1
    for d in shape[1:]:
        cells *= int(d)
    slab = cells * itemsize
    want = header_len + int(shape[0]) * slab
    if nbytes and want != nbytes:
        raise SystemExit(f"{name}: header says {want} bytes, the file is "
                         f"{nbytes} — the range arithmetic would be wrong")
    if len(chans) != int(shape[-1]):
        raise SystemExit(f"{name}: {len(chans)} channel names for "
                         f"{shape[-1]} channels")
    labels, units, signs, ramps = {}, {}, {}, {}
    for c in chans:
        lab, unit, sign, ramp = describe(c)
        labels[c], units[c], signs[c], ramps[c] = lab, unit, sign, ramp
    blk = dict(
        file=name, url=url, bytes=int(nbytes), sha256=sha,
        header_len=int(header_len), shape=[int(x) for x in shape],
        dtype=dtype, itemsize=itemsize, fortran_order=False,
        slab_bytes=int(slab), n_bins=int(shape[0]),
        chans=list(chans), norm=norm_rows(norm),
        labels=labels, units=units, sign=signs, ramp=ramps,
        grid=grid,
    )
    if extra:
        blk.update(extra)
    return blk


def grid_block(ny, nx, lat0, lon0, step):
    """The geometry of a point-aligned grid, as the browser needs it: the
    POINTS are at lat0 + step*i, and a cell is the half-step box around its
    point — so the west edge is lon0 − step/2 and the column index wraps at
    the dateline (the half cell either side of 180° is ONE cell)."""
    return dict(ny=int(ny), nx=int(nx), lat0=float(lat0), lon0=float(lon0),
                step=float(step), south_first=True, wrap=True)


def packed_class_grid(sphere, lats, lons, step, source):
    """`data/family7_sphere.json` — the `classGrid` + `packed:` convention
    (CLAUDE.md §3 and §2.3): one character per cell, the file carries its own
    palette, and the values are row-major FROM THE SOUTH, which is the axis
    order the tensor already has (no flip — see the note in E-070 §1)."""
    ny, nx = sphere.shape
    chars = []
    for iy in range(ny):
        row = sphere[iy]
        chars.append("".join("." if v < 0 else chr(48 + int(v)) for v in row))
    return dict(
        _source=source,
        west=float(lons[0]) - step / 2, east=float(lons[-1]) + step / 2,
        south=float(lats[0]) - step / 2, north=float(lats[-1]) + step / 2,
        dlon=step, dlat=step, nx=int(nx), ny=int(ny),
        units="", period=None,
        classes=SPHERE_CLASSES,
        packed="".join(chars),
    )


def elev_grid(elev, lats, lons, step, source):
    """`data/family7_elev.json` — an ordinary regular grid in whole metres
    (int16; ETOPO's decimetres are far below the 0.25°-cell mean's own
    honesty), row-major from the south, `null` where there is no value."""
    ny, nx = elev.shape
    vals = []
    for iy in range(ny):
        for v in elev[iy]:
            vals.append(None if v != v else int(round(float(v))))
    return dict(
        _source=source,
        west=float(lons[0]) - step / 2, east=float(lons[-1]) + step / 2,
        south=float(lats[0]) - step / 2, north=float(lats[-1]) + step / 2,
        dlon=step, dlat=step, nx=int(nx), ny=int(ny),
        units="m", period=None,
        values=vals,
    )


def write_json(path, obj, label):
    text = json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {label} ({len(text) / 1e6:.2f} MB)")


def now_utc():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


RECIPE_NOTE = (
    "python3 ml/build_family7.py --work <dir> --stage all   # the tensor "
    "(ml/plans/E070_family7_build.md); then "
    "python3 ml/publish_family7_index.py                    # this index"
)


# ------------------------------------------------------------- fixture mode
def build_fixture(work):
    import numpy as np
    meta_path = os.path.join(work, f"{STEM}.npz")
    if not os.path.exists(meta_path):
        raise SystemExit(f"no {meta_path} — run "
                         f"`python3 ml/build_family7.py --smoke --smoke-dir <dir>` first")
    meta = np.load(meta_path, allow_pickle=True)
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    groups, statics = {}, {}
    for g in GROUPS:
        src = os.path.join(work, f"{STEM}_X_{g}.npy")
        if not os.path.exists(src):
            continue
        stride = FIX_STRIDE[g]
        a = np.load(src, mmap_mode="r")
        small = np.ascontiguousarray(a[:, ::stride, ::stride, :])
        out = os.path.join(FIXTURE_DIR, f"{STEM}_X_{g}.npy")
        np.save(out, small)
        with open(out, "rb") as fh:
            head = fh.read(256)
        header_len, shape, dtype, fortran = parse_npy_header(head)
        nbytes = os.path.getsize(out)
        lat_key, lon_key = ("lats", "lons") if g == "g025" else ("lat1", "lon1")
        lats = meta[lat_key][::stride]
        lons = meta[lon_key][::stride]
        step = float(lats[1] - lats[0])
        extra = None
        if g == "rg100":
            extra = dict(live_only=True,
                         bin_index=[int(b) for b in meta["rg_bin_index"]])
        groups[g] = group_block(
            f"{STEM}_X_{g}.npy",
            # The URL is the REAL Hub URL even in the fixture, so the browser
            # code path (and the range read it does) is byte-for-byte the one
            # production runs; the tests route that URL to these local files.
            resolve_url(f"{STEM}_X_{g}.npy"),
            nbytes, sha256_file(out), header_len, shape, dtype, fortran,
            [str(c) for c in meta[f"chan_{g}"]], meta[f"norm_{g}"],
            grid_block(len(lats), len(lons), lats[0], lons[0], step),
            extra=extra)

    st = FIX_STRIDE["g025"]
    lats, lons = meta["lats"][::st], meta["lons"][::st]
    step = float(lats[1] - lats[0])
    sph = os.path.join(FIXTURE_DIR, "family7_sphere.json")
    elv = os.path.join(FIXTURE_DIR, "family7_elev.json")
    write_json(sph, packed_class_grid(np.asarray(meta["sphere"])[::st, ::st],
                                      lats, lons, step,
                                      "ml/publish_family7_index.py --fixture"),
               sph)
    write_json(elv, elev_grid(np.asarray(meta["elev"])[::st, ::st],
                              lats, lons, step,
                              "ml/publish_family7_index.py --fixture"), elv)
    statics = static_block("data/family7/fixture/family7_sphere.json",
                           "data/family7/fixture/family7_elev.json")

    bins = [int(b) for b in meta["bin_index"]]
    index = index_block(groups, statics, meta, bins, fixture=True,
                        cors=None, verified=False)
    write_json(os.path.join(FIXTURE_DIR, "family7_index.json"), index,
               os.path.join(FIXTURE_DIR, "family7_index.json"))
    return index


def static_block(sphere_file, elev_file):
    return {
        "sphere": dict(
            file=sphere_file, kind="classGrid",
            label="Surface sphere (ocean / land / ice sheet / inland water)",
            units="", note="a static, not a channel — it never changes with the date"),
        "elev": dict(
            file=elev_file, kind="grid",
            label="Surface elevation (negative under the sea)", units="m",
            ramp="terrain", vmin=-6000, vmax=6000,
            note="a static, not a channel — it never changes with the date"),
    }


def index_block(groups, statics, meta, bins, fixture, cors, verified):
    return dict(
        _source="ml/publish_family7_index.py — do not hand-edit",
        generated_utc=now_utc(),
        fixture=bool(fixture),
        recipe=str(meta["recipe"]),
        stem=STEM,
        window=str(meta["window"]),
        cadence=str(meta["cadence"]),
        repo=f"datasets/{REPO_ID}",
        base=f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{PREFIX}/",
        epoch=str(meta["epoch"]),
        pentad_days=int(meta["pentad_days"]),
        bin_first=bins[0], bin_last=bins[-1], n_bins=len(bins),
        # The ocean groups begin in 1993 (GLORYS12's own start); the shared
        # land/air channels reach back to the epoch. Both are stated because
        # the hover card has to say "Recorded" honestly for a layer whose
        # channels do not all start on the same day.
        recorded=dict(all="1982-01-01", ocean="1993-01-01",
                      last=str(meta["months"][-1]) if "months" in meta.files else None),
        groups=groups,
        statics=statics,
        restore_verified=bool(verified),
        cors_measured=cors,
        recipe_cmd=RECIPE_NOTE,
        plan="https://blauewelt.github.io/earth/docs.html?f=ml/plans/E070_family7_build.md",
        sources=str(meta["sources"]) if "sources" in meta.files else None,
    )


# ---------------------------------------------------------------- Hub mode
def build_from_hub(trust_manifest, out_index, out_sphere, out_elev):
    import numpy as np
    from huggingface_hub import hf_hub_download

    def pull(name):
        return hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                               filename=f"{PREFIX}/{name}")

    man = json.load(open(pull("manifest.json"), encoding="utf-8"))
    by_name = {f["name"]: f for f in man["files"]} if "files" in man else man
    meta = np.load(pull(f"{STEM}.npz"), allow_pickle=True)

    groups = {}
    for g in GROUPS:
        name = f"{STEM}_X_{g}.npy"
        rec = by_name.get(name)
        if rec is None:
            print(f"  {g}: not in the manifest — skipped")
            continue
        url = resolve_url(name)
        # (1) the header, read as a RANGE — the whole point of the index is
        # that this file is never downloaded to be understood.
        probe = measure_cors(url)
        header_len, shape, dtype, fortran = parse_npy_header(probe["head"])
        if probe["status"] != 206 or not probe["content_range"]:
            raise SystemExit(f"{name}: the Hub answered {probe['status']} to a "
                             f"ranged GET — the layer's whole design is one "
                             f"range read per frame")
        if not probe["access_control_allow_origin"]:
            raise SystemExit(f"{name}: no access-control-allow-origin for "
                             f"{ORIGIN} — a browser could not read it")
        # (2) the restore: the bytes, not the claim.
        if trust_manifest:
            print(f"  {g}: sha256 taken from the manifest (--trust-manifest)")
        else:
            local = pull(name)
            got = sha256_file(local)
            if got != rec["sha256"]:
                raise SystemExit(f"{name}: sha256 mismatch on restore "
                                 f"({got} != {rec['sha256']})")
            print(f"  {g}: restored ✓ {rec['bytes'] / 1e9:.1f} GB")
        extra = None
        if g == "rg100" and "rg_bin_index" in meta.files:
            extra = dict(live_only=True,
                         bin_index=[int(b) for b in meta["rg_bin_index"]])
        lat_key, lon_key = ("lats", "lons") if g == "g025" else ("lat1", "lon1")
        lats, lons = meta[lat_key], meta[lon_key]
        groups[g] = group_block(
            name, url, rec["bytes"], rec["sha256"], header_len, shape, dtype,
            fortran, [str(c) for c in meta[f"chan_{g}"]], meta[f"norm_{g}"],
            grid_block(len(lats), len(lons), lats[0], lons[0],
                       float(lats[1] - lats[0])),
            extra=extra)

    write_json(out_sphere, packed_class_grid(
        np.asarray(meta["sphere"]), meta["lats"], meta["lons"], 0.25,
        "ml/publish_family7_index.py (family 7 statics)"), out_sphere)
    write_json(out_elev, elev_grid(
        np.asarray(meta["elev"]), meta["lats"], meta["lons"], 0.25,
        "ml/publish_family7_index.py (family 7 statics)"), out_elev)

    cors = measure_cors(resolve_url(f"{STEM}_X_g100.npy"))
    cors_rec = dict(origin=ORIGIN, status=cors["status"],
                    final_url_host=cors["final_url"].split("/")[2],
                    access_control_allow_origin=cors["access_control_allow_origin"],
                    access_control_expose_headers=cors["access_control_expose_headers"],
                    accept_ranges=cors["accept_ranges"],
                    content_range=cors["content_range"],
                    measured_utc=now_utc())
    bins = [int(b) for b in meta["bin_index"]]
    index = index_block(groups,
                        static_block("data/family7_sphere.json",
                                     "data/family7_elev.json"),
                        meta, bins, fixture=False, cors=cors_rec,
                        verified=not trust_manifest)
    write_json(out_index, index, out_index)
    return index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixture", default="",
                    help="build data/family7/fixture/ from a --smoke work dir "
                         "instead of reading the Hub")
    ap.add_argument("--trust-manifest", action="store_true",
                    help="skip the re-download restore check (record it in the "
                         "index); only when the build job verified in-session")
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--sphere", default=SPHERE)
    ap.add_argument("--elev", default=ELEV)
    a = ap.parse_args(argv)
    if a.fixture:
        build_fixture(a.fixture)
        return 0
    build_from_hub(a.trust_manifest, a.index, a.sphere, a.elev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
