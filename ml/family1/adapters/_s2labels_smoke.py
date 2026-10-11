"""The synthetic archive of family 1.3's `s2labels` (E-092).

Both Zenodo records (licence, the class table in each description's own
format, the zip's size and md5), both zips in their REAL layouts —

  MARIDA.zip  patches/S2_<d-m-yy>_<tile>/S2_<d-m-yy>_<tile>_<k>.tif (11
              bands, UTM GeoTIFF) + _cl.tif + _conf.tif (float32 masks, as
              the real ones), splits/{train,val,test}_X.txt (CRLF),
              labels_mapping.txt
  MADOS.zip   MADOS/Scene_<n>/{10,20,60}/Scene_<n>_L2R_rhorc_<nm>_<k>.tif at
              native resolution with the satellite's own ACOLITE band names,
              10/ and 20/ _cl/_conf/_rep (uint8), NO georeferencing (identity
              transform, placeholder EPSG:4326, as the real files),
              MADOS/splits/{train,val,test}_X.txt

— the producers' two `assets.py` and the Copernicus catalogue's answers, all
mirrored by URL exactly as the adapter fetches them.

Hostile on purpose; every case is a named expectation in the test:
three MARIDA scenes (Sentinel-2A with one catalogue product, Sentinel-2B with
two reprocessed copies of one, and one the catalogue does not answer -> 12:00
UTC); a MADOS crop that straddles two MARIDA patches (its votes split between
them at ONE origin); pixels both sets label, one class unchanged, one changed
(Marine Water -> Oil Spill), one renamed (Sparse Sargassum -> Sparse Floating
Algae) and a confidence change; MADOS-only pixels inside a MARIDA train patch
and inside a MARIDA TEST patch from a MADOS train crop (the split leak); two
MADOS crops labelling the same ground pixels (duplicates, two with a
different class); MARIDA pixels a covering MADOS crop leaves unlabelled; an
unlocatable crop; two out-of-bounds reflectances. MADOS's values are an
increasing map of MARIDA's (0.9 x + 0.003), never equal to them — the real
sets' relation. The truth is computed HERE, in each patch's own grid
coordinates, never by calling the adapter.
"""
import datetime as dt
import hashlib
import io
import json
import os
import zipfile

import numpy as np

import build_family10_stores as f10b
from family1.adapters import s2labels as s2

NAN = float("nan")
PH = PW = 32                     # MARIDA patch
CH = CW = 24                     # MADOS crop at 10 m (/2 at 20 m, /6 at 60 m)
A = lambda x: np.float32(0.9) * np.asarray(x, np.float32) + np.float32(0.003)  # noqa: E731

SCENES = {
    # folder: (day, tile, epsg, satellite, catalogue products, time of day)
    "S2_20-4-16_30VWH": (dt.date(2016, 4, 20), "30VWH", 32630, "S2A", 0,
                         None),
    "S2_5-3-18_16PCC": (dt.date(2018, 3, 5), "16PCC", 32616, "S2A", 1,
                        (16, 8, 19)),
    "S2_7-9-19_48MYU": (dt.date(2019, 9, 7), "48MYU", 32748, "S2B", 2,
                        (3, 1, 1)),
}
# patch: (scene, k, origin x, origin y, split)
PATCHES = [
    ("S2_20-4-16_30VWH", 0, 500040.0, 6300000.0, "train"),
    ("S2_5-3-18_16PCC", 0, 400000.0, 1760000.0, "train"),
    ("S2_5-3-18_16PCC", 1, 400320.0, 1760000.0, "test"),    # right of 0
    ("S2_5-3-18_16PCC", 2, 400000.0, 1759680.0, "val"),     # below 0
    ("S2_7-9-19_48MYU", 0, 706740.0, 9340960.0, "test"),
]
# MARIDA labels: patch -> {(r, c): (class, conf)}
MARIDA_LABELS = {
    ("S2_20-4-16_30VWH", 0): {(4, 4): (6, 1), (5, 5): (13, 1)},
    ("S2_5-3-18_16PCC", 0): dict(
        [((r, c), (7, 1)) for r in range(2, 10) for c in range(10, 18)] +
        [((12, 12), (3, 2)), ((13, 13), (1, 3)), ((20, 20), (3, 1)),
         ((25, 25), (9, 1))]),
    ("S2_5-3-18_16PCC", 1): dict([((5, c), (1, 1)) for c in range(5, 10)] +
                                 [((20, 20), (7, 1))]),
    ("S2_5-3-18_16PCC", 2): {(3, 3): (8, 1)},
    ("S2_7-9-19_48MYU", 0): dict([((r, c), (10, 1)) for r in (1, 2, 3)
                                  for c in (1, 2, 3)] + [((10, 10), (1, 2))]),
}
# MARIDA out-of-bounds values: (patch, r, c, band index, value)
OOB = [(("S2_5-3-18_16PCC", 0), 12, 12, 10, -0.5),        # rho_B12
       (("S2_7-9-19_48MYU", 0), 10, 10, 2, 2.0)]          # rho_B03 (MADOS
#                                    keeps it out of bounds: 0.9 * 2 + .003)
# MADOS crops: (scene, num, k, satellite, (MARIDA patch, dr, dc) or None,
# split, {(r, c): (MADOS class, conf)})
CROPS = [
    ("Scene_0", 0, 1, "S2A", (("S2_5-3-18_16PCC", 0), 4, 14), "train", dict(
        [((r, c), (7, 1)) for r in range(0, 6) for c in range(0, 3)] +  # same
        [((r, 3), (6, 2)) for r in range(0, 6)] +        # water -> oil spill
        [((16, 6), (3, 1))] +                            # sparse renamed
        [((1, 23), (1, 1))] +                            # A1 debris, same
        [((22, c), (10, 1)) for c in range(4)] +         # only, in A0 train
        [((10, c), (5, 1)) for c in range(20, 24)])),    # only, in A1 TEST
    ("Scene_0", 0, 2, "S2A", (("S2_5-3-18_16PCC", 0), 4, 12), "val", dict(
        [((22, 2), (10, 1)), ((22, 3), (10, 1)),         # duplicates, same
         ((22, 4), (11, 1)), ((22, 5), (11, 1))])),      # duplicates, differ
    ("Scene_1", 1, 1, "S2B", None, "train",
     dict(((r, 0), (6, 1)) for r in range(10))),         # cannot be located
    ("Scene_2", 2, 1, "S2B", (("S2_7-9-19_48MYU", 0), 0, 2), "test", dict(
        [((r, c), (10, 1)) for r in (1, 2, 3) for c in (0, 1)] +
        [((10, 8), (1, 2))])),
]


def _mirror_path(root, url=None, name=None):
    p = os.path.join(root, "s2labels", name or s2._mirror(url))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _w(root, data, url=None, name=None):
    with open(_mirror_path(root, url, name), "wb") as fh:
        fh.write(data if isinstance(data, bytes) else data.encode("utf-8"))


def _tif(arr, transform=None, crs=None, dtype=None):
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    kw = dict(driver="GTiff", count=arr.shape[0], height=arr.shape[1],
              width=arr.shape[2], dtype=dtype or arr.dtype.name,
              crs=crs or "EPSG:4326",
              transform=transform if transform is not None else Affine.identity())
    with MemoryFile() as m:
        with m.open(**kw) as d:
            d.write(arr.astype(kw["dtype"]))
        return m.read()


def _zip(members):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        dirs = set()
        for n in members:
            parts = n.split("/")[:-1]
            for i in range(1, len(parts) + 1):
                dirs.add("/".join(parts[:i]) + "/")
        for d in sorted(dirs):
            z.writestr(d, b"")
        for n, data in members.items():
            z.writestr(n, data)
    return b.getvalue()


def _record(ds, key, data):
    if ds == "MARIDA":
        table = " ".join(f"<p>{k}: {v}</p>" for k, v in
                         s2.MARIDA_CLASSES.items())
        desc = ("<p>MARIne Debris Archive (MARIDA) ...</p><p>The mapping "
                "between Digital Numbers and Classes is:</p>" + table +
                "<p>The mapping between Digital Numbers and Confidence level "
                "is:</p><p>1: High 2: Moderate 3: Low</p>")
    else:
        table = " ".join(f"<p>{k}.&nbsp; {v}</p>" for k, v in
                         [(0, "Non-annotated")] +
                         list(s2.MADOS_CLASSES.items()))
        desc = ("<p>Marine Debris and Oil Spill (MADOS) ...</p><p>The mapping "
                "between Digital Numbers and Classes in _cl files is:</p>" +
                table + "<p>The mapping between Digital Numbers and "
                "Confidence level in _conf files is: 1: High</p>")
    rec, _k = s2.RECORDS[ds]
    return {"id": int(rec), "metadata": {
        "title": ds, "license": {"id": "cc-by-4.0"}, "description": desc},
        "files": [{"key": key, "size": len(data),
                   "checksum": "md5:" + hashlib.md5(data).hexdigest(),
                   "links": {"self": f"https://zenodo.org/api/records/{rec}/"
                                     f"files/{key}/content"}}]}


def _assets(ds):
    var = s2.ASSETS[ds][1]
    s2m = {nm: i for i, nm in enumerate(s2.ASSET_NM)}
    s2m.update({"Class": 11, "Confidence": 12})
    cats = {v: k for k, v in s2.SOURCE_CLASSES[ds].items()}
    return (f"import numpy as np\n\n{var} = {cats!r}\n\nlabels = "
            f"{list(cats)!r}\n\ns2_mapping = {s2m!r}\n\nconf_mapping = "
            f"{{'High': 1, 'Moderate': 2, 'Low': 3}}\n")


def _catalogue(folder):
    day, tile, _e, sat, n, tod = SCENES[folder]
    vals = []
    for i in range(n):
        st = dt.datetime.combine(day, dt.time(*tod))
        name = (f"{sat}_MSIL1C_{st:%Y%m%dT%H%M%S}_N0500_R140_T{tile}_"
                f"2023073{i}T125342.SAFE")
        vals.append({"Name": name, "ContentDate": {
            "Start": f"{st:%Y-%m-%dT%H:%M:%S}.024000Z",
            "End": f"{st:%Y-%m-%dT%H:%M:%S}.024000Z"}})
    return {"@odata.context": "$metadata#Products", "value": vals}


def _t_of(folder):
    day, _t, _e, _s, n, tod = SCENES[folder]
    base = (day - dt.date(1982, 1, 1)).days * 86400
    if n == 0:
        return base + 43200, s2.QC_NO_TIME
    return base + tod[0] * 3600 + tod[1] * 60 + tod[2], 0


def write_all(root):
    from rasterio.transform import Affine
    from rasterio.warp import transform as warp
    rng = np.random.default_rng(20261011)
    # ---------------------------------------------------------- MARIDA ---
    bands, meta, members = {}, {}, {}
    for folder, k, x0, y0, split in PATCHES:
        _d, tile, epsg, *_ = SCENES[folder]
        a = rng.uniform(0.02, 0.2, (11, PH, PW)).astype(np.float32)
        for (pk, r, c, b, val) in OOB:
            if pk == (folder, k):
                a[b, r, c] = val
        bands[(folder, k)] = a
        T = Affine(10.0, 0.0, x0, 0.0, -10.0, y0)
        meta[(folder, k)] = (T, f"EPSG:{epsg}", split)
        cl = np.zeros((PH, PW), np.float32)
        cf = np.zeros((PH, PW), np.float32)
        for (r, c), (q, f) in MARIDA_LABELS[(folder, k)].items():
            cl[r, c], cf[r, c] = q, f
        base = f"patches/{folder}/{folder}_{k}"
        members[base + ".tif"] = _tif(a, T, f"EPSG:{epsg}")
        members[base + "_cl.tif"] = _tif(cl, T, f"EPSG:{epsg}")
        members[base + "_conf.tif"] = _tif(cf, T, f"EPSG:{epsg}")
    for s in ("train", "val", "test"):
        members[f"splits/{s}_X.txt"] = "\r\n".join(
            f"{f[3:]}_{k}" for f, k, _x, _y, sp in PATCHES if sp == s) + "\r\n"
    members["labels_mapping.txt"] = json.dumps(
        {f"{f}_{k}.tif": [0] * 15 for f, k, *_ in PATCHES})
    mz = _zip(members)
    rec = _record("MARIDA", "MARIDA.zip", mz)
    _w(root, json.dumps(rec), url=s2.ZENODO_RECORD.format(rec="5151941"))
    _w(root, mz, name=s2.zip_name("MARIDA"))
    _w(root, _assets("MARIDA"), url=s2.ASSETS["MARIDA"][0])
    for folder, (day, tile, *_r) in SCENES.items():
        _w(root, json.dumps(_catalogue(folder)), url=s2.cdse_url(tile, day))

    # ----------------------------------------------------------- MADOS ---
    def marida_at(folder, gx, gy):
        """The MARIDA patch of `folder` holding grid cell (gx, gy) of the
        scene's 10 m grid (metres of the cell's upper-left corner)."""
        for (f, k), (T, _crs, _s) in meta.items():
            if f != folder:
                continue
            c = int(round((gx - T.c) / 10.0))
            r = int(round((T.f - gy) / 10.0))
            if 0 <= r < PH and 0 <= c < PW:
                return (f, k), r, c
        return None

    members = {}
    crop_vals = {}
    for sc, num, k, sat, where, split, labels in CROPS:
        v10 = rng.uniform(0.02, 0.2, (11, CH, CW)).astype(np.float32)
        if where is not None:
            (pf, pk), dr, dc = where
            T = meta[(pf, pk)][0]
            for r in range(CH):
                for c in range(CW):
                    hit = marida_at(pf, T.c + (c + dc) * 10.0,
                                    T.f - (r + dr) * 10.0)
                    if hit is not None:
                        (f2, k2), r2, c2 = hit
                        v10[:, r, c] = A(bands[(f2, k2)][:, r2, c2])
        crop_vals[(sc, k)] = v10
        cl = np.zeros((CH, CW), np.uint8)
        cf = np.zeros((CH, CW), np.uint8)
        for (r, c), (q, f) in labels.items():
            cl[r, c], cf[r, c] = q, f
        d10 = f"MADOS/{sc}/10/{sc}_L2R_"
        d20 = f"MADOS/{sc}/20/{sc}_L2R_"
        for nm, res, b in zip(s2.ACOLITE_NM[sat], s2.RES_M, range(11)):
            f = res // 10
            native = v10[b][::f, ::f]          # the producer's nearest stack
            members[f"MADOS/{sc}/{res}/{sc}_L2R_rhorc_{nm}_{k}.tif"] = \
                _tif(native.astype(np.float32))
        for m_, arr in (("cl", cl), ("conf", cf),
                        ("rep", np.where(cl > 0, 3, 0).astype(np.uint8))):
            members[f"{d10}{m_}_{k}.tif"] = _tif(arr)
            members[f"{d20}{m_}_{k}.tif"] = _tif(arr[::2, ::2])
    for s in ("train", "val", "test"):
        members[f"MADOS/splits/{s}_X.txt"] = "\n".join(
            f"{sc}_{k}" for sc, _n, k, _s, _w_, sp, _l in CROPS
            if sp == s) + "\n"
    dz = _zip(members)
    rec = _record("MADOS", "MADOS.zip", dz)
    _w(root, json.dumps(rec), url=s2.ZENODO_RECORD.format(rec="10664073"))
    _w(root, dz, name=s2.zip_name("MADOS"))
    _w(root, _assets("MADOS"), url=s2.ASSETS["MADOS"][0])

    # ----------------------------------------------------------- truth ---
    def stored(v):
        v = np.asarray(v, np.float64).copy()
        b = v[:11]
        b[(b < s2.RHO_LO) | (b > s2.RHO_HI)] = np.nan
        v[:11] = b
        return v

    def latlon(epsg, x, y):
        lon, lat = warp(f"EPSG:{epsg}", "EPSG:4326", [x], [y])
        return lat[0], lon[0]

    order = sorted(PATCHES, key=lambda p: (SCENES[p[0]][0], SCENES[p[0]][1],
                                           p[1]))
    rows, by_ground = [], {}
    for folder, k, x0, y0, split in order:
        day, tile, epsg, sat, n, _tod = SCENES[folder]
        t, q0 = _t_of(folder)
        for (r, c) in sorted(MARIDA_LABELS[(folder, k)]):
            q, f = MARIDA_LABELS[(folder, k)][(r, c)]
            v = list(bands[(folder, k)][:, r, c]) + [
                f, s2.SPLIT_CODE[split],
                s2.SAT_CODE[sat] if n else NAN]
            x, y = x0 + (c + 0.5) * 10, y0 - (r + 0.5) * 10
            row = {"t": t, "x": x, "y": y, "epsg": epsg, "v": stored(v),
                   "qc": q | q0, "plat": f"MARIDA|{folder}",
                   "ds": "MARIDA", "split_name": f"MARIDA {split}"}
            by_ground[(day, tile, round(x, 1), round(y, 1))] = row
            rows.append(row)
    expect = {"overlap": {}, "leak": {}, "dup": 0, "dup_diff": 0,
              "kept_inside": {}, "unlocated": {}, "mados_only": 0,
              "conf_changed": 0, "located": 0, "mados_labelled": 0}
    seen, only_rows, footprints = {}, [], []
    for sc, num, k, sat, where, split, labels in CROPS:
        expect["mados_labelled"] += len(labels)
        if where is None:
            for q, _f in labels.values():
                n_ = s2.MADOS_CLASSES[q]
                expect["unlocated"][n_] = expect["unlocated"].get(n_, 0) + 1
            continue
        expect["located"] += 1
        (pf, pk), dr, dc = where
        T = meta[(pf, pk)][0]
        day, tile, epsg, psat, n, _tod = SCENES[pf]
        t, q0 = _t_of(pf)
        ox, oy = T.c + dc * 10.0, T.f - dr * 10.0
        footprints.append((day, tile, ox, ox + CW * 10, oy - CH * 10, oy))
        for (r, c) in sorted(labels):
            q, f = labels[(r, c)]
            jc = s2.TO_JOINT["MADOS"][q]
            x, y = ox + (c + 0.5) * 10, oy - (r + 0.5) * 10
            g = (day, tile, round(x, 1), round(y, 1))
            if g in seen:
                expect["dup"] += 1
                expect["dup_diff"] += int(seen[g] != jc)
                continue
            seen[g] = jc
            v10 = crop_vals[(sc, k)]
            vb = [v10[b][(r // (res // 10)) * (res // 10),
                         (c // (res // 10)) * (res // 10)]
                  for b, res in enumerate(s2.RES_M)]
            if g in by_ground:
                old = by_ground[g]
                key = (f"{s2.JOINT_CLASSES[old['qc'] & 31]} -> "
                       f"{s2.MADOS_CLASSES[q]}")
                expect["overlap"][key] = expect["overlap"].get(key, 0) + 1
                expect["conf_changed"] += int(old["v"][11] != f)
                old["v"] = stored(vb + [f, old["v"][12],
                                        s2.SAT_CODE[psat] if n else NAN])
                old["qc"] = jc | q0 | s2.QC_BOTH | s2.QC_MADOS
                old["ds"] = "both"
                continue
            hit = marida_at(pf, x - 5, y + 5)
            if hit is not None:
                ps = meta[hit[0]][2]
                if ps != split:
                    kk = f"inside a MARIDA {ps} patch, MADOS {split}"
                    expect["leak"][kk] = expect["leak"].get(kk, 0) + 1
            only_rows.append({
                "t": t, "x": x, "y": y, "epsg": epsg,
                "v": stored(vb + [f, s2.SPLIT_CODE[split],
                                  s2.SAT_CODE[psat] if n else NAN]),
                "qc": jc | q0 | s2.QC_MADOS, "plat": f"MADOS|{sc}",
                "ds": "MADOS", "split_name": f"MADOS {split}"})
    expect["mados_only"] = len(only_rows)
    for g, row in by_ground.items():
        if row["ds"] != "MARIDA":
            continue
        day, tile, x, y = g
        for d2, t2, x0, x1, y0, y1 in footprints:
            if (d2, t2) == (day, tile) and x0 <= x < x1 and y0 <= y < y1:
                n_ = s2.JOINT_CLASSES[row["qc"] & 31]
                expect["kept_inside"][n_] = expect["kept_inside"].get(n_, 0) + 1
                break
    rows = rows + only_rows
    # the store's tie order: MARIDA rows of a scene, then its MADOS-only rows
    rows.sort(key=lambda r: r["t"])
    truth = []
    for r in rows:
        lat, lon = latlon(r["epsg"], r["x"], r["y"])
        truth.append({"t": int(r["t"]), "lat": lat, "lon": lon,
                      "platform": f10b.platform_hash(r["plat"]),
                      "v": [float(x) for x in r["v"]], "qc": int(r["qc"]),
                      "ds": r["ds"]})
    return truth, expect


EXPECT = {}


def make_smoke_sources(root, d_lo, d_hi):
    truth, expect = write_all(root)
    EXPECT.clear()
    EXPECT.update(expect)
    lo = f10b.seconds_since_epoch(d_lo)
    hi = f10b.seconds_since_epoch(d_hi) + 86399
    return [r for r in truth if lo <= r["t"] <= hi]
