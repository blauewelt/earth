#!/usr/bin/env python3
"""E-093 · PENTAD-OF-YEAR CLIMATOLOGY sums of the five-day global tensor's
gridded groups (family 7.2 / family 10: g025, g100, oc025), CUMULATIVE over the
years, so the Data tab can give "the mean of every five-day slot of the year
over any span of years" from two small range reads per slot — E-091's idea
(ml/export_gridded_doy.py) for five-day bins instead of days.

THE SLOT OF A BIN. Bin b covers the five days starting 1982-01-01 + 5·b days
(UTC). Its slot is

    slot = min(72, floor((day_of_year(bin start) − 1) / 5))       0 … 72

so slot s holds the bins that START on days 5s+1 … 5s+5 of their year, and
slot 72 those starting on days 361 … 365 (366 in a leap year). Each bin is
charged to exactly one slot and one year — those of its start. The bins drift
one day per leap year since 1982, so a year normally contributes one bin to a
slot but can contribute 0 or 2 (a leap year that starts its first bin on 1
January puts bins on days 361 and 366 into slot 72); the counts record that
and nothing is forced.

WHAT THIS WRITES, per group and per slot s (73 slots, `ss` = 00 … 72):

    <out>/sum_p{ss}.npy    [C, Y, H, W] float32 — plane (c, y) is the sum, in
                           PHYSICAL units (z × sd + mean with the group's norm
                           from data/family7_index.json, in float64), of every
                           finite value of channel c in the bins of slot s in
                           the years year_first … year_first + y (float64
                           running total, rounded once per plane)
    <out>/count_p{ss}.npy  [C, Y, H, W] uint8 — how many bins that sum holds
    <out>/manifest_c{kk}.json  per chunk of slots: bytes, sha256, headers, the
                           years, the slot table, the falsifier's numbers

    mean of slot s over years a … b = (S[b] − S[a−1]) / (N[b] − N[a−1])
    leaving a year e out            = subtract (S[e] − S[e−1]) and (N[e] − N[e−1])

    plane (c, y) starts at header_len + (c·Y + y)·H·W·itemsize; rows are SOUTH
    FIRST (row 0 = lat0 = −90), columns from lon0 = −180, as in the tensor.

THE SOURCE is the bin-major tensor file on the Hub, one HTTP range per bin —
the whole bin slab, offset header_len + (b − bin_first)·H·W·C·2 — each answer
checked for 206, its length, and the file's total size; the .npy header is
read and checked against the index first. NaN = missing, excluded.

THE FALSIFIER, in every chunk job before anything is uploaded: for the first
and last slot of the chunk every native slab is kept, and the composed mean
over several spans — the whole record, a 20-year span, single years at both
ends and in the middle, all but the end years, and the whole record with a
year left out — is compared with numpy's mean of those bins. Counts must be
equal exactly; the mean must lie within ½ ulp(S[b]) + ½ ulp(S[a−1]) (+ the
same for the two planes of a left-out year) divided by the count — the bound
E-091 corrected in commit 88015db. A failure uploads nothing.

One job does ONE CHUNK of slots (12 chunks: 6 slots each, the last 7), so
twelve hosted runners build a group in parallel.

    python3 ml/export_gridded_poy.py chunk --group g100 --chunk 0 --out DIR [--upload]
                                     [--index data/family7_index.json] [--base URL-or-dir]
    python3 ml/export_gridded_poy.py index [--groups g025,g100,oc025] [--local DIR]
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PLAN = "ml/plans/E093_pentad_climatology.md"
VERSION = "v1"
REPO_ID = "chfrank/earth-tensors"
STEM = "family7_global025_pentad_l2"
SRC_INDEX = os.path.join(ROOT, "data", "family7_index.json")
INDEX = os.path.join(ROOT, "data", "gridded_poy_index.json")
GROUPS = ("g025", "g100", "oc025")
EPOCH = dt.date(1982, 1, 1)
NSLOT = 73
NCHUNK = 12
HUB_RESOLVE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/"

SLOT_RULE = ("slot = min(72, floor((day_of_year(bin start, UTC) − 1) / 5)); bin b "
             "starts on 1982-01-01 + 5·b days; slot s holds the bins starting on "
             "days 5s+1 … 5s+5 of their year, slot 72 those starting on days "
             "361 … 366; a bin belongs to the year of its start")
LAYOUT = ("sum_pSS.npy [channel, year, lat, lon] float32 in PHYSICAL units, "
          "count_pSS.npy the same shape uint8, both CUMULATIVE over the years: "
          "plane (c, y) = every bin of slot SS in years year_first … year_first + y. "
          "Plane offset = header_len + (c * n_years + y) * H * W * itemsize; "
          "row 0 is the SOUTHERNMOST latitude (lat0), column 0 is lon0.")
COMBINE = ("mean of a slot over years a…b = (S[b] − S[a−1]) / (N[b] − N[a−1]) "
           "(S[a−1], N[a−1] = 0 when a is the first year); a year e left out "
           "subtracts (S[e] − S[e−1]) and (N[e] − N[e−1]); N = 0 → no value (NaN)")


# ============================================================== calendar ====
def bin_date(b):
    return EPOCH + dt.timedelta(days=5 * int(b))


def slot_of_date(d):
    return min(NSLOT - 1, (d.timetuple().tm_yday - 1) // 5)


def slot_of_bin(b):
    d = bin_date(b)
    return d.year, slot_of_date(d)


def slots_of_chunk(k):
    if not 0 <= k < NCHUNK:
        raise SystemExit(f"chunk must be 0..{NCHUNK - 1}")
    return list(range(6 * k, NSLOT if k == NCHUNK - 1 else 6 * k + 6))


def slot_table():
    """Per slot: the first day of year, the number of days (5, slot 72: 5 or
    6), and — in the nominal leap year 2000 the tab's time axis uses — the
    month and day of its first day (the month a slot is filed under)."""
    out = {}
    for s in range(NSLOT):
        d0 = dt.date(2000, 1, 1) + dt.timedelta(days=5 * s)
        out[f"p{s:02d}"] = dict(slot=s, start_doy=5 * s + 1,
                                days="5 or 6" if s == NSLOT - 1 else 5,
                                month_2000=d0.month, day_2000=d0.day)
    return out


def now_utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_sha():
    import subprocess
    try:
        r = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or "unknown"
    except Exception:                                         # noqa: BLE001
        return "unknown"


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def sha256_file(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(buf), b""):
            h.update(blk)
    return h.hexdigest()


def parse_npy_header(head):
    import ast
    if head[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file (bad magic)")
    if head[6] == 1:
        n, off = int.from_bytes(head[8:10], "little"), 10
    else:
        n, off = int.from_bytes(head[8:12], "little"), 12
    d = ast.literal_eval(head[off:off + n].decode("latin1"))
    return off + n, list(d["shape"]), d["descr"], bool(d["fortran_order"])


# ================================================================ source ====
RATE_WAIT_CAP_S = 3600


def rate_limit_wait(headers):
    import random
    import re
    ra = headers.get("Retry-After")
    if ra and str(ra).strip().isdigit():
        base = float(ra)
    else:
        m = re.search(r"t=(\d+)", headers.get("RateLimit", "") or "")
        base = float(m.group(1)) if m else 30.0
    return min(310.0, base + 1.0 + random.random() * 4.0)


class BinSource:
    """One group's bin-major .npy, on the Hub (HTTP ranges) or a local path.
    Every read is one whole bin slab; a Hub answer must be 206 with the slab's
    length and the file's total size in Content-Range."""

    def __init__(self, loc, G, attempts=6):
        self.loc = loc
        self.url = str(loc).startswith(("http://", "https://"))
        self.G = G
        self.H, self.W, self.C = G["grid"]["ny"], G["grid"]["nx"], len(G["chans"])
        self.slab = self.H * self.W * self.C * 2
        self.bin_first = int(G.get("bin_first", 0))
        self.n_bins = int(G["n_bins"])
        self.total = int(G["bytes"])
        self.hdr = int(G["header_len"])
        self.attempts = attempts
        self.bytes = 0
        self._tls = None
        if int(G.get("slab_bytes", self.slab)) != self.slab:
            raise SystemExit(f"slab_bytes {G.get('slab_bytes')} ≠ H·W·C·2 = {self.slab}")
        if self.total != self.hdr + self.n_bins * self.slab:
            raise SystemExit(f"file bytes {self.total} ≠ header + n_bins·slab")

    def _session(self):
        import threading
        import requests
        if self._tls is None:
            self._tls = threading.local()
        s = getattr(self._tls, "s", None)
        if s is None:
            s = requests.Session()
            self._tls.s = s
        return s

    def _range(self, a, n):
        if not self.url:
            with open(self.loc, "rb") as fh:
                fh.seek(a)
                b = fh.read(n)
            if len(b) != n:
                raise SystemExit(f"{self.loc}: short read at {a}")
            self.bytes += n
            return b
        tok = os.environ.get("HF_TOKEN", "").strip()
        hdr = {"Range": f"bytes={a}-{a + n - 1}"}
        if tok:
            hdr["Authorization"] = f"Bearer {tok}"
        tries, waited = 0, 0.0
        while True:
            try:
                r = self._session().get(self.loc, headers=hdr, timeout=300)
                if r.status_code == 206:
                    cr = r.headers.get("Content-Range", "")
                    if not cr.endswith(f"/{self.total}") or len(r.content) != n:
                        raise ConnectionError(f"range answer {cr!r}, {len(r.content)} bytes "
                                              f"(asked {n} of a {self.total}-byte file)")
                    self.bytes += n
                    return r.content
                if r.status_code == 200:
                    raise SystemExit(f"{self.loc}: the server ignored the Range header")
                if r.status_code == 429:
                    nap = rate_limit_wait(r.headers)
                    waited += nap
                    if waited > RATE_WAIT_CAP_S:
                        raise SystemExit(f"{self.loc}: rate-limited for {waited:.0f} s — giving up")
                    print(f"    429 from the Hub — waiting {nap:.0f} s", flush=True)
                    time.sleep(nap)
                    continue
                if r.status_code in (500, 502, 503, 504):
                    raise ConnectionError(f"HTTP {r.status_code}")
                raise SystemExit(f"{self.loc}: HTTP {r.status_code}")
            except SystemExit:
                raise
            except Exception as e:                            # noqa: BLE001
                tries += 1
                if tries >= self.attempts:
                    raise SystemExit(f"{self.loc}: gave up ({e})")
                time.sleep(min(60, 3 * 2 ** tries))

    def check_header(self):
        head = self._range(0, min(4096, self.total))
        hl, shp, descr, fo = parse_npy_header(head)
        want = [self.n_bins, self.H, self.W, self.C]
        if hl != self.hdr or shp != want or descr != "<f2" or fo:
            raise SystemExit(f"{self.loc}: header {hl}/{shp}/{descr}/{fo}, index says "
                             f"{self.hdr}/{want}/<f2/False")
        return hl, shp

    def slab_of(self, b):
        """Bin b as float16 [H, W, C] (z-scored, as stored)."""
        i = int(b) - self.bin_first
        if not 0 <= i < self.n_bins:
            raise SystemExit(f"bin {b} outside {self.bin_first}…{self.bin_first + self.n_bins - 1}")
        raw = self._range(self.hdr + i * self.slab, self.slab)
        return np.frombuffer(raw, dtype="<f2").reshape(self.H, self.W, self.C)


def physical(slab16, norm):
    """[H, W, C] float16 z → [C, H, W] float64 physical (NaN stays NaN)."""
    z = np.moveaxis(slab16, -1, 0).astype(np.float64)
    mu = np.array([n[0] for n in norm], np.float64)[:, None, None]
    sd = np.array([n[1] for n in norm], np.float64)[:, None, None]
    return z * sd + mu


def spans_to_check(Y):
    """(a, b, e) year-index spans, e = a year left out or None."""
    out = {(0, Y - 1, None), (Y - 1, Y - 1, None), (0, 0, None)}
    if Y > 21:
        out.add((Y - 21, Y - 2, None))
    if Y > 3:
        out.add((Y // 2, Y // 2, None))
        out.add((1, Y - 2, None))
        out.add((0, Y - 1, Y // 2))
        out.add((1, Y - 2, Y - 3))
    return sorted(out, key=lambda t: (t[0], t[1], -1 if t[2] is None else t[2]))


def group_meta(index, group):
    ix = json.load(open(index)) if isinstance(index, str) else index
    G = ix["groups"][group]
    return ix, G


def source_loc(ix, G, base):
    if base:
        return base if base.startswith(("http://", "https://")) else os.path.join(base, G["file"])
    return G.get("url") or (ix["base"] + G["file"])


# ================================================================ export ====
def export_chunk(group, chunk, out, *, index=SRC_INDEX, base=None, threads=4,
                 source_file=None):
    from numpy.lib.format import open_memmap
    t0 = time.time()
    ix, G = group_meta(index, group)
    src = BinSource(source_loc(ix, G, base), G)
    src.check_header()
    C, H, W = src.C, src.H, src.W
    norm = G["norm"]
    if len(norm) != C:
        raise SystemExit(f"{group}: {len(norm)} norms for {C} channels")
    slots = slots_of_chunk(chunk)
    bins = range(src.bin_first, src.bin_first + src.n_bins)
    years = list(range(bin_date(bins[0]).year, bin_date(bins[-1]).year + 1))
    Y = len(years)
    per = {s: {y: [] for y in years} for s in range(NSLOT)}
    for b in bins:
        y, s = slot_of_bin(b)
        per[s][y].append(b)
    os.makedirs(out, exist_ok=True)
    S = {s: open_memmap(os.path.join(out, f"sum_p{s:02d}.npy"), "w+", np.float32, (C, Y, H, W)) for s in slots}
    N = {s: open_memmap(os.path.join(out, f"count_p{s:02d}.npy"), "w+", np.uint8, (C, Y, H, W)) for s in slots}
    cs = {s: np.zeros((C, H, W), np.float64) for s in slots}
    cn = {s: np.zeros((C, H, W), np.int32) for s in slots}
    sample = sorted({slots[0], slots[-1]})
    keep = {s: {} for s in sample}                        # s -> {yi: [slab16]}
    nread = 0
    with cf.ThreadPoolExecutor(max_workers=threads) as ex:
        for yi, y in enumerate(years):
            todo = [(s, b) for s in slots for b in per[s][y]]
            for (s, b), slab in zip(todo, ex.map(lambda sb: src.slab_of(sb[1]), todo)):
                v = physical(slab, norm)
                fin = np.isfinite(v)
                cs[s] += np.where(fin, v, 0.0)
                cn[s] += fin
                nread += 1
                if s in keep:
                    keep[s].setdefault(yi, []).append(slab.copy())
            for s in slots:
                if cn[s].max() > 255:
                    raise SystemExit(f"{group} slot {s}: a cumulative count passed 255 in {y}")
                S[s][:, yi] = cs[s].astype(np.float32)
                N[s][:, yi] = cn[s].astype(np.uint8)
            print(f"  {group} chunk {chunk:02d} · {y}: {len(todo)} bins "
                  f"({src.bytes / 1e9:.2f} GB read, {time.time() - t0:.0f} s)", flush=True)
    for a in list(S.values()) + list(N.values()):
        a.flush()
    del S, N

    # ---- the falsifier: composed means against numpy on the native bins
    worst, worst_ratio, checks = 0.0, 0.0, 0
    for s, by_year in keep.items():
        Ss = np.load(os.path.join(out, f"sum_p{s:02d}.npy"), mmap_mode="r")
        Ns = np.load(os.path.join(out, f"count_p{s:02d}.npy"), mmap_mode="r")
        for a, b, e in spans_to_check(Y):
            planes = [(b, 1)] + ([(a - 1, -1)] if a else [])
            if e is not None:
                planes += [(e, -1)] + ([(e - 1, 1)] if e else [])
            ssum = np.zeros((C, H, W), np.float64)
            n = np.zeros((C, H, W), np.int64)
            ulp = np.zeros((C, H, W), np.float64)
            for p, sg in planes:
                ssum += sg * Ss[:, p].astype(np.float64)
                n += sg * Ns[:, p].astype(np.int64)
                ulp += np.spacing(np.abs(Ss[:, p]).astype(np.float32)).astype(np.float64)
            frs = [f for yi in range(a, b + 1) if yi != e for f in by_year.get(yi, [])]
            if not frs:
                if n.any():
                    raise SystemExit(f"falsifier: p{s:02d} span {a}-{b} ex {e} has counts with no bin")
                continue
            ref_s = np.zeros((C, H, W), np.float64)
            rn = np.zeros((C, H, W), np.int64)
            for f in frs:
                v = physical(f, norm)
                fin = np.isfinite(v)
                ref_s += np.where(fin, v, 0.0)
                rn += fin
            if not np.array_equal(rn, n):
                raise SystemExit(f"falsifier: p{s:02d} span {a}-{b} ex {e}: counts differ "
                                 f"from the native bins")
            ok = rn > 0
            if not ok.any():
                continue
            ref = ref_s[ok] / rn[ok]
            mean = ssum[ok] / n[ok]
            err = np.abs(mean - ref)
            bound = 0.5 * ulp[ok] / n[ok] * (1 + 1e-9) + 1e-12 * np.abs(ref) + 1e-30
            worst = max(worst, float(err.max()))
            worst_ratio = max(worst_ratio, float((err / bound).max()))
            checks += 1
        del Ss, Ns
    print(f"  falsifier: {checks} spans over slots {sample}, max |Δmean| {worst:.3g}, "
          f"max Δ/bound {worst_ratio:.3f}", flush=True)
    if worst_ratio > 1.0:
        raise SystemExit("falsifier FAILED: a composed mean is outside the float32 "
                         "rounding bound — nothing is uploaded")

    files = {}
    for s in slots:
        for role in ("sum", "count"):
            name = f"{role}_p{s:02d}.npy"
            p = os.path.join(out, name)
            with open(p, "rb") as fh:
                hl, shp, dty, _ = parse_npy_header(fh.read(4096))
            files[name] = dict(bytes=os.path.getsize(p), sha256=sha256_file(p),
                               header_len=hl, shape=list(shp), dtype=dty)
    gr = G["grid"]
    tab = slot_table()
    man = dict(
        _source="ml/export_gridded_poy.py", plan=PLAN, group=group, stem=ix.get("stem", STEM),
        prefix=prefix_of(group, ix.get("stem", STEM)), chunk=chunk, slots=slots,
        years=years, year_first=years[0], year_last=years[-1], n_years=Y,
        chans=list(G["chans"]), units=G.get("units", {}), labels=G.get("labels", {}),
        norm=[list(map(float, n)) for n in norm], zscored_source=True,
        H=H, W=W, lat0=float(gr["lat0"]), dlat=float(gr["step"]), lon0=float(gr["lon0"]),
        dlon=float(gr["step"]), south_first=bool(gr.get("south_first", True)),
        bin_first=src.bin_first, bin_last=src.bin_first + src.n_bins - 1,
        record_first=bin_date(src.bin_first).isoformat(),
        record_last=(bin_date(src.bin_first + src.n_bins) - dt.timedelta(days=1)).isoformat(),
        source_file=source_file or f"tensors/{ix.get('stem', STEM)}/{G['file']}",
        source_sha256=G.get("sha256"), source_bytes=int(G["bytes"]),
        slot_rule=SLOT_RULE,
        slot_bins={f"p{s:02d}": dict(tab[f"p{s:02d}"], n_bins=sum(len(per[s][y]) for y in years),
                                     bins_per_year=[len(per[s][y]) for y in years]) for s in slots},
        falsifier=dict(spans=checks, max_abs_mean=worst, max_mean_over_bound=worst_ratio,
                       slots_sampled=sample),
        files=files, git_sha=git_sha(), generated_utc=now_utc(),
        seconds=round(time.time() - t0), bytes_read=src.bytes, bins_read=nread)
    write_json(os.path.join(out, f"manifest_c{chunk:02d}.json"), man)
    return man


def prefix_of(group, stem=STEM):
    return f"tensors/{stem}/poy/{VERSION}/{group}"


def upload_chunk(group, chunk, out):
    from huggingface_hub import HfApi
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        raise SystemExit("no HF_TOKEN in the environment")
    api = HfApi(token=tok)
    man = json.load(open(os.path.join(out, f"manifest_c{chunk:02d}.json")))
    pre = man["prefix"]
    names = list(man["files"]) + [f"manifest_c{chunk:02d}.json"]
    t0 = time.time()
    gb = sum(f["bytes"] for f in man["files"].values()) / 1e9
    print(f"  uploading {len(man['files'])} files, {gb:.2f} GB to {pre} …", flush=True)
    for attempt in range(5):
        try:
            api.upload_folder(repo_id=REPO_ID, repo_type="dataset", folder_path=out,
                              path_in_repo=pre, allow_patterns=names,
                              commit_message=f"E-093: pentad-of-year sums of {group}, chunk {chunk:02d}")
            break
        except Exception as e:                                # noqa: BLE001
            print(f"    upload attempt {attempt + 1} failed: {e}", flush=True)
            if attempt == 4:
                raise
            time.sleep(30 * (attempt + 1))
    infos = {i.path: i for i in api.get_paths_info(
        REPO_ID, [f"{pre}/{n}" for n in man["files"]], repo_type="dataset")}
    for n, f in man["files"].items():
        i = infos.get(f"{pre}/{n}")
        if i is None or i.size != f["bytes"] or (i.lfs and i.lfs.sha256 != f["sha256"]):
            raise SystemExit(f"read-back: {pre}/{n} is not what was written")
    print(f"  committed and read back in {time.time() - t0:.0f} s", flush=True)


# ================================================================= index ====
STORE_KEYS = ("group", "stem", "prefix", "years", "year_first", "year_last", "n_years", "chans",
              "units", "labels", "norm", "zscored_source", "H", "W", "lat0", "dlat", "lon0",
              "dlon", "south_first", "bin_first", "bin_last", "record_first", "record_last",
              "source_file", "source_sha256", "source_bytes")


def build_index(groups, *, local=None, base=HUB_RESOLVE, stem=STEM, fetch=None):
    """data/gridded_poy_index.json from the twelve chunk manifests of each
    group (local: DIR/<group>/manifest_cKK.json; else the Hub)."""
    out = dict(_source="ml/export_gridded_poy.py index — do not hand-edit",
               plan=f"https://blauewelt.github.io/earth/docs.html?f={PLAN}",
               base=base, layout=LAYOUT, combine=COMBINE, slot_rule=SLOT_RULE,
               n_slots=NSLOT, slot_table=slot_table(),
               time_rule=("the tab's time value of slot s is 2000-01-01 + 5·s days (a nominal "
                          "leap year); a slot is filed under the month of that day"),
               generated_utc=now_utc(), stores={})
    for group in groups:
        pre = prefix_of(group, stem)
        blk, slots, fals = None, {}, []
        for k in range(NCHUNK):
            if fetch:
                man = fetch(group, k)
            elif local:
                p = os.path.join(local, group, f"manifest_c{k:02d}.json")
                man = json.load(open(p)) if os.path.exists(p) else None
            else:
                import requests
                r = requests.get(f"{base}{pre}/manifest_c{k:02d}.json", timeout=120)
                if r.status_code == 404:
                    man = None
                else:
                    r.raise_for_status()
                    man = r.json()
            if man is None:
                continue
            if blk is None:
                blk = {kk: man[kk] for kk in STORE_KEYS}
            elif man["years"] != blk["years"] or man["chans"] != blk["chans"] or \
                    man["source_sha256"] != blk["source_sha256"]:
                raise SystemExit(f"{group}: chunk {k} was built over a different record than "
                                 f"the first chunk — rebuild the group")
            for s in man["slots"]:
                key = f"p{s:02d}"
                sf, cf_ = man["files"][f"sum_{key}.npy"], man["files"][f"count_{key}.npy"]
                want = [len(blk["chans"]), blk["n_years"], blk["H"], blk["W"]]
                if sf["shape"] != want or cf_["shape"] != want:
                    raise SystemExit(f"{group} {key}: shape {sf['shape']}, expected {want}")
                sb = man["slot_bins"][key]
                slots[key] = dict(sum_bytes=sf["bytes"], sum_header=sf["header_len"],
                                  count_bytes=cf_["bytes"], count_header=cf_["header_len"],
                                  sum_sha256=sf["sha256"], count_sha256=cf_["sha256"],
                                  start_doy=sb["start_doy"], month_2000=sb["month_2000"],
                                  day_2000=sb["day_2000"], n_bins=sb["n_bins"],
                                  bins_per_year=sb["bins_per_year"])
            fals.append(man["falsifier"])
        if blk is None:
            print(f"  {group}: nothing published yet")
            continue
        blk["slots"] = dict(sorted(slots.items()))
        blk["complete"] = len(slots) == NSLOT
        blk["falsifier"] = dict(max_abs_mean=max(f["max_abs_mean"] for f in fals),
                                max_mean_over_bound=max(f["max_mean_over_bound"] for f in fals))
        out["stores"][f"{stem}/{group}"] = blk
        print(f"  {group}: {len(slots)} of {NSLOT} slots")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("chunk")
    m.add_argument("--group", required=True)
    m.add_argument("--chunk", type=int, required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--index", default=SRC_INDEX)
    m.add_argument("--base")
    m.add_argument("--threads", type=int, default=4)
    m.add_argument("--upload", action="store_true")
    i = sub.add_parser("index")
    i.add_argument("--groups", default=",".join(GROUPS))
    i.add_argument("--index", default=INDEX)
    i.add_argument("--local")
    a = ap.parse_args(argv)
    if a.cmd == "chunk":
        out = os.path.join(a.out, a.group) if not a.out.rstrip("/").endswith(a.group) else a.out
        export_chunk(a.group, a.chunk, out, index=a.index, base=a.base, threads=a.threads)
        if a.upload:
            upload_chunk(a.group, a.chunk, out)
    else:
        idx = build_index([g for g in a.groups.split(",") if g], local=a.local)
        write_json(a.index, idx)
        print(f"wrote {a.index}")


if __name__ == "__main__":
    main()
