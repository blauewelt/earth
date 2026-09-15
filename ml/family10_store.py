#!/usr/bin/env python3
"""Family 10's TIER-P STORE reader — points, tracks and profiles, and the k nearest.

E-078 §2-§3 and E-079 §2, made readable. `ml/build_family10_stores.py` writes
the stores; this file is the only thing that opens them, so the layout is
defined once. It is `ml/family8_store.py` generalised in exactly three ways and
in no others:

  1. **C value columns instead of two named ones.** Family 8 carries `temp` and
     `psal` as separate (N, 16) blocks; family 10 carries one `values.npy` of
     shape (N, C) plus a `channels` list in `store.json`. The family-8 store is
     read through this class unchanged by treating its two blocks as one
     32-column value matrix (`temp` levels 0..15 then `psal` levels 0..15) —
     there is no rebuild and no copy: the blocks stay separate memmaps and only
     the k selected rows are ever concatenated.
  2. **A PER-ROW footprint.** `fp.npy` is (N, 2) float16 holding
     `(log2_fp, log2_dt)` from E-078 §2. For every source built so far it is
     constant down the column — a drifter is a point, an altimeter sample is
     7 km — but it is stored per row because a store that later mixes supports
     (a mooring that changed sampling, a tile-token store) must read
     identically. Family 8 has no `fp.npy`; its constants come out of its
     `store.json` `footprint` block, or default to (-4, -4).
  3. **A BIN RANGE THAT MAY START BEFORE THE EPOCH.** Drifters begin in 1979 and
     SOCAT in 1957, so `bin` is a SIGNED int16 and may be negative. The CSR
     index is over the store's own range: `bin_offsets[i]` opens bin
     `bin_first + i`, and `bin_first` is in `store.json`. Family 8's
     `bin_first` is 0 and its index is the familiar 3,143 entries.

WHAT IS IN A FAMILY-10 STORE (`tensors/family10_1/<store>/`, N rows sorted by
`(bin, time_s)` ascending):

    bin.npy         int16   [N]      five-day bin, floor(time_s / 432000 s);
                                     NEGATIVE before 1982 and kept
    time_s.npy      int32   [N]      SECONDS since 1982-01-01T00:00:00Z,
                                     negative before it (schema 2)
    lat.npy         float32 [N]      degrees north
    lon.npy         float32 [N]      degrees east, in [-180, 180)
    values.npy      float16 [N, C]   the channels, RAW units, NaN = not measured
    platform.npy    int64   [N]      drifter id / mooring code / expocode /
                                     mission, as an integer (see store.json)
    qc.npy          uint8   [N]      0 not assessed, 1 good, 2 probably good,
                                     3+ the source's worse grades
    fp.npy          float16 [N, 2]   (log2_fp, log2_dt), E-078 §2
    bin_offsets.npy int64   [B + 1]  CSR over bins bin_first .. bin_first+B-1
    store.json                       schema, footprint, QC policy, provenance,
                                     per-year counts, sha256 of every file

TWO SCHEMAS, AND THIS READER OPENS BOTH.

  schema 1 (family 10, family 8) carried the time as `time_days.npy`, float32
  days since the epoch. float32 has 24 bits of mantissa, so at 15,700 days —
  the end of the record — consecutive representable values are **0.977 ms x
  86400 = 84 seconds** apart (21 s in 1993, where the exponent is one lower).
  Against slatrack's 1 Hz along-track sampling that is a column in which up to
  316 consecutive samples share one timestamp: the order inside a bin is
  correct and the timestamps cannot express it (measured, E-079 §10.1).

  schema 2 (family 10.1) carries `time_s.npy`, int32 seconds. One second is
  the finest thing any of these four archives reports, so the column is exact
  everywhere, and `bin = floor(time_s / 432000)` is integer arithmetic with no
  float rounding anywhere in it. int32 spans **1913-12-13 20:45:52 ..
  2050-01-19 03:14:07**; the builder refuses a row outside that (E-079 §10.1),
  which is the 2050 limit stated where it can be acted on rather than
  discovered.

`store.json` says which: `schema_version` 1 or 2, and `family_version` "10.1"
on a schema-2 store. A schema-1 store is read by converting its days to
seconds on the fly — the precision it never had is NOT recovered, the two
schemas are only put in one unit — so family 8's Argo store and family 10's
published stores keep opening unchanged. `dt_days` out of `knearest` is
float64 days under both.

THE SEARCH (`knearest`) is family 8's, unchanged in behaviour:

  ONE-SIDED IN TIME is a hard constraint of the search, not a mask applied
  afterwards — the CSR slice stops at the anchor's own bin and never reads a
  later row. BOUNDED by `R_max_km` and `T_max_days`; inside them the k nearest
  under the metric d^2 = (dx^2 + dy^2)/L^2 + dt^2/T^2 are taken, and the
  remaining slots carry the MISS TOKEN (NaN values, `valid = False`). `dt` is
  measured from the END of the anchor's pentad, so an observation inside the
  anchor's own bin is 0 to 5 days old rather than arriving from the future.

    from family10_store import Store
    st = Store.open("ml/cache/family10_1/gdp")          # a directory
    st = Store.open("chfrank/earth-tensors:tensors/family10_1/gdp")  # the Hub
    tok = st.knearest(36.0, -70.0, bin=2411, k=8,
                      R_max_km=300.0, T_max_days=10.0)
    tok["values"]     # (8, C) float32, NaN where a channel or a slot is missing
    tok["mask"]       # (8, C) bool  — True where a real number was measured
    tok["dx_km"], tok["dy_km"], tok["dt_days"], tok["n_R"]
    tok["log2_fp"], tok["log2_dt"], tok["platform"], tok["qc"]

Pure numpy; no scipy, no torch. The search is a vectorised brute force over the
rows of the bins the time bound admits.
"""
import hashlib
import json
import os

import numpy as np

# ---------------------------------------------------------------- the axis --
EPOCH = "1982-01-01"
PENTAD_DAYS = 5
SECONDS_PER_DAY = 86400
# 5 days, in seconds, EXACTLY. Every bin derivation in family 10.1 is
# `floor_divide` by this integer; no float appears anywhere in it.
PENTAD_SECONDS = PENTAD_DAYS * SECONDS_PER_DAY            # 432_000

# ------------------------------------------------------------- the version --
# ONE constant. Every prefix, every directory name and every registry field
# below is derived from it, so the family's version is changed in one place
# and the paths cannot disagree with the schema they carry.
FAMILY = "family10"
FAMILY_VERSION = "10.1"
SCHEMA_VERSION = 2
_VER_SLUG = "family" + FAMILY_VERSION.replace(".", "_")   # family10_1
HF_ROOT = f"tensors/{_VER_SLUG}"                          # the stores + registry
HF_PARTIALS = f"partials/{_VER_SLUG}"                     # the slatrack lanes
REGISTRY_NAME = "family10.json"
CACHE_DIRNAME = _VER_SLUG                                 # ml/cache/family10_1

# int32 seconds since 1982-01-01T00:00:00Z. The bounds are the dtype's, stated
# as dates because that is the form in which anybody will meet them.
TIME_S_MIN = -2_147_483_648            # 1913-12-13T20:45:52Z
TIME_S_MAX = 2_147_483_647             # 2050-01-19T03:14:07Z
TIME_S_MIN_DATE = "1913-12-13T20:45:52Z"
TIME_S_MAX_DATE = "2050-01-19T03:14:07Z"

# ONE definition of the ground scale, shared with ml/family8_store.py,
# ml/temporal.py (KM_PER_DEG) and ml/cone.py (0.25 deg x 111.32 = 27.83 km a
# cell). Restated rather than imported for the same reason family 8 restates
# it: importing `cone` pulls in torch, and this reader must run anywhere.
KM_PER_DEG = 111.32
COS_FLOOR = 0.05

# E-078 §2's clamp. A value outside it is a builder bug, not a footprint.
LOG2_FP_RANGE = (-12.0, 12.0)

# What a family-8 store carries when it has no fp.npy (E-076 §2.6: a point
# profile taken in minutes; both clamp to the plan's -4 floor).
DEFAULT_LOG2_FP = -4.0
DEFAULT_LOG2_DT = -4.0

# The core columns, per schema. The TIME column is the only difference.
CORE_COLUMNS = ("bin", "time_s", "lat", "lon")                  # schema 2
CORE_COLUMNS_V1 = ("bin", "time_days", "lat", "lon")            # schema 1
CORE_DTYPES = {"bin": "int16", "time_s": "int32",
               "lat": "float32", "lon": "float32"}
CORE_DTYPES_V1 = {"bin": "int16", "time_days": "float32",
                  "lat": "float32", "lon": "float32"}
TIME_COLUMN = {1: "time_days", 2: "time_s"}

# The family-8 layout, recognised so its store opens here with no rebuild.
FAMILY8_BLOCKS = ("temp", "psal")


def anchor_time_days(b):
    """The instant `dt_days` is measured back from: the END of bin `b`.

    A bin is a five-day span, and an observation inside the anchor's own bin
    must not read as being in the future (family 8's rule, unchanged).
    """
    return float(PENTAD_DAYS * (int(b) + 1))


def anchor_time_s(b):
    """The same instant, in SECONDS — the END of bin `b`, exactly.

    `PENTAD_SECONDS * (b + 1)` is an integer for every bin, so the anchor of a
    schema-2 store is representable with no rounding at all. `anchor_time_days`
    is the same number divided by 86400 and stays the reader's public form,
    because `dt` is reported in days under both schemas.
    """
    return int(PENTAD_SECONDS * (int(b) + 1))


def offsets_km(lat0, lon0, lat, lon):
    """Signed (dx_km east, dy_km north) from anchor (lat0, lon0) to (lat, lon).

    Identical arithmetic to `ml/family8_store.offsets_km`; the test asserts the
    two agree on a grid of pairs rather than trusting that they were copied.
    """
    lat = np.asarray(lat, np.float64)
    lon = np.asarray(lon, np.float64)
    dy = (lat - float(lat0)) * KM_PER_DEG
    dlon = np.mod(lon - float(lon0) + 180.0, 360.0) - 180.0
    coslat = np.maximum(np.cos(np.radians(0.5 * (lat + float(lat0)))), COS_FLOOR)
    dx = dlon * KM_PER_DEG * coslat
    return dx, dy


def wrap_lon(lon):
    """Degrees east into [-180, 180). The store's invariant, in one place."""
    return np.mod(np.asarray(lon, np.float64) + 180.0, 360.0) - 180.0


def bin_of_seconds(sec):
    """floor(time_s / 432000), NEGATIVE before the epoch. INTEGER arithmetic.

    This is family 10.1's bin rule and there is no float in it: numpy's
    `floor_divide` on int64 floors toward minus infinity exactly as Python's
    `//` does, so -1 s is bin -1 and 431,999 s is bin 0 with nothing to round.
    The old float path (`bin_of_days`) could put a timestamp a microsecond
    before a boundary on the wrong side of it, which is why `_pack` had to
    compute the bin from the value it was about to STORE rather than from the
    value it had parsed. That whole class is gone.
    """
    return np.floor_divide(np.asarray(sec, np.int64), PENTAD_SECONDS)


def bin_of_days(days):
    """floor(days-since-1982-01-01 / 5), NEGATIVE before the epoch.

    The SCHEMA-1 rule, kept because schema-1 stores (family 8's Argo store and
    family 10's published four) are still opened by this reader and still have
    to be checked against their own float32 column. `np.floor` is a true floor,
    so -0.5 days is bin -1 and not bin 0 — which is the whole reason pre-1982
    rows can be kept at all. New stores use `bin_of_seconds`.
    """
    return np.floor(np.asarray(days, np.float64) / PENTAD_DAYS).astype(np.int64)


def seconds_of_days(days):
    """Schema-1 float32 DAYS -> int64 SECONDS, rounded to the nearest second.

    The one conversion the v1 read path needs. It does NOT recover precision
    the column never had — a slatrack v1 row is still only good to 84 s — it
    only puts the two schemas in one unit so a single code path can read both.
    """
    return np.rint(np.asarray(days, np.float64) * SECONDS_PER_DAY).astype(np.int64)


def check_part_schema(path):
    """REFUSE a schema-1 column part. A v1 part CANNOT be upgraded.

    `ml/family10_parts_hub.py` parks a fetch lane's per-year `.npz` parts on
    the Hub and a box assembles them, so the two halves of a slatrack build can
    be months and a schema apart. A part written by the v1 builder carries
    `time_days` — float32 days — and there is no arithmetic that turns it back
    into the seconds the archive had: the precision was destroyed at the cast,
    before the part was written. Multiplying by 86400 would produce an int
    column that LOOKS like schema 2 and is wrong by up to 84 seconds, which is
    the one outcome this whole change exists to remove. So the part is refused,
    by name, and the year is refetched from the source.

    Reads only the zip directory (npz is a zip), so it costs a seek on a
    multi-gigabyte part.
    """
    import zipfile
    with zipfile.ZipFile(path) as z:
        names = {n[:-4] for n in z.namelist() if n.endswith(".npy")}
    if "time_s" in names:
        return 2
    if "time_days" in names:
        raise ValueError(
            f"{path} is a SCHEMA-1 column part: it carries `time_days` "
            f"(float32 days) and family {FAMILY_VERSION} stores `time_s` "
            f"(int32 seconds). A v1 part cannot be upgraded — float32 days "
            f"resolve 21-84 s over this record, so the seconds the archive "
            f"published are already gone and multiplying by 86400 would write "
            f"a column that looks exact and is wrong by up to 84 s. Refetch "
            f"the year from the source with the current builder; the v1 parts "
            f"under the old `partials/family10/` prefix are not inputs to this "
            f"build and nothing reads them.")
    raise ValueError(
        f"{path} carries no time column at all (members {sorted(names)}) — it "
        f"is not a family-10 column part")


def csr_offsets(bin_sorted, bin_first, n_bins):
    """CSR offsets over bins `bin_first .. bin_first + n_bins - 1`.

    `bin_sorted` must be ascending. Returns int64 [n_bins + 1] with
    `off[0] == 0` and `off[-1] == len(bin_sorted)`.
    """
    b = np.asarray(bin_sorted, np.int64)
    edges = np.arange(int(bin_first), int(bin_first) + int(n_bins) + 1,
                      dtype=np.int64)
    return np.searchsorted(b, edges, side="left").astype(np.int64)


def sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def verify_store(path, chunk=1 << 22):
    """Every file's sha256 against `store.json`'s own record.

    Family 8's rule, one store wider: a truncated `values.npy` still memmaps and
    still answers `knearest`, with whatever the tail of the file happens to be.
    Raises on any mismatch; returns the number of files checked.
    """
    path = os.path.abspath(path)
    mp = os.path.join(path, "store.json")
    if not os.path.exists(mp):
        raise FileNotFoundError(
            f"{mp} is missing — {path} is not a family-10 store "
            f"(docs/FAMILY10_DATA_HANDOVER.md §2 lists the files)")
    with open(mp) as fh:
        want = json.load(fh).get("sha256") or {}
    if not want:
        raise ValueError(
            f"{mp} carries no sha256 block, so nothing about this store can be "
            f"verified. Refusing rather than reading bytes of unknown "
            f"provenance.")
    bad = []
    for name, digest in sorted(want.items()):
        p = os.path.join(path, name)
        if not os.path.exists(p):
            bad.append(f"{name}: missing")
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for blk in iter(lambda: fh.read(chunk), b""):
                h.update(blk)
        got = h.hexdigest()
        if got != digest:
            bad.append(f"{name}: {got[:12]} != {digest[:12]}")
    if bad:
        raise ValueError(
            f"family-10 store at {path} does not match its own store.json:\n  "
            + "\n  ".join(bad)
            + "\nA truncated file still memmaps and still answers a search, "
              "with whatever its tail happens to hold — re-fetch it.")
    return len(want)


# ----------------------------------------------------------- hub resolution --
def _looks_like_hub(spec):
    return ("://" in spec) or (":" in spec and "/" in spec.split(":", 1)[0])


def resolve(spec, cache_dir=None, token=None):
    """A directory on disk, or a Hub prefix -> a local directory.

    Accepts `"/some/dir"`, `"repo/name:prefix/in/repo"` and
    `"hf://repo/name/prefix/in/repo"`. A Hub prefix downloads `store.json`
    first, then exactly the files it names — never a whole-repo snapshot, which
    for `chfrank/earth-tensors` would be tens of gigabytes of family-7 tensors
    nobody asked for.
    """
    if os.path.isdir(spec):
        return os.path.abspath(spec)
    if not _looks_like_hub(spec):
        raise FileNotFoundError(
            f"{spec} is not a directory and is not a Hub prefix of the form "
            f"'<owner>/<repo>:<path/in/repo>' or 'hf://<owner>/<repo>/<path>'")
    if spec.startswith("hf://"):
        rest = spec[len("hf://"):]
        parts = rest.split("/")
        repo, prefix = "/".join(parts[:2]), "/".join(parts[2:])
    else:
        repo, prefix = spec.split(":", 1)
    prefix = prefix.strip("/")
    from huggingface_hub import hf_hub_download
    cache_dir = cache_dir or os.path.join(
        os.path.expanduser("~"), ".cache", "earth-family10",
        repo.replace("/", "_"), prefix.replace("/", "_"))
    os.makedirs(cache_dir, exist_ok=True)
    meta = hf_hub_download(repo, f"{prefix}/store.json", repo_type="dataset",
                           token=token, local_dir=cache_dir)
    with open(meta) as fh:
        names = sorted((json.load(fh).get("sha256") or {}).keys())
    if not names:
        raise ValueError(f"{repo}:{prefix}/store.json carries no sha256 block")
    for n in names:
        if n == "store.json":
            continue
        hf_hub_download(repo, f"{prefix}/{n}", repo_type="dataset",
                        token=token, local_dir=cache_dir)
    # `local_dir` preserves the repo path, so every file landed beside the
    # store.json we just read — that directory IS the store.
    return os.path.dirname(os.path.abspath(meta))


# ==================================================================== Store ==
class Store:
    """A tier-P observation store, opened with memmaps. Read-only.

    Open it with `Store.open(...)`, not the constructor, so a directory and a
    Hub prefix are the same call.
    """

    # -- construction ------------------------------------------------------
    @classmethod
    def open(cls, spec, verify=False, cache_dir=None, token=None):
        """`Store.open(dir_or_hub_prefix)`. `verify=True` re-hashes every file."""
        path = resolve(spec, cache_dir=cache_dir, token=token)
        if verify:
            verify_store(path)
        return cls(path)

    def __init__(self, path):
        self.path = os.path.abspath(path)
        mp = os.path.join(self.path, "store.json")
        if not os.path.exists(mp):
            raise FileNotFoundError(
                f"{mp} is missing — {self.path} is not a family-10 tier-P "
                f"store. Build one with ml/build_family10_stores.py, or open a "
                f"published one with Store.open('<owner>/<repo>:<prefix>').")
        with open(mp) as fh:
            self.meta = json.load(fh)

        # -- WHICH SCHEMA. The COLUMN ON DISK decides, and `store.json` is
        # then checked against it: a store.json that claims schema 2 over a
        # float32 `time_days` would have every consumer believing a precision
        # the bytes do not carry, so the disagreement is fatal rather than
        # resolved in either direction.
        has_s = os.path.exists(os.path.join(self.path, "time_s.npy"))
        has_d = os.path.exists(os.path.join(self.path, "time_days.npy"))
        if has_s:
            self.schema_version = 2
        elif has_d:
            self.schema_version = 1
        else:
            raise FileNotFoundError(
                f"{self.path} carries neither time_s.npy (schema 2) nor "
                f"time_days.npy (schema 1) — nothing in it has a time")
        claimed = int(self.meta.get("schema_version", 1) or 1)
        if claimed != self.schema_version:
            raise ValueError(
                f"{mp} says schema_version {claimed} and the store holds "
                f"{TIME_COLUMN[self.schema_version]}.npy, which is schema "
                f"{self.schema_version}. Refusing to guess which is right: "
                f"one of them is a store assembled under one schema and "
                f"described under the other.")
        self.time_column = TIME_COLUMN[self.schema_version]

        self._col = {}
        for name in (CORE_COLUMNS if self.schema_version == 2
                     else CORE_COLUMNS_V1):
            p = os.path.join(self.path, name + ".npy")
            if not os.path.exists(p):
                raise FileNotFoundError(f"{p} is missing from the store")
            self._col[name] = np.load(p, mmap_mode="r")
        self.N = int(self._col["bin"].shape[0])

        # -- the value blocks. ONE code path for both layouts: family 10 has a
        # single `values` block; family 8 has `temp` and `psal`, which are
        # concatenated LAZILY — only the k selected rows are ever materialised,
        # so opening the 2.68 M-profile Argo store costs no copy at all.
        self._blocks = []
        self.layout = "family10"
        vp = os.path.join(self.path, "values.npy")
        if os.path.exists(vp):
            a = np.load(vp, mmap_mode="r")
            if a.ndim != 2:
                raise ValueError(f"{vp} has shape {a.shape}; expected [N, C]")
            self._blocks.append(("values", a))
        else:
            for nm in FAMILY8_BLOCKS:
                p = os.path.join(self.path, nm + ".npy")
                if os.path.exists(p):
                    self._blocks.append((nm, np.load(p, mmap_mode="r")))
            if not self._blocks:
                raise FileNotFoundError(
                    f"neither values.npy nor {'/'.join(FAMILY8_BLOCKS)}.npy is "
                    f"in {self.path} — nothing in it is a measurement")
            self.layout = "family8"
        for nm, a in self._blocks:
            if a.shape[0] != self.N:
                raise ValueError(f"{nm}.npy has {a.shape[0]} rows, the store "
                                 f"has {self.N}")
        self._block_widths = [int(a.shape[1]) for _n, a in self._blocks]
        self.C = int(sum(self._block_widths))
        self.channels = self._channel_names()
        if len(self.channels) != self.C:
            raise ValueError(
                f"store.json names {len(self.channels)} channel(s) but the "
                f"value block(s) are {self.C} column(s) wide")

        # -- platform / qc / fp. Absent in a family-8 store; synthesised from
        # what it does carry, never invented: `wmo` IS the platform, an
        # unflagged store is "not assessed" (0), and the footprint constants
        # come from its own store.json.
        self._platform = self._optional("platform", np.int64)
        if self._platform is None and os.path.exists(
                os.path.join(self.path, "wmo.npy")):
            self._platform = np.load(os.path.join(self.path, "wmo.npy"),
                                     mmap_mode="r")
        self._qc = self._optional("qc", np.uint8)
        self._fp = self._optional("fp", np.float16)
        fpm = self.meta.get("footprint") or {}
        self.log2_fp_const = float(fpm.get("log2_fp", DEFAULT_LOG2_FP))
        self.log2_dt_const = float(fpm.get("log2_dt", DEFAULT_LOG2_DT))

        # -- the CSR index, over the store's OWN bin range ---------------------
        op = os.path.join(self.path, "bin_offsets.npy")
        if not os.path.exists(op):
            raise FileNotFoundError(f"{op} is missing from the store")
        self.bin_offsets = np.load(op)
        self.n_bins = int(len(self.bin_offsets) - 1)
        self.bin_first = int(self.meta.get("bin_first", 0))
        self.bin_last = self.bin_first + self.n_bins - 1
        if int(self.bin_offsets[0]) != 0 or int(self.bin_offsets[-1]) != self.N:
            raise ValueError(
                f"bin_offsets runs {int(self.bin_offsets[0])}.."
                f"{int(self.bin_offsets[-1])} over {self.N} rows — the CSR "
                f"index does not describe this store")
        if int(self.meta.get("N", self.N)) != self.N:
            raise ValueError(f"store.json says N={self.meta['N']}, the arrays "
                             f"hold {self.N}")
        # Levels, for a family-8 store read through this class: the 32 columns
        # are temp at 16 pressures then psal at the same 16.
        self.levels = np.asarray(self.meta.get("levels", []), np.float64)

    def _optional(self, name, dtype):
        p = os.path.join(self.path, name + ".npy")
        if not os.path.exists(p):
            return None
        a = np.load(p, mmap_mode="r")
        if a.shape[0] != self.N:
            raise ValueError(f"{name}.npy has {a.shape[0]} rows, the store "
                             f"has {self.N}")
        return a

    def _channel_names(self):
        """The C channel names, from store.json or derived for family 8."""
        ch = self.meta.get("channels")
        if ch:
            return [c["name"] if isinstance(c, dict) else str(c) for c in ch]
        if self.layout == "family8":
            lv = self.meta.get("levels") or list(range(self._block_widths[0]))
            return [f"{nm}_{int(round(float(p)))}"
                    for (nm, _a), w in zip(self._blocks, self._block_widths)
                    for p in (lv[:w] if len(lv) >= w else range(w))]
        return [f"c{i}" for i in range(self.C)]

    def channel_units(self):
        ch = self.meta.get("channels")
        if ch and isinstance(ch[0], dict):
            return [c.get("unit", "") for c in ch]
        if self.layout == "family8":
            n0 = self._block_widths[0]
            return ["degC"] * n0 + ["PSU"] * (self.C - n0)
        return [""] * self.C

    # -- columns -----------------------------------------------------------
    def __len__(self):
        return self.N

    def time_s(self, lo=0, hi=None):
        """Rows [lo, hi) as int64 SECONDS since the epoch, under BOTH schemas.

        Schema 2 reads the column; schema 1 converts its float32 days through
        `seconds_of_days`, which rounds and recovers nothing.
        """
        hi = self.N if hi is None else hi
        a = self._col[self.time_column][lo:hi]
        if self.schema_version == 2:
            return np.asarray(a, np.int64)
        return seconds_of_days(a)

    def time_days(self, lo=0, hi=None):
        """Rows [lo, hi) as float64 DAYS since the epoch, under BOTH schemas.

        This is what the search measures `dt` in, so it is the one number the
        two schemas must agree on for a timestamp they both represent.
        """
        hi = self.N if hi is None else hi
        a = self._col[self.time_column][lo:hi]
        if self.schema_version == 2:
            return np.asarray(a, np.int64) / float(SECONDS_PER_DAY)
        return np.asarray(a, np.float64)

    def __getitem__(self, name):
        # `time_s` and `time_days` both answer on either schema: the NATIVE one
        # hands back the memmap, the other computes the whole column. A caller
        # that wants a block (and on slatrack every caller should) asks
        # `st.time_s(lo, hi)` instead.
        if name == "time_s":
            return (self._col["time_s"] if self.schema_version == 2
                    else self.time_s())
        if name == "time_days":
            return (self._col["time_days"] if self.schema_version == 1
                    else self.time_days())
        if name in self._col:
            return self._col[name]
        for nm, a in self._blocks:
            if nm == name:
                return a
        for nm, a in (("platform", self._platform), ("qc", self._qc),
                      ("fp", self._fp)):
            if nm == name and a is not None:
                return a
        raise KeyError(name)

    def _slice(self, b0, b1):
        """Row range [lo, hi) of bins b0..b1 INCLUSIVE, clipped to the store."""
        b0 = max(self.bin_first, int(b0))
        b1 = min(self.bin_last, int(b1))
        if b1 < b0:
            return 0, 0
        return (int(self.bin_offsets[b0 - self.bin_first]),
                int(self.bin_offsets[b1 - self.bin_first + 1]))

    def count(self, b0, b1):
        """How many observations lie in bins b0..b1 inclusive."""
        lo, hi = self._slice(b0, b1)
        return hi - lo

    def bins(self, b0, b1):
        """Every row of bins b0..b1 inclusive, as a dict of array views.

        Carries BOTH time forms, whichever schema the store is in, so a caller
        never has to ask which one it opened.
        """
        lo, hi = self._slice(b0, b1)
        out = {k: v[lo:hi] for k, v in self._col.items()}
        out["time_s"] = self.time_s(lo, hi)
        out["time_days"] = self.time_days(lo, hi)
        out["values"] = self._gather_values(np.arange(lo, hi, dtype=np.int64))
        out["row"] = np.arange(lo, hi, dtype=np.int64)
        return out

    def _gather_values(self, rows):
        """(len(rows), C) float32 — the ONE place the two layouts differ."""
        rows = np.asarray(rows, np.int64)
        if len(self._blocks) == 1:
            return np.asarray(self._blocks[0][1][rows], np.float32)
        return np.concatenate(
            [np.asarray(a[rows], np.float32).reshape(len(rows), -1)
             for _n, a in self._blocks], axis=1)

    # -- the search --------------------------------------------------------
    def knearest(self, lat, lon, bin, k=5, R_max_km=1000.0, T_max_days=30.0,
                 L_km=None, T_days=None, qc_max=None):
        """The k observations nearest `(lat, lon)` at or BEFORE pentad `bin`.

        Returns a dict of fixed length `k` — always `k` slots, so a batch is
        never ragged. Slot i is real where `valid[i]` and a MISS TOKEN (NaN
        everywhere, `row = -1`) where it is not.

        Keys: `values` (k, C) float32 in the store's raw units with NaN for an
        unmeasured channel; `mask` (k, C) bool, True where a real number was
        measured; `dx_km`, `dy_km`, `dt_days`, `dist_km`, `d2` (k,);
        `log2_fp`, `log2_dt` (k,) — per row, from `fp.npy` where the store has
        one and the store's constants where it does not; `platform` (k,) int64;
        `qc` (k,) uint8; `n_R` (scalar) — how many observations lay inside
        `(R_max_km, T_max_days)`, the local density feature; `valid` (k,) bool;
        `row`, `lat`, `lon`, `time_days` and `time_s` (k,) for provenance —
        `time_s` is float64 seconds so that a miss slot can be NaN like every
        other per-slot field, and every real value in it is an exact integer;
        `n_found`, `channels`.

        `qc_max` additionally excludes rows whose stored flag is worse than it.
        The builder already applies the store's QC policy, so this is the
        reader-side tightening, not the policy — and it is applied BEFORE `n_R`
        is counted, because a token the reader will not return must not inflate
        the density feature.

        `L_km` and `T_days` are the metric's scales and default to the search
        bounds, which makes the metric's two axes commensurate at the boundary.
        """
        k = int(k)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        b = int(bin)
        L = float(L_km if L_km is not None else R_max_km)
        T = float(T_days if T_days is not None else T_max_days)
        if L <= 0 or T <= 0:
            raise ValueError(f"metric scales must be positive, got L={L}, T={T}")
        R_max_km = float(R_max_km)
        T_max_days = float(T_max_days)

        out = self._miss(k)
        if self.N == 0 or b < self.bin_first:
            return out
        t_anchor = anchor_time_days(b)

        # ONE-SIDED, AND BOUNDED, BY CONSTRUCTION: the slice ends at the
        # anchor's own bin, and starts at the oldest bin that could still hold
        # an observation no older than T_max.
        b_hi = min(b, self.bin_last)
        b_lo = int(np.floor((t_anchor - T_max_days) / PENTAD_DAYS))
        lo, hi = self._slice(b_lo, b_hi)
        if hi <= lo:
            return out

        # DAYS, float64, from whichever column this store carries. Under
        # schema 2 that is `time_s / 86400`, which is exact to the last bit for
        # every second the int32 can hold; under schema 1 it is the float32
        # column widened. Either way `dt_days` below is in DAYS, so the
        # search's bounds and its metric are unchanged by the schema.
        t = self.time_days(lo, hi)
        ok = (t_anchor - t >= 0.0) & (t_anchor - t <= T_max_days)
        if qc_max is not None and self._qc is not None:
            ok &= np.asarray(self._qc[lo:hi], np.int32) <= int(qc_max)
        if not ok.any():
            return out
        dt_days = t_anchor - t

        dx, dy = offsets_km(lat, lon,
                            np.asarray(self._col["lat"][lo:hi], np.float64),
                            np.asarray(self._col["lon"][lo:hi], np.float64))
        r2 = dx * dx + dy * dy
        ok &= r2 <= R_max_km * R_max_km
        n_R = int(ok.sum())
        out["n_R"] = n_R
        if n_R == 0:
            return out

        d2 = r2 / (L * L) + (dt_days * dt_days) / (T * T)
        d2 = np.where(ok, d2, np.inf)
        # A stable sort, not argpartition: ties must resolve the same way on
        # every machine.
        order = np.argsort(d2, kind="stable")[:min(k, n_R)]
        n = len(order)
        rows = order + lo

        vals = self._gather_values(rows)
        out["n_found"] = n
        out["valid"][:n] = True
        out["row"][:n] = rows
        out["values"][:n] = vals
        out["mask"][:n] = np.isfinite(vals)
        out["dx_km"][:n] = dx[order]
        out["dy_km"][:n] = dy[order]
        out["dt_days"][:n] = dt_days[order]
        out["dist_km"][:n] = np.sqrt(r2[order])
        out["d2"][:n] = d2[order]
        out["lat"][:n] = self._col["lat"][rows]
        out["lon"][:n] = self._col["lon"][rows]
        out["time_days"][:n] = t[order]
        out["time_s"][:n] = (np.asarray(self._col["time_s"][rows], np.int64)
                             if self.schema_version == 2
                             else seconds_of_days(
                                 self._col["time_days"][rows]))
        if self._platform is not None:
            out["platform"][:n] = self._platform[rows]
        if self._qc is not None:
            out["qc"][:n] = self._qc[rows]
        if self._fp is not None:
            fp = np.asarray(self._fp[rows], np.float32)
            out["log2_fp"][:n] = fp[:, 0]
            out["log2_dt"][:n] = fp[:, 1]
        return out

    def _miss(self, k):
        """k miss tokens — the shape every answer has before it is filled."""
        nan1 = np.full(k, np.nan, np.float32)
        # The TIME fields are float64: a schema-2 timestamp is an exact integer
        # number of seconds and float32 would throw that away in the very field
        # a consumer reads to find out when the observation was made (float32
        # resolves 84 s at the end of the record — the whole reason for 10.1).
        # NaN is the miss token, which is why `time_s` is a float64 second
        # count rather than an int64 column with no way to say "nothing here".
        nan64 = np.full(k, np.nan, np.float64)
        return {
            "k": k, "n_found": 0, "n_R": 0,
            "valid": np.zeros(k, bool),
            "row": np.full(k, -1, np.int64),
            "values": np.full((k, self.C), np.nan, np.float32),
            "mask": np.zeros((k, self.C), bool),
            "dx_km": nan1.copy(), "dy_km": nan1.copy(),
            "dt_days": nan64.copy(), "dist_km": nan1.copy(), "d2": nan1.copy(),
            "lat": nan1.copy(), "lon": nan1.copy(),
            "time_days": nan64.copy(), "time_s": nan64.copy(),
            "platform": np.zeros(k, np.int64),
            "qc": np.zeros(k, np.uint8),
            # Constants where the store has no fp column — which is exactly the
            # family-8 case, and is why they are FIELDS rather than metadata.
            "log2_fp": np.full(k, self.log2_fp_const, np.float32),
            "log2_dt": np.full(k, self.log2_dt_const, np.float32),
            "channels": self.channels,
            "levels": self.levels,
        }

    # -- self-description ---------------------------------------------------
    def summary(self):
        live = int((np.diff(self.bin_offsets) > 0).sum())
        return (f"{os.path.basename(self.path)}: N={self.N:,} · C={self.C} "
                f"({', '.join(self.channels[:6])}"
                f"{' …' if self.C > 6 else ''}) · bins {self.bin_first}.."
                f"{self.bin_last} ({live:,} live) · layout {self.layout} · "
                f"schema {self.schema_version} ({self.time_column}) · "
                f"fp ({self.log2_fp_const:g}, {self.log2_dt_const:g})")


def open_store(spec, **kw):
    """Module-level alias, for callers that prefer a function."""
    return Store.open(spec, **kw)


if __name__ == "__main__":
    import sys
    for s in sys.argv[1:]:
        print(Store.open(s).summary())
