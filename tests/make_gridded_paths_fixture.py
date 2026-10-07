#!/usr/bin/env python3
"""The NATIVE stores behind E-088's committed monthly-sums fixture, for the
Data tab's two-path test (tests/f1data.test.mjs, "E-088 …").

`tests/make_gridded_monthly_fixture.py` builds two tiny sharded tier-G stores
(fxdaily: daily, two channels, gaps and an absent day across two year
boundaries; fx6h: six-hourly, three "pressure levels") and
`ml/export_gridded_monthly.py fixture` turned them into
data/gridded_monthly/fixture/ (sum, count, m2 — committed) but kept no native
store. This script rebuilds those stores with the same deterministic writer
and commits them beside the sums, so src/f1data.js can answer ONE selection
both ways — from the precomputed sums and from every native frame — and the
test can demand that the two agree:

  data/gridded_monthly/fixture_native/
    registry.json                 a family-1-shaped registry, hf_root
                                  "tensors/fixture", paths tensors/fixture/<name>
                                  (= the sums' `source_store`)
    gridded_monthly_index.json    data/gridded_monthly/fixture/'s index with
                                  every file URL made relative to this folder
    fxdaily/  fx6h/               the native stores (store.json, <group>/ …)

Before writing, the rebuilt stores are exported again and the fresh sum,
count and m2 must be byte-identical to the committed ones — otherwise the
"native" side of the test would not be the store the sums came from.

    python3 tests/make_gridded_paths_fixture.py
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))
import make_family1_fixture as m1                               # noqa: E402
import make_gridded_monthly_fixture as F                        # noqa: E402
import export_gridded_monthly as X                              # noqa: E402

SUMS = os.path.join(ROOT, "data", "gridded_monthly", "fixture")
OUT = os.path.join(ROOT, "data", "gridded_monthly", "fixture_native")
CLASSES = {"fxdaily": F.FxDaily, "fx6h": F.Fx6h}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    tmp = tempfile.mkdtemp(prefix="gm_paths_")
    try:
        stores = F.build(tmp)                    # {"fixture/fxdaily": dir, …}
        # the falsifier of this fixture: the sums of the rebuilt stores are
        # the committed sums, byte for byte
        for key, base in stores.items():
            name = X.store_name(key)
            d = os.path.join(tmp, "_sums_" + name)
            X.export(key, d, base=base, workers=1, threads=1)
            for f in ("sum.npy", "count.npy", "m2.npy"):
                a, b = sha(os.path.join(d, f)), sha(os.path.join(SUMS, name, f))
                if a != b:
                    sys.exit(f"{name}/{f}: the rebuilt store's sums differ from "
                             f"the committed fixture ({a} vs {b})")
        if os.path.isdir(OUT):
            shutil.rmtree(OUT)
        os.makedirs(OUT)
        groups = []
        for key, base in stores.items():
            name = X.store_name(key)
            shutil.copytree(base, os.path.join(OUT, name))
            e = m1.registry_entry(os.path.join(OUT, name), name, CLASSES[name], "G")
            e["path"] = f"tensors/fixture/{name}"
            e["family_version"] = "fixture"
            if name == "fx6h":
                e["levels_hpa"] = list(F.LEVELS)
            groups.append(e)
        reg = {"family": "fixture", "family_version": "fixture",
               "hf_root": "tensors/fixture", "repo": "fixture",
               "epoch": "1982-01-01", "pentad_days": 5,
               "generated_utc": "fixture", "groups": groups,
               "_source": "tests/make_gridded_paths_fixture.py"}
        with open(os.path.join(OUT, "registry.json"), "w") as fh:
            json.dump(reg, fh, indent=1)
        ix = json.load(open(os.path.join(SUMS, "gridded_monthly_index.json")))
        ix["_source"] = ("tests/make_gridded_paths_fixture.py — a copy of "
                         "data/gridded_monthly/fixture/gridded_monthly_index.json "
                         "with every file URL relative to this folder")
        for key, st in ix["stores"].items():
            name = X.store_name(key)
            for f in ("sum", "count", "m2", "stats", "verify"):
                if st.get(f) and st[f].get("url"):
                    st[f]["url"] = f"../fixture/{name}/{os.path.basename(st[f]['url'])}"
        with open(os.path.join(OUT, "gridded_monthly_index.json"), "w") as fh:
            json.dump(ix, fh, indent=1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(OUT) for f in fs)
    print(f"wrote {OUT}: {total:,} bytes")
    if total > 2_000_000:
        sys.exit("the fixture is over 2 MB")


if __name__ == "__main__":
    main()
