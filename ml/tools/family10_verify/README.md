# family10_verify — measuring a tier-P observation store's search geometry

Two scripts behind `docs/FAMILY10_VERIFICATION_2026-09-14.md`, the independent
check of the published family-10 point stores (drifters, moorings, ship CO₂).
They need only `numpy` and this repository's reader, `ml/family10_store.py`.

```sh
python3 ml/tools/family10_verify/measure_knn.py --store gdp \
    --dir chfrank/earth-tensors:tensors/family10/gdp --json /tmp/gdp.json
python3 ml/tools/family10_verify/platform_diversity.py --store gdp \
    --dir chfrank/earth-tensors:tensors/family10/gdp
```

`measure_knn.py` reports how far away and how old the k-th nearest observation
is, at the store's own rows and at uniform globe anchors — the numbers that
replaced §6's guessed search radii. `platform_diversity.py` reports how many
distinct instruments the k tokens come from, which is how the "k tokens are one
platform" finding was measured. `--dir` takes a local store directory or a Hub
prefix; `--store` selects the suggested (k, R, T) the ranking is pinned to.
