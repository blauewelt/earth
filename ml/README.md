# ml/ — Earth-State Embeddings, Phase 1 (North-Atlantic pilot)

The working implementation of the private proposal's pilot (§9): a **masked
autoencoder over one pixel's channels** whose bottleneck embedding must
predict the pixel's own masked channels AND its neighbours in space and time.
If the embedding can do that, it carries *state*, not decoration — and the
held-out physical probe is whether a linear read-out of the 26.5°N section's
embeddings tracks the RAPID overturning transport the model never saw.

## What most accurately describes the AMOC (and what we have)

The overturning is geostrophy plus Ekman: to constrain it you want the
**density structure at depth** (thermal wind → shear), **SSH** (surface
geostrophy), **wind stress** (Ekman transport), **boundary/bottom pressure**
(barotropic compensation), and the **direct array transports** as truth.

| Constraint | Dataset | In repo today | Colab adds |
|---|---|---|---|
| Direct transport (truth/probe) | RAPID 26.5°N | `data/rapid_moc.json` ✔ | OSNAP, MOVE, SAMBA, Florida Current cable |
| T/S at depth (density) | RG-Argo, EN4, GLORYS12 | Argo column *latest month only* | full RG-Argo 2004→ (1°, monthly, 58 levels); EN4 1900→ |
| SSH | GLORYS zos, DUACS altimetry | latest month only | full DUACS SLA 1993→ |
| Surface circulation | GLORYS uo/vo | **speed, monthly 1993→2026** ✔ | u,v components |
| Mixed layer | GLORYS mlotst | **monthly 1993→2026** ✔ | — |
| Wind stress (Ekman) | ERA5 τx/τy | — | monthly means 1940→ |
| SST / SSS | MUR, OISST, SMAP | climatology only (tiles are imagery) | OISST monthly means 1982→; SMAP/SMOS SSS 2010→ |
| Bottom pressure | GRACE ocean OBP | land mascons only | GRACE/GRACE-FO ocean OBP |

The catalog (`data/catalog.json`) flags 58 AMOC-relevant records; the big
not-yet-ingested ones for this model are **EN4**, **full RG-Argo**, **ERA5
winds**, **DUACS altimetry**, and the **non-RAPID arrays** (OSNAP/MOVE/SAMBA)
as additional held-out probes.

## Files

- `build_dataset.py` — assembles `cache/na_pixels.npz` from the repo's baked
  grids: 1° × monthly, 1993-01→now, window 100°W–20°E / 0–70°N. Channels
  today: GLORYS current speed + log-MLD (monthly), OISST SST + GPCP precip
  climatologies (static). Every archive above lands as another channel here.
- `model.py` — `PixelMAE`: channel-tokens → transformer encoder (explicit
  *missing* tokens; absence is information) → bottleneck `z` (default 32-D) →
  a neural-field-style decoder `f(z, channel, Δlon, Δlat, Δmonth)` queried at
  offset 0 (reconstruction), ±1 space and ±1 month (neighbour prediction).
- `train.py` — training with **blocked splits** (whole held-out years + a
  held-out mid-Atlantic longitude block; never random splits — proposal §7),
  then eval: masked-channel skill vs channel-mean, t+1 prediction vs
  persistence, and the **RAPID ridge probe** scored on held-out years only.

## Run

```bash
python3 ml/build_dataset.py          # → ml/cache/na_pixels.npz  (~13 MB)
python3 ml/train.py --smoke          # CPU sanity, ~10 min
python3 ml/train.py                  # full run
```

## TPU/GPU via the Google Colab CLI

Google shipped an official Colab CLI (June 2026) aimed exactly at this:
terminal/agent-driven jobs on Colab GPUs and TPUs (T4/L4/A100/H100, TPU
v5e/v6e).

```bash
pip install google-colab-cli
colab auth                                     # OAuth2 or ADC
colab run --gpu v6e1 ml/train.py               # ephemeral one-shot job
# or a persistent session:
colab new -s amoc --gpu A100
colab upload -s amoc ml/cache/na_pixels.npz ml/cache/na_pixels.npz
colab exec -s amoc -f ml/train.py
colab download -s amoc ml/runs/pilot/pixelmae.pt runs/pixelmae.pt
colab stop -s amoc
```

`colab auth` needs a browser once (user-side); after that the CLI is fully
headless. The Colab runtime is also where the full archives get pulled
(copernicusmarine for GLORYS/DUACS, ERDDAP for RG-Argo, CDS for ERA5) —
they're too big to bake into this repo.

## Next (mirrors the proposal roadmap)

1. Channels: add RG-Argo T/S at 17 levels, ERA5 τ, OISST monthly SST, DUACS
   SLA → C ≈ 45.
2. Rate sweep: replace the linear bottleneck with RVQ, sweep codebooks, plot
   held-out skill vs bits (the dimensionality curve, proposal §6).
3. Probes: OSNAP/MOVE/SAMBA as additional never-seen transports.
