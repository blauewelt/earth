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
- `trainprobe.py` — predictive metrics on FROZEN embeddings, cheap enough
  to run every N training steps (`train.py --eval-every`): linear section
  probe + a fixed-seed mini temporal transformer, protocol v2. Rankings
  live in `LEADERBOARD.md`; backfill any finished run with
  `python3 ml/trainprobe.py --run <name>`.
- `train.py` — training with **blocked splits** (whole held-out years + a
  held-out mid-Atlantic longitude block; never random splits — proposal §7),
  then eval: masked-channel skill vs channel-mean, t+1 prediction vs
  persistence, and the **RAPID ridge probe** scored on held-out years only.

## Run

```bash
python3 ml/build_dataset.py          # → ml/cache/na_pixels.npz  (repo channels)
python3 ml/fetch_rg_channels.py      # + RG-Argo T/S at 10/200/700/1500 dbar
                                     #   (~2 GB of downloads; run on Colab or
                                     #    any machine — needs netCDF4)
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

## Pilot results so far (CPU, in-sandbox)

4-channel smoke (1,500 steps): masked-channel reconstruction beats the
channel-mean baseline on every channel (skill 0.45–0.99), and a ridge probe
from the 26.5°N section's 32-D embeddings correlates r≈0.47 with RAPID
transport on *held-out years* — from current speed + MLD alone, the transport
never seen in training. t+1 prediction does not beat persistence yet at this
scale. The 12-channel run (with the RG depth structure) is the current
experiment; results land in `runs/pilot12/eval.json`.

**Stage 2 (`temporal.py`, 2026-08-05, 4-channel in-sandbox pilot):** a small
causal transformer (d=96, 3 layers, K=24 months) over the frozen anomaly
codec's embedding sequences. Held-out months, protocol v2 throughout:
z-space t+1 MSE 1.53 vs 2.31 persistence (−34%); decoded channel-space t+1
0.90 vs 1.33 persistence (−32%, dynamic channels only) — the temporal stage
roughly doubles the codec's own single-step margin (which is diluted by
static channels; not directly comparable). RAPID probe from the
section-pooled hidden state: r_deseas 0.33 from 96 features, where the
linear K-concatenation probe swings −0.04…+0.48 across K on the same codec
(36 test months; up to 768 features) — compact and stable-by-construction,
not yet a probe win. The probe verdict needs the 12-channel tensor and more
than one seed; the dynamics verdict is already clear.

## Next (mirrors the proposal roadmap)

1. Channels: ERA5 τ (Ekman), OISST monthly SST, DUACS SLA, more RG levels
   → C ≈ 45.
2. Rate sweep: replace the linear bottleneck with RVQ, sweep codebooks, plot
   held-out skill vs bits (the dimensionality curve, proposal §6).
3. Probes: OSNAP/MOVE/SAMBA as additional never-seen transports.
