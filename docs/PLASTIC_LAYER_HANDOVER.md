# Handover — the Ocean Cleanup plastic layer, then family 1.3

*Written 2026-10-09 by the planning session for a Claude session on another
subscription. Nothing below has been built: no code, no bake, no commit other
than this note. Every data fact was measured by downloading the files on
2026-10-08/09. Facts marked "not verified" were not.*

## 0 · What Chris asked for, in order

1. **First: show The Ocean Cleanup's public survey data as a layer on
   blauewelt.org** (the globe app in this repo). Chris, 2026-10-09: *"Can you
   display the data as a layer on Blauewelt.org first?"*
2. **Then: draft family 1.3.** This is a new data family: family 1.2 (the
   global fine observation stores plus ERA5 upper air) inherited unchanged,
   **plus stores that could identify plastic in the ocean.** Chris,
   2026-10-08: *"Alright, let's draft family 1.3 with this"*. Not started; §4
   has the state of play.

## 1 · Rules you must read first

- **Root `CLAUDE.md`** governs the globe app. For a new layer, the key parts
  are:
  - Part 1 §2: the ten items every layer ships with
  - §3: static snapshots only, with no new browser-facing host
  - §4: sandbox testing with MIRROR and `scripts/run_tests.sh`
  - §4b: the dateless toast
  - §5: a provenance stamp on every value
  - §6: stamping and commits
- **"Deploy first":** commit, deploy, then run the affected tests. Run
  `python3 scripts/stamp_assets.py` before committing anything under `src/`.
- **Never force-push `main`.** Push with `node scripts/git_api_push.mjs
  --branch main` (from `blauewelt/base`, also shipped as the `git-api-push`
  skill). The token is **not** in this repo. Chris supplies it, and it sits in
  the Earth project's doc `claude/github-access.md`. Keep it in a file and
  never put it in argv.
- **Chris's standing rules for replies:**
  - Post links as clickable markdown, one per line, and open each one first.
  - Never ask through multiple-choice widgets. Lay the options out in prose,
    name your pick and proceed.
  - Spell out every code name in one plain-English sentence the first time
    it appears.

## 2 · The data (measured)

**Paper:** Lebreton et al. 2018, *Evidence that the Great Pacific Garbage
Patch is rapidly accumulating plastic*, Scientific Reports 8:4666 —
[nature.com](https://www.nature.com/articles/s41598-018-22939-w)

**Data:** figshare article 5873142, CC BY 4.0 (attribution required) —
[DOI](https://doi.org/10.6084/m9.figshare.5873142).

- The figshare *web page* answers 403 to scripts.
- The API `https://api.figshare.com/v2/articles/5873142/files` and the
  downloads `https://ndownloader.figshare.com/files/<id>` both work keylessly
  from a sandbox with curl.
- Resolve file ids through the API rather than typing them in.

| file id | name | what is in it |
|---|---|---|
| 10441962 | `Lebreton2018_Concentration.xls` | 4 sheets, one per size class: Microplastics 0.05–0.5 cm (501 rows, manta net), Mesoplastics 0.5–5 cm (501, manta net), Macroplastics 5–50 cm (151, "mega" net), Megaplastics > 50 cm (31, aerial RGB mosaic). Per event: midpoint / lower / higher estimates in pieces/km² and g/km² (corrected for wind mixing, i.e. plastic pushed below the net by waves), plus **raw** count and mass (what was actually caught; raw mass is empty for megaplastics) |
| 10441968 | `Lebreton2018_SamplingInformation.xls` | `StationInfo`: 683 events (501 manta, 151 mega, 31 RGB mosaic) — platform (RV Ocean Starr, 17 sailing vessels, "Aircraft C-130 Hercules"), UTC start day/month/year and decimal hour, start/end lat/lon, distance, width, area (km²), wind (kn), Beaufort sea state. `MosaicDebrisInfo`: **1,595 individual objects seen from the aircraft** — position, object type (container, buoy, net, unknown …), colour, length/width (m), "< 50 cm?", top-view area, weight low/mid/high (kg). Other sheets not needed |
| 10441965 | `Lebreton2018_HistoricalDataset.xls` | 3,532 earlier net tows **1972–2015**, Pacific-wide (lat −17…57): origin cruise, reference, year, month, lon, lat, gear (Neuston/neuston, Manta/manta, ovoid plankton, seston, ring net — normalise case), mesh (µm; mostly 335 or 500), tow depth, microplastic count (#/km²) and mass (g/km²), inside-the-patch flag. **Zeros are real measurements** (nothing caught) |
| 44299727, 44299730 | `GPGP_contours.zip`, `MassConcentrationAllSizes.tif` | the paper's modelled patch outline and mass map (201 × 126 grid) — derived, not observations; optional later |

**Dates (from StationInfo):**
- manta tows: 2015-07-25 → 2015-09-20
- mega tows: 2015-07-25 → 2015-08-18
- aircraft: 2016-10-02 → 2016-10-06

**Two traps, both seen in the files:**
- **The Megaplastics sheet has latitude and longitude swapped.** Its
  "Longitude" column holds about 30–34 and its "Latitude" column about −143…−135.
  - Take every position from `StationInfo`.
  - The bake should **assert** that each concentration row matches
    StationInfo within 0.01°, allowing the swap for that one sheet only, and
    refuse otherwise.
- **Event IDs look like decimals** (0, 0.1, 1.1, 2.1 …). A naive pandas join
  matched only ~93% of the Microplastics sheet to StationInfo, almost
  certainly because of float/text formatting. Normalise the IDs and **assert
  a 100% join** for all four sheets and for MosaicDebrisInfo.

**What is not public:** the raw aerial imagery, and the plane's lidar and
shortwave-infrared data. There is also the 2017 visual-observer technical
report — [The Ocean Cleanup report page](https://theoceancleanup.com/scientific-publications/quantification-of-medium-and-large-debris-in-the-great-pacific-garbage-patch-using-visual-observations-from-aerial-surveys).
No repository was found for any of them.

## 3 · The layer — design decisions already taken

### Bake

`python3 scripts/refresh_data.py gpgp` writes `data/gpgp_plastic.json`. Aim
well under 1.5 MB: coordinates to 1e-4°, values to 4 significant figures. It
contains:
- `events`: 683 entries. Each has id, type, platform, ISO UTC start, position
  (midpoint of start and end), area, distance, wind, Beaufort, and per size
  class the mid/lo/hi count and mass plus raw count and mass.
- `objects`: 1,595 entries, each with the event's date.
- `historical`: 3,532 entries.
- The source, licence, attribution and DOI.
- The counts and periods of each part, derived from the files.

The bake prints and asserts the counts, the joins, the swap check and the
coordinate ranges. It needs `xlrd` and `pandas`.

### Layer

- **Placement and title.** One row in the ocean group. Title: *"Floating
  plastic, North Pacific — net tows and aircraft survey (The Ocean Cleanup),
  points"*, linked to the paper. §2.1 requires the title's size to match the
  `sp` fact, so copy how the existing point layers pass that test.
- **Picker, default first:**
  1. Microplastics 0.05–0.5 cm — fine net, 2015 (501)
  2. Mesoplastics 0.5–5 cm — fine net, 2015 (501)
  3. Macroplastics 5–50 cm — large net, 2015 (151)
  4. Megaplastics > 50 cm — aircraft, 2016 (31)
  5. **Objects seen from the aircraft, 2016** (1,595): categorical colour by
     object type, dot size by length, and a swatch legend like the
     classification layers.
  6. Microplastics, earlier surveys 1972–2015 (compiled) (3,532)
- **Second control:** pieces per km² (default) or grams per km².
- **Colour:** a log ramp on the mixing-corrected midpoint, with a legend
  hover read-out. A measured zero is a small hollow grey dot reading "none
  caught", never the ramp's minimum colour.
- **Read-out for one dot**, every value stamped with its date (day for
  2015/16, month for the historical tows):
  - plain-English gear: fine-mesh manta net towed beside the ship, large
    "mega" net, or photo mosaic from a C-130 at about 400 m
  - platform and UTC time
  - the midpoint with its low–high range, the raw value, and one line
    explaining wind mixing
  - the other quantity, the area sampled, wind and sea state
  - for objects: type, colour, size, weight range and the < 50 cm flag
  - for historical tows: cruise, reference, gear, mesh, depth, and
    inside-the-patch
- **Hover card:**
  - A gist in plain English: an 18-vessel net expedition in summer 2015 and
    two C-130 flights in October 2016 across the Great Pacific Garbage Patch
    (the accumulation zone between Hawaii and California). Nets catch pieces
    down to 0.5 mm, and the photo mosaics count objects over 50 cm.
  - The paper's headline numbers. The planning session remembers them as at
    least 79,000 tonnes, with most of the mass in pieces over 5 cm.
    **Not verified — check against the abstract before writing them.**
  - Recorded: a closed set of surveys with the dates above, plus the
    1972–2015 compilation. Not updated.
  - Interval: a one-off campaign.
  - Spatial: one point per sampling event or per object.
- **Dateless toast (§4b):** a fixed 2015–16 set; the date selector doesn't
  change it, and each dot says when it was sampled.
- **Posture matrix (§2.5):** neither aggregate nor difference, because these
  are points.
- **Housekeeping:**
  - Register the chip in `STATIC_LAYER_CHIPS` if the layer is hand-written.
  - Add a catalog record with `globe: true`, and update the README counts.
  - Attribution "Data: Lebreton et al. 2018 / The Ocean Cleanup, CC BY 4.0"
    goes in the credit, the hover card and the footer, the way GFW's is done.
- **Build pattern:** use an existing point layer as the template, such as
  loitering, Climate TRACE or Argo, with a `PointPrimitiveCollection`
  rebuilt only on a picker change. The browser reads only the baked JSON.

### Tests

- **`tests/data.spec.js`:**
  - the five counts
  - every megaplastic point inside lat 25–40 / lon −160…−125, which catches
    the swap
  - attribution present
  - min/max checks only, never one `expect()` per point
- **`tests/app.spec.js`:**
  - the layer draws 501 points by default
  - the objects option draws 1,595 and shows swatches
  - the dateless toast fires
  - a pick on a known event shows its date and value
- **Docs:** update root `CLAUDE.md` Part 3 (a record paragraph that includes
  the swap quirk) and the posture matrix.

Local copies of the four files from 2026-10-09 were in the old session's
scratchpad, which is gone. Re-download them through the API.

## 4 · Family 1.3 — state of play (nothing written yet)

- **What it is meant to be:** family 1.2 inherited by reference, plus
  plastic-relevant stores.
  - Family 1.2's registry is `tensors/family1_2/family12.json` on the
    Hugging Face dataset `chfrank/earth-tensors`.
  - It is described in `docs/FAMILY12_HANDOVER.md` and
    `ml/plans/E085_family12_atmosphere.md`.
- **Finding from 2026-10-08: no existing family 1.2 channel can identify
  plastic.**
  - `oc4k` and `pace4k` are 4 km biology products: chlorophyll, carbon,
    apparent visible wavelength, plankton cell counts. They hold no
    reflectance bands, and a 4 km pixel is far too coarse.
  - SWOT's radar backscatter `sig0_karin` at 2 km could at most give an
    unvalidated "calm-patch" proxy.
- **Proposed additions, in Chris's order:**
  1. **An in-situ microplastic point store.** Candidates, **not verified
     yet:** NOAA NCEI Marine Microplastics, Japan's Atlas of Ocean
     Microplastics (AOMI), and EMODnet floating micro-litter, plus the
     figshare data above. Watch for unit mismatches (per m³ vs per km²),
     different mesh sizes, wind-mixing correction, and duplication between
     compilations.
  2. **Sentinel-2 at 10–20 m in the coastal band**, using a floating-debris
     index from near-infrared, red and shortwave-infrared bands. This
     belongs with family 1.tf's harmonised Landsat/Sentinel-2 catalogue.
  3. **PACE Level-2 reflectance at its native 1.2 km**, including the
     shortwave-infrared bands. Research-grade.
- **The planning session's pick:** item 1 first, because it is cheap and it
  is the ground truth the other two need.
- **Form Chris expects**, matching the family 1.gf and 1.2 notes:
  - an `ml/plans/E0xx_…md` plan (the next free experiment number was E-092
    when this was written — check `docs.html`'s DOCS list and `ml/plans/`
    first)
  - a LaTeX design note in `ml/paper/notes/`
  - a 1–2 page summary with a big table of every source's time and space
    granularity and a plain-English glossary
  - the plan registered in `docs.html` (the docs test fails otherwise)
- **Family rule:** raw observations first, and derived products only where
  they carry observations nothing else does. A dataset whose licence forbids
  redistribution still goes on the private track; it is never republished.
