# E-092 · Family 1.3 — family 1.2 plus the stores that could identify plastic in the ocean

*Written 2026-10-11, BEFORE any adapter, probe or build. Chris asked
(2026-10-08) to draft a new family: family 1.2 unchanged, plus stores that
could identify plastic in the ocean, and (2026-10-09) to show The Ocean
Cleanup's survey on blauewelt.org first, which shipped that day as the
"Floating plastic, North Pacific" layer. This plan records what every
candidate source actually serves (each was measured from the sandbox on
2026-10-11, not transcribed from its documentation), the stores proposed, the
order to build them in, and the test each must pass to be kept. Nothing has
been dispatched, rented or uploaded.*

The design note is
[`ml/paper/notes/family13.pdf`](https://github.com/blauewelt/earth/blob/main/ml/paper/notes/family13.pdf)
and the two-page summary with the glossary is
[`ml/paper/notes/family13_summary.pdf`](https://github.com/blauewelt/earth/blob/main/ml/paper/notes/family13_summary.pdf).

## 0 · In plain English

- **Family 1.2** is the global observation stores at 10 km / 5 days or finer
  (family 1.gf: ocean colour, satellite sea-surface temperature, ship
  reports, floats, bottles, moorings, the SWOT swaths …) plus ERA5's
  upper-air temperature, humidity and wind (E-085). It is published on the
  Hugging Face dataset `chfrank/earth-tensors`.
- **Family 1.3** is family 1.2 **inherited by reference** (no byte copied),
  plus a small number of new stores whose job is to say *where floating
  plastic is, and how much*.
- **Why it is a new family and not a new store in 1.2.** Plastic is not one
  of the programme's four prediction goals (El Niño, the ocean carbon sink,
  the Atlantic overturning circulation, sea-surface temperature). Family 1.3
  adds a **fifth read-out** — floating plastic concentration at a place and a
  time — and the stores that measure it. Keeping it a separate family keeps
  1.2's registry and its verdicts untouched.
- **The finding that shapes everything: plastic in the open ocean is
  measured almost only by nets, not by satellites.** A net towed at the
  surface catches pieces down to a third of a millimetre; a satellite pixel
  sees a 10 m or 1 km square of sea, and the densest open-ocean microplastic
  covers about 0.005 % of the surface (Hu 2021), far below what any optical
  sensor can see. Satellites see plastic only where it is piled up — windrows,
  river plumes, harbours — and even then mixed with seaweed and foam. So the
  first store is the nets, and the satellite stores are tested against them
  before anything large is built.

## 1 · What the family already holds that bears on plastic

Measured from the registries and adapters, 2026-10-11:

| store (inherited) | what it holds | why it matters for plastic |
|---|---|---|
| `oc4k` (1.gf) | daily 4 km ocean colour: log-chlorophyll, light attenuation `kd_490`, observation count, 1997-09 → 2022-12 | turbid or bloom water is what a debris index confuses with plastic |
| `pace4k` (1.gf) | daily 4 km PACE: log-chlorophyll, organic and phytoplankton carbon, apparent visible wavelength, three plankton cell counts, 2024-03 → | the same confusers; **no reflectance and no shortwave-infrared band** |
| `swot` (1.gf) | 2 km swath sea-surface height, radar backscatter `sig0_karin` (linear), uncertainty | currents converge where plastic gathers; backscatter dims over surface films (no published plastic use found) |
| `swh` (1.gf) | wave height along altimeter tracks | waves mix light plastic below a surface net — the wind-mixing correction |
| `icoads` (1.gf), `era5_u`/`era5_v` (1.2) | ship winds; upper-air wind on 13 pressure levels | wind drives both the drift and the mixing |

**No inherited channel can identify plastic.** Ocean colour at 4 km carries
no reflectance bands, and no family-1 store holds a shortwave-infrared band
over water. This is the finding of 2026-10-08, confirmed against the adapters
(`ml/family1/adapters/pace4k.py`, `oc4k.py`, `swot.py`).

## 2 · The sources, measured 2026-10-11

### 2a · Net tows and water samples (the ground truth)

| source | access (HTTP from the sandbox) | licence | rows | span | units as published |
|---|---|---|---|---|---|
| **NOAA NCEI Global Marine Microplastics** (the US national archive's harmonised compilation) | ArcGIS FeatureServer `services2.arcgis.com/C8EMgrsFcRFL6LrL/…/Hub_Microplastics_Replace/FeatureServer/0`, and the Hub CSV export of item `1b56a456ad8c43889652e62aa25a317e` — 200, identical; the product page names neither | CC0 1.0 (accession 0306475; citation requested, DOI 10.25921/v8gj-k896) | 29,076, of which **16,988 ocean-water samples** (893 beach, 580 sediment, 10,615 citizen nurdle counts) | water 1972-04-20 → 2024-10-23 | water and beach in pieces/m³; sediment pieces/kg; nurdles pieces/10 min |
| **The Ocean Cleanup, Lebreton et al. 2018** (figshare 5873142) | figshare API + ndownloader — 200 | CC BY 4.0 | 683 events + 1,595 aircraft objects + 3,532 compiled tows | 2015-07 → 2016-10 (compilation 1972–2015) | pieces/km² and g/km², four size classes, wind-corrected AND raw |
| **The Ocean Cleanup, Lebreton et al. 2024** (figshare 25674120) | figshare — 200 | CC BY 4.0 | 917 manta tows (510 are the 2015 expedition again, **407 new, 2018–2022**), 162 large-net tows, 74 aerial surveys, 40 cleanup-system catches | 2015 → 2022 | pieces/km² and g/km² over 7 size classes |
| **EMODnet Chemistry floating micro-litter** v2026 (the EU's marine data network) | `nodc.ogs.it/repository/aggregated-datasets/litter/EMODnet_floating_microlitter_2026_unrestricted.zip` — 200 | CC BY 4.0 (in every record) | ≈ 3,368 samples (32,067 lines: sample × litter type × size class) | 2011-06-14 → 2025-08-30; mostly European seas | **raw counts per net** with mouth width, track length (21,497 lines) or filtered volume (10,588) — normalised by the builder |
| **Isobe et al. 2021** (the multilevel dataset behind Japan's Atlas of Ocean Microplastics) | Springer supplement zip — 200 | article CC BY 4.0; data licence to confirm | 8,218 samples, 27 projects | 2000 → 2019 | pieces/m³ (level 0/1); pieces/km² wind-corrected (level 2) |
| **AOMI** (Japan's Atlas of Ocean Microplastics) | `aomi.env.go.jp/arcgis/rest/services/opendata/MapServer/1000` — 200 | **non-commercial only** (site data policy); per-record CC licences, 6,900 rows with none | 18,860 raw samples | 1972 → 2026-01 | pieces/m³ and pieces/km² by size class |
| **Eriksen et al. 2023** compilation ("plastic smog") | raw GitHub CSV of the authors' repository — 200 | **none in the repository**; the paper is CC BY 4.0 | 12,546 | 1971 → 2019 | pieces/km², g/km² |

**Three facts about these sources decide the design.**

1. **They are mostly the SAME tows, compiled again and again.** The Sea
   Education Association's (SEA's) cruises, The Ocean Cleanup's 2015
   expedition and Eriksen's 5 Gyres tows sit in NCEI, AOMI and Eriksen 2023
   alike, and the SEA Pacific tows are in AOMI twice (2,527 submitted by SEA,
   2,130 imported via NOAA). 12,160 of AOMI's 18,860 match NCEI on date and
   position; 7,758 of Eriksen 2023's 12,546 do; 644 of the 683 Ocean Cleanup
   2015–16 events have an NCEI row on the same day and place. A store that
   concatenated the compilations would count the same tow three times.
2. **"pieces/m³" for a surface net tow is not a measurement.** A neuston or
   manta net skims the top few tens of centimetres and measures pieces per
   AREA; NCEI converts to per VOLUME by dividing by a net depth the compiler
   assumed, and the assumption differs by study. Matched against The Ocean
   Cleanup's own pieces/km², the implied depths are 0.25 m for Law 2014,
   0.08 m for Eriksen 2014, 0.155–0.25 m for Goldstein, 0.075 m for The Ocean
   Cleanup's 2015 tows; Isobe 2021 uses 0.5 m for the same SEA tows. Values
   from different studies are not comparable until that division is undone.
3. **Zeros are real and common.** 4,958 of NCEI's 16,988 water samples, 5,566
   of AOMI's, 1,506 of Lebreton's compiled tows: the net came up empty. A
   zero is an observation and is stored as one.

**The SEA conflict — found while checking, SETTLED 2026-10-11 (§5 step 0).**
SEA's 2,529 Pacific tows (Law et al. 2014) exist in two published copies —
NCEI accession 0211008 (also carried by NCEI's harmonised table as 2,530 rows
of "Law et al. 2014") and the copy inside Lebreton 2018's compilation. They
agree on every year-month count and on the first ~46 rows, then diverge: only
399 positions match. Both are coherent ship tracks (median step 92 km and
100 km), so the tie was broken with two further copies, measured:

| witness | rows compared | matches Lebreton's copy (year, month, position) | matches NCEI 0211008 |
|---|---|---|---|
| **SEA's own submission to AOMI** (organisation "Sea Education Association", 2001-10-12 → 2012-12-20, west of 100° W) | 2,485 | **2,485** | 355 |
| **Eriksen et al. 2023** (10,170 SEA rows with ship, cruise and station numbers) | 2,487 | **2,485** | 357 |

**NCEI accession 0211008's positions are wrong; Lebreton's copy — the one the
globe's "earlier surveys" option draws — is right.** In September 2005, for
example, NCEI puts the tows off Hawaii (15° N, 160° W) where SEA's own record
and both compilations put them off San Diego (32.8° N, 117.5° W). Only the
Pacific accession is affected: NCEI's 6,135 SEA Atlantic tows (Law et al.
2010) match SEA's own AOMI submission at 5,808 of 5,851 positions. (The
remaining Lebreton rows lie east of 100° W, in the eastern tropical Pacific,
outside the longitude cut the comparison used — not missing.)

**A second, separate question about the VALUES remains open.** Where the
positions agree, the published Law 2014 value (Lebreton's copy; Eriksen 2023
agrees on 2,279 rows) equals SEA's own AOMI value (pieces/km², < 5 mm) for
1,440 of 2,487 tows and is within 5 % for 20 more; for the other 1,027 SEA's
resubmitted value is larger, typically about three times (median ratio 3.0;
10th–90th percentile 1.02–5.85). That looks like a later reprocessing by SEA
(a different tow area or size cut), not a transcription error, and nothing in
the files says which version SEA now stands by. **D9** (§3) records the
choice.

### 2b · Satellites

| source | what it is | access | licence | the measured fact that matters |
|---|---|---|---|---|
| **Sentinel-2 MSI** | 10–20 m optical imager, 13 bands incl. red-edge B06 (740 nm), NIR B08 (842 nm), shortwave-infrared B11 (1,610 nm), B12 (2,190 nm) | L2A COGs on AWS (`sentinel-cogs`): anonymous range read **206**, CORS `*`; L1C as JP2 in `sentinel-s2-l1c`, anonymous 206 from here (documented as requester-pays — recheck outside the proxy) | Copernicus free, full and open; notice "Contains modified Copernicus Sentinel data [year]" | **It does not image the open ocean.** ESA's plan covers land, coastal waters to 20 km, islands, the Mediterranean and closed seas; measured offshore reach is ~250 km where a swath clips land (off Portugal: items at 65–250 km, none at 350 km). The Great Pacific Garbage Patch box (30–35° N, 135–145° W) has **0 items in 2023 and in every year except one special campaign: 2,602 L2A items, 61 dates, 2021-07-02 → 2021-09-19** |
| … its L2A over water | Sen2Cor's atmospheric correction is tuned for land | measured on one clear tile 100+ km off Portugal | — | **11 % of B08, 27 % of B11, 8 % of B06 pixels pinned at the lowest value** — the bands the debris index differences. The debris papers start from L1C and correct with ACOLITE (an open-source water-tuned atmospheric correction) |
| **MARIDA** (Marine Debris Archive, Kikaki et al. 2022) | human-labelled Sentinel-2 pixels in 15 classes | Zenodo 5151941 — 200, 1.16 GB | CC BY 4.0 | 1,381 patches of 256², 63 scenes, 2015-11 → 2021-01, 11 countries; **837,377 labelled pixels, 3,399 of them "marine debris"** (373 patches) |
| **MADOS** (Kikaki et al. 2024) | the same kind, adds oil spills | Zenodo 10664073 — 200, 4.04 GB | CC BY 4.0 | 174 scenes, 2,803 crops of 240², 2015–2022 |
| **Plastic Litter Project** (Topouzelis, University of the Aegean) | KNOWN plastic targets moored in the Gulf of Gera, Lesvos, imaged by Sentinel-2 and drones | Zenodo 3752719 (2019), 7085112 (2021, 159.5 GB), 10046182 (2022–23) — 200 | CC BY 4.0 | 2021: 22 acquisition dates, 2021-06-11 → 2021-10-04; detection of ≥ 25 % pixel cover shown (Topouzelis 2020) |
| **FloatingObjects** (Mifdal et al. 2021) | line annotations of floating objects on 26 scenes | Google Drive zip 16.6 GB — 206 | Apache-2.0 | 2,270 features, 2018–2020 |
| **PACE OCI** level 2 | the 2024 hyperspectral ocean-colour instrument at ~1 km | CMR (NASA's catalogue) anonymous; downloads need an Earthdata login (302 → 401); the anonymous OB.DAAC OPeNDAP server holds only level 3 and 4 | NASA open | water-leaving reflectance stops at 719 nm; the only above-level-1 shortwave-infrared is the **surface reflectance product SFREFL, `rhos` at 1038, 1249, 1618, 2131, 2258 nm**; **no band sits on plastic's 1,732 nm absorption**; ~89 GB a day at level 2; anonymous only as level-3 2 km / 4 km daily maps |
| **CYGNSS microplastic L3** (Evans & Ruf 2021) | an empirical index from GPS-reflection radar roughness, fitted to a plastic MODEL | PO.DAAC, Earthdata login | NASA open | 0.25° daily files, but the authors' feature resolution is 1° and 30 days; ±~38° latitude; confounded by natural surfactants and the tropical rain band |

**One window where satellite and nets meet in the open gyre — narrower than
it first looked.** The Ocean Cleanup's North Pacific missions of 2021 (in
Lebreton 2024) flew 22 aerial surveys (34.8–35.4° N, 144.4–145.9° W,
2021-08-05 → 08-31) and towed nets inside the area and the weeks of the only
Sentinel-2 campaign over the patch: a 0.7° box around the surveys has 167
L2A items on 20 dates in August–September 2021. **But no acquisition on a
survey's own day covers its point** (the nearest same-day swath misses by
19–115 km, measured by the co-location probe); within ±1 day there are 10
(point, acquisition) pairs. §5 step 2 is what came of them.

## 3 · Decisions — the planning session's picks, each reversible

- **D1 — the first store is the nets.** `mplastic` (below) is built first;
  every satellite store is gated on a test against it or against labelled
  pixels.
- **D2 — one producer per cruise, the original producer first.** Precedence:
  The Ocean Cleanup (its own cruises, its own size classes) → NCEI (each
  study's own accession where one exists, else the harmonised table) →
  EMODnet (European cruises) → Isobe 2021 (rows not in the above, once its
  data licence is confirmed). A row from a later source is dropped when an
  earlier one holds the same day within 0.01° (and kept with its drop counted
  in the ledger, never silently). Compilations are sources of rows nobody
  else published, not of copies.
- **D3 — store what the net measured.** For surface net tows the stored
  quantity is pieces/km² (and g/km² where weighed) **as caught**; NCEI's
  pieces/m³ is multiplied back by the depth the compiler assumed, recorded per
  study in a committed table with the match that measured it. Pumps and grab
  samples are volumetric and stay in pieces/m³, in their own channel. The
  producer's wind-mixing correction is kept beside the raw value, never in
  place of it.
- **D4 — AOMI stays out of phase A** (non-commercial, mixed per-record
  licences, 64 % already in NCEI, 79 rows with latitude and longitude
  swapped). The probe measures what it would add; if it is worth it, it goes
  on the PRIVATE track.
- **D5 — Eriksen 2023 stays out until its licence is clear** (no licence in
  the repository). It is the third copy that can settle the SEA conflict, and
  that use needs no redistribution.
- **D6 — no satellite debris store is built at scale in this family's first
  pass.** Sentinel-2 cannot see the open ocean, PACE has no band on plastic's
  absorption, and CYGNSS's index is fitted to a model. What IS built is
  cheap: the labelled pixels (`s2labels`) and one probe at the 2021
  co-location window. A coastal Sentinel-2 debris store is a phase-C decision
  with a cost estimate, taken on the probe's result.
- **D7 — CYGNSS's microplastic index is rejected** under the family's rule
  (raw observations first; a derived product only where it carries
  observations nothing else does): it is a roughness anomaly fitted to a
  plastic model, at 1° and 30 days. The roughness itself (mean-square slope)
  is a phase-C candidate.
- **D8 — family code `13`, version `1.3`, root `tensors/family1_3/`, registry
  `tensors/family1_3/family13.json`.** It inherits 1.2, which inherits 1.gf:
  the registry's `inherits_block` lists the RESOLVED chain (1.gf's 13 stores
  under their own paths and 1.2's four), so a reader opens one parent list,
  not a chain. `ml/build_family1_registry.py` needs that small extension
  (today it resolves one parent).

- **D9 — SEA's Pacific tows come from Lebreton 2018's copy, with the
  published Law et al. 2014 values** (settled 2026-10-11, §2a). NCEI's
  accession 0211008 and its 2,530 "Law et al. 2014" rows are DROPPED by name
  (their positions are wrong); the Atlantic SEA rows (Law et al. 2010) stay
  with NCEI. The ledger records, per SEA Pacific tow, whether SEA's later AOMI
  value differs, so the choice is reversible without a rebuild of anything
  else. Whether SEA's resubmitted values supersede the published ones is a
  question for SEA.

- **D13 — phase C: all four stores are built (Chris, 2026-10-13: "can we
  add all 4? (the more data, the better)").** Each was measured first
  (`ml/family1/probes/{beachlitter,cygnss_mss,pace_swir,s2fdi_coast}_measure.json`);
  the designs below are what the measurements allow, not what the plan first
  hoped:
  - **`beachlitter`** (tier P, one row per survey): NCEI's Nurdle Patrol
    pellet counts (CC0; 10,615 rows, raw count recomputed — the published
    standardisation is inconsistent on 3,334 rows), NCEI's beach-sand
    microplastics (893 rows, pieces/m³, their own channel), EMODnet beach
    litter v2025 — **only surveys under CC BY 4.0 ("CB", 4,444 of 16,861)**;
    the 8,130 under the SeaDataNet licence (no redistribution without the
    data centre's consent) and the 4,287 "UN" go to the private track, the
    mplastic precedent — and NOAA MDMAP (8,771 surveys, 2009 → 2026, a NOAA
    programme on a NOAA portal; no licence file is posted, so it is taken as
    US-government public data and recorded as unconfirmed). EMODnet's
    material per litter code comes from a committed classification table;
    freshwater shores (Great Lakes, rivers) are kept and flagged.
  - **`cygnss_mss`** (tier G, the producer's native 0.2° grid, daily): the
    mean-square slope of the sea surface from CYGNSS L3 V3.2, its
    uncertainty, the sample count and CYGNSS's own wind (kept to study the
    confound, never as an independent wind), 2018-08 → present, ±40°. Read
    on hosted runners only (Earthdata). Not per-sample: L2 rows would cost
    30–60× the store for a question asked at 1° and 30 days.
  - **`pace_swir`** (tier G, pace4k's 4 km grid, daily): PACE OCI level-3
    surface reflectance `rhos` at 645, 740, 865, 1038, 1249, 1618, 2131 and
    2258 nm — the near/shortwave-infrared bands plus the two baselines a
    PACE floating-debris or floating-algae index needs — all surfaces, not
    ocean only (coasts are where floating debris shows), 2024-03-05 →
    present, anonymous OPeNDAP server-side subsets (≈ 3 min a day).
  - **`s2fdi_coast`** (tier G-fine, per acquisition, 20 m) — **a PILOT, not
    the coast**: the 63 MARIDA scenes and every acquisition under 20 % cloud
    of the Plastic Litter Project's tile (Gulf of Gera, Lesvos, 35SMD)
    2019–2023 — about 310 acquisitions, ~32 runner-hours, ~190 GB read,
    ~44 GB stored. L1C corrected by ACOLITE exactly as in the co-location
    probe; L2A is not usable over water (15–17 % of water pixels pinned at
    the floor, measured on a Mediterranean tile). The Mediterranean
    (≈ 2,500 runner-hours and 14 TB read per year) and the global coast
    (≈ 47,000 runner-hours, 282 TB read per year) are priced, not built: a
    scale-up waits on the pilot's falsifiers (§6).

## 4 · The stores

| store | tier | what one row / frame is | channels | rows / size | phase |
|---|---|---|---|---|---|
| **`mplastic`** | P (points) | one surface sample: a net tow, a pump or grab sample, or an aircraft survey line | `n_km2` (pieces/km², as caught), `g_km2`, `n_m3` (volumetric samples only), `n_km2_wc` (producer's wind-mixing-corrected value, where given), four size-class counts where the producer resolves them (0.05–0.5, 0.5–5, 5–50, > 50 cm), `mesh_um`, `depth_m` (net depth, assumed or measured), `wind_ms` at sampling; `qc` = gear class (manta, neuston, large net, pump, grab, aircraft); `platform` = cruise | ≈ 20,000 rows after de-duplication (NCEI ≈ 17 k water samples, of which ≈ 0.6 k are replaced by The Ocean Cleanup's own ≈ 1.2 k rows, + EMODnet ≈ 3.4 k − ≈ 0.6 k already in NCEI), < 5 MB | A |
| **`mobjects`** | P | one object identified in an aircraft photo mosaic | length, width, top-view area, weight low/mid/high; `qc` = object type (net, rope, buoy, container, unknown …) | 1,595 (2016) + the 2021–22 surveys if Lebreton 2024 resolves objects; < 1 MB | A |
| **`s2labels`** | P | one human-labelled Sentinel-2 pixel | the 11 ACOLITE-corrected band reflectances the label set ships; `qc` = class (marine debris, Sargassum, foam, ship, clear water, turbid water, …), confidence | MARIDA 837,377 (measured) + MADOS ≈ 1.5 M (its project page) + the Plastic Litter Project's target pixels (to count) — ≈ 2.4 M rows, ≈ 130 MB at 53 bytes a row | A |
| `s2fdi_gpgp` | G-fine | one 20 m Sentinel-2 frame over the patch, July–September 2021 | FDI, FAI, NDVI, PI and the four bands they use, from L1C via ACOLITE | only if step 2 passes; the campaign is 2,602 tiles × 5,490² pixels × 8 float16 channels ≈ 1.25 TB before any cut to water and cloud-free | B |
| `s2fdi_coast` | G-fine | the same over the water of the coastal band and enclosed seas | as above | a cost estimate, not a build | C |
| `pace_swir` | G | PACE `rhos` at 865, 1038, 1249, 1618, 2131, 2258 nm, daily 4 km (anonymous level 3) or 1.2 km (level 2, Earthdata) | 6 reflectances | ≈ 2 TB/yr at level 2 | C |
| `cygnss_mss` | P or G | GPS-reflection roughness, the observable under the CYGNSS index | mean-square slope, wind | to measure | C |
| `beachlitter` | P | shoreline litter surveys (NCEI beach and nurdle counts, EMODnet beach litter, NOAA MDMAP) | counts per transect or per 10 minutes | ≈ 12 k NCEI rows + others | C |

**The indices**, as their papers define them (Sentinel-2 band numbers):

- **FDI**, floating debris index (Biermann et al. 2020): FDI = R₈ − R′₈,
  R′₈ = R₆ + (R₁₁ − R₆) · (λ₈ − λ₄)/(λ₁₁ − λ₄) · 10, with the centre
  wavelengths of each satellite (the factor is 1.7722 for S2A, 1.7770 for S2B).
- **FAI**, floating algae index (Hu 2009): R₈ − [R₄ + (R₁₁ − R₄)(λ₈ − λ₄)/(λ₁₁ − λ₄)].
- **NDVI** = (R₈ − R₄)/(R₈ + R₄); **PI**, plastic index (Themistocleous et al.
  2020) = R₈/(R₈ + R₄).

**Footprints.** A manta tow sweeps ~0.9 m × 1–15 km: its token's area is the
swept area (≈ 0.001–0.02 km²) and its length the tow line; a pump sample is
a point (`log2_fp` −4, the family's point label); a Sentinel-2 pixel is 20 m.

## 5 · The ladder

0. **Settle the SEA conflict** (§2a) — **DONE 2026-10-11.** SEA's own AOMI
   submission and Eriksen 2023 both agree with Lebreton's copy (2,485 of
   2,485 and 2,485 of 2,487 positions) and not with NCEI's accession 0211008
   (355 and 357). The globe layer keeps Lebreton's copy; `mplastic` drops
   NCEI's SEA Pacific rows (D9). Left open: which VALUES SEA now stands by.
1. **Phase A, on a hosted runner, $0**: adapters `mplastic`, `mobjects`,
   `s2labels` against the family-10 point-store builder; a probe that runs
   the whole of each (they are small enough that the probe IS the build); the
   per-study depth table committed with the match that measured each depth;
   the de-duplication ledger (rows read, rows dropped as copies per pair of
   sources, rows kept). Publish to `tensors/family1_3/`, registry
   `family13.json` inheriting the resolved 1.2 chain.
2. **The co-location probe — one hour, $0.** For each 2021 aerial survey and
   manta tow in the patch with a Sentinel-2 acquisition within ±1 day, fetch
   the L1C tiles over the survey line, correct with ACOLITE, compute FDI, and
   compare FDI's distribution under the line (and its upper percentiles)
   with the same tile's open water away from it, and with the survey's
   megaplastic count. **Expected: nothing above noise** — the densest patch
   holds at most ~13 objects over 50 cm per km² (Lebreton 2018's highest mosaic), of the order of 0.001 % of the surface. A clear signal would be a
   finding worth a paragraph; no signal closes `s2fdi_gpgp` for good, with a
   measured reason.
   **RESULT (2026-10-11): no signal. `s2fdi_gpgp` is closed.**
   `ml/family1/colocation_probe.py`, result and figure in
   `ml/family1/probes/colocation_gpgp_2021.{json,png}`. Nine usable pairs (6
   aerial surveys, 3 manta tows; 2 at −1 day, 7 at +1 day — none on the
   day), L1C corrected with ACOLITE's dark-spectrum fitting (and TOA beside
   it). The fraction of 2 km-disc pixels above the surrounding water's 99.9th
   percentile of FDI is 0–0.16 % per pair; against 200 random 2 km discs in
   the same scene no pair ranks above the 95th percentile (median rank 0.49;
   Wilcoxon p = 0.16 over pairs and over scenes; TOA p = 0.73); against the
   aircraft's count of objects > 50 cm, Spearman ρ = −0.06 (n = 6, p = 0.91)
   — the densest survey (17 objects per km²) reads 0. The ten brightest disc
   pixels are 3–9-pixel clusters 45–204 m from cloud, with red ≈ NIR: cloud
   fringes, not debris. The physics says the same: ~0.001 % cover changes FDI
   by ~1e-6 against scene noise of ~1e-3. A 1.25 TB store over the patch
   would encode clouds. Cost: ~4.4 GB downloaded, a few minutes of compute.
3. **Phase B** only on step 2's result — **nothing to build** (`s2fdi_gpgp`
   closed above).
4. **Phase C** decisions, each with its own probe first: `s2fdi_coast`,
   `pace_swir`, `cygnss_mss`, `beachlitter`, and AOMI's unique remainder on
   the private track.

## 6 · What would falsify each store

- **`mplastic` as a read-out.** The family is worth having only if a model
  reading 1.3's inputs predicts held-out tows (left out by CRUISE, so a
  cruise's neighbouring tows cannot leak) better than two baselines: the
  nearest tows in space and time (k nearest neighbours, k = 5), and the
  published climatological map (Isobe 2021's level-3 grid). Score: error in
  log10(1 + pieces/km²) and, separately, whether an empty net is predicted
  empty. If it does not beat the neighbours, the inputs carry nothing the
  tows' own geography does not, and the read-out is shelved.
- **`mobjects`**: kept as the aircraft's own record; falsified only if its
  counts do not reproduce The Ocean Cleanup's published aircraft
  concentrations per mosaic — amended 2026-10-11 to the producer's 2024
  reprocessing, which they match on 31 of 31 (they match the 2018 paper's on
  2 of 31; §8).
- **`s2labels`**: an index threshold (FDI > its MARIDA-train 95th percentile
  over clear water) is the baseline on MARIDA's own test split; the store
  earns its place if a model reading the 11 bands plus the inherited ocean
  colour beats it on debris-vs-not F1. If the threshold already matches the
  model, the labels are a test set, not a training store.
- **`s2fdi_gpgp`**: §5 step 2.
- **`beachlitter`**: MDMAP material totals sum exactly to the total
  transect count and plastic never exceeds it; every EMODnet survey has as
  many type lines as its reference list; the recomputed pellet rate equals
  NCEI's published one on the ≥ 7,275 rows standardised the documented way.
- **`cygnss_mss`**: re-binning the good L2 samples of three days into the L3
  cells with weights 1/uncertainty² reproduces `num_mss_samples` in ≥ 99.9 %
  of cells and the mean to 1e-5 relative; else the daily grid is built from
  L2. Physics: over 2019, MSS rises with an independent reanalysis 10 m wind
  (rank correlation ≥ 0.8).
- **`pace_swir`**: a pixel re-read from the native file equals the store to
  float16 precision; over clear water `rhos` at 2258 nm has median < 0.01 and
  99th percentile < 0.03 (measured 0.0052 / 0.0217 on 2025-07-01).
- **`s2fdi_coast`**: (i) FDI from the store reproduces FDI computed from
  MARIDA's own corrected bands at its labelled pixels (Spearman ≥ 0.9 over
  clear water); (ii) the threshold baseline's debris-vs-water F1 on MARIDA's
  test split from the store's FDI is within 0.05 of the F1 from MARIDA's
  bands; (iii) on the Lesvos campaign dates the moored plastic targets rank
  above the 99th percentile of their scene's water. If (i) or (ii) fails the
  store is not kept.

## 8 · Phase A, built locally — what the adapters found (2026-10-11)

The adapters `mplastic` and `mobjects` (`ml/family1/adapters/`), the shared
Lebreton 2018 parser (`_lebreton2018.py`, now also used by the globe bake —
`data/gpgp_plastic.json` rebuilt identical apart from its snapshot date) and
the family-1.3 plumbing are written and tested (19 store tests, 20 registry
tests). Both stores were built end to end against the live sources in the
sandbox; nothing is published yet. Every number below is in
`ml/family1/probes/mplastic_ledger.json`, `mplastic_depth_table.json`,
`mplastic_emodnet_types.json` and the two probe files.

**`mplastic`: 18,504 surface samples, 1972-04-16 → 2024-10-23, 349 platforms,
0 values out of bounds.**

| source | rows read | kept | why the rest went |
|---|---|---|---|
| The Ocean Cleanup 2015–16 events (Lebreton 2018) | 683 | 683 | — |
| Lebreton 2018's compilation of earlier tows | 3,532 | 3,031 | 501 are the 2015 events again ("This study") |
| The Ocean Cleanup 2018–2022 (Lebreton 2024) | 1,193 | 496 | 692 are the 2015–16 expeditions, excluded by name and confirmed against 2018 |
| NOAA NCEI, ocean-water samples | 16,988 | 13,385 | 2,530 SEA Pacific rows by name (D9); 673 of The Ocean Cleanup's own submissions ("Egger et al.") matched to its files; 400 already in the compilation |
| EMODnet floating micro-litter | 3,385 samples | 909 | 657 already in NCEI; the rest have no plastic-only total (below) |

Decisions taken while building, each reversible by one constant:

- **D10 — concentrations are stored as log10(1 + value).** The store's values
  are float16, which stops at 65,504; real tows reach ~10⁷ pieces/km² and one
  NCEI pump sample 137,800 pieces/m³. A measured zero stays exactly 0. The
  channels are `log_n_km2`, `log_g_km2`, `log_n_m3`, `log_n_km2_wc` and four
  size classes; `sample_depth_m` was added (some pumps and bottles sample
  3–11 m down), and two time-precision flags ride in `qc` (no time of day →
  noon; month only → mid-month, the gbif/glodap precedent).
- **D11 — NCEI's per-m³ values are undone per study AND sampling method**
  (one study used two nets). 6,905 rows take an areal value from the study's
  own accession (44 accessions read; implied net depths 0.0675–0.5 m, e.g.
  Law et al. 2010 0.25 m); Eriksen 2014 is back-converted with a measured
  0.08 m (161 matches, cv 0); 3,076 net rows stay per m³ because no depth
  could be measured, and 3,028 volumetric samples are per m³ by nature.
  NCEI records several studies at the END of the tow, so matching tries both
  ends. The Ocean Cleanup's own NCEI rows are dropped only where they match its
  files (±1 day, 0.06°: its 2024 positions are rounded to 0.01°); the 96 that
  match nothing — 2018–2020 tows in no Ocean Cleanup file — are kept, per m³
  (NCEI's value there is the wind-corrected 0.5–5 mm count over 0.075 m, a
  quantity no areal channel holds).
- **D12 — EMODnet's shape-only lines COUNT as plastic, flagged (Chris,
  2026-10-13: "yes").** EMODnet's own vocabulary leaves a shape ("fragments",
  "pellets", "filaments", "films", "foams", "granules") with no polymer type
  materially undefined; the MSFD sampling protocol reads them as plastic, and
  so does the store now (`EMODNET_SHAPES_ARE_PLASTIC = True`). A sample whose
  total includes a shape-only line that caught something carries **qc bit 6
  (64)**, so the strict reading is one filter away; "undefined micro-litter
  items" (H0100010) is not a shape and stays ambiguous. Before the switch:
  1,566 samples with a plastic total (909 after removing NCEI copies);
  estimated ≈ 3,357 with it — the rebuild's ledger carries the measured count.
- **The Ocean Cleanup's 2024 file re-uses earlier numbers in specific ways**,
  checked on the shared 2015 tows: its manta values equal 2018's
  wind-CORRECTED midpoints (352 of 352), so its new 2018–22 manta tows fill
  only `log_n_km2_wc`; its large-net values equal the raw catch (18 of 18);
  its aircraft counts were REPROCESSED and equal 2018's on only 2 of 31
  mosaics (ratio 0.27–7.2, median 0.98).

**`s2labels`: 837,964 labelled Sentinel-2 pixels, 2015-11 → 2021-01**
(built locally; MARIDA's 837,377 pixels and 3,399 "marine debris" re-measured
and asserted). Two findings shape it. **MADOS's files carry no location and no
date** — every GeoTIFF has a placeholder identity georeference — so its crops
were placed by matching their spatial brightness pattern to MARIDA's: 1,488
of 2,803 crops located (each with ≥ 54 votes, median 6,196; no unlocated crop
scored above 3; satellite unit agrees with the Copernicus catalogue on all
1,488), and they are exactly MARIDA's 63 scenes. MADOS's other 111 scenes
(772,971 labelled pixels — every oil spill, oil platform, jellyfish and sea
snot) cannot be placed and are counted and left out. **And MADOS adds almost
nothing to MARIDA where it can be placed:** 707,583 pixels in both, 587 only
in MADOS; every label change in the overlap is a class RENAME (Sparse
Sargassum → Sparse Floating Algae, Waves/Wakes → Waves & Wakes, Mixed Water →
Marine Water); Marine Debris is unchanged. The producer's later label wins,
MARIDA's split is kept. **The falsifier's baseline** (§6): FDI above its
MARIDA-train 95th percentile over clear marine water, scored debris-vs-not on
MARIDA's test split — **F1 = 0.0063** (precision 0.0032, recall 0.987): the
index fires on sediment-laden water (93,037 false positives), cloud (12,686)
and turbid water (8,562). Against clear marine water alone it would score
0.81, which is why the plan's metric is the harder one. A model reading the 11
bands has a very low bar to beat; the store's use is to say whether it can.

**`mobjects`: 1,595 objects, 100 % joined, seven types.** The §6 falsifier as
written ("reproduce Lebreton 2018's megaplastic concentration per mosaic")
**fails on 29 of 31 mosaics — and the objects reproduce The Ocean Cleanup's
2024 reprocessed aircraft counts on all 31.** The producer re-derived its own
2016 aircraft numbers; the objects are consistent with the current version.
The falsifier is amended to the 2024 counts, which it passes.

**D9's per-tow note** (does SEA's later value differ?) is not in the store:
it needs AOMI's rows, which D4 keeps out. The comparison stays in §2a.

## 7 · Status

**Phase A `mplastic` and `mobjects` BUILT AND PUBLISHED 2026-10-11** (family1-build #1057 and #1058, hosted runners, $0; registry `tensors/family1_3/family13.json` by family1-registry #7; every file read back from the Hub). The co-location probe ran the same day and found NO signal — `s2fdi_gpgp` is closed (§5 step 2). `s2labels` BUILT AND PUBLISHED the same night (family1-build #1059; registry republished by family1-registry #8). **Phase A is complete: three stores, 858,063 rows.** What remains is phase C — each a decision for Chris (§5 step 4) — and the open questions below. Sources measured 2026-10-11 (the raw downloads are
in the session's scratch space, not committed). Open items, in the order they
block:

1. ~~the SEA Pacific conflict~~ — settled 2026-10-11 (§2a, D9): NCEI's
   accession 0211008 has the wrong positions. Still open: whether SEA's
   resubmitted values (about 3× for 1,027 tows) supersede the published ones
   — a question for SEA, not a blocker (the ledger keeps both);
2. Isobe 2021's data licence; Eriksen 2023's licence;
2b. ~~D12~~ — settled 2026-10-13: shape-only lines count as plastic, flagged qc bit 6 (rebuild of `mplastic`);
3. whether the repository's Earthdata account (secrets `EARTHDATA_USERNAME` /
   `EARTHDATA_PASSWORD`, working for LP DAAC, GES DISC and PO.DAAC) reads
   OB.DAAC's PACE level 2 — never tried;
4. the registry builder's resolved-chain inheritance (D8);
5. the L1C bucket's requester-pays status outside this sandbox's proxy.
