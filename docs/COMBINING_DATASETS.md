# Combining datasets in the earth catalog

*An analysis of which of the 241 cataloged datasets measure the same thing, which can be
aggregated, and where combination is scientifically meaningful — including the specific
question of merging land and ocean temperature. July 2026.*

---

## 1. The short answer

Yes — large parts of the catalog are families of datasets measuring the same physical
quantity by different methods, and combining them is not just possible but standard
scientific practice. In fact, several catalog entries *are already combinations* of other
catalog entries (GPCP merges gauge and satellite precipitation; IMBIE reconciles three
independent ice-sheet mass-balance methods; HadCRUT5 merges land stations with ship/buoy
SST). The productive way to think about the catalog is therefore in **families** (many
estimates of one quantity), **complements** (different quantities that constrain each other
through a physical budget), and **already-merged products** (where the combination work has
been done for you, with uncertainty quantified).

The single most important caution: **apparent redundancy is not independence.** Most
datasets in a family share raw inputs — the same ships, buoys, satellites and stations —
so their errors are correlated. Averaging five SST products does not reduce error by √5;
their spread is best read as a *lower bound* on structural uncertainty.

## 2. Ocean temperature — the deepest family in the catalog

The catalog contains at least nine distinct ocean-temperature datasets, and they form a
coherent hierarchy rather than mere duplication.

**Surface (SST).** Four gridded analyses overlap in purpose but differ in resolution,
record length and method: MUR (1 km daily, 2002–, the layer on the globe), OSTIA (0.05°
daily, 1982–), NOAA OISST v2.1 (0.25° daily, 1981–), and the two century-scale
reconstructions HadISST (1°, 1870–) and ERSSTv5 (2°, 1854–). ESA's SST CCI adds a
climate-quality satellite-only record. These divide naturally by role: **use MUR/OSTIA for
spatial detail, OISST for a robust daily workhorse, HadISST/ERSST for trends and long
context.** They are calibrated against overlapping in-situ networks (ICOADS ships, drifters,
Argo), so treat their agreement as consistency, not confirmation.

**Subsurface.** Argo profiles (plus Deep Argo) are the raw truth; EN4 and IAP grid those
profiles into monthly T/S fields; the World Ocean Database holds the full historical
archive; and the reanalyses (GLORYS12, ORAS5, ECCO, SODA) assimilate all of the above into
dynamically consistent states. This is a *vertical* combination chain: profiles → objective
analyses → reanalyses, each stage adding coverage and physics at the cost of model
dependence.

**Heat content.** NCEI/Levitus, IAP/Cheng and EN4-derived OHC are three estimates of the
same integral. Their spread (visible in every IPCC figure) is the community's de-facto
uncertainty band, and using all three is exactly how the annual "Indicators of Global
Climate Change" product does it.

**Sensible combinations:** an SST *ensemble layer* (mean and spread of OISST/OSTIA/MUR on
a common grid); long-record trend maps from HadISST/ERSST cross-checked against the
satellite era; OHC ensembles for the energy side of AMOC analysis; and validation of any
gridded product against raw Argo profiles, which are as close to independent ground truth
as the ocean offers.

## 3. Land + ocean temperature — sensible, with one physical subtlety

Combining land and ocean temperature is not only sensible — it is precisely how the
world's headline climate records are built. HadCRUT5, GISTEMP, NOAAGlobalTemp and Berkeley
Earth are all *blends*: land air temperature from station networks (GHCN and relatives)
merged with SST from ship/buoy archives (ERSST or HadSST). The catalog therefore already
contains four independent-ish implementations of the land+ocean merge, and their mutual
spread is the standard uncertainty statement for global mean surface temperature (GMST).

The subtlety worth understanding before building your own blend: **the two halves are
different physical quantities.** Over land these products use air temperature at 2 m;
over ocean they use *water* temperature (SST), because SST anomalies are a good and much
better-sampled proxy for marine air temperature anomalies. Three practical consequences:

1. **Blend anomalies, never absolutes.** A 15 °C SST and a 15 °C land air temperature are
   not the same thing, but their *departures from a common 1961–1990 (or similar) baseline*
   are comparable and can share a grid. Every serious product works in anomaly space.
2. **Sea-ice edges are the awkward seam.** Where ice comes and goes, products must switch
   between SST and air temperature, and the choices differ between products — a real
   contributor to their spread in polar regions.
3. **Comparison with models needs care.** Climate models output true global 2 m air
   temperature; blended observational records run slightly cooler in trend. When the
   prediction side of this project compares observed GMST to CMIP6, use the models'
   "blended-masked" diagnostics or apply the known adjustment.

For the globe, the practical recipe is: MUR/OISST as the ocean layer, ERA5-Land or
Berkeley Earth as the land layer, displayed as anomalies against a shared baseline — or
simply render an existing blended product (GISTEMP and Berkeley Earth ship gridded
anomaly fields) rather than re-deriving one.

## 4. Other high-value combinations in the catalog

**Precipitation** mirrors temperature: gauges (GPCC) are truth-but-sparse, satellites
(IMERG, CMORPH, PERSIANN) are complete-but-biased, and the merged products (GPCP for
climate quality, CHIRPS for high-resolution land) are the combinations. Building a new
merge is rarely worthwhile; comparing the existing ones quantifies uncertainty.

**Sea level is the showcase budget.** Altimetry (DUACS/AVISO, total sea level) should
equal ocean mass change (GRACE/GRACE-FO) plus steric expansion (Argo/EN4/IAP), with tide
gauges (PSMSL/UHSLC) anchoring the coast. Closing this three-way budget is a gold-standard
consistency check, and its residuals are scientifically interesting in themselves. A
"sea-level budget" dashboard would combine five catalog families in one panel.

**The planetary energy budget** links CERES (top-of-atmosphere imbalance, ~1 W/m²) to
ocean heat content change, which absorbs about 90 % of it. EEI from CERES and dOHC/dt from
IAP/NCEI are independent measurements of nearly the same number — their agreement over
multi-year windows is one of the strongest validation checks in climate science.

**The carbon budget** is an accounting identity across the catalog's GHG section:
emissions (Global Carbon Project, EDGAR, Climate TRACE) minus land and ocean sinks must
equal the observed atmospheric growth rate (NOAA GML). The Global Carbon Project performs
this reconciliation annually; Climate TRACE's facility-level totals can be validated
against national inventories (UNFCCC, PRIMAP) the same way.

**The cryosphere already demonstrates the method.** IMBIE exists because three independent
techniques — altimetry (ICESat-2), gravimetry (GRACE) and the input–output method
(velocity × thickness from ITS_LIVE/BedMachine minus SMB from RACMO/MAR) — gave divergent
ice-sheet mass balances until they were formally reconciled. Any combination work in this
project should aspire to IMBIE's pattern: independent methods, one reconciled series,
explicit uncertainty.

**The AMOC stack — most relevant to the prediction goal.** The catalog holds several
*semi-independent* estimates of AMOC state: direct transports (RAPID, OSNAP, MOVE, SAMBA,
Florida Current cable — unified by AMOCatlas), SST-fingerprint indices (computable from
OISST, HadISST *and* ERSST — three versions of the same index whose spread matters),
subsurface density/salinity signatures (EN4, Argo), altimetry-based proxies, and
reanalysis-derived streamfunctions (GLORYS, ORAS5, ECCO). A combined "AMOC state vector"
with these as parallel channels — rather than any single index — is exactly what a
credible early-warning analysis should ingest, because the tipping-point statistics
(variance, autocorrelation) are sensitive to dataset choice.

**Reanalyses as an ensemble.** ERA5, MERRA-2 and JRA-3Q assimilate largely the same
observations with different models; their spread is the honest uncertainty of "what the
atmosphere did." For any derived index (jet position, blocking, surface winds over the
subpolar gyre), computing it in at least two reanalyses is cheap insurance.

**Risk stacks multiply rather than average.** Hazard (IBTrACS tracks, FIRMS fires, GloFAS
floods, HadEX3 extremes) × exposure (WorldPop, GHSL) × vulnerability (ND-GAIN) is a
combination *across* families that yields impact layers — a different logic from
aggregating estimates of one quantity, and well suited to the globe.

## 5. Rules of engagement for any combination

Work in anomaly space with a common baseline period; regrid conservatively to the coarsest
common resolution rather than inventing detail; never treat products sharing inputs as
independent; keep reanalyses and observation-only products distinct in your bookkeeping
(reanalyses have *assimilated* the observations you might validate them against); check
whether the quantities are truly the same (SST vs marine air temperature, skin vs
foundation temperature, gauge point vs satellite areal average, calendar-day vs
period-mean); and propagate ensemble spread through to every derived number, presenting it
as a lower bound on uncertainty.

## 6. What this suggests for the project

Three concrete, buildable steps fall out of this analysis. First, an **SST ensemble
layer**: OISST + OSTIA + MUR on a common grid with a mean and a spread toggle — the spread
map itself is a compelling visualization of where the ocean is well- vs poorly-observed.
Second, the **sea-level budget panel**: altimetry vs GRACE + steric, the catalog's most
elegant cross-family check. Third, the **multi-channel AMOC state vector** feeding the
dashboard: RAPID/OSNAP transports, three SST-fingerprint variants, and an EN4 salinity
index side by side, with their disagreement displayed honestly — the right substrate for
the tipping-point statistics this project ultimately wants to compute.

A supporting improvement to the catalog itself: add a `family` field (e.g. `sst`,
`ocean-heat-content`, `gmst-blend`, `precipitation`, `sea-level`, `amoc-index`) so these
relationships become machine-readable and the app can show "related datasets" on every
entry.
