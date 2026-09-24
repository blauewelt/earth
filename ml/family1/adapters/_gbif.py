"""GBIF occurrence records — the body both tracks share (family 1.0.tf).

PLAIN ENGLISH. GBIF — the Global Biodiversity Information Facility — is the
world's pooled record of "an organism of this kind was here on this day":
museum specimens, bird lists, camera traps, DNA samples, four billion of
them. It is the biosphere's `icoads` (family 1.gf's archive of ship weather
reports): sparse, biased towards birds and towards wealthy countries, and
the only global record of species there is. This module reads the monthly
Parquet snapshot GBIF publishes on the Amazon Open Data registry and turns
it into tier-P rows, one per record.

TWO TRACKS, DECIDED BY THE ROW'S OWN `license` COLUMN. GBIF publishes each
record under one of three licences, and the snapshot carries which. CC0 and
CC BY 4.0 rows may be redistributed, so they go to the PUBLIC store `gbif`;
CC BY-NC 4.0 rows may not be put on a public track by this project's own
two-track rule (E-082 §1.3), so they go to the private sibling `gbif_nc`
(`distribution = "private"`, `licence["redistribution"] = "no"`), exactly as
`tide` and `tide_private` split GESLA's contributors. The two stores request
the same parts and keep disjoint rows: together they read the snapshot once.

SOURCE, VERIFIED 2026-09-20 FROM THIS SANDBOX (anonymous S3 over HTTPS, no
account, no signing):
  listing  https://gbif-open-data-us-east-1.s3.amazonaws.com/?list-type=2
           &prefix=occurrence/  — 65 monthly snapshots, 2021-04-13 ..
           2026-09-01.
  snapshot .../?list-type=2&prefix=occurrence/2026-09-01/ — ten pages,
           3,443,777 bytes of XML, **9,899 objects**: one `citation.txt`
           (88 B) and **9,898 parquet parts** under `occurrence.parquet/`,
           named `000001` .. `010418` with gaps and WITHOUT a `.parquet`
           suffix, **285,323,189,332 bytes** in all (the notes' 285.3 GB).
           Part sizes: 0.01 MB at the minimum, 15.1 / 27.2 / 47.8 MB at the
           quartiles, 102.1 MB at the maximum, 28.8 MB mean.
  a part   `occurrence/2026-09-01/occurrence.parquet/005147`, the
           median-sized one (27,226,457 B), read with pyarrow: **383,269
           rows, ONE row group, 50 columns**, written by parquet-mr 1.9.0,
           SNAPPY, every column but four dictionary-encoded.

COLUMN PROJECTION IS WHY THIS IS AFFORDABLE, and it is measured. The fifteen
columns a row needs are **6,024,510 of the part's 27,219,965 compressed
bytes — 22.1 %** (`occurrenceid` alone is 7.3 MB and `locality` 3.6 MB, and
neither is read). Parquet stores each column contiguously and lists its byte
range in the footer, so the adapter opens each part as a FILE-LIKE OVER HTTP
RANGE REQUESTS (`HttpParquet` below) and pyarrow fetches only those ranges:
measured on that part, **9 requests and 6,092,259 bytes, 22.38 % of the
file, in 5.1 s**. The whole snapshot is therefore ~63 GB of transfer rather
than 285 GB.

THE ROW RULE — a record is kept when it has a position, no geospatial issue,
and a date to the month or better.

  * **A geospatial issue is SEVEN issue names, not the four that get quoted,
    and the set was measured rather than recalled.** The snapshot has no
    `hasgeospatialissues` column; it has an `issue` list. GBIF's own search
    API was asked, on 2026-09-20, for `hasGeospatialIssue=true` (7,089,295
    records) and then, for each candidate name, for the count of records
    carrying it WITHOUT a geospatial issue. Seven names answered zero —
    ZERO_COORDINATE, COORDINATE_INVALID, COORDINATE_OUT_OF_RANGE,
    COUNTRY_COORDINATE_MISMATCH, PRESUMED_SWAPPED_COORDINATE,
    PRESUMED_NEGATED_LATITUDE, PRESUMED_NEGATED_LONGITUDE — and the union of
    exactly those seven counts **7,089,295**, the geospatial total to the
    record. Names that look like they belong and do NOT
    (COORDINATE_REPROJECTED, COORDINATE_REPROJECTION_SUSPICIOUS,
    GEODETIC_DATUM_INVALID, CONTINENT_COORDINATE_MISMATCH,
    COORDINATE_UNCERTAINTY_METERS_INVALID) all occur in millions of records
    with no geospatial issue. `index` re-runs that falsification against the
    live API — seven cheap calls — and REFUSES if a declared member turns out
    to occur without a geospatial issue; a network failure records
    `geospatial_check: "unavailable"` and the build continues, because an
    unreachable API is not evidence about the set.
  * **A date to the month or better** means `year` AND `month` are both
    present. `day` may be absent. A record whose `eventdate` is present while
    `year` is null (1,069 of the 383,269 in the part measured) is GBIF saying
    the date spans more than one year; it is dropped and counted.

WHAT A ROW IS.
  time_s    int64 seconds since 1982-01-01 (schema 3 — the record reaches the
            1600s and `first_year` is 1600, far outside int32's 1913 floor).
            With a day: that day at the `eventdate` time of day when the
            snapshot carries one and it agrees with year/month/day, else at
            NOON UTC. Without a day: the MIDDLE of the month, the month's
            first second plus half its length, which is the choice that
            minimises how far a month-only record can be from where it
            belongs.
  lat, lon  `decimallatitude`, `decimallongitude`.
  platform  platform_hash of the `taxonkey` STRING — GBIF's 2026 keys are
            short alphanumerics such as `6GQ7J`, not integers. A record whose
            taxon did not match the backbone (6,265 of 383,269 in the part
            measured, 1.6 %) is pooled under
            platform_hash("gbif-taxon-unmatched") and counted.
            `platforms.json` maps each hash to its taxon key, scientific
            name, kingdom and class.
  values    C = 4 codes and counts, every code an index into a table printed
            in store.json: kingdom_code (into `KINGDOMS`), class_code (into
            `CLASSES`), basis_of_record_code (into `BASIS_OF_RECORD`), and
            individual_count. A name absent from its table is NaN and counted
            by name, never coerced.
  qc        A BITFIELD, NOT A GRADE, and the only store in the family where
            that is so — `qc_policy` says it and store.json carries it:
              bit 0 (1)     the record states no coordinate uncertainty, so
                            the footprint below is the store's "unknown"
                            constant
              bit 1 (2)     the date is month-only, so this row's real time
                            footprint is log2(30.44/5) = +2.606, not the
                            store's -2.322
              bits 2-7      k = round(log2(coordinate uncertainty in metres))
                            clamped to [0, 20] and stored as k << 2; the
                            row's real log2_fp is k - log2(1000 * 27.83) =
                            k - 14.7639. k is 10 (1,024 m, the 1 km unknown
                            constant) whenever bit 0 is set.
            `--qc-keep` must NOT be read as a grade for this store.

WHY THE FOOTPRINT IS IN `qc` AND NOT IN `fp.npy`. The design asks for a
per-row footprint and a per-row time footprint, and the tier-P layout cannot
carry either: both assemblers fill `fp.npy` from the adapter's single
`log2_fp`/`log2_dt` pair and `check_store` ASSERTS the two columns are
constant ("the footprint columns are not constant for this source"). Changing
that touches `ROW_KEYS`, `_pack`, `PartWriter`, both assemblers and every
part already parked on the Hub, so it is not an additive change and was not
made here. The store's constant pair is the "unknown" footprint (1 km, the
value the design names for a record with no uncertainty, log2(1/27.83) =
-4.798) and the day support (log2(1/5) = -2.322); the six spare bits of `qc`
carry each row's own footprint to a factor of two, and bit 1 names the rows
whose support is a month. Nothing is lost and nothing is invented.

THE CODE TABLES ARE MEASURED FROM THIS SNAPSHOT, and how is on the record.
`KINGDOMS` is the 19 distinct `kingdom` values seen across 233 sampled parts
(58.9 M + 29.2 M rows) — and they are NOT the names GBIF's own species API
returns for kingdom keys 0..8 (`Bacteria`, `Archaea`): the 2026 backbone
renamed the prokaryote kingdoms (`Pseudomonadati`, `Bacillati`,
`Methanobacteriati`, `Thermotogati`, `Fusobacteriati`, `Thermoproteati`,
`Nanobdellati`) and added the virus realms, while `species/match?name=
Pseudomonadati` still answers `matchType: NONE`. `CLASSES` is the 450
distinct `class` values from the same sample; 194 of them cover 99.9 % of the
sampled rows, and the table is under float16's 2048-exact-integer ceiling by
a factor of four. `BASIS_OF_RECORD` is GBIF's own enumeration, fetched from
`api.gbif.org/v1/enumeration/basic/BasisOfRecord`. A table is EXTENDED BY
APPENDING, never re-sorted, so a code never moves once a store holds it.

THE PROBE MEASURES PART FILES AND IGNORES ITS MONTH ARGUMENT, because the
snapshot is not partitioned by time: a part holds records from the 1600s to
last week, and no part is "September 2026". `--probe-month` still chooses
which month the framework's own per-day and row counts describe — the probe
filters the rows it is handed to that month — while the measurement the store
is sized from is in `counts`: `probe_part`, `probe_rows_in_part`,
`probe_bytes`, `probe_seconds`, `probe_rows_by_licence`,
`probe_share_day_precision`, `probe_share_with_uncertainty` and
`probe_projection`, which multiplies the parts' mean kept rows by 9,898 parts
and by 31 + 2C = 39 bytes a row.

ONE PART WAS THE DESIGN, AND ONE PART IS NOT ENOUGH — measured, not decided.
The parts follow the PUBLISHERS, so one part can be one publisher's whole
dataset under one licence: part 005210 of the 2026-09-01 snapshot, the middle
of the listing, holds 585,040 records of which **585,034 are CC BY-NC**. A
one-part probe of the public store would therefore measure six rows and
project a store of nothing, and a one-part probe of either store would
extrapolate a licence split that does not exist. `GBIF_PROBE_PARTS` (default
8) spreads the probe evenly across the listing, `per_part` keeps every part's
own numbers so the spread is visible rather than averaged away, and
`keep_fraction_per_part` is the first thing to read in the projection.

THE PROBE, 2026-09-01 SNAPSHOT, MEASURED 2026-09-20 FROM THIS SANDBOX
(`--stage probe --probe-month 2020-12`, ml/family1/probes/gbif_2020-12.json
and gbif_nc_2020-12.json): eight evenly spaced parts, 3,028,011 records read
through **55,202,914 bytes and 54 s** — 18.2 source bytes a record, because
the column projection fetches 22 % of the parts. The two tracks keep
**2,094,778** (69.2 %) and **579,838** (19.1 %) of those records, 88.3 %
between them; the rest are the 297,988 with no position, the 53,531 with no
month, the 1,876 with a geospatial issue and the rounding. Of the public
rows, 1,777,756 are CC BY 4.0 and 317,022 CC0; **99.98 % carry a day rather
than a month** and **31.3 % state a coordinate uncertainty** (56.2 % on the
private track). 35,206 distinct taxa in the probe month alone; 8,546 records
whose taxon did not match the backbone; 37,733 uncertainties clamped into
[10 m, 100 km]; 13 individual counts past float16. **NOT ONE kingdom or
class name fell outside the code tables** in three million records, which is
what those 450 + 19 names were measured for.

PROJECTED: 2.59e9 rows and **101.1 GB** on the public track, 7.17e8 rows and
**28.0 GB** on the private one, at 31 + 2C = 39 bytes a row, with 68.3 GB of
transfer to build either (the same parts are read once per track). The
ledger's "~90 GB public" is 11 % low and its "~130 GB" in the wave plan 29 %
high. The per-part keep fractions run 0.045 .. 0.9999 — read
`keep_fraction_per_part` before trusting the mean.

TWO ENVIRONMENT KNOBS, both read at CONSTRUCTION time so the fresh adapter
`stage_probe` builds for itself sees them, exactly as `swot` reads
`SWOT_MAX_PASSES`. `GBIF_SNAPSHOT` (a `YYYY-MM-DD` prefix) pins the snapshot;
the default is the newest the bucket lists, and BOTH TRACKS MUST BE BUILT
FROM THE SAME ONE. `GBIF_PARTS` restricts the parts a run reads — `"0:500"`
or a comma list of part names — because the snapshot has no time axis to lane
on and a part range is the only way to split the work. A restricted run
writes the restriction into `notes`, which reaches store.json, so a partial
store can never look like a whole one.

A PART RANGE IS A LANE (2026-09-24). The whole pass was the resumable unit —
23 h on one box, and family1-build #837 lost nine hours of it when its runner
lost contact — so the snapshot is fetched as N PART-RANGE LANES on
GitHub-hosted runners, each `GBIF_PARTS=lo:hi` with `--push-parts`, and
assembled on a box with `--parts-from-hub --lanes parts:N`. A range's part
NAMES are its groups, so the lane is `g-<hash>` (`group_subset`), every year
it writes carries them as `lane_groups`, and N lanes of one year are disjoint.
`part_lanes(parts, n)` — or `python3 -m family1.adapters._gbif --lanes N`
from `ml/` — gives the dispatcher the N specs and lane names.

THE TAXON TABLE TRAVELS ON DISK. `platforms(ctx)` is called at ASSEMBLE time
and the taxa are only discovered during the FETCH, so the fetch writes
`<work>/<store>/taxa.json` atomically as it goes (flush, THEN mark —
ml/CLAUDE.md §5.21) and `platforms()` reads it back. For an assemble on a
machine that did not run the fetch — `--parts-from-hub` — the table ALSO
travels with the parts: `finish_fetch` writes each year-lane directory a
`taxa.json` SIDECAR holding the taxa of that directory's rows, the Hub carries
and verifies it like a part (`family10_parts_hub.SIDECAR_NAMES`), and
`platforms()` unions every sidecar of the years being assembled, refusing two
different records for one key and parts from two different snapshots. Only
with no table anywhere is it refused, rather than publishing a store whose
platforms.json is empty. It is also the one thing here that scales with the
ARCHIVE rather than with the block being read: the probe met 35,206 distinct
taxa in eight parts, so a whole-snapshot build holds a few million entries,
and the sidecar is rewritten every `TAXA_FLUSH_PARTS` parts rather than every
part because rewriting a growing file 9,898 times is quadratic.
"""
import datetime as dt
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _common as cm

BUCKET = "gbif-open-data-us-east-1"
BASE = f"https://{BUCKET}.s3.amazonaws.com/"
PREFIX = "occurrence/"
PARQUET_DIR = "occurrence.parquet/"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
GBIF_API = "https://api.gbif.org/v1/occurrence/search"

SNAPSHOT_RE = re.compile(r"^occurrence/(\d{4}-\d{2}-\d{2})/$")

# The fifteen columns a row needs — 22.1 % of a part's compressed bytes.
COLUMNS = ("decimallatitude", "decimallongitude",
           "coordinateuncertaintyinmeters",
           "eventdate", "year", "month", "day", "taxonkey", "scientificname",
           "kingdom", "class", "basisofrecord", "individualcount", "license",
           "issue")

# Measured 2026-09-20 against GBIF's own search API; see the module docstring.
# The union of exactly these seven is 7,089,295 records, which is the count
# of `hasGeospatialIssue=true` to the record, and each of them occurs in ZERO
# records without a geospatial issue.
GEOSPATIAL_ISSUES = ("COORDINATE_INVALID", "COORDINATE_OUT_OF_RANGE",
                     "COUNTRY_COORDINATE_MISMATCH",
                     "PRESUMED_NEGATED_LATITUDE",
                     "PRESUMED_NEGATED_LONGITUDE",
                     "PRESUMED_SWAPPED_COORDINATE", "ZERO_COORDINATE")
GEOSPATIAL_SET = frozenset(GEOSPATIAL_ISSUES)
GEOSPATIAL_MEASURED = {"at": "2026-09-20", "hasGeospatialIssue_true": 7089295,
                       "union_of_the_seven": 7089295}

# The licence column's three values (measured: no other value occurs in the
# 233 parts sampled) and the track each one belongs to. GBIF's own licence
# enumeration also names UNSPECIFIED and UNSUPPORTED; a row carrying either,
# or anything else, is DROPPED by both tracks and counted — an unknown
# licence never reaches a public store.
LICENCE_TRACK = {"CC0_1_0": "public", "CC_BY_4_0": "public",
                 "CC_BY_NC_4_0": "private"}
LICENCE_URL = {
    "CC0_1_0": "http://creativecommons.org/publicdomain/zero/1.0/legalcode",
    "CC_BY_4_0": "http://creativecommons.org/licenses/by/4.0/legalcode",
    "CC_BY_NC_4_0": "http://creativecommons.org/licenses/by-nc/4.0/legalcode",
}

# GBIF's own enumeration, fetched 2026-09-20 from
# api.gbif.org/v1/enumeration/basic/BasisOfRecord, in the order it returns.
BASIS_OF_RECORD = ("PRESERVED_SPECIMEN", "FOSSIL_SPECIMEN", "LIVING_SPECIMEN",
                   "OBSERVATION", "HUMAN_OBSERVATION", "MACHINE_OBSERVATION",
                   "MATERIAL_SAMPLE", "LITERATURE", "MATERIAL_CITATION",
                   "OCCURRENCE", "UNKNOWN")

# The 19 `kingdom` values of the 2026-09-01 snapshot, sorted, measured over
# 233 sampled parts. NOT the names GBIF's species API gives for kingdom keys
# 0..8 — see the module docstring.
KINGDOMS = ("Animalia", "Bacillati", "Bamfordvirae", "Chromista", "Fungi",
            "Fusobacteriati", "Heunggongvirae", "Loebvirae",
            "Methanobacteriati", "Nanobdellati", "Orthornavirae",
            "Pararnavirae", "Plantae", "Protozoa", "Pseudomonadati",
            "Sangervirae", "Shotokuvirae", "Thermoproteati", "Thermotogati")

# The 450 `class` values of the same sample, sorted. 194 of them cover
# 99.9 % of the sampled rows; the table is EXTENDED BY APPENDING so a code
# never moves once a store holds it.
CLASSES = (
    "Abditibacteriia", "Acantharia", "Acidimicrobiia",
    "Acidithiobacillia", "Aconoidasida", "Actinomycetes",
    "Agaricomycetes", "Agaricostilbomycetes", "Allomalorhagida",
    "Alphaproteobacteria", "Alsuviricetes", "Amabiliviricetes",
    "Amphibia", "Anaerolineae", "Anaeromonadea", "Andreaeopsida",
    "Anthocerotopsida", "Anthozoa", "Apicomonadea", "Appendicularia",
    "Aquificia", "Arachnida", "Archaeocyatha", "Archaeoglobi",
    "Archaeorhizomycetes", "Archiacanthocephala", "Ardenticatenia",
    "Arfiviricetes", "Armatimonadia", "Arthoniomycetes", "Ascetosporea",
    "Ascidiacea", "Asteroidea", "Atractiellomycetes", "Aves",
    "Bacillariophyceae", "Bacilli", "Bacteroidia", "Balneolia",
    "Bangiophyceae", "Betaproteobacteria", "Bicoecea", "Bicosoecophyceae",
    "Bigyromonadea", "Bivalvia", "Blastocatellia", "Blastocladiomycetes",
    "Blastocystea", "Bolidophyceae", "Branchiopoda", "Breviatea",
    "Bryopsida", "Bunyaviricetes", "Calcarea", "Caldilineae",
    "Caldisericia", "Calditrichia", "Candelariomycetes",
    "Cardeaviricetes", "Carpomonadea", "Caudofoveata", "Caudoviricetes",
    "Cephalaspidomorphi", "Cephalocarida", "Cephalopoda", "Cestoda",
    "Charophyceae", "Chileata", "Chilopoda", "Chitinispirillia",
    "Chitinivibrionia", "Chitinophagia", "Chlamydiia", "Chlorarachnea",
    "Chlorarachniophyceae", "Chlorobiia", "Chlorodendrophyceae",
    "Chloroflexia", "Chlorokybophyceae", "Chlorophyceae",
    "Chloropicophyceae", "Choanoflagellatea", "Chondrostei",
    "Chromadorea", "Chrymotiviricetes", "Chrysiogenia",
    "Chrysomerophyceae", "Chrysoparadoxophyceae", "Chrysophyceae",
    "Chthonomonadia", "Chuariophyceae", "Chytridiomycetes", "Cladistii",
    "Classiculomycetes", "Clitellata", "Clostridia", "Coccidiomorphea",
    "Coccolithophyceae", "Coelacanthi", "Coleochaetophyceae",
    "Collembola", "Colpodea", "Colponemea", "Compsopogonophyceae",
    "Coniocybomycetes", "Conjugatophyceae", "Conoidasida", "Copepoda",
    "Coriobacteriia", "Craniata", "Cricoconarida", "Crinoidea",
    "Cristidiscoidea", "Cryptophyceae", "Cubozoa", "Cyanidiophyceae",
    "Cyanophyceae", "Cycadopsida", "Cyclorhagida", "Cyrtophoria",
    "Cystobasidiomycetes", "Cytophagia", "Dacrymycetes",
    "Deferribacteres", "Deferrisomatia", "Dehalococcoidia", "Deinococci",
    "Deltaproteobacteria", "Demospongiae", "Desulfarculia",
    "Desulfobaccia", "Desulfobacteria", "Desulfobulbia", "Desulfomonilia",
    "Desulfovibrionia", "Desulfuromonadia", "Dictyochophyceae",
    "Dictyosteliomycetes", "Dinophyceae", "Diplonemea", "Diplopoda",
    "Diplura", "Dipneusti", "Discosea", "Dissulfuribacteria",
    "Dothideomycetes", "Duplopiviricetes", "Echinoidea", "Elasmobranchii",
    "Ellobiopsea", "Elusimicrobia", "Endogonomycetes", "Endomicrobiia",
    "Enoplea", "Enteropneusta", "Entomophthoromycetes",
    "Entorrhizomycetes", "Eoacanthocephala", "Eogyrea", "Eopharyngea",
    "Epsilonproteobacteria", "Erysipelotrichia", "Eucycliophora",
    "Euglenophyceae", "Eurotatoria", "Eurotiomycetes",
    "Eustigmatophyceae", "Eutardigrada", "Euthycarcinoidea",
    "Exobasidiomycetes", "Faserviricetes", "Fibrobacteria", "Filasterea",
    "Filosia", "Fimbriimonadia", "Flasuviricetes", "Flavobacteriia",
    "Florideophyceae", "Fusobacteriia", "Fusulinata",
    "Gammaproteobacteria", "Gastropoda", "Gemmatimonadia",
    "Geoglossomycetes", "Ginkgoopsida", "Glaucophyceae", "Globothalamea",
    "Glomeromycetes", "Gordioida", "Granofilosea", "Gymnolaemata",
    "Gymnostomatea", "Halobacteria", "Haplomitriopsida", "Haplosporea",
    "Herviviricetes", "Heterotardigrada", "Heterotrichea",
    "Hexactinellida", "Hilomonadea", "Holocephali", "Holophagae",
    "Holostei", "Holothuroidea", "Homoscleromorpha", "Hoplonemertea",
    "Hydrogenophilia", "Hydrozoa", "Hyphochytrea", "Hypotrichea",
    "Ichthyosporea", "Ignavibacteria", "Imbricatea", "Insecta",
    "Insthoviricetes", "Jakobea", "Jungermanniopsida", "Karyorelictea",
    "Kickxellomycetes", "Kinetofragminophora", "Kinetoplastea",
    "Kiritimatiellia", "Klebsormidiophyceae", "Ktedonobacteria",
    "Kutorginata", "Laboulbeniomycetes", "Labyrinthulea",
    "Langiophytopsida", "Lecanoromycetes", "Leiosporocerotopsida",
    "Lentisphaeria", "Leotiomycetes", "Leptocardii", "Leucocryptea",
    "Leviviricetes", "Lichinomycetes", "Liliopsida", "Limnochordia",
    "Lingulata", "Litostomatea", "Lobosa", "Lobulomycetes",
    "Longimicrobiia", "Lycopodiopsida", "Magnoliopsida", "Magsaviricetes",
    "Malacostraca", "Malasseziomycetes", "Malawimonadea",
    "Malgrandaviricetes", "Mamiellophyceae", "Mammalia",
    "Marchantiopsida", "Megaviricetes", "Merostomata",
    "Mesochytriomycetes", "Mesostigmatophyceae", "Methanobacteria",
    "Methanococci", "Methanomicrobia", "Methanosarcinia", "Metromonadea",
    "Miaviricetes", "Microbotryomycetes", "Microsporea", "Minisyncoccia",
    "Mixiomycetes", "Mollicutes", "Moniliellomycetes", "Monjiviricetes",
    "Monoblepharidomycetes", "Monogenea", "Monoplacophora",
    "Monothalamea", "Mortierellomycetes", "Mucoromycetes",
    "Mystacocarida", "Myxini", "Myxomycetes", "Myxozoa", "Naldaviricetes",
    "Nanomonadea", "Nassophorea", "Negativicutes",
    "Neocallimastigomycetes", "Neolectomycetes", "Nephroselmidophyceae",
    "Nitriliruptoria", "Nitrososphaeria", "Nitrospinia", "Nitrospiria",
    "Noctilucea", "Nodosariata", "Nuda", "Obolellata", "Octocorallia",
    "Oligoflexia", "Oligohymenophorea", "Oligosphaeria", "Oligotrichea",
    "Opalinea", "Ophiuroidea", "Opitutia", "Orbiliomycetes", "Ostracoda",
    "Oxyrrhea", "Palaeacanthocephala", "Palaeonemertea",
    "Palmophyllophyceae", "Palpitea", "Papovaviricetes", "Paterinata",
    "Pauropoda", "Pavlovophyceae", "Pedinophyceae", "Peranemea",
    "Perkinsea", "Peronosporea", "Petromyzonti", "Pezizomycetes",
    "Phaeophyceae", "Phaeothamniophyceae", "Pharingeaviricetes",
    "Phascolosomatidea", "Phycisphaerae", "Phylactolaemata",
    "Phyllopharyngea", "Phytomastigophora", "Phytomyxea",
    "Picocystophyceae", "Picomonadea", "Picophagophyceae", "Pilidiophora",
    "Pinguiophyceae", "Pinopsida", "Pisoniviricetes", "Planctomycetia",
    "Pneumocystomycetes", "Pokkesviricetes", "Polyacanthocephala",
    "Polychaeta", "Polycystina", "Polyplacophora", "Polypodiopsida",
    "Polytrichopsida", "Porphyridiophyceae", "Prostomatea",
    "Protosteliomycetes", "Protura", "Pterobranchia", "Pucciniomycetes",
    "Pycnogonida", "Pyramimonadophyceae", "Quintoviricetes",
    "Raphidophyceae", "Rappephyceae", "Remipedia", "Repensiviricetes",
    "Reptilia", "Resentoviricetes", "Revtraviricetes",
    "Rhizophydiomycetes", "Rhodellophyceae", "Rhodothermia", "Rhombozoa",
    "Rhynchonellata", "Rostroconchia", "Rubrobacteria", "Saccharomycetes",
    "Sagittoidea", "Sanchytriomycetes", "Saprospiria", "Sarcomonadea",
    "Sareomycetes", "Scaphopoda", "Schizosaccharomycetes", "Scyphozoa",
    "Sipunculidea", "Skiomonadea", "Solenogastres", "Sordariomycetes",
    "Sphagnopsida", "Sphingobacteriia", "Spiculogloeomycetes",
    "Spirochaetia", "Spirotrichea", "Staurozoa", "Stavomonadea",
    "Stelpaviricetes", "Stenolaemata", "Sticholonchea",
    "Stromatoporoidea", "Strophomenata", "Stylonematophyceae", "Symphyla",
    "Syndinea", "Synergistia", "Syntrophia", "Syntrophobacteria",
    "Syntrophomonadia", "Syntrophorhabdia", "Takakiopsida",
    "Taphrinomycetes", "Teleostei", "Telonemea", "Tentaculata",
    "Tepidiformia", "Terriglobia", "Terrimicrobiia", "Thaliacea",
    "Thecofilosea", "Thecomonadea", "Thecostraca", "Thermoanaerobaculia",
    "Thermococci", "Thermodesulfobacteria", "Thermodesulfovibrionia",
    "Thermoflexia", "Thermoleophilia", "Thermolithobacteria",
    "Thermomicrobia", "Thermoplasmata", "Thermoprotei", "Thermotogae",
    "Tolucaviricetes", "Trebouxiophyceae", "Trematoda", "Tremellomycetes",
    "Trichomonadea", "Trichonymphea", "Trilobita", "Tritirachiomycetes",
    "Tubothalamea", "Tubulinea", "Turbellaria", "Ulvophyceae",
    "Umbelopsidomycetes", "Ustilaginomycetes", "Vampyrellidea",
    "Variosea", "Vendophyceae", "Verrucomicrobiia", "Vicinamibacteria",
    "Vulcanimicrobiia", "Wallemiomycetes", "Xanthophyceae",
    "Xylobotryomycetes", "Xylonomycetes", "Zoopagomycetes", "Zygomycetes",
)

CHANNELS = (("kingdom_code", "index into KINGDOMS (store.json)", 0.0,
             float(len(KINGDOMS) - 1)),
            ("class_code", "index into CLASSES (store.json)", 0.0,
             float(len(CLASSES) - 1)),
            ("basis_of_record_code", "index into BASIS_OF_RECORD "
                                     "(store.json)", 0.0,
             float(len(BASIS_OF_RECORD) - 1)),
            # float16 tops out at 65504 and is exact on integers only to
            # 2048; a larger count is NaN and counted, never silently rounded
            # into infinity.
            ("individual_count", "individuals", 0.0, 60000.0))

UNKNOWN_FOOTPRINT_M = 1000.0            # the design's "unknown" constant
UNCERTAINTY_CLAMP = (10.0, 100000.0)    # 10 m .. 100 km
REFERENCE_KM = 27.83                    # the family's cell, E-078 §2
QC_NO_UNCERTAINTY = 1
QC_MONTH_ONLY = 2
QC_K_SHIFT = 2
QC_K_MAX = 20                           # 2^20 m = 1,049 km; clamped there
TAXON_UNMATCHED = "gbif-taxon-unmatched"
# How often the taxon sidecar is rewritten; see `_parts_rows`. The probe met
# 35,206 distinct taxa in eight parts, so a whole-snapshot build holds a few
# million entries — order a gigabyte of dicts, which is the one part of this
# adapter that scales with the ARCHIVE rather than with the block being read.
TAXA_FLUSH_PARTS = 32
# The year-lane sidecar `finish_fetch` writes; it must be one of
# `family10_parts_hub.SIDECAR_NAMES` for the Hub to carry it (a test says so).
TAXA_SIDECAR = "taxa.json"
NOON = 43200
DAYS_IN_MONTH = (0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
MEAN_MONTH_DAYS = 365.2425 / 12.0       # 30.4369

NOTE_ESTIMATE = {
    "bytes": 90e9,
    "what": ("family1tf.tex ledger: '~90 GB on the public track' for the "
             "GBIF occurrence snapshot, over 3.67e9 records with "
             "coordinates, no geospatial issue and a year"),
}


class FormatError(ValueError):
    """A listing or a parquet part that is not the snapshot measured above."""


# ============================================== parquet over HTTP ranges ====
class HttpParquet(io.RawIOBase):
    """A seekable file over anonymous HTTP Range requests.

    pyarrow asks a parquet file for its footer and then for exactly the byte
    ranges of the columns it was told to read, so handing it one of these
    fetches ~22 % of a GBIF part instead of all of it (measured: 9 requests,
    6,092,259 of 27,226,457 bytes). A server that ignores the Range header
    and answers 200 is REFUSED rather than consumed — that would be the whole
    file arriving under a request for a slice.
    """

    def __init__(self, url, size, attempts=4, count=None):
        self.url = url
        self.size = int(size)
        self.attempts = max(1, int(attempts))
        self._pos = 0
        self.requests = 0
        self.bytes_read = 0
        self._count = count

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, off, whence=0):
        if whence == 0:
            self._pos = int(off)
        elif whence == 1:
            self._pos += int(off)
        else:
            self._pos = self.size + int(off)
        return self._pos

    def tell(self):
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self._pos
        n = min(int(n), self.size - self._pos)
        if n <= 0:
            return b""
        end = self._pos + n - 1
        err = None
        for i in range(self.attempts):
            try:
                req = urllib.request.Request(
                    self.url, headers={**f10b.UA,
                                       "Range": f"bytes={self._pos}-{end}"})
                with urllib.request.urlopen(
                        req, timeout=f10b.SOCKET_TIMEOUT) as r:
                    if r.status != 206:
                        raise FormatError(
                            f"{self.url}: asked for bytes {self._pos}-{end} "
                            f"and got HTTP {r.status} — refusing to read a "
                            f"whole 28 MB part for one column")
                    b = r.read()
                if len(b) != n:
                    raise IOError(f"{self.url}: range {self._pos}-{end} "
                                  f"returned {len(b)} of {n} bytes")
                self.requests += 1
                self.bytes_read += len(b)
                self._pos += len(b)
                f10b.count_bytes(len(b))
                if self._count is not None:
                    self._count(len(b))
                return b
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise f10b._NotFound(self.url) from None
                err = e
            except (IOError, *cm.RETRY_ERRORS) as e:
                err = e
            if i < self.attempts - 1:
                time.sleep(2.0 * (2 ** i))
        raise IOError(f"{self.url} bytes {self._pos}-{end}: "
                      f"{type(err).__name__}: {err}")

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def _pq():
    try:
        import pyarrow.parquet as pq
    except ImportError:                                     # pragma: no cover
        sys.exit("gbif needs the `pyarrow` package (pip install "
                 "--break-system-packages pyarrow) — it is in the "
                 "family1-build workflow's install step")
    return pq


def read_part(url_or_path, size=None, columns=COLUMNS, attempts=4,
              count=None, want_meta=False):
    """One part -> (pyarrow Table, {bytes, requests, rows, row_groups}).

    `size` is None for a local file (the smoke's synthetic snapshot).
    """
    pq = _pq()
    if size is None:
        n = os.path.getsize(url_or_path)
        if count is not None:
            count(n)
        f = open(url_or_path, "rb")
        stat = {"bytes": n, "requests": 0, "whole_file": True}
    else:
        f = HttpParquet(url_or_path, size, attempts=attempts, count=count)
        stat = None
    try:
        pf = pq.ParquetFile(f)
        md = pf.metadata
        have = set(pf.schema_arrow.names)
        missing = [c for c in columns if c not in have]
        if missing:
            raise FormatError(
                f"the part is missing column(s) {missing}; it has "
                f"{md.num_columns} columns, {sorted(have)[:8]}...")
        table = pf.read(columns=list(columns))
        info = {"rows": int(md.num_rows), "row_groups": int(md.num_row_groups),
                "columns": int(md.num_columns),
                "created_by": str(md.created_by)}
        if want_meta:
            info["schema"] = [f"{n}:{t}" for n, t in
                              zip(pf.schema_arrow.names,
                                  pf.schema_arrow.types)]
        if stat is None:
            stat = {"bytes": f.bytes_read, "requests": f.requests,
                    "whole_file": False}
        stat.update(info)
        return table, stat
    finally:
        f.close()


# ================================================================ listing ===
def s3_list(prefix, delimiter="", attempts=4, count=None, max_pages=200):
    """Every object (or common prefix) under `prefix`. Refuses an endless
    listing; an empty answer is returned empty and the CALLER decides."""
    keys, prefixes, token, pages = [], [], None, 0
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        url = BASE + "?" + urllib.parse.urlencode(q)
        raw, why = cm.get_bytes(url, attempts=attempts)
        if raw is None:
            raise FormatError(f"the bucket listing answered {why} for {url}")
        if count is not None:
            count(len(raw))
        try:
            root = ET.fromstring(raw)
        except ET.ParseError as e:
            raise FormatError(f"{url}: not XML ({e})") from None
        for c in root.findall(S3_NS + "Contents"):
            keys.append((c.findtext(S3_NS + "Key"),
                         int(c.findtext(S3_NS + "Size") or 0),
                         c.findtext(S3_NS + "LastModified")))
        for c in root.findall(S3_NS + "CommonPrefixes"):
            prefixes.append(c.findtext(S3_NS + "Prefix"))
        token = root.findtext(S3_NS + "NextContinuationToken")
        pages += 1
        if not token:
            return keys, prefixes, pages
        if pages > max_pages:
            raise FormatError(f"the listing of {prefix} did not end after "
                              f"{max_pages} pages")


def parse_parts(keys, snapshot):
    """Snapshot objects -> ([(name, key, bytes)], counts). Refuses a
    listing whose shape is not the one measured."""
    pref = f"{PREFIX}{snapshot}/{PARQUET_DIR}"
    parts, other = [], []
    for key, size, _mod in keys:
        if key.startswith(pref):
            name = key[len(pref):]
            if "/" in name or not name:
                other.append(key)
                continue
            parts.append((name, key, int(size)))
        else:
            other.append(key)
    if not parts:
        raise FormatError(
            f"the listing of {PREFIX}{snapshot}/ holds no object under "
            f"{PARQUET_DIR} — an empty listing is a broken listing "
            f"(ml/CLAUDE.md, the 2026-09-14 rule)")
    parts.sort()
    return parts, {"objects": len(keys), "parts": len(parts),
                   "other_objects": sorted(other)[:8],
                   "other_objects_n": len(other)}


def check_geospatial_set(attempts=2):
    """Re-run the falsification of `GEOSPATIAL_ISSUES` against GBIF's API.

    Returns a dict for `plan.json`. A member that occurs WITHOUT a geospatial
    issue is a contradiction and RAISES; an unreachable API is recorded as
    unavailable, because a network failure is not evidence about the set.
    """
    out = {"measured": dict(GEOSPATIAL_MEASURED), "issues": {}}
    try:
        raw, why = cm.get_bytes(f"{GBIF_API}?limit=0&hasGeospatialIssue=true",
                                attempts=attempts)
        if raw is None:
            return {**out, "status": f"unavailable ({why})"}
        total = int(json.loads(raw)["count"])
        out["hasGeospatialIssue_true_now"] = total
        union = "&".join(f"issue={i}" for i in GEOSPATIAL_ISSUES)
        raw, why = cm.get_bytes(f"{GBIF_API}?limit=0&{union}",
                                attempts=attempts)
        if raw is None:
            return {**out, "status": f"unavailable ({why})"}
        out["union_of_the_declared_seven_now"] = int(json.loads(raw)["count"])
        for name in GEOSPATIAL_ISSUES:
            raw, why = cm.get_bytes(
                f"{GBIF_API}?limit=0&issue={name}&hasGeospatialIssue=false",
                attempts=attempts)
            if raw is None:
                return {**out, "status": f"unavailable ({why})"}
            n = int(json.loads(raw)["count"])
            out["issues"][name] = n
            if n:
                raise FormatError(
                    f"{name} is declared a geospatial issue and GBIF reports "
                    f"{n:,} record(s) carrying it with hasGeospatialIssue="
                    f"false — the declared set is wrong and the row filter "
                    f"would drop good records. Re-measure it (the module "
                    f"docstring says how) before building.")
    except (IOError, ValueError, KeyError) as e:
        if isinstance(e, FormatError):
            raise
        return {**out, "status": f"unavailable ({type(e).__name__}: {e})"}
    out["status"] = "verified"
    return out


# ================================================================ parsing ===
def _codes(names, table):
    """A column of strings -> (float64 codes, {unknown name: n}, n_absent)."""
    lookup = {n: float(i) for i, n in enumerate(table)}
    out = np.full(len(names), np.nan, np.float64)
    unknown, absent = {}, 0
    cache = {}
    for i, s in enumerate(names):
        if s is None:
            absent += 1
            continue
        c = cache.get(s)
        if c is None:
            c = lookup.get(s)
            if c is None:
                unknown[s] = unknown.get(s, 0) + 1
                cache[s] = np.nan
                continue
            cache[s] = c
        if c == c:                                   # not NaN
            out[i] = c
        else:
            unknown[s] = unknown.get(s, 0) + 1
    return out, unknown, absent


def _add(counts, key, n):
    if n:
        counts[key] = counts.get(key, 0) + int(n)


def _issue_names(cell):
    """One `issue` cell -> the issue strings.

    The column is `list<struct<array_element: string>>` in this snapshot
    (parquet-mr's two-level list encoding), so pyarrow hands back a list of
    one-key dicts; a plain list of strings is accepted too, because the
    encoding is the writer's choice and not a fact about the data.
    """
    if not cell:
        return ()
    out = []
    for x in cell:
        if isinstance(x, dict):
            out.extend(str(v) for v in x.values() if v is not None)
        elif x is not None:
            out.append(str(x))
    return tuple(out)


def rows_from_table(table, track, counts, t_lo=None, t_hi=None):
    """One part's table -> the columns of this TRACK's rows, everything
    counted.

    Returns (t, lat, lon, values[n, 4], platform, qc, taxa) where `taxa` is
    {platform_hash: {taxonkey, scientificname, kingdom, class}} for the rows
    kept.
    """
    n0 = table.num_rows
    _add(counts, "records_read", n0)
    col = {c: table.column(c).to_pylist() for c in
           ("decimallatitude", "decimallongitude",
            "coordinateuncertaintyinmeters", "year", "month", "day",
            "taxonkey", "scientificname", "kingdom", "class",
            "basisofrecord", "individualcount", "license", "issue")}
    ev = table.column("eventdate").cast("int64").to_pylist()

    lat = np.array([np.nan if x is None else float(x)
                    for x in col["decimallatitude"]], np.float64)
    lon = np.array([np.nan if x is None else float(x)
                    for x in col["decimallongitude"]], np.float64)
    keep = np.isfinite(lat) & np.isfinite(lon)
    _add(counts, "records_no_position", int((~keep).sum()))
    bad = keep & ((np.abs(lat) > 90.0) | (np.abs(lon) > 180.0))
    _add(counts, "records_position_out_of_range", int(bad.sum()))
    keep &= ~bad

    geo = np.zeros(n0, bool)
    seen_issue = {}
    for i, cell in enumerate(col["issue"]):
        names = _issue_names(cell)
        for s in names:
            if s in GEOSPATIAL_SET:
                geo[i] = True
                seen_issue[s] = seen_issue.get(s, 0) + 1
    if seen_issue:
        counts.setdefault("geospatial_issue_by_name", {})
        for k, v in seen_issue.items():
            counts["geospatial_issue_by_name"][k] = \
                counts["geospatial_issue_by_name"].get(k, 0) + v
    _add(counts, "records_geospatial_issue", int((keep & geo).sum()))
    keep &= ~geo

    yr = np.array([-32768 if x is None else int(x) for x in col["year"]],
                  np.int64)
    mo = np.array([0 if x is None else int(x) for x in col["month"]], np.int64)
    dy = np.array([0 if x is None else int(x) for x in col["day"]], np.int64)
    no_month = (yr == -32768) | (mo < 1) | (mo > 12)
    _add(counts, "records_no_month", int((keep & no_month).sum()))
    keep &= ~no_month
    month_only = dy < 1
    baddate = keep & ~month_only & ~cm.valid_date(yr, mo, np.maximum(dy, 1))
    _add(counts, "records_bad_date", int(baddate.sum()))
    keep &= ~baddate

    lic = np.array([x or "" for x in col["license"]], dtype=object)
    tr = np.array([LICENCE_TRACK.get(s, "") for s in lic], dtype=object)
    unknown_lic = keep & (tr == "")
    if unknown_lic.any():
        d = counts.setdefault("records_licence_unknown_by_name", {})
        for s in lic[unknown_lic]:
            d[str(s)] = d.get(str(s), 0) + 1
        _add(counts, "records_licence_unknown", int(unknown_lic.sum()))
    keep &= ~unknown_lic
    other = keep & (tr != track)
    _add(counts, "records_other_track", int(other.sum()))
    keep &= ~other
    for s in np.unique(lic[keep]) if keep.any() else ():
        d = counts.setdefault("records_by_licence", {})
        d[str(s)] = d.get(str(s), 0) + int((keep & (lic == s)).sum())

    idx = np.flatnonzero(keep)
    n = idx.size
    if not n:
        return None
    mo_k, dy_k, yr_k = mo[idx], dy[idx], yr[idx]
    monthly = month_only[idx]
    # the date: a day at its own time of day (noon when none), or the middle
    # of the month
    day_s = cm.days_from_civil(yr_k, mo_k, np.maximum(dy_k, 1)) * 86400
    mlen = np.where((mo_k == 2) & cm.is_leap(yr_k), 29,
                    np.take(np.array(DAYS_IN_MONTH, np.int64), mo_k))
    month_start = cm.days_from_civil(yr_k, mo_k, np.ones_like(mo_k)) * 86400
    t = np.where(monthly, month_start + (mlen * 86400) // 2, day_s + NOON)
    # a time of day out of `eventdate`, only where it agrees with y/m/d
    n_tod = 0
    for j, i in enumerate(idx):
        if monthly[j] or ev[i] is None:
            continue
        tod = int(ev[i]) % 86400000
        if tod == 0:
            continue
        # `eventdate` is milliseconds since 1970 UTC and `day_s` is seconds
        # since 1982; its own DATE must be the row's date, or the two
        # disagree and the year/month/day columns win.
        if int(ev[i]) // 86400000 - cm.EPOCH_DAYS_1970 != \
                int(day_s[j]) // 86400:
            continue
        t[j] = day_s[j] + tod // 1000
        n_tod += 1
    _add(counts, "records_with_time_of_day", n_tod)
    _add(counts, "records_month_only", int(monthly.sum()))

    if t_lo is not None:
        inside = (t >= t_lo) & (t <= t_hi)
        _add(counts, "records_outside_window", int((~inside).sum()))
        if not inside.all():
            idx = idx[inside]
            t = t[inside]
            monthly = monthly[inside]
            n = idx.size
            if not n:
                return None

    lat_k = lat[idx].astype(np.float32)
    lon_k = lon[idx].astype(np.float32)
    cu = np.array([np.nan if col["coordinateuncertaintyinmeters"][i] is None
                   else float(col["coordinateuncertaintyinmeters"][i])
                   for i in idx], np.float64)
    known = np.isfinite(cu) & (cu > 0)
    _add(counts, "records_no_uncertainty", int((~known).sum()))
    clamped = np.clip(np.where(known, cu, UNKNOWN_FOOTPRINT_M),
                      UNCERTAINTY_CLAMP[0], UNCERTAINTY_CLAMP[1])
    _add(counts, "records_uncertainty_clamped",
         int((known & ((cu < UNCERTAINTY_CLAMP[0])
                       | (cu > UNCERTAINTY_CLAMP[1]))).sum()))
    k = np.clip(np.rint(np.log2(clamped)), 0, QC_K_MAX).astype(np.int64)
    qc = (k << QC_K_SHIFT).astype(np.int64)
    qc |= np.where(known, 0, QC_NO_UNCERTAINTY)
    qc |= np.where(monthly, QC_MONTH_ONLY, 0)
    qc = qc.astype(np.uint8)

    vals = np.full((n, len(CHANNELS)), np.nan, np.float64)
    for ci, (names, tbl, key) in enumerate((
            ([col["kingdom"][i] for i in idx], KINGDOMS, "kingdom"),
            ([col["class"][i] for i in idx], CLASSES, "class"),
            ([col["basisofrecord"][i] for i in idx], BASIS_OF_RECORD,
             "basis_of_record"))):
        codes, unknown, absent = _codes(names, tbl)
        vals[:, ci] = codes
        _add(counts, f"{key}_absent", absent)
        if unknown:
            d = counts.setdefault(f"{key}_not_in_table", {})
            for s, c in unknown.items():
                d[s] = d.get(s, 0) + c
            _add(counts, f"{key}_not_in_table_rows", sum(unknown.values()))
    vals[:, 3] = [np.nan if col["individualcount"][i] is None
                  else float(col["individualcount"][i]) for i in idx]

    taxa = {}
    plat = np.empty(n, np.int64)
    unmatched = int(f10b.platform_hash(TAXON_UNMATCHED))
    n_unmatched = 0
    cache = {}
    for j, i in enumerate(idx):
        key = col["taxonkey"][i]
        if not key:
            plat[j] = unmatched
            n_unmatched += 1
            continue
        h = cache.get(key)
        if h is None:
            h = cache[key] = int(f10b.platform_hash(str(key)))
            taxa[h] = {"taxonkey": str(key),
                       "scientificname": col["scientificname"][i],
                       "kingdom": col["kingdom"][i],
                       "class": col["class"][i]}
        plat[j] = h
    if n_unmatched:
        _add(counts, "records_taxon_unmatched", n_unmatched)
        taxa[unmatched] = {"taxonkey": None, "scientificname": None,
                           "kingdom": None, "class": None,
                           "note": "the pooled platform for records whose "
                                   "taxon did not match GBIF's backbone"}
    _add(counts, "rows_kept", n)
    return t, lat_k, lon_k, vals, plat, qc, taxa


# ================================================================ adapter ===
class GBIFBase(f10b.SourceAdapter):
    """The two tracks' common body; `track` picks the side of the licence."""

    track = "public"
    family = "1tf"
    time_dtype = "int64"
    platform_meta = True
    credentials = ()
    channels = CHANNELS
    # the "unknown" footprint the design names, 1 km over the family's cell
    log2_fp = float(np.log2(UNKNOWN_FOOTPRINT_M / 1000.0 / REFERENCE_KM))
    log2_dt = float(np.log2(1.0 / 5.0))              # one day; see qc bit 1
    per_year = False
    first_year = 1600
    fetch_month_scope = "one_part"
    qc_policy = (
        "qc IS A BITFIELD FOR THIS STORE, NOT A GRADE, and --qc-keep must "
        "not be read as one. bit 0 (1): the record states no coordinate "
        "uncertainty, so the footprint is the store's 1 km 'unknown' "
        "constant. bit 1 (2): the date is month-only, so the row's real time "
        "footprint is log2(30.44/5) = +2.606 rather than the store's -2.322. "
        "bits 2-7: k = round(log2(coordinate uncertainty in metres)) clamped "
        "to [0, 20], stored as k << 2, so the row's real log2_fp is "
        "k - log2(1000 * 27.83) = k - 14.7639; k is 10 (1,024 m) whenever "
        "bit 0 is set. The uncertainty itself is clamped to [10 m, 100 km] "
        "before k is taken and the clamping is counted. Records DROPPED and "
        "counted: no position, a position out of range, any of the seven "
        "measured geospatial issues, no year or no month, an impossible "
        "date, a licence outside GBIF's three (an unknown licence never "
        "reaches a public store), and the other track's licence. A kingdom, "
        "class or basis-of-record name absent from its code table is NaN for "
        "that channel and counted BY NAME; an individual count above 60,000 "
        "is out of bounds and NaN, because float16 stops being exact at "
        "2,048 and overflows to infinity at 65,504.")
    sources = (BASE + "?list-type=2&prefix=" + PREFIX,
               BASE + PREFIX + "<YYYY-MM-01>/" + PARQUET_DIR + "<part>",
               GBIF_API + " (the geospatial-issue falsification only)")
    verified = (
        "2026-09-20 from this sandbox, anonymously: the bucket's snapshot "
        "listing (65 snapshots, 2021-04-13 .. 2026-09-01); the 2026-09-01 "
        "snapshot in full (ten pages, 9,899 objects, 9,898 parquet parts, "
        "285,323,189,332 bytes); part 005147 read with pyarrow (383,269 "
        "rows, one row group, 50 columns, parquet-mr 1.9.0) and its whole "
        "column-size table, from which the fifteen columns a row needs are "
        "22.1 % of the compressed bytes; a projected read of that part over "
        "HTTP Range (9 requests, 6,092,259 bytes, 5.1 s); the kingdom, class, "
        "basis-of-record and licence vocabularies over 233 sampled parts "
        "(88 M rows); and GBIF's own search API for the record counts "
        "(3,960,167,524 occurrences; 3,782,087,285 with coordinates and no "
        "geospatial issue; 3,668,229,700 of those with a year, split CC BY "
        "72.57 % / CC BY-NC 16.38 % / CC0 11.05 %) and for the "
        "seven-name geospatial-issue set.")
    smoke_window = ("1899-12-01", "1900-12-31")
    smoke_probe_month = "1900-01"

    def __init__(self):
        self.snapshot = (os.environ.get("GBIF_SNAPSHOT") or "").strip()
        self.parts_spec = (os.environ.get("GBIF_PARTS") or "").strip()
        try:
            self.probe_n = int(os.environ.get("GBIF_PROBE_PARTS") or 8)
        except ValueError:
            sys.exit("GBIF_PROBE_PARTS must be an integer")
        self._listing = None
        self._taxa = {}
        self._taxa_path = None
        # the part NAMES `GBIF_PARTS` resolved to (`wanted_parts`), and the
        # platforms each year's rows carried in this fetch (`fetch_stream`),
        # which is what each year-lane's taxon sidecar is cut from
        self._wanted_names = None
        self._year_plats = {}
        self.notes = (
            "Two tracks decided by the row's own `license` column: CC0 and "
            "CC BY 4.0 to the public store `gbif`, CC BY-NC 4.0 to the "
            "private sibling `gbif_nc`. BOTH TRACKS MUST BE BUILT FROM THE "
            "SAME SNAPSHOT (GBIF_SNAPSHOT), or they are two different months "
            "of the archive. The snapshot has no time axis, so `per_year` is "
            "False and the resumable unit is the whole pass; GBIF_PARTS is "
            "the only way to lane it — each `lo:hi` range is a g-<hash> "
            "group lane, and an assembly from N such lanes declares them "
            "with --lanes parts:N. Each part is read by COLUMN "
            "PROJECTION over HTTP Range requests — 22.1 % of its bytes, "
            "measured. qc is a BITFIELD here (see qc_policy): the layout "
            "carries one (log2_fp, log2_dt) pair per store and `check_store` "
            "asserts it, so each row's own footprint rides in qc's upper six "
            "bits and its month-only flag in bit 1."
            + (f" SNAPSHOT PINNED: GBIF_SNAPSHOT={self.snapshot}."
               if self.snapshot else "")
            + (f" RESTRICTED BUILD: GBIF_PARTS={self.parts_spec!r} — this "
               f"store is NOT the whole snapshot." if self.parts_spec else ""))

    # ------------------------------------------------------------- listing --
    def listing(self, ctx):
        """(snapshot, [(name, key, bytes)], counts), listed once."""
        if self._listing is not None:
            return self._listing
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, "gbif", "listing.json")
            if not os.path.exists(p):
                sys.exit(f"REFUSING {self.store}: no {p} — the smoke's "
                         f"synthetic bucket listing is missing")
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
            js = json.loads(raw)
            snap = self.snapshot or js["snapshots"][-1]
            keys = [(k["key"], int(k["size"]), "") for k in js["keys"]]
        else:
            snap = self.snapshot
            if not snap:
                _k, prefixes, _p = s3_list(PREFIX, delimiter="/",
                                           attempts=ctx.a.attempts,
                                           count=ctx.count_bytes)
                snaps = sorted(m.group(1) for m in
                               (SNAPSHOT_RE.match(p) for p in prefixes) if m)
                if not snaps:
                    sys.exit(f"REFUSING {self.store}: the bucket lists no "
                             f"{PREFIX}<YYYY-MM-DD>/ snapshot at all — an "
                             f"empty listing is a refusal")
                snap = snaps[-1]
            keys, _pref, _pages = s3_list(f"{PREFIX}{snap}/",
                                          attempts=ctx.a.attempts,
                                          count=ctx.count_bytes)
        try:
            parts, counts = parse_parts(keys, snap)
        except FormatError as e:
            sys.exit(f"REFUSING {self.store}: {e}")
        counts["snapshot"] = snap
        self._listing = (snap, parts, counts)
        return self._listing

    def wanted_parts(self, ctx):
        """The parts this run reads, after GBIF_PARTS."""
        _snap, parts, _c = self.listing(ctx)
        spec = self.parts_spec
        if not spec:
            return parts
        if ":" in spec:
            a, _, b = spec.partition(":")
            try:
                lo = int(a or 0)
                hi = int(b) if b else len(parts)
            except ValueError:
                sys.exit(f"GBIF_PARTS={spec!r}: expected `lo:hi` or a comma "
                         f"list of part names")
            out = parts[lo:hi]
        else:
            want = {s.strip() for s in spec.split(",") if s.strip()}
            out = [p for p in parts if p[0] in want]
            missing = sorted(want - {p[0] for p in out})
            if missing:
                sys.exit(f"REFUSING {self.store}: GBIF_PARTS names part(s) "
                         f"the snapshot does not list: {missing[:8]}")
        self._wanted_names = sorted(p[0] for p in out)
        return out

    # --------------------------------------------------------------- lanes --
    # A PART RANGE IS A GROUP LANE (2026-09-24). The whole pass was the
    # resumable unit — 23 h on one box, and family1-build #837 lost nine hours
    # of it when its runner lost contact — so the snapshot is split by part
    # range across GitHub-hosted runners, each one `GBIF_PARTS=lo:hi`. The
    # lane's GROUPS are the part NAMES it reads, so it is named
    # `g-<sha1 of the sorted names>` by the lane machinery that already names
    # canopy30's tile subsets (`build_family1_stores.group_lane`), writes
    # `parts/<year>/<lane>/` for every year of its window, and carries the
    # names in each year's `counts.json` as `lane_groups` — so N part lanes of
    # one year are DISJOINT to `lanes_preflight` and assemble as one year.
    #
    # WHY `group_subset` TAKES A CONTEXT. `lo:hi` is an index into the
    # listing, and the listing needs the context (`--source-dir`, attempts),
    # while `apply_lane` runs in `main` before any stage. `apply_lane` now
    # hands the context to a hook that declares one (`adapter_group_subset`),
    # so the lane is known BEFORE the index and the fetch, the listing is made
    # once and cached, and no stage has to re-derive a lane afterwards.
    def group_subset(self, ctx=None):
        """The sorted part NAMES `GBIF_PARTS` restricts this run to, or None
        (the whole snapshot — the unnamed lane, exactly as before)."""
        if not self.parts_spec:
            return None
        if ctx is not None:
            self.wanted_parts(ctx)
        elif self._wanted_names is None:
            sys.exit(f"{self.store}: GBIF_PARTS={self.parts_spec!r} names its "
                     f"lane only once the snapshot is listed — call "
                     f"group_subset(ctx)")
        if not self._wanted_names:
            # An empty range would read as NO subset and write the UNNAMED
            # lane — a whole-snapshot claim over zero parts.
            sys.exit(f"REFUSING {self.store}: GBIF_PARTS={self.parts_spec!r} "
                     f"selects no part of the snapshot's "
                     f"{len(self.listing(ctx)[1]) if ctx is not None else '?'}"
                     f" — a lane over nothing would claim the whole snapshot")
        return list(self._wanted_names)

    def lanes_for_parts(self, ctx, n):
        """`--lanes parts:<N>`: the N lane names of the snapshot split into
        N contiguous part ranges (`part_lanes`), for plan.json's
        `lanes_expected`. Declared from the WHOLE listing, so a declaring run
        with `GBIF_PARTS` set is a contradiction and is refused."""
        if self.parts_spec:
            sys.exit(f"--lanes parts:{n}: GBIF_PARTS={self.parts_spec!r} is "
                     f"set — the run that DECLARES the part lanes is the "
                     f"assembly of all of them; unset GBIF_PARTS there")
        snap, parts, _c = self.listing(ctx)
        try:
            lanes = part_lanes(parts, n)
        except ValueError as e:
            sys.exit(f"--lanes parts:{n}: {e}")
        print(f"  {self.store}: snapshot {snap}, {len(parts):,} parts in "
              f"{n} part-range lane(s): "
              + ", ".join(f"{lo}:{hi}={name}" for lo, hi, name in lanes[:4])
              + (" …" if len(lanes) > 4 else ""))
        return [name for _lo, _hi, name in lanes]

    def probe_part(self, ctx):
        """The ONE part `index` opens to check the schema: the middle of the
        listing, or whatever GBIF_PARTS names first."""
        parts = self.wanted_parts(ctx)
        if not parts:
            sys.exit(f"REFUSING {self.store}: no part to probe")
        return parts[len(parts) // 2] if not self.parts_spec else parts[0]

    def probe_parts(self, ctx):
        """The parts `--stage probe` measures, EVENLY SPACED.

        ONE PART WAS THE DESIGN AND ONE PART IS NOT ENOUGH, and that is a
        measurement rather than a preference. The snapshot's parts follow the
        publishers, so a part can be one publisher's whole dataset under one
        licence: part 005210 of the 2026-09-01 snapshot — the middle of the
        listing, which is what `probe_part` picks — holds 585,040 records of
        which **585,034 are CC BY-NC**, so a one-part probe of the PUBLIC
        store measures six rows and a one-part probe of either store
        extrapolates a licence split that does not exist. `GBIF_PROBE_PARTS`
        (default 8) spreads the measurement evenly across the listing; the
        per-part breakdown stays in `per_part` so the spread between them is
        visible rather than averaged away.
        """
        parts = self.wanted_parts(ctx)
        if not parts:
            sys.exit(f"REFUSING {self.store}: no part to probe")
        n = min(max(1, self.probe_n), len(parts))
        step = len(parts) / float(n)
        idx = sorted({min(len(parts) - 1, int(i * step + step / 2))
                      for i in range(n)})
        return [parts[i] for i in idx]

    # ---------------------------------------------------------------- read --
    def _read(self, ctx, part, want_meta=False):
        name, key, size = part
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, "gbif", key)
            if not os.path.exists(p):
                raise f10b._NotFound(p)
            return read_part(p, None, count=ctx.count_bytes,
                             want_meta=want_meta)
        return read_part(BASE + urllib.parse.quote(key), size,
                         attempts=max(1, ctx.a.attempts),
                         count=None, want_meta=want_meta)

    def _taxa_file(self, ctx):
        # `ctx.root` already names this store (<work>/<store>); `ctx.store` is
        # the ASSEMBLED store directory, and a file written there would sit
        # beside the arrays without being in store.json's sha256 block.
        return os.path.join(ctx.root, "taxa.json")

    def _flush_taxa(self, ctx):
        """Flush, THEN mark (§5.21): the table is on disk before the part
        that produced it is counted as read."""
        p = self._taxa_file(ctx)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump({str(k): v for k, v in sorted(self._taxa.items())}, fh,
                      separators=(",", ":"))
        os.replace(tmp, p)

    def _parts_rows(self, ctx, parts, label, t_lo, t_hi, counts):
        """Yield packed rows for each part in turn; counts accumulate."""
        t0 = time.time()
        per_part = []
        for name, key, size in parts:
            p0, b0 = time.time(), f10b.NET_BYTES["n"]
            try:
                table, stat = self._read(ctx, (name, key, size))
            except f10b._NotFound:
                ctx.note_absent(label, f"{key}: listed in the bucket, and 404")
                continue
            except FormatError as e:
                sys.exit(f"REFUSING {self.store}: {key}: {e}")
            except IOError as e:
                ctx.note_absent(label, f"{key}: {type(e).__name__}: {e}")
                continue
            c = {}
            got = rows_from_table(table, self.track, c, t_lo, t_hi)
            del table
            f10b._merge_counts(counts, c)
            per_part.append({"part": name, "bytes": stat["bytes"],
                             "requests": stat["requests"],
                             "rows_in_part": stat["rows"],
                             "rows_kept": int(c.get("rows_kept", 0)),
                             "seconds": round(time.time() - p0, 2),
                             "net_bytes": int(f10b.NET_BYTES["n"] - b0)})
            _add(counts, "parts_read", 1)
            if got is None:
                continue
            t, lat, lon, vals, plat, qc, taxa = got
            oob = self.mask_bounds(vals)
            if oob:
                f10b._merge_counts(counts, {"out_of_bounds": oob})
            self._taxa.update(taxa)
            # FLUSH EVERY `TAXA_FLUSH_PARTS` PARTS, NOT EVERY PART. The file
            # is rewritten whole, and it grows with the taxa: rewriting a
            # table of millions of entries after each of 9,898 parts is
            # quadratic and would cost more than the fetch. Every flush still
            # precedes the rows it describes reaching a part file (§5.21:
            # flush, THEN mark), because a part is only counted as read once
            # this generator has yielded past it.
            if len(per_part) % TAXA_FLUSH_PARTS == 0:
                self._flush_taxa(ctx)
            yield t, lat, lon, vals, plat, qc
        self._flush_taxa(ctx)
        counts["fetch_seconds"] = round(time.time() - t0, 1)
        counts["per_part"] = per_part[:40]
        counts["taxa_seen"] = len(self._taxa)

    # ------------------------------------------------------------ contract --
    def index(self, ctx):
        snap, parts, counts = self.listing(ctx)
        mine = self.wanted_parts(ctx)
        nb = sum(p[2] for p in parts)
        out = {"dataset": f"GBIF occurrence snapshot {snap} (parquet, AWS "
                          f"Open Data registry)",
               "url": BASE + PREFIX + snap + "/",
               "snapshot": snap,
               "track": self.track,
               "parts": len(parts),
               "parts_this_run": len(mine),
               "parts_spec": self.parts_spec,
               "bytes": int(nb),
               "bytes_per_part_mean": (round(nb / len(parts), 1) if parts
                                       else None),
               "columns_read": list(COLUMNS),
               "code_tables": {"kingdoms": len(KINGDOMS),
                               "classes": len(CLASSES),
                               "basis_of_record": len(BASIS_OF_RECORD)},
               "licence_tracks": dict(LICENCE_TRACK),
               "listing": counts}
        if not ctx.source_dir:
            out["geospatial_issue_check"] = check_geospatial_set(
                attempts=max(1, min(2, ctx.a.attempts)))
        else:
            out["geospatial_issue_check"] = {
                "status": "skipped (--source-dir)",
                "measured": dict(GEOSPATIAL_MEASURED)}
        # ONE REAL PART, read and checked against the declared schema.
        part = self.probe_part(ctx)
        try:
            table, stat = self._read(ctx, part, want_meta=True)
        except (f10b._NotFound, IOError) as e:
            sys.exit(f"REFUSING {self.store}: {part[1]} is listed and cannot "
                     f"be read ({e})")
        except FormatError as e:
            sys.exit(f"REFUSING {self.store}: {part[1]}: {e}")
        c = {}
        got = rows_from_table(table, self.track, c)
        del table
        # THE CODE TABLES ARE A HYPOTHESIS AND THIS IS WHERE IT IS FALSIFIED
        # (ADAPTER_CONTRACT rule 4, the shape `lossyear` uses for its tile
        # tuple): one real part is read and the class names it carries that
        # the table cannot code are NAMED, so a snapshot whose vocabulary has
        # moved is visible in plan.json instead of arriving as a column of
        # NaN nobody looked at.
        names = sorted((c.get("class_not_in_table") or {}))
        out["first_part"] = {"part": part[0], "listed_bytes": part[2],
                             **stat,
                             "rows_this_track": (0 if got is None
                                                 else int(got[0].size)),
                             "counts": c,
                             "class_names_not_in_table": names[:20],
                             "class_names_not_in_table_n": len(names)}
        return out

    def fetch_stream(self, ctx):
        """ONE PASS over the snapshot's parts; rows routed to their year."""
        snap, parts, _c = self.listing(ctx)
        mine = self.wanted_parts(ctx)
        counts = {"snapshot": snap, "parts_listed": len(parts),
                  "parts_wanted": len(mine), "track": self.track}
        self._year_plats = {}
        for t, lat, lon, vals, plat, qc in self._parts_rows(
                ctx, mine, snap, ctx.t_lo, ctx.t_hi, counts):
            years = years_of(t)
            for y in np.unique(years):
                m = years == y
                self._note_year_platforms(int(y), plat[m])
                # AN INT, NOT A STRING: `stage_fetch`'s one-stream branch
                # tests `year in ctx.years`, and ctx.years holds ints — a
                # string year makes it close a SECOND writer over the same
                # directory and write a counts.json claiming zero parts.
                yield int(y), self.pack(t[m], lat[m], lon[m], vals[m],
                                        plat[m], qc[m]), None
        counts["note_estimate"] = [NOTE_ESTIMATE]
        yield None, None, counts

    def fetch_month(self, ctx, year, month):
        """PART FILES, and the month argument is IGNORED — see the docstring.

        The framework's probe filters what it is handed to the month it was
        asked for, so `rows` and `rows_per_day` describe that month; the
        measurement the store is SIZED from is in `counts` under `probe_*`,
        because a GBIF part holds records from the 1600s to last week and no
        part is a month. `probe_parts` says why the default reads eight
        evenly spaced parts rather than the one the design asked for.
        """
        snap, parts, _c = self.listing(ctx)
        mine = self.probe_parts(ctx)
        counts = {"snapshot": snap, "parts_listed": len(parts),
                  "track": self.track,
                  "probe_part": ",".join(p[0] for p in mine),
                  "probe_parts_n": len(mine),
                  "probe_part_listed_bytes": sum(p[2] for p in mine),
                  "probe_month_argument_ignored": (
                      f"{year}-{int(month):02d}: the GBIF snapshot is not "
                      f"partitioned by time, so one part holds records from "
                      f"the 1600s to last week and no part is a month. The "
                      f"probe measures the PARTS; the month only selects "
                      f"which rows the framework's own per-day counters "
                      f"describe.")}
        t0, b0 = time.time(), f10b.NET_BYTES["n"]
        label = f"{snap} parts {mine[0][0]}..{mine[-1][0]}"
        batches = []
        for t, lat, lon, vals, plat, qc in self._parts_rows(
                ctx, mine, label, None, None, counts):
            batches.append(self.pack(t, lat, lon, vals, plat, qc))
        counts["probe_seconds"] = round(time.time() - t0, 2)
        counts["probe_bytes"] = int(f10b.NET_BYTES["n"] - b0) or \
            int(sum(p["bytes"] for p in counts.get("per_part", [])))
        counts["probe_rows_in_part"] = int(counts.get("records_read", 0))
        counts["probe_rows_kept"] = int(counts.get("rows_kept", 0))
        counts["probe_rows_by_licence"] = dict(
            counts.get("records_by_licence", {}))
        kept = int(counts.get("rows_kept", 0))
        counts["probe_share_day_precision"] = (
            None if not kept else
            round(1.0 - int(counts.get("records_month_only", 0)) / kept, 6))
        counts["probe_share_with_uncertainty"] = (
            None if not kept else
            round(1.0 - int(counts.get("records_no_uncertainty", 0)) / kept,
                  6))
        # A LIST OF ONE, not a dict: `f10b._merge_counts` merges a dict ONE
        # level deep and ADDS its values, so a nested dict carrying a string
        # raises there. A list concatenates, and only this one yield makes it
        # (the shape `swot` uses for `storage_options`).
        proj = _projection(counts, len(parts), self.C, self.time_dtype)
        counts["probe_projection"] = [proj] if proj else []
        counts["note_estimate"] = [NOTE_ESTIMATE]
        rows = cm.concat_rows(batches, self.C, self.time_dtype)
        yield from cm.batch_rows(self, rows, counts, label)

    # ------------------------------------------------------- taxon sidecars --
    def _note_year_platforms(self, year, plat):
        """Remember which platforms `year`'s rows carried, as sorted int64
        arrays compacted as they grow — eight bytes a (taxon, year) pair
        rather than a Python set's sixty."""
        got = self._year_plats.setdefault(int(year), [])
        got.append(np.unique(np.asarray(plat, np.int64)))
        if len(got) > 32:
            self._year_plats[int(year)] = [np.unique(np.concatenate(got))]

    def year_platforms(self, year):
        got = self._year_plats.get(int(year))
        if not got:
            return np.empty(0, np.int64)
        return np.unique(np.concatenate(got))

    def finish_fetch(self, ctx, year_dirs):
        """THE TAXON TABLE TRAVELS WITH THE PARTS (2026-09-24).

        `platforms(ctx)` runs at ASSEMBLE time and the taxa are discovered
        during the FETCH, so a box assembling part-range lanes pulled from the
        Hub (`--parts-from-hub`) has no `<work>/<store>/taxa.json` of its own.
        Each year-lane directory therefore gets a `taxa.json` SIDECAR
        (`family10_parts_hub.SIDECAR_NAMES`): pushed, restore-verified, listed
        in `done.json` and pulled like a part, and never read as one.

        EACH DIRECTORY GETS THE TAXA ITS OWN ROWS CARRY, NOT THE WHOLE TABLE.
        A lane's window is 1600..2026, i.e. 427 year directories, and the
        table is the one thing here that scales with the archive (a few
        million taxa over the snapshot): a full copy in every directory would
        be hundreds of copies of the largest file the lane writes, on a
        runner with ~86 GB of disk. A per-year subset is exactly what that
        year's parts need, the union over any set of years and lanes is
        exactly the table those rows need, and a year with no row gets no
        sidecar. Called by `stage_fetch` BEFORE the years are marked, so a
        marked year always carries its sidecar (§5.21, flush THEN mark).
        """
        for y, d in sorted(year_dirs.items()):
            plats = self.year_platforms(y)
            if not plats.size:
                continue
            table = {}
            for h in plats.tolist():
                rec = self._taxa.get(int(h))
                if rec is None:
                    raise ValueError(
                        f"{self.store}: year {y} holds platform {h} and the "
                        f"fetch's taxon table has no record of it — refusing "
                        f"to mark a year whose sidecar would be short")
                table[str(int(h))] = rec
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, TAXA_SIDECAR)
            tmp = f"{p}.tmp{os.getpid()}"
            with open(tmp, "w") as fh:
                json.dump(table, fh, separators=(",", ":"))
            os.replace(tmp, p)

    def platforms(self, ctx):
        """The taxa this build met: the fetch's own table, UNIONED with every
        year-lane sidecar of the years being assembled.

        Sources, all optional, at least one required: the table in memory
        when this process ran the fetch, else `<work>/<store>/taxa.json`; and
        `parts/<year>[/<lane>]/taxa.json` for every year of the window and
        every lane of it on disk (`ctx.lanes_of`) — which is how a box that
        pulled N part-range lanes from the Hub gets the whole table. A taxon
        key maps to one record by construction (the platform is a hash of the
        key), so two sources that both know a key must agree, and a
        disagreement is REFUSED by name rather than resolved by picking one.
        The same pass refuses parts from two different SNAPSHOTS (their
        ledgers' `snapshot`, and plan.json's): lanes dispatched without a
        pinned GBIF_SNAPSHOT across a monthly release would otherwise
        assemble two months of the archive into one store.
        """
        sources = []
        if self._taxa:
            sources.append(("this process's fetch", self._taxa))
        else:
            p = self._taxa_file(ctx)
            if os.path.exists(p):
                with open(p, "rb") as fh:
                    sources.append((p, json.load(fh)))
        snaps = {}
        plan = os.path.join(ctx.root, "plan.json")
        if os.path.exists(plan):
            with open(plan, "rb") as fh:
                s = json.load(fh).get("snapshot")
            if s:
                snaps.setdefault(str(s), plan)
        for y in ctx.years:
            for lane in ctx.lanes_of(y):
                d = ctx.year_dir(y, lane)
                p = os.path.join(d, TAXA_SIDECAR)
                if os.path.exists(p):
                    with open(p, "rb") as fh:
                        sources.append((p, json.load(fh)))
                cp = os.path.join(d, "counts.json")
                if os.path.exists(cp):
                    with open(cp, "rb") as fh:
                        s = (json.load(fh).get("counts") or {}).get("snapshot")
                    if s:
                        snaps.setdefault(str(s), cp)
        if len(snaps) > 1:
            raise ValueError(
                f"{self.store}: the parts being assembled come from "
                f"{len(snaps)} different GBIF snapshots — "
                + "; ".join(f"{s} ({w})" for s, w in sorted(snaps.items()))
                + ". Every lane and the assembling box must pin the SAME "
                  "GBIF_SNAPSHOT.")
        if not sources:
            raise ValueError(
                f"{self.store}: platforms.json needs the taxon table the "
                f"FETCH writes ({self._taxa_file(ctx)}, or a {TAXA_SIDECAR} "
                f"sidecar in the year-lane directories under {ctx.parts}), "
                f"and none is there. The taxa are discovered while the parts "
                f"are read, so parts pushed by a builder older than the "
                f"sidecar (2026-09-24) cannot write platforms.json on a "
                f"machine that did not run the fetch. Re-fetch those lanes, "
                f"or assemble where the parts were fetched.")
        out = {}
        for where, table in sources:
            for k, rec in table.items():
                k = int(k)
                have = out.get(k)
                if have is None:
                    out[k] = rec
                elif have != rec:
                    raise ValueError(
                        f"{self.store}: platform {k} (taxon key "
                        f"{rec.get('taxonkey')!r}) has two different records "
                        f"— {have!r} and {rec!r} (the second from {where}). "
                        f"Refusing to pick one.")
        return out

    def smoke_sources(self, root, d_lo, d_hi, seed=20260920):
        truth = make_smoke_sources(root, d_lo, d_hi, self.track, seed)
        self.__init__()
        return truth


def years_of(t):
    """The calendar year of every second since 1982-01-01, vectorised.

    Howard Hinnant's `civil_from_days`, the inverse of the `days_from_civil`
    in `_common.py`, and vectorised for the same reason: a Python loop over a
    part's 383,269 rows times the snapshot's 9,898 parts is an hour of
    arithmetic nobody needs.
    """
    z = np.floor_divide(np.asarray(t, np.int64), 86400) \
        + cm.EPOCH_DAYS_1970 + 719468
    era = np.floor_divide(np.where(z >= 0, z, z - 146096), 146097)
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    return y + (mp >= 10)


def part_ranges(n_parts, n):
    """`n` contiguous `(lo, hi)` index ranges over `n_parts` parts: equal
    size `n_parts // n`, the LAST one taking the remainder. Refuses an `n`
    that would leave a range empty."""
    n_parts, n = int(n_parts), int(n)
    if n < 1 or n > n_parts:
        raise ValueError(f"cannot split {n_parts} part(s) into {n} non-empty "
                         f"range(s)")
    size = n_parts // n
    return [(i * size, (i + 1) * size if i < n - 1 else n_parts)
            for i in range(n)]


def part_lanes(parts, n):
    """The N part-range lanes of a snapshot: `[(lo, hi, lane_name)]`.

    `parts` is the listing as `listing()` returns it — `(name, key, bytes)`
    tuples — or bare part names, in LISTING ORDER (sorted, as `parse_parts`
    sorts them). `lo:hi` is the `GBIF_PARTS` value a hosted lane is dispatched
    with, and `lane_name` is `build_family1_stores.group_lane` of the names in
    that range — the very name that lane's `apply_lane` will derive, and the
    name `--lanes parts:<N>` declares for the assembly. Every part is in
    exactly one range.

    The dispatcher, from the sandbox (one anonymous S3 listing):
        cd ml && python3 -m family1.adapters._gbif --lanes 16 \\
            [--snapshot 2026-09-01]
    """
    import build_family1_stores as b1          # not at import: b1 imports us
    names = [p[0] if isinstance(p, (tuple, list)) else str(p) for p in parts]
    if names != sorted(set(names)):
        raise ValueError("the part names are not the listing's sorted, "
                         "distinct names")
    return [(lo, hi, b1.group_lane(names[lo:hi]))
            for lo, hi in part_ranges(len(names), n)]


def main(argv=None):
    """Print the N `GBIF_PARTS=lo:hi` specs of a snapshot and their lane
    names — what a dispatcher needs to launch N hosted part lanes and the one
    assembly that declares them."""
    import argparse
    import types
    from family1.adapters import gbif as _pub
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--lanes", type=int, required=True,
                    help="how many part-range lanes")
    ap.add_argument("--snapshot", default=os.environ.get("GBIF_SNAPSHOT", ""),
                    help="YYYY-MM-DD (default GBIF_SNAPSHOT, else the newest "
                         "the bucket lists)")
    ap.add_argument("--source-dir", default="",
                    help="read listing.json from here (tests) instead of S3")
    ap.add_argument("--json", action="store_true",
                    help="print the lanes as JSON")
    a = ap.parse_args(argv)
    ad = _pub.GBIFAdapter()
    ad.snapshot, ad.parts_spec = a.snapshot.strip(), ""
    ctx = types.SimpleNamespace(
        source_dir=os.path.abspath(a.source_dir) if a.source_dir else None,
        a=types.SimpleNamespace(attempts=4), count_bytes=lambda n: None)
    snap, parts, _c = ad.listing(ctx)
    lanes = part_lanes(parts, a.lanes)
    out = [{"lane": i + 1, "GBIF_PARTS": f"{lo}:{hi}", "lane_name": name,
            "first_part": parts[lo][0], "last_part": parts[hi - 1][0],
            "n_parts": hi - lo} for i, (lo, hi, name) in enumerate(lanes)]
    if a.json:
        print(json.dumps({"snapshot": snap, "parts": len(parts),
                          "lanes": out}, indent=1))
        return out
    print(f"snapshot {snap} · {len(parts):,} parts · {a.lanes} lane(s). Pin "
          f"GBIF_SNAPSHOT={snap} on EVERY lane and on the assembling box.")
    for r in out:
        print(f"  lane {r['lane']:>3}  GBIF_PARTS={r['GBIF_PARTS']:<12} "
              f"{r['lane_name']}  parts {r['first_part']}..{r['last_part']} "
              f"({r['n_parts']})")
    print(f"each lane:  --store gbif --stage index,fetch --end 2026-12-31 "
          f"--push-parts   (no --start: the whole record, so the lane name "
          f"is the g-<hash> alone)")
    print(f"the box:    --store gbif --stage all --parts-from-hub "
          f"--end 2026-12-31 --lanes parts:{a.lanes}   (GBIF_PARTS unset)")
    return out


def _projection(counts, n_parts, C, time_dtype):
    """The whole snapshot, from the parts the probe read.

    Every input is in `counts`, and the spread BETWEEN the parts is left in
    `per_part` rather than averaged away: parts follow publishers, so their
    keep fractions differ by orders of magnitude and a mean is the number
    least able to say so.
    """
    read = int(counts.get("records_read", 0))
    kept = int(counts.get("rows_kept", 0))
    per_part = [p for p in (counts.get("per_part") or [])]
    n_read = max(1, int(counts.get("parts_read", len(per_part) or 1)))
    nb = sum(p["bytes"] for p in per_part) or None
    if not read:
        return None
    row = f10b.row_bytes(C, time_dtype)
    keeps = sorted(p["rows_kept"] / p["rows_in_part"] for p in per_part
                   if p.get("rows_in_part"))
    return {
        "parts_in_snapshot": int(n_parts),
        "parts_read": n_read,
        "rows_read": read,
        "rows_per_part": round(read / n_read, 1),
        "kept_per_part": round(kept / n_read, 1),
        "keep_fraction": round(kept / read, 6),
        "keep_fraction_per_part": [round(k, 6) for k in keeps],
        "projected_rows": int(round(kept / n_read * n_parts)),
        "stored_bytes_per_row": row,
        "projected_store_bytes": int(round(kept / n_read * n_parts * row)),
        "projected_fetch_bytes": (int(round(nb / n_read * n_parts))
                                  if nb else None),
        "basis": ("the probe's parts' kept rows, averaged per part and "
                  "multiplied by the snapshot's part count. Parts differ in "
                  "size (0.01-102 MB, median 27.2 MB) AND in licence (part "
                  "005210 of the 2026-09-01 snapshot is 585,034 CC BY-NC of "
                  "585,040 records), so read keep_fraction_per_part before "
                  "trusting the mean: this is a first order estimate, not a "
                  "census"),
    }


# ================================================================== smoke ==
# A synthetic snapshot in the bucket's real shape: a `listing.json` holding
# what ListObjectsV2 returns, a `citation.txt`, and three real parquet parts
# written with pyarrow under the snapshot's own column names.
#
# THE PARTS ARE LAID OUT SO THE PROBE'S RULE IS EXERCISED RATHER THAN WORKED
# AROUND. `--stage probe` reads ONE part — the middle of the listing — while
# the fetch reads all three, and `build_family1_stores.run_smoke` asserts the
# probe's row count equals the truth's rows in `smoke_probe_month`. So part
# 000002 holds every record of January 1900 and nothing else, part 000001
# holds December 1899 and part 000003 June 1900: the two counts agree by
# construction, the store still spans two years, and the fetch still meets
# three parts.
#
# THE EXPECTED ROW OF EVERY RECORD IS WRITTEN OUT BY HAND BELOW, by a second
# implementation of the rules (`_expect`) that shares no code with
# `rows_from_table`. That is the point of a smoke: a typo in one of the two
# does not match the other.
SMOKE_SNAPSHOT = "1900-01-01"
SMOKE_PARTS = ("000001", "000002", "000003")
SMOKE_LAT, SMOKE_LON = 51.5, -0.12


def _expect(rec, track):
    """The row `rec` must become for `track`, or None. Written longhand.

    A second implementation of the row rules, on purpose. It takes the
    record exactly as the parquet part carries it and returns the store row
    with no reference to the adapter's own parser.
    """
    import math
    lat, lon = rec["decimallatitude"], rec["decimallongitude"]
    if lat is None or lon is None:
        return None
    if abs(lat) > 90.0 or abs(lon) > 180.0:
        return None
    for name in (rec["issue"] or ()):
        if name in GEOSPATIAL_ISSUES:
            return None
    y, m, d = rec["year"], rec["month"], rec["day"]
    if y is None or m is None or not (1 <= m <= 12):
        return None
    lic = rec["license"]
    if LICENCE_TRACK.get(lic) != track:
        return None
    mlen = [0, 31, 29 if (y % 4 == 0 and (y % 100 or y % 400 == 0)) else 28,
            31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m]
    if d is not None and not (1 <= d <= mlen):
        return None
    epoch = dt.date(1982, 1, 1)
    if d is None:
        t = (dt.date(y, m, 1) - epoch).days * 86400 + (mlen * 86400) // 2
        month_only = True
    else:
        t = (dt.date(y, m, d) - epoch).days * 86400 + 43200
        month_only = False
        ev = rec["eventdate"]
        if ev is not None:
            tod = ev.hour * 3600 + ev.minute * 60 + ev.second
            if tod and (ev.year, ev.month, ev.day) == (y, m, d):
                t = (dt.date(y, m, d) - epoch).days * 86400 + tod
    unc = rec["coordinateuncertaintyinmeters"]
    if unc is None or not unc > 0:
        k, bit0 = 10, 1
    else:
        k = int(round(math.log2(min(max(unc, 10.0), 100000.0))))
        k, bit0 = max(0, min(20, k)), 0
    qc = (k << 2) | bit0 | (2 if month_only else 0)
    def code(value, table):
        return float(table.index(value)) if value in table else np.nan
    v = [code(rec["kingdom"], KINGDOMS), code(rec["class"], CLASSES),
         code(rec["basisofrecord"], BASIS_OF_RECORD),
         np.nan if rec["individualcount"] is None
         else float(rec["individualcount"])]
    for j, (_n, _u, lo, hi) in enumerate(CHANNELS):
        if not (v[j] == v[j]) or not (lo <= v[j] <= hi):
            v[j] = np.nan
    key = rec["taxonkey"]
    plat = int(f10b.platform_hash(str(key) if key else TAXON_UNMATCHED))
    return {"t": int(t), "lat": float(lat), "lon": float(lon),
            "platform": plat, "v": v, "qc": int(qc)}


def _rec(**kw):
    """One occurrence record with the snapshot's own column names."""
    r = {"decimallatitude": SMOKE_LAT, "decimallongitude": SMOKE_LON,
         "coordinateuncertaintyinmeters": 100.0, "eventdate": None,
         "year": 1900, "month": 1, "day": 2, "taxonkey": "6GQ7J",
         "scientificname": "Turdus merula Linnaeus, 1758",
         "kingdom": "Animalia", "class": "Aves",
         "basisofrecord": "HUMAN_OBSERVATION", "individualcount": 1,
         "license": "CC_BY_4_0", "issue": []}
    r.update(kw)
    return r


def smoke_parts():
    """{part name: [record]} — the synthetic snapshot, hostile on purpose.

    Every rule has a witness: both tracks, all three licences and an unknown
    one, each of the seven geospatial issues and two harmless issues, no
    position, a position out of range, no month, a month and no day, a time
    of day, a time of day that disagrees with the date columns, no
    coordinate uncertainty, an uncertainty under 10 m and one over 100 km, an
    unknown kingdom and an unknown class, an individual count past float16's
    exact range, a record with no taxon key, and records on both sides of
    the window.

    Every record gets its OWN day and hour, so no two rows share a
    (bin, time_s) and the store's order is decided by the data rather than
    by a tie-break.
    """
    jan, dec, jun = [], [], []
    # --- part 000002, January 1900: the probe's part ----------------------
    day = 1
    for j, lic in enumerate(("CC0_1_0", "CC_BY_4_0", "CC_BY_NC_4_0")):
        for k in range(3):
            jan.append(_rec(
                year=1900, month=1, day=day, license=lic,
                decimallatitude=round(-60.0 + 9.5 * (3 * j + k), 4),
                decimallongitude=round(-170.0 + 23.0 * (3 * j + k), 4),
                coordinateuncertaintyinmeters=float(10 ** (1 + k)),
                taxonkey=f"TAX{j}{k}", scientificname=f"Taxon {j}{k}",
                kingdom=KINGDOMS[j + k], **{"class": CLASSES[17 * (j + k)]},
                basisofrecord=BASIS_OF_RECORD[j + k], individualcount=1 + k))
            day += 1
    jan += [
        _rec(month=1, day=None, license="CC0_1_0", taxonkey="MONTHONLY",
             scientificname="Month only"),
        _rec(month=1, day=11, license="CC_BY_4_0", taxonkey="TIMEOFDAY",
             scientificname="A time of day",
             eventdate=dt.datetime(1900, 1, 11, 9, 30, 15)),
        _rec(month=1, day=12, license="CC_BY_NC_4_0", taxonkey="TIMENC",
             scientificname="A time of day, non-commercial",
             eventdate=dt.datetime(1900, 1, 12, 17, 5, 0)),
        _rec(month=1, day=13, license="CC0_1_0", taxonkey="BADEVENT",
             scientificname="Event date disagreeing with the columns",
             eventdate=dt.datetime(1899, 7, 4, 8, 0, 0)),
        _rec(month=1, day=14, license="CC_BY_4_0", taxonkey="NOUNC",
             scientificname="No coordinate uncertainty",
             coordinateuncertaintyinmeters=None),
        _rec(month=1, day=15, license="CC0_1_0", taxonkey="TINYUNC",
             scientificname="One metre of uncertainty",
             coordinateuncertaintyinmeters=1.0),
        _rec(month=1, day=16, license="CC_BY_NC_4_0", taxonkey="HUGEUNC",
             scientificname="Five hundred kilometres of uncertainty",
             coordinateuncertaintyinmeters=5.0e5),
        _rec(month=1, day=17, license="CC_BY_4_0", taxonkey="BADKINGDOM",
             scientificname="A kingdom no table knows", kingdom="Nowhereia"),
        _rec(month=1, day=18, license="CC0_1_0", taxonkey="BADCLASS",
             scientificname="A class no table knows",
             **{"class": "Nowhereopsida"}),
        _rec(month=1, day=19, license="CC_BY_4_0", taxonkey="BIGCOUNT",
             scientificname="A count past float16", individualcount=100000),
        _rec(month=1, day=20, license="CC0_1_0", taxonkey=None,
             scientificname=None, kingdom=None, **{"class": None}),
        _rec(month=1, day=21, license="CC_BY_NC_4_0", taxonkey=None,
             scientificname=None, kingdom=None, **{"class": None}),
        _rec(month=1, day=22, license="CC_BY_4_0", taxonkey="HARMLESS",
             scientificname="Issues that are not geospatial",
             issue=["COORDINATE_ROUNDED", "GEODETIC_DATUM_ASSUMED_WGS84"]),
    ]
    # the droppers: one per geospatial issue, then one per other rule
    for i, name in enumerate(GEOSPATIAL_ISSUES):
        jan.append(_rec(month=1, day=23, license="CC0_1_0",
                        taxonkey=f"DROPGEO{i}", issue=[name]))
    jan += [
        _rec(month=1, day=24, decimallatitude=None, decimallongitude=None,
             taxonkey="DROPPOS"),
        _rec(month=1, day=25, decimallatitude=999.0, taxonkey="DROPRANGE"),
        _rec(year=1900, month=None, day=None, taxonkey="DROPNOMONTH"),
        _rec(month=None, year=None, day=None, taxonkey="DROPNOYEAR",
             eventdate=dt.datetime(1899, 6, 1)),
        _rec(month=2, day=30, taxonkey="DROPBADDATE"),
        _rec(month=1, day=26, license="UNSPECIFIED", taxonkey="DROPLICENCE"),
        _rec(month=1, day=27, license="UNSUPPORTED", taxonkey="DROPLICENCE2"),
    ]
    # --- part 000001, December 1899, and part 000003, June 1900 -----------
    # Two more months so the store spans two years and its per-year ledger is
    # something to check, and so the probe's rule is visible: these rows are
    # in the store and NOT in the part the probe reads.
    for i, lic in enumerate(("CC0_1_0", "CC_BY_4_0", "CC_BY_NC_4_0") * 5):
        dec.append(_rec(year=1899, month=12, day=1 + i, license=lic,
                        taxonkey=f"DEC{i}", scientificname=f"December {i}",
                        coordinateuncertaintyinmeters=250.0,
                        individualcount=None))
        jun.append(_rec(year=1900, month=6, day=1 + i, license=lic,
                        taxonkey=f"JUN{i}", scientificname=f"June {i}",
                        coordinateuncertaintyinmeters=None))
    dec.append(_rec(year=1899, month=12, day=28, issue=["ZERO_COORDINATE"],
                    taxonkey="DECDROP"))
    jun.append(_rec(year=1900, month=6, day=28, decimallatitude=None,
                    decimallongitude=None, taxonkey="JUNDROP"))
    # outside the window on both sides, and never in the store
    dec.append(_rec(year=1897, month=5, day=4, taxonkey="BEFORE"))
    jun.append(_rec(year=1903, month=5, day=4, taxonkey="AFTER"))
    return {SMOKE_PARTS[0]: dec, SMOKE_PARTS[1]: jan, SMOKE_PARTS[2]: jun}


def smoke_schema():
    import pyarrow as pa
    return pa.schema([("decimallatitude", pa.float64()),
                      ("decimallongitude", pa.float64()),
                      ("coordinateuncertaintyinmeters", pa.float64()),
                      ("eventdate", pa.timestamp("ms", tz="UTC")),
                      ("year", pa.int32()), ("month", pa.int32()),
                      ("day", pa.int32()), ("taxonkey", pa.string()),
                      ("scientificname", pa.string()),
                      ("kingdom", pa.string()), ("class", pa.string()),
                      ("basisofrecord", pa.string()),
                      ("individualcount", pa.int32()),
                      ("license", pa.string()),
                      ("issue", pa.list_(pa.string()))])


def make_smoke_sources(root, d_lo, d_hi, track, seed=20260920):
    """The bucket in its real shape, and the truth rows of `track`.

    The truth is ordered the way the store breaks ties among equal
    (bin, time_s): the order the adapter yields, i.e. part order then the
    part's own row order. No two records share a second here, so the order is
    decided by the data.
    """
    import pyarrow as pa
    pq = _pq()
    os.environ.pop("GBIF_PARTS", None)
    os.environ["GBIF_SNAPSHOT"] = SMOKE_SNAPSHOT
    # ONE part for the smoke's probe, so the rule the real probe departs from
    # is still exercised: the fetch reads all three parts, the probe reads
    # the middle one, and `run_smoke` compares the two counts.
    os.environ["GBIF_PROBE_PARTS"] = "1"
    parts = smoke_parts()
    base = os.path.join(root, "gbif", PREFIX, SMOKE_SNAPSHOT, PARQUET_DIR)
    os.makedirs(base, exist_ok=True)
    keys = [{"key": f"{PREFIX}{SMOKE_SNAPSHOT}/citation.txt", "size": 88}]
    with open(os.path.join(root, "gbif", PREFIX, SMOKE_SNAPSHOT,
                           "citation.txt"), "w") as fh:
        fh.write("GBIF.org (1900) GBIF Occurrence Download\n")
    t_lo = f10b.seconds_since_epoch(d_lo)
    t_hi = f10b.seconds_since_epoch(d_hi) + 86399
    truth = []
    for name in SMOKE_PARTS:
        recs = parts[name]
        p = os.path.join(base, name)
        pq.write_table(pa.Table.from_pylist(recs, schema=smoke_schema()), p)
        keys.append({"key": f"{PREFIX}{SMOKE_SNAPSHOT}/{PARQUET_DIR}{name}",
                     "size": os.path.getsize(p)})
        for r in recs:
            row = _expect(r, track)
            if row is not None and t_lo <= row["t"] <= t_hi:
                truth.append(row)
    with open(os.path.join(root, "gbif", "listing.json"), "w") as fh:
        json.dump({"snapshots": [SMOKE_SNAPSHOT], "keys": keys}, fh)
    return truth


if __name__ == "__main__":                                  # pragma: no cover
    main()
