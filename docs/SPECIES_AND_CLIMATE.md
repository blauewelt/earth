# Species data and climate change — is it relevant?

*A short answer to "which species live where, how many individuals are there, and does
this matter for global warming?" — and what the globe's Biodiversity layer shows.*

## Yes — biodiversity is one of the clearest fingerprints of warming

Species distributions are among the most sensitive and best-documented indicators of
climate change. As the climate warms, organisms track the temperatures they can tolerate,
and they do so in three consistent directions: **poleward** (toward the poles), **upward**
(to higher elevations), and, in the ocean, **deeper and northward**. Meta-analyses find
land species moving poleward on the order of ~17 km per decade and marine species faster —
tens of km per decade — because the ocean has fewer barriers. Alongside range shifts,
warming advances **phenology**: earlier springs, earlier breeding, earlier migration and
flowering, which can desynchronise predators from prey and pollinators from plants.

These shifts are not projections — they are observed. The globe's **Biodiversity
(GBIF)** layer lets you see the raw evidence: the recorded distribution of climate-indicator
species, drawn from ~3.9 billion occurrence records.

## What the data does — and does not — tell you

There is an important distinction the layer makes explicit:

**Occurrence (presence) data** — *where a species has been recorded* — is abundant and
openly available. GBIF (the Global Biodiversity Information Facility) aggregates ~3.9
billion georeferenced records from museums, surveys, and citizen science. This is what the
globe layer renders. Its great caveat is **observer bias**: the map partly reflects where
people *look* (Europe, North America, Australia are dwarfed by their true biodiversity),
not only where species are. The "all recorded life" view is as much a map of human effort
as of nature.

**Abundance (how many individuals)** — is much scarcer. Counting populations requires
sustained monitoring, and no single dataset covers all species. The closest global product
is the **Living Planet Index** (WWF/ZSL), which tracks ~35,000 populations of ~5,500
vertebrate species and reports an average **~73% decline in monitored wildlife populations
since 1970** — a headline abundance signal, though climate is only one driver alongside
habitat loss and exploitation. Specific, well-counted cases exist (breeding-bird surveys,
fisheries stock assessments, Antarctic penguin colony counts from satellite guano), but a
complete "how many individuals live where" map of life does not exist.

So: presence, yes, richly; absolute abundance, only patchily.

## The indicator species on the globe

Each was chosen for a documented, climate-linked range change spanning marine, terrestrial,
and polar systems:

- **Atlantic mackerel** — a marine fish whose feeding range has pushed north into Icelandic
  and Greenland waters as seas warm, even triggering international fishing disputes.
- **European bee-eater** — a Mediterranean bird now breeding in Britain and Scandinavia.
- **Comma butterfly** — a textbook northward expansion across Britain over recent decades.
- **Little egret** — a wetland bird that has expanded rapidly northward in Europe.
- **Emperor penguin** — sea-ice-dependent; colonies are directly threatened by Antarctic
  sea-ice loss (a link to this globe's cryosphere and AMOC themes).
- **Staghorn coral (*Acropora*)** — reef-builders hit hardest by marine heatwaves and
  bleaching.
- **Arctic fox** — a tundra species squeezed as the red fox advances north with warming.
- **Buff-tailed bumblebee** — a pollinator whose range is compressing at its warm southern
  edge, a pattern seen across many bumblebees.

Selecting one shows its recorded occurrences in warm colours over the globe; the pattern
(e.g. mackerel clustering in the North Sea and reaching toward the Arctic) is the visible
signature of a warming-driven shift.

## How this connects to the rest of the project

Biodiversity closes a loop with the physical layers. The same North Atlantic warming visible
in the SST and sea-level tabs is what pushes mackerel north; the Antarctic ice loss in the
cryosphere data is what threatens emperor penguins; a weakening AMOC would reorganise the
very SST patterns that marine species track. In a prediction pipeline, occurrence data feeds
**species distribution models** (correlating presence with climate variables to project
future ranges) — a natural future extension: overlay a species' recorded range on the SST
field it depends on, and project the range forward under the scenario data already cataloged.

## Sources

GBIF occurrence data and map API (gbif.org); Living Planet Report 2024 (WWF/ZSL); IPCC AR6
WGII Chapter 2 (terrestrial and freshwater ecosystems) and Chapter 3 (ocean) for the
observed range-shift and phenology assessments. See the catalog entry for GBIF and OBIS
(marine biodiversity) for programmatic access.
