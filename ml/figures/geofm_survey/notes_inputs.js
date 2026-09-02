// Plain-English speaker notes for slides 34–45 (the generic-embedding input proposal), in deck order.
// Spliced into notes.js before the Sources entry.
module.exports = [

// 35 — what should a generic Earth embedding see?
`This slide opens the last part of the deck: if we want one embedding that is generic — usable for the ocean, the atmosphere, the land and both biospheres — what should it be allowed to see?

The rule at the top is the whole answer in one sentence: every signal that is relevant to the state of the four spheres at a point, and that cannot be derived from the other inputs. "Cannot be derived" is the important half. A vegetation index is a formula applied to two satellite bands; if the bands are in, the index adds nothing. Sunlight angle follows from position and time. The slope of a hill follows from the elevation map. None of those belong in the input — they are fine as things to predict, which costs nothing and gives the model free supervision.

The ladder on the left lists thirteen rungs, finest at the bottom. The bar is the native pixel size on a logarithmic scale, so 10-metre Sentinel imagery is a short bar and the 31-kilometre reanalysis is a long one. Colours are the sphere the rung mostly serves. Two rungs deserve a mention: rung 3, the 300-metre-to-1-kilometre daily radiometers, is the first rung that covers the ocean — the fine imagery of rungs 1 and 2 only sees land and coast. And rung 8 is the first rung that looks below the sea surface.

The box on the right is about derived products — a "reanalysis" is a weather or ocean model that is continuously nudged towards observations, and a "gap-filled map" is an observation product with the clouds interpolated away. They are admitted on three conditions: they must carry information we do not otherwise ingest (ERA5 has weather balloons and aircraft inside it), they are flagged so the model knows they are second-hand and are hidden from the input more often, and their time stamp is shifted by the window of observations they were built from, so they cannot smuggle in the future. That last point gets its own slide.`,

// 36 — input ladder I
`The first table walks up the fine end of the ladder.

Rung 0 is not data at all but the frame: where and when we are looking, and how deep. The model needs it because the same temperature means different things in January at sixty degrees north and in July at the equator.

Rung 1 is the 10-metre imagery: Sentinel-2, which is a colour-plus-infrared photograph of the sunlit surface with the atmosphere corrected out, and Sentinel-1, a radar that works day and night and through cloud, and that measures how rough and how wet a surface is. Between them they give field-scale texture — crops, forests, floods, sea-ice edges, ships. The constellation facts are current as of this week: Sentinel-2 has three satellites in orbit until the end of 2026, and Sentinel-1 is now the C and D satellites, since 1A was switched off at the end of June.

Rung 2 is Landsat, at 30 metres: the same kind of photograph plus a thermal image — how warm the ground is — and a forty-year archive.

Rung 3 is where the ocean appears. Sentinel-3 and VIIRS take a daily picture of the whole planet at 300 metres to a kilometre. Over the ocean the colour of the water tells you how much phytoplankton is in it — that is the ocean biosphere — and the thermal channels give the skin temperature of sea and land. MODIS, the older sensor everyone has used since 2000, is being retired at the end of this year, which is why VIIRS is named here.

Rung 4 is the geostationary satellites — the ones parked over one spot, like GOES over the Americas or Meteosat over Europe and Africa. They take a picture every ten minutes, so they are the only place where you can actually watch the atmosphere move. That is the fast end of the dependency cone.

The "O / D" column says whether each rung is observed or derived; the whole of this first table is observed.`,

// 37 — input ladder II
`The second table covers the physical fields and the ocean's memory.

Rung 5 is a collection of kilometre-scale sensors that are sparse — they see along a line or a narrow strip rather than everywhere. The important one for the ocean is altimetry: a radar that measures the height of the sea surface to a few centimetres along the satellite's track. Sea-surface height is the ocean's pressure map; its slopes drive the currents. SWOT is the new altimeter that sees a 2-kilometre-resolution strip instead of a line. The same rung holds ground weather radar, lightning, the laser altimeters that measure tree height and sea-ice thickness, and the satellites that measure greenhouse gases in the air column. Everything here is sparse in space or time, which is exactly the case where a set of points beats a grid — a theme that returns in the proposal.

Rung 6 is the cleaned-up daily maps: sea temperature with the clouds filled in, sea level everywhere and not only under the track, rain everywhere, wind over the whole ocean, sea-ice cover and drift, and the wetness of the top centimetres of soil. Most of these are derived — they merge several sensors and interpolate the gaps — so they are flagged, masked more heavily, and time-shifted. A few in this rung are genuine observations put on a grid, like the ASCAT scatterometer winds.

Rung 7 is the reanalysis atmosphere, ERA5: the best physically consistent estimate of the whole atmosphere at every hour since 1940, made by running a weather model and nudging it towards millions of observations. It is what forces everything else — heat and momentum into the ocean and land. One warning that surprised us in verification: ERA5's sea-surface temperature and sea ice are inputs to ERA5, taken from other products, not things ERA5 estimates, so they must never be treated as ocean observations. For real time, where ERA5 lags five days, the free ECMWF or GFS analyses at a quarter degree fill in.

Rung 8 is the first look below the surface: Argo floats, which drift at depth and surface every ten days to report temperature and salinity down to 2,000 metres — there are about 4,400 of them right now; deep floats to 6,000 metres; biogeochemical floats that add oxygen, nitrate, acidity and chlorophyll; moorings; surface drifters, which give currents directly; and GLORYS, the ocean reanalysis, a model's gridded best guess around all of these. This rung holds the heat capacity, the density gradients and the AMOC — the slow memory of the whole system. The note in the last column matters for the proposal: only points do this justice; gridding Argo throws away its depth resolution.`,

// 38 — input ladder III
`The third table completes the ladder with the two biospheres, the ground truth, and the stage.

Rung 9 is the ocean biosphere and carbon: ocean colour as observed, with the cloud gaps left as gaps; PACE, the new hyperspectral colour sensor launched in 2024; and the carbon measurements — SOCAT, which is forty-four million ship measurements of the CO2 content of surface water, telling you whether the ocean is absorbing or releasing carbon at that spot; plus the machine-learned monthly carbon maps built on top of it, which are derived and flagged as such.

Rung 10 is the land biosphere and land water: leaf area, fluorescence — plants give off a faint glow when they photosynthesise, and a satellite can measure it — surface temperature, evaporation, soil moisture, snow, fire, the total water stored underground as measured by the GRACE gravity satellites, rivers, and the flux towers that measure carbon and water exchange every half hour at a few hundred sites. Almost everything here stays where it is: soil, plants and snow do not move sideways. That is the column-only cone family on slide 39.

Rung 11 is the ground truth: weather stations, balloons, aircraft, ships, buoys, tide gauges. These are the measurements the reanalysis of rung 7 is fitted to, so they are redundant while the reanalysis is present — and essential when it is masked out, because the model must then be able to rebuild the atmosphere from points.

Rung 12 is the static context: elevation, sea-floor depth, land cover, soil, tree height, glaciers, people. Depth sets what currents can do; terrain sets rain and rivers; soil sets what water does. Each is a single dot with zero speed and infinite memory.

Rung 13 is optional in the first phase: aerosols and greenhouse-gas columns, which matter for radiative forcing and close the carbon loop between land, ocean and air.

If one had to start with one rung per sphere, it would be rung 7 for the atmosphere, rungs 6 and 8 for ocean physics, rung 9 for the ocean biosphere, and rungs 1 and 10 for land — which is the sequencing agreed earlier: ocean interior first, then ocean colour, then the atmosphere as forcing, then land.`,

// 39 — observed, derived, derivable — and the leakage trap
`This slide sorts every input into three bins and then explains a trap that is easy to fall into.

Left, always in: observations — a number a sensor produced. A reflectance, a radar echo, a float's thermometer reading. Nothing else contains them, so they are irreducible.

Middle, admitted but flagged: derived products — a number a program produced from observations plus a model or a statistical assumption. Reanalyses, gap-filled maps, machine-learned reconstructions. They come in because they carry information we do not ingest raw, but every dot carries a "source token" saying what kind of thing it is, and derived sources are removed from the input at least half the time. That way the model cannot get lazy and simply copy the reanalysis; it must also be able to work from observations alone.

Right, dropped: things that are pure functions of inputs already present — vegetation indices, sun angle, slope, the currents you can compute from the sea-level slope. They are not inputs. They are optional prediction targets.

The orange band is the trap. A gap-filled map or a reanalysis for a given day is built from observations on both sides of that day. The delayed-time sea-level maps, for example, use satellite tracks from up to six weeks before and after the day in question — only the near-real-time stream is built from the past alone; the ocean reanalysis works in seven-day chunks; ERA5 in twelve-hour chunks. If we train the codec to predict the present or the future from the past, then a derived dot labelled "ten days ago" secretly contains information from a few days ago — or even from after the anchor time. The model would learn to peek. The rule is simple: shift each derived source's effective time by its window, never use a delayed-time product at lag zero, and train the forecasting part of the objective on the near-real-time streams, which by construction only saw the past. This gives two embeddings per anchor — a near-real-time one and a final one — exactly the way ECMWF publishes ERA5T first and ERA5 later.`,

// 40 — five cone families
`Now we apply the dependency cone from the earlier slides to every rung of the ladder. The recap: a process that travels at speed v can only influence a point from within a distance v times the time elapsed, and only lags shorter than the process's memory tau are worth reading. When we do that for all thirteen rungs, five shapes come out.

Family A, fast, wide and short, is the atmosphere: wind carries information at about ten metres per second, which is 860 kilometres a day, but the atmosphere forgets in three to ten days. So the support is a wide flat disc — you need dots from far away but only from the last few days.

Family B, slow, narrow and long, is the ocean interior and sea level: currents move at a tenth of a metre per second, 13 kilometres a day; sea-level anomalies drift westward as Rossby waves at 3 kilometres a day. But the memory is months for eddies and years to decades for the interior. The support is a thin tall column, tilted upstream.

Family C is the L-shaped union we met before: sea-surface temperature and salinity, sea ice, surface chlorophyll. They are pushed around by the fast atmosphere and remembered by the slow ocean, so the support is wide at short lags and narrow at long lags.

Family D is land. Soil, plants and snow do not move sideways at all; they are forced from above by the atmosphere and remember locally for weeks to seasons. The support is a narrow column with a wide flat hat on top — the atmosphere disc at the shortest lags. Rivers and sea ice are the exceptions that move along a path.

Family E is static: one dot, zero speed, infinite memory.

The paragraph at the bottom adds depth as a sixth axis. The top ten to a hundred metres of the ocean, the mixed layer, are stirred within hours and belong to the surface cone; below it memory grows with depth — months at a few hundred metres, decades below two thousand. So depth dots are spaced logarithmically, dense near the surface, each depth band with its own memory. The consequence for sampling is the point of the slide: fine rungs want a small dense patch and a long local history; coarse physical rungs want a large sparse far field and a short history; the interior wants depth and years. No single grid and window serves all three. A per-channel dot sampler does.`,

// 41 — the cone-native codec
`This is the proposal itself: a cone-native codec — dots in, embedding out.

Until now the codec embedded one pixel-month, and a second stage assembled a stencil of those embeddings across space and time. Here the codec itself reads a set of dots — each dot is a sample of one channel at some offset in space, some depth, and some lag into the past — drawn from that channel's cone around an anchor point. The picture on the left shows it: the anchor is the yellow dot at the origin; the orange wind dots spread wide but stay near the present; the blue current and sea-level dots stay close in space but reach far back in time; the teal temperature dots fill the L. Depth dots are the small stack at the anchor — a float profile enters as it is, not regridded.

Step 1, the sampler, is fixed by physics. For gridded channels we draw dots in log-polar space and log lag, denser near the anchor and near the present, with the slot budget from the family. For point, profile and track channels — Argo, drifters, stations, altimeter tracks — the dots are the actual observations that fall inside the cone. Nothing is interpolated or invented. Fine imagery arrives as tokens from a small local codec, which the hybrid slide explains. Missing dots are simply absent: a set has no holes to fill.

Step 2 turns each dot into a token: the value, the coordinates as Fourier features on a log scale so that ten metres and ten thousand kilometres fit in the same code, a channel embedding, and the source flag from the previous slide.

Step 3 is the encoder, Perceiver-style: a couple of hundred learned latent vectors cross-attend to the two to four thousand dots, then a few self-attention blocks over the latents. The cost grows linearly with the number of dots, and per anchor it is a few billion operations — less than running a standard vision transformer on one image.

Step 4 pools the latents into the dynamic embedding, 256 numbers stored as 8-bit integers.

Step 5 is the objective. A held-out dot becomes a query — coordinates and channel without the value — and a light decoder predicts its value as a distribution. We mix several ways of hiding dots: whole channels, as before; the recent past, so that the model must forecast from the older past; dots from the future, so that the embedding learns tendency; an upstream sector; a depth band; and always the anchor's own channels, so the embedding still means "the state at this point". The distinction between "never measured" and "hidden" survives on the target side: there is no loss where nothing was ever observed.`,

// 42 — where velocity comes from
`This slide answers the question in the request directly: can an embedding capture momentum features — velocity, water flow — if it sees several points in space and time? Yes, and here is why.

A single photograph does not show speed. Two photographs a known time apart do: whatever moved, moved by speed times time. In the diagram, position runs left to right and lag runs upward, the present at the bottom. A feature carried along by the flow — a warm patch, an eddy, a cloud — traces a tilted band. The yellow dot is the anchor now; the blue dot is the same water some time ago, upstream by exactly speed times lag. The grey dots are candidates at the same lag that the model has to reject.

The argument in prose: if a field is carried along, its value here now is best predicted by its value upstream a while ago. The correlation between the anchor's neighbourhood now and the field some time ago peaks at a displacement equal to speed times lag. Oceanographers have used exactly this — the maximum cross-correlation method — since the 1980s to compute currents from pairs of satellite temperature images. Attention over dots that carry their relative position and lag can implement that search, and the winning ratio of displacement to lag is a velocity. The rate of change at one place comes from two lags at that place; convergence comes from dots around the anchor. A per-pixel snapshot codec has one lag and cannot do any of this.

Having the capacity is not the same as using it, so three measures make sure it is used. First, flow enters as maskable channels — drifter velocities, sea-ice drift, wind, river discharge, and the flagged reanalysis currents — so the masked objective directly asks the embedding to produce velocity whenever the velocity channel is hidden. Second, the slow families always contain several lags of the same field, by construction of the cone budgets. Third, optionally, a fraction of the dots can be drawn along the path a first-guess velocity would predict — the semi-Lagrangian idea from the PARADIS paper used as a sampler rather than as a layer.

The test is a velocity probe: a simple linear head from the frozen embedding to drifter velocity or reanalysis currents at the anchor, with the derived channels hidden at test time. The snapshot embedding should score near zero; the cone codec should not. It is the cheapest decisive experiment in the plan.`,

// 43 — is it zero-sum? pros and cons
`This is the slide the request asked for explicitly: is the architecture zero-sum? Either we keep the embedding fine-grained and local and let stage 2 make the spatial and temporal connections, or we put that work into the embedding — and does one side's gain equal the other's loss?

Column A is the thin codec with a thick stage 2: embed one pixel-month, then a second-stage model assembles stencils of those embeddings. That is today's Earth 2, and it is what every model in the survey does — they embed images, and the user connects them. Column B is the cone codec from the previous slides with a thin stage 2.

Reading down the rows, green is an advantage and red a cost. A wins on cost per embedding, on caching — an embedding is this place at this time and never changes — on provenance, on simplicity, and on direct comparability with published models. B wins on everything that involves time: dynamics, which a snapshot cannot represent at all; the information bottleneck, because a per-pixel codec has to compress before it can know what the dynamics need, and whatever it throws away as noise — the tiny sub-pixel shifts and gradients that carry velocity — is gone for good; sparse observations, because a lone Argo profile with no context is nearly empty but a profile among its neighbours is not; the cone geometry, which A has to re-learn for every task from labels and B fixes once from physics; sample efficiency of the probes, which matters when the AMOC has only twenty years of monthly labels; and forecasting, which B gets for free inside pretraining.

Two rows are honest costs of B: it is a more complex training pipeline, and there is a real risk that the model copies the derived channels instead of learning — which is what the source flags and drop rates are for. And for purely static tasks like land cover, A's snapshot semantics are actually better.

So the table is not one-sided, but it is also not symmetric — which is the argument of the next slide.`,

// 44 — not zero-sum
`The answer to "is it zero-sum" is no, and the reason is an asymmetry.

First: B contains A. A cone codec that ignores every dot except the anchor's own is a per-pixel codec. Whatever A can represent, B can. The reverse is false, because of the bottleneck: information that the per-pixel code discards before the dynamics are visible cannot be recovered by stage 2 — this is the data-processing inequality, the formal statement that you cannot get information back by processing. So in capability the trade is one-sided; what B pays is compute and a less tidy meaning of "the embedding of this place".

Second: the compute is paid in different places. A pays little per embedding but pays the context assembly again inside every stage-2 model, for every task. B pays once per anchor and amortises it across tasks — provided the cone is generic. And that is exactly what a physics-set support promises: the shape depends on the channel, not on the task.

That gives a principled split rather than a compromise. Into the embedding goes everything whose relevance is decided by physics — the cone, depth coupling, the local past — because that is identical for every task. Into stage 2 goes everything whose relevance is decided by the task: which section to integrate for the AMOC, which basin to sum for carbon, which teleconnection matters — El Niño's Pacific state is ten thousand kilometres and six months away, outside any single cone — and which label.

The right side shows the resulting hierarchy. Raw observations at the bottom. A local codec, per tile and date, cached, with snapshot semantics — the existing per-pixel codec generalised to a tile — producing a small local code that mapping tasks use directly. A cone codec per anchor, which reads local tokens plus coarse dots plus native point observations inside each channel's cone, and produces the dynamic code where the momentum features live. And a thin stage 2 for teleconnections, aggregation and task heads.

What does existing evidence say? Nothing decisive yet, and it is worth being precise about why. The attribution matrix's contrast between a 3-by-3 codec and the same 3-by-3 raw input — 0.672 against 0.659 — is a single-seed probe number, and our own record shows probe numbers moving by far more than that between seeds, so the honest reading is parity. And on 2 September the paper was reset: every rolled forecast number from before that date was withdrawn, because the second-stage heads had been trained under a pool that had seen the held-out years. The first clean measurement of the bottleneck question is the one the reset paper puts first on its list — a Linear Inverse Model fitted in pixel space against one fitted in the codec's embedding space. If the embedding-space model is worse, the codec has discarded predictable signal, which is exactly column A's weakness. Phase 0, next slide, is built to run after that measurement exists.`,

// 44b — two stencils, one cone (E-067)
`This slide is the one Chris asked to see first: two stencils that together implement the whole dependency cone. It is also the first slide in the deck about something that has been built rather than proposed.

The picture on the left is the dependency space for a family-B channel — currents or sea level — with lag along the bottom on a logarithmic scale from one pentad, that is five days, to 144 pentads, two years, and distance from the anchor up the side, from one cell to 4,444 kilometres. The purple region is the codec's inner cone: raw channel values at lags one to six, within a reach that grows from 130 to 907 kilometres, sampled as sunflower dots. The orange region is stage 2's outer cone: embeddings at lags seven to 143, from 111 kilometres out to a reach that grows with lag until it hits the 4,444-kilometre cap. The yellow bar along the bottom is the anchor's own column, which both stages see, and it is the only place they overlap. One consequence is worth saying out loud: the outer stencil is empty for the first six lags, by construction, because the codec has already read that entire disc. Stage 2 keeps only the anchor's own embedding there.

The right-hand box lists what exists. Data: the pentad tensor gets the east and north components of the surface current — the same GLORYS numbers the speed channel is the hypotenuse of, so nothing new is downloaded; direction is what the cone needs and what the first hypothesis supervises. Cone: a small module holds the three channel families, the reach per lag, the dot budget — 748 tokens per anchor, 42 patches and 706 dots — and the stage-2 spiral, 3,432 cells over 144 lags against 20,880 for the old cylinder. Codec: a seven-million-parameter Perceiver whose decoder can be asked for any channel at any offset and lag and answers with a mean and an uncertainty; one fix from the implementation is that the headline loss reads the bottleneck vector alone, because a decoder that could also read the latents would let the bottleneck carry nothing while every curve looked healthy. Trainer: the held-out years are held out the strict way — every dot and every future target of a training anchor must lie in training bins, checked by brute force before training — and a velocity probe reads the current from the frozen embedding with the current channels hidden. On a synthetic flow whose velocity cannot be read from a single frame, the cone codec's probe finds it and the snapshot codec's does not: a check that the pipeline works, not an ocean result.

The paragraph at the bottom answers "how much velocity should go into the embedding" with a resolvability rule: a displacement can be read once it exceeds one grid cell, six centimetres per second per five-day lag. Eddies clear that at the first lag, the slow Rossby drift within six, the atmosphere forgets within one or two, and sea-surface temperature's months of memory are deliberately left to stage 2 — so thirty days at 0.3 metres per second is the shortest inner window in which every fast process has moved by a cell and the slow ones by three.

Three hypotheses are registered with their falsifiers before anything runs, and the order is deliberate: the velocity probe first, because it is cheap and decides whether the expensive five-seed stage-2 comparison is worth buying.`,

// 45 — phases and the decisive experiment
`The plan starts with an experiment that needs no new data.

Phase 0 uses exactly the tokens the second stage already reads: ocean pixel-months, the 3-by-3 codec, and the sunflower stencil over 24 months. We train a cone codec over those same tokens with the masked-dot objective, and compare three things at equal total context: the current codec with its stage-2 attention head; the cone codec with a plain ridge regression on top; and the cone codec with an attention head. Three things about the protocol are non-negotiable after the 2 September paper reset. The verdict is rolled skill under the corrected window-scope pool — mean skill against climatology and against damped persistence at each lead, trained and held-out longitudes reported separately, with block-bootstrap intervals and the Linear-Inverse-Model null beside every number. Probes — the velocity probe and the RAPID transport probe — are diagnostics of what the embedding encodes, never verdicts. And at least three seeds per arm, because anything probe-scored is unreadable at one seed. The predictions are written down so they can be wrong: the cone codec should match or beat the current setup on rolled skill with only a linear head, and beat the embedding-space LIM where the current codec does not; it should win by a wide margin on the velocity probe; and it should tie on static probes. If the cone codec only matches when it also gets an attention head, then the "work in the embedding" argument loses on sample efficiency and the hybrid keeps a thick stage 2.

After that, one sphere per phase, in the order agreed earlier. Phase 1 adds the ocean surface and the atmosphere as forcing — the gap-filled surface maps and ERA5 — and is scored on forecasting sea temperature and sea level one to four weeks ahead against persistence and against the operational Copernicus forecast. Phase 2 adds the interior — Argo as native dots, GLORYS as a flagged, heavily masked prior — and is scored on the AMOC again, on mixed-layer depth, and on upper-ocean heat content. Phase 3 adds the ocean biosphere and is scored on reconstructing the carbon flux with the carbon maps held out. Phase 4 adds land, with the local codec on Sentinel and Landsat tiles, and is scored on vegetation and soil-moisture forecasts as well as the standard benchmarks for comparability. Phase 5 adds the fast atmosphere at ten-minute cadence through a second local codec.

Six ablations run in every phase; the first — cone inside versus cone on top at equal context — is the one that keeps answering the zero-sum question as the inputs grow. The engineering note at the bottom is a warning: the network is cheap, under an hour per year of ocean embeddings on a single modern accelerator; the sampler that finds which tiles, dates and profiles fall into which cones is the thing to build first.`,

// 46 — data continuity 2026–27
`The verification pass this week turned up an unusual amount of change in the satellite fleet, and it affects which channels are safe to build on.

Sentinel-1 lost its A satellite at the end of June; the constellation is now C and D, and there is a two-year single-satellite gap in the radar record. Sentinel-2 has three satellites only until the end of this year. MODIS, the sensor behind most 500-metre products since the year 2000, is being switched off around the turn of the year, and Suomi-NPP, the first VIIRS satellite, stops delivering data on the first of November; the newer NOAA-20 and NOAA-21 continue. The passive-microwave sea-ice record is in the middle of a hand-over from the retiring military DMSP satellites to Japan's AMSR2 and, eventually, AMSR3 — which means that for a while near-real-time sea ice depends on a single satellite launched in 2012. Sea level is stable; Sentinel-6B is up but not yet the reference. Several climate-quality records have hard ends — the ocean-colour climate record at the end of 2024, the FLUXNET collection frozen in 2020 — and their near-real-time twins must be used for later years.

The architectural answer is the one the whole deck has been building towards: with channel-drop training, a sensor that disappears is a masked channel, not a failure. But the training archive has to be assembled sensor-agnostic, with the sensor identity in the source flag, so that the model learns "thermal radiometer at 1 km" rather than "MODIS".

The licence line matters for anything meant to be published: the EN4 ocean analysis is non-commercial, the river-gauge archive is research-only and cannot be redistributed, the European radar composite is for members, and ECMWF's native 9-kilometre analysis is licensed while the quarter-degree open data is free.

The paragraph at the bottom lists smaller corrections that came out of the same pass — worth knowing because they are commonly assumed the other way: ERA5's sea-surface temperature is an input, not a product; ERA5-Land has no data assimilation; the Copernicus wave model is MFWAM, not WAVEWATCH; and the European radar composite is now 1 kilometre every 5 minutes.`

];
