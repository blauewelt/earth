// Plain-English speaker notes, one entry per slide, in deck order.
// Written for a listener who has not seen the papers: every technical term and symbol is explained
// where it is first used, and again briefly where it matters.
module.exports = [

// 1 — title
`This deck compares six "representation models" for Earth observation, and then uses them to sharpen the design of our own model for the ocean.

A representation model (also called an embedding model or foundation model) is a neural network that is trained on huge amounts of raw data without labels, and whose job is to turn each place on Earth into a short list of numbers — a vector — that summarises what is going on there. That vector is called an embedding. Once you have good embeddings, many downstream tasks (mapping crops, detecting floods, estimating an ocean current) become much easier because a small model can be trained on top of the embeddings with very few labels.

The word "stencil" runs through the whole deck. It is our term for the piece of the world that feeds one embedding: how large an area around a pixel the model looks at, and how long a stretch of time it summarises. The little drawing on the right shows the idea — a grid of pixels with one output pixel (yellow) and its spatial context (blue), and a timeline with the observations that go into it (orange).

The six models are: AlphaEarth (Google DeepMind), TESSERA (University of Cambridge), OlmoEarth (Allen Institute for AI), TerraMind (IBM and ESA), Prithvi-EO 2.0 (IBM and NASA), and IBM's ocean model, Granite-Geospatial-Ocean. The last third of the deck applies what we learn from them to "Earth 2", our own ocean project.`,

// 2 — how to read the stencil diagrams
`Every model slide uses the same three-panel picture, so this slide is the legend.

SPACE panel (left). The grid is the tile — the square block of pixels the model processes at once. The blue outline marks the tile edge. The small blue square is one "token": a token is the unit the neural network actually reasons about, usually a small patch of pixels (for example 16 by 16) squashed into one vector. The yellow square is the single output pixel we are asking about. The stacked sheets behind the grid stand for the different data sources — for example optical imagery, radar and elevation — that are laid on top of each other ("co-registered") so that every pixel has a value from each source.

TIME panel (middle). The horizontal axis is time. The shaded band is the "temporal window": the span of time that one embedding summarises. The dots are the individual satellite observations inside that window. If instead there is a single tall bar, the model looks at one moment only — a snapshot.

OUTPUT panel (right). The purple stack represents the embedding vector. "D" is its dimension, the number of entries in the vector — 64, 128, 768 and so on. The caption says whether you get one vector per pixel, per token, or per whole tile.

The four questions on the right are the ones we ask of every model. "Per-pixel or patch" asks whether the model uses neighbouring pixels at all. "Snapshot or period summary" asks whether time is folded into the vector. "Product or model" asks whether the makers hand you pre-computed embeddings, the trained network ("weights"), or both. And "does it touch the ocean" is our own concern: almost all of these models deliberately throw away sea pixels.`,

// 3 — AlphaEarth
`AlphaEarth Foundations is Google DeepMind's model, published in July 2025. Its central idea is the "embedding field": every 10-metre pixel of land on Earth gets a 64-number vector for every year from 2017 to 2024, and Google publishes those vectors as a dataset inside Google Earth Engine.

Inputs: a very wide mix — optical imagery (Sentinel-2, Landsat), radar (Sentinel-1, PALSAR-2), laser altimetry (GEDI), climate reanalysis (ERA5-Land), gravity (GRACE), elevation, land cover and even text descriptions. Over three billion individual observations went in.

Architecture: the encoder is called Space-Time-Precision, STP, and has 15 blocks. The next four slides walk through it. The training uses several losses at once: reconstructing masked-out inputs, keeping the embeddings spread evenly over a sphere ("batch uniformity"), consistency between a teacher and a student network, and matching to text. The deployed network has about 480 million parameters.

The stencil is the key point for us. Output pixel: 10 m. Spatial context: the model uses attention over the whole input frame at coarse resolution, so a pixel's embedding is influenced by everything in its frame — and the supplement tells us how large the frame is: 1.28 km by 1.28 km, 128 pixels a side, for both training and the global inference run (which tiles the world in 960-m squares padded by 160 m on each side). So an AlphaEarth pixel never sees further than about 640 m in any direction. Time: the model takes a "valid period" — a start and end date — and produces a summary of everything observed in it. In principle any window; in practice the published product uses calendar years.

Output: 64 dimensions, normalised to length one (a "unit vector"), stored as 64 bytes. Coverage is land plus coastal and inland water; no open ocean.

Availability is the big caveat: only the dataset is released. The trained network is not, so nobody can run it on new data, new time windows, or the ocean.

"Error reduction ~23.9 %" means: across their 15 evaluation tasks, using AlphaEarth embeddings instead of the next-best method cut the error by about a quarter on average (about 10 % when only ten labels per class are available, about 4 % with one). The per-task numbers are published only as charts, so we do not quote them.`,

// 4 — inside the STP encoder
`This slide opens up the encoder. First the symbol: L is the side length of the square input frame, measured in pixels. The main text keeps it symbolic — "given a square input of L pixels a side" — but the supplement fixes it: frames are 128 by 128 pixels, 1.28 km at 10 m, so the three grids below are 8 by 8, 16 by 16 and 64 by 64 cells.

There are three "operators" — three different kinds of neural-network layer — that run in parallel inside every block:

The SPACE operator works on a coarse grid of L/16 by L/16 cells and uses self-attention, the mechanism from transformers in which every cell can look at every other cell. This gives global context across the frame at low resolution.

The TIME operator works on an L/8 grid and uses "time-axial" attention: each location looks only at its own history across all the dated frames. Every frame carries a sinusoidal "timecode" (a numeric encoding of its date), which is how the model knows when things happened.

The PRECISION operator works on the finest grid, L/2, with ordinary 3-by-3 convolutions to keep sharp spatial detail.

A "block" is one round of all three operators. After each block, a learned "Laplacian pyramid" rescaling step lets every operator hand its result to every operator of the next block: coarse results are up-sampled, fine results are down-sampled, so that information flows between scales. Repeat for 15 blocks (supplement S2.4, which also gives the widths: 1024 numbers per cell on the space path, 512 on the time path, 128 on the precision path).

Input projectors bring each source's frames down to the L/2 grid first. The output head resamples back up to the precision resolution, then to 10 m, and produces the 64-number vector per pixel.

"Teacher/student" refers to the training recipe: one network (teacher) produces targets that a second, identical network (student) learns to match; a third model aligns embeddings with text descriptions in the style of CLIP.

Bottom line: because the space operator uses attention over the whole frame, the effective receptive field is the frame — 1.28 km, so at most about 640 m from the pixel in any direction. What the paper still does not spell out is the exact form of the pyramid resampling and how the three states are merged.`,

// 5 — STP step 1
`Steps 1 to 4 animate the previous slide with the paper's own frame: L = 128 pixels at 10 m, so the frame is 1.28 km across (supplement S2.1 and S8.2). Two things in these slides are simplified and the footers say so: we draw 4 blocks where the real network has 15, and the per-year frame counts are typical values rather than the paper's.

Step 1 is about the input projectors. Each data source arrives at its own native resolution over the same 1.28-km footprint: Sentinel-2 at 10 m gives a 128 by 128 grid (drawn coarser here); Landsat at 30 m gives about 43 by 43; ERA5-Land, a weather reanalysis at 9 km, gives just one value for the whole frame. Each source also arrives many times a year — Sentinel-2 roughly every five days, Landsat every 16.

A learned projector per source converts each frame into a "feature map" — a grid of learned numbers rather than raw pixel values — at half the input resolution: 64 by 64 cells of 20 m. This is the only place where resolution is permanently reduced before the output head restores it.

The important detail is that each frame keeps its timestamp: it is carried along as a sinusoidal timecode so the time operator later knows when each observation was made. "Σ Nᵢ frames in, Σ Nᵢ maps out" simply means: however many frames come in from all sources combined, that many feature maps come out — nothing is merged yet.

The yellow pixel is our 10-m output pixel; after this step it lives inside a 20-m feature cell.`,

// 6 — STP step 2
`Step 2 shows what one cell can "see" inside each of the three operators, on the same 1.28-km footprint.

SPACE (left): the grid has 8 by 8 cells of 160 m (that is L/16 with L = 128). The highlighted cell attends to every other cell — the dashed lines. So the space operator gives every cell a view of the whole frame, but only at coarse grain.

TIME (middle): 16 by 16 cells of 80 m. The stacked sheets are the dated frames. The highlighted cell attends only to itself across all the sheets — this is "time-axial" attention. It knows the whole history of one spot, but nothing about the neighbours.

PRECISION (right): 64 by 64 cells of 20 m. The operator is a 3-by-3 convolution, so a cell sees only its eight immediate neighbours — 60 m across — but at the finest grain the block carries. The magnifier shows that 3-by-3 neighbourhood.

The point of running three views at once: none alone is the stencil. Space knows the whole frame coarsely, time knows the whole year at one spot, precision knows the fine local texture. Step 3 is where they are combined.`,

// 7 — STP step 3
`This is the sentence the walk-through was built to explain: "after each block, learned Laplacian-pyramid rescaling lets every operator feed every operator of the next block".

On the left are the three outputs of block k — the coarse space map, the medium time map, the fine precision map — each with one feature highlighted. In the middle is the rescaling stage. A Laplacian pyramid is a classic image-processing idea: an image is represented at several resolutions at once, and there are learned operations to move information up (coarse to fine, up-sampling) and down (fine to coarse, down-sampling). Here it is used to translate every map to every resolution.

On the right are the three inputs of block k+1. Each now contains all three colours: the blue space feature arrived in the fine map as an 8-by-8 block of 20-m cells, the green precision patch arrived in the coarse map as a fraction of one 160-m cell, the orange time cell landed as 2-by-2 or 4-by-4 blocks. Nine paths in total (three sources times three destinations), then a sum per level — the sum is our reading; the paper says "pass its state to each of the operators" without spelling out the merge.

Repeat 15 times (we draw 4). The consequence for our stencil question: after a few blocks, even the fine 20-m path has a receptive field that spans the whole frame, because it has received the space operator's global view several times — while keeping its own fine grain.`,

// 8 — STP step 4
`Step 4 closes the loop. After the last block, the three maps are merged and a final learned spatial resampling takes the result from the L/2 grid (64 by 64 at 20 m) back to L (128 by 128 at 10 m).

The output frame in the middle shows, for one yellow 10-m pixel, what it has inherited: the tinted background stands for the whole frame reaching it through the space path; the dashed squares are the coarse cells it inherited from the space and time operators; the green patch is the fine local texture from the precision path.

Each output pixel becomes a 64-number vector. "Unit vector on S⁶³" is the mathematical way of saying the 64 numbers are scaled so their total length is exactly one — they sit on the surface of a 64-dimensional sphere. The batch-uniformity loss during training spreads the embeddings evenly over that sphere. In the released dataset each vector is stored as 64 bytes (one "int8", a whole number between −128 and 127, per dimension).

"Per valid period": the pair of dates (t_s, t_e) — start and end of the window — was fed into the time operators in every block as timecodes, so the whole frame of vectors is a summary of that window. In the public product the window is one calendar year.

So what is AlphaEarth's stencil? Spatially, the whole 1.28-km frame coarsely (8 by 8 cells of 160 m through the space path) plus a fine local neighbourhood (20-m cells through the precision path); temporally, every dated observation in the window. The number to remember: an AlphaEarth pixel never sees further than about 640 m.`,

// 9 — TESSERA
`TESSERA comes from the University of Cambridge (published June 2025, accepted at CVPR 2026) and is the most radical design in the deck: it uses no spatial context at all.

Inputs: Sentinel-2 optical (10 bands) and Sentinel-1 radar (2 polarisations, VV and VH). Sentinel-1 is radar, so it sees through clouds; Sentinel-2 is optical.

Architecture: for each 10-m pixel separately, a small transformer reads the pixel's own time series — 40 dates sampled from one calendar year for each sensor — and a GRU (a type of recurrent network) pools them into one vector. The two sensors have their own branches which are fused at the end. The encoder is only about 46 million parameters; a much larger "projector" is used only during training and then discarded. Training uses Barlow Twins, a self-supervised objective that asks two differently-sampled views of the same pixel-year to produce the same embedding while keeping the embedding's dimensions non-redundant — no reconstruction of inputs is involved.

Stencil: strictly one 10-m pixel; neighbours never enter. One calendar year; dates are irregular, and the model is told the day of year of each observation. If a pixel has fewer than 40 clear observations, dates are sampled with replacement to fill the 40 slots.

Output: 128 numbers per pixel per year, stored as int8. Years 2017–2025, land only (water masked). Everything is released: weights (CC0, the most permissive licence), code (MIT), and precomputed embeddings through the GeoTessera service.

Results quoted are with the embeddings frozen and only a small head trained on top: PASTIS-R crop segmentation mIoU 50.7, TreeSatAI tree-species F1 78.0, canopy height error 12.2 m. mIoU is "mean intersection over union", the standard segmentation score (100 = perfect overlap). F1 is the balanced average of precision and recall for classification.

Why it matters for us: TESSERA proves that a per-pixel temporal encoder with zero spatial context is competitive on land. That is the opposite extreme from our 3-by-3 patch, and a useful reference point.`,

// 10 — OlmoEarth
`OlmoEarth is the Allen Institute for AI's family of open models, first released November 2025; version 1.1 (May 2026) merged Sentinel-2's band groups into a single token to cut compute, and version 1.2 added rotary position encodings. It is a "space-time ViT": a Vision Transformer that tokenises not just space but also time and sensor band groups.

Inputs: Sentinel-1, Sentinel-2 and Landsat-8 monthly time series, all resampled to 10 m, plus static map layers (OpenStreetMap, land cover, crop data, elevation, canopy height) used as extra inputs and as prediction targets.

Architecture: the tile is 2.56 km square (256 by 256 pixels at 10 m). Tokens are flexible patches of 1 to 8 pixels — so 10 to 80 m — and up to 12 monthly steps are tokenised separately. Attention runs across space, time and modality together. The training objective, "Latent MIM Lite", is masked modelling: hide some tokens, predict them in the embedding space rather than as pixels, plus contrastive terms. Four sizes: Nano (1.4 M parameters), Tiny (6 M), Base (89 M), Large (308 M).

Output: one vector per patch per month (768 dimensions for Base, 1024 for Large) plus a pooled vector for the tile. There is no global precomputed product; instead the "OlmoEarth Studio" computes embeddings on demand for any region you draw.

Results: in the authors' own benchmark the model is best on 15 of 24 tasks with frozen embeddings and 19 of 29 when fine-tuned; on five tasks a fine-tuned OlmoEarth matches or beats AlphaEarth.

Licence: open weights, but under a custom licence that forbids some uses (military, extractive industries).

For us: OlmoEarth is the closest analogue to a general space-time transformer we could actually retrain — open weights, monthly tokens, a masked latent objective — but its pretraining sample is land-biased.`,

// 11 — TerraMind
`TerraMind is a joint IBM Research and ESA Φ-lab model (April 2025, ICCV 2025). Its distinguishing feature is that it is generative and "any-to-any": given some modalities it can produce the missing ones.

Inputs come in two kinds. Pixel-level: Sentinel-2 optical, Sentinel-1 radar, elevation, RGB, vegetation index (NDVI). Token-level: land-use class maps, text captions, geographic coordinates. All are aligned on 10-m, 224-pixel tiles.

Architecture: each modality has its own "tokenizer" — a small network that turns its raster into discrete codes from a 16,000-entry codebook (an FSQ-VAE, a kind of quantising autoencoder). The main encoder–decoder then learns to predict masked codes of one modality from the others. "Thinking in Modalities" (TiM) is the trick of generating a missing modality — say land cover from optical — and feeding it back in as extra input. Trained on 500 billion tokens from about 9 million TerraMesh samples.

Stencil: tile 2.24 km (224 pixels at 10 m); token 16 pixels = 160 m, so 14 by 14 tokens per modality. One co-registered timestamp per sample — every modality comes from the same date and place; there is no multi-temporal fusion in pretraining.

Output: 196 tokens per modality, 768 dimensions for the Base model and 1024 for Large, or one merged vector per tile, plus generated rasters. No precomputed embedding archive; weights are Apache 2.0 (fully open) on Hugging Face.

Results: on the PANGAEA benchmark (a standard suite of segmentation tasks) TerraMind Large scores 59.6 mean IoU, the only foundation model to beat U-Nets trained from scratch on that suite.

Ocean: the training data is stratified by land ecoregion; there are no marine modalities and open-ocean behaviour is untested.`,

// 12 — Prithvi-EO 2.0
`Prithvi-EO 2.0 is IBM and NASA's second-generation Earth observation model (December 2024). It matters here for two reasons: it is a widely used open baseline, and IBM's ocean model on the next slide reuses its recipe.

Inputs: HLS, the Harmonized Landsat–Sentinel-2 product, at 30 m with six spectral bands (blue, green, red, near-infrared, and two shortwave-infrared bands). 4.2 million training samples from 2014–2023. Importantly for us, sea-only tiles were explicitly removed using a cloud/water mask, and Greenland was excluded.

Architecture: a masked autoencoder (MAE) — the network sees a tile with 75 % of its patches hidden and learns to reconstruct them. Two sizes: 300 million parameters (a "ViT-Large", 1024-dimensional embeddings, 24 layers) and 600 million (ViT-Huge, 1280 dimensions, 32 layers). The "TL" variants additionally receive the acquisition date and latitude/longitude as extra encodings.

Stencil: tile 224 pixels at 30 m = 6.72 km. Token 16 pixels = 480 m for the 300 M model, 14 pixels = 420 m for the 600 M model (we checked the published configuration files). Time: four frames, each 1–6 months apart to capture seasons; the "tubelet depth" is 1, meaning each frame gets its own grid of tokens rather than frames being merged into one token — time is fused only inside attention. Single-frame use is also supported.

Output: one vector per token per frame; no precomputed product. The typical way to use it is full fine-tuning — retraining the encoder together with a small decoder for your task — through IBM's TerraTorch toolkit.

Results: on GEO-Bench (a 12-task standard benchmark) the 600 M-TL model is the best on average, about 8 % above version 1.0 and ahead of six other foundation models; the per-dataset scores are published as charts, so we quote no single number. In May 2026 the model was deployed on a satellite for onboard flood and cloud detection.`,

// 13 — Granite-Geospatial-Ocean
`This is IBM's Prithvi-based ocean model, developed with the UK's STFC Hartree Centre, Plymouth Marine Laboratory and the University of Exeter, released October 2025. The code lives in the ibm-granite GitHub organisation; we read its configuration files directly to get the numbers below.

Inputs: Sentinel-3 OLCI, the ocean-colour instrument, 16 spectral bands at 300 m, plus sea-surface temperature from the SLSTR instrument — 17 channels in total. About 512,000 tiles from 2017–2021 (the paper also quotes 470,000 for training plus 50,000 for validation), sampled evenly across 83 "Longhurst provinces" (a standard division of the ocean into biogeochemical regions) and across months.

Architecture: the Prithvi masked-autoencoder recipe with the network shrunk to fit the coarser sensor: embedding size 512, 12 layers, about 50 million parameters, trained to reconstruct masked patches under a root-mean-square-error loss. The authors report that larger networks gave no gain.

Stencil: tile 42 by 42 pixels at 300 m = 12.6 km square. Token 2 by 2 pixels = 600 m, so 21 by 21 = 441 tokens. num_frames = 1: a single acquisition, no temporal stacking — the four-frame option of Prithvi is switched off. For fine-tuning, each ship measurement is paired with cloud-free imagery from a six-day window centred on it.

Output: 441 vectors of 512 dimensions per tile. Downstream, a small regression head is trained to predict chlorophyll-a (a proxy for phytoplankton) and primary production (how much carbon the phytoplankton fix).

Results: the numbers are RMSE — root-mean-square error — in log10 units, because these quantities vary over orders of magnitude. With the foundation model: 0.14 for chlorophyll, versus 0.16 training the same network from scratch; 0.39 versus 0.42 for primary production. The gains are small and within the fold-to-fold spread; the authors' stronger claim is that the advantage grows when labels are scarce.

Caveats: surface-only, optical (so cloud-limited), one snapshot in time. It is a model of ocean colour — biology at the surface — not of the ocean's physical state.`,

// 14 — stencils side by side
`This chart puts all six models — and our own — on two axes that summarise the whole deck.

Horizontal axis (logarithmic): the spatial extent of the input that shapes one embedding, from 10 m to 1000 km. Vertical axis (categorical): how much time one embedding summarises, from a single snapshot at the bottom to a full year of all sources at the top.

Reading the points: TESSERA sits at 10 m on the far left — a per-pixel output with no spatial context at all — and AlphaEarth at 1.28 km, the frame each of its 10-m pixels is computed from; both sit at the top because both fold a whole year into one vector. TerraMind and Granite are single snapshots at 2–13 km tile scale. Prithvi stacks four dates; OlmoEarth stacks twelve monthly steps.

Earth 2, our codec, is the purple dot at about 84 km: each channel token sees a 3-by-3 neighbourhood of quarter-degree cells (a quarter degree is about 28 km) at one monthly or five-day mean. It is a snapshot stencil roughly seven times wider than the widest Earth-observation tile here (Granite, 12.6 km). The hollow dot is the proposed "pixel-year" experiment from a later slide, which would move us up to the "one year folded" row.

Two design axes separate all these models: per-pixel versus patch, and snapshot versus period summary. Every later recommendation in the deck comes back to these two axes.`,

// 15 — summary I
`A reference table of what goes into each embedding, condensing the model slides. Column by column:

Inputs: the sensors and data sources. S-1 and S-2 are Sentinel-1 (radar) and Sentinel-2 (optical); HLS is the harmonised Landsat–Sentinel product; OLCI and SLSTR are the Sentinel-3 ocean-colour and temperature instruments.

Spatial stencil: what area feeds one embedding — the output pixel size, the token size (the patch the network reasons about) and the tile size (the block it processes at once).

Temporal stencil: whether the model summarises a period (AlphaEarth, TESSERA), stacks dated frames (OlmoEarth, Prithvi), or takes a single snapshot (TerraMind, Granite).

Output: the vector dimension and what one vector describes — a pixel-year, a token, a tile.

The table is designed to be read across a row to reconstruct a model's stencil, or down a column to compare one design choice.`,

// 16 — summary II
`The companion table: what each project actually hands you, which is often as important as the architecture.

Open weights: whether the trained network is downloadable, so you can run it on your own data or fine-tune it. AlphaEarth is the exception — no weights, no code.

Precomputed embeddings: whether a ready-made global product exists. AlphaEarth and TESSERA yes; OlmoEarth on demand through its Studio; the others no.

Licence: CC-BY and CC0 are Creative Commons licences (attribution required, or no restrictions); Apache 2.0 and MIT are permissive software licences; OlmoEarth's custom licence has use restrictions. Prithvi's licence is stated differently on Hugging Face and GitHub, hence "verify".

Ocean: green means open ocean is in the training data, red means it is masked out or not targeted. Only Granite is green among the six.

Intended use: "frozen embeddings" means you keep the network fixed and train only a small head on top; "fine-tuning" means you retrain the network for your task; kNN is k-nearest-neighbours, the simplest possible classifier, used to test how good raw embeddings are.`,

// 17 — who embeds the ocean, and how deep
`This slide answers a question we kept returning to: which representation models have ever seen the sea in their training data, and how deep do they go?

Earth 2 (ours) is in the first row because it is the only entry whose inputs include the ocean interior — temperature and salinity at depth from Argo floats (the global fleet of autonomous profiling instruments), plus sea-surface height, currents and wind at the surface. Our stencil is a 3-by-3 patch of quarter-degree cells, one monthly or five-day snapshot, with 24 months of history in the second-stage forecasting model. Output: 64 numbers per pixel-month.

Granite-Geospatial-Ocean (previous slide) and OceanSAR are the two published encoders trained deliberately on sea pixels — but both are surface-only. Granite sees ocean colour and temperature; OceanSAR (from the company Galeio) sees Sentinel-1 radar "wave mode" images of the sea surface, about 12 million of them, and predicts sea state, wave height and wind speed.

Copernicus-FM (TU Munich) covers "the whole land surface and near-land ocean" on a quarter-degree grid — coastal seas are incidental. Weather and climate models (Prithvi WxC, Aurora, ClimaX) contain ocean grid cells only because reanalysis fields are global; the sea is background, not the learning target. Location encoders such as SatCLIP work on coordinates, not ocean state.

Verdict: nothing else embeds the ocean interior through time. Forecast systems such as GLONET or Samudra learn ocean dynamics but publish forecasts, not reusable embeddings.`,

// 18 — four interacting spheres
`Chris's framing of the long-term goal: the Earth system has four interacting "spheres", and eventually we want one representation that spans them all. This slide maps who covers which today and proposes an order for joining them.

a) Atmospheric weather: covered by forecast models (Prithvi WxC, Aurora, GraphCast and others) that produce states, not reusable embeddings. Native timescale hours to days; the atmosphere is predictable for about ten days.

b) Ocean weather: the surface is covered by OceanSAR, Granite and ECMWF's 2026 AIFS ocean extension; the interior only by Earth 2. Timescales run from days at the surface to decades in the deep. A physical fact drives everything: the top two and a half metres of ocean hold as much heat as the entire atmosphere above it, so the ocean is the slow, remembering partner.

c) Ocean biosphere: phytoplankton, ocean colour, carbon uptake. Granite plus task-specific carbon-flux models (SOM-FFN, CSIR-ML6, CANYON-B). Timescales: blooms in weeks, seasons, decadal trends.

d) Land biosphere: the entire mainstream of Earth observation models, all of which fold a year and none of which touch the sea.

Nobody represents two spheres jointly. The proposed sequence: start from the ocean interior (longest memory, least covered by anyone else — what we already have), add ocean colour (same grid, Granite's recipe exists, and it closes the physics-to-biology loop through mixed-layer depth, light and temperature), then treat the atmosphere as a forcing (already in our channels as wind and fluxes), and add land last.

Three design consequences follow: each sphere needs its own speed-and-memory "cone" (introduced on slide 26); biospheres run on the calendar so they want year-summary tokens while physics wants dated snapshots; and the observing systems differ in shape (grids, sparse profiles, cloud-gapped swaths, 10-m tiles), which our distinction between "hidden from the model" and "never measured" tokens is built to handle.`,

// 19 — head-to-head I
`The first of four slides with actual benchmark numbers, transcribed from the papers. This one uses the widest comparison available: Table 2 of the OlmoEarth paper, which evaluates many models with a single frozen-embedding protocol.

"Frozen" means the encoder is not retrained; only a trivial classifier sits on top. kNN (k-nearest neighbours) assigns a test image the majority label of its 20 nearest training embeddings — the purest test of embedding quality. "Linear probe" fits a single linear layer, used for segmentation and multi-temporal tasks.

Columns are benchmark datasets: BigEarthNet and So2Sat (land-cover classification), EuroSAT (scene classification), CropHarvest-Togo (crop vs non-crop), m-cashew and SA-crop (crop segmentation), PASTIS (crop parcels through time), MADOS (marine debris and oil), Sen1Floods11 (flood mapping from radar). Scores are accuracy or micro-F1 for classification and mIoU for segmentation — all 0 to 100, higher is better.

Shaded rows are the models in this deck; grey rows are other models for context. Green is the best in each column across all rows.

What it says: on frozen features OlmoEarth leads the classification columns by large margins (So2Sat 68 vs 47 for TerraMind vs 35 for Prithvi), TerraMind leads BigEarthNet and the cashew and SA-crop segmentation, and Prithvi trails on every classification column. TESSERA appears once, on CropHarvest-Togo, within one point of OlmoEarth. AlphaEarth is absent from this table because its embeddings are annual and could not be run under this protocol.

Caveat, which applies to every table in this section: it was authored by one of the contestants.`,

// 20 — head-to-head II
`Two more tables. The top one is OlmoEarth's Table 3: the same models, now fully fine-tuned — the whole network is retrained for each task (the encoder is unfrozen after the first 20 % of training). The bottom one is TerraMind's Table 6 on the PANGAEA benchmark, a community suite of about eleven segmentation, change-detection and regression tasks, evaluated with a frozen encoder plus a standard segmentation decoder (UPerNet).

Top table: once fine-tuned, the gap between models closes. TerraMind Large and OlmoEarth Large split the columns roughly evenly, and Prithvi-EO 2.0 (600 M) sits one to three points behind on the classification columns but eight to thirteen points behind on the PASTIS and MADOS segmentation columns. The last row is the same OlmoEarth architecture trained from random initialisation — no pretraining at all. The distance between that row and the others (10 to 37 points) is the value of pretraining itself.

Bottom table: mIoU per dataset plus an average and an average rank. TerraMind Large is the only foundation model whose average (59.6) beats a U-Net trained from scratch (57.6). Prithvi appears only as version 1.0 and comes last. HLS Burn Scars, FiveBillionPixels, DynamicEarthNet, SpaceNet 7 and AI4SmallFarms are the other PANGAEA datasets.

Not shown: Prithvi-EO 2.0's own GEO-Bench table, because none of the other models in this deck appear in it.`,

// 21 — head-to-head III
`This slide compares the two "products" — the models whose main deliverable is a precomputed global embedding layer: AlphaEarth and TESSERA.

Left table, from the TESSERA paper: frozen embeddings plus a light head, on six tasks. F1 and mIoU are higher-is-better (arrows up); RMSE — root-mean-square error, in tonnes per hectare for biomass and metres for canopy height — is lower-is-better (arrows down). TESSERA beats AlphaEarth on five of six, with the largest margins on pixel-wise crop tasks where a per-pixel time series matters most (Austrian crop F1 82 vs 56). AlphaEarth edges PASTIS-R by 0.4 points. Both products beat every frozen ViT-style model by wide margins.

Top-right table, from the OlmoEarth paper: AlphaEarth versus OlmoEarth under three protocols. With both frozen, results are mixed. Fine-tuned OlmoEarth wins all five tasks — and the "n/a" row is the point: AlphaEarth cannot be fine-tuned, because only the embeddings, not the network, are released. LFMC is live fuel moisture content, an L1 (absolute) error, lower is better.

Bottom-right, an independent study: mapping Local Climate Zones in Bern with an Attention U-Net fed either raw Sentinel composites, AlphaEarth, or TESSERA. The two products are within one point of each other and two to five points above raw imagery.

AlphaEarth's own paper reports a 23.9 % average error reduction versus the next-best method across 15 evaluations, losing only on land-use change; its per-task values are published as charts without numbers, which is why they are not tabulated here.`,

// 22 — head-to-head IV
`The two ocean encoders cannot be compared with the land models — no shared benchmark exists — so this slide shows each against its own baselines.

Left: Granite-Geospatial-Ocean, Table 1. Five-fold cross-validated RMSE in log10 units for chlorophyll-a and primary production, comparing the fine-tuned foundation model with the same network trained from scratch and with a random forest on raw pixel values. The foundation model is best in both columns, but the margins (0.02 and 0.03) are smaller than the spread between folds. The authors' stronger claims are label efficiency (the advantage grows as training labels shrink) and better spatial pattern (SSIM — structural similarity — of 0.88 versus 0.82 against the operational product).

Right: OceanSAR-1, kNN on frozen features against other SAR foundation models. TenGeoP is a ten-class geophysical-phenomena dataset (accuracy, higher is better); wave height and wind speed are RMSE in metres and metres per second (lower is better). Ocean-only pretraining beats land-trained SAR models by 9 to 23 points on classification and cuts wind-speed error by 30 to 45 %.

Bottom box: the only thing the ocean and land models share is the protocol — frozen features plus a small head, and "foundation model versus same network from scratch" as the control for the value of pretraining. That is also exactly what our attribution matrix does: our pixel and patch codecs versus raw data at matched receptive field. Our pretraining margin (0.617 vs 0.613 correlation for a single pixel; 0.672 vs 0.659 for the 3-by-3 patch) is the same kind of statement as Granite's 0.14 versus 0.16 — a small one.`,

// 23 — digest
`A one-page digest of the four head-to-head slides: for each evaluation protocol and metric family, how the models in this deck rank, with the margin and the source.

Reading down: with frozen features OlmoEarth leads classification and shares segmentation with TerraMind; fine-tuned, TerraMind Large and OlmoEarth Large tie with Prithvi behind; on PANGAEA TerraMind is the only model above a from-scratch U-Net; GEO-Bench-2 (an independent 2025 benchmark that publishes ranks, not scores) places Prithvi 600M-TL fifth and TerraMind Large sixth of fourteen; among the two products TESSERA beats AlphaEarth on five of six tasks in TESSERA's own table but they tie in an independent study; a fine-tuned OlmoEarth beats AlphaEarth everywhere because AlphaEarth cannot be fine-tuned; and the ocean models are measured only against themselves.

The two structural caveats at the bottom are the honest summary. First, no benchmark contains all six models — the widest single table has five and omits AlphaEarth, and the only independent multi-model source has two. Second, no public benchmark has an ocean-state, transport or interior task, so none of these rankings transfer to our problem. What transfers is the protocol: frozen probe versus fine-tune, and a from-scratch control at matched receptive field, which our attribution matrix already implements.`,

// 24 — what this means for our ocean work
`Four takeaways for Earth 2, each tied to something measured.

Axis 1, pixel versus patch. TESSERA shows a per-pixel encoder with no spatial context is competitive on land; every other model buys context with a patch grid. We have already priced this axis in our attribution matrix: reading the RAPID transport with a single-pixel codec gives a correlation of 0.617 (raw data 0.613); with the 3-by-3 patch, 0.672 averaged over two seeds (raw 0.659). Taken at face value that would make the neighbourhood worth about +0.05 and pretraining about +0.01. (RAPID is the mooring array at 26.5°N that measures the Atlantic overturning circulation; "correlation" is how well our read-out tracks it.) Two cautions. These are probe numbers, whose seed-to-seed spread in our own record runs from 0.04 to 0.25, so the +0.05 sits at the edge of the noise band and the +0.01 inside it — the honest reading is "the neighbourhood probably matters; pretraining is unresolved". And on 2 September the paper was reset, because its rolled forecast numbers came from second-stage heads that had seen the held-out years; these probe numbers are unaffected by that (the reset paper keeps them in its appendix), but no forecast number from before 2 September may be quoted anywhere.

Axis 2, snapshot versus period. TerraMind and Granite are snapshots; Prithvi and OlmoEarth stack dated frames; AlphaEarth and TESSERA fold a full year into one vector. Our first-stage codec is a snapshot of a monthly (or five-day) mean, and the month-block codec folds only about six five-day bins. The year-folded design is the axis we have not tested — hence the proposed experiment on the next slide.

Product versus model. AlphaEarth is a dataset you cannot re-run; TESSERA and OlmoEarth ship both. For the ocean nothing precomputed exists at all.

The gap. Only two published encoders are trained on sea pixels — Granite and OceanSAR — and both are surface, single-instant models. To our knowledge no published model embeds the gridded ocean state through time. That is the territory our codec occupies.`,

// 25 — proposed arm
`A concrete experiment proposal, framed honestly. An earlier draft of this deck suggested a "1-by-1 pixel-only control"; it turned out our paper already contains it (the attribution matrix has both single-pixel and 3-by-3 versions of raw and codec inputs). So the spatial question has been measured, at the resolution probe numbers allow; what TESSERA adds that we have not tried is the temporal fold.

What TESSERA does that we don't: it produces one embedding per pixel-YEAR from about 40 dated observations of one pixel, using a temporal transformer and day-of-year encodings, trained with Barlow Twins (no reconstruction) and quantised to int8 during training. Our first-stage vector z describes one pixel-MONTH; a second-stage model then runs over the sequence of monthly vectors.

Arm A — the pixel-year codec: an encoder that reads the 12 (or 24) pixel-months of one quarter-degree cell at once, with month-of-year tokens, and emits one 64-number vector per pixel-year. We keep the patch size at 1 so that only the time axis changes. Token count about 12 times (channels + 2), roughly 310 at 24 channels — the same order as our existing 282-token month-block codec, so it fits the current training recipe.

Arm B — swap the training objective: same encoder, but Barlow Twins on two random subsets of dates from the same pixel-year, instead of masked reconstruction.

Controls: the average of the 12 monthly vectors (does folding beat pooling?), the current stage-2 pathway, and the raw 12-month stack fed to the same attention head (the end-to-end control from the attribution matrix). At least two codec seeds, because head numbers from one seed have proven noisy.

Read-outs: our probe ladder to the RAPID transport (correlation and RMSE in Sverdrups), the fraction of variance lost on the round trip, and rolled skill under the corrected protocol — the mean skill against climatology and against damped persistence at each lead, with trained and held-out longitudes reported separately; the old "corridor AUC" name was retired on 2 September 2026 when the paper was reset. Success means the folded vector beats the control beyond block-bootstrap intervals, with at least three seeds.`,

// 26 — dependency cone
`This slide introduces the physical idea Chris proposed: what can influence a pixel Δt ahead is bounded by how fast each process propagates and how long it remembers.

Notation first. t is "now", the last observed step. Δt is the forecast step — we predict the target at t+Δt. ℓ (a script L) is the lag of an input: it was observed at t−ℓ, so it sits Δt+ℓ before the target; ℓ = 0 is the newest input. v is a process's signal speed; τ (tau) is its memory — how long it stays predictable.

The model: for a process p, an input at lag ℓ can matter only if it is close enough to have travelled to the target in time, |Δx| ≤ v·(Δt+ℓ), and only if the process still remembers, Δt+ℓ ≤ τ. The first condition is the domain-of-dependence or "light-cone" argument from numerical physics (the CFL condition); the second says that beyond a process's predictability its history is noise.

The chart is log-log: time separation on the horizontal axis (1 day to 10 years), distance on the vertical (10 km to 20,000 km), so a constant speed is a straight diagonal. Each process is a region: the atmosphere (amber) is wide but shallow — 10 m/s but only 10 days of memory; the surface ocean (blue) is narrow but months deep — 0.1 m/s, 8 months; the interior (teal) is narrower still but years deep — 0.03 m/s. The dashed line is the Kelvin-wave speed along coasts and the equator, an exception that is fast but anisotropic.

The union rule: a channel forced by several processes has the union of their cones. Sea-surface temperature is forced by the atmosphere (heat flux) and by currents (advection), so its region is wide-and-shallow plus narrow-and-deep — an L-shape, not a cone.

Where our stencils sit: a fixed stencil is a cylinder — same reach at every lag. Our ring-8 at 222 km per month is an ocean-advection speed (0.085 m/s); our sunflower-89 at 4444 km is 1.7 m/s, inside the atmosphere's cone at a one-month step. The dashed vertical line marks the 24 months of history our second-stage model sees.`,

// 27 — what the cone predicts
`The cone model applied channel group by channel group, with the useful reach per monthly step, the useful history, and a design consequence for each.

Wind, pressure and fluxes: the atmosphere moves at 10 m/s and forgets in 5–10 days. Its causal reach in one month is the whole globe, but because its memory is shorter than our step, monthly wind is not predictable from its own history — only through slow ocean modes that force it. So treat it as a wide, shallow forcing: large stencil, no depth. At the five-day step the memory and the step are similar, so wind becomes partly self-predictable — a gain only a pentad model could show, and one that has not been shown: the earlier pentad rolls were withdrawn with the paper reset.

Sea-surface and mixed-layer temperature: advected at 0.1–0.3 m/s (260–800 km per month), forced by the atmosphere, remembered for 3–12 months (including "re-emergence" — anomalies that sink in autumn and reappear next spring). It needs both the wide wind stencil and months of history: the one group that justifies the full cylinder.

Interior temperature and salinity: 0.02–0.1 m/s, memory of years to decades. Per monthly step the useful reach is 50–300 km, which a 3-by-3 patch at quarter degree already covers; our 4444-km stencil is 15–90 times too wide for one step. Small reach, long history — the concrete answer to the open question "can reach come down?" from the project's dependency-cone note.

Sea-surface height: slow Rossby waves in the interior (0.03 m/s at mid-latitudes), fast Kelvin and coastal waves along boundaries (1–3 m/s). An anisotropic stencil — an arm along coasts, narrow elsewhere.

The transport target (RAPID): the overturning transport is set by the east-west density difference across the basin ("thermal wind"), which means its cone is the whole section — and why a read-out must see the whole section rather than a spatial mean, which would average away exactly that east–west contrast (an argument from the mechanism; the measured pooled-versus-unpooled gap is a two-interval comparison our own rules do not allow as evidence).

Bottom left: four falsifiable predictions, each an ablation to run under the corrected roll protocol of the reset paper (none has been run; the earlier rolled numbers were withdrawn). Bottom right: where the simple picture needs care — v is the fastest signal speed, not the mean flow; τ is predictability, not causality; cones are tilted, not symmetric; monthly averaging turns anything faster than the step into forcing.`,

// 28 — worked example
`A concrete walk-through of the cone idea for one target: the surface current (u, v) at a quarter-degree pixel, one month ahead. Each row is a driver — something that pushes the current around — with its mechanism and its two numbers, speed and memory, plus a correlation length.

Wind stress acts through Ekman transport (the wind drags the surface layer, which turns at right angles because of Earth's rotation), within about a day. Because its 10-day memory is shorter than the monthly step, only the newest input (lag 0) is useful, and only as forcing that is not itself predictable.

Air temperature acts through heat flux, which changes mixed-layer temperature, which changes density gradients, which drive geostrophic currents. This is the subtle row: the atmosphere forgets in 10 days, but the mixed layer remembers the heat it received for months and carries it along at ocean speed — so the channel inherits the ocean's memory (6 months) and speed (0.15 m/s). Its reach is the 1,500-km atmospheric correlation length plus what the ocean advects.

Ocean surface temperature drives currents through horizontal density gradients ("thermal wind") — which is why neighbours are mandatory: a gradient does not exist inside one pixel.

Interior temperature and salinity: same mechanism, integrated from depth; at 0.03 m/s (78 km per month) the 3-by-3 patch already covers lag 0.

Sea-surface height sets the pressure gradient; slow Rossby waves in the interior plus a fast Kelvin arm along coasts.

Own history: momentum persistence and eddies. Individual eddies live four months and longer and drift westward at a few centimetres per second, but at a fixed pixel the current decorrelates in weeks to a few months because they pass through — that decorrelation time, about three months, is the memory that matters here.

The "two rules" box states the arithmetic used throughout: reach at lag ℓ is the larger of v·(Δt+ℓ) and the correlation length, capped at 10,000 km; a lag is useful if Δt+ℓ ≤ τ. There is one deliberate exception, stated in the box: for a fast driver acting through a slow medium (air temperature → heat flux → mixed layer) the two lengths add rather than compete — the flux pattern is laid down over the atmospheric correlation length and then carried away by the ocean — which is why the air-temperature row reads 1,889 to 3,833 km. The notation panel defines ℓ with a timeline. All parameter values are order-of-magnitude illustrations and can be changed live in the Dependency-Cone Explorer page.`,

// 29 — step 1: wedges
`Each driver from the previous slide becomes a wedge on this chart. Horizontal axis: lag ℓ, how many months before now the input was observed (0 to 24). Vertical axis (logarithmic): how far from the pixel the input may sit and still matter, 10 km to 10,000 km.

A wedge rises to the right because older inputs have had more time to travel — reach grows as v·(Δt+ℓ) — and stops abruptly where the driver's memory τ runs out. Wind is a tall spike at lag 0 only. Air temperature is a wide band for six lags. Surface temperature is a true wedge for eight lags. Interior temperature and salinity and sea-surface height are low and long. Own history is short.

The bold black line is the union of all wedges: the region a stencil for this target should follow. Its shape is an L: a spike at lag 0 (the atmosphere, wide but instant), a shoulder out to 6–8 months (mixed-layer temperature and integrated heat flux, 2,000–4,000 km), then a long low tail to 24 months (the interior, under 1,900 km).

Two reference lines: the dashed rectangle is our current "cylinder" — 89 stencil slots reaching 4,444 km at every one of 24 lags; the dotted line at 84 km is the 3-by-3 patch. The cylinder over-covers the tail by a factor of 2 to 50 and cannot cover the lag-0 spike at all.`,

// 30 — step 2: slot map
`This slide turns the wedges into a stencil budget. Our production stencil, the "sunflower-89", spreads 89 slots over a disc of radius 4,444 km and is applied at all 24 lags: 89 × 24 = 2,136 "slot-lags" per channel — the cylinder.

The rule: slots needed for a given reach r scale with area, 89·(r/4,444 km)², floored at 9 (the 3-by-3 patch) and capped at 89. Each row is a driver; each bar is a lag; bar height is the slots needed at that lag; grey stubs are lags outside the driver's memory, which are dropped.

Wind: one full-width sheet at lag 0, nothing else — 89 slot-lags. Air temperature: a wide sheet for six lags, 232. Surface temperature: a true cone from 9 to 44 slots over eight lags, 157. Interior temperature and salinity: a thin 24-lag column at the 9-slot floor for a year and a half before it widens, 238. Sea-surface height: a thin 12-lag column, 108 (its coastal arm is not counted by this isotropic formula). Own history: three short lags, 27.

Total: 851 cone-shaped slot-lags for the six channels against 12,816 for the cylinder — 93 % fewer. Equivalently, the saving could buy 48 months of history for the interior channels at no extra cost. These numbers follow directly from the illustrative speeds and memories; the Explorer page recomputes them if you change any parameter.`,

// 31 — step 3: physics
`The physics behind each wedge, in four cards, and then what is fixed versus what is learned.

Ekman: wind stress τ drives a surface transport of about τ/(ρ f) at right angles to the wind — ρ is water density, f the Coriolis parameter set by latitude — within about one inertial period, roughly a day at 30°. Local, fast, shallow. At monthly cadence this is same-step forcing.

Thermal wind: the vertical shear of the geostrophic current is proportional to the horizontal density gradient, ∂u/∂z = (g/(f ρ₀)) ∂ρ/∂y. A gradient does not exist inside one pixel, which is why the 3-by-3 patch is the floor for temperature and salinity channels, and why heat flux matters through its spatial pattern rather than its local value.

Pressure gradient: the surface geostrophic current is u = −(g/f) ∂η/∂y, where η is sea-surface height. SSH anomalies arrive slowly as westward Rossby waves in the interior and fast as Kelvin or coastal waves along boundaries.

Advection and eddies: momentum persists and mesoscale eddies drift westward at a few centimetres per second. The eddies themselves live four months and more, but at a fixed pixel the current decorrelates in weeks to a few months as they pass through — that is the memory used for "own history". Upstream cells carry what will arrive.

What the stencil fixes versus what attention learns: the cone is the support — which (cell, lag, channel) tokens may enter at all. It encodes only speed and memory per driver. Everything else — which upstream cell matters this month, how strongly an SSH gradient projects onto the current — is learned by the attention weights inside the support. A support too small cannot be learned around; one too large costs slots and admits noise.

Right box: the four ablations specialised to this target, with the predicted outcomes. They are hypotheses; nothing here has been run. The Dependency-Cone Explorer page uses the same formulas with editable parameters.`,

// 32 — related work I
`Where does the dependency-cone idea stand against the literature? Row by row.

The origin is the Courant–Friedrichs–Lewy condition from 1928: a numerical scheme's stencil must contain the physical domain of dependence, |Δx| ≤ v·Δt. Our first rule is exactly this; the CFL condition has no notion of memory.

OceanNet (2024), a regional neural forecaster for the Gulf Stream, uses precisely a wave-speed-times-time argument to justify dropping lateral boundary conditions: Rossby waves at 40°N move about 1 cm/s, so eddies travel only 30–120 km from the open boundary over the forecast horizon. Same arithmetic, used to shrink a boundary rather than to shape a per-channel stencil.

PARADIS (2026) is the architectural version: a "neural semi-Lagrangian" layer gathers features at the departure point x − Δt·u — where the flow came from — turning the CFL limit into a smoothness condition on the velocity. A 20-parameter layer moves a tracer that a U-Net's fixed receptive field cannot. That is the tilted, anisotropic cone made operational, though with one transport field for all variables and no memory truncation.

ClimODE uses advection as an inductive bias but with fixed local and global branches. Limited-area weather models use fixed boundary halos, not sized by wind speed times lead time. Global models such as GraphCast, Pangu and FuXi set reach per step by hop count or window size and grow it with lead time by changing the step — the same reach for every channel. Per-variable tokenisation (ClimaX, Aurora, Prithvi WxC) gives each variable its own embedding, not its own spatial or temporal extent. Ocean emulators such as Samudra prescribe the atmosphere as forcing — our monthly-step inversion in practice.

What we did not find is the combination: truncating each process's cone by its own memory, taking the union per channel, and letting that dictate a channel-specific stencil. A one-day search is not a literature review; treat it as "not yet found", and cite OceanNet and PARADIS as the nearest neighbours.`,

// 33 — related work II
`The four-sphere ambition against the literature, by family.

Coupled emulators: Samudra is a 3-D global ocean emulator (temperature, salinity, currents at depth) trained on reanalysis; SamudrACE (2026) couples it to Ai2's machine-learned atmosphere ACE2. This is the closest structural analogue to our order — ocean interior first, atmosphere joined later — but it emulates a simulator rather than embedding observations, and has no biosphere. ACE2-SOM and ACE2-NEMO couple an ML atmosphere to a slab or dynamical ocean.

ECMWF's 2026 AIFS surface-ocean paper forecasts atmosphere plus sea-surface temperature, salinity, height, currents, waves and sea ice in one "component-agnostic" model, and explicitly gives the slowly evolving ocean and ice fields larger loss weights — the first ML system we found that names and handles the cross-sphere timescale mismatch. Surface only, and a forecast model rather than a reusable embedding.

Aurora (Nature 2025, titled "A foundation model for the Earth system") is one pretrained backbone with sphere-specific fine-tunes; atmosphere-centric, no interior ocean.

Observation-native learning: Aardvark Weather and GraphDOP learn directly from raw observations, skipping reanalysis; 4DVarNet learns to fuse sparse altimetry and temperature into gap-free surface fields. This is the recipe for taking Argo profiles, swaths and ship tracks natively — what "combine all relevant data" needs — but none of them has an Argo-depth encoder.

"Earth-system" foundation models: ESFM (2026) and Copernicus-FM unify heterogeneous sources with missingness tokens — the same primitive as our "masked" versus "never measured" tokens — but neither touches the ocean. The Prithvi family is a set of separate sphere-specific models; AlphaEarth uses ERA5-Land only as an input.

Ocean-biosphere and carbon ML (SOM-FFN, CSIR-ML6, CANYON-B, Granite) are task models with hand-chosen predictors. Coupled data assimilation (Penny & Hamill 2017) is where the timescale problem was first named. Platforms and twins (NVIDIA Earth-2, Destination Earth, EDITO) share the goal but are not learned embeddings — and NVIDIA's "Earth-2" is a name collision we must disambiguate, as is Aurora's title.`,

// 34 — sources
`Every paper, model card and repository the deck draws on, grouped by model and topic, as short URLs. A few notes on how they were used.

For the six model slides, numbers were taken from the papers and, where possible, verified against the published configuration files: Prithvi-EO 2.0's token sizes come from its Hugging Face config.json (16-pixel patches for the 300 M model, 14 for the 600 M), and Granite-Geospatial-Ocean's architecture from the config and model file in the ibm-granite GitHub repository (42-pixel tiles, 17 channels, embedding 512, 12 layers, patch size 2 by 2, one frame).

The head-to-head tables were transcribed from the papers' own tables: OlmoEarth Tables 2, 3 and 7; TerraMind Table 6; TESSERA Table 1; Granite Table 1; OceanSAR-1 Tables 1–3; GEO-Bench-2 for independent ranks; and an independent Local Climate Zone study for AlphaEarth versus TESSERA.

For the ocean-including models and related work, sources were checked in August 2026 and include several 2026 papers (AIFS surface ocean, ESFM, SamudrACE, PARADIS, ACE2-NEMO). Earth 2 numbers — the attribution matrix and probe ladder — are quoted from the project's own paper draft.

Items listed as "also checked" are things we looked at and excluded, with the reason, so that nobody re-chases them.`

];

// slides 35–46: the generic-embedding input proposal, spliced in before the Sources entry
module.exports.splice(module.exports.length - 1, 0, ...require("./notes_inputs.js"));
