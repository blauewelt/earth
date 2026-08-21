# Hosting: GitHub Pages today, Cloudflare Pages on standby

The globe is served by GitHub Pages from the `gh-pages` branch, under a
**soft** bandwidth limit of 100 GB/month. Soft means GitHub does not bill for
the overage — it may simply stop serving the site. There is no alert, no
grace purchase, and no button that fixes it in the hour it happens.

So the escape hatch is cut in advance. `.github/workflows/deploy-cloudflare.yml`
is a complete, tested Cloudflare Pages deployment that **does nothing at all
until two secrets exist**. GitHub Pages stays live and untouched; nothing about
today's deploy changes. When it is needed, enabling it is five minutes of
clicking, not a migration under load.

Everything below was measured on 2026-08-21 against the real repo and the real
live site, not estimated.

---

## 1 · The two hosts, side by side

| | GitHub Pages (today) | Cloudflare Pages (standby) |
|---|---|---|
| Bandwidth | **100 GB/month, soft** | **Unlimited** |
| Requests | rate-limited, unspecified | **Unlimited static requests** |
| Published site size | **1 GB hard** | not published as a limit |
| Files per site | not published | **20,000** (free) / 100,000 (paid) |
| Max single file | not published | **25 MiB** |
| Builds | 10/hour soft (waived for Actions workflows) | **500/month** |
| Custom domains | 1 per site | **100 per project** |
| Cost | free | free |

Sources: [GitHub Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits),
[Cloudflare Pages limits](https://developers.cloudflare.com/pages/platform/limits/),
[pages.cloudflare.com](https://pages.cloudflare.com/) pricing panel. All five
figures in bold were read from those pages on 2026-08-21.

### Where this site sits against each limit

Measured from the tracked tree at `140c9a2`:

| Quantity | Measured | Limit | Headroom |
|---|---|---|---|
| Browser-facing files | **179** | 20,000 (CF) | 0.9% used |
| Browser-facing bytes | **400 MiB** | — | — |
| Whole published site (what Pages serves today) | **412.7 MiB**, 446 files | 1 GB (GH, hard) | **40% used** |
| Largest single file (`data/glaciers.json`) | **7.05 MiB** | 25 MiB (CF) | 28% used |

**Nothing exceeds any limit, and the tightest one is GitHub's, not
Cloudflare's.** The 1 GB published-site ceiling is 40% consumed, and the three
per-year directories grow every year: `data/oisst_y` (46 files, 162 MB),
`data/currents_y` (34, 116 MB), `data/mld_y` (34, 85 MB). That is the number to
watch — it bites before bandwidth does, and it is a *hard* limit.

---

## 2 · The bandwidth arithmetic

What matters is **bytes off our origin**, and most of what the app draws never
touches it. Measured with a headless Chromium against `index.html`, cold
cache, 1280×720, default layers; transfer sizes are the real
`content-length` values from the live site, which gzips everything
(`content-encoding: gzip`, verified).

A cold first visit fetches **nine files from our origin**:

| File | On the wire (gzip) |
|---|---|
| `data/cities.json` | 168,461 |
| `src/app.js` | 150,612 |
| `data/islands.json` | 108,639 |
| `icon-192.png` | 35,490 |
| `data/catalog.json` | 34,769 |
| `index.html` | 21,065 |
| `src/style.css` | 8,649 |
| `data/stations.geojson` | 1,482 |
| `data/species.json` | 1,507 |
| **total** | **530,674 B ≈ 518 KiB** |

Two large things are deliberately *not* in that list. **Cesium (7.13 MB over
33 requests) comes from cdnjs**, not from us — `index.html` points
`CESIUM_BASE_URL` at the CDN. And **11 tile requests went straight to NASA
GIBS**. Neither is our bandwidth; see §6.

Three visitor profiles, and what each costs:

| Profile | Origin bytes | Visits before 100 GB |
|---|---|---|
| **A** — opens the globe, looks, leaves | 530,674 (0.51 MiB) | **188,000** |
| **B** — A + one pixel-inspector click (`ocean_column`, `ocean_surface`, `climatetrace`, `argo`, `oisst_monthly`, one clim month, one SST year) | 2,617,280 (2.50 MiB) | **38,000** |
| **C** — B + turns on glaciers (2,096,747) + searches a place, pulling the gazetteer (1,205,561) | 5,919,588 (5.65 MiB) | **16,900** |

**The honest headline: between ~17,000 and ~188,000 visits a month, depending
on what people do.** Profile C is the realistic ceiling for an engaged user and
works out to about **555 such visits a day** before the soft limit bites. A
link that goes around and brings twenty thousand curious people in a week would
exceed it.

Two things make the real number worse than the table:

- **`cache-control: max-age=600`.** GitHub Pages caches for ten minutes.
  A returning visitor an hour later pays the full 518 KiB again. There is no
  setting to change this on Pages; Cloudflare Pages lets you set
  `_headers` and would cut repeat-visit bandwidth substantially.
- Assets are stamped (`?v=<sha8>`), so every deploy invalidates `app.js` and
  `style.css` for everyone — correct behaviour, and ~160 KB per returning
  visitor per deploy.

---

## 3 · The build budget — why the workflow has a `paths:` filter

Cloudflare's free tier allows **500 builds/month**, and this repo commits far
faster than that:

| | 14 days to 2026-08-21 | per 30.44-day month | vs 500/month |
|---|---|---|---|
| All commits | 594 | **1,292** | **2.6× over** |
| Commits touching browser-served files | 99 | **215** | 43% of quota |

A naive Git-integration deploy — Cloudflare building on every push — would burn
the whole month's quota in eleven days on ML research commits that change
nothing a browser fetches. So the workflow triggers only on the paths the
browser is actually served:

```
index.html  status.html  docs.html  manifest.json  .nojekyll
icon-*.png  src/**  lib/**  data/**  docs/**
```

That list was verified against the real asset graph, not guessed:

- **`_vendor/` is absent** because it is gitignored and never deployed —
  Cesium is loaded from cdnjs in production; `_vendor/cesium` exists only so the
  sandbox test suite can run without egress.
- **`ml/` is absent** because `docs.html` reads every markdown file from
  `raw.githubusercontent.com` on the deployed site (it only reads the working
  copy on `localhost`). No `.md` under `ml/` is ever fetched from our origin.
- **`docs/` is present** because `index.html` links `docs/PIXEL_STATE.md`,
  `docs/COMBINING_DATASETS.md`, `docs/SPECIES_AND_CLIMATE.md` and
  `docs/CATALOG.md` with *relative* hrefs, which would 404 without it.

The filter was then **replayed against real commit history** rather than
eyeballed: every commit in the last 14 days was re-run through the glob
semantics GitHub uses (`*` stops at `/`, `**` does not). Result: 99 build, 495
skip, and of the **297 commits in the last 30 days that touch only `ml/`, zero
trigger a build.**

---

## 4 · Turning it on — the exact click-path

Nothing here changes the live site. Do it now, while there is no fire.

1. **Sign in** at [dash.cloudflare.com](https://dash.cloudflare.com/) (create a
   free account if needed — no card).
2. **Copy the Account ID.** It is in the right-hand sidebar of the account home
   page, and also in the URL: `dash.cloudflare.com/<account-id>/...`. A 32-character hex string.
3. **Create the Pages project.** *Workers & Pages → Create → Pages → Use direct
   upload* (**not** "Connect to Git" — Git integration is the thing that would
   burn 1,292 builds a month). Name it **`earth`**. If you name it something
   else, add a repository *variable* `CLOUDFLARE_PAGES_PROJECT` with that name.
   Cloudflare will ask for an initial upload; any single placeholder file is
   fine, the workflow replaces it.
4. **Create the API token.** *My Profile → API Tokens → Create Token → Create
   Custom Token*. Permissions: **Account → Cloudflare Pages → Edit**. Nothing
   else — no Zone permissions, no DNS. Copy the token; it is shown once.
5. **Add the two repo secrets.** GitHub →
   [Settings → Secrets and variables → Actions](https://github.com/blauewelt/earth/settings/secrets/actions)
   → *New repository secret*, twice:
   - `CLOUDFLARE_API_TOKEN` — the token from step 4
   - `CLOUDFLARE_ACCOUNT_ID` — the ID from step 2
6. **Fire it once by hand.** Actions → *Deploy to Cloudflare Pages (standby)* →
   *Run workflow*. It will assemble 179 files (~400 MiB) and upload them. First
   upload takes a few minutes; later ones only send changed files.
7. **Check it.** Open `https://earth.pages.dev/` (or whatever the project's
   `*.pages.dev` name is). The globe should look identical — every asset path
   in the app is relative, so it works equally at `/earth/` and at `/`.
   If the run's output shows a *preview* URL with a random subdomain instead,
   the project's production branch is not `main`: *Settings → Builds &
   deployments → Production branch* → set it to `main`, and re-run. The
   workflow deploys with `--branch main` deliberately, so that production and
   the repo's default branch are the same thing.

Until step 5 is done the workflow still runs on qualifying pushes and **exits
green**, printing a `::notice::` that says it is on standby and what to
configure. It never shows a red X for a feature nobody enabled.

---

## 5 · What a cutover actually looks like

Both hosts can run indefinitely, serving the same content from the same branch.
So "cutover" means *changing where the links point*, and it is a settings
change, not a deploy:

1. Confirm the Cloudflare copy is current (Actions → last green run).
2. Point people at the Cloudflare URL. If a custom domain exists, move the DNS
   record; if not, this is a link change everywhere.
3. Optionally leave GitHub Pages up. It costs nothing and it is the rollback.

**Three things in the repo hardcode the GitHub Pages origin** and would need
updating on a real cutover (they are correct today and are listed so nobody has
to find them under pressure):

- `status.html` — `DOCS_BASE = "https://blauewelt.github.io/earth/docs.html"`,
  and a "Live app" link in the header
- `tests/status.spec.js` asserts that exact absolute URL, so it fails the moment
  the page changes and will not let the change go unnoticed
- `README.md` (four links) and `CLAUDE.md` (§0b's examples, and the header line)

`manifest.json` needs nothing: `start_url` and `scope` are relative, so the
installed PWA follows whichever origin it was installed from.

### The part that is not free: the URL changes

The site is `https://blauewelt.github.io/earth/`. On Cloudflare it becomes
`https://earth.pages.dev/` (or a custom domain). **Every link ever shared
breaks** — and this project shares a *lot* of links, by standing rule:

- every `docs.html?f=ml/EXPERIMENTS.md#e-026b` posted in chat
- every `status.html#run-427` in a session report
- every figure and PDF link in the paper's trail

Those live in chat history, in the paper, and on Chris's phone. Nothing rewrites
them.

**The fix is a custom domain, and its value is that it is bought before it
matters.** A domain pointed at GitHub Pages *today* makes every link issued
from today onward portable: a cutover then becomes one DNS record and no broken
link at all. Bought after the traffic spike, it fixes nothing that was already
shared. GitHub Pages supports a custom domain (Settings → Pages → Custom
domain, plus a `CNAME` file) and so does Cloudflare Pages (100 per project,
free) — the same domain can point at either. This is the single highest-value
five-minute action in this document, and it is the one with a deadline.

---

## 6 · What Cloudflare does not fix

**Tiles.** Every map tile is a **direct browser → NASA GIBS request**, and GBIF
occurrence tiles are the same shape. Nothing of ours stands in front of either
— no origin, no CDN, no cache we control. Moving our static files to Cloudflare
changes that not at all: the 11 GIBS requests measured on a cold page load in §2
went to `gibs.earthdata.nasa.gov` and would go there from any host. A CDN in
front of *our* files cannot shield a service our files never proxy.

That budget is a separate subject with its own document and its own rule:
**[docs/TILE_BUDGET.md](?f=docs/TILE_BUDGET.md)** — the measured GIBS request count per
click, drag, window and playback frame, the unbounded paths that were found and
closed, and the check to run before adding any tile-issuing feature. It is the
more important of the two documents, because GIBS is a public, taxpayer-funded
service where behaving badly gets you blocked, and no amount of bandwidth buys
that back.

**The Open-Meteo and Hugging Face calls** (CLAUDE.md §3's two deliberate
exceptions) are likewise browser-to-third-party and unaffected.

**The 1 GB published-site limit** is GitHub's, and it applies to the repo we
would keep serving from Pages as a fallback. Cloudflare does not raise it; only
moving `data/`'s per-year archives off the repo would.

---

## 7 · Design notes on the workflow

Three choices in `.github/workflows/deploy-cloudflare.yml` worth knowing:

- **`runs-on: ubuntu-latest`, always.** `ml/CLAUDE.md` §6: the one line of
  defence for the self-hosted GPU boxes is that no fork-triggerable workflow
  ever reaches them. This workflow has no `pull_request`, no `workflow_run`, no
  `schedule`, and never `runs-on: gpu`.
- **The upload set is an allowlist, not a subtraction.** The job runs
  `git ls-files -z -- <allowlist> | tar --null -T - -cf - | tar -xf - -C _site`.
  An allowlist fails *closed*: a new top-level directory of model checkpoints is
  excluded by default instead of silently shipping 300 MB. It also means
  untracked files and `_vendor/` can never ride along. `ml/`, `tests/`,
  `scripts/`, `.github/`, `README.md`, `CLAUDE.md` and `node_modules/` are
  excluded because they are simply not on the list.
- **Secrets are read from `env:`, never argv.** `wrangler` reads
  `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` from the environment; only
  the non-secret project name appears on the command line. The presence check
  that keeps the job inert is `secrets.X != ''` reduced to a boolean in the
  job's `env:` (where the `secrets` context is available) and read as
  `env.CONFIGURED` in step conditions (where it is available and `secrets` is
  not). The boolean is all that is stored; no token value reaches a log.
- **The job checks Cloudflare's own limits before uploading** — over 20,000
  files or any file over 25 MiB fails the run with an explicit error, rather
  than failing halfway through a 400 MiB upload.
