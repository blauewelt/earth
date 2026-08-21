# Hosting: GitHub Pages today, Cloudflare Pages on standby

The globe is served by **GitHub Pages**, published by the `pages.yml` Actions
workflow from `main` (`actions/upload-pages-artifact` with `path: .`, so the
*whole tracked repo* is the published site). It runs under a **soft** bandwidth
limit of 100 GB/month. Soft means GitHub does not bill for the overage — it may
simply stop serving the site. There is no alert, no grace purchase, and no
button that fixes it in the hour it happens. The same page also forbids using
Pages "as a free web-hosting service to run your online business".

> **Correction, 2026-08-21.** This document and CLAUDE.md §6 both used to say
> "GitHub Pages serves the `gh-pages` branch". It does not, and has not for a
> while. `GET /repos/blauewelt/earth/pages` returns `"build_type": "workflow"`,
> and the live deployment on 2026-08-21 was for `0c84d800` (main's tip) while
> `gh-pages` still stood at `27319fd4`, two commits behind. The `gh-pages`
> branch is maintained by habit and is harmless; it is not what browsers get.
> This matters for a cutover, because "what is live" is the thing being copied.

So the escape hatch is cut in advance. `.github/workflows/deploy-cloudflare.yml`
is a complete Cloudflare Pages deployment that **does nothing at all until two
secrets exist**. GitHub Pages stays live and untouched; nothing about today's
deploy changes. When it is needed, enabling it is five minutes of clicking, not
a migration under load.

Everything below was measured on 2026-08-21 against the real repo, the real
live site and the two hosts' own documentation, not estimated.

---

## 1 · The two hosts, side by side

| | GitHub Pages (today) | Cloudflare Pages (standby) |
|---|---|---|
| Bandwidth | **100 GB/month, soft** | **Unlimited** |
| Requests | rate-limited, unspecified | **Unlimited static requests** |
| Published site size | **1 GB hard** | not published as a limit |
| Files per site | not published | **20,000** (free) / 100,000 (paid) |
| Max single file | not published | **25 MiB** |
| Builds | 10/hour soft, waived for Actions workflows | 500/month — **only for Git-integrated projects**, §3 |
| Custom domains | 1 per site | **100 per project** |
| Commercial use | **prohibited** | permitted |
| Cost | free | free |

Sources: [GitHub Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits),
[Cloudflare Pages limits](https://developers.cloudflare.com/pages/platform/limits/).
Every figure in bold was read from those pages on 2026-08-21.

### Where this site sits against each limit

| Quantity | Measured | Limit | Headroom |
|---|---|---|---|
| Browser-facing files (the Cloudflare upload set) | **184** | 20,000 (CF) | 0.9% used |
| Browser-facing bytes | **400.8 MiB** | — | — |
| Whole published site (what Pages serves today) | **412.9 MiB**, 454 files | 1 GB (GH, hard) | **40% used** |
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
GIBS**. Neither is our bandwidth; see §7.

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
  setting to change this on Pages. Cloudflare Pages sends
  `public, max-age=0, must-revalidate` instead — every request revalidates, and
  a revalidation that matches the `ETag` costs a 304 with no body rather than
  518 KiB. For repeat visitors that is a large reduction, and `_headers` would
  let us go further and mark the `?v=`-stamped assets immutable.
- Assets are stamped (`?v=<sha8>`), so every deploy invalidates `app.js` and
  `style.css` for everyone — correct behaviour, and ~160 KB per returning
  visitor per deploy.

---

## 3 · The build budget — and why the 500/month number does *not* apply

Cloudflare's free tier allows **500 builds/month**, and this repo commits far
faster than that. That was the original reason for the workflow's `paths:`
filter. **The reason has since been checked against the source and is
narrower than it looked**, and the filter survives on a different argument.

Cloudflare's limits page introduces the number like this:

> "**Each time you push new code to your Git repository, Pages will build and
> deploy your site.** Build limits depend on your plan: … Builds per month:
> 500."
> — [Cloudflare Pages limits](https://developers.cloudflare.com/pages/platform/limits/)

That quota meters Cloudflare's *own build system*, which only runs for
[Git-integrated](https://developers.cloudflare.com/pages/get-started/git-integration/)
projects. This workflow uses
[Direct Upload](https://developers.cloudflare.com/pages/get-started/direct-upload/):
GitHub Actions assembles `_site` and `wrangler pages deploy` uploads it.
Cloudflare builds nothing, so there is no build to meter. **Do not treat 500 as
a hard ceiling we are spending against, and do not use it to justify a decision
on its own.**

The filter stays, for reasons that hold regardless:

| | 30 days to 2026-08-21 | vs an unfiltered deploy |
|---|---|---|
| Commits on `main` | 759 | — |
| Commits matching the filter (would deploy) | **214** | 28% |
| Commits touching only `ml/` | 298 | — |
| …of those, deploying | **5** | all five change an HTML figure the site serves |

Re-verified by replaying every commit in the window through GitHub's own glob
semantics (`*` stops at `/`, `**` does not), not by eyeballing the list.
Publishing an identical 400 MiB site 545 extra times a month buys nothing, adds
545 entries of deployment history to a project that
[cannot be deleted past ~100 deployments](https://developers.cloudflare.com/pages/platform/known-issues/)
without deleting them one at a time, and is 545 more chances for a partial
upload to be the live one.

The filtered set is the files a browser is served:

```
index.html  status.html  docs.html  404.html  manifest.json  .nojekyll
icon-*.png  src/**  lib/**  data/**  docs/**
ml/index.html  ml/figs/*.html  ml/paper/figs/*.html
```

Verified against the real asset graph *and* against what the live site returns
200 for:

- **`_vendor/` is absent** because it is gitignored and never deployed —
  Cesium is loaded from cdnjs in production; `_vendor/cesium` exists only so the
  sandbox test suite can run without egress.
- **Most of `ml/` is absent** because `docs.html` reads every markdown file from
  `raw.githubusercontent.com` on the deployed site (it only reads the working
  copy on `localhost`), and the paper PDFs are linked as `github.com/blob`
  URLs, which GitHub renders. No `.md` and no `.pdf` under `ml/` is ever
  fetched from our origin.
- **`docs/` is present** because `index.html` and the pixel card link
  `docs/PIXEL_STATE.md`, `docs/COMBINING_DATASETS.md`,
  `docs/SPECIES_AND_CLIMATE.md` and `docs/CATALOG.md` with *relative* hrefs,
  which would 404 without it.
- **`ml/index.html` is present** — and it is the one an earlier draft of this
  allowlist got wrong. It is the `/earth/ml` shortcut Chris asked for (a meta
  refresh to `../status.html`), it is live and returns **200** today, and
  excluding it would have taken the shortcut away silently. The workflow now
  asserts a manifest of entry points after assembling `_site`, so an allowlist
  that stops matching a required file fails the run instead of deploying a
  complete-looking broken site.
- **`ml/figs/*.html` and `ml/paper/figs/*.html` are present** because CLAUDE.md
  §0b requires an HTML figure to be linked at the *Pages* origin — a
  `github.com/blob` URL renders it as source. Both are self-contained (no
  sibling fetches). They cost 5 deploys a month.

---

## 4 · Is deploying to Cloudflare faster or slower than to GitHub Pages?

Chris asked this directly. The measured answer: **to the origin it is roughly a
tie, around 40–60 seconds either way. To the person looking at the page,
Cloudflare is faster by up to ten minutes**, and the difference is caching, not
deployment.

### GitHub Pages — measured, from this repo's own Actions history

Ten successful `pages.yml` runs, deploy-job timings read from
`GET /repos/blauewelt/earth/actions/runs/{id}/jobs`:

| Stage | Median | Range |
|---|---|---|
| Push → runner picks the job up | **4 s** | 3–5 s |
| `deploy` job wall clock | **40 s** | 34–83 s |
| ↳ `actions/checkout@v4` | 6 s | |
| ↳ `actions/upload-pages-artifact@v3` (whole repo, 412 MiB) | 19 s | |
| ↳ `actions/deploy-pages@v4` | 11 s | |
| **Push → origin serves the new bytes** | **≈ 44 s** | 38–88 s |

Confirmed end to end on run #710: pushed at 13:39:40 UTC, the live
`index.html` carried `last-modified: 13:40:21 GMT` — **41 s**. GitHub's own
documentation states the pessimistic case: *"It can take up to 10 minutes for
changes to your site to publish after you push the changes to GitHub"*
([Creating a GitHub Pages site](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site)).

Then the CDN. GitHub Pages serves through Fastly with a fixed
`cache-control: max-age=600`, and it really binds — measured live on
2026-08-21, four minutes after a deploy:

```
$ curl -sSI https://blauewelt.github.io/earth/
cache-control: max-age=600
x-cache: HIT
age: 245
```

**A visitor whose edge already holds the page can be served a build up to ten
minutes old, and there is no way to shorten it.** (The in-app new-build check
is exempt: `checkForNewBuild()` requests `index.html?fresh=<now>` with
`cache: "no-store"`, and the nonce is part of Fastly's cache key — measured,
that request returns `x-cache: MISS, age: 0` and reaches the origin. So the
reload toast is honest even while the plain URL is stale. It only fires on
foregrounding, every 15 minutes, and 10 s after load.)

### Cloudflare Pages — from Cloudflare's documentation

- **The deploy.** `wrangler pages deploy` uploads `_site` and Cloudflare
  distributes it: *"Every deployment will be distributed right to the Cloudflare
  network within seconds"*
  ([Introducing Direct Uploads](https://blog.cloudflare.com/cloudflare-pages-direct-uploads/)).
  Our job is the same shape as the GitHub one — checkout, assemble, upload —
  minus the 19 s artifact tar of the whole repo and plus an `npx` fetch of
  wrangler, so **expect the same 40–60 s band**. Wrangler uploads only assets
  Cloudflare does not already hold (the documented `--skip-caching` flag exists
  precisely to turn that off), so a typical deploy sends one or two changed
  files, not 400 MiB. The **first** deploy does send 400 MiB and will take
  several minutes.
- **The cache.** This is where it wins. *"Every time you deploy an asset to
  Pages, the asset remains cached on the Cloudflare CDN until your next
  deployment"*, and every cacheable asset is served with
  `Cache-Control: public, max-age=0, must-revalidate`
  ([Serving Pages](https://developers.cloudflare.com/pages/configuration/serving-pages/)).
  A deployment is the invalidation event, and the browser revalidates on every
  request. **There is no staleness window at all.**
  The one documented exception is a *custom domain* with custom Cache Rules on
  it — Cloudflare explicitly warns against adding caching there and tells you to
  Purge Everything if you do. On `*.pages.dev`, and on a custom domain left at
  its defaults, this does not apply.

### The answer

| | GitHub Pages | Cloudflare Pages |
|---|---|---|
| Push → origin has the new bytes | ~44 s (measured) | ~40–60 s (same job shape) |
| Origin → a returning visitor sees it | **0–600 s** (measured `age: 245`) | **0 s** — `max-age=0, must-revalidate` |
| Worst case, someone reloading after a deploy | **~11 minutes** | **~1 minute** |
| First deploy of the whole site | 19 s upload | several minutes (400 MiB) |

**Cloudflare is faster where it is felt.** It is not a faster CI pipeline; it is
a host that stops lying to the browser about how fresh the page is. The
practical effect is that "I pushed a fix, reload it" stops needing the
`?v=<sha8>` stamps and the reload toast to be *believed* — those stay, because
they solve the installed-PWA problem too, but they stop being the only thing
standing between a user and a ten-minute-old page.

---

## 5 · Turning it on — the exact click-path

Nothing here changes the live site. Do it now, while there is no fire. The two
secret names below were read from
`.github/workflows/deploy-cloudflare.yml` itself, not from memory, and they are
the names Cloudflare's own CI guide uses.

1. **Sign in** at [dash.cloudflare.com](https://dash.cloudflare.com/) — create a
   free account if needed, no card required.
2. **Copy the Account ID.** Right-hand sidebar of the account home page, under
   **API**; it is also the 32-character hex string in the dashboard URL,
   `dash.cloudflare.com/<account-id>/...`.
3. **Create the Pages project.** *Workers & Pages → Create → Pages → Upload
   assets* (**not** "Connect to Git"). Name it **`earth`**. Cloudflare asks for
   an initial upload — any single placeholder file is fine, the workflow
   replaces everything.
   Three things about this step are one-way, so get them right now:
   - **Direct Upload cannot be switched to Git integration later** (Cloudflare
     documents this). That is the direction we want.
   - **The `*.pages.dev` subdomain cannot be changed later.** If `earth` is
     taken, Cloudflare appends random characters, and the only fix is to delete
     the project and make a new one. Note the real name it gives you.
   - If the project ends up named something other than `earth`, add a
     repository **variable** (not a secret) `CLOUDFLARE_PAGES_PROJECT` with the
     real name.
4. **Set the production branch to `main`.** *The project → Settings → Builds &
   deployments → Production branch*. The workflow deploys with `--branch main`
   deliberately, so that "production" and the repo's default branch are the same
   thing. If this is wrong, the run succeeds and publishes to a *preview* URL
   with a random subdomain, which looks like a broken deploy and is not one.
5. **Create the API token.** *My Profile → API Tokens → Create Token → Create
   Custom Token*. Permissions: **Account → Cloudflare Pages → Edit**. Nothing
   else — no Zone permissions, no DNS. Copy it; it is shown once.
6. **Add the two repo secrets.** GitHub →
   [Settings → Secrets and variables → Actions](https://github.com/blauewelt/earth/settings/secrets/actions)
   → *New repository secret*, twice, spelled exactly:
   - `CLOUDFLARE_API_TOKEN` — the token from step 5
   - `CLOUDFLARE_ACCOUNT_ID` — the ID from step 2
7. **Fire it once by hand.** Actions → *Deploy to Cloudflare Pages (standby)* →
   *Run workflow*. It assembles 184 files (~400 MiB) and uploads them; the first
   upload takes a few minutes, later ones send only what changed. The run
   summary prints the file count, the size and the deployed commit.
8. **Check it** — §6 has the byte-for-byte verification, which is the step that
   decides whether any link ever moves.

Until step 6 is done the workflow still runs on qualifying pushes and **exits
green**, printing a `::notice::` that says it is on standby. It never shows a
red X for a feature nobody enabled. It is loud in the two cases that are
mistakes rather than choices: **exactly one** of the two secrets set (a typo,
not a standby) fails the run, and **Run workflow** pressed while unconfigured
fails the run, because someone asking for a deploy by hand and getting a green
tick and no deploy is the worst possible answer.

---

## 6 · The cutover, as a reversible sequence

**Nothing has to be switched off, ever.** Both hosts can serve the same content
from the same branch indefinitely, at no cost, and GitHub Pages costs nothing to
leave running. So there is no big-bang: "cutover" means *changing where links
point*, and every step below is undone by changing them back.

### Phase 0 — parallel, indefinite (this is the default state)

`pages.yml` and `deploy-cloudflare.yml` both fire on a push to `main`. GitHub
Pages publishes the whole repo; Cloudflare publishes the 184-file browser set.
Both are current within a minute of every push. Nobody's links change. **Stay
here as long as you like** — this is not a staging phase to be got through, it
is a supported configuration.

### Phase 1 — verify the Cloudflare copy is byte-correct

Do not compare by looking at it. The globe looks identical when a data file is
missing, because almost every fetch failure degrades to `null` by design.

1. **Every file, by hash.** For each of the 184 paths in the upload set, fetch
   `https://<project>.pages.dev/<path>` and compare the sha256 with the local
   file. Zero mismatches, zero non-200s. This is the whole test; the rest are
   checks that the *shape* of the site matches.
2. **The four entry points return 200 and the right content type:** `/`,
   `/docs.html`, `/status.html`, `/ml/` (which must land on the status page).
3. **A path that does not exist returns 404, not 200.** Without a top-level
   `404.html`, Cloudflare treats a site as a single-page app and answers *every*
   unknown path with the root document at HTTP 200 — so a dropped data file
   would arrive as HTML and a missing page would look like the globe. `404.html`
   is in the repo and in the upload set for exactly this reason; confirm it is
   working before trusting anything else on this list.
4. **The `.html` → extensionless redirect is understood.** Cloudflare Pages
   *"will redirect HTML pages to their extension-less counterparts"*, so
   `/docs.html?f=x#y` answers with a redirect to `/docs?f=x` (query preserved,
   fragment is client-side and survives). Every shared link still resolves; it
   costs one hop. Check one `docs.html?f=…#…` and one `status.html#run-N` by
   hand, on a phone.
5. **Run the suite against it.** `PLAYWRIGHT_BASE_URL=https://<project>.pages.dev
   npx playwright test tests/app.spec.js tests/docs.spec.js` — note that
   `docs.html` reads from `raw.githubusercontent.com` on any non-localhost host,
   so this exercises the real deployed path rather than the working copy.
6. **Watch it for a week.** Deploy normally; confirm the Cloudflare copy tracks
   `main` and that no ML-only day triggers a deploy.

**Rollback at this phase: none needed.** Nothing has been pointed anywhere.

### Phase 2 — move the links

This is the only irreversible-feeling step, and it is irreversible only in the
sense that links already sent cannot be edited.

1. Update the three places in the repo that name the origin (§8).
2. Update `docs.html`'s and `status.html`'s own links if they moved.
3. Start posting Cloudflare URLs in chat and in session reports.

**Rollback: change them back.** GitHub Pages is still live and still current,
because `pages.yml` was never touched. The rollback is a commit, not a
migration, and it takes the same 44 seconds as any other deploy.

### Phase 3 — retire GitHub Pages (optional, and not recommended)

Only if the 1 GB published-site limit is actually being hit. Deleting the Pages
deployment removes the rollback. There is no cost reason to do it.

### What actually breaks at the moment of cutover

The site is `https://blauewelt.github.io/earth/`. On Cloudflare it becomes
`https://earth.pages.dev/` (or a custom domain). **Every link ever shared
stops resolving** — and this project shares a *lot* of links, by standing rule
(CLAUDE.md §0b):

- every `docs.html?f=ml/EXPERIMENTS.md#e-026b` posted in chat
- every `status.html#run-427` in a session report
- every `ml/figs/*.html` figure link
- the `/earth/ml` shortcut, if anyone typed it into a phone's home screen

Those live in chat history, in the paper's trail, and on Chris's phone. Nothing
rewrites them. Chris has accepted this explicitly — *"the deep links are
normally not saved, and our status page can be changed by us"* — so it is not a
blocker. It is written down so that nobody rediscovers it under pressure.

**Two things reduce the blast radius, and both are cheap:**

- **Leave GitHub Pages running.** Every old link keeps working. This is free and
  it is the reason Phase 3 is optional.
- **A custom domain is the real fix, and its value is that it is bought before
  it matters.** A domain pointed at GitHub Pages *today* makes every link issued
  from today onward portable: the cutover then becomes one DNS record and no
  broken link at all. Bought after the traffic spike, it fixes nothing already
  shared. GitHub Pages supports one (Settings → Pages → Custom domain, plus a
  `CNAME` file) and Cloudflare Pages supports 100 per project, free — the same
  domain can point at either. This is the single highest-value five-minute
  action in this document, and it is the one with a deadline.

---

## 7 · What Cloudflare does not fix

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
exceptions) are likewise browser-to-third-party and unaffected. Note in
particular that E-040's daily-SST reads are HTTP **range** requests against
`huggingface.co`, and that Cloudflare Pages *"currently returns 200 responses
for HTTP range requests"* — so if that archive were ever moved onto our own
origin, Pages would answer a 730-byte range read with a 757 MB file. It is not
on our origin and must not be moved there.

**The 1 GB published-site limit** is GitHub's, and it applies to the copy we
would keep serving from Pages as a fallback. Cloudflare does not raise it; only
moving `data/`'s per-year archives off the repo would.

---

## 8 · Everything that names the origin

Swept 2026-08-21 across `index.html`, `status.html`, `docs.html`,
`manifest.json`, `src/`, `data/`, `lib/` and `.github/workflows/`.

**Nothing in the app breaks when the origin changes.** Every asset, data file
and internal page is fetched by a *relative* path, and `manifest.json` uses
`"start_url": "./index.html"` and `"scope": "./"` — confirmed — so an installed
PWA follows whichever origin it was installed from. The build check
(`checkForNewBuild()`, `src/app.js:8288`) fetches `index.html?fresh=…`
relatively and is origin-agnostic. `docs.html` reads markdown from
`raw.githubusercontent.com` on any non-`localhost` host
(`docs.html:368–372`) and `status.html` reads only `api.github.com` and
`raw.githubusercontent.com` — all cross-origin already, all unaffected.

Four hits name the GitHub Pages origin. All four **degrade** (they keep working,
they just point back at GitHub Pages) as long as Pages stays up, which is the
plan:

| File · line | What | Verdict |
|---|---|---|
| `status.html:204` | `DOCS_BASE = "https://blauewelt.github.io/earth/docs.html"` | degrades — every experiment badge links back to the GitHub Pages copy |
| `status.html:160` | `<a href="https://blauewelt.github.io/earth/">Live app</a>` | degrades — sends readers to the old copy |
| `tests/status.spec.js:480` | asserts that exact absolute URL | fine, and deliberate — it is the tripwire that stops the change going unnoticed |
| `README.md` ×4, `CLAUDE.md` ×2 | documentation links | fine — they are about where the site is, and can be edited any time |

`DOCS_BASE` is absolute on purpose: `status.html` is opened from Pages, from a
`file://` copy and from `localhost` in the suite, and the badge must resolve in
all three. **Do not simply make it relative** — that breaks the `file://` case.
The origin-following form that keeps all three working is:

```js
var DOCS_BASE = (location.protocol === "file:" || !location.host)
  ? "https://blauewelt.github.io/earth/docs.html"          // a saved copy
  : new URL("docs.html", location.href).href;              // follow the origin
```

On GitHub Pages that evaluates to exactly today's string, so the change is
behaviour-preserving in production and only the suite's expectation moves. It is
a Phase-2 edit, listed here so it is a two-line diff rather than a search.

Non-hits worth recording so nobody looks again: `index.html:866`,
`status.html:153–154` and `src/app.js:439` contain `github.com/blauewelt/earth/blob/…`
URLs — those point at the *repository*, not the site, and are correct on any
host.

---

## 9 · Design notes on the workflow

Five choices in `.github/workflows/deploy-cloudflare.yml` worth knowing:

- **`runs-on: ubuntu-latest`, always.** `ml/CLAUDE.md` §6: the one line of
  defence for the self-hosted GPU boxes is that no fork-triggerable workflow
  ever reaches them. This workflow has no `pull_request`, no `workflow_run`, no
  `schedule`, and never `runs-on: gpu`.
- **The upload set is an allowlist, not a subtraction.** The job runs
  `git ls-files -z -- <allowlist> | tar --null -T - -cf - | tar -xf - -C _site`.
  An allowlist fails *closed*: a new top-level directory of model checkpoints is
  excluded by default instead of silently shipping 300 MB. It also means
  untracked files and `_vendor/` can never ride along. `tests/`, `scripts/`,
  `.github/`, `README.md`, `CLAUDE.md` and `node_modules/` are excluded because
  they are simply not on the list.
- **Failing closed has a cost, and the job pays it explicitly.** A file the site
  *needs* is also excluded by default, and nothing about the upload would look
  wrong — which is exactly how `ml/index.html` was missed. So after assembling
  `_site` the job asserts a manifest of entry points (`index.html`,
  `status.html`, `docs.html`, `404.html`, `manifest.json`, `src/app.js`,
  `src/style.css`, `lib/marked.umd.js`, `icon-192.png`, `data/catalog.json`,
  `docs/PIXEL_STATE.md`, `ml/index.html`) and fails the run if any is absent.
- **Secrets are read from `env:`, never argv.** `wrangler` reads
  `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` from the environment (those
  names are Cloudflare's own, from its Direct-Upload CI guide); only the
  non-secret project name, branch and commit sha appear on the command line. The
  presence checks that keep the job inert are `secrets.X != ''` reduced to
  booleans in the job's `env:` (where the `secrets` context is available) and
  read as `env.` in step conditions (where it is available and `secrets` is not).
  Only the booleans are stored; no token value reaches a log.
- **Three "not configured" states, three answers.** Neither secret set, on a
  push: green, with a notice — a feature nobody enabled must not paint a red X.
  Exactly one secret set: **red**, because that is a typo, not a standby.
  `workflow_dispatch` while unconfigured: **red**, because someone pressed *Run
  workflow* and a green tick with no deploy is the worst possible answer.
- **The job checks Cloudflare's own limits before uploading** — over 20,000
  files or any file over 25 MiB fails the run with an explicit error, rather
  than failing halfway through a 400 MiB upload. `wrangler@4` is a MAJOR pin,
  not a freeze: 4 is the current line (4.125.0, published 2026-08-20).
- **`concurrency: cancel-in-progress` is safe here.** A Pages deployment becomes
  live only once the upload completes, so an interrupted run leaves the previous
  deployment serving rather than a half-uploaded site.
