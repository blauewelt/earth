# Hosting: GitHub Pages today, Cloudflare Pages and blauewelt.org next

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

**Where this is going.** The escape hatch has a destination now:
**`blauewelt.org`**, registered at Infomaniak, with the app at the **apex** and
`www.blauewelt.org` and `blauewelt.ch` both 301-ing to it. That is a nameserver
move, not a DNS record — Infomaniak cannot put a CNAME-like record at a zone
apex, and Cloudflare Redirect Rules only run on traffic its proxy sees. **§6 is
the runbook**, in the order the clicks happen, with the DNSSEC ordering that
decides whether the domain stays reachable. Neither domain carries mail
(Chris, 2026-08-21), which removes the half of the move that used to break
silently — see §6's risk framing. GitHub Pages keeps running throughout,
indefinitely, so every link ever shared keeps resolving and no phase needs a
site rollback.

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
8. **Check it** — §6.0b has the byte-for-byte verification, which is the
   prerequisite for every domain step in §6 and the thing that decides whether
   any link ever moves.

Until step 6 is done the workflow still runs on qualifying pushes and **exits
green**, printing a `::notice::` that says it is on standby. It never shows a
red X for a feature nobody enabled. It is loud in the two cases that are
mistakes rather than choices: **exactly one** of the two secrets set (a typo,
not a standby) fails the run, and **Run workflow** pressed while unconfigured
fails the run, because someone asking for a deploy by hand and getting a green
tick and no deploy is the worst possible answer.

---

## 6 · The cutover: `blauewelt.github.io/earth` → `blauewelt.org`

**The site moves to a domain Chris owns, in one step, and never moves again.**
Everything below is a runbook to be followed with the Infomaniak Manager open in
one tab and the Cloudflare dashboard in another. It is written in the order the
clicks happen, and every phase names its own rollback.

### How risky this actually is

An earlier draft of this section braced the reader for a dangerous migration.
Three answers from Chris on 2026-08-21 removed most of that danger, and the
document should say so plainly rather than leave the warning standing over
something that is no longer there:

- *"I get no email"* — **neither `blauewelt.org` nor `blauewelt.ch` receives
  mail.**
- *"no the site is just registered"* — **`blauewelt.ch` hosts nothing**: no
  site, no mail, no redirect.
- The DNS-zone screenshots are **still outstanding**, and are the one thing
  blocking a start (Phase A).

So the mail migration that used to dominate this runbook — MX, SPF, DMARC, and
above all the `_domainkey` **NS delegation** that has to become a literal TXT
record or stop signing without a single bounce — **is not part of this move.**
It is preserved, in full, as a clearly-marked subsection of Phase C, because the
answer could change and the finding was expensive. It is no longer the main
path. What is left is a nearly empty zone, a static site already deployed and
byte-verified at a second origin, and a GitHub Pages copy that keeps serving
every link ever shared no matter what happens here.

**One real risk remains, at full strength: DNSSEC.** Move the nameservers while
the DS record still stands at the `.org` registry and the domain goes
**completely dark** — SERVFAIL from every validating resolver, which is not a
wrong answer but *no* answer, and it looks fine from any network that does not
validate. Phase B exists for that one step; its ordering is not negotiable and
its 1.5 × TTL wait is not padding.

Two smaller risks, both real and both bounded. A nameserver move that has to be
reversed costs **24–48 h** at registry TTLs — which is why Phase C populates the
zone *before* the switch rather than after. And a certificate that cannot issue
(a CAA record forbidding Let's Encrypt, an Advanced Certificate ordered by
mistake, a redirect rule swallowing `/.well-known/`) shows as a browser warning
on a live domain; Phases D and E check all three.

Everything else on the old risk list was email, and the email is gone.

### 6.0 · The shape of the decision, and why it is this shape

| | Decision |
|---|---|
| Canonical name | **`blauewelt.org`**, and the app lives at the **apex** — `https://blauewelt.org/`, no `www`, no path prefix |
| `www.blauewelt.org` | **301** to the apex, path and query preserved |
| `blauewelt.ch` | **301** to `blauewelt.org`, path and query preserved. Never an origin |
| Nameservers | **both zones move to Cloudflare** |
| `earth.pages.dev` | stays reachable as the verification surface. **Never advertised** |
| `blauewelt.github.io/earth/` | **stays live indefinitely.** Free, unchanged, and the rollback |
| Order | **`.org` first and alone.** `.ch` is untouched until `.org` is verified healthy |

Five of those follow from constraints rather than taste, and each is worth
knowing before someone "simplifies" it back:

- **The apex forces the nameserver move.** A Pages custom domain at a zone apex
  needs a record that behaves like a CNAME at the root, and Infomaniak cannot do
  it. Their own FAQ:
  > *"Créer un ANAME revient à créer un CNAME directement à la racine du domaine
  > et cela n'est pas possible."*
  > — [infomaniak.com/en/support/faq/2493](https://www.infomaniak.com/en/support/faq/2493)

  Cloudflare states the other half:
  > *"If you are deploying to an apex domain… you will need to add your site as a
  > Cloudflare zone and configure your nameservers."*

  A **subdomain** would not need this — `earth.blauewelt.org CNAME earth.pages.dev`
  works on Infomaniak DNS, provided the domain is added in the Pages dashboard
  **first** (a CNAME pointed at `earth.pages.dev` before Pages knows the name
  answers **522**). That option was considered and rejected: the apex is the name
  worth having, and a subdomain would spend the one permitted link-breakage on a
  URL nobody wants to type.
- **The `.ch` redirect forces its nameservers too.** Cloudflare Redirect Rules
  only run on traffic the Cloudflare proxy actually sees, so `blauewelt.ch` has
  to be a proxied zone on Cloudflare. It becomes a second zone **on the same
  account** — which is required anyway, because a Pages apex custom domain must
  be a zone on the account that owns the project.
- **One breakage, not two.** Do **not** cut over to `earth.pages.dev` and then
  again to the domain. Go straight from `blauewelt.github.io/earth/` to
  `blauewelt.org/`. Links shared before the move break exactly once, and
  everything issued afterwards is permanent.
- **GitHub Pages keeps running, in parallel, indefinitely.** `pages.yml` is not
  touched by any step in this document. It costs nothing, every old link keeps
  resolving, and it is the reason no phase below needs a site rollback at all —
  the only rollback in the document is a *DNS* rollback.
- **`.org` moves alone.** While `.ch` is still on Infomaniak DNS it is both the
  control and the fallback: "is this Infomaniak or is this us?" is answerable in
  one `dig`.

### 6.0b · Phase 0 — the prerequisite, and it is not optional

Everything in §5 must be done and **verified byte-for-byte** before a single DNS
record changes. Attaching a domain to a broken deployment converts a private
problem into a public one.

1. `pages.yml` and `deploy-cloudflare.yml` both fire on a push to `main`. Both
   are current within a minute of every push. Nobody's links have changed yet.
   **This is a supported configuration, not a staging phase to get through** —
   stay here as long as you like.
2. **Every file, by hash.** For each of the 184 paths in the upload set, fetch
   `https://earth.pages.dev/<path>` and compare the sha256 with the local file.
   Zero mismatches, zero non-200s. Do not compare by looking at it: the globe
   looks identical when a data file is missing, because almost every fetch
   failure degrades to `null` by design.
3. **The four entry points return 200 and the right content type:** `/`,
   `/docs.html`, `/status.html`, `/ml` (which must land on the status page).
4. **A path that does not exist returns 404, not 200.** Without a top-level
   `404.html`, Cloudflare treats a site as a single-page app and answers *every*
   unknown path with the root document at HTTP 200 — a dropped data file would
   arrive as HTML and a missing page would look like the globe. `404.html` is in
   the repo and in the upload set for exactly this reason; confirm it works
   before trusting anything else on this list.
5. **The `.html` → extensionless redirect is understood.** Cloudflare Pages
   *"will redirect HTML pages to their extension-less counterparts"*, so
   `/docs.html?f=x#y` answers with a redirect to `/docs?f=x` (query preserved;
   the fragment is client-side and survives). Every shared link still resolves;
   it costs one hop. Check one `docs.html?f=…#…` and one `status.html#run-N` by
   hand, on a phone.
6. **Run the suite against it.**
   `PLAYWRIGHT_BASE_URL=https://earth.pages.dev npx playwright test tests/app.spec.js tests/docs.spec.js`
7. **Watch it for a week.** Confirm the Cloudflare copy tracks `main` and that no
   ML-only day triggers a deploy.

**Rollback at Phase 0: none needed.** Nothing has been pointed anywhere.

---

### Phase A — capture both zones. There is less there than there looks

**Do this for `blauewelt.org` and `blauewelt.ch` on the same day, before
touching either.** This is still the only phase whose output cannot be recreated
afterwards: once the nameservers move, the Infomaniak zone is still there, but
the moment anyone edits the Cloudflare copy you have lost the reference you
would have compared it against.

What changed is the *size* of the capture, not its necessity. Neither domain
carries mail, so the mail records — MX, SPF, DKIM, DMARC, `autoconfig`,
`autodiscover`, the SRV set — are not expected to exist. **A mail-less zone is
not a recordless zone**, and what remains is exactly the kind of thing that is
missed because nobody thinks to look for it: an existing **CAA** record set that
does not permit Let's Encrypt (which makes the Pages certificate impossible to
issue), a third-party **verification TXT** that un-verifies a property weeks
later, whatever Infomaniak provisions **by default** on a domain nobody has
configured, and the **DS record's TTL**, from which Phase B's wait is derived.

> **BLOCKING — the DNS-zone screenshots are still outstanding.** Chris will send
> them in a later session. **This is the one thing standing between here and a
> start.** Everything below can be read from outside with `dig` *except* what
> Infomaniak shows only in the Manager: the redirections list, and any record
> whose name is not guessable from outside. Do not begin Phase B on the `dig`
> output alone.

**Cloudflare's automatic record scan is not a capture and must not be trusted as
one.** There is no zone transfer on the public internet (no AXFR), so Cloudflare
cannot enumerate a zone — it guesses from a list of common names. The records it
misses are precisely the ones that break silently: third-party verification
TXTs, and anything whose name is not on somebody's list of common names. Capture
by hand, then use the scan only as a second opinion.

**From the Manager:** Domains → the domain → **DNS zone**. Export the zone if
Infomaniak offers a zone-file download; otherwise screenshot **every record, of
every type, with its TTL column visible**. Then take the same capture from
outside, because the dashboard and the wire occasionally disagree:

```bash
D=blauewelt.org
for t in SOA NS A AAAA MX TXT CAA; do dig +noall +answer "$D" $t; done
dig +noall +answer DS $D                     # the PARENT's record. Read its TTL
```

`+noall +answer` rather than `+short` throughout, because **the TTL column is
half the point** and `+short` throws it away.

`MX` and `TXT` are in that loop as a *check on the answer*, not as a capture.
The expectation is that `MX` returns nothing and that no `v=spf1` TXT exists. If
either turns out otherwise — or if
`dig +noall +answer _domainkey.$D NS` returns a delegation — then this domain
has mail history after all, and **Phase C's boxed subsection applies in full
before anything else happens.**

What to look for, and why each one matters:

| Record | Why it matters |
|---|---|
| **CAA** | If any CAA record exists it **must** permit `letsencrypt.org`, `pki.goog` and `ssl.com`, or Cloudflare's Universal SSL cannot issue and the domain serves a certificate error. **No CAA record at all is fine** — it means anyone may issue. Do not invent one |
| **Verification TXTs** | Google/Microsoft/anything ending `-site-verification`. Dropping one un-verifies a property weeks later, silently |
| **The DS record, and its TTL** | The wait in Phase B is derived from this number. Write it down. This is the single most load-bearing value in the whole capture |
| **Whatever Infomaniak provisions by default** | A registered-but-unused domain does not arrive empty: parking A/AAAA records, a webmail vanity CNAME, an SOA with a long TTL. None of it has to be carried over, but it has to be *seen* before it is dropped |
| **A / AAAA / CNAME for anything else** | An old subdomain, something pointed at a service nobody remembers |
| **MX / apex TXT** | Expected empty. If not, see above — the mail path is back on |
| **Web redirections** | Infomaniak's redirections are a **hosting** feature, not DNS. They do **not** move with the zone and will simply stop existing. List them |

**Rollback at Phase A: nothing has changed, so nothing to undo.** The failure
mode of this phase is a *thin* capture, and you only discover it in Phase E when
something is missing and there is nothing to compare against. A short capture is
correct here; a capture that was never taken because "the zone is empty anyway"
is not. If the screenshots have not arrived, this phase is not done.

---

### Phase B — DNSSEC off, wait, *then* nameservers

**This is the highest-risk step in the whole document.** Infomaniak enables
DNSSEC by default on purchase, and Cloudflare is blunt about the consequence:

> *"Changing nameservers while DNSSEC is active can cause your domain to become
> unreachable."*

The mechanism: the DS record at the `.org` registry commits to a signing key.
Move the nameservers while it stands, and Cloudflare's answers are signed by a
key the registry has never heard of. Every **validating** resolver — 1.1.1.1,
8.8.8.8, most ISPs — then refuses to return anything at all. Not a wrong answer:
**no** answer.

**The order is not negotiable.**

1. **Infomaniak Manager → the domain → DNSSEC → off.** This is what removes the
   DS record from the registry. Do it *at the registrar*, not by deleting keys in
   a zone.
2. **Confirm the DS is gone at the parent**, not just in the dashboard:
   ```bash
   dig +noall +answer DS blauewelt.org        # must return nothing
   dig +dnssec blauewelt.org @1.1.1.1 | grep -c ' ad'   # the AD flag should stop appearing
   ```
3. **Wait 1.5 × the DS record's TTL** — the number written down in Phase A. A
   3600 s TTL means **90 minutes**. Why 1.5 and not 1.0: a resolver may have
   cached the DS one second before you removed it, so a full TTL is the *floor*,
   not the guarantee; the extra half covers that plus clock skew. This wait is
   the cheapest insurance in the runbook. Do not shorten it because the
   dashboard already says DNSSEC is off — the dashboard is not what resolvers
   read.
4. **Add the zone to Cloudflare.** *Add a site → `blauewelt.org` → Free plan.*
   Cloudflare scans and shows you what it found, and it assigns two nameservers.
   Note that **partial (CNAME) zone setup is Business-plan-only**, so "keep
   Infomaniak DNS and just proxy the one hostname" is not available here.
5. **STOP.** Do **Phase C** now — populate the zone and check it against the
   Phase A capture. It is a short list, and it is short work. A Cloudflare zone
   can be edited freely while it is still inactive, and every record you get
   right before the switch is a record that never has a broken minute. **Come
   back here for step 6.**
6. **Now change the nameservers at Infomaniak** to the two Cloudflare assigned.
   Manager → the domain → nameservers → custom.
7. Wait for Cloudflare to mark the zone **Active** (it emails, and the Overview
   page says so). This is usually minutes and can take hours.
8. **Re-enable DNSSEC, in the right direction.** Cloudflare → DNS → Settings →
   DNSSEC → Enable. Cloudflare gives you a DS record; publish it at **Infomaniak**,
   which documents this exact combination at
   [faq/2187](https://www.infomaniak.com/en/support/faq/2187) with the field
   mapping spelled out — Cloudflare's **Digest** → Infomaniak's *Hash*,
   **Digest Type** → *Hash Type*, **Algorithm** → *Algorithm*, **Key Tag** →
   *Key*. Do this only after the zone is Active and Phase E has passed.

**What it looks like if the order is wrong.** The signature is unmistakable once
you know it, and invisible if you do not, because *non-validating* resolvers keep
working — so the domain will look fine from one network and dead from another:

```bash
$ dig blauewelt.org @1.1.1.1
;; ->>HEADER<<- opcode: QUERY, status: SERVFAIL, id: 12345

$ dig +cd blauewelt.org @1.1.1.1        # +cd = checking disabled
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 12346
;; ANSWER SECTION: ... 104.21.x.x
```

**SERVFAIL without `+cd`, NOERROR with it, is a DNSSEC validation failure, not an
outage.** Nothing about the website is wrong; resolvers are refusing to speak
about the domain at all. Every name under it is gone, everywhere that validates
— and with no mail on the domain that is the whole blast radius, which makes it
smaller than it used to be and no less total.

| Rollback at Phase B | |
|---|---|
| **Symptom** | Step 1 done, nothing else: none. Removing a DS never breaks resolution — it only stops validation |
| **Action** | Re-enable DNSSEC at Infomaniak if you change your mind |
| **Time** | Registry publish: minutes to an hour |
| **Symptom** | SERVFAIL as above (nameservers moved while the DS still stood) |
| **Action** | Turn DNSSEC **off** at Infomaniak — remove the DS. Do not try to fix it forward |
| **Time** | Registry publish, then up to the **old DS TTL**. Typically under an hour. This is why that TTL was written down |
| **Symptom** | Nameservers moved, zone is wrong, site down |
| **Action** | **Point the nameservers back at Infomaniak.** The Infomaniak zone is not deleted by delegating away — every record is still there |
| **Time** | Up to the delegation's TTL at the registry, commonly **24–48 h**. This is the slowest rollback in the document, and it is the entire reason Phase C is done before step B6 |

---

### Phase C — populate the zone. It is nearly empty

Do this **while the zone is still inactive** (between B5 and B6). Work from the
**Phase A capture**, not from Cloudflare's scan; use the scan only to notice
something the capture missed, never the other way round.

**This used to be the most dangerous phase in the document, and it is now the
shortest.** It was dangerous because of mail: MX, SPF, DMARC and a DKIM
delegation that breaks without a bounce. Neither domain carries mail, so none of
that is here. Read the boxed subsection below before deciding this paragraph
applies to you, then do these four things:

1. **Delete whatever the scan invented.** Cloudflare guesses from a list of
   common names, and on a zone this empty most of what it offers is a guess.
   **A record that is not in the Phase A capture does not go into the zone.**
   Delete any **MX** the scan added in particular: an MX on a domain that
   receives no mail is not harmless — it advertises a destination that will not
   accept mail, and it is exactly the sort of record the next person copies
   forward on the assumption that it was put there deliberately.
2. **Everything from the capture, verbatim**, TTL Auto: the verification TXTs
   and any remaining A/AAAA/CNAME. TTL Auto is 300 s while the zone is proxied,
   which is what makes Phase E's fixes five-minute fixes. The free plan's
   200-records-per-zone ceiling is not a consideration at this size.
3. **CAA.** If the capture found CAA records, make sure the set permits
   `letsencrypt.org`, `pki.goog` and `ssl.com`, or Universal SSL cannot issue
   and the domain serves a certificate error. If it found none, **add none** —
   an absent CAA record set permits every issuer, which is what you want.
4. **Do not add an A or CNAME for the site itself.** Phase D creates the right
   record automatically when the domain is attached to the Pages project, and a
   hand-made one will fight it.

Then **go back to Phase B step 6** and move the nameservers.

> #### If mail is ever added to these domains, read this first
>
> **Everything in this box is inactive as of 2026-08-21**, because Chris
> answered *"I get no email"* for both domains. It is kept in full — not
> summarised, not deleted — because the answer can change with one mailbox, the
> DKIM finding below is genuinely hard-won, and its failure mode is silence.
> **If Phase A found an MX, a `v=spf1` TXT, or a `_domainkey` NS delegation, or
> if mail is ever added to `blauewelt.org` or `blauewelt.ch`, this box is the
> main path and the four steps above are the footnote.**
>
> Mail goes in **first**, before anything else in the zone, because it is the
> only thing here that breaks *silently*. A broken website is a phone call
> within the hour. A broken DKIM is three weeks of your mail quietly scoring as
> spam.
>
> **What Phase A must additionally capture**
>
> ```bash
> D=blauewelt.org
> dig +noall +answer _dmarc.$D TXT
> dig +noall +answer _domainkey.$D NS          # delegation, or nothing — see below
> dig +noall +answer autoconfig.$D CNAME
> dig +noall +answer autodiscover.$D CNAME
> for s in _autodiscover._tcp _imaps._tcp _submission._tcp _caldavs._tcp; do
>   dig +noall +answer $s.$D SRV
> done
> ```
>
> | Record | Why it matters |
> |---|---|
> | **MX** | Infomaniak Mail expects **exactly one**: `mta-gw.infomaniak.ch`, **priority 5**. They warn that any other MX, or more than one, voids their delivery guarantee. This is the record that stops inbound mail dead |
> | **SPF** (TXT at the apex) | `v=spf1 include:spf.infomaniak.ch -all`. Lose it and everything you send starts landing in spam. **Two** SPF records is a permerror, which is worse than none |
> | **`_domainkey`** | **The trap. Record which of two shapes it has.** See below |
> | **`_dmarc`** (TXT) | Copy the value verbatim. A DMARC policy of `p=reject` over a broken DKIM is how a mail domain deletes its own outbound mail without a single bounce reaching you |
> | **`autoconfig` / `autodiscover` / SRV** | Mail-client auto-setup. Nothing breaks today; it breaks the day someone adds the account to a new phone, months later, and reads as "your mail server is broken" |
>
> **The DKIM trap, stated plainly.** Infomaniak's own words: DKIM is *"enabled
> by default for all Mail Services whose DNS zone is managed with Infomaniak"* —
> and the way it is enabled is usually an **NS delegation**: `_domainkey` is
> delegated to Infomaniak's nameservers, and they publish the selector record
> inside that delegated subtree. That arrangement is a *consequence of Infomaniak
> hosting the zone*, and it ends the instant the zone leaves. So the capture must
> answer one question: **is `_domainkey` an NS delegation, or a literal TXT at
> `<selector>._domainkey`?**
>
> - `dig +noall +answer _domainkey.blauewelt.org NS` returns records →
>   **delegation**. You cannot copy this to Cloudflare; it must be converted.
> - It returns nothing, and `dig +short <selector>._domainkey.blauewelt.org TXT`
>   returns a `v=DKIM1…` string → **a literal TXT**. Copy it verbatim, whole.
>
> Either way, **write down the selector name.** It is needed for the conversion
> and again for the verification, and it is not guessable.
>
> **What Phase C must additionally do**
>
> 1. **MX — exactly one.** Add record → MX. Name `@`, server
>    `mta-gw.infomaniak.ch`, **priority 5**, TTL Auto. Then **delete every other
>    MX Cloudflare's scan invented.**
> 2. **SPF — exactly one.** TXT at `@`: `v=spf1 include:spf.infomaniak.ch -all`.
>    If the capture showed extra `include:`s for other senders, keep them, in the
>    same single record. Two SPF TXTs is a permerror.
> 3. **DMARC.** TXT at `_dmarc`, value copied verbatim from the capture.
> 4. **DKIM — the NS → TXT conversion.** If the capture found `_domainkey` as an
>    **NS delegation**, do **not** recreate those NS records in Cloudflare.
>    Infomaniak's own Cloudflare guide
>    ([faq/1619](https://www.infomaniak.com/en/support/faq/1619)) says to remove
>    them and publish a literal TXT instead.
>    - Get the value: **Manager → Mail Service → the domain → the DKIM /
>      signature panel.** It shows a Name (the selector) and a long
>      `v=DKIM1; k=rsa; p=…` value. Copy the **whole** value — the `p=` blob is
>      long and a copy that drops its middle produces a record that exists, looks
>      right, and fails every verification.
>    - Publish it in Cloudflare. Cloudflare's **DKIM record helper** (the
>      email-records shortcut in the DNS tab) takes the **selector alone** in the
>      Name field and appends `._domainkey` plus the zone for you; the plain
>      **TXT** form does not, and there the Name must be written out as
>      `<selector>._domainkey`. **Both are correct and they produce the same
>      name** — which is why the arbiter is not the form you used but this:
>      ```bash
>      dig +short <selector>._domainkey.blauewelt.org TXT
>      ```
>      It must return the `v=DKIM1…` string, **once**. A result at
>      `<selector>._domainkey._domainkey.blauewelt.org` is the classic error, and
>      an empty result with two records elsewhere is the second classic error.
>    - **Proxy status: DNS only.**
> 5. **Grey cloud on everything mail touches.** Cloudflare's proxy handles HTTP
>    and HTTPS and nothing else. An orange-clouded hostname resolves to
>    Cloudflare's anycast addresses, and SMTP or IMAP to those goes nowhere. TXT
>    and MX have no proxy toggle, but `autoconfig`, `autodiscover`, `webmail` and
>    any mail-server A/CNAME do — and they **default to proxied**. Set every one
>    of them to **DNS only (grey cloud)**.
> 6. **The SRV records, verbatim**, TTL Auto.
>
> **What Phase E must additionally verify — and this becomes the gate**
>
> ```bash
> D=blauewelt.org
> dig +short MX $D                           # exactly: 5 mta-gw.infomaniak.ch.
> dig +short TXT $D                          # ONE v=spf1 …
> dig +short TXT _dmarc.$D
> dig +short TXT <selector>._domainkey.$D    # the v=DKIM1 string, exactly once
> ```
>
> 1. **Inbound.** From an account at a completely unrelated provider — a phone,
>    not this machine, not an alias on the same domain — send a message to a real
>    mailbox at `blauewelt.org`. It must arrive within a minute. Open the received
>    message's full headers and confirm `Received: … by mta-gw.infomaniak.ch`.
> 2. **Outbound.** Reply from that Infomaniak mailbox to the external address.
>    In the copy that arrives, open *Show original* / full headers and read the
>    `Authentication-Results:` line at the **receiving** end:
>    - `spf=pass` — the SPF TXT survived.
>    - `dkim=pass header.d=blauewelt.org` — the NS→TXT conversion worked.
>    - `dmarc=pass` — both of the above agree with the `_dmarc` policy.
> 3. **Read a failure correctly:**
>    - `dkim=none` or `dkim=fail` → the conversion. In order of likelihood: the
>      record landed at `<selector>._domainkey._domainkey`, the `p=` value was
>      truncated in the copy, or there are two TXT records at the name.
>    - `spf=softfail` / `spf=fail` → the SPF TXT is missing, duplicated, or lost
>      its `include:spf.infomaniak.ch`.
>    - `dmarc=fail` with the other two passing → the `_dmarc` value was not
>      copied verbatim.
> 4. **Infomaniak's own checker**
>    ([faq/2692](https://www.infomaniak.com/en/support/faq/2692)) — run it against
>    the domain from the Manager. It checks MX, SPF, DKIM and DMARC *as
>    Infomaniak expects them*, which makes it the authority on the only question
>    that matters: will Infomaniak still deliver for this domain now that it does
>    not own the zone.
> 5. **A second opinion from outside.** Send one message from the Infomaniak
>    mailbox to a mail-tester-style external checker and read the score. It flags
>    things Infomaniak has no reason to flag — a missing PTR, a DMARC policy now
>    enforcing against a DKIM that stopped signing.
> 6. **If `autoconfig`/`autodiscover`/SRV records existed**, add the account from
>    scratch in a mail client. That path is exercised by nobody until the day
>    someone gets a new phone, and it fails as "your mail is broken".
> 7. **Give it a few days of real mail** before calling it done. DKIM failures
>    present as spam foldering, not as bounces, and low-volume domains take days
>    to show it.
>
> **Rollback, in this world:** fix the record in Cloudflare, never the
> nameservers. Check the **MX** first — a wrong MX is a two-minute fix and a
> nameserver rollback is a 24–48 hour one. Only if the zone is comprehensively
> wrong is Phase B's nameserver rollback the right tool.

| Rollback at Phase C | |
|---|---|
| **Symptom** | A record is missing or wrong (found in Phase E, or by diffing against the capture) |
| **Action** | Fix it in Cloudflare, against the Phase A capture |
| **Time** | Cloudflare is authoritative and TTL Auto is 300 s → **~5 minutes**, worldwide |
| **Symptom** | The zone turned out to carry mail after all |
| **Action** | Stop and work the boxed subsection above, in its own order — mail first, then everything else |
| **Time** | An hour of careful copying, and a few days of watching |

---

### Phase D — attach the apex to the Pages project

Do not start until the Cloudflare Overview page says the zone is **Active**.
Universal SSL is issued **only after that**, and everything in this phase depends
on the certificate.

1. **Set the SSL mode first.** SSL/TLS → Overview → **Full (strict)**.
   **Not Flexible.** Flexible sends plain HTTP to the origin; Pages redirects
   HTTP → HTTPS; the browser follows that back into Cloudflare, which strips it
   to HTTP again, and the visitor gets **`ERR_TOO_MANY_REDIRECTS`** on a site
   that is otherwise perfectly deployed. Pages presents a valid, publicly
   trusted certificate, so **Full (strict)** is not merely safe here, it is
   correct.
2. **Attach the domain in the Pages project, not in DNS.** Workers & Pages →
   `earth` → **Custom domains** → *Set up a custom domain* → `blauewelt.org`.
   Because the zone is on the same Cloudflare account, Cloudflare creates the
   apex record itself. **The registration in the Pages dashboard must come
   first**: a record pointed at `earth.pages.dev` before Pages knows the name
   answers **522**, which reads exactly like an origin outage and is not one.
3. **Wait for the custom domain to read *Active*** and for the certificate to
   issue. Universal SSL covers the **apex and first-level subdomains only** —
   here that is exactly `blauewelt.org` and `www.blauewelt.org`, which is all
   this design needs. **Do not order an Advanced Certificate: they are
   incompatible with Pages.**
4. **Never flip a custom domain's DNS away and back to "retry".** Cloudflare
   marks the domain inactive when its record stops pointing at Pages, and
   visitors get errors until it reactivates. If something looks stuck, wait and
   read the status text; do not poke the record.
5. **`www`, which exists only to redirect.** It needs a record so that the
   hostname resolves, and it needs to be **proxied** so a Redirect Rule can run
   on it — but it must never reach an origin. Add:
   - **AAAA**, name `www`, address `100::` (the IPv6 discard prefix), **Proxied
     (orange cloud)**.

   The proxy answers, the rule below fires, and nothing is ever forwarded.
6. **The redirect rule.** Rules → **Redirect Rules** → *Create rule* → Single
   Redirect. The free plan allows **10 per zone**; this is one.
   - **Wildcard form** — *When incoming requests match*: wildcard pattern
     `https://www.blauewelt.org/*`; *Then*: target
     `https://blauewelt.org/${1}`, status **301**, and switch
     **Preserve query string** **on**.
   - **Expression form**, identical in effect and easier to read back later:
     when `http.host eq "www.blauewelt.org"`, then *Dynamic* →
     `concat("https://blauewelt.org", http.request.uri.path)`, **301**,
     **Preserve query string on**.
   - **Exclude the ACME path.** Add
     `and not starts_with(http.request.uri.path, "/.well-known/")` to the match.
     A redirect rule that swallows `/.well-known/acme-challenge/` is the single
     most common way a certificate silently stops renewing thirty days later.
   - **Preserve query string is an explicit toggle and it is off by default.**
     Every `docs.html?f=…` link in this project's history depends on it.
     Fragments (`#anchor`) need nothing: browsers never send them, so they
     survive any redirect by construction.
7. **Do not add `www` — or anything else — as a second Pages custom domain.** A
   second origin name means two live copies of the app and every future link
   split between them.

| Rollback at Phase D | |
|---|---|
| **Symptom** | `ERR_TOO_MANY_REDIRECTS` |
| **Action** | SSL mode is Flexible. Set **Full (strict)** |
| **Time** | ~1 minute |
| **Symptom** | **522** on the custom domain |
| **Action** | The name was not registered in the Pages project first, or it is still initialising. Add it in the Pages project and wait. Do **not** edit the DNS record |
| **Time** | Minutes |
| **Symptom** | Certificate warning in the browser |
| **Action** | Zone not Active yet, or a CAA record forbids the issuer (Phase A). Universal SSL issues only after Active |
| **Time** | Minutes to a few hours |
| **Symptom** | Anything worse |
| **Action** | Remove the custom domain from the Pages project. The site is still live and current at `blauewelt.github.io/earth/` and at `earth.pages.dev` — **there is no site rollback to perform**, only a domain to detach |
| **Time** | Immediate |

---

### Phase E — verify, and do not hand-wave it

This is the phase that decides whether `.ch` is touched at all. Run every
command. Compare every answer against the **Phase A capture**, line by line:
anything in the capture that does not appear here has been dropped, and nothing
will tell you so.

The email round-trip that used to be the gate here is gone with the mail — if
that ever changes, the gate comes back, and it is written out in Phase C's boxed
subsection. **What did not shrink** is everything below: DNSSEC, the
certificate, the ACME path, CAA, and the byte-correctness of the deployed site.

**DNS — is the zone what it used to be?**

```bash
D=blauewelt.org
dig +short NS $D                      # the two Cloudflare nameservers, nothing else
dig +short A $D                       # Cloudflare anycast addresses
dig +short AAAA $D
dig +short TXT $D                     # the verification TXTs from the capture — and nothing invented
dig +short MX $D                      # expected: nothing. An MX here was invented by the scan
dig +noall +answer CAA $D             # empty is fine; if not, letsencrypt.org must be allowed
dig +noall +answer DS $D              # empty until Phase B8 re-enables DNSSEC
```

**DNSSEC, both directions — the one step that can take the domain dark.**

```bash
dig blauewelt.org @1.1.1.1 | head -3          # NOERROR, not SERVFAIL
dig +cd blauewelt.org @1.1.1.1 | head -3      # if this differs from the line above, see Phase B
dig blauewelt.org @8.8.8.8 | head -3          # a second validating resolver, on a different network
```

A domain that answers here and SERVFAILs there is not intermittent — it is a
validation failure being masked by whichever resolver happens not to validate.
After **Phase B8** re-publishes the DS at Infomaniak, run these three again:
re-enabling DNSSEC is the same hazard as removing it, pointed the other way.

**HTTP and the certificate.**

```bash
curl -sSI https://blauewelt.org/ | head -20              # 200, and a cf-ray header
curl -sSI http://blauewelt.org/ | head -5                # 301 → https, ONE hop
curl -sS -o /dev/null -w '%{num_redirects}\n' -L https://blauewelt.org/
                                                          # 0 or 1. Never climbing
curl -sSI 'https://www.blauewelt.org/docs.html?f=ml/EXPERIMENTS.md' | head -8
     # 301 → https://blauewelt.org/docs.html?f=ml/EXPERIMENTS.md   ← query preserved
curl -sSI https://blauewelt.org/this-path-does-not-exist | head -3   # 404, NOT 200
curl -sSI https://blauewelt.org/ml | head -8             # the /ml shortcut still lands
openssl s_client -connect blauewelt.org:443 -servername blauewelt.org </dev/null 2>/dev/null \
  | openssl x509 -noout -subject -issuer -dates
```

**The ACME probe — the check for a failure that is thirty days away.**

```bash
curl -sSI https://blauewelt.org/.well-known/acme-challenge/probe | head -3
curl -sSI https://www.blauewelt.org/.well-known/acme-challenge/probe | head -3
```

A **404** is the healthy answer. A **301 to somewhere else** means a redirect
rule is eating the path certificate authorities use, and the failure surfaces at
the next renewal, not today. Cloudflare validates Universal SSL over DNS while it
runs the zone, so this path is not load-bearing for the *first* certificate —
which is exactly why it can be broken for a month without anyone noticing. Check
`www` in particular: its whole existence is a redirect rule.

**The site itself, byte for byte — this is the gate now.**

The site is what the move is *for*, so verify it at the new hostname the same way
§6.0b verified it at `earth.pages.dev`, and do not substitute looking at the
globe: almost every fetch failure in this app degrades to `null` by design, so a
missing data file renders as a perfectly convincing planet.

1. **Every file, by hash.** For each of the 184 paths in the upload set, fetch
   `https://blauewelt.org/<path>` and compare the sha256 with the local file.
   Zero mismatches, zero non-200s.
2. **The four entry points**, with the right content type: `/`, `/docs.html`,
   `/status.html`, `/ml` (which must land on the status page).
3. **The suite, against the live domain:**
   `PLAYWRIGHT_BASE_URL=https://blauewelt.org npx playwright test tests/app.spec.js tests/docs.spec.js`
4. **One `docs.html?f=…#…` and one `status.html#run-N` by hand, on a phone** —
   the `.html` → extensionless redirect (§6.0b step 5) is now happening behind a
   custom domain, and the query string has to survive both it and the `www`
   rule.

**`blauewelt.ch` is not touched until every check above passes.** Not the
nameservers, not the DNSSEC toggle, nothing. While `.ch` is still wholly on
Infomaniak it is a working control: any question of the form "is this Infomaniak
misbehaving or is this our change?" is answered by running the same query against
`.ch`.

| Rollback at Phase E | |
|---|---|
| **Symptom** | A DNS answer disagrees with the capture |
| **Action** | Fix the record in Cloudflare |
| **Time** | ~5 minutes (TTL Auto = 300 s) |
| **Symptom** | SERVFAIL on one resolver and not another |
| **Action** | DNSSEC. Phase B's rollback table, and do not try to fix it forward |
| **Time** | Registry publish, then up to the old DS TTL |
| **Symptom** | A file's hash disagrees, or an entry point 404s |
| **Action** | Not a DNS problem — the deployment is wrong. Detach the custom domain, fix the deploy, re-attach. `blauewelt.github.io/earth/` is still live and current throughout |
| **Time** | Immediate to detach |

---

### Phase F — `blauewelt.ch`, once and only once `.org` is healthy

**`blauewelt.ch` hosts nothing.** Chris, 2026-08-21: *"no the site is just
registered"* — no site, no mail, no redirect. So there is nothing to preserve
here and nothing that can break by being moved. This phase is a nameserver move
with the same DNSSEC care as Phase B, and then three clicks.

1. **A short Phase A anyway**, for the two things that are not about mail: the
   **DS record's TTL**, which sets the wait in step 2 and is a different number
   from `.org`'s; and any **CAA** record, which would stop a certificate
   issuing for `blauewelt.ch` exactly as it would for `.org`. Also glance for a
   verification TXT — a registered-and-unused domain is where one gets
   forgotten. The rest of the capture is `dig +noall +answer` on the same seven
   types and takes a minute.
   If it turns out `.ch` carries mail after all, **Phase C's boxed subsection
   applies in full** before anything else.
2. **Phase B again, unchanged and at full strength:** DNSSEC off at Infomaniak →
   confirm the DS is gone at the parent → wait 1.5 × the DS TTL → add the zone
   to Cloudflare → populate it (step 3) → move the nameservers → wait for
   **Active** → re-enable DNSSEC and publish the new DS at Infomaniak. The
   domain being empty makes the *zone* trivial; it does not make the DNSSEC
   ordering any less capable of taking the name completely dark.
3. **Two placeholder records**, because both names must resolve and both must be
   proxied for a rule to run on them, and neither may reach an origin:
   - **AAAA**, name `@`, `100::` (the IPv6 discard prefix), **Proxied**
   - **AAAA**, name `www`, `100::`, **Proxied**
4. **One redirect rule for both hostnames.** Rules → Redirect Rules → Create:
   - *When*: `http.host in {"blauewelt.ch" "www.blauewelt.ch"} and not starts_with(http.request.uri.path, "/.well-known/")`
   - *Then*: Dynamic → `concat("https://blauewelt.org", http.request.uri.path)`,
     **301**, **Preserve query string on**.

   Both halves of that are deliberate. The `/.well-known/` exclusion keeps the
   ACME path reachable, and **Preserve query string is an explicit toggle that
   is off by default** — every `docs.html?f=…` link depends on it.
5. **`blauewelt.ch` is never added to the Pages project.** It is a redirect. Add
   it as a custom domain and you have two origins serving the same app under two
   names, and every link ever shared afterwards is a coin flip.
6. **Verify:**
   ```bash
   curl -sSI 'https://blauewelt.ch/docs.html?f=ml/EXPERIMENTS.md' | head -8
   curl -sSI 'https://www.blauewelt.ch/status.html' | head -8
   curl -sSI https://blauewelt.ch/.well-known/acme-challenge/probe | head -3
   dig blauewelt.ch @1.1.1.1 | head -3      # NOERROR, not SERVFAIL
   dig blauewelt.ch @8.8.8.8 | head -3      # a second validating resolver
   ```
   The first two must be **301** to the same path and query on `blauewelt.org`;
   the third must be a **404**, not a redirect; the last two must agree.

| Rollback at Phase F | |
|---|---|
| **Symptom** | The redirect loops, or does not fire |
| **Action** | Disable the rule — one toggle. `.org` is untouched by construction: the rule is scoped to `.ch` hostnames |
| **Time** | Immediate |
| **Symptom** | SERVFAIL on `blauewelt.ch` |
| **Action** | DNSSEC, exactly as in Phase B: turn it **off** at Infomaniak, do not fix it forward |
| **Time** | Registry publish, then up to the old DS TTL |
| **Symptom** | `.ch` is unreachable and the zone looks comprehensively wrong |
| **Action** | Nameservers back to Infomaniak; the zone there is intact. Note that `.ch` served nothing before this phase, so the cost of leaving it broken while you think is close to zero |
| **Time** | **24–48 h** |
| **Symptom** | Anything at all wrong with `.org` |
| **Action** | Not caused by this phase. Stop, and go back to Phase E |
| **Time** | — |

---

### What actually breaks, and what it costs

The site is `https://blauewelt.github.io/earth/`; it becomes
`https://blauewelt.org/`. **Every link ever shared points at the old address**,
and this project shares a *lot* of links, by standing rule (CLAUDE.md §0b):

- every `docs.html?f=ml/EXPERIMENTS.md#e-026b` posted in chat
- every `status.html#run-427` in a session report
- every `ml/figs/*.html` figure link
- the `/earth/ml` shortcut, if it is on a phone's home screen

Nothing rewrites them, and Chris has accepted this explicitly — *"the deep links
are normally not saved, and our status page can be changed by us"*. Two things
reduce it to almost nothing:

- **GitHub Pages keeps running.** Every old link keeps resolving, forever, at no
  cost. That is the whole reason nothing in this document needs a site rollback.
- **It happens once.** `blauewelt.org` is a name we control; any future host
  change is a DNS record and breaks nothing. This is the last time these links
  move.

### Retiring GitHub Pages — optional, and not recommended

Only if the 1 GB published-site limit is actually being hit. Deleting the Pages
deployment removes the rollback and the permanent home of every link shared
before the cutover. There is no cost reason to do it.

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

Swept 2026-08-21 across `index.html`, `status.html`, `docs.html`, `404.html`,
`ml/index.html`, `manifest.json`, `src/`, `lib/`, `data/`, `scripts/` and
`.github/workflows/`. Re-swept and **closed** the same day, when the destination
stopped being hypothetical.

**Nothing in the app breaks when the origin changes, and nothing points at the
old one any more.**

### The parts that were already right (verified, not assumed)

- **Every asset, data file and internal page is fetched by a *relative* path.**
  No `href="/…"` or `src="/…"` exists anywhere in the browser-facing set — which
  is what makes the `/earth/` path prefix a non-issue: the same files work at
  `/earth/` on Pages, at `/` on Cloudflare, and at `/` on the test server.
- **`manifest.json` — confirmed by reading it, not by memory:**
  `"start_url": "./index.html"`, `"scope": "./"`, and all three icons are
  relative (`icon-192.png?v=…`). An installed PWA therefore follows whichever
  origin it was installed from, and an install made from `blauewelt.org` scopes
  to `blauewelt.org`.
- **The build check** (`checkForNewBuild()`, `src/app.js:8288`) fetches
  `index.html?fresh=…` relatively. Origin-agnostic.
- **`404.html`** links `./index.html`, `./docs.html`, `./status.html` — relative,
  and a relative link on a 404 resolves against the URL that failed, which is
  exact for the mistyped-path case it exists for.
- **`ml/index.html`** redirects to `../status.html`. Relative.
- **`docs.html`** reads markdown from `raw.githubusercontent.com` on any
  non-`localhost` host (`docs.html:369–374`) and **`status.html`** reads only
  `api.github.com` and `raw.githubusercontent.com`. All cross-origin already,
  all unaffected by the move — and note that this is why the docs reader stays
  correct on a brand-new origin from its first minute.
- **No `<link rel="canonical">`, no `og:url`, no Content-Security-Policy** in any
  page. Nothing to re-point, and nothing that would refuse to load on a new
  hostname.
- **No absolute self-origin URL in `data/`.** Checked across every file.

### The two hits, and how they were closed

`status.html` was the only browser-facing file that named the GitHub Pages
origin, in two places, and both were on the reader's path rather than in a
comment: the experiment badge on every run card, and the "Live app" link. Left
alone they would have *degraded* rather than broken — a reader on
`blauewelt.org` would have been quietly handed back to the GitHub Pages copy,
which is only as fresh as its last deploy. Both are now **origin-following with
a saved-copy fallback**:

```js
var SAVED_COPY = "https://blauewelt.github.io/earth/";
var ORIGINLESS = (location.protocol === "file:" || !location.host);
var DOCS_BASE = ORIGINLESS ? SAVED_COPY + "docs.html"
                           : new URL("docs.html", location.href).href;
var APP_BASE  = ORIGINLESS ? SAVED_COPY
                           : new URL("./", location.href).href;
```

The `file://` branch is the reason this is not simply a relative URL: a saved
copy has no origin to follow, and a relative link there would resolve to a path
on the reader's own disk. GitHub Pages stays live indefinitely (§6.0), so the
fallback target is a real page and not a hopeful one.

**On GitHub Pages both expressions evaluate to exactly the strings that were
hardcoded before**, so the change is behaviour-preserving in production; only
the suite's expectation moved.

| File · line | What it is now |
|---|---|
| `status.html:223–228` | `SAVED_COPY` / `ORIGINLESS` / `DOCS_BASE` / `APP_BASE` — the block above |
| `status.html:163` | `<a id="live-app" href="https://blauewelt.github.io/earth/">` — the literal is the no-JS and saved-copy value; `status.html:245–249` re-points it at `APP_BASE` on load |
| `tests/status.spec.js:487` | the tripwire, **moved rather than deleted**: it now asserts the behaviour in all three contexts — the serving origin, a *different* serving origin (127.0.0.1 standing in for `earth.pages.dev` / `blauewelt.org`), and a real `file://` load. The second of those is the one that fails if anything is ever pinned to a host again |

### Deliberate non-hits, recorded so nobody sweeps them twice

- `index.html:866`, `status.html:153–154` and `src/app.js:439` contain
  `github.com/blauewelt/earth/blob/…` URLs. Those point at the **repository**,
  not the site, and are correct on any host.
- `README.md` ×4 and `CLAUDE.md` ×2 name the live site in prose. They are
  *about* where the site is; they get edited when it moves, and a stale one is
  visible rather than silent.
- `scripts/run_index.mjs:39` defaults `--docs-base` to the Pages URL and
  `scripts/dispatch_run.mjs:206` prints the Pages status URL to a console. Both
  are agent-side tooling, neither is served to a browser, `run_index.mjs`
  already takes an override on the command line, and both keep working because
  GitHub Pages keeps running. Change them when the domain is live and verified —
  not before, or they will print a URL that does not resolve yet.

### `_headers` and `_redirects`: audited, and deliberately absent

Both were considered and neither is added, because the audit produced no
requirement for either:

- **`_headers`** — there is no CSP to extend, no cross-origin font or asset to
  permit, and no header the app depends on. The one *speculative* use (marking
  the `?v=<sha8>`-stamped assets `immutable`, §2) is a caching optimisation with
  no measurement behind it and no bearing on the domain move. Adding
  configuration "while we're in there" is how a file nobody understands ends up
  in front of every request. If the stamped-asset caching is ever wanted, it is
  its own change with its own before/after numbers.
- **`_redirects`** — it cannot do what this cutover needs. Pages' `_redirects`
  is scoped to paths within one deployment: it cannot redirect one *domain* to
  another, and it does not preserve query strings. `www.blauewelt.org` and
  `blauewelt.ch` are therefore handled by **Cloudflare Single Redirect Rules**
  (§6, Phases D and F), which do both and expose "Preserve query string" as an
  explicit toggle. Putting a `_redirects` file in the repo would look like it
  was doing that job and would not be.

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
