// Data-integrity tests — run without a browser.
// These guard the bundled snapshots and catalog against corruption/regressions.
"use strict";
const { test, expect } = require("@playwright/test");
const fs = require("fs");
const path = require("path");

const DATA = path.join(__dirname, "..", "data");
const read = (f) => JSON.parse(fs.readFileSync(path.join(DATA, f), "utf8"));

test.describe("catalog.json", () => {
  const cat = read("catalog.json");

  // Lower bound, not equality: the catalog only grows. The title tracks the
  // current size so it doesn't quietly drift the way the README's did.
  test("the OISST monthly normals are a usable, seasonally-correct baseline", () => {
    // The pixel card subtracts these from OISST monthly means to report an
    // anomaly the GIBS palette cannot express (its end bins are catch-alls at
    // ±3 °C). Two things have to hold or that subtraction is meaningless.
    const idx = read("oisst_clim.json");
    expect(idx.period).toBe("1991-2020");
    expect(idx.monthsAvailable).toHaveLength(12);
    expect(idx.nx * idx.ny).toBe(64800);
    const jan = read("oisst_clim/01.json").values;
    const jul = read("oisst_clim/07.json").values;
    expect(jan).toHaveLength(idx.nx * idx.ny);
    // (1) it is PER MONTH, not an annual mean — an annual baseline would make
    // the "anomaly" mostly the seasonal cycle, which is why a derived
    // SST-vs-normal row was refused before this file existed.
    let diffs = 0, maxSwing = 0;
    for (let i = 0; i < jan.length; i++) {
      if (jan[i] == null || jul[i] == null) continue;
      const d = Math.abs(jan[i] - jul[i]);
      if (d > 0.5) diffs++;
      maxSwing = Math.max(maxSwing, d);
    }
    expect(diffs).toBeGreaterThan(20000);       // most of the ocean has a season
    expect(maxSwing).toBeGreaterThan(10);       // midlatitude shelves swing hard
    // (2) the values are sea temperatures, and land is absent rather than zero
    let lo = Infinity, hi = -Infinity, nulls = 0;
    for (const v of jul) {
      if (v == null) { nulls++; continue; }
      lo = Math.min(lo, v); hi = Math.max(hi, v);
    }
    expect(lo).toBeGreaterThan(-2.5);           // freezing point of seawater
    expect(hi).toBeLessThan(36);
    expect(nulls).toBeGreaterThan(15000);       // land is null, not 0 °C
  });

  test("has 244+ records with required fields", () => {
    expect(cat.record_count).toBeGreaterThanOrEqual(244);
    expect(cat.records.length).toBeGreaterThanOrEqual(244);
    for (const r of cat.records) {
      for (const field of ["id", "name", "domain", "provider", "url", "access", "license"]) {
        expect(r[field], `${r.id || r.name} missing ${field}`).toBeTruthy();
      }
      expect(r.url).toMatch(/^https?:\/\//);
    }
  });

  test("the fine tier (≤30 m) is catalogued, live-flagged, and keyless", () => {
    // §2.6: every live layer has a record with the live note. The ten below
    // are the fine-tier layers; the reference-only rows (Sentinel-1/2 at the
    // source, SWOT, GSW, Copernicus DEM, WorldCover WMTS, AlphaEarth,
    // swisstopo, ECOSTRESS) document what is NOT a layer and why.
    const live = ["hls", "opera-rtc-s1", "nisar-gcov", "opera-dswx-hls", "opera-dswx-s1",
      "aster-gdem", "sedac-hbase", "sedac-gmis", "landsat-weld"];
    // the third backend: live too, but keyless tile hosts other than GIBS
    const liveXyz = ["esa-worldcover-wmts", "jrc-gsw", "eox-s2cloudless", "swisstopo-wmts"];
    const ref = ["sentinel-2-msi", "sentinel-1-sar", "swot-karin", "copernicus-dem",
      "alphaearth", "ecostress"];
    const byId = new Map(cat.records.map((r) => [r.id, r]));
    for (const id of live) {
      const r = byId.get(id);
      expect(r, id).toBeTruthy();
      expect(r.globe, id).toBe(true);
      expect(r.notes, id).toMatch(/Live globe layer in this app\./);
      expect(r.access, id).toMatch(/GIBS/);          // keyless tiles, CLAUDE.md §3
      expect(r.spatial, id).toMatch(/\b(10|15|20|30) m\b/);
    }
    for (const id of liveXyz) {
      const r = byId.get(id);
      expect(r, id).toBeTruthy();
      expect(r.globe, id).toBe(true);
      expect(r.notes, id).toMatch(/Live globe layer in this app\./);
      expect(r.access, id).toMatch(/no key/);
      expect(r.license, id).toBeTruthy();
    }
    for (const id of ref) {
      const r = byId.get(id);
      expect(r, id).toBeTruthy();
      expect(r.notes, id).not.toMatch(/Live globe layer/);
    }
    expect(cat.record_count).toBeGreaterThanOrEqual(274);
    expect(cat.record_count).toBe(cat.records.length);
    const byDomain = {};
    for (const r of cat.records) byDomain[r.domain] = (byDomain[r.domain] || 0) + 1;
    expect(cat.records_by_domain).toEqual(byDomain);
  });

  test("ids are unique", () => {
    const ids = cat.records.map((r) => r.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  test("flags are consistent with the summary counts", () => {
    const globe = cat.records.filter((r) => r.globe).length;
    const amoc = cat.records.filter((r) => r.amoc).length;
    expect(globe).toBeGreaterThan(80);
    expect(amoc).toBeGreaterThan(40);
  });
});

test.describe("rapid_moc.json", () => {
  const r = read("rapid_moc.json");

  test("series are aligned and span 2004 to at least 2024", () => {
    expect(r.t.length).toBeGreaterThan(700);
    for (const k of ["moc", "gulf_stream", "ekman", "upper_mid_ocean"]) {
      expect(r[k], `${k} length`).toHaveLength(r.t.length);
    }
    expect(r.t[0] <= "2004-05-01").toBeTruthy();
    expect(r.t[r.t.length - 1] >= "2024-01-01").toBeTruthy();
  });

  test("MOC values are physically plausible (Sv)", () => {
    const vals = r.moc.filter((v) => v != null);
    expect(vals.length).toBeGreaterThan(600);
    for (const v of vals) {
      expect(v).toBeGreaterThan(-10);
      expect(v).toBeLessThan(45);
    }
    const mean = vals.reduce((s, x) => s + x, 0) / vals.length;
    expect(mean).toBeGreaterThan(10);
    expect(mean).toBeLessThan(25);
  });
});

test.describe("argo.json", () => {
  const a = read("argo.json");

  test("has a plausible active fleet with valid coordinates", () => {
    expect(a.floats.length).toBeGreaterThan(2000);
    expect(a.floats.length).toBeLessThan(10000);
    for (const [lon, lat] of a.floats) {
      expect(lon).toBeGreaterThanOrEqual(-180);
      expect(lon).toBeLessThanOrEqual(180);
      expect(lat).toBeGreaterThanOrEqual(-90);
      expect(lat).toBeLessThanOrEqual(90);
    }
  });
});

test.describe("climatetrace.json", () => {
  const c = read("climatetrace.json");

  test("multiple years, each a valid top-N sorted by emissions", () => {
    expect(c.years.length).toBeGreaterThanOrEqual(4);       // ~2021-2025
    expect(c.years).toEqual([...c.years].sort((a, b) => a - b));
    for (const yr of c.years) {
      const assets = c.assets_by_year[String(yr)];
      expect(assets, `year ${yr} present`).toBeTruthy();
      expect(assets.length).toBeGreaterThan(500);
      // sorted by emissions, valid coords, positive magnitudes
      expect(assets[0][2]).toBeGreaterThan(assets[assets.length - 1][2]);
      for (const [lon, lat, mt] of assets) {
        expect(Math.abs(lon)).toBeLessThanOrEqual(180);
        expect(Math.abs(lat)).toBeLessThanOrEqual(90);
        expect(mt).toBeGreaterThan(0);
      }
    }
    // the years genuinely differ (not the same payload cloned)
    const [a, b] = [c.years[0], c.years[c.years.length - 1]].map(
      (y) => c.assets_by_year[String(y)][0][2]);
    expect(a).not.toBe(b);
  });
});

test.describe("stations.geojson", () => {
  const s = read("stations.geojson");

  test("all stations have coordinates, name, type, url", () => {
    expect(s.features.length).toBeGreaterThanOrEqual(13);
    for (const f of s.features) {
      const [lon, lat] = f.geometry.coordinates;
      expect(Math.abs(lon)).toBeLessThanOrEqual(180);
      expect(Math.abs(lat)).toBeLessThanOrEqual(90);
      expect(f.properties.name).toBeTruthy();
      expect(f.properties.type).toBeTruthy();
      expect(f.properties.url).toMatch(/^https?:\/\//);
    }
  });
});

test.describe("sealevel.json", () => {
  const s = read("sealevel.json");

  test("budget components and altimetry are present and aligned", () => {
    expect(s.years[0]).toBe(1900);
    expect(s.years[s.years.length - 1]).toBeGreaterThanOrEqual(2018);
    for (const k of ["observed", "sum", "steric", "glaciers", "greenland", "antarctica", "tws"]) {
      expect(s.components[k], `${k} length`).toHaveLength(s.years.length);
    }
    expect(s.altimetry.t.length).toBeGreaterThan(500);
    expect(s.altimetry.t.length).toBe(s.altimetry.v.length);
  });

  test("budget approximately closes and shows the expected rise", () => {
    const i = s.years.length - 1;
    const rise = s.components.observed[i] - s.components.observed[0];
    expect(rise).toBeGreaterThan(150);  // ~200 mm over the 20th century
    expect(rise).toBeLessThan(260);
    // summed budget tracks observed within a reasonable residual
    const gap = Math.abs(s.components.observed[i] - s.components.sum[i]);
    expect(gap).toBeLessThan(20);
    // steric is a major positive contributor by the end
    expect(s.components.steric[i]).toBeGreaterThan(0);
  });
});

test.describe("species.json", () => {
  const s = read("species.json");

  test("curated indicator species have keys, notes and record counts", () => {
    expect(s.species.length).toBeGreaterThanOrEqual(8);
    for (const sp of s.species) {
      expect(Number.isInteger(sp.key)).toBe(true);
      expect(sp.common).toBeTruthy();
      expect(sp.note.length).toBeGreaterThan(10);
      expect(sp.records).toBeGreaterThan(0);
    }
    // The payload's note explains how the 3.9 B total decomposes — the three
    // things a reader gets wrong otherwise: what the unplaced remainder is,
    // that the mix reflects observer effort rather than actual abundance, and
    // why humans are almost absent. (The presence-vs-abundance caveat itself
    // is UI copy, asserted in app.spec.js.)
    const note = s.note.toLowerCase();
    expect(note).toContain("kingdom");
    expect(note).toContain("birds");
    expect(note).toContain("privacy");
    // the composition figures the note quotes must match the data
    expect(s.total).toBeGreaterThan(3.5e9);
    expect(s.unplaced).toBeGreaterThan(0);
    expect(s.unplaced).toBeLessThan(s.total);
  });
});

test.describe("glaciers.json", () => {
  const g = read("glaciers.json");

  test("RGI v7 glaciers: aligned arrays, plausible count/area, valid coords", () => {
    expect(g.count).toBeGreaterThan(250000);           // RGI7 G-product ~274k
    for (const k of ["lon", "lat", "area", "dhdt"]) expect(g[k]).toHaveLength(g.count);
    expect(g.total_area_km2).toBeGreaterThan(600000);  // ~706k km² global total
    expect(g.total_area_km2).toBeLessThan(800000);
    // most glaciers carry a 2000-2020 melt rate, and the majority are thinning
    expect(g.dhdt_matched).toBeGreaterThan(200000);
    const rates = g.dhdt.filter((v) => v != null);
    const thinning = rates.filter((v) => v < 0).length / rates.length;
    expect(thinning).toBeGreaterThan(0.6);             // ~78% thinning
    // spot-check coordinate/area validity across the array
    for (let i = 0; i < g.count; i += 5000) {
      expect(Math.abs(g.lon[i])).toBeLessThanOrEqual(180);
      expect(Math.abs(g.lat[i])).toBeLessThanOrEqual(90);
      expect(g.area[i]).toBeGreaterThan(0);
    }
  });
});

test.describe("gistemp.json", () => {
  const g = read("gistemp.json");
  test("global temperature series, land warms faster than land+ocean", () => {
    expect(g.years[0]).toBe(1880);
    expect(g.years[g.years.length - 1]).toBeGreaterThanOrEqual(2024);
    expect(g.land_ocean).toHaveLength(g.years.length);
    expect(g.land_only).toHaveLength(g.years.length);
    // recent warming is well above the 1951-1980 baseline
    const i2024 = g.years.indexOf(2024);
    expect(g.land_ocean[i2024]).toBeGreaterThan(1.0);
    // land anomaly exceeds land+ocean (land warms faster)
    expect(g.land_only[i2024]).toBeGreaterThan(g.land_ocean[i2024]);
  });
});

test.describe("gridded climatology snapshots", () => {
  const specs = {
    "gpcp.json": { units: "mm/yr", ramp: "precip", nx: 144, ny: 72, global: true, vmaxData: 8000 },
    "oisst.json": { units: "°C", ramp: "sst", nx: 360, ny: 180, global: true, vmaxData: 40 },
    "eobs.json": { units: "mm/yr", ramp: "precip", regional: true, vmaxData: 5000 },
    "meteoswiss.json": { units: "mm/yr", ramp: "precip", regional: true, vmaxData: 6000 },
  };
  for (const [file, s] of Object.entries(specs)) {
    test(`${file} is a valid regular lon/lat grid`, () => {
      const g = read(file);
      for (const f of ["id", "title", "units", "ramp", "vmin", "vmax",
                       "west", "south", "east", "north", "dlon", "dlat", "nx", "ny", "values"]) {
        expect(g[f], `${file} missing ${f}`).not.toBeUndefined();
      }
      expect(g.units).toBe(s.units);
      expect(g.ramp).toBe(s.ramp);
      expect(g.values.length).toBe(g.nx * g.ny);
      if (s.nx) { expect(g.nx).toBe(s.nx); expect(g.ny).toBe(s.ny); }
      // bounds sane
      expect(g.east).toBeGreaterThan(g.west);
      expect(g.north).toBeGreaterThan(g.south);
      if (s.global) {
        expect(g.west).toBe(-180); expect(g.east).toBe(180);
      } else {
        expect(g.east - g.west).toBeLessThan(180);   // regional patch
      }
      // dlon/dlat match bounds & dims
      expect(g.dlon).toBeCloseTo((g.east - g.west) / g.nx, 3);
      expect(g.dlat).toBeCloseTo((g.north - g.south) / g.ny, 3);
      // Some cells filled, values within a physical range. Reduce to min/max
      // and assert twice rather than calling expect() per cell: OISST alone is
      // 64,800 cells, and per-cell matchers make this test take minutes.
      const finite = g.values.filter((v) => v != null);
      expect(finite.length).toBeGreaterThan(1000);
      let lo = Infinity, hi = -Infinity;
      for (const v of finite) { if (v < lo) lo = v; if (v > hi) hi = v; }
      expect(lo, `${file} minimum value`).toBeGreaterThanOrEqual(s.ramp === "sst" ? -5 : 0);
      expect(hi, `${file} maximum value`).toBeLessThan(s.vmaxData);
    });
  }
});

test.describe("ocean_column.json (Argo RG)", () => {
  const c = read("ocean_column.json");

  test("column grid: shape, physical plausibility, land masked", () => {
    expect(c.levels.length).toBeGreaterThanOrEqual(15);
    expect(c.levels[0]).toBeLessThan(5);              // near-surface
    expect(c.levels[c.levels.length - 1]).toBeGreaterThan(1900);
    expect(c.month).toMatch(/^\d{4}-\d{2}$/);
    for (const f of ["t_now", "t_norm", "s_now", "s_norm"]) {
      expect(c[f].length).toBe(c.levels.length);
      for (const lev of c[f]) expect(lev.length).toBe(c.nx * c.ny);
    }
    const cell = (lon, lat) =>
      (Math.floor((lat - c.south) / c.dlat)) * c.nx + Math.floor((lon - c.west) / c.dlon);
    // tropical Pacific: warm surface, cold abyss, monotonically stratified-ish
    const i = cell(-170, 0);
    const surf = c.t_now[0][i] / 100, deep = c.t_now[c.levels.length - 1][i] / 100;
    expect(surf).toBeGreaterThan(20);
    expect(surf).toBeLessThan(35);
    expect(deep).toBeGreaterThan(0);
    expect(deep).toBeLessThan(6);
    // salinity is ocean-like
    expect(c.s_now[0][i] / 100).toBeGreaterThan(30);
    expect(c.s_now[0][i] / 100).toBeLessThan(40);
    // land is null, not zero: central Sahara and central Asia
    expect(c.t_now[0][cell(10, 21)]).toBeNull();
    expect(c.t_now[0][cell(90, 47)]).toBeNull();
    // ocean coverage is roughly the ocean fraction of the domain
    const filled = c.t_now[0].filter((v) => v != null).length;
    expect(filled / (c.nx * c.ny)).toBeGreaterThan(0.5);
    expect(filled / (c.nx * c.ny)).toBeLessThan(0.8);
  });
});

test.describe("argo_t300.json (subsurface anomaly grid)", () => {
  const g = read("argo_t300.json");

  test("valid 1-degree grid of modest anomalies, land masked", () => {
    expect(g.ramp).toBe("anom");
    expect(g.units).toBe("°C");
    expect(g.values.length).toBe(g.nx * g.ny);
    const fin = g.values.filter((v) => v != null);
    // subsurface anomalies are modest — a symmetric-ish diverging field
    expect(Math.min(...fin)).toBeGreaterThan(-8);
    expect(Math.max(...fin)).toBeLessThan(8);
    const warm = fin.filter((v) => v > 0).length / fin.length;
    expect(warm).toBeGreaterThan(0.3);                 // both signs well represented
    expect(warm).toBeLessThan(0.9);                    // (warming ocean → skewed warm is fine)
    // fill fraction ≈ ocean fraction; a land-as-zero bug would push this to ~1
    expect(fin.length / g.values.length).toBeGreaterThan(0.5);
    expect(fin.length / g.values.length).toBeLessThan(0.75);
  });
});

test.describe("GLORYS surface snapshots", () => {
  const c = read("currents.json");
  const m = read("mld.json");
  const s = read("ocean_surface.json");
  const cell = (g, lon, lat) =>
    (Math.floor((lat - g.south) / g.dlat)) * g.nx + Math.floor((lon - g.west) / g.dlon);

  test("currents: the Gulf Stream is faster than the gyre interior; land null", () => {
    expect(c.units).toBe("m/s");
    const gulf = c.values[cell(c, -74, 36)];
    const gyre = c.values[cell(c, -40, 30)];
    expect(gulf).toBeGreaterThan(0.5);
    expect(gyre).toBeLessThan(0.3);
    expect(gulf).toBeGreaterThan(gyre * 3);
    expect(c.values[cell(c, 10, 21)]).toBeNull();       // Sahara
    const fin = c.values.filter((v) => v != null);
    expect(Math.max(...fin)).toBeLessThan(4);           // fill-value bug guard
    expect(fin.length / c.values.length).toBeGreaterThan(0.6);
    expect(fin.length / c.values.length).toBeLessThan(0.85);
  });

  test("mixed-layer depth is physical and land-masked", () => {
    const fin = m.values.filter((v) => v != null);
    expect(Math.min(...fin)).toBeGreaterThanOrEqual(0); // a depth, never negative
    expect(Math.max(...fin)).toBeLessThan(2500);
    expect(m.values[cell(m, 90, 47)]).toBeNull();       // central Asia
  });

  test("currents/mld are month-keyed indexes over the full 1993→ archive", () => {
    for (const [g, dir] of [[c, "currents_y"], [m, "mld_y"]]) {
      expect(g.latest).toMatch(/^\d{4}-\d{2}$/);
      const avail = g.monthsAvailable;
      expect(avail[0]).toBe("1993-01");                    // GREP era starts here
      expect(avail.length).toBeGreaterThanOrEqual(300);    // ~full monthly archive
      expect([...avail].sort()).toEqual(avail);            // sorted stamps
      expect(g.latest).toBe(avail[avail.length - 1]);
      expect(g.yearDir).toBe(`data/${dir}`);
      // the index inlines only the latest year; values mirrors the latest
      // month (the backward-compatible view the tests above use)
      for (const k of Object.keys(g.months)) {
        expect(k.slice(0, 4)).toBe(g.latest.slice(0, 4));
        expect(g.months[k].length).toBe(g.nx * g.ny);
      }
      expect(g.values).toEqual(g.months[g.latest]);
      // year files: complete years carry 12 full-size month arrays
      for (const yr of ["1993", "2010"]) {
        const y = read(`${dir}/${yr}.json`);
        expect(Object.keys(y.months).length).toBe(12);
        for (const arr of Object.values(y.months)) expect(arr.length).toBe(g.nx * g.ny);
      }
      // seam sanity: the 1/4° GREP era and the 1/12° GLORYS12 era should
      // agree on what is ocean (a mask bug would jump the fill fraction)
      const y93 = read(`${dir}/1993.json`);
      const fin93 = y93.months["1993-07"].filter((v) => v != null).length;
      const finNow = g.values.filter((v) => v != null).length;
      expect(Math.abs(fin93 - finNow) / finNow).toBeLessThan(0.1);
    }
  });

  test("historic months carry real circulation (Gulf Stream, 1993 and 2010)", () => {
    const y93 = read("currents_y/1993.json");
    const y10 = read("currents_y/2010.json");
    for (const [months, k] of [[y93.months, "1993-07"], [y10.months, "2010-01"]]) {
      const gulf = months[k][cell(c, -74, 36)];
      const gyre = months[k][cell(c, -40, 30)];
      expect(gulf).toBeGreaterThan(0.25);                 // 1/4° binned: a bit smoother
      expect(gulf).toBeGreaterThan(gyre * 2);
      expect(months[k][cell(c, 10, 21)]).toBeNull();      // Sahara stays land
    }
  });

  test("packed surface fields align and carry a month stamp", () => {
    expect(s.month).toMatch(/^\d{4}-\d{2}$/);
    for (const f of ["u", "v", "zos", "mld"]) expect(s[f].length).toBe(s.nx * s.ny);
    // Gulf Stream flows broadly northeastward off Hatteras: u > 0
    const i = cell(s, -74, 36);
    expect(s.u[i] / 100).toBeGreaterThan(0.2);
  });
});

test.describe("GFS forecast bakes", () => {
  const t = read("gfs_temp.json");
  const p = read("gfs_precip.json");
  const cell = (g, lon, lat) =>
    (Math.floor((lat - g.south) / g.dlat)) * g.nx + Math.floor((lon - g.west) / g.dlon);

  test("temperature: day-keyed, contiguous, full coverage, physical", () => {
    expect(t.keyLen).toBe(10);
    expect(t.init).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}Z$/);
    expect(t.monthsAvailable.length).toBeGreaterThanOrEqual(10);   // ~11 daily frames
    for (let i = 1; i < t.monthsAvailable.length; i++) {
      const gap = new Date(t.monthsAvailable[i]) - new Date(t.monthsAvailable[i - 1]);
      expect(gap).toBe(864e5);                                     // strictly daily
    }
    expect(t.latest).toBe(t.monthsAvailable[t.monthsAvailable.length - 1]);
    for (const s of t.monthsAvailable) {
      const v = t.months[s];
      expect(v.length).toBe(t.nx * t.ny);
      const fin = v.filter((x) => x != null);
      expect(fin.length / v.length).toBeGreaterThan(0.99);         // temp covers the globe
      expect(Math.min(...fin)).toBeGreaterThan(-95);
      expect(Math.max(...fin)).toBeLessThan(60);
    }
    // season-agnostic physics: the tropical Pacific is warm year-round, and
    // whichever pole is in winter is far colder than the tropics
    const v = t.months[t.monthsAvailable[0]];
    const tropics = v[cell(t, -140, 0)];
    expect(tropics).toBeGreaterThan(15);
    const poles = Math.min(v[cell(t, 0, -80)], v[cell(t, 0, 85)]);
    expect(tropics - poles).toBeGreaterThan(20);
  });

  test("precipitation: full days only, non-negative, rain exists, dry transparent", () => {
    expect(p.keyLen).toBe(10);
    expect(p.init).toBe(t.init);                                   // same model run
    expect(p.monthsAvailable.length).toBeGreaterThanOrEqual(8);
    for (const s of p.monthsAvailable) {
      const v = p.months[s];
      expect(v.length).toBe(p.nx * p.ny);
      const fin = v.filter((x) => x != null);
      expect(Math.min(...fin)).toBeGreaterThanOrEqual(1);          // <0.5 mm baked as null
      expect(Math.max(...fin)).toBeGreaterThan(20);                // the ITCZ always rains
      expect(Math.max(...fin)).toBeLessThan(1000);
      const wet = fin.length / v.length;
      expect(wet).toBeGreaterThan(0.2);                            // neither empty
      expect(wet).toBeLessThan(0.85);                              // nor raining everywhere
    }
    expect(p.values).toEqual(p.months[p.monthsAvailable[0]]);
  });
});

test.describe("eei.json (ocean heat / energy imbalance)", () => {
  const e = read("eei.json");

  test("OHC series span, rise, and a plausible planetary imbalance", () => {
    expect(e.y700[0]).toBe(1955);
    expect(e.y700[e.y700.length - 1]).toBeGreaterThanOrEqual(2024);
    expect(e.y2000[0]).toBe(2005);
    expect(e.ohc700.length).toBe(e.y700.length);
    expect(e.ohc2000.length).toBe(e.y2000.length);
    // the ocean has gained heat, a lot of it, and faster over 0-2000 m
    const rise2000 = e.ohc2000[e.ohc2000.length - 1] - e.ohc2000[0];
    expect(rise2000).toBeGreaterThan(15);              // ×10^22 J over ~20 yr
    expect(e.zj_gained).toBeGreaterThan(150);
    expect(e.zj_gained).toBeLessThan(400);
    // the headline rate: CERES-era EEI is ~0.5-1.5 W/m² — outside that,
    // something broke in units (the 0.6213 J/yr→W/m² chain) or parsing
    expect(e.rate10).toBeGreaterThan(0.4);
    expect(e.rate10).toBeLessThan(1.3);
    expect(e.eei10).toBeCloseTo(e.rate10 / 0.9, 2);
    // rolling rates align with their series and are mostly positive recently
    expect(e.rate2000.length).toBe(e.y2000.length);
    const recent = e.rate2000.slice(-8).filter((v) => v != null);
    expect(recent.every((v) => v > 0)).toBe(true);
  });

  test("ERF forcing curves: human push climbs, natural hugs zero except eruptions", () => {
    expect(e.erf_years[0]).toBe(e.y700[0]);              // aligned to the OHC record
    expect(e.erf_years.length).toBe(e.erf_anthro.length);
    expect(e.erf_years.length).toBe(e.erf_natural.length);
    const last = e.erf_anthro[e.erf_anthro.length - 1];
    expect(last).toBeGreaterThan(2.3);                   // AR6-consistent present-day push
    expect(last).toBeLessThan(3.5);
    expect(e.erf_anthro[0]).toBeLessThan(0.8);           // and it was small in the 1950s
    // natural = solar + volcanic: small except eruption dips
    const nat = e.erf_natural;
    const mean = nat.reduce((s, v) => s + v, 0) / nat.length;
    expect(Math.abs(mean)).toBeLessThan(0.25);
    const pin = nat[e.erf_years.indexOf(1992)];          // year after Pinatubo
    expect(pin).toBeLessThan(-1);                        // the dip is real and large
    const quiet = nat.filter((v) => v > -0.5);
    expect(quiet.length / nat.length).toBeGreaterThan(0.85);  // ...and rare
  });

  test("ENSO and volcano annotations are present and sane", () => {
    // the canonical events classify correctly from NOAA's ONI (DJF convention)
    expect(e.oni["1998"]).toBeGreaterThan(1.5);        // 97/98 El Niño
    expect(e.oni["2016"]).toBeGreaterThan(1.5);        // 15/16 El Niño
    expect(e.oni["2011"]).toBeLessThan(-0.5);          // 10/11 La Niña
    // a balanced mix of events over the record, not a parser artefact
    const vals = Object.values(e.oni);
    const nino = vals.filter((v) => v >= 0.5).length;
    const nina = vals.filter((v) => v <= -0.5).length;
    expect(nino).toBeGreaterThan(15);
    expect(nina).toBeGreaterThan(15);
    expect(nino + nina).toBeLessThan(vals.length);     // neutral years exist
    expect(e.volcanoes.map((v) => v.n)).toContain("Pinatubo");
    expect(e.volcanoes.find((v) => v.n === "Pinatubo").y).toBe(1991);
  });
});

test.describe("drivers.json (categorical grid)", () => {
  const g = read("drivers.json");

  test("is a packed categorical grid, self-describing its own palette", () => {
    for (const f of ["id", "title", "units", "source", "citation", "classes",
                     "west", "south", "east", "north", "dlon", "dlat", "nx", "ny",
                     "period", "packed"]) {
      expect(g[f], `drivers.json missing ${f}`).not.toBeUndefined();
    }
    // No `values` array and no ramp on purpose: the cell is a CLASS, so it
    // indexes the palette shipped in this very file rather than running
    // through a colour ramp. A ramp here would invent an ordering between
    // "logging" and "wildfire" that does not exist.
    expect(g.values).toBeUndefined();
    expect(g.ramp).toBeUndefined();
    expect(g.vmin).toBeUndefined();

    // WRI's seven drivers, each with a label and an rgb triple the globe paints
    // from. The palette travelling WITH the values is the point — it cannot
    // drift out of sync with them the way a copy in the layer config would.
    expect(g.classes.length).toBe(7);
    const codes = g.classes.map((c) => c.code);
    expect(codes).toEqual([1, 2, 3, 4, 5, 6, 7]);
    for (const c of g.classes) {
      expect(typeof c.label).toBe("string");
      expect(c.label.length).toBeGreaterThan(3);
      expect(c.rgb.length).toBe(3);
      for (const ch of c.rgb) { expect(ch).toBeGreaterThanOrEqual(0); expect(ch).toBeLessThan(256); }
    }

    // geometry: 0.25° global-in-lon, truncated in lat where the source stops
    expect(g.packed.length).toBe(g.nx * g.ny);
    expect(g.dlon).toBeCloseTo((g.east - g.west) / g.nx, 6);
    expect(g.dlat).toBeCloseTo((g.north - g.south) / g.ny, 6);
    expect(g.west).toBe(-180); expect(g.east).toBe(180);
    expect(g.north).toBeLessThanOrEqual(90);   // the product stops at 84N/56S:
    expect(g.south).toBeGreaterThan(-90);      // no forest to lose beyond
  });

  test("every packed character is a mapped driver or empty, and the shares are plausible", () => {
    // One pass with a counter rather than expect() per cell — 806k cells, and
    // per-cell matchers turn this into a multi-minute test.
    const counts = new Map();
    for (let i = 0; i < g.packed.length; i++) {
      const c = g.packed[i];
      counts.set(c, (counts.get(c) ?? 0) + 1);
    }
    const legal = new Set([".", "1", "2", "3", "4", "5", "6", "7"]);
    for (const c of counts.keys()) expect(legal.has(c), `illegal packed char ${JSON.stringify(c)}`).toBe(true);

    // Most of the planet has no mapped forest loss (ocean, desert, ice, intact
    // forest) — a grid that came out mostly filled would mean the nodata value
    // leaked into a class.
    const filled = g.packed.length - (counts.get(".") ?? 0);
    expect(filled).toBeGreaterThan(50000);
    expect(filled / g.packed.length).toBeLessThan(0.5);

    // Agriculture dominates globally and mining is a sliver — the published
    // result. If a re-bake ever inverted the class order this would catch it.
    const share = (code) => (counts.get(String(code)) ?? 0) / filled;
    expect(share(1)).toBeGreaterThan(0.2);    // permanent agriculture
    expect(share(2)).toBeLessThan(0.1);       // hard commodities
    expect(share(5)).toBeGreaterThan(0.05);   // wildfire — boreal Canada, Siberia
    for (let c = 1; c <= 7; c++) expect(share(c), `class ${c} is empty`).toBeGreaterThan(0);
  });
});

test.describe("amoc_eval_mask.json (what the forecaster actually computes)", () => {
  const g = read("amoc_eval_mask.json");

  // This file is not a measurement of the world — it is a picture of an
  // EXPERIMENT, written by the experiment's own scoring code
  // (ml/rollout_spatial.py --export-mask). So the tests here are about it
  // still describing the same experiment: the same window, the same nesting,
  // the same counts the evaluator reports in rollout_spatial.json. If the
  // tensor window or the corridor recipe changes and this file is not re-baked,
  // the globe would be drawing last week's experiment — which is the one
  // failure a picture of a model can have and still look perfectly fine.
  test("is a packed categorical grid over the family3 window", () => {
    for (const f of ["id", "title", "units", "source", "citation", "classes",
                     "west", "south", "east", "north", "dlon", "dlat", "nx", "ny",
                     "period", "counts", "corridor_def", "packed"]) {
      expect(g[f], `amoc_eval_mask.json missing ${f}`).not.toBeUndefined();
    }
    expect(g.values).toBeUndefined();     // categorical: palette, not a ramp
    expect(g.ramp).toBeUndefined();
    expect(g.doc).toMatch(/^https:\/\/github\.com\/blauewelt\/earth\/blob\/main\/ml\//);

    // the ML window: lat 0..70 N, lon -100..+20 E at a quarter degree, cell
    // CENTRES on the tensor's own lat/lon vectors (hence the half-cell bounds)
    expect(g.nx).toBe(481); expect(g.ny).toBe(281);
    expect(g.dlon).toBeCloseTo(0.25, 6); expect(g.dlat).toBeCloseTo(0.25, 6);
    expect(g.west).toBeCloseTo(-100.125, 6); expect(g.east).toBeCloseTo(20.125, 6);
    expect(g.south).toBeCloseTo(-0.125, 6); expect(g.north).toBeCloseTo(70.125, 6);
    expect(g.packed.length).toBe(g.nx * g.ny);

    expect(g.classes.map((c) => c.code)).toEqual([1, 2, 3]);
    for (const c of g.classes) {
      expect(c.label.length).toBeGreaterThan(3);
      expect(c.rgb.length).toBe(3);
      for (const ch of c.rgb) { expect(ch).toBeGreaterThanOrEqual(0); expect(ch).toBeLessThan(256); }
    }
  });

  test("the roles NEST and the counts match the pixels actually drawn", () => {
    // One pass with counters, never expect() per cell (135k cells).
    const counts = new Map();
    for (let i = 0; i < g.packed.length; i++) {
      const c = g.packed[i];
      counts.set(c, (counts.get(c) ?? 0) + 1);
    }
    for (const c of counts.keys()) {
      expect(new Set([".", "1", "2", "3"]).has(c), `illegal char ${JSON.stringify(c)}`).toBe(true);
    }
    const n1 = counts.get("1") ?? 0, n2 = counts.get("2") ?? 0, n3 = counts.get("3") ?? 0;
    // The header's counts are the NESTED sets (rolled ⊇ corridor ⊇ section);
    // the packed cells carry only the most specific role. Checking one against
    // the other is what makes the legend's promise true.
    expect(n1 + n2 + n3).toBe(g.counts.rolled);
    expect(n2 + n3).toBe(g.counts.corridor);
    expect(n3).toBe(g.counts.section);

    // the eval's own numbers, as reported by rollout_spatial.py on the
    // sha-pinned family3 tensor — 84,405 ocean pixels, 265 section cells
    expect(g.counts.rolled).toBe(84405);
    expect(g.counts.section).toBe(265);
    expect(g.counts.corridor).toBe(g.corridor_def.n_px);
    expect(g.corridor_def.of).toBe(g.counts.rolled);
    // the corridor is a MINORITY of the window (it is the fast quarter plus a
    // dilation): a corridor that swallowed the basin would score nothing in
    // particular, and one that vanished would score almost nothing
    expect(g.counts.corridor / g.counts.rolled).toBeGreaterThan(0.1);
    expect(g.counts.corridor / g.counts.rolled).toBeLessThan(0.6);
    expect(g.corridor_def.pctl).toBe(75);
    expect(g.corridor_def.dilate_cells).toBe(2);
    expect(g.section_row.lat).toBeCloseTo(26.5, 2);   // RAPID's latitude
  });

  test("the section really lies on the RAPID row, and the Gulf Stream is in the corridor", () => {
    const at = (lon, lat) => {
      const ix = Math.floor((lon - g.west) / g.dlon);
      const iy = Math.floor((lat - g.south) / g.dlat);
      return g.packed[iy * g.nx + ix];
    };
    // RAPID sits at 26.5°N between the Bahamas and the African shelf
    expect(at(-70, 26.5)).toBe("3");
    expect(at(-40, 26.5)).toBe("3");
    // one row north is corridor or plain ocean, never section — a section that
    // smeared across rows would mean the row index is off by one
    expect(["1", "2"]).toContain(at(-40, 26.8));
    // Gulf Stream off Cape Hatteras — the one place in this basin where a
    // "fastest quarter by mean current speed" rule cannot miss
    expect(at(-73, 36)).toBe("2");
    // outside the window there is nothing at all (the model has no state there)
    expect(at(-99, 40)).toBe(".");     // Pacific side of the window edge: land

    // Chris's spec for the corridor was a ROUTE — "from the Gulf of Mexico to
    // northern Europe, and back" — so test the route as coverage per region,
    // not at hand-picked points. (Written after a point test at 62°N/10°W
    // failed: the North Atlantic Current is there, but its fastest quarter
    // hugs the shelf edge a few hundred km away. The percentile threshold is
    // the data's opinion about where the flow is, and a single coordinate is
    // mine.)
    const arr = new Uint8Array(g.packed.length);
    for (let i = 0; i < g.packed.length; i++) arr[i] = g.packed[i] === "." ? 0 : +g.packed[i];
    const frac = (la0, la1, lo0, lo1) => {
      let cor = 0, ocean = 0;
      for (let iy = 0; iy < g.ny; iy++) {
        const la = g.south + (iy + 0.5) * g.dlat;
        if (la < la0 || la >= la1) continue;
        for (let ix = 0; ix < g.nx; ix++) {
          const lo = g.west + (ix + 0.5) * g.dlon;
          if (lo < lo0 || lo >= lo1) continue;
          const v = arr[iy * g.nx + ix];
          if (v >= 1) ocean++;
          if (v >= 2) cor++;
        }
      }
      return ocean ? cor / ocean : 0;
    };
    expect(frac(20, 30, -98, -80)).toBeGreaterThan(0.3);   // Gulf of Mexico / Loop
    expect(frac(30, 45, -80, -60)).toBeGreaterThan(0.3);   // Gulf Stream
    expect(frac(45, 60, -50, -10)).toBeGreaterThan(0.1);   // North Atlantic Current
    expect(frac(55, 70, -20, 20)).toBeGreaterThan(0.05);   // reaches northern Europe
    expect(frac(0, 20, -60, -20)).toBeGreaterThan(0.2);    // and the tropical return
  });
});

test.describe("cities.json (place-name reference points)", () => {
  const d = read("cities.json");

  // These are the map's *reference points*, not a dataset — nothing here gets a
  // legend or a date. What they must be is trustworthy anchors: a name at the
  // wrong coordinate is worse than no name, because the reader believes it.
  test("is a self-describing, public-domain gazetteer with the required fields", () => {
    expect(d.id).toBe("cities");
    expect(d.source).toMatch(/Natural Earth/i);
    expect(d.citation).toMatch(/[Pp]ublic domain/);
    expect(d.doc).toMatch(/^https:\/\//);
    expect(d.snapshot).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(d.count).toBe(d.places.length);
    // Enough to name the world at every altitude, few enough that the whole
    // file is one modest fetch. Natural Earth 10m ships ~7.3k.
    expect(d.places.length).toBeGreaterThan(5000);
    expect(d.places.length).toBeLessThan(20000);
  });

  test("every place is nameable, locatable and carries its declutter rung", () => {
    // Single pass with plain conditionals; 7k places × six matchers each would
    // dominate the suite's runtime for no extra signal.
    let bad = null;
    const seen = new Set();
    for (const c of d.places) {
      const why =
        typeof c.n !== "string" || !c.n.length ? "name" :
        !(c.o >= -180 && c.o <= 180) ? "lon" :
        !(c.a >= -90 && c.a <= 90) ? "lat" :
        !(c.z >= 1 && c.z <= 12) ? "min_zoom" :
        !(Number.isInteger(c.p) && c.p >= 0) ? "pop" :
        typeof c.c !== "string" ? "country" :
        (c.cap !== 0 && c.cap !== 1) ? "capital flag" : null;
      if (why && !bad) bad = `${c.n}: bad ${why} (${JSON.stringify(c)})`;
      seen.add(`${c.n}|${c.c}`);
    }
    expect(bad).toBeNull();
    // Names repeat across countries (Springfield, San José); wholesale
    // duplication inside one country would mean the source was concatenated.
    expect(seen.size).toBeGreaterThan(d.places.length * 0.95);
  });

  test("is sorted most-important-first, so a client can truncate the tail", () => {
    // The client turns `z` into a per-label DistanceDisplayCondition, but the
    // sort is what lets anything downstream cut the file short without losing
    // the world's capitals — and it settles draw order for overlapping labels.
    for (let i = 1; i < d.places.length; i++) {
      expect(d.places[i].z, `unsorted at ${i}`).toBeGreaterThanOrEqual(d.places[i - 1].z);
    }
    expect(d.places[0].z).toBeLessThan(2);          // Natural Earth's top tier
    expect(d.places[0].p).toBeGreaterThan(1e7);     // …and it's a megacity
  });

  test("the globe-zoom tier is a readable handful, and the capitals are all there", () => {
    // What survives the full-globe view is the whole point of the feature: a
    // dozen-ish names orient you, a hundred is a smear of white text.
    const globe = d.places.filter((c) => c.z <= 3);
    expect(globe.length).toBeGreaterThan(10);
    expect(globe.length).toBeLessThan(120);

    const caps = d.places.filter((c) => c.cap === 1);
    expect(caps.length).toBeGreaterThan(150);       // ~200 sovereign capitals
    expect(caps.length).toBeLessThan(300);
    const capNames = new Set(caps.map((c) => c.n));
    // Accents intact: the names are baked with ensure_ascii off, and "Brasilia"
    // for "Brasília" would mean the encoding got flattened somewhere.
    for (const n of ["Paris", "Tokyo", "Nairobi", "Brasília", "Canberra"]) {
      expect(capNames.has(n), `${n} is not flagged a capital`).toBe(true);
    }

    // Spot-check two anchors against known coordinates: this is the test that
    // would catch a lon/lat swap, which every other assertion here would pass.
    const at = (n) => d.places.find((c) => c.n === n);
    expect(at("Paris").o).toBeCloseTo(2.33, 1);
    expect(at("Paris").a).toBeCloseTo(48.87, 1);
    expect(at("Sydney").o).toBeCloseTo(151.18, 1);
    expect(at("Sydney").a).toBeCloseTo(-33.92, 1);
  });
});

test.describe("gazetteer.json (the deep tier under Natural Earth)", () => {
  const d = read("gazetteer.json");
  const ne = read("cities.json");

  // cities.json is a cartographic SELECTION — the right list to keep a map
  // legible, the wrong list to answer "where is Peniche". This file is the
  // other job, and its whole contract is that it CONTINUES the same ladder
  // rather than starting a second one.
  test("is self-describing, attributed, and picks up exactly where Natural Earth stops", () => {
    expect(d.id).toBe("gazetteer");
    expect(d.source).toMatch(/GeoNames/i);
    // CC BY 4.0 obliges us to say so — in the file as well as in the UI.
    expect(d.citation).toMatch(/GeoNames/);
    expect(d.citation).toMatch(/CC BY 4\.0/);
    expect(d.license).toMatch(/CC BY 4\.0/);
    expect(d.doc).toMatch(/^https:\/\//);
    expect(d.snapshot).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(d.count).toBe(d.places.length);
    expect(d.places.length).toBeGreaterThan(20000);   // a gazetteer, not a selection

    // The seam. zFrom must equal the rung Natural Earth ends on, or the two
    // tiers either overlap (double labels) or leave an altitude band empty.
    expect(d.zFrom).toBeCloseTo(ne.places[ne.places.length - 1].z, 5);
    expect(d.places[0].z).toBeCloseTo(d.zFrom, 5);

    // The rung spacing is MEASURED from Natural Earth's own cumulative counts,
    // not chosen: one rung down quarters the visible area, so it can carry a
    // few times the places at the same on-screen density. Anything near 1 would
    // mean the ladder had stopped decluttering; anything huge would mean the
    // whole file lands on one rung.
    expect(d.growth).toBeGreaterThan(2);
    expect(d.growth).toBeLessThan(6);
  });

  test("every place is nameable, locatable, and carries a rung below the seam", () => {
    let bad = null;
    for (let i = 0; i < d.places.length; i++) {
      const p = d.places[i];
      const why =
        typeof p.n !== "string" || !p.n.length ? "name" :
        !(p.o >= -180 && p.o <= 180) ? "lon" :
        !(p.a >= -90 && p.a <= 90) ? "lat" :
        !(p.z >= d.zFrom && p.z <= 16) ? "rung" :
        !(Number.isInteger(p.p) && p.p >= 0) ? "pop" :
        !(typeof p.c === "string" && p.c.length === 2) ? "country code" :
        p.z < (i ? d.places[i - 1].z : -1) ? "sort order" : null;
      if (why && !bad) bad = `${p.n}: bad ${why} (${JSON.stringify(p)})`;
    }
    expect(bad).toBeNull();

    // Country codes are ISO-3166 alpha-2 and the file ships its own lookup —
    // "Portugal" × 500 is the single most compressible thing in it, and a code
    // with no expansion would render as "PT" in the search results.
    const missing = [...new Set(d.places.map((p) => p.c))].filter((c) => !d.countries[c]);
    expect(missing).toEqual([]);
    expect(d.countries.PT).toMatch(/Portugal/);
  });

  test("Peniche is in it, in the right place, at a town's altitude", () => {
    // The literal bug report. A user looking at the sea off Peniche got a globe
    // that could neither name the town nor be asked about it.
    const p = d.places.find((x) => x.n === "Peniche" && x.c === "PT");
    expect(p, "Peniche is missing from the gazetteer").toBeTruthy();
    expect(p.a).toBeCloseTo(39.36, 1);
    expect(p.o).toBeCloseTo(-9.38, 1);
    expect(p.p).toBeGreaterThan(5000);
    // Below the seam: a town of 15k has no business labelling the globe view.
    expect(p.z).toBeGreaterThan(d.zFrom);

    // Accents survive the bake here too — the client folds diacritics when
    // searching, but the LABEL must still read "Guimarães".
    expect(d.places.some((x) => x.n === "Guimarães")).toBe(true);
  });

  test("adds to Natural Earth instead of repeating it", () => {
    // Deduplication is what keeps the two tiers from drawing "Lisboa" next to
    // "Lisbon" a kilometre apart. Same name within ~half a degree is the case
    // that actually bites, because the two projects place big cities from
    // different sources and can differ by several kilometres.
    const fold = (s) => s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
    const near = new Map();
    for (const c of ne.places) {
      const k = fold(c.n);
      if (!near.has(k)) near.set(k, []);
      near.get(k).push(c);
    }
    let dupes = 0, first = null;
    for (const p of d.places) {
      const k = fold(p.n);
      for (const c of near.get(k) || []) {
        if (Math.abs(c.a - p.a) < 0.5 && Math.abs(c.o - p.o) < 0.6) {
          dupes++; if (!first) first = `${p.n} (${p.a}, ${p.o})`;
          break;
        }
      }
    }
    expect(dupes, `duplicated across tiers, e.g. ${first}`).toBe(0);

    // And the two together must actually cover more of the world than one:
    // Portugal is 24 places in Natural Earth, which is why the report happened.
    const pt = d.places.filter((p) => p.c === "PT").length;
    expect(pt).toBeGreaterThan(100);
  });
});

test.describe("islands.json (the tier that is not a settlement)", () => {
  const d = read("islands.json");

  // The first place tier in the app that names a piece of GROUND rather than a
  // population. Both gazetteers carry only populated places, which is why Sylt
  // — 43 km of German North Sea coast — was on the globe and unnamed while the
  // town of Westerland standing on it was labelled.
  test("is self-describing and attributed to both of its sources", () => {
    expect(d.id).toBe("islands");
    expect(d.source).toMatch(/Natural Earth/i);
    expect(d.source).toMatch(/GeoNames/i);
    // Natural Earth is public domain, GeoNames is CC BY 4.0 — the obligation
    // is the union, so both must be stated here and in the UI footer.
    expect(d.citation).toMatch(/Natural Earth/);
    expect(d.citation).toMatch(/GeoNames/);
    expect(d.license).toMatch(/CC BY 4\.0/);
    expect(d.doc).toMatch(/^https:\/\//);
    expect(d.snapshot).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(d.count).toBe(d.islands.length);
    expect(d.islands.length).toBeGreaterThan(2000);
  });

  test("every island is nameable, locatable, measured, and sorted biggest first", () => {
    let bad = null;
    for (let i = 0; i < d.islands.length; i++) {
      const p = d.islands[i];
      const why =
        typeof p.n !== "string" || !p.n.length ? "name" :
        !(p.o >= -180 && p.o <= 180) ? "lon" :
        !(p.a >= -90 && p.a <= 90) ? "lat" :
        !(p.e > 0 && p.e < 5000) ? "extent" :
        !(p.s === 0 || p.s === 1) ? "source flag" :
        typeof p.c !== "string" ? "country code" :
        // The extent IS the ladder here — there is no min_zoom to borrow, so
        // the file's sort order is what makes "everything that could be
        // visible now" a contiguous prefix the client can walk.
        p.e > (i ? d.islands[i - 1].e : Infinity) ? "sort order" : null;
      if (why && !bad) bad = `${p.n}: bad ${why} (${JSON.stringify(p)})`;
    }
    expect(bad).toBeNull();

    const missing = [...new Set(d.islands.map((p) => p.c))]
      .filter((c) => c && !d.countries[c]);
    expect(missing).toEqual([]);
    expect(d.countries.DE).toMatch(/Germany/);
  });

  test("the continent cut keeps Greenland and drops Australia", () => {
    // The standard line — Australia is a continent, Greenland is the largest
    // island — falls out of one measured threshold rather than a list of
    // exceptions. The gap either side of it is a factor of ~3.6 in area, so
    // nothing is near the cut and no ring is a judgement call.
    expect(d.continentCutKm2).toBeGreaterThan(2.2e6);
    expect(d.continentCutKm2).toBeLessThan(7e6);
    expect(d.islands[0].n).toBe("Greenland");
    // …and exactly one of it: Natural Earth draws GREENLAND as sixteen label
    // patches, and naming the largest ring under each produced sixteen.
    expect(d.islands.filter((p) => p.n === "Greenland").length).toBe(1);
    for (const n of ["Australia", "Antarctica", "Africa", "Eurasia"]) {
      expect(d.islands.some((p) => p.n === n)).toBe(false);
    }
  });

  test("Sylt is in it — the literal report — and so are the ones GeoNames cannot name", () => {
    const s = d.islands.find((p) => p.n === "Sylt");
    expect(s, "Sylt is missing from islands.json").toBeTruthy();
    expect(s.a).toBeCloseTo(54.9, 1);
    expect(s.o).toBeCloseTo(8.35, 1);
    expect(s.e).toBeGreaterThan(30);       // ~43 km of dune, north to south
    expect(s.e).toBeLessThan(60);
    expect(s.c).toBe("DE");

    // The curated Natural Earth tier exists for the islands GeoNames has no
    // T-class entry for: its only "Ireland" is an ISLF in the UAE, so a
    // GeoNames-only join labelled the whole island "Coney Island".
    const ie = d.islands.find((p) => p.n === "Ireland");
    expect(ie, "Ireland is missing").toBeTruthy();
    expect(ie.s).toBe(1);                  // named by Natural Earth, not GeoNames
    expect(ie.c).toBe("IE");               // the ring's majority country, not one entry's
    expect(ie.e).toBeGreaterThan(400);

    // Both naming passes carry real weight; neither is a rounding error.
    const ne = d.islands.filter((p) => p.s === 1).length;
    expect(ne).toBeGreaterThan(100);
    expect(d.islands.length - ne).toBeGreaterThan(1000);
  });

  test("the label anchor is a point on the island, not in the sea beside it", () => {
    // A crescent atoll or a fjord coast puts its own centroid in the water, so
    // the baker falls back to a representative point. Spot-check the shape of
    // the fix on a ring notorious for it rather than re-implementing
    // point-in-polygon here: the anchor must at least sit inside the island's
    // own bounding circle.
    const a = d.islands.find((p) => p.n === "Ireland");
    expect(Math.abs(a.a - 53.2)).toBeLessThan(1.5);
    expect(Math.abs(a.o + 8.0)).toBeLessThan(1.5);
  });
});

test.describe("observation times travel with the data", () => {
  // Every number the app prints is stamped with WHEN it was observed. That date
  // has to live in the file — a value whose only date is `snapshot` can be
  // stamped with the day it was DOWNLOADED, which is not an observation time at
  // all. These fields are the app's only honest source for those rows.

  test("gridded climatologies name the years they average", () => {
    const spans = {
      "gpcp.json": [1975, 2030],        // whole GPCP record, extended by re-bakes
      "oisst.json": [1991, 2020],       // WOA-style 30-year normal
      "eobs.json": [1945, 2030],
      "meteoswiss.json": [1991, 2020],  // RnormY9120
    };
    for (const [file, [lo, hi]] of Object.entries(spans)) {
      const g = read(file);
      expect(g.period, `${file} has no period`).toMatch(/^\d{4}-\d{4}$/);
      const [a, b] = g.period.split("-").map(Number);
      expect(b, `${file} period runs backwards`).toBeGreaterThan(a);
      expect(a).toBeGreaterThanOrEqual(lo);
      expect(b).toBeLessThanOrEqual(hi);
      // A period is not a bake date. If they were ever the same field the app
      // would print "downloaded today" as if it meant "observed today".
      expect(g.period).not.toBe(g.snapshot);
    }
  });

  test("moving grids name the month they observed", () => {
    for (const file of ["argo_t300.json", "ocean_column.json", "ocean_surface.json"]) {
      const g = read(file);
      expect(g.month, `${file} has no month`).toMatch(/^\d{4}-(0[1-9]|1[0-2])$/);
      // the observed month precedes the bake — Argo/GLORYS both lag real time
      if (g.snapshot) expect(g.month <= g.snapshot.slice(0, 7)).toBe(true);
      expect(g.month <= new Date().toISOString().slice(0, 7)).toBe(true);
    }
  });

  test("glacier thinning names the window it was measured over", () => {
    // The RATE has a definite window (Hugonnet et al.); the inventory COUNT
    // does not, which is why only one of the two rows carries a stamp.
    const g = read("glaciers.json");
    expect(g.dhdt_period).toMatch(/^\d{4}-\d{4}$/);
    const [a, b] = g.dhdt_period.split("-").map(Number);
    expect(b - a).toBeGreaterThanOrEqual(10);
    expect(b).toBeLessThanOrEqual(new Date().getUTCFullYear());
  });

  test("the driver map and the emitter inventory are year-addressed", () => {
    expect(read("drivers.json").period).toMatch(/^\d{4}-\d{4}$/);
    const t = read("climatetrace.json");
    // The card reads assets_by_year[year], not a bare `assets` key — a row that
    // reached for `assets` rendered nothing at all for as long as it existed.
    expect(Array.isArray(t.years)).toBe(true);
    expect(t.assets).toBeUndefined();
    for (const y of t.years) {
      expect(Array.isArray(t.assets_by_year[String(y)]), `no assets for ${y}`).toBe(true);
    }
  });

  test("every Argo float reports the date it last surfaced", () => {
    const a = read("argo.json");
    // The card stamps the "nearest float" row with that float's own last
    // report, so the date has to be per-float, not one date for the file.
    for (let i = 0; i < a.floats.length; i += 97) {
      expect(a.floats[i][3], `float ${i}`).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      expect(a.floats[i][3] <= a.snapshot).toBe(true);
    }
  });
});

/* A deploy that a browser refuses to pick up is indistinguishable, from the
 * user's side, from a deploy that never happened — and that is exactly how
 * this was reported: "I reloaded the tab twice and see no change." index.html
 * asked for `src/app.js`, the browser had `src/app.js`, and the CDN had said
 * it was fresh. Every local asset now carries the first 8 hex of its own
 * sha256, so its URL changes whenever its bytes do. These tests re-derive the
 * hashes from the files themselves, which means a forgotten
 * `scripts/stamp_assets.py` fails here rather than on someone's phone. */
test.describe("cache-busting stamps and the web app manifest", () => {
  const ROOT = path.join(__dirname, "..");
  const crypto = require("crypto");
  const slurp = (f) => fs.readFileSync(path.join(ROOT, f));
  const digest = (f) => crypto.createHash("sha256").update(slurp(f)).digest("hex").slice(0, 8);
  const html = slurp("index.html").toString("utf8");
  const manifest = JSON.parse(slurp("manifest.json").toString("utf8"));

  const ICONS = ["icon-192.png", "icon-512.png", "icon-512-maskable.png"];

  test("index.html versions every local asset with that file's own hash", () => {
    for (const asset of ["src/app.js", "src/style.css", "manifest.json", "icon-192.png"]) {
      const m = html.match(
        new RegExp(`(?<![\\w./-])${asset.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\$&")}\\?v=([0-9a-f]{8})`));
      expect(m, `${asset} is not version-stamped in index.html`).not.toBeNull();
      expect(m[1], `${asset} stamp is stale — run scripts/stamp_assets.py`).toBe(digest(asset));
    }
    // and nothing local is left unstamped
    expect(html).not.toMatch(/(?:href|src)="(?:src\/[\w.-]+|icon-[\w.-]+\.png|manifest\.json)"/);
  });

  test("the visible build marker matches the script it labels", () => {
    // The About tab prints this so a user can tell a stale cache from a
    // missing feature without opening dev tools.
    const m = html.match(/<code id="build-id">([0-9a-f]{8})<\/code>/);
    expect(m).not.toBeNull();
    expect(m[1]).toBe(digest("src/app.js"));
  });

  test("the manifest is installable: icons exist, are square, and are stamped", () => {
    expect(manifest.name).toBeTruthy();
    expect(manifest.short_name.length).toBeLessThanOrEqual(12); // home-screen label
    expect(manifest.display).toBe("standalone");
    // Relative, so the same file works at / under the test server and at
    // /earth/ on GitHub Pages. A leading slash would break one of the two.
    expect(manifest.start_url.startsWith("/")).toBe(false);
    expect(manifest.scope.startsWith("/")).toBe(false);
    expect(manifest.background_color).toMatch(/^#[0-9a-f]{6}$/i);

    const sizes = manifest.icons.map((i) => i.sizes);
    expect(sizes).toContain("192x192");           // Android's minimum pair
    expect(sizes).toContain("512x512");
    expect(manifest.icons.some((i) => i.purpose === "maskable")).toBe(true);

    for (const icon of manifest.icons) {
      const [file, v] = icon.src.split("?v=");
      expect(ICONS).toContain(file);
      expect(v, `${file} stamp is stale — run scripts/stamp_assets.py`).toBe(digest(file));
      const buf = slurp(file);
      // PNG signature + IHDR width/height, read straight from the bytes: an
      // icon whose declared size is a lie fails installability silently.
      expect(buf.slice(0, 8).toString("hex")).toBe("89504e470d0a1a0a");
      const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
      expect(`${w}x${h}`).toBe(icon.sizes);
      expect(buf.length).toBeGreaterThan(1000);   // not a placeholder
    }
  });

  test("the icon is drawn from the data it claims to show", () => {
    // scripts/make_icons.py renders the Blue Marble snapshot in data/icon/
    // through the blauewelt accent-blue ramp. If the icon were hand-drawn art
    // this test would be meaningless; it is here so the icon stays a render
    // of a real NASA raster, reproducible from the committed snapshot.
    const src = fs.readFileSync(path.join(ROOT, "scripts", "make_icons.py"), "utf8");
    expect(src).toContain("base.png");
    // the snapshot exists, is a PNG, and is equirectangular (2:1) — the
    // orthographic un-projection in the generator depends on that
    const png = fs.readFileSync(path.join(ROOT, "data", "icon", "base.png"));
    expect(png.subarray(1, 4).toString()).toBe("PNG");
    const w = png.readUInt32BE(16), h = png.readUInt32BE(20);
    expect(w).toBe(2 * h);
    // the ramp's bright end is the app's accent (#4493f8 = 68,147,248) — the
    // icon must stay in the UI's own blue, not a near-miss of it
    const css = fs.readFileSync(path.join(ROOT, "src", "style.css"), "utf8");
    expect(css).toContain("#4493f8");
    expect(src.replace(/[()]/g, "")).toContain("68, 147, 248");
    // and the snapshot has a maintained writer in the data pipeline
    const refresh = fs.readFileSync(path.join(ROOT, "scripts", "refresh_data.py"), "utf8");
    expect(refresh).toContain("def icon_sources");
  });
});

test.describe("tides.json + tide_constituents.json (EOT20)", () => {
  test("tidal-range grid: schema, physical range, shelf vs open ocean", () => {
    const g = read("tides.json");
    for (const f of ["id", "title", "units", "ramp", "vmin", "vmax", "west",
                     "south", "east", "north", "nx", "ny", "values", "period"]) {
      expect(g[f], `tides.json missing ${f}`).not.toBeUndefined();
    }
    expect(g.units).toBe("m");
    expect(g.ramp).toBe("speed");
    expect(g.nx).toBe(360); expect(g.ny).toBe(180);
    expect(g.period).toBe("1992-2019");
    expect(g.values.length).toBe(g.nx * g.ny);
    // reduce, then assert — never expect() per data point (house rule)
    const finite = g.values.filter((v) => v != null);
    const max = Math.max(...finite), min = Math.min(...finite);
    expect(finite.length).toBeGreaterThan(40000);     // ocean coverage
    expect(min).toBeGreaterThanOrEqual(0);
    expect(max).toBeGreaterThan(8);                   // Fundy/Severn/Ungava exist
    expect(max).toBeLessThan(16);                     // and estuary artifacts don't
    const cell = (lon, lat) => g.values[(Math.floor(lat) + 90) * 360 + (Math.floor(lon) + 180)];
    // macrotidal Bay of Fundy vs a quiet subtropical gyre cell
    expect(cell(-65.5, 45.5)).toBeGreaterThan(5);
    expect(cell(-140.5, 30.5)).toBeLessThan(2);
  });

  test("constituents: the five main harmonics reconstruct a real tide", () => {
    const c = read("tide_constituents.json");
    const SPEED = { M2: 28.9841042, S2: 30.0, N2: 28.4397295, K1: 15.0410686, O1: 13.9430356 };
    expect(c.constituents.map((k) => k.id).sort()).toEqual(["K1", "M2", "N2", "O1", "S2"]);
    for (const k of c.constituents) {
      expect(Math.abs(k.speed - SPEED[k.id])).toBeLessThan(1e-4);
      expect(k.V0).toBeGreaterThanOrEqual(0);
      expect(k.V0).toBeLessThan(360);
      expect(k.amp.length).toBe(c.nx * c.ny);
      expect(k.phase.length).toBe(c.nx * c.ny);
      const amps = k.amp.filter((v) => v != null);
      expect(amps.length).toBeGreaterThan(45000);
      expect(Math.max(...amps)).toBeLessThan(800);    // the estuary-artifact cap held
    }
    expect(Number.isFinite(Date.parse(c.epoch))).toBe(true);
    // Ocean fraction (2026-08-07): per-cell share of 0.125° source subcells,
    // used by the renderer as alpha so coastal cells feather instead of
    // stamping opaque 1° squares over land (the British-Isles artifact).
    expect(c.frac.length).toBe(c.nx * c.ny);
    expect(Math.min(...c.frac)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...c.frac)).toBeLessThanOrEqual(1);
    const cellOf = (lon, lat) => (Math.floor(lat) + 90) * 360 + (Math.floor(lon) + 180);
    expect(c.frac[cellOf(-1.5, 52.5)]).toBe(0);     // Birmingham: inland
    expect(c.frac[cellOf(-20.5, 45.5)]).toBe(1);    // open Atlantic
    // Reconstruct h(t) per the file's own note and check the physics: a
    // macrotidal cell swings metres over a month, a gyre cell barely moves.
    const h = (i, hours) => c.constituents.reduce((sum, k) => {
      const a = k.amp[i], g = k.phase[i];
      if (a == null) return NaN;
      return sum + a * Math.cos((k.speed * hours + k.V0 - g) * Math.PI / 180);
    }, 0);
    const idx = (lon, lat) => (Math.floor(lat) + 90) * 360 + (Math.floor(lon) + 180);
    const swing = (i) => {
      let lo = Infinity, hi = -Infinity;
      for (let t = 0; t <= 720; t++) {                // hourly, 30 days
        const v = h(i, t); lo = Math.min(lo, v); hi = Math.max(hi, v);
      }
      return hi - lo;
    };
    expect(swing(idx(-65.5, 45.5))).toBeGreaterThan(500);   // Fundy: > 5 m
    expect(swing(idx(-140.5, 30.5))).toBeLessThan(150);     // gyre: < 1.5 m
  });
});

test.describe("oisst_monthly.json (SST from the numbers, to 36°)", () => {
  test("month-keyed contract; the hottest seas actually show above 32", () => {
    const g = read("oisst_monthly.json");
    for (const f of ["latest", "monthsAvailable", "yearDir", "months", "values",
                     "vmin", "vmax", "nx", "ny"]) {
      expect(g[f], `missing ${f}`).not.toBeUndefined();
    }
    expect(g.vmax).toBe(36);
    expect(g.monthsAvailable[0]).toBe("1981-09");
    expect(g.monthsAvailable.length).toBeGreaterThan(500);
    expect(g.values.length).toBe(360 * 180);
    // the reason the layer exists: an August Persian Gulf cell reads >32.5
    const aug = g.monthsAvailable.filter((s) => s.endsWith("-08")).pop();
    const yr = read(`oisst_y/${aug.slice(0, 4)}.json`).months[aug];
    const cell = (lon, lat) => yr[(Math.floor(lat) + 90) * 360 + (Math.floor(lon) + 180)];
    expect(cell(52.5, 26.5)).toBeGreaterThan(32.5);   // Persian Gulf, August
    // reduce-then-assert (house rule): global August max in a sane band
    const finite = yr.filter((v) => v != null);
    expect(Math.max(...finite)).toBeGreaterThan(33);
    expect(Math.max(...finite)).toBeLessThan(38);
    expect(Math.min(...finite)).toBeGreaterThan(-3);
  });
});

/* The dependency cone (E-069). Two tests, and they answer different questions.
 *
 * The first reads the file: `data/cone_geometry.json` is a BUILD ARTEFACT of
 * ml/cone.py, and its shape is what the Cones tab draws from — the token
 * budget, the reach tables, the fact that stage 2's annulus is empty inside
 * the codec's own window. (`tests/test_cone_geometry_export.py` is the other
 * half: it proves the committed file IS a fresh export.)
 *
 * The second is a CERTIFICATION, in the sense the JAX port's gate tests use
 * the word. The tab ports the sunflower to JavaScript because the offsets
 * depend on the anchor's latitude and baking 281 rows would be 281 copies of
 * one formula. A port is only a port if it reproduces the original's own
 * output, so the exported reference sets — every family at five latitudes,
 * every outer lag at three — are replayed through the browser's functions and
 * compared whole. One deep-equal per set, never an expect per dot. */
test.describe("cone_geometry.json (E-069, exported from ml/cone.py)", () => {
  const G = read("cone_geometry.json");

  test("the file carries the cone's tables, budget and window", () => {
    for (const f of ["constants", "families", "channel_family", "reach_km",
                     "slots", "counts", "reference", "window"]) {
      expect(G[f], `missing ${f}`).not.toBeUndefined();
    }
    // the token budget, spelled out: 4 wind × 8 + 4 surface-B × 80
    // + 32 depth × 6 + 2 C × 81 = 706 dots, plus one patch token per channel
    const c = G.counts;
    expect(c.inner_dots_A).toBe(8);
    expect(c.inner_dots_B).toBe(80);
    expect(c.inner_dots_rg).toBe(6);
    expect(c.inner_dots_C).toBe(81);
    expect(c.dot_tokens).toBe(706);
    expect(4 * 8 + 4 * 80 + 32 * 6 + 2 * 81).toBe(c.dot_tokens);
    expect(c.patch_tokens).toBe(42);
    expect(c.total_tokens).toBe(748);
    expect(Object.keys(G.channel_family).length).toBe(42);

    // reach: A dies with its 10-day memory, B grows one-and-a-bit cells per
    // pentad, C is L-shaped and DROPS at lag 2 (the stirring going out of
    // memory, by design — so this is asserted, not tolerated).
    expect(G.reach_km.inner.A).toEqual([500, 500, 0, 0, 0, 0, 0]);
    expect(G.reach_km.inner.B.map((r) => Math.round(r * 10) / 10))
      .toEqual([129.6, 259.2, 388.8, 518.4, 648, 777.6, 907.2]);
    expect(G.reach_km.inner.C[1]).toBe(500);
    expect(G.reach_km.inner.C[2]).toBeLessThan(G.reach_km.inner.C[1]);
    expect(G.reach_km.inner.B.every((r, i, a) => i === 0 || r > a[i - 1])).toBe(true);
    const out = G.reach_km.outer;
    expect(out.length).toBe(144);
    expect(out.every((r, i, a) => i === 0 || r >= a[i - 1])).toBe(true);
    expect(out[33]).toBeCloseTo(4406.4, 6);          // last below the cap
    expect(out[34]).toBe(G.constants.OUTER_CAP_KM);  // the cap binds from k=34
    expect(out[143]).toBe(G.constants.OUTER_CAP_KM);

    // slots follow the DISC's area: 6 bearings at 130 km, 24 at 907 km
    expect(G.slots.B).toEqual([6, 6, 6, 8, 12, 18, 24]);
    expect(G.slots.C).toEqual([7, 7, 6, 8, 12, 18, 24]);
    // and the rule's own three numbers are EXPORTED, not retyped in the page:
    // the Cones tab lets a reader move them, and its reset has to land back on
    // ml/cone.py::slots rather than on a JS literal that has drifted from it
    expect([G.constants.SLOT_MAX, G.constants.SLOT_MIN, G.constants.SLOT_REF_KM])
      .toEqual([24, 6, 900]);

    // the tensor window, point-aligned at 0.25°
    const w = G.window;
    expect([w.lat0, w.lat1, w.lon0, w.lon1, w.dlat]).toEqual([0, 70, -100, 20, 0.25]);
    expect([w.ny, w.nx]).toEqual([281, 481]);
  });

  test("the outer spiral is empty inside the codec's own window", () => {
    // For k ≤ 6 the inner and outer reaches are the same number by
    // construction, so stage 2's annulus has nothing in it — the design, not a
    // degenerate case. The reference sets start at the first non-empty lag.
    for (const [lat, ks] of Object.entries(G.reference.outer)) {
      for (const [k, pts] of Object.entries(ks)) {
        expect(Number(k), `reference outer lag at ${lat}`).toBeGreaterThan(6);
        expect(pts.length, `outer spiral size at lat ${lat} k ${k}`).toBe(24);
      }
    }
    // and every dot count is latitude-independent: latitude converts km to
    // cells, it never changes how many cells the cone reads
    const per = { A: new Set(), B: new Set(), C: new Set(), rg: new Set() };
    for (const fams of Object.values(G.reference.inner)) {
      for (const f of Object.keys(per)) per[f].add(fams[f].length);
    }
    expect([...per.A]).toEqual([8]);
    expect([...per.B]).toEqual([80]);
    expect([...per.C]).toEqual([81]);
    expect([...per.rg]).toEqual([6]);
  });

  /* The port lives in the browser, so this one needs a page. Only the Cesium
   * CDN has to be mirrored (app.js will not define window.__earth without it);
   * no tiles are fetched, because no layer is switched on. */
  test.describe("the JS port certified against ml/cone.py's own output", () => {
    const CESIUM = "https://cdnjs.cloudflare.com/ajax/libs/cesium/1.133.1";
    test.beforeEach(async ({ page, baseURL }) => {
      if (process.env.MIRROR) {
        await page.route(/https:\/\/cdnjs\.cloudflare\.com\/.*/, async (route) => {
          try {
            const url = route.request().url()
              .replace(CESIUM, `${baseURL}/_vendor/cesium`)
              .replace("widgets.min.css", "widgets.css");
            await route.fulfill({ response: await page.request.get(url) });
          } catch { await route.abort().catch(() => {}); }
        });
        await page.route(/https:\/\/gibs\.earthdata\.nasa\.gov\/.*/, async (route) => {
          try {
            const url = route.request().url()
              .replace("https://gibs.earthdata.nasa.gov", "http://localhost:8081");
            await route.fulfill({ response: await page.request.get(url) });
          } catch { await route.abort().catch(() => {}); }
        });
      }
      await page.goto("/");
      await page.waitForFunction(() => window.__earth?.viewer, null, { timeout: 30000 });
    });

    test("every exported reference dot set is reproduced exactly", async ({ page }) => {
      const got = await page.evaluate(async () => {
        await window.__earth.loadCones();
        const g = window.__earth.coneGeometry;
        const inner = {}, outer = {};
        for (const [lat, fams] of Object.entries(g.reference.inner)) {
          inner[lat] = {};
          for (const f of Object.keys(fams)) {
            inner[lat][f] = window.__earth.coneInnerDots(Number(lat), f);
          }
        }
        for (const [lat, ks] of Object.entries(g.reference.outer)) {
          outer[lat] = {};
          for (const k of Object.keys(ks)) {
            outer[lat][k] = window.__earth.coneOuterSpiral(Number(lat), Number(k));
          }
        }
        // and the two tables the port derives rather than reads
        const reach = {}, slots = [];
        for (const f of ["A", "B", "C"]) {
          reach[f] = [];
          for (let l = 0; l <= 6; l++) reach[f].push(window.__earth.coneReachKm(f, l));
        }
        for (let l = 0; l <= 6; l++) slots.push(window.__earth.coneSlots(window.__earth.coneReachKm("B", l)));
        return { inner, outer, reach, slots,
                 outer143: window.__earth.coneOuterReachKm(143),
                 emptyInside: [0, 3, 6].map((k) => window.__earth.coneOuterSpiral(40, k).length) };
      });
      // whole sets, compared whole — one assertion per set, never one per dot
      expect(got.inner).toEqual(G.reference.inner);
      expect(got.outer).toEqual(G.reference.outer);
      expect(got.reach).toEqual(G.reach_km.inner);
      expect(got.slots).toEqual(G.slots.B);
      expect(got.outer143).toBe(G.reach_km.outer[143]);
      expect(got.emptyInside).toEqual([0, 0, 0]);
    });
  });
});

/* The cone SAMPLES (E-069 data mode). `data/cone_geometry.json` above says
 * which cells the model reads; these two files say what is IN them.
 *
 * The index is committed and tiny; the samples themselves are megabytes and
 * live on the Hugging Face Hub, so what is checkable here is that the index
 * describes them honestly and that the in-repo FIXTURE — the copy the browser
 * tests run against with no network — carries the real schema rather than a
 * convenient one. A fixture with its own shape would let an app test pass
 * against a format the deployed page never sees.
 *
 * The Python side is the other half: `tests/test_export_cone_sample.py`
 * asserts the exporter's `valid`/`obs` are bit-identical to a direct
 * `ConeSampler.sample` call, and that its anomaly is bit-identical to
 * `trainprobe.anomaly_transform`. */
test.describe("cone_samples.json + the fixture (E-069 data mode)", () => {
  const idx = read("cone_samples.json");
  const fx = read("cone_samples/fixture.json");
  const G = read("cone_geometry.json");

  test("the index names the tensor, the dates and every anchor's URL", () => {
    // the exact tensor the numbers came out of — a sample from another build
    // would be a different experiment wearing this one's label
    expect(idx.tensor.name).toBe("family4_na025_pentad_r3_fa460837fa");
    expect(idx.tensor.sha256).toBe(
      "fa460837fa172825ee76c8fc6fc4da75fa7b96d64519a2e2186f5c306cf03ea9");
    expect(idx.tensor.shape).toEqual([3142, 281, 481, 42]);
    expect(idx.produced_by).toContain("ConeSampler.sample");
    expect(idx.produced_by).toContain("outer_spiral");
    expect(idx.exporter).toBe("ml/export_cone_sample.py");
    expect(idx.exporter_commit).toMatch(/^[0-9a-f]{40}$/);
    expect(idx.L_in).toBe(G.constants.L_IN);
    expect(idx.holdout_years).toEqual([2009, 2017, 2023]);
    expect(idx.channels).toHaveLength(42);
    expect(idx.scoreable_channels).toHaveLength(8);
    // 24 consecutive pentads: every gap is exactly one bin, i.e. five days
    expect(idx.dates).toHaveLength(24);
    expect(idx.bins).toHaveLength(24);
    for (let i = 1; i < idx.bins.length; i++) {
      expect(idx.bins[i] - idx.bins[i - 1]).toBe(1);
    }
    // the outer stencil starts where cone.py says it can: lag 7
    expect(idx.outer_lags[0]).toBe(7);
    expect(idx.outer.empty_below).toBe(G.constants.L_IN + 1);
    expect(Math.max(...idx.outer_lags)).toBeLessThanOrEqual(G.constants.K_OUTER - 1);

    // the anchors, and the CORS measurement that lets a browser read them
    expect(idx.anchors.length).toBeGreaterThanOrEqual(5);
    const ids = idx.anchors.map((a) => a.id);
    for (const want of ["gulf_stream", "rapid", "labrador", "equator"]) {
      expect(ids).toContain(want);
    }
    for (const a of idx.anchors) {
      expect(a.url).toMatch(
        /^https:\/\/huggingface\.co\/datasets\/chfrank\/earth-tensors\/resolve\/main\/cone_samples\/.+\.json$/);
      expect(a.sha256).toMatch(/^[0-9a-f]{64}$/);
      expect(a.bytes).toBeGreaterThan(1e5);
      expect(a.bytes).toBeLessThan(6.5e6);   // the per-file budget
      expect(a.row).toBeGreaterThanOrEqual(0);
      expect(a.col).toBeGreaterThanOrEqual(0);
    }
    // The Hub ECHOES our origin rather than answering `*`, which is what makes
    // it readable from the deployed page (CLAUDE.md §3, second exception). The
    // probe is a ranged read, so 206 is the healthy answer and 200 is the
    // whole-file variant — both are the same CORS fact.
    expect(idx.cors_measured.access_control_allow_origin)
      .toBe("https://blauewelt.github.io");
    expect([200, 206]).toContain(idx.cors_measured.status);
    expect(idx.fixture).toBe("data/cone_samples/fixture.json");
  });

  test("the fixture is the real schema, trimmed", () => {
    expect(Object.keys(fx).sort()).toEqual(
      ["future", "inner", "meta", "outer", "patch"]);
    expect(fx.meta.produced_by).toBe(idx.produced_by);
    expect(fx.meta.tensor.sha256).toBe(idx.tensor.sha256);
    expect(fx.meta.channels).toEqual(idx.channels);
    expect(fx.meta.fixture).toContain("trimmed copy");
    expect(fx.meta.dates.length).toBeLessThanOrEqual(3);
    expect(fx.meta.dates).toEqual(idx.dates.slice(0, fx.meta.dates.length));
    expect(fx.outer.lags[0]).toBe(7);
    expect(fx.outer.lags).toHaveLength(3);

    // every declared shape matches the array behind it — one assertion per
    // array, never one per value
    const nT = fx.meta.dates.length, nC = fx.meta.channels.length;
    const nD = fx.inner.n_dots;
    expect(fx.inner.shape).toEqual([nT, nD]);
    expect(nD).toBe(G.counts.dot_tokens);          // 706, ml/cone.py::budget
    expect(fx.inner.raw).toHaveLength(nT);
    expect(fx.inner.raw.every((r) => r.length === nD)).toBe(true);
    expect(fx.inner.obs).toHaveLength(nT * nD);
    expect(fx.inner.valid).toHaveLength(nT * nD);
    expect(/^[01]+$/.test(fx.inner.obs)).toBe(true);
    expect(fx.patch.shape).toEqual([nT, nC, 9]);
    expect(fx.patch.raw).toHaveLength(nT * nC * 9);
    expect(fx.future.raw).toHaveLength(nT * nC * fx.future.lags.length);
    const [oT, oK, oD, oC] = fx.outer.shape;
    expect([oT, oK, oD, oC]).toEqual(
      [nT, 3, G.constants.OUTER_N_PTS, fx.outer.channels.length]);
    expect(fx.outer.raw).toHaveLength(oT * oK * oD * oC);
    expect(fx.outer.obs).toHaveLength(oT * oK * oD * oC);
    expect(fx.outer.valid).toHaveLength(oK * oD);

    // lag 0 is the PATCH, so no dot carries it; dots stop at L_in
    expect(Math.min(...fx.inner.lag)).toBe(1);
    expect(Math.max(...fx.inner.lag)).toBe(G.constants.L_IN);

    // there are real numbers in here, and they are in two spaces — the raw
    // measurement and the z-scored anomaly the codec is actually given
    const finite = (a) => a.filter((v) => typeof v === "number");
    const raw0 = finite(fx.inner.raw[0]), anom0 = finite(fx.inner.anom[0]);
    expect(raw0.length).toBeGreaterThan(100);
    expect(anom0.length).toBe(raw0.length);
    expect(Math.max(...anom0.map(Math.abs))).toBeLessThan(30);
  });
});

/* The cone samples exported from FAMILY 7 — the tensor that covers the whole
 * globe at 0.25° rather than only the North Atlantic window. Same exporter,
 * same production sampler, same file schema; three things differ and each one
 * is what this block is about:
 *
 *   · a `grid` block. Family 4's (row, col) are indices into a window with
 *     EDGES, so the page maps them through `data/cone_geometry.json`. Family
 *     7's are indices into a grid that CLOSES at the dateline, and the file
 *     says so itself — which is what lets one drawing routine serve both.
 *   · three CHANNEL GROUPS. 54 channels on three different grids (0.25° ocean,
 *     1° atmosphere and land, 1° Argo depth column), so the (mean, sd) the
 *     builder z-scored with is keyed by group and indexed within it.
 *   · twelve anchors instead of five, and they are global: the Antarctic
 *     Circumpolar Current, the Sahara, the dateline itself.
 *
 * The full files are on the Hub; the in-repo fixture is one anchor cut down to
 * two dates and four channels — one from each group plus a second ocean one,
 * which is the whole schema in the smallest honest file. */
test.describe("cone_samples_f7.json + the fixture (family 7, the global anchors)", () => {
  const idx = read("cone_samples_f7.json");
  const fx = read("cone_samples_f7/fixture.json");
  const G = read("cone_geometry.json");
  const GRID = { lat0: -90, lon0: -180, nx: 1440, ny: 721, step: 0.25, wrap: true };

  test("the index names the global tensor, its groups and every anchor's URL", () => {
    expect(idx.recipe).toBe("f7l0");
    expect(idx.tensor.name).toBe("family7_global025_pentad_l0");
    expect(idx.tensor.recipe).toBe("f7l0");
    expect(idx.tensor.window).toBe("global025");
    expect(idx.tensor.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(idx.tensor.shape).toEqual([3142, 721, 1440, 54]);
    expect(idx.prefix).toBe("cone_samples_f7");
    expect(idx.produced_by).toContain("ConeSampler.sample");
    expect(idx.produced_by).toContain("outer_spiral");
    expect(idx.exporter).toBe("ml/export_cone_sample.py");
    expect(idx.exporter_commit).toMatch(/^[0-9a-f]{40}$/);
    expect(idx.L_in).toBe(G.constants.L_IN);
    expect(idx.holdout_years).toEqual([2009, 2017, 2023]);

    // THE GRID. This is the block the page branches on, and it is the whole
    // difference between a window with edges and a globe that closes.
    expect(idx.grid).toEqual(GRID);
    expect(idx.grid.wrap).toBe(true);

    // THREE GROUPS, in the order the channel list is laid out in
    expect(idx.groups.names).toEqual(["g025", "g100", "rg100"]);
    expect(idx.channels).toHaveLength(54);
    expect(idx.groups.channels.g025).toHaveLength(7);
    expect(idx.groups.channels.g100).toHaveLength(15);
    expect(idx.groups.channels.rg100).toHaveLength(32);
    // the three lists together ARE the channel list, in order — the page
    // indexes a channel by its position in `channels` and looks its (mean, sd)
    // up by its position within its GROUP, so the two must not drift
    expect([...idx.groups.channels.g025, ...idx.groups.channels.g100,
            ...idx.groups.channels.rg100]).toEqual(idx.channels);
    for (const c of idx.channels) {
      expect(idx.groups.names).toContain(idx.channel_group[c]);
    }
    expect(idx.scoreable_channels).toHaveLength(8);

    // 24 consecutive pentads, exactly as the North Atlantic set
    expect(idx.dates).toHaveLength(24);
    expect(idx.bins).toHaveLength(24);
    for (let i = 1; i < idx.bins.length; i++) {
      expect(idx.bins[i] - idx.bins[i - 1]).toBe(1);
    }
    expect(idx.outer_lags[0]).toBe(7);
    expect(idx.outer.empty_below).toBe(G.constants.L_IN + 1);
    expect(Math.max(...idx.outer_lags)).toBeLessThanOrEqual(G.constants.K_OUTER - 1);

    // TWELVE anchors, and they are global rather than one ocean's
    expect(idx.anchors).toHaveLength(12);
    const ids = idx.anchors.map((a) => a.id);
    for (const want of ["acc", "antarctica_ice", "dateline", "sahara",
                        "nino34", "kuroshio", "rapid", "greenland"]) {
      expect(ids).toContain(want);
    }
    for (const a of idx.anchors) {
      expect(a.url).toBe(
        `https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/` +
        `cone_samples_f7/${a.id}.json`);
      expect(a.url).toMatch(
        /^https:\/\/huggingface\.co\/datasets\/chfrank\/earth-tensors\/resolve\/main\/cone_samples_f7\/[a-z0-9_]+\.json$/);
      expect(a.file).toBe(`cone_samples_f7/${a.id}.json`);
      expect(a.sha256).toMatch(/^[0-9a-f]{64}$/);
      expect(a.bytes).toBeGreaterThan(1e5);
      expect(a.bytes).toBeLessThan(6.5e6);         // the per-file budget
      // every anchor is a cell of the GLOBAL grid, and its (row, col) and its
      // (lat, lon) are the same place said twice
      expect(a.row).toBeGreaterThanOrEqual(0);
      expect(a.row).toBeLessThan(GRID.ny);
      expect(a.col).toBeGreaterThanOrEqual(0);
      expect(a.col).toBeLessThan(GRID.nx);
      expect(GRID.lat0 + a.row * GRID.step).toBeCloseTo(a.lat, 6);
      expect(GRID.lon0 + a.col * GRID.step).toBeCloseTo(a.lon, 6);
    }
    // the dateline anchor is the last column — the one that has no eastern
    // neighbour on a window and has one here
    expect(idx.anchors.find((a) => a.id === "dateline").col).toBe(GRID.nx - 1);

    expect(idx.cors_measured.access_control_allow_origin)
      .toBe("https://blauewelt.github.io");
    expect(idx.cors_measured.status).toBe(206);
    expect(idx.fixture).toBe("data/cone_samples_f7/fixture.json");
  });

  test("the fixture is the real schema, trimmed in dates AND channels", () => {
    expect(Object.keys(fx).sort()).toEqual(
      ["future", "inner", "meta", "outer", "patch"]);
    expect(fx.meta.produced_by).toBe(idx.produced_by);
    expect(fx.meta.recipe).toBe("f7l0");
    expect(fx.meta.tensor.sha256).toBe(idx.tensor.sha256);
    expect(fx.meta.grid).toEqual(idx.grid);
    expect(fx.meta.fixture).toContain("trimmed copy");
    expect(fx.meta.dates.length).toBeLessThanOrEqual(3);
    expect(fx.meta.dates).toEqual(idx.dates.slice(0, fx.meta.dates.length));
    expect(fx.outer.lags[0]).toBe(7);
    expect(fx.outer.lags).toHaveLength(3);

    // the five kept channels are a subset of the real list: one per group, a
    // second ocean one, and one channel of every CONE FAMILY the page can
    // select — B (cur_speed), C (sst, skt), A (tau_x) and rg (rg_t10), which
    // is what lets the browser test check that the family select follows the
    // channel in data mode
    expect(fx.meta.channels).toEqual(
      ["cur_speed", "sst", "tau_x", "skt", "rg_t10"]);
    expect([...new Set(fx.meta.channels.map((c) => fx.meta.channel_family[c]))]
      .sort()).toEqual(["A", "B", "C"]);
    for (const c of fx.meta.channels) expect(idx.channels).toContain(c);
    expect(fx.meta.groups.names).toEqual(["g025", "g100", "rg100"]);
    expect([...fx.meta.groups.channels.g025, ...fx.meta.groups.channels.g100,
            ...fx.meta.groups.channels.rg100]).toEqual(fx.meta.channels);

    // the (mean, sd) table is keyed by GROUP and indexed within it — the page
    // puts the unit back on a stored value with exactly this lookup, so a
    // group whose norm list is a different length from its channel list would
    // print the wrong number in the right unit
    const norm = fx.meta.value_space.tensor_norm;
    expect(Array.isArray(norm)).toBe(false);
    for (const g of fx.meta.groups.names) {
      expect(norm[g]).toHaveLength(fx.meta.groups.channels[g].length);
      for (const row of norm[g]) {
        expect(row).toHaveLength(2);
        expect(Number.isFinite(row[0]) && Number.isFinite(row[1])).toBe(true);
      }
    }
    // and every kept channel still has its group, its family and its unit
    for (const c of fx.meta.channels) {
      expect(fx.meta.groups.names).toContain(fx.meta.channel_group[c]);
      expect(fx.meta.channel_family[c]).toBeTruthy();
      expect(typeof fx.meta.units[c]).toBe("string");
      // and the unit is the CHANNEL'S own. The exporter's fallback used to be
      // the Argo depth column's composite unit, applied to everything it did
      // not recognise, so the file said air and skin temperature were measured
      // in dbar-levels — the one thing a unit must never do is describe a
      // different quantity.
      expect(fx.meta.units[c], `${c} unit`).not.toContain("dbar-level");
    }
    expect(fx.meta.units.skt).toBe("°C");
    expect(fx.meta.units.tau_x).toBe("N/m²");
    expect(fx.meta.units.rg_t10).toBe("°C");

    // the anchor's own statics — which surface it stands on and how high
    expect(fx.meta.anchor.id).toBe("dateline");
    expect(fx.meta.anchor.col).toBe(idx.grid.nx - 1);
    expect([0, 1, 2, 3]).toContain(fx.meta.sphere_at_anchor);
    expect(Number.isFinite(fx.meta.elev_at_anchor)).toBe(true);

    // every declared shape matches the array behind it
    const nT = fx.meta.dates.length, nC = fx.meta.channels.length;
    const nD = fx.inner.n_dots;
    expect(fx.inner.shape).toEqual([nT, nD]);
    expect(fx.inner.raw).toHaveLength(nT);
    expect(fx.inner.raw.every((r) => r.length === nD)).toBe(true);
    expect(fx.inner.obs).toHaveLength(nT * nD);
    expect(fx.inner.valid).toHaveLength(nT * nD);
    expect(/^[01]+$/.test(fx.inner.obs)).toBe(true);
    expect([...new Set(fx.inner.chan)].sort((a, b) => a - b))
      .toEqual([0, 1, 2, 3, 4]);
    expect(fx.patch.shape).toEqual([nT, nC, 9]);
    expect(fx.patch.raw).toHaveLength(nT * nC * 9);
    expect(fx.future.raw).toHaveLength(nT * nC * fx.future.lags.length);
    const [oT, oK, oD, oC] = fx.outer.shape;
    expect([oT, oK, oD, oC]).toEqual(
      [nT, 3, G.constants.OUTER_N_PTS, fx.outer.channels.length]);
    expect(fx.outer.raw).toHaveLength(oT * oK * oD * oC);
    expect(fx.outer.obs).toHaveLength(oT * oK * oD * oC);
    expect(fx.outer.valid).toHaveLength(oK * oD);
    // stage 2's channel list is a subset of the tensor's, and `chan_index`
    // points back into it — the read-out follows that pointer
    for (let i = 0; i < fx.outer.channels.length; i++) {
      expect(fx.meta.channels[fx.outer.chan_index[i]]).toBe(fx.outer.channels[i]);
    }

    expect(Math.min(...fx.inner.lag)).toBe(1);
    expect(Math.max(...fx.inner.lag)).toBe(G.constants.L_IN);

    // THE DATELINE, which is the point of the whole family: the anchor sits in
    // the last column, so its cone reaches across the join and the columns
    // come back round rather than falling off an edge
    expect(Math.min(...fx.inner.col)).toBe(0);
    expect(Math.max(...fx.inner.col)).toBe(idx.grid.nx - 1);
    expect(fx.inner.lon.some((v) => v > 170)).toBe(true);
    expect(fx.inner.lon.some((v) => v < -170)).toBe(true);
    expect(fx.inner.lon.every((v) => v >= -180 && v < 180)).toBe(true);
    // nothing is off the grid: a global cone has no invalid cells
    expect(/^1+$/.test(fx.inner.valid)).toBe(true);

    // there are real numbers in here, and they are in two spaces — the raw
    // measurement and the z-scored anomaly the codec is actually given. The
    // ANOMALY is the thing live mode cannot offer and this mode can.
    const finite = (a) => a.filter((v) => typeof v === "number");
    const raw0 = finite(fx.inner.raw[0]), anom0 = finite(fx.inner.anom[0]);
    expect(raw0.length).toBeGreaterThan(50);
    expect(anom0.length).toBe(raw0.length);
    expect(Math.max(...anom0.map(Math.abs))).toBeLessThan(30);
  });
});

/* ============================ family7_index.json + the in-repo fixture ======
 *
 * FAMILY 7 is the first input tensor covering the whole globe rather than the
 * North Atlantic window: 721 × 1440 at 0.25°, one value per channel per
 * five-day bin from 1982 to 2024. The globe's "Global tensor" layer paints one
 * channel of one bin by a single HTTP range read of a `.npy` on the Hugging
 * Face Hub, and everything it needs to compute that read — header length,
 * shape, dtype, slab size, grid geometry, channel vocabulary, the (mean, sd)
 * the builder z-scored with — lives in an index written by
 * `ml/publish_family7_index.py`.
 *
 * The real index is absent until the build job lands, so what is pinned here
 * is `data/family7/fixture/` — the same schema over the T=5 tensor
 * `ml/build_family7.py --smoke` writes, decimated to a 5°/10° grid so it can
 * live in git. Its `.npy` HEADERS are parsed here from the real bytes, which
 * is the check that matters: if the index's `header_len` and `slab_bytes` do
 * not agree with the file, the browser reads the wrong bytes and paints a
 * plausible-looking wrong map. */
test.describe("family7 index + fixture (the global tensor's range-read contract)", () => {
  const F7 = path.join(DATA, "family7", "fixture");
  const idx = JSON.parse(fs.readFileSync(path.join(F7, "family7_index.json"), "utf8"));

  // The .npy header, parsed the same way ml/publish_family7_index.py parses it.
  function npyHeader(file) {
    const buf = fs.readFileSync(file);
    expect(buf.slice(0, 6).toString("latin1")).toBe("\x93NUMPY");
    const major = buf[6];
    const n = major === 1 ? buf.readUInt16LE(8) : buf.readUInt32LE(8);
    const off = major === 1 ? 10 : 12;
    const txt = buf.slice(off, off + n).toString("latin1");
    const shape = /'shape':\s*\(([^)]*)\)/.exec(txt)[1]
      .split(",").map((x) => x.trim()).filter(Boolean).map(Number);
    const descr = /'descr':\s*'([^']+)'/.exec(txt)[1];
    const fortran = /'fortran_order':\s*(True|False)/.exec(txt)[1] === "True";
    return { headerLen: off + n, shape, descr, fortran, bytes: buf.length };
  }

  test("the index says what a browser needs to address one pentad", () => {
    expect(idx._source).toContain("publish_family7_index.py");
    expect(idx.recipe).toBe("f7l0");
    expect(idx.epoch).toBe("1982-01-01");
    expect(idx.pentad_days).toBe(5);
    expect(idx.bin_last).toBeGreaterThanOrEqual(idx.bin_first);
    expect(idx.n_bins).toBe(idx.bin_last - idx.bin_first + 1);
    expect(idx.base).toMatch(
      /^https:\/\/huggingface\.co\/datasets\/chfrank\/earth-tensors\/resolve\/main\/tensors\//);
    expect(idx.plan).toContain("E070_family7_build.md");
    expect(Object.keys(idx.groups).sort()).toEqual(["g025", "g100", "rg100"]);
    expect(Object.keys(idx.statics).sort()).toEqual(["elev", "sphere"]);
  });

  test("every group's header, shape and slab size are the file's own", () => {
    for (const [name, g] of Object.entries(idx.groups)) {
      const h = npyHeader(path.join(F7, g.file));
      expect(h.headerLen, `${name} header_len`).toBe(g.header_len);
      expect(h.shape, `${name} shape`).toEqual(g.shape);
      expect(h.descr, `${name} dtype`).toBe(g.dtype);
      // C order is not a detail: the whole range read assumes bin-major.
      expect(h.fortran, `${name} fortran_order`).toBe(false);
      expect(g.fortran_order).toBe(false);
      expect(h.bytes, `${name} bytes`).toBe(g.bytes);
      expect(g.dtype).toBe("<f2");           // float16, 2 bytes a value
      expect(g.itemsize).toBe(2);
      // offset = header_len + row·slab_bytes, length = slab_bytes — and the
      // arithmetic has to close on the file size exactly
      const cells = g.shape.slice(1).reduce((a, b) => a * b, 1);
      expect(g.slab_bytes).toBe(cells * g.itemsize);
      expect(g.header_len + g.shape[0] * g.slab_bytes).toBe(g.bytes);
      // the channel vocabulary matches the last axis, with a label and a unit
      expect(g.chans).toHaveLength(g.shape[3]);
      expect(g.norm).toHaveLength(g.shape[3]);
      for (const c of g.chans) {
        expect(g.labels[c], `${name}/${c} label`).toBeTruthy();
        expect(g.units[c], `${name}/${c} unit`).not.toBe(undefined);
        expect(["seq", "div"]).toContain(g.sign[c]);
        expect(g.ramp[c]).toBeTruthy();
      }
      // every sd is positive, or un-z-scoring divides by zero
      expect(g.norm.every(([, sd]) => sd > 0)).toBe(true);
    }
  });

  test("the grids are point-aligned, south-first and closed at the dateline", () => {
    for (const [name, g] of Object.entries(idx.groups)) {
      const gr = g.grid;
      expect(gr.ny, `${name} ny`).toBe(g.shape[1]);
      expect(gr.nx, `${name} nx`).toBe(g.shape[2]);
      expect(gr.lat0, `${name} lat0`).toBe(-90);       // SOUTH-first, like every family
      expect(gr.lon0, `${name} lon0`).toBe(-180);
      expect(gr.south_first).toBe(true);
      expect(gr.wrap).toBe(true);
      // the northmost row is the pole and the eastmost column stops one step
      // short of it — a point grid, not a cell grid
      expect(gr.lat0 + (gr.ny - 1) * gr.step).toBeCloseTo(90, 9);
      expect(gr.lon0 + gr.nx * gr.step).toBeCloseTo(180, 9);
    }
    // the coarse group is a whole multiple of the fine one, which is what
    // makes "every Nth fine point IS a coarse point" true
    const r = idx.groups.g100.grid.step / idx.groups.g025.grid.step;
    expect(Number.isInteger(r)).toBe(true);
    expect(r).toBeGreaterThan(1);
  });

  test("row 0 is the SOUTH pole — asserted on the bytes, not on a comment", () => {
    /* The one orientation mistake that looks entirely plausible: the AMOC eval
     * mask shipped with a `[::-1]` row flip copied from another bake and put
     * the Gulf Stream in the Norwegian Sea. The tensor is south-first and the
     * app's grid convention is row-major from the south, so no flip — and the
     * way to know is to read a known cell. The smoke tensor's `sst` is a
     * smooth synthetic field of latitude, so the southernmost row must be
     * colder than the equatorial one. */
    const g = idx.groups.g025;
    const buf = fs.readFileSync(path.join(F7, g.file));
    const ci = g.chans.indexOf("sst"), nC = g.chans.length;
    const [mean, sd] = g.norm[ci];
    // The smoke tensor's FIRST bin has no source day behind it and is all NaN,
    // which is itself the honest answer — read the third, which does.
    const row0 = 2;
    const at = (row, col) => {
      const off = g.header_len + row0 * g.slab_bytes +
                  ((row * g.grid.nx + col) * nC + ci) * g.itemsize;
      const h = buf.readUInt16LE(off);
      const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      const z = e === 0 ? s * 6.103515625e-5 * (f / 1024)
              : e === 31 ? (f ? NaN : s * Infinity)
              : s * Math.pow(2, e - 15) * (1 + f / 1024);
      return z * sd + mean;
    };
    const eqRow = Math.round((0 - g.grid.lat0) / g.grid.step);
    const southRow = 1;                   // row 0 is the pole itself
    /* `sst` is OISST and therefore OCEAN ONLY — NaN over land, which is the
     * honest answer and not a gap. So the column is CHOSEN rather than typed:
     * the first meridian that is sea at both rows. */
    let col = -1;
    for (let c = 0; c < g.grid.nx && col < 0; c++) {
      if (Number.isFinite(at(southRow, c)) && Number.isFinite(at(eqRow, c))) col = c;
    }
    expect(col, "no meridian is sea at both rows").toBeGreaterThanOrEqual(0);
    const south = at(southRow, col), equator = at(eqRow, col);
    expect(Number.isFinite(south) && Number.isFinite(equator)).toBe(true);
    expect(equator, `row ${eqRow} (equator) vs row ${southRow} (south)`)
      .toBeGreaterThan(south);
  });

  test("the two statics are ordinary grids, and sphere carries its own palette", () => {
    const sph = JSON.parse(fs.readFileSync(
      path.join(F7, "family7_sphere.json"), "utf8"));
    const elv = JSON.parse(fs.readFileSync(
      path.join(F7, "family7_elev.json"), "utf8"));
    const gr = idx.groups.g025.grid;
    for (const [name, g] of [["sphere", sph], ["elev", elv]]) {
      expect(g.nx, `${name} nx`).toBe(gr.nx);
      expect(g.ny, `${name} ny`).toBe(gr.ny);
      expect(g.dlon).toBeCloseTo(gr.step, 9);
      expect(g.dlat).toBeCloseTo(gr.step, 9);
      // point-aligned: the cell is the half-step box around its point
      expect(g.west).toBeCloseTo(gr.lon0 - gr.step / 2, 9);
      expect(g.south).toBeCloseTo(gr.lat0 - gr.step / 2, 9);
    }
    // sphere is CATEGORICAL and ships the producer's palette in the file
    // (CLAUDE.md §2.3) — one character per cell, "." for empty
    expect(sph.packed).toHaveLength(sph.nx * sph.ny);
    expect(/^[0-9.]+$/.test(sph.packed)).toBe(true);
    expect(sph.values).toBe(undefined);
    expect(sph.classes.map((c) => c.code)).toEqual([0, 1, 2, 3]);
    expect(sph.classes.map((c) => c.label))
      .toEqual(["ocean", "land", "ice sheet or glacier", "inland water"]);
    for (const c of sph.classes) expect(c.rgb).toHaveLength(3);
    // every code actually in the file is a code the palette names
    const codes = new Set(sph.packed.replace(/\./g, "").split("").map(Number));
    for (const c of codes) expect(sph.classes.some((k) => k.code === c)).toBe(true);
    // elev is a plain numeric grid in whole metres, null where unknown
    expect(elv.values).toHaveLength(elv.nx * elv.ny);
    expect(elv.units).toBe("m");
    const nums = elv.values.filter((v) => v !== null);
    expect(nums.length).toBeGreaterThan(elv.values.length / 2);
    expect(nums.every((v) => Number.isInteger(v))).toBe(true);
    // the fixture index points at the fixture's own copies
    expect(idx.statics.sphere.file).toBe("data/family7/fixture/family7_sphere.json");
    expect(idx.statics.elev.file).toBe("data/family7/fixture/family7_elev.json");
    expect(idx.statics.sphere.kind).toBe("classGrid");
    expect(idx.fixture).toBe(true);
  });
});

/* ================= the cone geometry on the GLOBAL grid (the dateline wrap) ==
 *
 * The Python side exports a `global` block into data/cone_geometry.json: the
 * same sunflower, evaluated on the 721 × 1440 global grid, with its columns
 * WRAPPED — because on a globe there is no eastern edge, and a dot 4,000 km
 * east of 175°E is a dot in the western Pacific rather than a dot the model
 * reads as missing. The JS port has to agree, or the Cones tab's live mode is
 * a second definition of the cone.
 *
 * SKIPPED, loudly, until that export lands: guarding on the key rather than on
 * a date keeps the suite green today and turns red the moment the two could
 * disagree. */
test.describe("cone_geometry.json · the global block (E-070 live cones)", () => {
  const G = read("cone_geometry.json");
  const CESIUM = "https://cdnjs.cloudflare.com/ajax/libs/cesium/1.133.1";

  test.beforeEach(async ({ page, baseURL }) => {
    if (!G.global) return;
    if (process.env.MIRROR) {
      await page.route(/https:\/\/cdnjs\.cloudflare\.com\/.*/, async (route) => {
        try {
          const url = route.request().url()
            .replace(CESIUM, `${baseURL}/_vendor/cesium`)
            .replace("widgets.min.css", "widgets.css");
          await route.fulfill({ response: await page.request.get(url) });
        } catch { await route.abort().catch(() => {}); }
      });
      await page.route(/https:\/\/gibs\.earthdata\.nasa\.gov\/.*/, async (route) => {
        try {
          const url = route.request().url()
            .replace("https://gibs.earthdata.nasa.gov", "http://localhost:8081");
          await route.fulfill({ response: await page.request.get(url) });
        } catch { await route.abort().catch(() => {}); }
      });
    }
    await page.goto("/");
    await page.waitForFunction(() => window.__earth?.viewer, null, { timeout: 30000 });
  });

  test("the global grid's own header is a point-aligned wrapping 0.25° grid", () => {
    if (!G.global) {
      console.log("SKIPPED: data/cone_geometry.json has no `global` block yet — " +
                  "ml/export_cone_geometry.py has not exported one. This test " +
                  "certifies the JS port against it and will run as soon as it does.");
      test.skip(true, "no `global` block exported yet");
      return;
    }
    expect(G.global.ny).toBe(721);
    expect(G.global.nx).toBe(1440);
    expect(G.global.lat0).toBe(-90);
    expect(G.global.lon0).toBe(-180);
    expect(G.global.step).toBe(0.25);
    expect(G.global.wrap).toBe(true);
    expect(Array.isArray(G.global.refs)).toBe(true);
    expect(G.global.refs.length).toBeGreaterThan(0);
  });

  test("every global reference set is reproduced by the JS port, wrap included",
       async ({ page }) => {
    if (!G.global) {
      console.log("SKIPPED: data/cone_geometry.json has no `global` block yet — " +
                  "the JS port's dateline wrap is uncertified until it lands.");
      test.skip(true, "no `global` block exported yet");
      return;
    }
    /* The export's own schema line: `cells` is
     * `cone.inner_dots(anchor.lat, family)` filtered to that lag, in its own
     * order — the anchor column (0,0) first, then `spiral_offsets`' points
     * deduplicated on the rounded cell — with `row = anchor.row + dy` (which
     * MAY leave the latitude axis: latitude is clipped, never wrapped) and
     * `col = (anchor.col + dx) mod nx` (which never does: longitude closes).
     * `outer_refs` is the same four keys over `cone.outer_spiral`. */
    const inner = G.global.refs, outer = G.global.outer_refs || [];
    const got = await page.evaluate(async ({ inner, outer }) => {
      const E = window.__earth;
      await E.loadCones();
      const g = E.coneGeometry.global;
      const wrap = (a, dy, dx) => [a.row + dy, E.coneWrapCol(a.col + dx, g.nx, g.wrap)];
      const gotInner = inner.map((ref) =>
        E.coneInnerDots(ref.anchor.lat, ref.family)
          .filter(([lag]) => lag === ref.lag)
          .map(([, dy, dx]) => wrap(ref.anchor, dy, dx)));
      const gotOuter = outer.map((ref) =>
        E.coneOuterSpiral(ref.anchor.lat, ref.lag)
          .map(([dy, dx]) => wrap(ref.anchor, dy, dx)));
      // lag 0 is the 3×3 patch, exported once rather than per family
      const patch = [];
      for (let i = 0; i < 9; i++) patch.push([Math.floor(i / 3) - 1, (i % 3) - 1]);
      return { gotInner, gotOuter, patch };
    }, { inner, outer });
    // one assertion per REF, never one per dot (CLAUDE.md §4)
    for (let i = 0; i < inner.length; i++) {
      const r = inner[i];
      expect(got.gotInner[i], `family ${r.family} lag ${r.lag} at row ` +
             `${r.anchor.row} col ${r.anchor.col}`).toEqual(r.cells);
    }
    for (let i = 0; i < outer.length; i++) {
      const r = outer[i];
      expect(got.gotOuter[i], `outer lag ${r.lag} at row ${r.anchor.row} ` +
             `col ${r.anchor.col}`).toEqual(r.cells);
    }
    expect(got.patch).toEqual(G.global.patch_cells);
    // family L (land: snow, soil moisture, soil temperature) is the family the
    // North Atlantic tensor never had, so its presence here is the test that
    // the port reads families from the FILE rather than from a list of three
    expect(inner.some((r) => r.family === "L")).toBe(true);
    // THE WRAP ITSELF: every column the export names is inside the grid. On the
    // North Atlantic window an off-edge column stays off the edge and is drawn
    // hollow; on the globe there is no edge, so a column outside [0, nx) would
    // mean the export and this port disagree about what "east of the dateline"
    // is — and both would still draw a perfectly plausible cone.
    const outside = [...inner, ...outer].flatMap((r) =>
      r.cells.filter(([, c]) => c < 0 || c >= G.global.nx));
    expect(outside).toEqual([]);
    // …and at least one anchor genuinely sits against the dateline, or the
    // wrap is being certified by a set that never exercises it
    const nearEdge = [...inner, ...outer].filter((r) =>
      r.anchor.col < 4 || r.anchor.col > G.global.nx - 5);
    expect(nearEdge.length).toBeGreaterThan(0);
    expect(nearEdge.some((r) => {
      const spread = r.cells.map(([, c]) => c);
      return spread.length > 1 && Math.max(...spread) - Math.min(...spread) > G.global.nx / 2;
    })).toBe(true);
  });
});
