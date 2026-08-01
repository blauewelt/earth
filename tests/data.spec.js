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
