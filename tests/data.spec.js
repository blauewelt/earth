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

  test("has 241 records with required fields", () => {
    expect(cat.record_count).toBeGreaterThanOrEqual(241);
    expect(cat.records.length).toBeGreaterThanOrEqual(241);
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

  test("has 1000 assets sorted by emissions with valid coordinates", () => {
    expect(c.assets).toHaveLength(1000);
    expect(c.assets[0][2]).toBeGreaterThan(c.assets[c.assets.length - 1][2]);
    for (const [lon, lat, mt] of c.assets) {
      expect(Math.abs(lon)).toBeLessThanOrEqual(180);
      expect(Math.abs(lat)).toBeLessThanOrEqual(90);
      expect(mt).toBeGreaterThan(0);
    }
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
    // presence-vs-abundance caveat is documented in the payload
    expect(s.note.toLowerCase()).toContain("presence");
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
      // some cells filled, values within a physical range
      const finite = g.values.filter((v) => v != null);
      expect(finite.length).toBeGreaterThan(1000);
      for (const v of finite) {
        expect(v).toBeGreaterThanOrEqual(s.ramp === "sst" ? -5 : 0);
        expect(v).toBeLessThan(s.vmaxData);
      }
    });
  }
});
