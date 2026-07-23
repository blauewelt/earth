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
