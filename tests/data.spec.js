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
    expect(cat.record_count).toBe(241);
    expect(cat.records).toHaveLength(241);
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
