// Playwright config for the earth test suite.
// CI: real network (cdnjs + NASA GIBS). Local sandbox: MIRROR=1 + CHROMIUM_PATH.
"use strict";
const { defineConfig } = require("@playwright/test");

module.exports = defineConfig({
  testDir: "tests",
  timeout: 90000,
  retries: process.env.CI ? 1 : 0,
  workers: 2,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: "http://localhost:8080",
    launchOptions: {
      ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}),
      args: ["--use-gl=swiftshader"],
    },
  },
  webServer: {
    command: "python3 -m http.server 8080",
    url: "http://localhost:8080/",
    reuseExistingServer: true,
  },
});
