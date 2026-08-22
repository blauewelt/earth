// Run the browser suite against a DEPLOYED site instead of the local server.
//   PLAYWRIGHT_BASE_URL=https://blauewelt.pages.dev npx playwright test -c playwright.remote.config.js tests/app.spec.js tests/docs.spec.js
// The base config pins localhost:8080 and starts python's http.server, so
// PLAYWRIGHT_BASE_URL alone is ignored there. docs/HOSTING.md §6.0b step 6.
"use strict";
const base = require("./playwright.config.js");
if (!process.env.PLAYWRIGHT_BASE_URL) throw new Error("set PLAYWRIGHT_BASE_URL");
module.exports = { ...base, webServer: undefined,
  use: { ...base.use, baseURL: process.env.PLAYWRIGHT_BASE_URL } };
