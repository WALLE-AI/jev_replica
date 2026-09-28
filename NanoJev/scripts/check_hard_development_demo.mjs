#!/usr/bin/env node
/** Run strict hard-task browser verification with an independently pinned navigation receipt. */
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

if (process.argv.includes('--help')) {
  console.log(`Usage:
  PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs CHROME_EXECUTABLE=/path/to/chrome \\
  node scripts/check_hard_development_demo.mjs --url DEVELOPMENT_DIRECTORY_URL \\
    --output NEW_SCREENSHOT_DIRECTORY --report NEW_REPORT_FILE \\
    --navigation-receipt VERIFIED_NAVIGATION_RECEIPT_JSON \\
    --navigation-receipt-sha256 INDEPENDENTLY_PINNED_RECEIPT_SHA256

Maze must be 50x50; Snake must retain its 12x12, 256-step continuing-food challenge.
Navigation JSON must match the pinned replay/export receipt and current checkpoint.
Basic retains its 3 cases. Predict Position retains exactly its 2 NanoJev-only wins.
No model or API calls are made.`);
  process.exit(0);
}
assert.ok(process.argv.includes('--navigation-receipt') && process.argv.includes('--navigation-receipt-sha256'),
  'A verified navigation receipt and its independently pinned SHA256 are required.');
const result = spawnSync(process.execPath, [fileURLToPath(new URL('./check_unified_development_demo.mjs', import.meta.url)),
  ...process.argv.slice(2)], {stdio: 'inherit', env: process.env});
if (result.error) throw result.error;
process.exit(result.status ?? 1);
