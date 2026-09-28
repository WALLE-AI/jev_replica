#!/usr/bin/env node
// Real-browser checks against the exact JSON served to the comparison page.
// Uses an isolated browser context: no user profile, login, model or API calls.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const HELP = `Usage:
  PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs \\
  CHROME_EXECUTABLE=/path/to/chrome \\
  node scripts/check_side_by_side.mjs --web-root web --output-dir runs/comparison_check

  node scripts/check_side_by_side.mjs --url https://example.chatgpt.site/ \\
    --output-dir runs/comparison_remote_check

Choose exactly one of --url or --web-root. The output directory must be new.
PLAYWRIGHT_MODULE is optional if the playwright package is locally resolvable.
CHROME_EXECUTABLE is optional when Playwright has an installed browser.
The local server exposes only the HTML, CSS, JS and recording JSON.
`;
if (process.argv.includes('--help')) { console.log(HELP); process.exit(0); }
const args = {};
for (let index = 2; index < process.argv.length; index += 2) {
  const key = process.argv[index];
  assert.ok(['--url', '--web-root', '--output-dir'].includes(key), `Unknown option: ${key}`);
  assert.ok(process.argv[index + 1] && !process.argv[index + 1].startsWith('--'), `Missing value: ${key}`);
  assert.ok(!(key.slice(2) in args), `Duplicate option: ${key}`);
  args[key.slice(2)] = process.argv[index + 1];
}
assert.ok(Boolean(args.url) !== Boolean(args['web-root']), 'Choose exactly one of --url or --web-root.');
assert.ok(args['output-dir'], '--output-dir is required.');
const output = path.resolve(args['output-dir']);
await fs.mkdir(path.dirname(output), {recursive: true});
await fs.mkdir(output, {recursive: false});
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const report = {schema: 'nanojev-side-by-side-browser-check-v1', passed: false,
  mode: args.url ? 'remote' : 'local_allowlisted_server', started_at: new Date().toISOString(),
  model_calls: 0, screenshots: {}, scenes: [], controls: [], page_errors: [], failed_asset_requests: []};
let browser, server;

async function localServer(rootPath) {
  const root = await fs.realpath(path.resolve(rootPath));
  let html = 'side-by-side.html';
  try { await fs.access(path.join(root, html)); } catch { html = 'index.html'; }
  const names = [html, 'side-by-side.css', 'side-by-side.js', 'side_by_side_results.json'];
  const content = new Map();
  report.local_asset_sha256 = {};
  for (const name of names) {
    const filename = await fs.realpath(path.join(root, name));
    assert.ok(!path.relative(root, filename).startsWith('..'), 'An allowlisted asset resolves outside the web root.');
    const bytes = await fs.readFile(filename); content.set(name, bytes);
    report.local_asset_sha256[name] = digest(bytes);
  }
  const types = {'.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8', '.json': 'application/json'};
  const routes = new Map([['/', html], ['/index.html', html], ['/side-by-side.html', html],
    ...names.slice(1).map((name) => ['/' + name, name])]);
  server = http.createServer((request, response) => {
    const name = routes.get(new URL(request.url, 'http://localhost').pathname);
    if (!name || !['GET', 'HEAD'].includes(request.method)) { response.writeHead(404).end(); return; }
    response.writeHead(200, {'Content-Type': types[path.extname(name)], 'Cache-Control': 'no-store'});
    response.end(request.method === 'HEAD' ? undefined : content.get(name));
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  return `http://127.0.0.1:${server.address().port}/`;
}

try {
  const modulePath = process.env.PLAYWRIGHT_MODULE;
  const {chromium} = await import(modulePath ? pathToFileURL(path.resolve(modulePath)).href : 'playwright');
  const url = args.url || await localServer(args['web-root']);
  assert.ok(['http:', 'https:'].includes(new URL(url).protocol), '--url must use HTTP or HTTPS.');
  report.url = url;
  browser = await chromium.launch({headless: true,
    ...(process.env.CHROME_EXECUTABLE ? {executablePath: process.env.CHROME_EXECUTABLE} : {}),
    args: ['--disable-background-networking', '--no-first-run', '--no-default-browser-check']});
  report.browser = browser.version();
  const context = await browser.newContext({viewport: {width: 1440, height: 1060}, deviceScaleFactor: 1});
  const page = await context.newPage();
  const assetPattern = /\/(?:side-by-side\.(?:css|js)|side_by_side_results\.json)(?:\?|$)/;
  page.on('pageerror', (error) => report.page_errors.push(error.message));
  page.on('requestfailed', (request) => {
    if (assetPattern.test(request.url())) report.failed_asset_requests.push({url: request.url(), error: request.failure()?.errorText});
  });
  const initial = new URL(url); initial.hash = 'snake';
  const recordingResponse = page.waitForResponse((response) => new URL(response.url()).pathname.endsWith('/side_by_side_results.json'));
  const [, response] = await Promise.all([page.goto(initial.href, {waitUntil: 'networkidle'}), recordingResponse]);
  assert.equal(response.status(), 200, 'Recording JSON must load successfully.');
  const raw = await response.body(); const data = JSON.parse(raw.toString('utf8'));
  report.data_sha256 = digest(raw);
  assert.equal(data.schema, 'nanojev-arcade-v1'); assert.equal(data.examples.length, 2);
  assert.deepEqual(data.examples.map((example) => example.game).sort(), ['maze', 'snake']);
  const order = ['jev', 'nanojev', 'base'];
  for (const example of data.examples) {
    assert.equal(example.systems.length, 3);
    assert.deepEqual(example.systems.map((system) => system.id).sort(), [...order].sort());
  }
  await page.waitForFunction(() => window.nanojevComparison?.ready || window.nanojevComparison?.error);
  assert.equal(await page.evaluate(() => window.nanojevComparison.error), null);
  assert.equal((await page.evaluate(() => window.nanojevComparison.getSnapshot())).game, 'snake');
  report.controls.push('direct_snake_hash');
  const snap = () => page.evaluate(() => window.nanojevComparison.getSnapshot());
  const frameAt = (index, step) => page.evaluate(([i, s]) => window.nanojevComparison.setFrame(i, s), [index, step]);

  function verifySnapshot(snapshot, index, step) {
    const example = data.examples[index]; const maxStep = Math.max(...example.systems.map((system) => system.frames.length - 1));
    assert.equal(snapshot.ready, true); assert.equal(snapshot.error, null); assert.equal(snapshot.playing, false);
    assert.equal(snapshot.exampleIndex, index); assert.equal(snapshot.exampleId, example.id); assert.equal(snapshot.game, example.game);
    assert.equal(snapshot.globalStep, step); assert.equal(snapshot.step, step); assert.equal(snapshot.totalSteps, maxStep);
    assert.deepEqual(snapshot.systems.map((system) => system.id), order);
    for (const actual of snapshot.systems) {
      const expected = example.systems.find((system) => system.id === actual.id);
      const local = Math.min(step, expected.frames.length - 1);
      assert.equal(actual.name, expected.name, 'Displayed identities must retain the source model name.');
      assert.equal(actual.localStep, local); assert.equal(actual.step, local);
      assert.equal(actual.totalSteps, expected.frames.length - 1);
      assert.equal(actual.finished, local === expected.frames.length - 1);
      assert.deepEqual(actual.frame, expected.frames[local], 'Visible frame must equal the real source frame.');
      assert.deepEqual(actual.summary, expected.summary || {});
    }
  }

  for (let index = 0; index < data.examples.length; index++) {
    const example = data.examples[index]; const maxStep = Math.max(...example.systems.map((system) => system.frames.length - 1));
    const steps = [...new Set([0, Math.floor(maxStep / 2), ...example.systems.map((system) => system.frames.length - 1), maxStep])].sort((a, b) => a - b);
    for (const step of steps) {
      const snapshot = await frameAt(index, step); verifySnapshot(snapshot, index, step);
      assert.equal(await page.locator('.model-panel').count(), 3);
      assert.deepEqual(await page.locator('.model-name').allTextContents(), snapshot.systems.map((system) => system.name));
      assert.deepEqual(await page.locator('.stat-steps').allTextContents(), snapshot.systems.map((system) => String(system.localStep)));
      for (let panel = 0; panel < 3; panel++) assert.ok(await page.locator('.model-panel').nth(panel).isVisible());
    }
    report.scenes.push({example_id: example.id, game: example.game, checked_global_steps: steps,
      source_lengths: Object.fromEntries(example.systems.map((system) => [system.id, system.frames.length])),
      exact_frame_comparisons: steps.length * 3, terminal_freeze_verified: true});
  }

  const snakeIndex = data.examples.findIndex((example) => example.game === 'snake');
  const mazeIndex = data.examples.findIndex((example) => example.game === 'maze');
  await page.locator('[data-game="maze"]').click();
  await page.waitForFunction(() => location.hash === '#maze' && window.nanojevComparison.getSnapshot().game === 'maze');
  await page.locator('[data-game="snake"]').click();
  await page.waitForFunction(() => location.hash === '#snake' && window.nanojevComparison.getSnapshot().game === 'snake');
  report.controls.push('maze_tab_hash', 'snake_tab_hash');
  await frameAt(snakeIndex, 0);
  await page.locator('#next').click(); verifySnapshot(await snap(), snakeIndex, 1);
  await page.locator('#previous').click(); verifySnapshot(await snap(), snakeIndex, 0);
  const seek = Math.min(50, (await snap()).totalSteps);
  await page.locator('#timeline').fill(String(seek)); verifySnapshot(await snap(), snakeIndex, seek);
  // Compare the actual canvas pixels, excluding parent CSS clipping/compositing.
  // The full product screenshots below still use real browser page captures.
  const canvasPixelHash = async () => digest(await page.locator('canvas').first().evaluate((canvas) => canvas.toDataURL('image/png')));
  const trailOn = await canvasPixelHash();
  assert.equal(await page.locator('#showTrail').isChecked(), true);
  await page.locator('.trail-control').click();
  assert.equal(await page.locator('#showTrail').isChecked(), false);
  assert.equal((await snap()).showTrail, false);
  const trailOff = await canvasPixelHash();
  assert.notEqual(trailOn, trailOff, 'The trail control must change the rendered canvas.');
  await page.locator('.trail-control').click();
  assert.equal(await page.locator('#showTrail').isChecked(), true);
  assert.equal((await snap()).showTrail, true);
  assert.equal(await canvasPixelHash(), trailOn, 'Restoring trails must reproduce the canvas.');
  report.trail_canvas_sha256 = {on: trailOn, off: trailOff};
  report.trail_hash_scope = 'Strict SHA256 of canvas PNG data URLs; parent CSS clipping is excluded.';
  await page.locator('#restart').click(); verifySnapshot(await snap(), snakeIndex, 0);
  await page.locator('#speed').selectOption('32'); assert.equal((await snap()).stepsPerSecond, 32);
  await page.locator('#speed').selectOption('16'); assert.equal((await snap()).stepsPerSecond, 16);
  await page.locator('#play').click();
  await page.waitForFunction(() => window.nanojevComparison.getSnapshot().globalStep >= 2, undefined, {timeout: 10000});
  await page.locator('#play').click(); const paused = await snap();
  assert.equal(paused.playing, false);
  await page.waitForTimeout(180); verifySnapshot(await snap(), snakeIndex, paused.globalStep);
  report.controls.push('next', 'previous', 'seek', 'trail_off', 'trail_on', 'restart', 'speed', 'play', 'pause');

  const screenshot = async (name, fullPage = false) => {
    const filename = `${name}.png`; await page.screenshot({path: path.join(output, filename), fullPage, animations: 'disabled'});
    const bytes = await fs.readFile(path.join(output, filename));
    report.screenshots[name] = {file: filename, bytes: bytes.length, sha256: digest(bytes), snapshot: await snap()};
  };
  assert.ok(data.examples[snakeIndex].systems.some((system) => system.frames.length > 220), 'The required Snake screenshot step 220 is absent.');
  await frameAt(snakeIndex, 220); verifySnapshot(await snap(), snakeIndex, 220);
  await screenshot('desktop_snake_step220');
  assert.ok(data.examples[mazeIndex].systems.some((system) => system.frames.length > 244), 'The required maze screenshot step 244 is absent.');
  await frameAt(mazeIndex, 244); verifySnapshot(await snap(), mazeIndex, 244);
  await screenshot('desktop_maze_step244');
  // Confirm the desktop boards share a row and fit in the viewport together.
  const desktop = await page.locator('.model-panel').evaluateAll((panels) => panels.map((panel) => {
    const rect = panel.getBoundingClientRect(); return {x: rect.x, y: rect.y, width: rect.width, right: rect.right};
  }));
  assert.equal(desktop.length, 3);
  assert.ok(Math.max(...desktop.map((rect) => rect.y)) - Math.min(...desktop.map((rect) => rect.y)) < 2);
  assert.ok(Math.max(...desktop.map((rect) => rect.width)) - Math.min(...desktop.map((rect) => rect.width)) < 2);
  assert.ok(desktop.every((rect) => rect.x >= 0 && rect.right <= 1440));
  report.desktop_panel_bounds = desktop;

  await page.setViewportSize({width: 390, height: 844});
  await frameAt(snakeIndex, 220);
  report.mobile = await page.evaluate(() => ({viewport: innerWidth, document: document.documentElement.scrollWidth,
    panels: [...document.querySelectorAll('.model-panel')].map((panel) => {
      const rect = panel.getBoundingClientRect(); const style = getComputedStyle(panel);
      return {width: rect.width, left: rect.left, right: rect.right, height: rect.height,
        rendered: style.display !== 'none' && style.visibility !== 'hidden'};
    })}));
  assert.ok(report.mobile.document <= report.mobile.viewport + 1, 'Mobile document overflows horizontally.');
  assert.equal(report.mobile.panels.length, 3);
  for (let index = 0; index < 3; index++) {
    const bounds = report.mobile.panels[index];
    assert.ok(bounds.rendered && bounds.width > 0 && bounds.height > 0 && bounds.left >= 0 && bounds.right <= 391);
    const panel = page.locator('.model-panel').nth(index);
    await panel.scrollIntoViewIfNeeded(); assert.ok(await panel.isVisible());
  }
  report.mobile.visibility_scope = 'All three panels are rendered and individually visible when scrolled into view.';
  await page.evaluate(() => scrollTo(0, 0));
  await screenshot('mobile_snake_step220');
  assert.deepEqual(report.page_errors, []); assert.deepEqual(report.failed_asset_requests, []);
  report.passed = true;
} catch (error) {
  report.failure = {name: error.name, message: error.message};
  process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  if (server) await new Promise((resolve) => server.close(resolve));
  report.completed_at = new Date().toISOString();
  await fs.writeFile(path.join(output, 'browser_check.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({passed: report.passed, output, scenes: report.scenes.length,
    screenshots: Object.keys(report.screenshots), failure: report.failure}));
}
