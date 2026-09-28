#!/usr/bin/env node
/** Verify the independent shooting viewer against its recorded frames. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, index, all) => {
  if (index % 2 === 0) pairs.push([value.replace(/^--/, ''), all[index + 1]]);
  return pairs;
}, []));
assert.ok(args.url && args.output, 'Use --url LOCAL_DEVELOPMENT_URL --output NEW_DIRECTORY');
const out = path.resolve(args.output);
await fs.mkdir(out, {recursive: false});
const {chromium} = await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE).href);
const browser = await chromium.launch({headless: true, executablePath: process.env.CHROME_EXECUTABLE,
  args: ['--disable-background-networking', '--no-first-run', '--no-default-browser-check']});
const report = {url: args.url, browser: browser.version(), page_errors: [], checks: [], cases: []};
try {
  const page = await browser.newPage({viewport: {width: 1600, height: 1100}, deviceScaleFactor: 1});
  page.on('pageerror', error => report.page_errors.push(error.message));
  const responsePromise = page.waitForResponse(response => new URL(response.url()).pathname.endsWith('/shooting_results.json'));
  await page.goto(args.url, {waitUntil: 'networkidle'});
  const response = await responsePromise;
  assert.equal(response.status(), 200);
  const raw = await response.body(), data = JSON.parse(raw.toString());
  report.data_sha256 = crypto.createHash('sha256').update(raw).digest('hex');
  await page.waitForFunction(() => window.nanojevShooting?.ready || window.nanojevShooting?.error);
  assert.equal(await page.evaluate(() => window.nanojevShooting.error), null);
  assert.equal((await page.evaluate(() => window.nanojevShooting.getSnapshot())).caseId, data.default_case_id);
  assert.equal(await page.locator('#caseSelect option').count(), data.cases.length);

  for (const item of data.cases) {
    await page.evaluate(id => window.nanojevShooting.setCase(id), item.id);
    const max = Math.max(...item.systems.map(system => system.total_ticks));
    const ticks = [...new Set([0, 1, Math.floor(max / 2), max,
      ...item.systems.flatMap(system => [system.total_ticks, Math.max(0, system.total_ticks - 1),
        Math.min(max, system.total_ticks + 1)])])].sort((a, b) => a - b);
    const result = await page.evaluate(async ({item, ticks}) => {
      let pixels = 0;
      const cache = new Map();
      const readImage = async src => {
        if (!cache.has(src)) {
          const image = new Image(); image.src = src; await image.decode(); cache.set(src, image);
        }
        return cache.get(src);
      };
      for (const tick of ticks) {
        const snapshot = window.nanojevShooting.setTick(tick);
        if (snapshot.tick !== tick || snapshot.playing || snapshot.caseId !== item.id) throw new Error('Shared tick mismatch');
        for (const system of item.systems) {
          const frame = system.frames[Math.min(tick, system.total_ticks)];
          const actual = snapshot.systems.find(row => row.id === system.id);
          if (JSON.stringify(actual.frame) !== JSON.stringify(frame)) throw new Error('Source frame mismatch');
          const card = document.querySelector(`[data-model="${system.id}"]`);
          if (+card.dataset.tick !== frame.tick || +card.dataset.imageTick !== frame.image_tick ||
              card.dataset.action !== (frame.action || '') || card.dataset.terminal !== String(frame.terminal)) throw new Error('Rendered card timing mismatch');
          for (const row of card.querySelectorAll('.probability-row')) {
            const expected = frame.probabilities?.[row.dataset.action];
            if (row.dataset.probability !== (expected == null ? '' : String(expected))) throw new Error('Probability bar mismatch');
            if (row.classList.contains('selected') !== (frame.action === row.dataset.action)) throw new Error('Executed action highlight mismatch');
          }
          const canvas = card.querySelector('canvas');
          const expectedCanvas = document.createElement('canvas'); expectedCanvas.width = 320; expectedCanvas.height = 240;
          const expectedContext = expectedCanvas.getContext('2d', {alpha: false});
          const sprite = frame.sprite;
          expectedContext.drawImage(await readImage(sprite.src), sprite.x, sprite.y, 320, 240, 0, 0, 320, 240);
          const expected = expectedContext.getImageData(0, 0, 320, 240).data;
          const rendered = canvas.getContext('2d').getImageData(0, 0, 320, 240).data;
          for (let i = 0; i < expected.length; i++) if (expected[i] !== rendered[i]) throw new Error(`Actual canvas differs from recorded sprite: ${system.id}/${tick}`);
          pixels++;
        }
      }
      return {case: item.id, checked_ticks: ticks, canvas_checks: pixels};
    }, {item, ticks});
    report.cases.push(result);
  }
  report.checks.push('all_cases_source_frames_actions_probabilities_and_canvas_pixels', 'terminal_hold_on_shared_physical_clock');
  await page.evaluate(id => window.nanojevShooting.setCase(id), data.default_case_id);
  await page.locator('#next').click();
  assert.equal((await page.evaluate(() => window.nanojevShooting.getSnapshot())).tick, 1);
  await page.locator('#previous').click();
  assert.equal((await page.evaluate(() => window.nanojevShooting.getSnapshot())).tick, 0);
  await page.locator('#speed').selectOption('2');
  await page.locator('#play').click();
  await page.waitForFunction(() => window.nanojevShooting.getSnapshot().tick >= 3);
  await page.locator('#play').click();
  const paused = await page.evaluate(() => window.nanojevShooting.getSnapshot());
  assert.equal(paused.playing, false); assert.equal(paused.speed, 2);
  await page.locator('#restart').click();
  assert.equal((await page.evaluate(() => window.nanojevShooting.getSnapshot())).tick, 0);
  await page.locator('#timeline').evaluate(element => {
    element.value = '12'; element.dispatchEvent(new Event('input', {bubbles: true}));
  });
  assert.equal((await page.evaluate(() => window.nanojevShooting.getSnapshot())).tick, 12);
  await page.locator('h1').click(); await page.keyboard.press('ArrowRight');
  assert.equal((await page.evaluate(() => window.nanojevShooting.getSnapshot())).tick, 13);
  report.checks.push('play_pause_restart_step_seek_speed_keyboard');
  const choice = data.cases.find(item => item.id !== data.default_case_id);
  await page.locator('#caseSelect').selectOption(choice.id);
  await page.waitForFunction(id => window.nanojevShooting.ready && window.nanojevShooting.getSnapshot().caseId === id && window.nanojevShooting.getSnapshot().ready, choice.id);
  report.checks.push('visible_case_selector');
  await page.evaluate(id => window.nanojevShooting.setCase(id), data.default_case_id);
  const defaultCase = data.cases.find(item => item.id === data.default_case_id);
  const posterTick = Math.min(Math.max(...defaultCase.systems.map(system => system.total_ticks)),
                             defaultCase.systems.find(system => system.id === 'nanojev').total_ticks);
  await page.evaluate(tick => window.nanojevShooting.setTick(tick), posterTick);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.screenshot({path: path.join(out, 'shooting_desktop.png'), fullPage: true});
  await page.setViewportSize({width: 390, height: 844});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  assert.equal(await page.locator('[data-model]').count(), 3);
  await page.screenshot({path: path.join(out, 'shooting_mobile.png'), fullPage: true});
  report.checks.push('desktop_and_mobile_without_horizontal_overflow');
  assert.deepEqual(report.page_errors, []);
  report.passed = true;
} finally {
  await fs.writeFile(path.join(out, 'browser_check.json'), JSON.stringify(report, null, 2) + '\n');
  await browser.close();
}
console.log(JSON.stringify({passed: report.passed, cases: report.cases.length,
  canvas_checks: report.cases.reduce((n, c) => n + c.canvas_checks, 0), output: out}));
