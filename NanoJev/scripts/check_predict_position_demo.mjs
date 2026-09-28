#!/usr/bin/env node
/** Verify the Predict Position viewer against every recorded physical tick. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';

const args = {};
for (let index = 2; index < process.argv.length; index += 2) {
  const key = process.argv[index];
  assert.ok(['--url', '--output', '--poster-tick'].includes(key), `Unknown option: ${key}`);
  assert.ok(process.argv[index + 1] && !process.argv[index + 1].startsWith('--'), `Missing value: ${key}`);
  assert.ok(!(key.slice(2) in args), `Duplicate option: ${key}`);
  args[key.slice(2)] = process.argv[index + 1];
}
assert.ok(args.url && args.output, 'Use --url LOCAL_DEVELOPMENT_URL --output NEW_DIRECTORY');
assert.ok(process.env.PLAYWRIGHT_MODULE && process.env.CHROME_EXECUTABLE, 'Set PLAYWRIGHT_MODULE and CHROME_EXECUTABLE.');
const out = path.resolve(args.output);
await fs.mkdir(out, {recursive: false});
const {chromium} = await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE).href);
const browser = await chromium.launch({headless: true, executablePath: process.env.CHROME_EXECUTABLE,
  args: ['--disable-background-networking', '--no-first-run', '--no-default-browser-check']});
const report = {schema: 'nanojev-predict-position-browser-check-v1', passed: false,
  url: args.url, browser: browser.version(), page_errors: [], checks: [], cases: []};
try {
  const page = await browser.newPage({viewport: {width: 1600, height: 1200}, deviceScaleFactor: 1});
  page.on('pageerror', error => report.page_errors.push(error.message));
  const responsePromise = page.waitForResponse(response => new URL(response.url()).pathname.endsWith('/predict_position_results.json'));
  await page.goto(args.url, {waitUntil: 'networkidle'});
  const response = await responsePromise;
  assert.equal(response.status(), 200);
  const raw = await response.body(), data = JSON.parse(raw.toString());
  report.data_sha256 = crypto.createHash('sha256').update(raw).digest('hex');
  await page.waitForFunction(() => window.nanojevShooting?.ready || window.nanojevShooting?.error);
  assert.equal(await page.evaluate(() => window.nanojevShooting.error), null);
  assert.equal((await page.evaluate(() => window.nanojevShooting.getSnapshot())).caseId, data.default_case_id);
  assert.equal(await page.locator('#caseSelect option').count(), data.cases.length);
  assert.equal(await page.locator('#benchmarkStrip').isVisible(), true);
  report.benchmark_text = await page.locator('#benchmarkStrip').innerText();
  assert.match(report.benchmark_text, /128/, 'Benchmark strip must disclose the full held-out denominator.');
  for (const model of data.models) {
    const cell = page.locator(`#benchmarkStrip .benchmark-cell[data-model="${model.id}"]`);
    assert.equal(await cell.getAttribute('data-successes'), String(data.summary[model.id].test.successes));
    assert.equal(await cell.getAttribute('data-episodes'), String(data.summary[model.id].test.episodes));
    assert.equal(await cell.locator('.benchmark-rate').innerText(), `${(100 * data.summary[model.id].test.successes / data.summary[model.id].test.episodes).toFixed(1)}%`);
  }

  for (const item of data.cases) {
    await page.evaluate(id => window.nanojevShooting.setCase(id), item.id);
    const result = await page.evaluate(async item => {
      const finalTick = Math.max(...item.systems.map(system => system.total_ticks));
      const cache = new Map(), verifiedCrops = new Set();
      const readImage = async src => {
        if (!cache.has(src)) {
          const image = new Image(); image.src = src; await image.decode(); cache.set(src, image);
        }
        return cache.get(src);
      };
      const expectedCanvas = document.createElement('canvas'); expectedCanvas.width = 320; expectedCanvas.height = 240;
      const expectedContext = expectedCanvas.getContext('2d', {alpha: false, willReadFrequently: true});
      let pixels = 0, metadata = 0, heldImages = 0, terminalHolds = 0;
      const expectedShots = item.systems.flatMap(system => system.frames.flatMap((frame, index) =>
        index > 0 && Number.isFinite(frame.ammo) && Number.isFinite(system.frames[index - 1].ammo) &&
        frame.ammo < system.frames[index - 1].ammo ? [{model: system.id, tick: frame.tick}] : []));
      const actualShots = [...document.querySelectorAll('.shot-marker')].map(element => ({model: element.dataset.model, tick: +element.dataset.tick}));
      const shotKey = shot => `${shot.model}:${shot.tick}`;
      if (JSON.stringify(actualShots.map(shotKey).sort()) !== JSON.stringify(expectedShots.map(shotKey).sort())) throw new Error('Shot markers disagree with actual ammo-consumption ticks.');
      for (const tick of Array.from({length: finalTick + 1}, (_, index) => index)) {
        const snapshot = window.nanojevShooting.setTick(tick);
        if (snapshot.tick !== tick || snapshot.playing || snapshot.caseId !== item.id) throw new Error('Shared physical tick mismatch');
        for (const system of item.systems) {
          const frame = system.frames[Math.min(tick, system.total_ticks)];
          const actual = snapshot.systems.find(row => row.id === system.id);
          if (JSON.stringify(actual.frame) !== JSON.stringify(frame)) throw new Error('Source frame mismatch');
          const card = document.querySelector(`.model-card[data-model="${system.id}"]`);
          if (+card.dataset.tick !== frame.tick || +card.dataset.imageTick !== frame.image_tick ||
              card.dataset.action !== (frame.action || '') || card.dataset.terminal !== String(frame.terminal)) throw new Error('Rendered card timing mismatch');
          const firstShot = expectedShots.find(shot => shot.model === system.id)?.tick;
          const launched = firstShot !== undefined && frame.tick >= firstShot;
          const expectedPhase = frame.terminal ? (system.success ? 'Target hit' : 'Target missed') : launched ? 'Rocket fired' : 'Rocket ready';
          if (card.querySelector('.shot-phase')?.textContent !== expectedPhase || card.dataset.launched !== String(launched)) throw new Error('Shot phase disagrees with actual launch or terminal outcome.');
          if (card.querySelector('.stat-ammo').textContent !== (frame.ammo == null ? '—' : String(frame.ammo))) throw new Error('Ammo display mismatch');
          const note = card.querySelector('.image-note');
          if (note.hidden !== (frame.image_tick === frame.tick)) throw new Error('Image holding disclosure mismatch');
          if (!note.hidden && !note.textContent.includes(String(frame.image_tick))) throw new Error('Held image tick is not disclosed');
          if (frame.image_tick !== frame.tick) heldImages++;
          if (tick > system.total_ticks) terminalHolds++;
          for (const row of card.querySelectorAll('.probability-row')) {
            const expected = frame.probabilities?.[row.dataset.action];
            if (row.dataset.probability !== (expected == null ? '' : String(expected))) throw new Error('Probability bar mismatch');
            if (row.classList.contains('selected') !== (frame.action === row.dataset.action)) throw new Error('Executed action highlight mismatch');
          }
          metadata++;
          const sprite = frame.sprite, cropKey = `${system.id}:${sprite.src}:${sprite.x}:${sprite.y}`;
          // Compare every unique real image crop; held images still get complete metadata checks above.
          if (!verifiedCrops.has(cropKey)) {
            const image = await readImage(sprite.src);
            if (sprite.x + 320 > image.naturalWidth || sprite.y + 240 > image.naturalHeight) throw new Error('Atlas crop exceeds recorded image bounds');
            expectedContext.drawImage(image, sprite.x, sprite.y, 320, 240, 0, 0, 320, 240);
            const expected = new Uint32Array(expectedContext.getImageData(0, 0, 320, 240).data.buffer);
            const rendered = new Uint32Array(card.querySelector('canvas').getContext('2d').getImageData(0, 0, 320, 240).data.buffer);
            for (let i = 0; i < expected.length; i++) if (expected[i] !== rendered[i]) throw new Error(`Canvas differs from its real recorded sprite: ${system.id}/${tick}`);
            verifiedCrops.add(cropKey); pixels++;
          }
        }
      }
      return {case: item.id, physical_ticks_checked: finalTick + 1, frame_metadata_checks: metadata,
        unique_canvas_crop_checks: pixels, held_image_checks: heldImages, terminal_hold_checks: terminalHolds,
        shot_markers: expectedShots};
    }, item);
    for (const shot of result.shot_markers) {
      await page.locator(`.shot-marker[data-model="${shot.model}"][data-tick="${shot.tick}"]`).click();
      const snapshot = await page.evaluate(() => window.nanojevShooting.getSnapshot());
      assert.equal(snapshot.tick, shot.tick); assert.equal(snapshot.playing, false);
    }
    report.cases.push(result);
    console.log(JSON.stringify({checked_case: item.id, physical_ticks: result.physical_ticks_checked,
      unique_canvas_crops: result.unique_canvas_crop_checks, shot_markers: result.shot_markers.length}));
  }
  report.checks.push('every_physical_tick_source_metadata_actions_and_probabilities',
    'every_unique_recorded_canvas_crop_pixel_identical', 'terminal_hold_on_shared_physical_clock',
    'actual_ammo_consumption_shot_markers_and_navigation', 'held_image_tick_disclosures');
  await page.evaluate(id => window.nanojevShooting.setCase(id), data.default_case_id);
  await page.locator('#next').click();
  assert.equal((await page.evaluate(() => window.nanojevShooting.getSnapshot())).tick, 1);
  await page.locator('#previous').click();
  assert.equal((await page.evaluate(() => window.nanojevShooting.getSnapshot())).tick, 0);
  await page.locator('#speed').selectOption('0.5');
  await page.locator('#play').click();
  await page.waitForFunction(() => window.nanojevShooting.getSnapshot().tick >= 3);
  await page.locator('#play').click();
  const paused = await page.evaluate(() => window.nanojevShooting.getSnapshot());
  assert.equal(paused.playing, false); assert.equal(paused.speed, 0.5);
  await page.locator('#restart').click();
  assert.equal((await page.evaluate(() => window.nanojevShooting.getSnapshot())).tick, 0);
  await page.locator('#timeline').evaluate(element => {
    element.value = '12'; element.dispatchEvent(new Event('input', {bubbles: true}));
  });
  assert.equal((await page.evaluate(() => window.nanojevShooting.getSnapshot())).tick, 12);
  await page.locator('h1').click(); await page.keyboard.press('ArrowRight');
  assert.equal((await page.evaluate(() => window.nanojevShooting.getSnapshot())).tick, 13);
  report.checks.push('play_pause_restart_step_seek_slow_motion_speed_keyboard');
  const choice = data.cases.find(item => item.id !== data.default_case_id);
  if (choice) {
    await page.locator('#caseSelect').selectOption(choice.id);
    await page.waitForFunction(id => window.nanojevShooting.ready && window.nanojevShooting.getSnapshot().caseId === id && window.nanojevShooting.getSnapshot().ready, choice.id);
    report.checks.push('visible_case_selector');
  }
  await page.evaluate(id => window.nanojevShooting.setCase(id), data.default_case_id);
  const defaultCase = data.cases.find(item => item.id === data.default_case_id);
  const nanojev = defaultCase.systems.find(system => system.id === 'nanojev');
  const shot = nanojev.frames.find((frame, index) => index > 0 && Number.isFinite(frame.ammo) &&
    Number.isFinite(nanojev.frames[index - 1].ammo) && frame.ammo < nanojev.frames[index - 1].ammo);
  const posterTick = args['poster-tick'] == null ? Math.min(nanojev.total_ticks, (shot?.tick || 0) + 12) : Number(args['poster-tick']);
  assert.ok(Number.isInteger(posterTick) && posterTick >= 0);
  await page.evaluate(tick => window.nanojevShooting.setTick(tick), posterTick);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.screenshot({path: path.join(out, 'predict_position_desktop.png'), fullPage: true, animations: 'disabled'});
  await page.setViewportSize({width: 390, height: 844});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  assert.equal(await page.locator('.model-card[data-model]').count(), 3);
  await page.screenshot({path: path.join(out, 'predict_position_mobile.png'), fullPage: true, animations: 'disabled'});
  report.screenshots = {desktop: 'predict_position_desktop.png', mobile: 'predict_position_mobile.png', case: defaultCase.id, physical_tick: posterTick};
  report.checks.push('desktop_and_mobile_without_horizontal_overflow', 'full_held_out_benchmark_display');
  assert.deepEqual(report.page_errors, []);
  report.passed = true;
} catch (error) {
  report.failure = {name: error.name, message: error.message};
  throw error;
} finally {
  await fs.writeFile(path.join(out, 'browser_check.json'), JSON.stringify(report, null, 2) + '\n');
  await browser.close();
}
console.log(JSON.stringify({passed: report.passed, cases: report.cases.length,
  canvas_checks: report.cases.reduce((n, c) => n + c.unique_canvas_crop_checks, 0), output: out}));
