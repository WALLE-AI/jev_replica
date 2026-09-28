#!/usr/bin/env node
/** Verify all four development tasks against one frozen checkpoint and their served recordings. */
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const EXPECTED_CHECKPOINT = 'f68c47d66998231b86b7e91b4ed5e82ae23acf104c8b7cd6d165c3ac7b7ffe1b';
const EXPECTED_EPISODES = {
  jev: 'f5d54efe726e737cd394b32b11bde81b59e72fa107e2ba00e9b44413728fb945',
  nanojev: '0a6fd7e0ce9498472add6277a869d4b9e04bcb366c3241ed823993f3d21b6077',
  base: '998f3d2eeebfab5e5d6415e2fb0e7b5cb36f3b66f333ff8be12276755c7a375d',
};
const args = {};
for (let index = 2; index < process.argv.length; index += 2) {
  const key = process.argv[index];
  assert.ok(['--url', '--output', '--report', '--navigation-receipt', '--navigation-receipt-sha256'].includes(key), `Unknown option: ${key}`);
  assert.ok(process.argv[index + 1] && !process.argv[index + 1].startsWith('--'), `Missing value: ${key}`);
  assert.ok(!(key.slice(2) in args), `Duplicate option: ${key}`);
  args[key.slice(2)] = process.argv[index + 1];
}
assert.ok(args.url && args.output, 'Use --url DEVELOPMENT_DIRECTORY_URL --output NEW_SCREENSHOT_DIRECTORY [--report NEW_REPORT_FILE].');
const hardMode = Boolean(args['navigation-receipt']);
assert.equal(hardMode, Boolean(args['navigation-receipt-sha256']), 'Hard-task verification requires both the navigation receipt and its independently pinned SHA256.');
assert.ok(process.env.PLAYWRIGHT_MODULE && process.env.CHROME_EXECUTABLE, 'Set PLAYWRIGHT_MODULE and CHROME_EXECUTABLE.');
const base = new URL(args.url);
assert.ok(['http:', 'https:'].includes(base.protocol));
assert.ok(base.pathname.endsWith('/'), '--url must end with / and point to the development site directory.');
const output = path.resolve(args.output);
await fs.mkdir(path.dirname(output), {recursive: true}); await fs.mkdir(output, {recursive: false});
const reportPath = path.resolve(args.report || path.join(output, 'browser_check.json'));
try { await fs.lstat(reportPath); throw new Error(`Refusing to overwrite ${reportPath}`); } catch (error) { if (error.code !== 'ENOENT') throw error; }
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const report = {schema: hardMode ? 'nanojev-hard-development-browser-check-v3' : 'nanojev-unified-development-browser-check-v2', passed: false, url: base.href,
  started_at: new Date().toISOString(), expected_checkpoint_sha256: EXPECTED_CHECKPOINT,
  expected_source_episodes_sha256: EXPECTED_EPISODES, source_episode_scope: hardMode ? 'Basic and Predict Position only; navigation is independently receipt-bound.' : 'All four tasks.',
  model_calls: 0, api_calls: 0,
  datasets: {}, tasks: [], navigation: [], screenshots: {}, page_errors: [], failed_asset_requests: []};
let browser;
try {
  let navigationReceipt;
  if (hardMode) {
    assert.match(args['navigation-receipt-sha256'], /^[a-f0-9]{64}$/);
    const bytes = await fs.readFile(path.resolve(args['navigation-receipt']));
    assert.equal(digest(bytes), args['navigation-receipt-sha256'], 'Navigation receipt must equal its independently pinned SHA256.');
    navigationReceipt = JSON.parse(bytes.toString());
    assert.equal(navigationReceipt.passed, true, 'Navigation export/replay receipt must have passed.');
    assert.equal(navigationReceipt.selected_checkpoint_sha256, EXPECTED_CHECKPOINT);
    assert.match(navigationReceipt.export_sha256, /^[a-f0-9]{64}$/);
    report.navigation_receipt = {file: args['navigation-receipt'], sha256: digest(bytes),
      export_sha256: navigationReceipt.export_sha256, selected_checkpoint_sha256: navigationReceipt.selected_checkpoint_sha256};
  }
  const {chromium} = await import(pathToFileURL(path.resolve(process.env.PLAYWRIGHT_MODULE)).href);
  browser = await chromium.launch({headless: true, executablePath: process.env.CHROME_EXECUTABLE,
    args: ['--disable-background-networking', '--no-first-run', '--no-default-browser-check']});
  report.browser = browser.version();
  const context = await browser.newContext({viewport: {width: 1600, height: 1200}, deviceScaleFactor: 1});
  const page = await context.newPage();
  page.on('pageerror', error => report.page_errors.push({url: page.url(), message: error.message}));
  page.on('requestfailed', request => {
    if (/\.(?:js|css|json|webp|png)(?:\?|$)/.test(request.url())) report.failed_asset_requests.push({url: request.url(), error: request.failure()?.errorText});
  });
  for (const filename of ['side_by_side_results.json', 'shooting_results.json', 'predict_position_results.json']) {
    const response = await context.request.get(new URL(filename, base).href);
    assert.equal(response.status(), 200, `${filename} must load successfully.`);
    const raw = await response.body(), data = JSON.parse(raw.toString());
    assert.equal(data.protocol?.selected_checkpoint_sha256, EXPECTED_CHECKPOINT, `${filename} must use the current unified model.`);
    if (hardMode && filename === 'side_by_side_results.json') {
      assert.equal(digest(raw), navigationReceipt.export_sha256, 'Served hard navigation data must equal the independently verified export receipt.');
    } else {
      assert.deepEqual(data.protocol.source_episodes_sha256, EXPECTED_EPISODES, `${filename} must use the same frozen final evaluation episodes.`);
    }
    report.datasets[filename] = {sha256: digest(raw), bytes: raw.length, checkpoint_sha256: data.protocol.selected_checkpoint_sha256,
      source_episodes_sha256: data.protocol.source_episodes_sha256};
  }
  const getPageData = async (filename, relative) => {
    const responsePromise = page.waitForResponse(response => new URL(response.url()).pathname.endsWith('/' + filename));
    const [response] = await Promise.all([responsePromise, page.goto(new URL(relative, base).href, {waitUntil: 'networkidle'})]);
    assert.equal(response.status(), 200);
    const raw = await response.body(); assert.equal(digest(raw), report.datasets[filename].sha256, 'Displayed data must equal the independently fetched current recording.');
    return JSON.parse(raw.toString());
  };
  const ready = async kind => {
    await page.waitForFunction(kind => { const api = kind === 'navigation' ? window.nanojevComparison : window.nanojevShooting; return api?.ready || api?.error; }, kind);
    assert.equal(await page.evaluate(kind => (kind === 'navigation' ? window.nanojevComparison : window.nanojevShooting).error, kind), null);
  };
  async function screenshot(name, snapshot) {
    await page.evaluate(() => scrollTo(0, 0));
    const layout = await page.evaluate(() => ({viewport_width: innerWidth, document_width: document.documentElement.scrollWidth,
      panels: [...document.querySelectorAll('.model-card, .model-panel')].map(element => {
        const rect = element.getBoundingClientRect(), style = getComputedStyle(element);
        return {left: rect.left, right: rect.right, width: rect.width, height: rect.height,
          visible: style.display !== 'none' && style.visibility !== 'hidden'};
      })}));
    assert.ok(layout.document_width <= layout.viewport_width + 1, `${name}: horizontal overflow.`);
    assert.equal(layout.panels.length, 3);
    assert.ok(layout.panels.every(panel => panel.visible && panel.width > 0 && panel.height > 0 && panel.left >= -1 && panel.right <= layout.viewport_width + 1));
    const filename = `${name}.png`, bytes = await page.screenshot({path: path.join(output, filename), fullPage: true, animations: 'disabled'});
    report.screenshots[name] = {file: filename, bytes: bytes.length, sha256: digest(bytes), layout, snapshot};
  }
  async function controls(kind, maxStep) {
    const snapshot = () => page.evaluate(kind => (kind === 'navigation' ? window.nanojevComparison : window.nanojevShooting).getSnapshot(), kind);
    const index = kind === 'navigation' ? (await snapshot()).exampleIndex : undefined;
    const reset = () => page.evaluate(({kind, index}) => kind === 'navigation' ? window.nanojevComparison.setFrame(index, 0) : window.nanojevShooting.setTick(0), {kind, index});
    const position = snapshot => kind === 'navigation' ? snapshot.globalStep : snapshot.tick;
    await reset();
    if (maxStep > 0) {
      await page.locator('#next').click(); assert.equal(position(await snapshot()), 1);
      await page.locator('#previous').click(); assert.equal(position(await snapshot()), 0);
      const seek = Math.min(12, maxStep);
      await page.locator('#timeline').evaluate((element, value) => { element.value = String(value); element.dispatchEvent(new Event('input', {bubbles: true})); }, seek);
      assert.equal(position(await snapshot()), seek);
      await page.locator('#restart').click(); assert.equal(position(await snapshot()), 0);
      const speed = kind === 'navigation' ? '16' : '0.5';
      await page.locator('#speed').selectOption(speed);
      await page.locator('#play').click();
      await page.waitForFunction(kind => {
        const snapshot = (kind === 'navigation' ? window.nanojevComparison : window.nanojevShooting).getSnapshot();
        return (kind === 'navigation' ? snapshot.globalStep : snapshot.tick) >= 1;
      }, kind);
      const playing = await snapshot();
      if (playing.playing) await page.locator('#play').click();
      const paused = await snapshot(); assert.equal(paused.playing, false);
      assert.equal(kind === 'navigation' ? paused.stepsPerSecond : paused.speed, Number(speed));
      await page.waitForTimeout(100); assert.equal(position(await snapshot()), position(paused));
      await reset();
    }
    return ['next', 'previous', 'seek', 'restart', 'speed', 'play', 'pause'];
  }

  const navData = await getPageData('side_by_side_results.json', 'side-by-side.html#maze');
  await ready('navigation');
  assert.deepEqual(navData.examples.map(example => example.game).sort(), ['maze', 'snake']);
  assert.equal((await page.evaluate(() => window.nanojevComparison.getSnapshot())).game, 'maze');
  if (hardMode) {
    const maze = navData.examples.find(example => example.game === 'maze');
    const snake = navData.examples.find(example => example.game === 'snake');
    assert.equal(maze.id, 'maze:ood:50:24310922'); assert.equal(maze.size, 50);
    assert.equal(snake.id, 'snake:showcase:12:61005'); assert.equal(snake.size, 12);
    assert.equal(snake.horizon, 256); assert.equal(snake.max_steps, 256);
    assert.equal(maze.horizon, 5000); assert.equal(maze.max_steps, 5000);
    assert.deepEqual(navigationReceipt.examples.map(example => example.case_id).sort(), navData.examples.map(example => example.id).sort());
    for (const example of navData.examples) {
      const receipt = navigationReceipt.examples.find(recorded => recorded.case_id === example.id);
      assert.equal(receipt.game, example.game); assert.equal(receipt.passed, true);
      assert.deepEqual(receipt.controller, navData.protocol.controllers[example.game]);
      assert.equal(receipt.horizon, example.horizon); assert.equal(receipt.size, example.size);
      assert.deepEqual(receipt.sources.map(source => source.system_id).sort(), ['base', 'jev', 'nanojev']);
      assert.ok(example.systems.every(system => system.frames.length - 1 <= example.max_steps));
      for (const system of example.systems) {
        const source = receipt.sources.find(source => source.system_id === system.id);
        assert.equal(source.passed, true);
        assert.equal(source.initial_state_sha256, receipt.initial_state_sha256);
        assert.equal(source.sha256, navData.protocol.source_recordings_sha256[example.game][system.id]);
        assert.equal(source.verified_transitions, system.frames.length - 1);
        assert.deepEqual(receipt.results[system.id], system.summary);
        if (system.id !== 'nanojev') assert.equal(source.frames_unchanged_from_original_public, true);
        assert.equal(system.summary.steps, system.frames.length - 1, 'Hard-task endpoint length must match its verified summary.');
        assert.equal(system.summary.score, system.frames.at(-1).score, 'Hard-task final score must match its real terminal frame.');
      }
    }
    report.hard_task_scope = {maze: {id: maze.id, size: maze.size, horizon: maze.horizon},
      snake: {id: snake.id, size: snake.size, horizon: snake.horizon}};
  }
  for (let exampleIndex = 0; exampleIndex < navData.examples.length; exampleIndex++) {
    const example = navData.examples[exampleIndex];
    const result = await page.evaluate(async ({example, exampleIndex, hardMode}) => {
      const max = Math.max(...example.systems.map(system => system.frames.length - 1));
      const finalCanvases = {}, directions = ['north', 'east', 'south', 'west'];
      let compared = 0, frozen = 0, frozenMetadata = 0;
      for (let step = 0; step <= max; step++) {
        // Long 50x50 trajectories retain every metadata check while yielding to the browser event loop.
        if (step > 0 && step % 50 === 0) await new Promise(resolve => setTimeout(resolve, 0));
        const snapshot = window.nanojevComparison.setFrame(exampleIndex, step);
        if (snapshot.globalStep !== step || snapshot.playing || snapshot.exampleId !== example.id) throw new Error('Navigation shared-step mismatch.');
        for (const source of example.systems) {
          const actual = snapshot.systems.find(system => system.id === source.id), local = Math.min(step, source.frames.length - 1);
          const frame = source.frames[local];
          if (actual.localStep !== local || actual.finished !== (local === source.frames.length - 1) ||
            JSON.stringify(actual.frame) !== JSON.stringify(frame) || JSON.stringify(actual.summary) !== JSON.stringify(source.summary || {})) throw new Error('Navigation source frame/summary mismatch.');
          const card = document.querySelector(`.model-panel[data-system="${source.id}"]`);
          if (card.querySelector('.stat-steps').textContent !== String(local)) throw new Error('Navigation step display mismatch.');
          if (card.querySelector('.board-status').hidden !== (local !== source.frames.length - 1)) throw new Error('Navigation terminal badge appears at the wrong source step.');
          if (hardMode && local === source.frames.length - 1) {
            const labels = {goal: 'Goal reached', goal_reached: 'Goal reached', horizon_survived: 'Survived',
              trapped: 'Trapped', step_limit: 'Step limit', timeout: 'Step limit'};
            const expectedLabel = labels[source.summary?.outcome];
            if (expectedLabel && card.querySelector('.stat-status').textContent !== expectedLabel) throw new Error('Hard-task terminal status differs from the verified outcome.');
          }
          if (example.game === 'snake' && card.querySelector('.stat-secondary').textContent !== String(frame.score)) throw new Error('Snake food score display mismatch.');
          if (local === source.frames.length - 1 && source.summary?.outcome === 'target_reached' &&
            card.querySelector('.stat-status').textContent !== 'Food target reached') throw new Error('Snake food-target terminal label mismatch.');
          const rows = [...card.querySelectorAll('.probability-item')];
          if (rows.length !== directions.length) throw new Error('Navigation directions are incomplete.');
          rows.forEach((row, index) => {
            const expected = local ? frame.probabilities?.[directions[index]] : undefined;
            if (row.querySelector('.probability-value').textContent !== (expected === undefined ? '—' : `${(100 * expected).toFixed(1)}%`)) throw new Error('Navigation probability display mismatch.');
            if (row.classList.contains('chosen') !== (local > 0 && frame.action === directions[index])) throw new Error('Navigation executed action highlight mismatch.');
          });
          if (local > 0 && (frame.forced === true || /forced/i.test(frame.decision_source || ''))) {
            const entries = Object.entries(frame.probabilities || {});
            const codeOneHot = hardMode && entries.length === 1 && entries[0][0] === frame.action && entries[0][1] === 1;
            if (entries.length !== 0 && !codeOneHot) throw new Error('Shared code moves must not reuse stale model probabilities.');
            if (hardMode && !/shared planner move|code.forced move/i.test(card.querySelector('.probability-kind').textContent)) throw new Error('Deterministic code action must be labelled separately from model probabilities.');
          }
          if (local === source.frames.length - 1) {
            if (step > local) frozenMetadata++;
            // Bitmap equality at the terminal transition, the next shared step, and the final shared step.
            // Complete source-frame equality above still runs at every intervening step.
            if (step === local || step === local + 1 || step === max) {
              const pixels = card.querySelector('canvas').toDataURL('image/png');
              if (source.id in finalCanvases && finalCanvases[source.id] !== pixels) throw new Error('Navigation final canvas did not freeze.');
              finalCanvases[source.id] = pixels;
              if (step > local) frozen++;
            }
          }
          compared++;
        }
      }
      return {task: example.game, case_id: example.id, shared_steps_checked: max + 1, frame_metadata_checks: compared,
        frozen_terminal_canvas_checks: frozen, frozen_terminal_metadata_checks: frozenMetadata,
        source_lengths: Object.fromEntries(example.systems.map(system => [system.id, system.frames.length])),
        outcomes: Object.fromEntries(example.systems.map(system => [system.id, system.summary || {}]))};
    }, {example, exampleIndex, hardMode});
    if (hardMode && example.game === 'snake') {
      assert.doesNotMatch(await page.locator('body').innerText(), /\bthree bites\b|\bthree food\b|\b3[- ]food\b|\bfood\s*\/\s*3(?:\s|$)|\bfood target reached\b/i,
        'The restored continuing-food challenge must not display stale three-food target text.');
      result.no_stale_three_food_target_text = true;
    }
    result.controls = await controls('navigation', Math.max(...example.systems.map(system => system.frames.length - 1)));
    const nanojev = example.systems.find(system => system.id === 'nanojev');
    const poster = nanojev.frames.length - 1;
    await page.evaluate(({index, step}) => window.nanojevComparison.setFrame(index, step), {index: exampleIndex, step: poster});
    await screenshot(`${example.game}_desktop`, await page.evaluate(() => window.nanojevComparison.getSnapshot()));
    await page.setViewportSize({width: 390, height: 844});
    await screenshot(`${example.game}_mobile`, await page.evaluate(() => window.nanojevComparison.getSnapshot()));
    await page.setViewportSize({width: 1600, height: 1200});
    if (hardMode) {
      const final = Math.max(...example.systems.map(system => system.frames.length - 1));
      await page.evaluate(({index, step}) => window.nanojevComparison.setFrame(index, step), {index: exampleIndex, step: final});
      await screenshot(`${example.game}_final_all_desktop`, await page.evaluate(() => window.nanojevComparison.getSnapshot()));
      await page.setViewportSize({width: 390, height: 844});
      await screenshot(`${example.game}_final_all_mobile`, await page.evaluate(() => window.nanojevComparison.getSnapshot()));
      await page.setViewportSize({width: 1600, height: 1200});
    }
    report.tasks.push(result);
    console.log(JSON.stringify({task: result.task, checked_steps: result.shared_steps_checked, checked_frames: result.frame_metadata_checks}));
  }

  for (const [task, filename, relative] of [['basic', 'shooting_results.json', 'index.html'],
    ['predict_position', 'predict_position_results.json', 'predict-position.html']]) {
    const data = await getPageData(filename, relative); await ready('shooting');
    if (hardMode) {
      if (task === 'basic') assert.equal(data.cases.length, 3, 'Basic must preserve the existing three selected cases.');
      if (task === 'predict_position') {
        assert.deepEqual(data.cases.map(item => item.id).sort(), ['test-sonic_predict_position-9300720', 'test-sonic_predict_position-9300738']);
        for (const item of data.cases) {
          assert.equal(item.systems.find(system => system.id === 'nanojev').success, true);
          assert.equal(item.systems.find(system => system.id === 'jev').success, false);
          assert.equal(item.systems.find(system => system.id === 'base').success, false);
        }
      }
    }
    const defaultCase = data.cases.find(item => item.id === data.default_case_id);
    assert.ok(defaultCase, `${task}: default case must exist.`);
    assert.equal(defaultCase.systems.find(system => system.id === 'nanojev').success, true);
    assert.equal(defaultCase.systems.find(system => system.id === 'jev').success, false);
    assert.equal(defaultCase.systems.find(system => system.id === 'base').success, false);
    const initialSnapshot = await page.evaluate(() => window.nanojevShooting.getSnapshot());
    assert.equal(initialSnapshot.caseId, data.default_case_id);
    assert.equal(initialSnapshot.tick, 0); assert.equal(initialSnapshot.playing, false);
    assert.equal(await page.locator('#caseSelect option').count(), data.cases.length);
    const result = {task, default_case: data.default_case_id, default_is_nanojev_only_success: true, cases: []};
    if (await page.locator('#benchmarkStrip').count()) {
      assert.equal(await page.locator('#benchmarkStrip').isVisible(), true);
      for (const model of data.models) {
        const cell = page.locator(`#benchmarkStrip .benchmark-cell[data-model="${model.id}"]`);
        assert.equal(await cell.getAttribute('data-successes'), String(data.summary[model.id].test.successes));
        assert.equal(await cell.getAttribute('data-episodes'), String(data.summary[model.id].test.episodes));
        assert.equal(await cell.locator('.benchmark-rate').innerText(), `${(100 * data.summary[model.id].test.successes / data.summary[model.id].test.episodes).toFixed(1)}%`);
      }
      result.full_benchmark_verified = true;
    }
    for (const item of data.cases) {
      await page.evaluate(id => window.nanojevShooting.setCase(id), item.id);
      const checked = await page.evaluate(async ({item, task}) => {
        const max = Math.max(...item.systems.map(system => system.total_ticks));
        const canvas = document.createElement('canvas'); canvas.width = 320; canvas.height = 240;
        const context = canvas.getContext('2d', {alpha: false, willReadFrequently: true});
        const images = new Map(), crops = new Set();
        const shots = item.systems.flatMap(system => system.frames.flatMap((frame, index) => index > 0 &&
          Number.isFinite(frame.ammo) && Number.isFinite(system.frames[index - 1].ammo) && frame.ammo < system.frames[index - 1].ammo ? [{model: system.id, tick: frame.tick}] : []));
        if (task === 'predict_position') {
          const markers = [...document.querySelectorAll('.shot-marker')].map(element => `${element.dataset.model}:${element.dataset.tick}`).sort();
          if (JSON.stringify(markers) !== JSON.stringify(shots.map(shot => `${shot.model}:${shot.tick}`).sort())) throw new Error('Shot marker must be an actual ammo decrease.');
        }
        let compared = 0, pixels = 0, held = 0, frozen = 0;
        for (let tick = 0; tick <= max; tick++) {
          const snapshot = window.nanojevShooting.setTick(tick);
          if (snapshot.tick !== tick || snapshot.playing || snapshot.caseId !== item.id) throw new Error('Shooting shared-clock mismatch.');
          for (const source of item.systems) {
            const frame = source.frames[Math.min(tick, source.total_ticks)], actual = snapshot.systems.find(system => system.id === source.id);
            if (JSON.stringify(actual.frame) !== JSON.stringify(frame)) throw new Error('Shooting source-frame mismatch.');
            const card = document.querySelector(`.model-card[data-model="${source.id}"]`);
            if (+card.dataset.tick !== frame.tick || +card.dataset.imageTick !== frame.image_tick || card.dataset.action !== (frame.action || '') ||
              card.dataset.terminal !== String(frame.terminal)) throw new Error('Shooting rendered timing mismatch.');
            if (card.querySelector('.stat-ammo').textContent !== (frame.ammo == null ? '—' : String(frame.ammo))) throw new Error('Shooting ammo display mismatch.');
            const note = card.querySelector('.image-note');
            if (note.hidden !== (frame.image_tick === frame.tick)) throw new Error('Held real image disclosure mismatch.');
            if (frame.image_tick !== frame.tick) held++;
            if (tick > source.total_ticks) frozen++;
            for (const row of card.querySelectorAll('.probability-row')) {
              const expected = frame.probabilities?.[row.dataset.action];
              if (row.dataset.probability !== (expected == null ? '' : String(expected)) ||
                row.classList.contains('selected') !== (frame.action === row.dataset.action)) throw new Error('Shooting probability or executed-action mismatch.');
            }
            if (task === 'predict_position') {
              const launch = shots.find(shot => shot.model === source.id)?.tick;
              const launched = launch !== undefined && frame.tick >= launch;
              const phase = frame.terminal ? (source.success ? 'Target hit' : 'Target missed') : launched ? 'Rocket fired' : 'Rocket ready';
              if (card.querySelector('.shot-phase').textContent !== phase || card.dataset.launched !== String(launched)) throw new Error('Rocket phase does not match recorded launch/outcome.');
            }
            const sprite = frame.sprite, key = `${source.id}:${sprite.src}:${sprite.x}:${sprite.y}`;
            if (!crops.has(key)) {
              if (!images.has(sprite.src)) { const image = new Image(); image.src = sprite.src; await image.decode(); images.set(sprite.src, image); }
              const image = images.get(sprite.src);
              if (sprite.x + 320 > image.naturalWidth || sprite.y + 240 > image.naturalHeight) throw new Error('Sprite crop exceeds its real image atlas.');
              context.drawImage(image, sprite.x, sprite.y, 320, 240, 0, 0, 320, 240);
              const expected = new Uint32Array(context.getImageData(0, 0, 320, 240).data.buffer);
              const rendered = new Uint32Array(card.querySelector('canvas').getContext('2d').getImageData(0, 0, 320, 240).data.buffer);
              for (let i = 0; i < expected.length; i++) if (expected[i] !== rendered[i]) throw new Error(`Real canvas pixel mismatch: ${source.id}/${tick}`);
              crops.add(key); pixels++;
            }
            compared++;
          }
        }
        return {case_id: item.id, physical_ticks_checked: max + 1, frame_metadata_checks: compared,
          unique_canvas_crop_checks: pixels, held_image_checks: held, terminal_hold_checks: frozen,
          shot_markers: task === 'predict_position' ? shots : []};
      }, {item, task});
      for (const shot of checked.shot_markers) {
        await page.locator(`.shot-marker[data-model="${shot.model}"][data-tick="${shot.tick}"]`).click();
        assert.equal((await page.evaluate(() => window.nanojevShooting.getSnapshot())).tick, shot.tick);
      }
      result.cases.push(checked);
    }
    const alternate = data.cases.find(item => item.id !== data.default_case_id);
    if (alternate) {
      await page.locator('#caseSelect').selectOption(alternate.id);
      await page.waitForFunction(id => window.nanojevShooting?.ready && window.nanojevShooting.getSnapshot().caseId === id, alternate.id);
      result.visible_case_selector_verified = true;
    }
    assert.equal(await page.locator('#playFeatured').isVisible(), true);
    await page.locator('#playFeatured').click();
    await page.waitForFunction(id => window.nanojevShooting?.ready && window.nanojevShooting.getSnapshot().caseId === id &&
      window.nanojevShooting.getSnapshot().playing, data.default_case_id);
    await page.evaluate(() => window.nanojevShooting.pause());
    result.featured_button_resets_to_default_and_plays = true;
    await page.evaluate(id => window.nanojevShooting.setCase(id), data.default_case_id);
    result.controls = await controls('shooting', Math.max(...defaultCase.systems.map(system => system.total_ticks)));
    const nanojev = defaultCase.systems.find(system => system.id === 'nanojev');
    await page.evaluate(tick => window.nanojevShooting.setTick(tick), nanojev.total_ticks);
    await screenshot(`${task}_desktop`, await page.evaluate(() => window.nanojevShooting.getSnapshot()));
    await page.setViewportSize({width: 390, height: 844});
    await screenshot(`${task}_mobile`, await page.evaluate(() => window.nanojevShooting.getSnapshot()));
    await page.setViewportSize({width: 1600, height: 1200});
    report.tasks.push(result);
    console.log(JSON.stringify({task, cases: result.cases.length, checked_frames: result.cases.reduce((n, item) => n + item.frame_metadata_checks, 0)}));
  }

  const routes = {maze: 'side-by-side.html#maze', snake: 'side-by-side.html#snake', basic: 'index.html', predict_position: 'predict-position.html'};
  function identify(text) {
    const value = text.trim().toLowerCase();
    if (value.includes('predict')) return 'predict_position';
    if (value.includes('maze')) return 'maze';
    if (value.includes('snake')) return 'snake';
    if (value.includes('basic')) return 'basic';
    return null;
  }
  for (const [sourceTask, route] of Object.entries(routes)) {
    await page.goto(new URL(route, base).href, {waitUntil: 'networkidle'});
    await ready(['maze', 'snake'].includes(sourceTask) ? 'navigation' : 'shooting');
    const items = await page.locator('nav a, nav button[data-game]').evaluateAll(elements => elements.map((element, index) => ({
      index, text: element.textContent.trim(), href: element.tagName === 'A' ? element.href : null, game: element.dataset.game || null})));
    const taskItems = Object.fromEntries(items.map(item => [identify(item.text), item]).filter(([task]) => task));
    assert.deepEqual(Object.keys(taskItems).sort(), Object.keys(routes).sort(), `${sourceTask}: navigation must expose all four tasks.`);
    for (const [targetTask, item] of Object.entries(taskItems)) {
      if (item.href) {
        const target = new URL(item.href);
        assert.equal(target.origin, base.origin, 'Task navigation must stay on the development site.');
        assert.ok(target.pathname.startsWith(base.pathname), 'Task navigation must stay inside the development directory.');
      }
      await page.locator('nav a, nav button[data-game]').nth(item.index).click();
      const kind = ['maze', 'snake'].includes(targetTask) ? 'navigation' : 'shooting';
      await ready(kind);
      if (kind === 'navigation') {
        await page.waitForFunction(game => window.nanojevComparison?.ready && window.nanojevComparison.getSnapshot().game === game, targetTask);
        assert.equal(new URL(page.url()).hash, '#' + targetTask);
      } else {
        const actual = await page.evaluate(() => window.nanojevShooting.getSnapshot());
        const filename = targetTask === 'basic' ? 'shooting_results.json' : 'predict_position_results.json';
        const raw = await context.request.get(new URL(filename, base).href);
        assert.equal(actual.caseId, (await raw.json()).default_case_id);
      }
      report.navigation.push({from: sourceTask, to: targetTask, passed: true, url: page.url()});
      await page.goto(new URL(route, base).href, {waitUntil: 'networkidle'});
      await ready(['maze', 'snake'].includes(sourceTask) ? 'navigation' : 'shooting');
    }
  }
  assert.deepEqual(report.page_errors, []); assert.deepEqual(report.failed_asset_requests, []);
  assert.deepEqual(report.tasks.map(task => task.task).sort(), ['basic', 'maze', 'predict_position', 'snake']);
  report.passed = true;
} catch (error) {
  report.failure = {name: error.name, message: error.message}; process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  report.completed_at = new Date().toISOString();
  await fs.mkdir(path.dirname(reportPath), {recursive: true});
  await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n', {flag: 'wx'});
  console.log(JSON.stringify({passed: report.passed, report: reportPath, output,
    tasks: report.tasks.map(task => task.task), navigation_checks: report.navigation.length, failure: report.failure}));
}
