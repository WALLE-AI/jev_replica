#!/usr/bin/env node
/** Capture looping README GIFs from the real development webpage and recorded trajectories. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath, pathToFileURL} from 'node:url';

const HELP = `Usage:
  node scripts/capture_readme_autoplay.mjs --web-root web/dev --output assets \\
    --work runs/readme_autoplay_v1 --manifest results/readme_autoplay_v1/capture_manifest.json \\
    --playwright-module /path/to/playwright/index.mjs --chrome /path/to/chrome --ffmpeg /path/to/ffmpeg

Creates maze_unified_autoplay.gif and predict_position_unified_autoplay.gif by default.
Use --tasks basic to create only basic_unified_autoplay.gif; task names may be comma-separated.
Use --fps 10 for a smaller GIF; the default frame rate is 12.5.
Every displayed state is an original webpage render on a shared physical timeline.
The existing website, recordings, and output files are never modified.
`;
if (process.argv.includes('--help')) { console.log(HELP); process.exit(0); }
const args = {};
for (let index = 2; index < process.argv.length; index += 2) {
  const key = process.argv[index];
  assert.ok(['--web-root', '--output', '--work', '--manifest', '--playwright-module', '--chrome', '--ffmpeg', '--tasks', '--fps'].includes(key), `Unknown option: ${key}`);
  assert.ok(process.argv[index + 1] && !process.argv[index + 1].startsWith('--'), `Missing value: ${key}`);
  assert.ok(!(key.slice(2) in args), `Duplicate option: ${key}`);
  args[key.slice(2)] = process.argv[index + 1];
}
assert.ok(args['playwright-module'] && args.chrome && args.ffmpeg, 'Provide local Playwright, Chrome, and FFmpeg paths.');
const root = await fs.realpath(path.resolve(args['web-root'] || 'web/dev'));
const output = path.resolve(args.output || 'assets'), work = path.resolve(args.work || 'runs/readme_autoplay_v1');
const manifestPath = path.resolve(args.manifest || 'results/readme_autoplay_v1/capture_manifest.json');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const exists = async filename => { try { await fs.lstat(filename); return true; } catch (error) { if (error.code === 'ENOENT') return false; throw error; } };
const CHECKPOINT = 'f68c47d66998231b86b7e91b4ed5e82ae23acf104c8b7cd6d165c3ac7b7ffe1b';
const ORDER = ['jev', 'nanojev', 'base'], FPS = Number(args.fps || 12.5), OUTPUT_WIDTH = 1200;
assert.ok(Number.isFinite(FPS) && FPS >= 1 && FPS <= 50 && Number.isInteger(100 / FPS),
  'The GIF frame rate must be between 1 and 50 and have an exact integer-centisecond frame delay.');
const INITIAL_FRAMES = Math.round(FPS), FINAL_FRAMES = Math.round(2 * FPS);
const availableJobs = [
  {id: 'maze', html: 'side-by-side.html', data: 'side_by_side_results.json', api: 'nanojevComparison',
    hash: '#maze', caseId: 'maze:ood:50:24310922', rate: 256, selectorSpeed: '256', output: 'maze_unified_autoplay.gif',
    rendererAssets: ['side-by-side.css', 'side-by-side.js']},
  {id: 'predict_position', html: 'predict-position.html', data: 'predict_position_results.json', api: 'nanojevShooting',
    hash: '', caseId: 'test-sonic_predict_position-9300720', rate: 17.5, selectorSpeed: '0.5', output: 'predict_position_unified_autoplay.gif',
    rendererAssets: ['predict-position.css', 'predict-position.js', 'shooting.css', 'shooting.js']},
  {id: 'basic', html: 'index.html', data: 'shooting_results.json', api: 'nanojevShooting',
    hash: '', caseId: 'test-appo_basic-9030060', rate: 17.5, selectorSpeed: '0.5', output: 'basic_unified_autoplay.gif',
    rendererAssets: ['predict-position.css', 'basic.js', 'shooting.css', 'shooting.js']},
];
const selectedTasks = (args.tasks || 'maze,predict_position').split(',');
assert.equal(new Set(selectedTasks).size, selectedTasks.length, 'Task names must be unique.');
const jobs = selectedTasks.map(id => {
  const job = availableJobs.find(item => item.id === id);
  assert.ok(job, `Unknown task: ${id}`);
  return job;
});
assert.ok(!await exists(work), `Use a new work directory: ${work}`);
assert.ok(!await exists(manifestPath), `Refusing to overwrite ${manifestPath}`);
for (const job of jobs) assert.ok(!await exists(path.join(output, job.output)), `Refusing to overwrite ${job.output}`);
const assetNames = [...new Set(jobs.flatMap(job => [job.html, job.data, ...job.rendererAssets]))];
const assets = new Map();
for (const name of assetNames) assets.set(name, await fs.readFile(path.join(root, name)));
for (const job of jobs) {
  const data = JSON.parse(assets.get(job.data));
  assert.equal(data.protocol.selected_checkpoint_sha256, CHECKPOINT);
  job.dataObject = data;
  job.exampleIndex = job.id === 'maze' ? data.examples.findIndex(item => item.id === job.caseId) : null;
  job.item = job.id === 'maze' ? data.examples[job.exampleIndex] : data.cases.find(item => item.id === job.caseId);
  assert.ok(job.item, `The fixed ${job.id} case must exist.`);
  assert.deepEqual(job.item.systems.map(system => system.id).sort(), [...ORDER].sort());
  if (job.id === 'maze') assert.equal(job.item.size, 50);
  else {
    assert.equal(data.default_case_id, job.caseId);
    for (const system of job.item.systems) for (const frame of system.frames) {
      const name = frame.sprite.src;
      if (!assets.has(name)) {
        const resolved = await fs.realpath(path.join(root, name));
        assert.ok(!path.relative(root, resolved).startsWith('..'));
        assets.set(name, await fs.readFile(resolved));
      }
    }
  }
  job.terminal = Object.fromEntries(job.item.systems.map(system => [system.id,
    job.id === 'maze' ? system.frames.length - 1 : system.total_ticks]));
  job.maxStep = Math.max(...Object.values(job.terminal));
  job.movementFrames = Math.ceil(job.maxStep * FPS / job.rate);
  job.frameSteps = [...Array(INITIAL_FRAMES).fill(0),
    ...Array.from({length: job.movementFrames}, (_, index) => Math.min(job.maxStep, Math.floor((index + 1) * job.rate / FPS))),
    ...Array(FINAL_FRAMES).fill(job.maxStep)];
}
await fs.mkdir(output, {recursive: true}); await fs.mkdir(path.dirname(work), {recursive: true}); await fs.mkdir(work);
const manifest = {schema: 'nanojev-readme-autoplay-capture-v1', passed: false, started_at: new Date().toISOString(),
  checkpoint_sha256: CHECKPOINT, model_calls: 0, api_calls: 0, generated_gameplay_images: 0,
  capture_script_sha256: digest(await fs.readFile(fileURLToPath(import.meta.url))),
  renderer_asset_sha256: Object.fromEntries([...assets].map(([name, bytes]) => [name, digest(bytes)])),
  rendering: 'Actual original webpage screenshots, cropped to the scene title, three model panels, and shared playback controls. No DOM or CSS changes.',
  encoding: 'Only spatial scaling and GIF color quantization; no interpolation or synthetic game frames.',
  frame_rate: FPS, gif_frame_delay_centiseconds: 100 / FPS, output_width: OUTPUT_WIDTH,
  initial_hold_seconds: INITIAL_FRAMES / FPS, final_hold_seconds: FINAL_FRAMES / FPS,
  jobs: [], browser_checks: [], page_errors: [], failed_requests: []};
const runFFmpeg = command => {
  const result = spawnSync(args.ffmpeg, ['-hide_banner', ...command], {encoding: 'utf8', maxBuffer: 8 * 1024 * 1024});
  if (result.error) throw result.error;
  assert.equal(result.status, 0, `FFmpeg failed: ${(result.stderr || '').slice(-4000)}`);
  return result;
};
function inspectGif(bytes) {
  assert.ok(['GIF87a', 'GIF89a'].includes(bytes.toString('ascii', 0, 6)));
  const width = bytes.readUInt16LE(6), height = bytes.readUInt16LE(8);
  let offset = 13, pendingDelay = 0, frames = 0, durationCentiseconds = 0, loopCount = null;
  if (bytes[10] & 0x80) offset += 3 * 2 ** ((bytes[10] & 7) + 1);
  const blocks = () => {
    const chunks = [];
    while (true) { const size = bytes[offset++]; if (!size) break; chunks.push(bytes.subarray(offset, offset + size)); offset += size; }
    return Buffer.concat(chunks);
  };
  while (offset < bytes.length) {
    const marker = bytes[offset++];
    if (marker === 0x3b) break;
    if (marker === 0x21) {
      const label = bytes[offset++];
      if (label === 0xf9) {
        assert.equal(bytes[offset++], 4); offset++; pendingDelay = bytes.readUInt16LE(offset); offset += 3;
        assert.equal(bytes[offset++], 0);
      } else if (label === 0xff) {
        const count = bytes[offset++], application = bytes.toString('ascii', offset, offset + count); offset += count;
        const data = blocks();
        if (application === 'NETSCAPE2.0') { assert.equal(data[0], 1); loopCount = data.readUInt16LE(1); }
      } else blocks();
    } else if (marker === 0x2c) {
      const packed = bytes[offset + 8]; offset += 9;
      if (packed & 0x80) offset += 3 * 2 ** ((packed & 7) + 1);
      offset++; blocks(); frames++; durationCentiseconds += pendingDelay; pendingDelay = 0;
    } else throw new Error(`Unexpected GIF marker ${marker} at ${offset - 1}`);
  }
  return {width, height, frames, duration_seconds: durationCentiseconds / 100, loop_count: loopCount};
}
let browser, server;
try {
  const types = {'.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.json': 'application/json', '.webp': 'image/webp'};
  server = http.createServer((request, response) => {
    const name = decodeURIComponent(new URL(request.url, 'http://localhost').pathname).slice(1), body = assets.get(name);
    if (!body || !['GET', 'HEAD'].includes(request.method)) { response.writeHead(404).end(); return; }
    response.writeHead(200, {'Content-Type': types[path.extname(name)] || 'application/octet-stream', 'Cache-Control': 'no-store'});
    response.end(request.method === 'HEAD' ? undefined : body);
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  const origin = `http://127.0.0.1:${server.address().port}/`;
  const {chromium} = await import(pathToFileURL(path.resolve(args['playwright-module'])).href);
  browser = await chromium.launch({headless: true, executablePath: args.chrome,
    args: ['--disable-background-networking', '--no-first-run', '--no-default-browser-check']});
  manifest.browser = browser.version();
  for (const job of jobs) {
    const taskDirectory = path.join(work, job.id), captures = path.join(taskDirectory, 'captures'), frames = path.join(taskDirectory, 'frames');
    await fs.mkdir(taskDirectory); await fs.mkdir(captures); await fs.mkdir(frames);
    const context = await browser.newContext({viewport: {width: 1320, height: 1800}, deviceScaleFactor: 1});
    const page = await context.newPage();
    page.on('pageerror', error => manifest.page_errors.push({task: job.id, message: error.message}));
    page.on('requestfailed', request => manifest.failed_requests.push({task: job.id, url: request.url(), error: request.failure()?.errorText}));
    const ready = async () => {
      await page.waitForFunction(name => window[name]?.ready || window[name]?.error, job.api);
      assert.equal(await page.evaluate(name => window[name].error, job.api), null);
    };
    await page.goto(origin + job.html + job.hash, {waitUntil: 'networkidle'}); await ready();
    let snapshot = await page.evaluate(name => window[name].getSnapshot(), job.api);
    assert.equal(snapshot.playing, false); assert.equal(job.id === 'maze' ? snapshot.globalStep : snapshot.tick, 0);
    await page.waitForTimeout(150);
    snapshot = await page.evaluate(name => window[name].getSnapshot(), job.api);
    assert.equal(snapshot.playing, false); assert.equal(job.id === 'maze' ? snapshot.globalStep : snapshot.tick, 0);
    await page.locator('#speed').selectOption(job.selectorSpeed);
    if (job.id === 'maze') assert.equal(await page.locator('#showTrail').isChecked(), true);
    const clip = await page.evaluate(task => {
      const first = document.querySelector(task === 'maze' ? '.scene-heading' : '.case-toolbar').getBoundingClientRect();
      const last = document.querySelector(task === 'maze' ? '.viewer-footer' : '.viewer-note').getBoundingClientRect();
      return {x: Math.max(0, Math.floor(first.left - 12)), y: Math.max(0, Math.floor(first.top - 12)),
        width: Math.ceil(first.width + 24), height: Math.ceil(last.bottom - first.top + 24)};
    }, job.id);
    assert.ok(clip.x + clip.width <= 1320 && clip.y + clip.height <= 1800, 'The scene crop must fit inside the real browser viewport.');
    const record = {task: job.id, case_id: job.caseId, source_json_sha256: digest(assets.get(job.data)),
      selected_checkpoint_sha256: CHECKPOINT, source_terminal_steps: job.terminal,
      shared_timebase: job.id === 'maze' ? 'physical environment attempts including collisions' : 'physical game ticks at 35 Hz',
      displayed_playback_speed: job.selectorSpeed, physical_steps_per_video_second: job.rate,
      source_game_ticks_per_second: job.id === 'maze' ? null : 35, viewport: {width: 1320, height: 1800, device_scale_factor: 1},
      crop: clip, requested_output_frames: job.frameSteps.length, requested_duration_seconds: job.frameSteps.length / FPS,
      initial_hold_frames: INITIAL_FRAMES, movement_frames: job.movementFrames, final_hold_frames: FINAL_FRAMES,
      sampled_shared_steps: job.frameSteps, unique_sampled_shared_steps: [...new Set(job.frameSteps)],
      sampling_rule: 'Fixed shared speed. Movement output frame i samples floor((i+1)*physical_steps_per_video_second/fps), clamped at the true global terminal.',
      terminal_freeze: 'Each panel retains its own exact terminal source frame while the common clock continues.',
      source_outcomes: Object.fromEntries(job.item.systems.map(system => [system.id, job.id === 'maze' ? system.summary :
        {success: system.success, total_ticks: system.total_ticks, total_decisions: system.total_decisions}])),
      verification: {exact_source_frame_comparisons: 0, original_canvas_pixel_comparisons: 0, terminal_panels_observed: {},
        held_image_disclosures: 0, probability_displays_checked: 0}, encoded_attempts: []};
    const captured = new Map(), frameDigest = createHash('sha256');
    for (let index = 0; index < job.frameSteps.length; index++) {
      const step = job.frameSteps[index];
      if (!captured.has(step)) {
        snapshot = await page.evaluate(({task, api, exampleIndex, step}) => task === 'maze' ?
          window[api].setFrame(exampleIndex, step) : window[api].setTick(step),
        {task: job.id, api: job.api, exampleIndex: job.exampleIndex, step});
        assert.equal(snapshot.playing, false); assert.equal(snapshot.ready, true);
        assert.equal(job.id === 'maze' ? snapshot.globalStep : snapshot.tick, step);
        assert.equal(job.id === 'maze' ? snapshot.exampleId : snapshot.caseId, job.caseId);
        assert.equal(job.id === 'maze' ? snapshot.stepsPerSecond : snapshot.speed, Number(job.selectorSpeed));
        assert.deepEqual(snapshot.systems.map(system => system.id), ORDER);
        for (const system of snapshot.systems) {
          const source = job.item.systems.find(item => item.id === system.id), localStep = Math.min(step, job.terminal[system.id]);
          assert.deepEqual(system.frame, source.frames[localStep]); record.verification.exact_source_frame_comparisons++;
          if (step >= job.terminal[system.id]) record.verification.terminal_panels_observed[system.id] = true;
        }
        const verification = await page.evaluate(async ({task, snapshot}) => {
          let probabilityDisplays = 0, pixels = 0, held = 0;
          for (const system of snapshot.systems) {
            const card = document.querySelector(task === 'maze' ? `.model-panel[data-system="${system.id}"]` : `.model-card[data-model="${system.id}"]`);
            const frame = system.frame;
            if (task === 'maze') {
              const directions = ['north', 'east', 'south', 'west'];
              [...card.querySelectorAll('.probability-item')].forEach((row, index) => {
                const expected = system.localStep ? frame.probabilities?.[directions[index]] : undefined;
                if (row.querySelector('.probability-value').textContent !== (expected === undefined ? '—' : `${(100 * expected).toFixed(1)}%`)) throw new Error('Maze probability display differs from the source.');
                if (row.classList.contains('chosen') !== (system.localStep > 0 && frame.action === directions[index])) throw new Error('Maze actual-action highlight differs from the source.');
                probabilityDisplays++;
              });
              if (card.querySelector('.stat-steps').textContent !== String(system.localStep)) throw new Error('Maze local step differs from the source.');
            } else {
              for (const row of card.querySelectorAll('.probability-row')) {
                const expected = frame.probabilities?.[row.dataset.action];
                if (row.dataset.probability !== (expected == null ? '' : String(expected)) ||
                  row.classList.contains('selected') !== (frame.action === row.dataset.action)) throw new Error('Shooting probability/action display differs from the source.');
                probabilityDisplays++;
              }
              const note = card.querySelector('.image-note');
              if (note.hidden !== (frame.image_tick === frame.tick)) throw new Error('Held image must disclose the actual original image tick.');
              if (frame.image_tick !== frame.tick) held++;
              window.__readmeSourceImages ||= new Map();
              const sprite = frame.sprite;
              if (!window.__readmeSourceImages.has(sprite.src)) {
                const image = new Image(); image.src = sprite.src; await image.decode(); window.__readmeSourceImages.set(sprite.src, image);
              }
              const canvas = document.createElement('canvas'); canvas.width = 320; canvas.height = 240;
              const ctx = canvas.getContext('2d', {alpha: false, willReadFrequently: true});
              ctx.drawImage(window.__readmeSourceImages.get(sprite.src), sprite.x, sprite.y, 320, 240, 0, 0, 320, 240);
              const expected = new Uint32Array(ctx.getImageData(0, 0, 320, 240).data.buffer);
              const actual = new Uint32Array(card.querySelector('canvas').getContext('2d').getImageData(0, 0, 320, 240).data.buffer);
              for (let i = 0; i < expected.length; i++) if (expected[i] !== actual[i]) throw new Error('Shooting canvas pixels differ from the real original image.');
              pixels++;
            }
          }
          return {probabilityDisplays, pixels, held};
        }, {task: job.id, snapshot});
        record.verification.probability_displays_checked += verification.probabilityDisplays;
        record.verification.original_canvas_pixel_comparisons += verification.pixels;
        record.verification.held_image_disclosures += verification.held;
        const filename = path.join(captures, `step_${String(step).padStart(6, '0')}.png`);
        const png = await page.screenshot({path: filename, clip, animations: 'disabled'});
        frameDigest.update(`${step}:${digest(png)}\n`); captured.set(step, filename);
        if (captured.size === 1 || captured.size % 50 === 0) console.log(JSON.stringify({task: job.id, captured_frames: captured.size, shared_step: step, max_step: job.maxStep}));
      }
      await fs.link(captured.get(step), path.join(frames, `frame_${String(index).padStart(6, '0')}.png`));
    }
    assert.ok(ORDER.every(id => record.verification.terminal_panels_observed[id]));
    assert.equal(job.frameSteps.at(-1), job.maxStep);
    record.verification.unique_screenshot_frames = captured.size;
    record.verification.captured_png_sequence_sha256 = frameDigest.digest('hex');
    record.verification.png_hash_sequence_encoding = 'Unique sampled shared steps in order: UTF-8 step:PNG_SHA256 plus newline.';
    await page.goto(origin + job.html + '?autoplay=1' + job.hash, {waitUntil: 'networkidle'}); await ready();
    await page.waitForFunction(({api, task}) => { const state = window[api].getSnapshot(); return (task === 'maze' ? state.globalStep : state.tick) > 0; }, {api: job.api, task: job.id});
    const autoplay = await page.evaluate(name => window[name].getSnapshot(), job.api);
    assert.equal(autoplay.playing, true);
    manifest.browser_checks.push({task: job.id, default_paused_at_zero: true, autoplay_query_advances_clock: true,
      autoplay_observed_step: job.id === 'maze' ? autoplay.globalStep : autoplay.tick});
    await context.close();
    const scale = `scale=${OUTPUT_WIDTH}:-2:flags=lanczos`;
    // The moving Doom scene needs a full-frame palette so static success colors survive quantization.
    const paletteStatistics = job.id === 'maze' ? 'diff' : 'full';
    const paletteSizes = job.id === 'maze' ? [128, 96, 64] : [256, 128, 96, 64];
    let selectedFile;
    for (const colors of paletteSizes) {
      const palette = path.join(taskDirectory, `palette_${colors}.png`), candidate = path.join(taskDirectory, `${job.id}_${colors}.gif`);
      runFFmpeg(['-loglevel', 'error', '-n', '-framerate', String(FPS), '-i', path.join(frames, 'frame_%06d.png'),
        '-vf', `${scale},palettegen=max_colors=${colors}:reserve_transparent=1:stats_mode=${paletteStatistics}`, '-frames:v', '1', '-update', '1', palette]);
      runFFmpeg(['-loglevel', 'error', '-n', '-framerate', String(FPS), '-i', path.join(frames, 'frame_%06d.png'), '-i', palette,
        '-lavfi', `[0:v]${scale}[scaled];[scaled][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle`,
        '-frames:v', String(job.frameSteps.length), '-gifflags', '+offsetting+transdiff', '-loop', '0', candidate]);
      const bytes = await fs.readFile(candidate), parsed = inspectGif(bytes);
      assert.equal(parsed.frames, job.frameSteps.length); assert.equal(parsed.width, OUTPUT_WIDTH); assert.equal(parsed.loop_count, 0);
      assert.ok(Math.abs(parsed.duration_seconds - job.frameSteps.length / FPS) <= .011);
      record.encoded_attempts.push({max_colors: colors, palette_statistics: paletteStatistics, bytes: bytes.length, sha256: digest(bytes), ...parsed});
      selectedFile = candidate;
      if (bytes.length < 10_000_000) break;
    }
    const gif = await fs.readFile(selectedFile), parsed = inspectGif(gif);
    const decoded = runFFmpeg(['-loglevel', 'error', '-nostats', '-progress', 'pipe:1', '-ignore_loop', '1', '-i', selectedFile,
      '-map', '0:v:0', '-f', 'null', '-']);
    const decodedFrames = [...decoded.stdout.matchAll(/^frame=(\d+)$/gm)].map(match => Number(match[1])).at(-1);
    assert.equal(decodedFrames, job.frameSteps.length); assert.ok(decoded.stdout.includes('progress=end'));
    await fs.writeFile(path.join(output, job.output), gif, {flag: 'wx'});
    const selectedAttempt = record.encoded_attempts.at(-1);
    record.gif = {file: job.output, bytes: gif.length, sha256: digest(gif), ...parsed,
      max_colors: selectedAttempt.max_colors, palette_statistics: paletteStatistics, dither: 'bayer', bayer_scale: 4, scale_filter: 'lanczos',
      diff_mode: 'rectangle', gif_flags: ['offsetting', 'transdiff'], full_decode_passed: true,
      decoded_frames: decodedFrames, under_ten_megabytes: gif.length < 10_000_000};
    const times = [{label: 'initial', seconds: 0}, {label: 'nanojev_terminal', seconds: INITIAL_FRAMES / FPS + job.terminal.nanojev / job.rate + .1},
      {label: 'all_terminal', seconds: parsed.duration_seconds - 1}];
    record.preview_frames = [];
    for (const preview of times) {
      const filename = path.join(taskDirectory, `preview_${preview.label}.png`);
      runFFmpeg(['-loglevel', 'error', '-n', '-ignore_loop', '1', '-ss', String(preview.seconds), '-i', path.join(output, job.output), '-frames:v', '1', filename]);
      record.preview_frames.push({label: preview.label, seconds: preview.seconds, file: path.relative(work, filename),
        sha256: digest(await fs.readFile(filename))});
    }
    manifest.jobs.push(record);
    console.log(JSON.stringify({task: job.id, gif: path.join(output, job.output), ...record.gif}));
  }
  assert.deepEqual(manifest.page_errors, []); assert.deepEqual(manifest.failed_requests, []);
  // Ensure the captured immutable assets still identify the final product files on disk.
  for (const [name, bytes] of assets) assert.equal(digest(await fs.readFile(path.join(root, name))), digest(bytes), `The product changed during capture: ${name}`);
  manifest.passed = true;
} catch (error) {
  manifest.failure = {name: error.name, message: error.message}; process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  if (server) await new Promise(resolve => server.close(resolve));
  manifest.completed_at = new Date().toISOString();
  await fs.mkdir(path.dirname(manifestPath), {recursive: true});
  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2) + '\n', {flag: 'wx'});
  console.log(JSON.stringify({passed: manifest.passed, manifest: manifestPath, jobs: manifest.jobs.map(job => job.task), failure: manifest.failure}));
}
