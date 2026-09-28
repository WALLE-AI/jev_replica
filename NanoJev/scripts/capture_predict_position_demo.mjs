#!/usr/bin/env node
// Capture original Predict Position gameplay on one shared physical clock.
// Browser playback is paused during deterministic capture. No inference occurs.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath, pathToFileURL} from 'node:url';

const HELP = `Usage:
  node scripts/capture_predict_position_demo.mjs --web-root web/dev --output assets \\
    --work runs/predict_position_video_capture --playwright-module /path/to/playwright/index.mjs \\
    --ffmpeg /path/to/ffmpeg --chrome /path/to/chrome [--case RECORDED_CASE_ID]

Creates predict_position.mp4 and predict_position_video_manifest.json.
Use a new work directory. Existing output files are never overwritten.
The complete recorded case plays at a visible 0.5x speed, preceded by a 1-second
initial hold and followed by a 3-second terminal hold. All panels share physical ticks.
`;
if (process.argv.includes('--help')) { console.log(HELP); process.exit(0); }
const args = {};
for (let index = 2; index < process.argv.length; index += 2) {
  const key = process.argv[index];
  assert.ok(['--web-root', '--output', '--work', '--playwright-module', '--ffmpeg', '--chrome', '--case'].includes(key), `Unknown option: ${key}`);
  assert.ok(process.argv[index + 1] && !process.argv[index + 1].startsWith('--'), `Missing value: ${key}`);
  assert.ok(!(key.slice(2) in args), `Duplicate option: ${key}`);
  args[key.slice(2)] = process.argv[index + 1];
}
assert.ok(args['playwright-module'] && args.ffmpeg && args.chrome, 'Playwright module, FFmpeg and Chrome paths are required.');
const root = await fs.realpath(path.resolve(args['web-root'] || 'web/dev'));
const output = path.resolve(args.output || 'assets');
const work = path.resolve(args.work || 'runs/predict_position_video_capture');
const movie = path.join(output, 'predict_position.mp4');
const manifestPath = path.join(output, 'predict_position_video_manifest.json');
const FPS = 30, WIDTH = 1600, SPEED = 0.5;
const ORDER = ['jev', 'nanojev', 'base'];
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const exists = async filename => { try { await fs.lstat(filename); return true; } catch (error) { if (error.code === 'ENOENT') return false; throw error; } };
assert.ok(!await exists(movie), `Refusing to overwrite ${movie}`);
assert.ok(!await exists(manifestPath), `Refusing to overwrite ${manifestPath}`);
assert.ok(!await exists(work), `Use a new capture work directory: ${work}`);
const dataBytes = await fs.readFile(path.join(root, 'predict_position_results.json'));
const data = JSON.parse(dataBytes.toString('utf8'));
assert.equal(data.schema || data.schema_version, 'nanojev-shooting-demo-v1');
const item = data.cases.find(item => item.id === (args.case || data.default_case_id));
assert.ok(item, 'The requested case must exist in the recorded data.');
assert.equal(item.systems.length, 3);
assert.deepEqual(item.systems.map(system => system.id).sort(), [...ORDER].sort());
const systems = ORDER.map(id => item.systems.find(system => system.id === id));
for (const system of systems) assert.equal(system.frames.length, system.total_ticks + 1);
const maxTick = Math.max(...systems.map(system => system.total_ticks));
const ticksPerSecond = data.protocol?.ticks_per_second ?? 35;
assert.ok(Number.isFinite(ticksPerSecond) && ticksPerSecond > 0);
const physicalTicksPerVideoSecond = ticksPerSecond * SPEED;
assert.ok(physicalTicksPerVideoSecond <= FPS, 'Every recorded physical tick must appear in the video.');
const movementFrames = Math.ceil(maxTick * FPS / physicalTicksPerVideoSecond);
const totalFrames = FPS + movementFrames + 3 * FPS;
assert.ok(Number.isSafeInteger(totalFrames) && totalFrames > 0, 'The output frame count must be a positive safe integer.');
const segments = [
  {kind: 'hold', label: 'initial_state', tick: 0, first_output_frame: 0, frame_count: FPS},
  {kind: 'movement', start_tick: 0, end_tick: maxTick, first_output_frame: FPS, frame_count: movementFrames,
    mapping: 'For zero-based segment frame i: min(end_tick, floor((i+1)*physical_ticks_per_video_second/fps)).'},
  {kind: 'hold', label: 'global_terminal', tick: maxTick, first_output_frame: FPS + movementFrames, frame_count: 3 * FPS},
];
const ffmpeg = command => {
  const result = spawnSync(args.ffmpeg, ['-hide_banner', ...command], {encoding: 'utf8', maxBuffer: 8 * 1024 * 1024});
  if (result.error) throw result.error;
  assert.equal(result.status, 0, `FFmpeg failed: ${(result.stderr || '').slice(-4000)}`);
  return result;
};
await fs.mkdir(output, {recursive: true});
await fs.mkdir(path.dirname(work), {recursive: true});
await fs.mkdir(work);
const capturesDirectory = path.join(work, 'captures'), framesDirectory = path.join(work, 'frames');
await fs.mkdir(capturesDirectory); await fs.mkdir(framesDirectory);
const report = {schema: 'nanojev-predict-position-video-v1', passed: false,
  data_sha256: digest(dataBytes), capture_script_sha256: digest(await fs.readFile(fileURLToPath(import.meta.url))),
  case_id: item.id, game: 'doom_predict_position', source_split: item.split, source_seed: item.seed,
  model_calls: 0, api_calls: 0, model_gpu_calls: 0,
  synchronization: 'One shared global physical tick; completed systems hold their exact final recorded frame.',
  frame_pixels: 'Original recorded canvas crops; no generated, interpolated, or enhanced gameplay images.',
  image_tick: 'A held original image retains its recorded image_tick, visibly disclosed by the webpage.',
  play_button_state: 'Paused during deterministic capture. The visible speed selector shows 0.5x.',
  fps: FPS, source_ticks_per_second: ticksPerSecond, playback_speed: SPEED,
  physical_ticks_per_video_second: physicalTicksPerVideoSecond,
  total_output_frames: totalFrames, requested_duration_seconds: totalFrames / FPS, segments,
  terminal_outcomes: systems.map(system => ({id: system.id, name: system.name, success: system.success,
    terminal_tick: system.total_ticks, total_decisions: system.total_decisions,
    terminal_frame_sha256: digest(JSON.stringify(system.frames.at(-1)))})),
  page_errors: [], verification: {unique_rendered_ticks: 0, exact_system_frame_comparisons: 0,
    canvas_pixel_comparisons: 0, terminal_ticks_included: {}, original_image_holds_checked: 0}};
let browser, server;
try {
  const types = {'.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
    '.json': 'application/json', '.webp': 'image/webp', '.png': 'image/png', '.svg': 'image/svg+xml', '.ico': 'image/x-icon'};
  server = http.createServer(async (request, response) => {
    try {
      if (!['GET', 'HEAD'].includes(request.method)) { response.writeHead(405).end(); return; }
      const requested = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
      const filename = await fs.realpath(path.join(root, requested === '/' ? 'predict-position.html' : requested.slice(1)));
      const relative = path.relative(root, filename);
      if (relative.startsWith('..') || path.isAbsolute(relative)) { response.writeHead(403).end(); return; }
      const body = await fs.readFile(filename);
      response.writeHead(200, {'Content-Type': types[path.extname(filename)] || 'application/octet-stream', 'Cache-Control': 'no-store'});
      response.end(request.method === 'HEAD' ? undefined : body);
    } catch { response.writeHead(404).end(); }
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  const {chromium} = await import(pathToFileURL(path.resolve(args['playwright-module'])).href);
  browser = await chromium.launch({headless: true, executablePath: args.chrome,
    args: ['--disable-background-networking', '--no-first-run', '--no-default-browser-check']});
  report.browser = browser.version();
  const page = await browser.newPage({viewport: {width: WIDTH, height: 1200}, deviceScaleFactor: 1});
  page.on('pageerror', error => report.page_errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/predict-position.html?capture=1`, {waitUntil: 'networkidle'});
  await page.waitForFunction(() => window.nanojevShooting?.ready || window.nanojevShooting?.error);
  assert.equal(await page.evaluate(() => window.nanojevShooting.error), null);
  await page.evaluate(id => window.nanojevShooting.setCase(id), item.id);
  await page.locator('#speed').selectOption(String(SPEED));
  assert.equal(await page.locator('.model-card[data-model]').count(), 3);
  assert.equal(await page.locator('#benchmarkStrip').isVisible(), true);
  const height = Math.ceil(Math.max(1080, await page.evaluate(() => document.documentElement.scrollHeight)) / 2) * 2;
  await page.setViewportSize({width: WIDTH, height});
  await page.evaluate(() => scrollTo(0, 0));
  assert.ok(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight && document.documentElement.scrollWidth <= innerWidth),
    'The complete page and playback controls must fit the video without clipping.');
  report.viewport = {width: WIDTH, height, device_scale_factor: 1};
  report.benchmark_text = await page.locator('#benchmarkStrip').innerText();
  const resourceNames = await page.evaluate(() => [...new Set(performance.getEntriesByType('resource')
    .map(entry => new URL(entry.name)).filter(url => url.origin === location.origin).map(url => url.pathname.slice(1)))]);
  resourceNames.push('predict-position.html');
  report.rendered_asset_sha256 = {};
  report.unavailable_optional_assets = [];
  for (const name of resourceNames.sort()) {
    if (name === 'favicon.ico' && !await exists(path.join(root, name))) {
      report.unavailable_optional_assets.push(name); continue;
    }
    report.rendered_asset_sha256[name] = digest(await fs.readFile(path.join(root, name)));
  }
  const captured = new Map(), frameHash = createHash('sha256');
  let outputIndex = 0;
  async function captureTick(tick) {
    if (captured.has(tick)) return captured.get(tick);
    const snapshot = await page.evaluate(async tick => {
      const snapshot = window.nanojevShooting.setTick(tick);
      // Independently verify the three actual canvas bitmaps before each PNG capture.
      window.__captureSourceImages ||= new Map();
      const expectedCanvas = document.createElement('canvas'); expectedCanvas.width = 320; expectedCanvas.height = 240;
      const context = expectedCanvas.getContext('2d', {alpha: false, willReadFrequently: true});
      for (const system of snapshot.systems) {
        const sprite = system.frame.sprite;
        if (!window.__captureSourceImages.has(sprite.src)) {
          const image = new Image(); image.src = sprite.src; await image.decode(); window.__captureSourceImages.set(sprite.src, image);
        }
        context.drawImage(window.__captureSourceImages.get(sprite.src), sprite.x, sprite.y, 320, 240, 0, 0, 320, 240);
        const expected = new Uint32Array(context.getImageData(0, 0, 320, 240).data.buffer);
        const card = document.querySelector(`.model-card[data-model="${system.id}"]`);
        const actual = new Uint32Array(card.querySelector('canvas').getContext('2d').getImageData(0, 0, 320, 240).data.buffer);
        for (let i = 0; i < expected.length; i++) if (expected[i] !== actual[i]) throw new Error(`Canvas source mismatch at ${system.id}/${tick}`);
        const note = card.querySelector('.image-note');
        if (note.hidden !== (system.frame.image_tick === system.frame.tick)) throw new Error('Held original image is not disclosed.');
      }
      return snapshot;
    }, tick);
    assert.equal(snapshot.ready, true); assert.equal(snapshot.caseId, item.id);
    assert.equal(snapshot.tick, tick); assert.equal(snapshot.totalTicks, maxTick);
    assert.equal(snapshot.playing, false); assert.equal(snapshot.speed, SPEED);
    assert.deepEqual(snapshot.systems.map(system => system.id), ORDER);
    for (const actual of snapshot.systems) {
      const source = systems.find(system => system.id === actual.id);
      const frame = source.frames[Math.min(tick, source.total_ticks)];
      assert.deepEqual(actual.frame, frame); assert.equal(actual.success, source.success);
      report.verification.exact_system_frame_comparisons++;
      report.verification.canvas_pixel_comparisons++;
      if (frame.image_tick !== frame.tick) report.verification.original_image_holds_checked++;
      if (tick === source.total_ticks) report.verification.terminal_ticks_included[source.id] = true;
    }
    const filename = path.join(capturesDirectory, `tick_${String(tick).padStart(6, '0')}.png`);
    const png = await page.screenshot({path: filename, animations: 'disabled', fullPage: false});
    frameHash.update(`${tick}:${digest(png)}\n`);
    captured.set(tick, filename); report.verification.unique_rendered_ticks++;
    if (captured.size === 1 || captured.size % 50 === 0) console.log(JSON.stringify({captured_ticks: captured.size,
      current_tick: tick, final_tick: maxTick, output_frames_written: outputIndex, total_output_frames: totalFrames}));
    return filename;
  }
  for (const segment of segments) {
    assert.equal(outputIndex, segment.first_output_frame);
    for (let index = 0; index < segment.frame_count; index++) {
      const tick = segment.kind === 'hold' ? segment.tick : Math.min(maxTick, Math.floor((index + 1) * physicalTicksPerVideoSecond / FPS));
      const png = await captureTick(tick);
      await fs.link(png, path.join(framesDirectory, `frame_${String(outputIndex++).padStart(6, '0')}.png`));
    }
  }
  assert.equal(outputIndex, totalFrames); assert.equal(captured.size, maxTick + 1);
  assert.ok(ORDER.every(id => report.verification.terminal_ticks_included[id]));
  assert.deepEqual(report.page_errors, []);
  report.verification.all_recorded_physical_ticks_captured = true;
  report.verification.captured_png_sequence_sha256 = frameHash.digest('hex');
  report.verification.png_hash_sequence_encoding = 'Unique captured ticks in order: UTF-8 tick:PNG_SHA256 plus newline.';
  ffmpeg(['-loglevel', 'error', '-n', '-framerate', String(FPS), '-start_number', '0',
    '-i', path.join(framesDirectory, 'frame_%06d.png'), '-frames:v', String(totalFrames),
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
    '-r', String(FPS), '-movflags', '+faststart', movie]);
  const decoded = ffmpeg(['-loglevel', 'error', '-nostats', '-progress', 'pipe:1', '-i', movie, '-map', '0:v:0', '-f', 'null', '-']);
  const decodedFrames = [...decoded.stdout.matchAll(/^frame=(\d+)$/gm)].map(match => Number(match[1])).at(-1);
  assert.equal(decodedFrames, totalFrames); assert.ok(decoded.stdout.includes('progress=end'));
  const probe = spawnSync(args.ffmpeg, ['-hide_banner', '-i', movie], {encoding: 'utf8', maxBuffer: 4 * 1024 * 1024});
  if (probe.error) throw probe.error;
  const duration = probe.stderr.match(/Duration: (\d+):(\d+):(\d+(?:\.\d+)?)/);
  assert.ok(duration, 'Encoded duration must be readable.');
  const seconds = Number(duration[1]) * 3600 + Number(duration[2]) * 60 + Number(duration[3]);
  assert.ok(Math.abs(seconds - totalFrames / FPS) <= .011);
  assert.match(probe.stderr, /Video: h264/); assert.match(probe.stderr, /yuv420p/);
  assert.ok(probe.stderr.includes(`${WIDTH}x${height}`)); assert.match(probe.stderr, /\b30 fps\b/);
  const bytes = await fs.readFile(movie);
  report.encoding = {codec: 'H.264', pixel_format: 'yuv420p', fps: FPS, crf: 18, preset: 'medium', faststart: true,
    width: WIDTH, height, frame_count: decodedFrames, duration_seconds: decodedFrames / FPS,
    displayed_container_duration_seconds: seconds, full_decode_passed: true};
  report.movie = {file: path.basename(movie), bytes: bytes.length, sha256: digest(bytes)};
  report.passed = true;
  await fs.writeFile(manifestPath, JSON.stringify(report, null, 2) + '\n', {flag: 'wx'});
  console.log(JSON.stringify({passed: true, movie, manifest: manifestPath, encoding: report.encoding,
    unique_captures: report.verification.unique_rendered_ticks, bytes: bytes.length}));
} catch (error) {
  report.failure = {name: error.name, message: error.message};
  await fs.writeFile(path.join(work, 'capture_failure.json'), JSON.stringify(report, null, 2) + '\n');
  throw error;
} finally {
  if (browser) await browser.close();
  if (server) await new Promise(resolve => server.close(resolve));
}
