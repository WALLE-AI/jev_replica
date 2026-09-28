#!/usr/bin/env node
// Export the real Maze comparison on one shared environment-step timeline.
// Playback stays paused during capture. No model inference or API calls occur.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath, pathToFileURL} from 'node:url';

const HELP = `Usage:
  node scripts/capture_side_by_side_video.mjs --web-root web --output assets \\
    --work runs/side_by_side_video_capture \\
    --playwright-module /path/to/playwright/index.mjs \\
    --ffmpeg /path/to/ffmpeg --chrome /path/to/chrome

The work directory, MP4 and manifest must not already exist.
Outputs: side_by_side_maze.mp4 and side_by_side_video_manifest.json.
The existing poster is never modified. No GIF is generated.
`;
if (process.argv.includes('--help')) { console.log(HELP); process.exit(0); }
const args = {};
for (let index = 2; index < process.argv.length; index += 2) {
  const key = process.argv[index];
  assert.ok(['--web-root', '--output', '--work', '--playwright-module', '--ffmpeg', '--chrome'].includes(key), `Unknown option: ${key}`);
  assert.ok(process.argv[index + 1] && !process.argv[index + 1].startsWith('--'), `Missing value: ${key}`);
  assert.ok(!(key.slice(2) in args), `Duplicate option: ${key}`);
  args[key.slice(2)] = process.argv[index + 1];
}
assert.ok(args['playwright-module'] && args.ffmpeg && args.chrome, 'Playwright module, FFmpeg and Chrome paths are required.');
const root = await fs.realpath(path.resolve(args['web-root'] || 'web'));
const output = path.resolve(args.output || 'assets');
const work = path.resolve(args.work || 'runs/side_by_side_video_capture');
const movie = path.join(output, 'side_by_side_maze.mp4');
const manifestPath = path.join(output, 'side_by_side_video_manifest.json');
const FPS = 30;
const WIDTH = 1440;
const ORDER = ['jev', 'nanojev', 'base'];
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const exists = async (filename) => { try { await fs.lstat(filename); return true; } catch (error) { if (error.code === 'ENOENT') return false; throw error; } };
assert.ok(!await exists(movie), `Refusing to overwrite ${movie}`);
assert.ok(!await exists(manifestPath), `Refusing to overwrite ${manifestPath}`);
assert.ok(!await exists(work), `Use a new capture work directory: ${work}`);

const assetNames = ['side-by-side.html', 'side-by-side.css', 'side-by-side.js', 'side_by_side_results.json'];
const assets = new Map();
for (const name of assetNames) {
  const filename = await fs.realpath(path.join(root, name));
  assert.ok(!path.relative(root, filename).startsWith('..'), 'An allowlisted asset resolves outside the web root.');
  assets.set(name, await fs.readFile(filename));
}
const data = JSON.parse(assets.get('side_by_side_results.json').toString('utf8'));
assert.equal(data.schema, 'nanojev-arcade-v1');
const mazeIndices = data.examples.map((example, index) => ({example, index})).filter(({example}) => example.game === 'maze');
assert.equal(mazeIndices.length, 1, 'The export requires exactly one recorded Maze example.');
const {example, index: exampleIndex} = mazeIndices[0];
assert.equal(example.systems.length, 3);
assert.deepEqual(example.systems.map((system) => system.id).sort(), [...ORDER].sort());
const systems = ORDER.map((id) => example.systems.find((system) => system.id === id));
for (const system of systems) assert.ok(system.frames.length > 1, 'Every system needs an initial and terminal frame.');
const terminal = Object.fromEntries(systems.map((system) => [system.id, system.frames.length - 1]));
// The requested pacing follows these recorded milestones; no outcome is inferred.
assert.ok(terminal.nanojev < terminal.jev && terminal.jev < terminal.base,
  'This pacing requires the recorded NanoJev, Jev and baseline terminal steps in that order.');
const maxStep = terminal.base;
const segments = [];
let totalFrames = 0;
function hold(step, seconds, speed, label) {
  const count = Math.round(seconds * FPS);
  segments.push({kind: 'hold', label, global_step: step, displayed_steps_per_second: speed,
    first_output_frame: totalFrames, frame_count: count, duration_seconds: count / FPS});
  totalFrames += count;
}
function movement(start, end, speed, terminalSystem) {
  const count = Math.ceil((end - start) * FPS / speed);
  segments.push({kind: 'movement', start_step: start, end_step: end,
    displayed_steps_per_second: speed, terminal_system: terminalSystem,
    first_output_frame: totalFrames, frame_count: count, duration_seconds: count / FPS,
    ideal_duration_seconds: (end - start) / speed,
    mapping: 'For zero-based segment frame i: min(end_step, start_step + floor((i+1)*displayed_steps_per_second/fps)).'});
  totalFrames += count;
}
hold(0, 1, 128, 'initial_state');
movement(0, terminal.nanojev, 128, 'nanojev');
hold(terminal.nanojev, 2, 128, 'nanojev_terminal');
movement(terminal.nanojev, terminal.jev, 256, 'jev');
hold(terminal.jev, 1.5, 256, 'jev_terminal');
movement(terminal.jev, terminal.base, 256, 'base');
hold(terminal.base, 3, 256, 'global_terminal');
const frameName = (index) => `frame_${String(index).padStart(6, '0')}.png`;
const ffmpeg = (command) => {
  const result = spawnSync(args.ffmpeg, ['-hide_banner', ...command], {encoding: 'utf8', maxBuffer: 8 * 1024 * 1024});
  if (result.error) throw result.error;
  assert.equal(result.status, 0, `FFmpeg failed: ${(result.stderr || '').slice(-4000)}`);
  return result;
};

await fs.mkdir(output, {recursive: true});
await fs.mkdir(path.dirname(work), {recursive: true});
await fs.mkdir(work);
const capturesDirectory = path.join(work, 'captures');
const framesDirectory = path.join(work, 'frames');
await fs.mkdir(capturesDirectory); await fs.mkdir(framesDirectory);
const report = {schema: 'nanojev-side-by-side-video-v1', passed: false,
  data_sha256: digest(assets.get('side_by_side_results.json')),
  renderer_sources: Object.fromEntries(assetNames.slice(0, 3).map((name) => [name, digest(assets.get(name))])),
  capture_script_sha256: digest(await fs.readFile(fileURLToPath(import.meta.url))),
  example_id: example.id, game: example.game, controller: example.controller,
  model_calls: 0, api_calls: 0, model_gpu_calls: 0,
  synchronization: 'One shared global environment step; each system holds its exact final recorded frame after finishing.',
  play_button_state: 'paused during deterministic frame capture',
  fps: FPS, total_output_frames: totalFrames, requested_duration_seconds: totalFrames / FPS,
  segments, source_terminal_steps: terminal,
  all_recorded_steps_individually_captured: false,
  path_history: 'The webpage draws cumulative paths from the complete source trajectory up to the shared sampled step.',
  holds: 'Repeated output frames hard-link the same captured PNG, avoiding repeated browser capture.',
  terminal_outcomes: systems.map((system) => ({id: system.id, name: system.name,
    terminal_step: system.frames.length - 1, summary: system.summary || {},
    terminal_frame_sha256: digest(JSON.stringify(system.frames.at(-1)))})),
  page_errors: [], verification: {unique_rendered_frames: 0, exact_system_frame_comparisons: 0, paused_captures: 0,
    terminal_frames_included: {}, global_step_min: null, global_step_max: null}};
let browser, server;
try {
  const types = {'.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.json': 'application/json'};
  const routes = new Map([['/', 'side-by-side.html'], ...assetNames.map((name) => ['/' + name, name])]);
  server = http.createServer((request, response) => {
    const name = routes.get(new URL(request.url, 'http://localhost').pathname);
    if (!name || !['GET', 'HEAD'].includes(request.method)) { response.writeHead(404).end(); return; }
    response.writeHead(200, {'Content-Type': types[path.extname(name)], 'Cache-Control': 'no-store'});
    response.end(request.method === 'HEAD' ? undefined : assets.get(name));
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  const {chromium} = await import(pathToFileURL(path.resolve(args['playwright-module'])).href);
  browser = await chromium.launch({headless: true, executablePath: args.chrome,
    args: ['--disable-background-networking', '--no-first-run', '--no-default-browser-check']});
  report.browser = browser.version();
  const page = await browser.newPage({viewport: {width: WIDTH, height: 1080}, deviceScaleFactor: 1});
  page.on('pageerror', (error) => report.page_errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/?capture=1#maze`, {waitUntil: 'networkidle'});
  await page.waitForFunction(() => window.nanojevComparison?.ready || window.nanojevComparison?.error);
  assert.equal(await page.evaluate(() => window.nanojevComparison.error), null);
  assert.equal(await page.locator('.model-panel').count(), 3);
  assert.deepEqual(await page.locator('.model-name').allTextContents(), systems.map((system) => system.name));
  if (!await page.locator('#showTrail').isChecked()) await page.locator('.trail-control').click();
  const layout = await page.evaluate(() => ({scroll_height: document.documentElement.scrollHeight,
    stage_bottom: document.querySelector('.stage').getBoundingClientRect().bottom + scrollY,
    footer_bottom: document.querySelector('.site-footer').getBoundingClientRect().bottom + scrollY}));
  const neededHeight = Math.ceil(Math.max(layout.scroll_height, layout.stage_bottom, layout.footer_bottom + 18) / 2) * 2;
  const height = neededHeight > 1080 ? Math.max(1120, neededHeight) : 1080;
  await page.setViewportSize({width: WIDTH, height});
  await page.evaluate(() => scrollTo(0, 0));
  report.viewport = {width: WIDTH, height, device_scale_factor: 1, initial_layout: layout};
  const finalLayout = await page.evaluate(() => ({scroll_height: document.documentElement.scrollHeight,
    footer_bottom: document.querySelector('.site-footer').getBoundingClientRect().bottom,
    timeline_bottom: document.querySelector('.transport').getBoundingClientRect().bottom}));
  assert.ok(finalLayout.scroll_height <= height && finalLayout.footer_bottom < height && finalLayout.timeline_bottom < height,
    'The page footer and complete timeline must fit without clipping.');
  report.viewport.final_layout = finalLayout;

  const captured = new Map();
  const frameHash = createHash('sha256');
  let outputIndex = 0;
  async function captureStep(step, speed) {
    const key = `${step}:${speed}`;
    if (captured.has(key)) return captured.get(key);
    const snapshot = await page.evaluate(([index, globalStep, rate]) => {
      window.nanojevComparison.setFrame(index, globalStep);
      const control = document.getElementById('speed');
      control.value = String(rate); control.dispatchEvent(new Event('change', {bubbles: true}));
      return window.nanojevComparison.getSnapshot();
    }, [exampleIndex, step, speed]);
    assert.equal(snapshot.ready, true); assert.equal(snapshot.exampleId, example.id);
    assert.equal(snapshot.globalStep, step); assert.equal(snapshot.totalSteps, maxStep);
    assert.equal(snapshot.playing, false, 'Frame capture requires paused playback.');
    assert.equal(snapshot.stepsPerSecond, speed); assert.equal(snapshot.showTrail, true);
    assert.deepEqual(snapshot.systems.map((system) => system.id), ORDER);
    for (const actual of snapshot.systems) {
      const source = systems.find((system) => system.id === actual.id);
      const localStep = Math.min(step, source.frames.length - 1);
      assert.equal(actual.name, source.name); assert.equal(actual.localStep, localStep);
      assert.equal(actual.finished, localStep === source.frames.length - 1);
      assert.deepEqual(actual.frame, source.frames[localStep]);
      assert.deepEqual(actual.summary, source.summary || {});
      report.verification.exact_system_frame_comparisons++;
      if (step === source.frames.length - 1) report.verification.terminal_frames_included[source.id] = true;
    }
    const filename = path.join(capturesDirectory, `step_${String(step).padStart(6, '0')}_${speed}.png`);
    const png = await page.screenshot({path: filename, animations: 'disabled', fullPage: false});
    frameHash.update(`${step}:${speed}:${digest(png)}\n`);
    captured.set(key, filename);
    report.verification.unique_rendered_frames++; report.verification.paused_captures++;
    report.verification.global_step_min ??= step; report.verification.global_step_max = step;
    if (captured.size === 1 || captured.size % 100 === 0) console.log(JSON.stringify({captured_frames: captured.size,
      global_step: step, total_global_steps: maxStep, output_frames_written: outputIndex, total_output_frames: totalFrames}));
    return filename;
  }
  for (const segment of segments) {
    assert.equal(outputIndex, segment.first_output_frame);
    for (let index = 0; index < segment.frame_count; index++) {
      const step = segment.kind === 'hold' ? segment.global_step :
        Math.min(segment.end_step, segment.start_step + Math.floor((index + 1) * segment.displayed_steps_per_second / FPS));
      const image = await captureStep(step, segment.displayed_steps_per_second);
      await fs.link(image, path.join(framesDirectory, frameName(outputIndex++)));
    }
  }
  assert.equal(outputIndex, totalFrames);
  assert.deepEqual(report.page_errors, []);
  assert.ok(ORDER.every((id) => report.verification.terminal_frames_included[id]));
  assert.equal(report.verification.global_step_min, 0); assert.equal(report.verification.global_step_max, maxStep);
  report.verification.captured_png_sequence_sha256 = frameHash.digest('hex');
  report.verification.png_hash_sequence_encoding = 'Capture order, unique (step,speed) only: UTF-8 step:speed:PNG_SHA256 plus newline.';

  ffmpeg(['-loglevel', 'error', '-n', '-framerate', String(FPS), '-start_number', '0',
    '-i', path.join(framesDirectory, 'frame_%06d.png'), '-frames:v', String(totalFrames),
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
    '-r', String(FPS), '-movflags', '+faststart', movie]);
  const decoded = ffmpeg(['-loglevel', 'error', '-nostats', '-progress', 'pipe:1', '-i', movie, '-map', '0:v:0', '-f', 'null', '-']);
  const decodedFrames = [...decoded.stdout.matchAll(/^frame=(\d+)$/gm)].map((match) => Number(match[1])).at(-1);
  assert.equal(decodedFrames, totalFrames, 'Full decoding must recover every planned video frame.');
  assert.ok(decoded.stdout.includes('progress=end'));
  const probe = spawnSync(args.ffmpeg, ['-hide_banner', '-i', movie], {encoding: 'utf8', maxBuffer: 4 * 1024 * 1024});
  if (probe.error) throw probe.error;
  const duration = probe.stderr.match(/Duration: (\d+):(\d+):(\d+(?:\.\d+)?)/);
  assert.ok(duration, 'Encoded duration must be readable.');
  const seconds = Number(duration[1]) * 3600 + Number(duration[2]) * 60 + Number(duration[3]);
  assert.ok(Math.abs(seconds - totalFrames / FPS) <= .011, 'Encoded duration must match the exact frame schedule within displayed timestamp precision.');
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
    unique_captures: report.verification.unique_rendered_frames, bytes: bytes.length}));
} catch (error) {
  report.failure = {name: error.name, message: error.message};
  await fs.writeFile(path.join(work, 'capture_failure.json'), JSON.stringify(report, null, 2) + '\n');
  throw error;
} finally {
  if (browser) await browser.close();
  if (server) await new Promise((resolve) => server.close(resolve));
}
