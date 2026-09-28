// Render complete, recorded showcase trajectories to posters, GIFs and MP4s.
// The local server exposes only authored demo assets. No model/API calls occur.
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';

const args = {};
for (let i = 2; i < process.argv.length; i += 2) {
  assert.ok(process.argv[i].startsWith('--') && process.argv[i + 1]);
  args[process.argv[i].slice(2)] = process.argv[i + 1];
}
const root = path.resolve(args['web-root'] ?? 'web');
const output = path.resolve(args.output ?? 'assets');
const work = path.resolve(args.work ?? 'runs/arcade_capture');
const dataFile = path.resolve(args.data ?? path.join(root, 'arcade_results.json'));
const fps = Number(args['steps-per-second'] ?? 16);
assert.ok(Number.isFinite(fps) && fps >= 1 && fps <= 30);
assert.ok(args.ffmpeg && args['playwright-module']);
const {chromium} = await import(pathToFileURL(path.resolve(args['playwright-module'])).href);
const dataBytes = await fs.readFile(dataFile);
const data = JSON.parse(dataBytes);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const run = command => {
  const result = spawnSync(args.ffmpeg, ['-hide_banner', '-loglevel', 'error', '-y', ...command],
    {encoding: 'utf8', maxBuffer: 4 * 1024 * 1024});
  if (result.status !== 0) throw Error(result.stderr.slice(-3000));
};
const encodedDuration = file => {
  const probe = spawnSync(args.ffmpeg, ['-hide_banner', '-i', file], {encoding: 'utf8'});
  const value = probe.stderr.match(/Duration: (\d+):(\d+):(\d+(?:\.\d+)?)/);
  assert.ok(value, `Encoded media duration is missing: ${file}`);
  return Number(value[1]) * 3600 + Number(value[2]) * 60 + Number(value[3]);
};
const allowed = new Map([['/', 'arcade.html'], ['/arcade.html', 'arcade.html'],
  ['/arcade.js', 'arcade.js'], ['/arcade.css', 'arcade.css']]);
const server = http.createServer(async (req, res) => {
  try {
    const route = new URL(req.url, 'http://localhost').pathname;
    const file = route === '/arcade_results.json' ? dataFile : allowed.has(route) ? path.join(root, allowed.get(route)) : null;
    if (!file) { res.writeHead(404).end(); return; }
    const bytes = await fs.readFile(file);
    res.writeHead(200, {'Content-Type': file.endsWith('.js') ? 'text/javascript' :
      file.endsWith('.css') ? 'text/css' : file.endsWith('.json') ? 'application/json' : 'text/html'}).end(bytes);
  } catch { res.writeHead(500).end(); }
});
await fs.mkdir(output, {recursive: true});
await fs.mkdir(work, {recursive: true});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const browser = await chromium.launch({executablePath: args.chrome ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true, args: ['--disable-background-networking', '--no-first-run', '--no-default-browser-check']});
const report = {schema: 'nanojev-arcade-media-v1', data_sha256: hash(dataBytes), browser: browser.version(),
  model_calls: 0, errors: [], examples: [], replay_steps_per_second: fps};
try {
  const page = await browser.newPage({viewport: {width: 1480, height: 1060}, deviceScaleFactor: 1});
  page.on('pageerror', error => report.errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/?capture=1`, {waitUntil: 'networkidle'});
  await page.waitForFunction(() => window.nanojevArcade?.ready || window.nanojevArcade?.error);
  assert.ok(!await page.evaluate(() => window.nanojevArcade.error));
  await page.selectOption('#speedSelect', String(fps));
  // Test the visible controls before exporting deterministic frame captures.
  await page.locator('[data-game="snake"]').click();
  assert.equal((await page.evaluate(() => window.nanojevArcade.getSnapshot())).game, 'snake');
  await page.locator('#systemTabs button').nth(2).click();
  assert.equal((await page.evaluate(() => window.nanojevArcade.getSnapshot())).systemId, 'base');
  await page.locator('#nextButton').click();
  assert.equal((await page.evaluate(() => window.nanojevArcade.getSnapshot())).step, 1);
  await page.locator('#previousButton').click();
  assert.equal((await page.evaluate(() => window.nanojevArcade.getSnapshot())).step, 0);
  await page.locator('#timeline').fill('50');
  assert.equal((await page.evaluate(() => window.nanojevArcade.getSnapshot())).step, 50);
  await page.locator('#playButton').click();
  await page.waitForFunction(() => window.nanojevArcade.getSnapshot().step > 50);
  await page.locator('#playButton').click();
  assert.equal((await page.evaluate(() => window.nanojevArcade.getSnapshot())).playing, false);
  report.visible_controls_passed = ['game', 'model', 'next', 'previous', 'timeline', 'play', 'pause', 'speed'];
  const stage = page.locator('#arcadeStage');
  for (let ei = 0; ei < data.examples.length; ei++) {
    const example = data.examples[ei], si = example.systems.findIndex(system => system.id === 'nanojev');
    assert.ok(si >= 0, 'Showcase requires an explicitly named NanoJev system');
    const system = example.systems[si];
    const folder = path.join(work, example.game);
    await fs.mkdir(folder, {recursive: true});
    const frames = [];
    for (let step = 0; step < system.frames.length; step++) {
      const snapshot = await page.evaluate(([e, s, f]) => window.nanojevArcade.setFrame(e, s, f), [ei, si, step]);
      assert.deepEqual(snapshot.frame, system.frames[step], 'The rendered state must equal the recorded state');
      assert.equal(snapshot.exampleId, example.id);
      assert.equal(snapshot.systemId, system.id);
      assert.equal(snapshot.step, step);
      assert.equal(snapshot.playing, false);
      assert.equal(snapshot.stepsPerSecond, fps);
      const name = `frame_${String(step).padStart(5, '0')}.png`;
      await stage.screenshot({path: path.join(folder, name), animations: 'disabled'});
      frames.push({step, file: name, duration: step === 0 ? 1.1 : step === system.frames.length - 1 ? 2.2 : 1 / fps});
      if (step % 64 === 0) console.log(JSON.stringify({game: example.game, captured_step: step, total_steps: system.frames.length - 1}));
    }
    const timeline = frames.map(frame => `file '${frame.file}'\nduration ${frame.duration}`).join('\n') +
      `\nfile '${frames.at(-1).file}'\n`;
    const concat = path.join(folder, 'frames.txt');
    await fs.writeFile(concat, timeline);
    const stem = path.join(output, `arcade_${example.game}`);
    run(['-f', 'concat', '-safe', '0', '-i', concat, '-vf', 'fps=30,scale=trunc(iw/2)*2:trunc(ih/2)*2',
      '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', `${stem}.mp4`]);
    run(['-f', 'concat', '-safe', '0', '-i', concat, '-filter_complex',
      '[0:v]fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle',
      '-loop', '0', `${stem}.gif`]);
    const posterIndex = example.game === 'maze' ? system.frames.length - 1 : Math.max(1, Math.floor((system.frames.length - 1) * .88));
    await fs.copyFile(path.join(folder, frames[posterIndex].file), `${stem}.png`);
    const media = {};
    for (const extension of ['mp4', 'gif', 'png']) {
      const bytes = await fs.readFile(`${stem}.${extension}`);
      media[extension] = {path: `assets/arcade_${example.game}.${extension}`, bytes: bytes.length, sha256: hash(bytes)};
      if (extension !== 'png') media[extension].duration_seconds = encodedDuration(`${stem}.${extension}`);
    }
    report.examples.push({id: example.id, game: example.game, system: system.id, controller: example.controller,
      frames: frames.length, actual_steps: frames.length - 1, duration_seconds: media.mp4.duration_seconds,
      requested_timeline_duration_seconds: frames.reduce((sum, frame) => sum + frame.duration, 0),
      complete_trajectory: true, terminal_frame_included: true, poster_step: posterIndex, media});
    console.log(JSON.stringify(report.examples.at(-1)));
  }
  // Confirm every model's complete terminal state remains available.
  for (let ei = 0; ei < data.examples.length; ei++) {
    for (let si = 0; si < data.examples[ei].systems.length; si++) {
      const count = data.examples[ei].systems[si].frames.length;
      const snapshot = await page.evaluate(([e, s, n]) => window.nanojevArcade.setFrame(e, s, n), [ei, si, count - 1]);
      assert.deepEqual(snapshot.frame, data.examples[ei].systems[si].frames.at(-1));
      const outcome = data.examples[ei].systems[si].summary.outcome;
      if (outcome === 'horizon_survived') assert.equal(await page.locator('#metricStatus').textContent(), 'Survived');
      if (outcome === 'trapped') assert.equal(await page.locator('#metricStatus').textContent(), 'Trapped');
    }
  }
  await page.setViewportSize({width: 390, height: 844});
  await page.evaluate(() => window.nanojevArcade.setFrame(1, 0, 0));
  report.mobile = await page.evaluate(() => ({viewport: innerWidth, document: document.documentElement.scrollWidth}));
  assert.ok(report.mobile.document <= report.mobile.viewport + 1, 'Mobile page must not overflow horizontally');
  await page.screenshot({path: path.join(work, 'mobile.png'), fullPage: true});
  assert.deepEqual(report.errors, []);
  report.renderer_sources = {};
  for (const name of ['arcade.html', 'arcade.css', 'arcade.js']) {
    report.renderer_sources[`web/${name}`] = hash(await fs.readFile(path.join(root, name)));
  }
  report.passed = true;
  await fs.writeFile(path.join(output, 'arcade_media_manifest.json'), JSON.stringify(report, null, 2) + '\n');
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
