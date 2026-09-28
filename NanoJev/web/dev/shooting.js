/* Real recorded frames only; all three systems share elapsed physical ticks. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const ORDER = ['jev', 'nanojev', 'base'];
  const ACTIONS = ['left', 'right', 'shoot', 'noop'];
  const LABELS = { left: 'Left', right: 'Right', shoot: 'Shoot', noop: 'Wait' };
  const images = new Map();
  const state = { data: null, caseIndex: 0, tick: 0, speed: Number($('speed').value), playing: false, loading: false,
    animation: null, previousTime: 0, fraction: 0, loadVersion: 0, panels: [] };
  const api = { ready: false, error: null, getSnapshot, setTick, setCase, play, pause };
  window.nanojevShooting = api;
  const currentCase = () => state.data.cases[state.caseIndex];
  const orderedSystems = () => ORDER.map((id) => currentCase().systems.find((system) => system.id === id));
  const finalTick = () => Math.max(...orderedSystems().map((system) => system.total_ticks));
  const number = (value) => typeof value === 'number' && Number.isFinite(value);
  const integer = (value) => Number.isInteger(value) && value >= 0;

  function assetUrl(src) {
    if (typeof src !== 'string' || !src.trim()) throw new Error('A recording has no sprite image.');
    const url = new URL(src, location.href);
    const directory = new URL('./', location.href);
    if (url.origin !== directory.origin || !url.pathname.startsWith(directory.pathname) || !['http:', 'https:'].includes(url.protocol)) {
      throw new Error('Recording images must belong to this development page.');
    }
    return url.href;
  }

  function validateData(data) {
    if (!data || (data.schema || data.schema_version) !== 'nanojev-shooting-demo-v1' || !Array.isArray(data.cases) || !data.cases.length || !Array.isArray(data.models)) {
      throw new Error('The shooting recording has an unsupported format.');
    }
    if (ORDER.some((id) => data.models.filter((model) => model.id === id).length !== 1) || data.models.length !== 3) throw new Error('The recording must identify all three models.');
    if (new Set(data.cases.map((item) => item.id)).size !== data.cases.length || !data.cases.some((item) => item.id === data.default_case_id)) throw new Error('The case list or default case is invalid.');
    for (const item of data.cases) {
      if (typeof item.id !== 'string' || !item.id || !Array.isArray(item.systems) || item.systems.length !== 3 || ORDER.some((id) => item.systems.filter((system) => system.id === id).length !== 1)) throw new Error('A case is missing a recorded system.');
      if (!integer(item.max_ticks) || item.max_ticks < 1 || !integer(item.frame_skip) || item.frame_skip < 1) throw new Error('A case has invalid timing metadata.');
      for (const system of item.systems) {
        if (typeof system.success !== 'boolean' || !integer(system.total_ticks) || system.total_ticks > item.max_ticks || !integer(system.total_decisions) || !Array.isArray(system.frames) || system.frames.length !== system.total_ticks + 1) throw new Error('A run must contain every tick from reset to its recorded end.');
        system.frames.forEach((frame, index) => {
          if (frame.tick !== index || !integer(frame.decision) || frame.decision > system.total_decisions || typeof frame.terminal !== 'boolean' || frame.terminal !== (index === system.total_ticks)) throw new Error('A recording has a gap or an invalid terminal frame.');
          if (!integer(frame.image_tick) || frame.image_tick > frame.tick || (frame.action !== null && !ACTIONS.includes(frame.action))) throw new Error('A recorded frame has invalid image/action timing.');
          if (frame.probabilities !== null) {
            if (!frame.probabilities || typeof frame.probabilities !== 'object' || Object.keys(frame.probabilities).length !== 4 || ACTIONS.some((key) => !Object.hasOwn(frame.probabilities, key))) throw new Error('A decision distribution does not cover all offered actions.');
            const values = Object.values(frame.probabilities);
            if (values.some((value) => !number(value) || value < 0 || value > 1) || Math.abs(values.reduce((a, b) => a + b, 0) - 1) > 1e-5) throw new Error('A recorded action distribution is invalid.');
          }
          if ((frame.action === null) !== (frame.probabilities === null)) throw new Error('An action is missing its recorded distribution.');
          if ((frame.ammo !== null && !number(frame.ammo)) || !number(frame.kills) || frame.kills < 0) throw new Error('A frame is missing its recorded ammo or kill count.');
          const sprite = frame.sprite;
          if (!sprite || !integer(sprite.x) || !integer(sprite.y) || sprite.width !== 320 || sprite.height !== 240) throw new Error('The recording needs original 320 × 240 image crops.');
          assetUrl(sprite.src);
          if (frame.terminal && frame.success !== system.success) throw new Error('A terminal frame disagrees with the recorded result.');
        });
      }
    }
  }

  function loadImage(src) {
    const url = assetUrl(src);
    if (images.has(url)) return images.get(url).promise;
    const entry = { image: new Image(), ready: false };
    entry.promise = new Promise((resolve, reject) => {
      entry.image.onload = () => { entry.ready = true; resolve(entry.image); };
      entry.image.onerror = () => reject(new Error('A recorded image atlas could not be loaded.'));
      entry.image.src = url;
    });
    images.set(url, entry);
    return entry.promise;
  }

  function localFrame(system) {
    // Frames are validated as contiguous, but the timeline is always physical tick time.
    return system.frames[Math.min(state.tick, system.total_ticks)];
  }

  function setControlsDisabled(disabled) {
    for (const id of ['play', 'restart', 'timeline', 'previous', 'next', 'speed']) $(id).disabled = disabled;
  }

  function pause() {
    state.playing = false;
    if (state.animation !== null) cancelAnimationFrame(state.animation);
    state.animation = null;
    $('playIcon').textContent = '▶'; $('playLabel').textContent = 'Play';
    $('play').setAttribute('aria-label', 'Play all recordings');
    $('shootingStage').dataset.playing = 'false';
  }

  function play() {
    if (!api.ready || state.loading || state.playing) return;
    if (state.tick >= finalTick()) { state.tick = 0; state.fraction = 0; renderTick(); }
    state.playing = true; state.previousTime = performance.now();
    $('playIcon').textContent = 'Ⅱ'; $('playLabel').textContent = 'Pause';
    $('play').setAttribute('aria-label', 'Pause all recordings');
    $('shootingStage').dataset.playing = 'true';
    state.animation = requestAnimationFrame(animate);
  }

  function animate(time) {
    if (!state.playing) return;
    const elapsed = Math.max(0, time - state.previousTime) / 1000;
    state.previousTime = time;
    const rate = number(state.data.protocol?.ticks_per_second) ? state.data.protocol.ticks_per_second : 35;
    state.fraction += elapsed * rate * state.speed;
    const advance = Math.floor(state.fraction);
    if (advance > 0) {
      state.fraction -= advance;
      state.tick = Math.min(finalTick(), state.tick + advance);
      renderTick();
    }
    if (state.tick >= finalTick()) { pause(); return; }
    state.animation = requestAnimationFrame(animate);
  }

  function setTick(tick) {
    if (!api.ready || state.loading) throw new Error('The recorded arena is still loading.');
    if (!integer(tick) || tick > finalTick()) throw new RangeError('This physical tick is outside the recording.');
    pause(); state.tick = tick; state.fraction = 0; renderTick();
    return getSnapshot();
  }

  async function setCase(id) {
    if (!state.data) throw new Error('The case list is still loading.');
    const index = state.data.cases.findIndex((item) => item.id === id);
    if (index < 0) throw new RangeError('Unknown recorded case.');
    pause(); api.ready = false; state.caseIndex = index; state.tick = 0; state.fraction = 0; state.loading = true;
    const version = ++state.loadVersion;
    $('caseSelect').value = id; $('caseLoading').hidden = false;
    $('viewer').setAttribute('aria-busy', 'true'); setControlsDisabled(true);
    renderCase();
    try {
      const sources = new Set(orderedSystems().flatMap((system) => system.frames.map((frame) => frame.sprite.src)));
      await Promise.all([...sources].map(loadImage));
      if (version !== state.loadVersion) return getSnapshot();
      for (const system of orderedSystems()) for (const frame of system.frames) {
        const sprite = frame.sprite, entry = images.get(assetUrl(sprite.src));
        if (sprite.x + sprite.width > entry.image.naturalWidth || sprite.y + sprite.height > entry.image.naturalHeight) throw new Error('A sprite crop lies outside its recorded image atlas.');
      }
      const activeUrls = new Set([...sources].map(assetUrl));
      for (const url of images.keys()) if (!activeUrls.has(url)) images.delete(url);
      state.loading = false; api.ready = true; api.error = null;
      $('loadState').hidden = true; $('viewer').hidden = false; $('caseLoading').hidden = true;
      $('viewer').setAttribute('aria-busy', 'false'); $('shootingStage').dataset.ready = 'true';
      setControlsDisabled(false); renderTick();
      return getSnapshot();
    } catch (error) {
      if (version === state.loadVersion) showError(error);
      throw error;
    }
  }

  function renderCase() {
    const item = currentCase();
    $('caseTitle').textContent = item.title || item.id;
    $('splitBadge').textContent = String(item.split || 'Recorded');
    $('splitBadge').classList.toggle('ood', item.split === 'ood');
    $('caseMetadata').textContent = `SEED ${item.seed}  ·  ${item.frame_skip} TICKS / ACTION  ·  ${item.max_ticks}-TICK BUDGET`;
    const protocol = state.data.protocol || {};
    const labels = [protocol.controller ? `${String(protocol.controller).replaceAll('_', ' ')} controller` : '',
      number(protocol.epsilon) ? `ε = ${protocol.epsilon}` : '', integer(protocol.sampling_seed) ? `sampling seed ${protocol.sampling_seed}` : ''];
    $('protocolLabel').textContent = labels.filter(Boolean).join('  ·  ');
    $('timeline').max = String(finalTick()); $('totalTicks').textContent = String(finalTick());
    $('shootingStage').dataset.caseId = item.id;
    state.panels = orderedSystems().map((system) => {
      const model = state.data.models.find((entry) => entry.id === system.id);
      const card = $('cardTemplate').content.firstElementChild.cloneNode(true);
      card.dataset.model = system.id;
      card.querySelector('.model-name').textContent = model.name || system.name;
      card.querySelector('.model-description').textContent = model.description || '';
      card.querySelector('.model-description').title = model.description || '';
      const canvas = card.querySelector('canvas');
      const context = canvas.getContext('2d', { alpha: false });
      if (!context) throw new Error('This browser cannot display the recorded frames.');
      context.imageSmoothingEnabled = false;
      canvas.setAttribute('aria-label', `${model.name || system.name}: recorded Doom gameplay`);
      const bars = ACTIONS.map((action) => {
        const row = document.createElement('div'); row.className = 'probability-row'; row.dataset.action = action;
        const name = document.createElement('span'); name.className = 'probability-name'; name.textContent = LABELS[action];
        const track = document.createElement('div'); track.className = 'probability-track';
        const fill = document.createElement('div'); fill.className = 'probability-fill'; track.append(fill);
        const value = document.createElement('span'); value.className = 'probability-value'; value.textContent = '—';
        row.append(name, track, value); card.querySelector('.probability-bars').append(row);
        return { action, row, fill, value };
      });
      return { system, card, canvas, context, bars };
    });
    $('modelCards').replaceChildren(...state.panels.map((panel) => panel.card));
    document.dispatchEvent(new CustomEvent('nanojev:case', { detail: item }));
  }

  function renderTick() {
    for (const panel of state.panels) {
      const { system, card, context, canvas, bars } = panel;
      const frame = localFrame(system), sprite = frame.sprite;
      const image = images.get(assetUrl(sprite.src))?.image;
      context.drawImage(image, sprite.x, sprite.y, sprite.width, sprite.height, 0, 0, canvas.width, canvas.height);
      card.dataset.tick = String(frame.tick); card.dataset.decision = String(frame.decision);
      card.dataset.terminal = String(frame.terminal); card.dataset.success = String(frame.terminal && system.success);
      card.dataset.action = frame.action || ''; card.dataset.imageTick = String(frame.image_tick);
      card.dataset.probabilities = JSON.stringify(frame.probabilities);
      card.dataset.sprite = JSON.stringify(sprite);
      canvas.dataset.imageReady = 'true';
      card.querySelector('.frame-stamp').textContent = `TICK ${String(frame.tick).padStart(3, '0')}`;
      const held = card.querySelector('.image-note');
      held.hidden = frame.image_tick === frame.tick;
      held.textContent = `Last available image · tick ${frame.image_tick}`;
      const badge = card.querySelector('.status-badge');
      badge.textContent = frame.terminal ? (system.success ? 'Success' : 'Finished') : frame.tick === 0 ? 'Ready' : 'In progress';
      badge.className = `status-badge${frame.terminal && system.success ? ' success' : frame.tick > 0 && !frame.terminal ? ' live' : ''}`;
      const terminal = card.querySelector('.terminal-banner'); terminal.hidden = !frame.terminal;
      terminal.classList.toggle('success', system.success);
      card.querySelector('.terminal-symbol').textContent = system.success ? '✓' : '■';
      card.querySelector('.terminal-title').textContent = system.success ? 'Target eliminated' : 'Run completed';
      card.querySelector('.terminal-detail').textContent = `${system.total_ticks} ticks · ${system.total_decisions} decisions${system.success ? '' : ' · No elimination'}`;
      card.querySelector('.run-progress > span').style.width = `${system.total_ticks ? 100 * frame.tick / system.total_ticks : 100}%`;
      card.querySelector('.stat-ammo').textContent = frame.ammo == null ? '—' : String(frame.ammo);
      const ticks = card.querySelector('.stat-ticks'); ticks.textContent = String(frame.tick);
      const total = document.createElement('small'); total.textContent = `/ ${system.total_ticks}`; ticks.append(total);
      card.querySelector('.stat-decision').textContent = frame.decision === 0 ? '—' : String(frame.decision);
      card.querySelector('.action-heading').textContent = frame.terminal ? 'LAST RECORDED ACTION' : 'RECORDED ACTION';
      card.querySelector('.action-name').textContent = frame.action === null ? 'Initial state' : LABELS[frame.action];
      for (const bar of bars) {
        const value = frame.probabilities?.[bar.action];
        bar.row.classList.toggle('selected', frame.action === bar.action);
        bar.row.dataset.probability = value == null ? '' : String(value);
        bar.value.textContent = value == null ? '—' : `${(100 * value).toFixed(1)}%`;
        bar.fill.style.width = value == null ? '0%' : `${100 * value}%`;
      }
      card.querySelector('.decision-note').textContent = frame.action === null ? 'No action has been executed yet.' :
        frame.terminal ? 'Final decision held at the recorded end.' : 'Highlight = executed action, including exploration.';
    }
    $('timeline').value = String(state.tick);
    $('timeline').style.setProperty('--progress', `${finalTick() ? 100 * state.tick / finalTick() : 100}%`);
    $('timeline').setAttribute('aria-valuetext', `Physical tick ${state.tick} of ${finalTick()}`);
    $('currentTick').textContent = String(state.tick);
    $('shootingStage').dataset.tick = String(state.tick);
    document.dispatchEvent(new CustomEvent('nanojev:tick', { detail: getSnapshot() }));
  }

  function getSnapshot() {
    if (!state.data) return { ready: false, error: api.error };
    return JSON.parse(JSON.stringify({ ready: api.ready && !state.loading, error: api.error,
      caseId: currentCase().id, caseIndex: state.caseIndex, caseCount: state.data.cases.length,
      tick: state.tick, totalTicks: finalTick(), playing: state.playing, speed: state.speed,
      synchronization: 'physical_tick', probabilitySemantics: 'recorded_model_action_distribution',
      systems: orderedSystems().map((system) => ({ id: system.id, totalTicks: system.total_ticks,
        totalDecisions: system.total_decisions, success: system.success, frame: localFrame(system) })) }));
  }

  function showError(error) {
    pause(); api.ready = false; api.error = error.message || String(error); state.loading = false;
    $('shootingStage').dataset.ready = 'false'; $('viewer').hidden = true;
    $('loadState').hidden = false; $('loadState').querySelector('.spinner').hidden = true;
    $('loadState').querySelector('strong').textContent = 'The recording could not be loaded';
    $('loadState').querySelector('p').textContent = api.error;
  }

  $('play').addEventListener('click', () => state.playing ? pause() : play());
  $('restart').addEventListener('click', () => setTick(0));
  $('previous').addEventListener('click', () => setTick(Math.max(0, state.tick - 1)));
  $('next').addEventListener('click', () => setTick(Math.min(finalTick(), state.tick + 1)));
  $('timeline').addEventListener('input', (event) => setTick(Number(event.target.value)));
  $('speed').addEventListener('change', (event) => { state.speed = Number(event.target.value); state.previousTime = performance.now(); });
  $('caseSelect').addEventListener('change', (event) => { setCase(event.target.value).catch(() => {}); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  document.addEventListener('keydown', (event) => {
    if (!api.ready || state.loading || event.altKey || event.ctrlKey || event.metaKey || event.target.closest('input,select,textarea,button,[contenteditable="true"]')) return;
    if (event.key === ' ' || event.key.toLowerCase() === 'k') { event.preventDefault(); state.playing ? pause() : play(); }
    else if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); setTick(Math.max(0, Math.min(finalTick(), state.tick + (event.key === 'ArrowLeft' ? -1 : 1) * (event.shiftKey ? 10 : 1)))); }
    else if (event.key === 'Home' || event.key.toLowerCase() === 'r') { event.preventDefault(); setTick(0); }
    else if (event.key === 'End') { event.preventDefault(); setTick(finalTick()); }
  });

  async function boot() {
    setControlsDisabled(true);
    try {
      const recording = $('shootingStage').dataset.recording || 'shooting_results.json';
      const response = await fetch(assetUrl(recording), { cache: 'no-cache' });
      if (!response.ok) throw new Error(`Recorded data could not be read (HTTP ${response.status}).`);
      const data = await response.json(); validateData(data); state.data = data;
      document.dispatchEvent(new CustomEvent('nanojev:recordings', { detail: data }));
      for (const [index, item] of data.cases.entries()) {
        const option = document.createElement('option'); option.value = item.id;
        option.textContent = item.title ?
          `${String(index + 1).padStart(2, '0')} · ${item.title} · ${item.seed}` :
          `${String(index + 1).padStart(2, '0')} · ${String(item.split).toUpperCase()} · Seed ${item.seed}`;
        $('caseSelect').append(option);
      }
      $('recordingDetail').textContent = `${data.cases.length} recorded cases · 3 model perspectives`;
      const matchingActions = data.cases.every((item) => {
        const jev = item.systems.find((system) => system.id === 'jev');
        const base = item.systems.find((system) => system.id === 'base');
        return jev.frames.length === base.frames.length && jev.frames.every((frame, index) =>
          frame.action === base.frames[index].action && frame.decision === base.frames[index].decision);
      });
      const differingProbabilities = data.cases.some((item) => {
        const jev = item.systems.find((system) => system.id === 'jev');
        const base = item.systems.find((system) => system.id === 'base');
        return jev.frames.some((frame, index) => frame.probabilities && base.frames[index]?.probabilities &&
          ACTIONS.some((action) => frame.probabilities[action] !== base.frames[index].probabilities[action]));
      });
      $('cohortNote').hidden = !(matchingActions && differingProbabilities);
      $('viewer').hidden = false;
      await setCase(data.default_case_id);
      if (new URLSearchParams(location.search).get('autoplay') === '1') play();
    } catch (error) { showError(error); }
  }
  boot();
})();
