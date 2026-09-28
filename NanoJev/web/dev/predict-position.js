/* Explain actual shot timing without changing any recorded action or image. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const ORDER = ['jev', 'nanojev', 'base'];
  let recordings = null, current = null;
  const name = (id) => recordings.models.find(model => model.id === id).name;
  const seconds = (tick) => (tick / (recordings.protocol.ticks_per_second || 35)).toFixed(2);
  const shotTicks = (system) => system.frames.filter((frame, i, frames) => i > 0 &&
    Number.isFinite(frame.ammo) && Number.isFinite(frames[i - 1].ammo) && frame.ammo < frames[i - 1].ammo).map(frame => frame.tick);

  document.addEventListener('nanojev:recordings', ({ detail }) => {
    recordings = detail;
    const featured = detail.cases.find(row => row.id === detail.default_case_id);
    if (!featured || featured.systems.some(system => system.success !== (system.id === 'nanojev'))) {
      throw new Error('The featured replay must be a recorded NanoJev-only success.');
    }
    const cells = ORDER.map(id => {
      const row = detail.summary[id].test;
      if (!row || row.episodes !== 128 || !Number.isInteger(row.successes) || row.successes < 0 || row.successes > row.episodes) {
        throw new Error('The Predict Position benchmark requires the complete 128-case test cohort.');
      }
      const cell = document.createElement('article'); cell.className = 'benchmark-cell'; cell.dataset.model = id;
      cell.dataset.successes = String(row.successes); cell.dataset.episodes = String(row.episodes);
      const label = document.createElement('div'); label.className = 'benchmark-label';
      const labelName = document.createElement('span'); labelName.textContent = name(id);
      const rate = document.createElement('strong'); rate.className = 'benchmark-rate'; rate.textContent = `${(100 * row.successes / row.episodes).toFixed(1)}%`;
      label.append(labelName, rate);
      const count = document.createElement('strong'); count.className = 'benchmark-count';
      count.textContent = String(row.successes);
      const total = document.createElement('small'); total.textContent = ` / ${row.episodes} targets`; count.append(total);
      const track = document.createElement('div'); track.className = 'benchmark-track';
      const fill = document.createElement('span'); fill.className = 'benchmark-fill'; fill.style.width = `${100 * row.successes / row.episodes}%`; track.append(fill);
      cell.append(label, count, track); return cell;
    });
    $('benchmarkStrip').replaceChildren(...cells);
    $('benchmarkNote').textContent = 'Held-out test set · greedy actions with 10% exploration · 4 game ticks per decision. These scores include every test case, beyond the selected replays above.';
  });

  document.addEventListener('nanojev:case', ({ detail }) => {
    current = detail;
    $('caseDescription').textContent = current.description || '';
    const markers = [];
    for (const id of ORDER) {
      const system = current.systems.find(system => system.id === id);
      const ticks = shotTicks(system);
      for (const tick of ticks) {
        const button = document.createElement('button'); button.type = 'button'; button.className = 'shot-marker';
        button.dataset.model = id; button.dataset.tick = String(tick);
        button.textContent = `${name(id)} · ${seconds(tick)} s`;
        button.setAttribute('aria-label', `Jump to ${name(id)} rocket launch at tick ${tick}`);
        button.addEventListener('click', () => window.nanojevShooting.setTick(tick)); markers.push(button);
      }
      if (!ticks.length) {
        const note = document.createElement('span'); note.className = 'no-shot'; note.textContent = `${name(id)} · no rocket fired`; markers.push(note);
      }
    }
    $('shotMarkers').replaceChildren(...markers);
  });

  function playbackNote() {
    const speed = Number($('speed').value);
    $('timelineNote').textContent = `${speed < 1 ? 'Slow motion' : speed === 1 ? 'Real time' : 'Fast playback'} · ${speed}× playback. All panels share the same game clock.`;
  }
  $('speed').addEventListener('change', playbackNote);
  $('playFeatured').addEventListener('click', async () => {
    $('playFeatured').disabled = true;
    try {
      await window.nanojevShooting.setCase(recordings.default_case_id);
      window.nanojevShooting.play();
    } finally { $('playFeatured').disabled = !window.nanojevShooting.ready; }
  });
  document.addEventListener('nanojev:tick', ({ detail }) => {
    $('playFeatured').disabled = false;
    for (const system of detail.systems) {
      const card = document.querySelector(`.model-card[data-model="${system.id}"]`);
      const recorded = current.systems.find(row => row.id === system.id);
      const firstShot = shotTicks(recorded)[0];
      const launched = firstShot !== undefined && system.frame.tick >= firstShot;
      card.dataset.launched = String(launched);
      card.querySelector('.shot-phase').textContent = system.frame.terminal ? (system.success ? 'Target hit' : 'Target missed') : launched ? 'Rocket fired' : 'Rocket ready';
      card.querySelector('.shot-time').textContent = launched ? `Fired at ${seconds(firstShot)} s` : '1 rocket · not fired';
      for (const button of $('shotMarkers').querySelectorAll(`button[data-model="${system.id}"]`)) {
        button.dataset.reached = String(system.frame.tick >= Number(button.dataset.tick));
      }
    }
    playbackNote();
  });
})();
