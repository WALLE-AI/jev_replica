/* Present the current unified checkpoint's recorded Basic runs and full test scores. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let recordings;
  document.addEventListener('nanojev:recordings', ({ detail }) => {
    recordings = detail;
    const featured = detail.cases.find(row => row.id === detail.default_case_id);
    if (!featured || featured.systems.some(system => system.success !== (system.id === 'nanojev'))) {
      throw new Error('The featured Basic replay must be a recorded NanoJev-only success.');
    }
    const cells = ['jev', 'nanojev', 'base'].map(id => {
      const row = detail.summary[id].test;
      if (!row || row.episodes !== 128 || !Number.isInteger(row.successes) || row.successes < 0 || row.successes > 128) {
        throw new Error('Basic requires the complete 128-case test cohort.');
      }
      const cell = document.createElement('article'); cell.className = 'benchmark-cell'; cell.dataset.model = id;
      cell.dataset.successes = String(row.successes); cell.dataset.episodes = String(row.episodes);
      const label = document.createElement('div'); label.className = 'benchmark-label';
      const name = document.createElement('span'); name.textContent = detail.models.find(model => model.id === id).name;
      const rate = document.createElement('strong'); rate.className = 'benchmark-rate';
      rate.textContent = `${(100 * row.successes / row.episodes).toFixed(1)}%`; label.append(name, rate);
      const count = document.createElement('strong'); count.className = 'benchmark-count'; count.textContent = String(row.successes);
      const total = document.createElement('small'); total.textContent = ` / ${row.episodes} targets`; count.append(total);
      const track = document.createElement('div'); track.className = 'benchmark-track';
      const fill = document.createElement('span'); fill.className = 'benchmark-fill'; fill.style.width = `${100 * row.successes / row.episodes}%`; track.append(fill);
      cell.append(label, count, track); return cell;
    });
    $('benchmarkStrip').replaceChildren(...cells);
    $('benchmarkNote').textContent = 'Held-out test set · greedy actions with 10% exploration · 4 game ticks per decision. All 128 test cases are included; the player above features selected NanoJev wins.';
  });
  document.addEventListener('nanojev:case', ({ detail }) => {
    $('caseDescription').textContent = detail.description || 'A recorded NanoJev success from the held-out test set. Both comparison models miss the target.';
  });
  document.addEventListener('nanojev:tick', () => { $('playFeatured').disabled = false; });
  $('playFeatured').addEventListener('click', async () => {
    $('playFeatured').disabled = true;
    try {
      await window.nanojevShooting.setCase(recordings.default_case_id);
      window.nanojevShooting.play();
    } finally { $('playFeatured').disabled = !window.nanojevShooting.ready; }
  });
})();
