import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { evaluateTeacher } from './teachers.mjs';

// Run locally: node --env-file=.env scripts/label_toy_decisions.mjs
// Credentials never enter the dataset or the remote training machine.
const dir = process.argv[2] ?? 'research/private_toy';
const output = path.join(dir, 'labeled.jsonl');
const budget = 0.25;
const maxRequests = 200;
const concurrency = 3;
const hash = row => createHash('sha256').update(JSON.stringify({state: row.state, questions: row.questions})).digest('hex');
const rows = [];
for (const split of ['train', 'dev', 'calibration', 'test', 'ood']) {
  const data = await fs.readFile(path.join(dir, `${split}.jsonl`), 'utf8');
  rows.push(...data.trim().split('\n').filter(Boolean).map(JSON.parse));
}
const existing = new Map();
try {
  for (const line of (await fs.readFile(output, 'utf8')).split('\n').filter(Boolean)) {
    const row = JSON.parse(line);
    if (existing.has(row.id)) throw new Error('Duplicate saved ID');
    existing.set(row.id, row);
  }
} catch (error) { if (error.code !== 'ENOENT') throw error; }
for (const row of rows) {
  if (existing.has(row.id) && existing.get(row.id).input_sha256 !== hash(row)) {
    throw new Error('Input changed for an already labeled ID; use a new dataset directory.');
  }
}
const pending = rows.filter(row => !existing.has(row.id));
let spent = [...existing.values()].reduce((s, r) => s + Number(r.teacher.provider_metadata?.gateway?.cost ?? 0), 0);
let reserved = 0, next = 0, attempted = 0, completed = 0, failed = 0;
let stopped = false;
let writing = Promise.resolve();
const started = new Date().toISOString();
async function worker() {
  while (!stopped && next < pending.length) {
    const row = pending[next];
    // Bytes + a large allowance upper-bounds normal token accounting for these tiny requests.
    // This is an application guard, not a provider-enforced spending limit.
    const reserve = (Buffer.byteLength(JSON.stringify({state:row.state, questions:row.questions})) + 4096) * 0.042 / 1e6;
    if (attempted >= maxRequests || spent + reserved + reserve > budget) { stopped = true; break; }
    next++; attempted++; reserved += reserve;
    try {
      const teacher = await evaluateTeacher({teacher:'jev', model:'typesafe-ai/jev', state:row.state, questions:row.questions});
      const actual = Number(teacher.provider_metadata?.gateway?.cost);
      if (!Number.isFinite(actual) || actual < 0) throw new Error('MISSING_COST');
      spent += actual;
      const saved = {...row, input_sha256:hash(row), teacher, labeled_at:new Date().toISOString()};
      writing = writing.then(() => fs.appendFile(output, JSON.stringify(saved) + '\n'));
      await writing;
      completed++;
      if (completed % 24 === 0) console.log(JSON.stringify({completed, attempted, provider_cost_usd:spent}));
    } catch (error) {
      failed++; stopped = true;
      console.error(JSON.stringify({id:row.id, error_code:error.code ?? 'LABEL_OR_PERSIST_FAILED', auto_retry:false}));
    } finally { reserved -= reserve; }
  }
}
await Promise.all(Array.from({length:concurrency}, worker));
await writing;
const summary = {started, finished:new Date().toISOString(), dataset_states:rows.length,
  previous_states:existing.size, new_states:completed, attempted, failed,
  provider_cost_usd:spent, application_budget_usd:budget, complete:existing.size+completed === rows.length,
  unknown_failed_request_cost:failed > 0, concurrency, automatic_retries:0};
await fs.writeFile(path.join(dir, 'label_summary.json'), JSON.stringify(summary,null,2)+'\n');
console.log(JSON.stringify(summary));
if (!summary.complete) process.exitCode = 1;
