import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
  CONFIG, buildCases, inputHash, analyzeCase, nearestRoundingDiagnostic,
  runProbe, readLedger, inspectLedger, createSummary,
} from './probe_jev_distributions.mjs';

// 完全离线：只注入 mock evaluator，不导入 teachers.mjs，不读取 .env。
const cases = buildCases();
assert.equal(cases.length, 40);
assert.equal(new Set(cases.map(row => row.input_sha256)).size, 40);
for (const row of cases) {
  assert.equal(row.input_sha256, inputHash(row));
  assert.equal(Object.keys(row.questions).length, 1);
  assert.equal(Object.keys(row.questions.next_label.criteria).length, row.K);
  assert.ok(Math.abs(Object.values(row.reference_distribution).reduce((a, b) => a + b) - 1) < 1e-12);
  assert.deepEqual(Object.keys(row.questions.next_label.criteria), row.candidate_id_mapping.map(item => item.candidate_id));
  const pair = cases.find(other => other.K === row.K && other.case_family === row.case_family && other.permutation !== row.permutation);
  assert.equal(row.state, pair.state);
  assert.deepEqual(Object.keys(row.questions.next_label.criteria), Object.keys(pair.questions.next_label.criteria).reverse());
  assert.deepEqual(row.reference_distribution, pair.reference_distribution);
}
const makeTeacher = (row, probs = row.reference_distribution, cost = '0.0001') => ({
  native_probs: { next_label: { ...probs } },
  rounding: { probabilityDecimals: 2, scoreDecimals: 2 },
  provider_metadata: { gateway: { cost } },
});
const uniform255 = cases.find(row => row.K === 255 && row.case_family === 'unrevealed_uniform');
const allZero = Object.fromEntries(uniform255.canonical_candidate_ids.map(id => [id, 0]));
const allZeroStats = analyzeCase(uniform255, makeTeacher(uniform255, allZero));
assert.equal(allZeroStats.raw_sum, 0);
assert.equal(allZeroStats.native_pmax, null);
assert.equal(allZeroStats.native_entropy_nats, null);
assert.equal(allZeroStats.proxy_normalized, null);
assert.deepEqual(allZeroStats.native_argmax_candidate_ids, []);
assert.equal(allZeroStats.nearest_rounding.simplex_feasible, true);
assert.equal(allZeroStats.nearest_rounding.reference_inside_all_intervals, true);
const uniform64 = cases.find(row => row.K === 64 && row.case_family === 'unrevealed_uniform');
const rounded64 = Object.fromEntries(uniform64.canonical_candidate_ids.map(id => [id, 0.02]));
const nonunit = analyzeCase(uniform64, makeTeacher(uniform64, rounded64));
assert.ok(Math.abs(nonunit.raw_sum - 1.28) < 1e-12);
assert.equal(nonunit.native_entropy_nats, null);
assert.equal(nonunit.proxy_normalized.total_variation_from_reference < 1e-12, true);
assert.equal(nonunit.nearest_rounding.reference_inside_all_intervals, true);
assert.equal(nearestRoundingDiagnostic(allZero, uniform255.reference_distribution, null).available, false);

const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'openjev-probe-test-'));
try {
  let active = 0, peak = 0, calls = 0;
  let releaseFirstBatch;
  const firstBatch = new Promise(resolve => { releaseFirstBatch = resolve; });
  const lookup = new Map(cases.map(row => [row.input_sha256, row]));
  const evaluate = async input => {
    const call = ++calls;
    active++; peak = Math.max(peak, active);
    assert.ok(input.signal instanceof AbortSignal);
    if (call === 2) releaseFirstBatch();
    if (call <= 2) await firstBatch;
    await new Promise(resolve => setTimeout(resolve, 1));
    active--;
    const row = lookup.get(inputHash(input));
    return makeTeacher(row);
  };
  const completeDir = path.join(tmp, 'complete');
  const complete = await runProbe({ outputDir: completeDir, evaluate });
  assert.equal(complete.complete, true);
  assert.equal(calls, 40);
  assert.equal(peak, 2);
  assert.ok(Math.abs(complete.actual_known_cost_usd - 0.004) < 1e-12);
  assert.equal(complete.permutation_pairs.length, 20);
  assert.equal(complete.permutation_pairs.every(row => row.native_max_abs_probability_difference === 0), true);
  const resumed = await runProbe({ outputDir: completeDir, evaluate: () => { throw new Error('MUST_NOT_CALL'); } });
  assert.equal(resumed.complete, true);
  assert.equal(resumed.dispatched_evaluations, 40);

  let guardedCalls = 0;
  const guarded = await runProbe({ outputDir: path.join(tmp, 'budget'), budget: 0.03, evaluate: async input => {
    guardedCalls++;
    await new Promise(resolve => setTimeout(resolve, 1));
    return makeTeacher(lookup.get(inputHash(input)));
  } });
  assert.equal(guardedCalls, 1, '并发第二条必须考虑第一条 $0.02 在途预留');
  assert.equal(guarded.stopped_reason, 'BUDGET_GUARD');

  const failedDir = path.join(tmp, 'failure');
  let failingCalls = 0;
  const failed = await runProbe({ outputDir: failedDir, evaluate: async input => {
    const call = ++failingCalls;
    if (call === 1) throw Object.assign(new Error('不应落盘的敏感供应商文本'), { code: 'TEACHER_REQUEST_FAILED' });
    await new Promise(resolve => setTimeout(resolve, 5));
    return makeTeacher(lookup.get(inputHash(input)));
  } });
  assert.ok(failingCalls <= 2);
  assert.equal(failed.failed_evaluations, 1);
  assert.ok(failed.reserved_unknown_cost_usd >= CONFIG.minimum_request_reserve_usd);
  assert.ok(failed.successful_evaluations <= 1);
  const failedText = await fs.readFile(path.join(failedDir, 'private_distribution_probe.jsonl'), 'utf8');
  assert.equal(failedText.includes('敏感供应商文本'), false);
  const failedResume = await runProbe({ outputDir: failedDir, evaluate: () => { throw new Error('MUST_NOT_RETRY'); } });
  assert.equal(failedResume.stopped_reason, 'PRIOR_FAILURE_OR_UNRESOLVED_REQUEST');

  const events = await readLedger(path.join(completeDir, 'private_distribution_probe.jsonl'));
  const unfinishedDir = path.join(tmp, 'unfinished');
  await fs.mkdir(unfinishedDir);
  await fs.writeFile(path.join(unfinishedDir, 'private_distribution_probe.jsonl'), JSON.stringify(events[0]) + '\n');
  const unfinished = await runProbe({ outputDir: unfinishedDir, evaluate: () => { throw new Error('MUST_NOT_REPEAT_UNCERTAIN_REQUEST'); } });
  assert.equal(unfinished.unresolved_evaluations, 1);
  assert.equal(unfinished.stopped_reason, 'PRIOR_FAILURE_OR_UNRESOLVED_REQUEST');
  assert.throws(() => inspectLedger([events[0], events[0]], cases), /DUPLICATE_DISPATCH/);
  const corrupt = structuredClone(events[0]);
  corrupt.input_sha256 = 'changed';
  assert.throws(() => inspectLedger([corrupt], cases), /LEDGER_INPUT_MISMATCH/);
  const summary = createSummary(cases, events);
  assert.equal(summary.by_k.every(group => group.completed_cases === 8), true);
  console.log(JSON.stringify({ offline: true, paid_api_calls: 0, env_reads: 0, planned_cases: 40,
    checks: ['case_semantics_and_order', 'all_zero_rounding', 'nonunit_native_preserved', 'concurrency_2', 'inflight_budget', 'failure_stop_and_sanitization', 'completed_resume', 'orphan_dispatch_block', 'hash_mismatch', 'permutation_restore'], passed: true }));
} finally {
  await fs.rm(tmp, { recursive: true, force: true });
}
