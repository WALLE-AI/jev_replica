#!/usr/bin/env python3
'''Tests for scripts/summarize_unified_games.py.

All fixtures are synthetic temporary files created by the tests; they are not
model results and carry no performance claims.
'''
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_unified_games as sug  # noqa: E402

SPEC = {'task': 'maze', 'size': 8, 'topology': 'corridor', 'max_steps': 128}
TRAINING = {'best_step': 10, 'best_dev_selection_ce': 0.5, 'selected_on': 'dev only', 'stage': 'critic',
            'loss': 'paired_brier_pg', 'completed_steps': 300, 'training_seconds': 100, 'max_gpu_allocated_gb': 10,
            'weights_sha256': 'W1', 'continuation_policy_id': 'OLD_POLICY', 'temperature': 1.0, 'temperature_fitted': False,
            'metrics_by_split': {'test': {'selection_ce': 0.5, 'by_task_role': {
                'maze/outcome': {'questions': 10, 'ce': 0.5, 'brier': 0.3, 'observed_accuracy': 0.7},
                'maze/policy': {'questions': 10, 'ce': 0.9, 'kl': 0.2, 'tv': 0.1}}, 'by_role_macro_task': {}}}}


def episode(cid, success, split='test', task='maze', variant=None, seed=1, steps=3, metrics=None,
            policy='polA', complete=True, truncate=False, spec=None):
    sp = dict(spec or SPEC, task=task)
    st = [{'observation': {'cell': i}, 'action': 'north', 'scores': {}, 'answers': {}, 'forced': False,
           'behavior_probs': {}, 'reward': 0, 'terminated': i == steps - 1 and not truncate,
           'truncated': truncate and i == steps - 1, 'info': {}} for i in range(steps)]
    return {'case': {'id': cid, 'split': split, 'seed': seed, 'variant': variant or f'{task}8', 'spec': sp},
            'continuation_policy_id': policy, 'steps': st, 'complete': complete, 'success': success,
            'final_info': {'success': success, 'episode_metrics': metrics or {}}}


def write_run(d, name, rows, policy=None, manifest=None):
    ep = Path(d) / f'{name}.jsonl'
    ep.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    pol = {'engine': 'checkpoint', 'controller': 'greedy', 'epsilon': 0.0, 'sampling_seed': 17, 'temperature': 1.0,
           'source_sha256': {}, 'tie_break': 'lexicographic_first', 'environment_contract': 'finite_task_deadline_v1',
           'checkpoint_sha256': {'config.json': 'cfg', 'best.safetensors': 'W1'}}
    pol.update(policy or {})
    man = {'schema_version': sug.SCHEMA, 'policy': pol, 'continuation_policy_id': rows[0]['continuation_policy_id'],
           'cases_sha256': 'srcA', 'selected_cases': [r['case']['id'] for r in rows], 'finished': True,
           'episode_sha256': sug.sha256_file(ep), 'summary': {'success_rate': 0.99}}
    man.update(manifest or {})
    ep.with_suffix('.manifest.json').write_text(json.dumps(man), encoding='utf-8')
    return str(ep)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = self.tmp.name

    def run_path(self, name, rows, **kw):
        return write_run(self.d, name, rows, **kw)

    def test_success_counts_use_episode_denominator(self):
        rows = [episode('t-maze8-1', True, steps=40), episode('t-maze8-2', True, seed=2, steps=60),
                episode('t-maze8-3', False, seed=3, steps=50)]
        s = sug.build_summary({'A': self.run_path('a', rows)})
        g = s['runs']['A']['by_split_task']['test/maze']
        self.assertEqual((g['n'], g['successes']), (3, 2))
        self.assertAlmostEqual(g['success_rate'], 2 / 3)
        self.assertAlmostEqual(g['mean_decisions'], 50.0)
        self.assertEqual(len(s['runs']['A']['per_episode']), 3)
        self.assertNotIn('observation', json.dumps(s))
        self.assertEqual(s['inputs']['A']['episodes_sha256'], sug.sha256_file(s['inputs']['A']['episodes_path']))

    def test_metrics_only_observed_finite_numbers(self):
        rows = [episode('c1', True, metrics={'collisions': 3, 'flag': True, 'food_score': float('nan')}),
                episode('c2', False, seed=2, metrics={'collisions': 1, 'food_score': 2.5}),
                episode('c3', False, seed=3, metrics={})]
        g = sug.build_summary({'A': self.run_path('a', rows)})['runs']['A']['by_split_task']['test/maze']['metrics']
        self.assertEqual(set(g), {'collisions', 'food_score'})
        self.assertEqual((g['collisions']['n'], g['collisions']['mean']), (2, 2.0))
        self.assertEqual((g['food_score']['n'], g['food_score']['mean']), (1, 2.5))

    def test_wilson_endpoints(self):
        self.assertEqual(sug.wilson(0, 12)[0], 0.0)
        self.assertLess(sug.wilson(0, 12)[1], 0.3)
        self.assertEqual(sug.wilson(12, 12)[1], 1.0)
        self.assertGreater(sug.wilson(12, 12)[0], 0.7)
        lo, hi = sug.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        self.assertIsNone(sug.wilson(0, 0))
        for k, n in ((2, 1), (-1, 3), (True, 3), (1, -1)):
            with self.assertRaises(ValueError):
                sug.wilson(k, n)

    def test_complete_flag_without_terminal_transition_rejected(self):
        row = episode('c1', True)
        row['steps'][-1]['terminated'] = False
        with self.assertRaisesRegex(ValueError, 'terminated transition'):
            sug.load_run('A', self.run_path('a', [row]))

    def test_terminal_reset_requires_explicit_evidence_and_has_zero_decisions(self):
        row = episode('reset-terminal', True, steps=0, metrics={'physical_ticks': 0})
        with self.assertRaisesRegex(ValueError, 'terminal reset info'):
            sug.load_run('A', self.run_path('invalid-reset', [row]))
        row['final_info']['terminated'] = True
        summary = sug.build_summary({'A': self.run_path('valid-reset', [row])})
        group = summary['runs']['A']['by_split_task']['test/maze']
        self.assertEqual((group['n'], group['successes'], group['mean_steps']), (1, 1, 0))
        self.assertEqual(group['metrics']['physical_ticks']['mean'], 0.)
        self.assertIn('Mean decision transitions', sug.render_markdown(summary))
        row['final_info']['truncated'] = True
        with self.assertRaisesRegex(ValueError, 'without truncation'):
            sug.load_run('A', self.run_path('truncated-reset', [row]))

    def test_macro_equal_task_weight_vs_pooled_and_split_separation(self):
        rows = [episode(f'm{i}', True, seed=i) for i in range(8)]
        rows += [episode(f's{i}', False, task='snake', seed=i) for i in range(2)]
        rows += [episode('h0', True, task='shooting', seed=0), episode('h1', False, task='shooting', seed=1)]
        rows += [episode('tr0', False, split='train', seed=0)]
        s = sug.build_summary({'A': self.run_path('a', rows)})
        m = s['runs']['A']['by_split_task_macro']['test']
        self.assertEqual((m['tasks'], m['task_count']), (['maze', 'shooting', 'snake'], 3))
        self.assertAlmostEqual(m['task_macro_success_rate'], 0.5)
        self.assertNotIn('wilson95', m)
        test_groups = [g for k, g in s['runs']['A']['by_split_task'].items() if k.startswith('test/')]
        self.assertAlmostEqual(sum(g['successes'] for g in test_groups) / sum(g['n'] for g in test_groups), 0.75)
        self.assertEqual(s['runs']['A']['by_split_task']['train/maze']['n'], 1)
        self.assertEqual(s['runs']['A']['by_split_task']['test/maze']['n'], 8)
        self.assertEqual(s['runs']['A']['by_split_task_macro']['train']['tasks'], ['maze'])

    def test_cohort_mismatch_rejected(self):
        a = self.run_path('a', [episode('c1', True), episode('c2', False, seed=2)])
        b = self.run_path('b', [episode('c1', True), episode('c3', False, seed=3)])
        with self.assertRaisesRegex(ValueError, 'cohort mismatch'):
            sug.build_summary({'A': a, 'B': b})
        c = self.run_path('c', [episode('c1', True, spec=dict(SPEC, size=9)), episode('c2', False, seed=2)])
        with self.assertRaisesRegex(ValueError, 'definitions differ'):
            sug.build_summary({'A': a, 'C': c})

    def test_duplicate_ids_rejected(self):
        p = self.run_path('a', [episode('c1', True), episode('c1', False)], manifest={'selected_cases': ['c1']})
        with self.assertRaisesRegex(ValueError, 'duplicate case id'):
            sug.load_run('A', p)

    def test_incomplete_truncated_mixed_and_nonboolean_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not complete'):
            sug.load_run('A', self.run_path('a', [episode('c1', True, complete=False)]))
        with self.assertRaisesRegex(ValueError, 'truncated'):
            sug.load_run('B', self.run_path('b', [episode('c1', True, truncate=True)]))
        with self.assertRaisesRegex(ValueError, 'mixed policy'):
            sug.load_run('C', self.run_path('c', [episode('c1', True), episode('c2', True, seed=2, policy='polB')]))
        rows = [episode('c1', True)]
        rows[0]['success'] = rows[0]['final_info']['success'] = 'yes'
        with self.assertRaisesRegex(ValueError, 'boolean'):
            sug.load_run('D', self.run_path('d', rows))

    def test_hash_and_manifest_consistency(self):
        with self.assertRaisesRegex(ValueError, 'episode_sha256'):
            sug.load_run('A', self.run_path('a', [episode('c1', True)], manifest={'episode_sha256': '0' * 64}))
        with self.assertRaisesRegex(ValueError, 'not finished'):
            sug.load_run('B', self.run_path('b', [episode('c1', True)], manifest={'finished': False}))
        with self.assertRaisesRegex(ValueError, 'selected_cases'):
            sug.load_run('C', self.run_path('c', [episode('c1', True)], manifest={'selected_cases': ['c1', 'c2']}))

    def test_controller_epsilon_differences_explicit(self):
        a = self.run_path('a', [episode('c1', True), episode('c2', False, seed=2)])
        b = self.run_path('b', [episode('c1', False, policy='polB'), episode('c2', True, seed=2, policy='polB')],
                          policy={'controller': 'sample', 'epsilon': 0.15}, manifest={'cases_sha256': 'srcB'})
        s = sug.build_summary({'A': a, 'B': b})
        d = s['policy_differences'][0]
        self.assertFalse(d['same_controller'])
        self.assertEqual(d['differs']['controller'], ['greedy', 'sample'])
        self.assertEqual(d['differs']['epsilon'], [0.0, 0.15])
        self.assertEqual(d['differs']['continuation_policy_id'], ['polA', 'polB'])
        self.assertEqual(s['cohort']['cases_sha256_by_run'], {'A': 'srcA', 'B': 'srcB'})
        self.assertEqual(s['cohort']['case_count'], 2)
        self.assertEqual(s['comparison']['by_split_task']['test/maze']['B']['successes'], 1)
        self.assertIn('vs', sug.render_markdown(s))

    def test_sampling_seed_and_tie_break_are_controller_differences(self):
        a = self.run_path('a', [episode('c1', True)])
        b = self.run_path('b', [episode('c1', False, policy='polB')],
                          policy={'sampling_seed': 123, 'tie_break': 'different'})
        s = sug.build_summary({'A': a, 'B': b})
        self.assertFalse(s['policy_differences'][0]['same_controller'])
        self.assertEqual(s['policy_differences'][0]['differing_case_counts']['sampling_seed'], 1)

    def test_random_engine_uses_effective_uniform_controller_without_rewriting_manifest(self):
        a = self.run_path('starting', [episode('c1', True)])
        b = self.run_path('random', [episode('c1', False, policy='random-policy')],
                          policy={'engine': 'random', 'controller': 'greedy'})
        summary = sug.build_summary({'Starting': a, 'Random': b})
        diff = summary['policy_differences'][0]
        self.assertFalse(diff['same_controller'])
        self.assertEqual(diff['differs']['effective_controller'], ['greedy', 'uniform_random'])
        source = summary['policies']['Random']['sources'][0]
        self.assertEqual(source['policy']['controller'], 'greedy')
        self.assertEqual(source['effective_controller'], 'uniform_random')
        self.assertIn('Effective controller', sug.render_markdown(summary))
        self.assertIn('uniform_random', sug.render_markdown(summary))

    def test_same_named_disjoint_sources_match_one_combined_run(self):
        a = self.run_path('a', [episode('c1', True)])
        b = self.run_path('b', [episode('c2', False, seed=2, policy='extension')],
                          policy={'source_sha256': {'collector': 'new'}}, manifest={'cases_sha256': 'extension-cohort'})
        combined = self.run_path('c', [episode('c1', False, policy='new-model'),
                                      episode('c2', True, seed=2, policy='new-model')])
        specs = sug.parse_named([f'Old={a}', f'Old={b}', f'New={combined}'], '--run')
        s = sug.build_summary(specs)
        self.assertEqual(s['cohort']['case_count'], 2)
        self.assertEqual(s['policies']['Old']['continuation_policy_ids'], ['polA', 'extension'])
        self.assertIsNone(s['policies']['Old']['continuation_policy_id'])
        self.assertEqual(len(s['inputs']['Old']['sources']), 2)
        self.assertEqual(s['runs']['Old']['by_split_task']['test/maze']['successes'], 1)
        self.assertEqual({row['continuation_policy_id'] for row in s['runs']['Old']['per_episode']}, {'polA', 'extension'})
        self.assertIn('extension', sug.render_markdown(s))

    def test_duplicate_cases_and_conflicting_policy_declarations_across_sources_rejected(self):
        a = self.run_path('a', [episode('c1', True)])
        b = self.run_path('b', [episode('c1', False, policy='other')])
        with self.assertRaisesRegex(ValueError, 'duplicate case ids across source'):
            sug.build_summary({'A': [a, b]})
        changed = self.run_path('c', [episode('c2', False, seed=2)], policy={'epsilon': .4})
        with self.assertRaisesRegex(ValueError, 'policy ID has conflicting'):
            sug.build_summary({'A': [a, changed]})

    def test_casewise_mixed_controllers_cannot_hide_behind_identical_value_sets(self):
        a = self.run_path('a', [episode('c1', True, policy='A0')], policy={'epsilon': 0.})
        b = self.run_path('b', [episode('c2', False, seed=2, policy='A1')], policy={'epsilon': .1})
        c = self.run_path('c', [episode('c1', False, policy='B0')], policy={'epsilon': .1})
        d = self.run_path('d', [episode('c2', True, seed=2, policy='B1')], policy={'epsilon': 0.})
        s = sug.build_summary({'A': [a, b], 'B': [c, d]})
        diff = s['policy_differences'][0]
        self.assertFalse(diff['same_controller'])
        self.assertEqual(diff['differing_case_counts']['epsilon'], 2)

    def test_training_diagnostics_separate_and_checked(self):
        a = self.run_path('a', [episode('c1', True), episode('c2', True, seed=2)])
        t = Path(self.d) / 'trainer_summary.json'
        t.write_text(json.dumps(TRAINING), encoding='utf-8')
        s = sug.build_summary({'A': a}, {'A': str(t)})
        d = s['training_diagnostics']['A']
        self.assertEqual(d['fields']['continuation_policy_id'], 'OLD_POLICY')
        self.assertFalse(d['collection_policy_matches_run_policy'])
        self.assertTrue(d['weights_match_run_checkpoint'])
        self.assertIn('NOT', d['label'])
        self.assertEqual(d['metrics_by_split']['test']['by_task_role']['maze/outcome']['observed_accuracy'], 0.7)
        self.assertEqual(s['runs']['A']['by_split_task']['test/maze']['successes'], 2)
        bad = Path(self.d) / 'bad.json'
        bad.write_text(json.dumps(dict(TRAINING, weights_sha256='W2')), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'weights_sha256'):
            sug.build_summary({'A': a}, {'A': str(bad)})
        with self.assertRaisesRegex(ValueError, '--training'):
            sug.build_summary({'A': a}, {'Z': str(t)})

    def test_cli_end_to_end(self):
        a = self.run_path('a', [episode('c1', True), episode('c2', False, seed=2, task='snake')])
        b = self.run_path('b', [episode('c1', False, policy='polB'), episode('c2', True, seed=2, task='snake', policy='polB')])
        t = Path(self.d) / 'trainer_summary.json'
        t.write_text(json.dumps(TRAINING), encoding='utf-8')
        out = Path(self.d) / 'report'
        rc = sug.main(['--run', f'A={a}', '--run', f'B={b}', '--training', f'A={t}', '--output-dir', str(out)])
        self.assertEqual(rc, 0)
        js = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
        md = (out / 'summary.md').read_text(encoding='utf-8')
        self.assertEqual(set(js['runs']), {'A', 'B'})
        self.assertEqual(js['inputs']['A']['episodes_sha256'], sug.sha256_file(a))
        self.assertEqual(js['comparison']['by_split_task']['test/maze']['A']['successes'], 1)
        for needle in ('## Inputs and provenance', '## Success by split/task', '## Training diagnostics',
                       'test/maze', 'test/snake', 'OLD_POLICY'):
            self.assertIn(needle, md)
        self.assertEqual(sug.main(['--run', f'A={a}', '--output-dir', str(out)]), 2)
        with self.assertRaisesRegex(ValueError, 'unique'):
            sug.parse_named(['A=x', 'A=y'], '--training')


if __name__ == '__main__':
    unittest.main()
