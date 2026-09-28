#!/usr/bin/env python3
"""CPU-only synthetic fixtures for the MC/TD result verifier; no model scores."""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import summarize_unified_td as td


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value) + '\n')


def write_rows(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(v) + '\n' for v in values))


class TDSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases = []
        self.dataset = self.root / 'data'
        self.manifest = {'continuation_policy_id': 'frozen-pi1', 'max_states_per_episode': 0,
                         'episode_sha256': 'source-episodes', 'split_sha256': {}}
        for split in td.games.SPLITS:
            rows = []
            for task in td.games.TASKS:
                for index in range(2):
                    cid = f'{split}-{task}-{index}'
                    self.cases.append({'id': cid, 'split': split, 'seed': index,
                                       'variant': task + '8', 'spec': {'task': task, 'size': 8, 'max_steps': 32}})
                    rows.append({'id': cid, 'split': split, 'questions': {'success_north': {'type': 'boolean'}},
                                 'gold': {'success_north': bool(index)},
                                 'gold_label_kind': {'success_north': 'observed_outcome'},
                                 'metadata': {'task': task, 'record_role': 'outcome',
                                              'continuation_policy_id': 'frozen-pi1'}})
            path = self.dataset / f'{split}.jsonl'
            write_rows(path, rows)
            self.manifest['split_sha256'][split] = td.games.sha256_file(path)
        write_json(self.dataset / 'manifest.json', self.manifest)
        write_rows(self.root / 'cases.jsonl', self.cases)
        self.experiment = {'path_root': '.', 'cases': 'cases.jsonl', 'frozen_dataset': 'data',
                           'expected_case_count': len(self.cases), 'seeds': [17, 29], 'mc_condition': 'mc',
                           'primary_condition': 'mix',
                           'conditions': {'mc': {'config': {'td_n_step': 3, 'td_weight': 0}},
                                          'mix': {'config': {'td_n_step': 3, 'td_weight': .5}}},
                           'controls': {'steps': 2, 'target_update_every': 25}, 'arms': []}
        for condition in ('mc', 'mix'):
            for seed in (17, 29):
                name = f'{condition}_{seed}'
                self.experiment['arms'].append({'name': name, 'condition': condition, 'seed': seed,
                                               'run_dir': name, 'episodes': name + '.jsonl'})
                self.write_arm(name, condition, seed)
        self.exp_path = self.root / 'experiment.json'
        write_json(self.exp_path, self.experiment)

    def write_arm(self, name, condition, seed):
        run_dir = self.root / name
        weight = self.experiment['conditions'][condition]['config']['td_weight']
        hashes = {f'data/{split}.jsonl': self.manifest['split_sha256'][split] for split in td.games.SPLITS}
        hashes['data/manifest.json'] = td.games.sha256_file(self.dataset / 'manifest.json')
        config = {**td.DEFAULT_CONTROLS, 'output_dir': name, 'seed': seed, 'steps': 2,
                  'td_n_step': 3, 'td_weight': weight, 'target_update_every': 25,
                  'data_sha256': hashes, 'continuation_policy_id': 'frozen-pi1',
                  'init_weights_sha256': 'same-start', 'backbone_lr': 2e-5,
                  'td_enabled': bool(weight), 'td_audit': {'bootstrap_hash': condition}}
        write_json(run_dir / 'config.json', config)
        summary = {'completed_steps': 2, 'best_step': 2, 'best_dev_selection_ce': .5,
                   'selected_on': 'dev only', 'stage': 'critic', 'loss': 'brier',
                   'continuation_policy_id': 'frozen-pi1', 'weights_sha256': name + '-weights',
                   'training_seconds': 3., 'max_gpu_allocated_gb': 1., 'metrics_by_split': {},
                   'td': {'enabled': bool(weight), 'n_step': 3, 'weight': weight,
                          'target_update_every': 25, 'target_refresh_steps': [], 'target_forward_calls': 2 if weight else 0,
                          'target_padded_tokens': 32 if weight else 0}}
        for split in ('dev', 'calibration', 'test', 'ood'):
            predictions = []
            for case in [c for c in self.cases if c['split'] == split]:
                y = case['seed']
                predictions.append({'id': case['id'] + ':success_north', 'split': split,
                                    'task': case['spec']['task'], 'record_role': 'outcome',
                                    'gold_label_kind': 'observed_outcome', 'gold_index': y,
                                    'training_target': [float(y == 0), float(y == 1)],
                                    'target_objective': 'observed_outcome', 'candidate_ids': ['false', 'true'],
                                    'continuation_policy_id': 'frozen-pi1', 'student_logits': [0., 0.],
                                    'student_probs': [.5, .5]})
            write_rows(run_dir / f'predictions_{split}.jsonl', predictions)
            by_task = {task + '/outcome': {'questions': 2, 'excluded_questions': 0,
                                         'ce': math.log(2), 'brier': .5} for task in td.games.TASKS}
            summary['metrics_by_split'][split] = {
                'by_task_role': by_task,
                'by_role_macro_task': {'outcome': {'questions': 6, 'ce': math.log(2), 'brier': .5},
                                      'policy': {'questions': 6, 'ce': .7}}}
        write_json(run_dir / 'summary.json', summary)
        write_json(run_dir / 'train_log.json', [{'step': i, 'batch_question_ids_sha256': f'{seed}-{i}'} for i in (1, 2)])
        policy = {**td.DEFAULT_CONTROLLER, 'sampling_seed': seed, 'source_sha256': {'collector': 'same'},
                  'checkpoint_sha256': {'config.json': td.games.sha256_file(run_dir / 'config.json'),
                                        'best.safetensors': summary['weights_sha256']}}
        pid = td.digest(policy)
        episodes = []
        for case in self.cases:
            # MC is one success per task; mix wins both at seed 17 and loses both at seed 29.
            success = bool(case['seed']) if condition == 'mc' else seed == 17
            episodes.append({'case': case, 'complete': True, 'continuation_policy_id': pid,
                             'success': success, 'steps': [{'action': 'north', 'terminated': True, 'truncated': False}],
                             'final_info': {'success': success, 'episode_metrics': {'physical_steps': 1}}})
        ep_path = self.root / f'{name}.jsonl'
        write_rows(ep_path, episodes)
        write_json(ep_path.with_suffix('.manifest.json'), {
            'schema_version': td.games.SCHEMA, 'finished': True, 'policy': policy,
            'continuation_policy_id': pid, 'episode_sha256': td.games.sha256_file(ep_path),
            'selected_cases': [c['id'] for c in self.cases], 'cases_sha256': td.games.sha256_file(self.root / 'cases.jsonl')})

    def build(self):
        return td.build_summary(self.exp_path)

    def mutate_json(self, path, update):
        path = self.root / path
        data = td.read_json(path)
        update(data)
        write_json(path, data)

    def mutate_prediction(self, update):
        path = self.root / 'mix_17/predictions_test.jsonl'
        rows = list(td.read_rows(path)); update(rows); write_rows(path, rows)

    def refresh_rollout(self, name, update=None):
        ep = self.root / f'{name}.jsonl'
        manifest = td.read_json(ep.with_suffix('.manifest.json'))
        if update:
            update(manifest)
        manifest['continuation_policy_id'] = td.digest(manifest['policy'])
        rows = list(td.read_rows(ep))
        for row in rows:
            row['continuation_policy_id'] = manifest['continuation_policy_id']
        write_rows(ep, rows)
        manifest['episode_sha256'] = td.games.sha256_file(ep)
        write_json(ep.with_suffix('.manifest.json'), manifest)

    def test_complete_pairing_and_seed_range(self):
        result = self.build()
        self.assertTrue(result['complete'])
        self.assertEqual(result['paired_vs_mc']['mix']['17']['games']['test']['task_macro_success_delta'], .5)
        self.assertEqual(result['paired_vs_mc']['mix']['29']['games']['test']['task_macro_success_delta'], -.5)
        spread = result['paired_game_aggregates']['mix']['test']
        self.assertEqual((spread['mean'], spread['min'], spread['max']), (0, -.5, .5))
        self.assertEqual(result['runs']['mix_29']['by_split_task_variant']['test/maze/maze8']['successes'], 0)
        self.assertEqual(result['training']['mc_17']['probabilities']['test']['macro_equal_task']['vector_brier'], .5)
        self.assertTrue(result['paired_vs_mc']['mix']['17']['same_training_batch_sequence'])
        self.assertIn('TD', td.render_markdown(result))

    def test_missing_seed_has_no_partial_mean(self):
        (self.root / 'mix_29.jsonl').unlink()
        result = self.build()
        self.assertFalse(result['complete'])
        spread = result['condition_aggregates']['mix']['game_task_macro']['test']
        self.assertEqual(spread['values_by_seed'], {'17': 1.0})
        self.assertIsNone(spread['mean'])
        self.assertIsNone(result['paired_game_aggregates']['mix']['test']['mean'])

    def test_unfinished_manifest_is_pending(self):
        self.mutate_json('mix_29.manifest.json', lambda d: d.update(finished=False))
        result = self.build()
        self.assertEqual(result['missing'][0]['status'], 'unfinished')
        self.assertNotIn('mix_29', result['runs'])

    def test_episode_hash_mismatch_rejected(self):
        with (self.root / 'mix_17.jsonl').open('a') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(ValueError, 'episode_sha256'):
            self.build()

    def test_checkpoint_mismatch_rejected(self):
        self.refresh_rollout('mix_17', lambda m: m['policy']['checkpoint_sha256'].update({'best.safetensors': 'wrong'}))
        with self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
            self.build()

    def test_uncontrolled_lr_change_rejected(self):
        self.mutate_json('mix_17/config.json', lambda d: d.update(backbone_lr=.001))
        self.refresh_rollout('mix_17', lambda m: m['policy']['checkpoint_sha256'].update(
            {'config.json': td.games.sha256_file(self.root / 'mix_17/config.json')}))
        with self.assertRaisesRegex(ValueError, 'Uncontrolled training config'):
            self.build()

    def test_incorrect_condition_rejected(self):
        self.mutate_json('mix_17/config.json', lambda d: d.update(td_weight=1))
        with self.assertRaisesRegex(ValueError, 'td_weight differs'):
            self.build()

    def test_td_target_cannot_enter_heldout_metrics(self):
        self.mutate_prediction(lambda rows: rows[0].update(training_target=[.25, .75]))
        with self.assertRaisesRegex(ValueError, 'actual terminal Y'):
            self.build()

    def test_missing_prediction_rejected(self):
        self.mutate_prediction(lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, 'Missing prediction questions'):
            self.build()

    def test_summary_probability_mismatch_rejected(self):
        self.mutate_json('mix_17/summary.json', lambda d: d['metrics_by_split']['test']['by_task_role']['maze/outcome'].update(ce=0))
        with self.assertRaisesRegex(ValueError, 'summary metrics mismatch'):
            self.build()

    def test_dataset_hash_mismatch_rejected(self):
        with (self.dataset / 'test.jsonl').open('a') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(ValueError, 'Dataset hash mismatch'):
            self.build()

    def test_cohort_spec_change_rejected(self):
        path = self.root / 'mix_17.jsonl'
        rows = list(td.read_rows(path)); rows[0]['case']['spec']['size'] = 16; write_rows(path, rows)
        self.refresh_rollout('mix_17')
        with self.assertRaisesRegex(ValueError, 'cohort differs'):
            self.build()

    def test_pair_controller_seed_difference_rejected(self):
        self.refresh_rollout('mix_17', lambda m: m['policy'].update(sampling_seed=999))
        with self.assertRaisesRegex(ValueError, 'paired controllers differ'):
            self.build()

    def test_missing_declared_arm_rejected(self):
        self.experiment['arms'].pop(); write_json(self.exp_path, self.experiment)
        with self.assertRaisesRegex(ValueError, 'every condition/seed'):
            self.build()

    def test_registered_schema_and_controller_alias(self):
        self.experiment['training_seeds'] = self.experiment.pop('seeds')
        self.experiment['shared_training'] = self.experiment.pop('controls')
        self.experiment['case_sha256'] = td.games.sha256_file(self.root / 'cases.jsonl')
        self.experiment['initial_checkpoint_sha256'] = 'same-start'
        self.experiment['continuation_policy_id'] = 'frozen-pi1'
        self.experiment['controller'] = {'controller': 'q_greedy', 'epsilon': .15, 'seed': 17,
                                         'env_batch': 16, 'batch_questions': 16, 'max_length': 8192}
        for name in ('mc_29', 'mix_29'):
            self.refresh_rollout(name, lambda m: m['policy'].update(sampling_seed=17))
        write_json(self.exp_path, self.experiment)
        result = self.build()
        self.assertTrue(result['complete'])
        self.assertFalse(result['training']['mc_17']['weights_rehashed_locally'])
        self.assertEqual(result['rollout_execution_parameters']['env_batch'], 16)

    def test_registered_case_hash_mismatch(self):
        self.experiment['case_sha256'] = 'wrong'
        write_json(self.exp_path, self.experiment)
        with self.assertRaisesRegex(ValueError, 'Registered case hash'):
            self.build()

    def test_registered_warm_start_mismatch(self):
        self.experiment['initial_checkpoint_sha256'] = 'other-start'
        write_json(self.exp_path, self.experiment)
        with self.assertRaisesRegex(ValueError, 'registered warm start'):
            self.build()

    def test_registered_frozen_policy_mismatch(self):
        self.experiment['continuation_policy_id'] = 'wrong-policy'
        write_json(self.exp_path, self.experiment)
        with self.assertRaisesRegex(ValueError, 'Registered continuation policy'):
            self.build()

    def test_zero_predicted_probability_retains_large_unclipped_nll(self):
        self.mutate_prediction(lambda rows: rows[0].update(student_logits=[-1000., 0.], student_probs=[0., 1.]))
        nll = (1000. + math.log(2)) / 2
        def update(summary):
            metrics = summary['metrics_by_split']['test']
            metrics['by_task_role']['maze/outcome'].update(ce=nll, brier=1.25)
            metrics['by_role_macro_task']['outcome'].update(ce=(nll + 2 * math.log(2)) / 3, brier=.75)
        self.mutate_json('mix_17/summary.json', update)
        result = self.build()
        metric = result['training']['mix_17']['probabilities']['test']['by_task']['maze']
        self.assertEqual(metric['questions'], 2)
        self.assertGreater(metric['nll'], 500)
        self.assertEqual(metric['vector_brier'], 1.25)

    def test_nonfinite_logits_rejected(self):
        self.mutate_prediction(lambda rows: rows[0].update(student_logits=[float('nan'), 0.]))
        with self.assertRaisesRegex(ValueError, 'finite number'):
            self.build()

    def test_mc_target_computation_cannot_be_hidden(self):
        self.mutate_json('mc_17/summary.json', lambda d: d['td'].update(target_forward_calls=1))
        with self.assertRaisesRegex(ValueError, 'MC unexpectedly has target forwards'):
            self.build()

    def test_refresh_schedule_is_checked(self):
        self.mutate_json('mix_17/summary.json', lambda d: d['td'].update(target_refresh_steps=[1]))
        with self.assertRaisesRegex(ValueError, 'refresh schedule'):
            self.build()

    def test_same_seed_online_question_sequences_must_match(self):
        self.mutate_json('mix_17/train_log.json', lambda rows: rows[0].update(batch_question_ids_sha256='changed'))
        with self.assertRaisesRegex(ValueError, 'online question sequence differs'):
            self.build()

    def test_best_checkpoint_is_checked_when_initial_dev_available(self):
        write_json(self.root / 'mix_17/initial_dev_metrics.json', {'selection_ce': .1})
        self.mutate_json('mix_17/train_log.json', lambda rows: rows[-1].update(dev={'selection_ce': .5}))
        with self.assertRaisesRegex(ValueError, 'minimum recorded dev CE'):
            self.build()

    def test_td_derived_cache_and_audit_are_checked(self):
        reference = {'continuation_policy_id': 'pi1', 'targets': {'train': {
            'q1': {'record_role': 'outcome'}, 'q2': {'record_role': 'outcome'}}}}
        counts = {'train': {'terminal': 1, 'bootstrap': 1}}
        source = {'episodes_sha256': 'episode-bytes', 'manifest_sha256': 'manifest-bytes', 'episodes': 1,
                  'by_n': {3: {'by_split': counts, 'train_bootstrap_questions': 3}}}
        config = {'td_weight': .5, 'td_n_step': 3, 'td_enabled': True, 'target_update_every': 25,
                  'microbatch_questions': 8, 'max_microbatch_tokens': 32768, 'max_length': 8192,
                  'objective': 'Choice full-distribution CE plus observed Boolean MC Brier and frozen-policy soft TD Brier',
                  'td_audit': {'episodes_sha256': 'episode-bytes', 'manifest_sha256': 'manifest-bytes', 'episodes': 1,
                               'n_step': 3, 'gamma': 1, 'continuation_policy_id': 'pi1', 'outcome_rows': 2,
                               'by_split': counts, 'complete_transition_coverage': True, 'cross_split_environment_groups': 0,
                               'native_game_rewards_used': False, 'behavior_distribution_and_rng_verified': True},
                  'td_contract': {'active': True, 'gamma': 1, 'continuation_policy_id': 'pi1',
                                  'heldout_target': 'observed terminal outcome only',
                                  'bootstrap_policy': 'recorded frozen behavior_probs; never online greedy',
                                  'target_microbatch_questions': 4, 'max_microbatch_tokens': 32768,
                                  'bootstrap_cache': {'outcome_questions': 2, 'bootstrap_states': 1,
                                                      'tokenized_boolean_questions': 3, 'max_path_tokens': 42}}}
        summary = {'completed_steps': 50, 'td': {'enabled': True, 'n_step': 3, 'weight': .5,
                    'target_update_every': 25, 'target_refresh_steps': [25, 50]}}
        td.verify_td_metadata('fixture', config, summary, reference, source)
        config['td_contract']['bootstrap_cache']['tokenized_boolean_questions'] = 4
        with self.assertRaisesRegex(ValueError, 'bootstrap cache counts'):
            td.verify_td_metadata('fixture', config, summary, reference, source)

    def test_cli_fresh_output_and_english_report(self):
        out = self.root / 'report'
        args = ['--experiment', str(self.exp_path), '--output-dir', str(out)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(td.main(args), 0)
            self.assertEqual(td.main(args), 2)
        self.assertTrue((out / 'summary.json').is_file())
        self.assertIn('Fixed-policy probability', (out / 'summary.md').read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
