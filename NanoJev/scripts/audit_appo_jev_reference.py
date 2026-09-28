#!/usr/bin/env python3
"""Audit the existing Jev Basic reference without inference or API requests.

Python standard library plus Node.js are required. Node reproduces the journal's
original JSON.stringify hashing exactly. Original files are read, never changed.
The output path must be fresh. Run from the repository root:

  python scripts/audit_appo_jev_reference.py --output runs/appo_basic_v1/jev_audit.json
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import subprocess

from unified_game_pipeline import behavior_distribution, choose, digest, file_digest, policy_request

SPLITS = ('train', 'dev', 'calibration', 'test', 'ood')
ACTIONS = {'left', 'right', 'shoot', 'noop'}


def require(value, message):
    if not value:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def parse(text):
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)


def rows(path):
    return [parse(line) for line in Path(path).read_text().splitlines() if line.strip()]


def js_hashes(receipts, node):
    source = """const fs=require('node:fs'),{createHash}=require('node:crypto');
const digest=x=>createHash('sha256').update(JSON.stringify(x)).digest('hex');
const data=JSON.parse(fs.readFileSync(0,'utf8'));
process.stdout.write(JSON.stringify(data.map(r=>({input:digest(r.input),response:digest({native_probs:r.native_probs,rounding:r.rounding})}))));
"""
    payload = [{key: row[key] for key in ('input', 'native_probs', 'rounding')} for row in receipts]
    result = subprocess.run([node, '-e', source], input=json.dumps(payload, ensure_ascii=False, allow_nan=False),
                            text=True, capture_output=True, check=True, timeout=30)
    hashes = parse(result.stdout)
    require(len(hashes) == len(receipts), 'Incomplete journal hash results')
    return hashes


def audit(args):
    paths = {'episodes': args.episodes, 'manifest': args.episodes.with_suffix('.manifest.json'),
             'journal': args.journal, 'cases': args.cases, 'original_cpu_replay': args.replay}
    hashes_before = {name: file_digest(path) for name, path in paths.items()}
    episodes, cases = rows(args.episodes), rows(args.cases)
    manifest, replay = parse(paths['manifest'].read_text()), parse(args.replay.read_text())
    require(manifest.get('finished') is True, 'Reference manifest is not finished')
    require(manifest['episode_sha256'] == hashes_before['episodes'], 'Reference episode SHA mismatch')
    policy = manifest['policy']
    expected_policy = {'engine': 'jev', 'model': 'typesafe-ai/jev', 'controller': 'greedy',
                       'epsilon': .1, 'sampling_seed': 17, 'temperature': 1.,
                       'tie_break': 'lexicographic_first', 'environment_contract': 'finite_task_deadline_v1'}
    require(all(policy.get(key) == value for key, value in expected_policy.items()), 'Reference policy differs from the registered comparison')
    require(digest(policy) == manifest['continuation_policy_id'], 'Reference policy identity mismatch')
    all_ids = [e['case']['id'] for e in episodes]
    require(len(all_ids) == len(set(all_ids)) == 192 and set(all_ids) == set(manifest['selected_cases']),
            'Reference must contain its complete original 192-case cohort')
    selected = {c['id']: c for c in cases if c.get('variant') == 'doom_basic'}
    basic = [e for e in episodes if e['case']['variant'] == 'doom_basic']
    require(len(selected) == len(basic) == 48 and {e['case']['id'] for e in basic} == set(selected),
            'Basic reference must cover the exact 48 declared cases')
    require(Counter(c['split'] for c in selected.values()) == {'train': 24, 'dev': 6, 'calibration': 6, 'test': 6, 'ood': 6},
            'Unexpected Basic split population')
    require(len({c['seed'] for c in selected.values()}) == 48, 'Basic seeds overlap between cases')
    journal = rows(args.journal)
    started, succeeded = {}, {}
    for row in journal:
        table = started if row['status'] == 'started' else succeeded if row['status'] == 'succeeded' else None
        if table is not None:
            require(row['id'] not in table, 'Duplicate journal status/call ID')
            table[row['id']] = row
    used, cache_hits, nonargmax, decisions = set(), 0, 0, 0
    normalization_max_delta = 0.0
    per_case = []
    for ep in basic:
        case, steps, final = ep['case'], ep['steps'], ep['final_info']
        cid = case['id']
        require(case == selected[cid], 'Changed source case definition: ' + cid)
        require(ep['complete'] is True and type(ep['success']) is bool and bool(steps), 'Incomplete Basic episode: ' + cid)
        require(ep['continuation_policy_id'] == manifest['continuation_policy_id'], 'Mixed reference policies')
        require(final['success'] is ep['success'] and final['terminated'] is True and final['truncated'] is False,
                'Final terminal information disagrees: ' + cid)
        require(final['episode_start_tick'] == 14 and final['native_timeout_tick'] == 300, 'Changed Basic native clock')
        require(final['scenario'] == 'basic' and final['mode'] == 'PLAYER' and
                final['buttons'] == ['MOVE_LEFT', 'MOVE_RIGHT', 'ATTACK'], 'Changed physical action contract')
        require(final['frame_skip'] == case['spec']['frame_skip'], 'Changed action repeat')
        rng = random.Random(int(digest([cid, 17])[:16], 16))
        for index, step in enumerate(steps):
            where = f'{cid} decision {index}'
            require(step['terminated'] is (index == len(steps)-1) and step['truncated'] is False,
                    'Invalid terminal sequence: ' + where)
            require(set(step['observation']['candidates']) == ACTIONS and set(step['scores']) == ACTIONS,
                    'Changed Basic candidate set: ' + where)
            answer = step['answers']['action']
            require(answer['type'] == 'choice', 'Reference is not an action Choice')
            call = succeeded.get(answer['source_api_call_id'])
            require(call is not None and call['id'] in started, 'Missing successful receipt chain: ' + where)
            request = policy_request(step['observation'], cid)
            expected_input = {'model': 'typesafe-ai/jev', 'state': request['state'], 'questions': request['questions']}
            require(call['input'] == expected_input, 'Receipt input differs from executed state/question: ' + where)
            require(answer['source_input_sha256'] == call['input_sha256'] == started[call['id']]['input_sha256'],
                    'Receipt input hash references disagree: ' + where)
            native = call['native_probs']['action']
            require(native == answer['native_probabilities'] and set(native) == ACTIONS and
                    all(type(p) in (int,float) and math.isfinite(p) and 0 <= p <= 1 for p in native.values()),
                    'Invalid native action probabilities: ' + where)
            mass = sum(native.values())
            require(mass > 0 and abs(mass-answer['native_sum']) <= 1e-12, 'Invalid native probability mass: ' + where)
            proxy = {a: native[a]/mass for a in native}
            # The merged episode sorts object keys. Summation order can differ
            # from the original JavaScript worker by one floating-point ulp.
            require(set(answer['probabilities']) == ACTIONS and answer['probabilities'] == step['scores'],
                    'Recorded answer/controller probabilities differ: ' + where)
            delta = max(abs(proxy[a]-answer['probabilities'][a]) for a in ACTIONS)
            normalization_max_delta = max(normalization_max_delta, delta)
            require(delta <= 1e-12, 'Recorded controller normalization mismatch: ' + where)
            behavior = behavior_distribution(step['scores'], 'greedy', .1)
            require(behavior == step['behavior_probs'] and choose(behavior, rng) == step['action'],
                    'Recorded action differs from actual frozen RNG/controller: ' + where)
            require(sorted(behavior.values()) == [.025,.025,.025,.925], 'Unexpected exploration probabilities')
            info = step['info']
            require(0 < info['actual_ticks'] <= info['requested_ticks'] <= case['spec']['frame_skip'],
                    'Invalid physical tick count: ' + where)
            require(info['terminated'] is step['terminated'] and info['truncated'] is False, 'Transition flags disagree')
            used.add(call['id']); cache_hits += bool(answer['cache_hit']); decisions += 1
            best = min(proxy, key=lambda action: (-proxy[action], action))
            nonargmax += step['action'] != best
        metric = final['episode_metrics']
        require(steps[-1]['info'] == final, 'Final information differs from last transition')
        require(sum(s['info']['actual_ticks'] for s in steps) == metric['physical_ticks'] <= 286,
                'Physical tick sum/deadline mismatch: ' + cid)
        require(metric['decisions'] == len(steps) and (metric['kills'] > 0) is ep['success'],
                'Success must be a positive observed kill delta: ' + cid)
        require(abs(math.fsum(s['reward'] for s in steps)-metric['native_reward']) <= 1e-9,
                'Native reward sum mismatch: ' + cid)
        per_case.append({'id': cid, 'split': case['split'], 'seed': case['seed'], 'spec': case['spec'],
                         'success': ep['success'], 'decisions': len(steps), 'metrics': metric,
                         'initial_observation_sha256': digest(steps[0]['observation']),
                         'episode_start_tick': 14, 'native_timeout_tick': 300, 'available_physical_ticks': 286})
    receipts = [succeeded[key] for key in sorted(used)]
    for receipt, hashes in zip(receipts, js_hashes(receipts, args.node)):
        require(hashes['input'] == receipt['input_sha256'] and hashes['response'] == receipt['response_sha256'],
                'Journal JSON.stringify SHA mismatch: ' + receipt['id'])
    require(replay['passed'] is True and not replay['errors'] and replay['episodes_sha256'] == hashes_before['episodes'],
            'Original CPU replay does not attest this source')
    require(replay['summary']['episodes'] == replay['summary']['passed_episodes'] == 192 and
            replay['summary']['mismatches'] == replay['summary']['failed_episodes'] == 0, 'Original CPU replay is incomplete')
    replay_eps = {e['id']: e for e in replay['episodes']}
    require(set(replay_eps) == set(all_ids) and all(e['passed'] and not e['errors'] and not e['mismatches']
                                                 for e in replay_eps.values()), 'Invalid per-episode CPU replay')
    for ep in basic:
        require(replay_eps[ep['case']['id']]['replayed_final_metrics'] == ep['final_info']['episode_metrics'],
                'Basic final metrics differ from original CPU replay')
    collected_sources, replay_sources = replay['collection_declared_source_sha256'], replay['replay_source_sha256']
    require(collected_sources == policy['source_sha256'], 'Original replay collection-source declaration differs')
    require(collected_sources['unified_doom_env.py'] == replay_sources['unified_doom_env.py'], 'Doom replay environment source differs')
    source_differences = {name: {'collection': collected_sources.get(name), 'replay': replay_sources.get(name)}
                          for name in collected_sources.keys() | replay_sources.keys()
                          if collected_sources.get(name) != replay_sources.get(name)}
    first = basic[0]['final_info']
    require(all(e['final_info']['scenario_config_sha256'] == first['scenario_config_sha256'] and
                e['final_info']['scenario_wad_sha256'] == first['scenario_wad_sha256'] and
                e['final_info']['vizdoom_version'] == first['vizdoom_version'] for e in basic), 'Mixed scenario binaries')
    require(hashes_before == {name: file_digest(path) for name,path in paths.items()}, 'Audit inputs changed during reading')
    return {'schema': 'nanojev-appo-jev-basic-reference-audit-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
            'passed': True, 'script_sha256': file_digest(__file__),
            'inputs': {name: {'path': str(path), 'sha256': hashes_before[name]} for name,path in paths.items()},
            'policy': policy, 'continuation_policy_id': manifest['continuation_policy_id'],
            'original_source_episodes': 192, 'basic_episodes': 48, 'decisions': decisions,
            'unique_bound_successful_receipts': len(used), 'cache_hits': cache_hits,
            'normalization_absolute_tolerance': 1e-12, 'normalization_max_absolute_difference': normalization_max_delta,
            'executed_nonargmax_actions': nonargmax, 'journal_status_counts': dict(Counter(r['status'] for r in journal)),
            'checks': {'all_48_case_definitions_exact': True, 'all_receipt_inputs_and_json_hashes': True,
                       'all_native_and_controller_probabilities': True, 'all_actual_action_rng': True,
                       'all_tick_reward_sums_and_kill_success': True, 'original_192_episode_cpu_replay_passed': True},
            'scenario': {key: first[key] for key in ('vizdoom_version','scenario_config_sha256','scenario_wad_sha256',
                         'mode','buttons','observation_source','history_length','episode_start_tick','native_timeout_tick',
                         'counter_source','physical_tick_source','action_repeat')},
            'split_summary': {split: {'episodes': len(selected_rows), 'successes': sum(r['success'] for r in selected_rows),
                'decisions': sum(r['decisions'] for r in selected_rows),
                'physical_ticks': sum(r['metrics']['physical_ticks'] for r in selected_rows),
                'mean_native_reward': math.fsum(r['metrics']['native_reward'] for r in selected_rows)/len(selected_rows)}
                for split in SPLITS if (selected_rows := [r for r in per_case if r['split'] == split])},
            'cases': per_case, 'original_cpu_replay_metadata': {key:value for key,value in replay.items() if key != 'episodes'},
            'collection_vs_original_replay_source_differences': source_differences,
            'comparison_contract': {'primary_controller': 'greedy', 'primary_epsilon': .1, 'sampling_seed': 17,
                'auxiliary_controllers': [{'controller':'greedy','epsilon':0.}, {'controller':'sample','epsilon':0.}],
                'same_case_ids_and_native_ticks_required': True,
                'observation_difference': 'Jev uses simulator-provided visible semantic boxes and player health/ammo/pose with history; pixel APPO has a different perception pipeline.',
                'claim_scope': 'Matched game cases and controls, not a controlled comparison of model capacity or identical observations.',
                'small_heldout_cohort': 'Six test and six OOD episodes; report each split without inferring statistical significance.'},
            'model_inference_calls': 0, 'api_calls': 0, 'new_physical_replays': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--episodes', type=Path, default=Path('data/unified_v2/jev_complete.jsonl'))
    parser.add_argument('--journal', type=Path, default=Path('runs/unified_jev_v2_journal/calls.jsonl'))
    parser.add_argument('--cases', type=Path, default=Path('configs/unified_games_v1_cases.jsonl'))
    parser.add_argument('--replay', type=Path, default=Path('runs/unified_jev_replay.json'))
    parser.add_argument('--node', default='node')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path; an existing audit is never overwritten')
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps({'passed': True, 'episodes': result['basic_episodes'], 'decisions': result['decisions'],
                      'output': str(args.output), 'sha256': file_digest(args.output)}))


if __name__ == '__main__':
    main()
