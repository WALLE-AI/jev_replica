#!/usr/bin/env python3
"""Verify and summarize the predeclared APPO Basic versus Jev experiment."""
import argparse
import json
from pathlib import Path

from summarize_unified_games import load_run_sources, summarize_run, check_cohort
from summarize_unified_cohort import cohort_view
from unified_game_pipeline import file_digest, read_rows, SPLITS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize_rendering(path, primary):
    raw = json.loads(Path(path).read_text())
    require(raw['exploratory'] and raw['not_part_of_primary_comparison'], 'Follow-up must remain separate')
    require(raw['implementation_sha256'] == file_digest(Path(__file__).with_name('diagnose_appo_doom_rendering.py')),
            'Rendering diagnostic source hash mismatch')
    cases = {c['id']: c for c in read_rows(primary['protocol']['case_file']) if c['spec'].get('scenario') == 'basic'}
    expected_models = [primary['protocol']['primary']['checkpoint_repository'],
                       *primary['protocol']['secondary_repositories']]
    require(set(r['model'] for r in raw['episodes']) == set(expected_models), 'Incomplete model set')
    result = {'source': str(path), 'source_sha256': file_digest(path), 'exploratory': True,
              'rendering': raw['rendering'], 'controller': 'greedy', 'epsilon': .1, 'sampling_seed': 17,
              'summary': {}, 'per_episode': [], 'independent_replay_passed': True,
              'replayed_decisions': sum(len(r['actions']) for r in raw['episodes'])}
    for model in expected_models:
        rows = [r for r in raw['episodes'] if r['model'] == model]
        require(len(rows) == len(cases) and {r['case']['id']: r['case'] for r in rows} == cases,
                'Follow-up must cover the same 48 cases exactly once per model')
        model_summary = {}
        for row in rows:
            require(row['independent_replay_passed'] and row['success'] == (row['metrics']['kills'] > 0),
                    'Missing independent replay or invalid success')
            require(sum(s['actual_ticks'] for s in row['actions']) == row['metrics']['physical_ticks'],
                    'Physical tick sum mismatch')
            result['per_episode'].append({k:v for k,v in row.items() if k != 'actions'})
        for split in SPLITS:
            split_rows = [r for r in rows if r['case']['split'] == split]
            model_summary[split] = {'n': len(split_rows), 'successes': sum(r['success'] for r in split_rows),
                'mean_metrics': {key: sum(r['metrics'][key] for r in split_rows) / len(split_rows)
                                 for key in ('physical_ticks', 'ammo_consumed', 'native_reward')}}
        result['summary'][model] = model_summary
    return result


def build(protocol_path, run_paths):
    protocol = json.loads(protocol_path.read_text())
    cases = {c['id']: c for c in read_rows(protocol['case_file'])
             if c['spec'].get('scenario') == 'basic' and c['spec']['task'] == 'shooting'}
    reference = protocol['reference']
    require(file_digest(reference['episodes']) == reference['sha256'], 'Jev source hash mismatch')
    complete_jev = load_run_sources('Jev', reference['episodes'])
    jev, selection = cohort_view(complete_jev, {'cases': cases})
    require(jev['policy']['epsilon'] == .1 and jev['policy']['controller'] == 'greedy', 'Jev controller mismatch')
    raw_jev = {r['case']['id']: r for r in read_rows(reference['episodes']) if r['case']['id'] in cases}
    repositories = [protocol['primary']['checkpoint_repository'], *protocol['secondary_repositories']]
    expected = {(repo, c['controller'], c['epsilon']) for repo in repositories
                for c in protocol['controllers_for_every_repository']}
    found, views, checks, primary_name = set(), [jev], {}, None
    for name, path in run_paths.items():
        run = load_run_sources(name, path)
        policy = run['policy']
        key = (policy['model'], policy['controller'], policy['epsilon'])
        require(key in expected and key not in found, f'Unexpected or repeated experiment: {key}')
        require(run['cases'] == cases, f'{name}: expected all original 48 Basic cases')
        require(policy['sampling_seed'] == 17 and policy['actor_updated_during_evaluation'] is False,
                f'{name}: sampling seed or inference contract changed')
        require(policy['strict_model_state_load'] and policy['torch_weights_only'], f'{name}: unsafe or partial load')
        require(policy['source_sha256']['unified_doom_env.py'] == jev['policy']['source_sha256']['unified_doom_env.py'],
                f'{name}: environment adapter changed')
        fields = ('vizdoom_version', 'scenario_config_sha256', 'scenario_wad_sha256', 'mode', 'buttons',
                  'frame_skip', 'max_steps', 'max_ticks', 'history_length', 'episode_start_tick',
                  'native_timeout_tick', 'counter_source', 'physical_tick_source', 'action_repeat')
        comparisons = []
        for row in read_rows(path):
            previous = raw_jev[row['case']['id']]
            a, b = row['final_info'], previous['final_info']
            require(all(a[f] == b[f] for f in fields), f'{name}: environment contract mismatch')
            require(row['steps'][0]['observation'] == previous['steps'][0]['observation'],
                    f'{name}: initial visible state mismatch')
            require(row['success'] == (a['episode_metrics']['kills'] > 0), 'Success must use kill count')
            comparisons.append({'id': row['case']['id'], 'split': row['case']['split'],
                                'appo_success': row['success'], 'jev_success': previous['success']})
        paired = {}
        for split in SPLITS:
            rows = [r for r in comparisons if r['split'] == split]
            paired[split] = {'n': len(rows),
                'appo_only_success': [r['id'] for r in rows if r['appo_success'] and not r['jev_success']],
                'jev_only_success': [r['id'] for r in rows if r['jev_success'] and not r['appo_success']],
                'both_success': sum(r['appo_success'] and r['jev_success'] for r in rows),
                'both_fail': sum(not r['appo_success'] and not r['jev_success'] for r in rows)}
        checks[name] = {'environment_and_initial_state_match': True, 'environment_fields': list(fields),
                        'paired_by_split': paired}
        main = protocol['primary']
        if key == (main['checkpoint_repository'], main['controller'], main['epsilon']):
            require(policy['model_revision'] == main['checkpoint_revision'] and
                    policy['checkpoint_sha256']['model'] == main['checkpoint_sha256'], 'Primary weights mismatch')
            primary_name = name
        found.add(key)
        views.append(run)
    require(found == expected and primary_name is not None, 'All nine predeclared runs must finish')
    return {'schema_version': 'nanojev-appo-basic-comparison-v1',
            'protocol': protocol, 'protocol_sha256': file_digest(protocol_path),
            'cohort': check_cohort(views), 'reference_source_selection': selection,
            'primary_run': primary_name, 'checks': checks,
            'runs': {r['name']: summarize_run(r) for r in views},
            'sources': {r['name']: r['sources'] for r in views},
            'notes': ['Success is a positive KILLCOUNT delta, not positive native reward.',
                      'Test and OOD have six independent environment cases each; checkpoints reuse these cases.',
                      'APPO receives RGB and recurrent image history; Jev receives structured visible-object text.',
                      'This measures complete game controllers with different observation representations.',
                      'Only the primary and epsilon=0.1 controls match the existing Jev exploration rate.',
                      'No API calls, optimization, or checkpoint uploads were performed.']}


def markdown(report):
    lines = ['# APPO Basic versus Jev', '',
             f"Primary run, fixed before APPO evaluation: **{report['primary_run']}**.", '',
             '| Controller | Train | Dev | Calibration | Test | OOD |',
             '|---|---:|---:|---:|---:|---:|']
    for name, run in report['runs'].items():
        values = []
        for split in SPLITS:
            s = run['by_split_task'][f'{split}/shooting']
            values.append(f"{s['successes']}/{s['n']} ({s['success_rate']:.1%})")
        lines.append('| ' + ' | '.join([name, *values]) + ' |')
    lines += ['', 'All rows cover the same 48 Basic cases. No controller was chosen using test outcomes.', '',
              '| Controller | Test mean ticks | OOD mean ticks | Test mean ammo consumed | OOD mean ammo consumed |',
              '|---|---:|---:|---:|---:|']
    for name, run in report['runs'].items():
        values = [run['by_split_task'][f'{split}/shooting']['metrics'][metric]['mean']
                  for metric in ['physical_ticks', 'ammo_consumed'] for split in ['test', 'ood']]
        lines.append('| ' + ' | '.join([name, *[f'{v:.2f}' for v in values]]) + ' |')
    lines += ['', *[f'- {note}' for note in report['notes']], '']
    if 'rendering_followup' in report:
        lines += ['## Training-rendering follow-up', '',
                  'Exploratory follow-up after verifying the original rendering settings on development cases.', '',
                  '| Checkpoint | Train | Dev | Calibration | Test | OOD |',
                  '|---|---:|---:|---:|---:|---:|']
        for model, cells in report['rendering_followup']['summary'].items():
            lines.append('| ' + ' | '.join([model, *[f"{cells[s]['successes']}/{cells[s]['n']}" for s in SPLITS]]) + ' |')
        lines += ['', 'All use greedy action choice with 10% uniform exploration and the original case deadlines.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('configs/appo_basic_v1.json'))
    parser.add_argument('--run', action='append', required=True, metavar='NAME=EPISODES')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rendering-report', type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), 'Use a fresh output directory')
    paths = {}
    for item in args.run:
        name, path = item.split('=', 1)
        require(name and name not in paths and name != 'Jev', 'Duplicate/reserved run name')
        paths[name] = path
    report = build(args.protocol, paths)
    if args.rendering_report:
        report['rendering_followup'] = summarize_rendering(args.rendering_report, report)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (args.output_dir / 'summary.md').write_text(markdown(report))
    print(markdown(report))


if __name__ == '__main__':
    main()
