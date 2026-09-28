#!/usr/bin/env python3
'''Summarize and compare completed NanoJev unified-game episode runs.

Recomputes episode-level performance from episode JSONL files validated against
their sidecar manifests, optionally attaches offline training diagnostics, and
writes summary.json plus summary.md into a new directory.

Standard library only. Never invents scores, never reads credentials, never
touches the network or a model. A comparison/metrics tool, not a trainer.
Initial implementation by Claude Fable 5.1 (xhigh), reviewed and patched locally.
'''
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 'nanojev-unified-episodes-v1'
SPLITS = ('train', 'dev', 'calibration', 'test', 'ood')
TASKS = ('maze', 'snake', 'shooting')
Z95 = 1.959963984540054
POLICY_FIELDS = ('engine', 'controller', 'effective_controller', 'epsilon', 'temperature', 'sampling_seed', 'tie_break',
                 'environment_contract', 'checkpoint_sha256', 'source_sha256')
CONTROL_FIELDS = ('effective_controller', 'epsilon', 'temperature', 'sampling_seed', 'tie_break', 'environment_contract')
TRAINING_FIELDS = ('best_step', 'best_dev_selection_ce', 'selected_on', 'stage', 'loss', 'completed_steps',
                   'training_seconds', 'max_gpu_allocated_gb', 'weights_sha256', 'continuation_policy_id',
                   'temperature', 'temperature_fitted')
NOTES = [
    'Performance results come only from the supplied completed episodes; manifest summaries are ignored and recomputed.',
    'Wilson 95% intervals are descriptive episode-level binomial intervals (the episode is the unit; transitions are never '
    'treated as independent samples). They are not paired significance tests.',
    'Task-macro success rates equally weight the tasks present in a split and carry no interval.',
    'Splits are reported separately and never combined into one headline average.',
    'Compared runs share an identical case cohort; the compared object is the policy (engine/controller/epsilon), '
    'so continuation_policy_id is expected to differ between runs.',
    'Training diagnostics, when present, are offline prediction metrics under the frozen collection policy and never '
    'replace rollout performance.',
    'A named run may combine disjoint verified source files. All source policy IDs and declarations are retained; '
    'a mixed-source run is not assigned a fabricated single policy ID.',
    'Steps/mean_steps count environment decision transitions, not physical movement attempts or game ticks. '
    'Non-forced decisions exclude explicitly forced transitions and are not a model-call count. '
    'Physical steps/ticks are reported separately only when supplied in episode_metrics.',
    'The random engine executes uniform sampling regardless of its historical controller argument; its effective '
    'controller is uniform_random. Original manifest declarations are preserved.',
]


def _require(cond, msg):
    if not cond:
        raise ValueError(msg)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def effective_controller(policy):
    return 'uniform_random' if policy.get('engine') == 'random' else policy.get('controller')


def policy_field(policy, field):
    return effective_controller(policy) if field == 'effective_controller' else policy.get(field)


def wilson(k, n, z=Z95):
    '''Wilson score interval [low, high] for k successes in n episodes; None when n == 0.'''
    _require(type(k) is int and type(n) is int and 0 <= k <= n, 'Wilson requires integer 0 <= k <= n')
    _require(is_num(z) and z > 0, 'Wilson z must be finite and positive')
    if n == 0:
        return None
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2.0 * n)
    m = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return [0.0 if k == 0 else max(0.0, (c - m) / d),
            1.0 if k == n else min(1.0, (c + m) / d)]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _validate_row(row, where, pid):
    _require(isinstance(row, dict), f'{where}: row must be a JSON object')
    case = row.get('case')
    _require(isinstance(case, dict), f'{where}: missing case')
    cid, split, seed, variant, spec = (case.get(k) for k in ('id', 'split', 'seed', 'variant', 'spec'))
    _require(isinstance(cid, str) and cid, f'{where}: case.id missing')
    _require(split in SPLITS, f'{where}: unknown split {split!r} (expected one of {SPLITS})')
    _require(isinstance(seed, int) and not isinstance(seed, bool), f'{where}: case.seed must be an integer')
    _require(isinstance(variant, str) and variant, f'{where}: case.variant missing')
    _require(isinstance(spec, dict) and spec.get('task') in TASKS, f'{where}: spec.task must be one of {TASKS}')
    _require(row.get('continuation_policy_id') == pid,
             f'{where}: continuation_policy_id {row.get("continuation_policy_id")!r} differs from manifest {pid!r} (mixed policy versions)')
    _require(row.get('complete') is True, f'{where}: episode is not complete')
    success = row.get('success')
    _require(isinstance(success, bool), f'{where}: success must be a real boolean outcome')
    final = row.get('final_info')
    _require(isinstance(final, dict) and final.get('success') is success,
             f'{where}: final_info.success must be a boolean equal to success')
    steps = row.get('steps')
    _require(isinstance(steps, list), f'{where}: steps must be a list')
    if not steps:
        _require(final.get('terminated') is True and final.get('truncated') is not True,
                 f'{where}: empty episode requires explicit terminal reset info without truncation')
    for i, st in enumerate(steps):
        _require(isinstance(st, dict) and 'action' in st and isinstance(st.get('terminated'), bool)
                 and isinstance(st.get('truncated'), bool), f'{where}: step {i} must have action and boolean terminated/truncated')
        _require(not st['truncated'], f'{where}: step {i} is a truncated transition')
        _require(not st['terminated'] or i == len(steps) - 1, f'{where}: step {i} terminated before the last step')
    if steps:
        _require(steps[-1]['terminated'], f'{where}: complete episode must end in a terminated transition')
    em = final.get('episode_metrics') or {}
    _require(isinstance(em, dict), f'{where}: episode_metrics must be an object')
    return ({'id': cid, 'split': split, 'seed': seed, 'variant': variant, 'spec': spec},
            {'id': cid, 'split': split, 'task': spec['task'], 'variant': variant, 'seed': seed, 'success': success,
             'steps': len(steps), 'decisions': sum(1 for st in steps if st.get('forced') is not True),
             'metrics': {k: float(v) for k, v in em.items() if is_num(v)}})


def load_run(name, episodes_path):
    ep_path = Path(episodes_path)
    man_path = ep_path.with_suffix('.manifest.json')
    _require(ep_path.is_file(), f'run {name!r}: episodes file not found: {ep_path}')
    _require(man_path.is_file(), f'run {name!r}: manifest not found: {man_path}')
    man = json.loads(man_path.read_text(encoding='utf-8'))
    _require(isinstance(man, dict) and man.get('schema_version') == SCHEMA, f'run {name!r}: manifest schema_version must be {SCHEMA!r}')
    _require(man.get('finished') is True, f'run {name!r}: manifest is not finished')
    ep_sha = sha256_file(ep_path)
    _require(man.get('episode_sha256') == ep_sha,
             f'run {name!r}: manifest episode_sha256 {man.get("episode_sha256")!r} does not match actual {ep_sha!r}')
    pid = man.get('continuation_policy_id')
    _require(isinstance(pid, str) and pid, f'run {name!r}: manifest continuation_policy_id missing')
    policy = man.get('policy')
    _require(isinstance(policy, dict) and isinstance(policy.get('engine'), str) and isinstance(policy.get('controller'), str),
             f'run {name!r}: manifest policy must include engine and controller')
    selected = man.get('selected_cases')
    _require(isinstance(selected, list) and selected and all(isinstance(c, str) for c in selected),
             f'run {name!r}: selected_cases must be a nonempty list of case ids')
    _require(len(set(selected)) == len(selected), f'run {name!r}: selected_cases contains duplicate ids')
    cases, episodes = {}, []
    with open(ep_path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f'run {name!r} line {lineno}'
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{where}: invalid JSON ({exc})') from None
            case, ep = _validate_row(row, where, pid)
            _require(case['id'] not in cases, f'{where}: duplicate case id {case["id"]!r}')
            cases[case['id']] = case
            episodes.append(ep)
    missing, extra = sorted(set(selected) - set(cases)), sorted(set(cases) - set(selected))
    _require(not missing and not extra,
             f'run {name!r}: selected_cases do not exactly match episode rows (missing {missing[:5]}, extra {extra[:5]})')
    return {'name': name, 'episodes_path': str(ep_path), 'manifest_path': str(man_path), 'episodes_sha256': ep_sha,
            'manifest_sha256': sha256_file(man_path), 'continuation_policy_id': pid, 'policy': policy,
            'cases_sha256': man.get('cases_sha256'), 'cases': cases, 'episodes': episodes}


def load_run_sources(name, paths):
    '''Join disjoint, individually verified files without inventing one policy identity.'''
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    _require(paths, f'run {name!r}: at least one source file is required')
    sources = [load_run(name, path) for path in paths]
    cases, episodes, case_policies, declared_ids = {}, [], {}, {}
    for source in sources:
        pid, declaration = source['continuation_policy_id'], canon(source['policy'])
        _require(pid not in declared_ids or declared_ids[pid] == declaration,
                 f'run {name!r}: one policy ID has conflicting source declarations')
        declared_ids[pid] = declaration
        overlap = sorted(set(cases) & set(source['cases']))
        _require(not overlap, f'run {name!r}: duplicate case ids across source files: {overlap[:5]}')
        cases.update(source['cases'])
        case_policies.update({cid: source['policy'] for cid in source['cases']})
        episodes.extend({**ep, 'continuation_policy_id': source['continuation_policy_id'],
                         'source_episodes_sha256': source['episodes_sha256'],
                         'effective_controller': effective_controller(source['policy'])} for ep in source['episodes'])
    ids = list(dict.fromkeys(source['continuation_policy_id'] for source in sources))
    # This is a view of common declared fields, not a new behavior-policy declaration.
    common = {key: value for key, value in sources[0]['policy'].items()
              if all(key in source['policy'] and canon(source['policy'][key]) == canon(value) for source in sources[1:])}
    provenance = [{key: source[key] for key in ('episodes_path', 'manifest_path', 'episodes_sha256',
                   'manifest_sha256', 'cases_sha256', 'continuation_policy_id', 'policy')}
                  | {'case_ids': sorted(source['cases']), 'effective_controller': effective_controller(source['policy'])}
                  for source in sources]
    result = {'name': name, 'cases': cases, 'episodes': episodes, 'case_policies': case_policies,
              'policy': common, 'continuation_policy_id': ids[0] if len(ids) == 1 else None,
              'continuation_policy_ids': ids, 'sources': provenance}
    for key in ('episodes_path', 'manifest_path', 'episodes_sha256', 'manifest_sha256', 'cases_sha256'):
        result[key] = sources[0][key] if len(sources) == 1 else [source[key] for source in sources]
    return result


def group_stats(eps):
    k, n = sum(1 for e in eps if e['success']), len(eps)
    metrics = {}
    for m in sorted({m for e in eps for m in e['metrics']}):
        vals = [e['metrics'][m] for e in eps if m in e['metrics']]
        metrics[m] = {'n': len(vals), 'mean': mean(vals), 'min': min(vals), 'max': max(vals)}
    return {'n': n, 'successes': k, 'success_rate': k / n, 'wilson95': wilson(k, n),
            'mean_steps': mean([e['steps'] for e in eps]), 'mean_decisions': mean([e['decisions'] for e in eps]),
            'metrics': metrics}


def summarize_run(run):
    eps = run['episodes']
    by_stv = {'/'.join(key): group_stats([e for e in eps if (e['split'], e['task'], e['variant']) == key])
              for key in sorted({(e['split'], e['task'], e['variant']) for e in eps})}
    by_st = {'/'.join(key): group_stats([e for e in eps if (e['split'], e['task']) == key])
             for key in sorted({(e['split'], e['task']) for e in eps})}
    macro = {}
    for split in sorted({e['split'] for e in eps}, key=SPLITS.index):
        tasks = sorted({e['task'] for e in eps if e['split'] == split})
        macro[split] = {'tasks': tasks, 'task_count': len(tasks),
                        'task_macro_success_rate': mean([by_st[f'{split}/{t}']['success_rate'] for t in tasks]),
                        'episodes': sum(1 for e in eps if e['split'] == split),
                        'note': 'Equally weighted mean of per-task pooled success rates; no interval is given for macro averages.'}
    return {'by_split_task_variant': by_stv, 'by_split_task': by_st, 'by_split_task_macro': macro, 'per_episode': eps}


def check_cohort(runs):
    base = runs[0]
    for other in runs[1:]:
        a, b = set(base['cases']), set(other['cases'])
        if a != b:
            raise ValueError(f'cohort mismatch between runs {base["name"]!r} and {other["name"]!r}: case id sets differ '
                             f'(only in first: {sorted(a - b)[:5]}, only in second: {sorted(b - a)[:5]})')
        changed = [cid for cid in sorted(a) if canon(base['cases'][cid]) != canon(other['cases'][cid])]
        if changed:
            raise ValueError(f'cohort mismatch between runs {base["name"]!r} and {other["name"]!r}: '
                             f'case definitions differ for ids {changed[:5]}')
    ids = sorted(base['cases'])
    return {'case_count': len(ids),
            'cohort_sha256': hashlib.sha256(canon([base['cases'][c] for c in ids]).encode('utf-8')).hexdigest(),
            'cases_sha256_by_run': {r['name']: r['cases_sha256'] for r in runs}}


def policy_differences(runs):
    out = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = runs[i], runs[j]
            def values(run, field):
                unique = {canon(policy_field(source['policy'], field)): policy_field(source['policy'], field) for source in run['sources']}
                found = list(unique.values())
                return found[0] if len(found) == 1 else found
            diff_counts = {f: sum(canon(policy_field(a['case_policies'][cid], f)) != canon(policy_field(b['case_policies'][cid], f))
                                  for cid in a['cases']) for f in POLICY_FIELDS}
            diff = {f: [values(a, f), values(b, f)] for f, count in diff_counts.items() if count}
            if a['continuation_policy_ids'] != b['continuation_policy_ids']:
                diff['continuation_policy_id'] = [a['continuation_policy_ids'][0] if len(a['continuation_policy_ids']) == 1 else a['continuation_policy_ids'],
                                                   b['continuation_policy_ids'][0] if len(b['continuation_policy_ids']) == 1 else b['continuation_policy_ids']]
            same = not any(diff_counts[f] for f in CONTROL_FIELDS)
            out.append({'runs': [a['name'], b['name']], 'differs': diff, 'same_controller': same,
                        'differing_case_counts': {f: n for f, n in diff_counts.items() if n},
                        'note': ('Effective controller settings and sampling seeds match case by case; inspect source/runtime provenance before attribution.' if same else
                                 'Controller, exploration, sampling seed, tie-break or environment contract differ; this is not an isolated same-controller model effect.')})
    return out


def comparison_tables(names, summaries):
    tables = {}
    for section, fields in (('by_split_task', ('n', 'successes', 'success_rate', 'wilson95', 'mean_decisions')),
                            ('by_split_task_variant', ('n', 'successes', 'success_rate', 'wilson95', 'mean_decisions')),
                            ('by_split_task_macro', ('tasks', 'task_count', 'task_macro_success_rate'))):
        keys = sorted({k for s in summaries.values() for k in s[section]},
                      key=(SPLITS.index if section == 'by_split_task_macro' else None))
        tables[section] = {k: {n: ({f: summaries[n][section][k][f] for f in fields} if k in summaries[n][section] else None)
                               for n in names} for k in keys}
    return tables


def load_training(name, path, run):
    p = Path(path)
    _require(p.is_file(), f'training summary for run {name!r} not found: {p}')
    t = json.loads(p.read_text(encoding='utf-8'))
    _require(isinstance(t, dict), f'training summary for run {name!r} must be a JSON object')
    run_weights = list(dict.fromkeys((source['policy'].get('checkpoint_sha256') or {}).get('best.safetensors')
                                    for source in run['sources']))
    run_weights = [weight for weight in run_weights if weight]
    tw = t.get('weights_sha256')
    if run_weights and tw:
        _require(all(tw == weight for weight in run_weights), f'training summary for run {name!r}: weights_sha256 {tw!r} disagrees with a run '
                              f'checkpoint best.safetensors in {run_weights!r}')
    mbs = t.get('metrics_by_split') or {}
    _require(isinstance(mbs, dict), f'training summary for run {name!r}: metrics_by_split must be an object')
    coll = t.get('continuation_policy_id')
    controllers = list(dict.fromkeys(source['effective_controller'] for source in run['sources']))
    return {'path': str(p), 'sha256': sha256_file(p), 'fields': {k: t.get(k) for k in TRAINING_FIELDS},
            'weights_match_run_checkpoint': all(tw == weight for weight in run_weights) if (run_weights and tw) else None,
            'collection_policy_matches_run_policy': all(coll == pid for pid in run['continuation_policy_ids']) if coll else None,
            'rollout_continuation_policy_ids': run['continuation_policy_ids'],
            'deployed_effective_controllers': controllers,
            'metrics_by_split': {sp: {'selection_ce': v.get('selection_ce'), 'by_task_role': v.get('by_task_role'),
                                      'by_role_macro_task': v.get('by_role_macro_task')}
                                 for sp, v in mbs.items() if isinstance(v, dict)},
            'label': (f'Offline prediction diagnostics from train_unified_games.py for run {name!r}. Outcome metrics '
                      f'(ce/brier/observed_accuracy) concern the named frozen collection policy {coll!r}, NOT automatic '
                      f'calibration under the newly deployed effective controller(s) {controllers!r} (run policies {run["continuation_policy_ids"]!r}). '
                      'The sampled paired-loss surrogate is neither a likelihood nor a direct game reward. '
                      'These metrics never replace actual rollout performance.')}


def build_summary(run_specs, training_specs=None):
    _require(run_specs, 'at least one --run NAME=path is required')
    runs = [load_run_sources(n, p) for n, p in run_specs.items()]
    by_name = {r['name']: r for r in runs}
    cohort = check_cohort(runs)
    summaries = {r['name']: summarize_run(r) for r in runs}
    training = {}
    for n, p in (training_specs or {}).items():
        _require(n in by_name, f'--training name {n!r} does not refer to a --run name (runs: {sorted(by_name)})')
        training[n] = load_training(n, p, by_name[n])
    return {'tool': 'scripts/summarize_unified_games.py',
            'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'inputs': {r['name']: {k: r[k] for k in ('episodes_path', 'episodes_sha256', 'manifest_path', 'manifest_sha256')}
                       | {'sources': r['sources']} for r in runs},
            'policies': {r['name']: {'continuation_policy_id': r['continuation_policy_id'],
                                    'continuation_policy_ids': r['continuation_policy_ids'],
                                    'policy': r['policy'], 'sources': r['sources']} for r in runs},
            'cohort': cohort, 'policy_differences': policy_differences(runs), 'runs': summaries,
            'comparison': comparison_tables(list(by_name), summaries), 'training_diagnostics': training, 'notes': NOTES}


def _f(x):
    if x is None:
        return 'n/a'
    return f'{x:.3f}' if isinstance(x, float) else str(x)


def _cell(c):
    if not c:
        return 'n/a'
    lo, hi = c['wilson95']
    return f'{c["successes"]}/{c["n"]} = {_f(c["success_rate"])} [{_f(lo)}, {_f(hi)}]'


def _table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows]
    return '\n'.join(lines)


def render_markdown(s):
    names = list(s['runs'])
    cmp_ = s['comparison']
    out = ['# NanoJev unified games summary', '',
           'Generated by scripts/summarize_unified_games.py from the supplied completed episodes only. '
           'Intervals are descriptive episode-level Wilson 95% binomial intervals, not paired significance tests. '
           'Splits are reported separately and never pooled into one headline number.', '',
           '## Inputs and provenance',
           _table(['Run', 'Episodes file', 'Episodes SHA256', 'Manifest SHA256'],
                  [[n, source['episodes_path'], source['episodes_sha256'], source['manifest_sha256']]
                   for n, i in s['inputs'].items() for source in i['sources']]), '',
           '## Policies',
           _table(['Run', 'Policy id', 'Engine', 'Declared controller', 'Effective controller', 'Epsilon', 'Temperature', 'best.safetensors SHA256'],
                  [[n, source['continuation_policy_id'], source['policy'].get('engine'), source['policy'].get('controller'),
                    source['effective_controller'], source['policy'].get('epsilon'), source['policy'].get('temperature'),
                    (source['policy'].get('checkpoint_sha256') or {}).get('best.safetensors', 'n/a')]
                   for n, p in s['policies'].items() for source in p['sources']]), '']
    c = s['cohort']
    out += ['## Cohort',
            f'- Cases: {c["case_count"]} (identical ids and canonical definitions across all runs); cohort SHA256 `{c["cohort_sha256"]}`',
            '- Source cases_sha256 by run: ' + ', '.join(f'{n}: `{v}`' for n, v in c['cases_sha256_by_run'].items()), '']
    if s['policy_differences']:
        out.append('## Policy differences between compared runs')
        for d in s['policy_differences']:
            diffs = ', '.join(f'{k}: {v[0]!r} vs {v[1]!r}' for k, v in d['differs'].items()) or 'no policy field differences'
            out.append(f'- {d["runs"][0]} vs {d["runs"][1]}: {diffs}. {d["note"]}')
        out.append('')
    out += ['## Success by split/task (episodes pooled within task; Wilson 95%)',
            _table(['Split/Task'] + names, [[k] + [_cell(v[n]) for n in names] for k, v in cmp_['by_split_task'].items()]), '',
            '## Success by split/task/variant (Wilson 95%)',
            _table(['Split/Task/Variant'] + names,
                   [[k] + [_cell(v[n]) for n in names] for k, v in cmp_['by_split_task_variant'].items()]), '',
            '## Split task-macro success rate (equally weighted present tasks; no interval)']
    rows = []
    for k, v in cmp_['by_split_task_macro'].items():
        tasks = next((v[n]['tasks'] for n in names if v[n]), [])
        rows.append([k, f'{len(tasks)}: ' + ', '.join(tasks)] + [_f(v[n]['task_macro_success_rate']) if v[n] else 'n/a' for n in names])
    out += [_table(['Split', 'Tasks'] + names, rows), '', '## Episode metrics by split/task (observed numeric metrics only)']
    rows = []
    for n in names:
        for k, g in s['runs'][n]['by_split_task'].items():
            for m, v in (list(g['metrics'].items()) or [('n/a', None)]):
                rows.append([n, k, g['n'], _f(g['mean_steps']), _f(g['mean_decisions']), m]
                            + ([v['n'], _f(v['mean']), _f(v['min']), _f(v['max'])] if v else ['n/a'] * 4))
    out += [_table(['Run', 'Split/Task', 'Episodes', 'Mean decision transitions', 'Mean non-forced decision transitions', 'Metric', 'Samples', 'Mean', 'Min', 'Max'], rows), '']
    if s['training_diagnostics']:
        out.append('## Training diagnostics (offline prediction metrics; not rollout performance)')
        for n, t in s['training_diagnostics'].items():
            out += [f'### {n}', t['label'], '', f'- Source: `{t["path"]}` (SHA256 `{t["sha256"]}`)',
                    '- ' + ', '.join(f'{k}: {v}' for k, v in t['fields'].items()),
                    f'- weights_match_run_checkpoint: {t["weights_match_run_checkpoint"]}; '
                    f'collection_policy_matches_run_policy: {t["collection_policy_matches_run_policy"]}']
            for split, m in t['metrics_by_split'].items():
                rows = [[tr] + [_f(v.get(f)) for f in ('questions', 'ce', 'brier', 'observed_accuracy', 'kl', 'tv')]
                        for tr, v in (m.get('by_task_role') or {}).items() if isinstance(v, dict)]
                out += ['', f'Split `{split}` selection_ce: {_f(m.get("selection_ce"))}',
                        _table(['Task/Role', 'Questions', 'CE', 'Brier', 'Observed accuracy', 'KL', 'TV'], rows)]
            out.append('')
    out += ['## Notes'] + [f'- {n}' for n in s['notes']] + ['']
    return '\n'.join(out)


def parse_named(items, flag):
    out = {}
    for item in items or []:
        name, sep, path = item.partition('=')
        _require(sep and name and path, f'{flag} expects NAME=path, got {item!r}')
        if flag == '--run':
            out.setdefault(name, []).append(path)
        else:
            _require(name not in out, f'duplicate {flag} name {name!r}; names must be unique')
            out[name] = path
    return out


def write_outputs(summary, output_dir):
    out = Path(output_dir)
    _require(not out.exists(), f'--output-dir must be a new directory, but {out} already exists')
    out.mkdir(parents=True)
    with open(out / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    (out / 'summary.md').write_text(render_markdown(summary), encoding='utf-8')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='Summarize and compare completed NanoJev unified-game episode runs.')
    ap.add_argument('--run', action='append', required=True, metavar='NAME=EPISODES_JSONL',
                    help='named completed run (repeat same name to merge disjoint verified sources)')
    ap.add_argument('--training', action='append', default=[], metavar='NAME=TRAINER_SUMMARY_JSON',
                    help='optional train_unified_games.py summary attached to the named run as offline diagnostics')
    ap.add_argument('--output-dir', required=True, help='new directory that will receive summary.json and summary.md')
    args = ap.parse_args(argv)
    try:
        summary = build_summary(parse_named(args.run, '--run'), parse_named(args.training, '--training'))
        out = write_outputs(summary, args.output_dir)
    except ValueError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    print(f'wrote {out / "summary.json"} and {out / "summary.md"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
