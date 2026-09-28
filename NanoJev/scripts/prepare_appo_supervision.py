#!/usr/bin/env python3
"""Replace only Basic policy rows with verified APPO expert demonstrations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from train_pipeline_decisions import SPLITS, validate_training_row
from train_unified_games import file_sha256, read_unified_dataset

DEFAULT_PROTOCOL = Path('configs/appo_basic_supervision_v1.json')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def load_protocol(path=DEFAULT_PROTOCOL):
    path = Path(path)
    protocol = json.loads(path.read_text())
    if protocol.get('schema_version') != 'nanojev-appo-supervision-v1':
        raise ValueError('Wrong Basic supervision protocol')
    case_path = Path(protocol['case_file'])
    if file_sha256(case_path) != protocol['case_sha256']:
        raise ValueError('Frozen cases changed')
    cases = [json.loads(line) for line in case_path.read_text().splitlines() if line.strip()]
    ids = [c['id'] for c in cases]
    counts = dict(Counter(c['split'] for c in cases))
    if (not cases or len(set(ids)) != len(ids) or set(counts) != set(SPLITS)
            or counts != protocol['case_counts']):
        raise ValueError('Protocol must declare unique cases and complete coverage of all five splits')
    groups = set()
    for case in cases:
        spec = case['spec']
        group = (spec.get('scenario'), case['seed'])
        if spec.get('task') != 'shooting' or spec.get('scenario') != 'basic' or group in groups:
            raise ValueError('Protocol cases must be distinct Basic environment seeds')
        groups.add(group)
    return protocol, cases, file_sha256(path)


def validate_collection_binding(collection, protocol, cases, protocol_sha256):
    if collection.get('finished') is not True or type(collection.get('limit')) is not int or collection['limit'] != 0:
        raise ValueError('Only a completed full collection may enter the unified experiment')
    if collection.get('cases_sha256') != protocol['case_sha256'] or collection.get('config_sha256') != protocol_sha256:
        raise ValueError('Expert collection belongs to a different case file or protocol')
    if collection.get('selected_cases') != [c['id'] for c in cases] or collection.get('selected_cases_sha256') != canonical_hash(cases):
        raise ValueError('Expert collection must cover the full registered case sequence')
    if any(collection.get(k) != len(cases) for k in ('episodes', 'selected_episodes', 'selected_before_limit')):
        raise ValueError('Expert episode counts do not cover the complete protocol')
    if collection.get('split_episodes') != protocol['case_counts']:
        raise ValueError('Expert collection is missing registered split coverage')
    policy = collection['policy']
    if collection.get('continuation_policy_id') != canonical_hash(policy):
        raise ValueError('Expert policy identity hash mismatch')
    primary, identity = protocol['primary'], policy.get('model', {})
    expected_identity = {'model': primary['checkpoint_repository'],
                         'revision': primary['checkpoint_revision'],
                         'checkpoint_sha256': primary['checkpoint_sha256']}
    if any(identity.get(key) != value for key, value in expected_identity.items()):
        raise ValueError('Expert model identity differs from the registered checkpoint')
    if policy.get('checkpoint_sha256', {}).get('model') != primary['checkpoint_sha256']:
        raise ValueError('Expert checkpoint hash differs from the protocol')
    for key in ('controller', 'epsilon', 'sampling_seed'):
        if policy.get(key) != primary[key]:
            raise ValueError('Expert collection controller differs from the protocol: ' + key)
    if policy.get('engine') != 'sample_factory_appo' or policy.get('render_adapter') != 'mirrored_native':
        raise ValueError('Expert collection must use the registered mirrored native APPO actor')


def validate_expert_rows(records, manifest, collection, cases, kind):
    """Bind complete Basic rows to cases, policy identity and target provenance."""
    if (manifest.get('episode_sha256') != collection['episode_sha256'] or
            manifest.get('continuation_policy_id') != collection['continuation_policy_id'] or
            manifest.get('collection_policy') != collection['policy'] or
            manifest.get('policy_target_kind') != kind):
        raise ValueError('Expert dataset and collection provenance disagree')
    registry = {case['id']: case for case in cases}
    visits, counts = {}, Counter()
    identity = collection['policy']['model']
    for row in records:
        if not basic(row):
            continue  # The runner also validates the retained non-Basic corpus.
        meta = row['metadata']
        case = registry.get(meta.get('episode_id'))
        if (case is None or row['split'] != case['split'] or meta.get('spec') != case['spec'] or
                meta.get('continuation_policy_id') != collection['continuation_policy_id'] or
                meta.get('policy_target_kind') != kind):
            raise ValueError('Basic training row differs from its registered case or policy')
        expert = row.get('expert', {})
        if any(expert.get(key) != identity.get(key) for key in ('model', 'revision', 'checkpoint_sha256')):
            raise ValueError('Basic training target names a different expert checkpoint')
        index = meta.get('decision_index')
        if type(index) is not int or index < 0 or index in visits.setdefault(case['id'], set()):
            raise ValueError('Repeated or invalid expert decision index')
        visits[case['id']].add(index)
        counts[row['split']] += 1
    if set(visits) != set(registry) or any(indices != set(range(len(indices))) for indices in visits.values()):
        raise ValueError('Basic rows must contain every registered episode and all contiguous decisions')
    if dict(counts) != collection.get('split_records'):
        raise ValueError('Basic row counts disagree with the full collection')


def validate_episode_registry(path, collection, cases):
    registry = {case['id']: case for case in cases}
    seen, counts = [], Counter()
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            episode = json.loads(line)
            case = episode['case']
            if (registry.get(case['id']) != case or episode.get('complete') is not True or
                    episode.get('continuation_policy_id') != collection['continuation_policy_id']):
                raise ValueError('Expert episode is incomplete or outside the registered protocol')
            seen.append(case['id'])
            counts[case['split']] += len(episode['steps'])
    if seen != collection['selected_cases'] or dict(counts) != collection['split_records']:
        raise ValueError('Physical episode file does not contain the declared full collection')


def basic(row):
    meta = row['metadata']
    return meta['task'] == 'shooting' and meta.get('spec', {}).get('scenario') == 'basic'


def prepare(original, expert, output, protocol_path=DEFAULT_PROTOCOL):
    original, expert, output = map(Path, (original, expert, output))
    if output.exists():
        raise ValueError('Use a new output directory')
    protocol, cases, protocol_sha256 = load_protocol(protocol_path)
    old_rows, old_manifest, old_files, _ = read_unified_dataset(original)
    for split in SPLITS:
        if old_manifest.get('split_sha256', {}).get(split) != file_sha256(original / (split + '.jsonl')):
            raise ValueError('Original corpus split hash mismatch')
    collection = json.loads((expert / 'episodes.manifest.json').read_text())
    validate_collection_binding(collection, protocol, cases, protocol_sha256)
    if collection.get('episode_sha256') != file_sha256(expert / 'episodes.jsonl'):
        raise ValueError('Expert episode file hash mismatch')
    validate_episode_registry(expert / 'episodes.jsonl', collection, cases)
    if any(row['metadata']['record_role'] != 'policy' for row in old_rows):
        raise ValueError('The retained corpus must contain policy rows only')
    outputs, audits = {}, {}
    for arm, kind in [('hard', 'expert_action'), ('soft', 'expert_distribution')]:
        source = expert / arm
        new_rows, new_manifest, new_files, _ = read_unified_dataset(source)
        if new_manifest.get('episode_sha256') != collection['episode_sha256']:
            raise ValueError('Expert dataset and collection manifest disagree')
        if not new_rows or any(not basic(r) or r['metadata'].get('policy_target_kind') != kind for r in new_rows):
            raise ValueError('Only explicitly labeled Basic expert rows may replace Basic')
        for split in SPLITS:
            expected = new_manifest.get('split_sha256', {}).get(split)
            if expected != file_sha256(source / (split + '.jsonl')):
                raise ValueError('Expert dataset split hash mismatch')
        for row in new_rows:
            validate_training_row(row)
        validate_expert_rows(new_rows, new_manifest, collection, cases, kind)
        source_by_id = {r['id']: r for r in new_rows}
        if arm == 'soft':
            hard_rows = outputs['hard']['rows']
            if set(source_by_id) != set(hard_rows):
                raise ValueError('Hard and soft arms must have identical expert states')
            for key, row in source_by_id.items():
                hard = hard_rows[key]
                if any(row[field] != hard[field] for field in ('state', 'questions', 'gold', 'split', 'state_id')):
                    raise ValueError('Hard and soft inputs or argmax labels disagree')
        outputs[arm] = {'rows': source_by_id, 'splits': {}, 'manifest': new_manifest}
        retained, removed, added = Counter(), Counter(), Counter()
        for split in SPLITS:
            # Retention copies exact JSONL bytes, including source metadata.
            lines = []
            for line in (original / (split + '.jsonl')).read_text().splitlines(keepends=True):
                if not line.strip():
                    continue
                row = json.loads(line)
                if basic(row):
                    removed[split] += 1
                else:
                    lines.append(line if line.endswith('\n') else line + '\n')
                    retained[(split, row['metadata']['task'], row['metadata'].get('spec', {}).get('scenario', ''))] += 1
            for line in (source / (split + '.jsonl')).read_text().splitlines(keepends=True):
                if line.strip():
                    lines.append(line if line.endswith('\n') else line + '\n')
                    added[split] += 1
            outputs[arm]['splits'][split] = ''.join(lines)
        audits[arm] = {'original_source_sha256': {str(p): file_sha256(p) for p in old_files},
                       'expert_source_sha256': {str(p): file_sha256(p) for p in new_files},
                       'retained_rows': {'/'.join(k): v for k, v in sorted(retained.items())},
                       'removed_basic_rows': dict(removed), 'added_basic_rows': dict(added)}
    output.mkdir(parents=True)
    for arm, data in outputs.items():
        dest = output / arm
        dest.mkdir()
        for split, text in data['splits'].items():
            (dest / (split + '.jsonl')).write_text(text)
        manifest = {'schema_version': 'nanojev-unified-training-v1',
                    'protocol_sha256': protocol_sha256,
                    'continuation_policy_id': None,
                    'policy_only': True,
                    'target': 'expert_action' if arm == 'hard' else 'expert_distribution',
                    'retention_rule': 'All original non-Basic rows retained verbatim; Basic replaced in every split.',
                    'original_manifest': old_manifest, 'expert_manifest': data['manifest'],
                    'collection_manifest': collection,
                    'split_sha256': {s: file_sha256(dest / (s + '.jsonl')) for s in SPLITS}, **audits[arm]}
        (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        _, _, _, audit = read_unified_dataset(dest)
        audits[arm]['validation'] = audit
    (output / 'preparation_audit.json').write_text(json.dumps(audits, indent=2) + '\n')
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--expert', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    audits = prepare(args.original, args.expert, args.output, args.protocol)
    print(json.dumps({arm: {'added': a['added_basic_rows'], 'retained': a['retained_rows']}
                      for arm, a in audits.items()}))


if __name__ == '__main__':
    main()
