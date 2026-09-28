#!/usr/bin/env python3
"""Prepare complete, credential-free private Hub packages for the unified model.

No network calls occur here. Checkpoint bytes, data splits, frozen evaluation
records and the source implementation are retained exactly and SHA256-bound.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

SELECTED_SHA = 'f68c47d66998231b86b7e91b4ed5e82ae23acf104c8b7cd6d165c3ac7b7ffe1b'
INITIAL_SHA = '38116340795de1c82369b7fe15819d92d79600a7b4dc7a3cd0d4390cb6782639'
MODEL_REPO = 'C-Tianyu/NanoJev-dev'
DATA_REPO = 'C-Tianyu/NanoJev-Data-dev'
SPLITS = ('train', 'dev', 'calibration', 'test', 'ood')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def copy_file(source, target):
    source, target = Path(source), Path(target)
    if not source.is_file() or source.is_symlink():
        raise ValueError('Expected a regular source file: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(source) != sha(target): raise ValueError('Package path collision: ' + str(target))
        return
    # Only immutable weights share inodes; source/data snapshots are independent.
    if source.suffix == '.safetensors':
        try: os.link(source, target)
        except OSError: shutil.copyfile(source, target)
    else:
        shutil.copyfile(source, target)


def copy_tree(source, target):
    for file in sorted(Path(source).rglob('*')):
        if file.is_file() and not any(part.startswith('.') or part == '__pycache__' for part in file.relative_to(source).parts):
            copy_file(file, Path(target) / file.relative_to(source))


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def stats(data):
    out = {}
    for target in ('hard', 'soft'):
        manifest = json.loads((data / target / 'manifest.json').read_text())
        out[target] = {}
        for split in SPLITS:
            path = data / target / (split + '.jsonl')
            assert sha(path) == manifest['split_sha256'][split]
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            pools = Counter(row['metadata']['task'] + ('/' + row['metadata']['spec']['scenario']
                            if row['metadata']['task'] == 'shooting' else '') for row in rows)
            out[target][split] = {'rows': len(rows), 'by_task': dict(pools), 'sha256': sha(path), 'bytes': path.stat().st_size}
        assert sum(r['rows'] for r in out[target].values()) == 18760
    return out


def manifest(root, kind, extra):
    records = []
    forbidden = re.compile(rb'(?:hf_[A-Za-z0-9]{24,}|sk-(?:proj-)?[A-Za-z0-9_-]{32,})')
    for path in sorted(root.rglob('*')):
        if not path.is_file(): continue
        relative = path.relative_to(root).as_posix()
        assert not any(part.startswith('.') for part in path.relative_to(root).parts)
        assert path.name not in {'.env', 'calls.jsonl', 'label_journal.jsonl'}
        if path.suffix in ('.json', '.jsonl', '.py', '.mjs', '.md', '.txt', '.jinja'):
            with path.open('rb') as handle:
                for line in handle:
                    if forbidden.search(line): raise ValueError('Potential credential in staged file: ' + relative)
        records.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    payload = {'schema': 'nanojev-unified-hf-package-v1', 'kind': kind,
               'created_at': datetime.now(timezone.utc).isoformat(), 'files': records, **extra}
    write(root / 'SHA256_MANIFEST.json', json.dumps(payload, indent=2) + '\n')
    return {'files': len(records) + 1, 'bytes': sum(r['bytes'] for r in records) + (root / 'SHA256_MANIFEST.json').stat().st_size,
            'manifest_sha256': sha(root / 'SHA256_MANIFEST.json')}


RECIPE = '''# Reproduce the unified supervised training run

The selected artifact is `hard_lr1e5`, seed 17, checkpoint step 400 from a
600-update run. Selection uses development data: each arm's minimum weighted
dev CE, then weighted game success across the four arms and initialization.
The test and OOD cohorts do not select a checkpoint. No RL or TD update is active
in this release. The loss is complete-question categorical cross entropy.

## Exact inputs

- Model root: selected step-400 bundle, SHA256 `SELECTED_SHA`.
- `training_initialization/`: the exact prior unified Basic-supervised bundle,
  SHA256 `INITIAL_SHA`; a fresh optimizer is created for this round.
- Dataset `unified/hard/`: all five original mixed-task split files, unchanged.
  `unified/soft/` is the matched comparison arm's input.
- Raw expert data: `expert/predict_position/episodes.jsonl` (896 episodes).
- Preparation input: `preparation/original_unified_hard/`, containing the exact
  previous mixed dataset. Maze, Snake and Basic rows are retained byte-for-byte.
- `configs/sonic_unified_sft_v1.json`: all four arms and the frozen selection rule.
- `evaluation/experiment/`: initial and four-arm dev trajectories, selection,
  selected/initial full test trajectories, metrics and replay checks.
- `evaluation/{jev,native}/`: the other full 548-case test/OOD recordings.
- `demonstrations/`: fixed 50x50 Maze / 12x12 Snake case registry, six source
  recordings and validated export. These use their recorded code planners and
  remain separate from the 548-case benchmark.

Stored training rows total 18,760 per target variant. The trainer quarantines
11 invalid-target questions across the splits, leaving 10,893 eligible training
questions out of 10,898 stored training rows. The eligible training pools are
Maze 651, Snake 400, Basic 3,054 and Predict Position 6,788.

## Environment and selected arm

The recorded environment is Python 3.14.4, PyTorch 2.14.0, Transformers 5.17.0,
Safetensors 0.8.0, NumPy 2.5.3, ViZDoom 1.3.0 and A100 80GB.
FP32 weights are trained with BF16 forward computation and gradient checkpointing.
Install the matching packages from `source/requirements-toy.txt` and
`source/requirements-vizdoom.txt` in a GPU environment.

After authenticated `snapshot_download` of both private development repositories,
set `MODEL_DIR` and `DATA_DIR` to those absolute local paths and run from the model folder:

```bash
python source/scripts/train_unified_games.py \\
  --input "$DATA_DIR/unified/hard" \\
  --init-checkpoint "$MODEL_DIR/training_initialization" \\
  --output-dir ./reproduced_hard_lr1e5 --stage sft --loss ce --balance task \\
  --policy-pool-weights "$DATA_DIR/configs/sonic_policy_pool_weights.json" \\
  --steps 600 --head-steps 0 --seed 17 \\
  --batch-questions 24 --microbatch-questions 8 --max-microbatch-tokens 32768 \\
  --max-length 8192 --eval-every 100 \\
  --backbone-lr 1e-5 --head-lr 1e-4 --weight-decay 0.01 \\
  --precision bf16 --gradient-checkpointing --disable-native-triton
```

Every batch contains 8 Maze, 8 Snake, 4 Basic and 4 Predict Position questions.
Their loss weights are respectively 1/3, 1/3, 1/6 and 1/6. Rows are sampled with
replacement. Across 600 updates the PP sampler draws 2,400 questions covering
2,021 distinct questions; the selected step-400 checkpoint sees the matching
shorter prefix. The backbone/head learning-rate comparison is 1e-5/1e-4 versus
2e-5/2e-4, for both hard and soft PP targets. All arms keep the same sample order.

The complete original configs, hashes, logs, target audit, predictions and
selection receipts are included. Run the data-only check before training:

```bash
python source/scripts/train_unified_games.py \\
  --input "$DATA_DIR/unified/hard" --stage sft --loss ce \\
  --policy-pool-weights "$DATA_DIR/configs/sonic_policy_pool_weights.json" \\
  --validate-only
```

To replay the six hard navigation examples without model or API calls, change
into `$DATA_DIR/demonstrations` and run:

```bash
ln -s "$MODEL_DIR/source/scripts" scripts
python "$MODEL_DIR/source/scripts/build_hard_navigation_demo.py" --output ./rebuilt_hard_navigation.json --receipt ./rebuilt_hard_navigation_receipt.json
```

The `scripts` link lets the replay validator resolve the original relative runner
paths while checking their exact source hashes. The frozen source paths are
preserved under that folder. Source paths, recorded protocols
and all referenced source hashes appear in its `results/hard_navigation_demo_v1`
receipt. Inference uses `source/scripts/predict_toy_decisions.py`, whose
`DecisionPredictor` loads the custom dynamic candidate-scoring head and backbone.
'''.replace('SELECTED_SHA', SELECTED_SHA).replace('INITIAL_SHA', INITIAL_SHA)

MODEL_CARD = '''---
language:
- en
base_model: Qwen/Qwen3-0.6B
library_name: pytorch
tags:
- nanojev
- decision-model
- reinforcement-learning-environment
- supervised-fine-tuning
- maze
- snake
- vizdoom
---
# NanoJev-dev — A nano replica of Jev

One 0.6B model returns action distributions for Maze, Snake, ViZDoom Basic and
Predict Position. This private development release contains the current unified
step-400 model, its exact training initialization and the full inference source.
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) inspired the
state, question and candidate-set interface.

Each candidate is encoded with its state and question. A shared Qwen3-0.6B
backbone, end-of-sequence representations and an attention-based Choice head
score the offered set. Softmax returns all candidate probabilities together,
without output-token decoding. Boolean and ordered-score request types remain
available through the same structured interface.

| Matched test task | NanoJev | Jev | Untuned Qwen |
|---|---:|---:|---:|
| Maze | 4/10 | 7/10 | 2/10 |
| Snake | 8/8 | 8/8 | 0/8 |
| Basic | 128/128 | 56/128 | 56/128 |
| Predict Position | 27/128 | 11/128 | 11/128 |

All systems share the recorded structured observation interface, frozen cases,
controller epsilon 0.1 and sampling seed 17. Full test and OOD reports, source
trajectories and independent replay checks are in the linked dataset. The hard
50x50 Maze and 12x12 Snake demonstrations are a separately recorded showcase
using common code planners: current NanoJev reaches the maze goal in 225 attempts
and collects 30 food items while surviving all 256 Snake steps.

## Load the model

[Complete mixed dataset and recordings](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data-dev)
· [Training recipe](TRAINING_RECIPE.md)
· [File hashes](SHA256_MANIFEST.json)

Authenticate with a Hugging Face account that can read this private repository.
The root directory is the current complete checkpoint bundle; it is loaded by
NanoJev's `DecisionPredictor`.

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("C-Tianyu/NanoJev-dev", revision="unified-games-v1")
sys.path.insert(0, path + "/source/scripts")
from predict_toy_decisions import DecisionPredictor

model = DecisionPredictor(path, device_name="cuda:0", precision="bf16",
                          disable_native_triton=True)
request = {"states": [{"id": "example", "state": "The target is left of the aim.",
    "questions": {"action": {"type": "choice", "instructions": "Choose the next action.",
        "criteria": {"left": "Move the aim left.", "right": "Move the aim right.",
                     "shoot": "Fire.", "noop": "Wait."}}}}]}
answer = model.predict(request)
print(answer)
```

The saved format contains `best.safetensors`, `config.json`, `backbone_config/`
and `tokenizer/`, plus the original training audits. It is the project's decision
model format. The base revision is
`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`.

Training uses mixed-task supervised cross entropy. Predict Position action
supervision comes from the released Sonic Doom visual expert. The existing
Maze, Snake and Basic data remain in their original splits. Four matched runs
compare hard/soft PP targets and two learning rates; development performance
selects `hard_lr1e5` at step 400. The selected weight SHA256 is
`SELECTED_SHA`. The full original 15-file bundle is preserved.

Source code carries the included MIT license; the Qwen base model retains its
upstream license. Model and data repositories stay private during this review.
'''.replace('SELECTED_SHA', SELECTED_SHA)

DATA_CARD = '''---
language:
- en
tags:
- nanojev
- supervised-fine-tuning
- decision-making
- maze
- snake
- vizdoom
size_categories:
- 10K<n<100K
configs:
- config_name: hard
  default: true
  data_files:
  - split: train
    path: unified/hard/train.jsonl
  - split: validation
    path: unified/hard/dev.jsonl
  - split: calibration
    path: unified/hard/calibration.jsonl
  - split: test
    path: unified/hard/test.jsonl
  - split: ood
    path: unified/hard/ood.jsonl
- config_name: soft
  data_files:
  - split: train
    path: unified/soft/train.jsonl
  - split: validation
    path: unified/soft/dev.jsonl
  - split: calibration
    path: unified/soft/calibration.jsonl
  - split: test
    path: unified/soft/test.jsonl
  - split: ood
    path: unified/soft/ood.jsonl
---
# NanoJev-Data-dev — Unified game supervision and recorded evaluation

The complete data package for the current [NanoJev-dev model](https://huggingface.co/C-Tianyu/NanoJev-dev):
Maze, Snake, ViZDoom Basic and Predict Position. It includes the exact mixed
supervised-learning inputs, full expert episodes, frozen evaluation cohorts,
recorded comparisons, and the six-source hard Maze/Snake demonstration.

## Training data

| Split | Rows per hard/soft variant |
|---|---:|
| Train | 10,898 |
| Dev | 1,715 |
| Calibration | 1,709 |
| Test | 2,496 |
| OOD | 1,942 |
| Total | 18,760 |

The hard and soft variants share the same states, questions and candidates.
Only the new Predict Position target changes: expert argmax action versus its
complete action distribution. The selected checkpoint uses `hard`.

Both variants preserve all 7,587 existing Maze, Snake and Basic rows byte for
byte. Predict Position contributes 11,173 questions across the five splits,
including 6,788 training questions. The raw 896 expert episodes contain 17,498
decisions; the policy-training view retains pre-action states with ammunition,
including unsuccessful episodes and the actual firing decisions. Full episodes
also retain later states for audit and evaluation.

Rows contain `id`, `state`, `questions`, `split`, metadata and recorded target
annotations. The policy input is only state/question/candidate content; target
annotations, terminal outcomes and expert state are excluded from inference.
The existing target-validity filter yields 10,893 eligible training questions
and quarantines 11 questions over all splits. Stored source rows remain intact.

Expert policy source: [Sonic Doom](https://github.com/thainv0212/sonic_doom),
ordinary no-sound Predict Position visual policy with a recurrent GRU. Basic
supervision uses the existing APPO expert collection. Maze and Snake preserve
their existing action annotations. Complete provenance stays in each exact
source manifest and record; no target annotations are rewritten during packaging.

## Evaluation and demonstrations

The frozen benchmark has 548 cases: 274 test and 274 OOD. Four primary systems
have complete recordings: current NanoJev, its initialization, Jev and untuned
Qwen. All 2,192 episodes pass independent simulator replay. Source manifests,
model identities, actual action probabilities and physical counters are included.

| Test task | Current NanoJev | Jev | Untuned Qwen |
|---|---:|---:|---:|
| Maze | 4/10 | 7/10 | 2/10 |
| Snake | 8/8 | 8/8 | 0/8 |
| Basic | 128/128 | 56/128 | 56/128 |
| Predict Position | 27/128 | 11/128 | 11/128 |

Hard demonstrations are separate fixed examples: 50x50 Maze, seed 24310922;
12x12 Snake, seed 61005. They preserve six real model trajectories and the exact
common controller definitions. They are not additional samples in the 548-case
benchmark. The included current visualization indices are derived artifacts;
original recorded trajectories are also present.

See [the exact training recipe](TRAINING_RECIPE.md), [dataset counts](DATA_STATISTICS.json),
and [all file hashes](SHA256_MANIFEST.json). Download this private review package
with an authorized Hugging Face login:

```python
from huggingface_hub import snapshot_download
path = snapshot_download("C-Tianyu/NanoJev-Data-dev", repo_type="dataset",
                         revision="unified-games-v1")
```

Use the JSONL split files directly with the bundled project training scripts.
Split names and bytes match the original experiment. Test/OOD are held out from
training and checkpoint selection. Scenario/seed groups are split-disjoint;
individual visible observations can repeat between different episodes.
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/hf_unified_release_v1/packages'))
    args = parser.parse_args(argv)
    assert not args.output.exists(), 'Use a new package output directory'
    root = Path.cwd()
    selected = root / 'runs/sonic_unified_sft_v1/experiment/hard_lr1e5'
    initial = root / 'runs/appo_basic_supervision_v1/experiment/hard_s17'
    assert sha(selected / 'best.safetensors') == SELECTED_SHA
    assert sha(initial / 'best.safetensors') == INITIAL_SHA
    config = json.loads((selected / 'config.json').read_text())
    summary = json.loads((selected / 'summary.json').read_text())
    assert summary['best_step'] == 400 and summary['completed_steps'] == 600
    assert config['implementation_sha256'] == sha(root / 'scripts/train_unified_games.py')
    data_root = root / 'data/sonic_supervision_v1/unified'
    data_stats = stats(data_root)
    for source_path, expected in config['data_sha256'].items():
        assert sha(data_root / 'hard' / Path(source_path).name) == expected
    model, dataset = args.output / 'model', args.output / 'dataset'
    copy_tree(selected, model)
    copy_tree(initial, model / 'training_initialization')
    copy_tree(data_root, dataset / 'unified')
    copy_tree(root / 'data/appo_basic_supervision_v1/unified/hard', dataset / 'preparation/original_unified_hard')
    copy_tree(root / 'runs/sonic_unified_sft_v1/expert', dataset / 'expert/predict_position')
    # Exact source files for loading, training, environment replay and export.
    for p in sorted((root / 'scripts').iterdir()):
        if p.is_file() and p.suffix in ('.py', '.mjs') and not p.name.startswith(('test_', 'check_')):
            copy_file(p, model / 'source/scripts' / p.name)
    for name in ('requirements-toy.txt', 'requirements-vizdoom.txt', 'requirements-shooting-demo.txt', 'package.json', 'package-lock.json', 'LICENSE'):
        copy_file(root / name, model / 'source' / name)
    for p in (root / 'configs').iterdir():
        if p.is_file() and (p.name.startswith(('sonic_', 'hard_navigation_demo_', 'unified_basic_demo_', 'predict_position_demo_'))):
            copy_file(p, dataset / 'configs' / p.name)
    run = root / 'runs/sonic_unified_sft_v1'
    for name in ('dev_cases.jsonl', 'test_cases.jsonl', 'case_selection.json', 'data_quality.json', 'sampling_audit.json'):
        copy_file(run / name, dataset / 'evaluation' / name)
    exp = run / 'experiment'
    for p in exp.iterdir():
        if p.is_file() and not p.name.startswith('.'):
            copy_file(p, dataset / 'evaluation/experiment' / p.name)
        elif p.is_dir() and p.name.endswith('.sources'):
            copy_tree(p, dataset / 'evaluation/experiment' / p.name)
    for arm in ('hard_lr1e5', 'hard_lr2e5', 'soft_lr1e5', 'soft_lr2e5'):
        for name in ('config.json', 'summary.json', 'train_log.json'):
            copy_file(exp / arm / name, dataset / 'evaluation/experiment' / arm / name)
    for target, source in [('jev', run / 'jev_parallel_recovery_v2/episodes.jsonl'), ('native', run / 'native_test.jsonl')]:
        copy_file(source, dataset / 'evaluation' / target / source.name)
        copy_file(source.with_suffix('.manifest.json'), dataset / 'evaluation' / target / source.with_suffix('.manifest.json').name)
        copy_tree(source.with_suffix('.sources'), dataset / 'evaluation' / target / source.with_suffix('.sources').name)
        replay = run / ('jev_test_replay.json' if target == 'jev' else 'native_test_replay.json')
        copy_file(replay, dataset / 'evaluation' / target / 'independent_replay.json')
    copy_tree(run / 'final_results', dataset / 'evaluation/final_results')
    demo = dataset / 'demonstrations'
    demo_receipt = json.loads((root / 'results/hard_navigation_demo_v1/build_manifest.json').read_text())
    copy_file(root / 'configs/hard_navigation_demo_v1_cases.jsonl', demo / 'configs/hard_navigation_demo_v1_cases.jsonl')
    for example in demo_receipt['examples']:
        for source in example['sources']:
            p = root / source['path']
            assert sha(p) == source['sha256']
            copy_file(p, demo / source['path'])
    for name in ('web/arcade_results.json', 'web/side_by_side_results.json',
                 'assets/arcade_data_manifest.json', 'assets/side_by_side_data_manifest.json',
                 'results/rollout_pilot_episodes.jsonl', 'results/side_by_side_maze_episode.jsonl',
                 'results/arcade_snake_cohort.jsonl', 'results/model_edges_local_atomic.json',
                 'results/model_edges_initial.json', 'runs/arcade_snake/trained_greedy.json'):
        copy_file(root / name, demo / name)
    for name in ('side_by_side_results.json', 'shooting_results.json', 'predict_position_results.json'):
        copy_file(root / 'web/dev' / name, demo / 'web/dev' / name)
    for folder in ('hard_navigation_demo_v1', 'unified_basic_demo_v1', 'predict_position_demo_v1', 'predict_position_wins_v2'):
        copy_tree(root / 'results' / folder, demo / 'results' / folder)
    for name in ('SONIC_PREDICT_POSITION.md', 'SONIC_PREDICT_POSITION_RESULTS.md'):
        copy_file(root / 'docs' / name, dataset / 'docs' / name)
    write(model / 'README.md', MODEL_CARD)
    write(dataset / 'README.md', DATA_CARD)
    write(model / 'TRAINING_RECIPE.md', RECIPE)
    write(dataset / 'TRAINING_RECIPE.md', RECIPE)
    write(dataset / 'DATA_STATISTICS.json', json.dumps({'stored': data_stats,
        'eligible_by_split_sampling_pool': config['eligible_by_split_sampling_pool'],
        'quarantined_policy_questions': config['schema_counts']['quarantined_policy_questions'],
        'selected_target': 'hard', 'selected_checkpoint_step': 400}, indent=2) + '\n')
    common = {'selected_checkpoint_sha256': SELECTED_SHA, 'initial_checkpoint_sha256': INITIAL_SHA,
              'model_repo': MODEL_REPO, 'dataset_repo': DATA_REPO, 'private': True,
              'source_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'package_builder_sha256': sha(__file__)}
    reports = {'model': manifest(model, 'model', common), 'dataset': manifest(dataset, 'dataset', common)}
    output = {'passed': True, **common, 'packages': reports, 'data_statistics': data_stats}
    write(args.output.parent / 'package_inventory.json', json.dumps(output, indent=2) + '\n')
    print(json.dumps({'passed': True, 'packages': reports, 'checkpoint_sha256': SELECTED_SHA}), flush=True)


if __name__ == '__main__':
    main()
