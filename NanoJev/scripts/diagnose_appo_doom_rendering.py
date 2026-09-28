#!/usr/bin/env python3
"""Exploratory rendering diagnostic, separate from the frozen primary benchmark.

Restore the public policy's training rendering settings without changing game
actions, map, case seeds, deadlines, or success detection. This changes visual
observations and must never be merged into the predeclared primary results.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from evaluate_appo_doom import (SampleFactoryPolicy, map_action_distribution,
                               sample_with_receipt, preprocess_frame, TRAIN_SOURCE_COMMIT)
from unified_game_pipeline import read_rows, behavior_distribution, digest, file_digest
from unified_doom_env import UnifiedDoomEnv


def native_env(spec):
    env = UnifiedDoomEnv(spec)
    try:
        env._initialize()
        game = env._game
        game.close()
        game.set_screen_resolution(env._vzd.ScreenResolution.RES_160X120)
        game.set_render_hud(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.init()
        return env
    except Exception:
        env.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--cases', type=Path, default=Path('configs/unified_games_v1_cases.jsonl'))
    parser.add_argument('--splits', default='dev')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    import cv2
    import numpy as np
    cases = [c for c in read_rows(args.cases) if c['split'] in args.splits.split(',')
             and c['spec'].get('scenario') == 'basic']
    results = []
    for seed in ('1111', '2222', '3333'):
        model_dir = args.model_dir / f'doom_basic_{seed}'
        manifest = json.loads((model_dir / 'download_manifest.json').read_text())
        weight = next(x for x in manifest['files'] if x['file'].endswith('.pth'))
        checkpoint = model_dir / weight['file']
        assert file_digest(checkpoint) == weight['sha256']
        policy = SampleFactoryPolicy(checkpoint, model_dir / 'cfg.json', runtime_source_commit=TRAIN_SOURCE_COMMIT)
        for case in cases:
            actions = []
            rng = random.Random(int(digest([case['id'], 17])[:16], 16))
            with native_env(case['spec']) as env:
                game = env._game
                _, info = env.reset(case['seed'])
                initial_player_variables = dict(env._current)
                initial_clock = info['episode_start_tick']
                policy.reset_episode()
                while not info['terminated']:
                    raw = game.get_state().screen_buffer.copy()
                    assert raw.shape == (120, 160, 3)
                    # Reuse the frozen loader. Integer nearest upsampling followed
                    # by its resize is checked to equal the native training resize.
                    expanded = cv2.resize(raw, (320, 240), interpolation=cv2.INTER_NEAREST)
                    expected = np.ascontiguousarray(cv2.resize(raw, (128, 72),
                                                   interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1))
                    assert np.array_equal(preprocess_frame(expanded)[0], expected)
                    result = policy.predict_pixels(expanded)
                    probs = map_action_distribution(result['sf_logits'], result['sf_probabilities'])['policy_probs']
                    behavior = behavior_distribution(probs, 'greedy', .1)
                    action, draw = sample_with_receipt(behavior, rng)
                    _, reward, _, _, info = env.step(action)
                    actions.append({'action': action, 'policy_probs': probs, 'behavior_probs': behavior,
                                    'sampling_draw': draw, 'actual_ticks': info['actual_ticks'],
                                    'reward': reward, 'native_frame_sha256': hashlib.sha256(raw.tobytes()).hexdigest()})
                record = {'model': manifest['repo'], 'checkpoint_sha256': weight['sha256'],
                          'case': case, 'success': info['success'], 'metrics': info['episode_metrics'],
                          'initial_player_variables': initial_player_variables,
                          'initial_episode_tick': initial_clock, 'actions': actions}
                # Independent environment replay checks physics and outcomes.
                with native_env(case['spec']) as replay:
                    replay.reset(case['seed'])
                    assert replay._current == initial_player_variables
                    for step in actions:
                        raw_replay = replay._game.get_state().screen_buffer.copy()
                        assert hashlib.sha256(raw_replay.tobytes()).hexdigest() == step['native_frame_sha256']
                        _, reward_replay, _, _, final_replay = replay.step(step['action'])
                        assert reward_replay == step['reward']
                        assert final_replay['actual_ticks'] == step['actual_ticks']
                    assert final_replay['terminated'] and final_replay['episode_metrics'] == record['metrics']
                record['independent_replay_passed'] = True
                results.append(record)
                print(json.dumps({k:v for k,v in record.items() if k!='actions'}), flush=True)
    summary = {}
    for model in sorted({r['model'] for r in results}):
        summary[model] = {}
        for split in args.splits.split(','):
            rows = [r for r in results if r['model']==model and r['case']['split']==split]
            summary[model][split] = {'successes':sum(r['success'] for r in rows), 'n':len(rows)}
    report = {'schema_version':'nanojev-appo-rendering-diagnostic-v1', 'exploratory':True,
              'selection':'All three public checkpoints; fixed greedy epsilon=0.1, sampling seed17',
              'rendering':{'size_wh':[160,120],'hud':True,'decals':False,'particles':False},
              'gameplay_adapter_sha256':file_digest(Path(__file__).with_name('unified_doom_env.py')),
              'implementation_sha256':file_digest(__file__), 'cases_sha256':file_digest(args.cases),
              'not_part_of_primary_comparison':True, 'summary':summary, 'episodes':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'summary':summary}),flush=True)


if __name__=='__main__':
    main()
