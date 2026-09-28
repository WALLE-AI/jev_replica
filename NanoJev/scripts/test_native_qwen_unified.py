"""CPU checks of the unified native actor; fake logits make no capability claim."""
import copy
import hashlib
import math
import random
import unittest

from evaluate_native_qwen_navigation import BACKEND, build_prompt
from evaluate_native_qwen_shooting import iter_episodes as basic_episodes
from evaluate_native_qwen_unified import (
    iter_episodes, select_unified_cases, validate_observation,
)
from replay_unified_episodes import replay_episode
import test_appo_basic_data as mirror_fixtures
from unified_game_pipeline import behavior_distribution, choose, digest, policy_request


class FakeNative:
    def __init__(self, corrupt=None):
        self.calls = 0
        self.requests = []
        self.corrupt = corrupt

    def predict(self, payload, batch_questions=0, temperature=1.0):
        assert batch_questions == 0 and temperature == 1
        self.calls += 1
        self.requests.extend(copy.deepcopy(payload['states']))
        states = []
        for request in payload['states']:
            prompt, labels = build_prompt(request)
            total = sum(range(1, len(labels) + 1))
            probs = {key: (i+1)/total for i, key in enumerate(labels)}
            answer = {'type': 'choice', 'backend': BACKEND,
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'candidate_to_token': {key: {'text': label, 'id': 32+i}
                    for i, (key, label) in enumerate(labels.items())},
                'probabilities': probs,
                'native_option_logits': {key: math.log(p) for key, p in probs.items()},
                'native_option_unconditional_probs': {key: .5*p for key, p in probs.items()},
                'offered_token_mass': .5}
            states.append({'id': request['id'], 'answers': {'action': answer}})
        response = {'states': states, 'execution': {'forward_passes': 1, 'generated_tokens': 0}}
        if self.corrupt:
            self.corrupt(response)
        return response


def case(identity='case', seed=17, split='test', task='shooting', scenario='basic'):
    spec = {'task': task, 'max_steps': 6}
    if task == 'shooting':
        spec.update(scenario=scenario, frame_skip=4)
    else:
        spec.update(size=8)
    return {'id': identity, 'seed': seed, 'split': split, 'variant': task, 'spec': spec}


class TinyEnvironments:
    """One forced step, then two scored steps, with a finite physical deadline."""
    def __init__(self, initial_terminal=False):
        self.initial_terminal = initial_terminal
        self.indices = {}
        self.closed = False
        self.actions = []

    def view(self, key):
        index = self.indices[key]
        terminal = index == 3
        return {'observation': {'task': 'snake',
            'state': 'Complete public state, including body and food. Step ' + str(index),
            'candidates': {} if terminal else ({'north': 'Move north'} if index == 0 else {
                'north': 'Move north; collision risk remains a candidate', 'west': 'Move west'}),
            'step': index, 'remaining_steps': 3-index},
            'info': {'terminated': terminal, 'truncated': False, 'success': False,
                     'episode_metrics': {'physical_steps': index}}}

    def reset(self, cases):
        self.indices = {c['id']: 3 if self.initial_terminal else 0 for c in cases}
        return {key: self.view(key) for key in self.indices}

    def step(self, actions):
        self.actions.append(copy.deepcopy(actions))
        result = {}
        for key, action in actions.items():
            assert action in self.view(key)['observation']['candidates']
            self.indices[key] += 1
            result[key] = self.view(key)
            result[key].update(reward=0., terminated=self.indices[key] == 3, truncated=False)
        return result

    def close(self):
        self.closed = True


class UnifiedNativeTests(unittest.TestCase):
    def test_selection_supports_four_games_and_preserves_case_order(self):
        cases = [case('pp', scenario='predict_position'), case('maze', task='maze'),
                 case('snake', task='snake'), case('basic')]
        self.assertEqual(select_unified_cases(cases, ['test']), cases)

    def test_split_leak_is_rejected_before_filtering(self):
        with self.assertRaisesRegex(ValueError, 'crosses splits'):
            select_unified_cases([case('train', split='train'), case('test')], ['test'])
        with self.assertRaises(ValueError):
            select_unified_cases([case()], ['test', 'test'])

    def test_forced_step_has_no_forward_but_consumes_rng_draw(self):
        env = TinyEnvironments()
        predictor = FakeNative()
        episode, = iter_episodes([case()], predictor, 'fixture', epsilon=.1,
                                 environments_factory=lambda: env)
        self.assertEqual(predictor.calls, 2)
        self.assertEqual(len(episode['steps']), 3)
        forced = episode['steps'][0]
        self.assertTrue(forced['forced'])
        self.assertFalse(forced['model_forward'])
        self.assertEqual(forced['answers'], {})
        self.assertIsNone(forced['prediction_execution'])
        rng = random.Random(int(digest(['case', 17])[:16], 16))
        for step in episode['steps']:
            distribution = behavior_distribution(step['scores'], 'greedy', .1)
            saved = rng.getstate()
            self.assertEqual(step['sampling_draw'], rng.random())
            rng.setstate(saved)
            self.assertEqual(step['action'], choose(distribution, rng))
            self.assertEqual(step['request'], policy_request(step['observation'], 'case'))
        self.assertEqual(episode['final_observation']['candidates'], {})
        self.assertFalse(episode['success'])
        self.assertTrue(episode['complete'])
        self.assertTrue(env.closed)

    def test_initial_terminal_does_not_call_model_or_step(self):
        env, predictor = TinyEnvironments(initial_terminal=True), FakeNative()
        episode, = iter_episodes([case()], predictor, 'fixture', environments_factory=lambda: env)
        self.assertEqual(predictor.calls, 0)
        self.assertEqual(env.actions, [])
        self.assertEqual(episode['steps'], [])
        self.assertTrue(episode['complete'])

    def test_batch_size_does_not_change_actions_or_rng(self):
        cases = [case(str(i), seed=100+i) for i in range(5)]
        for controller in ('greedy', 'sample'):
            serial = list(iter_episodes(cases, FakeNative(), 'fixture', controller,
                batch_states=1, environments_factory=TinyEnvironments))
            batched = list(iter_episodes(cases, FakeNative(), 'fixture', controller,
                batch_states=4, environments_factory=TinyEnvironments))
            self.assertEqual(serial, batched)

    def test_wrong_response_coverage_or_generation_fails_and_closes(self):
        for mutate in (lambda r: r['states'].append(r['states'][0]),
                       lambda r: r['states'].clear(),
                       lambda r: r['execution'].update(generated_tokens=1),
                       lambda r: r['states'][0]['answers']['action'].update(prompt_sha256='wrong')):
            env = TinyEnvironments()
            with self.assertRaises(ValueError):
                list(iter_episodes([case()], FakeNative(mutate), 'fixture',
                                   environments_factory=lambda: env))
            self.assertTrue(env.closed)

    def test_malformed_terminal_or_candidate_contract_fails(self):
        env = TinyEnvironments()
        baseline = env.reset([case()])['case']
        for mutate in (lambda x: x['info'].update(truncated=True),
                       lambda x: x['info'].update(terminated=True),
                       lambda x: x['info'].update(success=True),
                       lambda x: x['observation'].update(remaining_steps=0),
                       lambda x: x['observation'].update(remaining_steps=True),
                       lambda x: x['observation'].update(candidates={str(i): 'move' for i in range(5)})):
            broken = copy.deepcopy(baseline)
            mutate(broken)
            with self.assertRaises(ValueError):
                validate_observation(broken['observation'], broken['info'])

    def test_real_grid_simulators_replay_to_the_actual_deadline(self):
        cases = [case('maze', seed=47, task='maze'), case('snake', seed=48, task='snake')]
        episodes = list(iter_episodes(cases, FakeNative(), 'fixture'))
        for episode in episodes:
            self.assertTrue(episode['complete'])
            self.assertTrue(episode['steps'][-1]['terminated'])
            self.assertFalse(episode['steps'][-1]['truncated'])
            self.assertTrue(replay_episode(episode)['passed'])

    def test_basic_rollout_matches_existing_script_and_physical_replay(self):
        fixture = mirror_fixtures.MirrorTests()
        fixture.setUp()
        try:
            cases = [copy.deepcopy(fixture.case)]
            cases[0]['split'] = 'test'
            old, = basic_episodes(cases, FakeNative(), 'fixture')
            new, = iter_episodes(cases, FakeNative(), 'fixture')
            self.assertEqual(old['final_info'], new['final_info'])
            self.assertEqual(len(old['steps']), len(new['steps']))
            for a, b in zip(old['steps'], new['steps']):
                for key, value in a.items():
                    self.assertEqual(value, b[key], key)
            self.assertTrue(replay_episode(new)['passed'])
            self.assertEqual([s['info']['actual_ticks'] for s in new['steps']], [4, 1])
        finally:
            fixture.tearDown()


if __name__ == '__main__':
    unittest.main()
