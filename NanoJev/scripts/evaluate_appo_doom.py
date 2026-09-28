#!/usr/bin/env python3
"""Evaluate a local Sample Factory APPO Basic policy in the unchanged Doom adapter.

The policy receives RGB pixels only. Recorded text requests support the existing
environment replay format; they are not passed to the APPO network. Heavy optional
dependencies are imported only by SampleFactoryPolicy, never by CLI validation.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import random
import shutil
import time

from unified_game_pipeline import (
    SPLITS, behavior_distribution, digest, encode, file_digest, policy_request,
    read_rows, source_hashes, summarize, validate_cases,
)

MODEL_ID = "edbeeching/doom_basic_1111"
REVISION = "fae3bfcd5ab5f98804fe4327a0be982776c8d7d3"
CHECKPOINT_SHA256 = "cc6d1657c0ab8dc4a75c7159fac97108ecb83131e613f6473b096b19348024cb"
TRAIN_SOURCE_COMMIT = "9da68b57eecd73c3c884c1be2d938b46aa7a7f49"
SF_ACTIONS = ("noop", "left", "right", "shoot")


def validate_config(cfg):
    """Require the published Basic observation/action contract, not generic Doom."""
    expected = {"env": "doom_basic", "res_w": 128, "res_h": 72,
                "pixel_format": "CHW", "env_frameskip": 4, "env_framestack": 1,
                "use_rnn": True, "rnn_type": "lstm", "rnn_size": 512,
                "rnn_num_layers": 1, "normalize_input": True,
                "normalize_returns": True, "obs_subtract_mean": 0.0,
                "obs_scale": 255.0, "actor_critic_share_weights": True}
    if not isinstance(cfg, dict):
        raise ValueError("Checkpoint cfg must be an object")
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Unsupported checkpoint contract: {key} must equal {value!r}")
    return cfg


def select_cases(rows, splits):
    validate_cases(rows)
    if not splits or len(splits) != len(set(splits)) or set(splits) - set(SPLITS):
        raise ValueError("Select distinct known splits")
    selected = [row for row in rows if row["split"] in splits and
                row["spec"].get("task") == "shooting" and row["spec"].get("scenario") == "basic"]
    if not selected:
        raise ValueError("No Basic cases in selected splits")
    for row in selected:
        if row["spec"].get("frame_skip", 4) not in (4, 8):
            raise ValueError("Use the frozen four-tick test or eight-tick OOD interval")
    return selected


def map_action_distribution(logits, probabilities):
    """Preserve raw SF probabilities and explicitly normalize float32 roundoff."""
    if len(logits) != 4 or len(probabilities) != 4:
        raise ValueError("Basic checkpoint must return exactly four categorical actions")
    if any(not math.isfinite(x) for x in logits):
        raise ValueError("Nonfinite APPO logits")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Invalid APPO probabilities")
    total = math.fsum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-5):
        raise ValueError("APPO probabilities do not form a simplex")
    largest = max(logits)
    norm = math.fsum(math.exp(x-largest) for x in logits)
    if any(abs(p-math.exp(z-largest)/norm) > 1e-5 for p,z in zip(probabilities,logits)):
        raise ValueError("APPO logits and probabilities disagree")
    return {"logits": dict(zip(SF_ACTIONS, logits)),
            "policy_probs_raw": dict(zip(SF_ACTIONS, probabilities)),
            "policy_probs": {key: probabilities[i]/total for i,key in enumerate(SF_ACTIONS)},
            "raw_probability_sum": total, "normalization_applied": total != 1.0}


def sample_with_receipt(probs, rng):
    """Same sorted-key inverse-CDF selection as unified_game_pipeline.choose."""
    draw, cumulative = rng.random(), 0.0
    for key in sorted(probs):
        cumulative += probs[key]
        if draw < cumulative:
            return key, draw
    return sorted(probs)[-1], draw


def preprocess_frame(frame, width=128, height=72):
    import cv2
    import numpy as np
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.shape != (240,320,3):
        raise ValueError("UnifiedDoomEnv must supply uint8 320x240 RGB24 HWC pixels")
    if (width,height) != (128,72):
        raise ValueError("This Basic checkpoint requires 128x72")
    raw = np.ascontiguousarray(frame)
    resized = cv2.resize(raw, (width,height), interpolation=cv2.INTER_NEAREST)
    chw = np.ascontiguousarray(resized.transpose(2,0,1))
    return chw, {"source": "DoomGame.get_state().screen_buffer", "color": "RGB24",
                 "raw_shape_hwc": list(raw.shape), "input_shape_chw": list(chw.shape),
                 "raw_uint8_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                 "resized_uint8_sha256": hashlib.sha256(chw.tobytes()).hexdigest(),
                 "resize": "cv2.INTER_NEAREST", "text_request_fed_to_policy": False}


def tensor_digest(tensor):
    value = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


class SampleFactoryPolicy:
    """One frozen recurrent actor; each episode starts with zero h and c."""

    def __init__(self, checkpoint, cfg_path, device="cpu", torch_threads=1,
                 runtime_source_commit=None):
        import torch
        import numpy as np
        import gym
        from sample_factory.model import actor_critic as actor_module
        from sample_factory.model.model_utils import get_rnn_size
        from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
        from sample_factory.utils.attr_dict import AttrDict
        from sample_factory.algo.utils.context import global_model_factory
        from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
        self.torch = torch
        torch.set_num_threads(torch_threads)
        self.device = torch.device(device)
        saved = validate_config(json.loads(Path(cfg_path).read_text()))
        global_model_factory().register_encoder_factory(make_vizdoom_encoder)
        self.cfg = AttrDict({**saved, "device": device})
        # The pinned 2022 Sample Factory source uses gym, not gymnasium.
        space = gym.spaces.Dict({"obs": gym.spaces.Box(0,255,shape=(3,72,128),dtype=np.uint8)})
        model = actor_module.create_actor_critic(self.cfg, space, gym.spaces.Discrete(4))
        model.model_to_device(self.device)
        try:
            # Only legacy NumPy numeric metadata needs a scoped allowlist.
            core = getattr(np, "_core", None)
            if core is None:
                core = np.core
            numeric_types = [(core.multiarray.scalar, "numpy.core.multiarray.scalar"),
                             np.dtype, type(np.dtype(np.float64))]
            with torch.serialization.safe_globals(numeric_types):
                payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        except Exception as error:
            raise RuntimeError("weights_only=True could not load this checkpoint; no unsafe-pickle fallback was attempted") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
            raise ValueError("Expected Sample Factory checkpoint['model'] state dict")
        model.load_state_dict(payload["model"], strict=True)
        model.eval().requires_grad_(False)
        self.model, self.prepare = model, prepare_and_normalize_obs
        self.rnn_size = get_rnn_size(self.cfg)
        if self.rnn_size != 1024:
            raise ValueError("Expected one 512-unit LSTM: 1024 packed h+c values")
        self.rnn = None
        self.reset_count = self.forward_count = 0
        runtime_files = {"actor_critic": inspect.getfile(actor_module),
                         "prepare_and_normalize_obs": inspect.getfile(prepare_and_normalize_obs),
                         "get_rnn_size": inspect.getfile(get_rnn_size),
                         "vizdoom_encoder": inspect.getfile(make_vizdoom_encoder)}
        self.identity = {"sample_factory_distribution_version": importlib.metadata.version("sample-factory"),
                         "runtime_source_commit_declared": runtime_source_commit,
                         "runtime_source_file_sha256": {name:file_digest(path) for name,path in runtime_files.items()},
                         "defaults_added_to_saved_cfg": [],
                         "resolved_model_cfg_sha256": digest(dict(self.cfg)),
                         "strict_model_state_load": True, "torch_weights_only": True,
                         "weights_only_numeric_allowlist": ["numpy.core.multiarray.scalar", "numpy.dtype", "numpy.float64_dtype_class"],
                         "device": device, "precision": "float32", "torch_threads": torch_threads,
                         "rnn_type": "lstm", "rnn_packed_size": self.rnn_size,
                         "rnn_reset": "zero h+c at every episode reset",
                         "normalization": "checkpoint normalizer via prepare_and_normalize_obs in eval mode",
                         "training_render_size_wh": [160,120], "evaluation_render_size_wh": [320,240],
                         "network_image_size_wh": [128,72], "training_frame_skip": 4,
                         "evaluation_frame_skip": "Each original case specification, including eight-tick OOD",
                         "render_contract": "UnifiedDoomEnv unchanged; direct nearest resize from its RGB framebuffer",
                         "parameter_count": sum(p.numel() for p in model.parameters()),
                         "deps": {name:importlib.metadata.version(name) for name in ("torch","numpy","vizdoom")}}

    def reset_episode(self):
        self.rnn = self.torch.zeros((1,self.rnn_size),dtype=self.torch.float32,device=self.device)
        self.reset_count += 1

    def predict_pixels(self, frame):
        if self.rnn is None:
            raise RuntimeError("Reset recurrent state before inference")
        pixels, receipt = preprocess_frame(frame)
        before = tensor_digest(self.rnn)
        with self.torch.inference_mode():
            normalized = self.prepare(self.model,{"obs":self.torch.from_numpy(pixels).unsqueeze(0)})
            normalized_hash = tensor_digest(normalized["obs"])
            head = self.model.forward_head(normalized)
            core, following = self.model.forward_core(head,self.rnn)
            output = self.model.forward_tail(core,values_only=False,sample_actions=False)
            logits = output["action_logits"].detach().float().cpu()
            probabilities = self.model.action_distribution().probs.detach().float().cpu()
            if logits.shape != (1,4) or probabilities.shape != (1,4):
                raise ValueError("Unexpected APPO categorical shape")
            if following.shape != self.rnn.shape or not self.torch.isfinite(following).all():
                raise ValueError("Invalid next recurrent state")
            self.rnn = following.detach().clone()
        self.forward_count += 1
        receipt.update(normalized_float32_sha256=normalized_hash, rnn_before_sha256=before,
                       rnn_after_sha256=tensor_digest(self.rnn), rnn_shape=list(self.rnn.shape),
                       backbone_forwards=1, recurrent_update_interval="one per environment decision")
        return {"sf_logits":logits[0].tolist(),"sf_probabilities":probabilities[0].tolist(),"pixel_input":receipt}


def rollout_episode(case, predictor, policy_id, controller="greedy", epsilon=0.0, seed=17, env_factory=None):
    if controller not in ("greedy","sample") or not 0 <= epsilon <= 1:
        raise ValueError("Invalid action controller")
    if case["spec"].get("task") != "shooting" or case["spec"].get("scenario") != "basic":
        raise ValueError("APPO Basic cannot evaluate another task/scenario")
    if env_factory is None:
        from unified_doom_env import UnifiedDoomEnv
        env_factory = UnifiedDoomEnv
    env = env_factory(copy.deepcopy(case["spec"]))
    rng = random.Random(int(digest([case["id"],seed])[:16],16))
    episode = {"case":copy.deepcopy(case),"continuation_policy_id":policy_id,
               "steps":[],"complete":False,"success":None}
    try:
        obs,info = env.reset(case["seed"])
        predictor.reset_episode()
        while not info["terminated"]:
            if set(obs["candidates"]) != set(SF_ACTIONS):
                raise ValueError("Unified Basic must offer all four mapped actions")
            state = env._game.get_state()
            if state is None or state.screen_buffer is None:
                raise RuntimeError("A live environment has no pixel observation")
            result = predictor.predict_pixels(state.screen_buffer.copy())
            mapped = map_action_distribution(result["sf_logits"],result["sf_probabilities"])
            policy_probs = mapped["policy_probs"]
            behavior = behavior_distribution(policy_probs,controller,epsilon)
            action,draw = sample_with_receipt(behavior,rng)
            request = policy_request(obs,case["id"])
            next_obs,reward,terminated,truncated,next_info = env.step(action)
            if truncated:
                raise RuntimeError("External truncation cannot be recorded as failure")
            if not 1 <= next_info["actual_ticks"] <= next_info["requested_ticks"] <= case["spec"].get("frame_skip",4):
                raise RuntimeError("Invalid actual physical action duration")
            episode["steps"].append({"observation":copy.deepcopy(obs),"request":request,
                "request_role":"text reference for replay; APPO receives pixels only",
                "action":action,"scores":policy_probs,"policy_probs":policy_probs,
                "policy_probs_raw":mapped["policy_probs_raw"],"behavior_probs":behavior,
                "sampling_draw":draw,"sf_action_index":SF_ACTIONS.index(action),
                "sf_action_logits":mapped["logits"],"raw_probability_sum":mapped["raw_probability_sum"],
                "normalization_applied":mapped["normalization_applied"],"pixel_input":result["pixel_input"],
                "answers":{"action":{"type":"choice","probabilities":policy_probs,
                    "probability_semantics":"APPO categorical action policy; not event-success probability"}},
                "decision_source":"sample_factory_appo_pixel_policy","forced":False,
                "reward":reward,"terminated":terminated,"truncated":truncated,"info":copy.deepcopy(next_info)})
            obs,info = next_obs,next_info
        if info.get("truncated") or type(info["success"]) is not bool:
            raise ValueError("Invalid terminal outcome")
        episode.update(complete=True,success=info["success"],final_info=copy.deepcopy(info))
        return episode
    finally:
        env.close()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--cfg",type=Path,required=True)
    p.add_argument("--checkpoint-sha256",default=CHECKPOINT_SHA256)
    p.add_argument("--model-id",default=MODEL_ID)
    p.add_argument("--revision",default=REVISION)
    p.add_argument("--sf-source-commit",help="Declared commit of the imported SF source; actual module hashes are also recorded")
    p.add_argument("--cases",type=Path,default=Path("configs/unified_games_v1_cases.jsonl"))
    p.add_argument("--splits",default=",".join(SPLITS))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--controller",choices=("greedy","sample"),default="greedy")
    p.add_argument("--epsilon",type=float,default=0.0)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--torch-threads",type=int,default=1)
    p.add_argument("--validate-only",action="store_true")
    args=p.parse_args(argv)
    if not math.isfinite(args.epsilon) or not 0 <= args.epsilon <= 1 or args.torch_threads < 1:
        p.error("epsilon must be in [0,1]; torch-threads must be positive")
    return args


def main(argv=None):
    args=parse_args(argv)
    manifest_path=args.output.with_suffix(".manifest.json")
    snapshot=args.output.with_suffix(".sources")
    if args.output.exists() or manifest_path.exists() or snapshot.exists():
        raise ValueError("Use fresh output, manifest and source-snapshot paths")
    cfg=validate_config(json.loads(args.cfg.read_text()))
    checkpoint_sha=file_digest(args.checkpoint)
    if checkpoint_sha != args.checkpoint_sha256:
        raise ValueError("Checkpoint SHA256 does not match the explicitly selected public artifact")
    cases=select_cases(read_rows(args.cases),args.splits.split(","))
    cfg_sha=file_digest(args.cfg)
    if args.validate_only:
        print(encode({"validated":True,"selected_cases":len(cases),"case_ids":[c["id"] for c in cases],
                      "checkpoint_sha256":checkpoint_sha,"cfg_sha256":cfg_sha,
                      "action_index_mapping":dict(enumerate(SF_ACTIONS)),"model_forward_calls":0}))
        return
    predictor=SampleFactoryPolicy(args.checkpoint,args.cfg,args.device,args.torch_threads,args.sf_source_commit)
    sources={**source_hashes(),Path(__file__).name:file_digest(__file__)}
    declaration={"engine":"sample_factory_appo","controller":args.controller,"epsilon":args.epsilon,
        "sampling_seed":args.seed,"temperature":1.0,"tie_break":"lexicographic_first",
        "environment_contract":"finite_task_deadline_v1","model":args.model_id,"model_revision":args.revision,
        "checkpoint_file":args.checkpoint.name,"checkpoint_sha256":{"model":checkpoint_sha,"cfg.json":cfg_sha},
        "training_source_commit":TRAIN_SOURCE_COMMIT,"source_sha256":sources,
        "action_index_mapping":{str(i):a for i,a in enumerate(SF_ACTIONS)},
        "input_source":"pixels_only; recurrent history of images",
        "actor_updated_during_evaluation":False,**predictor.identity}
    policy_id=digest(declaration)
    manifest={"schema_version":"nanojev-unified-episodes-v1","policy":declaration,
        "continuation_policy_id":policy_id,"cases_sha256":file_digest(args.cases),
        "selected_cases":[c["id"] for c in cases],"source_snapshot":snapshot.name,"finished":False,
        "scope":"all Basic cases in requested splits; no filtering by outcomes",
        "comparison_note":"Same UnifiedDoomEnv actions and deadlines; APPO reads pixels, NanoJev/Jev read visible-object text.",
        "api_calls":0,"training_updates":0}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    snapshot.mkdir(exist_ok=False)
    for name in sources:
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    episodes=[];started=time.monotonic()
    try:
        with args.output.open("x") as handle:
            for case in cases:
                episode=rollout_episode(case,predictor,policy_id,args.controller,args.epsilon,args.seed)
                handle.write(encode(episode)+"\n");handle.flush();episodes.append(episode)
                print(encode({"complete":len(episodes),"total":len(cases),"id":case["id"],
                              "success":episode["success"],"decisions":len(episode["steps"])}),flush=True)
        if file_digest(args.cfg)!=cfg_sha or file_digest(args.checkpoint)!=checkpoint_sha:
            raise ValueError("Checkpoint/config changed during evaluation")
        if {**source_hashes(),Path(__file__).name:file_digest(__file__)}!=sources:
            raise ValueError("Evaluation source changed during collection")
        manifest.update(finished=True,episode_sha256=file_digest(args.output),summary=summarize(episodes),
                        elapsed_seconds=time.monotonic()-started,model_forward_calls=predictor.forward_count,
                        recurrent_resets=predictor.reset_count)
    except Exception as error:
        manifest.update(error={"type":type(error).__name__,"message":str(error)},
                        completed_episodes=len(episodes),elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    print(encode({"finished":True,"summary":manifest["summary"]}),flush=True)


if __name__=="__main__":
    main()
