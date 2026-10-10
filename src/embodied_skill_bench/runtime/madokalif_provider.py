"""Pinned native OpenPI Pi0.5 inference over the existing RPent VLA wire."""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import time

import numpy as np

from ..io import sha256_file

PROFILE = 'openpi-full-sft-60k'
REVISION = 'e8ba73940836f73c50b3fe58653e5bf53799458b'
WEIGHT_SHA256 = 'b72bdb98cf9549189b47c8572407277dcccc20da0d0ffe2c356ebb5b0f5877b1'
NORM_SHA256 = 'c0ee3c1a6ca5f1414d31180af3bc333dca1d78cf912f341943191b393fe68707'


def validate_actions(actions):
    array = np.asarray(actions, dtype=np.float32)
    if array.shape != (10, 7):
        raise ValueError(f'Madokalif native action horizon must be [10,7], got {array.shape}')
    if not np.isfinite(array).all():
        raise ValueError('Madokalif actions contain NaN or infinity')
    return array


def wire_observation(instruction, images, state):
    from .vla_policy_server import _decode_png
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('Madokalif requires a nonempty instruction')
    if not isinstance(images, dict) or not {'main', 'wrist'} <= images.keys():
        raise ValueError('Madokalif requires main and wrist images')
    vector = np.asarray(state, dtype=np.float32)
    if vector.shape != (1, 8) or not np.isfinite(vector).all():
        raise ValueError('Madokalif requires finite state [1,8]')
    # RPent already supplies both views in the LIBERO dataset orientation and
    # state = EEF xyz, axis-angle, two signed finger qpos. Do not rotate twice.
    return {'observation/image': _decode_png(images['main'], name='main'),
            'observation/wrist_image': _decode_png(images['wrist'], name='wrist'),
            'observation/state': vector[0], 'prompt': instruction.strip()}


class MadokalifLiberoProvider:
    profile_id = PROFILE
    revision = REVISION

    def __init__(self, checkpoint_path, *, seed=0):
        # JAX is used by OpenPI tree/transforms only; inference is PyTorch CUDA.
        os.environ.setdefault('JAX_PLATFORMS', 'cpu')
        import torch
        from openpi.policies import policy_config
        from openpi.shared import normalize
        from openpi.training import config

        self.checkpoint_path = Path(checkpoint_path).resolve()
        weight = self.checkpoint_path / 'model.safetensors'
        if sha256_file(weight) != WEIGHT_SHA256:
            raise ValueError('Madokalif checkpoint does not match the pinned revision')
        stats_path = self.checkpoint_path / 'assets/physical-intelligence/libero'
        self.norm_stats_sha256 = sha256_file(stats_path / 'norm_stats.json')
        if self.norm_stats_sha256 != NORM_SHA256:
            raise ValueError('Madokalif normalization statistics do not match the pinned checkpoint')
        stats = normalize.load(stats_path)
        if len(stats['state'].mean) != 8 or len(stats['actions'].mean) != 7:
            raise ValueError('Unexpected Madokalif normalization dimensions')
        train_config = config.get_config('pi05_libero')
        self.native_config = {key: getattr(train_config.model, key) for key in
                              ['pi05', 'action_dim', 'action_horizon', 'max_token_len',
                               'discrete_state_input', 'paligemma_variant', 'action_expert_variant']}
        if self.native_config != {'pi05': True, 'action_dim': 32, 'action_horizon': 10,
                                  'max_token_len': 200, 'discrete_state_input': False,
                                  'paligemma_variant': 'gemma_2b', 'action_expert_variant': 'gemma_300m'}:
            raise ValueError('Native pi05_libero configuration drift')
        if train_config.data.extra_delta_transform:
            raise ValueError('Madokalif LIBERO must not apply an extra delta transform')
        # Pin transforms to this checkpoint's assets as well as passing the
        # explicit statistics; no training-dataset assets are fetched.
        train_config = dataclasses.replace(train_config, data=dataclasses.replace(
            train_config.data, assets=config.AssetsConfig(assets_dir=str(self.checkpoint_path / 'assets'))))
        self.seed = int(seed)
        limit = os.environ.get('ESB_MADOKALIF_MEMORY_LIMIT_MIB')
        self.allocator_limit_mib = int(limit) if limit is not None else None
        if self.allocator_limit_mib is not None:
            total = torch.cuda.get_device_properties('cuda').total_memory
            fraction = self.allocator_limit_mib * 1024**2 / total
            if not 0 < fraction <= 1:
                raise ValueError('Invalid Madokalif GPU memory limit')
            torch.cuda.set_per_process_memory_fraction(fraction, device=torch.cuda.current_device())
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.set_num_threads(4)
        self.policy = policy_config.create_trained_policy(
            train_config, self.checkpoint_path, norm_stats=stats, pytorch_device='cuda',
            sample_kwargs={'num_steps': 10})
        # Native constructor wraps this function in torch.compile. Use its
        # original implementation to avoid costly cold-start compilation;
        # weights, transforms, Euler steps and sampled actions are unchanged.
        sampler = self.policy._model.sample_actions
        self.policy._model.sample_actions = getattr(sampler, '_torchdynamo_orig_callable', sampler)
        self.policy._sample_actions = self.policy._model.sample_actions
        self.config = train_config.model
        self.forward_calls = 0
        self.last_forward = None
        self.model_parameter_count = sum(p.numel() for p in self.policy._model.parameters())

    def predict(self, instruction, images, state, mode='eval'):
        import torch
        if mode != 'eval':
            raise ValueError('Madokalif supports evaluation only')
        observation = wire_observation(instruction, images, state)
        started = time.monotonic()
        with torch.inference_mode():
            result = self.policy.infer(observation)
        actions = validate_actions(result['actions'])
        self.forward_calls += 1
        self.last_forward = {'seconds': time.monotonic()-started, 'shape': list(actions.shape),
                             'min': float(actions.min()), 'max': float(actions.max()),
                             'cuda_max_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
                             'cuda_max_memory_reserved_bytes': torch.cuda.max_memory_reserved()}
        return {'actions': actions[None].tolist(), 'shape': [1, 10, 7], 'dtype': 'float32',
                'policy_profile': PROFILE, 'checkpoint_revision': REVISION}

    def healthz(self):
        import torch
        return {'status': 'ok', 'policy_profile': PROFILE, 'checkpoint_revision': REVISION,
                'checkpoint_path': str(self.checkpoint_path), 'weight_sha256': WEIGHT_SHA256,
                'norm_stats_sha256': self.norm_stats_sha256, 'native_action_shape': [10, 7],
                'state_shape': [8], 'camera_keys': ['main', 'wrist'], 'seed': self.seed,
                'model_parameter_count': self.model_parameter_count,
                'pytorch_allocator_limit_mib': self.allocator_limit_mib,
                'native_model_config': self.native_config,
                'forward_calls': self.forward_calls, 'last_forward': self.last_forward,
                'cuda_memory_allocated_bytes': torch.cuda.memory_allocated(),
                'cuda_memory_reserved_bytes': torch.cuda.memory_reserved(),
                'loader': 'openpi.create_trained_policy; safetensors.load_model(strict=True)',
                'sampler': 'native Euler flow matching, 10 steps, eager',
                'action_conversion': 'native OpenPI LIBERO unnormalization and first 7 dimensions; no extra gripper flip',
                'd0_status': 'strict-load-and-finite-forward-passed' if self.forward_calls else 'strict-load-passed-forward-pending'}
