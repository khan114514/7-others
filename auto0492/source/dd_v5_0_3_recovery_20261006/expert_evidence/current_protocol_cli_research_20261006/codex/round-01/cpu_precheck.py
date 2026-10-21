"""Offline CPU checks against the disclosed dd-selfproto-v1.3 contract."""
import hashlib
import importlib.util
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import traceback

import torch


ROOT = Path('/workspace')
OUTPUT = ROOT / 'agent-output/round-01'
METHOD = ROOT / 'solution/method.py'
CONTRACT = ROOT / 'environment/public_eval/implementation/engine/method_contract.py'
CHECKS = []


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check(name, function):
    started = time.perf_counter()
    try:
        detail = function()
        record = {'name': name, 'status': 'PASS', 'detail': detail}
    except Exception as error:
        record = {'name': name, 'status': 'FAIL',
                  'error': repr(error), 'traceback': traceback.format_exc()}
    record['cpu_wall_seconds'] = time.perf_counter() - started
    CHECKS.append(record)
    print(json.dumps(record, allow_nan=False), flush=True)


def expect_value_error(function):
    try:
        function()
    except ValueError as error:
        return str(error)
    raise AssertionError('Expected invalid input to raise ValueError')


def check_config():
    raw = asdict(method.get_method())
    checked = contract.checked_spec(raw)
    assert checked.protocol_version == 'dd-selfproto-v1.3'
    assert checked.initialization == 'warmstart_official_dd_ema'
    assert checked.teacher.num_samples == 128
    assert checked.teacher.reuse == 1
    assert checked.teacher.proposal_multiplier == 3
    assert checked.teacher.filter_rule == 'student_agreement_per_label'
    assert checked.teacher.weighting == 'uniform'
    assert set(raw) == {'name', 'teacher', 'time_sampler_note', 'train_notes'}
    return {'checked_spec': asdict(checked), 'proposal_count': 384,
            'planned_pair_visits': 128, 'planned_optimizer_updates': 8,
            'updates_are_not_completed_training': True}


def check_times():
    generator = torch.Generator(device='cpu').manual_seed(2003)
    shapes = [(1,), (2,), (16,), (128,), (256,), (2, 3), (0,), ()]
    records = []
    for shape in shapes:
        values = method.fm_time_sampler(shape, device='cpu', generator=generator)
        assert isinstance(values, torch.Tensor)
        assert tuple(values.shape) == shape
        assert values.device.type == 'cpu' and values.dtype == torch.float32
        assert torch.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        flat = values.flatten()
        if len(shape) == 1 and values.numel() > 0:
            contract.checked_times(values.tolist(), shape[0])
        pairs = flat[:flat.numel() // 2 * 2].reshape(-1, 2)
        if len(pairs):
            assert torch.all((pairs < .1).sum(dim=1) == 1)
        records.append({'shape': list(shape), 'count': values.numel(),
                        'min': float(flat.min()) if len(flat) else None,
                        'max': float(flat.max()) if len(flat) else None})
    return {'shapes': records, 'all_complete_pairs_have_one_k0': True}


def check_time_rng():
    first = torch.Generator(device='cpu').manual_seed(2003)
    second = torch.Generator(device='cpu').manual_seed(2003)
    global_before = torch.random.get_rng_state().clone()
    old_state = first.get_state().clone()
    a = method.fm_time_sampler((128,), 'cpu', first)
    b = method.fm_time_sampler((128,), 'cpu', second)
    assert torch.equal(a, b) and torch.equal(first.get_state(), second.get_state())
    assert not torch.equal(old_state, first.get_state())
    assert torch.equal(global_before, torch.random.get_rng_state())
    restored = torch.Generator(device='cpu')
    restored.set_state(first.get_state())
    c = method.fm_time_sampler((2,), 'cpu', first)
    d = method.fm_time_sampler((2,), 'cpu', restored)
    assert torch.equal(c, d)
    train_generator = torch.Generator(device='cpu').manual_seed(2003)
    actual_stream = torch.cat([method.fm_time_sampler((2,), 'cpu', train_generator)
                              for _ in range(64)])
    bins = (actual_stream * 10).floor().long().clamp(0, 9)
    histogram = torch.bincount(bins, minlength=10)
    assert histogram[0] == 64
    assert torch.all((bins.reshape(-1, 2) == 0).sum(dim=1) == 1)
    context = actual_stream.reshape(-1, 2)[:, 0] >= .1
    return {'equal_seed_equal_output': True, 'generator_state_advances': True,
            'transport_state_replay_equal': True, 'global_rng_unchanged': True,
            'simulated_128_pair_time_histogram': histogram.tolist(),
            'first_sample_context_count_in_64_microbatches': int(context.sum())}


def check_selection():
    generator = torch.Generator(device='cpu').manual_seed(1001)
    results = []
    for multiplier in range(1, 17):
        scores = torch.rand(128 * multiplier, generator=generator, device='cpu')
        selected = method.select_teacher_candidates(scores, multiplier)
        assert selected.device.type == 'cpu' and selected.dtype == torch.long
        assert selected.shape == (128,) and selected.unique().numel() == 128
        assert torch.equal(torch.div(selected, multiplier, rounding_mode='floor'),
                           torch.arange(128))
        contract.checked_indices(selected.tolist(), 128, multiplier)
        # Python's stable ordering supplies an independent rank oracle.
        expected = []
        for group in range(128):
            order = sorted(range(multiplier),
                           key=lambda offset: float(scores[group * multiplier + offset]))
            expected.append(group * multiplier + order[(multiplier - 1) // 2])
        assert selected.tolist() == expected
        results.append({'multiplier': multiplier, 'groups': 128,
                        'unique_group_legal_indices': True})
    scores = torch.tensor([.9, .1, .5, .2, .2, .8, .3, .1, .2])
    fixed = method.select_teacher_candidates(scores, 3)
    assert fixed.tolist() == [2, 4, 8]
    ties = method.select_teacher_candidates(torch.ones(12), 3)
    assert ties.tolist() == [1, 4, 7, 10]
    return {'random_cases': results, 'fixed_median_indices': fixed.tolist(),
            'stable_tie_indices': ties.tolist()}


def check_weights():
    generator = torch.Generator(device='cpu').manual_seed(2001)
    records = []
    for batch in (1, 2, 16, 128, 256):
        labels = torch.randint(0, 1000, (batch,), generator=generator, dtype=torch.long)
        for kind in ('all_zero_tokens', 'all_max_tokens', 'random_tokens'):
            if kind == 'random_tokens':
                tokens = torch.randint(0, 4096, (batch, 680), generator=generator,
                                       dtype=torch.long)
            else:
                tokens = torch.full((batch, 680), 0 if kind == 'all_zero_tokens' else 4095,
                                    dtype=torch.long)
            weights = method.pair_weights(labels, tokens)
            assert weights.shape == (batch,) and weights.device.type == 'cpu'
            assert weights.dtype == torch.float32 and torch.isfinite(weights).all()
            assert ((weights >= 0) & (weights <= 16)).all() and weights.sum() > 0
            normalized = contract.checked_weights(weights.tolist(), batch)
            assert normalized == [1.] * batch
            records.append({'batch': batch, 'targets': kind, 'min': float(weights.min()),
                            'max': float(weights.max()), 'positive_mass': float(weights.sum())})
    return {'batches': records, 'normalized_mean': 1.,
            'weight_hook_enabled_in_training': False}


def check_invalid_inputs():
    results = {}
    results['missing_cpu_generator'] = expect_value_error(
        lambda: method.fm_time_sampler((2,), 'cpu'))
    results['incomplete_group'] = expect_value_error(
        lambda: method.select_teacher_candidates(torch.ones(4), 3))
    for value in (float('nan'), float('inf')):
        results['nonfinite_agreement_' + str(value)] = expect_value_error(
            lambda: method.select_teacher_candidates(torch.tensor([0., value, 1.]), 3))
    results['zero_multiplier'] = expect_value_error(
        lambda: method.select_teacher_candidates(torch.ones(3), 0))
    results['empty_weight_batch'] = expect_value_error(
        lambda: method.pair_weights(torch.empty(0, dtype=torch.long),
                                   torch.empty((0, 680), dtype=torch.long)))
    results['mismatched_weight_batch'] = expect_value_error(
        lambda: method.pair_weights(torch.zeros(2, dtype=torch.long),
                                   torch.zeros((1, 680), dtype=torch.long)))
    return {'expected_invalid_inputs_rejected': results}


if __name__ == '__main__':
    torch.set_num_threads(1)
    method = load_module('round01_candidate', METHOD)
    contract = load_module('round01_public_contract', CONTRACT)
    check('get_method_contract', check_config)
    check('time_shape_range_pairing', check_times)
    check('time_cpu_rng_replay_and_training_stream', check_time_rng)
    check('selection_unique_group_indices_and_rank', check_selection)
    check('weights_finite_nonnegative_positive_mass', check_weights)
    check('invalid_input_rejections', check_invalid_inputs)
    result = {'protocol': contract.PROTOCOL,
              'time': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
              'torch_version': torch.__version__, 'device': 'cpu',
              'method_sha256': hashlib.sha256(METHOD.read_bytes()).hexdigest(),
              'status': 'PASS' if all(c['status'] == 'PASS' for c in CHECKS) else 'FAIL',
              'checks': CHECKS, 'evaluated': False, 'score': None}
    (OUTPUT / 'cpu_precheck_result.json').write_text(json.dumps(result, indent=2,
                                                               allow_nan=False) + '\n')
    print(json.dumps({'cpu_precheck': result['status'],
                      'method_sha256': result['method_sha256'],
                      'time': result['time']}, allow_nan=False), flush=True)
    sys.exit(0 if result['status'] == 'PASS' else 1)
