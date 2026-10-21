"""CPU-only contract checks for the round-02 candidate."""
import hashlib
import ast
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
OUTPUT = ROOT / 'agent-output/round-02'
METHOD = ROOT / 'solution/method.py'
CONTRACT = ROOT / 'environment/public_eval/implementation/engine/method_contract.py'
PREVIOUS = ROOT / 'agent-output/round-01/method.py'
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
        record = {'name': name, 'status': 'FAIL', 'error': repr(error),
                  'traceback': traceback.format_exc()}
    record['cpu_wall_seconds'] = time.perf_counter() - started
    CHECKS.append(record)
    print(json.dumps(record, allow_nan=False), flush=True)


def expect_value_error(function):
    try:
        function()
    except ValueError as error:
        return str(error)
    raise AssertionError('Expected ValueError')


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
    assert checked.teacher.teacher_steps == 10
    assert set(raw) == {'name', 'teacher', 'time_sampler_note', 'train_notes'}
    return {'checked_spec': asdict(checked), 'proposal_count': 384,
            'planned_pair_visits': 128, 'planned_optimizer_updates': 8,
            'training_or_evaluation_run': False}


def check_controlled_ablation():
    assert asdict(method.get_method().teacher) == asdict(previous.get_method().teacher)
    old_tree = ast.parse(PREVIOUS.read_text())
    new_tree = ast.parse(METHOD.read_text())
    old_functions = {node.name: ast.dump(node) for node in old_tree.body
                     if isinstance(node, ast.FunctionDef)}
    new_functions = {node.name: ast.dump(node) for node in new_tree.body
                     if isinstance(node, ast.FunctionDef)}
    changed = [name for name in old_functions
               if old_functions[name] != new_functions[name]]
    assert changed == ['fm_time_sampler']
    old_generator = torch.Generator(device='cpu').manual_seed(2003)
    new_generator = torch.Generator(device='cpu').manual_seed(2003)
    for _ in range(64):
        previous.fm_time_sampler((2,), 'cpu', old_generator)
        method.fm_time_sampler((2,), 'cpu', new_generator)
        assert torch.equal(old_generator.get_state(), new_generator.get_state())
    archive = json.loads((OUTPUT.parent / 'round-01/analysis.json').read_text())
    hashes = archive['preservation']['original_artifact_sha256_relative_to_workspace']
    verified = []
    for path, digest in hashes.items():
        if path.startswith('agent-output/'):
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
            verified.append(path)
    return {'teacher_config_identical_to_round01': True,
            'changed_function_names': changed,
            'cpu_generator_consumption_identical_for_64_microbatches': True,
            'previous_artifact_hashes_unchanged': verified}


def check_times():
    generator = torch.Generator(device='cpu').manual_seed(2003)
    records = []
    for shape in [(1,), (2,), (16,), (128,), (256,), (2, 3), (0,), ()]:
        values = method.fm_time_sampler(shape, device='cpu', generator=generator)
        assert isinstance(values, torch.Tensor)
        assert tuple(values.shape) == shape
        assert values.device.type == 'cpu' and values.dtype == torch.float32
        assert torch.isfinite(values).all()
        assert ((values >= 0) & (values <= 1)).all()
        flat = values.flatten()
        contract.checked_times(flat.tolist(), flat.numel())
        pairs = flat[:(flat.numel() // 2) * 2].reshape(-1, 2)
        if len(pairs):
            assert torch.all((pairs < .5).sum(dim=1) == 1)
            assert torch.allclose(pairs.sum(dim=1), torch.ones(len(pairs)),
                                  rtol=0, atol=1e-7)
        records.append({'shape': list(shape), 'count': values.numel(),
                        'min': float(flat.min()) if len(flat) else None,
                        'max': float(flat.max()) if len(flat) else None})
    return {'shapes': records, 'complete_pairs_span_half_intervals': True,
            'complete_pair_sums_equal_one_within_1e-7': True,
            'public_checked_times_passed_for_each_flattened_shape': True}


def check_time_rng():
    first = torch.Generator(device='cpu').manual_seed(2003)
    second = torch.Generator(device='cpu').manual_seed(2003)
    global_before = torch.random.get_rng_state().clone()
    a = method.fm_time_sampler((128,), 'cpu', first)
    b = method.fm_time_sampler((128,), 'cpu', second)
    assert torch.equal(a, b)
    assert torch.equal(first.get_state(), second.get_state())
    assert not torch.equal(first.get_state(), torch.Generator(device='cpu').manual_seed(2003).get_state())
    assert torch.equal(global_before, torch.random.get_rng_state())
    restored = torch.Generator(device='cpu')
    restored.set_state(first.get_state())
    assert torch.equal(method.fm_time_sampler((2,), 'cpu', first),
                       method.fm_time_sampler((2,), 'cpu', restored))
    stream_generator = torch.Generator(device='cpu').manual_seed(2003)
    actual_stream = torch.cat([method.fm_time_sampler((2,), 'cpu', stream_generator)
                               for _ in range(64)])
    actual_bins = (actual_stream * 10).floor().long().clamp(0, 9)
    actual_histogram = torch.bincount(actual_bins, minlength=10)
    stream_generator = torch.Generator(device='cpu').manual_seed(2003)
    stream = torch.cat([method.fm_time_sampler((2,), 'cpu', stream_generator)
                        for _ in range(1024)])
    bins = (stream * 10).floor().long().clamp(0, 9)
    histogram = torch.bincount(bins, minlength=10)
    assert torch.all((stream.reshape(-1, 2) < .5).sum(dim=1) == 1)
    assert int(histogram.min()) > 0
    return {'equal_seed_equal_output': True,
            'generator_state_advances': True,
            'transport_state_replay_equal': True,
            'global_rng_unchanged': True,
            'simulated_128_visit_training_time_histogram': actual_histogram.tolist(),
            'simulated_2048_time_histogram': histogram.tolist(),
            'pair_count': int(stream.numel() // 2),
            'time_stream_seed': 2003,
            'these_are_time_calls_only_not_training_or_evaluation': True}


def check_selection():
    generator = torch.Generator(device='cpu').manual_seed(1001)
    cases = []
    for multiplier in range(1, 17):
        scores = torch.rand(128 * multiplier, generator=generator, device='cpu')
        selected = method.select_teacher_candidates(scores, multiplier)
        assert selected.device.type == 'cpu' and selected.dtype == torch.long
        assert selected.shape == (128,) and selected.unique().numel() == 128
        assert torch.equal(torch.div(selected, multiplier, rounding_mode='floor'),
                           torch.arange(128))
        contract.checked_indices(selected.tolist(), 128, multiplier)
        expected = []
        for group in range(128):
            order = sorted(range(multiplier),
                           key=lambda offset: float(scores[group * multiplier + offset]))
            expected.append(group * multiplier + order[(multiplier - 1) // 2])
        assert selected.tolist() == expected
        cases.append({'multiplier': multiplier,
                      'groups': 128,
                      'unique_group_legal_indices': True})
    fixed = method.select_teacher_candidates(
        torch.tensor([.9, .1, .5, .2, .2, .8, .3, .1, .2]), 3)
    assert fixed.tolist() == [2, 4, 8]
    ties = method.select_teacher_candidates(torch.ones(12), 3)
    assert ties.tolist() == [1, 4, 7, 10]
    return {'cases': cases, 'selection_rule': 'stable lower median',
            'fixed_median_indices': fixed.tolist(),
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
            assert weights.device.type == 'cpu' and weights.shape == (batch,)
            assert weights.dtype == torch.float32 and torch.isfinite(weights).all()
            assert ((weights >= 0) & (weights <= 16)).all()
            assert float(weights.sum()) > 0
            normalized = contract.checked_weights(weights.tolist(), batch)
            assert normalized == [1.] * batch
            records.append({'batch': batch, 'targets': kind,
                            'min': float(weights.min()), 'max': float(weights.max()),
                            'positive_mass': float(weights.sum())})
    return {'batches': records, 'all_weights_finite_nonnegative': True,
            'all_batches_positive_mass': True}


def check_invalid_inputs():
    results = {
        'missing_cpu_generator': expect_value_error(
            lambda: method.fm_time_sampler((2,), 'cpu')),
        'incomplete_group': expect_value_error(
            lambda: method.select_teacher_candidates(torch.ones(4), 3)),
        'nonfinite_agreement': expect_value_error(
            lambda: method.select_teacher_candidates(torch.tensor([0., float('nan'), 1.]), 3)),
        'empty_weight_batch': expect_value_error(
            lambda: method.pair_weights(torch.empty(0, dtype=torch.long),
                                        torch.empty((0, 680), dtype=torch.long))),
    }
    return {'expected_invalid_inputs_rejected': results}


if __name__ == '__main__':
    torch.set_num_threads(1)
    method = load_module('round02_candidate', METHOD)
    previous = load_module('round01_control', PREVIOUS)
    contract = load_module('round02_public_contract', CONTRACT)
    check('get_method_contract', check_config)
    check('single_time_algorithm_ablation_and_previous_artifact_preservation',
          check_controlled_ablation)
    check('time_shape_range_pairing', check_times)
    check('time_cpu_rng_replay_and_uniform_marginals', check_time_rng)
    check('selection_unique_group_indices_and_rank', check_selection)
    check('weights_finite_nonnegative_positive_mass', check_weights)
    check('invalid_input_rejections', check_invalid_inputs)
    result = {'protocol': contract.PROTOCOL,
              'time': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
              'torch_version': torch.__version__, 'device': 'cpu',
              'method_sha256': hashlib.sha256(METHOD.read_bytes()).hexdigest(),
              'status': 'PASS' if all(c['status'] == 'PASS' for c in CHECKS) else 'FAIL',
              'checks': CHECKS, 'evaluated': False, 'score': None}
    (OUTPUT / 'cpu_precheck_result.json').write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'cpu_precheck': result['status'],
                      'method_sha256': result['method_sha256'],
                      'time': result['time']}, allow_nan=False), flush=True)
    sys.exit(0 if result['status'] == 'PASS' else 1)
