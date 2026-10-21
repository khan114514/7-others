"""Untrusted candidate code lives only in this restricted subprocess."""
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worker_guard import apply_guard


def main():
    candidate, temporary = sys.argv[1:3]
    guard = apply_guard(candidate, temporary)
    # stdout is reserved for the typed transport. Candidate diagnostics go to
    # a separately size-limited log, never to the verifier's reward channel.
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        torch.set_num_threads(1)
        spec = importlib.util.spec_from_file_location('candidate_method', candidate)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        method = module.get_method()
    for line in sys.stdin:
        request = json.loads(line)
        operation = request['op']
        with contextlib.redirect_stdout(sys.stderr):
            if operation == 'config':
                # Explicitly select allowed scalar fields, rather than
                # serializing arbitrary Python objects across the boundary.
                fields = ('num_samples', 'teacher_steps', 'guidance', 'filter_rule',
                          'weighting', 'reuse', 'top_k', 'top_p', 'ode_nfe',
                          'ode_grid', 'proposal_multiplier')
                teacher = {name: getattr(method.teacher, name) for name in fields
                           if hasattr(method.teacher, name)}
                result = {'name': str(method.name), 'teacher': teacher,
                          'time_sampler_note': str(getattr(method, 'time_sampler_note', '')),
                          'train_notes': str(getattr(method, 'train_notes', ''))}
                response = {'result': result, 'guard': guard}
            elif operation == 'time':
                generator = torch.Generator(device='cpu')
                generator.set_state(torch.tensor(request['rng'], dtype=torch.uint8))
                values = module.fm_time_sampler(tuple(request['shape']),
                                                device='cpu', generator=generator)
                if not isinstance(values, torch.Tensor) or list(values.shape) != request['shape']:
                    raise ValueError('Sampler must return a tensor of the requested shape')
                response = {'result': values.detach().cpu().tolist(),
                            'rng': generator.get_state().tolist()}
            elif operation == 'select':
                agreement = torch.tensor(request['agreement'], dtype=torch.float32)
                indices = module.select_teacher_candidates(agreement, request['multiplier'])
                if not isinstance(indices, torch.Tensor) or indices.dtype != torch.long:
                    raise ValueError('Selector must return a LongTensor')
                response = {'result': indices.detach().cpu().tolist()}
            else:
                raise ValueError('Unsupported method operation')
        sys.stdout.write(json.dumps(response, allow_nan=False, separators=(',', ':')) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
