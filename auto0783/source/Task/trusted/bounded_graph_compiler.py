"""Trusted compiler for validated, bounded data-only tensor graph regions.

No candidate source, paths, backend names or compiler flags are accepted.
Numerical prefix values belong to TensorDAG and are never captured here.
"""
import time
import torch

SCHEMA = 'bifrost-tensor-dag/2'
MAX_REGIONS = 16
MAX_VARIANTS = 4


def validate_regions(program):
    regions = program['compile_regions']
    if not isinstance(regions, list) or not 1 <= len(regions) <= MAX_REGIONS:
        raise ValueError('Compiled graph requires1..16 bounded regions')
    nodes = program['layer_body']
    positions = {node['id']: index for index, node in enumerate(nodes)}
    occupied, identities, planned = set(), set(), []
    for region in regions:
        if not isinstance(region, dict) or set(region) != {'id', 'nodes'}:
            raise ValueError('Unknown compiled region field')
        name = region['id']
        # Graph node IDs have already been validated by the enclosing parser.
        import re
        if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]{0,63}', name) or name in identities:
            raise ValueError('Invalid or duplicate compiled region identity')
        identities.add(name)
        members = region['nodes']
        if not isinstance(members, list) or not members or any(not isinstance(x, str) or x not in positions for x in members):
            raise ValueError('Compiled region contains unknown graph nodes')
        indices = [positions[x] for x in members]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError('Compiled region must be contiguous and ordered')
        if occupied.intersection(indices):
            raise ValueError('Compiled regions must not overlap')
        occupied.update(indices)
        body = nodes[indices[0]:indices[-1] + 1]
        lifetimes = {x.get('lifetime', 'stage') for x in body}
        if len(lifetimes) != 1:
            raise ValueError('Compiled region cannot cross prefix and stage lifetimes')
        member_set = set(members)
        external = tuple(dict.fromkeys(source for node in body for source in node['inputs'] if source not in member_set))
        referenced = {source for node in nodes[indices[-1] + 1:] for source in node['inputs']}
        referenced.add(program['layer_output'])
        # Every prefix node must remain individually counted and cached just as
        # on the eager path, including views and otherwise dead intermediates.
        prefix = next(iter(lifetimes)) == 'prefix'
        outputs = tuple(x for x in members if prefix or x in referenced)
        if not outputs:
            raise ValueError('Compiled region has no observable graph output')
        planned.append({'id': name, 'start': indices[0], 'stop': indices[-1] + 1,
                        'nodes': body, 'inputs': external, 'outputs': outputs,
                        'prefix': prefix})
    return sorted(planned, key=lambda x: x['start'])


def tensor_signature(value):
    return (tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device), value.requires_grad)


class BoundedGraphCompiler:
    def __init__(self, regions, apply_node):
        self.regions = regions
        self.apply_node = apply_node
        self.variants = {}
        self.observations = []

    def run(self, region, inputs, dimensions):
        if any(x.ndim > 5 or x.numel() > 100_000_000 for x in inputs):
            raise ValueError('Compiled input shape budget exceeded')
        key = (region['id'], tuple(tensor_signature(x) for x in inputs), tuple(sorted(dimensions.items())))
        cached = self.variants.get(key)
        if cached is None:
            if sum(k[0] == region['id'] for k in self.variants) >= MAX_VARIANTS:
                raise ValueError('Compiled region shape variant budget exceeded')
            # Closure captures only trusted operation implementations and
            # validated scalar graph metadata, never an input or weight tensor.
            external, outputs, body = region['inputs'], region['outputs'], region['nodes']
            apply_node = self.apply_node
            fixed_dimensions = dict(dimensions)

            def execute(*arguments):
                values = dict(zip(external, arguments))
                for node in body:
                    values[node['id']] = apply_node(node, [values[s] for s in node['inputs']], fixed_dimensions)
                return tuple(values[name] for name in outputs)

            compiled = torch.compile(execute, backend='inductor', fullgraph=True, dynamic=False,
                                     options={'triton.cudagraphs': False, 'max_autotune': False,
                                              'emulate_precision_casts': True})
            cached = {'function': compiled, 'first_complete': False}
            self.variants[key] = cached
        started = time.monotonic()
        # Fullgraph rejects graph breaks; suppress_errors is disabled rather
        # than allowing a process-wide setting to silently select eager code.
        with torch._dynamo.config.patch(suppress_errors=False, cache_size_limit=64,
                                       accumulated_cache_size_limit=256):
            result = cached['function'](*inputs)
        if not isinstance(result, tuple) or len(result) != len(region['outputs']):
            raise ValueError('Compiled region output structure differs')
        if any(x.ndim > 5 or x.numel() > 100_000_000 for x in result):
            raise ValueError('Compiled output shape budget exceeded')
        if not cached['first_complete']:
            if inputs and inputs[0].device.type == 'cuda':
                torch.cuda.synchronize()
            self.observations.append({'region': region['id'],
                                      'first_invocation_wall_seconds': time.monotonic() - started,
                                      'includes_compilation_and_first_execution': True,
                                      'compiler_backend': 'inductor', 'fullgraph': True,
                                      'CUDA_graphs': False, 'tensor_results_retained_by_compiler_wrapper': False})
            cached['first_complete'] = True
        return result

    def facts(self):
        return {'schema': SCHEMA, 'region_count': len(self.regions),
                'compiled_variant_count': len(self.variants),
                'first_invocations': list(self.observations),
                'first_invocation_wall_is_not_net_research_credit': True}
