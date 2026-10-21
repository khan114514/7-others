"""Only scalar configuration and checked tensors cross the method boundary."""
from dataclasses import dataclass, field, asdict
import math

PROTOCOL = 'dd-selfproto-v1.3'


@dataclass
class TeacherConfig:
    num_samples: int = 0
    teacher_steps: int = 10
    guidance: float = 2.0
    filter_rule: str = 'none'
    weighting: str = 'uniform'
    reuse: int = 1
    top_k: int = 900
    top_p: float = .95
    ode_nfe: int = 32
    ode_grid: str = 'logSNR'
    proposal_multiplier: int = 1


@dataclass
class MethodSpec:
    name: str = 'published_dd'
    protocol_version: str = PROTOCOL
    initialization: str = 'warmstart_official_dd_ema'
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    time_sampler_note: str = ''
    train_notes: str = ''


def checked_spec(value):
    if not isinstance(value, dict) or set(value) - {'name', 'teacher', 'time_sampler_note', 'train_notes'}:
        raise ValueError('Method configuration contains unsupported fields')
    config = value.get('teacher')
    if not isinstance(config, dict) or set(config) - set(asdict(TeacherConfig())):
        raise ValueError('Unsupported teacher configuration')
    teacher = TeacherConfig(**config)
    for name, low, high in [('num_samples', 0, 100000), ('teacher_steps', 10, 10),
                           ('reuse', 1, 100000), ('top_k', 0, 4096),
                           ('ode_nfe', 1, 128), ('proposal_multiplier', 1, 16)]:
        val = getattr(teacher, name)
        if type(val) is not int or not low <= val <= high:
            raise ValueError('Invalid integer teacher field: ' + name)
    for name, low, high in [('guidance', 0, 20), ('top_p', 0, 1)]:
        val = getattr(teacher, name)
        if type(val) not in (int, float) or not math.isfinite(val) or not low <= val <= high:
            raise ValueError('Invalid teacher field: ' + name)
    if teacher.filter_rule not in ('none', 'student_agreement_per_label'):
        raise ValueError('Unsupported teacher filter')
    if teacher.weighting != 'uniform' or teacher.ode_grid not in ('logSNR', 'time_uniform'):
        raise ValueError('Unsupported teacher solver or weighting')
    if teacher.filter_rule == 'none' and teacher.proposal_multiplier != 1:
        raise ValueError('Unfiltered targets require one proposal per label')
    if teacher.num_samples * teacher.proposal_multiplier > 131072:
        raise ValueError('Proposal buffer exceeds the declared host-memory capacity')
    if teacher.num_samples % 16:
        raise ValueError('Teacher sample count must fill complete fixed batches of 16')
    output = MethodSpec(teacher=teacher)
    for name in ('name', 'time_sampler_note', 'train_notes'):
        val = value.get(name, getattr(output, name))
        if not isinstance(val, str) or len(val) > 4096:
            raise ValueError('Invalid descriptive field')
        setattr(output, name, val)
    return output


def checked_times(values, count):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError('FM sampler returned the wrong shape')
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError('FM times must be finite and in [0,1]')
    return values


def checked_indices(values, groups, multiplier):
    if not isinstance(values, list) or len(values) != groups:
        raise ValueError('Teacher selector returned the wrong shape')
    for group, index in enumerate(values):
        if type(index) is not int or not group * multiplier <= index < (group + 1) * multiplier:
            raise ValueError('Teacher selector must preserve each original label group')
    return values
