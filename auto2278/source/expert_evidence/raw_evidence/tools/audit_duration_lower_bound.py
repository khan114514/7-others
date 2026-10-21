"""Evidence floor for QA, separate from the user's relaxed primary estimate.

Withholds uncertain work instead of declaring it a stall. Does not alter raw
logs or the primary accounting policy. Never sums overlapping wall windows.
"""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.recalculate_active_time import calculate, result_from_log, total, invalid_model_generation, relaxed_retry_exclusion


def session_floor(session, events, requests, grade):
    wall = max(0., session['end_unix'] - session['start_unix'])
    observed = min(wall, max(0., session.get('observed_seconds', 0.)))
    model = min(observed, max(0., session.get('model_seconds', observed)))
    grading = max(0., observed - model)
    if grade.get('status') in (None, 'timeout', 'infrastructure_error', 'model_error'):
        grading = 0.
    errors = []
    for line in events.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('type') == 'error' and any(t in event.get('message', '').lower() for t in ('network', 'connection failed', 'stream disconnected')):
            errors.append(event)
    missing = 'CLI runtime failure:' in events and 'thread.started' not in events
    rejected = 'Error loading config.toml' in events
    active_requests = [q for q in requests if session['start_unix'] <= q['started_unix'] < session['end_unix']]
    ambiguous = [q for q in active_requests if q.get('status') != 'completed' and not invalid_model_generation(q)]
    reason = 'complete original native events and monotonic session record'
    withheld = 0.
    if missing or rejected or (errors and not active_requests):
        withheld = model
        reason = 'model phase withheld: missing partial native log, rejected config, or network errors without request timestamps'
    elif active_requests:
        intervals = [(max(session['start_unix'], q['started_unix']), min(session['end_unix'], q['ended_unix'])) for q in ambiguous]
        for index, request in enumerate(requests):
            if session['start_unix'] <= request['started_unix'] < session['end_unix']:
                pause = relaxed_retry_exclusion(requests, index, session['start_unix'], session['end_unix'])
                if pause:
                    intervals.append(pause)
        # Failed-request wall duration upper-bounds its monotonic occupied time.
        # Suspension inside these intervals can make this floor deliberately low.
        withheld = min(model, total(intervals))
        reason = 'request-audited model phase; all ambiguous failed request windows and observed long retry pauses withheld'
    verified_model = max(0., model - withheld)
    return {'observed_monotonic_seconds': observed, 'model_phase_seconds': model,
            'verified_model_phase_lower_bound_seconds': verified_model,
            'verified_public_grading_seconds': grading,
            'withheld_model_phase_upper_bound_seconds': withheld,
            'lower_bound_seconds': verified_model + grading,
            'network_error_count': len(errors), 'missing_partial_log': missing,
            'ambiguous_failed_requests': len(ambiguous), 'basis': reason}


def calculate_floor(cutoff=None):
    primary = calculate(cutoff or time.time())
    tracks = {key: {'sessions': [], 'initial_scoring_withheld': [], 'wall_intervals': []} for key in ('gpt', 'seed')}
    for job in sorted((ROOT / 'campaigns').glob('asd-research-*')):
        for trial in job.glob('harbor_task__*'):
            config = json.loads((trial / 'config.json').read_text())
            track = 'seed' if 'seed' in config['agent']['model_name'] else 'gpt'
            a = trial / 'agent'
            initial = result_from_log(a / 'initial-baseline.log')
            state = json.loads((a / 'runtime.json').read_text()) if (a / 'runtime.json').exists() else {}
            if initial:
                measured = state.get('initial_scoring_observed_seconds')
                if measured is not None and initial.get('status') == 'ok':
                    interval = (initial['start_unix'], initial['end_unix'])
                    credited = max(0., min(measured, interval[1] - interval[0]))
                    tracks[track]['sessions'].append({'round': None, 'runtime_source': str((a / 'runtime.json').relative_to(ROOT)), 'public_grade': str((a / 'initial-baseline.log').relative_to(ROOT)), 'lower_bound_seconds': credited, 'start_unix': interval[0], 'end_unix': interval[1], 'basis': 'actual completed initial scoring and trusted host monotonic measurement'})
                    tracks[track]['wall_intervals'].append(interval)
                else:
                    tracks[track]['initial_scoring_withheld'].append({'source': str((a / 'initial-baseline.log').relative_to(ROOT)), 'status': initial.get('status'), 'reason': 'No recorded host monotonic duration for historical initial scoring; excluded from this evidence floor only.'})
            requests = [json.loads(line) for line in (a / 'coding-plan.audit.jsonl').read_text().splitlines()] if (a / 'coding-plan.audit.jsonl').exists() else []
            if track == 'seed':
                assert all(q.get('billing_mode') == 'coding_plan' and q.get('upstream_credential_in_actor') is False for q in requests)
            for session in state.get('sessions', []):
                ep = a / session['events']; events = ep.read_text() if ep.exists() else 'CLI runtime failure: missing original log'
                gp = a / session['public_result']
                try:
                    grade = json.loads(gp.read_text().splitlines()[-1])
                except (OSError, ValueError, IndexError):
                    grade = {}
                detail = session_floor(session, events, requests, grade)
                detail.update(round=session['round'], runtime_source=str((a / 'runtime.json').relative_to(ROOT)), native_events=str(ep.relative_to(ROOT)), public_grade=str(gp.relative_to(ROOT)), request_audit=str((a / 'coding-plan.audit.jsonl').relative_to(ROOT)) if requests else None, start_unix=session['start_unix'], end_unix=session['end_unix'])
                tracks[track]['sessions'].append(detail)
                tracks[track]['wall_intervals'].append((session['start_unix'], session['end_unix']))
    for track, data in tracks.items():
        intervals = data.pop('wall_intervals')
        overlap_upper = max(0., sum(end-start for start, end in intervals) - total(intervals))
        raw_floor = sum(x['lower_bound_seconds'] for x in data['sessions'])
        floor = min(primary['tracks'][track]['effective_seconds'], max(0., raw_floor - overlap_upper))
        data.update(overlap_withheld_upper_bound_seconds=overlap_upper, auditable_lower_bound_seconds=floor,
                    auditable_lower_bound_hours=floor/3600, remaining_to_ten_hours_seconds=max(0., 36000-floor),
                    primary_relaxed_effective_seconds=primary['tracks'][track]['effective_seconds'])
    return {'policy': 'qa_evidence_lower_bound_v1', 'cutoff_unix': cutoff or time.time(),
            'primary_policy_unchanged': primary['policy'], 'not_a_claim_that_withheld_time_was_a_stall': True,
            'original_records_modified': False, 'ongoing_rounds_not_yet_credited': True,
            'initial_scoring_not_credited_without_monotonic_receipt': True,
            'tracks': tracks}


if __name__ == '__main__':
    report = calculate_floor()
    output = ROOT / 'submission/expert_evidence/auditable_time_lower_bound.json'
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: {k: value[k] for k in ('auditable_lower_bound_hours', 'remaining_to_ten_hours_seconds', 'primary_relaxed_effective_seconds')} for key, value in report['tracks'].items()}))
