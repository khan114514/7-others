"""User-amended active execution accounting; keeps original runtime records intact."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def total(intervals):
    return sum(end - start for start, end in union(intervals))


def overlap(first, second):
    return union([(max(a, c), min(b, d)) for a, b in first for c, d in second if min(b, d) > max(a, c)])


def result_from_log(path):
    if not path.exists():
        return None
    for line in reversed(path.read_text().splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict) and 'start_unix' in value and 'end_unix' in value:
                return value
        except json.JSONDecodeError:
            continue
    return None


def invalid_model_generation(request):
    """A model's malformed tool output is a failed attempt, not a network stall."""
    return (request.get('http_status') == 200 and request.get('error_type') == 'ValueError' and
            (str(request.get('failure_reason', '')).startswith(('Native ', 'Orphan native ')) or
             (request.get('error_kind') == 'model_output_protocol' and
              request.get('model_returned') == request.get('model_requested'))))


def relaxed_retry_exclusion(requests, index, lower, upper):
    """Failure alone does not prove the whole request was idle or stalled."""
    request = requests[index]
    if index+1 >= len(requests):
        return None
    ended = request['ended_unix']
    following = requests[index+1]['started_unix']
    # Brief retry/transport overhead remains ordinary attempt work. Only a
    # clearly observed longer retry pause is deducted under the user amendment.
    if not 5 <= following-ended <= 15:
        return None
    start, end = max(lower, ended), min(upper, following)
    return (start, end) if end > start else None


def calculate(cutoff):
    tracks = {key: {'intervals': [], 'exclusions': [], 'sources': [], 'missing_partial_cli_log_seconds': 0., 'completed_method_cycles': 0, 'failed_model_output_generation_seconds_included': 0., 'system_suspension_gaps': [], 'failed_request_wait_seconds_included_estimated': 0.} for key in ['gpt', 'seed']}
    latest = max((ROOT / 'campaigns').glob('asd-research-*'), key=lambda p: p.stat().st_mtime)
    for job in sorted((ROOT / 'campaigns').glob('asd-research-*')):
        for trial in job.glob('harbor_task__*'):
            config = json.loads((trial / 'config.json').read_text())
            track = 'seed' if 'seed' in config['agent']['model_name'] else 'gpt'
            data = tracks[track]
            agent = trial / 'agent'
            initial = result_from_log(agent / 'initial-baseline.log')
            if initial:
                data['intervals'].append([initial['start_unix'], min(initial['end_unix'], cutoff)])
                data['sources'].append({'path': str((agent/'initial-baseline.log').relative_to(ROOT)), 'kind': 'actual_initial_resumed_own_method_scoring' if (agent/'resume-provenance.json').exists() else 'actual_initial_starter_scoring', 'start': initial['start_unix'], 'end': initial['end_unix']})
            state = json.loads((agent/'runtime.json').read_text()) if (agent/'runtime.json').exists() else {}
            sessions = state.get('sessions', [])
            for session in sessions:
                try:
                    grading=json.loads((agent/session['public_result']).read_text().splitlines()[-1])
                except (OSError,ValueError,IndexError):
                    grading={}
                closed=bool(session.get('model_completed') and session.get('semantic_method_changed') and session.get('successful_command_tools',0)>0 and session.get('next_direction') not in [None,'Inspect original model events before continuing.'] and grading.get('status') not in [None,'infrastructure_error','timeout','model_error'])
                data['completed_method_cycles'] += closed
            requests = [json.loads(line) for line in (agent/'coding-plan.audit.jsonl').read_text().splitlines()] if (agent/'coding-plan.audit.jsonl').exists() else []
            recorded_rounds = set()
            for session in sessions:
                recorded_rounds.add(session['round'])
                start, end = session['start_unix'], min(session['end_unix'], cutoff)
                model_end = start + session['model_seconds']
                event_path = agent/session['events']
                events = event_path.read_text() if event_path.exists() else ''
                if 'Error loading config.toml' in events:
                    # A rejected configuration has no model work. The protected
                    # benchmark that actually ran afterward is normal scoring.
                    benchmark = result_from_log(agent/session['public_result'])
                    if benchmark:
                        data['intervals'].append([benchmark['start_unix'], benchmark['end_unix']])
                    data['exclusions'].append({'kind': 'CLI_configuration_rejection', 'start': start, 'end': min(model_end, end), 'source': str(event_path.relative_to(ROOT))})
                else:
                    data['intervals'].append([start, end])
                    if session.get('observed_seconds') is not None and end == session['end_unix']:
                        gap = end-start-session['observed_seconds']
                        if gap > 10:
                            data['system_suspension_gaps'].append({'source':str((agent/'runtime.json').relative_to(ROOT)), 'round':session['round'], 'seconds':gap, 'evidence':'completed wall-clock interval minus recorded monotonic execution duration; quantity observed, exact subinterval not localized'})
                    if 'CLI runtime failure:' in events and 'thread.started' not in events:
                        data['missing_partial_cli_log_seconds'] += session['model_seconds']
                data['sources'].append({'path': str((agent/'runtime.json').relative_to(ROOT)), 'kind': 'observed_execution_session_including_failed_attempts', 'round': session['round'], 'start': start, 'end': end, 'model_finished': session['model_completed'], 'method_changed': session['semantic_method_changed']})
                for index, request in enumerate(requests):
                    if request.get('status') == 'completed' or not start <= request['started_unix'] < min(model_end, end):
                        continue
                    if invalid_model_generation(request):
                        generated_end = min(request['ended_unix'], model_end, end)
                        data['failed_model_output_generation_seconds_included'] += max(0., generated_end-request['started_unix'])
                    else:
                        data['failed_request_wait_seconds_included_estimated'] += max(0., min(request['ended_unix'],end)-request['started_unix'])
                    pause = relaxed_retry_exclusion(requests,index,start,end)
                    if pause:
                        data['exclusions'].append({'kind':'observed_retry_pause_at_least_5s','start':pause[0],'end':pause[1],'source':str((agent/'coding-plan.audit.jsonl').relative_to(ROOT)),'error_type':request.get('error_type')})
            # Current in-flight work counts separately from completed cycles.
            # macOS birthtime records creation immediately before CLI exec.
            if not (trial/'result.json').exists():
                for event_path in agent.glob('round_*.events.jsonl'):
                    number = int(event_path.name.split('_')[1].split('.')[0])
                    if number in recorded_rounds:
                        continue
                    created = getattr(event_path.stat(), 'st_birthtime', None)
                    if created is None:
                        continue
                    data['intervals'].append([created, cutoff])
                    data['sources'].append({'path':str(event_path.relative_to(ROOT)), 'kind':'ongoing_active_turn_snapshot','round':number,'start':created,'end':cutoff})
                    for index, request in enumerate(requests):
                        if request.get('status') == 'completed' or request['started_unix'] < created:
                            continue
                        if invalid_model_generation(request):
                            generated_end = min(request['ended_unix'], cutoff)
                            data['failed_model_output_generation_seconds_included'] += max(0., generated_end-request['started_unix'])
                        else:
                            data['failed_request_wait_seconds_included_estimated'] += max(0., min(request['ended_unix'],cutoff)-request['started_unix'])
                        pause = relaxed_retry_exclusion(requests,index,created,cutoff)
                        if pause:
                            data['exclusions'].append({'kind':'observed_retry_pause_at_least_5s','start':pause[0],'end':pause[1],'source':str((agent/'coding-plan.audit.jsonl').relative_to(ROOT))})
    for track, data in tracks.items():
        active = union(data.pop('intervals'))
        blocked = overlap(active, [(x['start'], x['end']) for x in data['exclusions']])
        suspension = sum(x['seconds'] for x in data['system_suspension_gaps'])
        deducted = min(total(active),total(blocked)+suspension)
        data.update(observed_active_windows=active, observed_window_seconds=total(active), deducted_observed_retry_pause_seconds=total(blocked), deducted_system_suspension_seconds=suspension, deducted_observed_stall_seconds=deducted, effective_seconds=total(active)-deducted)
        data['effective_hours'] = data['effective_seconds']/3600
        data['historical_precision'] = 'estimate: old partial CLI logs were discarded; any unobserved stall cannot be precisely deducted' if data['missing_partial_cli_log_seconds'] else 'recorded execution windows minus observed stalls'
        data['unquantified_native_network_errors'] = []
        if track == 'gpt':
            for trial in (ROOT/'campaigns').glob('asd-research-*/harbor_task__*'):
                config = json.loads((trial/'config.json').read_text())
                if 'gpt' not in config['agent']['model_name']:
                    continue
                for event_path in (trial/'agent').glob('round_*.events.jsonl'):
                    errors = []
                    for line in event_path.read_text().splitlines():
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if event.get('type') == 'error' and any(term in event.get('message','').lower() for term in ['network', 'connection failed', 'stream disconnected']):
                            errors.append(event['message'])
                    if errors:
                        data['unquantified_native_network_errors'].append({'source':str(event_path.relative_to(ROOT)), 'count':len(errors), 'duration_seconds':None, 'reason':'original native events lack per-event receipt timestamps; do not invent a duration or deduct normal model thought'})
            if data['unquantified_native_network_errors']:
                data['historical_precision'] += '; native GPT network errors observed without per-event timestamps, stall durations remain unquantified'
        if data['failed_request_wait_seconds_included_estimated']:
            data['historical_precision'] += '; failed-request ordinary waiting included by relaxed user policy, no assertion that every included second is model computation'
    return {'policy':'user_amended_active_execution_v2_relaxed','scope':'each model separately; all same-task native Harbor attempts','user_instruction':'有效时间宽松计入正常思考、读写代码、评测、失败尝试和正常等待；不把失败请求整段自动扣掉；只扣已记录系统挂起及明确较长重试暂停，下载与停止空档仍排除；完整闭环另计。','failed_model_output_accounting':'HTTP200 malformed output remains failed method work. A failed HTTP request does not by itself locate a stall; ordinary attempted-request waiting is included as an estimate. Quantified system suspension is deducted once and observed retry pauses >=5s are deducted.','cutoff_unix':cutoff,'cutoff_local':datetime.fromtimestamp(cutoff,ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S'),'excluded_without_counting':['downloads and image/dependency preparation before agent execution','connection probes','author Baseline/Reference experiments','time between stopped attempts','container stability-only uptime'],'original_runtime_records_modified':False,'tracks':tracks}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cutoff-unix',type=float);args=parser.parse_args()
    report=calculate(args.cutoff_unix or time.time())
    output=ROOT/'submission/expert_evidence/active_time_recalculation.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    status_path=ROOT/'STATUS.json';status=json.loads(status_path.read_text());status['time_accounting_policy']=report['policy'];status['time_accounting_cutoff']=report['cutoff_local']
    for track, data in report['tracks'].items():
        status['model_tracks'][track].update(effective_hours=data['effective_hours'],completed_method_cycles=data['completed_method_cycles'],historical_precision=data['historical_precision'])
    status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2))
    summary_path=ROOT/'submission/expert_evidence/run_summary.json';summary=json.loads(summary_path.read_text());summary['time_accounting_policy']=report['policy'];summary['time_accounting_evidence']='active_time_recalculation.json';summary['time_accounting_cutoff']=report['cutoff_local']
    for item in summary['agent_execution']:
        data=report['tracks']['seed' if 'seed' in item['model'] else 'gpt']
        item.update(duration_hours=data['effective_hours'],effective_rounds=data['completed_method_cycles'],duration_precision=data['historical_precision'])
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({'cutoff':report['cutoff_local'],'tracks':{key:{k:value[k] for k in ['effective_hours','deducted_observed_stall_seconds','completed_method_cycles','missing_partial_cli_log_seconds']} for key,value in report['tracks'].items()}},ensure_ascii=False))


if __name__=='__main__':
    main()
