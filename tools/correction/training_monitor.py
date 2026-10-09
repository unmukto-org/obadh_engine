"""Read-only, restart-safe monitor for Obadh GPU work through chpc-gpu only.

Run once per launchd interval. Publish a fresh heartbeat even when idle; notify
locally on completion, failure, stalled progress, or an allocated idle GPU.
Never allocate/release GPUs, restart training, or establish SSH connections.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def latest_progress(log):
    latest = None
    for line in log.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and 'event' in item:
            latest = {k: item[k] for k in ('event', 'arm', 'step', 'rows', 'total', 'selected') if k in item}
    return latest


def update_progress(previous, step_id, name, log, now, timeout):
    progress = latest_progress(log)
    fingerprint = hashlib.sha256(json.dumps(progress, sort_keys=True).encode()).hexdigest()
    old = previous.get(step_id, {})
    changed = old.get('fingerprint') != fingerprint
    updated = now if changed or not old else old['progress_epoch']
    return dict(name=name, step_id=step_id, progress=progress, fingerprint=fingerprint,
                progress_epoch=updated, unchanged_seconds=round(now-updated),
                stalled=now-updated >= timeout)


def parse_steps(text):
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 3 or not parts[0].replace('.', '').isdigit():
            raise ValueError('Unexpected GPU process listing: ' + line[:160])
        rows.append(dict(step_id=parts[0], name=parts[1], elapsed=parts[2]))
    return rows


def classification(steps, jobs, pipelines, idle_since, now, stale_seconds):
    failed = [p for p in pipelines if p.get('status') in ('blocked', 'backup_failed', 'stopped_before_final_milestone')]
    if failed:
        return 'pipeline_failed', 'Obadh pipeline needs attention: ' + failed[-1]['path']
    if any(j['stalled'] for j in jobs.values()):
        return 'progress_stalled', 'No training/evaluation progress for the configured interval. Inspect the GPU log.'
    if steps:
        return 'running', None
    pending = [p for p in pipelines if p.get('status') not in ('complete', 'completed', 'deadline_backup_completed')]
    if pending:
        if now-idle_since >= stale_seconds:
            return 'orchestration_stalled', 'GPU is idle while an Obadh pipeline is unfinished. Inspect its queue log.'
        return 'between_stages', None
    return 'completed_idle', 'Obadh work is complete and the GPU is idle. The allocation is still held; review results or release it.'


def run(config, state, now, command):
    """One tick. command is injectable so failures can be tested without a GPU."""
    result = dict(checked_epoch=now, checked_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                  monitor_pid=os.getpid(), allocation_released=False, job_progress={}, pipelines=[])
    if now >= config['allocation_end_epoch']:
        result.update(status='deadline_reached', alert='The configured GPU allocation deadline has passed. Monitor no longer polls it.')
        return result
    # Missing credentials require human intervention. Latch rather than retrying
    # authentication every minute. --reset-connection clears only this latch.
    if state.get('connection_latched'):
        result.update(status='connection_required', connection_latched=True, alert='SSH master is missing; password and Duo login are required.')
        return result
    try:
        steps = parse_steps(command('ps'))
        result['active_steps'] = steps
        for step in steps:
            if not step['name'].startswith(config.get('job_prefix', 'obadh-')):
                continue
            log = command('logs', step['name'])
            result['job_progress'][step['step_id']] = update_progress(state.get('job_progress', {}),
                step['step_id'], step['name'], log, now, config['stall_seconds'])
        for value in config['pipeline_status_files']:
            path = Path(value)
            if path.exists():
                data = json.loads(path.read_text())
                result['pipelines'].append(dict(data, path=str(path)))
        idle_since = None if steps else state.get('idle_since_epoch') or now
        result['idle_since_epoch'] = idle_since
        result['status'], result['alert'] = classification(steps, result['job_progress'], result['pipelines'],
                                                           idle_since, now, config['stall_seconds'])
        # A cached utilization reading must be explicitly timestamped. ps/logs
        # are sufficient for minute-by-minute progress checks and avoid a new
        # Slurm step each minute.
        if now-state.get('gpu_info_epoch', 0) >= config.get('info_interval_seconds', 300):
            result['gpu_info'] = command('info').strip()
            result['gpu_info_epoch'] = now
        else:
            result['gpu_info'] = state.get('gpu_info')
            result['gpu_info_epoch'] = state.get('gpu_info_epoch')
    except Exception as error:
        detail = str(error)
        if 'no ssh master' in detail.lower():
            result.update(status='connection_required', connection_latched=True,
                          alert='SSH master is missing; a human must log in with password and Duo.')
        elif 'no running hold' in detail.lower():
            result.update(status='allocation_unavailable', alert='The configured GPU allocation is no longer available.')
        else:
            result.update(status='monitor_error', alert='Training monitor could not verify state: ' + detail[-500:])
        result['error'] = detail[-2000:]
    return result


def notification_due(result, state, now, repeat):
    if not result.get('alert'):
        return False
    if result['status'] == 'deadline_reached':
        return result['status'] != state.get('status')
    return result['status'] != state.get('status') or now-state.get('last_notification_epoch', 0) >= repeat


def notify(message):
    # argv is passed directly; no shell evaluation of notification text.
    completed = subprocess.run(['/usr/bin/osascript', '-e',
        'display notification ' + json.dumps(message) + ' with title "Obadh training monitor"'],
        capture_output=True, text=True, timeout=15)
    return dict(exit_code=completed.returncode, error=completed.stderr[-500:],
                scope='macOS notification requested; display depends on local notification settings')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--reset-connection', action='store_true')
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    directory = Path(config['state_directory']);directory.mkdir(parents=True, exist_ok=True)
    with (directory/'monitor.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        path = directory/'status.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        if args.reset_connection:state.pop('connection_latched', None)
        env = dict(os.environ, CHPC_GPU_SITE='deltaai', CHPC_GPU_JOB=str(config['allocation_job']))
        def command(*argv):
            completed = subprocess.run([config['gpu_wrapper'], *argv], env=env,
                capture_output=True, text=True, timeout=config.get('command_timeout_seconds', 40))
            if completed.returncode:
                raise RuntimeError(completed.stdout + completed.stderr)
            return completed.stdout
        now = time.time();result = run(config, state, now, command)
        result['last_notification_epoch'] = state.get('last_notification_epoch', 0)
        if notification_due(result, state, now, config.get('repeat_alert_seconds', 1800)):
            try:
                result['notification'] = notify(result['alert'])
                if result['notification']['exit_code'] == 0:result['last_notification_epoch'] = now
            except Exception as error:
                result['notification'] = dict(error=str(error))
        atomic_json(path, result)
        if result['status'] != state.get('status'):
            events = directory/'events.jsonl'
            if events.exists() and events.stat().st_size > 5_000_000:
                events.replace(directory/'events.previous.jsonl')
            with events.open('a') as f:
                f.write(json.dumps(dict(epoch=now, previous=state.get('status'), status=result['status'],
                                        alert=result.get('alert'))) + '\n')
        print(json.dumps(dict(checked_utc=result['checked_utc'], status=result['status'],
                              jobs=list(result['job_progress']))), flush=True)


if __name__ == '__main__':
    main()
