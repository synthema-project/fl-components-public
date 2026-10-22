"""Run the MSI simulation reporting workflow and publish its HTML artifacts."""
import importlib.metadata
import json
import logging
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from interfaces.validation_policy import required_reports


def progress(work, stage, status):
    """Only emit our own stage labels; evaluator output may contain patient data."""
    with (work / 'progress.jsonl').open('a') as stream:
        stream.write(json.dumps({'stage': stage, 'status': status}) + '\n')


def memory_counters(root=Path('/sys/fs/cgroup')):
    """Container-scoped Linux counters; no process environment or dataset content."""
    counters = {}
    for name in ('memory.current', 'memory.peak', 'memory.max', 'memory.events',
                 'memory/memory.usage_in_bytes', 'memory/memory.limit_in_bytes',
                 'memory/memory.failcnt', 'memory/memory.oom_control'):
        try:
            for line in (root / name).read_text().splitlines():
                fields = line.split()
                key, value = (name, fields[0]) if len(fields) == 1 else (f'{name}.{fields[0]}', fields[1])
                if value.isdigit() or value == 'max':
                    counters[key] = int(value) if value.isdigit() else value
        except OSError:
            pass
    return counters


def run_report_process(command, timeout):
    """Reap the evaluator directly, without waiting for inherited output pipes."""
    logger = logging.getLogger(__name__)
    before = memory_counters()
    logger.warning('Validation container memory before evaluation: %s', before or 'unavailable')
    # Regular temporary files cannot keep wait() blocked when workers inherit them.
    # Raw output stays private and is discarded after byte counts are recorded.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        logger.warning('Validation evaluator launched: pid=%s process_group=%s', process.pid, process.pid)
        try:
            code = process.wait(timeout=timeout)
            if code < 0:
                logger.warning('Validation evaluator exited: pid=%s signal=%s (%s)',
                               process.pid, -code, signal.Signals(-code).name)
            else:
                logger.warning('Validation evaluator exited: pid=%s exit_code=%s', process.pid, code)
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            # This isolated process group belongs exclusively to this evaluation.
            # Reap a timed-out evaluator and stop any inherited joblib workers.
            try:
                os.killpg(process.pid, signal.SIGKILL)
                logger.warning('Validation evaluator process group cleaned up: %s', process.pid)
            except ProcessLookupError:
                pass
            process.wait()
            after = memory_counters()
            events = {key: value - before[key] for key, value in after.items()
                      if key in before and isinstance(value, int) and isinstance(before[key], int)
                      and ('memory.events.' in key or key.endswith('failcnt') or key.endswith('oom_kill'))}
            logger.warning('Validation container memory after evaluation: %s; event deltas=%s '
                           '(container-wide, not proof of which process was killed)', after or 'unavailable', events)
            logger.warning('Validation captured output bytes: stdout=%s stderr=%s (content withheld)',
                           os.fstat(stdout.fileno()).st_size, os.fstat(stderr.fileno()).st_size)


def process_snapshot(pid, proc=Path('/proc')):
    """Read Linux process counters, including children spawned by worker threads."""
    pending, result = [pid], {}
    ticks = os.sysconf('SC_CLK_TCK')
    page_size = os.sysconf('SC_PAGE_SIZE')
    while pending:
        current = pending.pop()
        if current in result:
            continue
        try:
            directory = proc / str(current)
            # comm can contain spaces/parentheses; fields after its final ')' are fixed.
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            result[current] = {
                'state': fields[0], 'cpu': (int(fields[11]) + int(fields[12])) / ticks,
                'threads': int(fields[17]), 'rss_mib': int(fields[21]) * page_size / 1048576,
                'start': fields[19],
            }
            for children in directory.glob('task/*/children'):
                try:
                    pending.extend(int(value) for value in children.read_text().split())
                except OSError:
                    pass  # A worker may exit while we inspect it.
        except (OSError, ValueError, IndexError):
            continue
    return result


def log_process_activity(work, previous, seconds):
    logger = logging.getLogger(__name__)
    try:
        pid = int((work / 'evaluator.pid').read_text())
        snapshot = process_snapshot(pid)
    except (OSError, ValueError):
        logger.warning('Validation process diagnostics unavailable: evaluator PID not recorded yet')
        return previous
    if not snapshot:
        logger.warning('Validation evaluator PID %s unavailable in /proc; process may have exited', pid)
        return {}
    cpu_delta = 0
    for current, stats in sorted(snapshot.items()):
        old = previous.get(current)
        delta = max(0, stats['cpu'] - old['cpu']) if old and old['start'] == stats['start'] else None
        cpu_delta += delta or 0
        logger.warning('Validation process pid=%s role=%s state=%s threads=%s rss_mib=%.1f '
                       'cpu_total_s=%.2f cpu_delta_s=%s',
                       current, 'evaluator' if current == pid else 'worker', stats['state'],
                       stats['threads'], stats['rss_mib'], stats['cpu'],
                       'first sample' if delta is None else f'{delta:.2f}')
    logger.warning('Validation process summary: processes=%s sampled_cpu_delta_s=%.2f interval_s=%.1f '
                   '(R=running, S=sleeping/waiting, D=uninterruptible wait, Z=zombie; '
                   'RSS includes shared pages)', len(snapshot), cpu_delta, seconds)
    return snapshot


def monitor_progress(work, stop, interval=1, heartbeat=30):
    logger = logging.getLogger(__name__)
    started = last_update = time.monotonic()
    stage = 'starting subprocess'
    offset = 0
    last_event = started
    previous = {}
    while True:
        path = work / 'progress.jsonl'
        if path.exists():
            with path.open() as stream:
                stream.seek(offset)
                while True:
                    line = stream.readline()
                    if not line.endswith('\n'):
                        break
                    event = json.loads(line)
                    last_event = time.monotonic()
                    if event['stage'] != 'evaluator stack':
                        stage = event['stage']
                    logger.warning('Validation progress: %s %s (elapsed %.1fs)',
                                   event['stage'], event['status'], time.monotonic() - started)
                    offset = stream.tell()
        now = time.monotonic()
        if now - last_update >= heartbeat:
            logger.warning('Validation parent waiting: last stage=%s, elapsed %.1fs, '
                           'seconds since evaluator progress=%.1f', stage, now - started, now - last_event)
            if os.getenv('VALIDATION_DEBUG_STACKS', '').lower() == 'true':
                previous = log_process_activity(work, previous, now - last_update)
            last_update = now
        if stop.is_set():
            break
        stop.wait(interval)


def monitor_evaluator_stacks(work, stop, interval=60):
    try:
        sample_evaluator_stacks(work, stop, interval)
    except Exception as error:
        # Report only the diagnostic failure type, not captured exception text.
        progress(work, 'stack diagnostics', f'failed ({type(error).__name__})')


def sample_evaluator_stacks(work, stop, interval):
    """Sample function locations only, never locals, source text or exception values."""
    own_thread = threading.get_ident()
    while not stop.wait(interval):
        stacks = set()
        for thread_id, frame in sys._current_frames().items():
            if thread_id == own_thread:
                continue
            locations = []
            while frame is not None and len(locations) < 8:
                locations.append(f'{Path(frame.f_code.co_filename).name}:{frame.f_lineno} ({frame.f_code.co_name})')
                frame = frame.f_back
            stacks.add(' <- '.join(locations))
        # Unique stacks expose nested parallel fitting without hundreds of repeats.
        for stack in sorted(stacks):
            progress(work, 'evaluator stack', stack)


def validator(kind, use_case):
    if kind == 'privacy':
        from synth_privacy import sd_privacy
        return sd_privacy
    entries = list(importlib.metadata.entry_points(group=f'synthema.validation.{kind}'))
    for name in (use_case.lower(), 'default'):
        matches = [entry for entry in entries if entry.name.lower() == name]
        if len(matches) > 1:
            raise RuntimeError(f'Ambiguous {kind} plugins for {name}')
        if matches:
            return matches[0].load()
    raise RuntimeError(f'No installed {use_case} {kind} validator; report generation cannot complete')


def report_file(result):
    """Resolve MSI's file-or-output-directory return without rewriting HTML."""
    path = Path(result)
    if path.is_file():
        return path
    if path.is_dir():
        for name in ('report.html', 'index.html'):
            if (path / name).is_file():
                return path / name
        reports = sorted(path.rglob('*.html'))
        if len(reports) == 1:
            return reports[0]
        raise RuntimeError(f'MSI returned a report directory with {len(reports)} HTML files and no report.html or index.html')
    raise RuntimeError('MSI did not return an existing report file or directory')


def evaluate(work):
    work = Path(work)
    progress(work, 'input preparation', 'started')
    import pandas as pd
    from model_store_interface.metadata.models.model import SynthemaDatasetModel
    from model_store_interface import validation
    work = Path(work)
    payload = json.loads((work / 'configuration.json').read_text())
    dataset_model = SynthemaDatasetModel.model_validate(payload['dataset_meta'])
    frames = {name: pd.read_pickle(work / f'{name}.pkl')
              for name in ('train', 'test', 'synthetic')}
    if (work / 'val.pkl').exists():
        frames['val'] = pd.read_pickle(work / 'val.pkl')
    # Source CSVs can contain fields outside the generated dataset schema.
    # Give both MSI evaluators the same original features, in metadata order.
    columns = [variable.name for variable in dataset_model.metadata.variables]
    for name, frame in frames.items():
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f'Validation {name} partition is missing metadata columns: {missing}')
        if frame.columns.duplicated().any():
            raise ValueError(f'Validation {name} partition contains duplicate columns')
        frames[name] = frame.loc[:, columns].copy()
    # Keep MSI Flower simulation's partition roles, using original feature data.
    val = frames.get('val', pd.DataFrame(columns=columns))
    progress(work, 'input preparation', 'completed')
    results = []
    for kind in required_reports():
        (work / 'stage.json').write_text(json.dumps({'kind': kind}))
        function = getattr(validation, f'validate_{kind}_report')
        folder = work / kind
        folder.mkdir()
        utility = kind == 'utility'
        failures = []
        class CaptureFailure(logging.Handler):
            def emit(self, record):
                args = record.args if isinstance(record.args, tuple) else ()
                for error in args:
                    if isinstance(error, Exception):
                        frames = traceback.extract_tb(error.__traceback__)
                        failures.append({'type': type(error).__name__, 'location': ' -> '.join(
                            f'{Path(frame.filename).name}:{frame.lineno} ({frame.name})' for frame in frames[-5:])})
        handler = CaptureFailure()
        msi_logger = logging.getLogger(validation.__name__)
        msi_logger.addHandler(handler)
        try:
            progress(work, kind, 'started')
            result = function(
                synthetic_data=frames['synthetic'],
                real_data=frames['test'] if utility else frames['train'],
                holdout_data=val if utility else frames['test'],
                dataset_meta=dataset_model, output_folder=str(folder),
                node_id=payload['node_id'],
            )
        finally:
            msi_logger.removeHandler(handler)
        source = report_file(result)
        (work / f'{kind}.html').write_bytes(source.read_bytes())
        entry = {'kind': kind, 'fallback': source.name == f'{kind}_fallback.html'}
        if failures:
            entry['evaluator_failures'] = failures
        results.append(entry)
        progress(work, kind, 'completed (MSI fallback)' if entry['fallback'] else 'completed')
    (work / 'results.json').write_text(json.dumps(results))


def generate_reports(pool_id, object_uri, use_case, state):
    import pandas as pd
    from fl_client.utils.minio_loader import download_object, upload_pool_report
    if os.getenv('VALIDATION_PROFILE') != use_case.lower() + '-full':
        raise ValueError('Reporting image does not match task disease')
    # Synthetic output and metadata use the original feature schema. Model
    # encodings are for training, not inputs to the metadata-aware validators.
    partitions = state['data_partitions']
    if 'test' not in partitions:
        raise ValueError('Privacy evaluation requires an independent test partition')
    synthetic = pd.read_csv(download_object(object_uri))
    with tempfile.TemporaryDirectory(prefix='pool-validation-') as directory:
        work = Path(directory)
        # Pickles are created and consumed only in this private temporary directory.
        for name, frame in [('train', partitions['train']), ('test', partitions['test']), ('synthetic', synthetic)]:
            frame.to_pickle(work / f'{name}.pkl')
        if 'val' in partitions:
            partitions['val'].to_pickle(work / 'val.pkl')
        (work / 'configuration.json').write_text(json.dumps({
            'dataset_meta': state['model_meta'].model_dump(mode='json'),
            'node_id': state['node_name'],
        }))
        stop = threading.Event()
        monitor = threading.Thread(target=monitor_progress, args=(work, stop), daemon=True)
        logging.getLogger(__name__).warning('Validation subprocess starting for node %s', state['node_name'])
        monitor.start()
        try:
            run_report_process([sys.executable, '-m', 'fl_client.validation_runner', str(work)],
                               timeout=int(os.getenv('VALIDATION_TIMEOUT_SECONDS', '1500')))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            # Never transmit raw evaluator stdout/stderr or exception messages:
            # third-party validators can include patient values in them.
            if isinstance(error, subprocess.TimeoutExpired):
                detail = 'evaluator timed out'
            elif (work / 'failure.json').is_file():
                failure = json.loads((work / 'failure.json').read_text())
                detail = f"{failure['kind']} evaluator: {failure['type']}; location: {failure['location']}"
            else:
                detail = (f'evaluator terminated by {signal.Signals(-error.returncode).name}'
                          if error.returncode < 0 else f'evaluator process exited with status {error.returncode}')
                detail += ' before producing diagnostics; check process and container memory logs'
            raise RuntimeError('Validation failed: ' + detail) from error
        finally:
            stop.set()
            monitor.join()
        logging.getLogger(__name__).warning('Validation subprocess completed; uploading reports')
        reports = []
        for result in json.loads((work / 'results.json').read_text()):
            kind, fallback = result['kind'], result['fallback']
            if fallback:
                logging.getLogger(__name__).warning(
                    'MSI %s report is a fallback; evaluator diagnostics: %s',
                    kind, result.get('evaluator_failures', 'No evaluator exception captured; check plugin discovery'))
            uri = upload_pool_report((work / f'{kind}.html').read_text(), object_uri=object_uri,
                                     pool_id=pool_id, node_name=state['node_name'], kind=kind)
            logging.getLogger(__name__).warning('Validation %s report uploaded', kind)
            reports.append({'report_uri': uri, 'metrics': {'scope': kind, 'fallback': fallback,
                            'privacy_evaluated': kind == 'privacy' and not fallback, 'provider': 'msi-simulation'}})
    return {'node_name': state['node_name'], 'reports': reports}


def run_evaluation(work):
    """Persist code locations, never exception text, values, or source lines."""
    work = Path(work)
    (work / 'evaluator.pid').write_text(str(os.getpid()))
    stop = threading.Event()
    monitor = None
    if os.getenv('VALIDATION_DEBUG_STACKS', '').lower() == 'true':
        monitor = threading.Thread(target=monitor_evaluator_stacks, args=(work, stop), daemon=True)
        monitor.start()
    try:
        evaluate(work)
    except Exception as error:
        stage = json.loads((work / 'stage.json').read_text()) if (work / 'stage.json').exists() else {}
        progress(work, stage.get('kind', 'input preparation'), 'failed')
        frames = traceback.extract_tb(error.__traceback__)
        location = ' -> '.join(f'{Path(frame.filename).name}:{frame.lineno} ({frame.name})'
                               for frame in frames[-5:])
        (work / 'failure.json').write_text(json.dumps({
            'kind': stage.get('kind', 'input preparation'),
            'type': type(error).__name__, 'location': location,
        }))
        raise
    finally:
        stop.set()
        if monitor is not None:
            monitor.join()


if __name__ == '__main__':
    run_evaluation(sys.argv[1])
