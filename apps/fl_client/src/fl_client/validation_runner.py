"""Run genuine MSI-compatible validators locally; never accept fallback reports."""
import base64
import html
import importlib.metadata
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


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


def standalone_report(folder):
    """Bundle HTML and local assets into one authenticated, portable document."""
    folder = Path(folder).resolve()
    reports = sorted(folder.rglob('*.html'))
    if not reports:
        raise RuntimeError('Evaluator did not generate an HTML report')
    sections = []
    for report in reports:
        def embed(match):
            attr, quote, uri = match.groups()
            if uri.startswith(('https://', 'http://', 'data:', '#', '//')):
                return match.group(0)
            asset = (report.parent / uri).resolve()
            if not asset.is_relative_to(folder) or not asset.is_file():
                raise RuntimeError('Report references an unavailable local asset')
            mime = mimetypes.guess_type(asset.name)[0] or 'application/octet-stream'
            return f'{attr}={quote}data:{mime};base64,{base64.b64encode(asset.read_bytes()).decode()}{quote}'
        content = report.read_text()
        content = re.sub(r'''(src|href)=(["'])(.*?)\2''', embed, content)
        sections.append('<h2>' + html.escape(report.stem) + '</h2><iframe sandbox="allow-scripts" '
                        'style="width:100%;height:90vh;border:0" srcdoc="' + html.escape(content, quote=True) + '"></iframe>')
    return '<!doctype html><html><head><meta charset="utf-8"></head><body>' + ''.join(sections) + '</body></html>'


def evaluate(work):
    import pandas as pd
    work = Path(work)
    payload = json.loads((work / 'configuration.json').read_text())
    metadata, configuration = payload['metadata'], payload['evaluation']
    frames = {name: pd.read_pickle(work / f'{name}.pkl') for name in ('real', 'synthetic', 'holdout')}
    expected = {v['name'] for v in metadata['variables']}
    for name, frame in frames.items():
        if frame.empty or not expected.issubset(frame.columns):
            raise ValueError(f'{name} data does not match the configured evaluation schema')
    results = []
    for kind in ('utility', 'privacy', 'fidelity'):
        function = validator(kind, payload['use_case'])
        folder = work / kind
        folder.mkdir()
        config = json.loads(json.dumps(configuration))
        if kind == 'privacy':
            config.setdefault('privacy_evaluation', {})['format'] = 'html'
        function(real_data=frames['real'], synth_data=frames['synthetic'],
                 holdout_data=frames['holdout'], metadata=metadata,
                 evaluation_config=config, output_folder=str(folder))
        (work / f'{kind}.html').write_text(standalone_report(folder))
        results.append(kind)
    (work / 'results.json').write_text(json.dumps(results))


def generate_reports(pool_id, object_uri, use_case, state):
    import pandas as pd
    from fl_client.metadata_contract import documents
    from fl_client.utils.minio_loader import download_object, upload_pool_report
    if os.getenv('VALIDATION_PROFILE') != use_case.lower() + '-full':
        raise ValueError('Reporting image does not match task disease')
    partitions = state['data_partitions']
    if 'test' not in partitions:
        raise ValueError('Privacy evaluation requires an independent test partition')
    metadata, evaluation = documents(state['model_meta'])
    synthetic = pd.read_csv(download_object(object_uri))
    with tempfile.TemporaryDirectory(prefix='pool-validation-') as directory:
        work = Path(directory)
        # Pickles are created and consumed only in this private temporary directory.
        for name, frame in [('real', partitions['train']), ('holdout', partitions['test']), ('synthetic', synthetic)]:
            frame.to_pickle(work / f'{name}.pkl')
        (work / 'configuration.json').write_text(json.dumps({'metadata': metadata, 'evaluation': evaluation, 'use_case': use_case}))
        try:
            subprocess.run([sys.executable, '-m', 'fl_client.validation_runner', str(work)],
                           check=True, timeout=int(os.getenv('VALIDATION_TIMEOUT_SECONDS', '1500')),
                           capture_output=True, text=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            # Do not transmit evaluator output that may contain patient records.
            raise RuntimeError('Validation failed; check evaluator compatibility, configuration and input types') from error
        reports = []
        for kind in json.loads((work / 'results.json').read_text()):
            uri = upload_pool_report((work / f'{kind}.html').read_text(), object_uri=object_uri,
                                     pool_id=pool_id, node_name=state['node_name'], kind=kind)
            reports.append({'report_uri': uri, 'metrics': {'scope': kind, 'fallback': False,
                            'privacy_evaluated': kind == 'privacy', 'provider': 'synth_privacy' if kind == 'privacy' else 'msi-plugin'}})
    return {'node_name': state['node_name'], 'reports': reports}


if __name__ == '__main__':
    evaluate(sys.argv[1])
