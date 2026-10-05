"""Build only verified source revisions; never use the historical vendored wheels."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def verify_source(source, expected):
    if not re.fullmatch(r'[a-fA-F0-9]{40}', expected):
        raise ValueError('A full 40-character Git commit SHA is required')
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual.lower() != expected.lower():
        raise ValueError(f'Source checkout does not match requested commit: {source.name}')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all'], text=True).strip():
        raise ValueError(f'Source checkout must be clean: {source.name}')
    return actual


def build_sources(root, env, python=sys.executable):
    profile = env['VALIDATION_PROFILE']
    if profile not in {'aml', 'none', 'aml-full', 'scd-full'}:
        raise ValueError('Unsupported validation profile')
    names = ['msi', 'aml'] if profile == 'aml' else ['msi']
    sources = []
    requested = {}
    for name in names:
        prefix = name.upper()
        branch = env.get(prefix + '_BRANCH', '').strip()
        if not branch:
            raise ValueError('Supply a branch for each selected package')
        source = root/'.ci-sources'/name
        subprocess.run(['git', 'check-ref-format', 'refs/heads/' + branch], check=True)
        expected = env[prefix + '_RESOLVED_COMMIT']
        snapshot = subprocess.check_output(['git', '-C', str(source), 'rev-parse',
            '--verify', 'refs/remotes/origin/' + branch], text=True).strip()
        if snapshot.lower() != expected.lower():
            raise ValueError('Resolved commit does not match checked-out branch snapshot')
        requested[name] = {'branch': branch}
        sources.append((name, expected))
    # Check all revisions before building anything. Do not fetch moving branches again.
    commits = {name: verify_source(root/'.ci-sources'/name, sha) for name, sha in sources}
    output = root/'.ci-wheels'
    if output.exists():
        shutil.rmtree(output)
    output.mkdir()
    for name, _ in sources:
        subprocess.run([python, '-m', 'build', '--wheel', '--outdir', str(output),
                        str(root/'.ci-sources'/name)], check=True)
    wheels = sorted(output.glob('*.whl'))
    if len(wheels) != len(sources):
        raise ValueError('Expected exactly one wheel per source checkout')
    provenance = {'validation_profile': profile, 'source_commits': commits, 'requested_sources': requested,
                  'wheels': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in wheels}}
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return provenance


if __name__ == '__main__':
    build_sources(Path(__file__).resolve().parents[1], os.environ)
