"""Validate versioned wheel inputs before they enter Docker build contexts.

Jenkins builds wheels from pinned source checkouts.
No fallback to stale vendored MSI wheels is allowed in the release pipeline.
"""
import argparse
from email.parser import BytesParser
from pathlib import Path
import shutil
import zipfile


def inspect_wheel(path):
    with zipfile.ZipFile(path) as wheel:
        metadata = [n for n in wheel.namelist() if n.endswith('.dist-info/METADATA')]
        if len(metadata) != 1:
            raise ValueError(f'Invalid wheel metadata: {path.name}')
        return BytesParser().parsebytes(wheel.read(metadata[0])), set(wheel.namelist())


def prepare(root, wheel_dir, profile):
    if profile not in {'none', 'aml', 'aml-full', 'scd-full'}:
        raise ValueError('Unsupported validation profile')
    packages = {}
    for path in sorted(wheel_dir.glob('*.whl')):
        meta, members = inspect_wheel(path)
        name = meta['Name'].lower().replace('_', '-')
        if name in packages:
            raise ValueError(f'Multiple wheel versions for {name}')
        packages[name] = (path, meta, members)
    if 'model-store-interface' not in packages:
        raise ValueError('Provide the model-store-interface wheel')
    msi, meta, members = packages['model-store-interface']
    if profile in {'aml-full', 'scd-full'}:
        disease = profile.split('-')[0]
        if disease not in meta.get_all('Provides-Extra', []):
            raise ValueError(f'MSI wheel lacks the {disease} disease extra')
        if 'model_store_interface/metadata/models/model.py' not in members:
            raise ValueError('MSI wheel lacks the current metadata/evaluation contract')
    if profile == 'aml':
        if 'model_store_interface/aml_utility.py' not in members:
            raise ValueError('MSI wheel lacks aml_utility adapter; select a source commit containing the adapter')
        if 'aml-utility' not in meta.get_all('Provides-Extra', []):
            raise ValueError('MSI wheel lacks aml-utility extra')
        if 'sd-quality' not in packages:
            raise ValueError('Provide the sd_quality wheel for AML reporting')
    for app in ['fl_client', 'fl_server']:
        dest = root / 'apps' / app / 'wheels'
        dest.mkdir(exist_ok=True)
        for old in dest.glob('*.whl'):
            old.unlink()
        shutil.copy2(msi, dest / msi.name)
        if app == 'fl_client' and profile == 'aml':
            quality = packages['sd-quality'][0]
            shutil.copy2(quality, dest / quality.name)
    print(f'Prepared MSI {meta["Version"]}, validation profile {profile}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheels', type=Path, required=True)
    parser.add_argument('--profile', choices=['none', 'aml', 'aml-full', 'scd-full'], required=True)
    args = parser.parse_args()
    prepare(Path(__file__).resolve().parents[1], args.wheels, args.profile)
