import importlib.util
import json
from pathlib import Path
import subprocess
from unittest.mock import patch
import pytest

spec=importlib.util.spec_from_file_location('validation_sources',Path(__file__).resolve().parents[1]/'build_validation_sources.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def checkout(root,name):
    source=root/'.ci-sources'/name
    source.mkdir(parents=True)
    subprocess.run(['git','init','-q',str(source)],check=True)
    (source/'pyproject.toml').write_text('[project]\nname="fixture"\nversion="1.0.0"\n')
    subprocess.run(['git','-C',str(source),'add','.'],check=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=Test','-c','user.email=test@example.test','-c','commit.gpgsign=false','commit','-qm','fixture'],check=True)
    sha=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    subprocess.run(['git','-C',str(source),'update-ref','refs/remotes/origin/working',sha],check=True)
    return source,sha


def test_verified_commit_and_dirty_rejection(tmp_path):
    source,sha=checkout(tmp_path,'msi')
    assert module.verify_source(source,sha)==sha
    with pytest.raises(ValueError,match='full 40'):module.verify_source(source,'main')
    with pytest.raises(ValueError,match='does not match'):module.verify_source(source,'0'*40)
    (source/'uncommitted.py').write_text('')
    with pytest.raises(ValueError,match='clean'):module.verify_source(source,sha)


@pytest.mark.parametrize('profile,count',[('none',1),('aml',2),('aml-full',1),('scd-full',1)])
def test_builds_selected_sources_and_records_hashes(tmp_path,profile,count):
    _,msi=checkout(tmp_path,'msi')
    _,aml=checkout(tmp_path,'aml')
    output=tmp_path/'.ci-wheels';output.mkdir();(output/'obsolete.whl').write_text('old')
    real_run=subprocess.run
    def run(args,**kwargs):
        if args[1:3]==['-m','build']:
            (output/(Path(args[-1]).name+'.whl')).write_bytes(b'fixture-wheel')
            return subprocess.CompletedProcess(args,0)
        return real_run(args,**kwargs)
    with patch.object(module.subprocess,'run',side_effect=run):
        provenance=module.build_sources(tmp_path,dict(VALIDATION_PROFILE=profile,MSI_BRANCH='working',AML_BRANCH='working',MSI_RESOLVED_COMMIT=msi,AML_RESOLVED_COMMIT=aml))
    assert len(provenance['wheels'])==count
    assert not (output/'obsolete.whl').exists()
    assert provenance['source_commits']['msi']==msi
    assert json.loads((output/'provenance.json').read_text())==provenance


def test_scd_rejected_before_checkout(tmp_path):
    with pytest.raises(ValueError,match='Unsupported'):module.build_sources(tmp_path,{'VALIDATION_PROFILE':'scd'})


def test_branch_build_records_resolved_revision(tmp_path):
    source,sha=checkout(tmp_path,'msi')
    subprocess.run(['git','-C',str(source),'update-ref','refs/remotes/origin/fl-validation-frozen',sha],check=True)
    env=dict(VALIDATION_PROFILE='none',MSI_BRANCH='fl-validation-frozen',MSI_RESOLVED_COMMIT=sha)
    real_run=subprocess.run
    def run(args,**kwargs):
        if args[1:3]==['-m','build']:
            (tmp_path/'.ci-wheels/msi.whl').write_bytes(b'wheel')
            return subprocess.CompletedProcess(args,0)
        return real_run(args,**kwargs)
    with patch.object(module.subprocess,'run',side_effect=run):
        result=module.build_sources(tmp_path,env)
    assert result['source_commits']=={'msi':sha}
    assert result['requested_sources']=={'msi':{'branch':'fl-validation-frozen'}}
    with pytest.raises(ValueError,match='Supply a branch'):
        module.build_sources(tmp_path,{**env,'MSI_BRANCH':''})
    with pytest.raises(ValueError,match='branch snapshot'):
        module.build_sources(tmp_path,{**env,'MSI_RESOLVED_COMMIT':'0'*40})
