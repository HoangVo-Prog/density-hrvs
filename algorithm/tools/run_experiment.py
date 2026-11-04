"""Freeze executable source and dataset/config identity before a resumable GPU batch."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare_experiment(output, data, config, question_ids, source):
    output=Path(output);data=Path(data).resolve();source=Path(source).resolve()
    output.mkdir(parents=True,exist_ok=True)
    dataset={str(p.relative_to(data)):digest(p) for p in sorted(data.rglob('*'))
             if p.is_file() and p.suffix.lower() in {'.json','.csv','.zip','.jpg','.jpeg','.png','.webp'}}
    requested={'dataset':str(data),'dataset_hashes':dataset,'config':config,
               'question_ids':question_ids or [],'workers':1}
    path=output/'experiment_manifest.json';snapshot=output/'frozen_source'
    if path.exists():
        manifest=json.loads(path.read_text())
        if manifest['requested'] != requested:
            raise ValueError('Dataset, selected questions or configuration changed; use a new output directory')
        if json.loads((output/'frozen_config.json').read_text()) != config:
            raise ValueError('Frozen configuration changed')
        for name,sha in manifest['source_hashes'].items():
            if not (snapshot/name).is_file() or digest(snapshot/name)!=sha:
                raise ValueError(f'Frozen executable source changed: {name}')
        return manifest
    if list(output.glob('*/completion.json')):
        raise ValueError('Existing predictions have no frozen-source manifest; use a new output directory')
    hashes={}
    for p in sorted((source/'algorithm').rglob('*.py')):
        relative=p.relative_to(source);target=snapshot/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        hashes[str(relative)]=digest(target)
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        commit='unavailable'
    versions={}
    for name in ('torch','torchvision','transformers','huggingface_hub','bitsandbytes','accelerate','numpy','scipy','scikit-learn','scikit-image','Pillow'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='not installed'
    (output/'frozen_config.json').write_text(json.dumps(config,indent=2)+'\n')
    archive=output/'source_snapshot.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(snapshot/'algorithm',arcname='algorithm')
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),'requested':requested,
              'source_commit':commit,'source_hashes':hashes,'runtime_versions':versions,
              'python':sys.version,'archive_sha256':digest(archive),
              'resume_policy':'Execute the verified frozen source, even if the working branch changes.'}
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--question-ids',nargs='*')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--save-stages',action='store_true')
    args=parser.parse_args();output=args.output_dir.resolve();data=args.data_dir.resolve()
    config=json.loads(args.config.read_text())
    manifest=prepare_experiment(output,data,config,args.question_ids,Path(__file__).resolve().parents[2])
    print(f"Frozen source {manifest['source_commit']}; output {output}",flush=True)
    if args.prepare_only:return
    command=[sys.executable,'-B','-u','-m','algorithm.tools.benchmark',
        '--data-dir',str(data),'--output-dir',str(output),'--workers','1','--limit-images','0',
        '--config-file',str(output/'frozen_config.json')]
    if not args.save_stages:command.append('--no-stages')
    if args.question_ids:command.extend(['--question-ids',*args.question_ids])
    env={**os.environ,'PYTHONPATH':str(output/'frozen_source'),'PYTHONDONTWRITEBYTECODE':'1'}
    state={'started_at':datetime.now(timezone.utc).isoformat(),'status':'running','driver_pid':os.getpid()}
    with (output/'benchmark.log').open('a',buffering=1) as log:
        child=subprocess.Popen(command,cwd=output/'frozen_source',env=env,stdout=log,stderr=subprocess.STDOUT)
        state['benchmark_pid']=child.pid
        (output/'run_state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(f"Native benchmark PID {child.pid}; log {output/'benchmark.log'}",flush=True)
        try:
            code=child.wait()
        except KeyboardInterrupt:
            child.terminate();child.wait();state['status']='interrupted'
            (output/'run_state.json').write_text(json.dumps(state,indent=2)+'\n')
            raise
    summary=output/'summary.json'
    result=json.loads(summary.read_text()) if summary.exists() else {}
    state.update(returncode=code,finished_at=datetime.now(timezone.utc).isoformat(),
        status='finished' if code==0 and result.get('completed')==result.get('total') and result.get('total',0)>0 else 'finished_with_errors')
    (output/'run_state.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state),flush=True)
    if state['status']!='finished':raise SystemExit(code or 1)


if __name__=='__main__':main()
