"""Private, revision-pinned code submission and durable outputs for HF Jobs."""

import json
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from huggingface_hub import HfApi, get_token, hf_hub_download

from architecture import BUILTINS
from preflight import check_metadata

HERE = Path(__file__).resolve().parent
IMAGE = 'pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel@sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6'
SOURCES = ('moediet.py', 'probes.py', 'routing.py', 'fidelity.py', 'gguf_export.py', 'llama_tools.py',
           'run.py', 'hf_jobs.py', 'requirements.txt', 'requirements-q4.txt', 'architecture.py', 'adapters.py',
           'preflight.py', *(f'architectures/{name}.json' for name in BUILTINS))
# No model, token, local commands, or user-supplied values are interpolated into this program.
BOOTSTRAP = '''
import os, sys, subprocess, tempfile
from pathlib import Path
subprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub==0.36.2'], check=True)
from huggingface_hub import snapshot_download
repo, prefix = os.environ['MOEDIET_REPO'], os.environ['MOEDIET_CODE']
with tempfile.TemporaryDirectory() as job:
    snapshot_download(repo, revision=os.environ['MOEDIET_COMMIT'], allow_patterns=[prefix + '/*'],
                      local_dir=job, token=os.environ['HF_TOKEN'])
    code = Path(job) / prefix
    import json
    cfg = json.loads((code / 'request.json').read_text())
    deps = ['-r', str(code / 'requirements-q4.txt'), 'cmake>=3.28,<4'] if cfg['quantization'] else ['-r', str(code / 'requirements.txt')]
    subprocess.run([sys.executable, '-m', 'pip', 'install', *deps], check=True)
    subprocess.run([sys.executable, str(code / 'hf_jobs.py')], cwd=job, check=True)
'''


def submit(cfg):
    token = get_token()
    if not token:
        raise ValueError('Autenticarsi con hf auth login oppure impostare HF_TOKEN.')
    api = HfApi(token=token)
    repo = cfg['output_repo']
    # Check access and resolve the input before submitting billable compute.
    source = api.model_info(cfg['model'], revision=cfg['build'].get('revision'))
    if not source.sha:
        raise ValueError('Commit del modello sorgente non disponibile.')
    metadata = hf_hub_download(cfg['model'], 'config.json', revision=source.sha, token=token)
    plan = check_metadata(cfg, json.loads(Path(metadata).read_text(encoding='utf-8-sig')))
    api.create_repo(repo_id=repo, private=True, exist_ok=True)
    if api.model_info(repo).private is not True:
        raise ValueError('La destinazione esiste ed è pubblica. Scegliere un repository privato.')
    prefix = 'requests/' + uuid4().hex
    with tempfile.TemporaryDirectory(prefix='moediet-job-') as tmp:
        stage = Path(tmp)
        for name in SOURCES:
            (stage / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE / name, stage / name)
        payload = {'model': cfg['model'], 'tags': cfg['tags'], 'quantization': cfg['quantization'],
                   'architecture': plan['architecture'],
                   'build': {**cfg['build'], 'revision': source.sha}}
        if cfg.get('corpus'):
            shutil.copyfile(cfg['corpus'], stage / 'corpus.json')
            payload['corpus'] = 'corpus.json'
        (stage / 'request.json').write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        commit = api.upload_folder(repo_id=repo, folder_path=stage, path_in_repo=prefix,
                                   commit_message='MoEDiet: codice e parametri del Job')
    job = api.run_job(image=IMAGE, command=['python', '-c', BOOTSTRAP], flavor=cfg['flavor'],
                      timeout=cfg['timeout'], secrets={'HF_TOKEN': token},
                      env={'MOEDIET_REPO': repo, 'MOEDIET_CODE': prefix, 'MOEDIET_COMMIT': commit.oid})
    print(f'Job: {job.url}\nOutput dopo il salvataggio: https://huggingface.co/{repo}/tree/main/runs/{job.id}')
    return job


def worker():
    from run import execute
    cfg = json.loads((HERE / 'request.json').read_text(encoding='utf-8'))
    if 'corpus' in cfg:
        cfg['corpus'] = str(HERE / 'corpus.json')
    repo = os.environ['MOEDIET_REPO']
    api = HfApi(token=os.environ['HF_TOKEN'])
    prefix = 'runs/' + os.environ['JOB_ID']

    def persist(folder, suffix):
        remote_path = prefix + ('/' + suffix if suffix else '')
        commit = api.upload_folder(repo_id=repo, folder_path=folder, path_in_repo=remote_path,
                                   commit_message=f'MoEDiet: {remote_path}')
        print(f'Salvato: https://huggingface.co/{repo}/tree/{commit.oid}/{remote_path}', flush=True)

    with tempfile.TemporaryDirectory(prefix='moediet-result-') as tmp:
        execute(cfg, Path(tmp) / 'result', publish=persist)


if __name__ == '__main__':
    worker()
