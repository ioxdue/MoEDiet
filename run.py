"""Launch the same MoEDiet pipeline locally or on Hugging Face Jobs."""

import argparse
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys

from gguf_export import QTYPES, export_run
from probes import clean_tags, make_probes

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = 'allenai/OLMoE-1B-7B-0125-Instruct'
BUILD_FLAGS = {'budgets': '--budgets', 'lambda': '--lambda', 'max_kl': '--max-kl',
               'eval_tokens': '--eval-tokens', 'dtype': '--dtype', 'device': '--device', 'revision': '--revision'}


def path_from(base, value):
    expanded = os.path.expandvars(str(value))
    path = Path(expanded).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((base / path).resolve())


def load_config(filename, tags=None, repo=None):
    filename = Path(os.path.expandvars(str(filename))).expanduser().resolve()
    raw = json.loads(filename.read_text(encoding='utf-8-sig'))
    if not isinstance(raw, dict) or raw.get('mode') not in ('local', 'hf'):
        raise ValueError('Il profilo JSON richiede mode uguale a local o hf.')
    local = raw['mode'] == 'local'
    common = {'mode', 'model', 'tags', 'quantization', 'build', 'corpus'}
    extra = {'out', 'llama_dir', 'bin_dir'} if local else {'output_repo', 'flavor', 'timeout'}
    if unknown := set(raw) - common - extra:
        raise ValueError(f'Chiavi di configurazione sconosciute: {sorted(unknown)}')
    names = raw.get('tags') if tags is None else tags
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError('tags deve essere una lista di stringhe.')
    cfg = {'mode': raw['mode'], 'tags': clean_tags(names), 'model': raw.get('model', DEFAULT_MODEL),
           'quantization': raw.get('quantization', 'Q4_K_M'), 'build': raw.get('build', {})}
    if cfg['quantization'] not in (*QTYPES, None):
        raise ValueError(f'quantization deve essere null oppure uno tra {QTYPES}.')
    if not isinstance(cfg['model'], str) or not cfg['model'].strip():
        raise ValueError('model deve contenere un ID Hub o una cartella locale.')
    build = cfg['build']
    if not isinstance(build, dict) or set(build) - set(BUILD_FLAGS):
        raise ValueError(f'Parametri build consentiti: {list(BUILD_FLAGS)}')
    for key in ('lambda', 'max_kl'):
        value = build.get(key, 0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Valore non valido per {key}.')
    if type(build.get('eval_tokens', 8)) is not int or build.get('eval_tokens', 8) < 1:
        raise ValueError('eval_tokens deve essere un intero positivo.')
    if 'budgets' in build and (not isinstance(build['budgets'], list) or not build['budgets'] or
                               any(type(n) is not int or n < 1 for n in build['budgets'])):
        raise ValueError('budgets deve essere una lista di interi positivi.')
    if build.get('device', 'auto') not in ('auto', 'cpu', 'cuda') or build.get('dtype', 'auto') not in (
            'auto', 'float32', 'bfloat16', 'float16'):
        raise ValueError('device o dtype non valido.')
    if 'revision' in build and (not isinstance(build['revision'], str) or not build['revision'].strip()):
        raise ValueError('revision deve essere una stringa non vuota.')
    if raw.get('corpus'):
        cfg['corpus'] = path_from(filename.parent, raw['corpus'])
        make_probes(cfg['tags'], cfg['corpus'])
    if local:
        if repo:
            raise ValueError('--repo si usa con mode=hf.')
        if not isinstance(raw.get('out'), str) or not raw['out'].strip():
            raise ValueError('Indicare una nuova cartella out.')
        cfg['out'] = path_from(filename.parent, raw['out'])
        local_model = Path(path_from(filename.parent, cfg['model']))
        if local_model.is_dir():
            cfg['model'] = str(local_model)
        for key in ('llama_dir', 'bin_dir'):
            if raw.get(key):
                cfg[key] = path_from(filename.parent, raw[key])
        if cfg.get('bin_dir') and not cfg.get('llama_dir'):
            raise ValueError('bin_dir richiede llama_dir.')
    else:
        from huggingface_hub.utils import validate_repo_id
        cfg['output_repo'] = repo or raw.get('output_repo')
        for name in (cfg['model'], cfg['output_repo']):
            if not isinstance(name, str) or name.count('/') != 1:
                raise ValueError('HF richiede ID nome/modello. Specificare --repo nome/repo-privato per il risultato.')
            validate_repo_id(name)
        if cfg['output_repo'].casefold() == cfg['model'].casefold():
            raise ValueError('Sorgente e repository di destinazione devono essere distinti.')
        cfg.update(flavor=raw.get('flavor', 'l4x1'), timeout=raw.get('timeout', '90m'))
        if not isinstance(cfg['timeout'], str) or not re.fullmatch(r'[1-9]\d*[smhd]', cfg['timeout']):
            raise ValueError('Timeout non valido (per esempio 90m).')
        if not isinstance(cfg['flavor'], str) or not re.fullmatch(r'[a-z0-9-]+', cfg['flavor']):
            raise ValueError('Flavor HF non valido.')
    return cfg


def build_command(cfg, out):
    cmd = [sys.executable, str(HERE / 'moediet.py'), 'build', '--model', cfg['model'],
           '--tags', *cfg['tags'], '--out', str(out)]
    for name, flag in BUILD_FLAGS.items():
        if name in cfg['build']:
            value = cfg['build'][name]
            values = value if isinstance(value, list) else [value]
            cmd.extend([flag, *map(str, values)])
    if cfg.get('corpus'):
        cmd.extend(['--corpus', cfg['corpus']])
    return cmd


def execute(cfg, out, publish=None):
    subprocess.run(build_command(cfg, out), check=True)
    if publish:
        publish(out, '')  # Keep the audited result even if the subsequent conversion fails.
    if cfg['quantization']:
        folder = export_run(out, cfg['quantization'], cfg.get('llama_dir'), cfg.get('bin_dir'))
        if folder and publish:
            publish(folder, folder.relative_to(out).as_posix())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=lambda p: Path(os.path.expandvars(p)).expanduser(), required=True)
    parser.add_argument('--tags', nargs='+', help='Sostituisce i tag del profilo')
    parser.add_argument('--repo', help='Repository HF privato di destinazione, nome/repo')
    parser.add_argument('--dry-run', action='store_true', help='Convalida il piano senza download, upload o Job')
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.tags, args.repo)
    if args.dry_run:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return
    if cfg['mode'] == 'local':
        execute(cfg, Path(cfg['out']))
    else:
        from hf_jobs import submit
        submit(cfg)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, ImportError, subprocess.SubprocessError) as error:
        sys.exit(f'Errore: {error}')
