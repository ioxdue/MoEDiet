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
from architecture import check_export, load_profile
from preflight import JOB_REQUIREMENTS

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = 'allenai/OLMoE-1B-7B-0125-Instruct'
BUILD_FLAGS = {'budgets': '--budgets', 'lambda': '--lambda', 'max_kl': '--max-kl',
               'eval_tokens': '--eval-tokens', 'dtype': '--dtype', 'device': '--device', 'revision': '--revision',
               'keep_fractions': '--keep-fractions',
               'post_quant_min_first_token': '--post-quant-min-first-token',
               'post_quant_min_full_match': '--post-quant-min-full-match'}


def normalize_v2(raw):
    """Translate the public v2 schema to one internal plan; reject ambiguous duplicates."""
    raw = dict(raw)
    version = raw.pop('schema_version', 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError('schema_version deve essere 1 o 2.')
    if version == 1:
        return raw
    model = raw.get('model')
    if not isinstance(model, dict) or 'id' not in model or set(model) - {'id', 'revision'}:
        raise ValueError('model v2 richiede id e opzionalmente revision.')
    raw['model'] = model['id']
    build = raw.get('build', {})
    if not isinstance(build, dict):
        raise ValueError('build deve essere un oggetto.')
    build = dict(build)

    def put(key, value):
        if key in build:
            raise ValueError(f'Parametro duplicato: {key}.')
        build[key] = value

    if 'revision' in model:
        put('revision', model['revision'])
    pruning = raw.pop('pruning', {})
    if not isinstance(pruning, dict) or set(pruning) - {'budgets', 'keep_fractions', 'preserve_top_k', 'shared_experts'}:
        raise ValueError('Parametri pruning sconosciuti.')
    if pruning.get('preserve_top_k', True) is not True or pruning.get('shared_experts', 'preserve') != 'preserve':
        raise ValueError('Questa versione preserva top-k ed esperti condivisi.')
    for key in ('budgets', 'keep_fractions'):
        if key in pruning:
            put(key, pruning[key])
    validation = raw.pop('validation', {})
    if not isinstance(validation, dict) or set(validation) - {
            'max_kl', 'eval_tokens', 'require_control', 'post_quant_min_first_token', 'post_quant_min_full_match'}:
        raise ValueError('Parametri validation sconosciuti.')
    if validation.get('require_control', True) is not True:
        raise ValueError('Il controllo separato resta obbligatorio.')
    for key in ('max_kl', 'eval_tokens', 'post_quant_min_first_token', 'post_quant_min_full_match'):
        if key in validation:
            put(key, validation[key])
    export = raw.pop('export', {'format': 'hf'})
    if (not isinstance(export, dict) or set(export) - {'format', 'quantization'}
            or export.get('format') not in ('hf', 'gguf') or 'quantization' in raw):
        raise ValueError('Export v2: scegliere format hf oppure gguf, senza quantization al livello principale.')
    if export['format'] == 'hf' and 'quantization' in export:
        raise ValueError('La quantizzazione si specifica solo con export.format=gguf.')
    raw['quantization'] = export.get('quantization', 'Q4_K_M') if export['format'] == 'gguf' else None
    if export['format'] == 'gguf' and raw['quantization'] not in QTYPES:
        raise ValueError('Quantizzazione GGUF non supportata.')
    raw['build'] = build
    return raw


def path_from(base, value):
    expanded = os.path.expandvars(str(value))
    path = Path(expanded).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((base / path).resolve())


def load_config(filename, tags=None, repo=None, metadata_only=False):
    filename = Path(os.path.expandvars(str(filename))).expanduser().resolve()
    raw = json.loads(filename.read_text(encoding='utf-8-sig'))
    if not isinstance(raw, dict):
        raise ValueError('La configurazione deve essere un oggetto JSON.')
    raw = normalize_v2(raw)
    if not isinstance(raw, dict) or raw.get('mode') not in ('local', 'hf'):
        raise ValueError('Il profilo JSON richiede mode uguale a local o hf.')
    local = raw['mode'] == 'local'
    common = {'mode', 'model', 'tags', 'quantization', 'build', 'corpus', 'architecture_profile'}
    extra = {'out', 'llama_dir', 'bin_dir'} if local else {
        'output_repo', 'flavor', 'timeout', 'requirements', 'allow_public_output'}
    if unknown := set(raw) - common - extra:
        raise ValueError(f'Chiavi di configurazione sconosciute: {sorted(unknown)}')
    names = raw.get('tags') if tags is None else tags
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError('tags deve essere una lista di stringhe.')
    cfg = {'mode': raw['mode'], 'tags': clean_tags(names), 'model': raw.get('model', DEFAULT_MODEL),
           'quantization': raw.get('quantization', 'Q4_K_M'), 'build': raw.get('build', {})}
    if cfg['quantization'] not in (*QTYPES, None):
        raise ValueError(f'quantization deve essere null oppure uno tra {QTYPES}.')
    if 'architecture_profile' in raw:
        value = raw['architecture_profile']
        if not isinstance(value, str) or not value.strip():
            raise ValueError('architecture_profile deve essere un nome o percorso JSON.')
        from architecture import BUILTINS
        if value != 'auto' and not metadata_only:
            cfg['architecture'] = load_profile(value if value in BUILTINS else path_from(filename.parent, value))
            check_export(cfg['architecture'], cfg['quantization'])
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
    if 'keep_fractions' in build:
        values = build['keep_fractions']
        if ('budgets' in build or not isinstance(values, list) or not values or any(type(x) not in (int, float)
                or not math.isfinite(x) or not 0 < x <= 1 for x in values)):
            raise ValueError('keep_fractions richiede frazioni in (0, 1] ed esclude budgets.')
    if build.get('device', 'auto') not in ('auto', 'cpu', 'cuda') or build.get('dtype', 'auto') not in (
            'auto', 'float32', 'bfloat16', 'float16'):
        raise ValueError('device o dtype non valido.')
    if 'revision' in build and (not isinstance(build['revision'], str) or not build['revision'].strip()):
        raise ValueError('revision deve essere una stringa non vuota.')
    if raw.get('corpus') and not metadata_only:
        cfg['corpus'] = path_from(filename.parent, raw['corpus'])
        make_probes(cfg['tags'], cfg['corpus'])
    if local:
        if repo:
            raise ValueError('--repo si usa con mode=hf.')
        if not isinstance(raw.get('out'), str) or not raw['out'].strip():
            raise ValueError('Indicare una nuova cartella out.')
        cfg['out'] = path_from(filename.parent, raw['out'])
        local_model = Path(path_from(filename.parent, cfg['model']))
        explicit_path = (cfg['model'].startswith(('.', '~', '$', '%')) or Path(cfg['model']).is_absolute()
                         or '\\' in cfg['model'])
        if local_model.is_dir() or explicit_path:
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
        cfg['allow_public_output'] = bool(raw.get('allow_public_output', False))
        requirements = raw.get('requirements', 'requirements.txt')
        if requirements not in JOB_REQUIREMENTS:
            raise ValueError('requirements deve essere uno tra: ' + ', '.join(JOB_REQUIREMENTS))
        cfg['requirements'] = requirements
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
    if cfg.get('architecture'):
        cmd.extend(['--architecture-profile', json.dumps(cfg['architecture'], separators=(',', ':'))])
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
    parser.add_argument('--preflight', action='store_true', help='Legge solo metadati del modello, senza pesi, upload o Job')
    parser.add_argument('--inspect-safetensors', action='store_true',
                        help='Inventaria header e config SafeTensors locali o Hub senza leggere i payload')
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.tags, args.repo, metadata_only=args.inspect_safetensors)
    selected_modes = sum((args.dry_run, args.preflight, args.inspect_safetensors))
    if selected_modes > 1:
        raise ValueError('Scegliere una sola tra --dry-run, --preflight e --inspect-safetensors.')
    if args.dry_run:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return
    if args.inspect_safetensors:
        from safetensors_metadata import inspect_hub_safetensors, inspect_safetensors
        if cfg['mode'] == 'local':
            if not Path(cfg['model']).is_dir():
                raise ValueError('--inspect-safetensors in mode=local richiede una directory modello esistente.')
            result = inspect_safetensors(cfg['model'])
        else:
            result = inspect_hub_safetensors(cfg['model'], cfg['build'].get('revision'))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.preflight:
        from preflight import inspect
        print(json.dumps(inspect(cfg), ensure_ascii=False, indent=2))
        return
    if cfg['mode'] == 'local':
        from preflight import inspect
        plan = inspect(cfg)
        cfg['architecture'] = plan['architecture']
        if plan['source_revision']:
            cfg['build']['revision'] = plan['source_revision']
        execute(cfg, Path(cfg['out']))
    else:
        from hf_jobs import submit
        submit(cfg)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, ImportError, subprocess.SubprocessError) as error:
        sys.exit(f'Errore: {error}')
