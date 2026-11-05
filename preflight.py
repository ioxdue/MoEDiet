"""Read small metadata files and reject unsupported work before starting HF compute."""

import json
from pathlib import Path

from architecture import budgets_for, validate_checkpoint_config


def read_model_config(name, revision=None, api=None):
    path = Path(name)
    if path.is_dir():
        if revision:
            raise ValueError('revision si applica solo a un modello sullo Hub.')
        return json.loads((path / 'config.json').read_text(encoding='utf-8-sig')), None
    if path.is_absolute() or str(name).startswith(('.', '~')) or '\\' in str(name):
        raise ValueError(f'Directory del modello locale non trovata: {name}')
    from huggingface_hub import HfApi, hf_hub_download
    api = api or HfApi()
    source = api.model_info(name, revision=revision)
    if not source.sha:
        raise ValueError('Commit del modello sorgente non disponibile.')
    path = hf_hub_download(name, 'config.json', revision=source.sha, token=api.token)
    return json.loads(Path(path).read_text(encoding='utf-8-sig')), source.sha


def check_metadata(cfg, raw):
    spec, routing = validate_checkpoint_config(raw, cfg.get('architecture'), cfg.get('quantization'))
    if cfg.get('mode') == 'hf' and spec['transformers_version'] != '4.55.4':
        raise ValueError('Il runtime HF distribuito richiede un profilo per Transformers 4.55.4.')
    budgets = budgets_for(routing['experts_per_layer'], routing['experts_per_token'],
                          cfg['build'].get('budgets'), cfg['build'].get('keep_fractions'))
    return {'architecture': spec, 'routing': routing, 'budgets': budgets,
            'checks': 'metadata_only', 'weights_loaded': False,
            'note': 'Controllo di compatibilita dei metadati: non misura VRAM, non scarica pesi e non prova la qualita.'}


def inspect(cfg):
    raw, revision = read_model_config(cfg['model'], cfg['build'].get('revision'))
    result = check_metadata(cfg, raw)
    result['source_revision'] = revision
    return result
