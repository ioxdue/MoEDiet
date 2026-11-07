"""Versioned, declarative architecture profiles. No torch import in the launcher."""

import copy
import hashlib
import json
import math
from pathlib import Path
import re

PROFILES = Path(__file__).resolve().parent / 'architectures'
IMPLEMENTATION = 'softmax_topk_modulelist_v1'
BUILTINS = ('olmoe', 'qwen2_moe', 'qwen3_moe', 'qwen3_next', 'qwen4_exp')


def read_path(obj, path):
    for name in path.split('.') if path else ():
        try:
            obj = obj[name] if isinstance(obj, dict) else getattr(obj, name)
        except (KeyError, AttributeError) as error:
            raise ValueError(f'Campo o modulo mancante: {path}') from error
    return obj


def write_path(obj, path, value):
    parent, _, name = path.rpartition('.')
    obj = read_path(obj, parent)
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def _keys(obj, required, optional=()):
    if not isinstance(obj, dict) or set(obj) - set(required) - set(optional) or set(required) - set(obj):
        raise ValueError(f'Profilo architettura: chiavi richieste {sorted(required)}, opzionali {sorted(optional)}.')


def _path(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)*', value):
        raise ValueError(f'Percorso dichiarativo non valido: {value!r}')


def load_profile(value=None, model_type=None):
    if value is None or value == 'auto':
        if model_type not in BUILTINS:
            raise ValueError(f'Architettura {model_type!r} non supportata. Specificare un profilo compatibile.')
        value = model_type
    if isinstance(value, str) and value.lstrip().startswith('{'):
        value = json.loads(value)
    if isinstance(value, dict):
        spec = copy.deepcopy(value)
    else:
        path = PROFILES / f'{value}.json' if value in BUILTINS else Path(value)
        spec = json.loads(path.read_text(encoding='utf-8-sig'))
    _keys(spec, {'schema_version', 'id', 'model_type', 'implementation'},
          {'reason', 'transformers_version', 'layout', 'config', 'layer_selection', 'shared_experts'})
    if type(spec['schema_version']) is not int or spec['schema_version'] != 1:
        raise ValueError('Versione del profilo architettura non supportata.')
    for field in ('id', 'model_type', 'implementation'):
        if not isinstance(spec[field], str) or not spec[field].strip():
            raise ValueError(f'{field} deve essere una stringa non vuota.')
    if model_type and spec['model_type'] != model_type:
        raise ValueError(f"Profilo {spec['id']} per {spec['model_type']}, checkpoint {model_type}.")
    if spec['implementation'] == 'unsupported':
        raise ValueError(f"{spec['model_type']}: {spec.get('reason', 'adattatore non implementato')}")
    if spec['implementation'] != IMPLEMENTATION:
        raise ValueError(f"Implementazione non disponibile: {spec['implementation']}")
    _keys(spec.get('layout'), {'backbone', 'layers', 'block', 'router', 'experts', 'count', 'top_k', 'normalize'})
    _keys(spec.get('config'), {'layers', 'experts', 'top_k', 'normalize', 'context'})
    for mapping in (spec['layout'], spec['config']):
        for path in mapping.values():
            _path(path)
    if spec.get('layer_selection') not in ('all', 'qwen_sparse_step'):
        raise ValueError('layer_selection non supportata.')
    if not isinstance(spec.get('transformers_version'), str) or not re.fullmatch(r'\d+\.\d+\.\d+', spec['transformers_version']):
        raise ValueError('Il profilo richiede una versione Transformers esatta.')
    if not isinstance(spec.get('shared_experts'), list):
        raise ValueError('shared_experts deve essere una lista (anche vuota).')
    for shared in spec['shared_experts']:
        _keys(shared, {'module', 'gate', 'activation'})
        _path(shared['module'])
        _path(shared['gate'])
        if shared['activation'] != 'sigmoid':
            raise ValueError('Solo shared expert con gate sigmoid supportati.')
    return spec


def profile_hash(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def resolve_profile(cfg, requested=None):
    embedded = cfg.get('moediet_architecture') if isinstance(cfg, dict) else getattr(cfg, 'moediet_architecture', None)
    selected = requested if requested is not None else embedded
    spec = load_profile(selected, read_path(cfg, 'model_type'))
    if embedded and requested is not None and spec != load_profile(embedded, read_path(cfg, 'model_type')):
        raise ValueError('Il profilo richiesto differisce da quello incorporato nel checkpoint.')
    return spec


def layer_indices(cfg, spec):
    count = read_path(cfg, spec['config']['layers'])
    if type(count) is not int or count < 1:
        raise ValueError('Numero di layer non valido.')
    if spec['layer_selection'] == 'all':
        return list(range(count))
    excluded, step = read_path(cfg, 'mlp_only_layers'), read_path(cfg, 'decoder_sparse_step')
    if (type(step) is not int or step < 1 or not isinstance(excluded, list)
            or any(type(i) is not int or not 0 <= i < count for i in excluded)):
        raise ValueError('Selezione dei layer MoE non valida.')
    return [i for i in range(count) if i not in excluded and (i + 1) % step == 0]


def topology(cfg, spec):
    fields = spec['config']
    total, top_k = read_path(cfg, fields['experts']), read_path(cfg, fields['top_k'])
    norm = read_path(cfg, fields['normalize'])
    indices = layer_indices(cfg, spec)
    if (not indices or type(total) is not int or type(top_k) is not int or not 1 <= top_k <= total
            or type(norm) is not bool):
        raise ValueError('Checkpoint senza layer MoE o con routing non valido.')
    context = read_path(cfg, fields['context'])
    if type(context) is not int or context < 1:
        raise ValueError('Lunghezza del contesto non valida.')
    return {'num_layers': len(indices), 'experts_per_layer': total, 'experts_per_token': top_k,
            'norm_topk_prob': norm, 'layer_indices': indices}


def budgets_for(total, top_k, requested=None, fractions=None):
    if type(total) is not int or type(top_k) is not int or not 1 <= top_k <= total:
        raise ValueError('Numero di esperti o top-k del modello non valido.')
    if requested is not None and fractions is not None:
        raise ValueError('Usare budgets oppure keep_fractions, non entrambi.')
    if fractions is not None:
        if (not isinstance(fractions, list) or not fractions or any(type(x) not in (int, float)
                or not math.isfinite(x) or not 0 < x <= 1 for x in fractions)):
            raise ValueError('keep_fractions deve contenere frazioni finite in (0, 1].')
        requested = [math.ceil(total * x) for x in fractions]
    if requested is None:
        requested, value = [], top_k
        while value < total:
            requested.append(value)
            value *= 2
    elif not isinstance(requested, list) or not requested:
        raise ValueError('budgets deve essere una lista non vuota.')
    if any(type(x) is not int or not top_k <= x <= total for x in requested):
        raise ValueError(f'Ogni budget deve essere fra {top_k} (routing nativo) e {total}.')
    return sorted(set(requested) | {total})


# Profiles whose native GGUF export path is verified end-to-end (fixture tests):
# 'olmoe' (AllenAI OLMoE 0125) and 'qwen3_next' (Qwen3-Next, llama.cpp Qwen3NextModel).
EXPORT_PROFILES = ('olmoe', 'qwen3_next')


def check_export(spec, quantization):
    if quantization:
        # Only the exact built-in layouts have a verified native GGUF exporter.
        if all(spec != load_profile(name) for name in EXPORT_PROFILES):
            raise ValueError('Export GGUF non supportato per questo profilo. Usare export.format=hf o quantization=null.')


def validate_checkpoint_config(cfg, requested=None, quantization=None):
    spec = resolve_profile(cfg, requested)
    if (cfg.get('quantization_config') if isinstance(cfg, dict) else getattr(cfg, 'quantization_config', None)):
        raise ValueError('Il pruning richiede pesi HF non quantizzati.')
    routing = topology(cfg, spec)
    check_export(spec, quantization)
    return spec, routing
