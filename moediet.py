"""MoEDiet: profile-driven MoE pruning with held-out audits and checkpoint verification."""

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from adapters import adapter_for, tolerances
from architecture import budgets_for, profile_hash, resolve_profile, validate_checkpoint_config
from fidelity import compare, equivalent, make_traces, passes, replay
from probes import clean_tags, make_probes
from routing import blocks, compact_in_place, continuation, encode, masked, profile, select_experts, validate_rows

DEFAULT_MODEL = 'allenai/OLMoE-1B-7B-0125-Instruct'
FORMAT = 'moediet.mask.v2'
LEGACY_FORMAT = 'moediet.olmoe.mask.v1'


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def local_signature(path):
    # Check local model identity even when a training run overwrites the same directory.
    names = ('config.json', 'generation_config.json', 'model.safetensors.index.json', 'tokenizer.json',
             'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json', 'vocab.json',
             'vocab.txt', 'merges.txt', 'chat_template.jinja')
    files = sorted({*path.glob('*.safetensors'), *path.glob('*.model'),
                    *(path / name for name in names if (path / name).is_file())})
    hashes = hashlib.sha256()
    for file in files:
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            while chunk := stream.read(4 << 20):
                digest.update(chunk)
        hashes.update(file.name.encode('utf-8') + b'\x00' + digest.digest())
    return hashes.hexdigest()


def resolve_source(name, revision, fingerprint, architecture_profile=None, quantization=None):
    name = os.path.expandvars(str(name))
    path = Path(name).expanduser()
    local = path.is_dir()
    if local and revision:
        raise ValueError('--revision si applica solo a un modello sullo Hub.')
    name = str(path.resolve()) if local else name
    from preflight import read_model_config
    raw, commit = read_model_config(name, revision)
    spec, _ = validate_checkpoint_config(raw, architecture_profile, quantization)
    cfg = AutoConfig.from_pretrained(name, revision=commit, trust_remote_code=False)
    cfg.moediet_architecture = spec
    adapter_for(cfg).check_runtime()
    source = {'kind': 'local' if local else 'hub', 'id': name}
    if local and fingerprint:
        source['sha256'] = local_signature(path)
    if not local:
        source['revision'] = commit
        if not source['revision']:
            raise ValueError('Impossibile determinare il commit del checkpoint sullo Hub.')
    return cfg, source


def load_model(source, device, dtype, architecture_profile=None):
    tp = 'auto' if dtype == 'auto' else getattr(torch, dtype)
    model, info = AutoModelForCausalLM.from_pretrained(
        source['id'], revision=source.get('revision'), torch_dtype=tp, use_safetensors=True,
        attn_implementation='eager', output_loading_info=True, trust_remote_code=False,
    )
    if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'Pesi incompatibili con il modello: {info}')
    model.config.moediet_architecture = resolve_profile(model.config, architecture_profile)
    model.to(device).eval()
    blocks(model)
    return model


def load_tokenizer(source):
    return AutoTokenizer.from_pretrained(source['id'], revision=source.get('revision'), trust_remote_code=False)


def topology(cfg):
    return adapter_for(cfg).topology(cfg)


def size_of(model):
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters())}


def verify_manifest(meta, model, source):
    native = topology(model.config)
    if meta.get('format') == LEGACY_FORMAT and model.config.model_type == 'olmoe':
        native = {key: value for key, value in native.items() if key != 'layer_indices'}
    elif meta.get('format') == FORMAT:
        spec = adapter_for(model).spec
        if meta.get('architecture') != spec or meta.get('architecture_sha256') != profile_hash(spec):
            raise ValueError('La maschera usa un profilo architettura differente o alterato.')
    else:
        raise ValueError('Formato maschera non riconosciuto.')
    if meta.get('source') != source or meta.get('routing') != native:
        raise ValueError('La maschera non corrisponde alla sorgente, alla revisione o al routing del modello.')
    rows = meta.get('expert_mask')
    validate_rows(model, rows)
    if meta.get('retained_per_layer') != len(rows[0]):
        raise ValueError('Il budget dichiarato non corrisponde agli ID nella maschera.')


@torch.inference_mode()
def post_quant_reference(traces, tokenizer):
    """Compact replay reference for the post-quantization gate (gguf_export/post_quant)."""
    eos = tokenizer.eos_token_id
    reference = []
    for trace in traces:
        cont = [int(t) for t in trace.generated if int(t) != eos]
        prompt_ids = [int(t) for t in trace.prefix[0, :trace.prompt_len]]
        reference.append({'group': trace.group, 'prompt': trace.prompt_text,
                          'prompt_ids': prompt_ids, 'continuation': cont})
    return reference


def build(args, work, device):
    if not (math.isfinite(args.penalty) and args.penalty >= 0 and math.isfinite(args.max_kl) and args.max_kl >= 0):
        raise ValueError('Lambda e soglia KL devono essere finite e non negative.')
    for name in ('post_quant_min_first_token', 'post_quant_min_full_match'):
        value = getattr(args, name)
        if not (math.isfinite(value) and 0.0 <= value <= 1.0):
            raise ValueError(f'{name} deve essere in [0, 1].')
    if args.eval_tokens < 1:
        raise ValueError('--eval-tokens deve essere positivo.')
    tags = clean_tags(args.tags)
    corpus = make_probes(tags, args.corpus)
    cfg, source = resolve_source(args.model, args.revision, fingerprint=True,
                                 architecture_profile=args.architecture_profile, quantization=args.gguf)
    adapter = adapter_for(cfg)
    native = adapter.topology(cfg)
    total = native['experts_per_layer']
    budgets = budgets_for(total, native['experts_per_token'], args.budgets, args.keep_fractions)
    tok = load_tokenizer(source)
    model = load_model(source, device, args.dtype or 'auto', adapter.spec)
    dtype = str(model.get_input_embeddings().weight.dtype).split('.')[-1]
    probe = encode(model, tok, corpus['profile'][tags[0]][0])[:, :8]
    contract = adapter.verify_contract(model, probe)
    full_rows = [list(range(total)) for _ in range(native['num_layers'])]
    original_logits = model(input_ids=probe, use_cache=False).logits
    with masked(model, full_rows):
        identity_logits = model(input_ids=probe, use_cache=False).logits
    if not torch.equal(original_logits, identity_logits):
        raise ValueError('La maschera identita modifica il modello originale.')
    print('Profilazione dei router...', flush=True)
    score, tokens = profile(model, tok, corpus['profile'], tags, args.penalty)
    full_size = size_of(model)
    report = {'source': source, 'routing': native, 'dtype': dtype, 'device': str(device),
              'architecture': adapter.spec, 'architecture_sha256': profile_hash(adapter.spec),
              'adapter_contract': contract, 'identity_mask_passed': True,
              'requested_keep_fractions': args.keep_fractions,
              'torch': str(torch.__version__), 'transformers': transformers.__version__, 'probes': corpus,
              'profile_tokens_per_layer_by_group': tokens, 'score_by_layer': score.tolist(),
              'lambda': args.penalty, 'budgets': budgets, 'max_kl_nats_per_token': args.max_kl,
              'max_continuation_tokens': args.eval_tokens, 'selection': [], 'audit': None,
              'full_model': full_size, 'checkpoint': None}
    keep = total
    status = 'unchanged_no_candidate'
    if budgets[0] < total:
        selection = make_traces(model, tok, corpus['select'], args.eval_tokens)
        for budget in budgets:
            if budget == total:
                break   # The identity fallback has KL = 0 by construction.
            with masked(model, select_experts(score, budget)):
                metrics = compare(selection, replay(model, selection))
            passed = passes(metrics, args.max_kl)
            report['selection'].append({'keep': budget, 'passed': passed, 'groups': metrics})
            print(f'Budget {budget}/{total}: {"accettato" if passed else "respinto"}', flush=True)
            if passed:
                keep = budget
                break
        del selection
    report['selected_before_audit'] = keep
    rows = select_experts(score, keep)
    if keep < total:
        audit = make_traces(model, tok, corpus['audit'], args.eval_tokens)
        with masked(model, rows):
            masked_logp = replay(model, audit)
        metrics = compare(audit, masked_logp)
        passed = passes(metrics, args.max_kl)
        report['audit'] = {'keep': keep, 'passed': passed, 'groups': metrics}
        if passed:
            status = 'mask_only' if args.mask_only else 'pruned'
        else:
            status, keep = 'unchanged_audit_failed', total
            rows = select_experts(score, total)
    manifest = {'format': FORMAT, 'source': source, 'routing': native, 'profile_dtype': dtype,
                'architecture': adapter.spec, 'architecture_sha256': profile_hash(adapter.spec),
                'target_tags': tags, 'retained_per_layer': keep, 'expert_mask': rows}
    verify_manifest(manifest, model, source)
    if status == 'pruned':
        print('Esportazione e ricaricamento del checkpoint...', flush=True)
        path = work / 'checkpoint'
        compact_in_place(model, rows)
        model.save_pretrained(path, safe_serialization=True, max_shard_size='2GB')
        tok.save_pretrained(path)
        del model
        gc.collect()
        compact = load_model({'id': str(path)}, device, dtype)
        compact_logp = replay(compact, audit)
        actual = compare(audit, compact_logp)
        equivalence = equivalent(masked_logp, compact_logp, **tolerances(getattr(torch, dtype)))
        passed = passes(actual, args.max_kl) and equivalence['passed']
        report['checkpoint_audit'] = {'passed': passed, 'vs_original': actual,
                                      'mask_equivalence': equivalence,
                                      'vs_mask': compare(audit, compact_logp, masked_logp)}
        if passed:
            report['checkpoint'] = {**size_of(compact), 'safetensors_bytes': sum(
                file.stat().st_size for file in path.glob('*.safetensors'))}
            # Greedy-continuations from the PRUNED checkpoint isolate quantization
            # damage in the post-quantization gate (gguf_export/post_quant.py).
            ref_traces = make_traces(compact, tok, corpus['audit'], args.eval_tokens)
            report['post_quant_reference'] = post_quant_reference(ref_traces, tok)
            report['post_quant_thresholds'] = {
                'first_token_agreement': args.post_quant_min_first_token,
                'full_continuation_match': args.post_quant_min_full_match}
        else:
            status = 'mask_only_checkpoint_audit_failed'
        del compact
        gc.collect()
        if not passed:
            if path.resolve().parent != work.resolve() or path.name != 'checkpoint':
                raise ValueError('Percorso del checkpoint temporaneo fuori dalla directory della build.')
            shutil.rmtree(path)  # Only this run's unpublished temporary export.
    report.update(status=status, retained_per_layer=keep, expert_fraction_removed=1 - keep / total)
    write_json(work / 'mask.json', manifest)
    write_json(work / 'report.json', report)
    return report


@torch.inference_mode()
def generate(args, device):
    if args.max_new_tokens < 1 or not args.prompt.strip():
        raise ValueError('Servono un prompt non vuoto e un numero di token positivo.')
    meta = None
    if args.mask:
        if args.revision:
            raise ValueError('La revisione è già fissata nella maschera.')
        meta = json.loads(Path(args.mask).read_text(encoding='utf-8'))
        if meta.get('format') not in (FORMAT, LEGACY_FORMAT):
            raise ValueError('Formato maschera non riconosciuto.')
        original = meta['source']
        if original['kind'] == 'local' and not Path(original['id']).is_dir():
            raise ValueError('La directory locale originale registrata nella maschera non è disponibile.')
        requested = meta.get('architecture') if meta.get('format') == FORMAT else 'olmoe'
        if args.architecture_profile is not None:
            declared_type = requested['model_type'] if isinstance(requested, dict) else 'olmoe'
            minimal_cfg = {'model_type': declared_type}
            if resolve_profile(minimal_cfg, args.architecture_profile) != resolve_profile(minimal_cfg, requested):
                raise ValueError('Il profilo richiesto differisce da quello della maschera.')
        cfg, source = resolve_source(original['id'], original.get('revision'), fingerprint=True, architecture_profile=requested)
        if source != original:
            raise ValueError('Il checkpoint originale è cambiato dalla profilazione.')
    else:
        cfg, source = resolve_source(args.model, args.revision, fingerprint=False, architecture_profile=args.architecture_profile)
    dtype = args.dtype or (meta['profile_dtype'] if meta else 'auto')
    model, tok = load_model(source, device, dtype, cfg.moediet_architecture), load_tokenizer(source)
    ids = encode(model, tok, args.prompt, reserve=args.max_new_tokens)
    if meta:
        verify_manifest(meta, model, source)
        with masked(model, meta['expert_mask']):
            seq = continuation(model, tok, ids, args.max_new_tokens)
    else:
        seq = continuation(model, tok, ids, args.max_new_tokens)
    return tok.decode(seq[0, ids.shape[1]:].tolist(), skip_special_tokens=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    builder = subs.add_parser('build', help='Profilazione, selezione, audit ed export')
    builder.add_argument('--model', default=DEFAULT_MODEL, help='ID Hub o directory HF locale')
    builder.add_argument('--tags', nargs='+', required=True)
    builder.add_argument('--out', type=Path, required=True, help='Nuova directory dei risultati')
    builder.add_argument('--budgets', nargs='+', type=int, help='Esperti conservati per strato, fallback intero automatico')
    builder.add_argument('--keep-fractions', nargs='+', type=float, help='Frazioni conservate per layer MoE, arrotondate per eccesso')
    builder.add_argument('--lambda', dest='penalty', type=float, default=0.5)
    builder.add_argument('--max-kl', type=float, default=0.10, help='Soglia per ogni gruppo, in nat/token')
    builder.add_argument('--eval-tokens', type=int, default=8)
    builder.add_argument('--post-quant-min-first-token', type=float, default=0.95,
                         help="Accettazione GGUF: accordo minimo sul primo token (globale)")
    builder.add_argument('--post-quant-min-full-match', type=float, default=0.40,
                         help="Accettazione GGUF: match completo minimo della continuazione (globale)")
    builder.add_argument('--corpus', type=Path, help='JSON di sonde personalizzate')
    builder.add_argument('--mask-only', action='store_true')
    builder.add_argument('--gguf', choices=['Q4_K_M', 'Q4_0'], help='Esporta anche il checkpoint in GGUF Q4')
    builder.add_argument('--llama-dir', type=Path, help='Checkout llama.cpp per --gguf')
    builder.add_argument('--bin-dir', type=Path, help='Directory degli eseguibili della stessa build')
    gen = subs.add_parser('generate', help='Inferenza greedy con maschera o checkpoint HF')
    sources = gen.add_mutually_exclusive_group(required=True)
    sources.add_argument('--model', help='Checkpoint HF completo o compatto')
    sources.add_argument('--mask', type=Path, help='Maschera JSON con riferimento al checkpoint originale')
    gen.add_argument('--prompt', required=True)
    gen.add_argument('--max-new-tokens', type=int, default=48)
    for sub in (builder, gen):
        sub.add_argument('--architecture-profile', help='Profilo JSON oppure nome incorporato; default rilevamento automatico')
        sub.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
        sub.add_argument('--dtype', choices=['auto', 'float32', 'float16', 'bfloat16'])
        sub.add_argument('--revision', help='Commit o ref del modello sullo Hub')
    args = parser.parse_args(argv)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if args.command == 'generate':
        if args.mask:
            args.mask = Path(os.path.expandvars(str(args.mask))).expanduser()
        if args.model:
            args.model = os.path.expandvars(str(args.model))
        print(generate(args, device))
        return
    if args.corpus:
        args.corpus = Path(os.path.expandvars(str(args.corpus))).expanduser()
    if args.llama_dir:
        args.llama_dir = Path(os.path.expandvars(str(args.llama_dir))).expanduser()
    if args.bin_dir:
        args.bin_dir = Path(os.path.expandvars(str(args.bin_dir))).expanduser()
    if args.mask_only and args.gguf:
        raise ValueError('--gguf richiede il checkpoint fisico: non si combina con --mask-only.')
    if (args.llama_dir or args.bin_dir) and not args.gguf:
        raise ValueError('--llama-dir e --bin-dir richiedono --gguf.')
    if args.bin_dir and not args.llama_dir:
        raise ValueError('--bin-dir richiede --llama-dir.')
    out = Path(os.path.expandvars(str(args.out))).expanduser().resolve()
    if out.exists():
        raise ValueError(f'La destinazione esiste già: {out}. Scegliere una nuova directory.')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.moediet-', dir=out.parent) as temp:
        stage = Path(temp) / 'result'
        stage.mkdir()
        report = build(args, stage, device)
        stage.rename(out)
    print(f"{report['status']} | {report['retained_per_layer']}/{report['routing']['experts_per_layer']} esperti/strato | {out}")
    if args.gguf:
        from gguf_export import export_run
        export_run(out, args.gguf, args.llama_dir, args.bin_dir)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, ImportError, subprocess.SubprocessError) as error:
        sys.exit(f'Errore: {error}')
