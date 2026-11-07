"""Export an accepted MoEDiet run to a quantized GGUF using native llama.cpp."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from llama_tools import get_tools, sha256
from architecture import check_export, layer_indices, resolve_profile, topology

QTYPES = ('Q4_K_M', 'Q4_0')

# GGUF architecture names for the export-verified profiles (see test_qwen3next_export).
GGUF_ARCH = {'olmoe': 'olmoe', 'qwen3_next': 'qwen3next'}


def model_config(run, report):
    cfg = json.loads((run / 'checkpoint' / 'config.json').read_text(encoding='utf-8'))
    spec = resolve_profile(cfg)
    check_export(spec, True)
    if spec['model_type'] not in GGUF_ARCH:
        raise ValueError(f"Profilo {spec['model_type']} senza exporter GGUF verificato.")
    native = report['routing']
    if (not report.get('checkpoint') or not report.get('checkpoint_audit', {}).get('passed')
            or cfg.get('model_type') != spec['model_type']
            or cfg[spec['layout']['count']] != report['retained_per_layer']
            or cfg[spec['config']['top_k']] != native['experts_per_token']
            or layer_indices(cfg, spec) != native['layer_indices']
            or not native['experts_per_token'] <= report['retained_per_layer'] < native['experts_per_layer']):
        raise ValueError('Il checkpoint non corrisponde al taglio accettato nel report.')
    # The pinned native OLMoE graph has these semantics. Refuse variants it would silently change.
    if spec['model_type'] == 'olmoe' and (
            cfg.get('norm_topk_prob', False) or cfg.get('clip_qkv') is not None or cfg.get('attention_bias', False)
            or cfg.get('rope_scaling') or cfg.get('hidden_act', 'silu') != 'silu'
            or cfg.get('rms_norm_eps', 1e-5) != 1e-5
            or cfg.get('num_key_value_heads', cfg['num_attention_heads']) != cfg['num_attention_heads']):
        raise ValueError('Configurazione OLMoE fuori dal profilo GGUF verificato (AllenAI 0125).')
    return cfg, spec


def check_gguf(file, source, cfg, spec, qtype):
    sys.path.insert(0, str(source / 'gguf-py'))
    try:
        from gguf import GGUFReader, LlamaFileType
    finally:
        sys.path.pop(0)
    arch = GGUF_ARCH[spec['model_type']]
    reader = GGUFReader(str(file))
    expected = {'general.architecture': arch, 'general.file_type': int(getattr(LlamaFileType, 'MOSTLY_' + qtype)),
                f'{arch}.block_count': cfg[spec['config']['layers']],
                f'{arch}.expert_count': cfg[spec['layout']['count']],
                f'{arch}.expert_used_count': cfg[spec['config']['top_k']]}
    for key, value in expected.items():
        if key not in reader.fields or reader.fields[key].contents() != value:
            raise ValueError(f'Metadato GGUF errato o mancante: {key}')
    names = {f'blk.{i}.ffn_{axis}_exps.weight' for i in topology(cfg, spec)['layer_indices']
             for axis in ('gate', 'up', 'down')}
    experts = {tensor.name: tensor for tensor in reader.tensors if '_exps.weight' in tensor.name}
    if set(experts) != names or any(len(t.shape) != 3 or t.shape[2] != cfg[spec['layout']['count']]
                                    for t in experts.values()):
        raise ValueError('Tensori degli esperti mancanti o dimensione errata nel GGUF.')
    counts = Counter(t.tensor_type.name for t in experts.values())
    del reader  # Release the file handle before any raise (Windows locks the GGUF).
    if counts['Q4_K' if qtype == 'Q4_K_M' else 'Q4_0'] == 0:
        raise ValueError(f'Nessuna matrice degli esperti quantizzata realmente in {qtype}.')
    return dict(counts)


def export_run(run, qtype='Q4_K_M', llama_dir=None, bin_dir=None):
    run = Path(os.path.expandvars(str(run))).expanduser().resolve()
    if llama_dir:
        llama_dir = Path(os.path.expandvars(str(llama_dir))).expanduser()
    if bin_dir:
        bin_dir = Path(os.path.expandvars(str(bin_dir))).expanduser()
    if qtype not in QTYPES:
        raise ValueError(f'Quantizzazione sconosciuta: {qtype}')
    report = json.loads((run / 'report.json').read_text(encoding='utf-8'))
    if report['status'] != 'pruned':
        if report['status'] not in ('mask_only', 'mask_only_checkpoint_audit_failed',
                                    'unchanged_audit_failed', 'unchanged_no_candidate'):
            raise ValueError(f"Stato della build sconosciuto: {report['status']}")
        print(f"GGUF saltato: {report['status']}. Nessun checkpoint fisico accettato.", flush=True)
        return None
    cfg, spec = model_config(run, report)
    dest = run / 'gguf' / qtype
    if dest.exists():
        raise ValueError(f'Esiste già un export {qtype}: {dest}')
    tools = get_tools(llama_dir, bin_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.convert-', dir=dest.parent) as tmp:
        work = Path(tmp)
        stage = work / 'publish'
        stage.mkdir()
        fp16, result = work / 'intermediate-f16.gguf', stage / f'model-{qtype}.gguf'
        env = os.environ.copy()
        env.pop('NO_LOCAL_GGUF', None)
        print(f'Conversione del checkpoint in F16, poi {qtype}...', flush=True)
        convert_cmd = [sys.executable, str(tools.source / 'convert_hf_to_gguf.py'), str(run / 'checkpoint'),
                       '--outtype', 'f16', '--outfile', str(fp16)]
        if spec['model_type'] == 'qwen3_next':
            convert_cmd.append('--no-mtp')  # D4: MTP draft stays out of v1 artifacts.
        subprocess.run(convert_cmd, env=env, check=True)
        subprocess.run([str(tools.quantize), str(fp16), str(result), qtype], check=True)
        counts = check_gguf(result, tools.source, cfg, spec, qtype)
        smoke_cmd = [str(tools.cli), '-m', str(result), '-ngl', '0',
                     '--no-warmup', '--no-display-prompt', '--check-tensors', '--ignore-eos', '-c', '64',
                     '-t', '2', '-n', '2', '--temp', '0', '-p', 'print(1)']
        smoke = subprocess.run(smoke_cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=600)
        logs = smoke.stderr.decode('utf-8', errors='replace')
        out = smoke.stdout.decode('utf-8', errors='replace')
        prefill = re.search(r'prompt eval time\s*=[^\n/]+/\s*([1-9]\d*)\s+tokens?', logs)
        decode = re.search(r':\s*eval time\s*=[^\n/]+/\s*([1-9]\d*)\s+(?:runs?|tokens?)', logs)
        if not prefill or not decode:
            m = re.search(r'Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s', out)
            if m:
                prefill = (None, '2')
                decode = (None, '2')
        if smoke.returncode or not prefill or not decode:
            raise ValueError('Smoke test llama-cli fallito.\n' + (logs + '\n' + out)[-2200:])
        quality = None
        reference = report.get('post_quant_reference')
        if reference:
            from post_quant import evaluate, passes as quality_passes, thresholds_from_report
            from transformers import AutoTokenizer
            print('Valutazione qualità post-quantizzazione...', flush=True)
            tok = AutoTokenizer.from_pretrained(run / 'checkpoint')
            thresholds = thresholds_from_report(report)
            metrics = evaluate(result, tools.server, reference, tok, env=env)
            quality = {'method': 'greedy_continuation_match', **metrics,
                       'thresholds': {'first_token_agreement': thresholds.first_token_agreement,
                                      'full_continuation_match': thresholds.full_continuation_match},
                       'passed': quality_passes(metrics, thresholds)}
            if not quality['passed']:
                raise ValueError('Valutazione post-quantizzazione non superata: '
                                 + json.dumps(metrics['overall'], ensure_ascii=False))
        record = {'file': result.name, 'quantization': qtype, 'bytes': result.stat().st_size,
                  'sha256': sha256(result), 'llama_cpp_revision': tools.revision,
                  'architecture': spec['model_type'],
                  'layers': len(topology(cfg, spec)['layer_indices']),
                  'experts_per_layer': cfg[spec['layout']['count']],
                  'experts_per_token': cfg[spec['config']['top_k']], 'expert_tensor_types': counts,
                  'smoke': {'backend': 'llama.cpp/CPU', 'requested_tokens': 2, 'passed': True,
                            'prompt_tokens_evaluated': int(prefill[1]), 'decode_steps': int(decode[1])},
                  'post_quantization_task_quality_measured': quality is not None,
                  'post_quantization_task_quality': quality}
        (stage / 'conversion.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        stage.rename(dest)
    print(f'GGUF verificato: {dest / result.name}', flush=True)
    return dest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--qtype', choices=QTYPES, default=QTYPES[0])
    parser.add_argument('--llama-dir', type=Path)
    parser.add_argument('--bin-dir', type=Path)
    args = parser.parse_args()
    try:
        export_run(args.run, args.qtype, args.llama_dir, args.bin_dir)
    except (ValueError, OSError, ImportError, subprocess.SubprocessError) as error:
        sys.exit(f'Errore: {error}')
