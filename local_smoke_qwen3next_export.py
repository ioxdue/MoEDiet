"""End-to-end export test for the qwen3_next GGUF branch on a tiny fixture.

Runs in .venv-qwen3next (transformers==4.57.6, CPU). Builds a tiny Qwen3-Next
(hidden 64, 4 MoE layers, 8 experts top-2, vocab = real Qwen3-Next tokenizer size,
random weights), prunes to 4 experts, saves the checkpoint with the REAL
Qwen3-Coder-Next tokenizer files (a synthetic tokenizer can never pass the
converter's chkhsh pre-tokenizer recognition), then runs the real
gguf_export.export_run against the user's llama.cpp b11185 checkout:
convert (--no-mtp, per D4) -> quantize Q4_K_M -> check_gguf -> llama-cli smoke.

The post-quantization gate is deliberately NOT exercised here: with random weights
greedy argmax is noise, so agreement would be meaningless. The gate is validated on
OLMoE by local_smoke_post_quant.py.

NOTE: the FastRPC env (ADSP_LIBRARY_PATH...) is required on this box because even
the "cpu" llama.cpp build probes the Hexagon backend at startup.
"""

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers.models.qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM

from adapters import adapter_for

# llama.cpp checkout and executables of the SAME build, supplied via environment
# (never hardcode local machine paths in the repository).
LLAMA_DIR = Path(os.environ["MOEDIET_LLAMA_DIR"])
BIN_DIR = Path(os.environ["MOEDIET_LLAMA_BIN"])
RUN = Path(__file__).resolve().parent / "runs" / "qwen3next-export-fixture"
KEEP = [0, 2, 5, 7]
TOK_REPO = "Qwen/Qwen3-Coder-Next"


def main():
    os.environ.setdefault("ADSP_LIBRARY_PATH", str(BIN_DIR))
    os.environ.setdefault("GGML_HEXAGON_OPBATCH", "512")
    os.environ.setdefault("GGML_HEXAGON_VTCM_SIZE", "8M")

    tok_dir = Path(snapshot_download(TOK_REPO, allow_patterns=["*.json", "*.txt", "*.jinja"]))
    tok = AutoTokenizer.from_pretrained(str(tok_dir))

    torch.manual_seed(19)
    tiny = Qwen3NextConfig(
        vocab_size=len(tok), hidden_size=256, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        intermediate_size=512, moe_intermediate_size=512, shared_expert_intermediate_size=512,
        num_experts=8, num_experts_per_tok=2,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=4, linear_num_value_heads=4, linear_conv_kernel_dim=4,
        full_attention_interval=4, decoder_sparse_step=1, mlp_only_layers=[],  # real Qwen3-Next: all layers MoE
        max_position_embeddings=128, norm_topk_prob=True,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id or 0,
        bos_token_id=tok.bos_token_id, attn_implementation="eager",
    )
    model = Qwen3NextForCausalLM(tiny).eval()
    adapter = adapter_for(model.config)
    routing = adapter.topology(model.config)
    keep_rows = [KEEP] * len(adapter.blocks(model))

    # mask/compact equivalence on a pristine fixture (compact mutates the config object).
    torch.manual_seed(7)
    probe = torch.randint(0, len(tok), (1, 6))
    fresh = Qwen3NextForCausalLM(Qwen3NextConfig(**tiny.to_dict())).eval()
    with torch.inference_mode():
        with adapter.masked(fresh, keep_rows):
            masked_logits = fresh(input_ids=probe, use_cache=False).logits
        adapter.compact(fresh, keep_rows)
        compact_logits = fresh(input_ids=probe, use_cache=False).logits
    assert torch.equal(masked_logits, compact_logits), "fixture mask/compact mismatch"
    del fresh
    with torch.inference_mode():
        adapter.compact(model, keep_rows)

    if RUN.exists():
        shutil.rmtree(RUN)
    (RUN / "checkpoint").mkdir(parents=True)
    model.save_pretrained(RUN / "checkpoint", safe_serialization=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja"):
        src = tok_dir / name
        if src.is_file():
            shutil.copy(src, RUN / "checkpoint" / name)

    report = {
        "status": "pruned", "model": "tiny-fixture", "profile": "qwen3_next",
        "routing": routing, "retained_per_layer": len(KEEP), "checkpoint": True,
        "checkpoint_audit": {"passed": True, "note": "fixture; mask/compact equivalence asserted in-script"},
        "audit_scope": "fixture_export_test",
    }
    (RUN / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    from gguf_export import export_run
    dest = export_run(RUN, "Q4_K_M", LLAMA_DIR, BIN_DIR)
    conversion = json.loads((dest / "conversion.json").read_text(encoding="utf-8"))
    print(json.dumps({k: conversion[k] for k in ("architecture", "layers", "experts_per_layer",
                                                 "experts_per_token", "expert_tensor_types",
                                                 "post_quantization_task_quality_measured")}, indent=2))
    out = Path(__file__).resolve().parent / "reports" / "qwen3next-export-fixture.json"
    out.write_text(json.dumps(conversion, indent=2) + "\n", encoding="utf-8")
    print("export ok ->", dest)


if __name__ == "__main__":
    main()
