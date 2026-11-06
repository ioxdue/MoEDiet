"""T2 diagnostic: verify installable Transformers commit supports qwen4_exp.

Isolated venv (.venv-qwen4), CPU only. Checks: registry/config load from the real
Hub checkpoint at the pinned revision, tiny-fixture construction (GDN + QSA + PLE +
MTP fields), forward pass, and save/reload round-trip. No global runtime touched.
"""

import json
import tempfile
from pathlib import Path

import torch

import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen4_exp import (Qwen4ExpConfig, Qwen4ExpForCausalLM,
                                           Qwen4ExpForConditionalGeneration, Qwen4ExpTextConfig)

REPO = "Qwen/Qwen3.8-Flash-Next"
REVISION = "7ecef55a2e549814c62143fcf8b5be457695e3b9"

report = {"transformers_version": transformers.__version__,
          "torch_version": torch.__version__, "pinned_commit": "cdb75de4adf71059c01f58dbd2d24d85084a397d"}

# 1. Registry: load the real checkpoint config through AutoConfig.
real = AutoConfig.from_pretrained(REPO, revision=REVISION, trust_remote_code=False)
report["real_config_class"] = type(real).__name__
report["real_model_type"] = real.model_type
report["real_architectures"] = real.architectures
report["real_text_model_type"] = real.text_config.model_type
report["real_norm_topk_prob"] = real.text_config.norm_topk_prob
report["real_text_layers"] = real.text_config.num_hidden_layers
report["real_text_experts"] = real.text_config.num_experts

# 2. Tiny fixture: 4 layers (3 GDN + 1 QSA at index 3), PLE on layer 2, small MoE.
torch.manual_seed(19)
tiny = Qwen4ExpTextConfig(
    vocab_size=128, hidden_size=64, num_hidden_layers=4,
    num_attention_heads=2, num_key_value_heads=1, head_dim=16,
    max_position_embeddings=128, dtype=torch.float32,
    linear_key_head_dim=8, linear_value_head_dim=8,
    linear_num_key_heads=2, linear_num_value_heads=2, linear_conv_kernel_dim=4,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    num_experts=8, num_experts_per_tok=2,
    moe_intermediate_size=32, shared_expert_intermediate_size=32,
    hc_count=2, hc_lowrank=8,
    ple_layer_ids=[2], ple_embed_dim=64, ple_conv_kernel_size=4,
    ngram_vocab_size_base=256, make_ngram_vocab_size_divisible_by=16,
    split_ngram_parts=2, heads_per_ngram=8,
    mtp_num_hidden_layers=1,
    indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16,
    indexer_budget=32, indexer_compress_ratio=4,
    eos_token_id=127, pad_token_id=0, bos_token_id=1,
)
report["fixture_num_experts"] = tiny.num_experts

model = Qwen4ExpForCausalLM(tiny)
model.eval()
ids = torch.randint(0, 128, (2, 7))
with torch.inference_mode():
    out = model(input_ids=ids, use_cache=False)
report["fixture_logits_shape"] = list(out.logits.shape)

# Router contract on the tiny model: triple output on the MoE block.
block = model.model.layers[0].mlp
x = torch.randn(3, 64)
with torch.inference_mode():
    triple = block.gate(x)
report["router_triple_shapes"] = [list(t.shape) for t in triple]

# 3. Round-trip: save and reload, compare logits and key sets.
with tempfile.TemporaryDirectory() as tmp:
    model.save_pretrained(tmp, safe_serialization=True)
    reloaded = Qwen4ExpForCausalLM.from_pretrained(tmp, torch_dtype=torch.float32)
    reloaded.eval()
    with torch.inference_mode():
        out2 = reloaded(input_ids=ids, use_cache=False)
    report["roundtrip_max_abs_diff"] = float((out.logits - out2.logits).abs().max())
    before = set(model.state_dict())
    after = set(reloaded.state_dict())
    report["roundtrip_missing_keys"] = sorted(before - after)
    report["roundtrip_unexpected_keys"] = sorted(after - before)

# 4. device_map smoke (accelerate) on the tiny fixture.
try:
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp, safe_serialization=True)
        sharded = AutoModelForCausalLM.from_pretrained(
            tmp, dtype=torch.float32, device_map="auto", low_cpu_mem_usage=True)
        with torch.inference_mode():
            out3 = sharded(input_ids=ids, use_cache=False)
        report["device_map_max_abs_diff"] = float((out.logits - out3.logits).abs().max())
except Exception as error:  # optional: accelerate absent or unsupported here
    report["device_map"] = f"SKIP/FAIL: {type(error).__name__}: {str(error)[:200]}"

out_path = Path(__file__).resolve().parent / "reports" / "qwen4-exp-t2-runtime.json"
out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
