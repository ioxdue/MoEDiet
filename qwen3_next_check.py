"""qwen3_next adapter registration check: profile + real MoEDiet adapter code.

Runs in .venv-qwen3next (transformers==4.57.6, CPU). Verifies on a tiny fixture:
contract check, identity mask, mask/compact equivalence, save/reload round-trip;
plus topology validation of the real Qwen3-Coder-Next config (metadata only).
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
import torch
import transformers
from transformers.models.qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM

from architecture import load_profile, validate_checkpoint_config
from adapters import adapter_for

REPO = "Qwen/Qwen3-Coder-Next"
KEEP = [0, 2, 5, 7]

report = {"transformers_version": transformers.__version__, "torch_version": torch.__version__}

torch.manual_seed(19)
tiny = Qwen3NextConfig(
    vocab_size=128, hidden_size=64, num_hidden_layers=4,
    num_attention_heads=2, num_key_value_heads=1, head_dim=16,
    intermediate_size=64, moe_intermediate_size=32, shared_expert_intermediate_size=32,
    num_experts=8, num_experts_per_tok=2,
    linear_key_head_dim=8, linear_value_head_dim=8,
    linear_num_key_heads=2, linear_num_value_heads=2, linear_conv_kernel_dim=4,
    full_attention_interval=4, decoder_sparse_step=1, mlp_only_layers=[0],
    max_position_embeddings=128, norm_topk_prob=True,
    eos_token_id=127, pad_token_id=0, bos_token_id=1,
    attn_implementation="eager",
)
model = Qwen3NextForCausalLM(tiny).eval()
ids = torch.randint(0, 128, (2, 7))

adapter = adapter_for(model.config)
report["layer_indices"] = adapter.topology(model.config)["layer_indices"]

with torch.inference_mode():
    report["contract"] = adapter.verify_contract(model, ids)
    original = model(input_ids=ids, use_cache=False).logits
    rows = [list(range(8))] * len(adapter.blocks(model))
    with adapter.masked(model, rows):
        identity = model(input_ids=ids, use_cache=False).logits
    report["identity_mask_exact"] = bool(torch.equal(original, identity))

    keep_rows = [KEEP] * len(adapter.blocks(model))
    with adapter.masked(model, keep_rows):
        masked_logits = model(input_ids=ids, use_cache=False).logits
    adapter.compact(model, keep_rows)
    compact_logits = model(input_ids=ids, use_cache=False).logits
    report["mask_vs_compact_max_abs_diff"] = float((masked_logits - compact_logits).abs().max())
    report["config_num_experts_after_compact"] = model.config.num_experts

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp, safe_serialization=True)
        reloaded = Qwen3NextForCausalLM.from_pretrained(tmp, torch_dtype=torch.float32).eval()
        reloaded_logits = reloaded(input_ids=ids, use_cache=False).logits
    report["roundtrip_max_abs_diff"] = float((compact_logits - reloaded_logits).abs().max())
    report["reloaded_num_experts"] = reloaded.config.num_experts
    adapter2 = adapter_for(reloaded.config)
    report["reloaded_blocks"] = len(adapter2.blocks(reloaded))

# Real checkpoint config (metadata only): topology + shared-expert + version validation.
raw = requests.get(f"https://huggingface.co/{REPO}/raw/main/config.json", timeout=30).json()
spec, routing = validate_checkpoint_config(raw, 'qwen3_next')
report["real_checkpoint"] = {
    "repo": REPO, "layers": routing["num_layers"], "layer_indices_head_tail":
        [routing["layer_indices"][:3], routing["layer_indices"][-2:]],
    "experts_per_layer": routing["experts_per_layer"], "experts_per_token": routing["experts_per_token"],
    "norm_topk_prob": routing["norm_topk_prob"],
    "mlp_only_layers": raw.get("mlp_only_layers"), "decoder_sparse_step": raw.get("decoder_sparse_step"),
    "profile_id": spec["id"],
}

out = Path(__file__).resolve().parent / "reports" / "qwen3-next-adapter-check.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
