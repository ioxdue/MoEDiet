"""Staged local smoke for OLMoE pruning on this machine (48 GB unified memory).

Runs the real MoEDiet pipeline code against the cached OLMoE checkpoint in two
independent stages so each fits a bounded execution window:

  check     load -> contract -> identity mask -> mask/compact equivalence (no save)
  roundtrip load -> compact -> save -> reload -> logits equivalence

Usage: python local_smoke_olmoe.py --stage check|roundtrip [--budget 32]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from moediet import load_model, load_tokenizer, resolve_source
from adapters import adapter_for
from routing import compact_in_place, masked

MODEL = "allenai/OLMoE-1B-7B-0125-Instruct"
PROMPT = "Design the control loop of a coding agent that must call tools, verify their outputs, and stop when acceptance checks pass."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "roundtrip"), required=True)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("runs/local-smoke"))
    args = parser.parse_args()

    report = {"stage": args.stage, "budget": args.budget, "model": MODEL,
              "torch": torch.__version__, "cuda": torch.cuda.is_available()}
    t0 = time.time()
    cfg, source = resolve_source(MODEL, None, fingerprint=False, architecture_profile="olmoe")
    model = load_model(source, "cpu", "auto")
    report["load_seconds"] = round(time.time() - t0, 1)
    report["dtype"] = str(model.get_input_embeddings().weight.dtype)
    report["ram_estimate_gib"] = round(sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30, 2)

    tok = load_tokenizer(source)
    ids = tok(PROMPT, return_tensors="pt").input_ids[:, :16]
    adapter = adapter_for(model.config)
    n_layers = len(adapter.blocks(model))
    total = adapter.topology(model.config)["experts_per_layer"]
    report["layers"] = n_layers
    report["experts_per_layer"] = total
    keep = sorted(range(args.budget))

    with torch.inference_mode():
        if args.stage == "check":
            t = time.time()
            report["contract"] = adapter.verify_contract(model, ids)
            report["contract_seconds"] = round(time.time() - t, 1)
            original = model(input_ids=ids, use_cache=False).logits
            with masked(model, [list(range(total))] * n_layers):
                identity = model(input_ids=ids, use_cache=False).logits
            report["identity_exact"] = bool(torch.equal(original, identity))
            with masked(model, [keep] * n_layers):
                masked_logits = model(input_ids=ids, use_cache=False).logits
            compact_in_place(model, [keep] * n_layers)
            compact_logits = model(input_ids=ids, use_cache=False).logits
            report["mask_vs_compact_max_abs_diff"] = float((masked_logits - compact_logits).abs().max())
            report["config_experts_after"] = model.config.num_experts
        else:
            with masked(model, [keep] * n_layers):
                masked_logits = model(input_ids=ids, use_cache=False).logits
            compact_in_place(model, [keep] * n_layers)
            t = time.time()
            args.out.parent.mkdir(parents=True, exist_ok=True)
            if args.out.exists():
                raise SystemExit(f"Destinazione esistente: {args.out}")
            model.save_pretrained(args.out, safe_serialization=True)
            tok.save_pretrained(args.out)
            report["save_seconds"] = round(time.time() - t, 1)
            t = time.time()
            reloaded = load_model({"id": str(args.out.resolve()), "kind": "local"}, "cpu", "auto")
            report["reload_seconds"] = round(time.time() - t, 1)
            reloaded_logits = reloaded(input_ids=ids, use_cache=False).logits
            report["roundtrip_max_abs_diff"] = float((masked_logits - reloaded_logits).abs().max())
            report["reloaded_experts"] = reloaded.config.num_experts

    report["total_seconds"] = round(time.time() - t0, 1)
    out = Path(__file__).resolve().parent / "reports" / f"local-smoke-olmoe-{args.stage}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
