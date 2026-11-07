"""Collaudo for the post-quantization gate (post_quant.evaluate via llama-server).

Three checks, all against agents-v1 audit probes (1 per group so the run fits a
bounded window):

  A. b8-pruned GGUF vs reference from the PRUNED bf16 checkpoint
     -> expect REJECT: at 8/64 experts the pruned model degenerates on agentic
        probes, and the gate must catch it.
  B. full-model Q4_K_M GGUF vs reference from the ORIGINAL model (identity)
     -> expect PASS: isolates pure quantization damage (calibration: with 8-token
        greedy continuations, Q4_K_M keeps first-token agreement 1.0 and
        full-continuation match ~0.5).
  C. same as B with a shifted reference -> expect REJECT (mechanism sanity).

Usage: .venv-test4554 python local_smoke_post_quant.py [--per-group N]
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from moediet import load_model, resolve_source, post_quant_reference
from fidelity import make_traces
from probes import make_probes
from post_quant import Thresholds, evaluate, passes
from transformers import AutoTokenizer

MODEL = "allenai/OLMoE-1B-7B-0125-Instruct"
RUN = Path(__file__).resolve().parent / "runs" / "gguf-smoke-b8"
GGUF_PRUNED = RUN / "gguf" / "Q4_K_M" / "model-Q4_K_M.gguf"
GGUF_FULL = Path(__file__).resolve().parent / "runs" / "gguf-smoke-full" / "model-Q4_K_M.gguf"
SERVER = Path(r"$MOEDIET_LLAMA_BIN\llama-server.exe")
BIN_DIR = SERVER.parent
PRUNED_CKPT = Path(__file__).resolve().parent / "runs" / "local-smoke-b8"
TAGS = ["agents", "tools", "evals"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-group", type=int, default=1)
    args = parser.parse_args()

    env = dict(os.environ, ADSP_LIBRARY_PATH=str(BIN_DIR), GGML_HEXAGON_OPBATCH="512",
               GGML_HEXAGON_VTCM_SIZE="8M")
    corpus = make_probes(TAGS, Path("corpora/agents-v1.json"))
    subset = {g: ps[:args.per_group] for g, ps in corpus["audit"].items()}
    tok = AutoTokenizer.from_pretrained(str(PRUNED_CKPT))
    thresholds = Thresholds()

    # Reference A: pruned bf16 checkpoint (what the b8 GGUF was exported from).
    pruned = load_model({"id": str(PRUNED_CKPT.resolve()), "kind": "local"}, "cpu", "auto")
    ref_pruned = post_quant_reference(make_traces(pruned, tok, subset, steps=8), tok)
    del pruned
    # Reference B: original model (identity case for the full GGUF).
    _, source = resolve_source(MODEL, None, fingerprint=False, architecture_profile="olmoe")
    original = load_model(source, "cpu", "auto")
    ref_original = post_quant_reference(make_traces(original, tok, subset, steps=8), tok)

    checks = {}
    m = evaluate(GGUF_PRUNED, SERVER, ref_pruned, tok, env=env)
    checks["pruned_b8_vs_pruned_reference"] = {"metrics": m, "passed": passes(m, thresholds),
                                               "expect_pass": False}
    m = evaluate(GGUF_FULL, SERVER, ref_original, tok, env=env)
    checks["full_vs_original_reference"] = {"metrics": m, "passed": passes(m, thresholds),
                                            "expect_pass": True}
    shifted = [{**r, "continuation": r["continuation"][1:]} for r in ref_original
               if len(r["continuation"]) > 1]
    m = evaluate(GGUF_FULL, SERVER, shifted, tok, env=env)
    checks["negative_control_shifted"] = {"metrics": m, "passed": passes(m, thresholds),
                                          "expect_pass": False}

    ok = all(c["passed"] == c["expect_pass"] for c in checks.values())
    for name, c in checks.items():
        print(f"{name}: passed={c['passed']} (expected {c['expect_pass']}) "
              f"overall={json.dumps(c['metrics']['overall'])}")
    result = {"ok": ok, "thresholds": thresholds.__dict__,
              **{k: {"passed": c["passed"], "expect_pass": c["expect_pass"],
                     "overall": c["metrics"]["overall"]} for k, c in checks.items()}}
    out = Path(__file__).resolve().parent / "reports" / "local-smoke-post-quant.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not ok:
        raise SystemExit("Collaudo fallito: esito inatteso in almeno un controllo.")
    print("collaudo ok ->", out)


if __name__ == "__main__":
    main()
