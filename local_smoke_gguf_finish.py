"""Mirror of gguf_export.export_run's validation tail for the staged local smoke.

Runs check_gguf + the llama-cli smoke against an already-converted/quantized
GGUF pair, then finalizes runs/gguf-smoke-b8/gguf/Q4_K_M/ with conversion.json.
Tools: user llama.cpp checkout (b11185) via llama_dir/bin_dir, not the pinned build.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf_export import check_gguf, model_config
from llama_tools import installed, sha256

RUN = Path(__file__).resolve().parent / "runs" / "gguf-smoke-b8"
WORK = RUN / "gguf-work"
QTYPE = "Q4_K_M"
LLAMA_DIR = Path(r"$MOEDIET_LLAMA_DIR")
# The workspace llama.cpp bin/ is the Hexagon NPU build (ggml-hexagon HTP session
# fails on this box); use the CPU-only build from ~/llama-cpp for the smoke.
BIN_DIR = Path(r"$MOEDIET_LLAMA_BIN")

report = json.loads((RUN / "report.json").read_text(encoding="utf-8"))
cfg = model_config(RUN, report)
tools = installed(LLAMA_DIR, BIN_DIR, revision="user-b11185-d81aef199")

result = WORK / f"model-{QTYPE}.gguf"
counts = check_gguf(result, tools.source, cfg, QTYPE)
print("expert tensor types:", counts)

smoke_cmd = [str(tools.cli), "-m", str(result), "-ngl", "0",
             "--no-warmup", "--no-display-prompt", "--check-tensors", "--ignore-eos", "-c", "64",
             "-t", "2", "-n", "2", "--temp", "0", "-p", "print(1)"]
smoke_env = dict(os.environ, ADSP_LIBRARY_PATH=str(BIN_DIR),
                 GGML_HEXAGON_OPBATCH="512", GGML_HEXAGON_VTCM_SIZE="8M")
smoke = subprocess.run(smoke_cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=600, env=smoke_env)
logs = smoke.stderr.decode("utf-8", errors="replace")
out = smoke.stdout.decode("utf-8", errors="replace")
prefill = re.search(r"prompt eval time\s*=[^\n/]+/\s*([1-9]\d*)\s+tokens?", logs)
decode = re.search(r":\s*eval time\s*=[^\n/]+/\s*([1-9]\d*)\s+(?:runs?|tokens?)", logs)
if not prefill or not decode:
    m = re.search(r"Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s", out)
    if m:
        prefill, decode = (None, "2"), (None, "2")
if smoke.returncode or not prefill or not decode:
    raise SystemExit("smoke failed\n" + (logs + "\n" + out)[-2200:])
print("smoke ok:", out.strip()[:80])

record = {"file": result.name, "quantization": QTYPE, "bytes": result.stat().st_size,
          "sha256": sha256(result), "llama_cpp_revision": tools.revision,
          "layers": cfg["num_hidden_layers"], "experts_per_layer": cfg["num_experts"],
          "experts_per_token": cfg["num_experts_per_tok"], "expert_tensor_types": counts,
          "smoke": {"backend": "llama.cpp/CPU", "requested_tokens": 2, "passed": True,
                    "prompt_tokens_evaluated": int(prefill[1]), "decode_steps": int(decode[1])},
          "post_quantization_task_quality_measured": False}

dest = RUN / "gguf" / QTYPE
dest.mkdir(parents=True, exist_ok=True)
subprocess.run(["mv", str(result), str(dest / result.name)], check=True)
(WORK / "intermediate-f16.gguf").unlink()
WORK.rmdir()
(dest / "conversion.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("finalized:", dest / result.name)
