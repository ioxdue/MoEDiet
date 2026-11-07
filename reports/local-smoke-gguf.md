# Local smoke — GGUF leg (OLMoE, budget 8)

Date: 2026-11-07. Completes the staged local smoke: pruning → checkpoint → GGUF.

## What ran

1. **Prune + save/reload** — `local_smoke_olmoe.py --stage roundtrip --budget 8 --out runs/local-smoke-b8`
   (`.venv-test4554`): roundtrip max abs diff 0.0, reloaded experts 8, 7.3 s.
2. **Run assembly** — `runs/gguf-smoke-b8/checkpoint/` (the saved model) + synthesized
   `report.json` (`status: pruned`, `retained_per_layer: 8`, routing 16 layers / 64 native
   experts / top-8). **Scope note:** the audit fields cover probe equivalence only
   (`masked vs compact` logits 0.0 on a 16-token prompt); corpus KL was not run
   (`audit_scope: probe_equivalence_only; corpus_kl_not_run`).
3. **Convert** — `convert_hf_to_gguf.py` (user llama.cpp checkout, b11185/d81aef199)
   → F16 GGUF, 2.56 GiB, 195 tensors, 26 s (`.venv-gguf`, PYTHONPATH to repo `gguf-py`).
4. **Quantize** — `llama-quantize … Q4_K_M`: 2 442 → 818 MiB (16.00 → 5.36 BPW), 3 s.
5. **Validate** — `gguf_export.check_gguf`: metadata exact (olmoe, 16 blocks, 8 experts,
   8 used); expert tensors all present, 3-D, last dim = 8; quantized counts
   `Q4_K: 40, Q6_K: 8` (down-proj Q6_K is normal for Q4_K_M). `llama-cli` smoke
   (2 prompt tokens, 2 decode steps, temp 0, `-ngl 0`) **passed**.

Artifact: `runs/gguf-smoke-b8/gguf/Q4_K_M/model-Q4_K_M.gguf` (859 903 936 bytes,
sha256 `6b7b3f9d…4350`) + `conversion.json` (`post_quantization_task_quality_measured: false`).

## Toolchain notes (local machine — paths supplied via environment variables)

- No cmake / Visual Studio on PATH → the pinned b6500 auto-build in `llama_tools.get_tools()`
  cannot run here. Used a pre-existing local llama.cpp checkout (b11185) via the supported
  `llama_dir` / `bin_dir` override; revision recorded as `user-b11185-d81aef199`.
  Both locations are read from the environment: `MOEDIET_LLAMA_DIR` (checkout) and
  `MOEDIET_LLAMA_BIN` (executables of the same build) — never hardcoded in the repo.
- The checkout's own `bin/` is a **Hexagon NPU build**: llama-cli fails with
  `ggml-hex: HTP0 failed to open session (0x80000406)` unless the FastRPC env from
  a Hexagon launcher script is set (`ADSP_LIBRARY_PATH=<bin dir>`,
  `GGML_HEXAGON_OPBATCH=512`, `GGML_HEXAGON_VTCM_SIZE=8M`). A CPU-only build
  of the same commit plus that env was used for the smoke; both quirks are handled in
  `local_smoke_gguf_finish.py`.
- The monolithic `gguf_export.export_run` exceeds the 280 s execution cap on this box
  for large models; the leg was mirrored step-by-step (convert → quantize →
  check+smoke) by `local_smoke_gguf_finish.py`.

## Honesty caveats

- Budget 8 = native top-k of OLMoE; the smallest meaningful prune.
- The GGUF passed structural validation and a 2-token smoke only. **There is no
  task-quality measurement after quantization, and that is a hard quality gap, not a
  footnote:** a structurally valid Q4_K_M file can still degrade badly on real probes,
  so this artifact must be treated as a pipeline smoke, not as a quality-accepted build.
  The same gap exists on the HF export path (`post_quantization_task_quality_measured: false`).
