# 本地冒烟 —— GGUF 环节（OLMoE，budget 8）

日期：2026-11-07。完成分阶段本地冒烟：剪枝 → checkpoint → GGUF。

## 运行内容

1. **剪枝 + save/reload** —— `local_smoke_olmoe.py --stage roundtrip --budget 8 --out runs/local-smoke-b8`
   （`.venv-test4554`）：roundtrip 最大绝对差异 0.0，重载后专家数 8，耗时 7.3 s。
2. **Run 组装** —— `runs/gguf-smoke-b8/checkpoint/`（保存的模型）+ 合成
   `report.json`（`status: pruned`、`retained_per_layer: 8`、routing 16 层 / 原生 64 专家 / top-8）。
   **范围说明：** audit 字段仅覆盖探针等价性（16 token 提示下 `masked vs compact`
   对数差异 0.0）；未跑语料 KL（`audit_scope: probe_equivalence_only; corpus_kl_not_run`）。
3. **转换** —— `convert_hf_to_gguf.py`（用户 llama.cpp checkout，b11185/d81aef199）
   → F16 GGUF，2.56 GiB，195 个张量，26 s（`.venv-gguf`，PYTHONPATH 指向仓库 `gguf-py`）。
4. **量化** —— `llama-quantize … Q4_K_M`：2 442 → 818 MiB（16.00 → 5.36 BPW），3 s。
5. **校验** —— `gguf_export.check_gguf`：元数据精确匹配（olmoe，16 块，8 专家，
   8 使用）；专家张量齐全，3 维，末维 = 8；量化计数
   `Q4_K: 40, Q6_K: 8`（down-proj 用 Q6_K 是 Q4_K_M 的正常形态）。`llama-cli` 冒烟
   （2 prompt token，2 decode 步，temp 0，`-ngl 0`）**通过**。

产物：`runs/gguf-smoke-b8/gguf/Q4_K_M/model-Q4_K_M.gguf`（859 903 936 字节，
sha256 `6b7b3f9d…4350`）+ `conversion.json`（`post_quantization_task_quality_measured: false`）。

## 工具链备注（本机）

- PATH 上没有 cmake / Visual Studio → `llama_tools.get_tools()` 钉死的 b6500
  自动构建在本机无法运行。改用用户现有的 llama.cpp checkout
  （`$MOEDIET_LLAMA_DIR`，b11185），通过其支持的
  `llama_dir` / `bin_dir` 覆盖传入；revision 记为 `user-b11185-d81aef199`。
- 该仓库 `bin/` 是 **Hexagon NPU 构建**：不设置 `the local Hexagon launcher script (launch-hexagon.ps1)`
  中的 FastRPC 环境（`ADSP_LIBRARY_PATH=<bin 目录>`、`GGML_HEXAGON_OPBATCH=512`、
  `GGML_HEXAGON_VTCM_SIZE=8M`），llama-cli 会报
  `ggml-hex: HTP0 failed to open session (0x80000406)`。冒烟实际使用的是 CPU 版构建
  `$MOEDIET_LLAMA_BIN`（同一 commit）+ 上述环境；
  这两个问题都由 `local_smoke_gguf_finish.py` 处理。
- 单体式的 `gguf_export.export_run` 在本机对大模型会超出 280 s 执行上限；
  本环节由 `local_smoke_gguf_finish.py` 逐步镜像（convert → quantize → check + smoke）。

## 诚实声明

- Budget 8 = OLMoE 原生 top-k，是最小的有意义的剪枝。
- 该 GGUF 仅通过结构校验和 2 token 冒烟。**量化后没有任务质量测量，这是一个硬性
  质量缺口，不是脚注：** 结构合法的 Q4_K_M 文件在真实探针上仍可能严重退化，
  因此该产物只能视为管线冒烟，不能视为质量验收的构建。HF 导出路径存在同样的
  缺口（`post_quantization_task_quality_measured: false`）。
