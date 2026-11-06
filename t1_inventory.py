"""T1 diagnostic: inventory Qwen/Qwen3.8-Flash-Next metadata without reading payloads.

Read-only. Resolves the requested revision to a commit SHA, lists refs to check
whether the experiment-pinned SHA exists anywhere, then inventories config,
safetensors index and headers via bounded HTTP Range reads.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from huggingface_hub import HfApi

from safetensors_metadata import inspect_hub_safetensors

REPO = "Qwen/Qwen3.8-Flash-Next"
PINNED = "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e"

CATEGORY_PATTERNS = [
    ("router_gate", re.compile(r"layers\.(\d+)\.mlp\.gate\.weight$")),
    ("experts_fused_gate_up", re.compile(r"layers\.(\d+)\.mlp\.experts\.gate_up_proj$")),
    ("experts_fused_down", re.compile(r"layers\.(\d+)\.mlp\.experts\.down_proj$")),
    ("expert_per_gate", re.compile(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight$")),
    ("expert_per_up", re.compile(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.weight$")),
    ("expert_per_down", re.compile(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.weight$")),
    ("shared_expert", re.compile(r"layers\.(\d+)\.mlp\.shared_expert\.")),
    ("shared_expert_gate", re.compile(r"layers\.(\d+)\.mlp\.shared_expert_gate\.weight$")),
    ("ple", re.compile(r"ple\.|ngram_embedding")),
    ("mtp", re.compile(r"(^|\.)mtp\.")),
    ("vision", re.compile(r"(^|\.)visual\.")),
    ("embed_tokens", re.compile(r"embed_tokens\.weight$")),
    ("lm_head", re.compile(r"lm_head\.weight$")),
]


def categorize(name):
    for label, pattern in CATEGORY_PATTERNS:
        if pattern.search(name):
            return label
    return "other"


def summarize_inventory(result):
    tensors = result["tensors"]
    groups = defaultdict(list)
    for name, meta in tensors.items():
        groups[categorize(name)].append((name, meta))

    summary = {"tensor_count": len(tensors), "payload_bytes": result["payload_bytes"],
               "shards": len(result["shards"]), "source_revision": result["source_revision"],
               "categories": {}}
    for label, items in sorted(groups.items()):
        dtypes = sorted({m["dtype"] for _, m in items})
        shapes = sorted({tuple(m["shape"]) for _, m in items})
        layers = sorted({int(match.group(1)) for n, _ in items
                         for match in [re.search(r"layers\.(\d+)\.", n)] if match})
        summary["categories"][label] = {
            "count": len(items),
            "dtypes": dtypes,
            "distinct_shapes": [list(s) for s in shapes[:8]],
            "distinct_shape_count": len(shapes),
            "bytes": sum(m["payload_bytes"] for _, m in items),
            "layer_count": len(layers),
            "layer_minmax": [layers[0], layers[-1]] if layers else None,
            "example": items[0][0],
        }
    return summary


def main():
    api = HfApi()
    report = {"repo": REPO, "pinned_revision": PINNED}

    refs = api.list_repo_refs(REPO)
    all_commits = ([c.target_commit for c in getattr(refs, "branches", [])]
                   + [c.target_commit for c in getattr(refs, "converts", [])]
                   + [t.target_commit for t in getattr(refs, "tags", [])])
    report["pinned_found_in_refs"] = PINNED in all_commits
    report["available_branches"] = [c.name for c in getattr(refs, "branches", [])]
    report["available_tags"] = [t.name for t in getattr(refs, "tags", [])][:20]

    try:
        info = api.model_info(REPO, revision=PINNED)
        report["pinned_resolves"] = info.sha
    except Exception as error:
        report["pinned_resolves"] = f"FAIL: {type(error).__name__}"

    result = inspect_hub_safetensors(REPO, revision="main")
    report["main_revision"] = result["source_revision"]
    report["model_config"] = result["model_config"]

    out_dir = Path(__file__).resolve().parent / "reports"
    out_dir.mkdir(exist_ok=True)
    inventory_path = out_dir / "qwen4-exp-t1-inventory.json"
    inventory_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    report["inventory_file"] = str(inventory_path)

    summary = summarize_inventory(result)
    report["summary"] = summary
    report_path = out_dir / "qwen4-exp-t1-summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:6000])
    print("... summary written to", report_path)


if __name__ == "__main__":
    main()
