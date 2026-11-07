"""Post-quantization task-quality gate: greedy continuation match via llama-server.

The reference continuations are produced by the original (unquantized) model during
build and stored in report.json ('post_quant_reference'). This module starts a local
llama-server on the exported GGUF and replays the same prompts with greedy decoding,
comparing generated token ids against the reference — the exact behavior local
llama.cpp inference will have.
"""

import json
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    first_token_agreement: float = 0.95
    full_continuation_match: float = 0.40


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _post(port, payload, timeout=300):
    # OpenAI-compatible completions: accepts prompt as a list of token ids.
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_ready(port, proc, deadline=180):
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise ValueError(f'llama-server terminato prematuramente (codice {proc.returncode}).')
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(1.0)
    raise ValueError('llama-server non pronto entro il limite di tempo.')


def evaluate(gguf, server, reference, tokenizer, threads=2, env=None):
    """Replay reference prompts on the GGUF via llama-server; compare token ids.

    `reference`: list of {'group', 'prompt', 'continuation': [token ids]}.
    `tokenizer`: transformers tokenizer for the checkpoint (retokenizes server output).
    Returns per-group and overall metrics.
    """
    port = _free_port()
    cmd = [str(server), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port),
           "-ngl", "0", "-t", str(threads), "--no-warmup"]
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, env=env)
    try:
        _wait_ready(port, proc, deadline=time.time() + 180)
        rows = {}
        for item in reference:
            cont = list(item["continuation"])
            row = rows.setdefault(item["group"], {"prompts": 0, "first": 0, "full": 0, "prefix_sum": 0.0})
            if not cont:
                continue  # Reference model stopped immediately; nothing comparable.
            result = _post(port, {"prompt": item["prompt_ids"], "add_special_tokens": False,
                                  "n_predict": len(cont),
                                  "temperature": 0.0, "top_k": 1, "ignore_eos": True,
                                  "cache_prompt": True})
            text = result["choices"][0]["text"]
            out_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            row["prompts"] += 1
            row["first"] += int(bool(out_ids) and out_ids[0] == cont[0])
            row["full"] += int(out_ids[:len(cont)] == cont)
            common = 0
            for x, y in zip(out_ids, cont):
                if x != y:
                    break
                common += 1
            row["prefix_sum"] += common / len(cont)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    metrics = {name: {"prompts": r["prompts"],
                      "first_token_agreement": r["first"] / r["prompts"],
                      "full_continuation_match": r["full"] / r["prompts"],
                      "mean_prefix_fraction": r["prefix_sum"] / r["prompts"]}
               for name, r in rows.items() if r["prompts"]}
    if not metrics:
        raise ValueError("Nessun riferimento post-quantizzazione valutabile.")
    overall = {"prompts": sum(m["prompts"] for m in metrics.values())}
    for key in ("first_token_agreement", "full_continuation_match", "mean_prefix_fraction"):
        overall[key] = sum(m[key] * m["prompts"] for m in metrics.values()) / overall["prompts"]
    return {"groups": metrics, "overall": overall}


def passes(metrics, thresholds):
    o = metrics["overall"]
    return (o["first_token_agreement"] >= thresholds.first_token_agreement
            and o["full_continuation_match"] >= thresholds.full_continuation_match)


def thresholds_from_report(report):
    raw = report.get("post_quant_thresholds") or {}
    return Thresholds(first_token_agreement=raw.get("first_token_agreement", Thresholds.first_token_agreement),
                      full_continuation_match=raw.get("full_continuation_match", Thresholds.full_continuation_match))
