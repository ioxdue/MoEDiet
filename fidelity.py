"""Compare next-token distributions on held-out, original-model greedy continuations."""

from dataclasses import dataclass

import torch

from routing import continuation, encode


@dataclass
class Trace:
    group: str
    prefix: torch.Tensor  # Input through the penultimate token, on CPU.
    original: torch.Tensor  # Log probabilities for the generated positions, on CPU.


@torch.inference_mode()
def log_probs(model, prefix, steps):
    out = model(input_ids=prefix.to(model.get_input_embeddings().weight.device), use_cache=False,
                output_router_logits=False, logits_to_keep=steps, return_dict=True)
    logp = out.logits[0].float().log_softmax(-1).cpu()
    if logp.shape[0] != steps or not torch.isfinite(logp).all():
        raise ValueError('Log-probabilità mancanti o non finite.')
    return logp


@torch.inference_mode()
def make_traces(model, tokenizer, groups, steps):
    traces = []
    for group, prompts in groups.items():
        for prompt in prompts:
            ids = encode(model, tokenizer, prompt, reserve=steps)
            sequence = continuation(model, tokenizer, ids, steps)
            used = sequence.shape[1] - ids.shape[1]  # EOS may shorten the reference.
            if used < 1:
                raise ValueError('Il modello di riferimento ha generato una continuazione vuota.')
            prefix = sequence[:, :-1].cpu()
            traces.append(Trace(group, prefix, log_probs(model, prefix, used)))
    return traces


def replay(model, traces):
    return [log_probs(model, trace.prefix, len(trace.original)) for trace in traces]


def compare(traces, candidate, reference=None):
    if reference is None:
        reference = [t.original for t in traces]
    totals = {}
    for trace, base, new in zip(traces, reference, candidate, strict=True):
        # Sum over vocab then positions; per-group means use the number of generated positions.
        base, new = base.double().log_softmax(-1), new.double().log_softmax(-1)
        difference = base - new
        delta = (base.exp() * difference).sum().item()
        hits = (base.argmax(-1) == new.argmax(-1)).sum().item()
        row = totals.setdefault(trace.group, {'prompts': 0, 'tokens': 0, 'kl_sum': 0.0, 'hits': 0, 'peak': 0.0})
        row['prompts'] += 1
        row['tokens'] += len(base)
        row['kl_sum'] += delta
        row['hits'] += hits
        row['peak'] = max(row['peak'], difference.abs().max().item())
    return {name: {'prompts': row['prompts'], 'tokens': row['tokens'],
                   'kl_nats_per_token': max(0.0, row['kl_sum'] / row['tokens']),
                   'top1_agreement': row['hits'] / row['tokens'],
                   'max_abs_logp_delta': row['peak']} for name, row in totals.items()}


def passes(metrics, limit):
    return bool(metrics) and all(m['kl_nats_per_token'] <= limit for m in metrics.values())


def equivalent(reference, candidate, atol, rtol):
    """Independent numerical gate: matching KL to the teacher is not sufficient."""
    passed = bool(reference) and len(reference) == len(candidate)
    peak = 0.0
    if passed:
        for base, new in zip(reference, candidate, strict=True):
            if base.shape != new.shape or not torch.isfinite(base).all() or not torch.isfinite(new).all():
                passed = False
                peak = None
                break
            peak = max(peak, (base - new).abs().max().item())
            passed = bool(torch.allclose(base, new, atol=atol, rtol=rtol)) and passed
    return {'passed': passed, 'max_abs_logp_delta': peak, 'atol': atol, 'rtol': rtol}
