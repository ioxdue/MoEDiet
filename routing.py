"""OLMoE adapter for Transformers 4.55.4, with native top-k preserved."""

from contextlib import ExitStack, contextmanager

import torch
from torch import nn


def blocks(model):
    cfg = model.config
    if cfg.model_type != 'olmoe' or not 1 <= cfg.num_experts_per_tok <= cfg.num_experts:
        raise ValueError('Richiesto un checkpoint OLMoE con top-k valido.')
    layers = [layer.mlp for layer in model.model.layers]
    if not layers or len(layers) != cfg.num_hidden_layers:
        raise ValueError('Numero di strati OLMoE incoerente.')
    for block in layers:
        if not (isinstance(block.gate, nn.Linear) and block.gate.bias is None
                and isinstance(block.experts, nn.ModuleList)
                and block.num_experts == block.gate.out_features == len(block.experts) == cfg.num_experts
                and block.top_k == cfg.num_experts_per_tok):
            raise ValueError('Layout del router non supportato. Usare la versione Transformers in requirements.txt.')
    return layers


def encode(model, tokenizer, prompt, reserve=0):
    if tokenizer.chat_template:
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                                            add_generation_prompt=True, return_tensors='pt')
    else:
        ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    if not ids.shape[1] or ids.shape[1] + reserve > model.config.max_position_embeddings:
        raise ValueError(f'Sonda vuota o oltre il contesto: {ids.shape[1]} token + {reserve} riservati.')
    return ids.to(model.get_input_embeddings().weight.device)


def continuation(model, tokenizer, ids, count):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=count,
                          do_sample=False, num_beams=1, use_cache=True, pad_token_id=pad_id,
                          return_dict_in_generate=False, output_router_logits=False)


@torch.inference_mode()
def profile(model, tokenizer, groups, tags, penalty):
    from probes import CONTROL

    layers = blocks(model)
    experts = model.config.num_experts
    means, counts = {}, {}
    for group, prompts in groups.items():
        sums = torch.zeros(len(layers), experts, dtype=torch.float64)
        counts[group] = 0
        for prompt in prompts:
            ids = encode(model, tokenizer, prompt)
            # Single unpadded prompt. The backbone returns logits without the LM head or auxiliary loss.
            routers = model.model(input_ids=ids, use_cache=False, output_router_logits=True,
                                  return_dict=True).router_logits
            count = ids.numel()
            if len(routers) != len(layers):
                raise ValueError('Logit di uno o più router mancanti.')
            for i, (block, logits) in enumerate(zip(layers, routers)):
                if tuple(logits.shape) != (count, experts):
                    raise ValueError(f'Forma inattesa dei logit del router {i}: {tuple(logits.shape)}')
                probs = logits.float().softmax(-1)
                weights, chosen = probs.topk(block.top_k, dim=-1)
                if block.norm_topk_prob:
                    weights = weights / weights.sum(-1, keepdim=True)
                mass = torch.zeros_like(probs).scatter_(-1, chosen, weights)
                sums[i] += mass.sum(dim=0, dtype=torch.float64).cpu()
            counts[group] += count   # Every layer has seen these same tokens exactly once.
        means[group] = sums / counts[group]
    target = torch.stack([means[tag] for tag in tags]).mean(0)
    score = target - penalty * means[CONTROL]
    if not torch.isfinite(score).all():
        raise ValueError('Lo score contiene valori non finiti.')
    return score, counts


def select_experts(score, keep):
    # Stable sort resolves equal scores in favour of the lower original ID.
    ranked = torch.argsort(score, dim=1, descending=True, stable=True)
    return ranked[:, :keep].sort(dim=1).values.tolist()


def validate_rows(model, rows):
    layers = blocks(model)
    cfg = model.config
    if not isinstance(rows, list) or len(rows) != len(layers):
        raise ValueError('La maschera deve contenere una lista per ogni strato, in ordine.')
    for row in rows:
        if (not isinstance(row, list) or not cfg.num_experts_per_tok <= len(row) <= cfg.num_experts
                or any(type(i) is not int or not 0 <= i < cfg.num_experts for i in row)
                or row != sorted(set(row)) or len(row) != len(rows[0])):
            raise ValueError('Maschera non valida: servono ID unici ordinati e un budget uniforme >= top-k.')
    return layers


@contextmanager
def masked(model, rows):
    layers = validate_rows(model, rows)
    with ExitStack() as stack:
        for block, ids in zip(layers, rows):
            if len(ids) == block.num_experts:
                continue
            excluded = torch.ones(block.num_experts, device=block.gate.weight.device, dtype=torch.bool)
            excluded[ids] = False

            def mask_logits(module, args, logits, excluded=excluded):
                return logits.masked_fill(excluded, -torch.inf)

            handle = block.gate.register_forward_hook(mask_logits)
            stack.callback(handle.remove)
        yield


def compact_in_place(model, rows):
    # This consumes the original model. Only gate rows are copied, retained MLPs are reused.
    layers = validate_rows(model, rows)
    keep = len(rows[0])
    for block, ids in zip(layers, rows):
        old = block.gate
        index = torch.tensor(ids, device=old.weight.device)
        gate = nn.Linear(old.in_features, keep, bias=False, device=old.weight.device, dtype=old.weight.dtype)
        gate.weight = nn.Parameter(old.weight.index_select(0, index), requires_grad=old.weight.requires_grad)
        block.gate = gate
        block.experts = nn.ModuleList([block.experts[i] for i in ids])
        block.num_experts = keep
    model.num_experts = model.config.num_experts = keep
    model.eval()
