"""Architecture-independent profiling and compatibility wrappers for adapters."""

import torch

from adapters import adapter_for
from architecture import read_path


def blocks(model):
    return adapter_for(model).blocks(model)


def encode(model, tokenizer, prompt, reserve=0):
    if tokenizer.chat_template:
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                                            add_generation_prompt=True, return_tensors='pt')
    else:
        ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    adapter = adapter_for(model)
    context = read_path(model.config, adapter.spec['config']['context'])
    if not ids.shape[1] or ids.shape[1] + reserve > context:
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

    adapter = adapter_for(model)
    layers = adapter.blocks(model)
    experts = adapter.topology(model.config)['experts_per_layer']
    means, counts = {}, {}
    for group, prompts in groups.items():
        sums = torch.zeros(len(layers), experts, dtype=torch.float64)
        counts[group] = 0
        for prompt in prompts:
            ids = encode(model, tokenizer, prompt)
            routers = adapter.router_logits(model, ids)
            count = ids.numel()
            for i, (block, logits) in enumerate(zip(layers, routers, strict=True)):
                if tuple(logits.shape) != (count, experts):
                    raise ValueError(f'Forma inattesa dei logit del router {i}: {tuple(logits.shape)}')
                weights, chosen = adapter.weights(block, logits)
                mass = torch.zeros_like(logits, dtype=torch.float32).scatter_(-1, chosen, weights)
                sums[i] += mass.sum(dim=0, dtype=torch.float64).cpu()
            counts[group] += count
        means[group] = sums / counts[group]
    target = torch.stack([means[tag] for tag in tags]).mean(0)
    score = target - penalty * means[CONTROL]
    if not torch.isfinite(score).all():
        raise ValueError('Lo score contiene valori non finiti.')
    return score, counts


def select_experts(score, keep):
    ranked = torch.argsort(score, dim=1, descending=True, stable=True)
    return ranked[:, :keep].sort(dim=1).values.tolist()


def validate_rows(model, rows):
    return adapter_for(model).validate_rows(model, rows)


def masked(model, rows):
    return adapter_for(model).masked(model, rows)


def compact_in_place(model, rows):
    return adapter_for(model).compact(model, rows)
