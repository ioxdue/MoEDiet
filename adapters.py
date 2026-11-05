"""Runtime implementation of the declarative softmax/top-k/ModuleList contract."""

from contextlib import ExitStack, contextmanager
import copy

import torch
from torch import nn
import transformers

from architecture import read_path, resolve_profile, topology, write_path


def tolerances(dtype):
    # Numerical equivalence limits, independent of the quality/KL threshold.
    if dtype == torch.bfloat16:
        return {'atol': 0.125, 'rtol': 0.001}
    if dtype == torch.float16:
        return {'atol': 0.02, 'rtol': 0.001}
    return {'atol': 0.0001, 'rtol': 0.00001}


class ModuleListAdapter:
    def __init__(self, spec):
        self.spec = spec
        self.layout = spec['layout']

    def check_runtime(self):
        expected = self.spec['transformers_version']
        if transformers.__version__ != expected:
            raise ValueError(f"Il profilo {self.spec['id']} richiede transformers=={expected}; installato {transformers.__version__}.")

    def topology(self, config):
        return topology(config, self.spec)

    def blocks(self, model):
        self.check_runtime()
        routing = self.topology(model.config)
        layers = read_path(model, self.layout['layers'])
        if len(layers) != read_path(model.config, self.spec['config']['layers']):
            raise ValueError('Numero di layer incompatibile con il profilo.')
        selected = [read_path(layers[i], self.layout['block']) for i in routing['layer_indices']]
        for block in selected:
            gate, experts = self.router(block), self.experts(block)
            if not (isinstance(gate, nn.Linear) and gate.bias is None and isinstance(experts, nn.ModuleList)
                    and read_path(block, self.layout['count']) == gate.out_features == len(experts) == routing['experts_per_layer']
                    and read_path(block, self.layout['top_k']) == routing['experts_per_token']
                    and type(read_path(block, self.layout['normalize'])) is bool
                    and read_path(block, self.layout['normalize']) == routing['norm_topk_prob']):
                raise ValueError('Layout del router/esperti incompatibile con il profilo ModuleList.')
            for shared in self.spec['shared_experts']:
                if not (isinstance(read_path(block, shared['module']), nn.Module)
                        and isinstance(read_path(block, shared['gate']), nn.Linear)):
                    raise ValueError('Shared expert incompatibile con il profilo.')
        return selected

    def router(self, block):
        return read_path(block, self.layout['router'])

    def experts(self, block):
        return read_path(block, self.layout['experts'])

    def weights(self, block, logits):
        probs = logits.float().softmax(-1)
        weights, chosen = probs.topk(read_path(block, self.layout['top_k']), dim=-1)
        if read_path(block, self.layout['normalize']):
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights, chosen

    def forward_backbone(self, model, ids):
        return read_path(model, self.layout['backbone'])(input_ids=ids, use_cache=False,
                                                        output_router_logits=False, return_dict=True)

    def router_logits(self, model, ids):
        layers = self.blocks(model)
        captured = [[] for _ in layers]
        with ExitStack() as stack:
            for i, block in enumerate(layers):
                def capture(module, args, output, i=i):
                    captured[i].append(output.detach())
                stack.callback(self.router(block).register_forward_hook(capture).remove)
            self.forward_backbone(model, ids)
        if any(len(values) != 1 for values in captured):
            raise ValueError('Ogni router deve essere eseguito esattamente una volta per sonda.')
        return [values[0] for values in captured]

    def validate_rows(self, model, rows):
        layers = self.blocks(model)
        routing = self.topology(model.config)
        top_k, total = routing['experts_per_token'], routing['experts_per_layer']
        if not isinstance(rows, list) or len(rows) != len(layers):
            raise ValueError('La maschera deve contenere una lista per ogni layer MoE, in ordine.')
        for row in rows:
            if (not isinstance(row, list) or not top_k <= len(row) <= total
                    or any(type(i) is not int or not 0 <= i < total for i in row)
                    or row != sorted(set(row)) or len(row) != len(rows[0])):
                raise ValueError('Maschera non valida: ID unici ordinati, budget uniforme >= top-k.')
        return layers

    @contextmanager
    def masked(self, model, rows):
        layers = self.validate_rows(model, rows)
        with ExitStack() as stack:
            for block, ids in zip(layers, rows):
                gate = self.router(block)
                if len(ids) == gate.out_features:
                    continue
                excluded = torch.ones(gate.out_features, device=gate.weight.device, dtype=torch.bool)
                excluded[ids] = False

                def mask_logits(module, args, logits, excluded=excluded):
                    return logits.masked_fill(excluded, -torch.inf)

                stack.callback(gate.register_forward_hook(mask_logits).remove)
            yield

    def compact(self, model, rows):
        layers = self.validate_rows(model, rows)
        keep = len(rows[0])
        for block, ids in zip(layers, rows):
            old = self.router(block)
            index = torch.tensor(ids, device=old.weight.device)
            gate = nn.Linear(old.in_features, keep, bias=False, device=old.weight.device, dtype=old.weight.dtype)
            gate.weight = nn.Parameter(old.weight.index_select(0, index), requires_grad=old.weight.requires_grad)
            write_path(block, self.layout['router'], gate)
            write_path(block, self.layout['experts'], nn.ModuleList([self.experts(block)[i] for i in ids]))
            write_path(block, self.layout['count'], keep)
        write_path(model.config, self.spec['config']['experts'], keep)
        if hasattr(model, 'num_experts'):
            model.num_experts = keep
        model.config.moediet_architecture = copy.deepcopy(self.spec)
        model.eval()

    @torch.inference_mode()
    def verify_contract(self, model, ids):
        """Check actual native MoE outputs against the declared routing mathematics."""
        layers = self.blocks(model)
        seen = [0] * len(layers)
        peak = [0.0]
        with ExitStack() as stack:
            for i, block in enumerate(layers):
                def check(module, args, output, i=i):
                    hidden = args[0]
                    flat = hidden.reshape(-1, hidden.shape[-1])
                    weights, chosen = self.weights(module, self.router(module)(flat))
                    weights = weights.to(flat.dtype)
                    result = torch.zeros_like(flat)
                    for expert_id in chosen.unique().tolist():
                        tokens, slots = torch.where(chosen == expert_id)
                        values = self.experts(module)[expert_id](flat[tokens]) * weights[tokens, slots, None]
                        result.index_add_(0, tokens, values.to(flat.dtype))
                    for shared in self.spec['shared_experts']:
                        result += read_path(module, shared['gate'])(flat).sigmoid() * read_path(module, shared['module'])(flat)
                    native = output[0] if isinstance(output, tuple) else output
                    result = result.reshape(hidden.shape)
                    if (native.shape != result.shape or not torch.isfinite(native).all()
                            or not torch.allclose(native, result, **tolerances(flat.dtype))):
                        raise ValueError(f'Il blocco MoE {i} non rispetta la matematica del profilo.')
                    peak[0] = max(peak[0], (native - result).abs().max().item())
                    seen[i] += 1
                stack.callback(block.register_forward_hook(check).remove)
            self.forward_backbone(model, ids)
        if seen != [1] * len(layers):
            raise ValueError('Verifica incompleta dei blocchi MoE.')
        return {'passed': True, 'tokens': ids.numel(), 'max_abs_delta': peak[0],
                'tolerance': tolerances(model.get_input_embeddings().weight.dtype)}


def adapter_for(model_or_config, requested=None):
    config = getattr(model_or_config, 'config', model_or_config)
    return ModuleListAdapter(resolve_profile(config, requested))
