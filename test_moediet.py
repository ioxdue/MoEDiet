"""Offline integration tests on a seeded, randomly initialized Hugging Face OLMoE."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer, OlmoeConfig, OlmoeForCausalLM, PreTrainedTokenizerFast

from moediet import budgets_for, main, resolve_source, verify_manifest
from probes import CONTROL, make_probes
from routing import compact_in_place, continuation, encode, masked, profile, select_experts, validate_rows


def make_tiny(folder):
    torch.manual_seed(719)
    vocab = {word: i for i, word in enumerate(['<pad>', '<bos>', '<eos>', '<unk>'] + sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token='<bos>', eos_token='<eos>',
                                  pad_token='<pad>', unk_token='<unk>')
    tok.chat_template = "{{ bos_token }}{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    cfg = OlmoeConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, num_experts=6, num_experts_per_tok=2,
                      max_position_embeddings=512, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    model = OlmoeForCausalLM(cfg).eval()
    model.save_pretrained(folder, safe_serialization=True)
    tok.save_pretrained(folder)


class TestMoEDiet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.root = Path(cls.tempdir.name)
        cls.source = cls.root / 'source'
        make_tiny(cls.source)

    def load(self):
        model = AutoModelForCausalLM.from_pretrained(self.source, attn_implementation='eager').eval()
        return model, AutoTokenizer.from_pretrained(self.source)

    def build_args(self, out):
        return ['build', '--model', str(self.source), '--tags', 'python', 'c++', 'devops', '--device', 'cpu',
                '--eval-tokens', '4', '--out', str(out)]

    @torch.inference_mode()
    def test_profile_matches_tokenwise_native_routing(self):
        model, tok = self.load()
        corpus = {'python': ['short', 'two samples with different lengths'], 'sql': ['aggregate'], CONTROL: ['history']}
        for renorm in (False, True):
            model.config.norm_topk_prob = renorm
            for layer in model.model.layers:
                layer.mlp.norm_topk_prob = renorm
            actual, counts = profile(model, tok, corpus, ['python', 'sql'], 0.5)
            means = {}
            for group, texts in corpus.items():
                sums = torch.zeros_like(actual)
                tokens = 0
                for text in texts:
                    ids = encode(model, tok, text)
                    tokens += ids.numel()
                    routers = model.model(input_ids=ids, output_router_logits=True, use_cache=False).router_logits
                    for layer, logits in enumerate(routers):
                        for row in logits:
                            probs = row.softmax(0)
                            ids_top = probs.argsort(descending=True)[:2]
                            values = probs[ids_top]
                            if renorm:
                                values = values / values.sum()
                            sums[layer, ids_top] += values.double()
                self.assertEqual(counts[group], tokens)
                means[group] = sums / tokens
            torch.testing.assert_close(actual, (means['python'] + means['sql']) / 2 - 0.5 * means[CONTROL])

    @torch.inference_mode()
    def test_mask_dispatch_cleanup_compact_and_cached_decode(self):
        model, tok = self.load()
        rows = [[0, 3, 4], [1, 2, 5]]
        ids = encode(model, tok, 'Use an iterator safely.')
        full = model(input_ids=ids).logits
        with self.assertRaisesRegex(RuntimeError, 'cleanup'):
            with masked(model, rows):
                raise RuntimeError('cleanup')
        torch.testing.assert_close(model(input_ids=ids).logits, full, rtol=0, atol=0)
        traffic = [[0] * 6 for _ in rows]
        with contextlib.ExitStack() as stack, masked(model, rows):
            for layer_no, layer in enumerate(model.model.layers):
                for exp_no, expert in enumerate(layer.mlp.experts):
                    def record(mod, args, layer_no=layer_no, exp_no=exp_no):
                        traffic[layer_no][exp_no] += args[0].shape[0]
                    stack.callback(expert.register_forward_pre_hook(record).remove)
            expected = model(input_ids=ids).logits
        for kept, row in zip(rows, traffic):
            self.assertEqual(sum(row), ids.numel() * 2)
            self.assertTrue(all(count == 0 for i, count in enumerate(row) if i not in kept))
        with masked(model, rows):
            generated = continuation(model, tok, ids, 5)
        compact_in_place(model, rows)
        path = self.root / 'roundtrip'
        model.save_pretrained(path, safe_serialization=True)
        reloaded = AutoModelForCausalLM.from_pretrained(path, attn_implementation='eager').eval()
        self.assertEqual((reloaded.config.num_experts, reloaded.config.num_experts_per_tok), (3, 2))
        torch.testing.assert_close(reloaded(input_ids=ids).logits, expected, atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.equal(continuation(reloaded, tok, ids, 5), generated))

    def test_cli_build_and_both_generate_paths(self):
        out = self.root / 'good'
        main(self.build_args(out) + ['--budgets', '3', '4'])
        stats = json.loads((out / 'report.json').read_text())
        self.assertEqual((stats['status'], stats['retained_per_layer']), ('pruned', 3))
        self.assertTrue(stats['checkpoint_audit']['passed'])
        self.assertLess(stats['checkpoint']['parameters'], stats['full_model']['parameters'])
        self.assertLess(stats['checkpoint']['safetensors_bytes'], (self.source / 'model.safetensors').stat().st_size)
        self.assertEqual(set(stats['audit']['groups']), {'python', 'c++', 'devops', CONTROL})
        args = ['generate', '--prompt', 'How does a closure work?', '--max-new-tokens', '4', '--device', 'cpu']
        texts = []
        for opt, path in (('--mask', out / 'mask.json'), ('--model', out / 'checkpoint')):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main(args + [opt, str(path)])
            texts.append(output.getvalue())
        self.assertEqual(*texts)
        with self.assertRaisesRegex(ValueError, 'esiste già'):
            main(self.build_args(out))
        model, _ = self.load()
        meta = json.loads((out / 'mask.json').read_text())
        _, source = resolve_source(str(self.source), None, fingerprint=True)
        verify_manifest(meta, model, source)
        with self.assertRaises(ValueError):
            verify_manifest(meta, model, {**source, 'sha256': 'different'})

    def test_fallbacks_and_mask_only(self):
        for name, flags, expected in (
                ('strict', ['--max-kl', '0'], 'unchanged_no_candidate'),
                ('audit', [], 'unchanged_audit_failed'), ('mask', ['--mask-only'], 'mask_only')):
            out = self.root / name
            # Exercise the final gate failure deterministically, independently of the random fixture's quality.
            guard = patch('moediet.passes', side_effect=[True, False]) if name == 'audit' else contextlib.nullcontext()
            with guard:
                main(self.build_args(out) + ['--budgets', '2'] + flags)
            stats = json.loads((out / 'report.json').read_text())
            self.assertEqual(stats['status'], expected)
            meta = json.loads((out / 'mask.json').read_text())
            self.assertEqual(len(meta['expert_mask'][0]), 2 if name == 'mask' else 6)
            self.assertFalse((out / 'checkpoint').exists())

    def test_invalid_budget_mask_and_overlap(self):
        self.assertEqual(budgets_for(64, 8, None), [8, 16, 32, 64])
        with self.assertRaises(ValueError):
            budgets_for(64, 8, [2])
        model, _ = self.load()
        for invalid in ([[0, 1]], [[0, 0], [0, 1]], [[0, 6], [0, 1]], [[0], [1]], [[0, 1, 2], [0, 1]]):
            with self.assertRaises(ValueError):
                validate_rows(model, invalid)
        self.assertEqual(select_experts(torch.zeros(2, 6), 3), [[0, 1, 2]] * 2)
        corpus = make_probes(['python'])
        corpus['audit']['python'][0] = corpus['profile']['python'][0]
        path = self.root / 'overlap.json'
        path.write_text(json.dumps(corpus))
        with self.assertRaisesRegex(ValueError, 'duplicata'):
            make_probes(['python'], path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
