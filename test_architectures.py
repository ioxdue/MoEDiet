"""Real tiny-model adapter round trips plus offline schema and HF boundary tests."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2MoeConfig, Qwen2MoeForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from adapters import adapter_for
from architecture import budgets_for, load_profile, validate_checkpoint_config
import hf_jobs
from moediet import main, verify_manifest, resolve_source
from probes import CONTROL
from routing import encode, masked, profile, compact_in_place, continuation
import run
from test_moediet import make_tiny


class TestProfiles(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def write_config(self, values):
        file = self.root / 'experiment.json'
        file.write_text(json.dumps(values))
        return file

    def v2(self):
        return {'schema_version': 2, 'mode': 'hf', 'model': {'id': 'example/qwen', 'revision': 'a' * 40},
                'architecture_profile': 'qwen3_moe', 'output_repo': 'example/result', 'tags': ['python'],
                'pruning': {'keep_fractions': [0.75, 0.875], 'preserve_top_k': True, 'shared_experts': 'preserve'},
                'validation': {'max_kl': 0.1, 'eval_tokens': 32, 'require_control': True}, 'export': {'format': 'hf'}}

    def test_v2_and_relative_custom_profile(self):
        spec = load_profile('qwen3_moe')
        (self.root / 'custom.json').write_text(json.dumps(spec))
        raw = self.v2()
        raw['architecture_profile'] = 'custom.json'
        cfg = run.load_config(self.write_config(raw))
        self.assertEqual(cfg['architecture'], spec)
        self.assertIsNone(cfg['quantization'])
        self.assertEqual(cfg['build']['revision'], 'a' * 40)
        command = run.build_command(cfg, self.root / 'result')
        self.assertEqual(load_profile(command[command.index('--architecture-profile') + 1]), spec)
        self.assertIn('--keep-fractions', command)

    def test_invalid_or_ambiguous_schema(self):
        for changes in ({'schema_version': True}, {'schema_version': 3},
                        {'pruning': {'keep_fractions': [0.75], 'budgets': [48]}},
                        {'pruning': {'keep_fractions': [False]}}, {'pruning': {'keep_fractions': [float('nan')]}},
                        {'pruning': {'preserve_top_k': False}}, {'validation': {'require_control': False}},
                        {'build': {'revision': 'different'}}, {'export': {'format': 'gguf'}},
                        {'export': {'format': 'hf', 'quantization': 'Q4_0'}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                run.load_config(self.write_config({**self.v2(), **changes}))
        spec = load_profile('olmoe')
        for change in ({'implementation': 'eval'}, {'layout': {**spec['layout'], 'router': '__import__(os)'}},
                       {'unknown': True}, {'transformers_version': '*'}):
            with self.assertRaises(ValueError):
                load_profile({**spec, **change})

    def test_fraction_resolution_and_unsupported_qwen4(self):
        fractions = [0.75, 0.875, 0.9375, 0.96875]
        self.assertEqual(budgets_for(64, 8, fractions=fractions), [48, 56, 60, 62, 64])
        self.assertEqual(budgets_for(512, 10, fractions=fractions), [384, 448, 480, 496, 512])
        self.assertEqual(budgets_for(6, 2, fractions=[0.51]), [4, 6])
        with self.assertRaises(ValueError):
            budgets_for(64, 8, fractions=[0.01])
        with self.assertRaisesRegex(ValueError, 'qwen4_exp'):
            validate_checkpoint_config({'model_type': 'qwen4_exp'})

    def test_preflight_and_submission_reject_before_writes(self):
        raw = {'model_type': 'qwen4_exp'}
        file = self.root / 'model.json'
        file.write_text(json.dumps(raw))
        cfg = {'model': 'Qwen/Qwen3.8-Flash-Next', 'mode': 'hf', 'build': {}, 'quantization': None,
               'output_repo': 'example/result', 'tags': ['python'], 'flavor': 'l4x1', 'timeout': '90m'}
        api = Mock()
        api.model_info.return_value = SimpleNamespace(sha='a' * 40)
        with patch('hf_jobs.get_token', return_value='test'), patch('hf_jobs.HfApi', return_value=api), patch(
                'hf_jobs.hf_hub_download', return_value=str(file)):
            with self.assertRaisesRegex(ValueError, 'qwen4_exp'):
                hf_jobs.submit(cfg)
        api.create_repo.assert_not_called()
        api.upload_folder.assert_not_called()
        api.run_job.assert_not_called()

    def test_launcher_import_does_not_require_torch(self):
        subprocess.run([sys.executable, '-c', "import run, hf_jobs; import sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"], check=True)

    def test_local_metadata_preflight_without_loading_weights(self):
        raw = self.v2()
        raw.update(mode='local', model={'id': './source'}, out='./output')
        raw.pop('output_repo')
        source = self.root / 'source'
        source.mkdir()
        (source / 'config.json').write_text(json.dumps({'model_type': 'qwen3_moe', 'num_hidden_layers': 4,
                'num_experts': 64, 'num_experts_per_tok': 8, 'norm_topk_prob': True, 'max_position_embeddings': 512,
                'mlp_only_layers': [], 'decoder_sparse_step': 2}))
        with patch('run.execute') as execute, patch('hf_jobs.submit') as submit, contextlib.redirect_stdout(io.StringIO()) as output:
            run.main(['--config', str(self.write_config(raw)), '--preflight'])
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['budgets'], [48, 56, 64])
        self.assertEqual(plan['routing']['layer_indices'], [1, 3])
        self.assertFalse(plan['weights_loaded'])
        execute.assert_not_called()
        submit.assert_not_called()


class TestQwenAdapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name)
        make_tiny(cls.root / 'tokenizer')
        cls.tok = AutoTokenizer.from_pretrained(cls.root / 'tokenizer')
        cls.sources = {}
        for name, config_class, model_class in [('qwen2_moe', Qwen2MoeConfig, Qwen2MoeForCausalLM),
                                                 ('qwen3_moe', Qwen3MoeConfig, Qwen3MoeForCausalLM)]:
            torch.manual_seed(91)
            cfg = config_class(vocab_size=len(cls.tok), hidden_size=32, intermediate_size=48,
                    moe_intermediate_size=24, shared_expert_intermediate_size=24, num_hidden_layers=4,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_experts=6,
                    num_experts_per_tok=2, max_position_embeddings=512, decoder_sparse_step=2,
                    norm_topk_prob=True, pad_token_id=0, bos_token_id=1, eos_token_id=2)
            source = cls.root / name
            model_class(cfg).save_pretrained(source, safe_serialization=True)
            cls.tok.save_pretrained(source)
            cls.sources[name] = source

    @torch.inference_mode()
    def test_native_profile_mask_compaction_reload_and_shared_experts(self):
        for name, source in self.sources.items():
            with self.subTest(model=name):
                model = AutoModelForCausalLM.from_pretrained(source, attn_implementation='eager').eval()
                adapter = adapter_for(model)
                self.assertEqual(adapter.topology(model.config)['layer_indices'], [1, 3])
                ids = encode(model, self.tok, 'safe code')
                self.assertTrue(adapter.verify_contract(model, ids[:, :8])['passed'])
                original = model(input_ids=ids).logits
                with masked(model, [list(range(6))] * 2):
                    torch.testing.assert_close(model(input_ids=ids).logits, original, atol=0, rtol=0)
                score, counts = profile(model, self.tok, {'python': ['a'], CONTROL: ['b']}, ['python'], 0)
                routers = model.model(input_ids=encode(model, self.tok, 'a'), output_router_logits=True).router_logits
                for i, logits in enumerate(routers):
                    weights, chosen = logits.float().softmax(-1).topk(2, dim=-1)
                    weights /= weights.sum(-1, keepdim=True)
                    expected = torch.zeros_like(logits).scatter_(-1, chosen, weights).double().mean(0)
                    torch.testing.assert_close(score[i], expected)
                self.assertEqual(counts['python'], encode(model, self.tok, 'a').numel())
                dense_before = {k: v.clone() for k, v in model.model.layers[0].mlp.state_dict().items()}
                shared_before = {k: v.clone() for k, v in model.state_dict().items() if '.shared_expert' in k}
                rows = [[0, 2, 4], [1, 3, 5]]
                with self.assertRaisesRegex(RuntimeError, 'cleanup'):
                    with masked(model, rows):
                        raise RuntimeError('cleanup')
                torch.testing.assert_close(model(input_ids=ids).logits, original, atol=0, rtol=0)
                with masked(model, rows):
                    expected = model(input_ids=ids).logits
                    sequence = continuation(model, self.tok, ids, 3)
                compact_in_place(model, rows)
                for k, v in dense_before.items():
                    torch.testing.assert_close(model.model.layers[0].mlp.state_dict()[k], v, atol=0, rtol=0)
                for k, v in shared_before.items():
                    torch.testing.assert_close(model.state_dict()[k], v, atol=0, rtol=0)
                folder = self.root / (name + '-compact')
                model.save_pretrained(folder, safe_serialization=True)
                reloaded = AutoModelForCausalLM.from_pretrained(folder, attn_implementation='eager').eval()
                self.assertEqual(reloaded.config.num_experts, 3)
                torch.testing.assert_close(reloaded(input_ids=ids).logits, expected, atol=1e-6, rtol=1e-5)
                self.assertTrue(torch.equal(continuation(reloaded, self.tok, ids, 3), sequence))

    def test_bad_shared_expert_profile_fails_contract(self):
        model = AutoModelForCausalLM.from_pretrained(self.sources['qwen2_moe'], attn_implementation='eager').eval()
        spec = load_profile('qwen2_moe')
        spec['shared_experts'] = []
        bad = adapter_for(model, spec)
        with self.assertRaisesRegex(ValueError, 'matematica'):
            bad.verify_contract(model, encode(model, self.tok, 'a')[:, :8])
        self.assertTrue(all(not block._forward_hooks for block in bad.blocks(model)))

    def test_cli_roundtrip_and_checkpoint_equivalence_gate(self):
        source = self.sources['qwen3_moe']
        for suffix, reject in [('good', False), ('rejected', True)]:
            out = self.root / ('cli-' + suffix)
            guard = patch('moediet.equivalent', return_value={'passed': False}) if reject else contextlib.nullcontext()
            with guard:
                main(['build', '--model', str(source), '--architecture-profile', 'qwen3_moe', '--tags', 'python',
                      '--device', 'cpu', '--out', str(out), '--keep-fractions', '0.5', '--eval-tokens', '2'])
            report = json.loads((out / 'report.json').read_text())
            self.assertEqual(report['status'], 'mask_only_checkpoint_audit_failed' if reject else 'pruned')
            self.assertEqual((out / 'checkpoint').exists(), not reject)
            if reject:
                continue
            self.assertTrue(report['checkpoint_audit']['mask_equivalence']['passed'])
            meta = json.loads((out / 'mask.json').read_text())
            model = AutoModelForCausalLM.from_pretrained(source).eval()
            _, identity = resolve_source(str(source), None, True)
            verify_manifest(meta, model, identity)
            meta['architecture_sha256'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'architettura'):
                verify_manifest(meta, model, identity)
            texts = []
            for flag, value in [('--mask', out / 'mask.json'), ('--model', out / 'checkpoint')]:
                with contextlib.redirect_stdout(io.StringIO()) as captured:
                    main(['generate', flag, str(value), '--device', 'cpu', '--prompt', 'a', '--max-new-tokens', '2'])
                texts.append(captured.getvalue())
            self.assertEqual(*texts)

    def test_old_olmoe_manifest_remains_usable(self):
        source_path = self.root / 'tokenizer'
        _, source = resolve_source(str(source_path), None, True)
        meta = {'format': 'moediet.olmoe.mask.v1', 'source': source, 'profile_dtype': 'float32',
                'routing': {'num_layers': 2, 'experts_per_layer': 6, 'experts_per_token': 2, 'norm_topk_prob': False},
                'retained_per_layer': 3, 'expert_mask': [[0, 2, 4], [1, 3, 5]], 'target_tags': ['python']}
        path = self.root / 'legacy-mask.json'
        path.write_text(json.dumps(meta))
        model = AutoModelForCausalLM.from_pretrained(source_path).eval()
        verify_manifest(meta, model, source)
        with contextlib.redirect_stdout(io.StringIO()):
            main(['generate', '--mask', str(path), '--prompt', 'a', '--max-new-tokens', '2', '--device', 'cpu'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
