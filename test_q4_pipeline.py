"""Offline launcher/worker regressions and an optional real native integration test."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gguf_export
import hf_jobs
import run


class TestExecutionModes(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def config(self, **values):
        file = self.root / 'cfg.json'
        file.write_text(json.dumps({'mode': 'hf', 'tags': ['python'], 'output_repo': 'example/moediet', **values}))
        return file

    def test_dry_run_does_not_submit_download_or_build(self):
        with patch('hf_jobs.submit') as submit, patch('run.execute') as execute, contextlib.redirect_stdout(io.StringIO()):
            run.main(['--config', str(self.config()), '--tags', 'cpp', '--dry-run'])
            run.main(['--config', str(run.HERE / 'config.local.json'), '--dry-run'])
            submit.assert_not_called()
            execute.assert_not_called()
        for invalid in ({'quantization': 'Q8_0'}, {'HF_TOKEN': 'forbidden'}, {'model': '/local/weights'},
                        {'build': {'eval_tokens': -1}}, {'output_repo': run.DEFAULT_MODEL}):
            with self.assertRaises(ValueError):
                run.load_config(self.config(**invalid))

    def test_private_pinned_job_submission(self):
        cfg = run.load_config(self.config())
        cfg['corpus'] = str(self.root / 'custom.json')
        (self.root / 'custom.json').write_text('{"sample": true}')
        api = Mock()
        api.model_info.side_effect = [SimpleNamespace(sha='a' * 40), SimpleNamespace(private=True)]
        metadata = self.root / 'metadata.json'
        metadata.write_text(json.dumps({'model_type': 'olmoe', 'num_hidden_layers': 2,
                'num_experts': 6, 'num_experts_per_tok': 2, 'norm_topk_prob': False, 'max_position_embeddings': 512}))
        def store_code(**kw):
            stage = Path(kw['folder_path'])
            self.assertEqual({p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()},
                             {*hf_jobs.SOURCES, 'corpus.json', 'request.json'})
            payload = json.loads((stage / 'request.json').read_text())
            self.assertEqual((payload['build']['revision'], payload['corpus']), ('a' * 40, 'corpus.json'))
            self.assertNotIn('canary-token', (stage / 'request.json').read_text())
            self.assertEqual(payload['architecture']['model_type'], 'olmoe')
            return SimpleNamespace(oid='b' * 40)
        api.upload_folder.side_effect = store_code
        with patch('hf_jobs.HfApi', return_value=api), patch('hf_jobs.get_token', return_value='canary-token'), patch(
                'hf_jobs.hf_hub_download', return_value=str(metadata)):
            hf_jobs.submit(cfg)
        spec = api.run_job.call_args.kwargs
        self.assertEqual(spec['env']['MOEDIET_COMMIT'], 'b' * 40)
        self.assertEqual(spec['secrets']['HF_TOKEN'], 'canary-token')
        self.assertNotIn('canary-token', str(spec['command']) + str(spec['env']))
        compile(spec['command'][2], '<hf-bootstrap>', 'exec')
        api.reset_mock()
        api.model_info.side_effect = [SimpleNamespace(sha='a' * 40), SimpleNamespace(private=False)]
        with patch('hf_jobs.HfApi', return_value=api), patch('hf_jobs.get_token', return_value='canary-token'), patch(
                'hf_jobs.hf_hub_download', return_value=str(metadata)):
            with self.assertRaisesRegex(ValueError, 'pubblica'):
                hf_jobs.submit(cfg)
        api.upload_folder.assert_not_called()
        api.run_job.assert_not_called()

    def test_fallback_and_native_failure_preserve_previous_outputs(self):
        out = self.root / 'run'
        out.mkdir()
        r = out / 'report.json'
        r.write_text(json.dumps({'status': 'unchanged_no_candidate'}))
        cfg = {'model': 'example/model', 'build': {}, 'tags': ['python'], 'quantization': 'Q4_K_M'}
        saved = Mock()
        with patch('run.subprocess.run'), patch('gguf_export.get_tools') as tools:
            run.execute(cfg, out, saved)
        saved.assert_called_once_with(out, '')
        tools.assert_not_called()
        self.assertFalse((out / 'gguf').exists())
        r.write_text(json.dumps({'status': 'pruned', 'checkpoint': {'parameters': 10}, 'retained_per_layer': 2,
                                 'checkpoint_audit': {'passed': True}, 'routing': {'num_layers': 1,
                                 'experts_per_layer': 4, 'experts_per_token': 2}}))
        (out / 'checkpoint').mkdir()
        (out / 'checkpoint/config.json').write_text(json.dumps({'model_type': 'olmoe', 'num_experts': 2,
                'num_experts_per_tok': 2, 'num_hidden_layers': 1, 'num_attention_heads': 4}))
        saved.reset_mock()
        fake_tools = SimpleNamespace(source=self.root, quantize=self.root / 'missing', cli=self.root / 'unused')
        # First subprocess is the successful build, second is the failing converter.
        with patch('run.subprocess.run', side_effect=[None, OSError('converter failed')]), patch(
                'gguf_export.get_tools', return_value=fake_tools):
            with self.assertRaisesRegex(OSError, 'converter failed'):
                run.execute(cfg, out, saved)
        saved.assert_called_once_with(out, '')
        self.assertTrue((out / 'checkpoint/config.json').exists())
        self.assertFalse((out / 'gguf/Q4_K_M').exists())
        self.assertEqual(list((out / 'gguf').iterdir()), [])

    def test_worker_persists_run_and_q4_separately(self):
        code = self.root / 'code'
        code.mkdir()
        (code / 'request.json').write_text(json.dumps({'model': run.DEFAULT_MODEL, 'tags': ['python'],
                                                     'build': {}, 'quantization': 'Q4_K_M'}))
        paths = []
        def fake_execute(cfg, out, publish):
            out.mkdir()
            (out / 'report.json').write_text('{}')
            publish(out, '')
            gguf = out / 'gguf' / 'Q4_K_M'
            gguf.mkdir(parents=True)
            (gguf / 'model-Q4_K_M.gguf').write_bytes(b'example transport payload')
            publish(gguf, gguf.relative_to(out).as_posix())
        def fake_upload(**kwargs):
            paths.append(kwargs['path_in_repo'])
            shutil.copytree(kwargs['folder_path'], self.root / kwargs['path_in_repo'], dirs_exist_ok=True)
            return SimpleNamespace(oid='saved')
        api = Mock()
        api.upload_folder.side_effect = fake_upload
        env = {'HF_TOKEN': 'test', 'MOEDIET_REPO': 'example/moediet', 'JOB_ID': 'job-123'}
        with patch.object(hf_jobs, 'HERE', code), patch('hf_jobs.HfApi', return_value=api), patch.dict(os.environ, env), patch(
                'run.execute', side_effect=fake_execute):
            hf_jobs.worker()
        self.assertEqual(paths, ['runs/job-123', 'runs/job-123/gguf/Q4_K_M'])
        self.assertTrue((self.root / 'runs/job-123/gguf/Q4_K_M/model-Q4_K_M.gguf').is_file())

    def test_environment_variable_expansion_in_config(self):
        env = {
            'TEST_OUT_DIR': str(self.root / 'env_out'),
            'TEST_LLAMA_DIR': str(self.root / 'fake_llama'),
            'TEST_BIN_DIR': str(self.root / 'fake_bin'),
        }
        cfg_file = self.root / 'env_cfg.json'
        cfg_file.write_text(json.dumps({
            'mode': 'local',
            'tags': ['python'],
            'out': '$TEST_OUT_DIR',
            'llama_dir': '$TEST_LLAMA_DIR',
            'bin_dir': '$TEST_BIN_DIR',
        }))
        with patch.dict(os.environ, env):
            parsed = run.load_config(cfg_file)
            self.assertEqual(parsed['out'], str((self.root / 'env_out').resolve()))
            self.assertEqual(parsed['llama_dir'], str((self.root / 'fake_llama').resolve()))
            self.assertEqual(parsed['bin_dir'], str((self.root / 'fake_bin').resolve()))


@unittest.skipUnless(os.environ.get('MOEDIET_TEST_TOKENIZER'), 'Richiede il tokenizer locale AllenAI 0125')
class TestNativeGGUF(unittest.TestCase):
    def test_both_formats_in_llama_cpp(self):
        import torch
        from transformers import AutoConfig, AutoTokenizer, OlmoeForCausalLM
        tokenizer_dir = os.environ['MOEDIET_TEST_TOKENIZER']
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source, out = work / 'original', work / 'result'
            cfg = AutoConfig.from_pretrained(tokenizer_dir, local_files_only=True)
            cfg.hidden_size = cfg.intermediate_size = 256
            cfg.num_hidden_layers = cfg.num_experts_per_tok = 2
            cfg.num_attention_heads = cfg.num_key_value_heads = 4
            cfg.num_experts, cfg.max_position_embeddings = 6, 512
            torch.manual_seed(719)
            OlmoeForCausalLM(cfg).save_pretrained(source, safe_serialization=True)
            AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True).save_pretrained(source)
            profile = {'mode': 'local', 'model': 'original', 'tags': ['python', 'c++', 'devops'],
                       'out': 'result', 'quantization': 'Q4_K_M', 'llama_dir': os.environ.get('MOEDIET_TEST_LLAMA_DIR'),
                       'build': {'device': 'cpu', 'max_kl': 5, 'eval_tokens': 2, 'budgets': [3]}}
            config_file = work / 'local.json'
            config_file.write_text(json.dumps(profile))
            with patch.dict(os.environ, {'OMP_NUM_THREADS': '2', 'HF_HUB_OFFLINE': '1'}):
                run.main(['--config', str(config_file)])
                gguf_export.export_run(out, 'Q4_0', profile['llama_dir'])
            for qtype in gguf_export.QTYPES:
                folder = out / 'gguf' / qtype
                meta = json.loads((folder / 'conversion.json').read_text())
                gguf = folder / meta['file']
                self.assertEqual(meta['experts_per_layer'], 3)
                self.assertEqual(meta['experts_per_token'], 2)
                self.assertGreater(meta['smoke']['decode_steps'], 0)
                self.assertGreater(meta['expert_tensor_types'].get('Q4_K' if qtype == 'Q4_K_M' else qtype, 0), 0)
                self.assertEqual(meta['sha256'], gguf_export.sha256(gguf))
                self.assertFalse(list(out.rglob('*f16.gguf')))
                with self.assertRaisesRegex(ValueError, 'già'):
                    gguf_export.export_run(out, qtype, profile['llama_dir'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
