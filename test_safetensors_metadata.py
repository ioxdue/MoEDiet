"""Offline tests for header-only safetensors inventory."""

import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import run
from safetensors_metadata import inspect_hub_safetensors, inspect_safetensors


def write_shard(path, tensors):
    header = json.dumps(tensors, separators=(',', ':')).encode()
    header += b' ' * (-len(header) % 8)
    with path.open('wb') as stream:
        stream.write(struct.pack('<Q', len(header)))
        stream.write(header)
        stream.write(b'\0' * max((item['data_offsets'][1] for item in tensors.values()), default=0))


def shard_bytes(tensors):
    header = json.dumps(tensors, separators=(',', ':')).encode()
    header += b' ' * (-len(header) % 8)
    payload_size = max((item['data_offsets'][1] for item in tensors.values()), default=0)
    return struct.pack('<Q', len(header)) + header + b'\0' * payload_size


class FakeResponse:
    def __init__(self, status_code, body, headers=None):
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        self.closed = False

    def iter_content(self, chunk_size):
        yield self.body

    def close(self):
        self.closed = True


class TestSafeTensorsMetadata(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_indexed_checkpoint_inventory_uses_headers_only(self):
        first = {'model.layers.0.gate.weight': {'dtype': 'F32', 'shape': [2, 3], 'data_offsets': [0, 24]}}
        second = {'model.layers.0.experts.weight': {'dtype': 'BF16', 'shape': [2, 2], 'data_offsets': [0, 8]}}
        write_shard(self.root / 'model-00001-of-00002.safetensors', first)
        write_shard(self.root / 'model-00002-of-00002.safetensors', second)
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({
            'weight_map': {
                'model.layers.0.gate.weight': 'model-00001-of-00002.safetensors',
                'model.layers.0.experts.weight': 'model-00002-of-00002.safetensors',
            },
        }))

        result = inspect_safetensors(self.root)

        self.assertEqual(result['tensor_count'], 2)
        self.assertEqual(result['payload_bytes'], 32)
        self.assertEqual(result['tensors']['model.layers.0.experts.weight']['shape'], [2, 2])
        self.assertFalse(result['payloads_read'])

    def test_index_mismatch_is_rejected(self):
        write_shard(self.root / 'model.safetensors', {
            'actual': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]},
        })
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({
            'weight_map': {'expected': 'model.safetensors'},
        }))

        with self.assertRaisesRegex(ValueError, 'Indice safetensors'):
            inspect_safetensors(self.root)

    def test_invalid_offsets_are_rejected(self):
        write_shard(self.root / 'model.safetensors', {
            'invalid': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 4]},
        })

        with self.assertRaisesRegex(ValueError, 'payload discordanti'):
            inspect_safetensors(self.root)

    def test_local_cli_inspection_does_not_run_preflight_or_build(self):
        write_shard(self.root / 'model.safetensors', {
            'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]},
        })
        config = self.root / 'config.json'
        config.write_text(json.dumps({
            'mode': 'local', 'model': str(self.root), 'out': str(self.root / 'out'),
            'tags': ['python'], 'quantization': None,
        }))

        with patch('run.execute') as execute, patch('preflight.inspect') as preflight, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            run.main(['--config', str(config), '--inspect-safetensors'])

        result = json.loads(output.getvalue())
        self.assertEqual(result['tensor_count'], 1)
        self.assertFalse(result['payloads_read'])
        execute.assert_not_called()
        preflight.assert_not_called()

    def test_hub_inventory_reads_config_index_and_headers_only(self):
        first = {'model.layers.0.gate.weight': {'dtype': 'F32', 'shape': [2, 3], 'data_offsets': [0, 24]}}
        second = {'model.layers.0.experts.weight': {'dtype': 'BF16', 'shape': [2, 2], 'data_offsets': [0, 8]}}
        blobs = {
            'config.json': b'{"model_type":"qwen4_exp"}',
            'model.safetensors.index.json': json.dumps({'weight_map': {
                'model.layers.0.gate.weight': 'model-00001-of-00002.safetensors',
                'model.layers.0.experts.weight': 'model-00002-of-00002.safetensors',
            }}).encode(),
            'model-00001-of-00002.safetensors': shard_bytes(first),
            'model-00002-of-00002.safetensors': shard_bytes(second),
        }
        source = SimpleNamespace(sha='a' * 40, siblings=[
            SimpleNamespace(rfilename=name, size=len(body)) for name, body in blobs.items()
        ])
        requested_ranges = []

        def fake_get(url, headers, stream, timeout):
            name = url.split('/')[-1].split('?')[0]
            body = blobs[name]
            if 'Range' not in headers:
                return FakeResponse(200, body, {'Content-Length': str(len(body))})
            start, end = map(int, headers['Range'][6:].split('-'))
            requested_ranges.append((name, start, end))
            return FakeResponse(206, body[start:end + 1], {
                'Content-Range': f'bytes {start}-{end}/{len(body)}',
            })

        with patch('huggingface_hub.HfApi') as api, \
                patch('huggingface_hub.hf_hub_url', side_effect=lambda repo, name, revision: f'https://hub/{name}?{revision}'), \
                patch('requests.get', side_effect=fake_get):
            api.return_value.model_info.return_value = source
            result = inspect_hub_safetensors('org/model', 'main')

        self.assertEqual(result['source_revision'], 'a' * 40)
        self.assertEqual(result['model_config']['model_type'], 'qwen4_exp')
        self.assertEqual(result['tensor_count'], 2)
        self.assertEqual(result['payload_bytes'], 32)
        self.assertFalse(result['payloads_read'])
        self.assertEqual(len(requested_ranges), 4)
        self.assertTrue(all((start == 0 and end == 7) or start == 8
                            for _, start, end in requested_ranges))

    def test_hub_inventory_rejects_server_that_ignores_range(self):
        body = shard_bytes({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}})
        siblings = [
            SimpleNamespace(rfilename='config.json', size=2),
            SimpleNamespace(rfilename='model.safetensors', size=len(body)),
        ]
        source = SimpleNamespace(sha='b' * 40, siblings=siblings)

        def fake_get(url, headers, stream, timeout):
            if 'Range' in headers:
                return FakeResponse(200, body)
            return FakeResponse(200, b'{}', {'Content-Length': '2'})

        with patch('huggingface_hub.HfApi') as api, \
                patch('huggingface_hub.hf_hub_url', side_effect=lambda repo, name, revision: f'https://hub/{name}'), \
                patch('requests.get', side_effect=fake_get):
            api.return_value.model_info.return_value = source
            with self.assertRaisesRegex(ValueError, 'HTTP 200'):
                inspect_hub_safetensors('org/model')

    def test_hub_inventory_rejects_incorrect_content_range(self):
        body = shard_bytes({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}})
        source = SimpleNamespace(sha='c' * 40, siblings=[
            SimpleNamespace(rfilename='config.json', size=2),
            SimpleNamespace(rfilename='model.safetensors', size=len(body)),
        ])

        def fake_get(url, headers, stream, timeout):
            if 'Range' in headers:
                return FakeResponse(206, body[:8], {'Content-Range': f'bytes 1-8/{len(body)}'})
            return FakeResponse(200, b'{}', {'Content-Length': '2'})

        with patch('huggingface_hub.HfApi') as api, \
                patch('huggingface_hub.hf_hub_url', side_effect=lambda repo, name, revision: f'https://hub/{name}'), \
                patch('requests.get', side_effect=fake_get):
            api.return_value.model_info.return_value = source
            with self.assertRaisesRegex(ValueError, 'Content-Range'):
                inspect_hub_safetensors('org/model')

    def test_hub_cli_inspection_skips_unsupported_profile_validation(self):
        config = self.root / 'config.json'
        config.write_text(json.dumps({
            'mode': 'hf', 'model': 'org/model', 'tags': ['python'], 'output_repo': 'org/output',
            'architecture_profile': 'architectures/qwen4_exp.json',
            'build': {'revision': 'a' * 40},
        }))

        with patch('safetensors_metadata.inspect_hub_safetensors',
                   return_value={'payloads_read': False}) as inspect, \
                contextlib.redirect_stdout(io.StringIO()):
            run.main(['--config', str(config), '--inspect-safetensors'])

        inspect.assert_called_once_with('org/model', 'a' * 40)


if __name__ == '__main__':
    unittest.main(verbosity=2)
