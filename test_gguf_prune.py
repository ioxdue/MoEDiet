"""Small real GGUF containers; these tests do not run a language model."""

from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import gguf_prune as prune

try:
    gguf = prune.gguf_module()
except ImportError:
    gguf = None


def fixture(directory, split=True, alignment=32, extra_tensor=None, extra_metadata=None):
    """Upstream writer generates independent packed byte patterns per expert."""
    directory.mkdir()
    class Writer(gguf.GGUFWriter):
        def add_shard_kv_data(self):
            super().add_shard_kv_data()
            if alignment != 32:
                for fields in self.kv_data:
                    fields['general.alignment'] = gguf.GGUFValue(alignment, gguf.GGUFValueType.UINT32)

    writer = Writer(directory / 'source.gguf', 'qwen4exp', split_max_tensors=10 if split else 0,
                    small_first_shard=split)
    writer.add_block_count(2)
    writer.add_expert_count(6)
    writer.add_expert_used_count(2)
    writer.add_embedding_length(256)
    writer.add_expert_feed_forward_length(512)
    writer.add_name('fixture: structural validation only')
    writer.add_string('tokenizer.chat_template', '{{ messages }}')
    writer.add_array('fixture.strings', ['alpha', 'βeta'])
    writer.add_key_value('fixture.uint64s', [0, 2**40], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT64)
    writer.add_key_value('fixture.floats', [0.25, -2.5], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.FLOAT32)
    if alignment != 32:
        writer.add_custom_alignment(alignment)
    for key, value in (extra_metadata or {}).items():
        writer.add_uint32(key, value)
    types = ['Q5_0', 'IQ3_XXS', 'IQ4_XS', 'IQ4_NL', 'Q2_0', 'IQ2_S']
    raw = {}

    def add(name, data, kind=None):
        raw[name] = data.tobytes()
        writer.add_tensor(name, data, raw_dtype=kind)

    for layer in range(2):
        for j, axis in enumerate(('gate', 'up', 'down')):
            kind = getattr(gguf.GGMLQuantizationType, types[layer * 3 + j])
            block, size = gguf.GGML_QUANT_SIZES[kind]
            rows, columns = (256, 512) if axis == 'down' else (512, 256)
            data = np.empty((6, rows, columns // block * size), dtype=np.uint8)
            for expert in range(6):
                data[expert].fill(13 + layer * 40 + j * 8 + expert)
            add(f'blk.{layer}.ffn_{axis}_exps.weight', data, kind)
        add(f'blk.{layer}.ffn_gate_inp.weight',
            np.arange(6 * 256, dtype=np.float32).reshape(6, 256) + layer * 10000)
        for name in ('ffn_gate_inp_shexp', 'ffn_gate_shexp', 'ffn_up_shexp', 'ffn_down_shexp',
                     'ssm_conv1d', 'hc_ffn_norm'):
            add(f'blk.{layer}.{name}.weight', np.full((2, 256), layer + 0.5, dtype=np.float32))
    add('token_embd.weight', np.ones((8, 256), dtype=np.float32))
    add('output.weight', np.full((8, 256), 0.125, dtype=np.float32))
    add(prune.PLE, np.full((32, 90), 19, dtype=np.uint8), gguf.GGMLQuantizationType.IQ4_NL)
    if extra_tensor:
        add(extra_tensor, np.ones((2, 256), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return sorted(directory.glob('*.gguf'))[0], raw


@unittest.skipIf(gguf is None, 'Installare requirements-gguf.txt: serve la revisione con Q2_0.')
class TestGGUFPrune(unittest.TestCase):
    def setUp(self):
        silent = patch.object(prune, 'progress')
        silent.start()
        self.addCleanup(silent.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.first, self.raw = fixture(self.root / 'source')
        self.inventory = prune.Inventory(self.first)
        self.manifest = prune.identity_selection(self.inventory, self.inventory.files(hashed=True))
        self.out = self.root / 'out'

    def reduced(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['provenance'] = {'method': 'manual', 'description': 'Synthetic noncontiguous selection'}
        manifest['layers'][0]['expert_ids'] = [0, 2, 5]
        manifest['layers'][1]['expert_ids'] = [1, 3, 4]
        return manifest

    def read_payloads(self, first):
        inventory = prune.Inventory(first)
        payloads = {}
        for shard in inventory.shards:
            with shard.path.open('rb') as stream:
                for tensor in shard.tensors:
                    stream.seek(tensor.offset)
                    payloads[tensor.name] = stream.read(tensor.nbytes)
        return inventory, payloads

    def assert_clean_failure(self, manifest=None):
        with self.assertRaises((ValueError, OSError)):
            prune.compact(self.first, manifest or self.manifest, self.out)
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.glob('.gguf-prune-*')), [])

    def edit_shard(self, path, edit):
        reader = gguf.GGUFReader(path, 'r+')
        try:
            edit(reader)
            reader.data.flush()
        finally:
            reader.data._mmap.close()

    def test_inspect_counts_estimates_without_reading_payloads(self):
        self.assertEqual(len(self.inventory.shards), 4)
        self.assertEqual(len(self.inventory.shards[0].tensors), 0)
        with patch.object(prune, 'hash_file', side_effect=AssertionError('Payload hashing forbidden')), redirect_stdout(io.StringIO()) as output:
            result = prune.main(['inspect', '--model', str(self.first), '--budgets', '6', '3', '2'])
        self.assertEqual(json.loads(output.getvalue()), result)
        self.assertFalse(result['source_hashes_verified'])
        self.assertEqual(set(result['expert_tensor_types']), prune.EXPERT_TYPES)
        sizes = result['bytes_by_component']
        self.assertEqual(sum(sizes.values()), result['file_bytes'])
        self.assertEqual(result['estimates'][1]['saved_payload_bytes'], (sizes['experts'] + sizes['routers']) // 2)
        self.assertEqual(result['estimates'][0]['saved_payload_bytes'], 0)
        for budget in (True, 1, 7):
            with self.assertRaises(ValueError):
                self.inventory.summary([budget])

    def test_identity_preserves_all_payloads_and_metadata(self):
        before = {s.path.name: hashlib.sha256(s.path.read_bytes()).hexdigest() for s in self.inventory.shards}
        report = prune.compact(self.first, self.manifest, self.out)
        new, payloads = self.read_payloads(self.out / report['entrypoint'])
        self.assertEqual(payloads, self.raw)
        self.assertEqual(new.top_k, 2)
        self.assertEqual(new.experts, 6)
        self.assertEqual(report['status'], 'compacted_unvalidated')
        self.assertEqual(report['quality_validation']['status'], 'not_run')
        self.assertEqual(report['runtime_smoke']['status'], 'not_run')
        self.assertTrue(report['structural_validation']['passed'])
        self.assertGreater(report['memory']['peak_resident_bytes'] or 0, 0)
        for entry in report['output_files']:
            self.assertEqual(entry['sha256'], hashlib.sha256((self.out / entry['name']).read_bytes()).hexdigest())
        for s in self.inventory.shards:
            self.assertEqual(before[s.path.name], hashlib.sha256(s.path.read_bytes()).hexdigest())
        self.assertEqual(json.loads((self.out / 'report.json').read_text(encoding='utf-8')), report)

    def test_noncontiguous_experts_router_rows_and_unchanged_tensors(self):
        manifest = self.reduced()
        report = prune.compact(self.first, manifest, self.out)
        new, payloads = self.read_payloads(self.out / report['entrypoint'])
        self.assertEqual(new.experts, 3)
        self.assertEqual(new.top_k, 2)
        for name, old in self.raw.items():
            tensor = self.inventory.tensors[name]
            if tensor.layer is None:
                expected = old
            else:
                stride = len(old) // 6
                expected = b''.join(old[i * stride:(i + 1) * stride]
                                    for i in manifest['layers'][tensor.layer]['expert_ids'])
            self.assertEqual(payloads[name], expected, name)
            self.assertEqual(new.tensors[name].kind, tensor.kind)
        self.assertGreater(report['saved_file_bytes'], 0)

    def test_single_file_and_custom_alignment(self):
        for split in (False, True):
            with self.subTest(split=split):
                first, raw = fixture(self.root / f'align-{split}', split=split, alignment=64)
                inv = prune.Inventory(first)
                manifest = prune.identity_selection(inv, inv.files(hashed=True))
                out = self.root / f'align-out-{split}'
                report = prune.compact(first, manifest, out)
                actual, payloads = self.read_payloads(out / report['entrypoint'])
                self.assertEqual(payloads, raw)
                self.assertTrue(all(s.alignment == 64 for s in actual.shards))

    def test_manifest_rejections_do_not_create_output(self):
        bad = []
        for ids in ([0, 0, 1], [2, 0, 1], [0], [0, 1, 6], [0, True, 2], ['0', 1, 2]):
            value = self.reduced()
            value['layers'][0]['expert_ids'] = ids
            bad.append(value)
        value = self.reduced()
        value['layers'][0]['expert_ids'] = [0, 1]
        bad.append(value)
        value = self.reduced()
        value['layers'].reverse()
        bad.append(value)
        value = self.reduced()
        value['layers'].pop()
        bad.append(value)
        value = self.reduced()
        value['provenance'] = {'method': 'identity'}
        bad.append(value)
        value = self.reduced()
        value['provenance'] = {'method': 'routing_trace'}
        bad.append(value)
        value = self.reduced()
        value['source']['files'][0]['sha256'] = '0' * 64
        bad.append(value)
        value = self.reduced()
        value['source']['files'][0]['name'] = '../other.gguf'
        bad.append(value)
        bad.append({'schema_version': 1, 'expert_mask': [[0, 1]]})
        for i, manifest in enumerate(bad):
            with self.subTest(case=i):
                self.assert_clean_failure(manifest)

    def test_missing_or_truncated_shards(self):
        path = self.inventory.shards[-1].path
        payload = path.read_bytes()
        path.unlink()
        with self.assertRaises(ValueError):
            prune.Inventory(self.first)
        path.write_bytes(payload[:-300])
        with self.assertRaises(ValueError):
            prune.Inventory(self.first)

    def test_duplicate_tensor_names_across_shards(self):
        target = self.inventory.tensors['blk.1.ffn_gate_exps.weight']
        def edit(reader):
            tensor = next(t for t in reader.tensors if t.name == target.name)
            tensor.field.parts[1][:] = np.frombuffer(b'blk.0.ffn_gate_exps.weight', dtype=np.uint8)
        self.edit_shard(self.inventory.shards[target.shard].path, edit)
        with self.assertRaises(ValueError):
            prune.Inventory(self.first)

    def test_bad_split_and_wrong_shape_or_type(self):
        cases = [
            lambda r: r.fields['split.tensors.count'].parts[-1].__setitem__(0, 1),
            lambda r: r.tensors[0].field.parts[3].__setitem__(-1, 5),
            lambda r: r.tensors[3].field.parts[4].__setitem__(0, int(gguf.GGMLQuantizationType.Q8_0)),
            lambda r: r.tensors[1].field.parts[-1].__setitem__(0, 0),
        ]
        path = self.inventory.shards[1].path
        raw = path.read_bytes()
        for i, edit in enumerate(cases):
            with self.subTest(case=i):
                path.write_bytes(raw)
                self.edit_shard(path, edit)
                with self.assertRaises(ValueError):
                    prune.Inventory(self.first)

    def test_conflicting_metadata_across_shards(self):
        def edit(reader):
            # The global writer keeps model metadata only in shard zero, so
            # change alignment metadata on a separate custom-alignment fixture.
            reader.fields['general.alignment'].parts[-1][0] = 128
        first, _ = fixture(self.root / 'conflict', alignment=64)
        inv = prune.Inventory(first)
        self.edit_shard(inv.shards[1].path, edit)
        with self.assertRaises(ValueError):
            prune.Inventory(first)

    def test_unsupported_extra_expert_tensor_or_routing_metadata(self):
        for name, kw in [('fused', {'extra_tensor': 'blk.0.ffn_gate_up_exps.weight'}),
                         ('group', {'extra_metadata': {'qwen4exp.expert_group_count': 2}})]:
            first, _ = fixture(self.root / name, **kw)
            with self.assertRaises(ValueError):
                prune.Inventory(first)

    def test_hash_mismatch_detects_modified_source_payload(self):
        tensor = self.inventory.tensors[prune.PLE]
        with self.inventory.shards[tensor.shard].path.open('r+b') as stream:
            stream.seek(tensor.offset)
            stream.write(b'\xff')
        self.assert_clean_failure()

    def test_corrupted_output_is_not_promoted(self):
        real_copy = prune.copy_payloads
        def corrupt(inv, rows, writers):
            hashes = real_copy(inv, rows, writers)
            fp = writers[-1].fout[0]
            fp.seek(-40, 2)
            fp.write(b'\x00' * 32)
            fp.flush()
            return hashes
        with patch.object(prune, 'copy_payloads', side_effect=corrupt):
            self.assert_clean_failure()

    def test_no_space_and_interruption_leave_no_published_files(self):
        with patch.object(prune.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)), patch.object(prune, 'hash_file') as hashing:
            self.assert_clean_failure()
            hashing.assert_not_called()
        for error in (OSError('disk full during copy'), KeyboardInterrupt()):
            with patch.object(prune, 'copy_payloads', side_effect=error), self.assertRaises(type(error)):
                prune.compact(self.first, self.manifest, self.out)
            self.assertFalse(self.out.exists())
            self.assertEqual(list(self.root.glob('.gguf-prune-*')), [])

    def test_existing_destination_and_selection_are_preserved(self):
        self.out.mkdir()
        sentinel = self.out / 'sentinel.txt'
        sentinel.write_text('keep')
        with self.assertRaises(ValueError):
            prune.compact(self.first, self.manifest, self.out)
        self.assertEqual(sentinel.read_text(), 'keep')
        selection = self.root / 'identity.json'
        selection.write_text('keep')
        with self.assertRaises(ValueError):
            prune.main(['inspect', '--model', str(self.first), '--identity-selection', str(selection)])
        self.assertEqual(selection.read_text(), 'keep')

    def test_identity_manifest_cli_and_real_compact_cli(self):
        selection = self.root / 'identity.json'
        with redirect_stdout(io.StringIO()):
            prune.main(['inspect', '--model', str(self.first), '--identity-selection', str(selection)])
        self.assertEqual(json.loads(selection.read_text()), self.manifest)
        with redirect_stdout(io.StringIO()) as output:
            prune.main(['compact', '--model', str(self.first), '--selection', str(selection), '--out', str(self.out)])
        self.assertEqual(json.loads(output.getvalue())['status'], 'compacted_unvalidated')

    def test_copy_has_bounded_reads_and_detects_short_read(self):
        data = bytes(range(256)) * 20
        sizes = []
        class Source(io.BytesIO):
            def read(self, size=-1):
                sizes.append(size)
                return super().read(size)
        result = io.BytesIO()
        digest = hashlib.sha256()
        with patch.object(prune, 'BUFFER_BYTES', 127):
            prune.consume(Source(data), len(data), (digest,), result)
        self.assertEqual(result.getvalue(), data)
        self.assertEqual(digest.hexdigest(), hashlib.sha256(data).hexdigest())
        self.assertLessEqual(max(sizes), 127)
        with self.assertRaises(ValueError):
            prune.consume(io.BytesIO(b'123'), 4)

    def test_grouped_ranges_merge_only_adjacent_payloads(self):
        self.assertEqual(prune.grouped_ranges([(32, 8), (0, 16), (16, 16), (64, 4)]),
                         [(0, 40), (64, 4)])
        for ranges in ([(0, 0)], [(-1, 1)], [(0, 4), (3, 2)]):
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                prune.grouped_ranges(ranges)

    def test_hub_range_source_is_bounded_and_fail_closed(self):
        class Response:
            def __init__(self, body, status=206, content_range='bytes 4-9/32',
                         etag='"source-etag"', content_length='6', scheme='https'):
                self.body = io.BytesIO(body)
                self.status = status
                self.headers = {'Content-Range': content_range, 'ETag': etag}
                if content_length is not None:
                    self.headers['Content-Length'] = content_length
                self.url = f'{scheme}://cdn.example/file'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def getcode(self):
                return self.status

            def geturl(self):
                return self.url

            def read(self, size=-1):
                return self.body.read(size)

        seen = []

        def opener(request, timeout):
            seen.append((request, timeout))
            return Response(b'abcdef')

        source = prune.HubRangeSource(
            'https://huggingface.co/repo/resolve/commit/file', 32, '"source-etag"',
            opener=opener, timeout=7,
        )
        output = io.BytesIO()
        digest = hashlib.sha256()
        with patch.object(prune, 'BUFFER_BYTES', 3):
            source.copy_range(4, 6, output, (digest,))
        self.assertEqual(output.getvalue(), b'abcdef')
        self.assertEqual(digest.hexdigest(), hashlib.sha256(b'abcdef').hexdigest())
        request, timeout = seen[0]
        self.assertEqual(request.get_header('Range'), 'bytes=4-9')
        self.assertEqual(request.get_header('If-match'), '"source-etag"')
        self.assertEqual(request.get_header('Accept-encoding'), 'identity')
        self.assertEqual(timeout, 7)
        pinned = prune.HubRangeSource.from_repo(
            'owner/repo', 'd96201194f103c906994f283f504209dc8d2a10c',
            'folder/model shard.gguf', 32, '"source-etag"',
        )
        self.assertEqual(
            pinned.url,
            'https://huggingface.co/owner/repo/resolve/'
            'd96201194f103c906994f283f504209dc8d2a10c/folder/model%20shard.gguf',
        )
        for revision, path in (('main', 'model.gguf'), ('d' * 40, '../model.gguf')):
            with self.subTest(revision=revision, path=path), self.assertRaises(ValueError):
                prune.HubRangeSource.from_repo('owner/repo', revision, path, 32, '"source-etag"')

        bad_responses = (
            Response(b'abcdef', status=200),
            Response(b'abcdef', status=412),
            Response(b'abcdef', content_range='bytes 3-8/32'),
            Response(b'abcdef', etag='"different"'),
            Response(b'abcdef', content_length='5'),
            Response(b'abc'),
            Response(b'abcdefg'),
            Response(b'abcdef', scheme='http'),
        )
        for bad in bad_responses:
            with self.subTest(status=bad.status, headers=bad.headers, url=bad.url):
                dest = io.BytesIO()
                with self.assertRaises(ValueError):
                    prune.HubRangeSource(
                        'https://huggingface.co/repo/resolve/commit/file',
                        32, '"source-etag"', opener=lambda request, timeout, response=bad: response,
                    ).copy_range(4, 6, dest)
                self.assertLessEqual(len(dest.getvalue()), 6)

        with self.assertRaises(ValueError):
            source.copy_range(30, 3, io.BytesIO())
        with self.assertRaises(ValueError):
            prune.HubRangeSource('http://example.test/file', 32, '"etag')
        with self.assertRaises(ValueError):
            prune.consume(io.BytesIO(b'123'), -1)

    def test_remote_header_reader_matches_local_inventory_and_compacts(self):
        class MemoryRangeSource:
            def __init__(self, path, size, etag):
                self.data = path.read_bytes()
                self.size = size
                self.etag = etag
                self.transferred_bytes = 0
                self.ranges = []

            def copy_range(self, offset, count, output, hashes=()):
                if offset < 0 or count <= 0 or offset + count > self.size:
                    raise ValueError('Bad in-memory range')
                block = self.data[offset:offset + count]
                if len(block) != count:
                    raise ValueError('Truncated in-memory range')
                output.write(block)
                self.transferred_bytes += len(block)
                self.ranges.append((offset, count))
                for digest in hashes:
                    digest.update(block)

        remote = object.__new__(prune.RemoteInventory)
        remote.is_remote = True
        remote.repo_id = 'public/model'
        remote.revision = 'd' * 40
        remote.subpath = 'gguf'
        remote.max_header_bytes = 1 << 20
        remote.remote_files = []
        remote.sources = []
        remote_shards = []
        tree_rows = []
        for index, local_shard in enumerate(self.inventory.shards):
            etag = f'"source-{index}"'
            row = {
                'path': f'gguf/{local_shard.path.name}',
                'name': local_shard.path.name,
                'size_bytes': local_shard.stamp[2],
                'oid': hashlib.sha1(local_shard.path.read_bytes()).hexdigest(),
                'etag': etag,
            }
            remote.remote_files.append(row)
            tree_rows.append({key: row[key] for key in ('path', 'name', 'size_bytes', 'oid')})
            source = MemoryRangeSource(local_shard.path, local_shard.stamp[2], etag)
            remote.sources.append(source)
            remote_shards.append(prune.read_remote_shard(
                row['name'], index, source, remote.max_header_bytes,
            ))
            self.assertLessEqual(source.transferred_bytes, remote.max_header_bytes)
            self.assertEqual(
                {key: value.fingerprint for key, value in remote_shards[-1].metadata.items()},
                {key: value.fingerprint for key, value in local_shard.metadata.items()},
            )
            self.assertEqual(remote_shards[-1].tensors, local_shard.tensors)
            self.assertTrue(all(offset + count <= remote_shards[-1].data_offset
                                for offset, count in source.ranges))
        remote._init_from_shards(remote_shards)
        manifest = prune.identity_selection(remote, remote.remote_files)
        manifest['provenance'] = {'method': 'manual', 'description': 'Remote noncontiguous selection'}
        manifest['layers'][0]['expert_ids'] = [0, 2, 5]
        manifest['layers'][1]['expert_ids'] = [1, 3, 4]
        self.assertIn('repo_id', manifest['source'])
        self.assertNotIn('sha256', manifest['source']['files'][0])
        target = self.root / 'remote-out'
        with patch.object(prune, 'hub_tree_files', return_value=tree_rows):
            report = prune.compact_inventory(remote, manifest, target)
        output_inventory, payloads = self.read_payloads(target / report['entrypoint'])
        self.assertEqual(output_inventory.experts, 3)
        for name, old in self.raw.items():
            tensor = self.inventory.tensors[name]
            if tensor.layer is None:
                expected = old
            else:
                stride = len(old) // 6
                selected = manifest['layers'][tensor.layer]['expert_ids']
                expected = b''.join(old[i * stride:(i + 1) * stride] for i in selected)
            self.assertEqual(payloads[name], expected, name)
        self.assertLess(sum(source.transferred_bytes for source in remote.sources),
                        sum(shard.stamp[2] for shard in self.inventory.shards))
        self.assertFalse(report['structural_validation']['source_hashes_verified'])
        self.assertTrue(report['structural_validation']['source_identity_verified'])
        self.assertTrue(report['structural_validation']['all_output_payloads_verified'])

    def test_hub_tree_inventory_requires_complete_shard_group(self):
        entries = [
            {'type': 'file', 'path': f'gguf/{shard.path.name}',
             'size': shard.stamp[2], 'oid': f'{index:040x}'}
            for index, shard in enumerate(self.inventory.shards)
        ]
        entries.append({'type': 'file', 'path': 'gguf/README.md', 'size': 12, 'oid': 'a' * 40})

        class Response:
            status = 200
            headers = {}

            def __init__(self, body):
                self.body = io.BytesIO(body)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def geturl(self):
                return 'https://huggingface.co/api/models/public/model/tree/revision/gguf'

            def read(self, size=-1):
                return self.body.read(size)

        seen = []

        def opener(request, timeout):
            seen.append((request, timeout))
            return Response(json.dumps(entries).encode())

        revision = 'd' * 40
        files = prune.hub_tree_files('public/model', revision, 'gguf', opener=opener)
        self.assertEqual([row['name'] for row in files], [shard.path.name for shard in self.inventory.shards])
        self.assertIn(revision, seen[0][0].full_url)
        self.assertEqual(seen[0][0].get_header('Accept'), 'application/json')
        with self.assertRaises(ValueError):
            prune.hub_tree_files(
                'public/model', revision, 'gguf',
                opener=lambda request, timeout: Response(json.dumps(entries[:-2]).encode()),
            )

    def test_remote_header_reader_rejects_truncated_directory(self):
        class TruncatedSource:
            def __init__(self, path):
                self.data = path.read_bytes()[:24]
                self.size = len(self.data)

            def copy_range(self, offset, count, output, hashes=()):
                block = self.data[offset:offset + count]
                if len(block) != count:
                    raise ValueError('Truncated source')
                output.write(block)

        source = TruncatedSource(self.inventory.shards[0].path)
        with self.assertRaises(ValueError):
            prune.read_remote_shard(
                self.inventory.shards[0].path.name, 0, source, 1 << 20,
            )

    def test_remote_header_reads_exact_ranges_without_payload_bytes(self):
        local_shard = next(shard for shard in self.inventory.shards if shard.tensors)

        class HeaderOnlySource:
            def __init__(self, path, tensors):
                self.data = path.read_bytes()
                self.size = len(self.data)
                self.forbidden = [(tensor.offset, tensor.offset + tensor.nbytes) for tensor in tensors]
                self.ranges = []

            def copy_range(self, offset, count, output, hashes=()):
                end = offset + count
                if offset < 0 or count <= 0 or end > self.size:
                    raise ValueError('Bad in-memory range')
                if any(offset < stop and start < end for start, stop in self.forbidden):
                    raise AssertionError('Inventory requested tensor payload bytes')
                self.ranges.append((offset, count))
                block = self.data[offset:end]
                if len(block) != count:
                    raise ValueError('Truncated in-memory range')
                output.write(block)
                for digest in hashes:
                    digest.update(block)

        source = HeaderOnlySource(local_shard.path, local_shard.tensors)
        remote = prune.read_remote_shard(
            local_shard.path.name, local_shard.tensors[0].shard,
            source, 1 << 20,
        )
        self.assertNotEqual(local_shard.data_offset % 4096, 0)
        self.assertTrue(source.ranges)
        self.assertTrue(all(offset + count <= local_shard.data_offset
                            for offset, count in source.ranges))
        self.assertEqual(remote.data_offset, local_shard.data_offset)
        self.assertEqual(remote.tensors, local_shard.tensors)
        self.assertEqual(
            {key: value.fingerprint for key, value in remote.metadata.items()},
            {key: value.fingerprint for key, value in local_shard.metadata.items()},
        )

    def test_remote_header_budgets_fail_before_reading_more(self):
        local_shard = next(shard for shard in self.inventory.shards if shard.tensors)

        class CountingSource:
            def __init__(self, path):
                self.data = path.read_bytes()
                self.size = len(self.data)
                self.ranges = []

            def copy_range(self, offset, count, output, hashes=()):
                self.ranges.append((offset, count))
                output.write(self.data[offset:offset + count])

        source = CountingSource(local_shard.path)
        with self.assertRaisesRegex(ValueError, 'limite header'):
            prune.read_remote_shard(local_shard.path.name, 0, source, 24)
        self.assertEqual(sum(count for _, count in source.ranges), 24)
        self.assertTrue(all(offset + count <= local_shard.data_offset
                            for offset, count in source.ranges))

        source = CountingSource(local_shard.path)
        with self.assertRaisesRegex(ValueError, 'limite elementi'):
            prune.read_remote_shard(local_shard.path.name, 0, source, 1 << 20, 1)
        self.assertEqual(sum(count for _, count in source.ranges), 24)

        source = CountingSource(local_shard.path)
        with patch.object(prune, 'MAX_REMOTE_HEADER_RANGES', 2):
            with self.assertRaisesRegex(ValueError, 'richieste Range'):
                prune.read_remote_shard(local_shard.path.name, 0, source, 1 << 20)
        self.assertEqual(len(source.ranges), 2)

    def test_remote_header_rejects_malformed_tensor_directory(self):
        local_shard = next(shard for shard in self.inventory.shards if len(shard.tensors) >= 2)
        reader = gguf.GGUFReader(local_shard.path, 'r')
        try:
            first, second = next(
                (left, right)
                for index, left in enumerate(reader.tensors)
                for right in reader.tensors[index + 1:]
                if len(left.name.encode('utf-8')) == len(right.name.encode('utf-8'))
            )
            def positions(tensor):
                field_offset = int(tensor.field.offset)
                name_length = struct.unpack_from('<Q', raw, field_offset)[0]
                dims_position = field_offset + 8 + name_length
                n_dims = struct.unpack_from('<I', raw, dims_position)[0]
                dimensions = dims_position + 4
                type_position = dimensions + n_dims * 8
                return field_offset + 8, name_length, dimensions, type_position, type_position + 4

            raw = bytearray(local_shard.path.read_bytes())
            first_name_pos, first_name_length, _, _, _ = positions(first)
            second_name_pos, second_name_length, _, _, _ = positions(second)
            duplicate = bytearray(raw)
            duplicate[second_name_pos:second_name_pos + second_name_length] = \
                raw[first_name_pos:first_name_pos + first_name_length]
            invalid_type = bytearray(raw)
            _, _, _, _, type_offset_position = positions(first)
            struct.pack_into('<I', invalid_type, type_offset_position - 4, 0xFFFFFFFF)
            invalid_shape = bytearray(raw)
            _, _, dimensions_position, _, _ = positions(first)
            struct.pack_into('<Q', invalid_shape, dimensions_position, 1)
            out_of_range = bytearray(raw)
            offset_position = positions(first)[4]
            relative = ((len(raw) // local_shard.alignment) + 1) * local_shard.alignment
            struct.pack_into('<Q', out_of_range, offset_position, relative)
        finally:
            reader.data._mmap.close()

        class HeaderOnlySource:
            def __init__(self, data):
                self.data = data
                self.size = len(data)
                self.forbidden = [(tensor.offset, tensor.offset + tensor.nbytes)
                                  for tensor in local_shard.tensors]

            def copy_range(self, offset, count, output, hashes=()):
                end = offset + count
                if any(offset < stop and start < end for start, stop in self.forbidden):
                    raise AssertionError('Malformed inventory requested payload bytes')
                output.write(self.data[offset:end])

        for label, data in (
            ('duplicate tensor name', duplicate),
            ('unsupported type', invalid_type),
            ('invalid block shape', invalid_shape),
            ('out-of-range payload', out_of_range),
        ):
            with self.subTest(case=label), self.assertRaises(ValueError):
                prune.read_remote_shard(
                    local_shard.path.name, 0, HeaderOnlySource(data), 1 << 20,
                )


if __name__ == '__main__':
    unittest.main()
