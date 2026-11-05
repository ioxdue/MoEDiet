"""Inspect and physically compact qwen4exp GGUF experts without requantization.

This is a file transformation, not a routing profiler or a quality validator.
Only the observed separate gate/up/down expert layout is supported.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

ARCH = 'qwen4exp'
GGUF_REVISION = '19cce6919992fdffd92e0ac152caaf1e2b8f254e'
BUFFER_BYTES = 4 << 20
GIB = 1 << 30
SPLIT_KEYS = {'split.no', 'split.count', 'split.tensors.count'}
EXPERT_TYPES = {'Q5_0', 'IQ3_XXS', 'IQ4_XS', 'IQ4_NL', 'Q2_0', 'IQ2_S'}
EXPERT_RE = re.compile(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight\Z')
ROUTER_RE = re.compile(r'blk\.(\d+)\.ffn_gate_inp\.weight\Z')
SHARD_RE = re.compile(r'(.+)-(\d{5})-of-(\d{5})\.gguf\Z')
PLE = 'per_layer_token_embd.weight'


def gguf_module():
    try:
        import gguf
        if not all(hasattr(gguf.GGMLQuantizationType, name) for name in EXPERT_TYPES):
            raise ImportError('La libreria installata non include i tipi richiesti, incluso Q2_0.')
    except ImportError as exc:
        raise ImportError('Installare requirements-gguf.txt in un ambiente separato. ' + str(exc)) from exc
    return gguf


def file_stamp(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def consume(stream, count, hashes=(), output=None):
    """Bounded reads; no tensor-sized allocation, dequantizer or NumPy gather."""
    if count < 0:
        raise ValueError('Intervallo di lettura negativo.')
    while count:
        block = stream.read(min(count, BUFFER_BYTES))
        if not block:
            raise ValueError('File troncato durante la lettura del payload.')
        if output is not None and output.write(block) != len(block):
            raise OSError('Scrittura incompleta del payload.')
        for digest in hashes:
            digest.update(block)
        count -= len(block)


def hash_file(path):
    stamp = file_stamp(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        consume(stream, stamp[2], (digest,))
    if file_stamp(path) != stamp:
        raise ValueError(f'Sorgente modificata durante il calcolo hash: {path.name}')
    return digest.hexdigest()


@dataclass(frozen=True)
class Metadata:
    value: object
    kind: int
    subtype: int | None
    fingerprint: str


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple
    kind: int
    type_name: str
    offset: int
    nbytes: int
    shard: int

    @property
    def layer(self):
        match = EXPERT_RE.fullmatch(self.name) or ROUTER_RE.fullmatch(self.name)
        return int(match[1]) if match else None

    @property
    def category(self):
        if EXPERT_RE.fullmatch(self.name):
            return 'experts'
        if ROUTER_RE.fullmatch(self.name):
            return 'routers'
        return 'ple' if self.name == PLE else 'other'


@dataclass
class Shard:
    path: Path
    stamp: tuple
    metadata: dict
    tensors: list
    alignment: int


def read_shard(path, index):
    gguf = gguf_module()
    reader = None
    stamp = file_stamp(path)
    try:
        reader = gguf.GGUFReader(path, 'r')
        if reader.endianess != gguf.GGUFEndian.LITTLE:
            raise ValueError('Sono supportati soltanto GGUF little-endian.')
        if reader.fields['GGUF.version'].contents() != 3:
            raise ValueError('Questa versione richiede GGUF v3.')
        metadata = {}
        for key, field in reader.fields.items():
            if key.startswith('GGUF.'):
                continue
            subtype = None
            if field.types[0] == gguf.GGUFValueType.ARRAY:
                subtype = int(field.parts[3][0])
                if subtype == int(gguf.GGUFValueType.ARRAY) or not field.data:
                    raise ValueError(f'Array annidato o vuoto non supportato da questo writer: {key}')
            digest = hashlib.sha256()
            for part in field.parts[2:]:
                digest.update(part.tobytes())
            metadata[key] = Metadata(field.contents(), int(field.types[0]), subtype, digest.hexdigest())
        alignment = int(reader.alignment)
        tensors = []
        for tensor in reader.tensors:
            shape = tuple(int(n) for n in tensor.shape)
            block, size = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
            if not shape or any(n <= 0 for n in shape) or shape[0] % block:
                raise ValueError(f'Dimensioni/blocchi incompatibili: {tensor.name}')
            nbytes = math.prod(shape) // block * size
            offset = int(tensor.data_offset)
            if (nbytes != tensor.n_bytes or offset < reader.data_offset or
                    (offset - reader.data_offset) % alignment or offset + nbytes > stamp[2]):
                raise ValueError(f'Offset o dimensione fuori file: {tensor.name}')
            tensors.append(Tensor(tensor.name, shape, int(tensor.tensor_type), tensor.tensor_type.name,
                                  offset, nbytes, index))
        end = int(reader.data_offset)
        for tensor in sorted(tensors, key=lambda t: t.offset):
            if tensor.offset < end:
                raise ValueError(f'Payload sovrapposti: {tensor.name}')
            end = tensor.offset + tensor.nbytes
        if file_stamp(path) != stamp:
            raise ValueError('File modificato durante la lettura delle intestazioni.')
        return Shard(path, stamp, metadata, tensors, alignment)
    except (IndexError, KeyError, OverflowError) as exc:
        raise ValueError(f'GGUF malformato {path.name}: {exc}') from exc
    finally:
        # Copy out header data only. Do not retain any ndarray backed by this mmap.
        if reader is not None:
            reader.data._mmap.close()


def metadata_int(metadata, key):
    if key not in metadata or type(metadata[key].value) is not int or metadata[key].value < 1:
        raise ValueError(f'Metadato intero positivo richiesto: {key}')
    return metadata[key].value


class Inventory:
    def __init__(self, model):
        first = Path(model).expanduser().resolve(strict=True)
        match = SHARD_RE.fullmatch(first.name)
        if match:
            count = int(match[3])
            if int(match[2]) != 1 or not 1 <= count <= 99999:
                raise ValueError('Indicare il primo shard (00001-of-...).')
            paths = [first.with_name(f'{match[1]}-{i:05d}-of-{count:05d}.gguf') for i in range(1, count + 1)]
        else:
            paths = [first]
        if any(not path.is_file() for path in paths):
            raise ValueError('Gruppo GGUF incompleto: manca almeno uno shard.')
        if len({(p.stat().st_dev, p.stat().st_ino) for p in paths}) != len(paths):
            raise ValueError('Gli shard devono essere file distinti.')
        self.shards = [read_shard(path, i) for i, path in enumerate(paths)]
        self.metadata, self.tensors = {}, {}
        total = sum(len(s.tensors) for s in self.shards)
        for index, shard in enumerate(self.shards):
            split = {key: f.value for key, f in shard.metadata.items() if key in SPLIT_KEYS}
            expected = {'split.no': index, 'split.count': len(paths), 'split.tensors.count': total}
            if (len(paths) > 1 or split) and split != expected:
                raise ValueError(f'Metadati split incoerenti: {shard.path.name}')
            if any(type(value) is not int for value in split.values()):
                raise ValueError('I metadati split devono essere interi.')
            if shard.alignment != self.shards[0].alignment:
                raise ValueError('Allineamento differente fra shard non supportato.')
            for key, field in shard.metadata.items():
                if key in SPLIT_KEYS:
                    continue
                if key not in self.shards[0].metadata:
                    raise ValueError(f'Metadato presente soltanto in uno shard secondario: {key}')
                if key in self.metadata and field.fingerprint != self.metadata[key].fingerprint:
                    raise ValueError(f'Metadato discordante fra shard: {key}')
                self.metadata[key] = field
            for tensor in shard.tensors:
                if tensor.name in self.tensors:
                    raise ValueError(f'Tensore duplicato fra shard: {tensor.name}')
                self.tensors[tensor.name] = tensor
        if self.value('general.architecture') != ARCH:
            raise ValueError(f'Architettura GGUF richiesta: {ARCH}.')
        self.layers = metadata_int(self.metadata, f'{ARCH}.block_count')
        self.experts = metadata_int(self.metadata, f'{ARCH}.expert_count')
        self.top_k = metadata_int(self.metadata, f'{ARCH}.expert_used_count')
        hidden = metadata_int(self.metadata, f'{ARCH}.embedding_length')
        intermediate = metadata_int(self.metadata, f'{ARCH}.expert_feed_forward_length')
        if self.top_k > self.experts:
            raise ValueError('Top-k maggiore del numero di esperti.')
        allowed_expert_metadata = {f'{ARCH}.{suffix}' for suffix in (
            'expert_count', 'expert_used_count', 'expert_feed_forward_length',
            'expert_shared_feed_forward_length', 'expert_shared_count', 'expert_weights_scale',
            'expert_weights_norm', 'expert_gating_func')}
        if any('.expert_' in key and key not in allowed_expert_metadata for key in self.metadata):
            raise ValueError('Metadati di routing non supportati (ad esempio routing a gruppi).')
        expected = {}
        for layer in range(self.layers):
            for axis in ('gate', 'up', 'down'):
                expected[f'blk.{layer}.ffn_{axis}_exps.weight'] = (
                    (intermediate, hidden, self.experts) if axis == 'down' else (hidden, intermediate, self.experts))
            expected[f'blk.{layer}.ffn_gate_inp.weight'] = (hidden, self.experts)
        for name, shape in expected.items():
            tensor = self.tensors.get(name)
            if tensor is None or tensor.shape != shape:
                raise ValueError(f'Tensore mancante o layout inatteso: {name}; atteso {shape}')
            if ((tensor.category == 'experts' and tensor.type_name not in EXPERT_TYPES) or
                    (tensor.category == 'routers' and tensor.type_name != 'F32')):
                raise ValueError(f'Tipo non supportato per {name}: {tensor.type_name}')
        for name in self.tensors:
            shared_gate = re.fullmatch(r'blk\.\d+\.ffn_gate_inp_shexp\.weight', name)
            if any(s in name for s in ('_exps', 'expert', 'ffn_gate_inp')) and name not in expected and not shared_gate:
                raise ValueError(f'Tensore associato agli esperti non supportato: {name}')

    def value(self, key):
        return self.metadata[key].value if key in self.metadata else None

    def unchanged(self):
        if any(file_stamp(s.path) != s.stamp for s in self.shards):
            raise ValueError('Gli shard sorgenti sono cambiati durante l\'operazione.')

    def files(self, hashed=False):
        result = []
        for shard in self.shards:
            row = {'name': shard.path.name, 'size_bytes': shard.stamp[2]}
            if hashed:
                progress(f'SHA-256: {shard.path.name}')
                row['sha256'] = hash_file(shard.path)
            result.append(row)
        self.unchanged()
        return result

    def summary(self, budgets=()):
        sizes = Counter()
        types = Counter()
        for t in self.tensors.values():
            sizes[t.category] += t.nbytes
            if t.category == 'experts':
                types[t.type_name] += 1
        total = sum(s.stamp[2] for s in self.shards)
        sizes['overhead'] = total - sum(sizes.values())
        estimates = []
        for b in budgets:
            if type(b) is not int or not self.top_k <= b <= self.experts:
                raise ValueError(f'Budget fuori intervallo [{self.top_k}, {self.experts}]: {b}')
            saved = sum(t.nbytes // self.experts * (self.experts - b)
                        for t in self.tensors.values() if t.layer is not None)
            estimates.append({'retained_per_layer': b, 'estimated_file_bytes': total - saved,
                              'saved_payload_bytes': saved,
                              'weights_excluding_ple_bytes': sum(t.nbytes for t in self.tensors.values()
                                                               if t.category != 'ple') - saved})
        return {'architecture': ARCH, 'layers': self.layers, 'experts_per_layer': self.experts,
                'top_k': self.top_k, 'tensor_count': len(self.tensors), 'file_bytes': total,
                'file_gib': total / GIB, 'bytes_by_component': dict(sizes),
                'expert_tensor_types': dict(types), 'estimates': estimates,
                'note': 'Stime file con overhead sorgente; pesi senza PLE non equivalgono alla RAM totale.'}


def identity_selection(inventory, files):
    return {'schema_version': 1, 'architecture': ARCH, 'source': {'files': files},
            'layers': [{'layer': i, 'expert_ids': list(range(inventory.experts))} for i in range(inventory.layers)],
            'provenance': {'method': 'identity'}}


def validate_selection(inventory, manifest):
    if (not isinstance(manifest, dict) or set(manifest) !=
            {'schema_version', 'architecture', 'source', 'layers', 'provenance'} or
            type(manifest['schema_version']) is not int or manifest['schema_version'] != 1 or
            manifest['architecture'] != ARCH):
        raise ValueError('Manifesto GGUF v1 non valido; le maschere HF non sono compatibili.')
    source = manifest['source']
    if not isinstance(source, dict) or set(source) - {'files', 'repo_id', 'revision'} or 'files' not in source:
        raise ValueError('Sorgente del manifesto non valida.')
    for key in ('repo_id', 'revision'):
        if key in source and (not isinstance(source[key], str) or not source[key].strip()):
            raise ValueError(f'Provenienza sorgente non valida: {key}')
    files = source['files']
    if not isinstance(files, list) or len(files) != len(inventory.shards):
        raise ValueError('Elenco shard del manifesto incompleto.')
    for row, shard in zip(files, inventory.shards):
        if (not isinstance(row, dict) or set(row) != {'name', 'size_bytes', 'sha256'} or
                row['name'] != shard.path.name or type(row['size_bytes']) is not int or
                row['size_bytes'] != shard.stamp[2] or not isinstance(row['sha256'], str) or
                not re.fullmatch(r'[0-9a-f]{64}', row['sha256'])):
            raise ValueError('Identità shard/hash del manifesto non valida.')
    provenance = manifest['provenance']
    if (not isinstance(provenance, dict) or set(provenance) - {'method', 'description', 'trace_sha256'} or
            provenance.get('method') not in ('identity', 'manual', 'routing_trace')):
        raise ValueError('Provenienza selezione non valida.')
    if 'description' in provenance and not isinstance(provenance['description'], str):
        raise ValueError('La descrizione della selezione deve essere testuale.')
    if provenance['method'] == 'routing_trace' and not re.fullmatch(r'[0-9a-f]{64}', str(provenance.get('trace_sha256', ''))):
        raise ValueError('Una selezione routing_trace richiede lo SHA-256 della traccia.')
    layers = manifest['layers']
    if not isinstance(layers, list) or len(layers) != inventory.layers:
        raise ValueError('Occorre una selezione per ogni layer.')
    rows = []
    for i, row in enumerate(layers):
        if (not isinstance(row, dict) or set(row) != {'layer', 'expert_ids'} or
                type(row['layer']) is not int or row['layer'] != i):
            raise ValueError('I layer devono essere espliciti e ordinati da zero.')
        ids = row['expert_ids']
        if (not isinstance(ids, list) or not inventory.top_k <= len(ids) <= inventory.experts or
                any(type(x) is not int or not 0 <= x < inventory.experts for x in ids) or
                ids != sorted(set(ids)) or (rows and len(ids) != len(rows[0]))):
            raise ValueError('ID unici ordinati, budget uniforme e almeno top-k esperti sono obbligatori.')
        rows.append(ids)
    if provenance['method'] == 'identity' and any(row != list(range(inventory.experts)) for row in rows):
        raise ValueError('Una selezione ridotta non può dichiarare provenance.method=identity.')
    return rows


def output_shape(tensor, rows):
    return (*tensor.shape[:-1], len(rows[tensor.layer])) if tensor.layer is not None else tensor.shape


def payload_ranges(tensor, rows):
    if tensor.layer is None:
        yield tensor.offset, tensor.nbytes
    else:
        stride = tensor.nbytes // tensor.shape[-1]
        for expert in rows[tensor.layer]:
            yield tensor.offset + expert * stride, stride


def prepare_headers(inventory, rows, directory):
    """Upstream serializes metadata/offsets; this module only streams raw payloads."""
    gguf = gguf_module()
    import numpy as np
    writers, paths, expected_bytes = [], [], 0
    try:
        for i, shard in enumerate(inventory.shards):
            name = f'model-{i + 1:05d}-of-{len(inventory.shards):05d}.gguf' if len(inventory.shards) > 1 else 'model.gguf'
            path = directory / name
            writer = gguf.GGUFWriter(path, ARCH)
            writers.append(writer)
            paths.append(path)
            writer.data_alignment = shard.alignment
            for key, field in shard.metadata.items():
                if key in SPLIT_KEYS or key == 'general.architecture':
                    continue
                value = len(rows[0]) if key == f'{ARCH}.expert_count' else field.value
                writer.add_key_value(key, value, gguf.GGUFValueType(field.kind),
                                     gguf.GGUFValueType(field.subtype) if field.subtype is not None else None)
            if len(inventory.shards) > 1 or 'split.count' in shard.metadata:
                writer.add_key_value('split.no', i, gguf.GGUFValueType.UINT16)
                writer.add_key_value('split.count', len(inventory.shards), gguf.GGUFValueType.UINT16)
                writer.add_key_value('split.tensors.count', len(inventory.tensors), gguf.GGUFValueType.INT32)
            payload_bytes = 0
            for tensor in shard.tensors:
                shape = output_shape(tensor, rows)
                nbytes = tensor.nbytes // tensor.shape[-1] * shape[-1]
                # np.float32 avoids treating logical dimensions as a packed uint8 shape.
                writer.add_tensor_info(tensor.name, tuple(reversed(shape)), np.dtype('float32'),
                                       nbytes, raw_dtype=gguf.GGMLQuantizationType(tensor.kind))
                payload_bytes += gguf.GGUFWriter.ggml_pad(nbytes, shard.alignment)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()
            position = writer.fout[0].tell()
            expected_bytes += (gguf.GGUFWriter.ggml_pad(position, shard.alignment) + payload_bytes
                               if shard.tensors else position)
        return writers, paths, expected_bytes
    except BaseException:
        for writer in writers:
            writer.close()
        raise


def copy_payloads(inventory, rows, writers):
    hashes = {}
    for shard, writer in zip(inventory.shards, writers):
        progress(f'Copia: {shard.path.name}')
        with shard.path.open('rb') as source:
            output = writer.fout[0]
            for tensor in shard.tensors:
                writer.write_padding(output, output.tell())
                digest = hashlib.sha256()
                nbytes = 0
                for offset, count in payload_ranges(tensor, rows):
                    source.seek(offset)
                    consume(source, count, (digest,), output)
                    nbytes += count
                writer.write_padding(output, nbytes)
                hashes[tensor.name] = digest.hexdigest()
            output.flush()
            os.fsync(output.fileno())
    inventory.unchanged()
    return hashes


def verify_output(original, result, rows, hashes):
    if result.experts != len(rows[0]) or set(result.tensors) != set(original.tensors):
        raise ValueError('Inventario finale diverso da quello pianificato.')
    if set(result.metadata) != set(original.metadata):
        raise ValueError('Metadati semantici aggiunti o rimossi durante la scrittura.')
    for key, field in original.metadata.items():
        new = result.metadata[key]
        if key == f'{ARCH}.expert_count':
            if (new.value, new.kind) != (len(rows[0]), field.kind):
                raise ValueError('Conteggio finale esperti non coerente.')
        elif new.fingerprint != field.fingerprint:
            raise ValueError(f'Metadato non preservato: {key}')
    output_files = []
    for shard in result.shards:
        progress(f'Verifica: {shard.path.name}')
        full_hash = hashlib.sha256()
        with shard.path.open('rb') as stream:
            for tensor in sorted(shard.tensors, key=lambda t: t.offset):
                old = original.tensors[tensor.name]
                if tensor.shape != output_shape(old, rows) or tensor.kind != old.kind:
                    raise ValueError(f'Shape/tipo finale inatteso: {tensor.name}')
                consume(stream, tensor.offset - stream.tell(), (full_hash,))
                digest = hashlib.sha256()
                consume(stream, tensor.nbytes, (full_hash, digest))
                if digest.hexdigest() != hashes[tensor.name]:
                    raise ValueError(f'Byte non preservati per {tensor.name}')
            consume(stream, shard.stamp[2] - stream.tell(), (full_hash,))
        output_files.append({'name': shard.path.name, 'size_bytes': shard.stamp[2], 'sha256': full_hash.hexdigest()})
    original.unchanged()
    result.unchanged()
    return output_files


def peak_memory():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (key, ctypes.c_size_t) for key in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        psapi, kernel = ctypes.WinDLL('psapi'), ctypes.WinDLL('kernel32')
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return {'method': 'GetProcessMemoryInfo', 'peak_resident_bytes': counters.PeakWorkingSetSize}
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {'method': 'getrusage', 'peak_resident_bytes': peak * (1 if sys.platform == 'darwin' else 1024)}
    return {'method': 'unavailable', 'peak_resident_bytes': None}


def tool_identity():
    gguf = gguf_module()
    root = Path(gguf.__file__).parent
    return {'python': sys.version.split()[0], 'gguf_version': importlib.metadata.version('gguf'),
            'tested_gguf_revision': GGUF_REVISION,
            'gguf_source_sha256': {name: hash_file(root / name) for name in
                                  ('constants.py', 'gguf_reader.py', 'gguf_writer.py')},
            'gguf_prune_sha256': hash_file(Path(__file__).resolve())}


def write_json_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def compact(model, manifest, out):
    started = time.monotonic()
    original = Inventory(model)
    rows = validate_selection(original, manifest)
    target = Path(out).expanduser().absolute()
    if os.path.lexists(target):
        raise ValueError('La directory di output esiste già; scegliere una nuova destinazione.')
    target.parent.mkdir(parents=True, exist_ok=True)
    target = target.parent.resolve() / target.name
    with tempfile.TemporaryDirectory(prefix='.gguf-prune-', dir=target.parent) as tmp:
        stage = Path(tmp)
        writers, paths, expected_bytes = prepare_headers(original, rows, stage)
        try:
            allocated = sum(p.stat().st_size for p in paths)
            # Includes final padding and space for manifest/report; no full intermediate model.
            if shutil.disk_usage(stage).free < expected_bytes - allocated + (8 << 20):
                raise OSError(f'Spazio insufficiente: output previsto {expected_bytes / GIB:.2f} GiB, più report.')
            source_files = original.files(hashed=True)
            if source_files != manifest['source']['files']:
                raise ValueError('SHA-256 sorgente diverso dal manifesto; compattazione annullata.')
            hashes = copy_payloads(original, rows, writers)
        finally:
            for writer in writers:
                writer.close()
        result = Inventory(paths[0])
        output_files = verify_output(original, result, rows, hashes)
        if sum(f['size_bytes'] for f in output_files) != expected_bytes:
            raise ValueError('Dimensione scritta diversa da quella calcolata dalle intestazioni.')
        report = {'schema_version': 1, 'status': 'compacted_unvalidated',
                  'source': {**manifest['source'], 'files': source_files},
                  'output_files': output_files, 'entrypoint': paths[0].name,
                  'before': original.summary(), 'after': result.summary(),
                  'saved_file_bytes': original.summary()['file_bytes'] - result.summary()['file_bytes'],
                  'selection': manifest, 'mapping': 'layers[i].expert_ids[new_id] = old_id',
                  'tensor_payload_sha256': hashes, 'tools': tool_identity(),
                  'structural_validation': {'passed': True, 'source_hashes_verified': True,
                                            'all_output_payloads_verified': True, 'metadata_preserved': True},
                  'runtime_smoke': {'status': 'not_run'}, 'quality_validation': {'status': 'not_run'},
                  'memory': {**peak_memory(), 'copy_buffer_bytes': BUFFER_BYTES,
                             'note': 'Picco del processo; non è una stima della RAM di inferenza.'},
                  'elapsed_seconds': time.monotonic() - started}
        write_json_new(stage / 'selection.json', manifest)
        write_json_new(stage / 'report.json', report)
        original.unchanged()
        if os.path.lexists(target):
            raise ValueError('La destinazione è comparsa durante l\'operazione; output non promosso.')
        stage.rename(target)
    return report


def progress(message):
    print(message, file=sys.stderr, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inspector = sub.add_parser('inspect', help='Intestazioni, layout e stime; nessuna inferenza')
    inspector.add_argument('--model', type=Path, required=True, help='Primo shard locale')
    inspector.add_argument('--budgets', type=int, nargs='+', default=[])
    inspector.add_argument('--hash', action='store_true', help='Legge anche i payload per SHA-256')
    inspector.add_argument('--identity-selection', type=Path, help='Salva un manifesto identità; implica --hash')
    writer = sub.add_parser('compact', help='Copia gli esperti scelti, senza valutare la qualità')
    writer.add_argument('--model', type=Path, required=True)
    writer.add_argument('--selection', type=Path, required=True)
    writer.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'inspect':
        if args.identity_selection and os.path.lexists(args.identity_selection):
            raise ValueError('Il file di selezione esiste già.')
        inventory = Inventory(args.model)
        result = inventory.summary(args.budgets)
        hashed = bool(args.hash or args.identity_selection)
        files = inventory.files(hashed=hashed)
        result.update(source_files=files, source_hashes_verified=hashed,
                      runtime_smoke={'status': 'not_run'}, quality_validation={'status': 'not_run'})
        if args.identity_selection:
            write_json_new(args.identity_selection, identity_selection(inventory, files))
    else:
        manifest = json.loads(args.selection.read_text(encoding='utf-8-sig'))
        result = compact(args.model, manifest, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    try:
        main()
    except (ValueError, OSError, ImportError) as error:
        sys.exit(f'Errore: {error}')
    except KeyboardInterrupt:
        sys.exit('Interrotto: nessun output parziale è stato promosso.')
