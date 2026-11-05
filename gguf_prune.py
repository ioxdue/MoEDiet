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
import struct
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

ARCH = 'qwen4exp'
GGUF_REVISION = '19cce6919992fdffd92e0ac152caaf1e2b8f254e'
BUFFER_BYTES = 4 << 20
MAX_REMOTE_HEADER_BYTES = 64 << 20
MAX_REMOTE_DIRECTORY_ITEMS = 1_000_000
MAX_REMOTE_HEADER_RANGES = 20_000
GGUF_MAX_STRING_LENGTH = 1 << 30
GGUF_U64_MAX = (1 << 64) - 1
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


def grouped_ranges(ranges):
    """Sort and merge adjacent payload spans without including gaps."""
    ordered = list(ranges)
    if any(type(offset) is not int or type(count) is not int for offset, count in ordered):
        raise ValueError('Intervallo remoto non intero.')
    ordered.sort()
    merged = []
    for offset, count in ordered:
        if offset < 0 or count <= 0:
            raise ValueError('Intervallo remoto non valido.')
        end = offset + count
        if merged and offset < merged[-1][0] + merged[-1][1]:
            raise ValueError('Intervalli remoti sovrapposti.')
        if merged and offset == merged[-1][0] + merged[-1][1]:
            start, previous = merged[-1]
            merged[-1] = (start, previous + count)
        else:
            merged.append((offset, count))
    return merged


class HubRangeSource:
    """Fail-closed, bounded Range transport; URLs must identify immutable Hub revisions."""

    @staticmethod
    def probe_etag(url, size, opener=None, timeout=30):
        if type(size) is not int or size <= 0:
            raise ValueError('Dimensione sorgente Range non valida.')
        opener = opener or urllib.request.urlopen
        request = urllib.request.Request(
            url, headers={'Range': 'bytes=0-0', 'Accept-Encoding': 'identity'},
        )
        try:
            response = opener(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            raise ValueError(f'Probe Range rifiutato: status {exc.code}.') from exc
        with response:
            if getattr(response, 'status', None) != 206:
                raise ValueError('Probe Range richiede status 206.')
            if urllib.parse.urlsplit(response.geturl()).scheme.lower() != 'https':
                raise ValueError('Redirect HTTP non HTTPS rifiutato.')
            if response.headers.get('Content-Range') != f'bytes 0-0/{size}':
                raise ValueError('Content-Range probe diverso dalla dimensione Hub.')
            etag = response.headers.get('ETag')
            if not etag or etag.startswith('W/'):
                raise ValueError('Il file Hub non espone un ETag strong.')
            content_length = response.headers.get('Content-Length')
            if content_length is not None and content_length != '1':
                raise ValueError('Content-Length probe diverso da un byte.')
            if response.headers.get('Content-Encoding', 'identity').lower() not in ('', 'identity'):
                raise ValueError('Encoding HTTP trasformata non supportata.')
            if response.read(1) == b'' or response.read(1):
                raise ValueError('Body probe non contiene esattamente un byte.')
            return etag

    @classmethod
    def from_repo(cls, repo_id, revision, path, size, etag, opener=None, timeout=30):
        if (not isinstance(repo_id, str) or len(repo_id.split('/')) != 2 or
                any(not part or part in ('.', '..') for part in repo_id.split('/'))):
            raise ValueError('Repo ID Hub non valido.')
        if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40,64}', revision):
            raise ValueError('La revisione Hub deve essere un commit SHA completo.')
        if (not isinstance(path, str) or not path or path.startswith('/') or
                any(part in ('.', '..') for part in path.split('/'))):
            raise ValueError('Percorso file Hub non valido.')
        url = (f'https://huggingface.co/{urllib.parse.quote(repo_id, safe="/")}'
               f'/resolve/{revision}/{urllib.parse.quote(path, safe="/")}')
        return cls(url, size, etag, opener=opener, timeout=timeout)

    def __init__(self, url, size, etag, opener=None, timeout=30):
        if not isinstance(url, str):
            raise ValueError('URL sorgente Range non valido.')
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != 'https' or not parsed.netloc:
            raise ValueError('La sorgente Range deve usare un URL HTTPS.')
        if type(size) is not int or size <= 0:
            raise ValueError('Dimensione sorgente Range non valida.')
        if not isinstance(etag, str) or not etag or etag.startswith('W/'):
            raise ValueError('È richiesto un ETag strong per la sorgente Range.')
        if type(timeout) not in (int, float) or timeout <= 0:
            raise ValueError('Timeout HTTP non valido.')
        self.url = url
        self.size = size
        self.etag = etag
        self.opener = opener or urllib.request.urlopen
        self.timeout = timeout
        self.transferred_bytes = 0
        self.range_requests = 0

    def copy_range(self, offset, count, output, hashes=()):
        if (type(offset) is not int or type(count) is not int or
                offset < 0 or count <= 0 or offset + count > self.size):
            raise ValueError('Intervallo HTTP fuori dai limiti del file sorgente.')
        end = offset + count - 1
        request = urllib.request.Request(
            self.url,
            headers={'Range': f'bytes={offset}-{end}', 'If-Match': self.etag,
                     'Accept-Encoding': 'identity'},
        )
        try:
            response = self.opener(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise ValueError(f'Range HTTP rifiutato: status {exc.code}.') from exc
        with response:
            status = getattr(response, 'status', None) or response.getcode()
            content_range = response.headers.get('Content-Range')
            expected_range = f'bytes {offset}-{end}/{self.size}'
            if status != 206:
                raise ValueError(f'Range HTTP richiede status 206, ricevuto {status}.')
            if urllib.parse.urlsplit(response.geturl()).scheme.lower() != 'https':
                raise ValueError('Redirect HTTP non HTTPS rifiutato.')
            if response.headers.get('Content-Encoding', 'identity').lower() not in ('', 'identity'):
                raise ValueError('Encoding HTTP trasformata non supportata per i byte GGUF.')
            if content_range != expected_range:
                raise ValueError(f'Content-Range inatteso: {content_range!r}.')
            if response.headers.get('ETag') != self.etag:
                raise ValueError('ETag sorgente cambiato durante il trasferimento.')
            content_length = response.headers.get('Content-Length')
            if content_length is not None and content_length != str(count):
                raise ValueError('Content-Length diversa dall’intervallo richiesto.')
            remaining = count
            while remaining:
                block = response.read(min(remaining, BUFFER_BYTES))
                if not block:
                    raise ValueError('Body HTTP Range troncato.')
                if output.write(block) != len(block):
                    raise OSError('Scrittura incompleta del payload remoto.')
                for digest in hashes:
                    digest.update(block)
                self.transferred_bytes += len(block)
                remaining -= len(block)
            if response.read(1):
                raise ValueError('Body HTTP Range più lungo dell’intervallo richiesto.')
            self.range_requests += 1


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
    data_offset: int


@dataclass(frozen=True)
class HeaderTensor:
    name: str
    shape: tuple
    tensor_type: object
    data_offset: int
    n_bytes: int


class RangeGGUFReader:
    """Parse only GGUF v3 little-endian directory fields through exact ranges."""

    SCALARS = {
        0: ('B', int), 1: ('b', int), 2: ('H', int), 3: ('h', int),
        4: ('I', int), 5: ('i', int), 6: ('f', float), 7: ('?', bool),
        10: ('Q', int), 11: ('q', int), 12: ('d', float),
    }

    def __init__(self, source, max_header_bytes=MAX_REMOTE_HEADER_BYTES,
                 max_directory_items=MAX_REMOTE_DIRECTORY_ITEMS):
        if (type(max_header_bytes) is not int or
                not 24 <= max_header_bytes <= MAX_REMOTE_HEADER_BYTES):
            raise ValueError('Limite header remoto non valido.')
        if (type(max_directory_items) is not int or
                not 1 <= max_directory_items <= MAX_REMOTE_DIRECTORY_ITEMS):
            raise ValueError('Limite elementi directory remoto non valido.')
        if type(source.size) is not int or not 0 < source.size <= GGUF_U64_MAX:
            raise ValueError('Dimensione sorgente GGUF non valida.')
        self.source = source
        self.max_header_bytes = max_header_bytes
        self.max_directory_items = max_directory_items
        self.header_bytes = 0
        self.range_requests = 0
        self.item_count = 0
        self.offset = 0
        self.gguf = gguf_module()
        self.alignment = self.gguf.GGUF_DEFAULT_ALIGNMENT
        self.metadata = {}
        self.tensors = []

        if self._read(4) != b'GGUF':
            raise ValueError('GGUF magic invalid')
        version = self._u32()
        if version != 3:
            raise ValueError(f'Versione GGUF non supportata: {version}; richiesto v3 little-endian.')
        tensor_count, kv_count = self._u64(), self._u64()
        if tensor_count + kv_count > max_directory_items:
            raise ValueError('Conteggio GGUF oltre il limite elementi remoto.')
        self._count_items(tensor_count + kv_count)

        for _ in range(kv_count):
            key = self._string()
            if key in self.metadata:
                raise ValueError(f'Campo GGUF duplicato: {key}')
            kind = self._u32()
            digest = hashlib.sha256(struct.pack('<I', kind))
            value, subtype = self._value(kind, digest)
            self.metadata[key] = Metadata(value, kind, subtype, digest.hexdigest())

        pending = []
        seen_names = set()
        for _ in range(tensor_count):
            name = self._string()
            if name in seen_names:
                raise ValueError(f'Tensore GGUF duplicato: {name}')
            seen_names.add(name)
            n_dims = self._u32()
            if not 1 <= n_dims <= self.gguf.GGML_MAX_DIMS:
                raise ValueError(f'Numero dimensioni GGUF non valido: {name}')
            shape = tuple(self._u64() for _ in range(n_dims))
            if any(dimension <= 0 for dimension in shape):
                raise ValueError(f'Dimensione tensor GGUF non valida: {name}')
            raw_type = self._u32()
            try:
                tensor_type = self.gguf.GGMLQuantizationType(raw_type)
                block, type_size = self.gguf.GGML_QUANT_SIZES[tensor_type]
            except (KeyError, ValueError) as exc:
                raise ValueError(f'Tipo tensor GGUF non supportato: {name}') from exc
            if type(block) is not int or type_size <= 0 or block <= 0 or shape[0] % block:
                raise ValueError(f'Dimensioni/blocchi incompatibili: {name}')
            elements = 1
            for dimension in shape:
                if elements > GGUF_U64_MAX // dimension:
                    raise ValueError(f'Numero elementi GGUF in overflow: {name}')
                elements *= dimension
            if elements % block:
                raise ValueError(f'Dimensioni/blocchi incompatibili: {name}')
            n_bytes = elements // block * type_size
            if n_bytes > GGUF_U64_MAX:
                raise ValueError(f'Dimensione tensor GGUF in overflow: {name}')
            relative_offset = self._u64()
            pending.append((name, shape, tensor_type, relative_offset, n_bytes))

        alignment_field = self.metadata.get('general.alignment')
        if alignment_field is not None:
            if alignment_field.kind != int(self.gguf.GGUFValueType.UINT32):
                raise ValueError('Tipo general.alignment non valido.')
            self.alignment = alignment_field.value
        if (type(self.alignment) is not int or self.alignment <= 0 or
                self.alignment & (self.alignment - 1)):
            raise ValueError('Allineamento GGUF non valido.')
        self.directory_end = self.offset
        self.data_offset = (self.directory_end + self.alignment - 1) & -self.alignment
        if self.data_offset > source.size:
            raise ValueError('Data start GGUF fuori dal file.')

        previous_end = self.data_offset
        absolute_offsets = {}
        for name, _, _, relative_offset, n_bytes in sorted(pending, key=lambda row: row[3]):
            if relative_offset % self.alignment:
                raise ValueError(f'Offset tensor non allineato: {name}')
            if relative_offset > GGUF_U64_MAX - self.data_offset:
                raise ValueError(f'Offset tensor GGUF in overflow: {name}')
            tensor_offset = self.data_offset + relative_offset
            if tensor_offset < previous_end or tensor_offset > source.size or n_bytes > source.size - tensor_offset:
                raise ValueError(f'Offset, sovrapposizione o dimensione fuori file: {name}')
            absolute_offsets[name] = tensor_offset
            previous_end = tensor_offset + n_bytes
        self.tensors = [
            HeaderTensor(name, shape, tensor_type, absolute_offsets[name], n_bytes)
            for name, shape, tensor_type, _, n_bytes in pending
        ]

    def _count_items(self, count):
        if count > self.max_directory_items - self.item_count:
            raise ValueError('Directory GGUF oltre il limite elementi remoto.')
        self.item_count += count

    def _read(self, count):
        if type(count) is not int or count < 0:
            raise ValueError('Lunghezza GGUF non valida.')
        if count == 0:
            return b''
        if self.offset > self.source.size or count > self.source.size - self.offset:
            raise ValueError('Directory GGUF troncata oltre la dimensione del file.')
        if count > self.max_header_bytes - self.header_bytes:
            raise ValueError('Directory GGUF oltre il limite header remoto.')
        if self.range_requests >= MAX_REMOTE_HEADER_RANGES:
            raise ValueError('Directory GGUF oltre il limite richieste Range remoto.')
        import io
        output = io.BytesIO()
        self.range_requests += 1
        self.source.copy_range(self.offset, count, output)
        data = output.getvalue()
        if len(data) != count:
            raise ValueError('Lettura header GGUF incompleta.')
        self.offset += count
        self.header_bytes += count
        return data

    def _u32(self):
        return struct.unpack('<I', self._read(4))[0]

    def _u64(self):
        return struct.unpack('<Q', self._read(8))[0]

    def _string(self):
        length = self._u64()
        if length > GGUF_MAX_STRING_LENGTH:
            raise ValueError('Stringa GGUF oltre il limite previsto dalla specifica.')
        return self._read(length).decode('utf-8')

    def _value(self, kind, digest):
        if kind == int(self.gguf.GGUFValueType.STRING):
            return self._value_string(digest), None
        if kind == int(self.gguf.GGUFValueType.ARRAY):
            raw_subtype = self._read(4)
            digest.update(raw_subtype)
            subtype = struct.unpack('<I', raw_subtype)[0]
            length_bytes = self._read(8)
            digest.update(length_bytes)
            length = struct.unpack('<Q', length_bytes)[0]
            if (subtype == int(self.gguf.GGUFValueType.ARRAY) or
                    subtype not in self.SCALARS and subtype != int(self.gguf.GGUFValueType.STRING)):
                raise ValueError('Tipo array GGUF annidato o non supportato.')
            if length == 0:
                raise ValueError('Array GGUF vuoto non supportato da questo writer.')
            self._count_items(length)
            if subtype == int(self.gguf.GGUFValueType.STRING):
                values = []
                for _ in range(length):
                    value, raw = self._value_string_with_raw()
                    digest.update(raw)
                    values.append(value)
            else:
                fmt, convert = self.SCALARS[subtype]
                item_size = struct.calcsize('<' + fmt)
                raw_values = self._read(length * item_size)
                digest.update(raw_values)
                values = [convert(value[0]) for value in struct.iter_unpack('<' + fmt, raw_values)]
            return values, subtype
        if kind in self.SCALARS:
            value, raw = self._scalar(kind)
            digest.update(raw)
            return value, None
        raise ValueError(f'Tipo valore GGUF non supportato: {kind}')

    def _value_string(self, digest):
        value, raw = self._value_string_with_raw()
        digest.update(raw)
        return value

    def _value_string_with_raw(self):
        length_bytes = self._read(8)
        length = struct.unpack('<Q', length_bytes)[0]
        if length > GGUF_MAX_STRING_LENGTH:
            raise ValueError('Stringa GGUF oltre il limite previsto dalla specifica.')
        content = self._read(length)
        return content.decode('utf-8'), length_bytes + content

    def _scalar(self, kind):
        fmt, convert = self.SCALARS[kind]
        raw = self._read(struct.calcsize('<' + fmt))
        return convert(struct.unpack('<' + fmt, raw)[0]), raw


def shard_from_reader(reader, path, index, stamp):
    gguf = gguf_module()
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
    return Shard(path, stamp, metadata, tensors, alignment, int(reader.data_offset))


def read_shard(path, index):
    gguf = gguf_module()
    reader = None
    stamp = file_stamp(path)
    try:
        reader = gguf.GGUFReader(path, 'r')
        return shard_from_reader(reader, path, index, stamp)
    except (IndexError, KeyError, OverflowError) as exc:
        raise ValueError(f'GGUF malformato {path.name}: {exc}') from exc
    finally:
        if reader is not None:
            reader.data._mmap.close()


def read_remote_shard(path, index, source, max_header_bytes,
                      max_directory_items=MAX_REMOTE_DIRECTORY_ITEMS):
    try:
        reader = RangeGGUFReader(source, max_header_bytes, max_directory_items)
        tensors = [
            Tensor(tensor.name, tensor.shape, int(tensor.tensor_type), tensor.tensor_type.name,
                   tensor.data_offset, tensor.n_bytes, index)
            for tensor in reader.tensors
        ]
        return Shard(Path(path), (0, 0, source.size, 0, 0),
                     reader.metadata, tensors, reader.alignment, reader.data_offset)
    except (IndexError, KeyError, OverflowError) as exc:
        raise ValueError(f'GGUF remoto malformato {Path(path).name}: {exc}') from exc


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
        self.is_remote = False
        self._init_from_shards([read_shard(path, i) for i, path in enumerate(paths)])

    def _init_from_shards(self, shards):
        self.shards = shards
        paths = [shard.path for shard in shards]
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


def hub_tree_files(repo_id, revision, subpath, opener=None, timeout=30):
    if (not isinstance(repo_id, str) or len(repo_id.split('/')) != 2 or
            any(not part or part in ('.', '..') for part in repo_id.split('/'))):
        raise ValueError('Repo ID Hub non valido.')
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40,64}', revision):
        raise ValueError('La revisione Hub deve essere un commit SHA completo.')
    if (not isinstance(subpath, str) or not subpath or subpath.startswith('/') or
            any(part in ('.', '..') for part in subpath.split('/'))):
        raise ValueError('Percorso GGUF Hub non valido.')
    url = (f'https://huggingface.co/api/models/{urllib.parse.quote(repo_id, safe="/")}'
           f'/tree/{revision}/{urllib.parse.quote(subpath, safe="/")}?recursive=false&expand=true')
    request = urllib.request.Request(url, headers={'Accept': 'application/json'})
    try:
        response = (opener or urllib.request.urlopen)(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise ValueError(f'API Hub non disponibile: status {exc.code}.') from exc
    with response:
        if getattr(response, 'status', None) != 200:
            raise ValueError('API Hub non ha restituito status 200.')
        if urllib.parse.urlsplit(response.geturl()).scheme.lower() != 'https':
            raise ValueError('Redirect API Hub non HTTPS rifiutato.')
        payload = response.read((2 << 20) + 1)
        if len(payload) > (2 << 20):
            raise ValueError('Risposta API Hub oltre il limite di 2 MiB.')
    try:
        entries = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('JSON API Hub non valido.') from exc
    if not isinstance(entries, list):
        raise ValueError('La risposta tree API Hub non è una lista.')
    prefix = subpath.rstrip('/') + '/'
    files = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get('type') != 'file':
            continue
        path = entry.get('path')
        size = entry.get('size')
        oid = entry.get('oid')
        if not isinstance(path, str) or not path.startswith(prefix):
            continue
        relative = path[len(prefix):]
        if '/' in relative or not relative.endswith('.gguf'):
            continue
        if (type(size) is not int or size <= 0 or not isinstance(oid, str) or
                not re.fullmatch(r'[0-9a-f]{40,64}', oid)):
            raise ValueError(f'Metadata API Hub non valida per {path}.')
        files.append({'path': path, 'name': relative, 'size_bytes': size, 'oid': oid})
    if not files:
        raise ValueError('La directory Hub non contiene shard GGUF.')
    if len(files) == 1 and not SHARD_RE.fullmatch(files[0]['name']):
        return files
    matches = [SHARD_RE.fullmatch(row['name']) for row in files]
    if any(match is None for match in matches):
        raise ValueError('La directory deve contenere un solo GGUF o un gruppo shard completo.')
    identities = {(match[1], int(match[3])) for match in matches}
    if len(identities) != 1:
        raise ValueError('La directory contiene più gruppi GGUF.')
    prefix_name, count = next(iter(identities))
    expected_names = [f'{prefix_name}-{index:05d}-of-{count:05d}.gguf'
                      for index in range(1, count + 1)]
    by_name = {row['name']: row for row in files}
    if len(files) != count or set(by_name) != set(expected_names):
        raise ValueError('Gruppo shard GGUF incompleto o duplicato nella directory Hub.')
    return [by_name[name] for name in expected_names]


class RemoteInventory(Inventory):
    def __init__(self, repo_id, revision, subpath, max_header_bytes=MAX_REMOTE_HEADER_BYTES,
                 max_directory_items=MAX_REMOTE_DIRECTORY_ITEMS):
        if (type(max_header_bytes) is not int or
                not 24 <= max_header_bytes <= MAX_REMOTE_HEADER_BYTES):
            raise ValueError('Limite header remoto non valido.')
        if (type(max_directory_items) is not int or
                not 1 <= max_directory_items <= MAX_REMOTE_DIRECTORY_ITEMS):
            raise ValueError('Limite elementi directory remoto non valido.')
        self.is_remote = True
        self.repo_id = repo_id
        self.revision = revision
        self.subpath = subpath
        self.max_header_bytes = max_header_bytes
        self.max_directory_items = max_directory_items
        self.remote_files = hub_tree_files(repo_id, revision, subpath)
        self.sources = []
        shards = []
        for index, row in enumerate(self.remote_files):
            url = (f'https://huggingface.co/{urllib.parse.quote(repo_id, safe="/")}'
                   f'/resolve/{revision}/{urllib.parse.quote(row["path"], safe="/")}')
            etag = HubRangeSource.probe_etag(url, row['size_bytes'])
            source = HubRangeSource.from_repo(
                repo_id, revision, row['path'], row['size_bytes'], etag,
            )
            row['etag'] = etag
            self.sources.append(source)
            shards.append(read_remote_shard(
                row['name'], index, source, max_header_bytes, max_directory_items,
            ))
        self._init_from_shards(shards)

    def unchanged(self):
        current = hub_tree_files(self.repo_id, self.revision, self.subpath)
        expected = [{key: row[key] for key in ('path', 'name', 'size_bytes', 'oid')}
                    for row in self.remote_files]
        if current != expected:
            raise ValueError('Identità Hub sorgente diversa dall’inventario iniziale.')

    def files(self, hashed=False):
        if hashed:
            raise ValueError('Lo SHA-256 integrale non è disponibile senza leggere tutti i byte sorgente.')
        self.unchanged()
        return [dict(row) for row in self.remote_files]


def identity_selection(inventory, files):
    source = {'files': files}
    if inventory.is_remote:
        source.update(repo_id=inventory.repo_id, revision=inventory.revision)
    return {'schema_version': 1, 'architecture': ARCH, 'source': source,
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
    if inventory.is_remote:
        if (source.get('repo_id') != inventory.repo_id or
                source.get('revision') != inventory.revision or files != inventory.remote_files):
            raise ValueError('Identità Hub del manifesto diversa dalla sorgente fissata.')
    else:
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
    for shard_index, (shard, writer) in enumerate(zip(inventory.shards, writers)):
        progress(f'Copia: {shard.path.name}')
        output = writer.fout[0]
        source = None if inventory.is_remote else shard.path.open('rb')
        try:
            for tensor in shard.tensors:
                writer.write_padding(output, output.tell())
                digest = hashlib.sha256()
                nbytes = 0
                spans = grouped_ranges(payload_ranges(tensor, rows)) if inventory.is_remote else payload_ranges(tensor, rows)
                for offset, count in spans:
                    if inventory.is_remote:
                        inventory.sources[shard_index].copy_range(offset, count, output, (digest,))
                    else:
                        source.seek(offset)
                        consume(source, count, (digest,), output)
                    nbytes += count
                writer.write_padding(output, nbytes)
                hashes[tensor.name] = digest.hexdigest()
            output.flush()
            os.fsync(output.fileno())
        finally:
            if source is not None:
                source.close()
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


def compact_inventory(original, manifest, out):
    started = time.monotonic()
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
            source_files = original.files(hashed=not original.is_remote)
            if source_files != manifest['source']['files']:
                raise ValueError('Identità sorgente diversa dal manifesto; compattazione annullata.')
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
                  'structural_validation': {'passed': True, 'source_hashes_verified': not original.is_remote,
                                            'source_identity_verified': True,
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


def compact(model, manifest, out):
    return compact_inventory(Inventory(model), manifest, out)


def compact_remote(repo_id, revision, subpath, manifest, out,
                   max_header_bytes=MAX_REMOTE_HEADER_BYTES,
                   max_directory_items=MAX_REMOTE_DIRECTORY_ITEMS):
    return compact_inventory(
        RemoteInventory(repo_id, revision, subpath, max_header_bytes, max_directory_items),
        manifest, out,
    )


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
    remote_inspector = sub.add_parser('inspect-hub', help='Ispeziona shard Hub fissati senza scaricare payload')
    remote_inspector.add_argument('--repo-id', required=True, help='Repo pubblico owner/name')
    remote_inspector.add_argument('--revision', required=True, help='Commit SHA completo')
    remote_inspector.add_argument('--path', required=True, help='Directory contenente il GGUF o il gruppo shard')
    remote_inspector.add_argument('--budgets', type=int, nargs='+', default=[])
    remote_inspector.add_argument('--identity-selection', type=Path)
    remote_inspector.add_argument('--max-header-bytes', type=int, default=MAX_REMOTE_HEADER_BYTES,
                                  help='Budget esatto di byte header per shard (default: 64 MiB)')
    remote_inspector.add_argument('--max-directory-items', type=int, default=MAX_REMOTE_DIRECTORY_ITEMS,
                                  help='Budget per KV, tensori ed elementi array per shard')
    remote_writer = sub.add_parser('compact-hub', help='Compattazione Hub via HTTP Range')
    remote_writer.add_argument('--repo-id', required=True)
    remote_writer.add_argument('--revision', required=True, help='Commit SHA completo')
    remote_writer.add_argument('--path', required=True, help='Directory contenente il GGUF o il gruppo shard')
    remote_writer.add_argument('--selection', type=Path, required=True)
    remote_writer.add_argument('--out', type=Path, required=True)
    remote_writer.add_argument('--max-header-bytes', type=int, default=MAX_REMOTE_HEADER_BYTES,
                               help='Budget esatto di byte header per shard (default: 64 MiB)')
    remote_writer.add_argument('--max-directory-items', type=int, default=MAX_REMOTE_DIRECTORY_ITEMS,
                               help='Budget per KV, tensori ed elementi array per shard')
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
    elif args.command == 'compact':
        manifest = json.loads(args.selection.read_text(encoding='utf-8-sig'))
        result = compact(args.model, manifest, args.out)
    elif args.command == 'inspect-hub':
        if args.identity_selection and os.path.lexists(args.identity_selection):
            raise ValueError('Il file di selezione esiste già.')
        inventory = RemoteInventory(args.repo_id, args.revision, args.path,
                                    args.max_header_bytes, args.max_directory_items)
        files = inventory.files()
        result = inventory.summary(args.budgets)
        result.update(source_files=files, source_hashes_verified=False,
                      source_identity_verified=True, runtime_smoke={'status': 'not_run'},
                      quality_validation={'status': 'not_run'})
        if args.identity_selection:
            write_json_new(args.identity_selection, identity_selection(inventory, files))
    else:
        manifest = json.loads(args.selection.read_text(encoding='utf-8-sig'))
        result = compact_remote(args.repo_id, args.revision, args.path, manifest, args.out,
                                args.max_header_bytes, args.max_directory_items)
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
