"""Inspect safetensors headers without reading tensor payloads."""

import json
from pathlib import Path
import re
import struct

MAX_HEADER_BYTES = 64 << 20
MAX_INDEX_BYTES = 64 << 20
MAX_CONFIG_BYTES = 4 << 20
MAX_SHARDS = 1024
MAX_TENSORS = 100_000
MAX_TOTAL_HEADER_BYTES = 128 << 20
DTYPE_BYTES = {
    'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1, 'F8_E4M3FN': 1,
    'F8_E4M3FNUZ': 1, 'F8_E5M2FNUZ': 1,
    'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2,
    'I32': 4, 'U32': 4, 'F32': 4,
    'I64': 8, 'U64': 8, 'F64': 8,
}


def _read_json(path, limit):
    size = path.stat().st_size
    if size > limit:
        raise ValueError(f'Metadati troppo grandi: {path.name}.')
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f'Metadati JSON non validi: {path.name}.') from error


def _parse_header(header_bytes, size, file_name):
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f'Header safetensors JSON non valido: {file_name}.') from error
    if not isinstance(header, dict):
        raise ValueError(f'Header safetensors non valido: {file_name}.')
    tensors = {}
    ranges = []
    payload_size = size - 8 - len(header_bytes)
    for name, metadata in header.items():
        if name == '__metadata__':
            if not isinstance(metadata, dict) or any(
                    not isinstance(key, str) or not isinstance(value, str)
                    for key, value in metadata.items()):
                raise ValueError(f'Metadati safetensors non validi: {file_name}.')
            continue
        if not isinstance(name, str) or not name or not isinstance(metadata, dict):
            raise ValueError(f'Descrittore tensore non valido: {file_name}.')
        if set(metadata) != {'dtype', 'shape', 'data_offsets'}:
            raise ValueError(f'Campi del tensore non validi ({name}): {file_name}.')
        dtype, shape, offsets = metadata['dtype'], metadata['shape'], metadata['data_offsets']
        if (not isinstance(dtype, str) or dtype not in DTYPE_BYTES
                or not isinstance(shape, list)
                or any(type(dimension) is not int or dimension < 0 for dimension in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)):
            raise ValueError(f'Metadati del tensore non validi ({name}): {file_name}.')
        start, end = offsets
        if start < 0 or end < start or end > payload_size:
            raise ValueError(f'Offset del tensore fuori intervallo ({name}): {file_name}.')
        elements = 1
        for dimension in shape:
            elements *= dimension
        if end - start != elements * DTYPE_BYTES[dtype]:
            raise ValueError(f'Forma e dimensione payload discordanti ({name}): {file_name}.')
        tensors[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': offsets,
                         'payload_bytes': end - start}
        ranges.append((start, end, name))

    cursor = 0
    for start, end, name in sorted(ranges):
        if start != cursor:
            raise ValueError(f'Offset safetensors sovrapposti o discontinui ({name}): {file_name}.')
        cursor = end
    if cursor != payload_size:
        raise ValueError(f'Payload safetensors non coperto dall\'header: {file_name}.')
    return tensors


def _read_header(path):
    size = path.stat().st_size
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f'Header safetensors troncato: {path.name}.')
        header_size, = struct.unpack('<Q', prefix)
        if header_size < 2 or header_size > MAX_HEADER_BYTES or header_size > size - 8:
            raise ValueError(f'Dimensione header safetensors non valida: {path.name}.')
        header_bytes = stream.read(header_size)
    return _parse_header(header_bytes, size, path.name)


def inspect_safetensors(directory):
    """Return a bounded metadata inventory for a local safetensors checkpoint."""
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f'Directory modello non trovata: {directory}')

    index_path = root / 'model.safetensors.index.json'
    if index_path.is_file():
        index = _read_json(index_path, MAX_INDEX_BYTES)
        weight_map = index.get('weight_map') if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError('Indice safetensors privo di weight_map valido.')
        if any(not isinstance(key, str) or not key or not isinstance(shard, str)
               or Path(shard).name != shard or not shard.endswith('.safetensors')
               for key, shard in weight_map.items()):
            raise ValueError('Indice safetensors contiene chiavi o nomi shard non validi.')
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = sorted(path.name for path in root.glob('*.safetensors'))
        if not shard_names:
            raise ValueError(f'Nessun file safetensors trovato in {root}.')
        weight_map = None
    if len(shard_names) > MAX_SHARDS:
        raise ValueError('Numero di shard safetensors oltre il limite di ispezione.')

    tensors = {}
    shards = []
    total_header_bytes = 0
    for name in shard_names:
        path = root / name
        if not path.is_file() or path.resolve().parent != root:
            raise ValueError(f'Shard safetensors non valido o mancante: {name}.')
        size = path.stat().st_size
        if size < 8:
            raise ValueError(f'Header safetensors troncato: {name}.')
        with path.open('rb') as stream:
            prefix = stream.read(8)
        header_size, = struct.unpack('<Q', prefix)
        total_header_bytes += header_size
        if total_header_bytes > MAX_TOTAL_HEADER_BYTES:
            raise ValueError('Dimensione totale degli header oltre il limite di ispezione.')
        shard_tensors = _read_header(path)
        shards.append({'file': name, 'tensor_count': len(shard_tensors),
                       'payload_bytes': sum(item['payload_bytes'] for item in shard_tensors.values())})
        for tensor_name, metadata in shard_tensors.items():
            if len(tensors) >= MAX_TENSORS:
                raise ValueError('Numero di tensori oltre il limite di ispezione.')
            if tensor_name in tensors:
                raise ValueError(f'Tensore duplicato tra shard: {tensor_name}.')
            if weight_map is not None and weight_map.get(tensor_name) != name:
                raise ValueError(f'Indice safetensors incoerente per il tensore: {tensor_name}.')
            tensors[tensor_name] = {'shard': name, **metadata}

    if weight_map is not None and set(weight_map) != set(tensors):
        raise ValueError('L\'indice safetensors non corrisponde agli header degli shard.')
    return {'shards': shards, 'tensor_count': len(tensors),
            'payload_bytes': sum(item['payload_bytes'] for item in tensors.values()),
            'tensors': tensors, 'payloads_read': False}


def _hub_fetch(repo_id, filename, revision, token, limit, byte_range=None):
    from huggingface_hub import hf_hub_url
    import requests

    url = hf_hub_url(repo_id, filename, revision=revision)
    headers = {'Accept-Encoding': 'identity'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    if byte_range is not None:
        start, end = byte_range
        headers['Range'] = f'bytes={start}-{end}'
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=(10, 60))
    except requests.RequestException as error:
        raise ValueError(f'Errore di rete nel recupero Hub di {filename}.') from error
    try:
        expected_status = 206 if byte_range is not None else 200
        if response.status_code != expected_status:
            raise ValueError(f'Risposta Hub non valida per {filename}: HTTP {response.status_code}.')
        if byte_range is not None:
            start, end = byte_range
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if not match:
                raise ValueError(f'Content-Range Hub non valido per {filename}.')
            range_start, range_end, total_size = map(int, match.groups())
            if range_start != start or range_end != end or range_end >= total_size:
                raise ValueError(f'Content-Range Hub non valido per {filename}.')
            expected_size = end - start + 1
            if expected_size > limit:
                raise ValueError(f'Range Hub oltre il limite di lettura: {filename}.')
        else:
            expected_size = None
            content_length = response.headers.get('Content-Length')
            if content_length and int(content_length) > limit:
                raise ValueError(f'Metadati Hub troppo grandi: {filename}.')

        chunks = []
        size = 0
        try:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                size += len(chunk)
                if size > limit or (expected_size is not None and size > expected_size):
                    raise ValueError(f'Dati Hub oltre il limite atteso: {filename}.')
                chunks.append(chunk)
        except requests.RequestException as error:
            raise ValueError(f'Errore leggendo la risposta Hub di {filename}.') from error
        if expected_size is not None and size != expected_size:
            raise ValueError(f'Range Hub troncato per {filename}.')
        return b''.join(chunks), total_size if byte_range is not None else None
    finally:
        response.close()


def inspect_hub_safetensors(repo_id, revision=None, token=None):
    """Inspect a Hub checkpoint using bounded metadata reads and safetensors byte ranges."""
    from huggingface_hub import HfApi, get_token

    token = token or get_token()
    source = HfApi(token=token).model_info(repo_id, revision=revision)
    if not source.sha:
        raise ValueError('Commit del modello Hub sorgente non disponibile.')
    siblings = {item.rfilename: item for item in source.siblings}
    if 'config.json' not in siblings:
        raise ValueError('config.json non trovato nel checkpoint Hub.')
    config_bytes, _ = _hub_fetch(repo_id, 'config.json', source.sha, token, MAX_CONFIG_BYTES)
    try:
        model_config = json.loads(config_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('config.json Hub non valido.') from error
    if not isinstance(model_config, dict):
        raise ValueError('config.json Hub deve contenere un oggetto JSON.')

    index_name = 'model.safetensors.index.json'
    if index_name in siblings:
        index_bytes, _ = _hub_fetch(repo_id, index_name, source.sha, token, MAX_INDEX_BYTES)
        try:
            index = json.loads(index_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError('Indice safetensors Hub non valido.') from error
        weight_map = index.get('weight_map') if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError('Indice safetensors Hub privo di weight_map valido.')
        if any(not isinstance(key, str) or not key or not isinstance(shard, str)
               or Path(shard).name != shard or not shard.endswith('.safetensors')
               for key, shard in weight_map.items()):
            raise ValueError('Indice safetensors Hub contiene chiavi o nomi shard non validi.')
        shard_names = sorted(set(weight_map.values()))
    elif 'model.safetensors' in siblings:
        extra_shards = {name for name in siblings if name.endswith('.safetensors')} - {'model.safetensors'}
        if extra_shards:
            raise ValueError('Checkpoint Hub con più shard privo di indice safetensors.')
        weight_map = None
        shard_names = ['model.safetensors']
    else:
        raise ValueError('Indice o file model.safetensors non trovato nel checkpoint Hub.')
    if len(shard_names) > MAX_SHARDS:
        raise ValueError('Numero di shard safetensors oltre il limite di ispezione.')

    tensors = {}
    shards = []
    total_header_bytes = 0
    for name in shard_names:
        sibling = siblings.get(name)
        if sibling is None:
            raise ValueError(f'Shard safetensors Hub mancante: {name}.')
        prefix, total_size = _hub_fetch(repo_id, name, source.sha, token, 8, (0, 7))
        if len(prefix) != 8:
            raise ValueError(f'Header safetensors troncato: {name}.')
        header_size, = struct.unpack('<Q', prefix)
        if header_size < 2 or header_size > MAX_HEADER_BYTES or header_size > total_size - 8:
            raise ValueError(f'Dimensione header safetensors non valida: {name}.')
        total_header_bytes += header_size
        if total_header_bytes > MAX_TOTAL_HEADER_BYTES:
            raise ValueError('Dimensione totale degli header oltre il limite di ispezione.')
        header_bytes, header_total = _hub_fetch(repo_id, name, source.sha, token, header_size,
                                                (8, 7 + header_size))
        if header_total != total_size or (sibling.size is not None and sibling.size != total_size):
            raise ValueError(f'Dimensione shard Hub incoerente: {name}.')
        shard_tensors = _parse_header(header_bytes, total_size, name)
        shards.append({'file': name, 'tensor_count': len(shard_tensors),
                       'payload_bytes': sum(item['payload_bytes'] for item in shard_tensors.values())})
        for tensor_name, metadata in shard_tensors.items():
            if len(tensors) >= MAX_TENSORS:
                raise ValueError('Numero di tensori oltre il limite di ispezione.')
            if tensor_name in tensors:
                raise ValueError(f'Tensore duplicato tra shard: {tensor_name}.')
            if weight_map is not None and weight_map.get(tensor_name) != name:
                raise ValueError(f'Indice safetensors Hub incoerente per il tensore: {tensor_name}.')
            tensors[tensor_name] = {'shard': name, **metadata}

    if weight_map is not None and set(weight_map) != set(tensors):
        raise ValueError('L\'indice safetensors Hub non corrisponde agli header degli shard.')
    return {'shards': shards, 'tensor_count': len(tensors),
            'payload_bytes': sum(item['payload_bytes'] for item in tensors.values()),
            'tensors': tensors, 'payloads_read': False,
            'source_revision': source.sha, 'model_config': model_config}
