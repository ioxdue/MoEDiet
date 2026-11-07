"""Pinned CPU tools for GGUF conversion and validation, also on Windows ARM64."""

import hashlib
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

COMMIT = 'a7a98e0fffed794396b3fbad4dcdbbc184963645'  # llama.cpp b6500
ARCHIVE_SHA256 = '4d78e6aa4a9124b58dff994416525294fbab2990fe905a640d4cbd26bf563a31'
TARGETS = ('llama-quantize', 'llama-cli', 'llama-server')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 << 20):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Tools:
    source: Path
    quantize: Path
    cli: Path
    server: Path
    revision: str


def installed(source, binary_dir=None, revision='user-supplied'):
    source = Path(os.path.expandvars(str(source))).expanduser().resolve()
    if not (source / 'convert_hf_to_gguf.py').is_file() or not (source / 'gguf-py' / 'gguf').is_dir():
        raise ValueError(f'Checkout llama.cpp incompleto: {source}')
    dirs = [Path(os.path.expandvars(str(binary_dir))).expanduser().resolve()] if binary_dir else [
        source / 'build/bin/Release', source / 'build/bin', source / 'bin']
    suffix = '.exe' if os.name == 'nt' else ''
    for directory in dirs:
        bins = [directory / (name + suffix) for name in TARGETS]
        if all(file.is_file() for file in bins):
            return Tools(source, *bins, revision)
    raise ValueError('Servono llama-quantize e llama-cli della stessa build. Verificare bin_dir.')


def get_tools(source=None, binary_dir=None):
    if source:
        return installed(source, binary_dir)
    if binary_dir:
        raise ValueError('bin_dir richiede llama_dir.')
    tools_cache = os.environ.get('MOEDIET_TOOLS_CACHE')
    if tools_cache:
        root = Path(os.path.expandvars(str(tools_cache))).expanduser().resolve()
    else:
        root = (Path.home() / '.cache' / 'moediet').resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f'llama-{COMMIT[:12]}-{platform.system()}-{platform.machine()}'
    if target.exists():
        return installed(target, revision=COMMIT)
    if not shutil.which('cmake'):
        raise ValueError('Installare CMake e un compilatore C++, oppure configurare llama_dir e bin_dir.')
    print('Preparazione una tantum degli strumenti llama.cpp b6500...', flush=True)
    with tempfile.TemporaryDirectory(prefix='.tools-', dir=root) as tmp:
        work = Path(tmp)
        archive = work / 'llama.tar.gz'
        url = f'https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{COMMIT}'
        with urllib.request.urlopen(url, timeout=120) as response, archive.open('wb') as output:
            shutil.copyfileobj(response, output)
        if sha256(archive) != ARCHIVE_SHA256:
            raise ValueError('SHA-256 inatteso per i sorgenti llama.cpp.')
        with tarfile.open(archive) as tar:
            tar.extractall(work, filter='data')
        src, build = work / ('llama.cpp-' + COMMIT), work / 'build'
        flags = ['-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DLLAMA_CURL=OFF',
                 '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_SERVER=ON',
                 '-DBUILD_SHARED_LIBS=OFF']
        if os.name == 'nt' and platform.machine().upper() in ('ARM64', 'AARCH64'):
            flags += ['-G', 'Visual Studio 17 2022', '-A', 'ARM64']
        subprocess.run(['cmake', '-S', str(src), '-B', str(build), *flags], check=True)
        subprocess.run(['cmake', '--build', str(build), '--config', 'Release', '--parallel', '4',
                        '--target', *TARGETS], check=True)
        compiled = build / 'bin'
        if (compiled / 'Release').is_dir():
            compiled = compiled / 'Release'
        shutil.copytree(compiled, src / 'bin', dirs_exist_ok=True)
        installed(src)
        try:
            src.rename(target)
        except OSError:
            if not target.is_dir():  # Two first runs may finish installing simultaneously.
                raise
    return installed(target, revision=COMMIT)
