"""Inspect a built native payload before it enters the candidate."""
import argparse
import ctypes
import json
import os
import re
import struct
import subprocess
from pathlib import Path
from catalog import ROOT, target
from inventory import abi_header_digest, digest

def validate_windows_dependencies(dependencies):
    forbidden_pattern = re.compile(r'^(MSVCP|VCRUNTIME).*\.DLL$', re.IGNORECASE)
    for dep in dependencies:
        base = Path(dep).name
        if forbidden_pattern.match(base):
            raise ValueError(f'Windows distribution library imports a host-provided MSVC runtime: {base}')

def parse_windows_exports(dumpbin_output):
    exports = set()
    in_exports = False
    for line in dumpbin_output.splitlines():
        if re.search(r'\border\s+hint\s+RVA\s+name\b', line, re.I) or ('ordinal' in line.lower() and 'hint' in line.lower() and 'rva' in line.lower() and 'name' in line.lower()):
            in_exports = True
            continue
        if in_exports:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('Summary') or stripped.startswith('Section'):
                break
            parts = stripped.split()
            # typical format: <ordinal> <hint> <RVA> <name> [ = <forwarder> ]
            if len(parts) >= 4 and parts[0].isdigit():
                exports.add(parts[3])
    return exports

def architecture(path):
    with Path(path).open('rb') as f:
        head = f.read(64)
        if head[:4] == b'\x7fELF':
            if head[4:6] != b'\x02\x01' or struct.unpack_from('<H', head, 18)[0] != 62:
                raise ValueError('Expected little-endian ELF64 x86-64')
            return 'linux', 'amd64'
        if head[:2] == b'MZ':
            f.seek(struct.unpack_from('<I', head, 60)[0])
            pe = f.read(6)
            if pe != b'PE\0\0\x64\x86':
                raise ValueError('Expected PE x86-64')
            return 'windows', 'amd64'
        if head[:8] == b'\xcf\xfa\xed\xfe\x0c\x00\x00\x01':
            return 'macos', 'aarch64'
        raise ValueError('Unrecognized native binary architecture')

def inspect(path, row):
    if architecture(path) != (row['system'], row['arch']):
        raise ValueError('Native architecture differs from catalog')
    # Python 3.8+ no longer searches PATH for DLL dependencies implicitly.
    dll_directories = []
    if os.name == 'nt':
        for directory in os.environ.get('PATH', '').split(os.pathsep):
            if Path(directory).is_dir(): dll_directories.append(os.add_dll_directory(directory))
    lib = ctypes.CDLL(str(Path(path).resolve()))
    lib.argus_build_features.restype = ctypes.c_uint64
    mask = lib.argus_build_features()
    if mask != row['mask']:
        raise ValueError(f'Feature mismatch: {mask} != {row["mask"]}')
    for symbol in ('argus_test_set_observer', 'argus_test_projector', 'mtmd_test_create_input_chunks',
                   'argus_test_set_force_rollback_fail', 'argus_test_emit_log', 'argus_test_fail_after_retain',
                   'argus_test_set_eval_result', 'argus_test_video', 'argus_test_chunks',
                   'argus_test_sampler_get_history_token', 'argus_test_draft_seq_pos_max',
                   'argus_test_has_logits'):
        if hasattr(lib, symbol):
            raise ValueError(f'Test/internal symbol exported: {symbol}')
    lib.argus_build_info_copy.argtypes = (ctypes.c_void_p, ctypes.c_int32)
    lib.argus_build_info_copy.restype = ctypes.c_int32
    buffer = ctypes.create_string_buffer(4096)
    size = lib.argus_build_info_copy(buffer, len(buffer))
    if size < 0 or size >= len(buffer):
        raise ValueError('Invalid build metadata length')
    info = dict(line.split('=', 1) for line in buffer.value.decode().splitlines())
    if info['cpu_baseline'] != row['baseline']:
        raise ValueError('Unexpected CPU baseline')
    if info['abi_header_sha256'] != abi_header_digest(ROOT / 'include/libargus.h') or info['abi_schema'] != '1':
        raise ValueError('Native ABI differs from checked-out public header')
    declarations = re.findall(r'ARGUS_API\s+[^;]+;', (ROOT / 'include/libargus.h').read_text())
    expected = {m.group(1) for d in declarations if (m := re.search(r'\b(argus_\w+)\s*\(', d))}
    jni_declarations = re.findall(
        r'\b(JNI_OnLoad|JNI_OnUnload|Java_cc_projectargus_libargus_internal_ArgusJni_\w+)\s*\(',
        (ROOT / 'src/jni/argus_jni.cc').read_text(),
    )
    expected_jni = set(jni_declarations)
    expected_all = expected | expected_jni
    for symbol in expected_all:
        if not hasattr(lib, symbol): raise ValueError('Missing public native symbol: ' + symbol)
    if row['system'] == 'linux':
        symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(path)], text=True)
        dependencies = subprocess.check_output(['readelf', '-d', str(path)], text=True)
        exports = {line.split()[-1] for line in symbols.splitlines() if line.split()}
        needed = re.findall(r'Shared library: \[([^]]+)\]', dependencies)
        if any(name.startswith(('argus_', 'mtmd_', 'llama_', 'whisper_', 'ggml_', 'Java_')) and name not in expected_all for name in exports):
            raise ValueError('Internal/test symbols escaped the release boundary')
    elif row['system'] == 'macos':
        symbols = subprocess.check_output(['nm', '-gU', str(path)], text=True)
        exports = {line.split()[-1].removeprefix('_') for line in symbols.splitlines() if line.split()}
        if exports != expected_all: raise ValueError('macOS export surface differs from public ABI')
        needed = subprocess.check_output(['otool', '-L', str(path)], text=True).splitlines()[1:]
    else:
        exports_raw = subprocess.check_output(['dumpbin', '/EXPORTS', str(path)], text=True)
        exports = parse_windows_exports(exports_raw)
        argus_exports = {name for name in exports if name.startswith('argus_')}
        jni_exports = {name for name in exports if name in ('JNI_OnLoad', 'JNI_OnUnload') or name.startswith('Java_')}
        if argus_exports != expected or jni_exports != expected_jni:
            diff = (argus_exports | jni_exports).symmetric_difference(expected_all)
            raise ValueError(f'Windows DLL export surface differs from public ABI: {diff}')
        if any(name.startswith(('argus_test_', 'mtmd_', 'llama_', 'whisper_', 'ggml_')) for name in exports):
            raise ValueError('Internal/test symbols escaped the release boundary')
        needed = subprocess.check_output(['dumpbin', '/DEPENDENTS', str(path)], text=True).splitlines()
        validate_windows_dependencies(needed)
    if any(re.search(r'lib(asan|ubsan|tsan)|clang_rt\.(asan|tsan)', item, re.I) for item in needed):
        raise ValueError('Sanitizer runtime in distribution library')
    return {'target': row['id'], 'sha256': digest(path), 'features': mask, 'build': info, 'dependencies': needed}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('target'); p.add_argument('binary'); p.add_argument('output')
    a = p.parse_args()
    Path(a.output).write_text(json.dumps(inspect(a.binary, target(a.target)), sort_keys=True, indent=2) + '\n')
