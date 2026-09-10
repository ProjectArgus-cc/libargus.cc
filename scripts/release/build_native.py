"""Catalog-driven native build and test; no artifact discovery heuristics."""
import argparse
import os
import shutil
import subprocess
from pathlib import Path
from catalog import target
from native import inspect
from inventory import digest
import json

def run(command):
    subprocess.run(command, check=True)

def build(row, testing=False, sanitizer='', media=False):
    backend = row['backend']
    flags = ['-DGGML_' + key + '=' + ('ON' if backend == value else 'OFF')
             for key, value in [('CUDA','cuda'), ('HIP','rocm'), ('VULKAN','vulkan'), ('METAL','metal')]]
    if backend == 'cuda':
        # The release API does not require CUDA virtual-memory management. Avoiding
        # a load-time driver dependency lets hosted runners inspect the binary;
        # CUDA runtime calls still validate the driver when a device is used.
        flags += ['-DGGML_CUDA_NO_VMM=ON']
    if backend == 'rocm':
        flags += ['-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang', '-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++',
                  '-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++', '-DAMDGPU_TARGETS=gfx906;gfx908;gfx90a;gfx1030;gfx1100']
    if row['system'] == 'macos': flags += ['-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0']
    if sanitizer: flags += ['-DGGML_OPENMP=OFF']
    run(['cmake', '-B', 'build', '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DARGUS_PORTABLE=ON',
         '-DARGUS_TESTING=' + ('ON' if testing and backend in ('cpu', 'metal') else 'OFF'), '-DARGUS_SANITIZER=' + sanitizer,
         '-DARGUS_MEDIA_REQUIRED=' + ('ON' if media else 'OFF')] + flags)
    run(['cmake', '--build', 'build', '-j', '4'])
    commands = json.loads(Path('build/compile_commands.json').read_text())
    if any('-march=native' in c['command'] or '/arch:AVX512' in c['command'] for c in commands):
        raise ValueError('Host-specific instruction tuning escaped portable configuration')
    extension = '.exe' if row['system'] == 'windows' else ''
    run([str(Path('build/bin/test_libargus' + extension)), '--features-only', '--expect-features', backend])
    if backend in ('cpu', 'metal'):
        selection = ['-R', 'test_concurrency'] if sanitizer == 'thread' else []
        run(['ctest', '--test-dir', 'build', '--output-on-failure', '--timeout', '120'] + selection)
    # Sanitizer and observer libraries are deliberately never staged.
    if not sanitizer:
        path = Path('build') / ('bin' if row['system'] == 'windows' else 'lib') / row['library']
        info = inspect(path, row)
        if backend == 'cuda':
            toolkit = subprocess.check_output(['nvcc', '--version'], text=True)
            if 'release 12.4' not in toolkit or 'V12.4.131' not in toolkit:
                raise ValueError('CUDA compiler differs from toolkit 12.4.1')
        elif backend == 'rocm':
            toolkit = subprocess.check_output(['/opt/rocm/bin/hipcc', '--version'], text=True)
            if 'HIP version: 6.1' not in toolkit: raise ValueError('HIP compiler differs from ROCm 6.1')
        else: toolkit = info['build']['compiler']
        info['toolchain'] = toolkit.strip()
        info['compile_commands_sha256'] = digest(Path('build/compile_commands.json'))
        destination = Path('staging'); destination.mkdir()
        shutil.copyfile(path, destination / row['library'])
        (destination / 'native.json').write_text(json.dumps(info, sort_keys=True, indent=2) + '\n')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('target'); p.add_argument('--testing', action='store_true')
    p.add_argument('--sanitizer', default='', choices=['', 'address', 'thread']); p.add_argument('--media', action='store_true')
    a = p.parse_args(); build(target(a.target), a.testing, a.sanitizer, a.media)
