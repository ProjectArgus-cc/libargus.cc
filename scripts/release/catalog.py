"""Validated distribution contract shared by CI, staging and verification."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def targets(path=None):
    data = json.loads((path or Path(__file__).with_name('platforms.json')).read_text())
    if data['schema'] != 1:
        raise ValueError('Unsupported catalog schema')
    rows = data['targets']
    expected = {(os, backend) for os, backends in (
        ('linux', ('cpu', 'cuda', 'rocm', 'vulkan')),
        ('windows', ('cpu', 'cuda', 'vulkan')), ('macos', ('metal',))) for backend in backends}
    if len(rows) != 8 or {(r['system'], r['backend']) for r in rows} != expected:
        raise ValueError('Incomplete or duplicate platform set')
    masks = {'cpu': 1, 'cuda': 3, 'rocm': 5, 'vulkan': 9, 'metal': 17}
    definitions = dict(re.findall(r'#define ARGUS_FEATURE_(\w+)\s+\(1ULL << (\d+)\)', (ROOT / 'include/libargus.h').read_text()))
    bits = {name: 1 << int(bit) for name, bit in definitions.items()}
    for backend, mask in masks.items():
        flag = 'HIP' if backend == 'rocm' else backend.upper()
        if mask != bits['CPU'] | bits[flag]:
            raise ValueError('Catalog masks disagree with the public ABI')
    for r in rows:
        if r['mask'] != masks[r['backend']]:
            raise ValueError('Incorrect feature mask')
        if r['id'] != f"{r['system']}-{r['arch']}-{r['backend']}":
            raise ValueError('Incorrect platform identity')
        if r['module'] != f"libargus-native-{r['system']}-{r['backend']}":
            raise ValueError('Incorrect module identity')
        library = {'linux': 'libargus.so', 'windows': 'argus.dll', 'macos': 'libargus.dylib'}[r['system']]
        arch = 'aarch64' if r['system'] == 'macos' else 'amd64'
        if r['arch'] != arch or r['library'] != library or r['resource'] != f"natives/{r['system']}-{arch}/{r['backend']}/{library}":
            raise ValueError('Incorrect native resource contract')
        if r['runner'] != {'linux': 'ubuntu-22.04', 'windows': 'windows-2022', 'macos': 'macos-14'}[r['system']] or r['baseline'] != ('armv8-a' if arch == 'aarch64' else 'x86-64-v3'):
            raise ValueError('Incorrect runner or CPU baseline')
    return rows

def target(identity):
    return next(r for r in targets() if r['id'] == identity)

def version():
    value = (ROOT / 'version.txt').read_text().strip()
    if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', value):
        raise ValueError('Invalid release version')
    return value

def modules():
    return ['libargus-core'] + [r['module'] for r in targets()]

if __name__ == '__main__':
    print(json.dumps({'include': targets()}, separators=(',', ':')))
