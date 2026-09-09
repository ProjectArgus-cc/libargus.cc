"""Validate workflow Action references independently of their execution."""
import argparse
import json
import os
import re
import urllib.request
from pathlib import Path
from catalog import ROOT

def references(root):
    import yaml
    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == 'uses':
                    if not isinstance(value, str): raise ValueError('Action reference is not a string')
                    yield value
                else: yield from walk(value)
        elif isinstance(node, list):
            for value in node: yield from walk(value)
    result = set()
    for path in sorted((root / '.github').rglob('*.y*ml')):
        for ref in walk(yaml.load(path.read_text(), Loader=yaml.BaseLoader)):
            if ref.startswith('./'):
                resolved = (root / ref).resolve()
                if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
                    raise ValueError(f'Invalid local workflow: {ref}')
            else:
                if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_./-]+)?@[0-9a-f]{40}', ref):
                    raise ValueError(f'Action is not pinned to a full commit: {ref}')
                if '..' in ref.split('@')[0].split('/'): raise ValueError('Unsafe Action subpath')
                result.add(ref)
    return sorted(result)

def api(path):
    request = urllib.request.Request('https://api.github.com/' + path, headers={'Accept': 'application/vnd.github+json'})
    if token := os.environ.get('GH_TOKEN', os.environ.get('GITHUB_TOKEN')):
        request.add_header('Authorization', 'Bearer ' + token)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)

def validate(root, online=True, query=api):
    refs = references(root)
    lock = json.loads((root / 'scripts/release/actions.lock.json').read_text())
    for ref in refs:
        path, sha = ref.split('@'); parts = path.split('/'); repo = '/'.join(parts[:2])
        if repo not in lock or lock[repo]['sha'] != sha:
            raise ValueError('Action pin differs from documented lock: ' + ref)
        for file in (root / '.github').rglob('*.y*ml'):
            for line in file.read_text().splitlines():
                if 'uses:' in line and ref in line:
                    if not line.rstrip().endswith('# ' + lock[repo]['tag']):
                        raise ValueError('Action tag comment differs from lock: ' + ref)
        if online:
            if query(f'repos/{repo}/commits/{sha}')['sha'] != sha:
                raise ValueError('Resolved commit differs from pin')
            if query(f'repos/{repo}/commits/{lock[repo]["tag"]}')['sha'] != sha:
                raise ValueError('Documented Action tag does not resolve to pin')
            directory = '/'.join(parts[2:])
            listing = query(f'repos/{repo}/contents/{directory}?ref={sha}')
            if not any(item['name'] in ('action.yml', 'action.yaml') for item in listing):
                raise ValueError(f'Action metadata missing: {ref}')
    return refs

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--offline', action='store_true'); a = p.parse_args()
    print(json.dumps({'validated': validate(ROOT, not a.offline)}, indent=2))
