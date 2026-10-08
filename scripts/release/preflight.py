"""Resolve the exact existing release tag before any expensive jobs start."""
import argparse
import json
import re
import subprocess
from catalog import ROOT, targets, version

def git(*args):
    return subprocess.check_output(['git', '-c', 'core.fsmonitor=false', *args], cwd=ROOT, text=True).strip()

def validate_header_version(root=ROOT):
    header_path = root / 'include' / 'libargus.h'
    if header_path.is_file():
        header_text = header_path.read_text(encoding='utf-8')
        m = re.search(r'@version\s+([0-9]+\.[0-9]+\.[0-9]+)', header_text)
        if not m:
            raise ValueError(f"No @version found in {header_path}")
        header_ver = m.group(1)
        ver = (root / 'version.txt').read_text().strip() if (root / 'version.txt').is_file() else version()
        if header_ver != ver:
            raise ValueError(f"include/libargus.h @version ({header_ver}) differs from version.txt ({ver})")

def preflight(tag):
    if not re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+', tag):
        raise ValueError('Expected an existing vMAJOR.MINOR.PATCH release tag')
    if tag != 'v' + version(): raise ValueError('Tag differs from version.txt')
    validate_header_version(ROOT)
    commit = git('rev-parse', '--verify', f'refs/tags/{tag}^{{commit}}')
    if commit != git('rev-parse', 'HEAD'): raise ValueError('Checkout is not the release tag commit')
    if git('status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('Release checkout is dirty')
    targets()
    return {'tag': tag, 'source': commit, 'version': version()}

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('tag'); a = p.parse_args()
    print(json.dumps(preflight(a.tag)))
