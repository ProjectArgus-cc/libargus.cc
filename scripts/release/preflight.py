"""Resolve the exact existing release tag before any expensive jobs start."""
import argparse
import json
import re
import subprocess
from catalog import ROOT, targets, version

def git(*args):
    return subprocess.check_output(['git', '-c', 'core.fsmonitor=false', *args], cwd=ROOT, text=True).strip()

def preflight(tag):
    if not re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+', tag):
        raise ValueError('Expected an existing vMAJOR.MINOR.PATCH release tag')
    if tag != 'v' + version(): raise ValueError('Tag differs from version.txt')
    commit = git('rev-parse', '--verify', f'refs/tags/{tag}^{{commit}}')
    if commit != git('rev-parse', 'HEAD'): raise ValueError('Checkout is not the release tag commit')
    if git('status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('Release checkout is dirty')
    targets()
    return {'tag': tag, 'source': commit, 'version': version()}

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('tag'); a = p.parse_args()
    print(json.dumps(preflight(a.tag)))
