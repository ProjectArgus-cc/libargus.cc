"""Select an exact set of immutable artifacts from this workflow run."""
import argparse
import json
import os
from pins import api
from catalog import targets

def select(kind, query=api):
    expected = {kind + '-' + r['id'] for r in targets()}
    artifacts = []
    page = 1
    while True:
        batch = query(f'repos/{os.environ["GITHUB_REPOSITORY"]}/actions/runs/{os.environ["GITHUB_RUN_ID"]}/artifacts?per_page=100&page={page}')['artifacts']
        artifacts.extend(a for a in batch if a['name'].startswith(kind + '-'))
        if len(batch) < 100: break
        page += 1
    if len(artifacts) != 8 or len({a['id'] for a in artifacts}) != 8 or {a['name'] for a in artifacts} != expected or any(a['expired'] for a in artifacts):
        raise ValueError('Missing, duplicate, expired, or unexpected workflow artifacts')
    return ','.join(str(a['id']) for a in sorted(artifacts, key=lambda a: a['name']))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('kind', choices=['native', 'receipt']); a = p.parse_args()
    print('ids=' + select(a.kind))
