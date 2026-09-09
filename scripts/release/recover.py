"""Resolve retained artifacts only from a successful gate for the exact tag commit."""
import argparse
import json
import os
from pins import api
from preflight import preflight

def recover(tag, run_id, query=api):
    if not run_id.isdigit(): raise ValueError('Invalid run ID')
    identity = preflight(tag)
    base = f'repos/{os.environ["GITHUB_REPOSITORY"]}/actions/runs/{run_id}'
    run = query(base)
    if run['head_sha'] != identity['source'] or run['path'] != '.github/workflows/release.yml' or run['event'] not in ('push', 'workflow_dispatch'):
        raise ValueError('Recovery run is not the original release for this tag commit')
    jobs = []; page = 1
    while True:
        batch = query(base + f'/jobs?filter=all&per_page=100&page={page}')['jobs']; jobs += batch
        if len(batch) < 100: break
        page += 1
    gates = [j for j in jobs if j['name'] == 'validation / gate']
    if not gates or not any(j['conclusion'] == 'success' for j in gates):
        raise ValueError('Original release has no successful validation gate')
    artifacts = []; page = 1
    while True:
        batch = query(base + f'/artifacts?per_page=100&page={page}')['artifacts']; artifacts += batch
        if len(batch) < 100: break
        page += 1
    for name, key in [('sealed-candidate', 'candidate'), ('validated-receipts', 'receipts')]:
        found = [a for a in artifacts if a['name'] == name and not a['expired']]
        if len(found) != 1: raise ValueError('Original artifact is missing, expired or ambiguous: ' + name)
        identity[key] = str(found[0]['id'])
    return identity

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('tag'); p.add_argument('run_id'); a = p.parse_args()
    for key, value in recover(a.tag, a.run_id).items(): print(key + '=' + value)
