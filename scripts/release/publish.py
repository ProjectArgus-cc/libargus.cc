"""Publish ONLY sealed candidate files. No Gradle, compiler, packaging or signing."""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from candidate import verify
from verify_classifier import aggregate
from destinations import Central, GitHub, credential, packages
from http_transport import Transport
from inventory import digest

def signatures(directory, manifest):
    with tempfile.TemporaryDirectory(prefix='argus-verify-signatures-') as home:
        os.chmod(home, 0o700)
        command = ['gpg', '--homedir', home, '--batch']
        subprocess.run(command + ['--import', str(directory / 'signing-key.asc')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for name in manifest['files']:
            if name.startswith('maven/') and name.endswith(('.jar', '.pom', '.module')):
                subprocess.run(command + ['--verify', str(directory / (name + '.asc')), str(directory / name)],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def publish(directory, receipts, tag, source, state_path, deployment=None):
    manifest = aggregate(directory, receipts)
    if tag != 'v' + manifest['version'] or source != manifest['source']: raise ValueError('Publication tag/source mismatch')
    signatures(directory, manifest)
    central_auth = credential(os.environ.get('MAVEN_CENTRAL_USERNAME'), os.environ.get('MAVEN_CENTRAL_PASSWORD'), 'Bearer')
    token = os.environ.get('GH_TOKEN'); repository = os.environ['GITHUB_REPOSITORY']; actor = os.environ['GITHUB_ACTOR']
    credential(actor, token, 'Basic')  # Validate every destination before the first mutation.
    transport = Transport(); github = GitHub(transport, repository, token)
    # Resolve the remote tag without moving/creating it.
    resolved = transport.request('GET', github.api + '/commits/' + tag, github.auth)
    if resolved['sha'] != source: raise ValueError('Remote release tag changed')
    release, state = github.release(tag, source, digest(directory / 'manifest.json'))
    def save():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, sort_keys=True, indent=2) + '\n')
        temporary.replace(state_path)
        github.save(release, state)
    save()
    verify(directory)
    github.assets(release, directory, manifest)
    central = Central(transport, central_auth)
    central.publish(directory, state, save, deployment)
    central.served(directory, manifest)
    state['central'] = 'PUBLISHED'; state['central_served_bytes'] = 'verified'; save()
    # Core publication can complete even if the separately reported secondary
    # destination fails. Its exception still fails the run and is resumable.
    github.expose(release, state)
    try:
        packages(transport, repository, actor, token, directory, manifest)
        state['packages'] = 'verified'; save()
    except Exception:
        state['packages'] = 'incomplete'; save(); raise
    print('Central and GitHub served bytes verified; GitHub Packages verified.')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('candidate', type=Path); p.add_argument('receipts', type=Path); p.add_argument('tag')
    p.add_argument('--source', required=True); p.add_argument('--state', type=Path, required=True); p.add_argument('--deployment-id')
    a = p.parse_args(); publish(a.candidate.resolve(), a.receipts, a.tag, a.source, a.state, a.deployment_id)
