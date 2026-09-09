"""Destination protocols and resumable state transitions. No build tools."""
import base64
import json
import re
import time
from urllib.parse import quote, urlencode
from http_transport import HTTPFailure
from inventory import digest

CENTRAL = 'https://central.sonatype.com/api/v1/publisher'
MAVEN = 'https://repo.maven.apache.org/maven2/'

def credential(username, password, scheme):
    if not username or not password: raise ValueError('Missing publication credentials')
    return scheme + ' ' + base64.b64encode((username + ':' + password).encode()).decode()

class GitHub:
    def __init__(self, transport, repository, token):
        if not re.fullmatch(r'[\w.-]+/[\w.-]+', repository) or not token: raise ValueError('Missing GitHub publication configuration')
        self.http = transport; self.repo = repository; self.auth = 'Bearer ' + token
        self.api = 'https://api.github.com/repos/' + repository
    def request(self, method, path, payload=None):
        return self.http.request(method, self.api + path, self.auth, payload)
    def release(self, tag, source, identity):
        releases = []; page = 1
        while True:
            batch = self.request('GET', '/releases?per_page=100&page=' + str(page)); releases += batch
            if len(batch) < 100: break
            page += 1
        found = [r for r in releases if r['tag_name'] == tag]
        if len(found) > 1: raise ValueError('Duplicate releases for tag')
        if found:
            release = found[0]
            match = re.search(r'<!-- argus-state (.*?) -->', release.get('body') or '')
            if not match: raise ValueError('Existing release lacks candidate identity; manual reconciliation required')
            state = json.loads(match[1])
            if state['candidate'] != identity or state['source'] != source: raise ValueError('Existing version belongs to different bytes')
            return release, state
        state = {'schema': 1, 'candidate': identity, 'source': source, 'central': 'not_started', 'packages': 'not_started'}
        release = self.request('POST', '/releases', {'tag_name': tag, 'target_commitish': source,
            'name': tag, 'draft': True, 'body': self.body(state)})
        return release, state
    @staticmethod
    def body(state):
        return ('Validated native builds and isolated JVM classifier checks for all eight targets. '
                'CPU/media execution is tested; feature masks do not establish accelerator device execution.\n\n'
                'Publication state: Central ' + state['central'] + '; GitHub Packages ' + state['packages'] + '.\n\n'
                '<!-- argus-state ' + json.dumps(state, sort_keys=True) + ' -->')
    def save(self, release, state):
        self.request('PATCH', '/releases/' + str(release['id']), {'body': self.body(state)})
    def assets(self, release, directory, manifest):
        paths = manifest['assets'] + ['manifest.json', 'checksums.txt']
        expected = {p.split('/')[-1]: directory / p for p in paths}
        existing = []; page = 1
        while True:
            batch = self.request('GET', f'/releases/{release["id"]}/assets?per_page=100&page={page}')
            existing += batch
            if len(batch) < 100: break
            page += 1
        if len({a['name'] for a in existing}) != len(existing) or {a['name'] for a in existing} - set(expected):
            raise ValueError('Unexpected or duplicate GitHub assets')
        by_name = {a['name']: a for a in existing}
        for name, path in expected.items():
            meta = {'size': path.stat().st_size, 'sha256': digest(path)}
            if name not in by_name:
                if not release['draft']: raise ValueError('Public release is missing required assets')
                url = f'https://uploads.github.com/repos/{self.repo}/releases/{release["id"]}/assets?' + urlencode({'name': name})
                # An uncertain upload is reconciled by listing and hashing on the next run.
                by_name[name] = json.loads(self.http.upload('POST', url, path, self.auth))
            asset = by_name[name]
            if asset['state'] != 'uploaded' or asset['size'] != meta['size']: raise ValueError('Incomplete GitHub asset; manual reconciliation required')
            # The octet-stream endpoint also works for authenticated draft assets.
            url = self.api + '/releases/assets/' + str(asset['id'])
            # Transport adds no JSON Accept header to digest reads.
            if not self.http.matches(url, meta, self.auth): raise ValueError('GitHub asset disappeared')
    def expose(self, release, state):
        self.request('PATCH', '/releases/' + str(release['id']), {'draft': False, 'body': self.body(state)})

class Central:
    def __init__(self, transport, auth, wait=time.sleep, now=time.monotonic):
        self.http = transport; self.auth = auth; self.wait = wait; self.now = now
    def publish(self, directory, state, save, deployment=None, timeout=1800):
        name = 'libargus-' + state['candidate']
        if deployment:
            if state.get('deployment') not in (None, deployment): raise ValueError('Conflicting deployment ID')
            state['deployment'] = deployment; save()
        if not state.get('deployment'):
            if state['central'] not in ('not_started', 'upload_rejected'):
                raise ValueError('Central upload outcome uncertain: recover deployment ID from Portal; do not rebuild or reupload')
            state['central'] = 'uploading'; save()
            try:
                identifier = self.http.upload('POST', CENTRAL + '/upload?' + urlencode({'name': name, 'publishingType': 'USER_MANAGED'}),
                                              directory / 'central-bundle.zip', self.auth, multipart=True).strip()
            except HTTPFailure as failure:
                # Definitive client rejection differs from an uncertain response.
                # Conflicts and server failures require reconciliation, not retry.
                if failure.status in (400, 401, 403, 413, 429):
                    state['central'] = 'upload_rejected'; state['upload_http_status'] = failure.status; save()
                raise
            if not re.fullmatch(r'[0-9a-fA-F-]{36}', identifier): raise ValueError('Uncertain Central upload response')
            state['deployment'] = identifier; state['central'] = 'uploaded'; save()
        identifier = state['deployment']
        deadline = self.now() + timeout
        while self.now() < deadline:
            info = self.http.request('POST', CENTRAL + '/status?' + urlencode({'id': identifier}), self.auth)
            if info['deploymentId'] != identifier or info['deploymentName'] != name: raise ValueError('Deployment identity mismatch')
            status = info['deploymentState']; state['central'] = status; save()
            if status == 'PUBLISHED': return
            if status == 'FAILED': raise ValueError('Central validation failed; inspect retained deployment')
            if status == 'VALIDATED':
                self.http.request('POST', CENTRAL + '/deployment/' + quote(identifier, safe=''), self.auth)
            elif status not in ('PENDING', 'VALIDATING', 'PUBLISHING'):
                raise ValueError('Unknown Central deployment state')
            self.wait(15)
        raise TimeoutError('Central has not reached PUBLISHED; resume the retained deployment')
    def served(self, directory, manifest, timeout=1800):
        deadline = self.now() + timeout
        for name, meta in manifest['files'].items():
            if not name.startswith('maven/'): continue
            # Repository services may regenerate checksum sidecars. Compare every
            # actual payload and signature, including POM and Gradle module metadata.
            if name.rsplit('.', 1)[-1] in ('md5', 'sha1', 'sha256', 'sha512'): continue
            url = MAVEN + quote(name.removeprefix('maven/'), safe='/')
            while not self.http.matches(url, meta):
                if self.now() >= deadline: raise TimeoutError('Central payload propagation incomplete')
                self.wait(15)

def packages(transport, repository, username, token, directory, manifest):
    auth = credential(username, token, 'Basic')
    base = 'https://maven.pkg.github.com/' + repository + '/'
    for name, meta in manifest['files'].items():
        if not name.startswith('maven/'): continue
        url = base + quote(name.removeprefix('maven/'), safe='/')
        if not transport.matches(url, meta, auth):
            transport.upload('PUT', url, directory / name, auth)
            if not transport.matches(url, meta, auth): raise ValueError('GitHub Packages payload not readable after upload')
