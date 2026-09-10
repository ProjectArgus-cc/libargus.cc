import copy
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from candidate import build, verify, stage_natives
from catalog import ROOT, modules, targets, version
from destinations import Central, credential, packages
from destinations import GitHub
from http_transport import Transport, HTTPFailure
from inventory import digest, zip_entries
from pins import validate
from verify_classifier import aggregate
from artifact_ids import select
from publish import signatures
import preflight as preflight_module
from ci_route import is_tagged_release
from native import validate_windows_dependencies

SOURCE = '1' * 40
PROVIDER = 'META-INF/services/cc.projectargus.libargus.spi.NativeLibraryProvider'

def fixture(root):
    maven, natives = root / 'maven', root / 'downloads'
    v = version()
    for row in targets():
        dest = natives / ('native-' + row['id']); dest.mkdir(parents=True)
        header = bytearray(64)
        if row['system'] == 'linux': header[:6] = b'\x7fELF\x02\x01'; struct.pack_into('<H', header, 18, 62)
        elif row['system'] == 'windows':
            header[:2] = b'MZ'; struct.pack_into('<I', header, 60, 64); header += b'PE\0\0\x64\x86'
        else: header[:8] = b'\xcf\xfa\xed\xfe\x0c\0\0\1'
        (dest / row['library']).write_bytes(header + row['id'].encode())
        info = {'target': row['id'], 'sha256': digest(dest / row['library']), 'features': row['mask'],
                'build': {'source': SOURCE, 'version': v, 'cpu_baseline': row['baseline'], 'abi_header_sha256': '0' * 64}}
        (dest / 'native.json').write_text(json.dumps(info))
    for module in modules():
        dest = maven / 'cc/projectargus' / module / v; dest.mkdir(parents=True)
        for suffix in ('', '-sources', '-javadoc'):
            with ZipFile(dest / f'{module}-{v}{suffix}.jar', 'w') as z:
                z.writestr('META-INF/MANIFEST.MF', 'Manifest-Version: 1.0\n')
                if module == 'libargus-core':
                    z.writestr('argus-build.properties', f'version={v}\nsource={SOURCE}\nabi_header_sha256={"0"*64}\n')
                if module != 'libargus-core' and not suffix:
                    row = next(r for r in targets() if r['module'] == module)
                    z.write(natives / ('native-' + row['id']) / row['library'], row['resource'])
                    z.writestr(PROVIDER, 'fixture.' + row['backend'])
        (dest / f'{module}-{v}.pom').write_text(f'<project xmlns="http://maven.apache.org/POM/4.0.0"><groupId>cc.projectargus</groupId><artifactId>{module}</artifactId><version>{v}</version></project>')
        (dest / f'{module}-{v}.module').write_text(json.dumps({'component': {'group': 'cc.projectargus', 'module': module, 'version': v}}))
    return maven, natives

def receipts(candidate, destination):
    manifest = verify(candidate, False); destination.mkdir()
    for row in targets():
        native = next(n for n in manifest['natives'] if n['target'] == row['id'])
        result = {**native['build'], 'features': row['mask'], 'native_sha256': native['sha256'], 'provider': 'fixture.' + row['backend'], 'cpu_decode': 'passed'}
        receipt = {'schema': 1, 'target': row['id'], 'source': SOURCE, 'manifest_sha256': digest(candidate / 'manifest.json'),
                   'core_sha256': digest(candidate / 'jars' / f'libargus-core-{version()}.jar'),
                   'classifier_sha256': digest(candidate / 'jars' / f'{row["module"]}-{version()}.jar'),
                   'native_sha256': native['sha256'], 'positive': {**result, 'result': 'ok', 'expected_features': row['mask']},
                   'negative': {**result, 'result': 'feature_mismatch', 'expected_features': 3 if row['backend'] == 'cpu' else 1}}
        (destination / (row['id'] + '.json')).write_text(json.dumps(receipt))

class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        self.maven, self.natives = fixture(self.root); self.candidate = self.root / 'candidate'
    def seal(self): build(self.maven, self.natives, self.candidate, SOURCE, development=True)
    def test_sealed_roundtrip_and_mutation(self):
        stage_natives(self.natives, self.root / 'bindings'); self.seal()
        manifest = verify(self.candidate, False); self.assertEqual(len(manifest['assets']), 35)
        with self.assertRaisesRegex(ValueError, 'signed release'): verify(self.candidate)
        payload = self.candidate / manifest['assets'][0]
        with payload.open('r+b') as f: f.seek(20); f.write(b'X')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'): verify(self.candidate, False)
    def test_staging_rejects_namespace_contamination(self):
        (self.natives / 'test-reports').mkdir()
        with self.assertRaisesRegex(ValueError, 'catalog'): stage_natives(self.natives, self.root / 'bindings')
    def test_wrong_native_metadata(self):
        path = self.natives / 'native-linux-amd64-cpu/native.json'; obj = json.loads(path.read_text()); obj['features'] = 0; path.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, 'metadata mismatch'): stage_natives(self.natives, self.root / 'bindings')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'): self.seal()
    def test_missing_and_wrong_version_payload(self):
        path = self.maven / f'cc/projectargus/libargus-core/{version()}/libargus-core-{version()}.pom'
        path.write_text(path.read_text().replace('<version>' + version(), '<version>0.0.0'))
        with self.assertRaisesRegex(ValueError, 'coordinates'): self.seal()
    def test_duplicate_zip_native_rejected(self):
        row = targets()[0]; jar = self.maven / f'cc/projectargus/{row["module"]}/{version()}/{row["module"]}-{version()}.jar'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with ZipFile(jar, 'a') as z: z.writestr(row['resource'], b'evil')
        with self.assertRaisesRegex(ValueError, 'Duplicate'): self.seal()
    def test_exact_receipt_set_and_identity(self):
        self.seal(); folder = self.root / 'receipts'; receipts(self.candidate, folder)
        aggregate(self.candidate, folder, True)
        path = folder / 'linux-amd64-cpu.json'; original = path.read_text(); obj = json.loads(original)
        for mutation in ('features', 'provider', 'source', 'native_sha256'):
            broken = copy.deepcopy(obj); broken['negative'][mutation] = 'wrong'; path.write_text(json.dumps(broken))
            with self.assertRaises(ValueError): aggregate(self.candidate, folder, True)
        path.write_text(original); path.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'): aggregate(self.candidate, folder, True)
    def test_bundle_and_public_exports_are_same_bytes(self):
        self.seal(); manifest = verify(self.candidate, False)
        with ZipFile(self.candidate / 'central-bundle.zip') as z:
            for name in manifest['assets']:
                if not name.startswith('jars/'): continue
                payload = next(p for p in manifest['files'] if p.startswith('maven/') and p.endswith('/' + Path(name).name))
                self.assertEqual(z.read(payload.removeprefix('maven/')), (self.candidate / name).read_bytes())
    def test_real_signatures_and_corrupted_signature(self):
        home = self.root / 'gnupg'; home.mkdir(mode=0o700)
        with patch.dict(os.environ, {'GNUPGHOME': str(home), 'GPG_PASSPHRASE': ''}):
            subprocess.run(['gpg', '--batch', '--pinentry-mode', 'loopback', '--passphrase', '', '--quick-generate-key',
                            'Argus fixture <fixture@example.invalid>', 'rsa2048', 'sign', '1d'],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            listing = subprocess.check_output(['gpg', '--batch', '--with-colons', '--list-secret-keys'], text=True, stderr=subprocess.PIPE)
            fingerprint = next(line.split(':')[9] for line in listing.splitlines() if line.startswith('fpr:'))
            build(self.maven, self.natives, self.candidate, SOURCE, signing_key=fingerprint)
        manifest = verify(self.candidate); signatures(self.candidate, manifest)
        path = next((self.candidate / 'maven').rglob('*.jar.asc'))
        path.write_text('corrupt signature')
        with self.assertRaises(subprocess.CalledProcessError): signatures(self.candidate, manifest)

class ContractTests(unittest.TestCase):
    def test_windows_native_rejects_host_cpp_runtime_dependency(self):
        validate_windows_dependencies(['KERNEL32.dll', 'VCOMP140.DLL'])
        for runtime in ('MSVCP140.dll', 'VCRUNTIME140.dll', 'vcruntime140_1.DLL'):
            with self.assertRaisesRegex(ValueError, 'host-provided MSVC runtime'):
                validate_windows_dependencies(['KERNEL32.dll', runtime])

    def test_exact_tag_routes_only_push_event_to_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def git(*args, capture=False):
                result = subprocess.run(
                    ['git', '-c', 'user.name=Argus fixture', '-c', 'user.email=fixture@example.invalid', *args],
                    cwd=root, check=True, stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                    stderr=subprocess.PIPE, text=True)
                return result.stdout.strip() if capture else None
            git('init')
            (root / 'version.txt').write_text('1.2.3\n')
            git('add', 'version.txt'); git('commit', '-m', 'fixture')
            source = git('rev-parse', 'HEAD', capture=True)
            self.assertFalse(is_tagged_release('push', source, root))
            git('tag', 'v1.2.2')
            self.assertFalse(is_tagged_release('push', source, root))
            git('tag', 'v1.2.3')
            self.assertTrue(is_tagged_release('push', source, root))
            self.assertFalse(is_tagged_release('pull_request', source, root))
            with self.assertRaisesRegex(ValueError, 'full source commit'):
                is_tagged_release('push', 'HEAD', root)

    def test_tag_checkout_and_dirty_tree_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def git(*args):
                return subprocess.run(['git', '-c', 'user.name=Argus fixture', '-c', 'user.email=fixture@example.invalid', *args],
                                      cwd=root, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            git('init'); (root / 'version.txt').write_text(version()); git('add', 'version.txt'); git('commit', '-m', 'fixture')
            tag = 'v' + version(); git('tag', tag)
            with patch.object(preflight_module, 'ROOT', root):
                self.assertEqual(preflight_module.preflight(tag)['tag'], tag)
                with self.assertRaisesRegex(ValueError, 'version.txt'): preflight_module.preflight('v0.0.0')
                (root / 'unexpected').write_text('mutation')
                with self.assertRaisesRegex(ValueError, 'dirty'): preflight_module.preflight(tag)
                git('add', 'unexpected'); git('commit', '-m', 'different checkout')
                with self.assertRaisesRegex(ValueError, 'tag commit'): preflight_module.preflight(tag)
    def test_immutable_artifact_selection(self):
        rows = [{'name': 'native-' + r['id'], 'id': i+1, 'expired': False} for i, r in enumerate(targets())]
        with patch.dict(os.environ, {'GITHUB_REPOSITORY': 'a/b', 'GITHUB_RUN_ID': '123'}):
            self.assertEqual(len(select('native', lambda p: {'artifacts': rows}).split(',')), 8)
            for broken in (rows[:-1], rows + [rows[0]], [dict(r, expired=True) for r in rows]):
                with self.assertRaises(ValueError): select('native', lambda p: {'artifacts': broken})
    def test_catalog_duplicate_and_unknown_mask(self):
        data = json.loads((ROOT / 'scripts/release/platforms.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'catalog.json'
            for rows in [
                data['targets'][:-1],
                data['targets'][:-1] + [data['targets'][0]],
                [{**r, 'mask': r['mask'] | 128} for r in data['targets']],
                [{**r, 'runner': 'ubuntu-latest'} if r['backend'] == 'vulkan' and r['system'] == 'linux' else r
                 for r in data['targets']],
            ]:
                path.write_text(json.dumps({'schema': 1, 'targets': rows}))
                with self.assertRaises(ValueError): targets(path)
    def test_archive_traversal_duplicate_and_symlink(self):
        for name in ('../escape', '/absolute', 'a\\b', 'a/./b', 'C:/evil'):
            data = io.BytesIO()
            with ZipFile(data, 'w') as z: z.writestr(name, b'bad')
            with ZipFile(data) as z:
                with self.assertRaises(ValueError): list(zip_entries(z))
        data = io.BytesIO(); entry = ZipInfo('link'); entry.external_attr = 0o120777 << 16
        with ZipFile(data, 'w') as z: z.writestr(entry, b'outside')
        with ZipFile(data) as z:
            with self.assertRaises(ValueError): list(zip_entries(z))
    def test_documented_pin_and_repository_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / '.github/workflows').mkdir(parents=True); (root / 'scripts/release').mkdir(parents=True)
            (root / 'scripts/release/actions.lock.json').write_text(json.dumps({'a/b': {'sha': SOURCE, 'tag': 'v1'}}))
            workflow = root / '.github/workflows/ci.yml'
            workflow.write_text(f'jobs:\n  test:\n    steps:\n      - uses: a/b@{SOURCE} # v1\n')
            calls = []
            def api(path):
                calls.append(path)
                return [{'name': 'action.yml'}] if '/contents/' in path else {'sha': SOURCE}
            validate(root, query=api); self.assertTrue(any('/commits/v1' in c for c in calls))
            with self.assertRaisesRegex(ValueError, 'differs'): validate(root, query=lambda p: {'sha': '2'*40})
            workflow.write_text(workflow.read_text().replace('# v1', '# v2'))
            with self.assertRaisesRegex(ValueError, 'comment'): validate(root, online=False)
            workflow.write_text('uses: a/b@deadbeef')
            with self.assertRaisesRegex(ValueError, 'full commit'): validate(root, online=False)

class FakeCentral:
    def __init__(self, states, lost_upload=False):
        self.states = iter(states); self.lost_upload = lost_upload; self.uploads = 0; self.publishes = 0
    def upload(self, *args, **kwargs):
        self.uploads += 1
        if self.lost_upload: raise TimeoutError('lost accepted upload response')
        return '00000000-0000-0000-0000-000000000001'
    def request(self, method, url, *args):
        if '/status?' in url:
            return {'deploymentId': '00000000-0000-0000-0000-000000000001', 'deploymentName': 'libargus-' + SOURCE, 'deploymentState': next(self.states)}
        self.publishes += 1

class RecoveryTests(unittest.TestCase):
    def test_definitively_rejected_upload_can_resume_with_same_candidate(self):
        http = FakeCentral(['PUBLISHED']); central = Central(http, 'auth', wait=lambda s: None)
        state = {'candidate': SOURCE, 'central': 'not_started'}
        with patch.object(http, 'upload', side_effect=HTTPFailure(401)):
            with self.assertRaises(HTTPFailure): central.publish(Path('.'), state, lambda: None)
        self.assertEqual(state['central'], 'upload_rejected'); self.assertEqual(state['upload_http_status'], 401)
        central.publish(Path('.'), state, lambda: None)
        self.assertEqual(state['central'], 'PUBLISHED')
    def test_accepted_github_upload_with_lost_response_is_reconciled(self):
        class HTTP:
            def __init__(self): self.assets = []; self.uploads = 0; self.checked = []
            def request(self, method, url, *args): return self.assets
            def upload(self, method, url, path, *args):
                self.uploads += 1
                self.assets.append({'id': self.uploads, 'name': path.name, 'state': 'uploaded', 'size': path.stat().st_size})
                if self.uploads == 1: raise TimeoutError('accepted upload, response lost')
                return json.dumps(self.assets[-1])
            def matches(self, url, meta, *args): self.checked.append(meta); return True
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('runtime.jar', 'manifest.json', 'checksums.txt'): (root / name).write_text(name)
            manifest = {'assets': ['runtime.jar']}; release = {'id': 7, 'draft': True}
            http = HTTP(); github = GitHub(http, 'a/b', 'token')
            with self.assertRaises(TimeoutError): github.assets(release, root, manifest)
            github.assets(release, root, manifest)
            self.assertEqual(http.uploads, 3)
            self.assertEqual(len(http.checked), 3)
            self.assertEqual(http.checked[0]['sha256'], digest(root / 'runtime.jar'))
    def test_existing_release_conflict_and_unidentified_release(self):
        class HTTP:
            def __init__(self, body): self.body = body; self.writes = 0
            def request(self, method, url, *args):
                if method != 'GET': self.writes += 1; raise AssertionError('Unexpected mutation')
                return [{'tag_name': 'v1.7.5', 'body': self.body}]
        for body in ('source-only release', '<!-- argus-state ' + json.dumps({'candidate': 'different', 'source': SOURCE}) + ' -->'):
            http = HTTP(body)
            with self.assertRaises(ValueError): GitHub(http, 'a/b', 'token').release('v1.7.5', SOURCE, SOURCE)
            self.assertEqual(http.writes, 0)
    def test_terminal_state_and_same_deployment_resume(self):
        http = FakeCentral(['PENDING', 'VALIDATED', 'PUBLISHING', 'PUBLISHED', 'PUBLISHED'])
        state = {'candidate': SOURCE, 'central': 'not_started'}; snapshots = []
        central = Central(http, 'auth', wait=lambda s: None)
        central.publish(Path('.'), state, lambda: snapshots.append(dict(state)))
        self.assertEqual(state['central'], 'PUBLISHED'); self.assertEqual(http.publishes, 1)
        central.publish(Path('.'), state, lambda: None)
        self.assertEqual(http.uploads, 1); self.assertEqual(snapshots[0]['central'], 'uploading')
    def test_lost_upload_response_cannot_trigger_duplicate(self):
        http = FakeCentral([], lost_upload=True); state = {'candidate': SOURCE, 'central': 'not_started'}
        central = Central(http, 'auth', wait=lambda s: None)
        with self.assertRaises(TimeoutError): central.publish(Path('.'), state, lambda: None)
        with self.assertRaisesRegex(ValueError, 'uncertain'): central.publish(Path('.'), state, lambda: None)
        self.assertEqual(http.uploads, 1)
    def test_failed_and_delayed_deployment_are_not_success(self):
        for status in ('FAILED', 'PUBLISHING'):
            ticks = iter([0, 0, 2000]); http = FakeCentral([status]); state = {'candidate': SOURCE, 'central': 'not_started'}
            central = Central(http, 'auth', wait=lambda s: None, now=lambda: next(ticks))
            with self.assertRaises((ValueError, TimeoutError)): central.publish(Path('.'), state, lambda: None)
            self.assertNotEqual(state['central'], 'PUBLISHED')
    def test_credentials_fail_before_transport(self):
        with self.assertRaisesRegex(ValueError, 'credentials'): credential('user', '', 'Bearer')
    def test_remote_bytes_conflict_and_bounded_reads(self):
        transport = Transport()
        for payload in (b'bad', b'too long'):
            with patch.object(transport, 'open', return_value=io.BytesIO(payload)):
                with self.assertRaisesRegex(ValueError, 'Conflicting'):
                    transport.matches('https://example.test/payload', {'size': 3, 'sha256': '0'*64})
    def test_packages_do_not_swallow_failed_upload(self):
        class HTTP:
            def matches(self, *args): return False
            def upload(self, *args): raise ConnectionError('incomplete upload')
        with self.assertRaises(ConnectionError):
            packages(HTTP(), 'a/b', 'user', 'token', Path('.'), {'files': {'maven/a.jar': {}}})

if __name__ == '__main__': unittest.main()
