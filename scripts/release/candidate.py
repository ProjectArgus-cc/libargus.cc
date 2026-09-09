"""Construct, seal and verify a release candidate. Never used to publish."""
import argparse
import os
import re
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile
from catalog import ROOT, modules, targets, version
from inventory import digest, digest_stream, files, write_zip, zip_entries
from native import architecture

def stage_natives(downloads, bindings):
    files(downloads)  # Reject symlinks before opening any payload.
    rows = targets()
    expected = {'native-' + r['id'] for r in rows}
    if {p.name for p in downloads.iterdir()} != expected:
        raise ValueError('Native downloads differ from catalog')
    for r in rows:
        directory = downloads / ('native-' + r['id'])
        if {p.name for p in directory.iterdir()} != {r['library'], 'native.json'}:
            raise ValueError('Unexpected native artifact contents')
        native = directory / r['library']
        metadata = json.loads((directory / 'native.json').read_text())
        if metadata['target'] != r['id'] or metadata['sha256'] != digest(native) or metadata['features'] != r['mask']:
            raise ValueError('Native metadata mismatch')
        if architecture(native) != (r['system'], r['arch']):
            raise ValueError('Wrong native architecture')
        dest = bindings / r['module'] / 'src/main/resources' / r['resource']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(native, dest)

def build(maven, downloads, destination, source, signing_key=None, development=False):
    if destination.exists():
        raise ValueError('Candidate destination must not already exist')
    destination.mkdir(parents=True)
    v = version()
    artifacts = []
    jar_paths = {}
    for module in modules():
        relative = Path('cc/projectargus') / module / v
        source_dir = maven / relative
        names = [f'{module}-{v}{suffix}' for suffix in ('.jar', '-sources.jar', '-javadoc.jar', '.pom', '.module')]
        # The Gradle Java publication explicitly includes module metadata.
        checksums = {name + '.' + alg for name in names for alg in ('md5', 'sha1', 'sha256', 'sha512')}
        observed_names = {p.name for p in source_dir.iterdir()}
        if not set(names).issubset(observed_names) or observed_names - set(names) - checksums:
            raise ValueError(f'Unexpected Maven payload for {module}')
        for name in observed_names & checksums:
            payload, algorithm = name.rsplit('.', 1)
            if (source_dir / name).read_text().strip() != digest(source_dir / payload, algorithm):
                raise ValueError('Gradle Maven checksum mismatch')
        pom = ET.parse(source_dir / f'{module}-{v}.pom').getroot()
        ns = {'m': 'http://maven.apache.org/POM/4.0.0'}
        if [pom.findtext('m:' + k, namespaces=ns) for k in ('groupId', 'artifactId', 'version')] != ['cc.projectargus', module, v]:
            raise ValueError('Wrong Maven coordinates')
        component = json.loads((source_dir / f'{module}-{v}.module').read_text())['component']
        if [component.get(k) for k in ('group', 'module', 'version')] != ['cc.projectargus', module, v]:
            raise ValueError('Wrong Gradle publication coordinates')
        for name in names:
            out = destination / 'maven' / relative / name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_dir / name, out)
            artifacts.append(out)
            if name.endswith('.jar'):
                with ZipFile(out) as z:
                    entries = list(zip_entries(z))
                    if not entries:
                        raise ValueError('Empty JAR')
                jar = destination / 'jars' / name
                jar.parent.mkdir(exist_ok=True)
                shutil.copyfile(out, jar)
                jar_paths[name] = jar
    native_records = []
    for r in targets():
        jar = jar_paths[f'{r["module"]}-{v}.jar']
        with ZipFile(jar) as z:
            natives = [e for e in zip_entries(z) if e.filename.startswith('natives/')]
            if [e.filename for e in natives] != [r['resource']]:
                raise ValueError(f'Incorrect native entries in {jar.name}')
            with z.open(r['resource']) as f:
                embedded = digest_stream(f)
        directory = downloads / ('native-' + r['id'])
        native = directory / r['library']
        info = json.loads((directory / 'native.json').read_text())
        if info['sha256'] != embedded or embedded != digest(native) or info['target'] != r['id'] or info['features'] != r['mask']:
            raise ValueError('Native payload identity mismatch')
        if info['build']['version'] != v or info['build']['cpu_baseline'] != r['baseline']:
            raise ValueError('Native build contract mismatch')
        if not development and info['build']['source'] != source:
            raise ValueError('Native source revision mismatch')
        archive = destination / 'native' / f'libargus-{v}-{r["id"]}.zip'
        archive.parent.mkdir(exist_ok=True)
        write_zip(archive, {r['library']: native})
        native_records.append(info)
    if len(jar_paths) != 27:
        raise ValueError('Expected exactly 27 JARs')
    with ZipFile(jar_paths[f'libargus-core-{v}.jar']) as z:
        core_info = dict(line.split('=', 1) for line in z.read('argus-build.properties').decode().splitlines())
    if core_info['version'] != v or (not development and core_info['source'] != source):
        raise ValueError('Core JAR build identity mismatch')
    if any(core_info['abi_header_sha256'] != n['build']['abi_header_sha256'] for n in native_records):
        raise ValueError('Core/native ABI header identities differ')
    if not development and not signing_key:
        raise ValueError('Release candidates require signatures')
    for payload in artifacts:
        checksum_inputs = [payload]
        if signing_key:
            subprocess.run(['gpg', '--batch', '--yes', '--pinentry-mode', 'loopback', '--passphrase-fd', '0',
                            '--armor', '--local-user', signing_key, '--detach-sign', str(payload)],
                           input=os.environ.get('GPG_PASSPHRASE', '').encode(), check=True, stderr=subprocess.PIPE)
            signature = Path(str(payload) + '.asc')
            subprocess.run(['gpg', '--batch', '--verify', str(signature), str(payload)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            checksum_inputs.append(signature)
        for item in checksum_inputs:
            for algorithm in ('md5', 'sha1', 'sha256', 'sha512'):
                Path(str(item) + '.' + algorithm).write_text(digest(item, algorithm))
    repository = destination / 'maven'
    if signing_key:
        with (destination / 'signing-key.asc').open('wb') as key:
            subprocess.run(['gpg', '--batch', '--armor', '--export', signing_key], stdout=key, check=True)
    write_zip(destination / 'central-bundle.zip', {p.relative_to(repository).as_posix(): p for p in repository.rglob('*') if p.is_file()})
    manifest = {'schema': 1, 'version': v, 'source': source, 'signed': bool(signing_key),
                'development': development, 'natives': native_records, 'files': files(destination),
                'assets': sorted('jars/' + name for name in jar_paths) + sorted(p.relative_to(destination).as_posix() for p in (destination / 'native').iterdir())}
    (destination / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    assets = manifest['assets'] + ['manifest.json']
    (destination / 'checksums.txt').write_text(''.join(f'{digest(destination / name)}  {Path(name).name}\n' for name in assets))
    verify(destination, require_release=not development)

def verify(directory, require_release=True):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['schema'] != 1 or (require_release and (not manifest['signed'] or manifest['development'])):
        raise ValueError('Candidate is not a signed release')
    if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', manifest['version']) or not re.fullmatch(r'[0-9a-f]{40}', manifest['source']):
        raise ValueError('Invalid candidate version/source')
    v = manifest['version']
    payloads = {f'maven/cc/projectargus/{m}/{v}/{m}-{v}{suffix}' for m in modules()
                for suffix in ('.jar', '-sources.jar', '-javadoc.jar', '.pom', '.module')}
    if manifest['signed']: payloads |= {name + '.asc' for name in tuple(payloads)}
    required_files = payloads | {name + '.' + alg for name in payloads for alg in ('md5', 'sha1', 'sha256', 'sha512')}
    required_files |= {f'jars/{m}-{v}{suffix}.jar' for m in modules() for suffix in ('', '-sources', '-javadoc')}
    required_files |= {f'native/libargus-{v}-{r["id"]}.zip' for r in targets()} | {'central-bundle.zip'}
    if manifest['signed']: required_files.add('signing-key.asc')
    if set(manifest['files']) != required_files:
        raise ValueError('Unexpected sealed candidate payload set')
    observed = files(directory)
    observed.pop('manifest.json'); observed.pop('checksums.txt')
    if observed != manifest['files']:
        raise ValueError('Candidate file inventory or digest mismatch')
    for module in modules():
        for suffix in ('', '-sources', '-javadoc'):
            name = f'{module}-{v}{suffix}.jar'
            if manifest['files']['jars/' + name] != manifest['files'][f'maven/cc/projectargus/{module}/{v}/{name}']:
                raise ValueError('Public JAR differs from Maven publication')
            with ZipFile(directory / 'jars' / name) as z: list(zip_entries(z))
    if len(manifest['natives']) != 8 or {n['target'] for n in manifest['natives']} != {r['id'] for r in targets()}:
        raise ValueError('Native build identities differ from catalog')
    for r in targets():
        info = next(n for n in manifest['natives'] if n['target'] == r['id'])
        if info['features'] != r['mask'] or info['build']['version'] != v or info['build']['cpu_baseline'] != r['baseline']:
            raise ValueError('Native build contract differs from catalog')
        if require_release and info['build']['source'] != manifest['source']:
            raise ValueError('Native source identity differs from release')
        for path, entry in [(f'jars/{r["module"]}-{v}.jar', r['resource']), (f'native/libargus-{v}-{r["id"]}.zip', r['library'])]:
            with ZipFile(directory / path) as z:
                entries = [e.filename for e in zip_entries(z)]
                actual_native = [e for e in entries if e.startswith('natives/')] if path.startswith('jars/') else entries
                if actual_native != [entry]: raise ValueError('Wrong native archive payload')
                with z.open(entry) as stream:
                    if digest_stream(stream) != info['sha256']: raise ValueError('Embedded native digest mismatch')
    expected_assets = {f'jars/{m}-{manifest["version"]}{suffix}.jar' for m in modules() for suffix in ('', '-sources', '-javadoc')}
    expected_assets |= {f'native/libargus-{manifest["version"]}-{r["id"]}.zip' for r in targets()}
    if set(manifest['assets']) != expected_assets or len(manifest['assets']) != 35:
        raise ValueError('Incorrect release asset set')
    expected_checksums = ''.join(f'{digest(directory / name)}  {Path(name).name}\n' for name in manifest['assets'] + ['manifest.json'])
    if (directory / 'checksums.txt').read_text() != expected_checksums:
        raise ValueError('Checksum export mismatch')
    with ZipFile(directory / 'central-bundle.zip') as z:
        entries = list(zip_entries(z))
        expected = {name.removeprefix('maven/'): meta for name, meta in manifest['files'].items() if name.startswith('maven/')}
        if {e.filename for e in entries} != set(expected):
            raise ValueError('Central bundle member mismatch')
        for e in entries:
            with z.open(e) as f:
                if digest_stream(f) != expected[e.filename]['sha256']:
                    raise ValueError('Central bundle differs from verified Maven payload')
    return manifest

if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='command', required=True)
    stage = sub.add_parser('stage'); stage.add_argument('downloads', type=Path)
    create = sub.add_parser('build')
    for name in ('maven', 'downloads', 'destination'): create.add_argument(name, type=Path)
    create.add_argument('--source', required=True); create.add_argument('--signing-key'); create.add_argument('--development', action='store_true')
    check = sub.add_parser('verify'); check.add_argument('directory', type=Path); check.add_argument('--development', action='store_true')
    a = p.parse_args()
    if a.command == 'stage': stage_natives(a.downloads, ROOT / 'bindings/java')
    elif a.command == 'build': build(a.maven, a.downloads, a.destination, a.source, a.signing_key, a.development)
    else: verify(a.directory, require_release=not a.development)
