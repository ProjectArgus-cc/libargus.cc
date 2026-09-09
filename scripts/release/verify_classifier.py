"""Load final JAR bytes in fresh JVMs and bind results to a sealed candidate."""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile
from candidate import verify
from catalog import ROOT, target, targets
from inventory import digest

def run(directory, row, java='java', development=False):
    manifest = verify(directory, require_release=not development)
    v = manifest['version']
    classifier = (directory / 'jars' / f'{row["module"]}-{v}.jar').resolve()
    core = (directory / 'jars' / f'libargus-core-{v}.jar').resolve()
    native = next(n for n in manifest['natives'] if n['target'] == row['id'])
    with ZipFile(classifier) as z:
        provider = z.read('META-INF/services/cc.projectargus.libargus.spi.NativeLibraryProvider').decode().strip()
    env = {k: v for k, v in os.environ.items() if k not in ('JAVA_TOOL_OPTIONS', '_JAVA_OPTIONS', 'JDK_JAVA_OPTIONS', 'CLASSPATH')}
    results = []
    for expected in (row['backend'], 'cuda' if row['backend'] == 'cpu' else 'cpu'):
        with tempfile.TemporaryDirectory(prefix='argus-verify-') as temporary:
            root = Path(temporary)
            command = [java, '--enable-native-access=ALL-UNNAMED', '-Dcc.projectargus.libargus.verifySpi=true',
                       '-Dcc.projectargus.libargus.nativeDir=' + str(root / 'native'),
                       '-cp', os.pathsep.join((str(core), str(classifier))),
                       'cc.projectargus.libargus.ClassifierVerifier', expected]
            if expected == row['backend'] and row['backend'] == 'cpu':
                command.append(str(ROOT / 'tests/data/tiny.gguf'))
            log = root / 'verifier.log'
            with log.open('wb') as out:
                completed = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=180)
            if log.stat().st_size > 4 * 1024**2: raise ValueError('Verifier output exceeded limit')
            output = log.read_text(errors='replace')
            messages = [line.removeprefix('ARGUS_RECEIPT ') for line in output.splitlines() if line.startswith('ARGUS_RECEIPT ')]
            if len(messages) != 1:
                raise ValueError('Verifier did not produce a unique receipt: ' + output[-2000:])
            result = json.loads(messages[0])
            positive = expected == row['backend']
            if completed.returncode != (0 if positive else 3) or result['result'] != ('ok' if positive else 'feature_mismatch'):
                raise ValueError('Verifier failed for an unexpected reason')
            if result['native_sha256'] != native['sha256'] or int(result['features']) != row['mask'] or result['provider'] != provider:
                raise ValueError('Loaded native/provider identity mismatch')
            if not Path(result['native_path']).resolve().is_relative_to(root.resolve()):
                raise ValueError('Verifier loaded a library outside its fresh extraction directory')
            for key, value in native['build'].items():
                if result.get(key) != value: raise ValueError(f'Build identity mismatch: {key}')
            if positive and row['backend'] == 'cpu' and result.get('cpu_decode') != 'passed':
                raise ValueError('Final CPU decode was not executed')
            result.pop('native_path')
            results.append(result)
    return {'schema': 1, 'target': row['id'], 'manifest_sha256': digest(directory / 'manifest.json'),
            'source': manifest['source'], 'core_sha256': digest(core), 'classifier_sha256': digest(classifier),
            'native_sha256': native['sha256'], 'positive': results[0], 'negative': results[1]}

def aggregate(directory, receipts, development=False):
    manifest = verify(directory, require_release=not development)
    expected = {r['id']: r for r in targets()}
    seen = set()
    if any(p.is_symlink() or (p.is_file() and p.suffix != '.json') for p in receipts.rglob('*')):
        raise ValueError('Unexpected receipt files')
    for path in receipts.rglob('*.json'):
        r = json.loads(path.read_text()); identity = r['target']
        if identity not in expected or identity in seen: raise ValueError('Unknown or duplicate receipt')
        seen.add(identity); row = expected[identity]
        if r['manifest_sha256'] != digest(directory / 'manifest.json') or r['source'] != manifest['source']:
            raise ValueError('Receipt is for a different candidate')
        for key, module in (('core_sha256', 'libargus-core'), ('classifier_sha256', row['module'])):
            if r[key] != digest(directory / 'jars' / f'{module}-{manifest["version"]}.jar'):
                raise ValueError('Receipt JAR mismatch')
        native = next(n for n in manifest['natives'] if n['target'] == identity)
        if r['native_sha256'] != native['sha256'] or r['positive']['result'] != 'ok' or r['negative']['result'] != 'feature_mismatch':
            raise ValueError('Receipt did not prove positive and negative identity checks')
        with ZipFile(directory / 'jars' / f'{row["module"]}-{manifest["version"]}.jar') as z:
            provider = z.read('META-INF/services/cc.projectargus.libargus.spi.NativeLibraryProvider').decode().strip()
        for result in (r['positive'], r['negative']):
            if result['native_sha256'] != native['sha256'] or int(result['features']) != row['mask'] or result['provider'] != provider:
                raise ValueError('Receipt loaded identity differs from candidate')
            if any(result.get(k) != v for k, v in native['build'].items()):
                raise ValueError('Receipt build metadata differs from candidate')
        if int(r['positive']['expected_features']) != row['mask'] or int(r['negative']['expected_features']) != (3 if row['backend'] == 'cpu' else 1):
            raise ValueError('Receipt expected masks do not establish the required comparison')
        if row['backend'] == 'cpu' and r['positive'].get('cpu_decode') != 'passed':
            raise ValueError('CPU classifier execution receipt missing')
    if seen != set(expected): raise ValueError('Missing classifier receipts')
    return manifest

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('candidate', type=Path); p.add_argument('target')
    p.add_argument('output', type=Path); p.add_argument('--development', action='store_true'); a = p.parse_args()
    if a.target == 'aggregate': aggregate(a.candidate, a.output, a.development)
    else:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(run(a.candidate, target(a.target), development=a.development), sort_keys=True, indent=2) + '\n')
