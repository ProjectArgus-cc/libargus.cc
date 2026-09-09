"""Trusted signing boundary. Assembly has already finished before secrets arrive."""
import os
import subprocess
import tempfile
from pathlib import Path
from candidate import build

def finalize():
    release = os.environ.get('RELEASE') == 'true'
    if not release:
        build(Path('build/maven'), Path('native-binaries'), Path('candidate'), os.environ['SOURCE'], development=True)
        return
    key = os.environ.pop('GPG_PRIVATE_KEY', '')
    if not key: raise ValueError('Release signing key is missing')
    with tempfile.TemporaryDirectory(prefix='argus-signing-') as home:
        os.chmod(home, 0o700)
        os.environ['GNUPGHOME'] = home
        subprocess.run(['gpg', '--batch', '--import'], input=key.encode(), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        listing = subprocess.check_output(['gpg', '--batch', '--with-colons', '--list-secret-keys'], text=True)
        primary = [line.split(':')[9] for i, line in enumerate(listing.splitlines())
                   if line.startswith('fpr:') and i > 0 and listing.splitlines()[i-1].startswith('sec:')]
        if len(primary) != 1: raise ValueError('Expected exactly one primary signing key')
        build(Path('build/maven'), Path('native-binaries'), Path('candidate'), os.environ['SOURCE'], primary[0])

if __name__ == '__main__': finalize()
