"""Decide whether an exact tagged commit is owned by the release workflow."""
import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def is_tagged_release(event, source, root=ROOT):
    if not re.fullmatch(r'[0-9a-f]{40}', source):
        raise ValueError('Expected a full source commit')
    if event != 'push':
        return False
    release_version = (root / 'version.txt').read_text().strip()
    if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', release_version):
        raise ValueError('Invalid release version')
    expected = 'v' + release_version
    tags = subprocess.check_output(
        ['git', '-c', 'core.fsmonitor=false', 'tag', '--points-at', source, '--list', expected],
        cwd=root, text=True).splitlines()
    return tags == [expected]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('event')
    parser.add_argument('source')
    arguments = parser.parse_args()
    print(str(is_tagged_release(arguments.event, arguments.source)).lower())
