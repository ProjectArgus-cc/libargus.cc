"""Resolve the exact existing release tag before any expensive jobs start."""
import argparse
import json
import os
import re
import subprocess
from catalog import ROOT, targets, version

def git(*args):
    return subprocess.check_output(['git', '-c', 'core.fsmonitor=false', *args], cwd=ROOT, text=True).strip()

def validate_header_version(root=ROOT):
    header_path = root / 'include' / 'libargus.h'
    if header_path.is_file():
        header_text = header_path.read_text(encoding='utf-8')
        m = re.search(r'@version\s+([0-9]+\.[0-9]+\.[0-9]+)', header_text)
        if not m:
            raise ValueError(f"No @version found in {header_path}")
        header_ver = m.group(1)
        ver = (root / 'version.txt').read_text().strip() if (root / 'version.txt').is_file() else version()
        if header_ver != ver:
            raise ValueError(f"include/libargus.h @version ({header_ver}) differs from version.txt ({ver})")

def validate_header_no_test_symbols(root=ROOT):
    header_path = root / 'include' / 'libargus.h'
    if header_path.is_file():
        header_text = header_path.read_text(encoding='utf-8')
        test_symbols = re.findall(r'\b(argus_test_\w+)\b', header_text)
        if test_symbols:
            raise ValueError(f"Test-only symbols found in public header {header_path}: {test_symbols}")

def inspect_tag_object(tag):
    obj_type = git('cat-file', '-t', f'refs/tags/{tag}')
    if obj_type == 'tag':
        tag_type = 'annotated'
        raw_tag = git('cat-file', 'tag', f'refs/tags/{tag}')
        tag_signed = (
            '-----BEGIN PGP SIGNATURE-----' in raw_tag
            or '-----BEGIN SSH SIGNATURE-----' in raw_tag
        )
    else:
        tag_type = 'lightweight'
        tag_signed = False
    if os.environ.get('REQUIRE_SIGNED_TAG') == '1' and (tag_type != 'annotated' or not tag_signed):
        raise ValueError(f"Release tag {tag} must be an annotated signed tag (got type={tag_type}, signed={tag_signed})")
    return tag_type, tag_signed

def preflight(tag):
    if not re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+', tag):
        raise ValueError('Expected an existing vMAJOR.MINOR.PATCH release tag')
    if tag != 'v' + version(): raise ValueError('Tag differs from version.txt')
    validate_header_version(ROOT)
    validate_header_no_test_symbols(ROOT)
    commit = git('rev-parse', '--verify', f'refs/tags/{tag}^{{commit}}')
    if commit != git('rev-parse', 'HEAD'): raise ValueError('Checkout is not the release tag commit')
    if git('status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('Release checkout is dirty')
    tag_type, tag_signed = inspect_tag_object(tag)
    targets()
    return {
        'tag': tag,
        'source': commit,
        'version': version(),
        'tag_type': tag_type,
        'tag_signed': tag_signed,
    }

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('tag'); a = p.parse_args()
    print(json.dumps(preflight(a.tag)))

