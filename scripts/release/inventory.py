"""Bounded streaming hashes and strict, portable ZIP inventories."""
import hashlib
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo

def digest_stream(stream, algorithm='sha256'):
    h = hashlib.new(algorithm)
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()

def digest(path, algorithm='sha256'):
    with Path(path).open('rb') as f:
        return digest_stream(f, algorithm)

def abi_header_digest(path):
    """Hash the public ABI header after canonical CRLF-to-LF normalization."""
    data = Path(path).read_bytes().replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()

def safe_name(name):
    p = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError(f'Unsafe archive path: {name}')
    return p

def zip_entries(archive):
    names = set()
    for entry in archive.infolist():
        name = entry.filename.rstrip('/')
        safe_name(name)
        if name in names or (entry.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f'Duplicate or symlink archive entry: {name}')
        names.add(name)
        if not entry.is_dir():
            if entry.file_size > 8 * 1024**3:
                raise ValueError('Archive entry exceeds native distribution limit')
            yield entry

def files(root):
    root = Path(root)
    result = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError(f'Symlink in candidate: {p}')
        if p.is_file():
            name = p.relative_to(root).as_posix()
            safe_name(name)
            result[name] = {'sha256': digest(p), 'size': p.stat().st_size}
    return result

def write_zip(path, members):
    with ZipFile(path, 'w', compression=ZIP_DEFLATED, allowZip64=True) as z:
        for name, source in sorted(members.items()):
            safe_name(name)
            entry = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            with Path(source).open('rb') as src, z.open(entry, 'w', force_zip64=True) as dest:
                for block in iter(lambda: src.read(1024 * 1024), b''):
                    dest.write(block)
