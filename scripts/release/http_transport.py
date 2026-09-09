"""Streaming HTTPS transport; credentials stay in headers and redirects strip them."""
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request

class HTTPFailure(RuntimeError):
    def __init__(self, status):
        super().__init__(f'HTTP request failed with status {status}')
        self.status = status

class Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).scheme != 'https' or req.method not in ('GET', 'HEAD'):
            raise ValueError('Refusing unsafe HTTP redirect')
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if urllib.parse.urlparse(req.full_url).netloc != urllib.parse.urlparse(newurl).netloc:
            redirected.remove_header('Authorization')
        return redirected

class Transport:
    def __init__(self): self.opener = urllib.request.build_opener(Redirect())
    def open(self, method, url, auth=None, data=None, headers=None):
        if not url.startswith('https://'): raise ValueError('HTTPS required')
        headers = {'User-Agent': 'libargus-release/1', **(headers or {})}
        if auth: headers['Authorization'] = auth
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try: return self.opener.open(request, timeout=60)
        except urllib.error.HTTPError as e: raise HTTPFailure(e.code) from None
    def request(self, method, url, auth=None, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        with self.open(method, url, auth, data, {'Content-Type': 'application/json', 'Accept': 'application/vnd.github+json'}) as response:
            body = response.read(4 * 1024**2 + 1)
            if len(body) > 4 * 1024**2: raise ValueError('Oversized API response')
            return json.loads(body) if body else None
    def upload(self, method, url, path, auth, multipart=False):
        boundary = 'argus-release-bundle-9e353491'
        prefix = (f'--{boundary}\r\nContent-Disposition: form-data; name="bundle"; filename="central-bundle.zip"\r\nContent-Type: application/octet-stream\r\n\r\n').encode() if multipart else b''
        suffix = f'\r\n--{boundary}--\r\n'.encode() if multipart else b''
        def chunks():
            yield prefix
            with path.open('rb') as source:
                for block in iter(lambda: source.read(1024**2), b''): yield block
            yield suffix
        headers = {'Content-Length': str(len(prefix) + path.stat().st_size + len(suffix)),
                   'Content-Type': 'multipart/form-data; boundary=' + boundary if multipart else 'application/octet-stream'}
        with self.open(method, url, auth, chunks(), headers) as response:
            body = response.read(4 * 1024**2 + 1)
            if len(body) > 4 * 1024**2: raise ValueError('Oversized upload response')
            return body.decode()
    def matches(self, url, meta, auth=None):
        try:
            with self.open('GET', url, auth, headers={'Accept': 'application/octet-stream'}) as response:
                h = hashlib.sha256(); total = 0
                for block in iter(lambda: response.read(1024**2), b''):
                    total += len(block)
                    if total > meta['size']: raise ValueError('Conflicting remote artifact size')
                    h.update(block)
                if total != meta['size'] or h.hexdigest() != meta['sha256']:
                    raise ValueError('Conflicting remote artifact bytes')
                return True
        except HTTPFailure as e:
            if e.status == 404: return False
            raise
