#!/usr/bin/env python3
"""Loci A2A client for agents and CLI callers.

Python:
    from client import LociClient
    c = LociClient()
    results = await c.memory_recall("DAMA ant colony telemetry")
    await c.memory_remember("Resolved the k3s issue at 03:00 UTC", sender="hermes-agent")

Signed requests (Ed25519, instead of a bearer token): set A2A_SIGNING_KEY_FILE to a PEM private
key whose public half the server has registered for HERMES_AGENT_ID (PEER_PUBKEYS_DIR / _JSON).
A2A_SIGNING_VERSION picks the wire format: 2 (default) binds timestamp, nonce, method, path and
body so a captured request cannot be replayed; 1 signs the raw body only (older servers).

CLI:
    python3 client.py recall "DAMA"
    python3 client.py stats
    python3 client.py sessions "A2A"
    python3 client.py remember "content here" --sender hermes-agent
"""

import os, sys, json, asyncio, uuid, base64, hashlib, time
from urllib.parse import urlsplit

# Load ~/.hermes/.env if present.
_ENV = os.path.expanduser('~/.hermes/.env')
if os.path.exists(_ENV):
    for _l in open(_ENV):
        _l = _l.strip()
        if _l and not _l.startswith('#') and '=' in _l:
            _k, _v = _l.split('=', 1)
            os.environ.setdefault(_k.strip(), _v.strip())

try:
    import aiohttp
except ImportError:
    sys.exit('aiohttp required: pip install aiohttp')

# Optional TOTP support.
try:
    import pyotp
    _PYOTP = True
except ImportError:
    _PYOTP = False

# Optional Ed25519 request signing.
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    _CRYPTO = True
except ImportError:
    _CRYPTO = False


def _load_signing_key(path: str):
    """The Ed25519 private key in `path`, or a RuntimeError: a key that was asked for and cannot be
    used must not quietly turn into an unsigned (bearer) request."""
    if not _CRYPTO:
        raise RuntimeError('signing key configured but the cryptography package is not installed')
    try:
        with open(os.path.expanduser(path), 'rb') as fh:
            key = load_pem_private_key(fh.read(), password=None)
    except Exception as e:
        raise RuntimeError(f'signing key unusable ({e.__class__.__name__})') from None
    if not isinstance(key, Ed25519PrivateKey):
        raise RuntimeError('signing key unusable (not an Ed25519 private key)')
    return key


def _sig_payload_v2(agent_id: str, ts: str, nonce: str, method: str, path: str, body: bytes) -> bytes:
    # Must match a2a_server/server.py _sig_payload_v2 byte for byte.
    return '\n'.join(['a2a-sig-v2', agent_id, ts, nonce, method.upper(), path,
                      hashlib.sha256(body).hexdigest()]).encode('utf-8')


class LociClient:
    """Async client for the Loci A2A memory server.

    Auth:
      - ****** in Authorization
      - X-TOTP only when a TOTP seed is configured
      - or, when a signing key is configured, an Ed25519 signature as `sender` (no bearer, no TOTP)
    """

    def __init__(
        self,
        endpoint: str = None,
        token: str = None,
        totp_seed: str = None,
        sender: str = None,
        signing_key_file: str = None,
        signature_version: int = None,
    ):
        self.endpoint  = (endpoint or os.environ.get('LOCI_A2A_URL',
                          'http://127.0.0.1:8201')).rstrip('/')
        self.token     = token or os.environ.get('LOCI_A2A_TOKEN', '')
        self.totp_seed = totp_seed or os.environ.get('LOCI_A2A_TOTP_SEED', '')
        self.sender    = sender or os.environ.get('HERMES_AGENT_ID', 'unknown')
        key_path       = signing_key_file or os.environ.get('A2A_SIGNING_KEY_FILE', '')
        self._signing_key = _load_signing_key(key_path) if key_path else None
        self.signature_version = int(signature_version or os.environ.get('A2A_SIGNING_VERSION', 2))
        if self.signature_version not in (1, 2):
            raise ValueError(f'signature_version must be 1 or 2, got {self.signature_version}')

    def _headers(self) -> dict:
        h = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        if self.totp_seed and _PYOTP:
            h['X-TOTP'] = pyotp.TOTP(self.totp_seed).now()
        return h

    def _signed_request(self, url: str, payload: dict) -> tuple:
        """(body bytes, headers) for a request signed as `self.sender`."""
        body = json.dumps(payload, separators=(',', ':')).encode('utf-8')
        ts, nonce = str(int(time.time())), uuid.uuid4().hex
        signed = (_sig_payload_v2(self.sender, ts, nonce, 'POST', urlsplit(url).path or '/', body)
                  if self.signature_version == 2 else body)
        headers = {
            'Content-Type': 'application/json',
            'X-Agent-ID': self.sender,
            'X-Timestamp': ts,
            'X-Request-ID': nonce,
            'X-Signature': base64.urlsafe_b64encode(self._signing_key.sign(signed)).decode('ascii'),
        }
        if self.signature_version == 2:
            headers['X-Signature-Version'] = '2'
        return body, headers

    async def _call(self, skill_id: str, message: str = '', input_data: dict = None) -> dict:
        payload = {
            'jsonrpc': '2.0',
            'id': str(uuid.uuid4()),
            'method': 'tasks/send',
            'params': {
                'skill_id': skill_id,
                'message': message,
                'input': input_data or {},
                'sender': self.sender
            }
        }
        url = f'{self.endpoint}/a2a'
        if self._signing_key is not None:
            data, headers = self._signed_request(url, payload)
            post_kwargs = {'data': data, 'headers': headers}
        else:
            post_kwargs = {'json': payload, 'headers': self._headers()}
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as sess, sess.post(url, **post_kwargs) as resp:
            data = await resp.json()
            if resp.status != 200:
                return {'error': f'HTTP {resp.status}', 'detail': data}
            return data.get('result', {}).get('output', data)

    async def memory_recall(
        self, query: str, top_k: int = 5, semantic: bool = True
    ) -> dict:
        return await self._call(
            'memory_recall', message=query,
            input_data={'query': query, 'top_k': top_k, 'semantic': semantic}
        )

    async def memory_remember(
        self, content: str, source: str = 'a2a',
        importance: float = 0.5, bank: str = 'default',
        sender: str = None
    ) -> dict:
        old = self.sender
        if sender:
            self.sender = sender
        result = await self._call(
            'memory_remember', message=content,
            input_data={'content': content, 'source': source,
                        'importance': importance, 'bank': bank}
        )
        self.sender = old
        return result

    async def memory_stats(self) -> dict:
        return await self._call('memory_stats')

    async def session_search(
        self, query: str, top_k: int = 5, agent_id: str = None
    ) -> dict:
        inp: dict = {'query': query, 'top_k': top_k}
        if agent_id:
            inp['agent_id'] = agent_id
        return await self._call('session_search', message=query, input_data=inp)

    async def memory_sleep(self, dry_run: bool = False) -> dict:
        return await self._call('memory_sleep', input_data={'dry_run': dry_run})

    async def health(self) -> dict:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as sess, sess.get(f'{self.endpoint}/health') as r:
            return await r.json()


# ── CLI ──────────────────────────────────────────────────────────────────────────
async def _cli_main(args: list[str]):
    if not args:
        print(__doc__)
        return

    cmd = args[0]
    c = LociClient()

    if cmd == 'health':
        r = await c.health()
    elif cmd == 'stats':
        r = await c.memory_stats()
    elif cmd in ('recall', 'search'):
        query = ' '.join(args[1:]) or 'test'
        r = await c.memory_recall(query)
    elif cmd == 'sessions':
        query = ' '.join(args[1:]) or 'test'
        r = await c.session_search(query)
    elif cmd == 'remember':
        content = ' '.join(args[1:])
        if not content:
            print('Usage: client.py remember <content> [--sender name]')
            return
        # Parse --sender flag.
        sender = None
        if '--sender' in args:
            idx = args.index('--sender')
            sender = args[idx + 1] if idx + 1 < len(args) else None
            content = content.replace(f'--sender {sender}', '').strip()
        r = await c.memory_remember(content, sender=sender)
    elif cmd == 'sleep':
        dry = '--dry' in args or '--dry-run' in args
        r = await c.memory_sleep(dry_run=dry)
    else:
        print(f'Unknown command: {cmd}')
        print('Commands: health, stats, recall, sessions, remember, sleep')
        return

    print(json.dumps(r, indent=2, default=str))


if __name__ == '__main__':
    asyncio.run(_cli_main(sys.argv[1:]))
