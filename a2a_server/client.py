#!/usr/bin/env python3
"""
Loci A2A Client — helper for calling the Loci memory server from other agents.

Usage from other agents (Python):
    from client import LociClient
    c = LociClient()
    results = await c.memory_recall("DAMA ant colony telemetry")
    await c.memory_remember("Resolved the k3s issue at 03:00 UTC", sender="hermes-agent")

Or from CLI:
    python3 client.py recall "DAMA"
    python3 client.py stats
    python3 client.py sessions "A2A"
    python3 client.py remember "content here" --sender hermes-agent
"""

import os, sys, json, asyncio, uuid

# Load .env
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

# Optionally import pyotp
try:
    import pyotp
    _PYOTP = True
except ImportError:
    _PYOTP = False


class LociClient:
    """
    Async A2A client for the Hermes Memory server.

    Auth conventions:
      - Bearer token in Authorization header
      - Optional X-TOTP header if TOTP seed is configured
    """

    def __init__(
        self,
        endpoint: str = None,
        token: str = None,
        totp_seed: str = None,
        sender: str = None
    ):
        self.endpoint  = (endpoint or os.environ.get('HERMES_A2A_URL',
                          'http://127.0.0.1:8201')).rstrip('/')
        self.token     = token or os.environ.get('HERMES_A2A_TOKEN', '')
        self.totp_seed = totp_seed or os.environ.get('HERMES_A2A_TOTP_SEED', '')
        self.sender    = sender or os.environ.get('HERMES_AGENT_ID', 'unknown')

    def _headers(self) -> dict:
        h = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        if self.totp_seed and _PYOTP:
            h['X-TOTP'] = pyotp.TOTP(self.totp_seed).now()
        return h

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
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as sess, sess.post(
            f'{self.endpoint}/a2a',
            json=payload,
            headers=self._headers()
        ) as resp:
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
        # parse --sender
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
