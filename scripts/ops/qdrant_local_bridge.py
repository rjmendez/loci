#!/usr/bin/env python3
"""
Localhost TCP bridge for Qdrant.

Why this exists:
- loci-mcp policy expects QDRANT_URL on loopback (127.0.0.1/localhost).
- In this environment Qdrant runs in k3s, and kubectl port-forward can flap during
  node/pod churn.
- This bridge resolves the in-cluster qdrant service IP and forwards local
  loopback traffic to it.
"""

from __future__ import annotations

import asyncio
import os
import subprocess


LISTEN_HOST = os.environ.get("QDRANT_BRIDGE_LISTEN_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("QDRANT_BRIDGE_LISTEN_PORT", "30633"))
TARGET_PORT = int(os.environ.get("QDRANT_BRIDGE_TARGET_PORT", "6333"))


def discover_target_host() -> str:
    override = os.environ.get("QDRANT_BRIDGE_TARGET_HOST", "").strip()
    if override:
        return override

    cmd = [
        "kubectl",
        "get",
        "svc",
        "-n",
        "infra",
        "qdrant",
        "-o",
        "jsonpath={.spec.clusterIP}",
    ]
    cluster_ip = subprocess.check_output(cmd, text=True, timeout=5).strip()
    if not cluster_ip:
        raise RuntimeError("qdrant service clusterIP lookup returned empty output")
    return cluster_ip


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


async def handle_client(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    target_host: str,
) -> None:
    try:
        target_reader, target_writer = await asyncio.open_connection(target_host, TARGET_PORT)
    except Exception:
        client_writer.close()
        await client_writer.wait_closed()
        return

    await asyncio.gather(
        pipe(client_reader, target_writer),
        pipe(target_reader, client_writer),
    )


async def main() -> int:
    target_host = discover_target_host()
    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, target_host),
        LISTEN_HOST,
        LISTEN_PORT,
    )
    print(
        f"qdrant bridge listening on {LISTEN_HOST}:{LISTEN_PORT} -> "
        f"{target_host}:{TARGET_PORT}"
    )
    async with server:
        await server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
