#!/usr/bin/env python3
"""Real-socket TCP forwarder: 0.0.0.0:LISTEN -> vLLM k8s NodePort.

Why this exists: on WSL2 (mirrored networking) the k8s NodePort is an iptables DNAT rule,
not a real listening socket, so the Windows-side tailscaled cannot proxy to it. A real
userspace LISTEN socket in WSL *is* mirrored to Windows localhost, which tailscaled can then
`tailscale serve`. This process is that real socket. Run it durably (systemd), then:
    tailscale serve --bg --https=8000 http://127.0.0.1:<LISTEN>
"""
import os
import socket
import threading

LISTEN = ("0.0.0.0", int(os.environ.get("VLLM_FWD_PORT", "18000")))
TARGET = (os.environ.get("VLLM_FWD_TARGET_HOST", "172.21.171.198"),
          int(os.environ.get("VLLM_FWD_TARGET_PORT", "32272")))


def _pipe(a: socket.socket, b: socket.socket) -> None:
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _handle(client: socket.socket) -> None:
    try:
        upstream = socket.create_connection(TARGET, timeout=10)
    except OSError:
        client.close()
        return
    threading.Thread(target=_pipe, args=(client, upstream), daemon=True).start()
    threading.Thread(target=_pipe, args=(upstream, client), daemon=True).start()


def main() -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(LISTEN)
    srv.listen(256)
    print(f"forwarding {LISTEN[0]}:{LISTEN[1]} -> {TARGET[0]}:{TARGET[1]}", flush=True)
    while True:
        client, _ = srv.accept()
        threading.Thread(target=_handle, args=(client,), daemon=True).start()


if __name__ == "__main__":
    main()
