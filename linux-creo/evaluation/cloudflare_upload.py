#!/usr/bin/env python3
"""Upload data over one TCP socket explicitly configured with CREO."""

from __future__ import annotations

import argparse
import json
import socket
import ssl
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="speed.cloudflare.com")
    parser.add_argument("--port", type=int, default=443)
    parser.add_argument("--bytes", type=int, default=20_000_000)
    parser.add_argument("--cc", default="creo")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bytes <= 0:
        raise SystemExit("--bytes must be positive")

    addresses = socket.getaddrinfo(
        args.host, args.port, socket.AF_INET, socket.SOCK_STREAM
    )
    if not addresses:
        raise SystemExit(f"no IPv4 address for {args.host}")
    family, socktype, protocol, _, address = addresses[0]

    raw_socket = socket.socket(family, socktype, protocol)
    raw_socket.settimeout(args.timeout)
    raw_socket.setsockopt(
        socket.IPPROTO_TCP, socket.TCP_CONGESTION, args.cc.encode("ascii")
    )
    selected_cc = raw_socket.getsockopt(
        socket.IPPROTO_TCP, socket.TCP_CONGESTION, 32
    ).rstrip(b"\0").decode("ascii")

    connected_started = time.monotonic()
    raw_socket.connect(address)
    tcp_connected = time.monotonic()
    context = ssl.create_default_context()
    tls_socket = context.wrap_socket(raw_socket, server_hostname=args.host)
    tls_connected = time.monotonic()

    path = "/__up?creo_drl_socket=1"
    headers = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {args.host}\r\n"
        "User-Agent: creo-drl-linux-evaluator/1\r\n"
        "Content-Type: application/octet-stream\r\n"
        f"Content-Length: {args.bytes}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii")
    tls_socket.sendall(headers)

    chunk = bytes(64 * 1024)
    remaining = args.bytes
    upload_started = time.monotonic()
    while remaining:
        current = chunk if remaining >= len(chunk) else chunk[:remaining]
        tls_socket.sendall(current)
        remaining -= len(current)
    upload_finished = time.monotonic()

    response = bytearray()
    while len(response) < 4096:
        data = tls_socket.recv(4096 - len(response))
        if not data:
            break
        response.extend(data)
        if b"\r\n" in response:
            break
    tls_socket.close()
    finished = time.monotonic()

    status_line = response.split(b"\r\n", 1)[0].decode("ascii", "replace")
    upload_s = upload_finished - upload_started
    result = {
        "host": args.host,
        "remote_ip": address[0],
        "selected_cc": selected_cc,
        "bytes_uploaded": args.bytes,
        "tcp_connect_ms": (tcp_connected - connected_started) * 1000.0,
        "tls_handshake_ms": (tls_connected - tcp_connected) * 1000.0,
        "upload_s": upload_s,
        "upload_mbps": args.bytes * 8.0 / upload_s / 1e6,
        "total_s": finished - connected_started,
        "http_status": status_line,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status_line.startswith("HTTP/1.1 200") else 1


if __name__ == "__main__":
    sys.exit(main())
