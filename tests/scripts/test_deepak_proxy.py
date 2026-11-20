# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[2] / "scripts" / "deepak_proxy.py"
_SPEC = importlib.util.spec_from_file_location("deepak_proxy", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
proxy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(proxy)


@pytest.mark.asyncio
async def test_forward_preserves_stream_and_bidirectional_upgrade(monkeypatch):
    async def upstream(reader, writer):
        assert await reader.readuntil(b"\r\n\r\n") == b"GET /websocket HTTP/1.1\r\nUpgrade: websocket\r\n\r\n"
        writer.write(b"HTTP/1.1 101 Switching Protocols\r\n\r\n")
        await writer.drain()
        writer.write(await reader.readexactly(4))
        await writer.drain()
        writer.close()

    backend = await asyncio.start_server(upstream, "127.0.0.1", 0)
    port = backend.sockets[0].getsockname()[1]
    open_connection = asyncio.open_connection

    async def connect(host, target_port):
        assert (host, target_port) == ("frontend", 3000)
        return await open_connection("127.0.0.1", port)

    monkeypatch.setattr(proxy.asyncio, "open_connection", connect)
    server = await asyncio.start_server(proxy.forward, "127.0.0.1", 0)
    reader, writer = await open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
    try:
        writer.write(b"GET /websocket HTTP/1.1\r\nUpgrade: websocket\r\n\r\n")
        await writer.drain()
        assert await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 2) == b"HTTP/1.1 101 Switching Protocols\r\n\r\n"
        writer.write(b"test")
        await writer.drain()
        assert await asyncio.wait_for(reader.readexactly(4), 2) == b"test"
    finally:
        writer.close()
        await writer.wait_closed()
        server.close()
        backend.close()
        await server.wait_closed()
        await backend.wait_closed()


@pytest.mark.asyncio
async def test_forward_reports_unavailable_frontend(monkeypatch, capsys):
    async def unavailable(*_args):
        raise ConnectionRefusedError("frontend unavailable")

    monkeypatch.setattr(proxy.asyncio, "open_connection", unavailable)
    writer = type("Writer", (), {"close": lambda self: None})()
    await proxy.forward(None, writer)
    assert "frontend unavailable" in capsys.readouterr().err
