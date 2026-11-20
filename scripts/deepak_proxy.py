# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward the Workbench application port to the Compose frontend."""

import asyncio
import os
import signal
import sys
from pathlib import Path

PID_FILE = Path.home() / ".cache" / "deep-researcher" / "deepak-proxy.pid"


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while data := await reader.read(65536):
        writer.write(data)
        await writer.drain()
    if writer.can_write_eof():
        writer.write_eof()


async def forward(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    upstream = None
    tasks: list[asyncio.Task[None]] = []
    try:
        upstream_reader, upstream = await asyncio.open_connection("frontend", 3000)
        tasks = [asyncio.create_task(pipe(reader, upstream)), asyncio.create_task(pipe(upstream_reader, writer))]
        await asyncio.gather(*tasks)
    except (ConnectionError, OSError) as exc:
        print(f"Deepak frontend connection failed: {exc}", file=sys.stderr, flush=True)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        writer.close()
        if upstream is not None:
            upstream.close()


async def serve() -> None:
    server = await asyncio.start_server(forward, "0.0.0.0", 3000)
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="ascii")
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stopped.set)
    try:
        async with server:
            await stopped.wait()
    finally:
        PID_FILE.unlink(missing_ok=True)


def stop() -> None:
    pid = int(PID_FILE.read_text(encoding="ascii"))
    if Path(f"/proc/{pid}").stat().st_uid != os.getuid():
        raise RuntimeError("PID file identifies a process owned by another user")
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if not any(Path(os.fsdecode(arg)).resolve() == Path(__file__).resolve() for arg in cmdline if arg):
        raise RuntimeError("PID file does not identify the Deepak proxy")
    os.kill(pid, signal.SIGTERM)


if __name__ == "__main__":
    if sys.argv[1:] == ["--stop"]:
        stop()
    elif not sys.argv[1:]:
        asyncio.run(serve())
    else:
        raise SystemExit("Usage: deepak_proxy.py [--stop]")
