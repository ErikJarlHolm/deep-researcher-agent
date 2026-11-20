# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from deep_researcher_agent import document_preview
from deep_researcher_agent.agents.document_qa.models import DocumentAnswer


def test_preview_only_accepts_authenticated_document_requests(monkeypatch):
    class Researcher:
        def __init__(self, **_):
            pass

        async def run(self, question, *, mode):
            assert (question, mode) == ("What is the retention period?", "quick")
            return DocumentAnswer("37 days [1]", "complete", "quick", 1, 2, 0.1)

    monkeypatch.setattr(document_preview, "DocumentResearcher", Researcher)
    monkeypatch.setenv("DOCUMENT_PREVIEW_TOKEN", "test-token")
    endpoint = "/v1/documents/answer"
    payload = {"question": " What is the retention period? ", "mode": "quick"}
    headers = {"Authorization": "Bearer test-token"}
    with TestClient(document_preview.create_app()) as client:
        assert client.post(endpoint, json=payload).status_code == 401
        assert client.post(endpoint, headers=headers, json={**payload, "mode": "deep"}).status_code == 422
        assert client.post(endpoint, headers=headers, json={"question": "  "}).status_code == 422
        response = client.post(endpoint, headers=headers, json=payload)
        assert response.status_code == 200
        assert response.json()["text"] == "37 days [1]"


def test_preview_without_token_fails_closed(monkeypatch):
    monkeypatch.delenv("DOCUMENT_PREVIEW_TOKEN", raising=False)
    with TestClient(document_preview.create_app()) as client:
        response = client.post("/v1/documents/answer", json={"question": "A test question"})
    assert response.status_code == 503


def test_stream_sends_progress_then_only_the_verified_answer(monkeypatch):
    class Researcher:
        def __init__(self, **_):
            pass

        async def run(self, _question, *, mode):
            await asyncio.sleep(0.03)
            return DocumentAnswer("37 days [1]\n\n### Sources\n[1] fixture.txt", "complete", mode, 1, 2, 0.03)

    monkeypatch.setattr(document_preview, "DocumentResearcher", Researcher)
    monkeypatch.setattr(document_preview, "HEARTBEAT_SECONDS", 0.005)
    monkeypatch.setenv("DOCUMENT_PREVIEW_TOKEN", "test-token")
    with TestClient(document_preview.create_app()) as client:
        response = client.post(
            "/v1/documents/answer/stream",
            headers={"Authorization": "Bearer test-token"},
            json={"question": "How long?", "mode": "quick"},
        )
    events = response.text.strip().split("\n\n")
    assert response.headers["content-type"].startswith("text/event-stream")
    assert events[0] == "event: accepted\ndata: {}"
    assert all(event == "event: working\ndata: {}" for event in events[1:-1])
    assert len(events) >= 3
    name, payload = events[-1].splitlines()
    assert name == "event: answer"
    assert json.loads(payload.removeprefix("data: "))["text"].endswith("[1] fixture.txt")


@pytest.mark.parametrize(("failure", "status"), [(TimeoutError, 504), (document_preview.DocumentResponseError, 502)])
def test_model_failures_are_not_reported_as_missing_evidence(monkeypatch, failure, status):
    class Researcher:
        def __init__(self, **_):
            pass

        async def run(self, _question, *, mode):
            raise failure("internal detail")

    monkeypatch.setattr(document_preview, "DocumentResearcher", Researcher)
    monkeypatch.setenv("DOCUMENT_PREVIEW_TOKEN", "test-token")
    headers = {"Authorization": "Bearer test-token"}
    payload = {"question": "How long?", "mode": "quick"}
    with TestClient(document_preview.create_app()) as client:
        assert client.post("/v1/documents/answer", headers=headers, json=payload).status_code == status
        response = client.post("/v1/documents/answer/stream", headers=headers, json=payload)

    assert response.status_code == 200
    _, terminal = response.text.strip().split("\n\n")
    name, data = terminal.splitlines()
    assert name == "event: error"
    assert json.loads(data.removeprefix("data: "))["status"] == status
    assert "internal detail" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnect_kind", ["json", "stream", "send_failure"])
async def test_busy_request_is_rejected_and_disconnect_releases_slot(monkeypatch, disconnect_kind):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class Researcher:
        def __init__(self, **_):
            pass

        async def run(self, question, *, mode):
            if question == "slow question":
                started.set()
                try:
                    await asyncio.Future()
                finally:
                    cancelled.set()
            return DocumentAnswer("37 days [1]", "complete", mode, 1, 2, 0.03)

    monkeypatch.setattr(document_preview, "DocumentResearcher", Researcher)
    monkeypatch.setenv("DOCUMENT_PREVIEW_TOKEN", "test-token")
    app = document_preview.create_app()

    async def request(question, disconnected=None, *, stream=False, send_failure=False):
        body = json.dumps({"question": question, "mode": "quick"}).encode()
        sent = asyncio.Event()
        messages = []

        async def receive():
            if not sent.is_set():
                sent.set()
                return {"type": "http.request", "body": body}
            if disconnected is not None:
                await disconnected.wait()
                return {"type": "http.disconnect"}
            await asyncio.Future()

        async def send(message):
            messages.append(message)
            if send_failure and message.get("body", b"").startswith(b"event: accepted"):
                await disconnected.wait()
                raise OSError("Connection lost")

        path = "/v1/documents/answer" + ("/stream" if stream else "")
        await app(
            {
                "type": "http",
                "asgi": {"spec_version": "2.4" if send_failure else "2.3"},
                "method": "POST",
                "path": path,
                "query_string": b"",
                "headers": [(b"authorization", b"Bearer test-token"), (b"content-type", b"application/json")],
            },
            receive,
            send,
        )
        return messages

    async with app.router.lifespan_context(app):
        disconnect = asyncio.Event()
        first = asyncio.create_task(
            request(
                "slow question",
                disconnect,
                stream=disconnect_kind != "json",
                send_failure=disconnect_kind == "send_failure",
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 2)
            busy = await asyncio.wait_for(request("second question"), 2)
            assert busy[0]["status"] == 429
            assert (b"retry-after", b"2") in busy[0]["headers"]
            busy_stream = await asyncio.wait_for(request("second question", stream=True), 2)
            events = b"".join(message.get("body", b"") for message in busy_stream)
            assert events.startswith(b"event: error\n")
            assert b"accepted" not in events
            assert b'"retry_after_seconds": 2' in events
        finally:
            disconnect.set()
            if disconnect_kind == "send_failure":
                with pytest.raises(ClientDisconnect):
                    await asyncio.wait_for(first, 2)
            else:
                await asyncio.wait_for(first, 2)
        assert cancelled.is_set()
        accepted = await asyncio.wait_for(request("third question"), 2)
        assert accepted[0]["status"] == 200
