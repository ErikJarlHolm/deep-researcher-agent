# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi.testclient import TestClient

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
