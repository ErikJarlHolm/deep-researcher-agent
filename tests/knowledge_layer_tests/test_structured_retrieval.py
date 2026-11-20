# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from knowledge_layer import register
from knowledge_layer.llamaindex.adapter import _CHROMA_EMBEDDING_MODEL_KEY
from knowledge_layer.llamaindex.adapter import LlamaIndexRetriever

from deep_researcher_agent.knowledge.schema import Chunk
from deep_researcher_agent.knowledge.schema import RetrievalResult


def chunk():
    return Chunk(
        chunk_id="chunk",
        content="The target is 2045.",
        file_name="Policy.pdf",
        page_number=3,
        display_citation="Policy.pdf, p.3",
        content_type="text",
        image_storage_uri="s3://internal/example",
        metadata={"internal": "not-model-evidence"},
    )


async def build_search(monkeypatch, retriever):
    monkeypatch.setattr(register, "_get_retriever", lambda config: retriever)
    monkeypatch.setattr(register, "_initialize_ingestor", lambda *args: MagicMock())
    monkeypatch.setattr("deep_researcher_agent.knowledge.factory.configure_summary_db", lambda *args: None)
    monkeypatch.setattr(register.Context, "get", lambda: SimpleNamespace(conversation_id="owned-conversation"))
    config = register.KnowledgeRetrievalConfig(backend="llamaindex", generate_summary=False, top_k=7)
    registration = register.knowledge_retrieval.__wrapped__(config, MagicMock())
    info = await anext(registration)

    async def invoke(query, response_format="text"):
        return await info.single_fn(info.input_schema(query=query, response_format=response_format))

    return invoke, registration


@pytest.mark.asyncio
async def test_registered_source_json_preserves_request_collection_citations_and_typed_failure(monkeypatch):
    retriever = SimpleNamespace(
        retrieve=AsyncMock(return_value=RetrievalResult(query="target", backend="llamaindex", chunks=[chunk()]))
    )
    search, registration = await build_search(monkeypatch, retriever)
    try:
        raw = await search("target", response_format="json")
        result = RetrievalResult.model_validate_json(raw)
        assert result.chunks[0].file_name == "Policy.pdf"
        assert result.chunks[0].page_number == 3
        assert "internal" not in raw
        retriever.retrieve.assert_awaited_once_with(query="target", collection_name="owned-conversation", top_k=7)

        text = await search("target")
        assert "Citation: Policy.pdf, p.3" in text
        assert "The target is 2045." in text

        retriever.retrieve.return_value = RetrievalResult(query="target", backend="llamaindex", success=False)
        failure = RetrievalResult.model_validate_json(await search("target", response_format="json"))
        assert failure.success is False
    finally:
        await registration.aclose()


@pytest.mark.asyncio
async def test_registered_structured_source_exception_is_not_a_successful_empty_result(monkeypatch):
    retriever = SimpleNamespace(retrieve=AsyncMock(side_effect=RuntimeError("index unavailable")))
    search, registration = await build_search(monkeypatch, retriever)
    try:
        with pytest.raises(RuntimeError, match="Structured knowledge search failed"):
            await search("target", response_format="json")
    finally:
        await registration.aclose()


def mock_llama_retriever(monkeypatch, inner):
    index = MagicMock()
    index.as_retriever.return_value = inner
    index_type = MagicMock()
    index_type.from_vector_store.return_value = index
    monkeypatch.setitem(sys.modules, "llama_index.core", SimpleNamespace(VectorStoreIndex=index_type))
    monkeypatch.setitem(sys.modules, "llama_index.vector_stores.chroma", SimpleNamespace(ChromaVectorStore=MagicMock()))
    monkeypatch.setattr(LlamaIndexRetriever, "_ensure_initialized", lambda self: None)
    retriever = LlamaIndexRetriever({"embed_model": "fixture-model"})
    retriever._chroma_client = MagicMock()
    retriever._chroma_client.get_collection.return_value.metadata = {_CHROMA_EMBEDDING_MODEL_KEY: "fixture-model"}
    monkeypatch.setattr(retriever, "normalize", lambda value: chunk())
    return retriever


@pytest.mark.asyncio
async def test_llamaindex_awaits_async_retrieval_instead_of_blocking_in_sync_embedding(monkeypatch):
    inner = SimpleNamespace(aretrieve=AsyncMock(return_value=["a node"]))
    retriever = mock_llama_retriever(monkeypatch, inner)
    result = await retriever.retrieve("target", "owned-conversation")
    assert result.success is True
    inner.aretrieve.assert_awaited_once_with("target")


@pytest.mark.asyncio
async def test_llamaindex_embedding_cancellation_propagates(monkeypatch):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def retrieve(query):
        started.set()
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    retriever = mock_llama_retriever(monkeypatch, SimpleNamespace(aretrieve=retrieve))
    task = asyncio.create_task(retriever.retrieve("target", "owned-conversation"))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
