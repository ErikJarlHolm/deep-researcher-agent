# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding-identity checks for persisted LlamaIndex/Chroma collections."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from knowledge_layer.llamaindex.adapter import _CHROMA_EMBEDDING_MODEL_KEY
from knowledge_layer.llamaindex.adapter import LlamaIndexIngestor
from knowledge_layer.llamaindex.adapter import LlamaIndexRetriever
from knowledge_layer.llamaindex.adapter import _get_nvidia_api_key
from knowledge_layer.llamaindex.adapter import _validate_chroma_embedding_model
from knowledge_layer.register import KnowledgeRetrievalConfig
from knowledge_layer.register import _setup_backend


def test_llamaindex_uses_configured_embedding_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DEEP_RESEARCHER_CHROMA_DIR", str(tmp_path))
    monkeypatch.setenv("DEEP_RESEARCHER_EMBED_MODEL", "hosted-model")
    monkeypatch.setenv("DEEP_RESEARCHER_EMBED_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setattr(LlamaIndexIngestor, "_start_ttl_cleanup_task", lambda *args: None)
    config = KnowledgeRetrievalConfig(
        backend="llamaindex",
        chroma_dir=str(tmp_path),
        embed_model="nvidia/Nemotron-3.5-Embed-1B-BF16",
        embed_base_url="http://embedder:8000/v1",
    )

    _, backend_config = _setup_backend(config)

    for adapter_type in (LlamaIndexIngestor, LlamaIndexRetriever):
        adapter = adapter_type(backend_config)
        assert adapter.embed_model_name == config.embed_model
        assert adapter.embed_base_url == config.embed_base_url


def test_chroma_collection_accepts_matching_embedding_model() -> None:
    collection = SimpleNamespace(metadata={_CHROMA_EMBEDDING_MODEL_KEY: "nvidia/test-embed"})

    _validate_chroma_embedding_model(collection, "docs", "nvidia/test-embed")


def test_embedding_api_key_override_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEP_RESEARCHER_EMBED_API_KEY", "embedding-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "default-key")

    assert _get_nvidia_api_key() == "embedding-key"


def test_embedding_api_key_falls_back_to_nvidia_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEP_RESEARCHER_EMBED_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "default-key")

    assert _get_nvidia_api_key() == "default-key"


@pytest.mark.parametrize("metadata", [{}, {_CHROMA_EMBEDDING_MODEL_KEY: "nvidia/old-embed"}])
def test_chroma_collection_rejects_unknown_or_mismatched_embedding_model(metadata: dict[str, str]) -> None:
    collection = SimpleNamespace(metadata=metadata)

    with pytest.raises(RuntimeError, match="Delete and re-ingest"):
        _validate_chroma_embedding_model(collection, "docs", "nvidia/new-embed")
