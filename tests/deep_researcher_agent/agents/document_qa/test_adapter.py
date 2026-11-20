# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from deep_researcher_agent.agents.document_qa import adapter
from deep_researcher_agent.agents.document_qa.models import DocumentQAConfig
from deep_researcher_agent.agents.shallow_researcher.models import ShallowResearchAgentState
from deep_researcher_agent.knowledge.schema import Chunk
from deep_researcher_agent.knowledge.schema import RetrievalResult


@pytest.mark.asyncio
async def test_adapter_uses_filtered_tool_structured_payload_and_verifies_before_returning(monkeypatch):
    seen = []

    @tool("knowledge_search")
    async def search(query: str, response_format: str = "text") -> str:
        """Search authorized documents."""
        seen.append((query, response_format))
        return RetrievalResult(
            query=query,
            backend="fixture",
            chunks=[
                Chunk(
                    chunk_id="original-id",
                    content="The target year is 2045.",
                    file_name="Policy.pdf",
                    page_number=3,
                    display_citation="Policy.pdf, p.3",
                    content_type="text",
                )
            ],
        ).model_dump_json()

    infer = AsyncMock(
        side_effect=[
            AIMessage(
                content=json.dumps(
                    {
                        "claims": [{"text": "The target is 2045.", "passage_ids": ["p1"]}],
                        "search_queries": [],
                        "needs_deep_research": False,
                    }
                )
            ),
            AIMessage(content='{"supported_claim_ids":[0],"answer_complete":true}'),
        ]
    )
    monkeypatch.setattr(adapter, "ainvoke_with_relay", infer)
    state = ShallowResearchAgentState(
        messages=[HumanMessage(content="What is the target year?")],
        data_sources=["knowledge_layer"],
    )
    model = MagicMock()
    result = await adapter.run_document_qa(
        state,
        config=DocumentQAConfig(),
        tools=[search],
        llm=model,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    assert seen == [("What is the target year?", "json")]
    assert result.document_answer.status == "complete"
    assert "Policy.pdf, p.3" in result.messages[-1].content
    assert len(result.messages) == 2
    assert state.messages == [HumanMessage(content="What is the target year?")]
    assert model.bind.call_args.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert all(SUPPRESS_TAG in call.kwargs["config"]["tags"] for call in infer.call_args_list)


SUPPRESS_TAG = adapter.SUPPRESS_OUTPUT_ARTIFACT_TAG


@pytest.mark.asyncio
async def test_adapter_does_not_replace_a_request_filtered_out_tool(monkeypatch):
    infer = AsyncMock()
    monkeypatch.setattr(adapter, "ainvoke_with_relay", infer)
    with pytest.raises(RuntimeError, match="not available for this request"):
        await adapter.run_document_qa(
            ShallowResearchAgentState(messages=[HumanMessage(content="Question")]),
            config=DocumentQAConfig(),
            tools=[],
            llm=MagicMock(),
            extra_body={},
        )
    infer.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_structured_retrieval_never_reaches_generation(monkeypatch):
    @tool("knowledge_search")
    async def search(query: str, response_format: str = "text") -> str:
        """Search authorized documents."""
        return RetrievalResult(query=query, backend="fixture", success=False).model_dump_json()

    infer = AsyncMock()
    monkeypatch.setattr(adapter, "ainvoke_with_relay", infer)
    with pytest.raises(RuntimeError, match="retrieval failed"):
        await adapter.run_document_qa(
            ShallowResearchAgentState(messages=[HumanMessage(content="Question")]),
            config=DocumentQAConfig(),
            tools=[search],
            llm=MagicMock(),
            extra_body={},
        )
    infer.assert_not_awaited()
