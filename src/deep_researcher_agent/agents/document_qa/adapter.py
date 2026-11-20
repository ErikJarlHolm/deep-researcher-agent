# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bridge existing request-filtered tools and inference into the document core."""

import logging
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from deep_researcher_agent.agents.shallow_researcher.models import ShallowResearchAgentState
from deep_researcher_agent.common import get_latest_user_query
from deep_researcher_agent.common import load_prompt
from deep_researcher_agent.common.callbacks import SUPPRESS_OUTPUT_ARTIFACT_TAG
from deep_researcher_agent.common.evidence import split_evidence_content
from deep_researcher_agent.knowledge.schema import RetrievalResult
from deep_researcher_agent.relay import ainvoke_with_relay

from .agent import DocumentResearcher
from .agent import DocumentResponseError
from .models import DocumentQAConfig
from .models import Passage

logger = logging.getLogger(__name__)
PROMPTS = Path(__file__).parent / "prompts"


async def run_document_qa(
    state: ShallowResearchAgentState,
    *,
    config: DocumentQAConfig,
    tools: list[BaseTool],
    llm: BaseChatModel,
    extra_body: dict[str, object],
) -> ShallowResearchAgentState:
    """Use only the tool set already filtered and authorized by the caller."""
    search_tool = next((tool for tool in tools if tool.name == config.search_tool), None)
    if search_tool is None:
        raise RuntimeError("The configured document search tool is not available for this request")

    async def search(query: str) -> list[Passage]:
        raw = await search_tool.ainvoke({"query": query, "response_format": "json"})
        if not isinstance(raw, str):
            raise DocumentResponseError("Document search did not return a structured string")
        result = RetrievalResult.model_validate_json(raw)
        if not result.success:
            raise RuntimeError("Document retrieval failed")
        return [
            Passage(chunk.chunk_id, text, chunk.file_name, chunk.page_number)
            for chunk in result.chunks
            for text in split_evidence_content(chunk.content)
        ]

    async def generate(system: str, payload: str, tokens: int) -> str:
        model = llm.bind(
            temperature=0.0,
            max_tokens=tokens,
            extra_body={**extra_body, "response_format": {"type": "json_object"}},
        )
        response = await ainvoke_with_relay(
            model,
            [SystemMessage(content=system), HumanMessage(content=payload)],
            config={"tags": [SUPPRESS_OUTPUT_ARTIFACT_TAG]},
        )
        if not isinstance(response.content, str):
            raise DocumentResponseError("Document inference returned non-text content")
        return response.content

    agent = DocumentResearcher(
        search=search,
        generate=generate,
        config=config,
        generation_prompt=load_prompt(PROMPTS, "generate"),
        verification_prompt=load_prompt(PROMPTS, "verify"),
    )
    prior = [m for m in state.messages[:-1] if isinstance(m, (HumanMessage, AIMessage))][-4:]
    history = "\n".join(f"{message.type}: {str(message.content)[:800]}" for message in prior)
    answer = await agent.run(get_latest_user_query(state.messages), mode=state.research_mode, history=history)
    logger.info(
        "Document QA finished status=%s effort=%s searches=%d model_calls=%d seconds=%.2f budget_exhausted=%s",
        answer.status,
        answer.effort,
        answer.searches,
        answer.model_calls,
        answer.elapsed_seconds,
        answer.budget_exhausted,
    )
    return state.model_copy(
        update={"messages": [*state.messages, AIMessage(content=answer.text)], "document_answer": answer}
    )
