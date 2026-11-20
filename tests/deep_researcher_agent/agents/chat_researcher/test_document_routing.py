# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from deep_researcher_agent.agents.chat_researcher.agent import ChatResearcherAgent
from deep_researcher_agent.agents.chat_researcher.models import ChatResearcherState
from deep_researcher_agent.agents.chat_researcher.models import DepthDecision
from deep_researcher_agent.agents.chat_researcher.models import IntentResult
from deep_researcher_agent.agents.chat_researcher.preclassification import preclassified_depth
from deep_researcher_agent.agents.document_qa.agent import DocumentBudgetExceeded
from deep_researcher_agent.agents.document_qa.models import DocumentAnswer
from deep_researcher_agent.agents.document_qa.models import DocumentQAConfig
from deep_researcher_agent.agents.shallow_researcher.models import ShallowResearchAgentState


def make_agent(**options):
    classifier = AsyncMock(
        return_value={
            "user_intent": IntentResult(intent="research"),
            "depth_decision": DepthDecision(decision="deep"),
        }
    )

    async def shallow(state):
        return state.model_copy(
            update={
                "messages": [*state.messages, AIMessage(content="A verified answer.")],
                "document_answer": DocumentAnswer(
                    text="A verified answer.",
                    status="complete",
                    effort="quick",
                    searches=1,
                    model_calls=2,
                    elapsed_seconds=1,
                ),
            }
        )

    async def deep(state):
        return ShallowResearchAgentState(messages=[AIMessage(content="Deep report.")])

    shallow_fn = AsyncMock(side_effect=shallow)
    deep_fn = AsyncMock(side_effect=deep)
    agent = ChatResearcherAgent(
        intent_classifier_fn=classifier,
        shallow_research_fn=shallow_fn,
        deep_research_fn=deep_fn,
        clarifier_fn=None,
        enable_clarifier=False,
        document_qa=DocumentQAConfig(),
        **options,
    )
    return agent, classifier, shallow_fn, deep_fn


def state(**options):
    return ChatResearcherState(
        messages=[HumanMessage(content="Compare the target and the measures in the documents.")],
        data_sources=["knowledge_layer"],
        **options,
    )


@pytest.mark.asyncio
async def test_comparison_starts_with_documents_instead_of_complexity_classifier():
    agent, classifier, shallow, deep = make_agent()
    result = await agent.run(state())
    classifier.assert_not_awaited()
    deep.assert_not_awaited()
    assert shallow.call_args.args[0].document_qa == DocumentQAConfig()
    assert result["shallow_result"].escalate_to_deep is False


@pytest.mark.asyncio
async def test_explicit_deep_mode_retains_planning_research_and_writer_path():
    agent, classifier, shallow, deep = make_agent()
    await agent.run(state(research_mode="deep"))
    classifier.assert_not_awaited()
    shallow.assert_not_awaited()
    deep.assert_awaited_once()


@pytest.mark.asyncio
async def test_preclassified_depth_is_honored_without_running_another_classifier():
    agent, classifier, shallow, deep = make_agent()
    with preclassified_depth("deep"):
        await agent.run(state())
    classifier.assert_not_awaited()
    shallow.assert_not_awaited()
    deep.assert_awaited_once()


@pytest.mark.asyncio
async def test_quick_mode_overrides_a_caller_classifier_depth_but_does_not_leak_next_turn():
    agent, classifier, shallow, deep = make_agent(checkpointer=MemorySaver())
    with preclassified_depth("deep"):
        await agent.run(state(research_mode="quick"), thread_id="one-conversation")
    assert shallow.call_args.args[0].research_mode == "quick"
    deep.assert_not_awaited()
    await agent.run(
        {
            "messages": [HumanMessage(content="What about the implementation measures?")],
            "data_sources": ["knowledge_layer"],
        },
        thread_id="one-conversation",
    )
    assert shallow.call_args.args[0].research_mode == "auto"
    classifier.assert_not_awaited()
    assert shallow.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context",
    [
        {"data_sources": []},
        {"data_sources": ["knowledge_layer", "web_search"]},
        {"active_report_job_id": "authorized-report"},
        {"last_report_markdown": "Prior report"},
        {"database_name": "finance"},
    ],
)
async def test_document_shortcut_never_bypasses_other_source_report_or_database_routing(context):
    agent, classifier, _, _ = make_agent()
    current = state().model_copy(update=context)
    await agent.run(current)
    classifier.assert_awaited_once()


@pytest.mark.asyncio
async def test_report_followup_precedes_explicit_deep_mode_and_document_shortcut():
    report_ask = AsyncMock(return_value="Answer from the authorized report.")
    agent, classifier, shallow, deep = make_agent(report_ask_fn=report_ask)
    classifier.return_value = {"user_intent": IntentResult(intent="research", target="report", report_action="ask")}
    await agent.run(state(active_report_job_id="report", research_mode="deep"))
    report_ask.assert_awaited_once()
    shallow.assert_not_awaited()
    deep.assert_not_awaited()


@pytest.mark.asyncio
async def test_greetings_still_use_the_meta_router():
    agent, classifier, _, _ = make_agent()
    classifier.return_value = {
        "user_intent": IntentResult(intent="meta", target="meta"),
        "messages": [AIMessage(content="Hello.")],
    }
    await agent.run(state().model_copy(update={"messages": [HumanMessage(content="Hei!")]}))
    classifier.assert_awaited_once()


@pytest.mark.asyncio
async def test_other_profiles_keep_existing_classifier_routing():
    agent, classifier, shallow, deep = make_agent()
    agent.document_qa = None
    await agent.run(state())
    classifier.assert_awaited_once()
    shallow.assert_not_awaited()
    deep.assert_awaited_once()


@pytest.mark.asyncio
async def test_natural_language_thorough_request_keeps_semantic_depth_routing():
    agent, classifier, shallow, deep = make_agent()
    current = state().model_copy(update={"messages": [HumanMessage(content="Gjor en grundig analyse av planen.")]})
    await agent.run(current)
    classifier.assert_awaited_once()
    shallow.assert_not_awaited()
    deep.assert_awaited_once()


@pytest.mark.asyncio
async def test_unverified_budget_expiry_is_explicit_failure_without_automatic_deep_retry():
    agent, _, shallow, deep = make_agent()
    shallow.side_effect = DocumentBudgetExceeded()
    result = await agent.run(state())
    assert result["workflow_outcome"].status == "failed"
    assert "time or work limit" in result["messages"][-1].content
    deep.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_document_question_after_a_report_can_return_to_adaptive_answers():
    agent, classifier, shallow, deep = make_agent(checkpointer=MemorySaver())
    await agent.run(state(research_mode="deep"), thread_id="one-conversation")
    assert deep.await_count == 1
    await agent.run(
        state().model_copy(update={"messages": [HumanMessage(content="What is the document's target year?")]}),
        thread_id="one-conversation",
    )
    classifier.assert_awaited_once()
    shallow.assert_awaited_once()
    assert shallow.call_args.args[0].document_qa is not None
    assert deep.await_count == 1
