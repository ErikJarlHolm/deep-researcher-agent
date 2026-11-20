# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from deep_researcher_agent.agents.document_qa.agent import DocumentBudgetExceeded
from deep_researcher_agent.agents.document_qa.agent import DocumentResearcher
from deep_researcher_agent.agents.document_qa.agent import DocumentResponseError
from deep_researcher_agent.agents.document_qa.models import DocumentQAConfig
from deep_researcher_agent.agents.document_qa.models import Passage

PASSAGES = [
    Passage("ignored", "The policy was adopted in 2022 and sets a target for 2045.", "Policy.pdf", 3),
    Passage("ignored", "The target is met by reducing energy use and reusing materials.", "Measures.pdf", 6),
]


def draft(*claims, queries=(), deep=False):
    return json.dumps(
        {
            "claims": [{"text": text, "passage_ids": ids} for text, ids in claims],
            "search_queries": list(queries),
            "needs_deep_research": deep,
        }
    )


def verdict(ids, complete=False):
    return json.dumps({"supported_claim_ids": ids, "answer_complete": complete})


def researcher(responses, *, search=None, **options):
    generate = AsyncMock(side_effect=responses)
    search = search or AsyncMock(return_value=PASSAGES)
    agent = DocumentResearcher(
        search=search,
        generate=generate,
        config=DocumentQAConfig(**options),
        generation_prompt="generate",
        verification_prompt="verify",
    )
    return agent, generate, search


@pytest.mark.asyncio
async def test_short_answer_uses_one_search_and_two_calls_with_code_owned_citations():
    agent, generate, search = researcher([draft(("The target year is 2045. [99]", ["p1"])), verdict([0], True)])
    result = await agent.run("What is the target year?")
    assert result.status == "complete"
    assert result.effort == "quick"
    assert result.searches == 1
    assert result.model_calls == 2
    assert result.text == "The target year is 2045. [1]\n\n### Sources\n[1] Policy.pdf, p.3"
    sent_to_verifier = json.loads(generate.call_args_list[1].args[1])
    assert sent_to_verifier["claims"][0]["text"] == "The target year is 2045."
    assert "Measures.pdf" not in result.text
    search.assert_awaited_once()


@pytest.mark.asyncio
async def test_rejecting_optional_detail_does_not_discard_supported_answer():
    agent, _, _ = researcher(
        [
            draft(("The target year is 2045.", ["p1"]), ("It also bans every fuel.", ["p1"])),
            verdict([0], True),
        ]
    )
    result = await agent.run("What is the target year?")
    assert result.status == "complete"
    assert "2045" in result.text
    assert "bans" not in result.text
    assert result.model_calls == 2


@pytest.mark.asyncio
async def test_valid_negative_verdict_can_repair_wrong_date_without_workflow_failure():
    agent, generate, search = researcher(
        [
            draft(("The target year is 2022.", ["p1"])),
            verdict([]),
            draft(("The target year is 2045.", ["p1"])),
            verdict([0], True),
        ]
    )
    result = await agent.run("What is the target year?")
    assert result.status == "complete"
    assert result.effort == "synthesis"
    assert "2045" in result.text
    assert "2022" not in result.text
    assert search.await_count == 1
    revision = json.loads(generate.call_args_list[2].args[1])
    assert revision["previous_rejected_claims"][0]["text"] == "The target year is 2022."


@pytest.mark.asyncio
async def test_compact_verdict_accepts_paraphrase_and_does_not_require_irrelevant_second_source():
    agent, _, _ = researcher(
        [draft(("Reusing materials contributes to achieving the target.", ["p2"])), verdict([0], True)]
    )
    result = await agent.run("How does material reuse contribute?")
    assert result.status == "complete"
    assert "Measures.pdf, p.6" in result.text
    assert "Policy.pdf" not in result.text


@pytest.mark.asyncio
async def test_missing_part_triggers_targeted_search_and_multisource_synthesis():
    search = AsyncMock(side_effect=[PASSAGES[:1], PASSAGES[1:]])
    agent, _, _ = researcher(
        [
            draft(("The target year is 2045.", ["p1"]), queries=["Which measures implement the target?"]),
            verdict([0]),
            draft(("The target year is 2045.", ["p1"]), ("Measures include reusing materials.", ["p2"])),
            verdict([0, 1], True),
        ],
        search=search,
    )
    result = await agent.run("What is the target year and which measures implement it?")
    assert result.status == "complete"
    assert result.effort == "synthesis"
    assert result.searches == 2
    assert "Policy.pdf, p.3" in result.text
    assert "Measures.pdf, p.6" in result.text
    assert result.escalate_to_deep is False


@pytest.mark.asyncio
async def test_quick_mode_never_launches_additional_search_or_deep_work():
    agent, _, search = researcher(
        [
            draft(("The target year is 2045.", ["p1"]), queries=["More detail"], deep=True),
            verdict([0]),
        ]
    )
    result = await agent.run("Compare the target and its implementation", mode="quick")
    assert result.status == "partial"
    assert "Partial answer" in result.text
    assert result.escalate_to_deep is False
    search.assert_awaited_once()


@pytest.mark.asyncio
async def test_deep_escalation_requires_verified_multisource_evidence_and_unresolved_investigation():
    claims = (("The target is 2045.", ["p1"]), ("The measure is material reuse.", ["p2"]))
    agent, _, _ = researcher(
        [
            draft(*claims, queries=["Unresolved implementation contradictions"], deep=True),
            verdict([0, 1]),
            draft(*claims, deep=True),
            verdict([0, 1]),
        ]
    )
    result = await agent.run("Investigate contradictions in implementation")
    assert result.status == "partial"
    assert result.escalate_to_deep is True


@pytest.mark.asyncio
async def test_absence_does_not_escalate_or_claim_whole_collection_was_searched():
    agent, _, _ = researcher(
        [draft(queries=["Any evidence for the requested identifier?"], deep=True), draft(deep=True)]
    )
    result = await agent.run("What is the unpublished identifier?")
    assert result.status == "not_found"
    assert result.escalate_to_deep is False
    assert "entire collection" in result.text
    assert "### Sources" not in result.text


@pytest.mark.asyncio
async def test_missing_optional_escalation_flag_does_not_fail_an_unanswered_question():
    response = json.dumps({"claims": [], "search_queries": []})
    agent, _, _ = researcher([response, response])
    result = await agent.run("What is the unpublished identifier?")
    assert result.status == "not_found"
    assert result.escalate_to_deep is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "contact"),
    [
        ("What is the author's private phone number?", "+1 (202) 555-0123"),
        ("Hva er forfatterens private e-postadresse?", "recipient@example.test"),
    ],
)
async def test_private_contact_request_does_not_repeat_recipient_contact(question, contact):
    claims = (
        ("The document identifies an automated author.", ["p1"]),
        (f"The recipient's contact detail is {contact}.", ["p1"]),
    )
    agent, generate, _ = researcher(
        [
            draft(*claims),
            verdict([0], True),
            draft(*claims),
            verdict([0], True),
        ]
    )
    result = await agent.run(question)
    assert result.status == "partial"
    assert contact not in result.text
    for call in (generate.call_args_list[1], generate.call_args_list[3]):
        assert contact not in call.args[1]


@pytest.mark.asyncio
async def test_private_contact_only_does_not_return_unverified_detail():
    agent, _, _ = researcher([draft(("The recipient's number is +1 202 555 0123.", ["p1"])), draft()])
    result = await agent.run("What is the author's private phone number?")
    assert result.status == "not_found"
    assert "555" not in result.text


@pytest.mark.asyncio
async def test_public_business_contact_request_is_not_blocked_by_private_contact_filter():
    agent, _, _ = researcher([draft(("The public office telephone is +1 202 555 0123.", ["p1"])), verdict([0], True)])
    result = await agent.run("What is the public office telephone number?")
    assert result.status == "complete"
    assert "+1 202 555 0123" in result.text


@pytest.mark.asyncio
async def test_empty_successful_retrieval_needs_no_inference():
    agent, generate, _ = researcher([], search=AsyncMock(return_value=[]))
    result = await agent.run("Find an identifier")
    assert result.status == "not_found"
    generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_truncated_json_gets_one_compact_protocol_repair():
    agent, generate, _ = researcher(
        [draft(("The target is 2045.", ["p1"])), '{"supported_claim_ids":[', verdict([0], True)]
    )
    result = await agent.run("What is the target?")
    assert result.status == "complete"
    assert result.model_calls == 3
    assert "format_correction" in json.loads(generate.call_args_list[2].args[1])


@pytest.mark.asyncio
async def test_overlong_followup_list_is_repaired_before_abstaining():
    agent, generate, search = researcher(
        [draft(queries=["one", "two", "three"]), draft(queries=["one", "two"]), draft()]
    )
    result = await agent.run("What detail have the documents not provided?")
    assert result.status == "not_found"
    assert result.model_calls == 3
    assert [call.args[0] for call in search.await_args_list][1:] == ["one", "two"]
    repair = json.loads(generate.await_args_list[1].args[1])["format_correction"]
    assert "at most 2 entries" in repair


@pytest.mark.asyncio
async def test_persistently_invalid_json_is_an_operational_error_not_an_abstention():
    agent, generate, _ = researcher(["not-json", "still-not-json"])
    with pytest.raises(DocumentResponseError, match="after repair"):
        await agent.run("What is the target?")
    assert generate.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [[99], [-1], [0, 0]])
async def test_invalid_claim_ids_cannot_establish_support(ids):
    agent, _, _ = researcher([draft(("The target is 2045.", ["p1"])), verdict(ids, True)])
    with pytest.raises(DocumentResponseError, match="claim IDs"):
        await agent.run("What is the target?")


@pytest.mark.asyncio
async def test_invented_passage_id_is_not_forwarded_to_verifier():
    agent, generate, _ = researcher([draft(("An invented answer.", ["not-supplied"]))])
    with pytest.raises(DocumentResponseError, match="missing evidence"):
        await agent.run("What is the target?")
    assert generate.await_count == 1


@pytest.mark.asyncio
async def test_search_failure_is_not_reported_as_missing_evidence():
    agent, _, _ = researcher([], search=AsyncMock(side_effect=RuntimeError("backend unavailable")))
    with pytest.raises(RuntimeError, match="backend unavailable"):
        await agent.run("What is the target?")


@pytest.mark.asyncio
async def test_provider_timeout_before_our_deadline_is_not_masked_as_our_budget():
    agent, _, _ = researcher([TimeoutError("provider timeout")])
    with pytest.raises(TimeoutError, match="provider timeout"):
        await agent.run("What is the target?")


@pytest.mark.asyncio
async def test_expired_budget_cancels_inference_and_keeps_only_already_verified_partial_answer():
    cancelled = asyncio.Event()
    responses = iter(
        [
            draft(("The target is 2045.", ["p1"]), queries=["Implementation evidence"]),
            verdict([0]),
        ]
    )

    async def generate(*args):
        response = next(responses, None)
        if response is not None:
            return response
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    agent, _, _ = researcher([], quick_seconds=0.05, synthesis_seconds=0.08)
    agent.generate = generate
    result = await agent.run("What is the target and how will it be achieved?")
    assert result.status == "partial"
    assert result.budget_exhausted is True
    assert "2045" in result.text
    assert "budget was exhausted" in result.text
    assert cancelled.is_set()
    assert result.elapsed_seconds < 0.5


@pytest.mark.asyncio
async def test_unverified_draft_cannot_be_returned_on_timeout():
    cancelled = asyncio.Event()

    async def slow_generate(*args):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    agent, _, _ = researcher([], quick_seconds=0.02, synthesis_seconds=0.04)
    agent.generate = slow_generate
    with pytest.raises(DocumentBudgetExceeded):
        await agent.run("What is the target?", mode="quick")
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_timed_out_initial_retrieval_is_retried_not_misreported_as_empty_collection():
    async def search(query):
        if search_mock.await_count == 1:
            await asyncio.sleep(10)
        return PASSAGES

    search_mock = AsyncMock(side_effect=search)
    agent, _, _ = researcher(
        [draft(("The target is 2045.", ["p1"])), verdict([0], True)],
        search=search_mock,
        quick_seconds=0.02,
        synthesis_seconds=0.2,
    )
    result = await agent.run("What is the target?")
    assert result.status == "complete"
    assert result.searches == 2


@pytest.mark.asyncio
async def test_external_cancellation_propagates_without_fallback_or_orphaned_inference():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def generate(*args):
        started.set()
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    agent, _, _ = researcher([])
    agent.generate = generate
    task = asyncio.create_task(agent.run("What is the target?"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_call_budget_is_enforced_without_launching_an_unverifiable_draft():
    agent, generate, _ = researcher(
        [draft(("The target is 2045.", ["p1"]), queries=["Implementation"]), verdict([0])],
        max_model_calls=2,
    )
    result = await agent.run("What is the target and the implementation?")
    assert result.status == "partial"
    assert result.budget_exhausted is True
    assert generate.await_count == 2
