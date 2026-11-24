# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from deep_researcher_agent.agents.deep_researcher import report_review
from deep_researcher_agent.agents.deep_researcher.report_review import ReportReviewError


@pytest.fixture
def notes():
    return {
        "/shared/research_note_01.json": {
            "content": json.dumps(
                {
                    "query_topic": "coverage",
                    "target_components": ["coverage"],
                    "summary": "Evidence summary",
                    "findings": [],
                    "gaps": [],
                    "sources": [],
                    "narrative_notes": "Evidence details",
                    "language": "en",
                }
            )
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verdict", "rejected"),
    [
        ({"approved": True, "missing_topics": [], "unsupported_claims": []}, False),
        ({"approved": False, "missing_topics": [], "unsupported_claims": []}, True),
        ({"approved": True, "missing_topics": ["A"], "unsupported_claims": []}, True),
        ({"approved": True, "missing_topics": [], "unsupported_claims": ["B"]}, True),
    ],
)
async def test_review_only_accepts_a_consistent_approval(monkeypatch, notes, verdict, rejected):
    invoke = AsyncMock(return_value=AIMessage(content=json.dumps(verdict)))
    monkeypatch.setattr(report_review, "ainvoke_with_relay", invoke)
    model = MagicMock()
    if rejected:
        with pytest.raises(ReportReviewError, match="report_review_rejected"):
            await report_review.review_report(model=model, question="Original", report="Report", files=notes)
    else:
        await report_review.review_report(model=model, question="Original", report="Report", files=notes)

    payload = json.loads(invoke.await_args.args[1][1].content)
    assert payload["question"] == "Original"
    assert len(payload["research_notes"]) == 1
    assert invoke.await_args.kwargs["config"]["tags"] == [
        report_review.SUPPRESS_OUTPUT_ARTIFACT_TAG,
        report_review.SUPPRESS_UNREVIEWED_CONTENT_TAG,
    ]


@pytest.mark.asyncio
async def test_review_does_not_accept_missing_notes_or_invalid_model_output(monkeypatch, notes):
    model = MagicMock()
    invoke = AsyncMock(return_value=AIMessage(content="not JSON"))
    monkeypatch.setattr(report_review, "ainvoke_with_relay", invoke)

    with pytest.raises(ReportReviewError, match="report_review_missing_notes"):
        await report_review.review_report(model=model, question="Question", report="Report", files={})
    with pytest.raises(ReportReviewError, match="report_review_missing_notes"):
        await report_review.review_report(model=model, question="Question", report="Report", files=None)
    invoke.assert_not_awaited()
    with pytest.raises(ReportReviewError, match="report_review_invalid_response"):
        await report_review.review_report(model=model, question="Question", report="Report", files=notes)


@pytest.mark.asyncio
async def test_review_rejects_invalid_note_and_oversized_input(monkeypatch, notes):
    invoke = AsyncMock()
    monkeypatch.setattr(report_review, "ainvoke_with_relay", invoke)
    with pytest.raises(ReportReviewError, match="report_review_invalid_note"):
        await report_review.review_report(
            model=MagicMock(),
            question="Question",
            report="Report",
            files={"/shared/research_note_1.json": {"content": "{}"}},
        )
    with pytest.raises(ReportReviewError, match="report_review_input_too_large"):
        await report_review.review_report(model=MagicMock(), question="Question", report="x" * 100_001, files=notes)
    invoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_review_propagates_provider_timeout(monkeypatch, notes):
    monkeypatch.setattr(report_review, "ainvoke_with_relay", AsyncMock(side_effect=TimeoutError))
    with pytest.raises(TimeoutError):
        await report_review.review_report(model=MagicMock(), question="Question", report="Report", files=notes)


@pytest.mark.asyncio
async def test_review_accepts_legacy_line_list_notes(monkeypatch, notes):
    verdict = {"approved": True, "missing_topics": [], "unsupported_claims": []}
    invoke = AsyncMock(return_value=AIMessage(content=json.dumps(verdict)))
    monkeypatch.setattr(report_review, "ainvoke_with_relay", invoke)
    entry = notes["/shared/research_note_01.json"]
    entry["content"] = entry["content"].splitlines()

    await report_review.review_report(model=MagicMock(), question="Question", report="Report", files=notes)

    payload = json.loads(invoke.await_args.args[1][1].content)
    assert payload["research_notes"][0]["query_topic"] == "coverage"
