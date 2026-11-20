# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[2] / "scripts" / "summarize_document_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_summary", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def report():
    return {
        "identity": {"label": "test"},
        "finished_at": "2026-01-01",
        "cases": [
            {
                "id": "test-1",
                "status": "answered",
                "answer": "I could not find evidence.",
                "answer_location": "documents",
                "latency_seconds": 2,
            },
            {"id": "test-2", "status": "error", "failure_category": "timeout"},
        ],
    }


def test_execution_completion_does_not_count_as_correctness():
    result = summary.summarize(report())
    assert result["execution"] == {"completed": 1, "timeout": 1}
    assert result["quality"] == {"not_reviewed": 1}


def test_review_is_bound_to_the_actual_answer():
    execution = report()
    reviews = summary.review_template(execution)
    reviews["reviewer"] = "manual"
    reviews["cases"][0].update(outcome="unjustified_abstention", notes="The expected answer is present in the sources.")
    assert summary.summarize(execution, reviews)["quality"] == {"unjustified_abstention": 1}
    execution["cases"][0]["answer"] = "Different answer."
    with pytest.raises(ValueError, match="does not match"):
        summary.summarize(execution, reviews)


def test_known_answer_cannot_be_reviewed_as_appropriate_abstention():
    execution = report()
    reviews = summary.review_template(execution)
    reviews["reviewer"] = "manual"
    reviews["cases"][0].update(outcome="appropriate_abstention", notes="Wrong category.")
    with pytest.raises(ValueError, match="Expected an answer"):
        summary.summarize(execution, reviews)


def test_comparison_rejects_changed_time_budget_and_surfaces_case_regressions():
    baseline = report()
    baseline["identity"].update(
        protocol_version=2,
        suite_sha256="suite",
        documents_sha256={"source.pdf": "pdf"},
        timeout_seconds=300,
        source_policy="documents_only",
    )
    reviews = summary.review_template(baseline)
    reviews["reviewer"] = "manual"
    reviews["cases"][0].update(outcome="correct_answer", source_support="supported", notes="Gold points supported.")
    candidate = deepcopy(baseline)
    new_reviews = deepcopy(reviews)
    new_reviews["cases"][0].update(outcome="unjustified_abstention", source_support="missing", notes="Answer withheld.")

    comparison = summary.compare_reports(baseline, candidate, reviews, new_reviews)

    assert len(comparison["regressions_requiring_explanation"]) == 2
    assert comparison["cases_awaiting_review"] == []
    new_reviews["adjudication_sha256"] = "different-policy"
    with pytest.raises(ValueError, match="adjudication"):
        summary.compare_reports(baseline, candidate, reviews, new_reviews)
    new_reviews.pop("adjudication_sha256")
    candidate["identity"]["timeout_seconds"] = 900
    with pytest.raises(ValueError, match="timeout_seconds"):
        summary.compare_reports(baseline, candidate, reviews, new_reviews)


def test_cli_binds_reviews_to_policy_and_rejects_changed_policy(tmp_path, monkeypatch, capsys):
    execution_path = tmp_path / "report.json"
    policy_path = tmp_path / "adjudication.json"
    template_path = tmp_path / "review.json"
    execution_path.write_text(json.dumps(report()), encoding="utf-8")
    policy_path.write_text('{"version":"1.1"}', encoding="utf-8")
    base_args = [
        "summarize_document_benchmark.py",
        "--report",
        str(execution_path),
        "--adjudication",
        str(policy_path),
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        base_args + ["--review-template", str(template_path), "--output", str(tmp_path / "execution-summary.json")],
    )
    summary.main()
    reviews = json.loads(template_path.read_text(encoding="utf-8"))
    assert reviews["adjudication_sha256"] == summary.hashlib.sha256(policy_path.read_bytes()).hexdigest()
    reviews["reviewer"] = "explicit review"
    reviews["cases"][0].update(outcome="unjustified_abstention", source_support="not_applicable", notes="Withheld.")
    template_path.write_text(json.dumps(reviews), encoding="utf-8")
    reviewed_path = tmp_path / "reviewed-summary.json"
    monkeypatch.setattr(sys, "argv", base_args + ["--reviews", str(template_path), "--output", str(reviewed_path)])
    summary.main()
    assert (
        json.loads(reviewed_path.read_text(encoding="utf-8"))["adjudication_sha256"] == reviews["adjudication_sha256"]
    )
    reviewed_path.unlink()
    policy_path.write_text('{"version":"changed"}', encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        summary.main()
    assert not reviewed_path.exists()
    assert "not bound" in capsys.readouterr().err
