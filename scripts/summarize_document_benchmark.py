# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize execution separately from hash-bound, explicit human quality assessments."""

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

OUTCOMES = {
    "correct_answer",
    "partial_answer",
    "incorrect_answer",
    "unjustified_abstention",
    "appropriate_abstention",
    "not_reviewed",
}
SOURCE_SUPPORT = {"supported", "missing", "not_applicable", "not_reviewed"}


def answer_hash(case: dict) -> str:
    return hashlib.sha256(case["answer"].encode("utf-8")).hexdigest()


def review_template(report: dict) -> dict:
    return {
        "reviewer": "",
        "cases": [
            {
                "id": case["id"],
                "answer_sha256": answer_hash(case),
                "outcome": "not_reviewed",
                "source_support": "not_reviewed",
                "notes": "",
            }
            for case in report["cases"]
            if case["status"] == "answered"
        ],
    }


def summarize(report: dict, reviews: dict | None = None) -> dict:
    """Never infer semantic correctness from strings or citation presence."""
    assessments = {}
    cases = {case["id"]: case for case in report["cases"]}
    if len(cases) != len(report["cases"]):
        raise ValueError("Duplicate case IDs in execution report")
    if reviews:
        if not isinstance(reviews.get("reviewer"), str) or not reviews["reviewer"].strip():
            raise ValueError("A quality review must identify its reviewer")
        for assessment in reviews["cases"]:
            case_id = assessment["id"]
            if case_id in assessments or case_id not in cases:
                raise ValueError(f"Unknown or duplicate reviewed case: {case_id}")
            case = cases[case_id]
            if case["status"] != "answered" or assessment["answer_sha256"] != answer_hash(case):
                raise ValueError(f"Review does not match the recorded answer: {case_id}")
            if assessment["outcome"] not in OUTCOMES or assessment["source_support"] not in SOURCE_SUPPORT:
                raise ValueError(f"Invalid quality classification: {case_id}")
            if assessment["outcome"] == "appropriate_abstention" and case["answer_location"] != "absent":
                raise ValueError(f"Expected an answer, not an appropriate abstention: {case_id}")
            if assessment["outcome"] == "unjustified_abstention" and case["answer_location"] == "absent":
                raise ValueError(f"Expected abstention for this case: {case_id}")
            if assessment["outcome"] != "not_reviewed" and not assessment.get("notes", "").strip():
                raise ValueError(f"A reviewed outcome needs an evidence/reason note: {case_id}")
            assessments[case_id] = assessment

    execution: Counter = Counter()
    quality: Counter = Counter()
    source_support: Counter = Counter()
    citation_flags = []
    durations = []
    for case in cases.values():
        if case["status"] != "answered":
            execution[case.get("failure_category", "unclassified_error")] += 1
            continue
        execution["completed"] += 1
        assessment = assessments.get(case["id"], {})
        quality[assessment.get("outcome", "not_reviewed")] += 1
        source_support[assessment.get("source_support", "not_reviewed")] += 1
        durations.append(case["latency_seconds"])
        audit = case.get("citation_audit", {})
        if audit.get("unresolved_inline_reference_ids") or audit.get("required_documents_not_cited"):
            citation_flags.append(case["id"])
    return {
        "identity": report["identity"],
        "execution_finished": bool(report.get("finished_at")),
        "recorded_cases": len(cases),
        "execution": dict(execution),
        "quality": dict(quality),
        "semantic_source_support": dict(source_support),
        "citation_audit_flagged_cases": citation_flags,
        "mean_completed_answer_seconds": round(statistics.mean(durations), 2) if durations else None,
        "reviewer": reviews["reviewer"] if reviews else None,
        "adjudication_sha256": reviews.get("adjudication_sha256") if reviews else None,
        "note": "Citation flags are structural checks, not semantic verdicts. No historical 0-100 score is inferred.",
    }


def compare_reports(baseline: dict, candidate: dict, baseline_reviews: dict, candidate_reviews: dict) -> dict:
    """Compare like-for-like runs without hiding per-case regressions in an average."""
    for key in ("protocol_version", "suite_sha256", "documents_sha256", "timeout_seconds", "source_policy"):
        if key not in baseline["identity"] or baseline["identity"][key] != candidate["identity"].get(key):
            raise ValueError(f"Runs are not comparable: {key}")
    baseline_summary = summarize(baseline, baseline_reviews)
    candidate_summary = summarize(candidate, candidate_reviews)
    if baseline_summary["adjudication_sha256"] != candidate_summary["adjudication_sha256"]:
        raise ValueError("Runs are not comparable: adjudication policy differs")
    before = {case["id"]: case for case in baseline["cases"]}
    after = {case["id"]: case for case in candidate["cases"]}
    if set(before) != set(after) or not baseline.get("finished_at") or not candidate.get("finished_at"):
        raise ValueError("Both runs must finish and contain the same cases")
    old_reviews = {case["id"]: case for case in baseline_reviews["cases"]}
    new_reviews = {case["id"]: case for case in candidate_reviews["cases"]}
    regressions = []
    pending = []
    for case_id, old_case in before.items():
        new_case = after[case_id]
        old = old_reviews.get(case_id, {})
        new = new_reviews.get(case_id, {})
        for case, assessment in ((old_case, old), (new_case, new)):
            if case["status"] == "answered" and (
                assessment.get("outcome", "not_reviewed") == "not_reviewed"
                or assessment.get("source_support", "not_reviewed") == "not_reviewed"
            ):
                pending.append(case_id)
        if old_case["status"] == "answered" and new_case["status"] != "answered":
            regressions.append({"id": case_id, "reason": "new execution error"})
        for field, good_values in (
            ("outcome", {"correct_answer", "appropriate_abstention"}),
            ("source_support", {"supported"}),
        ):
            if old.get(field) in good_values and new.get(field) != old[field]:
                regressions.append({"id": case_id, "reason": f"{field}: {old[field]} -> {new.get(field)}"})
    return {
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "regressions_requiring_explanation": regressions,
        "cases_awaiting_review": sorted(set(pending)),
        "note": (
            "Model, retrieval and prompt parity must also be verified by the operator. This is not a merge approval."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--review-template", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--baseline-reviews", type=Path)
    parser.add_argument("--adjudication", type=Path, help="Bind an explicit versioned errata policy to the review")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.review_template and args.review_template.exists()):
        parser.error("Choose new output paths; summaries and reviews must not overwrite archived evidence")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8")) if args.reviews else None
    template = review_template(report)
    if args.adjudication:
        policy_hash = hashlib.sha256(args.adjudication.read_bytes()).hexdigest()
        template["adjudication_sha256"] = policy_hash
        if reviews and reviews.get("adjudication_sha256") != policy_hash:
            parser.error("Quality review is not bound to the supplied adjudication policy")
    if args.baseline_report:
        if not args.baseline_reviews or not reviews:
            parser.error("Comparison requires --reviews and --baseline-reviews")
        result = compare_reports(
            json.loads(args.baseline_report.read_text(encoding="utf-8")),
            report,
            json.loads(args.baseline_reviews.read_text(encoding="utf-8")),
            reviews,
        )
    else:
        if args.baseline_reviews:
            parser.error("--baseline-reviews requires --baseline-report")
        result = summarize(report, reviews)
    for path, content in (
        (args.output, result),
        (args.review_template, template),
    ):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
