# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "run_document_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("document_benchmark", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


class FakeSocket:
    def __init__(self, events):
        self.events = iter(events)
        self.sent = []

    def send(self, message):
        self.sent.append(json.loads(message))

    def recv(self, **kwargs):
        return json.dumps(next(self.events))


def test_stream_preserves_answer_before_empty_completion() -> None:
    socket = FakeSocket(
        [
            {"type": "system_intermediate_message", "parent_id": "internal-step", "content": "Searching"},
            {
                "type": "system_response_message",
                "parent_id": "request",
                "content": {"text": "The answer."},
                "status": "in_progress",
            },
            {"type": "system_response_message", "content": {"text": None}, "status": "complete"},
        ]
    )

    answer, events = benchmark.read_answer(socket, "request", float("inf"))

    assert answer == "The answer."
    assert len(events) == 3


@pytest.mark.parametrize(
    "event",
    [
        {"type": "error_message", "content": {"message": "failed"}},
        {"type": "system_interaction_message", "content": {"text": "Please clarify"}},
        {"type": "system_response_message", "content": {"text": None}, "status": "complete"},
        {
            "type": "system_response_message",
            "content": {"text": "We couldn't process your request due to a temporary error."},
            "status": "complete",
        },
    ],
)
def test_stream_failures_are_not_successful_answers(event) -> None:
    with pytest.raises(RuntimeError):
        benchmark.read_answer(FakeSocket([event]), "request", float("inf"))


def test_invalid_escalation_cannot_be_used_as_an_api_path() -> None:
    with pytest.raises(ValueError, match="job identifier"):
        benchmark.parse_escalation('{"type":"job_escalation","job_id":"../../collections"}')


def test_suite_requires_all_original_documents(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "example",
                "defaults": {"source_policy": "documents_only"},
                "tests": [
                    {
                        "id": "single",
                        "question": "Question?",
                        "required_sources": ["missing.pdf"],
                        "web_search": "forbidden",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing.pdf"):
        benchmark.load_suite(suite_path, tmp_path)


def test_cases_are_isolated_and_reference_answers_are_not_sent(monkeypatch, tmp_path: Path) -> None:
    documents = [tmp_path / "first.pdf", tmp_path / "second.pdf"]
    for path in documents:
        path.write_bytes(b"synthetic upload")
    requests = []
    sockets = []

    def request(method, path, **kwargs):
        requests.append((method, path, kwargs))
        return {"job_id": "upload"}

    def connect(*args, **kwargs):
        socket = FakeSocket(
            [{"type": "system_response_message", "content": {"text": "Answer from first.pdf"}, "status": "complete"}]
        )
        sockets.append(socket)
        context = MagicMock()
        context.__enter__.return_value = socket
        return context

    monkeypatch.setattr(benchmark, "connect", connect)
    client = benchmark.BenchmarkClient("http://127.0.0.1:3300")
    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(client, "wait_ingestion", lambda *args: None)
    case = {
        "id": "single",
        "question": "What is the budget?",
        "required_sources": ["first.pdf"],
        "expected_answer": "REFERENCE MUST NOT REACH AGENT",
    }
    try:
        results = [client.run_case(case, documents, 60) for _ in range(2)]
    finally:
        client.http.close()

    assert results[0]["collection"] != results[1]["collection"]
    assert all(result["quality_status"] == "not_scored" for result in results)
    uploads = [kwargs["files"] for method, path, kwargs in requests if path.endswith("/documents")]
    assert [[item[1][0] for item in upload] for upload in uploads] == [
        ["first.pdf", "second.pdf"],
        ["first.pdf", "second.pdf"],
    ]
    for socket in sockets:
        payload = socket.sent[0]
        query = json.loads(payload["content"]["messages"][0]["content"][0]["text"])
        assert query["data_sources"] == ["knowledge_layer"]
        assert query["query"] == case["question"]
        assert case["expected_answer"] not in query["query"]
    deleted = [path for method, path, _ in requests if method == "DELETE"]
    assert deleted == [f"/api/v1/collections/{result['collection']}" for result in results]


def test_deep_escalation_waits_for_the_actual_report(monkeypatch) -> None:
    client = benchmark.BenchmarkClient("http://127.0.0.1:3300")
    request = MagicMock(
        side_effect=[
            {"status": "running"},
            {"status": "success"},
            {"has_report": True, "report": "Final research, not a job acknowledgement."},
        ]
    )
    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(benchmark.time, "sleep", lambda _: None)
    try:
        assert client.wait_report("job_123", float("inf")) == "Final research, not a job acknowledgement."
    finally:
        client.http.close()
    assert request.call_args.args == ("GET", "/api/jobs/async/job/job_123/report")


def test_deep_timeout_cancels_the_job_and_is_not_an_answer(monkeypatch) -> None:
    client = benchmark.BenchmarkClient("http://127.0.0.1:3300")
    request = MagicMock(return_value={"status": "interrupted", "task_cancelled": True})
    monkeypatch.setattr(client, "request", request)
    try:
        with pytest.raises(benchmark.WorkflowFailed):
            client.wait_report("job_123", 0)
    finally:
        client.http.close()
    request.assert_called_once_with("POST", "/api/jobs/async/job/job_123/cancel")


def test_failed_deep_job_is_a_terminal_workflow_failure(monkeypatch) -> None:
    client = benchmark.BenchmarkClient("http://127.0.0.1:3300")
    request = MagicMock(return_value={"status": "failure", "error": "job failed (RuntimeError)"})
    monkeypatch.setattr(client, "request", request)
    try:
        with pytest.raises(benchmark.WorkflowFailed, match="job failed"):
            client.wait_report("job_123", float("inf"))
    finally:
        client.http.close()
    request.assert_called_once_with("GET", "/api/jobs/async/job/job_123", deadline=float("inf"))


def test_unconfirmed_job_cancellation_stops_instead_of_starting_more_work(monkeypatch) -> None:
    client = benchmark.BenchmarkClient("http://127.0.0.1:3300")
    monkeypatch.setattr(client, "request", MagicMock(return_value={"status": "interrupted", "task_cancelled": False}))
    try:
        with pytest.raises(RuntimeError, match="cancellation is unconfirmed") as failure:
            client.wait_report("job_123", 0)
    finally:
        client.http.close()
    assert not isinstance(failure.value, benchmark.WorkflowFailed)


def test_terminal_model_failure_is_recorded_and_later_questions_still_run(monkeypatch, tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text("original suite bytes", encoding="utf-8")
    document = tmp_path / "source.pdf"
    document.write_bytes(b"synthetic upload")
    cases = [{"id": "first", "question": "First?"}, {"id": "second", "question": "Second?"}]
    monkeypatch.setattr(benchmark, "load_suite", lambda *_: ({"name": "example", "tests": cases}, [document]))
    output = tmp_path / "results.json"
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        [
            "run_document_benchmark.py",
            "--suite",
            str(suite_path),
            "--documents-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--label",
            "synthetic",
        ],
    )
    client = MagicMock()
    client.request.return_value = [{"id": "knowledge_layer"}]
    client.active_collection = "eval_failed"
    client.active_job_id = None
    client.active_events = []
    client.run_case.side_effect = [
        benchmark.WorkflowFailed("Model timed out"),
        {**cases[1], "status": "answered", "answer": "Second answer", "latency_seconds": 1.0},
    ]
    monkeypatch.setattr(benchmark, "BenchmarkClient", lambda _: client)

    assert benchmark.main() == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert [case["status"] for case in report["cases"]] == ["error", "answered"]
    assert report["cases"][0]["error"] == "Model timed out"
    assert "finished_at" in report
    monkeypatch.setattr(benchmark.sys, "argv", [*benchmark.sys.argv, "--resume"])
    assert benchmark.main() == 1
    assert client.run_case.call_count == 2
    assert json.loads(output.read_text(encoding="utf-8"))["cases"] == report["cases"]
