# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.28,<1", "PyYAML>=6,<7", "websockets>=15,<17"]
# ///
"""Run an existing documents-only YAML benchmark against the frontend gateway."""

import argparse
import hashlib
import json
import re
import sys
import time
import uuid
from contextlib import ExitStack
from datetime import UTC
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import httpx
import yaml
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect

FAILURE_MESSAGES = (
    "We couldn't process your request due to a temporary error.",
    "Research could not be completed.",
    "I'm unable to reach the model service right now.",
    "The model service took too long to respond",
)


class WorkflowFailed(RuntimeError):
    """The workflow terminated with an explicit failure, not a usable answer."""


def load_suite(path: Path, documents_dir: Path) -> tuple[dict, list[Path]]:
    suite = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(suite, dict) or suite.get("defaults", {}).get("source_policy") != "documents_only":
        raise ValueError("The suite must declare defaults.source_policy: documents_only")
    cases = suite.get("tests")
    if not isinstance(cases, list) or not cases:
        raise ValueError("The suite must contain a nonempty tests list")
    seen = set()
    sources = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"].strip():
            raise ValueError("Each test needs a nonempty string id")
        if case["id"] in seen:
            raise ValueError(f"Duplicate test id: {case['id']}")
        seen.add(case["id"])
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"Missing question: {case['id']}")
        if case.get("web_search") != "forbidden":
            raise ValueError(f"Web search must be forbidden: {case['id']}")
        if case.get("source_policy", "documents_only") != "documents_only":
            raise ValueError(f"Only documents_only cases are supported: {case['id']}")
        required_sources = case.get("required_sources")
        if not isinstance(required_sources, list):
            raise ValueError(f"required_sources must be a list: {case['id']}")
        for source in required_sources:
            if not isinstance(source, str) or Path(source).name != source or "\\" in source or "/" in source:
                raise ValueError(f"Expected a document filename, not a path: {source!r}")
            sources.add(source)
    if not sources:
        raise ValueError("At least one test must identify the benchmark documents")
    documents = [documents_dir / name for name in sorted(sources)]
    for document in documents:
        if not document.is_file():
            raise FileNotFoundError(f"Missing benchmark document: {document}")
    return suite, documents


def remaining(deadline: float) -> float:
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("Benchmark case exceeded its time limit")
    return seconds


def response_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        value = content.get("text") or content.get("output")
        if value is None:
            return ""
        if isinstance(value, str):
            return value
    raise ValueError("Unsupported chat response content")


def parse_escalation(answer: str) -> str | None:
    if not answer.lstrip().startswith("{"):
        return None
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and payload.get("type") == "job_escalation":
        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or re.fullmatch(r"[A-Za-z0-9_-]+", job_id) is None:
            raise ValueError("Invalid escalated job identifier")
        return job_id
    return None


def read_answer(
    connection, message_id: str, deadline: float, events: list[dict] | None = None
) -> tuple[str, list[dict]]:
    parts = []
    events = events if events is not None else []
    while True:
        event = json.loads(connection.recv(timeout=remaining(deadline)))
        if not isinstance(event, dict):
            raise ValueError("Expected a JSON object from the WebSocket")
        events.append(event)
        if event.get("type") in ("error_message", "system_interaction_message") or event.get("status") == "error":
            raise RuntimeError(f"Workflow did not complete unattended: {json.dumps(event, ensure_ascii=False)}")
        if event.get("type") != "system_response_message":
            continue
        parent_id = event.get("parent_id")
        if parent_id and parent_id != message_id:
            raise ValueError("Received a response for a different request")
        parts.append(response_text(event.get("content")))
        if event.get("status") == "complete":
            answer = "".join(parts).strip()
            if not answer:
                raise RuntimeError("The workflow completed without an answer")
            if any(message in answer for message in FAILURE_MESSAGES):
                raise WorkflowFailed(f"Workflow returned a failure response: {answer}")
            return answer, events


class BenchmarkClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        parts = urlsplit(self.base_url)
        if (
            parts.scheme not in ("http", "https")
            or not parts.netloc
            or parts.path not in ("", "/")
            or parts.query
            or parts.fragment
            or parts.username is not None
        ):
            raise ValueError("Use the frontend gateway origin without a URL path")
        self.websocket_url = urlunsplit(
            ("wss" if parts.scheme == "https" else "ws", parts.netloc, "/websocket", "", "")
        )
        self.http = httpx.Client(base_url=self.base_url, timeout=30, trust_env=False)
        self.active_collection: str | None = None
        self.active_job_id: str | None = None
        self.active_events: list[dict] = []

    def request(self, method: str, path: str, *, deadline: float | None = None, **kwargs):
        if deadline is not None:
            kwargs["timeout"] = min(30, remaining(deadline))
        response = self.http.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    def wait_ingestion(self, job_id: str, deadline: float) -> None:
        while True:
            remaining(deadline)
            job = self.request("GET", f"/api/v1/documents/{job_id}/status", deadline=deadline)
            status = job["status"]
            if status == "completed":
                if any(detail["status"] != "success" for detail in job["file_details"]):
                    raise RuntimeError(f"Ingestion completed with unsuccessful files: {job_id}")
                return
            if status == "failed":
                raise RuntimeError(f"Document ingestion failed: {json.dumps(job, ensure_ascii=False)}")
            if status not in ("pending", "processing"):
                raise ValueError(f"Unexpected ingestion status: {status!r}")
            time.sleep(min(2, remaining(deadline)))

    def wait_report(self, job_id: str, deadline: float) -> str:
        path = f"/api/jobs/async/job/{job_id}"
        try:
            while True:
                remaining(deadline)
                job = self.request("GET", path, deadline=deadline)
                if job["status"] == "success":
                    report = self.request("GET", f"{path}/report", deadline=deadline)
                    if not report.get("has_report") or not report.get("report"):
                        raise WorkflowFailed(f"Job {job_id} succeeded without a report")
                    return report["report"]
                if job["status"] == "failure":
                    raise WorkflowFailed(f"Research job failed: {job.get('error')}")
                if job["status"] not in ("submitted", "running"):
                    raise RuntimeError(f"Research job did not succeed: {json.dumps(job, ensure_ascii=False)}")
                time.sleep(min(2, remaining(deadline)))
        except TimeoutError as exc:
            cancellation = self.request("POST", f"{path}/cancel")
            if cancellation.get("status") != "interrupted" or cancellation.get("task_cancelled") is not True:
                raise RuntimeError(
                    f"Research job {job_id} exceeded its deadline; task cancellation is unconfirmed"
                ) from exc
            raise WorkflowFailed("Research job exceeded its deadline and was interrupted") from exc

    def run_case(self, case: dict, documents: list[Path], timeout: float) -> dict:
        collection = f"eval_{uuid.uuid4().hex}"
        self.active_collection = collection
        self.active_job_id = None
        self.active_events = []
        deadline = time.monotonic() + timeout
        self.request(
            "POST",
            "/api/v1/collections",
            json={"name": collection, "description": "Isolated benchmark case"},
            deadline=deadline,
        )
        with ExitStack() as stack:
            files = [
                ("files", (path.name, stack.enter_context(path.open("rb")), "application/pdf")) for path in documents
            ]
            upload = self.request("POST", f"/api/v1/collections/{collection}/documents", files=files, deadline=deadline)
        self.wait_ingestion(upload["job_id"], deadline)
        started = time.monotonic()
        message_id = f"eval_{uuid.uuid4().hex}"
        query = json.dumps({"query": case["question"], "data_sources": ["knowledge_layer"]}, ensure_ascii=False)
        payload = {
            "type": "user_message",
            "schema_type": "chat_stream",
            "id": message_id,
            "conversation_id": collection,
            "content": {"messages": [{"role": "user", "content": [{"type": "text", "text": query}]}]},
        }
        with connect(
            self.websocket_url,
            additional_headers={"X-DeepResearcher-Mode": "headless"},
            open_timeout=min(10, remaining(deadline)),
            close_timeout=5,
            max_size=16 * 1024 * 1024,
            proxy=None,
        ) as connection:
            connection.send(json.dumps(payload))
            answer, events = read_answer(connection, message_id, deadline, self.active_events)
        job_id = parse_escalation(answer)
        self.active_job_id = job_id
        if job_id:
            answer = self.wait_report(job_id, deadline)
        elapsed = time.monotonic() - started
        required = case["required_sources"]
        missing_mentions = [name for name in required if name.casefold() not in answer.casefold()]
        result = {
            **case,
            "status": "answered",
            "answer": answer,
            "latency_seconds": round(elapsed, 2),
            "missing_source_mentions": missing_mentions,
            "quality_status": "not_scored",
            "collection": collection,
            "deep_research_job_id": job_id,
            "events": events,
        }
        self.request("DELETE", f"/api/v1/collections/{collection}")
        return result


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--documents-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True, help="Model/configuration label; must match when resuming")
    parser.add_argument("--base-url", default="http://127.0.0.1:3300")
    parser.add_argument("--timeout", type=float, default=900, help="Maximum seconds per case, including ingestion")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    suite, documents = load_suite(args.suite, args.documents_dir)
    identity = {
        "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        "documents_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in documents},
        "label": args.label,
        "base_url": args.base_url,
        "source_policy": "documents_only",
    }
    report = {"identity": identity, "suite": suite["name"], "started_at": datetime.now(UTC).isoformat(), "cases": []}
    if args.output.exists():
        if not args.resume:
            parser.error("Output already exists; choose a new path or use --resume")
        report = json.loads(args.output.read_text(encoding="utf-8"))
        if report["identity"] != identity:
            parser.error("Cannot resume with different documents, suite, model label, policy or gateway")
    completed = {case["id"] for case in report["cases"]}
    client = BenchmarkClient(args.base_url)
    try:
        sources = client.request("GET", "/api/v1/data_sources")
        if {source["id"] for source in sources} != {"knowledge_layer"}:
            raise ValueError("Expected only the internal-document source; use the documents-only profile")
        save_report(args.output, report)
        for case in suite["tests"]:
            if case["id"] in completed:
                continue
            print(f"Running {case['id']}...", flush=True)
            started = time.monotonic()
            try:
                result = client.run_case(case, documents, args.timeout)
            except (httpx.HTTPError, WebSocketException, TimeoutError, ValueError, RuntimeError) as exc:
                report["cases"].append(
                    {
                        **case,
                        "status": "error",
                        "error": str(exc),
                        "collection": client.active_collection,
                        "deep_research_job_id": client.active_job_id,
                        "events": client.active_events,
                        "elapsed_seconds": round(time.monotonic() - started, 2),
                    }
                )
                save_report(args.output, report)
                print(f"FAILED {case['id']}: {exc}", file=sys.stderr, flush=True)
                if isinstance(exc, WorkflowFailed):
                    continue
                print(
                    "Stopped; an interrupted eval_* collection may remain for diagnosis and TTL cleanup.",
                    file=sys.stderr,
                )
                return 1
            report["cases"].append(result)
            save_report(args.output, report)
            print(f"Answered {case['id']} in {result['latency_seconds']:.1f}s (quality not yet scored)", flush=True)
        report["finished_at"] = datetime.now(UTC).isoformat()
        save_report(args.output, report)
        return int(any(case["status"] == "error" for case in report["cases"]))
    finally:
        client.http.close()


if __name__ == "__main__":
    raise SystemExit(main())
