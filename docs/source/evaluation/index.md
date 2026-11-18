<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Evaluation

Measure and improve the quality of the Deep Researcher Agent blueprint.

```{note}
To create custom evaluators or benchmarks, refer to the [NeMo Agent Toolkit Evaluation documentation](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/evaluate.html). The benchmarks below are pre-built for Deep Researcher Agent.
```

- **[Benchmarks](./benchmarks/index.md)** — Run standardized evaluation suites

## Local document regression tests

`scripts/run_document_benchmark.py` runs a documents-only YAML suite through the
same frontend gateway and WebSocket workflow as the UI. It uploads the original
documents and asks every question verbatim automatically; it requires neither manual chat
entry nor a hosted model service.

```bash
uv run scripts/run_document_benchmark.py \
  --suite /path/to/test_rag_climate.yaml \
  --documents-dir /path/to/original-pdfs \
  --output /path/to/private-results.json \
  --label local-model-and-config
```

The default gateway is `http://127.0.0.1:3300`, usable with the SSH tunnel described
in the [Workbench deployment instructions](../deployment/docker-compose.md#ai-workbench-with-a-managed-secret).
The runner currently targets the anonymous local test deployment; it does not
disable or bypass authentication on an authenticated deployment.

The suite format preserves the existing `workbench-rag-ejh-nvidia` climate
benchmark: `defaults.source_policy: documents_only` and a `tests` list containing
`id`, `question`, `expected_answer`, `expected_key_points`, `answer_location`,
`required_sources` (original filenames), and `web_search: forbidden`. All
documents named anywhere in the suite are uploaded for every question, including
unanswerable questions, so a case's expected sources are not retrieval hints.

Each question gets a fresh collection and conversation, preventing earlier answers
from contaminating later tests. Successful cases delete only their own temporary
collection. Terminal model-service and deep-job failures are recorded as errors and the next
question runs; protocol, ingestion, connection, and interaction failures stop the
run. Failed cases preserve their collections for diagnosis; existing user
collections are not deleted. Deep-research escalations are polled until their
final report is available. The per-case timeout, including ingestion, defaults
to 900 seconds. A timed-out deep job must confirm task interruption before the
runner starts another question.

Results are saved after every answer or error. Use `--resume` with the same
arguments to continue with the remaining questions; recorded failures are kept,
not silently retried into successes. Suite and document hashes, the gateway,
policy, and operator-supplied model/configuration label must match. Do not reuse
a label after changing the deployed model or configuration. Start a separate
output file for a fresh run.

An `answered` result means the workflow produced an answer, **not** that the answer
is correct. Reports retain the original gold answers, actual answers, intermediate
events, latency, and missing filename references for subsequent quality review.
`quality_status: not_scored` is explicit: filename mentions alone do not prove
source support, correct synthesis, or proper abstention. This runner does not
reproduce the old model-judged 100-point score.

Keep the PDFs and result files local and out of Git: intermediate events can
contain retrieved document text. The original climate suite is available in
`ErikJarlHolm/workbench-rag-ejh-nvidia`, branch `erikjarlholm-rag-agent-evaluation`,
at `evaluation-tests/test_rag_climate.yaml`; its five PDFs were separate user
uploads, not versioned assets.
