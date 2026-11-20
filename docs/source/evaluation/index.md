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
`quality_status: needs_review` is explicit: filename mentions alone do not prove
source support, correct synthesis, or proper abstention. This runner does not
reproduce the old model-judged 100-point score.

### Frozen migration baseline

The frozen v1 suite and manifest live in
`frontends/benchmarks/document_rag/v1/`. They preserve the exact suite used in
the final pre-migration iterations: 10 single-source questions, five synthesis
questions, and five intentionally absent questions. Prior corrections to the
affected-countries synthesis expectation are already included; no gold answers
were changed when freezing v1. The manifest binds the exact YAML bytes, ordered
case IDs, and all five original PDF hashes. Do not edit v1 or replace its PDFs;
edits to questions/gold or new domains require v2 and an explicit change log.
Non-destructive review errata are separately versioned and hash-bound in
`adjudication.json`; they never rewrite the suite or raw results. PDFs and raw
traces remain private and outside Git.

Adjudication 1.1 documents one verified source-expectation defect:
`multi-source-01-methane` can be fully supported by the Norway climate-measures
PDF alone (physical pp.3-4). The other nominally required PDF does not discuss
methane. Omission of that irrelevant source is not a semantic-source error;
preserve the raw structural missing-source flag and explain the exception in
the review. Do not reward unnecessary citations. The expected answer and strict
`>75%` qualification remain unchanged. Consequently the v1 IDs describe five
"multi-source" cases, but only four actually require multiple documents. Apply
the same adjudication to both sides of any migration comparison.

Run the runner **on the GB10 execution host**, not on the laptop through a
tunnel. For example, from a persistent directory in the agent container, use
the internal frontend origin `http://deep-researcher-blueprint-ui:3000`. Start
with `docker exec -d` and redirect stdout/stderr to files inside the persistent
directory, so closing SSH does not terminate the benchmark. Preserve an exit-code
file and verify the PID, growing log, and incremental report after launch.

Protocol 2 fixes a 300-second per-case wall-clock budget (including ingestion)
for the final bounded migration baseline. It is an operational budget, not a
correctness threshold. The on-prem research model reserves 4096 output tokens
instead of 8192 to leave more room in the 32768-token model window. This is not
a general token-budget guarantee: oversized prompts can still fail, and failures
must remain visible. The shallow answer limit and prompts are unchanged.

```bash
python3 scripts/run_document_benchmark.py \
  --suite frontends/benchmarks/document_rag/v1/suite.yaml \
  --manifest frontends/benchmarks/document_rag/v1/manifest.json \
  --documents-dir /private/original-pdfs \
  --output /private/nat-baseline-v1.json \
  --label nat-qwen35-passage-id-output4096 \
  --base-url http://deep-researcher-blueprint-ui:3000 \
  --timeout 300 \
  --runtime-file configs/config_web_onprem_documents.yml \
  --runtime-file src/deep_researcher_agent/agents/shallow_researcher/agent.py \
  --runtime-file src/deep_researcher_agent/agents/shallow_researcher/prompts/researcher.j2 \
  --runtime-file scripts/run_document_benchmark.py
```

Protocol version, time budget, and runtime-file hashes are recorded in the
identity and checked on resume along with the corpus and suite. Record deployed
container image IDs and model-service configuration beside the report as well.
The benchmark does not assert that a file supplied via `--runtime-file` is the
one used by a remote service; the operator must verify deployment first.

`execution_status` and `failure_category` distinguish completed responses,
timeouts, workflow failures, transport errors, and protocol/runner errors.
`latency_seconds` measures answer generation after ingestion;
`end_to_end_seconds` includes ingestion. Completed refusals are not execution
failures and are not automatically correct. A cleanup error is explicit but
does not discard an otherwise recorded answer.

`citation_audit` resolves inline `[N]` references to the reference section and
known corpus filenames. A unique filename without its date prefix is accepted;
ambiguous aliases, missing definitions, and duplicate definitions are flagged.
This only measures *cited* sources, never whether a passage supports a claim.
Retrieved sources are not automatically used sources. Semantic review must
check each required source's contribution to the adjacent claim.

Generate and fill a separate review file:

```bash
python3 scripts/summarize_document_benchmark.py \
  --report /private/nat-baseline-v1.json \
  --adjudication frontends/benchmarks/document_rag/v1/adjudication.json \
  --review-template /private/nat-review-v1.json \
  --output /private/nat-execution-v1.json
# Fill reviewer, notes, outcome and source_support in nat-review-v1.json.
python3 scripts/summarize_document_benchmark.py \
  --report /private/nat-baseline-v1.json \
  --adjudication frontends/benchmarks/document_rag/v1/adjudication.json \
  --reviews /private/nat-review-v1.json \
  --output /private/nat-reviewed-v1.json
# After running and reviewing the replacement with the same protocol:
python3 scripts/summarize_document_benchmark.py \
  --report /private/replacement-v1.json \
  --adjudication frontends/benchmarks/document_rag/v1/adjudication.json \
  --reviews /private/replacement-review-v1.json \
  --baseline-report /private/nat-baseline-v1.json \
  --baseline-reviews /private/nat-review-v1.json \
  --output /private/migration-comparison-v1.json
```

Review outcomes are `correct_answer`, `partial_answer`, `incorrect_answer`,
`unjustified_abstention`, or `appropriate_abstention`; unfinished review remains
`not_reviewed`. Source support is reviewed independently (`supported`, `missing`,
or `not_applicable`). Each assessment is bound to the exact answer's SHA-256.
`supported` means an identifiable corpus source named in the answer actually
supports its factual claims; a correctly named document/page can provide that
support even when inline numbering is malformed. Preserve the separate citation
defect in that case. Retrieval alone does not establish source attribution.
For refusals without factual claims use `not_applicable`, not a fabricated
source-support success.
These are explicit human/agent review judgments, not an independent evaluator
model's verdict. Check numeric qualifications strictly (for example, "more
than" is not "at least") while accepting equivalent units. Do not infer a
historical 0-100 score. The comparison rejects suite/corpus/protocol/time-budget
or adjudication-policy changes, lists missing reviews and per-case regressions,
and does not treat an improved average as justification for a newly incorrect answer. Model, prompt,
and retrieval parity still require operator verification.

Freeze the reviewed results even when quality is disappointing. Migration can
start once all 20 cases have a terminal outcome, operational errors are explicit,
and completed answers are reviewed. Before replacing NAT in production, compare
the same corpus, model, retrieval settings, prompts, protocol and time budget:
report correctness, abstention, semantic source support, citation defects, errors,
and latency separately. Require no new unsupported factual answers to the five
absent questions and no unexplained regression on previously correct answers.
Repeat only unstable or changed cases three times and show every result.
The climate corpus is a regression set, not proof of performance across topics;
add another domain in a separate suite rather than tuning v1 until it passes.

#### Recorded NAT reference, 2026-11-19

The host-local protocol-2 run finished all 20 cases in 32.46 minutes. Its label
is `nat-qwen35-passage-id-output4096-v1`; raw-result SHA-256 is
`56f3e72908d973d983aa253ec0d085ea4ec9c4719d7408a3366e74777103cbca`.
The original report, runtime source snapshot, model/container provenance, logs,
and hash-bound reviews are preserved privately on the execution host and in
session artifacts, not in this repository. Adjudication 1.1 was applied.

| Group | Quality outcomes | Execution failures |
| :-- | :-- | :-- |
| 10 single-source cases | 8 correct factual answers, 2 unjustified refusals | 0 |
| 5 nominal multi-source cases | 1 unjustified refusal, no factual answers | 3 deadlines, 1 workflow failure |
| 5 absent cases | 3 source-bounded refusals, 1 safe privacy refusal | 1 workflow failure |

All 15 completed responses were explicitly reviewed by Copilot against the
frozen gold and the five inspected PDF texts; this is not independent evaluation.
The privacy refusal did not establish source absence and is not evidence that
retrieval worked. Five of the eight factually correct responses had citation
defects; seven identified genuinely supporting corpus sources, while one gave
the correct content without attributing the corpus document. Refusal-only
responses have no factual source-support verdict.

The process exited with code 1 because failures are not suppressed; all cases
nevertheless reached a terminal outcome. The three timed-out deep jobs were
confirmed interrupted. No model context-length errors were found in the
captured logs restricted to this execution; this does not establish a general
context-budget guarantee or diagnose the remaining workflow failures.

This is a completed **migration reference**, not an acceptable production
quality level. Stop NAT prompt tuning and compare the replacement with this
reference under the same protocol and adjudication.

Keep the PDFs and result files local and out of Git: intermediate events can
contain retrieved document text. The original climate suite is available in
`ErikJarlHolm/workbench-rag-ejh-nvidia`, branch `erikjarlholm-rag-agent-evaluation`,
at `evaluation-tests/test_rag_climate.yaml`; its five PDFs were separate user
uploads, not versioned assets.
