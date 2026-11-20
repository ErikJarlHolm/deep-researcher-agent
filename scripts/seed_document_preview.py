# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ingest a local fixture into the preview's dedicated collection."""

import argparse
import time
from pathlib import Path

from knowledge_layer.llamaindex import LlamaIndexIngestor

from deep_researcher_agent.document_preview import COLLECTION
from deep_researcher_agent.document_preview import retrieval_config
from deep_researcher_agent.knowledge.schema import FileStatus
from deep_researcher_agent.knowledge.schema import JobState


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path, nargs="+")
    args = parser.parse_args()
    if any(not fixture.is_file() or fixture.suffix.lower() not in {".txt", ".md", ".pdf"} for fixture in args.fixtures):
        parser.error("Provide existing .txt, .md, or .pdf test fixtures")
    ingestor = LlamaIndexIngestor(retrieval_config())
    job = ingestor.submit_job([str(fixture) for fixture in args.fixtures], COLLECTION)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        result = ingestor.get_job_status(job)
        if result.status == JobState.COMPLETED:
            if len(result.file_details) != len(args.fixtures) or any(
                file.status != FileStatus.SUCCESS or not file.chunks_created for file in result.file_details
            ):
                raise SystemExit("Preview ingestion did not index every fixture")
            return
        if result.status == JobState.FAILED:
            raise SystemExit("Preview document ingestion failed")
        time.sleep(0.25)
    raise SystemExit("Preview document ingestion timed out")


if __name__ == "__main__":
    main()
