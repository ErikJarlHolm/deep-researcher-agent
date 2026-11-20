# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ingest a local fixture into the preview's dedicated collection."""

import argparse
import time
from pathlib import Path

from knowledge_layer.llamaindex import LlamaIndexIngestor

from deep_researcher_agent.document_preview import COLLECTION
from deep_researcher_agent.document_preview import retrieval_config
from deep_researcher_agent.knowledge.schema import JobState


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    if not args.fixture.is_file() or args.fixture.suffix.lower() not in {".txt", ".md"}:
        parser.error("Provide an existing .txt or .md test fixture")
    ingestor = LlamaIndexIngestor(retrieval_config())
    job = ingestor.submit_job([str(args.fixture)], COLLECTION)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        result = ingestor.get_job_status(job)
        if result.status == JobState.COMPLETED:
            if not result.metadata.get("total_chunks"):
                raise SystemExit("Preview ingestion produced no chunks")
            return
        if result.status == JobState.FAILED:
            raise SystemExit("Preview document ingestion failed")
        time.sleep(0.25)
    raise SystemExit("Preview document ingestion timed out")


if __name__ == "__main__":
    main()
