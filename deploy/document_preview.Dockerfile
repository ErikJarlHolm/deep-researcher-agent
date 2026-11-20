# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM python:3.13-slim

WORKDIR /app
COPY deploy/document_preview.requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY sources/knowledge_layer ./sources/knowledge_layer
RUN pip install --no-cache-dir --no-deps ./sources/knowledge_layer \
    && mkdir -p /data/chroma && chown -R 1000:1000 /data
COPY src/deep_researcher_agent ./src/deep_researcher_agent
COPY scripts/seed_document_preview.py ./seed_document_preview.py
ENV PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 DEEP_RESEARCHER_EMBED_API_KEY=local
USER 1000:1000
ENTRYPOINT ["python", "-m", "uvicorn", "deep_researcher_agent.document_preview:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
