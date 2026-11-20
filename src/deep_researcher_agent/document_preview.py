# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private, NAT-free preview of the adaptive document workflow."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from hmac import compare_digest
from pathlib import Path
from typing import Annotated
from typing import Literal

from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer
from knowledge_layer.llamaindex import LlamaIndexRetriever
from openai import AsyncOpenAI
from openai import OpenAIError
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import StringConstraints

from deep_researcher_agent.agents.document_qa.agent import DocumentResearcher
from deep_researcher_agent.agents.document_qa.agent import DocumentResponseError
from deep_researcher_agent.agents.document_qa.models import DocumentAnswer
from deep_researcher_agent.agents.document_qa.models import DocumentQAConfig
from deep_researcher_agent.agents.document_qa.models import Passage
from deep_researcher_agent.evidence import split_evidence_content

logger = logging.getLogger(__name__)
PROMPTS = Path(__file__).parent / "agents" / "document_qa" / "prompts"
COLLECTION = "document-preview"
MODEL = "Qwen/Qwen3.5-35B-A3B"


def retrieval_config() -> dict[str, str]:
    return {
        "persist_dir": os.environ.get("DOCUMENT_PREVIEW_CHROMA_DIR", "/data/chroma"),
        "embed_model": "nvidia/Nemotron-3.5-Embed-1B-BF16",
        "embed_base_url": os.environ.get("DOCUMENT_PREVIEW_EMBED_URL", "http://embedder:8000/v1"),
    }


class DocumentQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8192)]
    mode: Literal["quick", "auto"] = "auto"


def create_app() -> FastAPI:
    token = os.environ.get("DOCUMENT_PREVIEW_TOKEN")
    bearer = HTTPBearer(auto_error=False)
    researcher: DocumentResearcher

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        nonlocal researcher
        retriever = LlamaIndexRetriever(retrieval_config())
        async with AsyncOpenAI(
            base_url=os.environ.get("DOCUMENT_PREVIEW_MODEL_URL", "http://qwen:8000/v1"),
            api_key=os.environ.get("DOCUMENT_PREVIEW_MODEL_KEY", "local"),
            max_retries=3,
        ) as client:

            async def search(query: str) -> list[Passage]:
                result = await retriever.retrieve(query, COLLECTION, top_k=10)
                if not result.success:
                    raise DocumentResponseError("Document retrieval failed")
                return [
                    Passage(chunk.chunk_id, text, chunk.file_name, chunk.page_number)
                    for chunk in result.chunks
                    for text in split_evidence_content(chunk.content[:5000])
                ]

            async def generate(system: str, payload: str, tokens: int) -> str:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": payload}],
                    temperature=0.0,
                    max_tokens=tokens,
                    response_format={"type": "json_object"},
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                content = response.choices[0].message.content if response.choices else None
                if not isinstance(content, str):
                    raise DocumentResponseError("Document inference returned non-text content")
                return content

            researcher = DocumentResearcher(
                search=search,
                generate=generate,
                config=DocumentQAConfig(),
                generation_prompt=(PROMPTS / "generate.j2").read_text(encoding="utf-8"),
                verification_prompt=(PROMPTS / "verify.j2").read_text(encoding="utf-8"),
            )
            yield

    def authorize(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]) -> None:
        if not token:
            raise HTTPException(503, "Document preview access is not configured")
        if credentials is None or not compare_digest(credentials.credentials.encode(), token.encode()):
            raise HTTPException(401, "Invalid or missing preview token", headers={"WWW-Authenticate": "Bearer"})

    app = FastAPI(title="Deepak document preview", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/documents/answer", dependencies=[Depends(authorize)])
    async def answer(request: DocumentQuestion) -> DocumentAnswer:
        try:
            return await researcher.run(request.question, mode=request.mode)
        except (DocumentResponseError, OpenAIError) as error:
            logger.warning("Document preview request failed (%s)", type(error).__name__)
            raise HTTPException(502, "Document model or retrieval failed") from error

    return app
