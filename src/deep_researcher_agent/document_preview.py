# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private, NAT-free preview of the adaptive document workflow."""

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from hmac import compare_digest
from pathlib import Path
from typing import Annotated
from typing import Literal

from anyio import CancelScope
from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer
from knowledge_layer.llamaindex import LlamaIndexRetriever
from openai import AsyncOpenAI
from openai import OpenAIError
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import StringConstraints
from starlette.types import Receive
from starlette.types import Send

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
HEARTBEAT_SECONDS = 2.0


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


class DocumentStream(StreamingResponse):
    def __init__(self, events: AsyncGenerator[str, None]) -> None:
        super().__init__(
            events,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )
        self._events = events

    async def stream_response(self, send: Send) -> None:
        # A failed ASGI send can leave the iterator suspended at a yield.
        try:
            await super().stream_response(send)
        finally:
            with CancelScope(shield=True):
                await self._events.aclose()


async def wait_for_disconnect(receive: Receive) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass


def create_app() -> FastAPI:
    token = os.environ.get("DOCUMENT_PREVIEW_TOKEN")
    bearer = HTTPBearer(auto_error=False)
    researcher: DocumentResearcher
    active = asyncio.Lock()
    config = DocumentQAConfig()

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
                config=config,
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

    @asynccontextmanager
    async def admit() -> AsyncIterator[None]:
        if active.locked():
            raise HTTPException(429, "Document preview is busy", headers={"Retry-After": "2"})
        async with active:
            yield

    async def run(question: DocumentQuestion) -> DocumentAnswer:
        budget = config.quick_seconds if question.mode == "quick" else config.synthesis_seconds
        try:
            # The core handles expected budget exhaustion and may return verified partial claims.
            async with asyncio.timeout(budget + 2):
                return await researcher.run(question.question, mode=question.mode)
        except TimeoutError as error:
            logger.warning("Document preview exceeded the request deadline")
            raise HTTPException(504, "Document answer timed out") from error
        except (DocumentResponseError, OpenAIError) as error:
            logger.warning("Document preview request failed (%s)", type(error).__name__)
            raise HTTPException(502, "Document model or retrieval failed") from error

    @app.post("/v1/documents/answer", dependencies=[Depends(authorize)])
    async def answer(question: DocumentQuestion, request: Request) -> DocumentAnswer:
        async with admit():
            job = asyncio.create_task(run(question))
            disconnect = asyncio.create_task(wait_for_disconnect(request.receive))
            try:
                await asyncio.wait((job, disconnect), return_when=asyncio.FIRST_COMPLETED)
                if job.done():
                    return job.result()
                disconnect.result()
                raise HTTPException(499, "Client closed the request")
            finally:
                job.cancel()
                disconnect.cancel()
                await asyncio.gather(job, disconnect, return_exceptions=True)

    @app.post("/v1/documents/answer/stream", dependencies=[Depends(authorize)])
    async def stream(question: DocumentQuestion) -> StreamingResponse:
        def event(name: str, data: dict) -> str:
            return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        async def events() -> AsyncGenerator[str, None]:
            try:
                async with admit():
                    job = asyncio.create_task(run(question))
                    try:
                        yield event("accepted", {})
                        while not job.done():
                            done, _ = await asyncio.wait((job,), timeout=HEARTBEAT_SECONDS)
                            if not done:
                                yield event("working", {})
                        result = job.result()
                    finally:
                        job.cancel()
                        with CancelScope(shield=True):
                            await asyncio.gather(job, return_exceptions=True)
                yield event("answer", asdict(result))
            except HTTPException as error:
                details = {"status": error.status_code, "message": error.detail}
                if error.headers and "Retry-After" in error.headers:
                    details["retry_after_seconds"] = int(error.headers["Retry-After"])
                yield event("error", details)

        return DocumentStream(events())

    return app
