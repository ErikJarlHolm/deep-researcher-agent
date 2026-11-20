# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-independent document QA. No NAT, LangGraph, or model-provider imports."""

import asyncio
import json
import logging
import re
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import asdict
from typing import Literal
from typing import TypeVar

from pydantic import BaseModel
from pydantic import ValidationError

from .models import Claim
from .models import DocumentAnswer
from .models import DocumentQAConfig
from .models import Draft
from .models import Passage
from .models import ResearchMode
from .models import Verdict

logger = logging.getLogger(__name__)
Search = Callable[[str], Awaitable[Sequence[Passage]]]
Generate = Callable[[str, str, int], Awaitable[str]]
ResponseT = TypeVar("ResponseT", bound=BaseModel)

_PRIVATE_CONTACT_REQUEST = re.compile(
    r"\b(?:private|personal|privat\w*|personlig\w*)\b.*"
    r"\b(?:phone|telephone|mobile|contact|email|e-mail|telefon\w*|mobil\w*|kontakt\w*|epost\w*|e-post\w*)\b"
    r"|"
    r"\b(?:phone|telephone|mobile|contact|email|e-mail|telefon\w*|mobil\w*|kontakt\w*|epost\w*|e-post\w*)\b.*"
    r"\b(?:private|personal|privat\w*|personlig\w*)\b",
    re.IGNORECASE | re.DOTALL,
)
_CONTACT_VALUE = re.compile(
    r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}(?!\w)"
    r"|(?<!\w)\+?\d(?:[\d\s()./-]*\d){6,14}(?!\w)",
    re.IGNORECASE,
)


class DocumentResponseError(RuntimeError):
    """Inference returned an invalid evidence contract after one bounded repair."""


class DocumentBudgetExceeded(TimeoutError):
    """The request ended before any answer could be verified."""


def _render_claims(claims: Sequence[Claim], passages: dict[str, Passage], notes: Sequence[str] = ()) -> str:
    references: dict[tuple[str, int | None], int] = {}
    lines = []
    for claim in claims:
        citations = []
        for passage_id in claim.passage_ids:
            passage = passages[passage_id]
            key = (passage.source, passage.page)
            number = references.setdefault(key, len(references) + 1)
            if number not in citations:
                citations.append(number)
        lines.append(f"{claim.text} {' '.join(f'[{number}]' for number in citations)}")
    sources = []
    for (source, page), number in references.items():
        label = re.sub(r"([\\`*_[\]<>])", r"\\\1", " ".join(source.split()))
        location = f", p.{page}" if page is not None and page > 0 else ""
        sources.append(f"[{number}] {label}{location}")
    return "\n\n".join([*lines, *notes]) + "\n\n### Sources\n" + "\n".join(sources)


class DocumentResearcher:
    """Search first; expand only after identifying unsupported or unanswered parts."""

    def __init__(
        self,
        *,
        search: Search,
        generate: Generate,
        config: DocumentQAConfig,
        generation_prompt: str,
        verification_prompt: str,
    ) -> None:
        self.search = search
        self.generate = generate
        self.config = config
        self.generation_prompt = generation_prompt
        self.verification_prompt = verification_prompt

    async def run(self, question: str, *, mode: ResearchMode = "auto", history: str = "") -> DocumentAnswer:
        if not question.strip():
            raise ValueError("A document question cannot be empty")
        loop = asyncio.get_running_loop()
        started = loop.time()
        effort: Literal["quick", "synthesis"] = "quick"
        deadline = started + self.config.quick_seconds
        searches = 0
        model_calls = 0
        passages: dict[str, Passage] = {}
        seen_passages: set[tuple[str, int | None, str]] = set()
        searched_queries: set[str] = set()
        recent_ids: list[str] = []
        best_claims: list[Claim] = []
        rejected_claims: list[Claim] = []
        last_draft: Draft | None = None
        budget_exhausted = False
        private_contact_request = bool(_PRIVATE_CONTACT_REQUEST.search(question))
        norwegian = bool(
            re.search(r"\b(?:hva|hvem|hvilke?|hvor|hvordan|hvorfor|omtrent|if\u00f8lge)\b", question, re.IGNORECASE)
        )

        def finish(*, complete: bool = False) -> DocumentAnswer:
            status: Literal["complete", "partial", "not_found"]
            if best_claims:
                notes = []
                status = "complete" if complete else "partial"
                if not complete:
                    notes.append(
                        (
                            "**Delvis svar:** Dokumentutdragene underbygger punktene ovenfor, "
                            "men ikke hele sp\u00f8rsm\u00e5let."
                        )
                        if norwegian
                        else (
                            "**Partial answer:** The passages support the points above, "
                            "but do not answer the whole question."
                        )
                    )
                if budget_exhausted:
                    notes.append(
                        "Tids- eller arbeidsbudsjettet er brukt opp; videre unders\u00f8kelse ble stoppet."
                        if norwegian
                        else "The time or work budget was exhausted; further investigation was stopped."
                    )
                text = _render_claims(best_claims, passages, notes)
            else:
                if budget_exhausted:
                    raise DocumentBudgetExceeded("Document QA budget expired before an answer could be verified")
                text = (
                    (
                        "Jeg fant ikke tilstrekkelig st\u00f8tte i de unders\u00f8kte dokumentutdragene "
                        "til \u00e5 besvare sp\u00f8rsm\u00e5let. Dette viser ikke at opplysningene mangler "
                        "i hele dokumentsamlingen."
                    )
                    if norwegian
                    else (
                        "I found insufficient support in the retrieved passages to answer the question. "
                        "This does not establish that the information is absent from the entire collection."
                    )
                )
                status = "not_found"
            return DocumentAnswer(
                text=text,
                status=status,
                effort=effort,
                searches=searches,
                model_calls=model_calls,
                elapsed_seconds=loop.time() - started,
                budget_exhausted=budget_exhausted,
                escalate_to_deep=bool(
                    mode == "auto"
                    and not complete
                    and not budget_exhausted
                    and last_draft
                    and last_draft.needs_deep_research
                    and searches >= 2
                    and best_claims
                ),
            )

        async def retrieve(query: str) -> None:
            nonlocal searches
            query = query.strip()[:500]
            if not query or query.casefold() in searched_queries:
                return
            if searches >= self.config.max_searches:
                if not searched_queries:
                    raise DocumentBudgetExceeded("Document QA search budget exhausted")
                return
            searches += 1
            scope = asyncio.timeout_at(deadline)
            try:
                async with scope:
                    found = await self.search(query)
            except TimeoutError as error:
                if scope.expired():
                    raise DocumentBudgetExceeded("Document QA retrieval deadline expired") from error
                raise
            searched_queries.add(query.casefold())
            added = []
            for passage in found:
                key = (passage.source, passage.page, passage.text)
                if not passage.text.strip() or key in seen_passages:
                    continue
                seen_passages.add(key)
                passage_id = f"p{len(passages) + 1}"
                passages[passage_id] = Passage(passage_id, passage.text, passage.source, passage.page)
                added.append(passage_id)
            recent_ids.extend(added)
            logger.info("Document QA retrieval search=%d new_passages=%d", searches, len(added))

        def evidence() -> list[dict]:
            # Keep verified support and give new searches room instead of filling
            # the entire next prompt with the initial nearest neighbours again.
            required = {pid for claim in best_claims for pid in claim.passage_ids}
            ordered = [p for p in passages.values() if p.id in required]
            ordered += [passages[pid] for pid in recent_ids if pid not in required]
            preferred = required | set(recent_ids)
            ordered += [p for p in passages.values() if p.id not in preferred]
            selected = []
            chars = 0
            for passage in ordered:
                if chars + len(passage.text) > self.config.max_evidence_chars:
                    continue
                selected.append(asdict(passage))
                chars += len(passage.text)
            return selected

        async def infer(schema: type[ResponseT], prompt: str, payload: dict, tokens: int) -> ResponseT:
            nonlocal model_calls
            for attempt in range(2):
                if model_calls >= self.config.max_model_calls:
                    raise DocumentBudgetExceeded("Document QA model-call budget exhausted")
                model_calls += 1
                scope = asyncio.timeout_at(deadline)
                try:
                    async with scope:
                        raw = await self.generate(prompt, json.dumps(payload, ensure_ascii=False), tokens)
                except TimeoutError as error:
                    if scope.expired():
                        raise DocumentBudgetExceeded("Document QA inference deadline expired") from error
                    raise
                try:
                    return schema.model_validate_json(raw)
                except ValidationError as error:
                    error_types = sorted({item["type"] for item in error.errors(include_input=False)})
                    logger.warning(
                        "Document QA invalid response schema=%s attempt=%d errors=%d types=%s",
                        schema.__name__,
                        attempt + 1,
                        error.error_count(),
                        ",".join(error_types),
                    )
                    if attempt:
                        raise DocumentResponseError(f"Invalid {schema.__name__} response after repair") from error
                    limits = (
                        "Draft limits: claims has at most 12 entries (6 in quick effort); each claim has "
                        "nonempty text of at most 800 characters and 1-4 passage IDs. search_queries "
                        "has at most 2 entries. If no claim is supported, use claims: []; never fill it "
                        "with speculation. If no useful new search remains, use search_queries: []."
                        if schema is Draft
                        else "Verdict limits: supported_claim_ids has at most 12 integer entries, "
                        "and answer_complete must be a boolean."
                    )
                    payload = {
                        **payload,
                        "format_correction": (
                            "The preceding response did not satisfy the required JSON schema. "
                            f"Return only the requested compact JSON object. {limits} "
                            "Do not repeat passages or explain decisions."
                        ),
                    }
            raise AssertionError("unreachable")

        for round_index in range(self.config.max_rounds):
            recent_ids.clear()
            if round_index:
                if mode == "quick":
                    break
                effort = "synthesis"
                deadline = started + self.config.synthesis_seconds
            try:
                if not searched_queries:
                    await retrieve(question)
                elif last_draft:
                    for query in last_draft.search_queries:
                        await retrieve(query)
                if not passages:
                    return finish()
                visible_evidence = evidence()
                visible_ids = {p["id"] for p in visible_evidence}
                if model_calls + 2 > self.config.max_model_calls:
                    budget_exhausted = True
                    break
                draft = await infer(
                    Draft,
                    self.generation_prompt,
                    {
                        "question": question,
                        "history_for_resolving_references_only": history,
                        "passages": visible_evidence,
                        "previous_verified_claims": [claim.model_dump() for claim in best_claims],
                        "previous_rejected_claims": [claim.model_dump() for claim in rejected_claims],
                        "revision_requested": round_index > 0,
                        "effort": effort,
                    },
                    self.config.generation_tokens,
                )
                draft.claims = [
                    claim.model_copy(update={"text": re.sub(r"\[\d+\]", "", claim.text).strip()})
                    for claim in draft.claims
                ]
                last_draft = draft
                if any(not claim.text.strip() or not set(claim.passage_ids) <= visible_ids for claim in draft.claims):
                    raise DocumentResponseError("Draft refers to missing evidence or contains an empty claim")
                privacy_rejected = [
                    claim for claim in draft.claims if private_contact_request and _CONTACT_VALUE.search(claim.text)
                ]
                if privacy_rejected:
                    logger.info("Document QA suppressed %d private-contact claims", len(privacy_rejected))
                    draft.claims = [claim for claim in draft.claims if claim not in privacy_rejected]
                    rejected_claims = privacy_rejected
                if not draft.claims:
                    continue
                referenced_ids = {pid for claim in draft.claims for pid in claim.passage_ids}
                verdict = await infer(
                    Verdict,
                    self.verification_prompt,
                    {
                        "question": question,
                        "claims": [{"id": i, **claim.model_dump()} for i, claim in enumerate(draft.claims)],
                        "passages": [p for p in visible_evidence if p["id"] in referenced_ids],
                    },
                    self.config.verification_tokens,
                )
                supported_ids = verdict.supported_claim_ids
                if (
                    len(set(supported_ids)) != len(supported_ids)
                    or any(i < 0 or i >= len(draft.claims) for i in supported_ids)
                    or (verdict.answer_complete and not supported_ids)
                ):
                    raise DocumentResponseError("Verifier returned invalid claim IDs or empty complete answer")
                supported = [claim for i, claim in enumerate(draft.claims) if i in supported_ids]
                rejected_claims = privacy_rejected + [
                    claim for i, claim in enumerate(draft.claims) if i not in supported_ids
                ]
                logger.info(
                    "Document QA verification effort=%s supported=%d rejected=%d complete=%s",
                    effort,
                    len(supported),
                    len(draft.claims) - len(supported),
                    verdict.answer_complete,
                )
                if supported:
                    best_claims = supported
                if verdict.answer_complete and not privacy_rejected:
                    return finish(complete=True)
            except DocumentBudgetExceeded:
                if mode != "quick" and effort == "quick" and round_index + 1 < self.config.max_rounds:
                    logger.info("Document QA quick budget exhausted; extending within synthesis budget")
                    continue
                budget_exhausted = True
                break
        return finish()
