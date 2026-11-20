# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

ResearchMode = Literal["auto", "quick", "deep"]


class DocumentQAConfig(BaseModel):
    """Opt-in document route; budgets include retrieval, generation, and verification."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = "knowledge_layer"
    search_tool: str = "knowledge_search"
    quick_seconds: float = Field(default=45, gt=0, allow_inf_nan=False)
    synthesis_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    max_searches: int = Field(default=3, ge=1, le=6)
    max_model_calls: int = Field(default=6, ge=2, le=12)
    max_rounds: int = Field(default=2, ge=1, le=3)
    max_evidence_chars: int = Field(default=18000, ge=2000, le=24000)
    generation_tokens: int = Field(default=1536, ge=256, le=4096)
    verification_tokens: int = Field(default=512, ge=128, le=1024)

    @model_validator(mode="after")
    def _ordered_budgets(self):
        if self.synthesis_seconds < self.quick_seconds:
            raise ValueError("synthesis_seconds must be at least quick_seconds")
        return self


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    source: str
    page: int | None = None


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(min_length=1, max_length=800)
    passage_ids: list[str] = Field(min_length=1, max_length=4)


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    claims: list[Claim] = Field(max_length=12)
    search_queries: list[str] = Field(max_length=2)
    needs_deep_research: bool = False


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    supported_claim_ids: list[int] = Field(max_length=12)
    answer_complete: bool


@dataclass(frozen=True)
class DocumentAnswer:
    text: str
    status: Literal["complete", "partial", "not_found"]
    effort: Literal["quick", "synthesis"]
    searches: int
    model_calls: int
    elapsed_seconds: float
    budget_exhausted: bool = False
    escalate_to_deep: bool = False
