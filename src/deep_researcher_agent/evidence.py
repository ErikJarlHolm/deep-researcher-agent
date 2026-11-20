# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact source-passage splitting shared by both research runtimes."""


def split_evidence_content(content: str, *, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    if not 0 <= overlap < max_chars:
        raise ValueError("Evidence overlap must be nonnegative and smaller than the passage size")
    passages = []
    start = 0
    while start < len(content):
        end = min(start + max_chars, len(content))
        if end < len(content):
            boundary_start = start + max_chars // 2
            boundary = max(content.rfind("\n", boundary_start, end), content.rfind(" ", boundary_start, end))
            if boundary > start:
                end = boundary
        passage = content[start:end]
        if passage.strip():
            passages.append(passage)
        if end == len(content):
            break
        start = max(start + 1, end - overlap)
    return passages
