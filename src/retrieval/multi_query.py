"""Multi-query expansion and Reciprocal Rank Fusion for candidate retrieval.

Long Vietnamese patient narratives bundle several symptoms, diseases and questions
into one string. A single embedded query mixes all of them, so the neighbourhood
of the query vector is dominated by the most salient clause and BM25 dilutes term
statistics across clauses. Splitting the narrative into clauses and fusing the
per-clause result lists with RRF restores candidates that the full query missed.
"""

import re
from typing import Any

_CLAUSE_SPLIT = re.compile(r"[.!?;\n]+")
_COMMA_SPLIT = re.compile(r",+")


def _normalize(text: str) -> str:
    return " ".join(text.split())


def build_subqueries(
    query: str,
    max_subqueries: int = 4,
    min_chars: int = 12,
) -> list[str]:
    """Splits a narrative query into clause-level sub-queries.

    Sentence boundaries are preferred; a comma split is used only when the query
    is a single sentence, since short queries carry their medical entities in a
    comma-separated enumeration. The original full query is never returned.
    """
    base = _normalize(query)
    if not base or max_subqueries <= 0:
        return []

    clauses = [_normalize(c).strip(" .,!?;:") for c in _CLAUSE_SPLIT.split(base)]
    if sum(1 for c in clauses if c) < 2:
        clauses = [_normalize(c).strip(" .,!?;:") for c in _COMMA_SPLIT.split(base)]

    seen = {base.lower()}
    subqueries: list[str] = []
    for clause in clauses:
        key = clause.lower()
        if len(clause) < min_chars or key in seen:
            continue
        seen.add(key)
        subqueries.append(clause)
        if len(subqueries) >= max_subqueries:
            break
    return subqueries


def fuse_candidates(
    weighted_lists: list[tuple[float, list[dict[str, Any]]]],
    top_k: int,
    rrf_k: int = 60,
    id_key: str = "chunk_id",
) -> list[dict[str, Any]]:
    """Fuses ranked candidate lists with weighted Reciprocal Rank Fusion.

    Args:
        weighted_lists: (weight, candidates) pairs, each list already sorted by rank.
        top_k: Maximum number of fused candidates to return.
        rrf_k: RRF rank smoothing constant.
        id_key: Candidate field identifying a chunk across lists.

    Returns:
        Fused candidates sorted by descending RRF score, with `score` overwritten.
    """
    pool: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}

    for weight, candidates in weighted_lists:
        for rank, cand in enumerate(candidates, start=1):
            key = str(cand[id_key])
            if key not in pool:
                pool[key] = dict(cand)
                scores[key] = 0.0
            scores[key] += weight / (rrf_k + rank)

    for key, cand in pool.items():
        cand["score"] = scores[key]

    return sorted(pool.values(), key=lambda c: c["score"], reverse=True)[:top_k]
