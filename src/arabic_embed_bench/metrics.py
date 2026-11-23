"""Retrieval metrics over chunks that cite one or more verses."""

from __future__ import annotations

import numpy as np


def top_k(query_vecs: np.ndarray, doc_vecs: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k most similar docs per query (vectors must be L2-normalized)."""
    scores = query_vecs @ doc_vecs.T
    k = min(k, doc_vecs.shape[0])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(scores, part, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(part, order, axis=1)


def first_relevant_rank(ranked: list[int], chunk_verses: list[set[str]], gold: set[str]) -> int | None:
    """1-based rank of the first chunk that contains a gold verse, or None."""
    for rank, idx in enumerate(ranked, start=1):
        if chunk_verses[idx] & gold:
            return rank
    return None


def hit_at(ranks: list[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks)


def mrr_at(ranks: list[int | None], k: int) -> float:
    return sum(1 / r for r in ranks if r is not None and r <= k) / len(ranks)
