import numpy as np

from arabic_embed_bench.metrics import first_relevant_rank, hit_at, mrr_at, top_k


def test_top_k_orders_by_similarity():
    docs = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]])
    query = np.array([[1.0, 0.0]])
    assert top_k(query, docs, 3).tolist() == [[0, 2, 1]]


def test_top_k_caps_at_corpus_size():
    assert top_k(np.eye(2), np.eye(2), 10).shape == (2, 2)


def test_first_relevant_rank_uses_any_overlap():
    chunk_verses = [{"1:1"}, {"2:1", "2:2", "2:3"}, {"2:255"}]
    assert first_relevant_rank([0, 1, 2], chunk_verses, {"2:2"}) == 2
    assert first_relevant_rank([0, 2], chunk_verses, {"2:2"}) is None


def test_hit_and_mrr():
    ranks = [1, 3, None, 7]
    assert hit_at(ranks, 1) == 0.25
    assert hit_at(ranks, 5) == 0.5
    assert mrr_at(ranks, 5) == (1 + 1 / 3) / 4
