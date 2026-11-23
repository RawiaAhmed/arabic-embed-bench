"""Embed the Quran with each model and score retrieval on the question set."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quran_dataset.build import build, read_jsonl
from quran_dataset.chunk import STRATEGIES
from quran_dataset.normalize import normalize_arabic

from .metrics import first_relevant_rank, hit_at, mrr_at, top_k


@dataclass(frozen=True)
class Model:
    name: str
    hf_id: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 512


MODELS = [
    Model("minilm-l12", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", max_seq_length=128),
    Model("e5-base", "intfloat/multilingual-e5-base", "query: ", "passage: "),
    Model("bge-m3", "BAAI/bge-m3"),
]
LANGS = ["ar", "en"]
# Document indexes: Uthmani Arabic as published, the same text with diacritics stripped, and English.
DOC_LANGS = ["ar", "arplain", "en"]
K = 10


def load_questions(path: Path) -> list[dict]:
    return read_jsonl(path)


def load_verses(data_dir: Path) -> list[dict]:
    path = data_dir / "verses.jsonl"
    return read_jsonl(path) if path.exists() else build(data_dir)


class Embedder:
    def __init__(self, model: Model, cache_dir: Path):
        from sentence_transformers import SentenceTransformer

        self.model = model
        self.cache_dir = cache_dir / model.name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.st = SentenceTransformer(model.hf_id, device=_device())
        self.st.max_seq_length = model.max_seq_length

    def encode(self, texts: list[str], prefix: str) -> np.ndarray:
        return self.st.encode([prefix + t for t in texts], batch_size=32, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)

    def corpus(self, key: str, texts: list[str]) -> tuple[np.ndarray, float]:
        """Embed (or load cached) passages; return vectors and seconds spent embedding."""
        path = self.cache_dir / f"{key}.npy"
        if path.exists():
            return np.load(path), 0.0
        start = time.perf_counter()
        vecs = self.encode(texts, self.model.passage_prefix)
        seconds = time.perf_counter() - start
        np.save(path, vecs)
        return vecs, seconds


def doc_text(chunk: dict, doc_lang: str) -> str:
    if doc_lang == "arplain":
        return normalize_arabic(chunk["text_ar"])
    return chunk[f"text_{doc_lang}"]


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_model(model: Model, verses: list[dict], questions: list[dict], cache_dir: Path) -> dict:
    emb = Embedder(model, cache_dir)
    q_vecs = {lang: emb.encode([q["text"] for q in questions if q["lang"] == lang], model.query_prefix) for lang in LANGS}
    q_by_lang = {lang: [q for q in questions if q["lang"] == lang] for lang in LANGS}

    results: dict = {"model": model.name, "hf_id": model.hf_id, "strategies": {}}
    for strategy, chunker in STRATEGIES.items():
        chunks = chunker(verses)
        chunk_verses = [set(c["verse_ids"]) for c in chunks]
        row: dict = {"chunks": len(chunks)}
        for doc_lang in DOC_LANGS:
            d_vecs, secs = emb.corpus(f"{strategy}_{doc_lang}", [doc_text(c, doc_lang) for c in chunks])
            if strategy == "verse" and secs:
                row[f"embed_seconds_{doc_lang}"] = round(secs, 1)
            for q_lang in LANGS:
                ranked = top_k(q_vecs[q_lang], d_vecs, K)
                ranks = [first_relevant_rank(list(r), chunk_verses, set(q["gold"]))
                         for r, q in zip(ranked, q_by_lang[q_lang])]
                row[f"{q_lang}->{doc_lang}"] = {
                    "hit@1": round(hit_at(ranks, 1), 3),
                    "hit@5": round(hit_at(ranks, 5), 3),
                    "mrr@10": round(mrr_at(ranks, K), 3),
                    "misses@5": [q["id"] for q, r in zip(q_by_lang[q_lang], ranks) if r is None or r > 5],
                }
        results["strategies"][strategy] = row
    return results


def save(results: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
