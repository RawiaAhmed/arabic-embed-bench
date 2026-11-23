# arabic-embed-bench

[![CI](https://github.com/RawiaAhmed/arabic-embed-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/RawiaAhmed/arabic-embed-bench/actions/workflows/ci.yml)

Which open embedding model finds the right Quran verse for a question asked in Arabic or English? Three multilingual models, three chunking strategies, 50 hand-written questions with verified gold verses, and one result I did not expect: **as published, the Arabic text is almost unsearchable.**

Built on [quran-dataset](https://github.com/RawiaAhmed/quran-dataset) (Day 1).

## The headline finding

Searching the Arabic Quran with an Arabic question, every model scored close to zero. The same question against the English translation worked. The cause is the script, not the language: the Uthmani text is full of diacritics and Quranic marks (ٱ, small high signs, dagger alef) that modern Arabic, and these models' training data, never use.

Stripping them (the `text_ar_plain` field from Day 1) fixes most of it:

| model | Arabic question → Uthmani text | → same text, diacritics stripped | embedding time, 6,236 verses (Uthmani → stripped) |
|---|---|---|---|
| minilm-l12 | 0.00 | **0.32** | 48.6 s → 13.5 s |
| e5-base | 0.08 | **0.48** | 274.6 s → 33.6 s |
| bge-m3 | 0.12 | **0.52** | 342.5 s → 98.8 s |

Hit@5 at verse level, 25 Arabic questions. Embedding on an Apple M1 (MPS).

Stripping diacritics also makes embedding **3.5–8× faster**, because the marks split every word into several tokens.

## Full results

Hit@5: share of questions where a gold verse is in the top 5 results. `ar->en` means an Arabic question searched against the English index.

| model | chunks | ar→ar | ar→ar (stripped) | en→en | ar→en | en→ar | en→ar (stripped) |
|---|---|---|---|---|---|---|---|
| minilm-l12 | verse | 0.00 | 0.32 | 0.52 | 0.32 | 0.00 | 0.28 |
| minilm-l12 | window3 | 0.00 | 0.16 | 0.52 | 0.28 | 0.00 | 0.16 |
| minilm-l12 | ruku | 0.04 | 0.16 | 0.32 | 0.20 | 0.04 | 0.16 |
| e5-base | verse | 0.08 | 0.48 | 0.60 | 0.48 | 0.00 | 0.08 |
| e5-base | window3 | 0.04 | 0.52 | 0.52 | 0.44 | 0.04 | 0.12 |
| e5-base | ruku | 0.12 | 0.40 | 0.60 | 0.40 | 0.08 | 0.32 |
| **bge-m3** | verse | 0.12 | 0.52 | **0.68** | 0.56 | 0.04 | 0.40 |
| **bge-m3** | window3 | 0.20 | 0.52 | **0.68** | 0.60 | 0.20 | 0.40 |
| **bge-m3** | ruku | 0.20 | 0.56 | **0.68** | **0.64** | 0.16 | 0.52 |

Hit@1 and MRR@10 for every cell, plus the IDs of every missed question, are in [`results/results.json`](results/results.json).

## What the numbers say

- **bge-m3 wins every column.** It is also the largest (568M parameters) and the slowest.
- **The best way to answer an Arabic question is to search the English translation** (ar→en 0.56–0.64), not the Arabic text (0.52–0.56 even after stripping). Cross-lingual retrieval into English beats same-language retrieval into Classical Arabic.
- **Vocabulary is the remaining gap.** Arabic questions that still miss use modern words the Quran does not: "الفائدة على القروض" (interest on loans) where the text says "الربا", "القلق" (anxiety) where it says "تطمئن القلوب" (hearts find rest). An embedding cannot always bridge 1,400 years of vocabulary; keyword search with synonyms might (Day 3).
- **Chunking matters less than the model.** For bge-m3, verse, window and ruku chunks are within 0.08 of each other. Ruku chunks look slightly better, but each one covers ~11 verses, so a "hit" is a much less precise citation. 3-verse windows are the sensible default.
- **MiniLM gets worse with bigger chunks** because it truncates input at 128 tokens and never sees most of a ruku.

**Read this with the sample size in mind.** Each column is 25 questions, so one question is 0.04. Differences under ~0.12 could be noise; the diacritics effect (0.00 → 0.52) is not.

## The question set

[`questions.jsonl`](questions.jsonl): 25 topics, each asked once in English and once in Arabic, with the same gold verses. Questions are paraphrased in everyday language ("Is charging interest on loans allowed?"), not copied from the verse, because real users do not quote the text they are looking for. Topics range from rulings (interest, alcohol, fasting) to stories (Joseph's dream, the ant and Solomon) to comfort ("How can I find peace of mind when I feel anxious?").

Every gold reference was checked against the verse text, and a test checks each one exists in the standard numbering.

## Run it

```bash
uv sync --extra models                 # sentence-transformers + torch (~4 GB of model downloads on first run)
uv run arabic-embed-bench run          # embeds all chunk sets per model, caches vectors in cache/, writes results/
uv run arabic-embed-bench table --metric mrr@10
uv run pytest                          # metric and question-set tests, no models needed
```

Models: [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), [`multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) (with its required `query:` / `passage:` prefixes), [`bge-m3`](https://huggingface.co/BAAI/bge-m3) (truncated to 512 tokens).

## What this decides for the next step

The `quran-rag` retriever will use **bge-m3 over 3-verse windows, indexing the English translation and the diacritic-stripped Arabic**, and add BM25 keyword search to close the vocabulary gap. Day 3 measures whether that hybrid beats these numbers.

## Part of a series

1. [quran-dataset](https://github.com/RawiaAhmed/quran-dataset): data and chunking
2. **arabic-embed-bench**: which embedding model retrieves Arabic and English best (this repo)
3. quran-hybrid-search: BM25 + vectors
