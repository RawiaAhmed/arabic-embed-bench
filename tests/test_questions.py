"""The gold set must stay balanced and point at real verses."""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
QUESTIONS = [json.loads(line) for line in (ROOT / "questions.jsonl").open(encoding="utf-8")]

# Verse counts per surah, from the standard Hafs numbering (6,236 total).
SURAH_LENGTHS = [7, 286, 200, 176, 120, 165, 206, 75, 129, 109, 123, 111, 43, 52, 99, 128, 111, 110, 98, 135,
                 112, 78, 118, 64, 77, 227, 93, 88, 69, 60, 34, 30, 73, 54, 45, 83, 182, 88, 75, 85, 54, 53, 89,
                 59, 37, 35, 38, 29, 18, 45, 60, 49, 62, 55, 78, 96, 29, 22, 24, 13, 14, 11, 11, 18, 12, 12, 30,
                 52, 52, 44, 28, 28, 20, 56, 40, 31, 50, 40, 46, 42, 29, 19, 36, 25, 22, 17, 19, 26, 30, 20, 15,
                 21, 11, 8, 8, 19, 5, 8, 8, 11, 11, 8, 3, 9, 5, 4, 7, 3, 6, 3, 5, 4, 5, 6]


def test_surah_table_is_complete():
    assert len(SURAH_LENGTHS) == 114 and sum(SURAH_LENGTHS) == 6236


def test_fifty_questions_with_unique_ids():
    assert len(QUESTIONS) == 50
    assert len({q["id"] for q in QUESTIONS}) == 50


def test_every_topic_has_one_arabic_and_one_english_question_with_same_gold():
    by_topic: dict[str, list[dict]] = {}
    for q in QUESTIONS:
        by_topic.setdefault(q["topic"], []).append(q)
    for topic, qs in by_topic.items():
        assert sorted(q["lang"] for q in qs) == ["ar", "en"], topic
        assert qs[0]["gold"] == qs[1]["gold"], topic


def test_languages_balanced():
    assert Counter(q["lang"] for q in QUESTIONS) == {"ar": 25, "en": 25}


def test_gold_references_exist():
    for q in QUESTIONS:
        assert q["gold"], q["id"]
        for ref in q["gold"]:
            surah, ayah = map(int, ref.split(":"))
            assert 1 <= surah <= 114 and 1 <= ayah <= SURAH_LENGTHS[surah - 1], f"{q['id']}: {ref}"
