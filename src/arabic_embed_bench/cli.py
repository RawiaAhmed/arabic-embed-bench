"""`arabic-embed-bench run` embeds and scores; `arabic-embed-bench table` prints results as markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DIRECTIONS = ["ar->ar", "ar->arplain", "en->en", "ar->en", "en->ar", "en->arplain"]


def markdown(results: list[dict], metric: str = "hit@5") -> str:
    lines = [f"| model | chunks | {' | '.join(DIRECTIONS)} |", "|---" * (len(DIRECTIONS) + 2) + "|"]
    for res in results:
        for strategy, row in res["strategies"].items():
            cells = " | ".join(f"{row[d][metric]:.2f}" for d in DIRECTIONS)
            lines.append(f"| {res['model']} | {strategy} | {cells} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="arabic-embed-bench")
    parser.add_argument("command", choices=["run", "table"])
    parser.add_argument("--models", nargs="*", help="subset of model names")
    parser.add_argument("--metric", default="hit@5", choices=["hit@1", "hit@5", "mrr@10"])
    parser.add_argument("--results", type=Path, default=Path("results/results.json"))
    args = parser.parse_args()

    if args.command == "run":
        from .bench import MODELS, load_questions, load_verses, run_model, save

        verses = load_verses(Path("data"))
        questions = load_questions(Path("questions.jsonl"))
        chosen = [m for m in MODELS if not args.models or m.name in args.models]
        results = []
        for model in chosen:
            print(f"running {model.name} ...", flush=True)
            results.append(run_model(model, verses, questions, Path("cache")))
            save(results, args.results)
    else:
        results = json.loads(args.results.read_text(encoding="utf-8"))
    print(markdown(results, args.metric))


if __name__ == "__main__":
    main()
