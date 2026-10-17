"""Score instruction-following responses with an LLM judge (book chapter 7, ollama_evaluate.py).

The judge sees the instruction, the reference answer and the model's response, and returns
a 0-100 score. It runs on a local Ollama server (https://ollama.com):
  ollama serve &  &&  ollama pull llama3
  python score_responses.py runs/sft_gpt2_355m/test_responses.json
  python score_responses.py runs/a/test_responses.json runs/b/test_responses.json   # compare runs
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests
from tqdm import tqdm

from llmtrain.instruct import format_input


def query(prompt: str, model: str, url: str) -> str:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
               "options": {"seed": 123, "temperature": 0, "num_ctx": 2048}}  # deterministic judge
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]


def score_file(path: Path, model: str, url: str) -> list[int | None]:
    entries = json.loads(path.read_text())
    scores = []
    for e in tqdm(entries, desc=path.parent.name):
        prompt = (f"Given the input `{format_input(e)}` and correct output `{e['output']}`, "
                  f"score the model response `{e['model_response']}` on a scale from 0 to 100, "
                  f"where 100 is the best score. Respond with the integer number only.")
        reply = query(prompt, model, url)
        match = re.search(r"\d+", reply)
        score = min(100, int(match.group())) if match else None
        e["judge_score"] = score
        scores.append(score)
    path.with_name(path.stem + "_scored.json").write_text(json.dumps(entries, indent=2))
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="test_responses.json files written by finetune.py")
    p.add_argument("--judge", default="llama3", help="Ollama model name")
    p.add_argument("--url", default="http://localhost:11434/api/chat")
    args = p.parse_args()

    try:
        requests.get(args.url.rsplit("/api/", 1)[0], timeout=5)
    except requests.RequestException:
        raise SystemExit(f"No Ollama server at {args.url}. Start it with `ollama serve` and "
                         f"`ollama pull {args.judge}`.")

    summary = []
    for f in args.files:
        scores = score_file(Path(f), args.judge, args.url)
        valid = [s for s in scores if s is not None]
        avg = sum(valid) / len(valid) if valid else float("nan")
        summary.append((f, avg, len(valid), len(scores)))
    print(f"\njudge: {args.judge}")
    for f, avg, n, total in summary:
        print(f"{avg:6.2f}  ({n}/{total} scored)  {f}")


if __name__ == "__main__":
    main()
