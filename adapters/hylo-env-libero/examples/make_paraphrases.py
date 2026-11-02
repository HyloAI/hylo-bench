"""Write paraphrases.json for the `language` perturbation: three rewordings of every instruction of the four LIBERO
suites, by a chat model, each checked to keep the instruction's object and place words (so the task is unchanged).

    uv run python bench/adapters/hylo-env-libero/examples/make_paraphrases.py [--model ollama:qwen3.5:35b-a3b]

The table is data: review it before relying on it (instructions whose paraphrases fail the check keep fewer).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from hylo.reasoners.llm.chat import build_chat

HERE = Path(__file__).resolve().parents[1] / "src" / "hylo" / "envs" / "libero"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
STOP = {"the", "a", "an", "and", "it", "on", "in", "of", "to", "up", "put", "place", "pick", "both", "from", "its",
        "into", "inside", "top", "next", "between", "close", "open", "turn", "push", "front", "left", "right", "back"}
PROMPT = ("Rewrite this instruction for a robot in three different ways. Keep exactly the same task: the same "
          "objects (keep their names, colours and descriptive words), the same places and the same order of steps. "
          "Change only the wording and sentence structure. Reply with a JSON list of three strings.\n\n"
          "Instruction: {text}")


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ollama:qwen3.5:35b-a3b")
    ap.add_argument("--out", default=str(HERE / "paraphrases.json"))
    a = ap.parse_args()
    backend, _, name = a.model.partition(":")
    model = build_chat({"backend": backend, "model": name, "timeout_s": 600, "reasoning_effort": "none"})
    index = json.loads((HERE / "suites.json").read_text())
    out_path = Path(a.out)
    table: dict[str, list[str]] = json.loads(out_path.read_text()) if out_path.is_file() else {}
    for suite in SUITES:
        for row in index[suite]:
            text = row["language"]
            if len(table.get(text, [])) >= 3:
                continue
            keep: list[str] = []
            for _ in range(3):
                reply = model.chat([{"role": "user", "content": PROMPT.format(text=text)}], max_tokens=2000).text
                m = re.search(r"\[.*\]", reply, re.S)
                try:
                    cands = json.loads(m.group(0)) if m else []
                except json.JSONDecodeError:
                    cands = []
                need = content_words(text)
                for c in cands:
                    if isinstance(c, str) and c.strip().lower() != text.lower() and need <= content_words(c) \
                            and c not in keep:
                        keep.append(c.strip().rstrip("."))
                if len(keep) >= 3:
                    break
            table[text] = keep[:3]
            print(f"{len(keep[:3])}  {text}  ->  {keep[:3]}", flush=True)
            out_path.write_text(json.dumps(table, indent=1))
    print(f"wrote {out_path}: {len(table)} instructions")


if __name__ == "__main__":
    main()
