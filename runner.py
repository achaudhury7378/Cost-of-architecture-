"""
arch_cost/runner.py — Execute model x task x repeat over OpenRouter.

Lessons carried over from verdict:
  * shuffle execution order (spreads provider drift across models)
  * record query identity on every result (pairing survives to analysis)
  * one bad call never kills the experiment
Raw per-call records go to results.jsonl; analysis is a separate pass.
"""

import json
import os
import random
import time
from pathlib import Path

import requests

from pricing import cost_of
from tasks import SUITES, SYSTEM_QA, TOOL_SCHEMAS, execute_tool

from dotenv import load_dotenv

load_dotenv()  # reads .env from current directory by default



CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOOL_ROUNDS = 4


def _headers() -> dict:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY first.")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _post(payload: dict) -> dict:
    resp = requests.post(CHAT_URL, headers=_headers(), json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()


def _accumulate(total: dict, usage: dict) -> None:
    for k in ("prompt_tokens", "completion_tokens"):
        total[k] = total.get(k, 0) + (usage or {}).get(k, 0)


def run_one(model_id: str, task, catalog: dict) -> dict:
    """One repeat of one task on one model. Returns a flat record."""
    messages = [{"role": "system", "content": SYSTEM_QA},
                {"role": "user", "content": task.prompt}]
    usage_total: dict = {}
    tool_calls_made: list[str] = []
    rounds = 0
    start = time.perf_counter()
    error, final_text = None, ""

    try:
        if task.suite == "tool":
            while rounds < MAX_TOOL_ROUNDS:
                rounds += 1
                data = _post({"model": model_id, "messages": messages,
                              "tools": TOOL_SCHEMAS})
                _accumulate(usage_total, data.get("usage", {}))
                msg = data["choices"][0]["message"]
                calls = msg.get("tool_calls") or []
                if not calls:
                    final_text = msg.get("content") or ""
                    break
                messages.append(msg)
                for call in calls:
                    fn = call["function"]
                    name = fn["name"]
                    tool_calls_made.append(name)
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": execute_tool(name, args),
                    })
            else:
                error = "max tool rounds exceeded"
        else:
            data = _post({"model": model_id, "messages": messages})
            _accumulate(usage_total, data.get("usage", {}))
            final_text = data["choices"][0]["message"].get("content") or ""
            rounds = 1
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"

    latency_ms = (time.perf_counter() - start) * 1000
    correct = bool(task.check(final_text)) if error is None else False
    # right_tool = (task.expected_tool in tool_calls_made
    #               if task.expected_tool else None)

    return {
        "model": model_id,
        "suite": task.suite,
        "task_id": task.id,
        "correct": correct,
        # "used_expected_tool": right_tool,
        "tool_calls": tool_calls_made,
        "api_rounds": rounds,
        "prompt_tokens": usage_total.get("prompt_tokens", 0),
        "completion_tokens": usage_total.get("completion_tokens", 0),
        "cost_usd": cost_of(usage_total, model_id, catalog) if error is None else 0.0,
        "latency_ms": latency_ms,
        "error": error,
        "final_text_tail": (final_text or "")[-200:],
    }


def run_experiment(model_ids: list[str], catalog: dict, repeats: int = 3,
                   suites: list[str] | None = None, seed: int = 42,
                   out_path: str = "results.jsonl",
                   sleep_s: float = 0.5) -> Path:
    suites = suites or list(SUITES)
    jobs = [(m, t) for m in model_ids
            for s in suites for t in SUITES[s]
            for _ in range(repeats)]
    random.seed(seed)
    random.shuffle(jobs)  # blocking against time-varying provider conditions

    out = Path(out_path)
    done = 0
    with out.open("a") as fh:
        for model_id, task in jobs:
            record = run_one(model_id, task, catalog)
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            done += 1
            status = "OK " if record["error"] is None else "ERR"
            mark = "+" if record["correct"] else "-"
            print(f"  [{done}/{len(jobs)}] {status} {mark} "
                  f"{model_id.split('/')[-1]:32} {task.id:10} "
                  f"${record['cost_usd']:.5f}")
            time.sleep(sleep_s)
    return out
