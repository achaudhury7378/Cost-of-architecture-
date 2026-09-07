"""
arch_cost/tasks.py — Three task suites of increasing structural demand.

Design decisions worth defending on stage:

1. Every model answers the SAME queries -> paired design (per-query
   deltas are the unit of comparison, not pooled means).
2. Quality is graded DETERMINISTICALLY (exact numeric/keyword match),
   not by an LLM judge. For tasks with checkable answers this removes
   an entire noise source (judge variance) from the measurement.
3. Models are instructed to end with "ANSWER: <value>" so extraction
   is a regex, not a judgment call.

Suites:
  easy      — single-hop factual/arithmetic. Measures floor cost.
  reasoning — multi-step word problems. Measures how many tokens an
              architecture spends thinking (verbosity IS a cost trait).
  tool      — function-calling with locally-executed mock tools.
              Measures round-trips, tool-call correctness, total cost.
"""

import json
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

ANSWER_RE = re.compile(r"ANSWER:\s*([^\n]+)", re.IGNORECASE)

SYSTEM_QA = (
    "Answer the question. You may reason step by step, but the final "
    "line of your reply MUST be exactly: ANSWER: <value>"
)


@dataclass
class Task:
    id: str
    suite: str                      # "easy" | "reasoning" | "tool"
    prompt: str
    check: Callable[[str], bool]    # grader over the final text
    expected_tool: Optional[str] = None  # for the tool suite


def _num_check(expected: float, tol: float = 1e-4) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        m = ANSWER_RE.search(text or "")
        if not m:
            return False
        raw = m.group(1).replace(",", "").replace("$", "").replace("%", "").strip()
        try:
            return math.isclose(float(re.findall(r"-?\d+\.?\d*", raw)[0]),
                                expected, rel_tol=tol)
        except (ValueError, IndexError):
            return False
    return check


def _kw_check(*keywords: str) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        m = ANSWER_RE.search(text or "")
        blob = (m.group(1) if m else text or "").lower()
        return any(k.lower() in blob for k in keywords)
    return check


# ── Suite 1: easy ─────────────────────────────────────────────────
EASY = [
    Task("easy-1", "easy", "What is 17 * 23?", _num_check(391)),
    Task("easy-2", "easy", "What is the capital of Australia?", _kw_check("canberra")),
    Task("easy-3", "easy", "Convert 100 km to miles (1 km = 0.621371 mi).",
         _num_check(62.1371, tol=0.02)),
    Task("easy-4", "easy", "Which planet is known as the Red Planet?", _kw_check("mars")),
    Task("easy-5", "easy", "What is 15% of 240?", _num_check(36)),
]

# ── Suite 2: multi-step reasoning ─────────────────────────────────
REASONING = [
    Task("reason-1", "reasoning",
         "A train leaves station A at 60 km/h. Two hours later a second train "
         "leaves the same station on a parallel track at 90 km/h. How many hours "
         "after the SECOND train departs does it catch the first? ",
         _num_check(4)),
    Task("reason-2", "reasoning",
         "A shop discounts a $250 jacket by 20%, then applies a 10% member "
         "discount, then adds 8% sales tax. What is the final price in dollars?",
         _num_check(194.40, tol=1e-3)),
    Task("reason-3", "reasoning",
         "I have twice as many apples as Ben. Together we have 27 fewer than "
         "Carol, who has 90. How many apples do I have?",
         _num_check(42)),
    Task("reason-4", "reasoning",
         "A tank fills at 12 L/min and drains at 7 L/min through a leak. It "
         "must reach 600 L, but after 40 minutes the leak is fixed. How many "
         "TOTAL minutes from the start until the tank holds 600 L?",
         _num_check(73)),
    Task("reason-5", "reasoning",
         "Working alone, Priya paints a room in 6 hours and Raj in 4 hours. "
         "They work together for 1 hour, then Raj leaves. How many MORE hours "
         "does Priya need to finish alone?",
         _num_check(3.5)),
]

# ── Suite 3: tool calling ─────────────────────────────────────────
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression, e.g. '3*(4+5)'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get a (fixed, mock) currency exchange rate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "string", "description": "e.g. USD"},
                    "quote": {"type": "string", "description": "e.g. INR"},
                },
                "required": ["base", "quote"],
            },
        },
    },
]

_MOCK_RATES = {("USD", "INR"): 88.0, ("EUR", "USD"): 1.10, ("USD", "JPY"): 150.0}

_SAFE_EXPR = re.compile(r"^[\d\s+\-*/().%]+$")


def execute_tool(name: str, args: dict) -> str:
    """Locally execute mock tools. Deterministic on purpose."""
    if name == "calculator":
        expr = str(args.get("expression", ""))
        if not _SAFE_EXPR.match(expr):
            return json.dumps({"error": "invalid expression"})
        try:
            return json.dumps({"result": eval(expr, {"__builtins__": {}}, {})})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})
    if name == "get_exchange_rate":
        rate = _MOCK_RATES.get((str(args.get("base", "")).upper(),
                                str(args.get("quote", "")).upper()))
        return json.dumps({"rate": rate} if rate else {"error": "unknown pair"})
    return json.dumps({"error": f"unknown tool {name}"})


TOOL = [
    Task("tool-1", "tool",
         "Use the tools to compute 1234 * 5678, then state the result.",
         _num_check(7006652), expected_tool="calculator"),
    Task("tool-2", "tool",
         "Use the tools to find how many INR 350 USD is (use the live rate tool).",
         _num_check(30800), expected_tool="get_exchange_rate"),
    Task("tool-3", "tool",
         "A meal costs 45 EUR. Use the tools to convert it to USD, then add a "
         "15% tip in USD. State the final USD amount.",
         _num_check(56.925, tol=1e-3), expected_tool="get_exchange_rate"),
    Task("tool-4", "tool",
         "Use the tools to compute (89 * 76) - (54 * 33) and state the result.",
         _num_check(4982), expected_tool="calculator"),
]

SUITES = {"easy": EASY, "reasoning": REASONING, "tool": TOOL}
