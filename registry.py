"""
arch_cost/registry.py — Model registry with architecture metadata.

Architecture is tagged on TWO independent axes (they are not exclusive):
  ffn:  "dense" | "moe"          — sparsity in the feed-forward layers
  attn: "full" | "sparse" | "local-global" — sparsity in the attention pattern

FlashAttention is deliberately NOT an axis: it is a kernel-level
optimization (exact attention, IO-aware), orthogonal to architecture.

Param counts are from public model cards / provider listings and are
best-effort; entries marked verified=False should be re-checked before
quoting them on a slide. Model IDs are OpenRouter IDs; availability and
pricing are resolved LIVE against the OpenRouter models endpoint at run
time (see pricing.py), so a delisted ID degrades gracefully instead of
silently costing the wrong amount.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelSpec:
    id: str                      # OpenRouter model id
    family: str
    ffn: str                     # "dense" | "moe"
    attn: str                    # "full" | "sparse" | "local-global"
    total_params_b: Optional[float]   # billions; None if unknown
    active_params_b: Optional[float]  # equals total for dense models
    verified: bool = True        # False -> double-check before citing
    notes: str = ""

    @property
    def arch_label(self) -> str:
        ffn = "MoE" if self.ffn == "moe" else "Dense"
        attn = {"full": "full attn",
                "sparse": "sparse attn",
                "local-global": "local/global attn"}[self.attn]
        return f"{ffn} + {attn}"

    @property
    def activation_ratio(self) -> Optional[float]:
        if self.total_params_b and self.active_params_b:
            return self.active_params_b / self.total_params_b
        return None


MODELS: list[ModelSpec] = [
    # ── Dense, full attention (the baseline quadrant) ──────────────
    ModelSpec(
        id="meta-llama/llama-3.3-70b-instruct",
        family="Llama", ffn="dense", attn="full",
        total_params_b=70, active_params_b=70,
        notes="Classic dense decoder; every token touches all 70B params.",
    ),
    ModelSpec(
        id="qwen/qwen2.5-72b-instruct",
        family="Qwen", ffn="dense", attn="full",
        total_params_b=72, active_params_b=72,
        notes="Dense baseline #2, different lab/training recipe.",
    ),
    ModelSpec(
        id="mistralai/mistral-small-3.2-24b-instruct",
        family="Mistral", ffn="dense", attn="full",
        total_params_b=24, active_params_b=24, verified=False,
        notes="Small dense point for the cost curve. Verify exact ID on OpenRouter.",
    ),

    # ── Dense, local/global attention ──────────────────────────────
    ModelSpec(
        id="google/gemma-3-27b-it",
        family="Gemma", ffn="dense", attn="local-global",
        total_params_b=27, active_params_b=27,
        notes="Interleaved sliding-window (local) + global attention layers.",
    ),

    # ── MoE, full attention (FFN sparsity only) ────────────────────
    ModelSpec(
        id="openai/gpt-oss-120b",
        family="GPT-OSS", ffn="moe", attn="full",
        total_params_b=117, active_params_b=5.1,
        notes="Extreme activation ratio (~4%). Open-weights OpenAI model.",
    ),
    ModelSpec(
        id="qwen/qwen3-235b-a22b-instruct",
        family="Qwen", ffn="moe", attn="full",
        total_params_b=235, active_params_b=22, verified=False,
        notes="A22B = 22B active. Verify exact instruct-variant ID.",
    ),
    ModelSpec(
        id="moonshotai/kimi-k2",
        family="Kimi", ffn="moe", attn="full",
        total_params_b=1000, active_params_b=32, verified=False,
        notes="~1T total / 32B active. Strong tool-use reputation.",
    ),

    # ── MoE + sparse attention (both axes at once) ─────────────────
    ModelSpec(
        id="deepseek/deepseek-v3.2",
        family="DeepSeek", ffn="moe", attn="sparse",
        total_params_b=671, active_params_b=37,
        notes="DeepSeek Sparse Attention (DSA); supports tool calling.",
    ),
    ModelSpec(
        id="deepseek/deepseek-v4-flash",
        family="DeepSeek", ffn="moe", attn="sparse",
        total_params_b=284, active_params_b=13, verified=False,
        notes="284B/13B active, 1M context. Verify GA model ID.",
    ),
    ModelSpec(
        id="minimax/minimax-m3",
        family="MiniMax", ffn="moe", attn="sparse",
        total_params_b=428, active_params_b=23, verified=False,
        notes="Blockwise MiniMax Sparse Attention; 1M context. Verify ID.",
    ),
]


def by_id(model_id: str) -> Optional[ModelSpec]:
    for m in MODELS:
        if m.id == model_id:
            return m
    return None


def print_registry() -> None:
    print(f"{'model':46} {'arch':24} {'total':>7} {'active':>7} {'ratio':>6}")
    print("-" * 95)
    for m in MODELS:
        ratio = f"{m.activation_ratio:.1%}" if m.activation_ratio else "—"
        tot = f"{m.total_params_b:g}B" if m.total_params_b else "?"
        act = f"{m.active_params_b:g}B" if m.active_params_b else "?"
        flag = "" if m.verified else "  [verify!]"
        print(f"{m.id:46} {m.arch_label:24} {tot:>7} {act:>7} {ratio:>6}{flag}")
