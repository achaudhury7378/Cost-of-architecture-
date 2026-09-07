# arch-cost — What does an architecture cost?

Measure the **inference economics** of foundation-model architecture families —
dense, Mixture-of-Experts (MoE), and sparse-attention models — on three task
types of increasing structural demand: easy QA, multi-step reasoning, and tool
calling. Built as the empirical companion to a proposed SciPy India 2026 talk
(proposal at the bottom of this file).

## The taxonomy (and one deliberate exclusion)

Architecture is tagged on **two independent axes**, because modern models mix
them freely:

| Axis | Values | What it means |
|---|---|---|
| FFN sparsity | `dense` / `moe` | Does every token touch every FFN parameter, or only a routed subset (active ≪ total params)? |
| Attention sparsity | `full` / `sparse` / `local-global` | Does every token attend to every other token, or to a structured subset? |

Examples in the registry: Llama 3.3 70B (dense + full), Gemma 3 27B (dense +
local/global), GPT-OSS-120B (MoE + full, ~4% activation ratio), DeepSeek V3.2
and MiniMax M3 (MoE **and** sparse attention — both axes at once).

**FlashAttention is deliberately not an axis.** It is a kernel-level
optimization — exact attention computed IO-efficiently — orthogonal to
architecture. Dense, MoE, and sparse-attention models all run on
FlashAttention-style kernels. It belongs in the talk as "the systems layer
underneath," not in the comparison grid.

## Platform choice

Everything runs through **OpenRouter** (one API key, OpenAI-compatible,
hundreds of open models). Rationale:

- Sign-up needs no credit card; `:free` model variants exist.
- **Recommended path:** make a one-time **$10** credit purchase and use *cheap
  paid* open models rather than `:free` variants. The free roster rotates
  without notice (whole Llama/Qwen free tiers have been delisted in the past),
  which is poison for a reproducible experiment. Paid open models are
  $0.07–$1 per million tokens; the full default experiment costs well under $1.
- Pricing is fetched **live** from `openrouter.ai/api/v1/models` at run time —
  no hardcoded prices to go stale — and every call is priced from its actual
  reported token usage.

Alternatives (Groq, Together, Fireworks) work with minor URL changes but none
covers all architecture families in one place the way an aggregator does.

## Setup

```bash
pip install requests numpy
export OPENROUTER_API_KEY=sk-or-...   # from openrouter.ai/keys
```

## Usage

```bash
python -m arch_cost registry   # show the architecture grid
python -m arch_cost check      # verify registry IDs against the LIVE catalog
                               # (rotating rosters: expect some "NOT on OpenRouter";
                               #  fix IDs in registry.py before running)
python -m arch_cost run --repeats 3                # full experiment
python -m arch_cost run --suites easy tool \
    --models deepseek/deepseek-v3.2 openai/gpt-oss-120b   # subset
python -m arch_cost report     # tables + paired bootstrap comparisons
```

Raw per-call records append to `results.jsonl` (model, task, tokens, cost,
latency, correctness, tool-call trace). Analysis is a separate offline pass,
so you can re-analyze without re-spending.

## What the analysis does (and why)

1. **Unit of analysis = task, not run.** Repeats of the same task are averaged
   first. Treating 3 repeats as 3 independent samples inflates n
   (pseudo-replication) and manufactures false significance.
2. **Paired comparisons.** Every model answers the same tasks, so model-vs-model
   cost differences are computed per-task and tested with a **paired bootstrap
   95% CI** on the mean delta. Task difficulty — the dominant variance source —
   cancels in the pairing.
3. **Cost per *solved* task** is the headline metric: `total cost / tasks
   solved`. A model that is half the price but wrong twice as often is not
   cheaper.
4. **Deterministic grading.** Answers end in `ANSWER: <value>` and are graded
   by exact numeric/keyword match — no LLM judge, hence no judge variance in
   the measurement. Tool tasks additionally check that the *expected tool* was
   actually called (a model that mental-maths past your calculator is a
   correctness risk, and the trace catches it).
5. **Shuffled execution order** spreads time-varying provider conditions
   (load, rate limiting) across models instead of biasing whichever ran last.

## Honest limitations (also slide material)

- **You observe price, not FLOPs.** API pricing embeds provider margins,
  hardware, and quantization choices. Defensible claim: "architecture X costs
  *me* $Y per solved task via commodity APIs" — not "architecture X is
  computationally cheaper." Mitigation: multiple models per family; the MoE
  activation-ratio → price correlation survives the confound loudly.
- **Verbosity is confounded with architecture.** Some models think longer by
  default; output tokens are reported separately so you can see whether a
  model is expensive per-token or expensive because it talks.
- Cached-input pricing is ignored (small, conservative over-estimate).
- Small task suites (14 tasks) → CIs are wide by design honesty; grow the
  suites before making strong claims.
- Param counts marked `[verify!]` in the registry must be re-checked against
  model cards before appearing on a slide.

---

# SciPy India 2026 — 30-minute talk proposal (draft)

**Title:** What does an architecture cost? Measuring the inference economics
of dense, MoE, and sparse-attention language models with SciPy

**Track:** AI, machine learning, and data-driven discovery
(secondary: Reproducibility in research)

**Abstract (~100 words):**
Mixture-of-Experts models advertise a trillion parameters but activate 3–5%
of them per token; sparse attention promises cheap long context. What do
these architecture choices actually cost *you* per task? We built a small
open harness (requests + numpy + scipy) that runs dense, MoE, and
sparse-attention open models through three task types — easy QA, multi-step
reasoning, and tool calling — over a commodity API, pricing every call from
live per-token rates. Along the way we treat LLM evaluation as the
experimental-design problem it is: paired comparisons, bootstrap confidence
intervals, pseudo-replication traps, and cost-per-solved-task as the metric
that keeps cheap-but-wrong models honest.

**Outline (30 min):**
1. *The architecture zoo, in one grid* (5 min) — dense vs MoE (FFN sparsity)
   vs sparse attention (attention sparsity) as independent axes; activation
   ratios from 100% down to ~4%; why FlashAttention is not on the grid.
2. *Measurement design* (7 min) — why single runs of stochastic systems are
   noise; pairing on task identity; unit-of-analysis and pseudo-replication;
   deterministic grading vs LLM judges; execution-order shuffling as blocking.
3. *Live results walk-through* (10 min) — cost/accuracy tables per task type;
   paired bootstrap CIs on cost deltas; the cost-per-solved-task ranking vs
   the naive price-per-token ranking (they disagree, and that's the punchline);
   activation ratio vs realized price.
4. *What you can and cannot conclude* (5 min) — price ≠ FLOPs; provider
   confounds; verbosity as a cost trait; when a CI crossing zero is the
   honest headline.
5. *Take it home* (3 min) — the harness is ~600 lines of standard scientific
   Python; adapt the task suites to your own workload before believing anyone
   else's benchmark.

**Why SciPy India:** This is a scientific-computing talk wearing an LLM
costume: experimental design, resampling statistics, and honest uncertainty
reporting, applied to a decision every Indian engineering team is currently
making by vibes ("which model is cheapest for us?"). Attendees leave with a
runnable harness and a checklist for auditing any benchmark they read.

**Speaker notes for the proposal form:** results shown will be regenerated
the week before the conference (model rosters and prices rotate); the repo,
raw JSONL, and analysis notebook will be public.

---

### Poster hedge (submit alongside)

Same skeleton, one dominant visual: a slope graph of per-task cost pairing
two models (dense vs MoE) across all 14 tasks, with the paired-bootstrap CI
printed beside the naive pooled comparison — the pairing structure the naive
analysis throws away, visible as ink.
