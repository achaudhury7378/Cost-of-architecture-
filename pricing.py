"""
arch_cost/pricing.py — Resolve model availability + pricing LIVE.

Never hardcode prices: OpenRouter's /api/v1/models returns per-token
prompt/completion prices for every listed model. We fetch once per run,
cache to disk, and price every request from actual reported usage.

This also acts as the availability check: a registry ID missing from
the live listing is reported loudly instead of failing mid-experiment.
"""

import json
import time
from pathlib import Path

import requests

MODELS_URL = "https://openrouter.ai/api/v1/models"
CACHE = Path(".openrouter_models_cache.json")
CACHE_TTL_S = 6 * 3600


def fetch_catalog(force: bool = False) -> dict:
    """Return {model_id: {"prompt": $/token, "completion": $/token, "context": int}}."""
    if not force and CACHE.exists() and time.time() - CACHE.stat().st_mtime < CACHE_TTL_S:
        return json.loads(CACHE.read_text())

    resp = requests.get(MODELS_URL, timeout=30)
    resp.raise_for_status()
    catalog = {}
    for m in resp.json()["data"]:
        pricing = m.get("pricing", {})
        catalog[m["id"]] = {
            "prompt": float(pricing.get("prompt", 0) or 0),       # $ per token
            "completion": float(pricing.get("completion", 0) or 0),
            "context": m.get("context_length"),
            "name": m.get("name", m["id"]),
        }
    CACHE.write_text(json.dumps(catalog))
    return catalog


def resolve(model_ids: list[str], catalog: dict) -> tuple[list[str], list[str]]:
    """Split registry IDs into (available, missing) against the live catalog."""
    available = [m for m in model_ids if m in catalog]
    missing = [m for m in model_ids if m not in catalog]
    return available, missing


def cost_of(usage: dict, model_id: str, catalog: dict) -> float:
    """Price one response from its reported usage. Cached-token pricing is
    ignored (conservative over-estimate) — noted in README limitations."""
    p = catalog[model_id]
    return (usage.get("prompt_tokens", 0) * p["prompt"]
            + usage.get("completion_tokens", 0) * p["completion"])
