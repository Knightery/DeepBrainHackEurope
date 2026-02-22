from __future__ import annotations

from typing import Any, Callable

from .alpaca_historical import fetch_alpaca_historical_bars
from .solana_xstocks import fetch_solana_xstocks_bars

SkillRunner = Callable[[dict[str, Any]], dict[str, Any]]

_SKILL_REGISTRY: dict[str, SkillRunner] = {
    "alpaca_historical_bars": fetch_alpaca_historical_bars,
    "solana_xstocks_bars": fetch_solana_xstocks_bars,
}


def list_data_skills() -> list[str]:
    return sorted(_SKILL_REGISTRY.keys())


def run_data_skill(skill_name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = str(skill_name or "").strip().lower()
    runner = _SKILL_REGISTRY.get(normalized)
    if runner is None:
        return {
            "skill": normalized or "(missing)",
            "status": "fail",
            "summary": f"Unknown skill `{skill_name}`.",
            "artifacts": {"available_skills": list_data_skills()},
        }
    payload = params if isinstance(params, dict) else {}
    return runner(payload)
