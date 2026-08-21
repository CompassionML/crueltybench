"""Adaptive per-model generation parameters: sampling temperature and reasoning effort.

Some models honour the ``temperature`` parameter; the newest ones (e.g. Claude Sonnet 5 /
Opus 5) have deprecated it — OpenRouter silently drops it, while Bedrock rejects it with a
hard error. This module resolves, per model, whether a desired temperature is honoured, so
callers can pass it where it works and omit it where it doesn't (rather than erroring).

``resolve_reasoning_effort`` does the same job for reasoning: providers expose different
rungs (kimi-k3 has no ``medium``, gemini no ``max``, claude-haiku-4.5 no reasoning at all),
so a desired effort is snapped to the nearest rung the model actually supports, or omitted
entirely for models that don't reason.

Detection is authoritative where possible:
- ``openrouter/*`` — query OpenRouter's ``supported_parameters`` for the model.
- everything else (``bedrock/*`` …) — a one-time 1-token probe: send the temperature and see
  whether the provider rejects it.

Results are cached per model id, so detection happens once per model per process.
"""

from __future__ import annotations

import os

import httpx
from inspect_ai.model import GenerateConfig, Model

# model id -> whether `temperature` is honoured
_SUPPORT: dict[str, bool] = {}


async def _openrouter_supports_temperature(model_id: str) -> bool:
    """True if OpenRouter lists `temperature` among the model's supported_parameters."""
    slug = model_id.split("/", 1)[1]  # openrouter/anthropic/claude-sonnet-5 -> anthropic/claude-sonnet-5
    key = os.environ.get("OPENROUTER_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(f"https://openrouter.ai/api/v1/parameters/{slug}", headers=headers)
        r.raise_for_status()
        data = r.json().get("data", {})
        return "temperature" in (data.get("supported_parameters") or [])


async def _probe_supports_temperature(model: Model, desired: float) -> bool:
    """Send a 1-token request with the temperature set; False iff it's rejected as unsupported."""
    try:
        await model.generate("ok", config=GenerateConfig(temperature=desired, max_tokens=1))
        return True
    except Exception as e:  # noqa: BLE001 — provider errors vary; inspect the message
        if "temperature" in str(e).lower():
            return False
        # Unrelated failure (network, auth, …): don't block the run on the probe — assume
        # supported and let the real generation surface any genuine error.
        return True


async def resolve_temperature(model_id: str, model: Model, desired: float) -> float | None:
    """Return ``desired`` if the model honours temperature, else ``None`` (omit it).

    ``model`` is only used for the probe path (non-OpenRouter models). Detection is cached
    per model id.
    """
    if model_id not in _SUPPORT:
        if "mockllm" in model_id:
            _SUPPORT[model_id] = True  # mock accepts anything; never probe it
        elif model_id.startswith("openrouter/"):
            try:
                _SUPPORT[model_id] = await _openrouter_supports_temperature(model_id)
            except Exception:  # noqa: BLE001 — fall back to probing if the API call fails
                _SUPPORT[model_id] = await _probe_supports_temperature(model, desired)
        else:
            _SUPPORT[model_id] = await _probe_supports_temperature(model, desired)
    return desired if _SUPPORT[model_id] else None


# Reasoning effort rungs, weakest to strongest. Providers each expose a subset (OpenRouter names
# every provider's native levels from this shared vocabulary), so a desired rung may be absent.
_EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

# model id -> the reasoning efforts OpenRouter lists for it ([] = model doesn't reason)
_EFFORTS: dict[str, list[str]] = {}


async def _openrouter_supported_efforts(model_id: str) -> list[str]:
    """The reasoning efforts OpenRouter lists for a model ([] if it has none)."""
    slug = model_id.split("/", 1)[1]
    key = os.environ.get("OPENROUTER_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get("https://openrouter.ai/api/v1/models", headers=headers)
        r.raise_for_status()
        for entry in r.json().get("data", []):
            if entry.get("id") == slug:
                return list((entry.get("reasoning") or {}).get("supported_efforts") or [])
    return []


def _nearest_effort(desired: str, supported: list[str]) -> str | None:
    """Snap ``desired`` to the nearest supported rung, preferring the next one *up*.

    Reasoning a little harder than asked is the safer miss for a judge: it risks cost, whereas
    under-reasoning risks the gated rubric being applied holistically.
    """
    known = [e for e in supported if e in _EFFORT_ORDER]
    if not known:
        return None
    if desired in known:
        return desired
    target = _EFFORT_ORDER.index(desired)
    return min(known, key=lambda e: (abs(_EFFORT_ORDER.index(e) - target), -_EFFORT_ORDER.index(e)))


async def resolve_reasoning_effort(model_id: str, desired: str) -> str | None:
    """Return the reasoning effort to send for ``model_id``, or ``None`` to omit the parameter.

    ``desired`` is snapped to the nearest rung the model supports; models that expose no
    reasoning efforts (e.g. claude-haiku-4.5) get ``None`` so the parameter is never sent to a
    model that would reject it. Detection is cached per model id; on lookup failure the desired
    value is passed through unchanged rather than blocking the run.
    """
    if "mockllm" in model_id:
        return desired  # mock accepts anything; never look it up
    if model_id not in _EFFORTS:
        if model_id.startswith("openrouter/"):
            try:
                _EFFORTS[model_id] = await _openrouter_supported_efforts(model_id)
            except Exception:  # noqa: BLE001 — don't block a run on the metadata lookup
                return desired
        else:
            return desired  # non-OpenRouter: let the provider decide
    return _nearest_effort(desired, _EFFORTS[model_id])
