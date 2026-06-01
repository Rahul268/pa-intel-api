"""
groq_client.py
Centralised Groq API wrapper.
- Enforces model allowlist (only llama-3.3-70b-versatile and llama-3.1-8b-instant).
- temperature=0 for reproducibility.
- Exponential-backoff retry on transient errors (not on auth errors).
- Raises on auth errors immediately (no retry).
- Tracks every call in self.usage_log for notebook inspection.

Call pattern (per manifest row):
  Step 5  → complete()     model=llama-3.3-70b-versatile  step="extraction"
  Step 6  → repair_json()  model=llama-3.1-8b-instant     step="repair"  (only if JSON fails)
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)

ALLOWED_MODELS = {
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
}

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0   # seconds


@dataclass
class CallRecord:
    """One entry in the usage log per LLM call."""
    timestamp: str
    step: str           # "extraction" | "repair"
    model: str
    filename: str
    brand: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    success: bool
    attempt: int        # which retry succeeded (1 = first try)


class GroqClient:
    def __init__(self, api_key: str, allowed_models: list[str] | None = None) -> None:
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("groq package required. Install with: pip install groq")

        self._client = Groq(api_key=api_key)
        self._allowed = set(allowed_models or ALLOWED_MODELS)
        self.usage_log: list[CallRecord] = []   # appended after every successful call

    # ── Main completion method ────────────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        system: str | None = None,
        use_cache_control: bool = False,
        # Usage-tracking context (optional but strongly recommended)
        step: str = "unknown",
        filename: str = "",
        brand: str = "",
    ) -> str:
        """
        Send a single-turn completion request.
        Returns the assistant message content as a string.
        Appends a CallRecord to self.usage_log on success.

        use_cache_control: when True, marks the system message with Groq's
        cache_control hint so the static system prefix is cached across calls.
        Cached tokens do not count toward the TPM limit.
        The system message must be identical across calls for caching to apply.
        """
        if model not in self._allowed:
            raise ValueError(
                f"Model '{model}' is not in the allowed list: {self._allowed}. "
                "Only llama-3.3-70b-versatile and llama-3.1-8b-instant are permitted."
            )

        messages: list[dict] = []
        if system:
            if use_cache_control:
                # Groq prompt caching: mark the system message for prefix caching.
                # Format follows the Anthropic-compatible cache_control spec that
                # Groq honours for its hosted LLMs.
                messages.append({
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                })
            else:
                messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_exc: Exception | None = None
        t_start = time.time()

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency = round(time.time() - t_start, 2)
                content = response.choices[0].message.content or ""

                # ── Record usage ───────────────────────────────────────────
                usage = response.usage
                record = CallRecord(
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                    step=step,
                    model=model,
                    filename=filename,
                    brand=brand,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    latency_s=latency,
                    success=True,
                    attempt=attempt,
                )
                self.usage_log.append(record)
                # ──────────────────────────────────────────────────────────

                log.info(
                    "Groq [%s | %s | %s] → %d tokens in %.2fs (attempt %d)",
                    step, model.split("-")[1], f"{filename}/{brand}",
                    record.total_tokens, latency, attempt,
                )
                return content

            except Exception as exc:
                exc_str = str(exc).lower()
                if "401" in exc_str or "invalid api key" in exc_str or "authentication" in exc_str:
                    raise RuntimeError(
                        f"Groq authentication error. Check your GROQ_API_KEY. Details: {exc}"
                    ) from exc
                if "429" in exc_str or "rate limit" in exc_str:
                    wait = _BACKOFF_BASE ** attempt * 3
                    log.warning("Groq rate limit (attempt %d/%d). Waiting %.0fs…", attempt, _MAX_RETRIES, wait)
                else:
                    wait = _BACKOFF_BASE ** attempt
                    log.warning("Groq error (attempt %d/%d): %s. Retrying in %.0fs…", attempt, _MAX_RETRIES, exc, wait)
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)

        raise RuntimeError(
            f"Groq call failed after {_MAX_RETRIES} attempts. Last error: {last_exc}"
        )

    # ── JSON repair (Step 6 fallback) ─────────────────────────────────────────

    def repair_json(
        self,
        malformed: str,
        model: str,
        filename: str = "",
        brand: str = "",
    ) -> str:
        """
        Step 6 — JSON repair fallback using llama-3.1-8b-instant.
        Model must be passed explicitly from settings.model_repair.
        """
        prompt = (
            "The following text should be valid JSON but contains formatting errors. "
            "Fix it and return ONLY the corrected JSON with no explanation, "
            "no markdown, and no preamble.\n\n"
            f"{malformed}"
        )
        return self.complete(
            prompt,
            model=model,
            max_tokens=2000,
            step="repair",
            filename=filename,
            brand=brand,
        )

    # ── Usage reporting ───────────────────────────────────────────────────────

    def usage_summary(self) -> dict[str, Any]:
        """Return aggregated usage stats, broken down by step and model."""
        if not self.usage_log:
            return {"total_calls": 0, "total_tokens": 0}

        by_step: dict[str, dict] = {}
        for r in self.usage_log:
            if r.step not in by_step:
                by_step[r.step] = {
                    "calls": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "total_tokens": 0,
                    "total_latency_s": 0.0, "model": r.model,
                }
            s = by_step[r.step]
            s["calls"] += 1
            s["prompt_tokens"] += r.prompt_tokens
            s["completion_tokens"] += r.completion_tokens
            s["total_tokens"] += r.total_tokens
            s["total_latency_s"] += r.latency_s

        return {
            "total_calls": len(self.usage_log),
            "total_tokens": sum(r.total_tokens for r in self.usage_log),
            "total_latency_s": round(sum(r.latency_s for r in self.usage_log), 2),
            "by_step": by_step,
        }

    def print_usage(self, title: str = "LLM Usage Log") -> None:
        """Pretty-print the usage log to stdout. Call this from the notebook."""
        log_entries = self.usage_log
        if not log_entries:
            print("No LLM calls recorded yet.")
            return

        w = {"time": 10, "step": 11, "model": 30, "file": 26, "brand": 12,
             "prompt": 8, "compl": 8, "total": 7, "lat": 8, "att": 4}
        sep = "─" * (sum(w.values()) + len(w) - 1)

        print(f"\n{'━'*len(sep)}")
        print(f" {title}")
        print(f"{'━'*len(sep)}")
        hdr = (f"{'Time':<{w['time']}} {'Step':<{w['step']}} {'Model':<{w['model']}} "
               f"{'File':<{w['file']}} {'Brand':<{w['brand']}} "
               f"{'Prompt':>{w['prompt']}} {'Compl':>{w['compl']}} "
               f"{'Total':>{w['total']}} {'Lat(s)':>{w['lat']}} {'Try':>{w['att']}}")
        print(hdr)
        print(sep)

        for r in log_entries:
            fname = (r.filename[:23] + "…") if len(r.filename) > 24 else r.filename
            mname = r.model.replace("llama-", "").replace("-versatile", "").replace("-instant", "")
            print(
                f"{r.timestamp:<{w['time']}} {r.step:<{w['step']}} {mname:<{w['model']}} "
                f"{fname:<{w['file']}} {r.brand:<{w['brand']}} "
                f"{r.prompt_tokens:>{w['prompt']},} {r.completion_tokens:>{w['compl']},} "
                f"{r.total_tokens:>{w['total']},} {r.latency_s:>{w['lat']}.2f} "
                f"{r.attempt:>{w['att']}}"
            )

        print(sep)
        summ = self.usage_summary()
        print(f" Calls: {summ['total_calls']}  |  "
              f"Total tokens: {summ['total_tokens']:,}  |  "
              f"Total time: {summ['total_latency_s']}s")
        for step, s in summ["by_step"].items():
            avg = s['total_latency_s'] / s['calls'] if s['calls'] else 0
            print(f"   {step:<11}: {s['calls']} call(s), "
                  f"{s['prompt_tokens']:,} prompt + {s['completion_tokens']:,} completion "
                  f"= {s['total_tokens']:,} tokens, avg {avg:.2f}s/call")
        print(f"{'━'*len(sep)}\n")
