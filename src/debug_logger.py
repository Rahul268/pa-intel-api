"""
debug_logger.py
Step-level execution logging, chunk storage, and prompt storage.
All artifacts land in  intermediate_outputs/debug/<filename>/<brand>/
so they are easy to browse per row.

Enable by setting  settings.save_intermediate = True  (default: True).
"""

from __future__ import annotations
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class DebugLogger:
    """
    One instance per pipeline row.  Call step_start / step_end around each
    named step, then use the specialised helpers to dump chunks and prompts.
    """

    def __init__(
        self,
        filename: str,
        brand: str,
        base_dir: Path,
        enabled: bool = True,
    ) -> None:
        self.filename = filename
        self.brand = brand
        self.enabled = enabled
        self._step_log: list[dict[str, Any]] = []
        self._current_step: str = ""
        self._step_start_ts: float = 0.0

        # Row-specific folder: intermediate_outputs/debug/<stem>/<brand>/
        stem = Path(filename).stem
        safe_brand = brand.replace("/", "_")
        self.row_dir = base_dir / "debug" / stem / safe_brand
        if enabled:
            self.row_dir.mkdir(parents=True, exist_ok=True)

    # ── Step timing ───────────────────────────────────────────────────────────

    def step_start(self, step_name: str) -> None:
        """Record the beginning of a named pipeline step."""
        self._current_step = step_name
        self._step_start_ts = time.time()
        log.debug("[%s | %s] STEP START: %s", self.filename, self.brand, step_name)

    def step_end(self, step_name: str, summary: dict[str, Any] | None = None) -> None:
        """Record the end of a named pipeline step with optional metrics."""
        elapsed = round(time.time() - self._step_start_ts, 3)
        entry = {
            "step": step_name,
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "elapsed_s": elapsed,
        }
        if summary:
            entry["summary"] = summary
        self._step_log.append(entry)
        log.debug(
            "[%s | %s] STEP END:   %s  (%.3fs)",
            self.filename, self.brand, step_name, elapsed,
        )

    def log_step(self, step_name: str, data: dict[str, Any]) -> None:
        """One-shot step logging without start/end timing."""
        entry = {
            "step": step_name,
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "data": data,
        }
        self._step_log.append(entry)

    # ── Specialised dump helpers ───────────────────────────────────────────────

    def save_chunks(self, chunks: list[Any]) -> None:
        """Dump the retrieved evidence chunks to JSON."""
        if not self.enabled:
            return
        out = self.row_dir / "chunks.json"
        payload = [c.to_dict() for c in chunks]
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        log.debug("[%s | %s] Chunks saved → %s (%d chunks)", self.filename, self.brand, out, len(payload))

    def save_prompt(self, prompt: str) -> None:
        """Dump the full extraction prompt to a text file."""
        if not self.enabled:
            return
        out = self.row_dir / "prompt.txt"
        out.write_text(prompt, encoding="utf-8")
        log.debug("[%s | %s] Prompt saved → %s (%d chars)", self.filename, self.brand, out, len(prompt))

    def save_raw_llm_response(self, response: str) -> None:
        """Dump the raw LLM response (pre-parse) to a text file."""
        if not self.enabled:
            return
        out = self.row_dir / "llm_response_raw.txt"
        out.write_text(response, encoding="utf-8")
        log.debug("[%s | %s] Raw LLM response saved → %s", self.filename, self.brand, out)

    def save_extracted_json(self, data: dict[str, Any]) -> None:
        """Dump the parsed extraction JSON (pre-normalisation) to a file."""
        if not self.enabled:
            return
        out = self.row_dir / "extracted_raw.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_normalised_row(self, data: dict[str, Any]) -> None:
        """Dump the normalised row (post-normalisation, pre-scoring)."""
        if not self.enabled:
            return
        out = self.row_dir / "normalised_row.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_evidence_text(self, text: str) -> None:
        """Dump the assembled evidence string that is fed into the prompt."""
        if not self.enabled:
            return
        out = self.row_dir / "evidence_text.txt"
        out.write_text(text, encoding="utf-8")
        log.debug("[%s | %s] Evidence text saved → %s", self.filename, self.brand, out)

    # ── Finalise ───────────────────────────────────────────────────────────────

    def flush(self, final_row: dict[str, Any] | None = None) -> None:
        """Write the step execution log JSON and optionally the final output row."""
        if not self.enabled:
            return
        log_out = self.row_dir / "step_log.json"
        payload = {
            "filename": self.filename,
            "brand": self.brand,
            "generated_at": datetime.now().isoformat(),
            "steps": self._step_log,
        }
        if final_row:
            payload["final_row"] = final_row
        with open(log_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        log.info(
            "[%s | %s] Debug artifacts → %s", self.filename, self.brand, self.row_dir
        )

    # ── Directory listing helper (for notebook) ───────────────────────────────

    @staticmethod
    def list_debug_artifacts(intermediate_outputs_dir: Path) -> list[str]:
        """Return a sorted list of all debug artifact paths under debug/."""
        debug_root = intermediate_outputs_dir / "debug"
        if not debug_root.exists():
            return []
        return sorted(str(p) for p in debug_root.rglob("*.json"))
