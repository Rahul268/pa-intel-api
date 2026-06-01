"""
extractor.py
Two-call extraction pipeline: one call per parameter group.

Call 1 — step_therapy group (llama-3.3-70b-versatile by default)
  Parameters: Step Therapy Requirements text, Steps Brands, Steps Generic,
              Step through-Phototherapy
  Evidence:   Filtered to step-therapy-relevant chunks
  System msg: Cached across all rows (identical static content)

Call 2 — lookups group (llama-3.1-8b-instant by default)
  Parameters: Age, TB, QL, Specialist, Init Auth, Reauth Duration,
              Reauth Required, Reauth Requirements text
  Evidence:   Filtered to lookup-relevant chunks
  System msg: Cached across all rows

Both system messages are sent with cache_control: ephemeral so Groq does not
count cached tokens toward the TPM limit on calls 2–N.

Falls back to the legacy single-call path if extraction_groups is not configured.
"""

from __future__ import annotations
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REQUIRED_KEYS = list({
    "Filename", "Brand", "Age",
    "Step Therapy Requirements Documented in Policy",
    "Number of Steps through Brands",
    "Number of Steps through Generic",
    "Step through-Phototherapy",
    "TB Test required",
    "Quantity Limits",
    "Specialist Types",
    "Initial Authorization Duration(in-months)",
    "Reauthorization Duration(in-months)",
    "Reauthorization Required",
    "Reauthorization Requirements Documented in Policy",
})


def extract_row(
    filename: str,
    brand: str,
    evidence_chunks: list[Any],
    settings: Any,
    groq_client: Any,
) -> dict[str, Any]:
    """
    Run extraction for one manifest row.

    Uses the two-call grouped path when settings.extraction_groups is configured,
    falling back to the legacy single-call path otherwise.
    """
    from src.retrieval import assemble_evidence_text

    evidence_text = assemble_evidence_text(evidence_chunks)

    # ── Cache lookup (full-row cache key) ──────────────────────────────────
    cache_key = _cache_key(filename, brand, evidence_text)
    if settings.cache_extractions:
        cached = _load_cache(cache_key, settings.intermediate_outputs_dir)
        if cached is not None:
            log.info("Cache hit for %s | %s", filename, brand)
            return cached

    # ── Choose extraction path ─────────────────────────────────────────────
    extraction_groups = getattr(settings, "extraction_groups", {})
    if extraction_groups and len(extraction_groups) >= 2:
        extracted = _extract_grouped(
            filename, brand, evidence_chunks, evidence_text, settings, groq_client
        )
    else:
        log.info("extraction_groups not configured — using single-call fallback")
        extracted = _extract_single(
            filename, brand, evidence_chunks, evidence_text, settings, groq_client
        )

    # ── Always override Filename and Brand ────────────────────────────────
    extracted["Filename"] = filename
    extracted["Brand"]    = brand
    for key in REQUIRED_KEYS:
        extracted.setdefault(key, "NA")

    # ── Save intermediate ──────────────────────────────────────────────────
    if settings.save_intermediate:
        _save_intermediate(filename, brand, extracted, evidence_chunks, settings)
    if settings.cache_extractions:
        _save_cache(cache_key, extracted, settings.intermediate_outputs_dir)

    return extracted


# ── Two-call grouped extraction ───────────────────────────────────────────────

def _extract_grouped(
    filename: str,
    brand: str,
    evidence_chunks: list[Any],
    evidence_text: str,
    settings: Any,
    groq_client: Any,
) -> dict[str, Any]:
    """
    Run two separate LLM calls — one per parameter group — and merge results.
    """
    from src.prompts import build_group_messages, split_evidence_by_group
    from src.retrieval import assemble_evidence_text

    extraction_groups = settings.extraction_groups
    merged: dict[str, Any] = {}

    for group_name, group_cfg in extraction_groups.items():
        model      = group_cfg.get("model", settings.model_extraction)
        max_tokens = group_cfg.get("max_tokens", settings.max_tokens)

        # Filter evidence to chunks most relevant to this group
        group_chunks   = split_evidence_by_group(evidence_chunks, group_cfg)
        group_evidence = assemble_evidence_text(group_chunks)

        log.info(
            "Group '%s': %d → %d chunks | %d chars | model=%s",
            group_name, len(evidence_chunks), len(group_chunks),
            len(group_evidence), model,
        )

        # Build system (static, cacheable) + user (dynamic) messages
        system_msg, user_msg = build_group_messages(
            group_name   = group_name,
            group_config = group_cfg,
            filename     = filename,
            brand        = brand,
            evidence_text = group_evidence,
            parameter_schema  = settings.parameter_schema,
            brand_registry    = settings.brand_registry,
            business_rules    = getattr(settings, "business_rules", {}),
            reference_examples= getattr(settings, "reference_examples", {}),
        )

        tok_system = len(system_msg) // 4
        tok_user   = len(user_msg) // 4
        log.info(
            "Group '%s' prompt: system≈%d tok (cached), user≈%d tok, max_tokens=%d, total≈%d",
            group_name, tok_system, tok_user, max_tokens,
            tok_system + tok_user + max_tokens,
        )

        # Call LLM — system message gets cache_control for Groq prefix caching
        raw = groq_client.complete(
            prompt            = user_msg,
            model             = model,
            max_tokens        = max_tokens,
            temperature       = settings.temperature,
            system            = system_msg,
            use_cache_control = True,     # cache static system prefix → saves TPM
            step              = f"extraction_{group_name}",
            filename          = filename,
            brand             = brand,
        )

        group_result = _parse_json(raw)
        if group_result is None:
            log.warning("JSON parse failed for group '%s' — attempting 8B repair", group_name)
            repaired     = groq_client.repair_json(raw, model=settings.model_repair,
                                                   filename=filename, brand=brand)
            group_result = _parse_json(repaired)

        if group_result is None:
            log.error("JSON repair also failed for group '%s'. Using NA fallback.", group_name)
            group_result = {k: "NA" for k in group_cfg.get("parameters", [])}

        merged.update(group_result)
        log.info("Group '%s' extracted %d fields", group_name, len(group_result))

    return merged


# ── Single-call fallback ──────────────────────────────────────────────────────

def _extract_single(
    filename: str,
    brand: str,
    evidence_chunks: list[Any],
    evidence_text: str,
    settings: Any,
    groq_client: Any,
) -> dict[str, Any]:
    """Legacy single-call extraction path.
    Sends all 12 parameters in one prompt to the primary extraction model.
    Used when extraction_groups is not configured in default.yaml, or as a
    safety fallback if the grouped path is unavailable.
    """
    from src.prompts import build_extraction_prompt

    prompt = build_extraction_prompt(
        filename=filename, brand=brand, evidence_text=evidence_text,
        parameter_schema=settings.parameter_schema,
        brand_registry=settings.brand_registry,
        business_rules=getattr(settings, "business_rules", {}),
        reference_examples=getattr(settings, "reference_examples", {}),
    )

    raw = groq_client.complete(
        prompt=prompt, model=settings.model_extraction,
        max_tokens=settings.max_tokens, temperature=settings.temperature,
        step="extraction", filename=filename, brand=brand,
    )

    extracted = _parse_json(raw)
    if extracted is None:
        repaired  = groq_client.repair_json(raw, model=settings.model_repair,
                                            filename=filename, brand=brand)
        extracted = _parse_json(repaired)

    return extracted or _empty_skeleton(filename, brand)


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict[str, Any] | None:
    """Attempt to parse a dict from raw LLM text.
    First strips markdown code fences the model may have added, then tries
    a strict json.loads. On failure, extracts the first {...} substring and
    retries — handles models that prepend explanation text before the JSON.
    Returns None if no valid dict can be parsed (triggers repair call).
    """
    if not text:
        return None
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*",     "", text)
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start: end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(filename: str, brand: str, evidence: str) -> str:
    """Build a filesystem-safe cache key for a (filename, brand, evidence) triple.
    MD5 of the full triple ensures identical evidence → cache hit; different
    evidence (e.g. after context reduction changes) → cache miss.
    """
    h = hashlib.md5(f"{filename}|{brand}|{evidence}".encode()).hexdigest()[:12]
    return f"{filename}__{brand}__{h}".replace("/", "_")


def _load_cache(key: str, cache_dir: Path) -> dict[str, Any] | None:
    """Load a previously cached extraction result. Returns None on miss or error."""
    path = cache_dir / "cache" / f"{key}.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_cache(key: str, data: dict[str, Any], cache_dir: Path) -> None:
    """Persist an extraction result to the JSON cache for future re-use."""
    cache_path = cache_dir / "cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    with open(cache_path / f"{key}.json", "w") as f:
        json.dump(data, f, indent=2)


def _save_intermediate(filename, brand, extracted, chunks, settings):
    """Append the raw extraction result + evidence chunks to the audit JSONL file.
    Written to intermediate_outputs/extractions.jsonl for post-run analysis.
    """
    from src.io_utils import write_jsonl
    record = {"filename": filename, "brand": brand, "extracted": extracted,
              "evidence_chunks": [c.to_dict() for c in chunks]}
    write_jsonl([record], settings.intermediate_outputs_dir / "extractions.jsonl")


def _empty_skeleton(filename: str, brand: str) -> dict[str, Any]:
    """Return an all-NA row with Filename and Brand set.
    Used as the last-resort fallback when both extraction and JSON repair fail.
    """
    return {k: "NA" for k in REQUIRED_KEYS} | {"Filename": filename, "Brand": brand}
