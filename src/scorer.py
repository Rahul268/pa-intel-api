"""
scorer.py
Deterministic Access Score calculation.
Reads penalty rules from access_score_config.json (loaded via settings).
All logic is rule-based — no LLM involved.

Output: 0 | 25 | 50 | 75 | 100
"""

from __future__ import annotations
import logging
import re
from typing import Any

log = logging.getLogger(__name__)


def calculate_access_score(
    row: dict[str, Any],
    brand: str,
    settings: Any,
) -> int:
    """
    Apply penalty rules and return the bucketed Access Score.
    row: normalised parameter dict (post-normalizer.normalize_row)
    """
    config = settings.access_score_config
    brand_info = settings.brand_registry.get("brands", {}).get(brand, {})
    fda_min_age = brand_info.get("fda_pso_min_age")

    if fda_min_age is None:
        # Brand not in registry — skip age penalty (we have no FDA baseline to compare against)
        log.warning(
            "Brand '%s' has no fda_pso_min_age in brand_registry — AGE_001 penalty set to 0.", brand
        )
        fda_min_age = None  # sentinel: age rule will return 0 when None

    total_penalty = 0
    penalty_log: list[dict[str, Any]] = []

    for rule in config.get("penalty_rules", []):
        rule_id = rule["rule_id"]
        param   = rule["parameter"]
        value   = str(row.get(param, "NA")).strip()

        penalty = _apply_rule(rule_id, param, value, row, fda_min_age)
        if penalty > 0:
            penalty_log.append({"rule_id": rule_id, "param": param, "value": value, "penalty": penalty})
            total_penalty += penalty

    raw = max(0, min(100, config["baseline_score"] - total_penalty))

    # Bucket mapping
    bucket = _bucket(raw, config.get("bucket_mapping", []))

    log.debug(
        "Access Score for %s | %s: raw=%d → bucket=%d | penalties=%s",
        row.get("Filename", ""), brand, raw, bucket, penalty_log,
    )
    return bucket


def _apply_rule(
    rule_id: str,
    param: str,
    value: str,
    row: dict[str, Any],
    fda_min_age: int,
) -> int:
    """Return the penalty for a single rule given the current field value."""
    vl = value.lower()

    # ── Age ────────────────────────────────────────────────────────────────
    if rule_id == "AGE_001":
        if fda_min_age is None:
            return 0  # brand not in registry — cannot compute age delta
        if vl in ("na", "fda labelled age", ""):
            return 0
        payer_age = _parse_age_numeric(value)
        if payer_age is None:
            return 0
        delta = payer_age - fda_min_age
        if delta <= 0:
            return 0
        if delta <= 5:
            return 5
        if delta <= 10:
            return 10
        return 15

    # ── Branded steps ──────────────────────────────────────────────────────
    if rule_id == "BRANDS_001":
        n = _parse_int(value)
        if n is None or n == 0:
            return 0
        if n == 1:
            return 15
        if n == 2:
            return 25
        return 35

    # ── Generic steps ──────────────────────────────────────────────────────
    if rule_id == "GENERIC_001":
        n = _parse_int(value)
        if n is None or n == 0:
            return 0
        if n == 1:
            return 8
        if n == 2:
            return 15
        return 20

    # ── Phototherapy ───────────────────────────────────────────────────────
    if rule_id == "PHOTO_001":
        return 10 if vl == "yes" else 0

    # ── TB test ────────────────────────────────────────────────────────────
    if rule_id == "TB_001":
        return 5 if vl == "yes" else 0

    # ── Specialist ─────────────────────────────────────────────────────────
    if rule_id == "SPECIALIST_001":
        return 5 if vl not in ("na", "n/a", "", "none") else 0

    # ── Initial auth duration ──────────────────────────────────────────────
    if rule_id == "INIT_DUR_001":
        months = _parse_months(value)
        if months is None:
            return 0
        if months < 6:
            return 10
        if months == 6:
            return 5
        return 0

    # ── Reauth required ────────────────────────────────────────────────────
    if rule_id == "REAUTH_REQ_001":
        return 5 if vl == "yes" else 0

    # ── Reauth duration ────────────────────────────────────────────────────
    if rule_id == "REAUTH_DUR_001":
        reauth_required = str(row.get("Reauthorization Required", "")).lower()
        if reauth_required != "yes":
            return 0
        months = _parse_months(value)
        if months is None:
            return 0
        return 5 if months < 12 else 0

    # ── Quantity limits ────────────────────────────────────────────────────
    if rule_id == "QL_001":
        return 5 if vl not in ("na", "n/a", "", "none") else 0

    return 0


def _bucket(raw: int, mapping: list[dict]) -> int:
    """Map raw score to output bucket using config ranges."""
    for entry in mapping:
        r = entry.get("raw_score_range", "")
        bucket_val = entry.get("bucket", 0)
        lo, hi = _parse_range(r)
        if lo <= raw <= hi:
            return bucket_val
    # Fallback: linear approximation
    if raw <= 10:   return 0
    if raw <= 35:   return 25
    if raw <= 60:   return 50
    if raw <= 85:   return 75
    return 100


def _parse_range(r: str) -> tuple[int, int]:
    """Parse a score range string like '11-35' or '11–35' into (low, high).
    Used by _bucket() to map raw scores to the five submission buckets.
    """
    m = re.match(r"(\d+)\s*[–\-]\s*(\d+)", r)
    if m:
        return int(m.group(1)), int(m.group(2))
    return (0, 100)


def _parse_age_numeric(value: str) -> int | None:
    """Extract the first integer from an age string (e.g. '>=18' → 18).
    Returns None if no digit found — triggers 'no restriction' in age scoring.
    """
    m = re.search(r"(\d+)", value)
    return int(m.group(1)) if m else None


def _parse_int(value: str) -> int | None:
    """Parse step-count strings like '1', '2', 'NA' → integer or None.
    'NA' and empty strings return None (treated as zero restriction in scoring).
    """
    vl = value.strip().lower()
    if vl in ("na", "n/a", "none", ""):
        return None
    m = re.search(r"(\d+)", vl)
    return int(m.group(1)) if m else None


def _parse_months(value: str) -> int | None:
    """Parse 'X Months' or 'Unspecified' to int | None."""
    vl = value.strip().lower()
    if vl in ("na", "n/a", "none", "", "unspecified"):
        return None
    m = re.search(r"(\d+)", vl)
    return int(m.group(1)) if m else None
