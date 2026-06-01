"""
brand_detector.py
Two-mode brand detection:

1. detect_supported_brands(text, brand_registry)
   Fast path — checks only the 15 supported brands in brand_registry["brands"].
   Used when the filename IS in the manifest (brand is already known) and for
   quick sanity checks.

2. detect_all_policy_drugs(pages, brand_registry)
   General path — used when a filename is NOT in the manifest.
   Step A: check brand_registry aliases (covers supported brands if present).
   Step B: scan document structure for any drug name mentioned in standard PA
           policy sections ("Products Affected:", "Requires PA:", title text, etc.)
           — no restriction to a predefined brand list.
   Returns a deduplicated list of brand/drug names found, in the order they appear.
"""

from __future__ import annotations
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

# ── Ustekinumab biosimilar disambiguation ─────────────────────────────────────
USTEKINUMAB_BIOSIMILAR_SUFFIXES: dict[str, str] = {
    "ustekinumab-kfce": "YESINTEK",
    "ustekinumab-aauz": "OTULFI",
}
USTEKINUMAB_BASE       = "ustekinumab"
USTEKINUMAB_BASE_BRAND = "STELARA"

# Patterns that signal the drug name is what follows them in a PA policy document
_PA_DRUG_SECTION_PATTERNS = [
    # "Products Affected:\n• Tremfya (guselkumab)"  or  "Products Affected: Tremfya"
    r"Products\s+Affected[:\s]+[•\-]?\s*([A-Za-z][A-Za-z0-9\-]+(?:\s+[A-Za-z][A-Za-z0-9\-]+){0,3})",
    # "Requires PA:\n• Tremfya"
    r"Requires\s+PA[:\s]+[•\-]?\s*([A-Za-z][A-Za-z0-9\-]+(?:\s+[A-Za-z][A-Za-z0-9\-]+){0,2})",
    # "Policy for Tremfya" / "Drug Management Policy\nTremfya"
    r"(?:Drug\s+Management\s+Policy|Policy\s+(?:for|[-:–]))\s*\n?\s*([A-Z][a-z]+(?:[A-Za-z0-9\-]*[a-z]+)?)",
    # Standalone capitalised drug name on its own line (title position, page 1)
    r"^([A-Z][a-z]+(?:[A-Za-z0-9\-]{2,})?)\s*$",
    # Drug name in parenthetical note: "(guselkumab)"  →  extract the brand from prior word
    r"([A-Z][a-z]+[A-Za-z0-9\-]*)\s+\([a-z][a-z0-9\-]+(?:\s+[a-z0-9\-]+)?\)",
]

# Common non-drug words that appear in PA policy titles — avoid false positives
_STOP_WORDS = {
    "Prior", "Authorization", "Policy", "Criteria", "Coverage", "Drug",
    "Management", "Plan", "Health", "Medical", "Pharmacy", "Fee", "Service",
    "Oregon", "Medicaid", "Medicare", "Approval", "Clinical", "Benefit",
    "Products", "Affected", "Program", "Division", "Systems", "Overview",
    "Background", "Introduction", "General", "Contents", "Table", "Appendix",
    "Section", "Summary", "Version", "Effective", "Date", "Number", "Annual",
    "Review", "Committee", "Therapeutics", "Therapeutic", "Guideline",
    "Note", "Goal", "Goals", "Length",
}


def detect_supported_brands(text: str, brand_registry: dict[str, Any]) -> list[str]:
    """
    Check text against the 15 supported brands in brand_registry["brands"].
    Returns a sorted list of brand display names found.
    Used for manifest-mode sanity checks and the adhoc auto-mode.
    """
    text_lower = text.lower()
    found: set[str] = set()
    brands_section = brand_registry.get("brands", {})

    # --- Biosimilar suffixes first (most specific) ---
    for suffix, brand_name in USTEKINUMAB_BIOSIMILAR_SUFFIXES.items():
        if re.search(r"\b" + re.escape(suffix) + r"\b", text_lower):
            found.add(brand_name)

    # --- All other brands via aliases ---
    for brand_name, info in brands_section.items():
        if brand_name in found:
            continue
        aliases = info.get("aliases", []) + info.get("generic_names", [])
        for alias in aliases:
            al = alias.lower()
            if al == USTEKINUMAB_BASE:
                if any(b in found for b in USTEKINUMAB_BIOSIMILAR_SUFFIXES.values()):
                    if re.search(r"\bstelara\b", text_lower):
                        found.add(USTEKINUMAB_BASE_BRAND)
                    continue
            if re.search(r"\b" + re.escape(al) + r"\b", text_lower):
                found.add(brand_name)
                break

    return sorted(found)


def detect_all_policy_drugs(
    pages: list[Any],
    brand_registry: dict[str, Any],
) -> list[str]:
    """
    General drug detection for files NOT in the manifest.

    Step A — check brand_registry (fast, covers supported brands).
    Step B — parse document structure using PA policy patterns to find
             any drug name regardless of whether it's in our registry.

    Returns a deduplicated list of drug/brand names in detection order.
    """
    from src.pdf_parser import get_full_text

    full_text = get_full_text(pages)

    # ── Step A: check known brands ─────────────────────────────────────────
    known = detect_supported_brands(full_text, brand_registry)

    # ── Step B: scan document structure ───────────────────────────────────
    from_structure = _detect_from_structure(pages)

    # Merge: known brands first, then any new ones from structure
    known_lower = {b.lower() for b in known}
    extra = [d for d in from_structure if d.lower() not in known_lower]
    combined = known + extra

    if combined:
        log.info("General drug detection found: %s (known=%s, extra=%s)", combined, known, extra)
    else:
        log.warning("General drug detection found no drugs in document.")

    return combined


def _detect_from_structure(pages: list[Any]) -> list[str]:
    """
    Scan the first 3 pages of a policy document for drug names using
    standard PA policy section patterns.
    """
    # Scan first 3 pages (title + first body page)
    scan_pages = pages[:3]
    candidates: list[str] = []

    for page in scan_pages:
        text = page.get("text", "") if isinstance(page, dict) else ""
        if not text:
            continue

        # Pattern-based extraction
        for pattern in _PA_DRUG_SECTION_PATTERNS:
            for m in re.finditer(pattern, text, re.MULTILINE):
                raw = m.group(1).strip()
                cleaned = _clean_candidate(raw)
                if cleaned:
                    candidates.append(cleaned)

    # Deduplicate preserving order
    seen: set[str] = set()
    result: list[str] = []
    for c in candidates:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            result.append(c)

    return result


def _clean_candidate(raw: str) -> str:
    """
    Clean and validate a drug name candidate.
    Returns empty string if it looks like a non-drug word.
    """
    # Strip punctuation from ends
    raw = raw.strip(" \t\n•-:,.()")

    # Must be at least 3 chars
    if len(raw) < 3:
        return ""

    # First token must start with a capital letter
    first_token = raw.split()[0] if raw.split() else ""
    if not first_token or not first_token[0].isupper():
        return ""

    # Filter stop words (first token)
    if first_token in _STOP_WORDS:
        return ""

    # Must contain at least one letter
    if not re.search(r"[a-zA-Z]", raw):
        return ""

    return raw


def get_brand_display_name(raw: str, brand_registry: dict[str, Any]) -> str | None:
    """Normalise a raw brand string to its canonical display name. Returns None if not found."""
    raw_upper = raw.strip().upper()
    brands = brand_registry.get("brands", {})
    if raw_upper in brands:
        return raw_upper
    raw_lower = raw.strip().lower()
    for brand_name, info in brands.items():
        aliases = info.get("aliases", []) + info.get("generic_names", [])
        if any(a.lower() == raw_lower for a in aliases):
            return brand_name
    return None
