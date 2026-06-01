"""
step_logic.py
Drug classification for step therapy counting and validation.
Loads the 3-bucket taxonomy (biologic, targeted_synthetic, conventional)
from brand_registry.json and provides utilities used by normalizer and
the extraction prompt builder.

The LLM handles complex OR/AND interpretation during extraction.
This module provides:
  1. Drug-name → step_therapy_class classification (for validation)
  2. Least-restrictive OR path counting logic (deterministic fallback)
  3. Post-extraction validation of LLM step counts
"""

from __future__ import annotations
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

# Phototherapy terms — never count as branded or generic steps
PHOTOTHERAPY_TERMS = [
    r"\bphototherapy\b", r"\buvb\b", r"\bpuva\b", r"\bpsoralen\b",
    r"\bnarrowband\b", r"\bnb-uvb\b", r"\bbroad.?band uvb\b",
    r"\blight therapy\b", r"\bultraviolet\b",
]


class StepClassifier:
    """
    Classifies drug names found in step therapy text into:
      'biologic'           → counts toward Number of Steps through Brands
      'targeted_synthetic' → counts toward Number of Steps through Brands
      'conventional'       → counts toward Number of Steps through Generic
      'phototherapy'       → counts toward Step through-Phototherapy only
      'unknown'            → treated as conventional/generic by default
    """

    def __init__(self, brand_registry: dict[str, Any]) -> None:
        # Build lookup: lowercase_alias → step_therapy_class
        self._alias_class: dict[str, str] = {}
        brands = brand_registry.get("brands", {})
        for info in brands.values():
            cls = info.get("step_therapy_class", "biologic")
            for alias in info.get("aliases", []) + info.get("generic_names", []):
                self._alias_class[alias.lower()] = cls

        # Also load reference drugs (step therapy drugs not in submission manifest)
        ref_drugs = brand_registry.get("step_therapy_reference_drugs", {})
        for cls, entries in ref_drugs.items():
            for entry in entries:
                for alias in entry.get("aliases", []):
                    self._alias_class[alias.lower()] = cls

        log.debug(
            "StepClassifier loaded %d alias→class mappings", len(self._alias_class)
        )

    def classify(self, drug_name: str) -> str:
        """Return step_therapy_class for a drug name string."""
        dl = drug_name.strip().lower()
        # Exact match
        if dl in self._alias_class:
            return self._alias_class[dl]
        # Partial match (for drug class phrases like 'tnf inhibitor')
        for alias, cls in self._alias_class.items():
            if len(alias) > 4 and alias in dl:
                return cls
        return "unknown"

    def is_phototherapy(self, text: str) -> bool:
        """Return True if text matches any known phototherapy alias."""
        tl = text.lower()
        return any(re.search(p, tl) for p in PHOTOTHERAPY_TERMS)

    def extract_drug_mentions(self, text: str) -> list[tuple[str, str]]:
        """
        Find drug/class mentions in step therapy text.
        Returns list of (matched_text, step_therapy_class).
        """
        text_lower = text.lower()
        found: list[tuple[str, str]] = []
        seen_positions: set[int] = set()

        # Sort aliases by length desc (longest match first)
        sorted_aliases = sorted(self._alias_class.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            for m in re.finditer(r"\b" + re.escape(alias) + r"\b", text_lower):
                if m.start() not in seen_positions:
                    seen_positions.add(m.start())
                    found.append((alias, self._alias_class[alias]))

        return found


def validate_step_counts(
    extracted: dict[str, Any],
    classifier: StepClassifier,
) -> dict[str, Any]:
    """
    Cross-check LLM step counts against the step therapy text.
    If counts look reasonable, returns unchanged dict.
    If LLM count is significantly higher than detectable drugs,
    logs a warning but does NOT override (LLM has richer context).
    """
    step_text = extracted.get("Step Therapy Requirements Documented in Policy", "")
    if not step_text or step_text.strip().upper() == "NA":
        return extracted

    mentions = classifier.extract_drug_mentions(step_text)

    biologic_count = sum(
        1 for _, cls in mentions if cls in ("biologic", "targeted_synthetic")
    )
    generic_count = sum(
        1 for _, cls in mentions if cls == "conventional"
    )

    llm_brands  = _parse_count(extracted.get("Number of Steps through Brands", "NA"))
    llm_generic = _parse_count(extracted.get("Number of Steps through Generic", "NA"))

    # Sanity check — warn if LLM count seems implausibly high
    if llm_brands is not None and biologic_count > 0 and llm_brands > biologic_count + 2:
        log.warning(
            "Step count mismatch: LLM brands=%d, detected drug mentions=%d",
            llm_brands, biologic_count,
        )
    if llm_generic is not None and generic_count > 0 and llm_generic > generic_count + 2:
        log.warning(
            "Step count mismatch: LLM generic=%d, detected drug mentions=%d",
            llm_generic, generic_count,
        )

    return extracted  # LLM counts are authoritative; this is validation only


def phototherapy_is_mandatory(step_text: str) -> bool:
    """
    Returns True only if phototherapy appears in the step text AND is
    NOT inside an OR clause (i.e., it is a mandatory standalone requirement).
    Heuristic: if 'or' appears on the same line as a phototherapy term, treat as OR.
    """
    if not step_text:
        return False
    lines = step_text.lower().split("\n")
    for line in lines:
        has_photo = any(re.search(p, line) for p in PHOTOTHERAPY_TERMS)
        if has_photo:
            # If 'or' is on the same line → phototherapy is in an OR clause → not mandatory
            if re.search(r"\bor\b", line):
                return False
            return True
    return False


def _parse_count(value: Any) -> int | None:
    """Parse a step-count string to an integer.
    Returns None for NA or blank values (treated as no restriction by the scorer).
    """
    v = str(value).strip().lower()
    if v in ("na", "n/a", "none", ""):
        return None
    m = re.search(r"(\d+)", v)
    return int(m.group(1)) if m else None
