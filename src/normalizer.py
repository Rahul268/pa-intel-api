"""
normalizer.py
Deterministic normalization of all 12 extracted parameters.
Rules are derived directly from PA_Business_Rules.xlsx.
No LLM calls — pure Python string matching and regex.

Age note: When a document says "FDA labelled age", the LLM is instructed (via
the extraction prompt) to search the document itself for the actual numeric age.
normalizer.py does NOT look up brand_registry for this — that would bypass the
document-grounded requirement. If the LLM could not find the numeric age in the
document it returns "FDA labelled age" as-is, and this module preserves that.
"""

from __future__ import annotations
import logging
import re
from typing import Any

log = logging.getLogger(__name__)


def normalize_row(extracted: dict[str, Any], brand: str = "") -> dict[str, Any]:
    """Apply all normalization rules to a raw extracted dict. Returns a clean dict."""
    row = dict(extracted)

    row["Age"]                     = normalize_age(row.get("Age", ""))
    row["Step Therapy Requirements Documented in Policy"] = normalize_text_field(
        row.get("Step Therapy Requirements Documented in Policy", "")
    )
    row["Number of Steps through Brands"]  = normalize_step_count(
        row.get("Number of Steps through Brands", "")
    )
    row["Number of Steps through Generic"] = normalize_step_count(
        row.get("Number of Steps through Generic", "")
    )
    row["Step through-Phototherapy"]       = normalize_phototherapy(
        row.get("Step through-Phototherapy", "")
    )
    row["TB Test required"]                = normalize_yes_no(
        row.get("TB Test required", ""), fallback="No"
    )
    row["Quantity Limits"]                 = normalize_text_field(
        row.get("Quantity Limits", "")
    )
    row["Specialist Types"]                = normalize_specialist(
        row.get("Specialist Types", "")
    )
    row["Initial Authorization Duration(in-months)"] = normalize_duration(
        row.get("Initial Authorization Duration(in-months)", ""),
        allow_unspecified=True,
    )
    row["Reauthorization Duration(in-months)"] = normalize_duration(
        row.get("Reauthorization Duration(in-months)", ""),
        allow_unspecified=True,
    )
    # Reauth Required is derived post-extraction (business rule): if either
    # Reauth Duration or Reauth Requirements is non-NA this must be "Yes".
    row["Reauthorization Required"]        = normalize_reauth_required(row)
    row["Reauthorization Requirements Documented in Policy"] = normalize_text_field(
        row.get("Reauthorization Requirements Documented in Policy", "")
    )

    row.setdefault("Filename", "")
    row.setdefault("Brand", brand)

    return row


# ── Individual field normalizers ──────────────────────────────────────────────

def normalize_age(value: str) -> str:
    """
    Normalize age eligibility.
    Returns >=4 | >=6 | >=18 | FDA labelled age | NA

    "FDA labelled age" is preserved as-is when the LLM could not find a numeric
    age in the document. The LLM is instructed to search the document itself
    (dosing tables, appendices, background sections) before falling back to this
    value — so this module never needs to consult brand_registry.
    """
    v = str(value).strip()
    if not v or v.lower() in ("na", "n/a", "none", "not specified", ""):
        return "NA"

    vl = v.lower()

    # Explicit "FDA labelled age" fallback — LLM searched document and found nothing numeric
    if re.search(r"fda\s+label(l?ed)?\s+age|per\s+fda\s+label|as\s+fda\s+label", vl):
        return "FDA labelled age"

    # Extract numeric threshold
    match = re.search(
        r"(\d+)\s*(?:years?|yrs?)\s*(?:of\s*age\s*)?(?:and\s*)?(?:older|above|or\s*older)?",
        vl,
    )
    if match:
        age = int(match.group(1))
        if age <= 4:
            return ">=4"
        if age <= 6:
            return ">=6"
        if age <= 18:
            return ">=18"
        return f">={age}"

    if re.search(r"\badult(s)?\b", vl):
        return ">=18"
    if re.search(r"\bpediatric\b", vl) and not re.search(r"\d+", vl):
        return "FDA labelled age"

    if re.match(r"^>=?\s*\d+$", v):
        return v.replace(" ", "")

    log.debug("normalize_age: unrecognised value '%s', returning as-is", v)
    return v


def normalize_step_count(value: Any) -> str:
    """Convert LLM step-count value to a clean integer string or 'NA'.
    Zero explicit steps or NA-like phrases all normalise to 'NA'.
    """
    v = str(value).strip().lower()
    if v in ("na", "n/a", "none", "not applicable", "no", "", "0"):
        return "NA"
    match = re.search(r"(\d+)", v)
    if match:
        n = int(match.group(1))
        return "NA" if n == 0 else str(n)
    return "NA"


def normalize_phototherapy(value: str) -> str:
    """Normalise phototherapy field to 'Yes' | 'No' | 'N/A'.
    'N/A' (with slash) is the canonical null per PA_Business_Rules.xlsx — it
    means the policy lists no criteria at all, distinct from 'No' (criteria
    exist but phototherapy is not required).
    """
    v = str(value).strip().lower()
    if v in ("yes", "y", "true", "required", "mandatory"):
        return "Yes"
    if v in ("n/a", "na", "not applicable", "no criteria"):
        return "N/A"
    return "No"


def normalize_yes_no(value: str, fallback: str = "No") -> str:
    """Generic Yes/No normaliser used for TB Test required.
    fallback controls what is returned for absent/unrecognised values.
    """
    v = str(value).strip().lower()
    if v in ("yes", "y", "true", "required", "1"):
        return "Yes"
    if v in ("no", "n", "false", "not required", "0", "na", "n/a", ""):
        return fallback
    if v.startswith("y"):
        return "Yes"
    return fallback


def normalize_duration(value: str, allow_unspecified: bool = True) -> str:
    """Convert any duration expression to 'X Months' | 'Unspecified' | 'NA'.
    Handles years (×12), days (÷30), weeks (×4.33), and plain integers
    (assumed months). 'Unspecified' is returned when reauthorization is
    required but no explicit duration is stated in the policy.
    """
    v = str(value).strip()
    vl = v.lower()
    if not v or vl in ("na", "n/a", "none", "not applicable", "not found", ""):
        return "NA"
    if vl in ("unspecified", "not specified", "not stated"):
        return "Unspecified" if allow_unspecified else "NA"
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(month|months|mo|year|years|yr|yrs|day|days|wk|weeks?)", vl,
    )
    if match:
        num = float(match.group(1))
        unit = match.group(2)
        if "year" in unit or "yr" in unit:
            months = round(num * 12)
        elif "day" in unit:
            months = round(num / 30)
        elif "week" in unit or "wk" in unit:
            months = round(num * 4.33)
        else:
            months = round(num)
        return f"{months} Months"
    if re.match(r"^\d+$", v):
        return f"{int(v)} Months"
    if allow_unspecified and re.search(r"unspecif|not\s+stated|not\s+specified", vl):
        return "Unspecified"
    log.debug("normalize_duration: unrecognised value '%s'", v)
    return v


def normalize_reauth_required(row: dict[str, Any]) -> str:
    """Apply the business-rule derived logic for Reauthorization Required.
    If the LLM already returned Yes/No, honour it.
    Otherwise derive: if either Reauthorization Duration or Reauthorization
    Requirements is non-NA, the answer must be 'Yes' (business rule).
    Falls back to 'NA' only when both dependent fields are also NA.
    """
    existing = str(row.get("Reauthorization Required", "")).strip().lower()
    if existing in ("yes", "y"):
        return "Yes"
    if existing == "no":
        return "No"
    duration = str(row.get("Reauthorization Duration(in-months)", "")).strip()
    requirements = str(row.get("Reauthorization Requirements Documented in Policy", "")).strip()
    duration_na = duration.lower() in ("na", "n/a", "none", "", "not applicable")
    requirements_na = requirements.lower() in ("na", "n/a", "none", "", "not applicable")
    if not duration_na or not requirements_na:
        return "Yes"
    return "NA"


def normalize_specialist(value: str) -> str:
    """Normalise specialist types to standard names or 'NA'.
    Multiple specialties are joined with ' or ' to match submission format.
    """
    v = str(value).strip()
    if not v or v.lower() in ("na", "n/a", "none", "no restriction", "any", ""):
        return "NA"
    vl = v.lower()
    specialists: list[str] = []
    if "dermatolog" in vl:
        specialists.append("Dermatologist")
    if "rheumatolog" in vl:
        specialists.append("Rheumatologist")
    if "gastroenterolog" in vl:
        specialists.append("Gastroenterologist")
    if "neurolog" in vl:
        specialists.append("Neurologist")
    if "immunolog" in vl:
        specialists.append("Immunologist")
    if specialists:
        return " or ".join(specialists)
    return v


def normalize_text_field(value: str) -> str:
    """Trim whitespace, collapse excessive internal spaces, and normalise
    all NA-like values to the canonical 'NA' string.
    Used for free-text fields: Step Therapy text, Quantity Limits,
    Reauthorization Requirements.
    """
    v = str(value).strip()
    if not v or v.lower() in ("na", "n/a", "none", "not applicable", "not found", "null"):
        return "NA"
    v = re.sub(r"\s{3,}", "  ", v)
    return v
