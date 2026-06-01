"""
validator.py
Validation gates for the pipeline. Fail fast with clear messages.
Gates:  startup → manifest → extraction → output
"""

from __future__ import annotations
import csv
import logging
from pathlib import Path
from typing import Any

from src.io_utils import RESULT_COLUMNS

log = logging.getLogger(__name__)


# ── Startup validation ────────────────────────────────────────────────────────

def validate_startup(settings: Any) -> None:
    """
    Validate credentials, config files, model names, and required directories.
    Raises RuntimeError on any failure.
    """
    errors: list[str] = []

    # GROQ_API_KEY
    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        errors.append(
            "GROQ_API_KEY is missing or still set to the placeholder value. "
            "Copy .env.template to .env and set a valid key."
        )

    # Model allowlist
    for model in (settings.model_extraction, settings.model_repair):
        if model not in settings.allowed_models:
            errors.append(
                f"Model '{model}' is not in the allowed list: {settings.allowed_models}"
            )

    # Required directories exist (or can be created)
    for attr, label in (
        ("input_pdfs_dir",           "data/input_pdfs"),
        ("outputs_dir",              "outputs"),
        ("intermediate_outputs_dir", "intermediate_outputs"),
        ("logs_dir",                 "logs"),
    ):
        d = getattr(settings, attr)
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            errors.append(f"Cannot create directory {label}: {exc}")

    # Brand registry sanity check — at least one brand entry
    brands = settings.brand_registry.get("brands", {})
    if not brands:
        errors.append("brand_registry.json has no 'brands' entries.")

    # Parameter schema sanity check
    params = settings.parameter_schema.get("parameters", [])
    if len(params) < 12:
        errors.append(
            f"parameter_extraction_schema.json has {len(params)} parameters; expected 12."
        )

    # Access score config sanity check
    rules = settings.access_score_config.get("penalty_rules", [])
    if not rules:
        errors.append("access_score_config.json has no 'penalty_rules'.")

    if errors:
        raise RuntimeError(
            "Startup validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    log.info("Startup validation passed.")


# ── Manifest validation ───────────────────────────────────────────────────────

def validate_manifest(
    manifest: list[dict[str, str]],
    input_pdfs_dir: Path,
    brand_registry: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Check that:
    1. Every PDF listed in the manifest exists under input_pdfs_dir.
    2. Every brand in the manifest has a registry entry.
    Returns the manifest unchanged (raises on hard errors, warns on soft ones).
    """
    errors: list[str] = []
    warnings: list[str] = []

    brands_section = brand_registry.get("brands", {})
    missing_pdfs: list[str] = []

    for row in manifest:
        fn = row["Filename"]
        brand = row["Brand"]

        # Check PDF exists
        pdf_path = input_pdfs_dir / fn
        if not pdf_path.exists():
            missing_pdfs.append(fn)

        # Check brand is in registry
        if brand not in brands_section:
            warnings.append(
                f"Brand '{brand}' (file: {fn}) not found in brand_registry.json. "
                f"Extraction will proceed but retrieval keywords may be limited."
            )

    for w in warnings:
        log.warning("Manifest warning: %s", w)

    if missing_pdfs:
        # Log all missing, then raise
        for fn in missing_pdfs:
            log.error("Missing PDF: %s not found in %s", fn, input_pdfs_dir)
        errors.append(
            f"{len(missing_pdfs)} PDF(s) listed in the manifest are missing from "
            f"{input_pdfs_dir}:\n  " + "\n  ".join(missing_pdfs[:10])
            + ("\n  ... (truncated)" if len(missing_pdfs) > 10 else "")
        )

    if errors:
        raise RuntimeError(
            "Manifest validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    log.info(
        "Manifest validation passed: %d rows, %d unique PDFs",
        len(manifest),
        len({r["Filename"] for r in manifest}),
    )
    return manifest


# ── Extraction JSON validation ────────────────────────────────────────────────

def validate_extraction_json(
    data: dict[str, Any],
    filename: str,
    brand: str,
) -> bool:
    """
    Check that the extracted JSON has all required keys and no None values.
    Returns True if valid, False if issues found (logs warnings).
    """
    from src.extractor import REQUIRED_KEYS

    issues: list[str] = []

    for key in REQUIRED_KEYS:
        if key not in data:
            issues.append(f"Missing key: '{key}'")
        elif data[key] is None:
            issues.append(f"Null value for key: '{key}'")

    if issues:
        log.warning(
            "Extraction JSON issues for %s | %s: %s",
            filename, brand, "; ".join(issues),
        )
        return False

    return True


# ── Output validation ─────────────────────────────────────────────────────────

def validate_output_csv(
    output_path: Path,
    manifest: list[dict[str, str]],
) -> bool:
    """
    Validate the final result.csv:
    - Exact column order matches RESULT_COLUMNS.
    - No blank cells (should all be NA or a value).
    - All manifest rows are present.
    - No duplicate Filename + Brand rows.
    Returns True if valid.
    """
    if not output_path.exists():
        log.error("Output file does not exist: %s", output_path)
        return False

    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        actual_cols = reader.fieldnames or []
        rows = list(reader)

    # Column check
    if list(actual_cols) != RESULT_COLUMNS:
        log.error(
            "Column order mismatch.\n  Expected: %s\n  Got:      %s",
            RESULT_COLUMNS, list(actual_cols),
        )
        return False

    # Duplicate check
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.get("Filename", ""), row.get("Brand", ""))
        if key in seen:
            log.warning("Duplicate row in output: %s | %s", *key)
        seen.add(key)

    # Blank cell check
    blank_count = 0
    for row in rows:
        for col in RESULT_COLUMNS:
            if not row.get(col, "").strip():
                blank_count += 1
                log.warning("Blank cell: col='%s' | %s | %s", col, row.get("Filename"), row.get("Brand"))

    # Manifest coverage
    output_keys = {(r.get("Filename", ""), r.get("Brand", "")) for r in rows}
    for item in manifest:
        k = (item["Filename"], item["Brand"])
        if k not in output_keys:
            log.warning("Manifest row missing from output: %s | %s", *k)

    log.info(
        "Output validation: %d rows, %d blank cells, %d columns",
        len(rows), blank_count, len(actual_cols),
    )
    return blank_count == 0
