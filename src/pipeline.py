"""
pipeline.py
End-to-end pipeline orchestration.

Brand resolution logic (Issue 1):
  For each PDF in data/input_pdfs/:
    1. Check brand_registry["manifest"] (derived from PA_Business_Rules.xlsx Submissions tab).
       If the filename is listed there → use the manifest brand. This is the authoritative source.
    2. If the filename is NOT in the manifest → auto-detect brands using the 15 supported
       brands in brand_registry["brands"], then process one row per detected brand.

This means files in the hackathon manifest are always processed with the correct
brand regardless of what the LLM returns in its JSON.  Files outside the manifest
(e.g., adhoc PDFs dropped into data/input_pdfs/) are handled gracefully without
any manual configuration.
"""

from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def run(settings: Any, mode: str, input_dir: Path | None = None, output_path: Path | None = None) -> None:
    """Pipeline entry point.
    Validates startup conditions, initialises the Groq client, then delegates
    to the appropriate mode runner.
    Modes: manifest — process rows from PA_Business_Rules.xlsx Submissions tab.
           auto    — scan input_dir for PDFs and auto-detect brands.
           both    — manifest first, then auto on adhoc_pdfs dir.
    """
    from src.validator import validate_startup
    from src.groq_client import GroqClient
    from src.io_utils import configure_logging

    configure_logging(settings.logs_dir)
    validate_startup(settings)

    groq = GroqClient(
        api_key=settings.groq_api_key,
        allowed_models=settings.allowed_models,
    )

    if mode == "manifest":
        _run_manifest(settings, groq, output_path)
    elif mode == "auto":
        _run_auto(settings, groq, input_dir or settings.adhoc_pdfs_dir, output_path)
    elif mode == "both":
        _run_manifest(settings, groq, output_path)
        auto_out = settings.outputs_dir / "adhoc_result.csv"
        _run_auto(settings, groq, settings.adhoc_pdfs_dir, auto_out)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use: manifest | auto | both")


# ── Brand resolution ──────────────────────────────────────────────────────────

def _resolve_brands_for_pdfs(
    pdf_paths: list[Path],
    settings: Any,
) -> list[dict[str, str]]:
    """
    For each PDF, determine which brand(s) to process.

    Priority:
      1. brand_registry manifest (from PA_Business_Rules.xlsx Submissions tab)
         → exact brand, no document scanning needed
      2. Auto-detect from document content (limited to 15 supported brands)
         → used for PDFs not listed in the manifest

    Returns a list of {Filename, Brand, _source} dicts where _source is
    'manifest' or 'auto_detected' for logging/debug purposes.
    """
    from src.brand_detector import detect_supported_brands, detect_all_policy_drugs
    from src.pdf_parser import parse_pdf, get_full_text

    # Build a lookup from brand_registry["manifest"] — this is the runtime source of truth.
    # The Excel Submissions tab is the BUILD-TIME source; brand_registry.json is the pre-extracted
    # runtime artifact. Reading the Excel at runtime would add unnecessary I/O since
    # brand_registry.json is already loaded in settings.
    manifest_lookup: dict[str, list[str]] = {}
    for entry in settings.brand_registry.get("manifest", []):
        fn = entry.get("filename") or entry.get("Filename", "")
        br = entry.get("brand") or entry.get("Brand", "")
        if fn and br:
            manifest_lookup.setdefault(fn, []).append(br)

    resolved: list[dict[str, str]] = []

    for pdf_path in pdf_paths:
        fn = pdf_path.name

        if fn in manifest_lookup:
            # ── Manifest path: use brands from registry ────────────────────
            for brand in manifest_lookup[fn]:
                resolved.append({"Filename": fn, "Brand": brand, "_source": "manifest"})
            log.info(
                "Brand(s) from manifest for %s: %s",
                fn, manifest_lookup[fn],
            )
        else:
            # ── Auto-detect path: scan document without brand restriction ──
            log.info(
                "%s not in manifest — running general drug detection (unrestricted)", fn
            )
            try:
                pages = parse_pdf(pdf_path)
                detected = detect_all_policy_drugs(pages, settings.brand_registry)
                if detected:
                    for brand in detected:
                        resolved.append({"Filename": fn, "Brand": brand, "_source": "auto_detected"})
                    log.info("Auto-detected drugs in %s: %s", fn, detected)
                else:
                    log.warning(
                        "No drugs detected in %s — skipping", fn
                    )
            except Exception as exc:
                log.error("Failed to detect drugs in %s: %s", fn, exc)

    return resolved


# ── Run modes ─────────────────────────────────────────────────────────────────

def _run_manifest(settings: Any, groq: Any, output_path: Path | None) -> None:
    """Manifest mode: process every (Filename, Brand) row from brand_registry.
    PDFs in data/input_pdfs/ are matched against the manifest first; any PDF
    not in the manifest has its brands auto-detected from document content.
    Writes result.csv and result.xlsx to the outputs directory.
    """
    from src.io_utils import list_pdfs, write_result_csv, write_result_excel, timestamped_backup
    from src.validator import validate_output_csv

    out = output_path or settings.outputs_dir / "result.csv"

    if out.exists() and settings.idempotent == "timestamp_backup":
        timestamped_backup(out)

    # Resolve brands for all PDFs in input_pdfs_dir
    all_pdfs = list_pdfs(settings.input_pdfs_dir)
    if not all_pdfs:
        log.warning("No PDFs found in %s", settings.input_pdfs_dir)
        return

    manifest = _resolve_brands_for_pdfs(all_pdfs, settings)
    if not manifest:
        log.warning("No processable rows resolved from PDFs in %s", settings.input_pdfs_dir)
        return

    manifest_entries = [r for r in manifest if r["_source"] == "manifest"]
    auto_entries = [r for r in manifest if r["_source"] == "auto_detected"]
    log.info(
        "=== MANIFEST MODE: %d rows total (%d from manifest, %d auto-detected) ===",
        len(manifest), len(manifest_entries), len(auto_entries),
    )

    results = _process_manifest_rows(manifest, settings, groq)
    write_result_csv(results, out)
    write_result_excel(results, out.with_suffix(".xlsx"))

    # Validate only against manifest-driven rows (auto-detected rows are a bonus)
    if manifest_entries:
        validate_output_csv(out, manifest_entries)
    log.info("=== Done. Output → %s ===", out)


def _run_auto(settings: Any, groq: Any, input_dir: Path, output_path: Path | None) -> None:
    """Auto mode: scan an arbitrary directory and always auto-detect brands.
    Used for ad-hoc PDFs dropped into data/adhoc_pdfs/ that are not in the
    manifest. Brand detection uses text-based matching against the 15 supported
    brands in brand_registry.json.
    """

    from src.io_utils import list_pdfs, write_result_csv, write_result_excel
    from src.brand_detector import detect_supported_brands, detect_all_policy_drugs
    from src.pdf_parser import parse_pdf, get_full_text

    out = output_path or settings.outputs_dir / "adhoc_result.csv"
    pdfs = list_pdfs(input_dir)

    if not pdfs:
        log.warning("No PDFs found in %s", input_dir)
        return

    manifest: list[dict[str, str]] = []
    for pdf_path in pdfs:
        try:
            pages = parse_pdf(pdf_path)
            full_text = get_full_text(pages)
            brands = detect_supported_brands(full_text, settings.brand_registry)
            if not brands:
                log.info("No supported brands detected in %s — skipping", pdf_path.name)
                continue
            for brand in brands:
                manifest.append({"Filename": pdf_path.name, "Brand": brand, "_source": "auto_detected"})
        except Exception as exc:
            log.error("Auto-mode failed for %s: %s", pdf_path.name, exc)

    log.info("Auto mode: %d Filename+Brand rows detected across %d PDFs", len(manifest), len(pdfs))
    results = _process_manifest_rows(manifest, settings, groq)
    write_result_csv(results, out)
    write_result_excel(results, out.with_suffix(".xlsx"))
    log.info("Auto mode done. Output → %s", out)


# ── Core processing loop ──────────────────────────────────────────────────────

def _process_manifest_rows(
    manifest: list[dict[str, str]],
    settings: Any,
    groq: Any,
) -> list[dict[str, Any]]:
    """Process all manifest rows through the full pipeline.
    For each (Filename, Brand) pair: parse PDF, chunk, build BM25 index,
    retrieve evidence via RRF, reduce context, run two-call LLM extraction,
    normalise, validate, and score. Errors on individual rows are caught
    and recorded as NA rows so the run always completes.
    Returns a list of result dicts ready to write to CSV/Excel.
    """
    from src.pdf_parser import parse_pdf, extract_document_headings
    from src.chunker import chunk_pdf_pages
    from src.retrieval import build_index, retrieve_evidence, assemble_evidence_text
    from src.context_reducer import reduce_evidence_context
    from src.extractor import extract_row
    from src.normalizer import normalize_row
    from src.step_logic import StepClassifier, validate_step_counts
    from src.scorer import calculate_access_score
    from src.debug_logger import DebugLogger
    from src.prompts import build_extraction_prompt

    classifier = StepClassifier(settings.brand_registry)

    from collections import defaultdict
    by_file: dict[str, list[str]] = defaultdict(list)
    for row in manifest:
        by_file[row["Filename"]].append(row["Brand"])

    results: list[dict[str, Any]] = []

    for filename, brands in by_file.items():
        pdf_path = settings.input_pdfs_dir / filename

        try:
            pages = parse_pdf(pdf_path)
        except FileNotFoundError:
            log.error("PDF not found: %s — skipping", filename)
            for brand in brands:
                results.append(_error_row(filename, brand, "PDF not found"))
            continue

        doc_structure = extract_document_headings(pdf_path)
        chunks = chunk_pdf_pages(pages, filename, settings.brand_registry, doc_structure=doc_structure)
        if not chunks:
            log.warning("No chunks produced for %s", filename)
            for brand in brands:
                results.append(_error_row(filename, brand, "No chunks extracted"))
            continue

        index = build_index(chunks, document_headings=doc_structure.get('headings', []))

        for brand in brands:
            dbg = DebugLogger(
                filename=filename,
                brand=brand,
                base_dir=settings.intermediate_outputs_dir,
                enabled=settings.save_intermediate,
            )
            log.info("Processing: %s | %s", filename, brand)

            try:
                dbg.step_start("retrieve")
                evidence = retrieve_evidence(brand, settings.brand_registry, index, settings)
                evidence_text = assemble_evidence_text(evidence)
                dbg.step_end("retrieve", {"chunks_retrieved": len(evidence), "evidence_chars": len(evidence_text)})
                dbg.save_chunks(evidence)
                dbg.save_evidence_text(evidence_text)

                dbg.step_start("context_reduce")
                indication = settings.brand_registry.get("brands", {}).get(brand, {}).get(
                    "indication_scope", "plaque psoriasis"
                )
                evidence, evidence_text = reduce_evidence_context(
                    evidence_chunks  = evidence,
                    brand            = brand,
                    indication       = indication,
                    groq_client      = groq,
                    settings         = settings,
                    total_doc_chunks = len(chunks),
                )
                dbg.step_end("context_reduce", {
                    "chunks_after_reduction": len(evidence),
                    "chars_after_reduction":  len(evidence_text),
                })

                dbg.step_start("prompt_build")
                prompt = build_extraction_prompt(
                    filename=filename, brand=brand,
                    evidence_text=evidence_text,
                    parameter_schema=settings.parameter_schema,
                    brand_registry=settings.brand_registry,
                    business_rules=settings.business_rules,
                    reference_examples=settings.reference_examples,
                )
                dbg.step_end("prompt_build", {"prompt_chars": len(prompt)})
                dbg.save_prompt(prompt)

                dbg.step_start("llm_extraction")
                raw_extracted = extract_row(filename, brand, evidence, settings, groq)
                dbg.step_end("llm_extraction", {"keys_returned": list(raw_extracted.keys())})
                dbg.save_extracted_json(raw_extracted)

                dbg.step_start("step_validation")
                validated = validate_step_counts(raw_extracted, classifier)
                dbg.step_end("step_validation")

                dbg.step_start("normalise")
                normalised = normalize_row(validated, brand)
                dbg.step_end("normalise")
                dbg.save_normalised_row(normalised)

                dbg.step_start("score")
                normalised["Access Score"] = str(calculate_access_score(normalised, brand, settings))
                dbg.step_end("score", {"access_score": normalised["Access Score"]})

                dbg.flush(final_row=normalised)
                results.append(normalised)
                log.info("  ✓ %s | %s → Access Score = %s", filename, brand, normalised["Access Score"])

            except Exception as exc:
                dbg.log_step("error", {"message": str(exc)})
                dbg.flush()
                log.error("Error processing %s | %s: %s", filename, brand, exc, exc_info=True)
                results.append(_error_row(filename, brand, str(exc)))

    return results


def _error_row(filename: str, brand: str, reason: str) -> dict[str, Any]:
    """Build an all-NA output row to record a failed extraction in the CSV.
    Ensures the output always has a row for every manifest entry even when
    the pipeline fails for that row, so downstream validation can detect gaps.
    """
    from src.io_utils import RESULT_COLUMNS
    row = {col: "NA" for col in RESULT_COLUMNS}
    row["Filename"] = filename
    row["Brand"] = brand
    row["Access Score"] = "0"
    log.warning("Error row written for %s | %s: %s", filename, brand, reason)
    return row


# ── Single-row helper (notebook) ──────────────────────────────────────────────

def process_single_row(
    filename: str,
    brand: str,
    settings: Any,
    groq: Any | None = None,
) -> dict[str, Any]:
    """
    Process a single row end-to-end with full debug logging.
    Brand must be passed explicitly — this function does not auto-detect.
    The manifest brand always takes precedence; the LLM cannot override it.
    """
    from src.pdf_parser import parse_pdf, extract_document_headings
    from src.chunker import chunk_pdf_pages
    from src.retrieval import build_index, retrieve_evidence, assemble_evidence_text
    from src.context_reducer import reduce_evidence_context
    from src.extractor import extract_row
    from src.normalizer import normalize_row
    from src.step_logic import StepClassifier, validate_step_counts
    from src.scorer import calculate_access_score
    from src.groq_client import GroqClient
    from src.prompts import build_extraction_prompt
    from src.debug_logger import DebugLogger

    if groq is None:
        groq = GroqClient(settings.groq_api_key, settings.allowed_models)

    dbg = DebugLogger(filename=filename, brand=brand,
                      base_dir=settings.intermediate_outputs_dir,
                      enabled=settings.save_intermediate)

    dbg.step_start("parse_pdf")
    pdf_path = settings.input_pdfs_dir / filename
    pages = parse_pdf(pdf_path)
    doc_structure = extract_document_headings(pdf_path)
    dbg.step_end("parse_pdf", {"pages": len(pages), "total_chars": sum(len(p["text"]) for p in pages)})

    dbg.step_start("chunk")
    chunks = chunk_pdf_pages(pages, filename, settings.brand_registry, doc_structure=doc_structure)
    dbg.step_end("chunk", {"total_chunks": len(chunks)})

    dbg.step_start("retrieve")
    index = build_index(chunks, document_headings=doc_structure.get('headings', []))
    evidence = retrieve_evidence(brand, settings.brand_registry, index, settings)
    evidence_text = assemble_evidence_text(evidence)
    dbg.step_end("retrieve", {"chunks_retrieved": len(evidence), "evidence_chars": len(evidence_text)})
    dbg.save_chunks(evidence)
    dbg.save_evidence_text(evidence_text)

    # Context reduction: use Llama-3.1-8B-Instant to filter multi-indication
    # noise from the evidence before passing to the 70B extraction model.
    dbg.step_start("context_reduce")
    indication = settings.brand_registry.get("brands", {}).get(brand, {}).get(
        "indication_scope", "plaque psoriasis"
    )
    evidence, evidence_text = reduce_evidence_context(
        evidence_chunks    = evidence,
        brand              = brand,
        indication         = indication,
        groq_client        = groq,
        settings           = settings,
        total_doc_chunks   = len(chunks),
    )
    dbg.step_end("context_reduce", {
        "chunks_after_reduction": len(evidence),
        "chars_after_reduction": len(evidence_text),
    })

    dbg.step_start("prompt_build")
    prompt = build_extraction_prompt(
        filename=filename, brand=brand,
        evidence_text=evidence_text,
        parameter_schema=settings.parameter_schema,
        brand_registry=settings.brand_registry,
        business_rules=getattr(settings, "business_rules", {}),
            reference_examples=getattr(settings, "reference_examples", {}),
        )
    dbg.step_end("prompt_build", {"prompt_chars": len(prompt)})
    dbg.save_prompt(prompt)

    dbg.step_start("llm_extraction")
    raw_extracted = extract_row(filename, brand, evidence, settings, groq)
    dbg.step_end("llm_extraction")
    dbg.save_extracted_json(raw_extracted)

    dbg.step_start("step_validation")
    classifier = StepClassifier(settings.brand_registry)
    validated = validate_step_counts(raw_extracted, classifier)
    dbg.step_end("step_validation")

    dbg.step_start("normalise")
    normalised = normalize_row(validated, brand)
    dbg.step_end("normalise")
    dbg.save_normalised_row(normalised)

    dbg.step_start("score")
    normalised["Access Score"] = str(calculate_access_score(normalised, brand, settings))
    dbg.step_end("score", {"access_score": normalised["Access Score"]})

    dbg.flush(final_row=normalised)
    return normalised
