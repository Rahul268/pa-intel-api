"""
api.py
FastAPI wrapper around the PA extraction pipeline.

Exposes:
  GET  /health            — liveness probe
  POST /api/extract       — upload a PDF + brand, run full pipeline, return JSON

Full pipeline flow per request:
  1. parse_pdf            (PyMuPDF / pdfplumber)
  2. extract_document_headings  (structural heading detection)
  3. chunk_pdf_pages      (section-aware chunking with BM25 metadata)
  4. build_index          (BM25 + RRF retrieval index)
  5. retrieve_evidence    (adaptive top_k, section anchors, section completion)
  6. reduce_evidence_context  (Llama-3.1-8B noise filter for large docs)
  7. extract_row          (Llama-3.3-70B two-call grouped extraction)
  8. validate_step_counts (deterministic step count validation)
  9. normalize_row        (deterministic normalisation)
  10. calculate_access_score  (deterministic 0/25/50/75/100 bucketed score)
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(
    title="PA Intelligence API",
    description="Prior Authorization policy extraction pipeline",
    version="1.0.0",
)

# ── CORS — allow the deployed frontend (and localhost for dev) ───────────────
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "*",  # Set to your frontend URL in production e.g. "https://pa-intel.vercel.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-load settings + Groq client (cached after first request) ────────────
_settings: Any = None
_groq: Any = None


def _get_pipeline_deps():
    """Initialise settings and Groq client once, reuse across requests."""
    global _settings, _groq
    if _settings is None:
        from src.settings import load_settings
        from src.groq_client import GroqClient
        _settings = load_settings()
        _groq = GroqClient(
            api_key=_settings.groq_api_key,
            allowed_models=_settings.allowed_models,
        )
        log.info("Pipeline deps initialised. Model: %s", _settings.model_extraction)
    return _settings, _groq


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe — Render.com calls this to check the service is up."""
    return {"status": "ok", "service": "pa-intelligence-api"}


@app.post("/api/extract")
async def extract(
    file: UploadFile = File(..., description="PA policy PDF document"),
    brand: str = Form(..., description="Brand name e.g. TREMFYA, STELARA"),
):
    """
    Upload a PA policy PDF and run the full extraction pipeline.

    Returns a JSON object with all 12 extracted parameters + Access Score.
    Mirrors the structure of result.csv exactly.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    brand = brand.strip().upper()
    filename = file.filename

    log.info("Extraction request: %s | %s", filename, brand)

    # ── Save upload to a temporary file ──────────────────────────────────────
    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".pdf")
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(pdf_bytes)

        result = _run_pipeline(filename, brand, tmp_path)
        log.info(
            "Extraction complete: %s | %s → Access Score = %s",
            filename, brand, result.get("Access Score"),
        )
        return result

    except HTTPException:
        raise
    except Exception as exc:
        log.error("Pipeline error for %s | %s: %s", filename, brand, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ── Core pipeline execution ───────────────────────────────────────────────────

def _run_pipeline(filename: str, brand: str, pdf_path: Path) -> dict[str, Any]:
    """
    Run all pipeline steps end-to-end against an on-disk PDF file.
    Mirrors process_single_row() in pipeline.py but accepts an arbitrary path
    (not just settings.input_pdfs_dir) so it works with uploaded temp files.
    """
    from src.pdf_parser import parse_pdf, extract_document_headings
    from src.chunker import chunk_pdf_pages
    from src.retrieval import build_index, retrieve_evidence, assemble_evidence_text
    from src.context_reducer import reduce_evidence_context
    from src.extractor import extract_row
    from src.normalizer import normalize_row
    from src.step_logic import StepClassifier, validate_step_counts
    from src.scorer import calculate_access_score

    settings, groq = _get_pipeline_deps()

    # ── Step 1: Parse PDF ─────────────────────────────────────────────────────
    log.info("[1/8] Parsing PDF: %s (%d bytes)", filename, pdf_path.stat().st_size)
    pages = parse_pdf(pdf_path)
    doc_structure = extract_document_headings(pdf_path)
    log.info(
        "      Pages: %d | Chars: %d | Headings: %d",
        len(pages),
        sum(len(p["text"]) for p in pages),
        len(doc_structure.get("headings", [])),
    )

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    log.info("[2/8] Chunking")
    chunks = chunk_pdf_pages(
        pages, filename, settings.brand_registry, doc_structure=doc_structure
    )
    log.info("      Chunks: %d", len(chunks))

    # ── Step 3: Build BM25 index ──────────────────────────────────────────────
    log.info("[3/8] Building retrieval index")
    index = build_index(
        chunks, document_headings=doc_structure.get("headings", [])
    )

    # ── Step 4: Retrieve evidence chunks ─────────────────────────────────────
    log.info("[4/8] Retrieving evidence")
    evidence = retrieve_evidence(brand, settings.brand_registry, index, settings)
    evidence_text = assemble_evidence_text(evidence)
    log.info(
        "      Chunks retrieved: %d | Evidence chars: %d",
        len(evidence), len(evidence_text),
    )

    # ── Step 5: Context reduction (8B noise filter, large docs only) ─────────
    log.info("[5/8] Context reduction")
    indication = (
        settings.brand_registry.get("brands", {})
        .get(brand, {})
        .get("indication_scope", "plaque psoriasis")
    )
    evidence, evidence_text = reduce_evidence_context(
        evidence_chunks=evidence,
        brand=brand,
        indication=indication,
        groq_client=groq,
        settings=settings,
        total_doc_chunks=len(chunks),
    )
    log.info(
        "      After reduction: %d chunks | %d chars",
        len(evidence), len(evidence_text),
    )

    # ── Step 6: LLM extraction (70B two-call grouped) ─────────────────────────
    log.info("[6/8] Running LLM extraction (%s)", settings.model_extraction)
    raw_extracted = extract_row(filename, brand, evidence, settings, groq)

    # ── Step 7: Step count validation ─────────────────────────────────────────
    log.info("[7/8] Validating step counts")
    classifier = StepClassifier(settings.brand_registry)
    validated = validate_step_counts(raw_extracted, classifier)

    # ── Step 8: Normalise + Score ─────────────────────────────────────────────
    log.info("[8/8] Normalising + scoring")
    normalised = normalize_row(validated, brand)
    score = calculate_access_score(normalised, brand, settings)
    normalised["Access Score"] = str(score)
    normalised["Filename"] = filename
    normalised["Brand"] = brand

    return normalised
