"""
pdf_parser.py
Extract page-wise text from payer policy PDFs.

Primary engine    : PyMuPDF (fitz) — fast, accurate for digital PDFs
Fallback engine   : pdfplumber — used when PyMuPDF returns sparse text
Structure engine  : PyMuPDF dict mode — font-size + bold analysis for heading detection
Table enrichment  : pymupdf4llm — replaces raw text on table-heavy pages with markdown
                    (optional; pipeline degrades gracefully if not installed)
"""

from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import TypedDict

log = logging.getLogger(__name__)


class PageText(TypedDict):
    page_num: int       # 1-based
    text: str
    has_table: bool     # True when pymupdf4llm enriched this page with a markdown table


# ── Public API ────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: Path) -> list[PageText]:
    """
    Extract page-wise text.  Tries PyMuPDF first; falls back to pdfplumber
    for scanned/sparse pages.  If pymupdf4llm is installed, table pages are
    enriched with markdown table representation.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = _parse_with_pymupdf(pdf_path)

    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars < 200 and pages:
        log.warning("%s: sparse text (%d chars), trying pdfplumber", pdf_path.name, total_chars)
        pages_fb = _parse_with_pdfplumber(pdf_path)
        if sum(len(p["text"]) for p in pages_fb) > total_chars:
            pages = pages_fb

    # Optional: enrich table pages with pymupdf4llm markdown
    pages = _enrich_tables(pdf_path, pages)

    log.info("%s: %d pages, %d chars", pdf_path.name,
             len(pages), sum(len(p["text"]) for p in pages))
    return pages


def extract_document_headings(pdf_path: Path) -> dict:
    """
    Detect all structural headings in a PDF using font-size and bold analysis
    via PyMuPDF's dict-mode text extraction.  No regex patterns — works for
    any payer document regardless of section naming convention.

    Returns:
        {
            "headings": [{"text", "page_num", "level", "font_size"}, ...],
            "body_font_size": float,
        }

    Heading levels:
        1 = major section title  (font_size >= body * 1.25, e.g. 16pt for body 11.4pt)
        2 = subsection label     (bold, font_size >= body * 0.9, short text ≤ 100 chars)
    """
    pdf_path = Path(pdf_path)
    try:
        import fitz
    except ImportError:
        log.warning("PyMuPDF not available — heading extraction skipped")
        return {"headings": [], "body_font_size": 11.0}

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        log.error("Cannot open %s for heading extraction: %s", pdf_path.name, exc)
        return {"headings": [], "body_font_size": 11.0}

    # ── Pass 1: compute body font size (statistical mode of all character sizes) ──
    from collections import Counter
    size_counts: Counter = Counter()
    for page in doc:
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip():
                        size_counts[round(span["size"], 1)] += len(span["text"])

    if not size_counts:
        doc.close()
        return {"headings": [], "body_font_size": 11.0}

    body_size = size_counts.most_common(1)[0][0]
    h1_threshold = body_size * 1.25   # 25% larger than body → major heading
    h2_size_min  = body_size * 0.90   # same-ish size but bold + short → sub-label

    # ── Pass 2: collect H1 headings only ────────────────────────────────────
    #
    # H2 detection is deliberately omitted.  In PA policy documents, bold
    # text is used for table answer columns ("Yes: Go to #4", 514 occurrences)
    # and decision-tree routing ("No: Pass to RPh.", 514 occurrences) — the
    # same visual style as section labels.  Attempting H2 detection produces
    # hundreds of spurious heading entries that fragment content incorrectly.
    #
    # H1 headings (significantly larger font, ≥1.25× body size) are clean,
    # unambiguous major section boundaries — exactly what we need for
    # section-anchor detection and structural section splitting.
    headings: list[dict] = []
    seen_h1_texts: set[str] = set()     # de-dup exact same H1 on same page

    for page in doc:
        page_num = page.number + 1
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                line_text = "".join(s["text"] for s in spans).strip()
                if not line_text or len(line_text) < 6:
                    continue

                content_spans = [s for s in spans if s["text"].strip()]
                if not content_spans:
                    continue
                max_size = max(s["size"] for s in content_spans)

                if max_size < h1_threshold:
                    continue   # not a major heading

                key = f"{page_num}:{line_text[:60].lower()}"
                if key in seen_h1_texts:
                    continue
                seen_h1_texts.add(key)

                headings.append({
                    "text": line_text,
                    "page_num": page_num,
                    "level": 1,
                    "font_size": round(max_size, 1),
                })

    doc.close()

    log.info("%s: extracted %d H1 headings (body_size=%.1f, h1_threshold=%.1f)",
             pdf_path.name, len(headings), body_size, h1_threshold)
    return {"headings": headings, "body_font_size": body_size}


# ── Table enrichment via pymupdf4llm ──────────────────────────────────────────

def _enrich_tables(pdf_path: Path, pages: list[PageText]) -> list[PageText]:
    """
    Replace raw text on table-heavy pages with pymupdf4llm markdown.
    The markdown table format (|col|col|) is far more readable by the LLM
    than the jumbled whitespace-separated text from PyMuPDF get_text().

    Only applied when:
      - pymupdf4llm is installed
      - The page contains a table block (fitz block type 1 indicates images,
        but we detect tables by checking for grid-like text spans)
    """
    try:
        import pymupdf4llm
        import fitz
    except ImportError:
        return pages    # not installed — return unchanged

    # Skip enrichment for large documents.
    # pymupdf4llm triggers Tesseract OCR and takes 100-170s on 400+ page PDFs.
    # Single-brand PA docs (typically ≤ 50 pages) still get table enrichment.
    # Multi-drug documents rely on regular PyMuPDF text extraction which is
    # accurate enough for our extraction parameters.
    if len(pages) > 50:
        log.debug("pymupdf4llm enrichment skipped: doc has %d pages (> 50 limit)", len(pages))
        return pages

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return pages

    # Identify pages that have tables (fitz detects them as "type 0" blocks
    # where lines are tightly packed — simpler: pages where the raw text
    # has very short irregular lines typical of table cells)
    table_page_indices: set[int] = set()
    for i, page_obj in enumerate(doc):
        for block in page_obj.get_text("dict").get("blocks", []):
            if block.get("type") == 1:      # image block — table might be an image
                table_page_indices.add(i)
            elif block.get("type") == 0:
                # Heuristic: many very short lines = table cells
                lines = block.get("lines", [])
                short_lines = sum(
                    1 for ln in lines
                    if sum(len(s["text"]) for s in ln.get("spans", [])) < 25
                )
                if len(lines) >= 4 and short_lines / max(len(lines), 1) > 0.5:
                    table_page_indices.add(i)

    if not table_page_indices:
        doc.close()
        return pages

    # Convert only table pages to markdown (cheaper than whole doc)
    try:
        import contextlib, io as _io, os as _os
        # pymupdf4llm and Tesseract write to stdout via both Python's sys.stdout
        # AND the underlying C-level file descriptor 1. contextlib.redirect_stdout
        # only captures the Python layer; we also need to redirect fd 1 to /dev/null
        # so Tesseract OCR messages ("=== Document parser messages ===",
        # "Using Tesseract for OCR processing.") don't leak into the notebook.
        _devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
        _saved_fd   = _os.dup(1)
        _os.dup2(_devnull_fd, 1)
        _os.close(_devnull_fd)
        try:
            with contextlib.redirect_stdout(_io.StringIO()):
                md_pages = pymupdf4llm.to_markdown(
                    doc,
                    pages=list(table_page_indices),
                    page_chunks=True,         # returns per-page dicts
                    show_progress=False,
                )
        finally:
            # Always restore fd 1 even if to_markdown raises
            _os.dup2(_saved_fd, 1)
            _os.close(_saved_fd)
    except Exception as exc:
        log.debug("pymupdf4llm enrichment failed: %s", exc)
        doc.close()
        return pages

    doc.close()

    # Build page_num → markdown mapping
    md_map: dict[int, str] = {}
    for chunk in md_pages:
        pn = chunk.get("metadata", {}).get("page", -1) + 1  # 0-based → 1-based
        md_text = chunk.get("text", "")
        if md_text and "|" in md_text:          # only keep if it has table content
            md_map[pn] = clean_text(md_text)

    if not md_map:
        return pages

    enriched: list[PageText] = []
    for p in pages:
        if p["page_num"] in md_map:
            enriched.append({
                "page_num": p["page_num"],
                "text":     md_map[p["page_num"]],
                "has_table": True,
            })
            log.debug("Table enrichment applied to page %d", p["page_num"])
        else:
            enriched.append(p)
    return enriched


# ── PyMuPDF and pdfplumber extractors ────────────────────────────────────────

def _parse_with_pymupdf(pdf_path: Path) -> list[PageText]:
    """Extract pages using PyMuPDF (fitz) — primary parser.
    Returns a list of PageText dicts with page_num, text, has_table, font_sizes.
    Calls _enrich_tables() for small docs (<=50 pages) to add markdown table
    formatting for complex grid tables via pymupdf4llm.
    """
    try:
        import fitz
    except ImportError:
        log.warning("PyMuPDF not installed.")
        return []

    pages: list[PageText] = []
    try:
        doc = fitz.open(str(pdf_path))
        for i, page in enumerate(doc, start=1):
            raw = page.get_text("text") or ""
            pages.append({"page_num": i, "text": clean_text(raw), "has_table": False})
        doc.close()
    except Exception as exc:
        log.error("PyMuPDF failed on %s: %s", pdf_path.name, exc)
    return pages


def _parse_with_pdfplumber(pdf_path: Path) -> list[PageText]:
    """Fallback parser using pdfplumber when PyMuPDF cannot open the PDF.
    Returns the same PageText dict format as _parse_with_pymupdf.
    """
    try:
        import pdfplumber
    except ImportError:
        log.warning("pdfplumber not installed.")
        return []

    pages: list[PageText] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text() or ""
                pages.append({"page_num": i, "text": clean_text(raw), "has_table": False})
    except Exception as exc:
        log.error("pdfplumber failed on %s: %s", pdf_path.name, exc)
    return pages


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise extracted page text: collapse runs of whitespace, strip
    control characters, and normalise unicode dashes and ligatures.
    Called on every page after extraction to reduce BM25 tokenisation noise.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    lines = [line.rstrip() for line in text.split("\n")]
    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                cleaned.append("")
        else:
            blank_run = 0
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def get_full_text(pages: list[PageText]) -> str:
    """Concatenate all pages into a single string with page markers."""
    return "\n\n".join(f"[PAGE {p['page_num']}]\n{p['text']}" for p in pages)
