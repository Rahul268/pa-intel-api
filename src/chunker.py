"""
chunker.py
Split page-wise PDF text into section-aware chunks with metadata.

Heading detection strategy (in priority order):
  1. Structural — uses heading positions from pdf_parser.extract_document_headings()
     (font-size + bold analysis). Heading text determines chunk_type only when
     it is structurally a heading — NOT on every body-text line.
  2. Pattern fallback — original SECTION_PATTERNS regex, used only when
     doc_structure is absent or produced no headings.

This eliminates the false-positive section splits where any line containing
"diagnosis" or "coverage criteria" would incorrectly start a new section.
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ── Fallback patterns (used only when structural headings unavailable) ─────────
SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)(general\s+criteria|universal\s+criteria|criteria\s+for\s+all)",         "universal_criteria"),
    (r"(?i)(coverage\s+criteria|coverage\s+requirements|prior\s+authorization\s+criteria|pa\s+criteria)", "coverage_criteria"),
    (r"(?i)(initial\s+authorization|initial\s+approval\s+criteria|initial\s+coverage)", "initial_auth"),
    (r"(?i)(step\s+therapy|step-through|prior\s+treatment|preferred\s+drug)",        "step_therapy"),
    (r"(?i)(continuation|reauthorization|re-?authorization|renewal\s+criteria|renewal\s+requirements)", "reauthorization"),
    (r"(?i)(quantity\s+limit|quantity\s+level\s+limit|ql\s+limit|supply\s+limit)",  "quantity_limits"),
    (r"(?i)(age\s+requirement|age\s+eligibility|age\s+criteria|patient\s+population)", "age_criteria"),
    (r"(?i)(prescriber\s+requirement|specialist\s+requirement|prescribing\s+physician)", "prescriber"),
    (r"(?i)(tuberculosis|tb\s+test|tb\s+screening|latent\s+tb)",                    "tb_testing"),
    (r"(?i)(diagnosis|diagnostic\s+criteria|disease\s+criteria|indication)",         "diagnosis"),
    (r"(?i)(plaque\s+psoriasis|moderate.{0,10}severe|psoriasis\s+criteria)",         "indication_specific"),
    (r"(?i)(background|overview|policy\s+overview|description|scope)",               "background"),
    (r"(?i)(exclusions?|non.?covered|not\s+covered)",                                "exclusions"),
]

INDICATION_KEYWORDS = [
    "plaque psoriasis", "psoriasis", "pso", "moderate-to-severe",
    "moderate to severe", "bsa", "body surface area",
]


@dataclass
class Chunk:
    chunk_id:        str
    filename:        str
    page_num:        int
    section_title:   str
    chunk_type:      str
    text:            str
    char_start:      int
    char_end:        int
    detected_brands: list[str] = field(default_factory=list)
    has_indication:  bool = False
    metadata:        dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the chunk to a plain dict for JSONL audit logging."""
        return {
            "chunk_id":        self.chunk_id,
            "filename":        self.filename,
            "page_num":        self.page_num,
            "section_title":   self.section_title,
            "chunk_type":      self.chunk_type,
            "text":            self.text,
            "detected_brands": self.detected_brands,
            "has_indication":  self.has_indication,
        }


# ── Public API ─────────────────────────────────────────────────────────────────

def chunk_pdf_pages(
    pages: list[dict],
    filename: str,
    brand_registry: dict,
    doc_structure: dict | None = None,
    max_chunk_chars: int = 1500,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """
    Convert page-wise text into section-aware chunks.

    doc_structure: optional output of pdf_parser.extract_document_headings().
                   When present and non-empty, structural heading detection is
                   used instead of SECTION_PATTERNS.
    """
    all_brands_lower = _build_brand_alias_map(brand_registry)
    chunks: list[Chunk] = []
    chunk_counter = 0

    # Build page_num → headings mapping
    page_headings: dict[int, list[dict]] = {}
    if doc_structure:
        for h in doc_structure.get("headings", []):
            page_headings.setdefault(h["page_num"], []).append(h)

    use_structural = bool(page_headings)
    log.debug("%s: chunking with %s detection (%d heading pages)",
              filename,
              "structural" if use_structural else "pattern-fallback",
              len(page_headings))

    for page in pages:
        page_num = page["page_num"]
        text     = page["text"]
        if not text.strip():
            continue

        if use_structural:
            sections = _split_into_sections_structural(text, page_headings.get(page_num, []))
        else:
            sections = _split_into_sections_pattern(text)

        for sec_title, sec_type, sec_text in sections:
            if not sec_text.strip():
                continue

            sub_texts = _split_on_paragraphs(sec_text, max_chunk_chars, overlap_chars)
            for sub_text in sub_texts:
                if not sub_text.strip():
                    continue

                chunk_counter += 1
                chunk_id  = f"{filename}__p{page_num}__c{chunk_counter:04d}"
                detected  = _detect_brands_in_text(sub_text, all_brands_lower)
                has_ind   = _has_indication_keywords(sub_text)

                chunks.append(Chunk(
                    chunk_id        = chunk_id,
                    filename        = filename,
                    page_num        = page_num,
                    section_title   = sec_title,
                    chunk_type      = sec_type,
                    text            = sub_text.strip(),
                    char_start      = 0,
                    char_end        = len(sub_text),
                    detected_brands = detected,
                    has_indication  = has_ind,
                ))

    log.debug("%s: %d chunks from %d pages (structural=%s)",
              filename, len(chunks), len(pages), use_structural)
    return chunks


# ── Structural heading-based splitting ────────────────────────────────────────

def _split_into_sections_structural(
    text: str,
    page_headings: list[dict],
) -> list[tuple[str, str, str]]:
    """
    Split page text at H1 heading positions (major section boundaries from
    font-size analysis), then apply SECTION_PATTERNS within each H1 section
    for fine-grained chunk-type labelling.

    This combines structural precision (H1 = unambiguous major boundary) with
    pattern reliability (vocabulary-based typing for step-therapy, reauth, etc.).

    Pages without H1 headings fall through to pure SECTION_PATTERNS.
    """
    if not page_headings:
        # No structural headings on this page — use pattern-based detection
        return _split_into_sections_pattern(text)

    # Locate each H1 heading in the raw page text
    split_points: list[tuple[int, str]] = []  # (char_offset, heading_title)
    for heading in page_headings:
        h_text = heading["text"].strip()
        idx = text.find(h_text)
        if idx == -1:
            idx = text.find(h_text[:50])
        if idx >= 0:
            split_points.append((idx, h_text[:120]))

    if not split_points:
        # H1s detected but can't be located in raw text — fall back to patterns
        return _split_into_sections_pattern(text)

    split_points.sort(key=lambda x: x[0])

    h1_segments: list[tuple[str, str]] = []  # (h1_title, segment_text)

    # Text before the first H1
    if split_points[0][0] > 0:
        pre = text[: split_points[0][0]].strip()
        if pre:
            h1_segments.append(("General", pre))

    for i, (pos, title) in enumerate(split_points):
        end     = split_points[i + 1][0] if i + 1 < len(split_points) else len(text)
        seg_txt = text[pos:end].strip()
        if seg_txt:
            h1_segments.append((title, seg_txt))

    # Within each H1 segment, apply SECTION_PATTERNS for chunk-type labelling
    final: list[tuple[str, str, str]] = []
    for h1_title, seg_text in h1_segments:
        sub_sections = _split_into_sections_pattern(seg_text)
        for sub_title, sub_type, sub_text in sub_sections:
            # Use the H1 title as section_title when the sub-section has no
            # meaningful label of its own (i.e., it defaulted to "General")
            effective_title = sub_title if sub_title != "General" else h1_title
            final.append((effective_title, sub_type, sub_text))

    return final or _split_into_sections_pattern(text)


def _heading_to_chunk_type(heading_text: str) -> str:
    """
    Assign chunk_type from a structurally confirmed heading line.
    Applied ONLY to lines identified as headings by font analysis —
    never to arbitrary body-text lines.
    """
    t = heading_text.lower()
    if re.search(r"step\s+therapy|step-through|prior\s+treatment|preferred\s+drug", t):
        return "step_therapy"
    if re.search(r"re-?auth|renewal\s+crit|renewal\s+req|continuation\s+crit", t):
        return "reauthorization"
    if re.search(r"quantity\s+limit|qty\s+limit|ql\s+limit|supply\s+limit", t):
        return "quantity_limits"
    if re.search(r"tuberculosis|tb\s+test|tb\s+screen|latent\s+tb", t):
        return "tb_testing"
    if re.search(r"initial\s+auth|initial\s+approv|initial\s+cov", t):
        return "initial_auth"
    if re.search(r"age\s+req|age\s+elig|age\s+crit|patient\s+population", t):
        return "age_criteria"
    if re.search(r"prescrib|specialist\s+req", t):
        return "prescriber"
    if re.search(r"plaque\s+psoriasis|moderate.{0,10}severe\s+pso", t):
        return "indication_specific"
    if re.search(r"targeted\s+immune|biologic|immune\s+modulator", t):
        return "coverage_criteria"
    if re.search(r"coverage\s+crit|approval\s+crit|prior\s+auth|pa\s+crit", t):
        return "coverage_criteria"
    if re.search(r"background|overview|policy\s+overview|description", t):
        return "background"
    if re.search(r"exclusion|not\s+covered|non.?covered", t):
        return "exclusions"
    if re.search(r"diagnosis|diagnostic|indication", t):
        return "diagnosis"
    return "general"


# ── Pattern-based splitting (fallback) ────────────────────────────────────────

def _split_into_sections_pattern(text: str) -> list[tuple[str, str, str]]:
    """
    Original line-by-line regex detection.  Used only when structural
    headings are unavailable.
    """
    lines = text.split("\n")
    sections: list[tuple[str, str, str]] = []
    current_title = "General"
    current_type  = "general"
    current_lines: list[str] = []

    for line in lines:
        matched = False
        for pattern, ctype in SECTION_PATTERNS:
            if re.search(pattern, line):
                if current_lines:
                    sections.append((current_title, current_type,
                                     "\n".join(current_lines)))
                current_title = line.strip()[:120]
                current_type  = ctype
                current_lines = [line]
                matched = True
                break
        if not matched:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_type, "\n".join(current_lines)))

    return sections


# ── Paragraph splitting with overlap ──────────────────────────────────────────

def _split_on_paragraphs(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split long sections at paragraph breaks, with optional tail overlap."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n{2,}", text)
    result: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                result.append(current)
            tail    = current[-overlap:].strip() if overlap > 0 and current else ""
            current = (tail + "\n\n" + para).strip() if tail else para

    if current:
        result.append(current)

    return result or [text]


# ── Brand and indication helpers ──────────────────────────────────────────────

def _build_brand_alias_map(brand_registry: dict) -> dict[str, str]:
    """Build a lowercase alias → canonical brand name lookup from brand_registry.
    Includes brand display names, generic names, and any additional aliases.
    Used for fast O(n) brand presence detection during chunking.
    """
    mapping: dict[str, str] = {}
    for brand_name, info in brand_registry.get("brands", {}).items():
        for alias in info.get("aliases", [brand_name]):
            mapping[alias.lower()] = brand_name
        for gname in info.get("generic_names", []):
            mapping[gname.lower()] = brand_name
    return mapping


def _detect_brands_in_text(text: str, alias_map: dict[str, str]) -> list[str]:
    """Return all canonical brand names mentioned in a text block.
    Used to tag chunks with detected_brands for retrieval scoring.
    """
    text_lower = text.lower()
    found: set[str] = set()
    for alias, brand_name in alias_map.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", text_lower):
            found.add(brand_name)
    return sorted(found)


def _has_indication_keywords(text: str) -> bool:
    """Check if a text block contains plaque psoriasis indication keywords.
    Chunks flagged has_indication=True get a scoring boost during retrieval.
    """
    tl = text.lower()
    return any(kw in tl for kw in INDICATION_KEYWORDS)
