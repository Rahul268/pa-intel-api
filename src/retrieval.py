"""
retrieval.py
BM25 retrieval with Reciprocal Rank Fusion and auto-generated section anchors.

Key design points:

RRF (Reciprocal Rank Fusion)
  Replaces the previous max-score _upsert strategy.  A chunk scoring
  consistently across many query passes ranks higher than one that wins
  a single pass.  Formula: rrf(chunk) = Σ 1 / (k + rank_i), k=60.

Auto-generated section anchors
  extract_document_headings() in pdf_parser returns structurally detected
  headings.  find_section_anchor_page() runs BM25 over those heading texts
  to find the best-matching section for the current brand + indication —
  no manually curated section_anchor_keywords needed.
  Falls back to manual keywords when headings are unavailable.

Adaptive top_k (unchanged from previous version)
  Scales top_k_brand / top_k_param with document chunk count so large
  multi-drug documents aren't starved.

Section-completion (unchanged from previous version)
  For large documents, all chunks within the anchor page's forward window
  are added to the candidate pool.
"""

from __future__ import annotations
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

STEP_SECTION_TYPES = {"step_therapy", "coverage_criteria", "universal_criteria", "initial_auth"}
AUTH_SECTION_TYPES = {"initial_auth", "reauthorization", "coverage_criteria"}
QTY_SECTION_TYPES  = {"quantity_limits"}

UNIVERSAL_PHRASES = [
    "all products", "all indications", "all drugs", "general criteria",
    "universal criteria", "documentation for all",
]

LARGE_DOC_THRESHOLD = 500
_RRF_K = 60          # standard RRF constant


# ── RetrievalIndex ─────────────────────────────────────────────────────────────

class RetrievalIndex:
    """BM25 index over chunks, with optional document heading list for auto-anchoring."""

    def __init__(self, chunks: list[Any], document_headings: list[dict] | None = None) -> None:
        self.chunks            = chunks
        self.document_headings = document_headings or []
        self._bm25             = None
        self._tokenized        = None
        self._build(chunks)

    def _build(self, chunks: list[Any]) -> None:
        """Tokenise all chunks and build the BM25 index.
        Stores the raw token lists for RRF rescoring.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 is required: pip install rank-bm25")
        self._tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25      = BM25Okapi(self._tokenized)
        log.debug("BM25 index built over %d chunks", len(chunks))


def build_index(
    chunks: list[Any],
    document_headings: list[dict] | None = None,
) -> RetrievalIndex:
    """Build a retrieval index, optionally storing document headings for auto-anchoring."""
    return RetrievalIndex(chunks, document_headings=document_headings)


# ── Adaptive top_k ─────────────────────────────────────────────────────────────

def _adaptive_top_k(num_chunks: int, base_brand: int, base_param: int) -> tuple[int, int]:
    """Scale BM25 top-k per query pass based on document size.
    Large documents (>500 chunks, e.g. Oregon Medicaid 441-page PDF) need a
    higher top-k to ensure all relevant sections reach the RRF merge step.
    Returns (brand_top_k, param_top_k) as a tuple.
    """
    if num_chunks >= LARGE_DOC_THRESHOLD:
        return base_brand * 3, base_param * 3
    if num_chunks >= 200:
        return base_brand * 2, base_param * 2
    return base_brand, base_param


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def _rrf_combine(
    ranked_lists: list[list[tuple[Any, float]]],
    k: int = _RRF_K,
) -> dict[str, tuple[Any, float]]:
    """
    Combine multiple BM25 ranked lists using RRF.
    Each input list is [(chunk, bm25_score), ...] sorted descending by score.
    Returns {chunk_id: (chunk, rrf_score)}.
    """
    rrf: dict[str, list] = {}   # chunk_id → [chunk, cumulative_rrf]
    for ranked_list in ranked_lists:
        for rank, (chunk, _bm25_score) in enumerate(ranked_list, start=1):
            cid = chunk.chunk_id
            contribution = 1.0 / (k + rank)
            if cid not in rrf:
                rrf[cid] = [chunk, 0.0]
            rrf[cid][1] += contribution
    return {cid: (c, s) for cid, (c, s) in rrf.items()}


# ── Auto section anchor ────────────────────────────────────────────────────────

def find_section_anchor_page(
    brand: str,
    indication: str,
    document_headings: list[dict],
    all_chunks: list | None = None,
    brand_aliases: list[str] | None = None,
) -> int | None:
    """
    Find the page number of the most relevant H1 section for this brand.

    Three strategies in order:
    1. Direct name match — brand/alias appears in an H1 heading text.
    2. Brand-mention count — the H1 section whose page range contains the most
       brand mentions in chunks.  Handles multi-drug documents where the section
       heading describes a drug class, not a specific drug.
    3. BM25 fallback over heading texts.
    """
    if not document_headings:
        return None

    h1 = sorted(
        [h for h in document_headings if h.get("level", 1) == 1],
        key=lambda h: h["page_num"],
    )
    if not h1:
        return None

    aliases_lower = {a.lower() for a in (brand_aliases or [])} | {brand.lower()}

    # Strategy 1: brand/alias directly in heading text
    for h in h1:
        if any(a in h["text"].lower() for a in aliases_lower):
            log.info("Auto-anchor S1 (name-match): %s → pg %d ('%s')",
                     brand, h["page_num"], h["text"][:60])
            return h["page_num"]

    # Strategy 2: count brand mentions per H1 section
    if all_chunks:
        brand_pages: set[int] = set()
        for c in all_chunks:
            if any(a in c.text.lower() for a in aliases_lower):
                brand_pages.add(c.page_num)

        if brand_pages:
            best_pg, best_count = None, 0
            for i, h in enumerate(h1):
                start_pg = h["page_num"]
                end_pg   = h1[i + 1]["page_num"] if i + 1 < len(h1) else start_pg + 60
                count    = len(brand_pages & set(range(start_pg, end_pg)))
                if count > best_count:
                    best_count, best_pg = count, start_pg
            if best_pg is not None:
                heading_text = next((h["text"] for h in h1 if h["page_num"] == best_pg), "?")
                log.info("Auto-anchor S2 (brand-mentions): %s → pg %d ('%s', %d pages)",
                         brand, best_pg, heading_text[:60], best_count)
                return best_pg

    # Strategy 3: BM25 over heading texts
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return None

    query_tokens   = _tokenize(f"{brand} {indication}")
    heading_tokens = [_tokenize(h["text"]) for h in h1]
    if not any(heading_tokens):
        return None

    bm25   = BM25Okapi(heading_tokens)
    scores = bm25.get_scores(query_tokens)
    if float(max(scores)) <= 0:
        return None

    best_idx = int(scores.argmax())
    log.info("Auto-anchor S3 (BM25): %s → pg %d ('%s')",
             brand, h1[best_idx]["page_num"], h1[best_idx]["text"][:60])
    return h1[best_idx]["page_num"]


def _section_completion(
    candidate_pool: list[Any],
    all_chunks: list[Any],
    anchor_pages: list[int],
    backward_window: int = 3,
    forward_window: int = 9,
) -> list[Any]:
    """Guarantee representation of every page in a window around the anchor.
    BM25 may score an adjacent page slightly lower even when it contains
    critical content (e.g. pg 375 age table next to pg 374 section header).
    Adds any chunk from pages [anchor-window, anchor+window] not already
    in the retrieved set, ensuring section continuity.
    """
    page_to_chunks: dict[int, list[Any]] = {}
    for c in all_chunks:
        page_to_chunks.setdefault(c.page_num, []).append(c)

    existing_ids = {id(c) for c in candidate_pool}
    extra: list[Any] = []

    for anchor_pg in anchor_pages[:3]:
        for offset in range(-backward_window, forward_window + 1):
            for c in page_to_chunks.get(anchor_pg + offset, []):
                if id(c) not in existing_ids:
                    existing_ids.add(id(c))
                    extra.append(c)

    if extra:
        log.info("Section-completion: +%d chunks from anchor %s (window -%d/+%d)",
                 len(extra), anchor_pages[:3], backward_window, forward_window)
    return candidate_pool + extra


# ── Main retrieval function ────────────────────────────────────────────────────

def retrieve_evidence(
    brand: str,
    brand_registry: dict,
    index: RetrievalIndex,
    settings: Any,
) -> list[Any]:
    """
    Retrieve the most relevant chunks for a given brand row using RRF.

    Query passes (each produces a ranked list fed into RRF):
      1. Universal criteria phrases  (apply to all brands/indications)
      2. Brand query — indication-scoped for manifest brands
      3. Section anchor — auto-generated from structural headings
      4. Indication pass
      5–12. One pass per parameter cluster (step, auth, reauth, qty, TB,
             specialist, age, age-appendix)

    Post-RRF boosts applied per chunk:
      brand name present in text  → +0.050
      universal phrase match      → +0.030
      has_indication flag         → +0.020
      step/coverage section type  → +0.015
      auth section type           → +0.010
      quantity section type       → +0.010
    """
    brand_info  = brand_registry.get("brands", {}).get(brand, {})
    brand_kws   = brand_info.get("retrieval_keywords", [brand.lower()])
    num_chunks  = len(index.chunks)
    is_large    = num_chunks >= LARGE_DOC_THRESHOLD

    top_k_brand, top_k_param = _adaptive_top_k(
        num_chunks, settings.top_k_brand, settings.top_k_param
    )
    log.info("Retrieval: brand=%s | chunks=%d | large=%s | top_k=%d/%d",
             brand, num_chunks, is_large, top_k_brand, top_k_param)

    # Biosimilar reference brand
    ref_brand = brand_info.get("reference_brand")
    if ref_brand:
        ref_kws   = brand_registry.get("brands", {}).get(ref_brand, {}).get("retrieval_keywords", [])
        brand_kws = list(dict.fromkeys(brand_kws + ref_kws))

    indication_scope = brand_info.get("indication_scope", "")
    indication_suffix = f" {indication_scope}" if indication_scope else ""

    # ── Collect ranked lists for RRF ──────────────────────────────────────────
    ranked_lists: list[list[tuple[Any, float]]] = []

    # 1. Universal criteria
    for phrase in UNIVERSAL_PHRASES:
        rl = _score_chunks(phrase, index, boost_universal=True)
        if rl:
            ranked_lists.append(rl)

    # 2. Brand query (indication-scoped for manifest brands)
    brand_query_base = " ".join(brand_kws[:8])
    brand_query = f"{brand_query_base}{indication_suffix}" if indication_scope else brand_query_base
    brand_rl = _score_chunks(brand_query, index, top_k=top_k_brand)
    ranked_lists.append(brand_rl)
    brand_anchor_pages = [c.page_num for c, _ in brand_rl[:5]]


    # 3. Section anchor resolution (priority order):
    #    a. Manual section_anchor_keywords — proven reliable for known brands.
    #       Handles multi-drug documents where the section heading describes a
    #       drug class ("Targeted Immune Modulators") not the specific brand.
    #    b. Auto-anchor from structural headings — fallback for unknown brands.
    section_anchor_page: int | None = None

    manual_kw = brand_info.get("section_anchor_keywords", "")
    if manual_kw:
        sa_rl = _score_chunks(manual_kw, index, top_k=top_k_brand)
        ranked_lists.append(sa_rl)
        if sa_rl:
            section_anchor_page = sa_rl[0][0].page_num
            log.info("Manual anchor: page %d (kw: '%s')", section_anchor_page, manual_kw[:40])

    if section_anchor_page is None and index.document_headings:
        # Auto-anchor for brands without manual section_anchor_keywords
        brand_specific_aliases = brand_info.get("aliases", [brand.lower()])
        section_anchor_page = find_section_anchor_page(
            brand,
            indication_scope,
            index.document_headings,
            all_chunks=index.chunks,
            brand_aliases=brand_specific_aliases,
        )

    # 4. Indication pass
    indication_query = (
        f"{indication_scope} moderate severe prior authorization"
        if indication_scope
        else "plaque psoriasis moderate severe prior authorization"
    )
    ranked_lists.append(_score_chunks(indication_query, index, top_k=top_k_brand))

    # 5–12. Parameter cluster passes
    param_queries = {
        "step_therapy":  f"step therapy trial failure inadequate response biologic prior treatment{indication_suffix}",
        "authorization": f"initial authorization duration approval period months coverage granted{indication_suffix}",
        "reauth":        f"reauthorization continuation renewal criteria clinical response{indication_suffix}",
        "quantity":      "quantity limit quantity level limit vials syringes per days",
        "tb":            "tuberculosis tb test tb screening latent tb quantiferon ppd",
        "specialist":    "dermatologist rheumatologist specialist prescriber prescribing physician",
        "age":           "years of age adult pediatric minimum age eligibility",
        "age_appendix":  "FDA approved dosing appendix minimum age years older indication table",
    }
    for query in param_queries.values():
        rl = _score_chunks(query, index, top_k=top_k_param)
        if rl:
            ranked_lists.append(rl)

    # ── RRF combination ────────────────────────────────────────────────────────
    rrf_scored = _rrf_combine(ranked_lists)

    # ── Post-RRF boosts ────────────────────────────────────────────────────────
    brand_lower = brand.lower()
    boosted: list[tuple[Any, float]] = []
    for chunk, rrf_score in rrf_scored.values():
        boost = 0.0
        if brand_lower in chunk.text.lower():
            boost += 0.050
        if chunk.has_indication:
            boost += 0.020
        if chunk.chunk_type in STEP_SECTION_TYPES:
            boost += 0.015
        if chunk.chunk_type in AUTH_SECTION_TYPES:
            boost += 0.010
        if chunk.chunk_type in QTY_SECTION_TYPES:
            boost += 0.010
        boosted.append((chunk, rrf_score + boost))

    boosted.sort(key=lambda x: x[1], reverse=True)
    candidate_list = [c for c, _ in boosted]

    # ── Section-completion for large documents ─────────────────────────────────
    if is_large and section_anchor_page is not None:
        candidate_list = _section_completion(
            candidate_list, index.chunks,
            [section_anchor_page],
            backward_window=3, forward_window=9,
        )

    # ── Character budget allocation ────────────────────────────────────────────
    max_chars = settings.max_evidence_chars
    if section_anchor_page is not None and is_large:
        # Forward window only — backward pages are pre-section metadata
        forward_win = 10
        section_window_pages = set(range(section_anchor_page, section_anchor_page + forward_win + 1))
    else:
        section_window_pages = set(brand_anchor_pages)

    priority = [c for c in candidate_list if c.page_num in section_window_pages]
    rest     = [c for c in candidate_list if c.page_num not in section_window_pages]

    MIN_CHUNK_CHARS = 60
    selected: list[Any] = []
    total_chars = 0
    seen_in_selected: set[int] = set()

    for c in priority + rest:
        if id(c) in seen_in_selected:
            continue
        if len(c.text) < MIN_CHUNK_CHARS:
            continue
        if total_chars + len(c.text) > max_chars:
            continue
        selected.append(c)
        seen_in_selected.add(id(c))
        total_chars += len(c.text)
        if total_chars >= max_chars:
            break

    log.info("Retrieved %d chunks (%d chars) for brand=%s (large=%s)",
             len(selected), total_chars, brand, is_large)
    return selected


# ── Evidence text assembly ─────────────────────────────────────────────────────

def assemble_evidence_text(chunks: list[Any]) -> str:
    """Join retrieved chunks into a single evidence string for the LLM.
    Each chunk is prefixed with its section title, page number, and type so
    the model can orient itself within the document structure.
    """
    parts = []
    for c in chunks:
        header = f"[Section: {c.section_title} | Page {c.page_num} | Type: {c.chunk_type}]"
        parts.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(parts)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into tokens for BM25 scoring.
    Strips punctuation and removes single-character tokens to reduce noise.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    return text.split()


def _score_chunks(
    query: str,
    index: RetrievalIndex,
    top_k: int = 10,
    boost_universal: bool = False,
) -> list[tuple[Any, float]]:
    """Return [(chunk, bm25_score), ...] for the top_k matches."""
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = index._bm25.get_scores(tokens)
    pairs  = sorted(zip(index.chunks, scores), key=lambda x: x[1], reverse=True)
    if not boost_universal:
        pairs = [(c, s) for c, s in pairs if s > 0]
    return pairs[:top_k]
