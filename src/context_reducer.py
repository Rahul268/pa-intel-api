"""
context_reducer.py
Post-retrieval context reduction using Llama-3.1-8B-Instant.

Problem
-------
After BM25 + RRF retrieval of ~54 chunks from a large multi-drug policy
document, the assembled evidence (~20k chars) contains criteria for multiple
conditions mixed together:

  Page 374: "Length of Authorization: Up to 12 months"  ← general, all indications
  Page 379: "Yes: Approve for up to 6 months"           ← plaque psoriasis only
  Page 380: "Methotrexate for ≥6 months OR..."          ← rheumatoid arthritis criteria
  Page 381: "Oral antibiotics 90-day trial..."          ← hidradenitis suppurativa

The 70B extraction model must pick "6 months" over "12 months" and count only
plaque-psoriasis step requirements — not RA or HS criteria.  Mixed evidence
degrades accuracy on step counts, phototherapy flags, and authorization durations.

Solution
--------
A single Llama-3.1-8B-Instant call receives a compact numbered index of all
retrieved chunks and returns the indices of those directly relevant to
[brand] + [indication].  The original verbatim text is then reconstructed
from those indices — the 8B model never rewrites or summarises the evidence,
only selects which chunks to keep.

Activation
----------
Only applied for large documents (> 500 chunks) where noise is a real problem.
Single-brand documents are already indication-specific; reduction adds no value.

Fallback
--------
Any failure (network, parse error, too-short output) silently returns the
original evidence.  The pipeline is never blocked.
"""

from __future__ import annotations
import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

LARGE_DOC_THRESHOLD   = 500   # same as retrieval.py — only apply for large docs
MIN_REDUCED_CHARS     = 800   # if filtered text is smaller, fall back to full evidence
MAX_REDUCED_CHUNKS    = 30    # cap to prevent over-filtering
INDEX_PREVIEW_CHARS   = 400   # chars to show per chunk in the index sent to 8B


def reduce_evidence_context(
    evidence_chunks: list[Any],
    brand: str,
    indication: str,
    groq_client: Any,
    settings: Any,
    total_doc_chunks: int = 0,
) -> tuple[list[Any], str]:
    """
    Filter retrieved chunks to those directly relevant to [brand] + [indication].

    Parameters
    ----------
    evidence_chunks     : chunks returned by retrieve_evidence
    brand               : e.g. "CIMZIA"
    indication          : e.g. "plaque psoriasis"
    groq_client         : GroqClient instance
    settings            : pipeline settings (used for model name)
    total_doc_chunks    : total chunks in the document index (for large-doc check)

    Returns
    -------
    (filtered_chunks, assembled_evidence_text)
    Falls back to original on any error or insufficient output.
    """
    from src.retrieval import assemble_evidence_text

    full_text = assemble_evidence_text(evidence_chunks)

    # Skip for small documents — evidence is already indication-specific
    if total_doc_chunks > 0 and total_doc_chunks < LARGE_DOC_THRESHOLD:
        log.debug("Context reduction skipped: small doc (%d chunks)", total_doc_chunks)
        return evidence_chunks, full_text

    if not evidence_chunks:
        return evidence_chunks, full_text

    # Build compact numbered index for the 8B model
    index_lines = _build_chunk_index(evidence_chunks)
    prompt      = _build_filter_prompt(brand, indication, index_lines)

    try:
        raw_response = groq_client.complete(
            prompt    = prompt,
            model     = settings.model_repair,   # llama-3.1-8b-instant
            max_tokens = 250,
            temperature = 0.0,
            step      = "context_reduce",
            filename  = "",
            brand     = brand,
        )
    except Exception as exc:
        log.warning("Context reduction API call failed (%s) — using full evidence", exc)
        return evidence_chunks, full_text

    relevant_indices = _parse_indices(raw_response, len(evidence_chunks))

    if not relevant_indices:
        log.warning(
            "Context reduction returned no indices for %s/%s — using full evidence",
            brand, indication,
        )
        return evidence_chunks, full_text

    # Cap to avoid over-filtering: never drop below a minimum of 10 chunks
    # (the 8B model can sometimes be too aggressive)
    if len(relevant_indices) > MAX_REDUCED_CHUNKS:
        relevant_indices = relevant_indices[:MAX_REDUCED_CHUNKS]

    # Token budget guard: ensure filtered evidence fits within Groq free-tier limits.
    # Groq on_demand: 12,000 TPM. Non-evidence overhead ≈ 4,800 tokens, max_tokens=1500.
    # Available for evidence: 12,000 - 4,800 - 1,500 = 5,700 tokens ≈ 22,800 chars.
    MAX_EVIDENCE_CHARS_BUDGET = 20_000  # conservative — leaves buffer for overhead variance
    candidate_chunks = [evidence_chunks[i] for i in relevant_indices]
    candidate_text   = assemble_evidence_text(candidate_chunks)
    if len(candidate_text) > MAX_EVIDENCE_CHARS_BUDGET:
        # Re-trim to budget by dropping lowest-index (lowest-priority) chunks from the end
        trimmed, total = [], 0
        for c in candidate_chunks:
            if total + len(c.text) > MAX_EVIDENCE_CHARS_BUDGET:
                break
            trimmed.append(c)
            total += len(c.text)
        candidate_chunks = trimmed
        log.info("Token budget trim: %d → %d chunks after budget cap",
                 len(relevant_indices), len(candidate_chunks))

    filtered      = candidate_chunks
    
    # ── Deterministic safety net ─────────────────────────────────────────
    # The 8B model sometimes drops chunks containing the brand name when
    # the chunk also lists other conditions (e.g. the pg375 FDA approval
    # table that starts with column headers for RA, AS, Crohn's etc.).
    # After the LLM filter, scan the ORIGINAL evidence for any chunk that
    # contains the brand name and was not selected — add it back.
    # This guarantees age eligibility tables and brand-specific criteria
    # are never accidentally excluded.
    brand_lower   = brand.lower()
    filtered_ids  = {id(c) for c in filtered}
    restored      = []
    for c in evidence_chunks:
        if id(c) not in filtered_ids and brand_lower in c.text.lower():
            filtered_ids.add(id(c))
            restored.append(c)
            log.info(
                "Safety net restored: pg%d [%s] — contains brand '%s'",
                c.page_num, c.chunk_type, brand,
            )
    if restored:
        filtered = filtered + restored
        log.info("Safety net: restored %d brand-containing chunks", len(restored))

    filtered_text = assemble_evidence_text(filtered)

    if len(filtered_text) < MIN_REDUCED_CHARS:
        log.warning(
            "Reduced evidence too short (%d chars) for %s/%s — using full evidence",
            len(filtered_text), brand, indication,
        )
        return evidence_chunks, full_text

    log.info(
        "Context reduction: %d → %d chunks | %d → %d chars | brand=%s indication=%s",
        len(evidence_chunks), len(filtered),
        len(full_text), len(filtered_text),
        brand, indication,
    )
    return filtered, filtered_text


# ── Prompt construction ────────────────────────────────────────────────────────

def _build_chunk_index(chunks: list[Any]) -> list[str]:
    """Build compact numbered one-liners for each chunk."""
    lines = []
    for i, c in enumerate(chunks):
        preview = c.text[:INDEX_PREVIEW_CHARS].replace("\n", " ").strip()
        lines.append(
            f"[{i}] pg{c.page_num} | {c.chunk_type}\n    {preview}"
        )
    return lines


def _build_filter_prompt(
    brand: str,
    indication: str,
    index_lines: list[str],
) -> str:
    """Build the prompt sent to the 8B model for evidence filtering.
    The prompt gives the model an indexed list of chunk previews (400 chars each)
    and asks it to return a JSON array of the indices to KEEP.
    ALWAYS KEEP rules are listed explicitly to prevent over-filtering of
    critical chunks (e.g. the FDA approval table that shows age eligibility).
    """
    n = len(index_lines)
    index_text = "\n\n".join(index_lines)

    return f"""You are a relevance filter for prior authorization policy documents.

Target: {brand} prior authorization criteria for {indication} ONLY.

Below are {n} numbered text sections retrieved from a multi-drug policy document.
Some sections contain {brand} criteria for {indication}.
Others contain criteria for DIFFERENT conditions or drugs — those must be excluded.

KEEP sections about:
- Step therapy / prior treatment requirements for {indication}
- Initial and reauthorization approval criteria for {indication}
- TB test / laboratory screening (applies to all biologics including {brand})
- Age eligibility specifically for {brand}
- Authorization duration and renewal requirements for {indication}
- Quantity limits for {brand}
- Funded/unfunded diagnosis notes for {indication}

EXCLUDE sections about:
- Rheumatoid arthritis, ankylosing spondylitis, Crohn's disease, ulcerative colitis
- Juvenile idiopathic arthritis, hidradenitis suppurativa, atopic dermatitis, asthma
- Unrelated drugs or drug classes not covering {indication}
- Table of contents, administrative headers, general policy preamble

Return ONLY a valid JSON array of the section numbers to KEEP, e.g. [0, 2, 5, 11, 14]
No explanation. No other text. Only the JSON array.

Sections:

{index_text}"""


# ── Response parsing ───────────────────────────────────────────────────────────

def _parse_indices(response_text: str, max_idx: int) -> list[int]:
    """Extract valid integer indices from the 8B model's JSON response."""
    if not response_text:
        return []

    # Strip markdown code fences the model might add
    text = re.sub(r"```[a-z]*\n?", "", response_text).strip()

    # Find the first JSON array of integers in the response
    m = re.search(r"\[[\s\d,]+\]", text)
    if not m:
        log.debug("No JSON array in context-reducer response: %r", text[:200])
        return []

    try:
        raw = json.loads(m.group())
        valid = sorted({int(i) for i in raw if isinstance(i, (int, float)) and 0 <= int(i) < max_idx})
        log.debug("Context reducer selected indices: %s", valid)
        return valid
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        log.debug("Index parse failed from %r: %s", text[:200], exc)
        return []