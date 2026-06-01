"""
prompts.py
Build extraction prompts for one Filename + Brand row.

Two call modes:
  - Single call (legacy): build_extraction_prompt() → one large prompt
  - Two-call split: build_group_messages() → (system_msg, user_msg) per group
      system_msg  — static content, identical for every row in the same group
                    → Groq caches this automatically for repeated identical prefixes
      user_msg    — dynamic content: evidence + filename + brand
                    → changes per row

Group definitions live in config/default.yaml under extraction_groups.
"""

from __future__ import annotations
import json
import logging
from typing import Any

log = logging.getLogger(__name__)

OUTPUT_SCHEMA = {
    "Filename": "",
    "Brand": "",
    "Age": "",
    "Step Therapy Requirements Documented in Policy": "",
    "Number of Steps through Brands": "",
    "Number of Steps through Generic": "",
    "Step through-Phototherapy": "",
    "TB Test required": "",
    "Quantity Limits": "",
    "Specialist Types": "",
    "Initial Authorization Duration(in-months)": "",
    "Reauthorization Duration(in-months)": "",
    "Reauthorization Required": "",
    "Reauthorization Requirements Documented in Policy": "",
}

SYSTEM_PROMPT = (
    "You are a precise payer policy analyst. Extract structured data from payer "
    "Prior Authorization (PA) policy documents. Follow the business rules and worked "
    "example exactly. Return only valid JSON. Never invent information not in the evidence."
)


# ── Two-call split (primary path) ────────────────────────────────────────────

def build_group_messages(
    group_name: str,
    group_config: dict[str, Any],
    filename: str,
    brand: str,
    evidence_text: str,
    parameter_schema: dict[str, Any],
    brand_registry: dict[str, Any],
    business_rules: dict[str, str] | None = None,
    reference_examples: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """
    Build (system_message, user_message) for one extraction group.

    system_message  — STATIC for this group: instructions + business rules for
                      the group's parameters + reference example subset.
                      Identical for every row → Groq prefix-caches it after the
                      first call. Cached tokens don't count toward TPM.

    user_message    — DYNAMIC per row: evidence text + filename + brand.

    Returns (system_msg, user_msg).
    """
    group_params = group_config.get("parameters", [])

    # Filter schema, business rules, and reference examples to this group only
    filtered_schema = _filter_schema(parameter_schema, group_params)
    filtered_rules  = _filter_dict(business_rules or {}, group_params)
    filtered_refs   = _filter_dict(reference_examples or {}, group_params)

    # ── System message (static, cacheable) ────────────────────────────────
    brand_notes            = _build_brand_notes(brand, brand_registry)
    business_rules_section = _build_business_rules_section(filtered_rules)
    reference_section      = _build_reference_example_section(filtered_refs)
    param_instructions     = _build_param_instructions(filtered_schema, group_params)

    # Partial schema for this group only
    group_schema = {k: OUTPUT_SCHEMA[k] for k in group_params if k in OUTPUT_SCHEMA}
    schema_str   = json.dumps(group_schema, indent=2)

    system_msg = f"""{SYSTEM_PROMPT}

Group: {group_name} — {group_config.get('description', '')}
You are extracting ONLY these parameters: {', '.join(group_params)}

{business_rules_section}

{reference_section}

━━━ EXTRACTION INSTRUCTIONS ━━━

GENERAL RULES:
- Use ONLY the evidence in the user message. Do not use external knowledge.
- For OR conditions in step therapy: take the LEAST RESTRICTIVE path (fewest steps).
- If two step statements have NO explicit AND/OR connector, treat as OR.
- Do not count phototherapy as a branded or generic step — it has its own field.
- If information for a field is absent from the evidence, use the fallback value.

PARAMETER-BY-PARAMETER INSTRUCTIONS:
{param_instructions}

━━━ OUTPUT FORMAT ━━━

Return ONLY valid JSON matching this exact schema. No preamble, no explanation,
no markdown code fences. Fill every field — use fallback if absent.

{schema_str}"""

    # ── User message (dynamic per row) ────────────────────────────────────
    user_msg = f"""Filename: {filename}
Brand: {brand}
Target indication: Plaque Psoriasis / Psoriasis
{brand_notes}

━━━ EVIDENCE FROM POLICY DOCUMENT ━━━
{evidence_text}
━━━ END OF EVIDENCE ━━━

Extract only these parameters: {', '.join(group_params)}
Return JSON only."""

    return system_msg, user_msg


def split_evidence_by_group(
    evidence_chunks: list[Any],
    group_config: dict[str, Any],
    min_chunks: int = 6,
) -> list[Any]:
    """
    Filter evidence chunks to those most relevant to a parameter group.

    Uses the group's evidence_keywords to score each chunk. Chunks scoring
    above zero are kept. Falls back to all chunks if filtered set is too small.
    """
    keywords = [kw.lower() for kw in group_config.get("evidence_keywords", [])]
    if not keywords:
        return evidence_chunks

    def chunk_score(chunk: Any) -> int:
        """Count how many of this group's evidence_keywords appear in the chunk."""
        text_lower = chunk.text.lower()
        return sum(1 for kw in keywords if kw in text_lower)

    scored   = [(c, chunk_score(c)) for c in evidence_chunks]
    filtered = [c for c, s in scored if s > 0]

    if len(filtered) < min_chunks:
        log.debug(
            "Group '%s' evidence filter too aggressive (%d < %d) — using all %d chunks",
            group_config.get("description", "?"), len(filtered), min_chunks, len(evidence_chunks),
        )
        return evidence_chunks

    log.debug(
        "Group evidence filter: %d → %d chunks",
        len(evidence_chunks), len(filtered),
    )
    return filtered


# ── Single-call (legacy / fallback) ──────────────────────────────────────────

def build_extraction_prompt(
    filename: str,
    brand: str,
    evidence_text: str,
    parameter_schema: dict[str, Any],
    brand_registry: dict[str, Any],
    business_rules: dict[str, str] | None = None,
    reference_examples: dict[str, Any] | None = None,
) -> str:
    """Legacy single-call prompt. Used as fallback when extraction_groups not configured."""
    brand_notes            = _build_brand_notes(brand, brand_registry)
    business_rules_section = _build_business_rules_section(business_rules or {})
    reference_section      = _build_reference_example_section(reference_examples or {})
    param_instructions     = _build_param_instructions(parameter_schema, list(OUTPUT_SCHEMA.keys()))
    schema_str             = json.dumps(OUTPUT_SCHEMA, indent=2)

    return f"""You are extracting payer Prior Authorization (PA) access criteria.

Filename: {filename}
Brand: {brand}
Target indication: Plaque Psoriasis / Psoriasis

━━━ EVIDENCE FROM POLICY DOCUMENT ━━━
{evidence_text}
━━━ END OF EVIDENCE ━━━
{brand_notes}

{business_rules_section}

{reference_section}

━━━ EXTRACTION INSTRUCTIONS ━━━

GENERAL RULES:
- Use ONLY the evidence provided above. Do not use external knowledge.
- Combine universal criteria with brand-specific criteria using AND logic.
- For OR conditions in step therapy: take the LEAST RESTRICTIVE path (fewest steps).
- If two step statements have NO explicit AND/OR connector, treat as OR.
- Do not count phototherapy as a branded or generic step — it has its own field.
- If information for a field is absent from the evidence, use the fallback value.

PARAMETER-BY-PARAMETER INSTRUCTIONS:
{param_instructions}

━━━ OUTPUT FORMAT ━━━

Return ONLY valid JSON matching this exact schema. No preamble, no explanation,
no markdown code fences. Fill every field — use the fallback value if absent.

{schema_str}
"""


def build_repair_prompt(malformed_json: str) -> str:
    """Build a prompt asking the repair model to fix malformed JSON.
    Called by extractor.py when _parse_json fails on the primary model output.
    The repair model (llama-3.1-8b-instant) is cheaper and fast for this task.
    """
    return (
        "The following text should be a valid JSON object but contains formatting "
        "errors. Return ONLY the corrected JSON — no explanation, no markdown, "
        "no code blocks.\n\n"
        f"{malformed_json}"
    )


# ── Filtering helpers ─────────────────────────────────────────────────────────

def _filter_schema(parameter_schema: dict[str, Any], group_params: list[str]) -> dict[str, Any]:
    """Return a parameter_schema dict scoped to this group's parameters only.
    Ensures the system message only contains extraction instructions for the
    parameters being extracted in this call — keeps the prompt lean.
    """

    params = parameter_schema.get("parameters", [])
    return {"parameters": [p for p in params if p.get("param_name") in group_params]}


def _filter_dict(d: dict[str, Any], group_params: list[str]) -> dict[str, Any]:
    """Return entries from d whose key matches any parameter in group_params.
    Used to scope business_rules and reference_examples to the current group
    so the system message only injects rules relevant to this extraction call.
    Matching is bidirectional substring: handles cases like 'Reauthorization
    Required' matching inside 'Reauthorization Requirements Documented...'.
    """

    if not d:
        return {}
    result = {}
    for key, val in d.items():
        if any(key in p or p in key for p in group_params):
            result[key] = val
    return result


# ── Shared section builders ───────────────────────────────────────────────────

def _build_business_rules_section(business_rules: dict[str, str]) -> str:
    """Format the group-scoped business rules into the system message section.
    Each parameter's rule is indented under its name for readability.
    This section is STATIC — identical for every row in this group, which
    is what allows Groq to cache it across calls.
    """
    if not business_rules:
        return ""
    lines = ["━━━ BUSINESS RULES (follow these strictly) ━━━"]
    for param, definition in business_rules.items():
        lines.append(f"\n{param}:")
        for line in definition.split("\n"):
            s = line.strip()
            if s:
                lines.append(f"  {s}")
    lines.append("\n━━━ END BUSINESS RULES ━━━")
    return "\n".join(lines)


def _build_reference_example_section(reference_examples: dict[str, Any]) -> str:
    """Format the reference worked-example into the system message section.
    Shows the expected output values and counting-logic walkthroughs for each
    parameter in this group. The counting logic is the most valuable part —
    it teaches the LLM how the AND/OR chain logic was applied on a real case.
    """
    if not reference_examples:
        return ""
    lines = [
        "━━━ WORKED EXAMPLE (from PA_Business_Rules.xlsx Reference tab) ━━━",
        "Use this to understand the counting logic and output format.\n",
        "Expected parameter values:",
    ]
    for param, ex in reference_examples.items():
        val = ex.get("expected_value", "")
        if val:
            lines.append(f"  {param}: {val[:120]}")

    logic_notes = {p: ex["counting_logic"] for p, ex in reference_examples.items()
                   if ex.get("counting_logic", "").strip()}
    if logic_notes:
        lines.append("\nCounting logic notes:")
        for param, logic in logic_notes.items():
            lines.append(f"\n  [{param}]:")
            for lline in logic.split("\n"):
                s = lline.strip()
                if s:
                    lines.append(f"    {s}")

    lines.append("\n━━━ END WORKED EXAMPLE ━━━")
    return "\n".join(lines)


def _build_brand_notes(brand: str, brand_registry: dict[str, Any]) -> str:
    """Build brand-specific contextual notes injected into the user message.
    Covers: biosimilar relationships (so the LLM looks under the reference
    brand's section), conventional drug classification (no biologic step
    expectations), REMS programs (not a step therapy step), and targeted
    synthetic classification.
    """
    info  = brand_registry.get("brands", {}).get(brand, {})
    notes: list[str] = []

    ref = info.get("reference_brand")
    if ref:
        generic = ", ".join(info.get("generic_names", []))
        notes.append(
            f"\nBRAND NOTE — BIOSIMILAR:\n"
            f"{brand} ({generic}) is a biosimilar of {ref}. "
            f"If the policy covers {brand} under the {ref} section, treat those criteria "
            f"as applicable to {brand} unless the policy explicitly excludes biosimilars."
        )
    if info.get("is_conventional"):
        notes.append(
            f"\nBRAND NOTE — CONVENTIONAL DRUG:\n"
            f"{brand} is a conventional (non-biologic) oral drug. "
            f"Do not assume or expect prior biologic step therapy requirements."
        )
    if "REMS" in info.get("special_flags", []):
        notes.append(
            f"\nBRAND NOTE — REMS PROGRAM:\n"
            f"REMS enrollment is an administrative prerequisite, NOT a step therapy step. "
            f"Do not count it toward step counts."
        )
    if info.get("is_targeted_synthetic"):
        notes.append(
            f"\nBRAND NOTE — TARGETED SYNTHETIC:\n"
            f"{brand} is a targeted synthetic oral agent. Prior steps are likely "
            f"conventional systemic drugs, not biologics."
        )
    return "\n".join(notes) if notes else ""


def _build_param_instructions(
    parameter_schema: dict[str, Any],
    group_params: list[str],
) -> str:
    """Build the per-parameter instruction block for the EXTRACTION INSTRUCTIONS
    section of the system message. For each parameter in this group, shows:
    the extraction prompt (how to find/parse the value), the expected output
    format, and the fallback value when the field is absent from the evidence.
    '{brand}' is replaced with 'this brand' to keep the system message static
    (brand-agnostic) so Groq's prefix cache applies across all rows.
    """
    params = parameter_schema.get("parameters", [])
    lines: list[str] = []
    for param in params:
        name = param["param_name"]
        if name not in group_params:
            continue
        prompt_text = param.get("extraction_prompt", "")
        fallback    = param.get("normalization_rules", {}).get("fallback_value", "NA")
        fmt         = param.get("expected_output_format", "")
        lines.append(f'"{name}":')
        if prompt_text:
            lines.append(f"  {prompt_text.replace('{brand}', 'this brand')}")
        if fmt:
            lines.append(f"  Expected format: {fmt}")
        lines.append(f'  Fallback if not found: "{fallback}"')
        lines.append("")
    return "\n".join(lines)
