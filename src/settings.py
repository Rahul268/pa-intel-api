"""
settings.py
Load and validate runtime configuration from default.yaml, .env,
and all three JSON config files. Exposes a Settings dataclass used
by every other module.
"""

from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Repository root is two levels above this file: src/ -> repo_root/
REPO_ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger(__name__)


@dataclass
class Settings:
    # Core paths
    repo_root: Path
    input_pdfs_dir: Path
    adhoc_pdfs_dir: Path
    business_rules_path: Path
    outputs_dir: Path
    intermediate_outputs_dir: Path
    logs_dir: Path

    # Credentials
    groq_api_key: str

    # Models
    model_extraction: str
    model_repair: str
    allowed_models: list[str]

    # Loaded JSON configs
    brand_registry: dict[str, Any]
    parameter_schema: dict[str, Any]
    access_score_config: dict[str, Any]
    business_rules: dict[str, str]       # from config/business_rules.json (Excel Business Rules tab)
    reference_examples: dict[str, Any]   # from config/reference_examples.json (Excel Reference tab)

    # Extraction tuning
    top_k_universal: int
    top_k_brand: int
    top_k_param: int
    max_evidence_chars: int
    temperature: float
    max_tokens: int
    extraction_groups: dict  # model + params per group

    # Pipeline behaviour
    idempotent: str
    save_intermediate: bool
    cache_extractions: bool
    submissions_tab: str

    # Derived convenience lookups
    brand_keywords: dict[str, list[str]] = field(default_factory=dict)
    brand_info: dict[str, Any] = field(default_factory=dict)


def load_settings(env_path: Path | None = None, config_path: Path | None = None) -> Settings:
    """Load and validate all settings. Raises ValueError on missing critical items."""

    # --- .env ---
    env_file = env_path or REPO_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()  # fall back to environment variables

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Copy .env.template to .env and set your key."
        )

    # --- default.yaml ---
    cfg_file = config_path or REPO_ROOT / "config" / "default.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)

    def rp(rel: str) -> Path:
        """Resolve a relative path from the repo root."""
        return REPO_ROOT / rel

    # --- JSON configs ---
    brand_registry_path = rp(cfg["paths"]["brand_registry"])
    param_schema_path   = rp(cfg["paths"]["parameter_schema"])
    score_cfg_path      = rp(cfg["paths"]["access_score_config"])

    for p in (brand_registry_path, param_schema_path, score_cfg_path):
        if not p.exists():
            raise FileNotFoundError(f"Required config file missing: {p}")

    with open(brand_registry_path) as f:
        brand_registry = json.load(f)
    with open(param_schema_path) as f:
        parameter_schema = json.load(f)
    with open(score_cfg_path) as f:
        access_score_config = json.load(f)

    # Business rules and reference examples from Excel (pre-extracted to JSON at build time)
    br_cfg_path  = rp(cfg["paths"].get("business_rules_config",  "config/business_rules.json"))
    ref_ex_path  = rp(cfg["paths"].get("reference_examples",     "config/reference_examples.json"))
    with open(br_cfg_path) as f:
        business_rules = json.load(f)
    with open(ref_ex_path) as f:
        reference_examples = json.load(f)

    # --- Model allowlist validation ---
    allowed = cfg["models"].get("allowed", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ])
    for model_key in ("extraction", "repair"):
        m = cfg["models"][model_key]
        if m not in allowed:
            raise ValueError(
                f"Model '{m}' is not in the allowed list: {allowed}"
            )

    # --- Build brand convenience lookups ---
    brands_section = brand_registry.get("brands", {})
    brand_keywords = {
        name: info.get("retrieval_keywords", [])
        for name, info in brands_section.items()
    }

    s = Settings(
        repo_root=REPO_ROOT,
        input_pdfs_dir=rp(cfg["paths"]["input_pdfs"]),
        adhoc_pdfs_dir=rp(cfg["paths"]["adhoc_pdfs"]),
        business_rules_path=rp(cfg["paths"]["business_rules"]),
        outputs_dir=rp(cfg["paths"]["outputs"]),
        intermediate_outputs_dir=rp(cfg["paths"]["intermediate_outputs"]),
        logs_dir=rp(cfg["paths"]["logs"]),
        groq_api_key=api_key,
        model_extraction=cfg["models"]["extraction"],
        model_repair=cfg["models"]["repair"],
        allowed_models=allowed,
        brand_registry=brand_registry,
        parameter_schema=parameter_schema,
        access_score_config=access_score_config,
        business_rules=business_rules,
        reference_examples=reference_examples,
        top_k_universal=cfg["extraction"]["top_k_universal"],
        top_k_brand=cfg["extraction"]["top_k_brand"],
        top_k_param=cfg["extraction"]["top_k_param"],
        max_evidence_chars=cfg["extraction"]["max_evidence_chars"],
        temperature=cfg["extraction"]["temperature"],
        max_tokens=cfg["extraction"]["max_tokens"],
        extraction_groups=cfg.get("extraction_groups", {}),
        idempotent=cfg["pipeline"]["idempotent"],
        save_intermediate=cfg["pipeline"]["save_intermediate"],
        cache_extractions=cfg["pipeline"]["cache_extractions"],
        submissions_tab=cfg.get("submissions_tab", "Submissions"),
        brand_keywords=brand_keywords,
        brand_info=brands_section,
    )

    log.debug("Settings loaded. Extraction model: %s", s.model_extraction)
    return s
