# Payer Policy Intelligence — Reproducible RAG Pipeline

## Final LLM Models Used

| Pipeline Node | Model | Reason |
|---|---|---|
| PDF parsing, chunking, normalization, scoring | No LLM | Deterministic Python |
| Brand detection (manifest mode) | No LLM | Brand from workbook |
| Brand detection (auto mode) | Alias matching (deterministic) | Regex-based |
| Evidence extraction & 12-parameter JSON | `llama-3.3-70b-versatile` | Complex policy reasoning |
| JSON repair (fallback only) | `llama-3.1-8b-instant` | Cheap repair step |

## Evaluator Setup (4 Steps)

```bash
# 1. Unzip
unzip payer-policy-intelligence-submission.zip
cd payer-policy-intelligence-submission

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set credentials
cp .env.template .env
# Edit .env and set: GROQ_API_KEY=<your_key>

# 4. Run
python run_pipeline.py --mode manifest
```

**Output:** `outputs/result.csv`

## Folder Structure

```
payer-policy-intelligence-submission/
├── README.md
├── requirements.txt
├── .env.template
├── run_pipeline.py              ← mandatory driver
├── config/
│   ├── default.yaml             ← runtime config (paths, models, tuning)
│   ├── brand_registry.json      ← 79-row manifest + 15-brand keyword registry
│   ├── parameter_extraction_schema.json  ← 12 parameter extraction rules
│   └── access_score_config.json ← deterministic scoring rules
├── data/
│   ├── business_rules/
│   │   └── PA_Business_Rules.xlsx
│   ├── input_pdfs/              ← place evaluation PDFs here
│   └── adhoc_pdfs/              ← for auto mode
├── src/                         ← all pipeline source modules
├── notebooks/
│   └── pipeline_testing.ipynb   ← step-by-step testing notebook
├── tests/                       ← unit tests
├── outputs/                     ← result.csv written here
├── intermediate_outputs/        ← extraction JSONL + cache
└── logs/                        ← pipeline.log
```

## Runtime Modes

| Mode | Command | Description |
|---|---|---|
| **manifest** (default) | `python run_pipeline.py` | Process all rows from Submissions tab |
| **auto** | `python run_pipeline.py --mode auto --input_dir data/adhoc_pdfs` | Scan arbitrary PDFs, detect brands |
| **both** | `python run_pipeline.py --mode both` | Run both modes |

## Failure Conditions

- Missing `GROQ_API_KEY` → error at startup with clear message
- Missing PDF from manifest → logged and pipeline exits with list of missing files
- Groq API error → 3 retries with exponential backoff, then error row written
- Malformed LLM JSON → automatic 8B repair; if still fails, NA skeleton row written

## Testing

```bash
# Unit tests (no API key required)
python tests/test_normalizer.py
python tests/test_step_logic.py
python tests/test_scorer.py

# Interactive step-by-step testing
jupyter notebook notebooks/pipeline_testing.ipynb
```
