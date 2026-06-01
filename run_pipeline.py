#!/usr/bin/env python3
"""
run_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mandatory driver for the Payer Policy Intelligence pipeline.

Usage
─────
  Default (manifest mode):
      python run_pipeline.py

  Explicit manifest mode:
      python run_pipeline.py --mode manifest

  Auto mode (arbitrary PDFs):
      python run_pipeline.py --mode auto --input_dir data/adhoc_pdfs

  Both modes:
      python run_pipeline.py --mode both

  Custom output path:
      python run_pipeline.py --output_path outputs/my_result.csv

Prerequisites
─────────────
  1. pip install -r requirements.txt
  2. cp .env.template .env  →  set GROQ_API_KEY=<your_key>
  3. Place PDFs in data/input_pdfs/  (for manifest mode)

Output
──────
  outputs/result.csv          (manifest mode)
  outputs/adhoc_result.csv    (auto mode)
  intermediate_outputs/       (audit JSONL + cache)
  logs/pipeline.log
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import sys
from pathlib import Path

# Ensure the repo root is on the Python path regardless of how the script is invoked
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Payer Policy Intelligence — reproducible RAG extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["manifest", "auto", "both"],
        default="manifest",
        help="manifest (default): process Submissions tab rows. "
             "auto: scan input_dir for PDFs and detect brands. "
             "both: run manifest then auto.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Directory of PDFs for auto mode (default: data/adhoc_pdfs).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Override the default output CSV path.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override path to default.yaml.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=None,
        help="Override path to .env file.",
    )
    args = parser.parse_args()

    # Load settings (validates credentials and config upfront)
    from src.settings import load_settings
    try:
        settings = load_settings(env_path=args.env, config_path=args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] Settings failed to load:\n  {exc}\n", file=sys.stderr)
        sys.exit(1)

    # Run pipeline
    from src.pipeline import run
    try:
        run(
            settings=settings,
            mode=args.mode,
            input_dir=args.input_dir,
            output_path=args.output_path,
        )
    except RuntimeError as exc:
        print(f"\n[ERROR] Pipeline error:\n  {exc}\n", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Pipeline stopped by user.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
