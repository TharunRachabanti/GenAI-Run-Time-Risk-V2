"""
run_pipeline.py
===============
Master orchestration script for the GenAI Credit-Risk Decision Engine.

Usage examples:
    # Run everything (Phase 1 + Phase 2 + Phase 3 LLM experiments)
    python run_pipeline.py

    # Run only Phase 1 (no API key needed)
    python run_pipeline.py --phase1-only

    # Run Phase 1 + Phase 2 RAG check, but skip LLM API calls
    python run_pipeline.py --skip-llm

Requires:
    - .env file in the project root with GEMINI_API_KEY (for LLM phases)
    - data/raw/application_train.csv
    - policy_documents/ with the 6 .docx files
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# Load environment variables from .env before anything else 
load_dotenv()

# =====================================================================
# CLIENT CONFIGURATION: TEST MODE SETTINGS
# =====================================================================
# When running with the `--test-mode` flag, the pipeline will limit
# execution to a small subset of borrowers. 
#
# Option A: Specify a list of specific Applicant IDs to test (e.g., ['P038', 'P040']).
# If this is set to a list, the pipeline will only run those borrowers.
TEST_BORROWER_IDS = ['P038', 'P040', 'P042']

# Option B: If TEST_BORROWER_IDS is None or empty, it will select the first N borrowers.
TEST_BORROWER_COUNT = 5
# =====================================================================


# Logging setup

def _setup_logging(verbose: bool = False) -> None:
    """Configure root logger with timestamps.

    Args:
        verbose (bool): If True, sets level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# Config loader

def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load the YAML configuration from disk.

    Args:
        config_path (str): Relative or absolute path to settings.yaml.

    Returns:
        Dict[str, Any]: Parsed configuration as a nested dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# Phase runners

def run_phase1(cfg: Dict[str, Any], logger: logging.Logger) -> Any:
    """Execute Phase 1: data cleaning, PD model training, GT labelling.

    Args:
        cfg (Dict[str, Any]): Full settings dict.
        logger (logging.Logger): Root logger instance.

    Returns:
        Any: The final Ground Truth DataFrame.
    """
    from src.pd_model import PDModel
    from src.data_pipeline import run_phase1 as _run_phase1

    paths = cfg["paths"]
    pd_cfg = cfg["pd_model"]
    data_cfg = cfg["data"]

    model_path = str(
        Path(paths["model_output_dir"]) / paths["pd_model_filename"]
    )

    pd_model = PDModel(
        features=data_cfg["model_features"],
        model_path=model_path,
        random_state=pd_cfg["random_state"],
        max_iter=pd_cfg["max_iter"],
        test_size=pd_cfg["test_size"],
    )

    logger.info("=" * 60)
    logger.info("PHASE 1 — Traditional Quantitative Baseline")
    logger.info("=" * 60)
    t0 = time.time()
    df_gt = _run_phase1(cfg, pd_model)
    elapsed = time.time() - t0
    logger.info("Phase 1 complete in %.1f s.", elapsed)
    logger.info(
        "Ground Truth Decision distribution:\n%s",
        df_gt["GROUND_TRUTH_DECISION"].value_counts().to_string(),
    )
    return df_gt


def run_phase2(cfg: Dict[str, Any], logger: logging.Logger) -> Any:
    """Execute Phase 2: parse policy documents and retrieve context.

    Args:
        cfg (Dict[str, Any]): Full settings dict.
        logger (logging.Logger): Root logger instance.

    Returns:
        Any: Initialised PolicyDocumentParser instance.
    """
    from src.rag_engine import PolicyDocumentParser

    logger.info("=" * 60)
    logger.info("PHASE 2 — RAG Engine Initialisation")
    logger.info("=" * 60)
    t0 = time.time()

    parser = PolicyDocumentParser(cfg["paths"]["policy_docs_dir"])
    ctx_2026, _ = parser.retrieve_context(2026)
    ctx_2025, _ = parser.retrieve_context(2025)
    elapsed = time.time() - t0

    logger.info(
        "Parsed %d documents in %.1f s.", len(parser.documents), elapsed
    )
    logger.info(
        "  2026 context length: %d chars | 2025 context length: %d chars",
        len(ctx_2026),
        len(ctx_2025),
    )
    return parser


def run_phase3(
    cfg: Dict[str, Any],
    df_gt: Any,
    parser: Any,
    logger: logging.Logger,
    skip_llm: bool = False,
    test_mode: bool = False,
) -> Any:
    """Execute Phase 3: LLM experiments (EXP-001 through EXP-004).

    Args:
        cfg (Dict[str, Any]): Full settings dict.
        df_gt (Any): Ground Truth DataFrame from Phase 1.
        parser (Any): Initialised PolicyDocumentParser from Phase 2.
        logger (logging.Logger): Root logger instance.
        skip_llm (bool): If True, skips calling the API.
        test_mode (bool): If True, limits records to minimise API cost.

    Returns:
        Any: Final results DataFrame with all experiment columns.
    """
    from src.llm_experiments import ExperimentOrchestrator

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Set up per-borrower audit directory
    audit_dir = Path(cfg["paths"]["audit_dir"])
    audit_dir.mkdir(parents=True, exist_ok=True)


    logger.info("=" * 60)
    logger.info("PHASE 3 — LLM Experiments")
    logger.info("=" * 60)
    logger.info("Audit directory: %s", audit_dir)
    t0 = time.time()

    orchestrator = ExperimentOrchestrator(
        cfg=cfg, 
        parser=parser, 
        dry_run=skip_llm, 
        audit_dir=str(audit_dir)
    )
    df_results = orchestrator.run_all(df_gt)

    elapsed = time.time() - t0
    logger.info("Phase 3 complete in %.1f s.", elapsed)

    # Save timestamped results CSV
    out_path = processed_dir / f"{cfg['paths']['artifact_04_prefix']}_{timestamp}.csv"
    df_results.to_csv(out_path, index=False)
    logger.info("  ✔ Saved results → %s  (%d rows)", out_path, len(df_results))

    return df_results


# CLI argument parsing

def _parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed namespace with CLI flags.
    """
    parser = argparse.ArgumentParser(
        description="GenAI Credit-Risk Decision Engine — Master Pipeline Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="Run only Phase 1 (data cleaning, PD model, Ground Truth). No API key needed.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Run Phase 1 + Phase 2 (RAG init), but skip LLM API calls.\n"
             "Writes 'LLM_SKIPPED' for all experiment decision columns.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Limit execution to a subset of test borrowers (configurable in run_pipeline.py).",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to the settings YAML file (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


# Entry point

def main() -> None:
    """Orchestrate the full pipeline based on CLI flags."""
    args = _parse_args()
    _setup_logging(verbose=args.verbose)
    logger = logging.getLogger("run_pipeline")

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║   GenAI Credit-Risk Decision Engine Pipeline     ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    try:
        cfg = load_config(args.config)
        logger.info("Configuration loaded from: %s", args.config)
    except (FileNotFoundError, Exception) as exc:
        logger.error("Failed to load configuration: %s", exc)
        sys.exit(1)

    try:
        df_gt = run_phase1(cfg, logger)
        if args.test_mode:
            if TEST_BORROWER_IDS:
                logger.warning(f"--test-mode active: limiting to specific IDs: {TEST_BORROWER_IDS}.")
                df_gt = df_gt[df_gt['Applicant_ID'].isin(TEST_BORROWER_IDS)].copy()
            else:
                logger.warning(f"--test-mode active: limiting to the first {TEST_BORROWER_COUNT} borrowers.")
                df_gt = df_gt.head(TEST_BORROWER_COUNT).copy()
    except Exception as exc:
        logger.error("Phase 1 failed: %s", exc, exc_info=True)
        sys.exit(1)

    if args.phase1_only:
        logger.info("--phase1-only flag set. Pipeline complete.")
        return

    try:
        parser = run_phase2(cfg, logger)
    except Exception as exc:
        logger.error("Phase 2 failed: %s", exc, exc_info=True)
        sys.exit(1)

    if args.skip_llm:
        logger.info("--skip-llm flag passed. Exiting cleanly before Phase 3.")
        return

    try:
        df_results = run_phase3(
            cfg=cfg,
            df_gt=df_gt,
            parser=parser,
            logger=logger,
            skip_llm=args.skip_llm,
            test_mode=args.test_mode,
        )
    except Exception as exc:
        logger.error("Phase 3 failed: %s", exc, exc_info=True)
        sys.exit(1)

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║   Pipeline Complete — all artefacts saved.       ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    # Print a final summary table
    summary_cols = [
        "Applicant_Code",
        "GROUND_TRUTH_DECISION",
        "EXP_001_Decision",
        "EXP_002_Decision",
        "EXP_003_Decision",
        "EXP_004_Decision",
    ]
    existing = [c for c in summary_cols if c in df_results.columns]
    logger.info("\nSample of final results (first 5 rows):\n%s", df_results[existing].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
