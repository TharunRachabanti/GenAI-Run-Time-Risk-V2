"""
src/llm_experiments.py
=======================
LLM Experiment Orchestration for the GenAI Credit-Risk Decision Engine.

Executes four experiments (EXP-001 through EXP-004) against a dataset of
500 borrower records, each using a different prompt design and/or policy
knowledge version.

Experiment Matrix:
    EXP-001  Neutral baseline prompt          + 2026 policy context
    EXP-002  Conservative persona prompt      + 2026 policy context
    EXP-003  Chain-of-Thought (CoT) prompt    + 2026 policy context
    EXP-004  Neutral baseline prompt          + 2025 policy context

API Key: loaded from environment variable GEMINI_API_KEY (via python-dotenv).
         Never hardcoded. Never read from settings.yaml.

Google-style docstrings, full type annotations.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
import os
import time
import json
from typing import Any, Dict, List, Optional, Tuple, Literal, TypedDict

import pandas as pd
from tqdm import tqdm

class LLMDecision(TypedDict):
    reasoning: str
    final_decision: Literal["APPROVE", "APPROVE WITH CONDITIONS", "DECLINE"]

logger = logging.getLogger(__name__)

# Decision constants 
APPROVE               = "APPROVE"
APPROVE_WITH_COND     = "APPROVE WITH CONDITIONS"
DECLINE               = "DECLINE"
SKIPPED               = "LLM_SKIPPED"
PARSE_ERROR           = "PARSE_ERROR"

VALID_DECISIONS       = {APPROVE, APPROVE_WITH_COND, DECLINE}



# Prompt templates

def _build_borrower_block(row: pd.Series) -> str:
    """Format a single borrower's metrics into a structured text block.

    Uses the human-readable renamed column names from Artifact 01.

    Args:
        row: A row from the experimental DataFrame containing the readable
            financial columns and computed metrics.

    Returns:
        A formatted multi-line string describing the borrower.
    """
    applicant_id = row.get("Applicant_ID", row.get("Applicant_Code", "N/A"))
    return (
        f"--- BORROWER PROFILE ---\n"
        f"Applicant ID                  : {applicant_id}\n"
        f"Annual Income                 : {row['Income']:,.2f}\n"
        f"Loan Amount                   : {row['Loan_Amount']:,.2f}\n"
        f"Annuity Payment               : {row['Annuity_Payment']:,.2f}\n"
        f"Goods Price                   : {row['Goods_Price']:,.2f}\n"
        f"Payment-to-Income Ratio (PTI) : {row['PTI']}%\n"
        f"Credit-to-Income Ratio (CTI)  : {row['CTI']}x\n"
        f"Loan-to-Goods-Value (LGV)     : {row['LGV']}%\n"
        f"Probability of Default (PD)   : {row['FROZEN_PD']}%\n"
        f"--- END BORROWER PROFILE ---"
    )


_DECISION_RULES_BLOCK = """
--- FINAL DECISION AGGREGATION RULES ---
After classifying each metric into its risk tier using the Policy Context above, determine the Final Decision strictly using these rules in order:

RULE 1:
If PD = High AND any other metric = High
-> DECLINE

RULE 2:
If two or more metrics = High
-> DECLINE

RULE 3:
If exactly one metric = High
-> APPROVE WITH CONDITIONS

RULE 4:
If no metric = High, but at least one metric = Enhanced Review / Elevated / Moderate
-> APPROVE WITH CONDITIONS

RULE 5:
Otherwise (all metrics are Standard / Low)
-> APPROVE
--- END DECISION AGGREGATION RULES ---
"""

def _prompt_exp001(borrower_block: str, context: str) -> str:
    """Build the EXP-001 neutral baseline prompt.

    Args:
        borrower_block: Formatted borrower metrics string.
        context: Retrieved 2026 policy context.

    Returns:
        Complete prompt string.
    """
    return (
        "You are a credit risk assistant. Using ONLY the provided policy "
        "context below, evaluate the borrower and output your final decision "
        "as exactly one of: APPROVE, APPROVE WITH CONDITIONS, or DECLINE.\n\n"
        f"POLICY CONTEXT:\n{context}\n\n"
        f"{_DECISION_RULES_BLOCK}\n\n"
        f"{borrower_block}\n\n"
        "Final Decision:"
    )


def _prompt_exp002(borrower_block: str, context: str) -> str:
    """Build the EXP-002 conservative risk officer persona prompt.

    Args:
        borrower_block: Formatted borrower metrics string.
        context: Retrieved 2026 policy context.

    Returns:
        Complete prompt string.
    """
    return (
        "You are a conservative credit risk officer. Carefully apply the "
        "requirements and thresholds in the provided policy context when "
        "evaluating the borrower. Using ONLY the provided policy context below, "
        "output your final decision as exactly one of: APPROVE, APPROVE WITH "
        "CONDITIONS, or DECLINE.\n\n"
        f"POLICY CONTEXT:\n{context}\n\n"
        f"{_DECISION_RULES_BLOCK}\n\n"
        f"{borrower_block}\n\n"
        "Final Decision:"
    )





# Map experiment ID → prompt builder function
_PROMPT_BUILDERS = {
    "EXP_001": _prompt_exp001,
    "EXP_002": _prompt_exp002,
    "EXP_003": _prompt_exp001,
}


# Gemini API initialisation

def _init_gemini(model_name: str) -> Any:
    """Initialise the Google Generative AI client and return a GenerativeModel.

    Reads the API key exclusively from the ``GEMINI_API_KEY`` environment
    variable (set via python-dotenv / .env file).  Never reads from YAML
    or any hardcoded value.

    Args:
        model_name: Gemini model identifier (e.g. ``"gemini-1.5-flash"``).

    Returns:
        A configured ``google.generativeai.GenerativeModel`` instance.

    Raises:
        EnvironmentError: If ``GEMINI_API_KEY`` is not set.
        ImportError: If ``google-generativeai`` is not installed.
    """
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError(
            "google-generativeai is not installed. "
            "Run: pip install google-generativeai"
        ) from exc

    api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your key."
        )

    genai.configure(api_key=api_key)
    logger.info("Gemini API configured with model: %s", model_name)
    return genai.GenerativeModel(model_name)

def _save_excel(df: pd.DataFrame, csv_path: str) -> None:
    """Save the DataFrame to a nicely formatted Excel workbook alongside the CSV."""
    import openpyxl
    from openpyxl.styles import PatternFill, Font, Alignment
    
    excel_path = csv_path.replace(".csv", ".xlsx")
    try:
        # Use openpyxl via pandas first to just dump data
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Results")
        
        # Now reopen with openpyxl to apply formatting
        wb = openpyxl.load_workbook(excel_path)
        ws = wb["Results"]
        
        # 1. Header row
        header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)
        header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
        
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_align
            
        ws.freeze_panes = "A2"
        
        # Body cells alignment
        body_align = Alignment(wrap_text=True, vertical="top")
        
        # Colors for Decision columns
        color_approve = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
        color_cond = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
        color_decline = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")

        decision_cols = ["GROUND_TRUTH_DECISION", "EXP_001_Decision", "EXP_002_Decision", "EXP_003_Decision"]
        col_name_to_idx = {cell.value: idx for idx, cell in enumerate(ws[1], 1)}
        
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = body_align
                
                # Check if it's a decision column
                col_name = ws.cell(row=1, column=cell.column).value
                if col_name in decision_cols:
                    if cell.value == "APPROVE":
                        cell.fill = color_approve
                    elif cell.value == "APPROVE WITH CONDITIONS":
                        cell.fill = color_cond
                    elif cell.value == "DECLINE":
                        cell.fill = color_decline

        # Set Column Widths
        for col_name, idx in col_name_to_idx.items():
            col_letter = openpyxl.utils.get_column_letter(idx)
            if col_name and col_name.endswith("_Reasoning"):
                ws.column_dimensions[col_letter].width = 55
            elif col_name and col_name.endswith("_Retrieved_Policies"):
                ws.column_dimensions[col_letter].width = 50
            elif col_name and col_name.endswith("_Decision") or col_name == "GROUND_TRUTH_DECISION":
                ws.column_dimensions[col_letter].width = 26
            else:
                ws.column_dimensions[col_letter].width = 16

        wb.save(excel_path)
    except Exception as e:
        logger.warning(f"Failed to format Excel file {excel_path}: {e}")

# Experiment Orchestrator

class ExperimentOrchestrator:
    """Orchestrates EXP-001 through EXP-004 LLM experiments.

    Iterates over every row in the Ground Truth DataFrame, constructs
    experiment-specific prompts, calls the Gemini API, and aggregates
    decisions into a final results DataFrame.

    In ``dry_run`` mode no API calls are made; all experiment decisions
    are set to ``'LLM_SKIPPED'``.  This is safe to run without a
    ``GEMINI_API_KEY``.

    Attributes:
        cfg: Full settings dict from ``config/settings.yaml``.
        parser: Initialised ``PolicyDocumentParser`` instance.
        dry_run: If True, skip all API calls.
        model_name: Gemini model name from settings.
        temperature: Sampling temperature from settings.
        max_tokens: Max output tokens from settings.
        request_timeout: Per-request timeout in seconds.
        model: Initialised ``GenerativeModel`` (None in dry_run mode).
        context_2026: Retrieved 2026 policy context string.
        context_2025: Retrieved 2025 policy context string.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        parser: Any,
        dry_run: bool = False,
        audit_dir: Optional[str] = None,
    ) -> None:
        """Initialise the orchestrator and pre-fetch policy contexts.

        Args:
            cfg: Full settings dict loaded from ``config/settings.yaml``.
            parser: Initialised ``PolicyDocumentParser``.
            dry_run: If True, no LLM API calls are made.
            audit_dir: Optional path to write per-borrower textual audit logs.
        """
        self.cfg = cfg
        self.parser = parser
        self.dry_run = dry_run
        self.audit_dir = audit_dir

        llm_cfg = cfg["llm"]
        self.model_name:      str   = llm_cfg["model_name"]
        self.temperature:     float = float(llm_cfg["temperature"])
        self.max_tokens:      int   = int(llm_cfg["max_tokens"])
        self.request_timeout: int   = int(llm_cfg["request_timeout"])

        # Initialise LLM client
        self.model: Optional[Any] = None
        if not self.dry_run:
            self.model = _init_gemini(self.model_name)
        else:
            logger.info("ExperimentOrchestrator: dry_run=True — LLM calls skipped.")
        
        self._exp_top_k: Dict[str, int] = {
            "EXP_001": 6,
            "EXP_002": 6,
            "EXP_003": 4,
        }

    # Private helpers

    def _call_llm(self, prompt: str) -> str:
        """Send a single prompt to the Gemini API and return the response text.

        Implements an exponential backoff strategy: catches 429 Too Many Requests
        or ResourceExhausted errors, waits progressively, and retries.

        Args:
            prompt: The complete, formatted prompt string.

        Returns:
            Raw text from the model response.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        import google.generativeai as genai

        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
            response_schema=LLMDecision,
        )

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={"timeout": self.request_timeout},
                )
                return response.text
            except Exception as exc:
                exc_str = str(exc).lower()
                if "429" in exc_str or "exhausted" in exc_str or "quota" in exc_str:
                    wait_sec = 10 * attempt
                    logger.warning("Rate limit hit (Attempt %d). Waiting %ds...", attempt, wait_sec)
                    time.sleep(wait_sec)
                else:
                    logger.warning("LLM call attempt %d failed: %s", attempt, exc)
                    if attempt < max_attempts:
                        time.sleep(5)

        raise RuntimeError(
            f"LLM call failed after {max_attempts} attempts for prompt starting with: "
            f"{prompt[:80]!r}"
        )

    def _run_single_experiment(
        self,
        exp_id: str,
        row: pd.Series,
    ) -> Tuple[str, str, str, str, str]:
        """Execute one experiment for a single borrower row.

        Args:
            exp_id: Experiment identifier (``'EXP_001'`` … ``'EXP_003'``).
            row: A DataFrame row with PTI, CTI, LGV, FROZEN_PD columns.

        Returns:
            Tuple of (prompt, raw_response, extracted_decision, retrieved_sources, reasoning).
        """
        borrower_block = _build_borrower_block(row)
        top_k = self._exp_top_k[exp_id]
        context, sources_meta = self.parser.retrieve_context(top_k=top_k, borrower_row=row)
        prompt_fn = _PROMPT_BUILDERS[exp_id]
        prompt = prompt_fn(borrower_block, context)
        
        sources_str_parts = []
        for m in sources_meta:
            score_str = f"Score: {m['score']:.4f}" if 'score' in m else ""
            match_str = f"Matched: {', '.join(m.get('reason', []))}" if 'reason' in m else ""
            bracket_content = " | ".join(filter(bool, [score_str, match_str]))
            sources_str_parts.append(f"Rank {m['rank']}: {m['policy_id_version']} [{bracket_content}]")
        
        sources_str = "\n".join(sources_str_parts)

        if self.dry_run:
            return prompt, "", SKIPPED, sources_str, ""

        try:
            raw_response = self._call_llm(prompt)
            try:
                parsed_json = json.loads(raw_response)
                decision = parsed_json.get("final_decision", PARSE_ERROR)
                reasoning = parsed_json.get("reasoning", "")
            except json.JSONDecodeError:
                logger.warning(
                    "%s | %s — failed to decode JSON. Raw: %.100s",
                    row.get("Applicant_ID", "?"),
                    exp_id,
                    raw_response,
                )
                decision = PARSE_ERROR
                reasoning = ""

            if decision == PARSE_ERROR:
                logger.warning(
                    "%s | %s — could not extract valid decision. Raw: %.100s",
                    row.get("Applicant_ID", "?"),
                    exp_id,
                    raw_response,
                )
            return prompt, raw_response, decision, sources_str, reasoning
        except RuntimeError as exc:
            logger.error(
                "%s | %s — API error: %s",
                row.get("Applicant_ID", "?"),
                exp_id,
                exc,
            )
            return prompt, f"ERROR: {exc}", "API_ERROR", sources_str, ""

    # Public API

    def run_all(self, df: pd.DataFrame, output_csv_path: Optional[str] = None) -> pd.DataFrame:
        """Run all four experiments across every row in ``df``.

        Iterates rows in a single pass using ``tqdm`` for progress
        visibility.  For each row, all four experiment calls are made
        sequentially (to respect API rate limits).

        The returned DataFrame preserves all original columns from ``df``
        and appends four new decision columns.

        Args:
            df: Ground Truth DataFrame (output of ``apply_ground_truth``),
                must contain ``PTI``, ``CTI``, ``LGV``, ``FROZEN_PD``,
                and ``Applicant_Code``.

        Returns:
            A copy of ``df`` with additional columns::

                EXP_001_Decision  EXP_002_Decision
                EXP_003_Decision  EXP_004_Decision

        Raises:
            ValueError: If any required metric column is missing from ``df``.
        """
        required_cols = ["PTI", "CTI", "LGV", "FROZEN_PD"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"ExperimentOrchestrator.run_all() — missing columns: {missing}"
            )

        exp_ids = ["EXP_001", "EXP_002", "EXP_003"]

        # Prepare output DataFrame
        df_out = df.copy()
        for exp_id in exp_ids:
            df_out[f"{exp_id}_Decision"] = pd.Series(dtype=str)
            df_out[f"{exp_id}_Reasoning"] = pd.Series(dtype=str)
            df_out[f"{exp_id}_Retrieved_Policies"] = pd.Series(dtype=str)

        mode_label = "DRY RUN" if self.dry_run else f"API ({self.model_name})"
        logger.info(
            "Starting %d experiments × %d records [%s] …",
            len(exp_ids),
            len(df),
            mode_label,
        )

        title_map = {
            "EXP_001": "[2] EXPERIMENT 001: BASELINE (Combined Context)",
            "EXP_002": "[3] EXPERIMENT 002: CONSERVATIVE PERSONA (Combined Context)",
            "EXP_003": "[4] EXPERIMENT 003: BASELINE TOP-K=4 (Combined Context)",
        }

        for idx, row in tqdm(
            df_out.iterrows(),
            total=len(df),
            desc="LLM Experiments",
            unit="borrower",
            ncols=90,
        ):
            app_id = row.get("Applicant_ID", "N/A")
            hist_id = row.get("Historical_Data_ID", "N/A")
            
            audit_chunks = [
                f"======================================================================",
                f"BORROWER: {app_id} | Historical ID: {hist_id}",
                f"======================================================================",
                f"[1] QUANTITATIVE BASELINE & GROUND TRUTH",
                f"Income: {row.get('Income', 0):.2f} | Loan: {row.get('Loan_Amount', 0):.2f} | Annuity: {row.get('Annuity_Payment', 0):.2f} | Goods Price: {row.get('Goods_Price', 0):.2f}",
                f"Metrics -> PTI: {row.get('PTI', 0)}% | CTI: {row.get('CTI', 0)}x | LGV: {row.get('LGV', 0)}% | PD: {row.get('FROZEN_PD', 0)}%",
                f"Calculated Risk Tiers -> PTI: {row.get('PTI_RISK', 'N/A')} | CTI: {row.get('CTI_RISK', 'N/A')} | LGV: {row.get('LGV_RISK', 'N/A')} | PD: {row.get('PD_RISK', 'N/A')}",
                f"Final Ground Truth Decision: {row.get('GROUND_TRUTH_DECISION', 'N/A')}\n"
            ]

            for exp_id in exp_ids:
                prompt, raw_res, decision, sources_str, reasoning = self._run_single_experiment(exp_id, row)
                df_out.at[idx, f"{exp_id}_Decision"] = decision
                df_out.at[idx, f"{exp_id}_Reasoning"] = reasoning
                df_out.at[idx, f"{exp_id}_Retrieved_Policies"] = sources_str
                
                audit_chunks.extend([
                    f"----------------------------------------------------------------------",
                    f"{title_map.get(exp_id, exp_id)}",
                    f"=> RETRIEVED SOURCES (Top-K={self._exp_top_k[exp_id]}):\n{sources_str}\n",
                    f">>> EXACT PROMPT SENT TO LLM:\n{prompt}\n",
                    f"<<< RAW LLM RESPONSE (JSON):\n{raw_res}\n",
                    f"<<< REASONING:\n{reasoning}\n",
                    f"=> EXTRACTED DECISION: {decision}\n"
                ])
            
            if self.audit_dir:
                filepath = os.path.join(self.audit_dir, f"{app_id}_audit.txt")
                with open(filepath, "w", encoding="utf-8") as af:
                    af.write("\n".join(audit_chunks) + "\n")
                    
            if not self.dry_run:
                time.sleep(2)
            
            if output_csv_path:
                try:
                    df_out.to_csv(output_csv_path, index=False)
                    _save_excel(df_out, output_csv_path)
                except PermissionError:
                    logger.warning("CSV is currently open in Excel; keeping results in memory and will update once closed.")

        # Log decision distributions per experiment
        for exp_id in exp_ids:
            col = f"{exp_id}_Decision"
            dist = df_out[col].value_counts().to_dict()
            logger.info("  %s distribution: %s", col, dist)

        return df_out
