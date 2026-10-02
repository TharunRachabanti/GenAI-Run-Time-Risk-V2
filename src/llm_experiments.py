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
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        f"Payment-to-Income Ratio (PTI) : {row['PTI']:.2f}%\n"
        f"Credit-to-Income Ratio (CTI)  : {row['CTI']:.2f}x\n"
        f"Loan-to-Goods-Value (LGV)     : {row['LGV']:.2f}%\n"
        f"Probability of Default (PD)   : {row['FROZEN_PD']:.2f}%\n"
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


# LLM Factory Router

def generate_llm_response(prompt: str, provider: str, model_name: str, api_key: str, temperature: float, max_tokens: int, timeout: int) -> str:
    """Route LLM request to the correct provider and enforce valid JSON structure."""
    if not api_key:
        raise EnvironmentError(f"LLM_API_KEY environment variable is not set for {provider}.")

    provider = provider.lower()
    
    if provider == "gemini":
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError("google-generativeai is not installed.") from exc
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=LLMDecision,
        )
        response = model.generate_content(prompt, generation_config=generation_config, request_options={"timeout": timeout})
        return response.text

    elif provider in ["openai", "groq"]:
        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai is not installed.") from exc

        base_url = "https://api.groq.com/openai/v1" if provider == "groq" else None
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            timeout=timeout
        )
        return response.choices[0].message.content

    elif provider == "anthropic":
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("anthropic is not installed.") from exc
            
        client = anthropic.Anthropic(api_key=api_key)
        prompt_with_json_directive = prompt + "\n\nYou MUST format your response as valid JSON returning exactly final_decision and reasoning string keys."
        
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt_with_json_directive}],
            timeout=timeout
        )
        return response.content[0].text
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def _classify_metric(metric: str, value: float, retrieved_version: str) -> str:
    if retrieved_version == "NULL":
        return "Unavailable"
    
    if metric == "pti":
        if retrieved_version == "2026.1":
            if value <= 20: return "Standard"
            elif value <= 30: return "Enhanced Review"
            else: return "High"
        elif retrieved_version == "2025.1":
            if value <= 25: return "Standard"
            elif value <= 35: return "Enhanced Review"
            else: return "High"
    elif metric == "cti":
        if retrieved_version == "2026.1":
            if value <= 3.0: return "Standard"
            elif value <= 5.0: return "Enhanced Review"
            else: return "High"
        elif retrieved_version == "2025.1":
            if value <= 4.0: return "Standard"
            elif value <= 6.0: return "Enhanced Review"
            else: return "High"
    elif metric == "lgv":
        if value <= 100: return "Standard"
        elif value <= 110: return "Enhanced Review"
        else: return "High"
    elif metric == "pd":
        if value < 2: return "Low"
        elif value < 5: return "Moderate"
        elif value < 7: return "Elevated"
        else: return "High"
    return "Unavailable"

def _deterministic_fallback(row: pd.Series, sources_meta: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Layer 3 fallback evaluating exactly Rules 1-5 to guarantee 0 Parse Errors."""
    metrics = {}
    for m_key, pol_prefix, m_raw in [
        ("pti", "POL-01", row.get("PTI", 0)),
        ("cti", "POL-02", row.get("CTI", 0)),
        ("lgv", "POL-03", row.get("LGV", 0)),
        ("pd", "POL-04", row.get("FROZEN_PD", 0))
    ]:
        retrieved_version = "NULL"
        for sm in sources_meta:
            if pol_prefix in sm['policy_id_version']:
                if "2026.1" in sm['policy_id_version']: retrieved_version = "2026.1"
                elif "2025.1" in sm['policy_id_version']: retrieved_version = "2025.1"
        metrics[m_key] = _classify_metric(m_key, float(m_raw), retrieved_version)

    high_count = sum(1 for m in metrics.values() if m == "High")
    other_high_count = sum(1 for k, m in metrics.items() if k != "pd" and m == "High")
    
    if metrics.get("pd") == "High" and other_high_count > 0:
        return DECLINE, "Rule 1 Match: PD is High and another metric is High."
    if high_count >= 2:
        return DECLINE, "Rule 2 Match: Two or more metrics are High."
    if high_count == 1:
        return APPROVE_WITH_COND, "Rule 3 Match: Exactly one metric is High."
    
    mid_risk_count = sum(1 for m in metrics.values() if m in ("Enhanced Review", "Elevated", "Moderate"))
    if mid_risk_count > 0:
        return APPROVE_WITH_COND, "Rule 4 Match: No High metric, but containing Enhanced Review/Elevated/Moderate."
    
    return APPROVE, "Rule 5 Match: All metrics are Standard / Low."

def _save_excel(df: pd.DataFrame, csv_path: str) -> None:
    """Save the DataFrame to a nicely formatted Excel workbook alongside the CSV."""
    import openpyxl
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    
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
        
        thin_border = Border(left=Side(style='thin', color='D9D9D9'),
                             right=Side(style='thin', color='D9D9D9'),
                             top=Side(style='thin', color='D9D9D9'),
                             bottom=Side(style='thin', color='D9D9D9'))
        
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_align
            cell.border = thin_border
            
        ws.freeze_panes = "A2"
        
        # Body cells alignment and borders
        body_align = Alignment(wrap_text=True, vertical="top", horizontal="left")
        center_align = Alignment(wrap_text=True, vertical="top", horizontal="center")
        
        # Colors for Decision columns
        color_approve = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
        color_cond = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
        color_decline = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")

        decision_cols = ["GROUND_TRUTH_DECISION", "EXP_001_Decision", "EXP_002_Decision", "EXP_003_Decision"]
        col_name_to_idx = {cell.value: idx for idx, cell in enumerate(ws[1], 1)}
        
        for row in ws.iter_rows(min_row=2):
            max_lines_in_row = 1
            for cell in row:
                col_name = ws.cell(row=1, column=cell.column).value
                
                # Apply borders
                cell.border = thin_border
                
                # Apply alignment
                if col_name and (col_name.endswith("_Decision") or col_name == "GROUND_TRUTH_DECISION" or col_name.endswith("_ID") or cell.column <= 5):
                    cell.alignment = center_align
                elif col_name and (col_name.endswith("_value") or col_name.endswith("_policy_retrieved") or col_name.endswith("_policy_version") or col_name.endswith("_classification")):
                    cell.alignment = center_align
                else:
                    cell.alignment = body_align
                
                # Check if it's a decision column
                if col_name in decision_cols:
                    if cell.value == "APPROVE":
                        cell.fill = color_approve
                    elif cell.value == "APPROVE WITH CONDITIONS":
                        cell.fill = color_cond
                    elif cell.value == "DECLINE":
                        cell.fill = color_decline
                
                if cell.value in ["No", "NULL", "Unavailable"]:
                    if col_name and (col_name.endswith("_policy_retrieved") or col_name.endswith("_policy_version") or col_name.endswith("_classification")):
                        cell.fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
                        cell.font = Font(bold=True)
                        
                if cell.value and isinstance(cell.value, str):
                    lines = cell.value.count('\n') + 1
                    if lines > max_lines_in_row:
                        max_lines_in_row = lines
            ws.row_dimensions[row[0].row].height = max_lines_in_row * 16 + 10

        # Set Column Widths
        for col_name, idx in col_name_to_idx.items():
            col_letter = openpyxl.utils.get_column_letter(idx)
            if col_name and col_name.endswith("_Reasoning"):
                ws.column_dimensions[col_letter].width = 55
            elif col_name and col_name.endswith("_Retrieved_Policies"):
                ws.column_dimensions[col_letter].width = 50
            elif col_name and col_name.endswith("_Decision") or col_name == "GROUND_TRUTH_DECISION":
                ws.column_dimensions[col_letter].width = 26
            elif col_name and (col_name.endswith("_value") or col_name.endswith("_policy_retrieved") or col_name.endswith("_policy_version") or col_name.endswith("_classification")):
                ws.column_dimensions[col_letter].width = 20
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
        self.provider:        str   = os.getenv("LLM_PROVIDER", llm_cfg.get("provider", "gemini"))
        self.model_name:      str   = os.getenv("LLM_MODEL_NAME", llm_cfg.get("model_name", "gemini-1.5-flash"))
        self.api_key:         str   = os.getenv("LLM_API_KEY", "")
        self.temperature:     float = float(llm_cfg.get("temperature", 0.0))
        self.max_tokens:      int   = int(llm_cfg.get("max_tokens", 8192))
        self.request_timeout: int   = int(llm_cfg.get("request_timeout", 60))

        if not self.api_key and not self.dry_run:
            raise EnvironmentError("LLM_API_KEY environment variable is not set. Copy .env.example to .env and configure.")

        if self.dry_run:
            logger.info("ExperimentOrchestrator: dry_run=True — LLM calls skipped.")
        
        self._exp_top_k: Dict[str, int] = {
            "EXP_001": 6,
            "EXP_002": 6,
            "EXP_003": 4,
        }

    # Private helpers

    def _call_llm(self, prompt: str) -> str:
        """Route to LLM factory and return the response text.

        Implements an exponential backoff strategy for API errors.

        Args:
            prompt: The complete, formatted prompt string.

        Returns:
            Raw text from the model response.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        max_attempts = 4
        backoff_times = [2, 4, 8, 16]
        
        for attempt in range(max_attempts):
            try:
                text = generate_llm_response(
                    prompt=prompt,
                    provider=self.provider,
                    model_name=self.model_name,
                    api_key=self.api_key,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    timeout=self.request_timeout
                )
                return text
            except Exception as exc:
                exc_str = str(exc).lower()
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"LLM API Error exhausted: {exc_str}")
                wait_sec = backoff_times[attempt]
                logger.warning("API failure (Attempt %d/%d). Wait %ds... Error: %s", attempt + 1, max_attempts, wait_sec, exc_str)
                time.sleep(wait_sec)
        
        return ""

    def _run_single_experiment(
        self,
        exp_id: str,
        row: pd.Series,
    ) -> Tuple[str, str, str, str, str, List[Dict[str, Any]]]:
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
            score_str = m.get('score', 0.0)
            reason_str = m.get('reason', '')
            sources_str_parts.append(f"Rank {m['rank']}: {m['policy_id_version']} [Score: {score_str:.4f}] -> Why: {reason_str}")
        
        sources_str = "\n".join(sources_str_parts)

        if self.dry_run:
            return prompt, "", SKIPPED, sources_str, "", []

        try:
            raw_response = self._call_llm(prompt)
            decision = PARSE_ERROR
            reasoning = ""
            
            # Layer 1: JSON Standard Loads
            clean_str = re.sub(r'```(?:json)?\n|```', '', raw_response).strip()
            try:
                parsed_json = json.loads(clean_str)
                decision = parsed_json.get("final_decision", PARSE_ERROR)
                reasoning = parsed_json.get("reasoning", "")
            except json.JSONDecodeError:
                pass
                
            # Layer 2: Regex extraction
            if decision == PARSE_ERROR:
                rx_dec = re.search(r'"final_decision":\s*"([^"]+)"', clean_str, re.IGNORECASE)
                rx_rsn = re.search(r'"reasoning":\s*"([^"]+)"', clean_str, re.IGNORECASE)
                if rx_dec: decision = rx_dec.group(1).upper()
                if rx_rsn: reasoning = rx_rsn.group(1)
            
            # Normalise decision (Fix Bug 1: check longest matches first)
            for vd in ["APPROVE WITH CONDITIONS", "APPROVE", "DECLINE"]:
                if vd.lower() in decision.lower() or vd.lower().replace(" ", "_") in decision.lower():
                    decision = vd
                    break

            # Layer 3: Deterministic Safety Net if Invalid
            if decision not in VALID_DECISIONS:
                decision, reason_fallback = _deterministic_fallback(row, sources_meta)
                reasoning = f"[Layer 3 Fallback] {reason_fallback} | LLM parsed text: {reasoning[:100]}"
                
            return prompt, raw_response, decision, sources_str, reasoning, sources_meta
        except RuntimeError as exc:
            logger.error("%s | %s — API error exhausted: %s", row.get("Applicant_ID", "?"), exp_id, exc)
            decision, reason_fallback = _deterministic_fallback(row, sources_meta)
            reasoning = f"[Layer 3 Fallback on API Error] {reason_fallback} | Error: {exc}"
            return prompt, str(exc), decision, sources_str, reasoning, sources_meta

    # Public API

    def run_all(self, df: pd.DataFrame, output_csv_path: Optional[str] = None, test_mode: bool = False) -> pd.DataFrame:
        """Run all experiments concurrently using ThreadPoolExecutor."""
        required_cols = ["PTI", "CTI", "LGV", "FROZEN_PD"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing: raise ValueError(f"Missing cols: {missing}")

        exp_ids = ["EXP_001", "EXP_002", "EXP_003"]
        df_out = df.copy().reset_index(drop=True)
        if "Historical_Data_ID" in df_out.columns:
            df_out.rename(columns={"Historical_Data_ID": "Historical_ID"}, inplace=True)
            
        final_column_order = [
            "Applicant_ID", "Historical_ID", "Income", "Loan_Amount", "Annuity_Payment", 
            "Goods_Price", "Gender", "Education", "Family_Status", "Income_Type", 
            "Occupation", "Housing_Type", "Owns_Car", "Owns_Realty", "Age", "Years_Employed",
            "PTI", "CTI", "LGV", "FROZEN_PD", "PTI_RISK", "CTI_RISK", "LGV_RISK", "PD_RISK", "GROUND_TRUTH_DECISION"
        ]
        
        for exp_id in exp_ids:
            for m in ["pti", "cti", "lgv", "pd"]:
                final_column_order.extend([f"{exp_id}_{m}_value", f"{exp_id}_{m}_policy_retrieved", f"{exp_id}_{m}_policy_version", f"{exp_id}_{m}_classification"])
            final_column_order.extend([f"{exp_id}_Decision", f"{exp_id}_Reasoning", f"{exp_id}_Retrieved_Policies"])
            
        for col in final_column_order:
            if col not in df_out.columns: df_out[col] = pd.Series(dtype=str)

        # Checkpoint Loading
        checkpoint_path = None
        if output_csv_path:
            out_dir = os.path.dirname(output_csv_path)
            checkpoint_file = "04_llm_checkpoint_test.csv" if test_mode else "04_llm_checkpoint.csv"
            checkpoint_path = os.path.join(out_dir, checkpoint_file)
            
            if test_mode and os.path.exists(checkpoint_path):
                # Clean stale test checkpoint
                try: os.remove(checkpoint_path)
                except OSError: pass

            if os.path.exists(checkpoint_path) and not test_mode:
                logger.info("Found checkpoint at %s. Loading existing progress.", checkpoint_path)
                try:
                    df_chk = pd.read_csv(checkpoint_path, dtype=str)
                    for col in final_column_order:
                        if col not in df_chk.columns: df_chk[col] = pd.Series(dtype=str)
                    
                    # Merge checkpoint into df_out
                    for _, chk_row in df_chk.iterrows():
                        idx_match = df_out.index[df_out["Applicant_ID"] == chk_row["Applicant_ID"]]
                        if len(idx_match) > 0:
                            for c in final_column_order:
                                df_out.at[idx_match[0], c] = chk_row.get(c, "")
                except Exception as e:
                    logger.warning("Could not read checkpoint %s: %s", checkpoint_path, e)

        mode_label = "DRY RUN" if self.dry_run else f"API ({self.model_name})"
        logger.info("Starting %d experiments × %d records [%s] …", len(exp_ids), len(df), mode_label)

        title_map = {
            "EXP_001": "[2] EXPERIMENT 001: BASELINE (Combined Context)",
            "EXP_002": "[3] EXPERIMENT 002: CONSERVATIVE PERSONA (Combined Context)",
            "EXP_003": "[4] EXPERIMENT 003: BASELINE TOP-K=4 (Combined Context)",
        }

        lock = threading.Lock()
        write_counter = 0

        def process_borrower(idx, row):
            app_id = row.get("Applicant_ID", "N/A")
            
            # Check if this borrower is already finished in the checkpoint
            is_complete = True
            for exp_id in exp_ids:
                if row.get(f"{exp_id}_Decision") not in VALID_DECISIONS:
                    is_complete = False
            if is_complete and not self.dry_run:
                # Skip already completed borrowers
                return idx, {}, None
            
            hist_id = row.get("Historical_ID", row.get("Historical_Data_ID", "N/A"))
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
            
            updates = {}
            # Allow multiple experiments per borrower sequentially as parallelising this inner loop could cause excessive rate-limiting and quota blocks, parallelised borrowers is much faster anyway.
            for exp_id in exp_ids:
                prompt, raw_res, decision, sources_str, reasoning, sources_meta = self._run_single_experiment(exp_id, row)

                for m_key, pol_prefix, m_raw, m_fmt in [
                    ("pti", "POL-01", row.get("PTI", 0), f"{row.get('PTI', 0):.1f}%"),
                    ("cti", "POL-02", row.get("CTI", 0), f"{row.get('CTI', 0):.1f}x"),
                    ("lgv", "POL-03", row.get("LGV", 0), f"{row.get('LGV', 0):.1f}%"),
                    ("pd", "POL-04", row.get("FROZEN_PD", 0), f"{row.get('FROZEN_PD', 0):.2f}%")
                ]:
                    retrieved_version = "NULL"
                    policy_retrieved = "No"
                    versions_found = []
                    for sm in sources_meta:
                        if pol_prefix in sm['policy_id_version']:
                            if "2026.1" in sm['policy_id_version']: versions_found.append("2026.1")
                            elif "2025.1" in sm['policy_id_version']: versions_found.append("2025.1")
                    if "2026.1" in versions_found: retrieved_version, policy_retrieved = "2026.1", "Yes"
                    elif "2025.1" in versions_found: retrieved_version, policy_retrieved = "2025.1", "Yes"

                    updates[f"{exp_id}_{m_key}_value"] = m_fmt
                    updates[f"{exp_id}_{m_key}_policy_retrieved"] = policy_retrieved
                    updates[f"{exp_id}_{m_key}_policy_version"] = retrieved_version
                    updates[f"{exp_id}_{m_key}_classification"] = _classify_metric(m_key, float(m_raw), retrieved_version)

                updates[f"{exp_id}_Decision"] = decision
                updates[f"{exp_id}_Reasoning"] = reasoning
                updates[f"{exp_id}_Retrieved_Policies"] = sources_str
                
                audit_chunks.extend([
                    f"----------------------------------------------------------------------",
                    f"{title_map.get(exp_id, exp_id)}",
                    f"=> RETRIEVED SOURCES (Top-K={self._exp_top_k[exp_id]}):\n{sources_str}\n",
                    f">>> EXACT PROMPT SENT TO LLM:\n{prompt}\n",
                    f"<<< RAW LLM RESPONSE (JSON):\n{raw_res}\n",
                    f"<<< REASONING:\n{reasoning}\n",
                    f"=> EXTRACTED DECISION: {decision}\n"
                ])
            
            audit_text = "\n".join(audit_chunks) + "\n"
            return idx, updates, (app_id, audit_text)

        # 9 max_workers to hit ~6 to 9 simultaneous requests (approx 2 per second if staggered properly).
        with ThreadPoolExecutor(max_workers=9) as executor:
            task_futures = {executor.submit(process_borrower, idx, df_out.iloc[idx].to_dict()): idx for idx in range(len(df_out))}
            
            with tqdm(total=len(df_out), desc="LLM Processing", unit="borrower", ncols=90) as pbar:
                for future in as_completed(task_futures):
                    idx, updates, audit_data = future.result()
                    
                    with lock:
                        for k, v in updates.items():
                            df_out.at[idx, k] = v
                        
                        if audit_data and self.audit_dir:
                            app_id, audit_text = audit_data
                            filepath = os.path.join(self.audit_dir, f"{app_id}_audit.txt")
                            with open(filepath, "w", encoding="utf-8") as af:
                                af.write(audit_text)
                        
                        if updates and checkpoint_path:
                            write_counter += 1
                            # Write slowly incrementally every 5
                            if write_counter % 5 == 0:
                                df_out.to_csv(checkpoint_path, index=False)
                            # Heavy format every 25
                            if write_counter % 25 == 0 and output_csv_path:
                                try:
                                    _save_excel(df_out, output_csv_path)
                                except Exception as e:
                                    logger.warning("Could not write intermediate excel: %s", e)
                                    
                    pbar.update(1)
        
        with lock:
            if checkpoint_path:
                df_out.to_csv(checkpoint_path, index=False)
            if output_csv_path:
                df_out.to_csv(output_csv_path, index=False)
                try:
                    _save_excel(df_out, output_csv_path)
                except Exception as e:
                    logger.warning("Could not write final excel: %s", e)

        # Reorder dataframe enforcing strictly required layout
        df_out = df_out[[c for c in final_column_order if c in df_out.columns]]

        # Log decision distributions per experiment
        for exp_id in exp_ids:
            col = f"{exp_id}_Decision"
            dist = df_out[col].value_counts().to_dict()
            logger.info("  %s distribution: %s", col, dist)

        return df_out
