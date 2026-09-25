"""Data pipeline for the GenAI Credit-Risk Decision Engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must be non-null to qualify as an experimental record 
_REQUIRED_CLEAN_COLS: List[str] = [
    "SK_ID_CURR",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "TARGET",
]

# The exact 19 columns kept after computation (before rename) 
_STRICT_COLUMNS: List[str] = [
    "SK_ID_CURR",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
    "NAME_HOUSING_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "AGE_YEARS",
    "EMPLOYED_YEARS",
    "PTI",
    "CTI",
    "LGV",
    "FROZEN_PD",
]

# Rename map 
_RENAME_MAP: Dict[str, str] = {
    "SK_ID_CURR":          "Historical_Data_ID",
    "AMT_INCOME_TOTAL":    "Income",
    "AMT_CREDIT":          "Loan_Amount",
    "AMT_ANNUITY":         "Annuity_Payment",
    "AMT_GOODS_PRICE":     "Goods_Price",
    "CODE_GENDER":         "Gender",
    "NAME_EDUCATION_TYPE": "Education_Level",
    "NAME_FAMILY_STATUS":  "Family_Status",
    "NAME_INCOME_TYPE":    "Income_Type",
    "OCCUPATION_TYPE":     "Occupation",
    "NAME_HOUSING_TYPE":   "Housing_Type",
    "FLAG_OWN_CAR":        "Owns_Car",
    "FLAG_OWN_REALTY":     "Owns_Realty",
    "AGE_YEARS":           "Age",
    "EMPLOYED_YEARS":      "Years_Employed",
}

_ART01_COLS: List[str] = [
    "Applicant_ID", "Historical_Data_ID",
    "Income", "Loan_Amount", "Annuity_Payment", "Goods_Price",
    "Gender", "Education_Level", "Family_Status", "Income_Type",
    "Occupation", "Housing_Type", "Owns_Car", "Owns_Realty",
    "Age", "Years_Employed"
]
_ART02_COLS: List[str] = _ART01_COLS + ["PTI", "CTI", "LGV", "FROZEN_PD"]

def load_raw(raw_csv_path: str) -> pd.DataFrame:
    """Load raw training application CSV without dropping rows.

    Args:
        raw_csv_path (str): Path to application_train.csv.

    Returns:
        pd.DataFrame: Parsed DataFrame with derived age and employment data appended.
        
    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required raw columns are absent.
        RuntimeError: If the file cannot be parsed.
    """
    raw_path = Path(raw_csv_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    logger.info("Loading raw data from %s …", raw_path)
    try:
        df = pd.read_csv(raw_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV: {exc}") from exc

    logger.info("  Raw shape: %d rows × %d cols", *df.shape)

    missing_cols = [c for c in _REQUIRED_CLEAN_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Raw CSV missing required columns: {missing_cols}")

    df["AGE_YEARS"]      = (df["DAYS_BIRTH"].abs() / 365.25).round(2)
    df["EMPLOYED_YEARS"] = df["DAYS_EMPLOYED"].apply(
        lambda x: round(abs(x) / 365.25, 2) if pd.notna(x) and x < 0 else 0.0
    )
    return df


def run_phase1(cfg: Dict[str, Any], pd_model_instance: Any) -> pd.DataFrame:
    """Execute complete Phase 1 pipeline processing exactly 500 rows.

    Args:
        cfg (Dict[str, Any]): Settings dictionary from configuration file.
        pd_model_instance (Any): Initialised (unfitted) PDModel.

    Returns:
        pd.DataFrame: DataFrame containing experiment rows and calculated columns.
    """
    paths    = cfg["paths"]
    data_cfg = cfg["data"]
    gt_cfg   = cfg["ground_truth"]
    proc_dir = Path(paths["processed_dir"])

    df_raw = load_raw(raw_csv_path=paths["raw_data"])

    logger.info("Training PD model on full raw dataset (%d rows) …", len(df_raw))
    train_meta = pd_model_instance.train(df_raw)
    logger.info("PD model training complete → AUC: %.4f", train_meta["test_auc"])

    pool = df_raw[df_raw["TARGET"] == 0].copy()
    pool = pool.dropna(subset=_REQUIRED_CLEAN_COLS).reset_index(drop=True)
    logger.info("Clean pool: %d rows (TARGET==0, all required cols non-null).", len(pool))

    n = data_cfg["sample_size"]
    seed = data_cfg["random_seed"]
    if len(pool) < n:
        raise ValueError(f"Pool has only {len(pool)} rows but {n} requested.")

    df = pool.sample(n=n, random_state=seed).reset_index(drop=True)
    logger.info("Sampled %d experimental records.", len(df))

    logger.info("Scoring FROZEN_PD …")
    pd_scores = pd_model_instance.predict_pd(df)
    df["FROZEN_PD"] = pd_scores.round(2).values

    df["PTI"] = ((df["AMT_ANNUITY"]  / df["AMT_INCOME_TOTAL"]) * 100).round(2)
    df["CTI"] = ( df["AMT_CREDIT"]   / df["AMT_INCOME_TOTAL"]).round(2)
    df["LGV"] = ((df["AMT_CREDIT"]   / df["AMT_GOODS_PRICE"])  * 100).round(2)

    logger.info(
        "  PTI [mean=%.2f min=%.2f max=%.2f]  "
        "CTI [mean=%.2f min=%.2f max=%.2f]  "
        "LGV [mean=%.2f min=%.2f max=%.2f]  "
        "FROZEN_PD [mean=%.2f min=%.2f max=%.2f]",
        df["PTI"].mean(), df["PTI"].min(), df["PTI"].max(),
        df["CTI"].mean(), df["CTI"].min(), df["CTI"].max(),
        df["LGV"].mean(), df["LGV"].min(), df["LGV"].max(),
        df["FROZEN_PD"].mean(), df["FROZEN_PD"].min(), df["FROZEN_PD"].max(),
    )

    df = df[_STRICT_COLUMNS].copy()
    assert list(df.columns) == _STRICT_COLUMNS, \
        f"Column mismatch after strict slice: {list(df.columns)}"
    logger.info("Strict column slice applied — %d columns retained.", len(df.columns))

    df = df.rename(columns=_RENAME_MAP)

    df.insert(0, "Applicant_ID", [f"P{i+1:03d}" for i in range(len(df))])

    art01_path = proc_dir / paths["artifact_01_records"]
    art01_path.parent.mkdir(parents=True, exist_ok=True)
    df[_ART01_COLS].to_csv(art01_path, index=False)
    logger.info(
        "  ✔ Artifact 01 saved: %s  (%d rows × %d cols)",
        art01_path, len(df), len(_ART01_COLS),
    )

    art02_path = proc_dir / paths["artifact_02_features"]
    df[_ART02_COLS].to_csv(art02_path, index=False)
    logger.info(
        "  ✔ Artifact 02 saved: %s  (%d rows × %d cols)",
        art02_path, len(df), len(_ART02_COLS),
    )

    df_gt = _apply_ground_truth(df.copy(), gt_cfg)
    art03_path = proc_dir / paths["artifact_03_ground_truth"]
    try:
        df_gt.to_csv(art03_path, index=False)
        logger.info(
            "  ✔ Artifact 03 saved: %s  (%d rows × %d cols)",
            art03_path, len(df_gt), len(df_gt.columns),
        )
    except PermissionError:
        logger.warning(
            "  ⚠ PermissionError: %s is locked bypass mode active.", art03_path
        )

    gt_dist = df_gt["GROUND_TRUTH_DECISION"].value_counts().to_dict()
    logger.info("Ground Truth Decision distribution: %s", gt_dist)

    return df_gt


# Ground Truth helpers

def _label_metric(value: float, standard_max: float, enhanced_max: float) -> str:
    """Map a numeric metric to risk tier.

    Args:
        value (float): The computed metric value.
        standard_max (float): Inclusive upper bound for 'Standard'.
        enhanced_max (float): Inclusive upper bound for 'Enhanced Review'.

    Returns:
        str: Risk tier label ('Standard', 'Enhanced Review', or 'High').
    """
    if value <= standard_max:
        return "Standard"
    elif value <= enhanced_max:
        return "Enhanced Review"
    return "High"


def _label_pd(pd_pct: float, low_max: float, moderate_max: float, elevated_max: float) -> str:
    """Map a PD percentage to a risk tier label.

    Args:
        pd_pct (float): Probability of Default (%).
        low_max (float): Exclusive upper bound for 'Low'.
        moderate_max (float): Inclusive upper bound for 'Moderate'.
        elevated_max (float): Inclusive upper bound for 'Elevated'.

    Returns:
        str: Risk tier label ('Low', 'Moderate', 'Elevated', or 'High').
    """
    if pd_pct < low_max:
        return "Low"
    elif pd_pct <= moderate_max:
        return "Moderate"
    elif pd_pct <= elevated_max:
        return "Elevated"
    return "High"


def _derive_decision(row: pd.Series) -> str:
    """Derive Ground Truth credit decision from risk tier labels.

    Args:
        row (pd.Series): DataFrame row containing risk tier labels.

    Returns:
        str: Outcome ('APPROVE', 'APPROVE WITH CONDITIONS', or 'DECLINE').
    """
    ratings = [row["PTI_RISK"], row["CTI_RISK"], row["LGV_RISK"], row["PD_RISK"]]
    high_count = ratings.count('High')
    review_count = ratings.count('Enhanced Review') + ratings.count('Moderate') + ratings.count('Elevated')
    
    if high_count >= 2:
        return 'DECLINE'
    elif high_count == 1 or review_count >= 1:
        return 'APPROVE WITH CONDITIONS'
    else:
        return 'APPROVE'


def _apply_ground_truth(df: pd.DataFrame, thresholds: Dict[str, Any]) -> pd.DataFrame:
    """Append risk tier labels and ground truth decisions to DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing PTI, CTI, LGV, and FROZEN_PD.
        thresholds (Dict[str, Any]): Ground truth thresholds dictionary from configuration.

    Returns:
        pd.DataFrame: DataFrame augmented with risk labels and GT decision.

    Raises:
        ValueError: If required metric columns are missing.
    """
    required = ["PTI", "CTI", "LGV", "FROZEN_PD"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"_apply_ground_truth() — missing columns: {missing}")

    pti = thresholds["pti"]
    cti = thresholds["cti"]
    lgv = thresholds["lgv"]
    pd_ = thresholds["pd"]

    df["PTI_RISK"] = df["PTI"].apply(
        lambda v: _label_metric(v, pti["standard"], pti["enhanced"])
    )
    df["CTI_RISK"] = df["CTI"].apply(
        lambda v: _label_metric(v, cti["standard"], cti["enhanced"])
    )
    df["LGV_RISK"] = df["LGV"].apply(
        lambda v: _label_metric(v, lgv["standard"], lgv["enhanced"])
    )
    df["PD_RISK"] = df["FROZEN_PD"].apply(
        lambda v: _label_pd(v, pd_["low_max"], pd_["moderate_max"], pd_["elevated_max"])
    )
    df["GROUND_TRUTH_DECISION"] = df.apply(_derive_decision, axis=1)

    for col in ["PTI_RISK", "CTI_RISK", "LGV_RISK", "PD_RISK"]:
        logger.info("  %s: %s", col, df[col].value_counts().to_dict())

    return df


if __name__ == "__main__":
    import yaml
    import sys
    import os
    from pathlib import Path

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Allow importing from root src
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.pd_model import PDModel

    logger.info("Executing Standalone Data Pipeline...")
    
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    pd_cfg = cfg["pd_model"]
    
    model_path = str(Path(paths["model_output_dir"]) / paths["pd_model_filename"])
    
    pd_model_instance = PDModel(
        features=cfg["data"]["model_features"],
        model_path=model_path,
        random_state=pd_cfg["random_state"],
        max_iter=pd_cfg["max_iter"],
        test_size=pd_cfg["test_size"],
    )

    df_ground_truth = run_phase1(cfg, pd_model_instance)
    logger.info("Standalone execution completed gracefully without LLM.")
