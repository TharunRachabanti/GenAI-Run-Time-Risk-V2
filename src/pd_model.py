"""Logistic Regression Probability-of-Default (PD) model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PDModel:
    """Logistic Regression model that produces a Probability of Default (PD).

    Attributes:
        features (List[str]): Ordered list of feature column names.
        model_path (Path): Path where the serialised pipeline is stored.
        pipeline (Optional[Pipeline]): The fitted sklearn Pipeline.
        random_state (int): Random seed for reproducibility.
        max_iter (int): Maximum iterations for the LR solver.
        test_size (float): Fraction of training data reserved for AUC validation.
    """

    def __init__(
        self,
        features: List[str],
        model_path: str,
        random_state: int = 42,
        max_iter: int = 1000,
        test_size: float = 0.2,
    ) -> None:
        """Initialise the PDModel.

        Args:
            features (List[str]): Column names of the input features.
            model_path (str): File path to save/load the model.
            random_state (int): Seed for train/test split and LR solver.
            max_iter (int): Solver iteration limit for LogisticRegression.
            test_size (float): Proportion of data held out for validation metrics.
        """
        self.features: List[str] = features
        self.model_path: Path = Path(model_path)
        self.random_state: int = random_state
        self.max_iter: int = max_iter
        self.test_size: float = test_size
        self.pipeline: Optional[Pipeline] = None

    def _validate_columns(self, df: pd.DataFrame, context: str = "") -> None:
        """Raise ValueError if any required feature column is absent.

        Args:
            df (pd.DataFrame): DataFrame to validate.
            context (str): Short label for the error message.

        Raises:
            ValueError: If one or more feature columns are missing.
        """
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"PDModel [{context}] — missing columns: {missing}")

    # Public API

    def train(self, df: pd.DataFrame) -> dict:
        """Fit the PD pipeline on the raw DataFrame (nulls handled internally).

        The ``SimpleImputer`` at step 1 of the pipeline fills every null with
        the column **median**, computed exclusively on the training split to
        prevent data leakage.  No rows are dropped before training.

        After fitting, the pipeline is saved to ``self.model_path``.

        Args:
            df (pd.DataFrame): Raw DataFrame containing self.features and TARGET.

        Returns:
            dict: A dictionary containing training metadata.

        Raises:
            ValueError: If required columns are missing.
            RuntimeError: If the model fails to save to disk.
        """
        self._validate_columns(df, context="training")
        if "TARGET" not in df.columns:
            raise ValueError("PDModel [training] — 'TARGET' column is required.")

        # Select only the feature columns + TARGET; keep ALL rows (no dropna)
        df_model = df[self.features + ["TARGET"]].copy()

        # Drop rows where TARGET itself is null (rare but possible)
        df_model = df_model.dropna(subset=["TARGET"])

        X = df_model[self.features].values
        y = df_model["TARGET"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        logger.info(
            "PD model — training on %d rows | validation on %d rows",
            len(X_train), len(X_test),
        )

        self.pipeline = Pipeline(
            steps=[
                # Step 1: Impute nulls with column medians (fit on train split only)
                ("imputer", SimpleImputer(strategy="median")),
                # Step 2: Standardise features
                ("scaler", StandardScaler()),
                # Step 3: Logistic Regression
                (
                    "lr",
                    LogisticRegression(
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

        self.pipeline.fit(X_train, y_train)

        # Validation AUC
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_prob))
        logger.info("PD model AUC (held-out): %.4f", auc)

        self.save()
        return {
            "train_size": int(len(X_train)),
            "test_size":  int(len(X_test)),
            "test_auc":   round(auc, 4),
            "model_path": str(self.model_path),
        }

    def predict_pd(self, df: pd.DataFrame) -> pd.Series:
        """Return PD as a banking-style percentage rounded to 2 decimal places.

        Args:
            df (pd.DataFrame): DataFrame containing self.features.

        Returns:
            pd.Series: A pandas Series named FROZEN_PD with float values in % format.

        Raises:
            RuntimeError: If train has not been called and no model is loaded.
            ValueError: If required feature columns are missing.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "PDModel.predict_pd() called before training or loading a model."
            )

        self._validate_columns(df, context="inference")

        # Pass raw feature values; the pipeline's imputer handles any nulls
        X = df[self.features].values
        probs = self.pipeline.predict_proba(X)[:, 1]

        # Convert to percentage, round to 2 d.p. (banking-standard display)
        pd_pct = pd.Series(
            (probs * 100.0).round(2),
            index=df.index,
            name="FROZEN_PD",
        )
        return pd_pct

    def save(self) -> None:
        """Serialise the fitted pipeline to self.model_path.

        Raises:
            RuntimeError: If the pipeline has not been fitted yet.
            IOError: If the file cannot be written.
        """
        if self.pipeline is None:
            raise RuntimeError("Cannot save an untrained PDModel.")

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(self.pipeline, self.model_path)
            logger.info("PDModel saved → %s", self.model_path)
        except IOError as exc:
            raise IOError(
                f"Failed to save PDModel to {self.model_path}: {exc}"
            ) from exc

    def load(self) -> None:
        """Load a previously serialised pipeline from self.model_path.

        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If the loaded object is not a valid Pipeline.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"PDModel file not found: {self.model_path}. "
                "Run the pipeline with --phase1-only first."
            )
        try:
            loaded = joblib.load(self.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load PDModel from {self.model_path}: {exc}"
            ) from exc

        if not isinstance(loaded, Pipeline):
            raise RuntimeError(
                f"Loaded object is not an sklearn Pipeline: {type(loaded)}"
            )
        self.pipeline = loaded
        logger.info("PDModel loaded ← %s", self.model_path)
