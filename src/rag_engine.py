"""RAG (Retrieval-Augmented Generation) engine for the GenAI Credit-Risk Decision Engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import docx  # python-docx

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class PolicyDocumentParser:
    """Parses .docx policy documents and supports retrieval by policy year.
    
    Attributes:
        docs_dir (Path): Absolute path to the policy documents directory.
        documents (Dict[str, str]): Mapping of filename to extracted full text.
    """

    def __init__(self, docs_dir: str) -> None:
        """Initialise parser and parse all .docx files.

        Args:
            docs_dir (str): Relative or absolute path to policy files directory.

        Raises:
            FileNotFoundError: If docs_dir does not exist.
            RuntimeError: If no .docx files are found.
        """
        self.docs_dir: Path = Path(docs_dir).resolve()
        if not self.docs_dir.exists():
            raise FileNotFoundError(
                f"Policy documents directory not found: {self.docs_dir}"
            )

        self.documents: Dict[str, str] = {}
        self._parse_all()

        if not self.documents:
            raise RuntimeError(
                f"No .docx files found in: {self.docs_dir}"
            )

        logger.info(
            "PolicyDocumentParser ready — %d documents loaded from %s",
            len(self.documents),
            self.docs_dir,
        )

    def _extract_text(self, docx_path: Path) -> str:
        """Extract all paragraph text from a single .docx file.

        Args:
            docx_path (Path): Absolute path to the .docx file.

        Returns:
            str: Full extracted text joined with newlines.

        Raises:
            RuntimeError: If python-docx cannot open or read the file.
        """
        try:
            document = docx.Document(str(docx_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open document {docx_path.name}: {exc}"
            ) from exc

        full_text: List[str] = []
        
        for para in document.paragraphs:
            full_text.append(para.text)
            
        for table in document.tables:
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                full_text.append(" | ".join(row_data))
                
        return "\n".join(full_text)

    def _parse_all(self) -> None:
        """Scan docs_dir and populate the documents dictionary."""
        docx_files = sorted(self.docs_dir.glob("*.docx"))
        if not docx_files:
            # Try case variations on Windows
            docx_files = sorted(self.docs_dir.glob("*.DOCX"))

        for file_path in docx_files:
            try:
                text = self._extract_text(file_path)
                self.documents[file_path.name] = text
                logger.debug(
                    "  Parsed %-55s  (%d chars)", file_path.name, len(text)
                )
            except RuntimeError as exc:
                logger.warning("Skipping %s — %s", file_path.name, exc)

    def retrieve_context(self, top_k: int = 5, borrower_row: Optional[pd.Series] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Return concatenated policy text ranked dynamically by TF-IDF similarity.

        Args:
            top_k (int): Maximum number of documents to retrieve.
            borrower_row (Optional[pd.Series]): Row of metrics to create scoring context.

        Returns:
            Tuple[str, List[Dict[str, Any]]]: A tuple containing the concatenated text and list of retrieved metadata.
        """
        if not self.documents:
            logger.warning("No policy documents found.")
            return "", []

        # Construct query from borrower or fallback
        if borrower_row is not None:
            terms = [
                "amt_income_total", "amt_credit", "amt_annuity", "amt_goods_price",
                "pti", "cti", "lgv", "pd",
                str(borrower_row.get("Income", "")),
                str(borrower_row.get("Loan_Amount", "")),
                str(borrower_row.get("Annuity_Payment", "")),
                str(borrower_row.get("Goods_Price", ""))
            ]
            query = " ".join([t for t in terms if t.strip()])
        else:
            query = "credit risk policy affordability exposure financing value probability of default"

        doc_keys = list(self.documents.keys())
        doc_texts = [self.documents[k] for k in doc_keys]

        # TF-IDF Retrieval
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(doc_texts + [query])
        
        doc_vectors = tfidf_matrix[:-1]
        query_vector = tfidf_matrix[-1]
        similarities = cosine_similarity(query_vector, doc_vectors)[0]

        # Rank documents by highest cosine similarity
        ranked_indices = similarities.argsort()[::-1]
        feature_names = vectorizer.get_feature_names_out()

        matched_texts: List[str] = []
        retrieved_metadata: List[Dict[str, Any]] = []

        rank = 1
        for idx in ranked_indices:
            if rank > top_k:
                break
            
            filename = doc_keys[idx]
            text = doc_texts[idx]
            score = float(similarities[idx])
            
            overlap = doc_vectors[idx].multiply(query_vector).toarray()[0]
            top_term_indices = overlap.argsort()[::-1][:3] # Top 3 terms
            matched_terms = [feature_names[i] for i in top_term_indices if overlap[i] > 0]
            if not matched_terms:
                matched_terms = ["generic match"]

            name_parts = filename.replace('.docx', '').split('_', 2)
            if len(name_parts) >= 3:
                pol_prefix = name_parts[0]
                version_str = name_parts[1]
                policy_id_version = f"{pol_prefix} (v{version_str.replace('-', '.')}) - {name_parts[2].replace('_', ' ')}"
            else:
                pol_prefix = "UNKNOWN"
                version_str = "UNKNOWN"
                policy_id_version = filename.replace('.docx', '')

            if borrower_row is not None:
                if "2026" in version_str:
                    version_context = f"active {version_str.replace('-', '.')} version"
                elif "2025" in version_str:
                    version_context = f"duplicate {version_str.replace('-', '.')} version consuming a Top-K slot"
                else:
                    version_context = f"version {version_str.replace('-', '.')}"

                metric_val = ""
                if "POL-01" in pol_prefix:
                    val = next((borrower_row.get(k) for k in ["PTI", "pti"] if pd.notna(borrower_row.get(k))), "")
                    metric_val = f"Borrower PTI = {val}%"
                elif "POL-02" in pol_prefix:
                    val = next((borrower_row.get(k) for k in ["CTI", "cti"] if pd.notna(borrower_row.get(k))), "")
                    metric_val = f"Borrower CTI = {val}x"
                elif "POL-03" in pol_prefix:
                    val = next((borrower_row.get(k) for k in ["LGV", "lgv"] if pd.notna(borrower_row.get(k))), "")
                    metric_val = f"Borrower LGV = {val}%"
                elif "POL-04" in pol_prefix:
                    val = next((borrower_row.get(k) for k in ["FROZEN_PD", "PD", "pd"] if pd.notna(borrower_row.get(k))), "")
                    metric_val = f"Borrower PD = {val}%"
                else:
                    metric_val = "Unknown metric"

                matched_str = f"[{', '.join(matched_terms)}]"
                reason_str = f"Ranked #{rank} via natural TF-IDF match on {matched_str} for {metric_val} ({version_context})."
            else:
                reason_str = "Matched by semantic similarity."

            
            header = (
                f"\n{'=' * 70}\n"
                f"Rank {rank}: {policy_id_version}\n"
                f"{'=' * 70}\n"
            )
            matched_texts.append(header + text)
            retrieved_metadata.append({
                "policy_id_version": policy_id_version,
                "rank": rank,
                "text": text,
                "score": round(score, 4),
                "reason": reason_str
            })
            logger.debug("  Retrieved: %s (Rank %d, Score: %.4f)", filename, rank, score)
            rank += 1

        context = "\n\n".join(matched_texts)
        logger.info(
            "Retrieved %d document(s) (Top-K=%d request) "
            "(total context: %d chars)",
            len(matched_texts),
            top_k,
            len(context),
        )
        return context, retrieved_metadata

    def list_documents(self) -> List[str]:
        """Return a sorted list of all loaded document filenames.

        Returns:
            List[str]: Sorted list of filename strings.
        """
        return sorted(self.documents.keys())

    def get_document_text(self, filename: str) -> Optional[str]:
        """Return the text of a specific document by its filename.

        Args:
            filename (str): The exact filename key.

        Returns:
            Optional[str]: The document text, or None if not found.
        """
        return self.documents.get(filename)
