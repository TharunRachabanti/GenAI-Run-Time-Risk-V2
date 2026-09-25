"""RAG (Retrieval-Augmented Generation) engine for the GenAI Credit-Risk Decision Engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import docx  # python-docx

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

    def retrieve_context(self, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """Return concatenated policy text.

        Args:
            top_k (int): Maximum number of documents to retrieve.

        Returns:
            Tuple[str, List[Dict[str, Any]]]: A tuple containing the concatenated text and list of retrieved metadata.
        """
        matched_texts: List[str] = []
        retrieved_metadata: List[Dict[str, Any]] = []

        rank = 1
        for filename, text in sorted(self.documents.items()):
            policy_id_version = filename.replace('.docx', '')
            header = (
                f"\n{'=' * 70}\n"
                f"Rank {rank}: {policy_id_version}\n"
                f"{'=' * 70}\n"
            )
            matched_texts.append(header + text)
            retrieved_metadata.append({
                "policy_id_version": policy_id_version,
                "rank": rank,
                "text": text
            })
            logger.debug("  Retrieved: %s (Rank %d)", filename, rank)
            if len(matched_texts) == top_k:
                break
            rank += 1

        if not matched_texts:
            logger.warning(
                "No policy documents found. "
                "Available filenames: %s",
                list(self.documents.keys()),
            )
            return "", []

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
