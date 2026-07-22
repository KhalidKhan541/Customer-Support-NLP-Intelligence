"""spaCy NER module for customer support ticket entity extraction.

This module provides a NamedEntityExtractor class that uses spaCy's
Named Entity Recognition to identify and aggregate entities across
support tickets, with critical entity flagging and co-occurrence analysis.
"""

import logging
import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    import spacy
    from spacy.language import Language

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

TARGET_ENTITY_TYPES = frozenset(
    {"PERSON", "ORG", "PRODUCT", "DATE", "MONEY", "GPE"}
)

COMPLAINT_KEYWORDS = frozenset(
    {
        "complaint",
        "issue",
        "problem",
        "broken",
        "defective",
        "error",
        "fail",
        "failed",
        "failure",
        "refund",
        "return",
        "damaged",
        "wrong",
        "missing",
        "lost",
        "overcharged",
        "charge",
        "dispute",
    }
)

MONEY_PATTERN = re.compile(
    r"\$[\d,]+(?:\.\d{1,2})?|[\d,]+\s*(?:dollars?|usd)", re.IGNORECASE
)


class NamedEntityExtractor:
    """Extract and aggregate named entities from support tickets.

    Uses spaCy's NER pipeline (en_core_web_sm with blank NER fallback)
    to identify entities of interest: PERSON, ORG, PRODUCT, DATE, MONEY,
    GPE. Provides entity frequency counts, type distributions, critical
    entity flagging, and co-occurrence analysis.

    Attributes:
        nlp: spaCy Language pipeline.
        target_types: Set of entity labels to extract.
    """

    MODEL_NAME = "en_core_web_sm"
    MIN_TEXT_LENGTH = 2
    HIGH_VALUE_THRESHOLD = 100.0

    def __init__(
        self,
        target_types: Optional[Set[str]] = None,
        custom_complaint_keywords: Optional[Set[str]] = None,
    ) -> None:
        """Initialize the NamedEntityExtractor.

        Args:
            target_types: Set of spaCy entity labels to extract.
                Defaults to PERSON, ORG, PRODUCT, DATE, MONEY, GPE.
            custom_complaint_keywords: Additional keywords that indicate
                a complaint context.
        """
        if not SPACY_AVAILABLE:
            raise ImportError(
                "spacy is required. Install with: pip install spacy"
            )

        self.target_types = target_types or set(TARGET_ENTITY_TYPES)
        self.complaint_keywords = COMPLAINT_KEYWORDS | (
            custom_complaint_keywords or set()
        )

        self.nlp = self._load_model()
        logger.info(
            "NamedEntityExtractor initialized. Model: %s, Target types: %s",
            getattr(self.nlp, "meta", {}).get("name", "blank"),
            self.target_types,
        )

    def _load_model(self) -> Language:
        """Load spaCy model with fallback to blank NER pipeline.

        Attempts to load en_core_web_sm. If unavailable, creates a
        blank English pipeline with an NER component.

        Returns:
            Loaded spaCy Language instance.
        """
        try:
            nlp = spacy.load(self.MODEL_NAME)
            logger.info("Loaded spaCy model: %s", self.MODEL_NAME)
            return nlp
        except OSError:
            logger.warning(
                "Model '%s' not found. Creating blank English NER pipeline.",
                self.MODEL_NAME,
            )

        try:
            nlp = spacy.blank("en")
            if "ner" not in nlp.pipe_names:
                nlp.add_pipe("ner")
            logger.info("Blank English NER pipeline created.")
            return nlp
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create blank NER pipeline: {exc}"
            ) from exc

    def _is_complaint_context(self, text: str) -> bool:
        """Determine if the text describes a complaint.

        Args:
            text: Input text to analyze.

        Returns:
            True if complaint keywords are present.
        """
        lower_text = text.lower()
        return any(kw in lower_text for kw in self.complaint_keywords)

    @staticmethod
    def _parse_money_amount(entity_text: str) -> Optional[float]:
        """Extract numeric dollar amount from entity text.

        Args:
            entity_text: Text of a MONEY entity (e.g., '$150.00').

        Returns:
            Parsed float amount, or None if parsing fails.
        """
        cleaned = entity_text.replace("$", "").replace(",", "").strip()
        cleaned = re.sub(r"[^\d.]", "", cleaned)
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def extract_entities(
        self, documents: List[str]
    ) -> Dict[str, Any]:
        """Extract named entities from a list of documents.

        Args:
            documents: List of raw text support tickets.

        Returns:
            Dictionary with keys:
                - 'entities': List of lists, each inner list contains
                  dicts with 'text', 'label', 'start', 'end' for each
                  document.
                - 'entity_counts': Counter of (text, label) tuples
                  across all documents.
                - 'type_distribution': Counter of entity label frequencies.
                - 'n_documents_processed': int.

        Raises:
            ValueError: If documents list is empty.
        """
        if not documents:
            raise ValueError("Cannot extract entities from empty document list.")

        all_entities: List[List[Dict[str, Any]]] = []
        entity_counts: Counter = Counter()
        type_distribution: Counter = Counter()

        for doc_text in documents:
            if doc_text is None:
                all_entities.append([])
                continue

            text = str(doc_text).strip()
            if len(text) < self.MIN_TEXT_LENGTH:
                all_entities.append([])
                continue

            doc = self.nlp(text)
            doc_entities: List[Dict[str, Any]] = []

            for ent in doc.ents:
                if ent.label_ in self.target_types:
                    entity_info = {
                        "text": ent.text.strip(),
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                    doc_entities.append(entity_info)
                    entity_key = (ent.text.strip(), ent.label_)
                    entity_counts[entity_key] += 1
                    type_distribution[ent.label_] += 1

            all_entities.append(doc_entities)

        logger.info(
            "Entity extraction complete. %d documents processed, "
            "%d total entity mentions, %d unique entities.",
            len(documents),
            sum(entity_counts.values()),
            len(entity_counts),
        )

        return {
            "entities": all_entities,
            "entity_counts": entity_counts,
            "type_distribution": type_distribution,
            "n_documents_processed": len(documents),
        }

    def get_entity_summary(
        self,
        documents: List[str],
        top_n: int = 20,
    ) -> Dict[str, Any]:
        """Produce a full entity analysis summary for a document corpus.

        Args:
            documents: List of raw text support tickets.
            top_n: Number of top entities to include in summary.

        Returns:
            Dictionary with keys:
                - 'top_entities': List of (text, label, count) tuples.
                - 'type_distribution': dict mapping label -> count.
                - 'total_unique_entities': int.
                - 'total_mentions': int.
                - 'critical_entities': List of flagged entities.
                - 'co_occurrence': Dict mapping entity pair -> count.
                - 'entity_dataframe': DataFrame with columns
                  (text, label, count).
        """
        extraction = self.extract_entities(documents)
        entity_counts = extraction["entity_counts"]
        type_distribution = extraction["type_distribution"]

        if not entity_counts:
            return {
                "top_entities": [],
                "type_distribution": dict(type_distribution),
                "total_unique_entities": 0,
                "total_mentions": 0,
                "critical_entities": [],
                "co_occurrence": {},
                "entity_dataframe": pd.DataFrame(
                    columns=["text", "label", "count"]
                ),
            }

        sorted_entities = entity_counts.most_common(top_n)
        total_mentions = sum(entity_counts.values())

        critical = self._flag_critical_entities(
            documents, entity_counts, extraction["entities"]
        )

        co_occurrence = self._compute_co_occurrence(extraction["entities"])

        rows = [
            {"text": text, "label": label, "count": count}
            for (text, label), count in entity_counts.most_common()
        ]
        df = pd.DataFrame(rows)

        return {
            "top_entities": [
                (text, label, count) for (text, label), count in sorted_entities
            ],
            "type_distribution": dict(type_distribution),
            "total_unique_entities": len(entity_counts),
            "total_mentions": total_mentions,
            "critical_entities": critical,
            "co_occurrence": co_occurrence,
            "entity_dataframe": df,
        }

    def _flag_critical_entities(
        self,
        documents: List[str],
        entity_counts: Counter,
        per_doc_entities: List[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Identify critical entities: products in complaints, high-value money.

        Args:
            documents: Original document texts.
            entity_counts: Global entity frequency counter.
            per_doc_entities: Per-document entity lists.

        Returns:
            List of critical entity dicts with 'text', 'label', 'reason',
            and 'frequency' keys.
        """
        critical: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str]] = set()

        for doc_text, doc_ents in zip(documents, per_doc_entities):
            if doc_text is None:
                continue
            is_complaint = self._is_complaint_context(str(doc_text))

            for ent in doc_ents:
                key = (ent["text"], ent["label"])
                if key in seen:
                    continue

                reason: Optional[str] = None

                if ent["label"] == "PRODUCT" and is_complaint:
                    reason = "product_mentioned_in_complaint"

                if ent["label"] == "MONEY":
                    amount = self._parse_money_amount(ent["text"])
                    if amount is not None and amount > self.HIGH_VALUE_THRESHOLD:
                        reason = f"high_value_amount_{amount:.2f}"

                if reason:
                    seen.add(key)
                    critical.append(
                        {
                            "text": ent["text"],
                            "label": ent["label"],
                            "reason": reason,
                            "frequency": entity_counts[key],
                        }
                    )

        critical.sort(key=lambda x: x["frequency"], reverse=True)
        logger.info("Flagged %d critical entities.", len(critical))
        return critical

    def _compute_co_occurrence(
        self, per_doc_entities: List[List[Dict[str, Any]]]
    ) -> Dict[Tuple[str, str], int]:
        """Compute entity co-occurrence across documents.

        Counts how often pairs of distinct entities appear in the same
        document, regardless of entity type.

        Args:
            per_doc_entities: Per-document entity lists.

        Returns:
            Dictionary mapping (entity_a, entity_b) -> co-occurrence count.
        """
        co_occurrence: Counter = Counter()

        for doc_ents in per_doc_entities:
            unique_texts = list({e["text"] for e in doc_ents})
            if len(unique_texts) < 2:
                continue

            for ent_a, ent_b in combinations(sorted(unique_texts), 2):
                co_occurrence[(ent_a, ent_b)] += 1

        logger.info(
            "Co-occurrence computed: %d unique entity pairs.",
            len(co_occurrence),
        )
        return dict(co_occurrence)

    def extract_entities_dataframe(
        self, documents: List[str]
    ) -> pd.DataFrame:
        """Extract entities into a flat DataFrame for analysis.

        Each row represents one entity mention in one document.

        Args:
            documents: List of raw text support tickets.

        Returns:
            DataFrame with columns: doc_index, text, label, start, end.
        """
        extraction = self.extract_entities(documents)

        rows: List[Dict[str, Any]] = []
        for doc_idx, doc_ents in enumerate(extraction["entities"]):
            for ent in doc_ents:
                rows.append(
                    {
                        "doc_index": doc_idx,
                        "text": ent["text"],
                        "label": ent["label"],
                        "start": ent["start"],
                        "end": ent["end"],
                    }
                )

        return pd.DataFrame(rows)
