"""
Dual Sentiment Analysis Module
Combines VADER (lexicon-based) and DistilBERT (transformer-based) for robust sentiment scoring.

Provides batch/single analysis, trend detection, alerting, and per-topic breakdown.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SentimentResult:
    """Structured output of a single sentiment analysis."""
    text: str
    label: str                         # positive | negative | neutral
    confidence: float                  # 0.0 – 1.0
    vader_compound: float              # -1.0 – 1.0
    distilbert_score: float            # raw model score
    vader_pos: float = 0.0
    vader_neg: float = 0.0
    vader_neu: float = 0.0
    is_critical: bool = False          # compound < -0.7


@dataclass
class TopicSentiment:
    """Aggregated sentiment statistics for a topic."""
    topic: str
    mean_compound: float
    mean_confidence: float
    count: int
    label_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class TrendPoint:
    """A single point on a sentiment trend line."""
    period: datetime
    mean_compound: float
    mean_confidence: float
    ticket_count: int
    shift_detected: bool = False


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    """Normalise unicode, strip control chars, collapse whitespace."""
    try:
        text = unicodedata.normalize("NFKC", text)
    except Exception:
        logger.debug("Unicode normalisation failed; using raw text")
    text = _CONTROL_CHAR_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def _is_valid(text: str) -> bool:
    """Return True if the text is analyzable (non-empty after cleaning)."""
    return bool(text and len(text.strip()) > 0)


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """Hybrid sentiment analyzer combining VADER and DistilBERT.

    Parameters
    ----------
    distilbert_model:
        HuggingFace model identifier for the transformer pipeline.
    hybrid_alpha:
        Weight given to VADER in the hybrid score (0.0 = pure DistilBERT,
        1.0 = pure VADER).  Default ``0.4`` balances speed and nuance.
    critical_threshold:
        Compound score below which a ticket is flagged as critical.
    device:
        PyTorch device (``"cpu"``, ``"cuda"``, ``"mps"``).  Auto-detected
        when *None*.
    """

    def __init__(
        self,
        distilbert_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        hybrid_alpha: float = 0.4,
        critical_threshold: float = -0.7,
        device: Optional[str] = None,
    ) -> None:
        self.hybrid_alpha = hybrid_alpha
        self.critical_threshold = critical_threshold

        # --- VADER -----------------------------------------------------------
        try:
            import nltk
            nltk.data.find("sentiment/vader_lexicon")
        except LookupError:
            logger.info("Downloading VADER lexicon …")
            import nltk
            nltk.download("vader_lexicon", quiet=True)
        self._vader = SentimentIntensityAnalyzer()

        # --- DistilBERT ------------------------------------------------------
        resolved_device: int = -1
        if device is not None:
            resolved_device = (
                torch.device(device).index
                if torch.cuda.is_available() and "cuda" in device
                else -1
            )
        elif torch.cuda.is_available():
            resolved_device = 0

        logger.info(
            "Loading DistilBERT sentiment pipeline on device %s …",
            resolved_device if resolved_device >= 0 else "cpu",
        )
        self._transformer = pipeline(
            "sentiment-analysis",
            model=distilbert_model,
            device=resolved_device,
            truncation=True,
            max_length=512,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def analyze_single(self, text: str) -> SentimentResult:
        """Analyze a single text snippet.

        Returns a ``SentimentResult`` with combined metrics.
        """
        cleaned = _clean_text(text)
        if not _is_valid(cleaned):
            return SentimentResult(
                text=text,
                label="neutral",
                confidence=0.0,
                vader_compound=0.0,
                distilbert_score=0.0,
            )

        vader_scores = self._vader.polarity_scores(cleaned)
        db_label, db_score = self._run_transformer(cleaned)

        label, confidence = self._hybrid(vader_scores, db_label, db_score)
        is_critical = vader_scores["compound"] < self.critical_threshold

        return SentimentResult(
            text=text,
            label=label,
            confidence=round(confidence, 4),
            vader_compound=round(vader_scores["compound"], 4),
            distilbert_score=round(db_score, 4),
            vader_pos=round(vader_scores["pos"], 4),
            vader_neg=round(vader_scores["neg"], 4),
            vader_neu=round(vader_scores["neu"], 4),
            is_critical=is_critical,
        )

    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> List[SentimentResult]:
        """Analyze a list of texts efficiently.

        Parameters
        ----------
        texts:
            Raw text strings.
        batch_size:
            Number of samples per transformer forward pass.
        show_progress:
            Tqdm progress bar (if tqdm is installed).

        Returns
        -------
        list[SentimentResult]
        """
        if not texts:
            return []

        cleaned = [_clean_text(t) for t in texts]
        valid_mask = [_is_valid(c) for c in cleaned]

        # Gather VADER scores (vectorised for speed)
        vader_compounds = np.zeros(len(texts))
        vader_pos = np.zeros(len(texts))
        vader_neg = np.zeros(len(texts))
        vader_neu = np.zeros(len(texts))
        for i, (c, valid) in enumerate(zip(cleaned, valid_mask)):
            if valid:
                vs = self._vader.polarity_scores(c)
                vader_compounds[i] = vs["compound"]
                vader_pos[i] = vs["pos"]
                vader_neg[i] = vs["neg"]
                vader_neu[i] = vs["neu"]

        # Transformer batch inference
        db_results = self._run_transformer_batch(
            [c for c, v in zip(cleaned, valid_mask) if v],
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # Map results back
        results: List[SentimentResult] = []
        db_idx = 0
        for i, raw in enumerate(texts):
            if not valid_mask[i]:
                results.append(
                    SentimentResult(
                        text=raw,
                        label="neutral",
                        confidence=0.0,
                        vader_compound=0.0,
                        distilbert_score=0.0,
                    )
                )
            else:
                db_label, db_score = db_results[db_idx]
                db_idx += 1
                vader_scores = {
                    "compound": float(vader_compounds[i]),
                    "pos": float(vader_pos[i]),
                    "neg": float(vader_neg[i]),
                    "neu": float(vader_neu[i]),
                }
                label, confidence = self._hybrid(vader_scores, db_label, db_score)
                is_critical = vader_scores["compound"] < self.critical_threshold
                results.append(
                    SentimentResult(
                        text=raw,
                        label=label,
                        confidence=round(confidence, 4),
                        vader_compound=round(vader_scores["compound"], 4),
                        distilbert_score=round(db_score, 4),
                        vader_pos=round(vader_scores["pos"], 4),
                        vader_neg=round(vader_scores["neg"], 4),
                        vader_neu=round(vader_scores["neu"], 4),
                        is_critical=is_critical,
                    )
                )
        return results

    # -----------------------------------------------------------------------
    # Trend analysis
    # -----------------------------------------------------------------------

    @staticmethod
    def analyze_trend(
        df: pd.DataFrame,
        timestamp_col: str = "created_at",
        compound_col: str = "vader_compound",
        confidence_col: str = "confidence",
        freq: str = "D",
        shift_window: int = 3,
        shift_threshold: float = 0.3,
    ) -> List[TrendPoint]:
        """Compute rolling sentiment averages and detect shifts.

        Parameters
        ----------
        df:
            DataFrame with at least *timestamp_col*, *compound_col*, and
            *confidence_col*.
        freq:
            Pandas offset alias for grouping (``"D"`` = daily, ``"W"`` = weekly).
        shift_window:
            Rolling window size (in periods) for detecting sentiment shifts.
        shift_threshold:
            Absolute change in rolling mean that triggers a shift alert.

        Returns
        -------
        list[TrendPoint]
        """
        if df.empty:
            return []

        work = df.copy()
        work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
        work = work.dropna(subset=[timestamp_col])

        grouped = (
            work.groupby(pd.Grouper(key=timestamp_col, freq=freq))
            .agg(
                mean_compound=(compound_col, "mean"),
                mean_confidence=(confidence_col, "mean"),
                count=(compound_col, "size"),
            )
            .dropna(subset=["mean_compound"])
        )

        if grouped.empty:
            return []

        rolling_mean = grouped["mean_compound"].rolling(
            window=shift_window, min_periods=1
        ).mean()
        rolling_std = grouped["mean_compound"].rolling(
            window=shift_window, min_periods=1
        ).std().fillna(0.0)

        points: List[TrendPoint] = []
        prev_mean: Optional[float] = None
        for ts, row in grouped.iterrows():
            shift = False
            if prev_mean is not None:
                shift = abs(row["mean_compound"] - prev_mean) > shift_threshold
            prev_mean = row["mean_compound"]
            points.append(
                TrendPoint(
                    period=ts.to_pydatetime(),
                    mean_compound=round(float(row["mean_compound"]), 4),
                    mean_confidence=round(float(row["mean_confidence"]), 4),
                    ticket_count=int(row["count"]),
                    shift_detected=shift,
                )
            )
        return points

    # -----------------------------------------------------------------------
    # Alert generation
    # -----------------------------------------------------------------------

    def generate_alerts(
        self,
        results: List[SentimentResult],
    ) -> List[Dict[str, Any]]:
        """Return alert dicts for tickets with extreme negative sentiment."""
        alerts: List[Dict[str, Any]] = []
        for i, r in enumerate(results):
            if r.is_critical:
                alerts.append(
                    {
                        "ticket_index": i,
                        "label": r.label,
                        "confidence": r.confidence,
                        "vader_compound": r.vader_compound,
                        "distilbert_score": r.distilbert_score,
                        "preview": r.text[:120],
                    }
                )
        return alerts

    # -----------------------------------------------------------------------
    # Per-topic breakdown
    # -----------------------------------------------------------------------

    @staticmethod
    def topic_breakdown(
        results: List[SentimentResult],
        topics: List[str],
        topic_labels: Optional[List[str]] = None,
    ) -> List[TopicSentiment]:
        """Aggregate sentiment per topic.

        Parameters
        ----------
        results:
            SentimentResult list from :meth:`analyze_batch`.
        topics:
            Topic keyword list (one keyword per topic).
        topic_labels:
            Human-readable labels. Defaults to *topics*.

        Returns
        -------
        list[TopicSentiment]
        """
        labels = topic_labels or topics
        topic_map: Dict[str, List[SentimentResult]] = {lbl: [] for lbl in labels}

        for result in results:
            lower_text = result.text.lower()
            for kw, lbl in zip(topics, labels):
                if kw.lower() in lower_text:
                    topic_map[lbl].append(result)

        breakdown: List[TopicSentiment] = []
        for lbl, items in topic_map.items():
            if not items:
                continue
            compounds = [r.vader_compound for r in items]
            confs = [r.confidence for r in items]
            label_counts: Dict[str, int] = {}
            for r in items:
                label_counts[r.label] = label_counts.get(r.label, 0) + 1
            breakdown.append(
                TopicSentiment(
                    topic=lbl,
                    mean_compound=round(float(np.mean(compounds)), 4),
                    mean_confidence=round(float(np.mean(confs)), 4),
                    count=len(items),
                    label_distribution=label_counts,
                )
            )
        return breakdown

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _run_transformer(
        self, text: str, max_len: int = 512
    ) -> Tuple[str, float]:
        """Run DistilBERT on a single text and return (label, score)."""
        try:
            out = self._transformer(text[:max_len])[0]
            label = out["label"].lower()          # POSITIVE / NEGATIVE
            score = float(out["score"])
            return label, score
        except Exception:
            logger.warning("DistilBERT inference failed for text snippet; falling back to VADER only")
            return "neutral", 0.0

    def _run_transformer_batch(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> List[Tuple[str, float]]:
        """Run DistilBERT in batches and return (label, score) pairs."""
        if not texts:
            return []

        try:
            outputs = self._transformer(
                texts,
                batch_size=batch_size,
                truncation=True,
                max_length=512,
                show_progress_bar=show_progress,
            )
            results: List[Tuple[str, float]] = []
            for out in outputs:
                label = out["label"].lower()
                score = float(out["score"])
                results.append((label, score))
            return results
        except Exception:
            logger.warning(
                "DistilBERT batch inference failed; returning fallback scores"
            )
            return [("neutral", 0.0)] * len(texts)

    def _hybrid(
        self,
        vader_scores: Dict[str, float],
        db_label: str,
        db_score: float,
    ) -> Tuple[str, float]:
        """Combine VADER compound and DistilBERT output into a hybrid score.

        Returns (label, confidence).

        The hybrid compound is a weighted average:
            h = alpha * vader_compound + (1 - alpha) * db_normalised
        where *db_normalised* maps DistilBERT output to [-1, 1]:
            POSITIVE -> +score, NEGATIVE -> -score, NEUTRAL -> 0.
        """
        compound = float(vader_scores["compound"])

        if db_label == "positive":
            db_normalised = db_score
        elif db_label == "negative":
            db_normalised = -db_score
        else:
            db_normalised = 0.0

        hybrid = (
            self.hybrid_alpha * compound
            + (1 - self.hybrid_alpha) * db_normalised
        )

        # Label thresholds
        if hybrid >= 0.15:
            label = "positive"
        elif hybrid <= -0.15:
            label = "negative"
        else:
            label = "neutral"

        confidence = min(abs(hybrid), 1.0)
        return label, round(confidence, 4)

    # -----------------------------------------------------------------------
    # Convenience: DataFrame integration
    # -----------------------------------------------------------------------

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Add sentiment columns to *df* in place and return it."""
        results = self.analyze_batch(
            df[text_col].tolist(),
            batch_size=batch_size,
            show_progress=show_progress,
        )
        df["sentiment_label"] = [r.label for r in results]
        df["sentiment_confidence"] = [r.confidence for r in results]
        df["vader_compound"] = [r.vader_compound for r in results]
        df["distilbert_score"] = [r.distilbert_score for r in results]
        df["is_critical"] = [r.is_critical for r in results]
        return df
