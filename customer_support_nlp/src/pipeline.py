"""End-to-end NLP intelligence pipeline orchestrator.

Coordinates topic modelling, sentiment analysis, named entity recognition,
and weekly intelligence report generation for customer support tickets.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .named_entity import NamedEntityExtractor
from .reporting import ReportGenerator
from .sentiment import SentimentAnalyzer
from .topic_modeling import TopicModeler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for the intelligence pipeline.

    Attributes:
        input_path: Path to the raw tickets CSV or JSON file.
        output_dir: Directory where all outputs are written.
        report_title: Title for the generated HTML report.
        reporting_period_days: Number of days to include in the report.
        ticket_id_col: Column name for unique ticket identifier.
        text_col: Column name for ticket body text.
        timestamp_col: Column name for ticket creation timestamp.
        priority_col: Column name for ticket priority (optional).
        embedding_model: Sentence-transformers model for BERTopic.
        umap_neighbors: UMAP n_neighbors parameter.
        umap_components: UMAP n_components parameter.
        hdbscan_min_cluster_size: HDBSCAN min_cluster_size.
        hdbscan_min_samples: HDBSCAN min_samples.
        nr_topics: Number of output topics ('auto' or int).
        distilbert_model: HuggingFace model for sentiment analysis.
        hybrid_alpha: Weight given to VADER in hybrid sentiment (0-1).
        critical_sentiment_threshold: Compound score below which a ticket is
            flagged as critical.
        spacy_model: spaCy model name for NER extraction.
        entity_top_n: Number of top entities to include in summary.
        random_seed: Random seed for reproducibility.
    """

    # --- Input / output ---
    input_path: Union[str, Path] = Path("data/raw/tickets.csv")
    output_dir: Union[str, Path] = Path("outputs")
    report_title: str = "Weekly Customer Support Intelligence Report"
    reporting_period_days: int = 7

    # --- Column mapping ---
    ticket_id_col: str = "ticket_id"
    text_col: str = "text"
    timestamp_col: str = "created_at"
    priority_col: Optional[str] = "priority"

    # --- Topic modelling ---
    embedding_model: str = "all-MiniLM-L6-v2"
    umap_neighbors: int = 15
    umap_components: int = 5
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int = 5
    nr_topics: Union[str, int] = "auto"

    # --- Sentiment analysis ---
    distilbert_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    hybrid_alpha: float = 0.4
    critical_sentiment_threshold: float = -0.7

    # --- NER ---
    spacy_model: str = "en_core_web_sm"
    entity_top_n: int = 20

    # --- Reproducibility ---
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class IntelligencePipeline:
    """End-to-end NLP intelligence pipeline for support ticket analysis.

    Orchestrates data loading, topic modelling, sentiment analysis, entity
    extraction, and weekly intelligence report generation.  Each stage is
    wrapped in error handling so that a single failure does not abort the
    entire pipeline.

    Attributes:
        config: Pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialise the pipeline with the given configuration.

        Args:
            config: Pipeline configuration dataclass.
        """
        self.config = config
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "IntelligencePipeline initialised.  input=%s  output=%s",
            config.input_path,
            self._output_dir,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_tickets(self) -> pd.DataFrame:
        """Load raw tickets from a CSV or JSON file.

        Returns:
            DataFrame containing the raw ticket data.

        Raises:
            FileNotFoundError: If the input path does not exist.
            ValueError: If the file format is unsupported.
        """
        path = Path(self.config.input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        logger.info("Loading tickets from %s", path)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in (".json", ".jsonl"):
            df = pd.read_json(path, lines=suffix == ".jsonl")
        else:
            raise ValueError(
                f"Unsupported file format '{suffix}'. Use .csv, .json, or .jsonl."
            )

        logger.info("Loaded %d tickets with columns: %s", len(df), list(df.columns))

        # Validate required columns exist
        for col in (self.config.ticket_id_col, self.config.text_col):
            if col not in df.columns:
                raise ValueError(
                    f"Required column '{col}' not found in input data. "
                    f"Available columns: {list(df.columns)}"
                )

        return df

    # ------------------------------------------------------------------
    # Stage runners (graceful degradation)
    # ------------------------------------------------------------------

    def _run_topic_modelling(
        self, texts: List[str]
    ) -> Dict[str, Any]:
        """Run BERTopic topic modelling on ticket texts.

        Args:
            texts: List of ticket body strings.

        Returns:
            Dictionary with topic results.
        """
        logger.info("Stage: Topic Modelling — %d documents", len(texts))
        modeler = TopicModeler(
            embedding_model_name=self.config.embedding_model,
            n_neighbors=self.config.umap_neighbors,
            n_components=self.config.umap_components,
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            min_samples=self.config.hdbscan_min_samples,
            nr_topics=self.config.nr_topics,
        )
        results = modeler.fit_transform(texts)
        summary = modeler.get_topic_summary()
        results["topic_summary"] = summary
        logger.info(
            "Topic modelling complete. %d topics found.",
            len(results.get("topic_info", [])),
        )
        return results

    def _run_sentiment_analysis(
        self, texts: List[str]
    ) -> pd.DataFrame:
        """Run hybrid VADER + DistilBERT sentiment analysis.

        Args:
            texts: List of ticket body strings.

        Returns:
            DataFrame with sentiment columns.
        """
        logger.info("Stage: Sentiment Analysis — %d texts", len(texts))
        analyzer = SentimentAnalyzer(
            distilbert_model=self.config.distilbert_model,
            hybrid_alpha=self.config.hybrid_alpha,
            critical_threshold=self.config.critical_sentiment_threshold,
        )
        results = analyzer.analyze_batch(texts, batch_size=64)
        rows = []
        for r in results:
            rows.append({
                "sentiment_label": r.label,
                "sentiment_confidence": r.confidence,
                "vader_compound": r.vader_compound,
                "distilbert_score": r.distilbert_score,
                "is_critical": r.is_critical,
            })
        return pd.DataFrame(rows)

    def _run_entity_extraction(
        self, texts: List[str]
    ) -> pd.DataFrame:
        """Run spaCy NER entity extraction.

        Args:
            texts: List of ticket body strings.

        Returns:
            DataFrame with entity mentions.
        """
        logger.info("Stage: Entity Extraction — %d documents", len(texts))
        extractor = NamedEntityExtractor()
        return extractor.extract_entities_dataframe(texts)

    # ------------------------------------------------------------------
    # Output persistence
    # ------------------------------------------------------------------

    def _save_enriched_tickets(self, df: pd.DataFrame) -> Path:
        """Save the enriched tickets DataFrame to CSV.

        Args:
            df: Merged DataFrame with topic, sentiment, and entity columns.

        Returns:
            Path to the saved CSV file.
        """
        path = self._output_dir / "enriched_tickets.csv"
        df.to_csv(path, index=False)
        logger.info("Enriched tickets saved to %s", path)
        return path

    def _save_topic_summary(self, topic_info: Any) -> Path:
        """Save topic summary to CSV.

        Args:
            topic_info: BERTopic topic info DataFrame.

        Returns:
            Path to the saved CSV file.
        """
        path = self._output_dir / "topic_summary.csv"
        if isinstance(topic_info, pd.DataFrame):
            topic_info.to_csv(path, index=False)
        else:
            pd.DataFrame(topic_info).to_csv(path, index=False)
        logger.info("Topic summary saved to %s", path)
        return path

    def _save_entity_summary(self, entity_df: pd.DataFrame) -> Path:
        """Save entity summary to CSV.

        Args:
            entity_df: Entity extraction DataFrame.

        Returns:
            Path to the saved CSV file.
        """
        path = self._output_dir / "entity_summary.csv"
        if not entity_df.empty:
            summary = (
                entity_df.groupby(["text", "label"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            summary.to_csv(path, index=False)
        else:
            entity_df.to_csv(path, index=False)
        logger.info("Entity summary saved to %s", path)
        return path

    def _save_report(self, html: str) -> Path:
        """Save the HTML intelligence report.

        Args:
            html: Complete HTML report string.

        Returns:
            Path to the saved HTML file.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._output_dir / f"intelligence_report_{ts}.html"
        path.write_text(html, encoding="utf-8")
        logger.info("Report saved to %s", path)
        return path

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline and return all outputs.

        Stages run sequentially.  If any stage fails, the pipeline logs
        the error and continues with subsequent stages using empty/partial
        results so that the remaining pipeline can still produce output.

        Returns:
            Dictionary with keys:
                - 'enriched_tickets': Path to enriched tickets CSV.
                - 'topic_summary': Path to topic summary CSV.
                - 'entity_summary': Path to entity summary CSV.
                - 'report': Path to HTML report file.
                - 'n_tickets': Number of tickets processed.
                - 'errors': List of error messages from failed stages.
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("=" * 60)
        logger.info("Pipeline run %s started", run_id)
        logger.info("=" * 60)

        outputs: Dict[str, Any] = {
            "enriched_tickets": None,
            "topic_summary": None,
            "entity_summary": None,
            "report": None,
            "n_tickets": 0,
            "errors": [],
        }

        # --- Stage 1: Load data ----------------------------------------
        try:
            df = self._load_tickets()
            outputs["n_tickets"] = len(df)
        except Exception as exc:
            logger.exception("Pipeline aborted: failed to load data.")
            outputs["errors"].append(f"Data loading failed: {exc}")
            return outputs

        texts = df[self.config.text_col].fillna("").tolist()
        ticket_ids = df[self.config.ticket_id_col].tolist()

        # --- Stage 2: Topic modelling ----------------------------------
        topic_results: Optional[Dict[str, Any]] = None
        try:
            topic_results = self._run_topic_modelling(texts)
            self._save_topic_summary(topic_results["topic_info"])
            outputs["topic_summary"] = self._output_dir / "topic_summary.csv"
        except Exception as exc:
            logger.exception("Topic modelling stage failed.")
            outputs["errors"].append(f"Topic modelling failed: {exc}")

        # --- Stage 3: Sentiment analysis -------------------------------
        sentiment_df = pd.DataFrame()
        try:
            sentiment_df = self._run_sentiment_analysis(texts)
            sentiment_df.insert(0, self.config.ticket_id_col, ticket_ids)
        except Exception as exc:
            logger.exception("Sentiment analysis stage failed.")
            outputs["errors"].append(f"Sentiment analysis failed: {exc}")

        # --- Stage 4: Entity extraction --------------------------------
        entity_df = pd.DataFrame()
        try:
            entity_df = self._run_entity_extraction(texts)
            if not entity_df.empty:
                entity_df.insert(0, self.config.ticket_id_col,
                                 [ticket_ids[i] for i in entity_df["doc_index"]])
                entity_df.rename(columns={"text": "entity", "label": "entity_type"}, inplace=True)
            self._save_entity_summary(entity_df)
            outputs["entity_summary"] = self._output_dir / "entity_summary.csv"
        except Exception as exc:
            logger.exception("Entity extraction stage failed.")
            outputs["errors"].append(f"Entity extraction failed: {exc}")

        # --- Stage 5: Build enriched DataFrame -------------------------
        enriched = self._build_enriched_df(
            df, topic_results, sentiment_df, entity_df
        )
        enriched_path = self._save_enriched_tickets(enriched)
        outputs["enriched_tickets"] = enriched_path

        # --- Stage 6: Generate report ----------------------------------
        try:
            topics_for_report = self._prepare_topics_for_report(
                ticket_ids, topic_results
            )
            sentiment_for_report = self._prepare_sentiment_for_report(
                ticket_ids, sentiment_df
            )
            entities_for_report = self._prepare_entities_for_report(
                ticket_ids, entity_df
            )

            generator = ReportGenerator(
                topics_df=topics_for_report,
                sentiment_df=sentiment_for_report,
                entities_df=entities_for_report,
                tickets_df=df[[c for c in [self.config.ticket_id_col,
                                           self.config.timestamp_col,
                                           self.config.priority_col] if c and c in df.columns]],
                report_title=self.config.report_title,
                reporting_period_days=self.config.reporting_period_days,
                critical_sentiment_threshold=self.config.critical_sentiment_threshold,
            )
            html = generator.generate_report()
            report_path = self._save_report(html)
            outputs["report"] = report_path
        except Exception as exc:
            logger.exception("Report generation stage failed.")
            outputs["errors"].append(f"Report generation failed: {exc}")

        # --- Summary ---------------------------------------------------
        logger.info("=" * 60)
        logger.info("Pipeline run %s finished", run_id)
        logger.info(
            "Tickets: %d | Topics: %s | Errors: %d",
            outputs["n_tickets"],
            len(topic_results.get("topic_info", [])) if topic_results else "N/A",
            len(outputs["errors"]),
        )
        if outputs["errors"]:
            for err in outputs["errors"]:
                logger.warning("  - %s", err)
        logger.info("=" * 60)

        return outputs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_enriched_df(
        self,
        raw_df: pd.DataFrame,
        topic_results: Optional[Dict[str, Any]],
        sentiment_df: pd.DataFrame,
        entity_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge topic, sentiment, and entity data onto the raw tickets.

        Args:
            raw_df: Original tickets DataFrame.
            topic_results: Output from BERTopic fit_transform.
            sentiment_df: Sentiment analysis results.
            entity_df: Entity extraction results.

        Returns:
            Merged DataFrame.
        """
        enriched = raw_df.copy()
        ticket_ids = enriched[self.config.ticket_id_col].tolist()

        # Attach topic labels
        if topic_results is not None and "topics" in topic_results:
            topics = topic_results["topics"]
            topic_info = topic_results.get("topic_info")
            label_map: Dict[int, str] = {}
            if isinstance(topic_info, pd.DataFrame) and "Topic" in topic_info.columns and "Name" in topic_info.columns:
                for _, row in topic_info.iterrows():
                    label_map[row["Topic"]] = row.get("Name", f"Topic_{row['Topic']}")

            # Topics list may be shorter than ticket_ids if documents were
            # filtered; align by index (BERTopic preserves order).
            enriched["topic"] = [-1] * len(enriched)
            for i, t in enumerate(topics):
                if i < len(enriched):
                    enriched.iloc[i, enriched.columns.get_loc("topic")] = t

            enriched["topic_label"] = enriched["topic"].map(label_map).fillna("Unknown")

        # Attach sentiment columns
        if not sentiment_df.empty:
            for col in sentiment_df.columns:
                if col == self.config.ticket_id_col:
                    continue
                enriched[col] = sentiment_df[col].values

        return enriched

    def _prepare_topics_for_report(
        self, ticket_ids: List[Any], topic_results: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Format topic data for the report generator.

        Args:
            ticket_ids: List of ticket identifiers.
            topic_results: BERTopic results dictionary.

        Returns:
            DataFrame with ticket_id, topic, and topic_label columns.
        """
        if topic_results is None or "topics" not in topic_results:
            return pd.DataFrame(columns=["ticket_id", "topic", "topic_label"])

        topics = topic_results["topics"]
        topic_info = topic_results.get("topic_info")
        label_map: Dict[int, str] = {}
        if isinstance(topic_info, pd.DataFrame) and "Topic" in topic_info.columns and "Name" in topic_info.columns:
            for _, row in topic_info.iterrows():
                label_map[row["Topic"]] = row.get("Name", f"Topic_{row['Topic']}")

        rows = []
        for i, t in enumerate(topics):
            if i < len(ticket_ids):
                rows.append({
                    "ticket_id": ticket_ids[i],
                    "topic": t,
                    "topic_label": label_map.get(t, f"Topic_{t}"),
                })
        return pd.DataFrame(rows)

    def _prepare_sentiment_for_report(
        self, ticket_ids: List[Any], sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Format sentiment data for the report generator.

        Args:
            ticket_ids: List of ticket identifiers.
            sentiment_df: Raw sentiment analysis DataFrame.

        Returns:
            DataFrame with ticket_id, sentiment_label, and sentiment_score.
        """
        if sentiment_df.empty:
            return pd.DataFrame(columns=["ticket_id", "sentiment_label", "sentiment_score"])

        report_df = sentiment_df.copy()
        report_df[self.config.ticket_id_col] = ticket_ids[: len(report_df)]

        # Map to expected column names
        if "vader_compound" in report_df.columns:
            report_df["sentiment_score"] = report_df["vader_compound"]

        keep_cols = [self.config.ticket_id_col, "sentiment_label", "sentiment_score"]
        keep_cols = [c for c in keep_cols if c in report_df.columns]
        return report_df[keep_cols]

    def _prepare_entities_for_report(
        self, ticket_ids: List[Any], entity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Format entity data for the report generator.

        Args:
            ticket_ids: List of ticket identifiers.
            entity_df: Raw entity extraction DataFrame.

        Returns:
            DataFrame with ticket_id, entity, and entity_type columns.
        """
        if entity_df.empty:
            return pd.DataFrame(columns=["ticket_id", "entity", "entity_type"])

        report_df = entity_df.copy()
        if self.config.ticket_id_col not in report_df.columns:
            return pd.DataFrame(columns=["ticket_id", "entity", "entity_type"])

        keep_cols = [self.config.ticket_id_col, "entity", "entity_type"]
        keep_cols = [c for c in keep_cols if c in report_df.columns]
        return report_df[keep_cols]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Customer Support NLP Intelligence Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (optional).",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to input CSV/JSON file.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = PipelineConfig()
    if args.input:
        config.input_path = args.input
    if args.output:
        config.output_dir = args.output

    # Optionally load YAML overrides
    if args.config:
        try:
            from omegaconf import OmegaConf
            yaml_cfg = OmegaConf.load(args.config)
            for key, value in yaml_cfg.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        except ImportError:
            logger.warning("OmegaConf not installed; ignoring --config flag.")
        except Exception as exc:
            logger.warning("Failed to load config file: %s", exc)

    pipeline = IntelligencePipeline(config)
    results = pipeline.run()

    if results["errors"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
