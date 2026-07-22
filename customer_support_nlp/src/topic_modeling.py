"""BERTopic-based topic modeling module for customer support ticket analysis.

This module provides a TopicModeler class that embeds support tickets using
sentence-transformers, performs topic extraction with BERTopic, and generates
visualizations for topic analysis.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

try:
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from bertopic.viz import (
       HierarchyConfig,
        IntertopicDistanceMap,
        TopicVisualizer,
        visualize_hierarchy,
        visualize_barchart,
    )

    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class TopicModeler:
    """BERTopic-based topic modeling for customer support tickets.

    Embeds documents using sentence-transformers, runs BERTopic with UMAP
    dimensionality reduction and HDBSCAN clustering, and provides methods
    for topic assignment, summarization, and visualization.

    Attributes:
        embedding_model_name: Name of the sentence-transformers model.
        n_neighbors: UMAP n_neighbors parameter.
        n_components: UMAP n_components parameter.
        min_cluster_size: HDBSCAN min_cluster_size parameter.
        min_samples: HDBSCAN min_samples parameter.
        nr_topics: Number of topics to reduce to ('auto' or int).
        embedding_model: Loaded SentenceTransformer model.
        topic_model: Fitted BERTopic instance.
        topics: Topic assignments from the last fit_transform call.
        probs: Topic probability distributions from the last fit_transform call.
    """

    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MIN_DOCUMENTS_FOR_TOPICING = 2
    MIN_TEXT_LENGTH = 3

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        n_neighbors: int = 15,
        n_components: int = 5,
        min_cluster_size: int = 10,
        min_samples: int = 5,
        nr_topics: Union[str, int] = "auto",
        verbose: bool = False,
    ) -> None:
        """Initialize the TopicModeler.

        Args:
            embedding_model_name: Sentence-transformers model name.
            n_neighbors: UMAP neighborhood size.
            n_components: UMAP output dimensions.
            min_cluster_size: HDBSCAN minimum cluster size.
            min_samples: HDBSCAN minimum samples in a neighborhood.
            nr_topics: Number of output topics ('auto' or specific int).
            verbose: Enable verbose logging from BERTopic.
        """
        if not BERTOPIC_AVAILABLE:
            raise ImportError(
                "bertopic is required. Install with: pip install bertopic"
            )

        self.embedding_model_name = embedding_model_name
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.nr_topics = nr_topics
        self.verbose = verbose

        self.embedding_model: Optional[SentenceTransformer] = None
        self.topic_model: Optional[BERTopic] = None
        self.topics: Optional[List[int]] = None
        self.probs: Optional[np.ndarray] = None
        self._embeddings: Optional[np.ndarray] = None
        self._documents: Optional[List[str]] = None

        logger.info(
            "TopicModeler initialized with embedding_model=%s, "
            "n_neighbors=%d, n_components=%d, min_cluster_size=%d, "
            "nr_topics=%s",
            embedding_model_name,
            n_neighbors,
            n_components,
            min_cluster_size,
            nr_topics,
        )

    def _load_embedding_model(self) -> SentenceTransformer:
        """Load the sentence-transformers embedding model.

        Returns:
            Loaded SentenceTransformer instance.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        if self.embedding_model is None:
            logger.info("Loading embedding model: %s", self.embedding_model_name)
            try:
                self.embedding_model = SentenceTransformer(
                    self.embedding_model_name
                )
                logger.info("Embedding model loaded successfully.")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load embedding model '{self.embedding_model_name}': "
                    f"{exc}"
                ) from exc
        return self.embedding_model

    def _preprocess_texts(self, documents: List[str]) -> List[str]:
        """Clean and filter input documents.

        Removes empty strings, very short texts, and strips whitespace.

        Args:
            documents: Raw input documents.

        Returns:
            List of cleaned, non-empty documents with original indices preserved
            via a parallel list mapping cleaned index -> original index.
        """
        cleaned: List[str] = []
        for doc in documents:
            if doc is None:
                continue
            text = str(doc).strip()
            if len(text) >= self.MIN_TEXT_LENGTH:
                cleaned.append(text)
            else:
                logger.debug(
                    "Skipping document with length %d (< %d)",
                    len(text),
                    self.MIN_TEXT_LENGTH,
                )
        return cleaned

    def _build_topic_model(self) -> BERTopic:
        """Construct the BERTopic pipeline with UMAP and HDBSCAN.

        Returns:
            Configured BERTopic instance.
        """
        umap_model = UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            prediction_data=True,
        )

        vectorizer = CountVectorizer(
            stop_words="english",
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
        )

        representation_model = KeyBERTInspired()

        topic_model = BERTopic(
            embedding_model=self._load_embedding_model(),
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            representation_model=representation_model,
            top_n_words=10,
            verbose=self.verbose,
        )
        return topic_model

    def fit_transform(
        self, documents: List[str], calculate_probabilities: bool = True
    ) -> Dict[str, Any]:
        """Fit BERTopic on the document corpus and return results.

        Args:
            documents: List of raw text support tickets.
            calculate_probabilities: Whether to compute topic probabilities.

        Returns:
            Dictionary with keys:
                - 'topics': List[int] of topic assignments (-1 = outlier).
                - 'probs': np.ndarray of topic probability distributions,
                  or None if calculate_probabilities=False.
                - 'topic_info': DataFrame with topic metadata.
                - 'topic_frequencies': DataFrame with topic counts.
                - 'topic_hierarchy': Topic hierarchy data from BERTopic.
                - 'embeddings': np.ndarray of document embeddings.
                - 'n_outliers': int count of outlier documents.

        Raises:
            ValueError: If documents list is empty or all documents are filtered out.
        """
        if not documents:
            raise ValueError("Cannot fit on an empty document list.")

        cleaned_docs = self._preprocess_texts(documents)
        if len(cleaned_docs) == 0:
            raise ValueError(
                "All documents were filtered out during preprocessing "
                "(all shorter than minimum length)."
            )

        logger.info(
            "Fitting BERTopic on %d documents (%d original, %d after cleaning).",
            len(cleaned_docs),
            len(documents),
            len(cleaned_docs),
        )

        self._documents = cleaned_docs
        model = self._build_topic_model()

        embeddings = model.embedding_model.encode(
            cleaned_docs, show_progress_bar=self.verbose
        )
        self._embeddings = embeddings

        topics, probs = model.fit_transform(
            cleaned_docs, embeddings=embeddings
        )

        if (
            calculate_probabilities
            and probs is not None
            and isinstance(probs, np.ndarray)
            and probs.size > 0
        ):
            self.probs = probs
        else:
            self.probs = None

        self.topics = topics
        self.topic_model = model

        topic_info = model.get_topic_info()
        topic_frequencies = model.get_topic_freq()
        hierarchy = model.hierarchical_topics(cleaned_docs)

        n_outliers = topics.count(-1)
        logger.info(
            "BERTopic fitting complete. Found %d topics, %d outliers.",
            len(topic_info),
            n_outliers,
        )

        return {
            "topics": topics,
            "probs": self.probs,
            "topic_info": topic_info,
            "topic_frequencies": topic_frequencies,
            "topic_hierarchy": hierarchy,
            "embeddings": embeddings,
            "n_outliers": n_outliers,
        }

    def get_topic_summary(self) -> Dict[str, Any]:
        """Return a human-readable summary of discovered topics.

        Returns:
            Dictionary mapping topic_id -> dict with:
                - 'name': Topic name from BERTopic.
                - 'keywords': Top representative keywords.
                - 'count': Number of documents in topic.
                - 'representative_docs': Example documents for the topic.

        Raises:
            RuntimeError: If fit_transform has not been called yet.
        """
        if self.topic_model is None or self._documents is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit_transform() first."
            )

        topic_info = self.topic_model.get_topic_info()
        summary: Dict[str, Any] = {}

        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]
            if topic_id == -1:
                continue

            keywords = self.topic_model.get_topic(topic_id)
            keyword_list = [kw for kw, _ in keywords] if keywords else []

            representative_docs = []
            if hasattr(self.topic_model, "get_representative_docs"):
                rep_docs = self.topic_model.get_representative_docs(topic_id)
                if rep_docs:
                    representative_docs = rep_docs[:5]

            summary[topic_id] = {
                "name": row.get("Name", f"Topic_{topic_id}"),
                "keywords": keyword_list,
                "count": row.get("Count", 0),
                "representative_docs": representative_docs,
            }

        logger.info(
            "Topic summary generated for %d topics.", len(summary)
        )
        return summary

    def visualize_topics(
        self, output_dir: Union[str, Path], top_n_topics: int = 8
    ) -> Dict[str, str]:
        """Generate topic visualizations and save them to disk.

        Creates three HTML visualizations:
        1. Topic bar chart with top keywords.
        2. Topic hierarchy dendrogram.
        3. Intertopic distance map.

        Args:
            output_dir: Directory to save visualization files.
            top_n_topics: Number of topics to show in bar chart.

        Returns:
            Dictionary mapping visualization name to saved file path.

        Raises:
            RuntimeError: If fit_transform has not been called.
            OSError: If output directory cannot be created.
        """
        if self.topic_model is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit_transform() first."
            )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        saved: Dict[str, str] = {}

        try:
            fig_bar = self.topic_model.visualize_barchart(
                top_n_topics=top_n_topics, n_words=8
            )
            bar_path = out / "topic_barchart.html"
            fig_bar.write_html(str(bar_path))
            saved["barchart"] = str(bar_path)
            logger.info("Saved bar chart to %s", bar_path)
        except Exception as exc:
            logger.warning("Failed to generate bar chart: %s", exc)

        try:
            fig_hierarchy = self.topic_model.visualize_hierarchy(
                hierarchical_topics=self.topic_model.hierarchical_topics(
                    self._documents
                )
            )
            hier_path = out / "topic_hierarchy.html"
            fig_hierarchy.write_html(str(hier_path))
            saved["hierarchy"] = str(hier_path)
            logger.info("Saved hierarchy to %s", hier_path)
        except Exception as exc:
            logger.warning("Failed to generate hierarchy: %s", exc)

        try:
            fig_intertopic = self.topic_model.visualize_topics()
            inter_path = out / "intertopic_distance.html"
            fig_intertopic.write_html(str(inter_path))
            saved["intertopic_map"] = str(inter_path)
            logger.info("Saved intertopic map to %s", inter_path)
        except Exception as exc:
            logger.warning("Failed to generate intertopic map: %s", exc)

        return saved

    def predict_topic(self, text: str) -> Dict[str, Any]:
        """Predict the topic for a single new document.

        Args:
            text: Raw input text.

        Returns:
            Dictionary with 'topic_id', 'keywords', and 'probability'.
        """
        if self.topic_model is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit_transform() first."
            )

        cleaned = str(text).strip()
        if len(cleaned) < self.MIN_TEXT_LENGTH:
            return {"topic_id": -1, "keywords": [], "probability": 0.0}

        topic_id, prob = self.topic_model.transform([cleaned])
        tid = topic_id[0] if isinstance(topic_id, list) else int(topic_id)

        keywords = []
        if tid != -1:
            topic_words = self.topic_model.get_topic(tid)
            if topic_words:
                keywords = [kw for kw, _ in topic_words]

        probability = float(prob[0].max()) if prob is not None and prob.size > 0 else 0.0

        return {
            "topic_id": tid,
            "keywords": keywords,
            "probability": probability,
        }
