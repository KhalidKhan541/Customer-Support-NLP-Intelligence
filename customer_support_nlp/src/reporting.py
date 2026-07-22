"""
Weekly Intelligence Report Generator — IBM Consulting Deliverable Format.

Generates comprehensive HTML intelligence reports with executive summaries,
charts, and detailed analysis from NLP pipeline outputs.
"""

from __future__ import annotations

import base64
import io
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial"],
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

IBM_NAVY = "#1F70C1"
IBM_DARK = "#162050"
IBM_LIGHT = "#A6C8FF"
IBM_ACCENT = "#0F62FE"
IBM_WARN = "#DA1E28"
IBM_SUCCESS = "#198038"
IBM_BG = "#F4F4F4"
IBM_TEXT = "#161616"
IBM_GRAY = "#525252"
IBM_LIGHT_GRAY = "#E0E0E0"

PALETTE = [IBM_NAVY, IBM_ACCENT, IBM_SUCCESS, "#6929C4", "#EE5396", "#005D5D", "#8B5CF6", IBM_WARN]


class ReportError(Exception):
    """Raised when report generation fails."""


class ReportGenerator:
    """Generates IBM-style weekly intelligence HTML reports.

    Parameters
    ----------
    topics_df : pd.DataFrame
        Topic extraction results with columns ``ticket_id``, ``topic``,
        ``topic_label`` (optional), ``confidence`` (optional).
    sentiment_df : pd.DataFrame
        Sentiment analysis results with columns ``ticket_id``,
        ``sentiment_label``, ``sentiment_score`` (optional).
    entities_df : pd.DataFrame
        Entity extraction results with columns ``ticket_id``, ``entity``,
        ``entity_type``, ``entity_label`` (optional).
    tickets_df : pd.DataFrame, optional
        Raw ticket metadata. Expected columns include ``ticket_id``,
        ``created_at``, ``priority``, ``channel``, ``agent``.
    report_title : str
        Title rendered in the report header.
    reporting_period_days : int
        Number of days of data to include (default 7).
    critical_sentiment_threshold : float
        Sentiment score below which a ticket is flagged critical.
    """

    def __init__(
        self,
        topics_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        entities_df: pd.DataFrame,
        tickets_df: Optional[pd.DataFrame] = None,
        report_title: str = "Weekly Customer Support Intelligence Report",
        reporting_period_days: int = 7,
        critical_sentiment_threshold: float = -0.6,
    ) -> None:
        self.topics_df = self._ensure_dataframe(topics_df, "topics")
        self.sentiment_df = self._ensure_dataframe(sentiment_df, "sentiment")
        self.entities_df = self._ensure_dataframe(entities_df, "entities")
        self.tickets_df = self._ensure_dataframe(tickets_df, "tickets") if tickets_df is not None else pd.DataFrame()
        self.report_title = report_title
        self.reporting_period_days = reporting_period_days
        self.critical_sentiment_threshold = critical_sentiment_threshold
        self.report_generated_at: Optional[datetime] = None

    @staticmethod
    def _ensure_dataframe(df: Any, label: str) -> pd.DataFrame:
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            logger.warning("Empty DataFrame supplied for '%s'; report will handle gracefully.", label)
            return pd.DataFrame()
        if not isinstance(df, pd.DataFrame):
            raise ReportError(f"Expected pd.DataFrame for '{label}', got {type(df).__name__}")
        return df.copy()

    @staticmethod
    def _fig_to_base64(fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    @staticmethod
    def _pct(part: Union[int, float], whole: Union[int, float]) -> str:
        if whole == 0:
            return "0.0%"
        return f"{part / whole * 100:.1f}%"

    def _safe_value_counts(self, df: pd.DataFrame, col: str) -> pd.Series:
        if df.empty or col not in df.columns:
            return pd.Series(dtype=int)
        return df[col].value_counts()

    def _merged_df(self) -> pd.DataFrame:
        if self.topics_df.empty:
            return self.sentiment_df if not self.sentiment_df.empty else self.entities_df
        merged = self.topics_df.copy()
        if not self.sentiment_df.empty and "sentiment_label" in self.sentiment_df.columns:
            cols = ["ticket_id", "sentiment_label", "sentiment_score"]
            cols = [c for c in cols if c in self.sentiment_df.columns]
            merged = merged.merge(self.sentiment_df[cols], on="ticket_id", how="left")
        if not self.entities_df.empty:
            merged = merged.merge(
                self.entities_df[["ticket_id"]].drop_duplicates(),
                on="ticket_id",
                how="left",
                indicator=True,
            )
            merged["has_entities"] = merged["_merge"] == "both"
            merged.drop(columns=["_merge"], inplace=True)
        return merged

    # ------------------------------------------------------------------
    # Chart builders
    # ------------------------------------------------------------------

    def _chart_topic_bar(self) -> str:
        vc = self._safe_value_counts(self.topics_df, "topic_label" if "topic_label" in self.topics_df.columns else "topic")
        if vc.empty:
            return self._placeholder_chart("No topic data available")
        top = vc.head(12)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = PALETTE[: len(top)]
        bars = ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], edgecolor="white", height=0.6)
        ax.set_xlabel("Number of Tickets")
        ax.set_title("Top Complaint Categories")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        for bar in bars:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{int(bar.get_width()):,}", va="center", fontsize=9, color=IBM_GRAY)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        return self._fig_to_base64(fig)

    def _chart_topic_pie(self) -> str:
        vc = self._safe_value_counts(self.topics_df, "topic_label" if "topic_label" in self.topics_df.columns else "topic")
        if vc.empty:
            return self._placeholder_chart("No topic data available")
        top = vc.head(8)
        if len(vc) > 8:
            other = vc.iloc[8:].sum()
            top = pd.concat([top, pd.Series({"Other": other})])
        fig, ax = plt.subplots(figsize=(7, 7))
        wedges, texts, autotexts = ax.pie(
            top.values,
            labels=top.index,
            autopct=lambda p: f"{p:.1f}%",
            colors=PALETTE[: len(top)],
            startangle=140,
            pctdistance=0.8,
            wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
        )
        for t in autotexts:
            t.set_fontsize(9)
            t.set_color("white")
        ax.set_title("Topic Distribution", pad=20)
        fig.tight_layout()
        return self._fig_to_base64(fig)

    def _chart_word_cloud(self, topic_label: str, texts: pd.Series) -> str:
        """Generate a word cloud for a specific topic's ticket texts."""
        try:
            from wordcloud import WordCloud
            text_blob = " ".join(texts.dropna().astype(str).tolist())
            if not text_blob.strip():
                return self._placeholder_chart(f"No text data for topic: {topic_label}")
            wc = WordCloud(
                width=600, height=300,
                background_color="white",
                colormap="Blues",
                max_words=50,
                contour_width=1,
                contour_color=IBM_NAVY,
            ).generate(text_blob)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.imshow(wc, interpolation="bilinear")
            ax.set_title(f"Word Cloud — {topic_label}", fontsize=11, color=IBM_DARK)
            ax.axis("off")
            fig.tight_layout()
            return self._fig_to_base64(fig)
        except ImportError:
            logger.warning("wordcloud not installed; skipping word cloud generation.")
            return self._placeholder_chart("wordcloud package not installed")

    def _chart_word_clouds_per_topic(self) -> List[Tuple[str, str]]:
        """Generate word clouds for each major topic."""
        clouds: List[Tuple[str, str]] = []
        if self.topics_df.empty:
            return clouds
        topic_col = "topic_label" if "topic_label" in self.topics_df.columns else "topic"
        if topic_col not in self.topics_df.columns:
            return clouds
        text_col = None
        for c in ("text_clean", "text", "ticket_text", "body"):
            if c in self.topics_df.columns:
                text_col = c
                break
        if text_col is None and not self.tickets_df.empty:
            for c in ("text_clean", "text", "ticket_text", "body"):
                if c in self.tickets_df.columns:
                    text_col = c
                    break
        if text_col is None:
            return clouds
        topic_counts = self._safe_value_counts(self.topics_df, topic_col)
        for topic_label in topic_counts.head(6).index:
            topic_df = self.topics_df[self.topics_df[topic_col] == topic_label]
            if not self.tickets_df.empty and text_col in self.tickets_df.columns:
                topic_df = topic_df.merge(self.tickets_df[["ticket_id", text_col]], on="ticket_id", how="left")
            texts = topic_df[text_col] if text_col in topic_df.columns else pd.Series(dtype=str)
            b64 = self._chart_word_cloud(str(topic_label), texts)
            clouds.append((str(topic_label), b64))
        return clouds

    def _chart_sentiment_trend(self) -> str:
        if self.sentiment_df.empty or "sentiment_score" not in self.sentiment_df.columns:
            return self._placeholder_chart("No sentiment score data available")
        ts_col = None
        for candidate in ("created_at", "timestamp", "date"):
            if candidate in self.sentiment_df.columns:
                ts_col = candidate
                break
        if ts_col is None and not self.tickets_df.empty:
            for candidate in ("created_at", "timestamp", "date"):
                if candidate in self.tickets_df.columns:
                    ts_col = candidate
                    break
        if ts_col is None:
            return self._placeholder_chart("No timestamp column found for trend")
        df = self.sentiment_df.copy()
        if ts_col not in df.columns and not self.tickets_df.empty:
            df = df.merge(self.tickets_df[["ticket_id", ts_col]], on="ticket_id", how="left")
        if ts_col not in df.columns:
            return self._placeholder_chart("Timestamp not available after merge")
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col])
        if df.empty:
            return self._placeholder_chart("No valid timestamps")
        df = df.set_index(ts_col).sort_index()
        daily = df["sentiment_score"].resample("D").mean().dropna()
        if daily.empty:
            return self._placeholder_chart("Insufficient data for daily trend")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.fill_between(daily.index, daily.values, alpha=0.15, color=IBM_NAVY)
        ax.plot(daily.index, daily.values, color=IBM_NAVY, linewidth=2.2, marker="o", markersize=4)
        ax.axhline(y=0, color=IBM_LIGHT_GRAY, linewidth=0.8, linestyle="--")
        ax.set_ylabel("Mean Sentiment Score")
        ax.set_title("Sentiment Trend Over Time")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=30)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        return self._fig_to_base64(fig)

    def _chart_sentiment_distribution(self) -> str:
        vc = self._safe_value_counts(self.sentiment_df, "sentiment_label")
        if vc.empty:
            return self._placeholder_chart("No sentiment labels available")
        color_map = {"positive": IBM_SUCCESS, "neutral": IBM_NAVY, "negative": IBM_WARN}
        colors = [color_map.get(label.lower(), IBM_NAVY) for label in vc.index]
        fig, ax = plt.subplots(figsize=(8, 4))
        vc.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.55)
        ax.set_title("Sentiment Distribution")
        ax.set_xlabel("")
        ax.set_ylabel("Ticket Count")
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        for i, v in enumerate(vc.values):
            ax.text(i, v + 0.5, f"{v:,}", ha="center", fontsize=9, color=IBM_GRAY)
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=0)
        fig.tight_layout()
        return self._fig_to_base64(fig)

    def _chart_sentiment_by_topic(self) -> str:
        """Bar chart comparing mean sentiment score across topics."""
        if self.topics_df.empty or self.sentiment_df.empty:
            return self._placeholder_chart("Insufficient data for topic-sentiment comparison")
        topic_col = "topic_label" if "topic_label" in self.topics_df.columns else "topic"
        if topic_col not in self.topics_df.columns or "sentiment_score" not in self.sentiment_df.columns:
            return self._placeholder_chart("Missing topic or sentiment columns")
        merged = self.topics_df[["ticket_id", topic_col]].merge(
            self.sentiment_df[["ticket_id", "sentiment_score"]], on="topic_id" if "topic_id" in self.sentiment_df.columns else "ticket_id", how="inner"
        ) if "topic_id" in self.sentiment_df.columns else self.topics_df[["ticket_id", topic_col]].merge(
            self.sentiment_df[["ticket_id", "sentiment_score"]], on="ticket_id", how="inner"
        )
        if merged.empty:
            return self._placeholder_chart("No overlapping ticket IDs between topics and sentiment")
        agg = merged.groupby(topic_col)["sentiment_score"].mean().sort_values()
        if agg.empty:
            return self._placeholder_chart("No aggregated sentiment data")
        fig, ax = plt.subplots(figsize=(10, max(4, len(agg) * 0.5)))
        colors = [IBM_SUCCESS if v > 0 else IBM_WARN if v < 0 else IBM_NAVY for v in agg.values]
        bars = ax.barh(agg.index, agg.values, color=colors, edgecolor="white", height=0.6)
        ax.axvline(x=0, color=IBM_LIGHT_GRAY, linewidth=0.8, linestyle="--")
        ax.set_xlabel("Mean Sentiment Score")
        ax.set_title("Sentiment by Topic")
        for bar in bars:
            w = bar.get_width()
            offset = 0.02 if w >= 0 else -0.02
            ax.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", fontsize=9, color=IBM_GRAY, ha="left" if w >= 0 else "right")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        return self._fig_to_base64(fig)

    def _chart_entity_bar(self) -> str:
        if self.entities_df.empty or "entity" not in self.entities_df.columns:
            return self._placeholder_chart("No entity data available")
        vc = self.entities_df["entity"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        colors = PALETTE[: len(vc)]
        ax.barh(vc.index[::-1], vc.values[::-1], color=colors[::-1], edgecolor="white", height=0.55)
        ax.set_xlabel("Mentions")
        ax.set_title("Most Mentioned Entities")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        for i, (idx, val) in enumerate(vc.items()):
            ax.text(val + 0.3, len(vc) - 1 - i, f"{val:,}", va="center", fontsize=9, color=IBM_GRAY)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        return self._fig_to_base64(fig)

    def _chart_entity_by_type(self) -> str:
        """Bar chart of entity counts broken down by entity type."""
        if self.entities_df.empty or "entity_type" not in self.entities_df.columns:
            return self._placeholder_chart("No entity type data available")
        vc = self.entities_df["entity_type"].value_counts().head(10)
        if vc.empty:
            return self._placeholder_chart("No entity types to display")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = PALETTE[: len(vc)]
        bars = ax.bar(vc.index, vc.values, color=colors, edgecolor="white", width=0.6)
        ax.set_ylabel("Count")
        ax.set_title("Entities by Type")
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{int(bar.get_height()):,}", ha="center", fontsize=9, color=IBM_GRAY)
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        return self._fig_to_base64(fig)

    def _chart_entity_heatmap(self) -> str:
        if self.entities_df.empty or not {"ticket_id", "entity_type"}.issubset(self.entities_df.columns):
            return self._placeholder_chart("Insufficient entity columns for heatmap")
        pivot = self.entities_df.pivot_table(index="entity_type", columns="entity", values="ticket_id", aggfunc="nunique", fill_value=0)
        if pivot.empty or pivot.shape[0] < 2 or pivot.shape[1] < 2:
            return self._placeholder_chart("Not enough variety for entity co-occurrence heatmap")
        cooc = pivot.values @ pivot.values.T
        labels = list(pivot.index)
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = LinearSegmentedColormap.from_list("ibm_heat", [IBM_LIGHT, IBM_NAVY, IBM_DARK])
        sns.heatmap(cooc, ax=ax, cmap=cmap, annot=True, fmt="d", linewidths=0.5, xticklabels=labels, yticklabels=labels, square=True)
        ax.set_title("Entity Type Co-occurrence")
        fig.tight_layout()
        return self._fig_to_base64(fig)

    def _placeholder_chart(self, msg: str) -> str:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14, color=IBM_GRAY, style="italic", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.patch.set_facecolor("white")
        fig.tight_layout()
        return self._fig_to_base64(fig)

    # ------------------------------------------------------------------
    # Section content generators
    # ------------------------------------------------------------------

    def _executive_summary(self) -> Dict[str, Any]:
        total_tickets = max(len(self.topics_df), len(self.sentiment_df), len(self.entities_df))
        if total_tickets == 0 and not self.tickets_df.empty:
            total_tickets = len(self.tickets_df)

        sentiment_counts = self._safe_value_counts(self.sentiment_df, "sentiment_label")
        positive = int(sentiment_counts.get("positive", 0))
        neutral = int(sentiment_counts.get("neutral", 0))
        negative = int(sentiment_counts.get("negative", 0))

        avg_sentiment = None
        if not self.sentiment_df.empty and "sentiment_score" in self.sentiment_df.columns:
            avg_sentiment = round(float(self.sentiment_df["sentiment_score"].mean()), 3)

        topics_vc = self._safe_value_counts(self.topics_df, "topic_label" if "topic_label" in self.topics_df.columns else "topic")
        top_topic = topics_vc.index[0] if not topics_vc.empty else "N/A"
        top_topic_count = int(topics_vc.iloc[0]) if not topics_vc.empty else 0

        critical_tickets = self._flag_critical_tickets()
        critical_count = len(critical_tickets)

        return {
            "total_tickets": total_tickets,
            "positive": positive,
            "neutral": neutral,
            "negative": negative,
            "avg_sentiment": avg_sentiment,
            "top_topic": top_topic,
            "top_topic_count": top_topic_count,
            "top_topic_pct": self._pct(top_topic_count, total_tickets),
            "critical_count": critical_count,
            "critical_pct": self._pct(critical_count, total_tickets),
        }

    def _flag_critical_tickets(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        if not self.sentiment_df.empty and "sentiment_score" in self.sentiment_df.columns:
            neg = self.sentiment_df[self.sentiment_df["sentiment_score"] <= self.critical_sentiment_threshold].copy()
            if not neg.empty:
                neg["flag_reason"] = "High negative sentiment"
                frames.append(neg)

        if not self.entities_df.empty:
            expensive_keywords = {"refund", "lawsuit", "legal", "escalation", "complaint", "regulatory", "safety"}
            if "entity" in self.entities_df.columns:
                mask = self.entities_df["entity"].str.lower().isin(expensive_keywords)
                flagged = self.entities_df[mask].copy()
                if not flagged.empty:
                    flagged["flag_reason"] = "Expensive/critical entity mention"
                    frames.append(flagged)

        if not self.tickets_df.empty and "priority" in self.tickets_df.columns:
            high = self.tickets_df[self.tickets_df["priority"].str.lower().isin(["critical", "urgent", "p1"])].copy()
            if not high.empty:
                high["flag_reason"] = "High priority ticket"
                frames.append(high)

        if not frames:
            return pd.DataFrame(columns=["ticket_id", "flag_reason"])
        combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ticket_id"])
        return combined

    def _critical_alerts_section(self) -> List[Dict[str, str]]:
        crit = self._flag_critical_tickets()
        alerts: List[Dict[str, str]] = []
        for _, row in crit.head(20).iterrows():
            tid = row.get("ticket_id", "—")
            reason = row.get("flag_reason", "Flagged")
            alerts.append({"ticket_id": str(tid), "reason": str(reason)})
        return alerts

    def _recommendations(self) -> List[str]:
        recs: List[str] = []
        summary = self._executive_summary()
        total = summary["total_tickets"]

        if summary["top_topic"] != "N/A":
            recs.append(
                f"Focus resources on '{summary['top_topic']}' — the leading complaint category "
                f"accounting for {summary['top_topic_pct']} of all tickets ({summary['top_topic_count']:,} tickets)."
            )

        if summary["avg_sentiment"] is not None and summary["avg_sentiment"] < 0:
            recs.append(
                f"Overall average sentiment is negative ({summary['avg_sentiment']:.2f}). "
                "Investigate systemic service gaps driving customer dissatisfaction."
            )

        if summary["critical_count"] > 0:
            recs.append(
                f"{summary['critical_count']} tickets ({summary['critical_pct']}) were flagged as critical. "
                "Prioritise immediate follow-up and root-cause analysis on these."
            )

        neg_pct = summary["negative"] / total * 100 if total else 0
        if neg_pct > 30:
            recs.append(
                f"Negative sentiment accounts for {neg_pct:.1f}% of tickets — above the 30% threshold. "
                "Recommend an urgent cross-functional review."
            )

        topics_vc = self._safe_value_counts(self.topics_df, "topic_label" if "topic_label" in self.topics_df.columns else "topic")
        if len(topics_vc) >= 3:
            top3_pct = topics_vc.head(3).sum() / total * 100 if total else 0
            recs.append(
                f"The top 3 complaint categories cover {top3_pct:.1f}% of tickets. "
                "Tackling these will yield the highest impact."
            )

        if not recs:
            recs.append("Insufficient data to generate automated recommendations. Continue data collection.")

        return recs

    # ------------------------------------------------------------------
    # HTML rendering
    # ------------------------------------------------------------------

    def _render_html(
        self,
        summary: Dict[str, Any],
        charts: Dict[str, str],
        word_clouds: List[Tuple[str, str]],
        critical_alerts: List[Dict[str, str]],
        recommendations: List[str],
    ) -> str:
        period_end = datetime.now()
        period_start = period_end - timedelta(days=self.reporting_period_days)

        word_cloud_html = ""
        if word_clouds:
            cloud_cells = ""
            for topic_label, b64 in word_clouds:
                cloud_cells += f"""
                <div class="cloud-cell">
                    <img src="data:image/png;base64,{b64}" alt="Word Cloud — {topic_label}" />
                    <p class="cloud-label">{topic_label}</p>
                </div>"""
            word_cloud_html = f'<div class="cloud-row">{cloud_cells}</div>'

        topic_analysis_content = ""
        if not self.topics_df.empty:
            topic_analysis_content = f"""
            <div class="chart-row">
                <div class="chart-cell">
                    <img src="data:image/png;base64,{charts.get('topic_bar', '')}" alt="Topic Bar Chart" />
                </div>
                <div class="chart-cell">
                    <img src="data:image/png;base64,{charts.get('topic_pie', '')}" alt="Topic Pie Chart" />
                </div>
            </div>
            {word_cloud_html}"""
        else:
            topic_analysis_content = '<p class="no-data">No topic data available for this reporting period.</p>'

        sentiment_content = ""
        if not self.sentiment_df.empty:
            sentiment_content = f"""
            <div class="chart-row">
                <div class="chart-cell">
                    <img src="data:image/png;base64,{charts.get('sentiment_trend', '')}" alt="Sentiment Trend" />
                </div>
                <div class="chart-cell">
                    <img src="data:image/png;base64,{charts.get('sentiment_dist', '')}" alt="Sentiment Distribution" />
                </div>
            </div>
            <div class="chart-full">
                <img src="data:image/png;base64,{charts.get('sentiment_by_topic', '')}" alt="Sentiment by Topic" />
            </div>"""
        else:
            sentiment_content = '<p class="no-data">No sentiment data available for this reporting period.</p>'

        entity_content = ""
        if not self.entities_df.empty:
            entity_content = f"""
            <div class="chart-row">
                <div class="chart-cell">
                    <img src="data:image/png;base64,{charts.get('entity_bar', '')}" alt="Entity Mentions" />
                </div>
                <div class="chart-cell">
                    <img src="data:image/png;base64,{charts.get('entity_by_type', '')}" alt="Entities by Type" />
                </div>
            </div>
            <div class="chart-full">
                <img src="data:image/png;base64,{charts.get('entity_heatmap', '')}" alt="Entity Co-occurrence Heatmap" />
            </div>"""
        else:
            entity_content = '<p class="no-data">No entity data available for this reporting period.</p>'

        alert_rows = ""
        for i, a in enumerate(critical_alerts, 1):
            alert_rows += f"""
            <tr>
                <td>{i}</td>
                <td><code>{a['ticket_id']}</code></td>
                <td>{a['reason']}</td>
            </tr>"""

        if not critical_alerts:
            alert_rows = '<tr><td colspan="3" style="text-align:center;color:#525252;">No critical alerts this period.</td></tr>'

        rec_items = "".join(f"<li>{r}</li>" for r in recommendations)
        avg_sent_display = f'{summary["avg_sentiment"]:.3f}' if summary["avg_sentiment"] is not None else "N/A"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{self.report_title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
      font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
      background: #F4F4F4; color: #161616; line-height: 1.6;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; background: #fff; box-shadow: 0 1px 8px rgba(0,0,0,0.06); }}
  .header {{
      background: linear-gradient(135deg, #162050 0%, #1F70C1 100%);
      color: #fff; padding: 40px 48px 32px;
  }}
  .header h1 {{ font-size: 26px; font-weight: 600; margin-bottom: 4px; }}
  .header .subtitle {{ font-size: 14px; color: #A6C8FF; }}
  .meta-bar {{
      display: flex; gap: 32px; padding: 16px 48px;
      background: #F0F4F8; border-bottom: 1px solid #E0E0E0;
      font-size: 13px; color: #525252;
  }}
  .meta-bar span {{ font-weight: 600; color: #161616; }}
  .section {{ padding: 32px 48px; border-bottom: 1px solid #E0E0E0; }}
  .section:last-child {{ border-bottom: none; }}
  .section h2 {{
      font-size: 18px; font-weight: 600; color: #162050;
      margin-bottom: 16px; padding-bottom: 8px;
      border-bottom: 3px solid #0F62FE;
      display: inline-block;
  }}
  .section h3 {{ font-size: 15px; font-weight: 600; color: #1F70C1; margin: 20px 0 10px; }}
  .kpi-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px; margin-bottom: 20px;
  }}
  .kpi-card {{
      background: #F4F4F4; border-radius: 8px; padding: 20px 16px;
      text-align: center; border-left: 4px solid #0F62FE;
  }}
  .kpi-card.alert {{ border-left-color: #DA1E28; }}
  .kpi-card.success {{ border-left-color: #198038; }}
  .kpi-value {{ font-size: 28px; font-weight: 700; color: #162050; }}
  .kpi-label {{ font-size: 12px; color: #525252; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 16px 0; }}
  .chart-cell {{ text-align: center; }}
  .chart-cell img {{ max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #E0E0E0; }}
  .chart-full {{ margin: 16px 0; text-align: center; }}
  .chart-full img {{ max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #E0E0E0; }}
  .cloud-row {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 16px 0; justify-content: center; }}
  .cloud-cell {{ text-align: center; flex: 0 0 auto; }}
  .cloud-cell img {{ border-radius: 4px; border: 1px solid #E0E0E0; }}
  .cloud-label {{ font-size: 12px; color: #525252; margin-top: 4px; }}
  .no-data {{ color: #525252; font-style: italic; padding: 24px; text-align: center; background: #F4F4F4; border-radius: 8px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }}
  th {{ background: #162050; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #E0E0E0; }}
  tr:hover td {{ background: #F0F4F8; }}
  code {{ background: #F0F4F8; padding: 2px 6px; border-radius: 3px; font-size: 12px; color: #1F70C1; }}
  .rec-list {{ list-style: none; padding: 0; }}
  .rec-list li {{
      padding: 14px 18px; margin-bottom: 10px;
      background: #F4F4F4; border-left: 4px solid #0F62FE;
      border-radius: 0 6px 6px 0; font-size: 14px;
  }}
  .footer {{
      padding: 24px 48px; background: #162050; color: #A6C8FF;
      font-size: 12px; text-align: center;
  }}
  @media (max-width: 768px) {{
      .chart-row {{ grid-template-columns: 1fr; }}
      .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
      .header, .section, .meta-bar, .footer {{ padding-left: 20px; padding-right: 20px; }}
  }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>{self.report_title}</h1>
    <div class="subtitle">Intelligence Report &mdash; {period_start.strftime('%B %d')} to {period_end.strftime('%B %d, %Y')}</div>
  </div>

  <div class="meta-bar">
    <div>Report Generated: <span>{self.report_generated_at.strftime('%B %d, %Y %H:%M UTC') if self.report_generated_at else 'N/A'}</span></div>
    <div>Period: <span>{self.reporting_period_days} days</span></div>
    <div>Total Tickets Analyzed: <span>{summary['total_tickets']:,}</span></div>
  </div>

  <div class="section">
    <h2>1. Executive Summary</h2>
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-value">{summary['total_tickets']:,}</div>
        <div class="kpi-label">Total Tickets</div>
      </div>
      <div class="kpi-card success">
        <div class="kpi-value">{summary['positive']:,}</div>
        <div class="kpi-label">Positive ({self._pct(summary['positive'], summary['total_tickets'])})</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value">{summary['neutral']:,}</div>
        <div class="kpi-label">Neutral ({self._pct(summary['neutral'], summary['total_tickets'])})</div>
      </div>
      <div class="kpi-card alert">
        <div class="kpi-value">{summary['negative']:,}</div>
        <div class="kpi-label">Negative ({self._pct(summary['negative'], summary['total_tickets'])})</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value">{avg_sent_display}</div>
        <div class="kpi-label">Avg. Sentiment</div>
      </div>
      <div class="kpi-card alert">
        <div class="kpi-value">{summary['critical_count']:,}</div>
        <div class="kpi-label">Critical Alerts ({summary['critical_pct']})</div>
      </div>
    </div>
    <h3>Key Findings</h3>
    <ul style="padding-left:18px; font-size:14px;">
      <li>Top complaint category: <strong>{summary['top_topic']}</strong> ({summary['top_topic_pct']} of tickets).</li>
      <li>Average sentiment across all tickets: <strong>{avg_sent_display}</strong>.</li>
      <li>{summary['critical_count']} ticket(s) flagged as critical requiring immediate attention.</li>
    </ul>
  </div>

  <div class="section">
    <h2>2. Topic Analysis</h2>
    <p style="font-size:14px; color:#525252; margin-bottom:12px;">
      Complaint categories identified via NLP topic modelling and keyword extraction.
    </p>
    {topic_analysis_content}
  </div>

  <div class="section">
    <h2>3. Sentiment Trends</h2>
    <p style="font-size:14px; color:#525252; margin-bottom:12px;">
      Sentiment classification and temporal analysis across the reporting period.
    </p>
    {sentiment_content}
  </div>

  <div class="section">
    <h2>4. Entity Intelligence</h2>
    <p style="font-size:14px; color:#525252; margin-bottom:12px;">
      Named entities extracted from ticket text &mdash; products, services, people, and organisations most frequently referenced.
    </p>
    {entity_content}
  </div>

  <div class="section">
    <h2>5. Critical Alerts</h2>
    <p style="font-size:14px; color:#525252; margin-bottom:12px;">
      Tickets flagged due to high negative sentiment, expensive entity mentions, or high-priority designation.
    </p>
    <table>
      <thead>
        <tr><th>#</th><th>Ticket ID</th><th>Flag Reason</th></tr>
      </thead>
      <tbody>{alert_rows}</tbody>
    </table>
  </div>

  <div class="section">
    <h2>6. Recommendations</h2>
    <p style="font-size:14px; color:#525252; margin-bottom:12px;">
      Data-driven action items derived from this period's intelligence findings.
    </p>
    <ul class="rec-list">{rec_items}</ul>
  </div>

  <div class="footer">
    <p>Customer Support NLP Intelligence Pipeline &mdash; Confidential</p>
    <p style="margin-top:4px;">Generated automatically. For questions contact the Data &amp; Analytics team.</p>
  </div>

</div>
</body>
</html>"""
        return html

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_report(self) -> str:
        """Build charts, assemble analytics, and return the full HTML string.

        Returns
        -------
        str
            Complete HTML report ready to be saved or served.

        Raises
        ------
        ReportError
            If the report cannot be generated.
        """
        self.report_generated_at = datetime.utcnow()
        logger.info("Report generation started at %s", self.report_generated_at.isoformat())

        try:
            summary = self._executive_summary()

            charts: Dict[str, str] = {
                "topic_bar": self._chart_topic_bar(),
                "topic_pie": self._chart_topic_pie(),
                "sentiment_trend": self._chart_sentiment_trend(),
                "sentiment_dist": self._chart_sentiment_distribution(),
                "sentiment_by_topic": self._chart_sentiment_by_topic(),
                "entity_bar": self._chart_entity_bar(),
                "entity_by_type": self._chart_entity_by_type(),
                "entity_heatmap": self._chart_entity_heatmap(),
            }

            word_clouds = self._chart_word_clouds_per_topic()
            critical_alerts = self._critical_alerts_section()
            recommendations = self._recommendations()

            html = self._render_html(summary, charts, word_clouds, critical_alerts, recommendations)
            logger.info("Report generation completed successfully.")
            return html

        except Exception as exc:
            logger.exception("Report generation failed.")
            raise ReportError(f"Failed to generate report: {exc}") from exc

    def save_report(self, output_path: Union[str, Path], filename: Optional[str] = None) -> Path:
        """Generate and persist the report to an HTML file.

        Parameters
        ----------
        output_path : str or Path
            Directory where the file will be saved. Created if it does not exist.
        filename : str, optional
            Name of the HTML file. Defaults to a timestamped name.

        Returns
        -------
        Path
            Absolute path to the saved HTML file.
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"intelligence_report_{ts}.html"

        html = self.generate_report()
        file_path = output_dir / filename
        file_path.write_text(html, encoding="utf-8")
        logger.info("Report saved to %s", file_path.resolve())
        return file_path.resolve()
