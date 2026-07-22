# Customer Support NLP Intelligence Pipeline

Production-grade NLP pipeline that transforms raw customer support tickets into actionable intelligence reports. Combines BERTopic topic modeling, hybrid sentiment analysis (VADER + DistilBERT), and spaCy NER for entity extraction — all auto-generating a weekly intelligence report in IBM consulting deliverable format.

## Architecture

```
customer_support_nlp/
├── run.py                    # CLI entry point (run / generate-sample / report-only)
├── src/
│   ├── topic_modeling.py     # BERTopic-based topic extraction
│   ├── sentiment.py          # VADER + DistilBERT hybrid sentiment
│   ├── named_entity.py       # spaCy NER with entity flagging
│   ├── reporting.py          # IBM-style HTML intelligence report
│   ├── pipeline.py           # End-to-end orchestration
│   └── data_utils.py         # Data loading, validation, synthetic generation
├── configs/
│   └── default.yaml          # Default configuration
└── outputs/                  # Generated reports and enriched data
```

## Features

### Topic Modeling (BERTopic)
- Sentence-transformers embeddings (all-MiniLM-L6-v2)
- UMAP dimensionality reduction + HDBSCAN clustering
- Automatic topic count detection
- Topic hierarchy and intertopic distance visualization
- Per-topic representative documents and keyword extraction

### Sentiment Analysis (Hybrid)
- **VADER**: Fast lexicon-based compound scoring
- **DistilBERT**: Fine-tuned transformer (distilbert-base-uncased-finetuned-sst-2-english)
- Weighted hybrid score with confidence aggregation
- Trend detection with rolling averages and shift alerts
- Critical ticket flagging (compound < -0.7)

### Entity Extraction (spaCy NER)
- Entity types: PERSON, ORG, PRODUCT, DATE, MONEY, GPE
- Entity frequency and type distribution analysis
- Critical entity flagging: products in complaints, amounts > $100
- Entity co-occurrence analysis across tickets

### Intelligence Report (IBM Format)
- Executive summary with key metrics and critical alerts
- Topic analysis with bar charts, pie charts, word clouds
- Sentiment trends with time-series visualization
- Entity intelligence with co-occurrence heatmap
- Auto-generated recommendations based on data thresholds
- Professional navy/blue corporate styling

## Quick Start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Generate sample data and run full pipeline
python -m customer_support_nlp.run generate-sample --n 200 --output sample_tickets.csv
python -m customer_support_nlp.run run --input sample_tickets.csv --output outputs/

# Run on your own data
python -m customer_support_nlp.run run --input tickets.csv --config configs/default.yaml --output outputs/

# Generate report from pre-computed data
python -m customer_support_nlp.run report-only --data enriched_tickets.csv --output outputs/
```

## Configuration

Edit `configs/default.yaml` to tune:

```yaml
topic_modeling:
  n_topics: auto
  embedding_model: all-MiniLM-L6-v2
  umap_n_neighbors: 15
  hdbscan_min_cluster_size: 10

sentiment:
  vader_weight: 0.4
  distilbert_weight: 0.6
  critical_threshold: -0.7

ner:
  spacy_model: en_core_web_sm
  critical_money_threshold: 100.0
```

## Output Files

| File | Description |
|------|-------------|
| `enriched_tickets.csv` | Original tickets + topic, sentiment, entities |
| `topic_summary.csv` | Topic keywords, counts, representative docs |
| `entity_summary.csv` | Entity frequencies, types, critical flags |
| `intelligence_report_*.html` | Full IBM-style HTML report |

## Dependencies

- numpy, pandas, scikit-learn
- bertopic, sentence-transformers, umap-learn, hdbscan
- spacy, nltk, transformers, torch
- matplotlib, seaborn, wordcloud
- pyyaml
