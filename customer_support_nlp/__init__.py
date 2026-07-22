"""Customer Support NLP Intelligence Pipeline.

End-to-end NLP pipeline for analyzing customer support tickets —
topic modelling, sentiment analysis, named entity recognition, and
automated weekly intelligence reporting.
"""

from customer_support_nlp.src.pipeline import IntelligencePipeline, PipelineConfig

__all__ = ["IntelligencePipeline", "PipelineConfig"]
