"""Confidence scoring from retrieval similarity scores."""


def score_confidence(top_reranker_score: float) -> str:
    """Map a retrieval similarity score to High/Medium/Low.

    Thresholds are calibrated for cosine similarity derived from OpenAI
    text-embedding-3-small via ``1 - L2²/2`` on unit-normalized vectors.
    Real financial-doc queries typically score 0.35–0.65 against relevant
    chunks, so 0.55 / 0.40 puts strong semantic matches at High, reasonable
    matches at Medium, and weak matches at Low. Revisit once a real query
    log is available — if most answers still land Low, drop the bands.
    """
    if top_reranker_score >= 0.55:
        return "High"
    if top_reranker_score >= 0.40:
        return "Medium"
    return "Low"


def confidence_color(confidence_level: str) -> str:
    """Return a dark hex color for the confidence badge."""
    return {"High": "#27500A", "Medium": "#633806", "Low": "#791F1F"}.get(confidence_level, "#791F1F")
