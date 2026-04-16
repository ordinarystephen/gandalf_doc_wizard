"""Confidence scoring from cross-encoder reranker scores."""


def score_confidence(top_reranker_score: float) -> str:
    """Map a reranker score to High/Medium/Low.

    Thresholds are conservative for financial docs: a low-confidence
    answer about a DSCR or repayment source could have material consequences.
    """
    if top_reranker_score >= 0.85:
        return "High"
    if top_reranker_score >= 0.60:
        return "Medium"
    return "Low"


def confidence_color(confidence_level: str) -> str:
    """Return a dark hex color for the confidence badge."""
    return {"High": "#27500A", "Medium": "#633806", "Low": "#791F1F"}[confidence_level]
