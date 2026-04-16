import sys; sys.path.insert(0, '.')
from doc_qa.utils.confidence import score_confidence, confidence_color

def test_high_confidence():
    assert score_confidence(0.85) == "High"
    assert score_confidence(1.0) == "High"

def test_medium_confidence():
    assert score_confidence(0.60) == "Medium"
    assert score_confidence(0.84) == "Medium"

def test_low_confidence():
    assert score_confidence(0.59) == "Low"
    assert score_confidence(0.0) == "Low"

def test_confidence_colors():
    assert confidence_color("High") == "#27500A"
    assert confidence_color("Medium") == "#633806"
    assert confidence_color("Low") == "#791F1F"
