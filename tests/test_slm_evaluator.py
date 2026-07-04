import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slm_evaluator import (
    category_coverage,
    classification_exact_match,
    evaluate_generation,
    rouge_l,
)


COMPLETE_RESPONSE = (
    "AGING: Favorable aging trajectory.\n\n"
    "STRESS: Not measured from NHANES accelerometer.\n\n"
    "METABOLISM: Pre-diabetic HbA1c reading.\n\n"
    "INFLAMMATION: Elevated NLR indicating inflammaging.\n\n"
    "SLEEP: Short sleep duration observed.\n"
)


def test_category_coverage_all_present():
    cov = category_coverage(COMPLETE_RESPONSE)
    assert cov == {
        "AGING": True, "STRESS": True, "METABOLISM": True,
        "INFLAMMATION": True, "SLEEP": True,
    }


def test_category_coverage_missing_section():
    partial = COMPLETE_RESPONSE.replace("SLEEP:", "SUMMARY:")
    cov = category_coverage(partial)
    assert cov["SLEEP"] is False
    assert cov["AGING"] is True


def test_classification_exact_match_all_correct():
    expected = {
        "hba1c": ("Pre-diabetic", ""),
        "nlr": ("Elevated", ""),
        "aging": ("Favorable", ""),
        "sleep": ("Short", ""),
    }
    matches = classification_exact_match(COMPLETE_RESPONSE, expected)
    assert matches == {"hba1c": True, "nlr": True, "aging": True, "sleep": True}


def test_classification_exact_match_wrong_label():
    expected = {
        "hba1c": ("Diabetic", ""),
        "nlr": ("Normal", ""),
        "aging": ("Favorable", ""),
        "sleep": ("Adequate", ""),
    }
    matches = classification_exact_match(COMPLETE_RESPONSE, expected)
    assert matches["hba1c"] is False
    assert matches["nlr"] is False
    assert matches["aging"] is True
    assert matches["sleep"] is False


def test_rouge_l_identical_strings_is_one():
    assert rouge_l("a b c d", "a b c d") == 1.0


def test_rouge_l_disjoint_is_zero():
    assert rouge_l("a b c", "x y z") == 0.0


def test_rouge_l_partial_overlap_between_0_and_1():
    score = rouge_l("the cat sat on the mat", "the cat is on the mat")
    assert 0.0 < score < 1.0


def test_evaluate_generation_aggregates():
    expected = {
        "hba1c": ("Pre-diabetic", ""),
        "nlr": ("Elevated", ""),
        "aging": ("Favorable", ""),
        "sleep": ("Short", ""),
    }
    result = evaluate_generation(
        generated=COMPLETE_RESPONSE,
        reference=COMPLETE_RESPONSE,
        expected_classifications=expected,
    )
    assert result["category_coverage_all"] is True
    assert result["classification_match_rate"] == 1.0
    assert result["rouge_l"] == 1.0
