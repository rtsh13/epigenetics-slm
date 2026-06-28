import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomarker_classifier import (
    classify_aging,
    classify_circadian,
    classify_hba1c,
    classify_nlr,
    classify_sleep,
)


class TestClassifyHba1c:
    def test_normal_below_5_7(self):
        label, interp = classify_hba1c(5.4)
        assert label == "Normal"
        assert "stable" in interp.lower() or "normal" in interp.lower()

    def test_prediabetic_boundary_5_7(self):
        label, _ = classify_hba1c(5.7)
        assert label == "Pre-diabetic"

    def test_prediabetic_6_2(self):
        label, _ = classify_hba1c(6.2)
        assert label == "Pre-diabetic"

    def test_diabetic_boundary_6_5(self):
        label, _ = classify_hba1c(6.5)
        assert label == "Diabetic"

    def test_diabetic_high(self):
        label, interp = classify_hba1c(8.1)
        assert label == "Diabetic"
        assert "hyperglycemia" in interp.lower() or "diabetic" in interp.lower()


class TestClassifyNlr:
    def test_normal_below_3(self):
        label, interp = classify_nlr(2.1)
        assert label == "Normal"
        assert "no" in interp.lower() or "normal" in interp.lower()

    def test_elevated_boundary_3(self):
        label, _ = classify_nlr(3.0)
        assert label == "Elevated"

    def test_elevated_4_5(self):
        label, _ = classify_nlr(4.5)
        assert label == "Elevated"

    def test_high_boundary_6(self):
        label, _ = classify_nlr(6.0)
        assert label == "High"

    def test_high_8(self):
        label, interp = classify_nlr(8.0)
        assert label == "High"
        assert "inflammation" in interp.lower()


class TestClassifyAging:
    def test_close_to_chrono(self):
        label, interp = classify_aging(2.1)
        assert label == "Favorable"
        assert "favorable" in interp.lower() or "closely" in interp.lower()

    def test_modest_acceleration(self):
        label, _ = classify_aging(5.0)
        assert label == "Modest"

    def test_significant_acceleration(self):
        label, interp = classify_aging(12.0)
        assert label == "Significant"
        assert "accelerat" in interp.lower()


class TestClassifyCircadian:
    def test_strong_rhythm(self):
        label, interp = classify_circadian(0.65, 0.55, 0.78)
        assert label == "Stable"
        assert "rhythm" in interp.lower() or "consistent" in interp.lower() or "contrast" in interp.lower()

    def test_fragmented(self):
        label, interp = classify_circadian(0.30, 0.95, 0.50)
        assert label == "Fragmented"
        assert "fragment" in interp.lower()

    def test_borderline(self):
        label, _ = classify_circadian(0.45, 0.70, 0.65)
        assert label == "Borderline"


class TestClassifySleep:
    def test_adequate_regular(self):
        label, interp = classify_sleep(450.0, 75.0)
        assert label == "Adequate"
        assert "adequate" in interp.lower() or "consistent" in interp.lower()

    def test_short_sleep(self):
        label, interp = classify_sleep(300.0, 65.0)
        assert label == "Short"
        assert "short" in interp.lower()

    def test_irregular(self):
        label, interp = classify_sleep(440.0, 35.0)
        assert label == "Irregular"
        assert "irregular" in interp.lower() or "misalign" in interp.lower()
