"""
Tests for hrv_pipeline.py

All tests use NeuroKit2 synthetic ECG - no real Autonomic Aging data needed.
"""
import numpy as np
import neurokit2 as nk
import pytest


def make_ecg(duration=300, sampling_rate=1000, heart_rate=65):
    """Synthetic ECG at 1000 Hz, 5-minute default."""
    return nk.ecg_simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        heart_rate=heart_rate,
        noise=0.01,
        random_state=42,
    )


# process_ecg_segment
class TestProcessEcgSegment:
    def test_returns_dict_with_required_keys(self):
        from src.hrv_pipeline import process_ecg_segment

        result = process_ecg_segment(make_ecg(), sampling_rate=1000)
        assert isinstance(result, dict)
        for key in [
            "hrv_rmssd", "hrv_sdnn", "hrv_pnn50",
            "hrv_mean_rr", "hrv_mean_hr",
            "hrv_lf", "hrv_hf", "hrv_lf_hf", "hrv_total_power",
            "hrv_sample_entropy", "hrv_dfa_alpha1",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_all_values_finite_or_none(self):
        from src.hrv_pipeline import process_ecg_segment

        result = process_ecg_segment(make_ecg(), sampling_rate=1000)
        for key, val in result.items():
            assert val is None or np.isfinite(val), f"{key}={val} is not finite"

    def test_rmssd_in_physiological_range(self):
        from src.hrv_pipeline import process_ecg_segment

        result = process_ecg_segment(make_ecg(heart_rate=65), sampling_rate=1000)
        assert result["hrv_rmssd"] is not None
        assert 1 < result["hrv_rmssd"] < 500, f"RMSSD={result['hrv_rmssd']} out of range"

    def test_mean_hr_close_to_simulated(self):
        from src.hrv_pipeline import process_ecg_segment

        result = process_ecg_segment(make_ecg(heart_rate=60), sampling_rate=1000)
        assert result["hrv_mean_hr"] is not None
        assert abs(result["hrv_mean_hr"] - 60) < 5, f"Mean HR={result['hrv_mean_hr']}"

    def test_returns_none_for_signal_under_60s(self):
        from src.hrv_pipeline import process_ecg_segment

        short = make_ecg(duration=30)
        assert process_ecg_segment(short, sampling_rate=1000) is None

    def test_returns_none_for_flat_signal(self):
        from src.hrv_pipeline import process_ecg_segment

        assert process_ecg_segment(np.zeros(300_000), sampling_rate=1000) is None


# load_wfdb_record
class TestLoadWfdbRecord:
    def test_raises_on_missing_file(self):
        from src.hrv_pipeline import load_wfdb_record

        with pytest.raises(FileNotFoundError):
            load_wfdb_record("/nonexistent/path/record")

    def test_returns_ecg_array_fs_from_synthetic_record(self, tmp_path):
        import wfdb
        from src.hrv_pipeline import load_wfdb_record

        ecg = make_ecg(duration=10)
        bp = 120 + 20 * np.sin(2 * np.pi * np.linspace(0, 10, 10_000))
        wfdb.wrsamp(
            "test_rec",
            fs=1000,
            units=["mV", "mmHg"],
            sig_name=["ECG", "ABP"],
            p_signal=np.column_stack([ecg, bp]),
            write_dir=str(tmp_path),
        )
        ecg_out, bp_out, fs = load_wfdb_record(str(tmp_path / "test_rec"))
        assert fs == 1000
        assert ecg_out.shape == (10_000,)
        assert bp_out is not None and bp_out.shape == (10_000,)

    def test_bp_is_none_when_no_bp_channel(self, tmp_path):
        import wfdb
        from src.hrv_pipeline import load_wfdb_record

        ecg = make_ecg(duration=5)
        wfdb.wrsamp(
            "ecg_only",
            fs=1000,
            units=["mV"],
            sig_name=["ECG"],
            p_signal=ecg.reshape(-1, 1),
            write_dir=str(tmp_path),
        )
        _, bp_out, _ = load_wfdb_record(str(tmp_path / "ecg_only"))
        assert bp_out is None


# process_bp_segment
class TestProcessBpSegment:
    def test_returns_dict_with_required_keys(self):
        from src.hrv_pipeline import process_bp_segment

        t = np.linspace(0, 300, 300_000)
        bp = 120 + 20 * np.sin(2 * np.pi * 1.1 * t) + np.random.default_rng(0).normal(0, 2, 300_000)
        result = process_bp_segment(bp, sampling_rate=1000)

        assert isinstance(result, dict)
        for key in ["bp_sbp_mean", "bp_dbp_mean", "bp_sbp_std", "bp_dbp_std", "bp_pulse_pressure_mean"]:
            assert key in result, f"Missing key: {key}"

    def test_returns_none_for_flat_signal(self):
        from src.hrv_pipeline import process_bp_segment

        assert process_bp_segment(np.zeros(300_000), sampling_rate=1000) is None

    def test_returns_none_for_short_signal(self):
        from src.hrv_pipeline import process_bp_segment

        t = np.linspace(0, 10, 10_000)
        bp = 120 + 20 * np.sin(2 * np.pi * 1.1 * t)
        assert process_bp_segment(bp, sampling_rate=1000) is None


# load_subject_metadata
class TestLoadSubjectMetadata:
    def test_loads_csv_and_normalises_columns(self, tmp_path):
        from src.autonomic_aging_processor import load_subject_metadata

        csv_file = tmp_path / "subject-info.csv"
        # Use non-normalised column names to exercise the lowercasing/hyphen transform
        csv_file.write_text("Subject-ID,Age,Gender,BMI\n001,35,Male,24.5\n002,62,Female,27.1\n")

        df = load_subject_metadata(str(csv_file))
        assert list(df.columns) == ["subject_id", "age", "gender", "bmi"]
        assert len(df) == 2

    def test_raises_on_missing_file(self):
        from src.autonomic_aging_processor import load_subject_metadata

        with pytest.raises(FileNotFoundError):
            load_subject_metadata("/nonexistent/subject-info.csv")


# process_single_participant
class TestProcessSingleParticipant:
    def test_returns_feature_dict(self, tmp_path):
        import wfdb
        from src.autonomic_aging_processor import process_single_participant

        ecg = make_ecg(duration=300, heart_rate=65)
        t = np.linspace(0, 300, 300_000)
        bp = 120 + 20 * np.sin(2 * np.pi * 1.1 * t)
        wfdb.wrsamp(
            "sub001",
            fs=1000,
            units=["mV", "mmHg"],
            sig_name=["ECG", "ABP"],
            p_signal=np.column_stack([ecg, bp]),
            write_dir=str(tmp_path),
        )
        result = process_single_participant(str(tmp_path / "sub001"), subject_id="sub001")

        assert result is not None
        assert result["subject_id"] == "sub001"
        assert "hrv_rmssd" in result
        assert "bp_sbp_mean" in result

    def test_returns_none_on_missing_record(self):
        from src.autonomic_aging_processor import process_single_participant

        assert process_single_participant("/nonexistent/path/999", subject_id="999") is None
