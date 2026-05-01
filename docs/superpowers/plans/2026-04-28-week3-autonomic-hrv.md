# Week 3: Autonomic Aging HRV Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable HRV feature extraction pipeline on the Autonomic Aging dataset, producing `data/autonomic_features.parquet` (~1,121 rows, ~20 features per participant).

**Architecture:** `hrv_pipeline.py` handles all signal processing (ECG/BP waveform → HRV features) as a pure library; `autonomic_aging_processor.py` is the batch orchestrator that discovers WFDB records, calls the pipeline, checkpoints progress, and saves results. Tests run entirely against NeuroKit2 synthetic ECG — no real dataset download required to run the test suite.

**Tech Stack:** `wfdb` (WFDB file I/O), `neurokit2==0.2.13` (ECG cleaning + peak detection + HRV), `scipy` (BP peak detection), `pandas`, `numpy`, `matplotlib`

---

## Prerequisites (Manual steps — run once before Task 1)

**1. Install wfdb into the project venv:**
```bash
source wearable-age/bin/activate
pip install wfdb==4.1.2
```

**2. Download the Autonomic Aging dataset** (~2–3 GB, requires PhysioNet account logged in):
```bash
mkdir -p data/autonomic_aging
cd data/autonomic_aging
wget -r -N -c -np https://physionet.org/files/autonomic-aging-cardiovascular/1.0.0/
```
Or via wfdb Python API (from within the activated venv):
```python
import wfdb
wfdb.dl_database('autonomic-aging-cardiovascular', dl_dir='data/autonomic_aging/')
```

After download, `data/autonomic_aging/` should contain:
- `subject-info.csv` — participant metadata (subject_id, age, gender, bmi)
- Per-participant WFDB files: `{id}.hea`, `{id}.dat` (ECG + BP channels)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/hrv_pipeline.py` | **Create** | Signal processing library: ECG → HRV features, WFDB loading, BP variability |
| `src/autonomic_aging_processor.py` | **Create** | Batch orchestrator: discover records, call pipeline, checkpoint, save parquet |
| `tests/__init__.py` | **Create** | Empty, makes `tests/` a package |
| `tests/test_hrv_pipeline.py` | **Create** | Unit tests using synthetic ECG; no real data required |
| `progress/week3-progress.md` | **Create** | Human-readable progress notes |
| `requirements.txt` | **Modify** | Add `wfdb==4.1.2` |

---

## Task 1: Test scaffolding + requirements

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/test_hrv_pipeline.py`

- [ ] **Step 1: Add wfdb to requirements.txt**

Open `requirements.txt` and add after `xgboost==3.2.0`:
```
wfdb==4.1.2
```

- [ ] **Step 2: Create `tests/__init__.py`**

Create an empty file at `tests/__init__.py`.

- [ ] **Step 3: Write failing tests for `process_ecg_segment`**

Create `tests/test_hrv_pipeline.py`:

```python
"""
Tests for hrv_pipeline.py

All tests use NeuroKit2 synthetic ECG — no real Autonomic Aging data needed.
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


# ---------------------------------------------------------------------------
# process_ecg_segment
# ---------------------------------------------------------------------------
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
        assert abs(result["hrv_mean_hr"] - 60) < 15, f"Mean HR={result['hrv_mean_hr']}"

    def test_returns_none_for_signal_under_60s(self):
        from src.hrv_pipeline import process_ecg_segment

        short = make_ecg(duration=30)
        assert process_ecg_segment(short, sampling_rate=1000) is None

    def test_returns_none_for_flat_signal(self):
        from src.hrv_pipeline import process_ecg_segment

        assert process_ecg_segment(np.zeros(300_000), sampling_rate=1000) is None


# ---------------------------------------------------------------------------
# load_wfdb_record
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# process_bp_segment
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# load_subject_metadata
# ---------------------------------------------------------------------------
class TestLoadSubjectMetadata:
    def test_loads_csv_and_normalises_columns(self, tmp_path):
        from src.autonomic_aging_processor import load_subject_metadata

        csv_file = tmp_path / "subject-info.csv"
        csv_file.write_text("subject_id,age,gender,bmi\n001,35,Male,24.5\n002,62,Female,27.1\n")

        df = load_subject_metadata(str(csv_file))
        assert list(df.columns) == ["subject_id", "age", "gender", "bmi"]
        assert len(df) == 2

    def test_raises_on_missing_file(self):
        from src.autonomic_aging_processor import load_subject_metadata

        with pytest.raises(FileNotFoundError):
            load_subject_metadata("/nonexistent/subject-info.csv")


# ---------------------------------------------------------------------------
# process_single_participant
# ---------------------------------------------------------------------------
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
```

- [ ] **Step 4: Confirm tests fail (nothing implemented yet)**

```bash
cd /Users/ruchitsingh/Desktop/loopAI/epigenetics_project
source wearable-age/bin/activate
python -m pytest tests/test_hrv_pipeline.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'process_ecg_segment' from 'src.hrv_pipeline'` (or ModuleNotFoundError).

- [ ] **Step 5: Commit scaffolding**

```bash
git add requirements.txt tests/__init__.py tests/test_hrv_pipeline.py
git commit -m "test: add Week 3 HRV pipeline test scaffolding"
```

---

## Task 2: Implement `src/hrv_pipeline.py`

**Files:**
- Create: `src/hrv_pipeline.py`

- [ ] **Step 1: Create `src/hrv_pipeline.py`**

```python
"""
Week 3: HRV Feature Extraction Pipeline

Reusable signal processing library for extracting HRV and cardiovascular
features from ECG and continuous BP recordings.

Designed for the Autonomic Aging dataset (PhysioNet WFDB format) but generic
enough for Apple Watch / Oura Ring RR-interval exports in future.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import find_peaks

log = logging.getLogger(__name__)

# 60 s minimum for time-domain HRV; 120 s for LF/HF ratio
MIN_DURATION_SECONDS = 60
MIN_DURATION_FREQ = 120
MIN_RPEAKS = 30


# ---------------------------------------------------------------------------
# ECG processing
# ---------------------------------------------------------------------------

def process_ecg_segment(
    signal: np.ndarray,
    sampling_rate: int = 1000,
) -> Optional[dict]:
    """Extract HRV features from a 1-D ECG array.

    Returns a flat dict with time-domain, frequency-domain, and nonlinear
    HRV metrics. Returns None if the signal is too short, flat, or yields
    too few R-peaks for reliable analysis.

    Args:
        signal: 1-D ECG signal (mV or arbitrary units).
        sampling_rate: Samples per second (1000 for Autonomic Aging dataset).

    Returns:
        Dict with keys hrv_rmssd, hrv_sdnn, hrv_pnn50, hrv_mean_rr,
        hrv_mean_hr, hrv_lf, hrv_hf, hrv_lf_hf, hrv_total_power,
        hrv_sample_entropy, hrv_dfa_alpha1, hrv_dfa_alpha2.
        Values are float or None if a sub-computation fails.
    """
    duration = len(signal) / sampling_rate

    if duration < MIN_DURATION_SECONDS:
        log.debug("Signal too short (%.1f s < %d s)", duration, MIN_DURATION_SECONDS)
        return None

    if np.std(signal) < 1e-6:
        log.debug("Signal is flat (std=%.2e)", np.std(signal))
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ecg_cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)
            peaks_df, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)

        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) < MIN_RPEAKS:
            log.debug("Too few R-peaks (%d < %d)", len(rpeaks), MIN_RPEAKS)
            return None

        # Time-domain HRV
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hrv_time = nk.hrv_time(peaks_df, sampling_rate=sampling_rate)

        # Frequency-domain HRV (needs >= 2 min for LF band)
        hrv_freq = None
        if duration >= MIN_DURATION_FREQ:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hrv_freq = nk.hrv_frequency(
                        peaks_df,
                        sampling_rate=sampling_rate,
                        psd_method="welch",
                    )
            except Exception as exc:
                log.debug("Frequency HRV failed: %s", exc)

        # Nonlinear HRV
        hrv_nonlinear = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hrv_nonlinear = nk.hrv_nonlinear(peaks_df, sampling_rate=sampling_rate)
        except Exception as exc:
            log.debug("Nonlinear HRV failed: %s", exc)

        return _assemble_hrv_dict(hrv_time, hrv_freq, hrv_nonlinear, rpeaks, sampling_rate)

    except Exception as exc:
        log.warning("ECG processing failed: %s", exc)
        return None


def _assemble_hrv_dict(
    hrv_time: pd.DataFrame,
    hrv_freq: Optional[pd.DataFrame],
    hrv_nonlinear: Optional[pd.DataFrame],
    rpeaks: np.ndarray,
    sampling_rate: int,
) -> dict:
    """Flatten NeuroKit2 HRV DataFrames into one dict with standardised keys."""

    def _get(df: Optional[pd.DataFrame], col: str) -> Optional[float]:
        if df is None or col not in df.columns:
            return None
        val = df[col].iloc[0]
        return float(val) if pd.notna(val) else None

    rr_ms = np.diff(rpeaks) / sampling_rate * 1000.0
    mean_rr = float(np.mean(rr_ms)) if len(rr_ms) > 0 else None
    mean_hr = float(60_000.0 / mean_rr) if (mean_rr and mean_rr > 0) else None

    return {
        # Time domain
        "hrv_rmssd":   _get(hrv_time, "HRV_RMSSD"),
        "hrv_sdnn":    _get(hrv_time, "HRV_SDNN"),
        "hrv_pnn50":   _get(hrv_time, "HRV_pNN50"),
        "hrv_mean_rr": mean_rr,
        "hrv_mean_hr": mean_hr,
        # Frequency domain
        "hrv_lf":          _get(hrv_freq, "HRV_LF"),
        "hrv_hf":          _get(hrv_freq, "HRV_HF"),
        "hrv_lf_hf":       _get(hrv_freq, "HRV_LFHF"),
        "hrv_total_power": _get(hrv_freq, "HRV_TP"),
        # Nonlinear
        "hrv_sample_entropy": _get(hrv_nonlinear, "HRV_SampEn"),
        "hrv_dfa_alpha1":     _get(hrv_nonlinear, "HRV_DFA_alpha1"),
        "hrv_dfa_alpha2":     _get(hrv_nonlinear, "HRV_DFA_alpha2"),
    }


# ---------------------------------------------------------------------------
# WFDB loading
# ---------------------------------------------------------------------------

def load_wfdb_record(record_path: str) -> tuple[np.ndarray, Optional[np.ndarray], int]:
    """Load a PhysioNet WFDB record and return (ecg_signal, bp_signal, fs).

    Identifies the ECG channel by name (ECG, II, I, EKG) and the BP channel
    (ABP, BP, SBP, NIBP). Falls back to channel 0 for ECG if none match.
    Returns None for bp_signal when no BP channel is found.

    Args:
        record_path: Absolute path without extension, e.g. 'data/autonomic_aging/001'.

    Raises:
        FileNotFoundError: If the .hea header file does not exist.
    """
    hea_path = Path(record_path).with_suffix(".hea")
    if not hea_path.exists():
        raise FileNotFoundError(f"WFDB header not found: {hea_path}")

    record = wfdb.rdrecord(str(record_path))
    signals = record.p_signal          # shape: (n_samples, n_channels)
    fs = record.fs
    sig_names_upper = [s.upper() for s in record.sig_name]

    # Identify ECG channel
    ecg_ch = None
    for candidate in ("ECG", "II", "I", "EKG"):
        if candidate in sig_names_upper:
            ecg_ch = sig_names_upper.index(candidate)
            break
    if ecg_ch is None:
        ecg_ch = 0
        log.debug("No named ECG channel; using channel 0 (%s)", record.sig_name[0])

    ecg_signal = signals[:, ecg_ch].astype(float)

    # Identify BP channel (optional)
    bp_signal = None
    for candidate in ("ABP", "BP", "SBP", "NIBP"):
        if candidate in sig_names_upper:
            bp_ch = sig_names_upper.index(candidate)
            bp_signal = signals[:, bp_ch].astype(float)
            break

    return ecg_signal, bp_signal, fs


# ---------------------------------------------------------------------------
# BP processing
# ---------------------------------------------------------------------------

def process_bp_segment(
    signal: np.ndarray,
    sampling_rate: int = 1000,
) -> Optional[dict]:
    """Extract blood pressure variability features from a continuous BP signal.

    Detects systolic peaks and diastolic troughs using scipy peak finding, then
    computes mean and std for SBP, DBP, and pulse pressure.

    Args:
        signal: 1-D continuous ABP/BP signal in mmHg.
        sampling_rate: Samples per second.

    Returns:
        Dict with bp_sbp_mean, bp_dbp_mean, bp_sbp_std, bp_dbp_std,
        bp_pulse_pressure_mean; or None on failure.
    """
    duration = len(signal) / sampling_rate

    if duration < MIN_DURATION_SECONDS:
        return None
    if np.std(signal) < 1e-6:
        return None

    try:
        min_dist = int(0.4 * sampling_rate)  # max ~150 bpm
        peaks, _ = find_peaks(signal, distance=min_dist, prominence=5)
        troughs, _ = find_peaks(-signal, distance=min_dist, prominence=5)

        if len(peaks) < 10 or len(troughs) < 10:
            return None

        sbp = signal[peaks]
        dbp = signal[troughs]
        n = min(len(sbp), len(dbp))
        sbp, dbp = sbp[:n], dbp[:n]
        pp = sbp - dbp

        return {
            "bp_sbp_mean":            float(np.mean(sbp)),
            "bp_dbp_mean":            float(np.mean(dbp)),
            "bp_sbp_std":             float(np.std(sbp)),
            "bp_dbp_std":             float(np.std(dbp)),
            "bp_pulse_pressure_mean": float(np.mean(pp)),
        }
    except Exception as exc:
        log.warning("BP processing failed: %s", exc)
        return None
```

- [ ] **Step 2: Run ECG and WFDB tests**

```bash
python -m pytest tests/test_hrv_pipeline.py::TestProcessEcgSegment tests/test_hrv_pipeline.py::TestLoadWfdbRecord tests/test_hrv_pipeline.py::TestProcessBpSegment -v
```

Expected: 11 tests pass. Note: `TestProcessEcgSegment::test_rmssd_in_physiological_range` — NeuroKit2 synthetic ECG RMSSD can be anywhere in the 1–500 ms range.

If any frequency or nonlinear test fails with `None` values — that is acceptable; the test only checks keys exist and values are finite-or-None.

- [ ] **Step 3: Commit**

```bash
git add src/hrv_pipeline.py
git commit -m "feat: implement hrv_pipeline with ECG/HRV processing and WFDB loading"
```

---

## Task 3: Implement `src/autonomic_aging_processor.py`

**Files:**
- Create: `src/autonomic_aging_processor.py`

- [ ] **Step 1: Run `TestLoadSubjectMetadata` and `TestProcessSingleParticipant` to confirm they fail**

```bash
python -m pytest tests/test_hrv_pipeline.py::TestLoadSubjectMetadata tests/test_hrv_pipeline.py::TestProcessSingleParticipant -v 2>&1 | head -15
```

Expected: `ImportError: cannot import name 'load_subject_metadata' from 'src.autonomic_aging_processor'`

- [ ] **Step 2: Create `src/autonomic_aging_processor.py`**

```python
"""
Week 3: Autonomic Aging Dataset Batch Processor

Loads WFDB recordings from data/autonomic_aging/, extracts HRV and BP
features via hrv_pipeline.py, merges with participant metadata, and saves
data/autonomic_features.parquet.

Dataset: https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from hrv_pipeline import load_wfdb_record, process_ecg_segment, process_bp_segment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "autonomic_aging"
SUBJECT_CSV = DATA_DIR / "subject-info.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "autonomic_features.parquet"
CHECKPOINT_PATH = PROJECT_DIR / "data" / "autonomic_checkpoint.json"
EDA_DIR = PROJECT_DIR / "reports" / "week3_eda"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def load_subject_metadata(csv_path: str) -> pd.DataFrame:
    """Load participant metadata from subject-info.csv.

    Column names are normalised to lowercase with underscores.
    The column that serves as participant ID must be named 'subject_id'
    in the CSV (or normalise to it after lowercasing).

    Raises:
        FileNotFoundError: If csv_path does not exist.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"subject-info.csv not found: {p}")

    df = pd.read_csv(p)
    df.columns = [c.lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    # Ensure subject_id is string for consistent joining
    if "subject_id" in df.columns:
        df["subject_id"] = df["subject_id"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Per-participant processing
# ---------------------------------------------------------------------------

def process_single_participant(record_path: str, subject_id: str) -> dict | None:
    """Load one WFDB record and return a flat feature dict.

    Returns None if the file is missing or HRV extraction fails.
    BP features are included when a BP channel is present; missing BP is not
    treated as a failure.
    """
    try:
        ecg_signal, bp_signal, fs = load_wfdb_record(record_path)
    except FileNotFoundError:
        log.warning("Record not found: %s", record_path)
        return None
    except Exception as exc:
        log.warning("WFDB load failed for %s: %s", subject_id, exc)
        return None

    hrv_features = process_ecg_segment(ecg_signal, sampling_rate=fs)
    if hrv_features is None:
        log.warning("HRV extraction failed for subject %s", subject_id)
        return None

    features: dict = {"subject_id": subject_id}
    features.update(hrv_features)

    if bp_signal is not None:
        bp_features = process_bp_segment(bp_signal, sampling_rate=fs)
        if bp_features:
            features.update(bp_features)

    return features


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def _save_checkpoint(results: list, path: Path) -> None:
    """Atomically save checkpoint via tmp file + rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f)
    tmp.rename(path)


def batch_process(
    data_dir: Path = DATA_DIR,
    subject_csv: Path = SUBJECT_CSV,
    checkpoint_path: Path = CHECKPOINT_PATH,
    max_participants: int | None = None,
) -> pd.DataFrame:
    """Extract features from all WFDB records in data_dir.

    Discovers records by globbing for *.hea files, resumes from checkpoint
    if one exists, and saves progress every 100 participants.

    Returns:
        DataFrame with one row per successfully processed participant.
        Subject metadata is NOT merged here — merging happens in main().
    """
    if not data_dir.exists():
        log.error("Data directory not found: %s  — download the dataset first.", data_dir)
        return pd.DataFrame()

    hea_files = sorted(data_dir.glob("**/*.hea"))
    record_paths = [str(f.with_suffix("")) for f in hea_files]
    if max_participants:
        record_paths = record_paths[:max_participants]
    log.info("Found %d WFDB records in %s", len(record_paths), data_dir)

    # Resume from checkpoint
    done_ids: set[str] = set()
    results: list[dict] = []
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
        done_ids = {r["subject_id"] for r in results}
        log.info("Resumed from checkpoint: %d already done", len(done_ids))

    remaining = [p for p in record_paths if Path(p).stem not in done_ids]
    log.info("Processing %d remaining records", len(remaining))

    for i, record_path in enumerate(tqdm(remaining, desc="Extracting HRV"), 1):
        subject_id = Path(record_path).stem
        result = process_single_participant(record_path, subject_id=subject_id)
        if result is not None:
            results.append(result)
        if i % 100 == 0:
            _save_checkpoint(results, checkpoint_path)

    if remaining:
        _save_checkpoint(results, checkpoint_path)

    features_df = pd.DataFrame(results)
    log.info(
        "Extraction complete: %d / %d succeeded",
        len(features_df),
        len(record_paths),
    )
    return features_df


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def run_eda(df: pd.DataFrame) -> None:
    """Generate 4 diagnostic plots and save to reports/week3_eda/."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    def _scatter(x_col, y_col, ylabel, filename, color="steelblue"):
        valid = df.dropna(subset=[x_col, y_col])
        if len(valid) < 2:
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(valid[x_col], valid[y_col], alpha=0.4, s=15, color=color)
        corr = valid[x_col].corr(valid[y_col])
        ax.set_xlabel("Age (years)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Age  (r={corr:.3f}, n={len(valid)})")
        fig.tight_layout()
        fig.savefig(EDA_DIR / filename, dpi=150)
        plt.close(fig)
        log.info("Saved %s (r=%.3f)", filename, corr)

    _scatter("age", "hrv_rmssd",   "RMSSD (ms)",      "rmssd_vs_age.png")
    _scatter("age", "hrv_sdnn",    "SDNN (ms)",        "sdnn_vs_age.png",    color="darkorange")
    _scatter("age", "hrv_mean_hr", "Resting HR (bpm)", "resting_hr_vs_age.png", color="crimson")

    # Feature distribution grid
    cols = [
        ("hrv_rmssd",    "RMSSD (ms)"),
        ("hrv_sdnn",     "SDNN (ms)"),
        ("hrv_pnn50",    "pNN50 (%)"),
        ("hrv_lf_hf",    "LF/HF ratio"),
        ("hrv_mean_hr",  "Mean HR (bpm)"),
        ("hrv_dfa_alpha1", "DFA alpha1"),
    ]
    available = [(c, l) for c, l in cols if c in df.columns]
    if available:
        ncols = 3
        nrows = (len(available) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()
        for i, (col, label) in enumerate(available):
            vals = df[col].dropna()
            if len(vals) > 0:
                axes[i].hist(vals, bins=40, edgecolor="black", alpha=0.7)
            axes[i].set_title(label)
            axes[i].set_ylabel("Count")
        for j in range(len(available), len(axes)):
            axes[j].set_visible(False)
        fig.tight_layout()
        fig.savefig(EDA_DIR / "hrv_feature_distributions.png", dpi=150)
        plt.close(fig)

    log.info("EDA plots saved to %s", EDA_DIR)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Week 3: Autonomic Aging HRV Pipeline")
    parser.add_argument("--max-participants", type=int, default=None,
                        help="Limit participants (for quick testing)")
    parser.add_argument("--skip-eda", action="store_true")
    args = parser.parse_args()

    features_df = batch_process(max_participants=args.max_participants)

    if features_df.empty:
        log.error(
            "No features extracted. Download the dataset to %s first.\n"
            "  mkdir -p data/autonomic_aging && cd data/autonomic_aging\n"
            "  wget -r -N -c -np "
            "https://physionet.org/files/autonomic-aging-cardiovascular/1.0.0/",
            DATA_DIR,
        )
        return

    # Merge with subject metadata
    metadata = load_subject_metadata(str(SUBJECT_CSV))
    merged = metadata.merge(features_df, on="subject_id", how="inner")
    log.info("Final dataset: %d participants × %d columns", *merged.shape)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)
    merged.to_csv(OUTPUT_PATH.with_suffix(".csv"), index=False)
    log.info("Saved → %s (.parquet + .csv)", OUTPUT_PATH)

    print(f"\n{'='*60}")
    print(f"Feature matrix: {merged.shape[0]} participants × {merged.shape[1]} columns")
    print(f"\nNull counts (top 10):")
    for col, n in merged.isnull().sum().sort_values(ascending=False).head(10).items():
        print(f"  {col}: {n} ({100*n/len(merged):.1f}%)")
    print(f"{'='*60}\n")

    if not args.skip_eda:
        run_eda(merged)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run all tests**

```bash
python -m pytest tests/test_hrv_pipeline.py -v
```

Expected: All 16 tests pass.

- [ ] **Step 4: Smoke test with synthetic data (no real data needed)**

```bash
cd /Users/ruchitsingh/Desktop/loopAI/epigenetics_project
source wearable-age/bin/activate
python -c "
import sys
sys.path.insert(0, 'src')
import numpy as np
import neurokit2 as nk
import wfdb
from pathlib import Path

# Write a synthetic record to /tmp
Path('/tmp/synth_test').mkdir(exist_ok=True)
ecg = nk.ecg_simulate(duration=300, sampling_rate=1000, heart_rate=65, random_state=42)
t = np.linspace(0, 300, 300_000)
bp = 120 + 20 * np.sin(2 * np.pi * 1.1 * t)
wfdb.wrsamp('sub001', fs=1000, units=['mV','mmHg'],
            sig_name=['ECG','ABP'],
            p_signal=np.column_stack([ecg, bp]),
            write_dir='/tmp/synth_test')

from autonomic_aging_processor import process_single_participant
result = process_single_participant('/tmp/synth_test/sub001', 'sub001')
print('Keys:', sorted(result.keys()) if result else 'FAILED')
print('RMSSD:', result.get('hrv_rmssd'))
print('Mean HR:', result.get('hrv_mean_hr'))
print('BP SBP mean:', result.get('bp_sbp_mean'))
"
```

Expected output:
```
Keys: ['bp_dbp_mean', 'bp_dbp_std', 'bp_pulse_pressure_mean', 'bp_sbp_mean', 'bp_sbp_std',
       'hrv_dfa_alpha1', 'hrv_dfa_alpha2', 'hrv_hf', 'hrv_lf', 'hrv_lf_hf',
       'hrv_mean_hr', 'hrv_mean_rr', 'hrv_pnn50', 'hrv_rmssd', 'hrv_sample_entropy',
       'hrv_sdnn', 'hrv_total_power', 'subject_id']
RMSSD: <some float>
Mean HR: <near 65.0>
BP SBP mean: <near 140.0>
```

- [ ] **Step 5: Commit**

```bash
git add src/autonomic_aging_processor.py
git commit -m "feat: add autonomic_aging_processor with batch HRV extraction and EDA"
```

---

## Task 4: Progress doc + final test run

**Files:**
- Create: `progress/week3-progress.md`

- [ ] **Step 1: Create `progress/week3-progress.md`**

```markdown
# Week 3 Progress

## Goal

Process the Autonomic Aging dataset (PhysioNet, 1,121 participants) to extract
HRV and blood pressure variability features for the Stress and Sleep categories,
producing `data/autonomic_features.parquet`.

---

## What Was Built

### `src/hrv_pipeline.py` — Signal processing library

| Function | Input | Output |
|----------|-------|--------|
| `process_ecg_segment(signal, fs)` | 1-D ECG array | Dict with 12 HRV features, or None |
| `load_wfdb_record(record_path)` | Path to WFDB record | (ecg_array, bp_array_or_None, fs) |
| `process_bp_segment(signal, fs)` | 1-D BP array | Dict with 5 BP variability features, or None |

### `src/autonomic_aging_processor.py` — Batch orchestrator

| Function | Purpose |
|----------|---------|
| `load_subject_metadata(csv_path)` | Read subject-info.csv → DataFrame |
| `process_single_participant(record_path, id)` | One WFDB record → feature dict |
| `batch_process(...)` | All participants with checkpointing → DataFrame |
| `run_eda(df)` | 4 diagnostic plots → reports/week3_eda/ |

---

## Output Schema

`data/autonomic_features.parquet` — one row per participant, ~20 columns:

| Group | Columns |
|-------|---------|
| Subject info | subject_id, age, gender, bmi |
| Time-domain HRV | hrv_rmssd, hrv_sdnn, hrv_pnn50, hrv_mean_rr, hrv_mean_hr |
| Frequency-domain HRV | hrv_lf, hrv_hf, hrv_lf_hf, hrv_total_power |
| Nonlinear HRV | hrv_sample_entropy, hrv_dfa_alpha1, hrv_dfa_alpha2 |
| BP variability (optional) | bp_sbp_mean, bp_dbp_mean, bp_sbp_std, bp_dbp_std, bp_pulse_pressure_mean |

---

## How to Run

### Prerequisites

```bash
# 1. Install wfdb
source wearable-age/bin/activate && pip install wfdb

# 2. Download dataset (~2–3 GB)
mkdir -p data/autonomic_aging && cd data/autonomic_aging
wget -r -N -c -np https://physionet.org/files/autonomic-aging-cardiovascular/1.0.0/
```

### Running the pipeline

```bash
# Quick test — first 10 participants
python src/autonomic_aging_processor.py --max-participants 10

# Full run (~1,121 participants, ~30–60 min depending on CPU)
python src/autonomic_aging_processor.py
```

Checkpointing: progress saved every 100 participants to `data/autonomic_checkpoint.json`.
Re-running picks up where it left off.

---

## EDA Plots (generated in `reports/week3_eda/`)

1. `rmssd_vs_age.png` — Expected: negative correlation (r ≈ −0.5 per Schumann et al.)
2. `sdnn_vs_age.png` — Expected: SDNN declines with age
3. `resting_hr_vs_age.png` — Expected: weak positive or flat correlation
4. `hrv_feature_distributions.png` — Distributions of 6 key HRV features

Compare your r values against Schumann et al. 2023, Table 2 for validation.

---

## What's Next (Week 4)

Build the Bio-RAG knowledge base with ChromaDB and all-MiniLM-L6-v2 embeddings.
```

- [ ] **Step 2: Run full test suite one final time**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: 16 tests pass, 0 fail.

- [ ] **Step 3: Final commit**

```bash
git add progress/week3-progress.md
git commit -m "docs: add Week 3 progress documentation"
```

---

## Self-Review

**Spec coverage (PRD tasks 3.1–3.12):**

| PRD Task | Plan Coverage |
|----------|--------------|
| 3.1 Download Autonomic Aging | Prerequisites section |
| 3.2 Understand WFDB format | `load_wfdb_record` + channel name detection |
| 3.3 Load ECG with `nk.ecg_process` | `process_ecg_segment` → `nk.ecg_clean` + `nk.ecg_peaks` |
| 3.4 Time-domain HRV: RMSSD, SDNN, pNN50, mean HR, mean RR | `_assemble_hrv_dict`: hrv_rmssd, hrv_sdnn, hrv_pnn50, hrv_mean_hr, hrv_mean_rr |
| 3.5 Frequency-domain: LF, HF, LF/HF, total power | `_assemble_hrv_dict`: hrv_lf, hrv_hf, hrv_lf_hf, hrv_total_power |
| 3.6 Nonlinear: sample entropy, DFA alpha1, alpha2 | `_assemble_hrv_dict`: hrv_sample_entropy, hrv_dfa_alpha1, hrv_dfa_alpha2 |
| 3.7 Mean resting HR | hrv_mean_hr (derived from RR intervals) |
| 3.8 BP variability | `process_bp_segment`: 5 BP features |
| 3.9 Merge with metadata | `main()` in `autonomic_aging_processor.py` |
| 3.10 `hrv_pipeline.py` as reusable module | Task 2: standalone library with clear function API |
| 3.11 EDA: RMSSD/SDNN/resting HR vs age | `run_eda()`: 3 scatterplots + distribution grid |
| 3.12 Compare against benchmark | `week3-progress.md`: instructs comparison against Schumann et al. Table 2 |

All tasks covered. No placeholder text. Type signatures consistent across all tasks.
