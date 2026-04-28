"""
hrv_pipeline.py — HRV feature extraction from ECG/BP signals.

Public functions:
    process_ecg_segment(signal, sampling_rate=1000) -> dict | None
    load_wfdb_record(record_path: str) -> tuple[np.ndarray, np.ndarray | None, int]
    process_bp_segment(signal, sampling_rate=1000) -> dict | None
"""

import warnings
from pathlib import Path

import numpy as np
import neurokit2 as nk
from scipy.signal import find_peaks
import wfdb


# ---------------------------------------------------------------------------
# process_ecg_segment
# ---------------------------------------------------------------------------

def process_ecg_segment(signal: np.ndarray, sampling_rate: int = 1000) -> dict | None:
    """
    Extract HRV features from a 1-D ECG array.

    Returns None if:
    - Duration < 60 seconds
    - Signal is flat (std < 1e-6)
    - Fewer than 30 R-peaks detected after physiological filtering

    Returns a dict with keys:
        hrv_rmssd, hrv_sdnn, hrv_pnn50, hrv_mean_rr, hrv_mean_hr,
        hrv_lf, hrv_hf, hrv_lf_hf, hrv_total_power,
        hrv_sample_entropy, hrv_dfa_alpha1, hrv_dfa_alpha2
    """
    signal = np.asarray(signal, dtype=float)
    duration = len(signal) / sampling_rate

    # Guard: too short
    if duration < 60:
        return None

    # Guard: flat signal
    if np.std(signal) < 1e-6:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Clean and detect R-peaks
        ecg_cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)
        peaks_df, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        rpeaks = info["ECG_R_Peaks"]

    # Guard: too few peaks
    if len(rpeaks) < 30:
        return None

    # Issue 5: Filter ectopic/artefact beats — keep only 300–2000 ms (30–200 bpm)
    rr_ms = np.diff(rpeaks) / sampling_rate * 1000.0
    valid_mask = (rr_ms >= 300) & (rr_ms <= 2000)
    if valid_mask.sum() < 29:  # need ≥30 peaks → ≥29 intervals
        return None
    # Keep rpeaks where both adjacent RR intervals are valid
    valid_rpeaks = rpeaks[
        np.concatenate([[True], valid_mask]) & np.concatenate([valid_mask, [True]])
    ]
    if len(valid_rpeaks) < 30:
        return None
    rpeaks = valid_rpeaks
    rr_ms = rr_ms[valid_mask]

    # Issue 2: Guard against division by zero for mean_rr / mean_hr
    mean_rr = float(np.mean(rr_ms)) if len(rr_ms) > 0 else None
    mean_hr = float(60_000.0 / mean_rr) if (mean_rr is not None and mean_rr > 0) else None

    # Issue 1: Pass rpeaks (indices array) rather than peaks_df to HRV functions.
    # nk.hrv_time/frequency/nonlinear all accept R-peak sample indices directly, which
    # is the unambiguous API per NeuroKit2 docs (verified equivalent on v0.2.13).

    # Issue 4: Time-domain HRV — wrap in try/except, consistent with other branches
    hrv_rmssd = None
    hrv_sdnn = None
    hrv_pnn50 = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hrv_time = nk.hrv_time(rpeaks, sampling_rate=sampling_rate)
        hrv_rmssd = _safe_scalar(hrv_time, "HRV_RMSSD")
        hrv_sdnn = _safe_scalar(hrv_time, "HRV_SDNN")
        hrv_pnn50 = _safe_scalar(hrv_time, "HRV_pNN50")
    except Exception:
        pass

    # Frequency-domain HRV (only if >= 120s)
    hrv_lf = None
    hrv_hf = None
    hrv_lf_hf = None
    hrv_total_power = None

    if duration >= 120:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hrv_freq = nk.hrv_frequency(
                    rpeaks, sampling_rate=sampling_rate, psd_method="welch"
                )
            hrv_lf = _safe_scalar(hrv_freq, "HRV_LF")
            hrv_hf = _safe_scalar(hrv_freq, "HRV_HF")
            hrv_lf_hf = _safe_scalar(hrv_freq, "HRV_LFHF")
            hrv_total_power = _safe_scalar(hrv_freq, "HRV_TP")
        except Exception:
            pass

    # Nonlinear HRV
    hrv_sample_entropy = None
    hrv_dfa_alpha1 = None
    hrv_dfa_alpha2 = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hrv_nonlinear = nk.hrv_nonlinear(rpeaks, sampling_rate=sampling_rate)
        hrv_sample_entropy = _safe_scalar(hrv_nonlinear, "HRV_SampEn")
        hrv_dfa_alpha1 = _safe_scalar(hrv_nonlinear, "HRV_DFA_alpha1")
        hrv_dfa_alpha2 = _safe_scalar(hrv_nonlinear, "HRV_DFA_alpha2")
    except Exception:
        pass

    return {
        "hrv_rmssd": hrv_rmssd,
        "hrv_sdnn": hrv_sdnn,
        "hrv_pnn50": hrv_pnn50,
        "hrv_mean_rr": mean_rr,
        "hrv_mean_hr": mean_hr,
        "hrv_lf": hrv_lf,
        "hrv_hf": hrv_hf,
        "hrv_lf_hf": hrv_lf_hf,
        "hrv_total_power": hrv_total_power,
        "hrv_sample_entropy": hrv_sample_entropy,
        "hrv_dfa_alpha1": hrv_dfa_alpha1,
        "hrv_dfa_alpha2": hrv_dfa_alpha2,
    }


def _safe_scalar(df, column: str) -> float | None:
    """Extract a scalar float from a single-row DataFrame column; return None on failure."""
    try:
        val = df[column].iloc[0]
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return None
        return float(val)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# load_wfdb_record
# ---------------------------------------------------------------------------

def load_wfdb_record(record_path: str) -> tuple:
    """
    Load a WFDB record and return (ecg_signal, bp_signal_or_None, fs).

    ECG channel priority: ECG, II, I, EKG — falls back to channel 0.
    BP  channel priority: ABP, BP, SBP, NIBP — returns None if not found.

    Raises FileNotFoundError if the .hea header is missing.
    """
    hea_path = Path(record_path).with_suffix(".hea")
    if not hea_path.exists():
        raise FileNotFoundError(f"WFDB header not found: {hea_path}")

    record = wfdb.rdrecord(str(record_path))
    signals = record.p_signal  # (n_samples, n_channels)
    fs = record.fs
    sig_names_upper = [s.upper() for s in record.sig_name]

    # Select ECG channel
    ecg_idx = _find_channel(sig_names_upper, ["ECG", "II", "I", "EKG"], default=0)
    ecg_signal = signals[:, ecg_idx].astype(float)

    # Select BP channel
    bp_idx = _find_channel(sig_names_upper, ["ABP", "BP", "SBP", "NIBP"], default=None)
    if bp_idx is not None:
        bp_signal = signals[:, bp_idx].astype(float)
    else:
        bp_signal = None

    return ecg_signal, bp_signal, fs


def _find_channel(sig_names_upper: list, priority: list, default):
    """Return index of first matching channel name, or default."""
    for name in priority:
        if name in sig_names_upper:
            return sig_names_upper.index(name)
    return default


# ---------------------------------------------------------------------------
# process_bp_segment
# ---------------------------------------------------------------------------

def process_bp_segment(signal: np.ndarray, sampling_rate: int = 1000) -> dict | None:
    """
    Extract BP variability features from an arterial blood pressure signal.

    Returns None if:
    - Duration < 60s
    - Signal is flat (std < 1e-6)
    - Fewer than 10 systolic peaks or diastolic readings
    """
    signal = np.asarray(signal, dtype=float)
    duration = len(signal) / sampling_rate

    if duration < 60:
        return None

    if np.std(signal) < 1e-6:
        return None

    min_dist = int(0.4 * sampling_rate)  # max ~150 bpm

    # Issue 3: Find systolic peaks first, then locate diastole as the minimum
    # between consecutive systolic peaks, avoiding dicrotic notch false positives
    # that arise from find_peaks(-signal) on arterial BP waveforms.
    peaks, _ = find_peaks(signal, distance=min_dist, prominence=5)
    if len(peaks) < 10:
        return None

    dbp_values = []
    for i in range(len(peaks) - 1):
        segment = signal[peaks[i]:peaks[i + 1]]
        if len(segment) > 0:
            dbp_values.append(float(np.min(segment)))

    if len(dbp_values) < 10:
        return None

    sbp = signal[peaks[:len(dbp_values)]]
    dbp = np.array(dbp_values)

    return {
        "bp_sbp_mean": float(np.mean(sbp)),
        "bp_dbp_mean": float(np.mean(dbp)),
        "bp_sbp_std": float(np.std(sbp)),
        "bp_dbp_std": float(np.std(dbp)),
        "bp_pulse_pressure_mean": float(np.mean(sbp - dbp)),
    }
