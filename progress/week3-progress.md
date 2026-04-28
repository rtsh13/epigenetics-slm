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
# 1. Install wfdb (already in requirements.txt)
source wearable-age/bin/activate && pip install wfdb

# 2. Download dataset (~2–3 GB, requires PhysioNet account)
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

Checkpointing: progress saved to `data/autonomic_checkpoint.json` every 100 participants.
Re-running resumes where it left off.

---

## EDA Plots (generated in `reports/week3_eda/`)

1. `rmssd_vs_age.png` — Expected: negative correlation (r ≈ −0.5 per Schumann et al.)
2. `sdnn_vs_age.png` — Expected: SDNN declines with age
3. `resting_hr_vs_age.png` — Expected: weak positive or flat correlation
4. `hrv_feature_distributions.png` — Distributions of 6 key HRV features

Compare your r values against Schumann et al. 2023 Table 2 for validation.

---

## What's Next (Week 4)

Build the Bio-RAG knowledge base with ChromaDB and all-MiniLM-L6-v2 embeddings.
