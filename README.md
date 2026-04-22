# SLM-Powered Epigenetic Wellbeing Prototype

A fine-tuned Small Language Model (Llama 3.2 1B) augmented with RAG to generate grounded, interpretable health insights from wearable and clinical biomarker data across five epigenetic aging dimensions.

## Five Biomarker Categories

| Category | Biomarker | Data Source |
|----------|-----------|-------------|
| Aging | CosinorAge (biological age from circadian rhythms) | NHANES 2011-2012 accelerometry |
| Stress | HRV (RMSSD, SDNN, LF/HF) | Autonomic Aging dataset (PhysioNet) |
| Metabolism | HbA1c (glycohemoglobin) | NHANES lab file GHB_G |
| Inflammation | NLR (neutrophil-to-lymphocyte ratio) | NHANES lab file CBC_G |
| Sleep | Sleep architecture (TST, WASO, efficiency) + RHR | NHANES accelerometry + Autonomic Aging ECG |

## Architecture

```
Layer 1: Data Ingestion    → NHANES + Autonomic Aging datasets
Layer 2: Feature Engineering → CosinorAge, NeuroKit2, pandas
Layer 3: Baseline ML        → XGBoost (dual-stream age prediction)
Layer 4: Bio-RAG            → ChromaDB + all-MiniLM-L6-v2
Layer 5: SLM                → Llama 3.2 1B (QLoRA fine-tuned)
Layer 6: Demo               → Gradio dashboard
```

## Setup

### Prerequisites
- Python 3.12+
- ~10 GB disk space for NHANES data

### Environment Setup

```bash
# Create virtual environment
python3.12 -m venv wearable-age
source wearable-age/bin/activate

# Install dependencies
pip install --upgrade pip
pip install cosinorage neurokit2 xgboost scikit-learn shap pandas numpy matplotlib seaborn
pip install chromadb sentence-transformers gradio llama-cpp-python
pip install pyreadstat pyarrow

# Install CosinorAge sub-dependencies
pip install claid scikit-digital-health CosinorPy
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

### Data Download

#### NHANES 2011-2012 (4 of 5 categories)
Download the following `.XPT` files from [CDC NHANES](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2011) into `data/nhanes/`:

| File | Contents |
|------|----------|
| DEMO_G.XPT | Demographics |
| BMX_G.XPT | Body measures |
| GHB_G.XPT | Glycohemoglobin (HbA1c) |
| CBC_G.XPT | Complete blood count |
| PAXHD_G.XPT | PAM header |

For accelerometry, download from [PhysioNet NHANES](https://physionet.org/content/minute-level-step-count-nhanes/1.0.1/xpt/) (free account required):

| File | Contents | Size |
|------|----------|------|
| nhanes_1440_PAXMTSM.xpt.xz | Triaxial MIMS (minute-level) | 252 MB |
| nhanes_1440_PAXPREDM.xpt.xz | Wear/sleep predictions | 12 MB |
| subject-info.csv | Participant demographics | 1 MB |

Decompress after downloading:
```bash
cd data/nhanes
xz -dk nhanes_1440_PAXMTSM.xpt.xz
xz -dk nhanes_1440_PAXPREDM.xpt.xz
```

#### Autonomic Aging (Stress/HRV category)
Download from [PhysioNet](https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/) (free account required).

### Running the Pipeline

```bash
# 1. Load and merge NHANES data
python src/data_loader.py

# 2. Convert accelerometry to per-participant CSVs
python src/nhanes_accel_adapter.py
```

## Project Structure

```
epigenetics_project/
├── data/
│   ├── nhanes/              # Raw NHANES .XPT files
│   └── accel_csv/           # Converted per-participant accelerometry CSVs
├── src/
│   ├── data_loader.py       # NHANES data loading and merging
│   └── nhanes_accel_adapter.py  # PhysioNet → CosinorAge adapter
├── prd.md                   # Product requirements document
├── requirements.txt         # Pinned dependencies
├── .gitignore
└── README.md
```

## Datasets

- **NHANES 2011-2012**: [CDC](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2011) - Public, no account needed for core files
- **Autonomic Aging**: [PhysioNet](https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/) - Free account required

## Key References

- CosinorAge: [Shim et al. 2024, npj Digital Medicine](https://www.nature.com/articles/s41746-024-01111-x)
- Autonomic Aging: [Schumann et al. 2023](https://www.nature.com/articles/s41597-022-01202-y)
- NHANES accelerometry + inflammation: [Nature Sci Reports 2023](https://www.nature.com/articles/s41598-023-36062-y)
