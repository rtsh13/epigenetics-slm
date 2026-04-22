"""
NHANES 2011-2012 Data Loading and Merging Script
PRD Week 1 - Tasks 1.11, 1.12

Loads all NHANES .XPT files, verifies SEQN linkage,
and merges into a single participant-level DataFrame.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "nhanes"


def load_demographics():
    """Load DEMO_G: demographics (age, sex, race/ethnicity)."""
    df = pd.read_sas(DATA_DIR / "DEMO_G.XPT")
    cols = {
        "SEQN": "seqn",
        "RIDAGEYR": "age",          # Age in years at screening
        "RIAGENDR": "sex",          # 1=Male, 2=Female
        "RIDRETH1": "race_ethnicity",  # Race/ethnicity category
    }
    df = df[list(cols.keys())].rename(columns=cols)
    df["seqn"] = df["seqn"].astype(int)
    df["sex"] = df["sex"].map({1: "Male", 2: "Female"})
    return df


def load_body_measures():
    """Load BMX_G: body measures (BMI, waist circumference)."""
    df = pd.read_sas(DATA_DIR / "BMX_G.XPT")
    cols = {
        "SEQN": "seqn",
        "BMXBMI": "bmi",            # Body Mass Index (kg/m^2)
        "BMXWAIST": "waist_circumference",  # Waist circumference (cm)
        "BMXHT": "height",          # Standing height (cm)
        "BMXWT": "weight",          # Weight (kg)
    }
    df = df[list(cols.keys())].rename(columns=cols)
    df["seqn"] = df["seqn"].astype(int)
    return df


def load_glycohemoglobin():
    """Load GHB_G: glycohemoglobin (HbA1c %) - Metabolism category."""
    df = pd.read_sas(DATA_DIR / "GHB_G.XPT")
    cols = {
        "SEQN": "seqn",
        "LBXGH": "hba1c",           # Glycohemoglobin (%)
    }
    df = df[list(cols.keys())].rename(columns=cols)
    df["seqn"] = df["seqn"].astype(int)
    return df


def load_cbc():
    """Load CBC_G: Complete Blood Count - Inflammation category.

    Computes NLR (Neutrophil-to-Lymphocyte Ratio) as inflammation biomarker.
    """
    df = pd.read_sas(DATA_DIR / "CBC_G.XPT")
    cols = {
        "SEQN": "seqn",
        "LBXWBCSI": "wbc",          # White blood cell count (1000 cells/uL)
        "LBDNENO": "neutrophils",   # Segmented neutrophils number (1000 cells/uL)
        "LBDLYMNO": "lymphocytes",  # Lymphocyte number (1000 cells/uL)
    }
    df = df[list(cols.keys())].rename(columns=cols)
    df["seqn"] = df["seqn"].astype(int)
    # Compute Neutrophil-to-Lymphocyte Ratio
    df["nlr"] = df["neutrophils"] / df["lymphocytes"]
    # Replace inf/nan from division
    df["nlr"] = df["nlr"].replace([np.inf, -np.inf], np.nan)
    return df


def load_pam_header():
    """Load PAXHD_G: Physical Activity Monitor header (participant metadata)."""
    df = pd.read_sas(DATA_DIR / "PAXHD_G.XPT")
    cols = {
        "SEQN": "seqn",
        "PAXSTS": "pam_status",     # Device status
        "PAXCDAY": "pam_start_day", # Calendar day device started
    }
    available_cols = {k: v for k, v in cols.items() if k in df.columns}
    df = df[list(available_cols.keys())].rename(columns=available_cols)
    df["seqn"] = df["seqn"].astype(int)
    return df


def load_subject_info():
    """Load subject-info.csv from PhysioNet preprocessed NHANES data."""
    df = pd.read_csv(DATA_DIR / "subject-info.csv")
    # Standardize column name
    if "SEQN" in df.columns:
        df = df.rename(columns={"SEQN": "seqn"})
    return df


def build_participant_dataframe(adults_only=True, min_age=20):
    """Merge all NHANES data into a single participant-level DataFrame.

    Args:
        adults_only: If True, filter to adults (age >= min_age)
        min_age: Minimum age for adult filter (default 20, per CosinorAge)

    Returns:
        pd.DataFrame with columns from all sources, joined on seqn
    """
    print("Loading NHANES 2011-2012 data files...")

    # Load all files
    demo = load_demographics()
    print(f"  Demographics: {len(demo)} participants")

    bmx = load_body_measures()
    print(f"  Body measures: {len(bmx)} participants")

    ghb = load_glycohemoglobin()
    print(f"  Glycohemoglobin (HbA1c): {len(ghb)} participants")

    cbc = load_cbc()
    print(f"  CBC (WBC, NLR): {len(cbc)} participants")

    pam = load_pam_header()
    print(f"  PAM header: {len(pam)} participants")

    # Merge all on seqn
    print("\nMerging on SEQN...")
    df = demo.merge(bmx, on="seqn", how="left")
    df = df.merge(ghb, on="seqn", how="left")
    df = df.merge(cbc, on="seqn", how="left")
    df = df.merge(pam, on="seqn", how="left")

    print(f"  Merged DataFrame: {len(df)} participants, {len(df.columns)} columns")

    if adults_only:
        df = df[df["age"] >= min_age].copy()
        print(f"  After filtering to adults (age >= {min_age}): {len(df)} participants")

    # Add HbA1c classification
    df["hba1c_class"] = pd.cut(
        df["hba1c"],
        bins=[0, 5.7, 6.5, 100],
        labels=["Normal", "Pre-diabetic", "Diabetic"],
        right=False,
    )

    # Add NLR classification
    df["nlr_class"] = pd.cut(
        df["nlr"],
        bins=[0, 3.0, 6.0, 100],
        labels=["Normal", "Elevated", "High"],
        right=False,
    )

    # Summary stats
    print("\n--- Summary ---")
    print(f"Age range: {df['age'].min():.0f} - {df['age'].max():.0f}")
    print(f"Sex: {df['sex'].value_counts().to_dict()}")
    print(f"HbA1c available: {df['hba1c'].notna().sum()}")
    print(f"NLR available: {df['nlr'].notna().sum()}")
    print(f"BMI available: {df['bmi'].notna().sum()}")

    return df


if __name__ == "__main__":
    df = build_participant_dataframe()

    print("\n--- Column Types ---")
    print(df.dtypes)

    print("\n--- First 5 Rows ---")
    print(df.head())

    # Save merged DataFrame
    output_path = DATA_DIR.parent / "nhanes_participants.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
