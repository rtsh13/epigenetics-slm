"""
Pure biomarker classification functions.

Each function takes a numeric biomarker value (or values) and returns a
(label, interpretation) tuple. Label strings for HbA1c and NLR match the
parquet's hba1c_class / nlr_class columns so they can be used
interchangeably.

Used by scripts/demo.py (display) and src/slm_dataset_builder.py
(training-set generation).
"""


def classify_hba1c(value: float) -> tuple[str, str]:
    if value < 5.7:
        return ("Normal", "Stable glucose metabolism, low epigenetic oxidative stress risk.")
    if value < 6.5:
        return ("Pre-diabetic", "Elevated glucose may accelerate DNA methylation drift.")
    return ("Diabetic", "Chronic hyperglycemia drives oxidative stress and methylation damage.")


def classify_nlr(value: float) -> tuple[str, str]:
    if value < 3.0:
        return ("Normal", "No systemic inflammatory signal detected.")
    if value < 6.0:
        return ("Elevated", "Chronic low-grade inflammation indicates inflammaging risk.")
    return ("High", "Strong systemic inflammation, a primary driver of epigenetic erosion.")


def classify_aging(advance_years: float) -> tuple[str, str]:
    if advance_years < 3:
        return ("Favorable", "Biological age closely tracks chronological age. Favorable aging trajectory.")
    if advance_years < 8:
        return ("Modest", "Modest biological age acceleration. Circadian or lifestyle factors may contribute.")
    return ("Significant", "Significant biological age acceleration. Circadian disruption is a key driver.")


def classify_circadian(is_value: float, iv_value: float, ra_value: float) -> tuple[str, str]:
    flags = []
    if is_value >= 0.6:
        flags.append(f"strong day-to-day rhythm consistency (IS={is_value:.2f})")
    elif is_value < 0.35:
        flags.append(f"fragmented circadian rhythm (IS={is_value:.2f}, low)")
    if iv_value < 0.6:
        flags.append(f"stable within-day activity pattern (IV={iv_value:.2f}, low)")
    elif iv_value > 0.9:
        flags.append(f"high within-day fragmentation (IV={iv_value:.2f}, elevated)")
    if ra_value >= 0.75:
        flags.append(f"sharp day/night activity contrast (RA={ra_value:.2f}, high)")

    if not flags:
        return ("Borderline", "Circadian metrics within range, no strong signal in either direction.")

    if is_value < 0.35 or iv_value > 0.9:
        return ("Fragmented", "; ".join(flags) + ".")
    if is_value >= 0.6 and iv_value < 0.6 and ra_value >= 0.75:
        return ("Stable", "; ".join(flags) + ".")
    return ("Borderline", "; ".join(flags) + ".")


def classify_sleep(tst_minutes: float, sri: float) -> tuple[str, str]:
    tst_hours = tst_minutes / 60.0
    parts = []

    if tst_hours < 5.5:
        duration_flag = "short"
        parts.append(f"short sleep ({tst_hours:.1f}h/night, below recommended 7h)")
    elif tst_hours > 9.0:
        duration_flag = "long"
        parts.append(f"extended sleep ({tst_hours:.1f}h/night, may reflect fragmented quality)")
    else:
        duration_flag = "adequate"
        parts.append(f"adequate sleep duration ({tst_hours:.1f}h/night)")

    if sri < 40:
        reg_flag = "irregular"
        parts.append("very irregular sleep timing (low SRI, circadian misalignment risk)")
    elif sri < 60:
        reg_flag = "moderate"
        parts.append("moderate sleep regularity")
    else:
        reg_flag = "regular"
        parts.append("consistent sleep schedule")

    interpretation = "; ".join(parts).capitalize() + "."

    if duration_flag == "short" or reg_flag == "irregular":
        label = "Short" if duration_flag == "short" else "Irregular"
    elif duration_flag == "adequate" and reg_flag == "regular":
        label = "Adequate"
    else:
        label = "Adequate"

    return (label, interpretation)
