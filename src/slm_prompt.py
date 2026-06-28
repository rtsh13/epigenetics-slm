"""
Single source of truth for the SLM prompt and training-response templates.

Both src/slm_dataset_builder.py (training-time) and src/slm_generator.py
(inference-time) import from this module. Byte-identical prompt format
between training and inference is essential for fine-tuned model quality;
the parity test in tests/test_slm_generator.py enforces this contract.

If you change the prompt format, bump PROMPT_VERSION and retrain.
"""

PROMPT_VERSION = "v1"


def _format_evidence(rag_chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(rag_chunks, 1):
        title = chunk.get("title", "Untitled")
        source = chunk.get("source", "Unknown")
        category = chunk.get("category", "General")
        context = chunk.get("context", "").strip()
        lines.append(f"[{i}] {category}: {title} ({source})")
        lines.append(f"    {context}")
    return "\n".join(lines)


def build_prompt(biomarkers: dict, rag_chunks: list[dict]) -> str:
    """Build the instruction prompt fed to the SLM at both training and inference."""
    age = biomarkers["age"]
    sex = biomarkers["sex"]
    hba1c = biomarkers["hba1c"]
    nlr = biomarkers["nlr"]
    wbc = biomarkers["wbc"]
    advance = biomarkers["cosinorage_advance"]
    is_v = biomarkers["is_value"]
    iv_v = biomarkers["iv_value"]
    ra_v = biomarkers["ra_value"]
    tst_min = biomarkers["tst_minutes"]
    tst_h = tst_min / 60.0
    sri = biomarkers["sri"]

    evidence = _format_evidence(rag_chunks)

    return (
        "You are a wellness assistant generating a five-category epigenetic "
        "health assessment grounded in the provided evidence.\n\n"
        "Patient profile:\n"
        f"- Age: {int(age)} years\n"
        f"- Sex: {sex}\n"
        f"- HbA1c: {hba1c:.1f}%\n"
        f"- NLR: {nlr:.2f}\n"
        f"- WBC: {wbc:.1f} x10^3/uL\n"
        f"- CosinorAge acceleration: {advance:+.1f} years\n"
        f"- Circadian (IS/IV/RA): {is_v:.2f} / {iv_v:.2f} / {ra_v:.2f}\n"
        f"- Total sleep time: {tst_h:.1f} hours/night ({tst_min:.0f} min)\n"
        f"- Sleep Regularity Index: {sri:.1f}\n\n"
        "Evidence:\n"
        f"{evidence}\n\n"
        "Write a five-category assessment with headers AGING, STRESS, "
        "METABOLISM, INFLAMMATION, SLEEP. Reference the evidence above by "
        "citing the source in parentheses. Keep each section to 2-3 sentences."
    )


def build_response(
    biomarkers: dict,
    classifications: dict,
    rag_chunks: list[dict],
) -> str:
    """Build the training-time target response.

    Used only at training-set generation. At inference, the SLM produces
    its own response and this function is not called.
    """
    age = int(biomarkers["age"])
    advance = biomarkers["cosinorage_advance"]
    hba1c = biomarkers["hba1c"]
    nlr = biomarkers["nlr"]
    wbc = biomarkers["wbc"]
    tst_h = biomarkers["tst_minutes"] / 60.0
    sri = biomarkers["sri"]

    aging_label, aging_interp = classifications["aging"]
    hba1c_label, hba1c_interp = classifications["hba1c"]
    nlr_label, nlr_interp = classifications["nlr"]
    sleep_label, sleep_interp = classifications["sleep"]
    circ_label, circ_interp = classifications["circadian"]

    first_source = rag_chunks[0].get("source", "Bio-RAG") if rag_chunks else "Bio-RAG"
    second_source = rag_chunks[1].get("source", first_source) if len(rag_chunks) > 1 else first_source

    return (
        f"AGING: CosinorAge acceleration is {advance:+.1f} years against a "
        f"chronological age of {age} ({aging_label}). {aging_interp} "
        f"Circadian pattern: {circ_interp} ({first_source}).\n\n"
        f"STRESS: HRV-based stress estimation is not available from the "
        f"NHANES accelerometer cohort; circadian fragmentation is used as a "
        f"proxy for autonomic load here.\n\n"
        f"METABOLISM: HbA1c of {hba1c:.1f}% places this patient in the "
        f"{hba1c_label} range. {hba1c_interp} ({second_source}).\n\n"
        f"INFLAMMATION: NLR of {nlr:.2f} with WBC {wbc:.1f} indicates "
        f"{nlr_label} inflammatory burden. {nlr_interp} ({first_source}).\n\n"
        f"SLEEP: {sleep_label} sleep profile, {tst_h:.1f} hours per night "
        f"with SRI {sri:.1f}. {sleep_interp}"
    )
