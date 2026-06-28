import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slm_prompt import PROMPT_VERSION, build_prompt, build_response


BIOMARKERS = {
    "age": 71,
    "sex": "Female",
    "hba1c": 5.8,
    "nlr": 4.2,
    "wbc": 7.1,
    "cosinorage_advance": 2.4,
    "is_value": 0.42,
    "iv_value": 0.71,
    "ra_value": 0.74,
    "tst_minutes": 372.0,
    "sri": 58.0,
}

RAG_CHUNKS = [
    {
        "context": "Chronic low-grade inflammation as measured by NLR predicts methylation drift...",
        "source": "Liu et al. 2020",
        "category": "Inflammation",
        "title": "NLR and inflammaging",
    },
    {
        "context": "Circadian fragmentation contributes to epigenetic aging via cortisol dysregulation...",
        "source": "Smith et al. 2021",
        "category": "Aging",
        "title": "Circadian rhythm and epigenetic aging",
    },
]

CLASSIFICATIONS = {
    "hba1c": ("Pre-diabetic", "Elevated glucose may accelerate DNA methylation drift."),
    "nlr": ("Elevated", "Chronic low-grade inflammation indicates inflammaging risk."),
    "aging": ("Favorable", "Biological age closely tracks chronological age. Favorable aging trajectory."),
    "circadian": ("Borderline", "Some circadian flags present."),
    "sleep": ("Short", "Short sleep (6.2h/night, below recommended 7h); moderate sleep regularity."),
}


def test_prompt_version_is_string():
    assert isinstance(PROMPT_VERSION, str)
    assert PROMPT_VERSION  # non-empty


def test_build_prompt_contains_all_biomarkers():
    prompt = build_prompt(BIOMARKERS, RAG_CHUNKS)
    assert "71" in prompt
    assert "Female" in prompt
    assert "5.8" in prompt
    assert "4.2" in prompt
    assert "+2.4" in prompt or "2.4" in prompt
    assert "0.42" in prompt
    assert "0.71" in prompt
    assert "6.2" in prompt or "372" in prompt  # TST in hours or minutes
    assert "58" in prompt


def test_build_prompt_contains_rag_chunks():
    prompt = build_prompt(BIOMARKERS, RAG_CHUNKS)
    assert "Chronic low-grade inflammation" in prompt
    assert "Circadian fragmentation" in prompt
    assert "Liu et al. 2020" in prompt
    assert "Smith et al. 2021" in prompt


def test_build_prompt_contains_instruction():
    prompt = build_prompt(BIOMARKERS, RAG_CHUNKS)
    lower = prompt.lower()
    assert "five" in lower or "5" in lower
    assert "category" in lower or "assessment" in lower or "report" in lower


def test_build_prompt_has_section_markers():
    prompt = build_prompt(BIOMARKERS, RAG_CHUNKS)
    assert "Patient" in prompt or "patient" in prompt or "presents" in prompt.lower()
    assert "Evidence" in prompt or "EVIDENCE" in prompt or "evidence" in prompt.lower()


def test_build_response_has_five_section_headers():
    response = build_response(BIOMARKERS, CLASSIFICATIONS, RAG_CHUNKS)
    for header in ["AGING", "STRESS", "METABOLISM", "INFLAMMATION", "SLEEP"]:
        assert header in response, f"missing section header: {header}"


def test_build_response_uses_classifications():
    response = build_response(BIOMARKERS, CLASSIFICATIONS, RAG_CHUNKS)
    assert "Pre-diabetic" in response
    assert "Elevated" in response


def test_build_response_references_rag_evidence():
    response = build_response(BIOMARKERS, CLASSIFICATIONS, RAG_CHUNKS)
    assert "Liu et al. 2020" in response or "Smith et al. 2021" in response


def test_build_response_includes_biomarker_values():
    response = build_response(BIOMARKERS, CLASSIFICATIONS, RAG_CHUNKS)
    assert "5.8" in response
    assert "4.2" in response


def test_prompt_is_deterministic():
    """Same inputs must produce byte-identical output (critical for parity)."""
    p1 = build_prompt(BIOMARKERS, RAG_CHUNKS)
    p2 = build_prompt(BIOMARKERS, RAG_CHUNKS)
    assert p1 == p2


def test_response_is_deterministic():
    r1 = build_response(BIOMARKERS, CLASSIFICATIONS, RAG_CHUNKS)
    r2 = build_response(BIOMARKERS, CLASSIFICATIONS, RAG_CHUNKS)
    assert r1 == r2
