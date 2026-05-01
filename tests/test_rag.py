"""
Tests for rag_retriever.py and vector_db_builder.py

All tests use chromadb.EphemeralClient() — no disk I/O required.
"""
import sys
sys.path.insert(0, "src")

import chromadb
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_KNOWLEDGE = [
    {
        "id": "test_aging_001",
        "category": "Aging",
        "type": "research_summary",
        "title": "CosinorAge biological age clock",
        "content": (
            "CosinorAge is a biological age estimator derived from wrist accelerometry circadian "
            "rhythm parameters. Published in npj Digital Medicine (Shim et al., 2024), it uses "
            "MESOR, amplitude, acrophase, interdaily stability (IS), intradaily variability (IV), "
            "L5, M10, and relative amplitude (RA) computed from 7-day accelerometry. Higher IV and "
            "lower IS reflect circadian fragmentation and are associated with accelerated biological "
            "aging. CosinorAge was trained on UK Biobank (n=67,000) and validated on NHANES 2011-2014. "
            "A 1-year increase in CosinorAge acceleration (biological age minus chronological age) is "
            "associated with a 5.8% increase in all-cause mortality risk. Circadian disruption "
            "suppresses BMAL1 and CLOCK gene expression, altering methylation at CpG sites linked "
            "to inflammatory response pathways."
        ),
        "source": "Shim et al. 2024, npj Digital Medicine",
    },
    {
        "id": "test_stress_001",
        "category": "Stress",
        "type": "reference_range",
        "title": "RMSSD reference ranges by age decade",
        "content": (
            "RMSSD (root mean square of successive RR interval differences) is the primary time-domain "
            "HRV metric reflecting parasympathetic activity. Published normative values from the "
            "Autonomic Aging dataset (Schumann et al., 2023, n=1,121): age 20-29: 42-68 ms; "
            "age 30-39: 35-58 ms; age 40-49: 28-48 ms; age 50-59: 22-38 ms; age 60-69: 16-30 ms; "
            "age 70+: 12-24 ms. Values below the lower bound for age indicate autonomic aging "
            "acceleration and elevated cardiovascular risk. RMSSD below 20 ms in any adult is "
            "clinically significant regardless of age."
        ),
        "source": "Schumann et al. 2023, Frontiers in Aging Neuroscience",
    },
    {
        "id": "test_metabolism_001",
        "category": "Metabolism",
        "type": "reference_range",
        "title": "HbA1c ranges and metabolic health",
        "content": (
            "HbA1c (glycated hemoglobin) reflects average blood glucose over the prior 120 days. "
            "Clinical ranges: below 5.7% is normal; 5.7-6.4% is prediabetes; 6.5% and above is "
            "diabetes (ADA 2024 guidelines). From an epigenetic aging perspective, HbA1c above 5.5% "
            "is associated with accelerated GrimAge even in the prediabetic range. Each 1% increase "
            "in HbA1c above 5.5% is associated with approximately 1.4-year acceleration in epigenetic "
            "age (Lu et al., 2022, Nature Aging). Chronic hyperglycemia generates advanced glycation "
            "end-products (AGEs) that drive oxidative stress, DNA strand breaks, and altered "
            "methylation at TXNIP, ABCG1, and CPT1A gene promoters."
        ),
        "source": "ADA 2024; Lu et al. 2022, Nature Aging",
    },
    {
        "id": "test_inflammation_001",
        "category": "Inflammation",
        "type": "mechanism",
        "title": "NLR as inflammaging biomarker",
        "content": (
            "The neutrophil-to-lymphocyte ratio (NLR) is computed as absolute neutrophil count "
            "divided by absolute lymphocyte count from a complete blood count. NLR is a validated "
            "systemic inflammation marker that predicts all-cause mortality independently of CRP "
            "(Forget et al., 2017). Normal NLR range: 1.0-3.0. NLR 3.0-5.0 indicates low-grade "
            "chronic inflammation; above 5.0 indicates significant systemic inflammation. Elevated "
            "NLR reflects the inflammaging phenotype: age-related increase in neutrophil activity "
            "combined with lymphopenia. NLR correlates with CRP (r≈0.5-0.6) and IL-6, making it a "
            "practical blood-draw substitute for hs-CRP in population datasets."
        ),
        "source": "Forget et al. 2017, BioMed Research International; NHANES 2023",
    },
    {
        "id": "test_sleep_001",
        "category": "Sleep",
        "type": "mechanism",
        "title": "Sleep and epigenetic aging acceleration",
        "content": (
            "Short sleep duration and poor sleep quality are associated with accelerated epigenetic "
            "aging across multiple clocks. Carroll et al. (2024, Nature Communications) found that "
            "sleeping fewer than 6 hours per night is associated with 1.8-year GrimAge acceleration. "
            "Waking after sleep onset (WASO) above 60 minutes per night is associated with reduced "
            "slow-wave sleep, during which growth hormone secretion, cellular repair, and memory "
            "consolidation occur. Low sleep efficiency (below 85%) measured by actigraphy predicts "
            "elevated morning cortisol and suppressed heart rate variability the following day. "
            "Resting heart rate during sleep is a proxy for sympathetic tone: values above 65 bpm "
            "suggest incomplete autonomic recovery."
        ),
        "source": "Carroll et al. 2024, Nature Communications; Diao et al. 2025",
    },
]


@pytest.fixture
def ephemeral_client():
    return chromadb.EphemeralClient()


@pytest.fixture
def populated_rag(ephemeral_client):
    """BioRAG instance with 5 knowledge entries loaded."""
    import tempfile, json, os
    from vector_db_builder import BioRAGBuilder
    from rag_retriever import BioRAG

    builder = BioRAGBuilder(
        knowledge_entries=MINIMAL_KNOWLEDGE,
        chroma_client=ephemeral_client,
        collection_name="test_bio_rag",
    )
    builder.build()
    return BioRAG(client=ephemeral_client, collection_name="test_bio_rag")


# ---------------------------------------------------------------------------
# BioRAGBuilder tests
# ---------------------------------------------------------------------------

class TestBioRAGBuilder:
    def test_build_populates_collection(self, ephemeral_client):
        from vector_db_builder import BioRAGBuilder

        builder = BioRAGBuilder(
            knowledge_entries=MINIMAL_KNOWLEDGE,
            chroma_client=ephemeral_client,
            collection_name="test_col",
        )
        builder.build()
        col = ephemeral_client.get_collection("test_col")
        assert col.count() == 5

    def test_build_is_idempotent(self, ephemeral_client):
        """Calling build() twice on same collection should not duplicate documents."""
        from vector_db_builder import BioRAGBuilder

        builder = BioRAGBuilder(
            knowledge_entries=MINIMAL_KNOWLEDGE,
            chroma_client=ephemeral_client,
            collection_name="test_idem",
        )
        builder.build()
        builder.build()
        col = ephemeral_client.get_collection("test_idem")
        assert col.count() == 5

    def test_metadata_fields_stored(self, ephemeral_client):
        from vector_db_builder import BioRAGBuilder

        builder = BioRAGBuilder(
            knowledge_entries=MINIMAL_KNOWLEDGE[:1],
            chroma_client=ephemeral_client,
            collection_name="test_meta",
        )
        builder.build()
        col = ephemeral_client.get_collection("test_meta")
        result = col.get(ids=["test_aging_001_0"])
        meta = result["metadatas"][0]
        assert meta["category"] == "Aging"
        assert meta["type"] == "research_summary"
        assert meta["source"] == "Shim et al. 2024, npj Digital Medicine"

    def test_long_content_is_chunked(self, ephemeral_client):
        """An entry with content > 500 words should produce multiple chunks."""
        from vector_db_builder import BioRAGBuilder

        long_entry = {
            "id": "long_001",
            "category": "Aging",
            "type": "mechanism",
            "title": "Long entry",
            "content": " ".join(["word"] * 600),
            "source": "test",
        }
        builder = BioRAGBuilder(
            knowledge_entries=[long_entry],
            chroma_client=ephemeral_client,
            collection_name="test_chunk",
        )
        builder.build()
        col = ephemeral_client.get_collection("test_chunk")
        assert col.count() >= 2

    def test_raises_on_missing_required_fields(self, ephemeral_client):
        from vector_db_builder import BioRAGBuilder

        bad_entry = {"id": "bad_001", "content": "some text"}
        builder = BioRAGBuilder(
            knowledge_entries=[bad_entry],
            chroma_client=ephemeral_client,
            collection_name="test_bad",
        )
        with pytest.raises(ValueError, match="category"):
            builder.build()


# ---------------------------------------------------------------------------
# BioRAG.search tests
# ---------------------------------------------------------------------------

class TestBioRAGSearch:
    def test_returns_list_of_dicts(self, populated_rag):
        results = populated_rag.search("biological age HRV")
        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(results[0], dict)

    def test_each_result_has_required_keys(self, populated_rag):
        results = populated_rag.search("HbA1c glucose metabolism")
        for r in results:
            for key in ("context", "source", "type", "category", "title"):
                assert key in r, f"Missing key '{key}' in result: {r}"

    def test_n_results_respected(self, populated_rag):
        results = populated_rag.search("aging epigenetics", n_results=2)
        assert len(results) <= 2

    def test_category_filter_returns_only_that_category(self, populated_rag):
        results = populated_rag.search("health aging stress", n_results=5, category="Sleep")
        for r in results:
            assert r["category"] == "Sleep", f"Expected Sleep, got {r['category']}"

    def test_empty_query_raises(self, populated_rag):
        with pytest.raises(ValueError, match="query"):
            populated_rag.search("")

    def test_context_field_is_nonempty_string(self, populated_rag):
        results = populated_rag.search("RMSSD heart rate variability age")
        for r in results:
            assert isinstance(r["context"], str)
            assert len(r["context"]) > 20

    def test_unknown_category_raises(self, populated_rag):
        with pytest.raises(ValueError, match="category"):
            populated_rag.search("test", category="NotACategory")

    def test_search_returns_relevant_category(self, populated_rag):
        """Smoke test: searching for glucose should surface Metabolism result."""
        results = populated_rag.search("HbA1c glucose prediabetes epigenetic", n_results=3)
        categories = [r["category"] for r in results]
        assert "Metabolism" in categories


# ---------------------------------------------------------------------------
# BioRAG.from_knowledge_dir (file-based constructor) tests
# ---------------------------------------------------------------------------

class TestBioRAGFromDir:
    def test_from_knowledge_dir_loads_all_files(self, tmp_path, ephemeral_client):
        """Passing a dir with one JSON file should populate the DB."""
        import json
        from rag_retriever import BioRAG

        knowledge_file = tmp_path / "test_knowledge.json"
        knowledge_file.write_text(json.dumps(MINIMAL_KNOWLEDGE[:2]))

        rag = BioRAG.from_knowledge_dir(
            knowledge_dir=str(tmp_path),
            client=ephemeral_client,
            collection_name="test_dir",
        )
        results = rag.search("aging circadian")
        assert len(results) > 0

    def test_from_knowledge_dir_raises_on_missing_dir(self, ephemeral_client):
        from rag_retriever import BioRAG

        with pytest.raises(FileNotFoundError):
            BioRAG.from_knowledge_dir(
                knowledge_dir="/nonexistent/path",
                client=ephemeral_client,
            )
