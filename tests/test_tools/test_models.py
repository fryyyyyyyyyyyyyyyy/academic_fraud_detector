"""
Tests for Pydantic data models — ensure the output schemas are correct.
"""

import pytest
from datetime import datetime


class TestInvestigationReport:
    """Test the InvestigationReport model."""

    @pytest.fixture
    def report_class(self):
        from academic_fraud_detector.models.investigation_report import InvestigationReport
        return InvestigationReport

    def test_empty_report_creation(self, report_class):
        """Should be able to create a minimal report."""
        report = report_class()
        assert report.overall_risk["level"] == "none"
        assert report.overall_risk["score"] == 0.0
        assert report.executive_summary == []

    def test_summary_method(self, report_class):
        """summary() should return a string."""
        report = report_class()
        report.paper_under_investigation = {"title": "Test Paper", "doi": "10.0000/test"}
        report.overall_risk = {"level": "high", "score": 72.5, "confidence": "medium"}
        s = report.summary()
        assert "HIGH" in s
        assert "72.5" in s
        assert "Test Paper" in s

    def test_risk_scoring_methodology_present(self, report_class):
        """Risk scoring methodology should be populated by default."""
        report = report_class()
        weights = report.risk_scoring_methodology.get("weights", {})
        assert weights["data_fabrication"] == 0.30
        assert weights["plagiarism"] == 0.25
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_report_with_findings(self, report_class):
        """Should support adding category findings."""
        from academic_fraud_detector.models.investigation_report import (
            CategoryFindings, FraudCategory, RiskLevel, ConfidenceLevel, EvidenceItem,
        )

        report = report_class()
        report.findings_by_category["plagiarism"] = CategoryFindings(
            category=FraudCategory.PLAGIARISM,
            severity=RiskLevel.HIGH,
            confidence=ConfidenceLevel.HIGH,
            summary="Found significant text overlap.",
            findings=[
                EvidenceItem(
                    id="EVID-001",
                    finding="Verbatim copy from Smith et al.",
                    location="Section 3.2",
                    category=FraudCategory.PLAGIARISM,
                    confidence=ConfidenceLevel.HIGH,
                ),
            ],
            category_risk_score=70.0,
        )

        # Serialization round-trip
        data = report.model_dump()
        assert "findings_by_category" in data
        assert "plagiarism" in data["findings_by_category"]


class TestPaperMetadata:
    """Test the PaperMetadata model."""

    @pytest.fixture
    def paper_class(self):
        from academic_fraud_detector.models.paper import PaperMetadata, Author
        return PaperMetadata, Author

    def test_short_citation_one_author(self, paper_class):
        PaperMetadata, Author = paper_class
        paper = PaperMetadata(
            title="Test",
            authors=[Author(full_name="John Smith", last_name="Smith")],
            year=2023,
        )
        assert paper.short_citation() == "Smith (2023)"

    def test_short_citation_two_authors(self, paper_class):
        PaperMetadata, Author = paper_class
        paper = PaperMetadata(
            title="Test",
            authors=[
                Author(full_name="John Smith", last_name="Smith"),
                Author(full_name="Jane Doe", last_name="Doe"),
            ],
            year=2023,
        )
        assert "&" in paper.short_citation()
        assert "Smith" in paper.short_citation()
        assert "Doe" in paper.short_citation()

    def test_short_citation_three_authors(self, paper_class):
        PaperMetadata, Author = paper_class
        paper = PaperMetadata(
            title="Test",
            authors=[
                Author(full_name="John Smith", last_name="Smith"),
                Author(full_name="Jane Doe", last_name="Doe"),
                Author(full_name="Bob Jones", last_name="Jones"),
            ],
            year=2023,
        )
        assert "et al." in paper.short_citation()


class TestEvidenceChain:
    """Test the EvidenceChain model."""

    @pytest.fixture
    def chain_class(self):
        from academic_fraud_detector.models.evidence import EvidenceChain, EvidenceNode
        return EvidenceChain, EvidenceNode

    def test_add_node(self, chain_class):
        EvidenceChain, EvidenceNode = chain_class
        chain = EvidenceChain(investigation_id="TEST-001")
        node = EvidenceNode(
            evidence_id="EVID-001",
            source_agent="plagiarism_detective",
            category="plagiarism",
            short_description="Found copied text.",
            confidence_score=0.95,
        )
        chain.add_node(node)
        assert len(chain.nodes) == 1
        assert chain.nodes[0].evidence_id == "EVID-001"

    def test_stats(self, chain_class):
        EvidenceChain, EvidenceNode = chain_class
        chain = EvidenceChain(investigation_id="TEST-001")

        for i in range(5):
            node = EvidenceNode(
                evidence_id=f"EVID-{i:03d}",
                source_agent="agent",
                category="plagiarism" if i < 3 else "image_manipulation",
                short_description=f"Finding {i}",
                confidence_score=0.8,
            )
            chain.add_node(node)

        stats = chain.stats()
        assert stats["total"] == 5
        assert stats["by_category"]["plagiarism"] == 3
        assert stats["by_category"]["image_manipulation"] == 2

    def test_filter_by_status(self, chain_class):
        EvidenceChain, EvidenceNode = chain_class
        from academic_fraud_detector.models.evidence import EvidenceStatus

        chain = EvidenceChain(investigation_id="TEST-001")
        node = EvidenceNode(
            evidence_id="EVID-001",
            source_agent="agent",
            category="plagiarism",
            short_description="Test",
            status=EvidenceStatus.VERIFIED,
        )
        chain.add_node(node)

        verified = chain.get_verified()
        assert len(verified) == 1
        assert verified[0].status == EvidenceStatus.VERIFIED
