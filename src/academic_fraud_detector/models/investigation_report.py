"""
Investigation Report — the structured output schema for the final fraud investigation report.

This is the CONTRACT that the Evidence Synthesizer agent works toward.
All fields have semantic descriptions so the LLM knows exactly what to fill in.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════

class RiskLevel(str, Enum):
    """Overall risk level for a finding or the entire investigation."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConfidenceLevel(str, Enum):
    """Confidence in a single piece of evidence."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FraudCategory(str, Enum):
    """The five dimensions of academic fraud this system detects."""
    PLAGIARISM = "plagiarism"
    IMAGE_MANIPULATION = "image_manipulation"
    DATA_FABRICATION = "data_fabrication"
    CITATION_MANIPULATION = "citation_manipulation"
    PEER_REVIEW_FRAUD = "peer_review_fraud"


class RecommendationUrgency(str, Enum):
    IMMEDIATE = "immediate"
    ROUTINE = "routine"
    ADVISORY = "advisory"


# ═══════════════════════════════════════════════════════════════════════
# Evidence Items
# ═══════════════════════════════════════════════════════════════════════

class EvidenceItem(BaseModel):
    """A single piece of evidence in the investigation."""

    id: str = Field(
        ...,
        description="Unique evidence identifier, e.g. 'EVID-001'.",
    )
    finding: str = Field(
        ...,
        description=(
            "Clear, concise description of what was found. "
            "E.g., 'Figure 3B is a horizontally-flipped duplicate of Figure 2A from Smith et al. (2022).'"
        ),
    )
    location: str = Field(
        ...,
        description=(
            "Where in the target paper this finding applies: section, figure number, "
            "table number, or paragraph reference."
        ),
    )
    category: FraudCategory = Field(
        ...,
        description="Which dimension of fraud this evidence pertains to.",
    )
    confidence: ConfidenceLevel = Field(
        ...,
        description=(
            "How confident the agent is in this finding. "
            "HIGH = definitive, MEDIUM = probable, LOW = suggestive but needs human review."
        ),
    )
    supporting_data: dict = Field(
        default_factory=dict,
        description=(
            "Raw data supporting this finding. Contents vary by category: "
            "for plagiarism: similarity scores and aligned text; "
            "for images: ELA statistics and clone coordinates; "
            "for data: statistical test results (chi2, p-value); "
            "for citation: graph metrics and self-citation rates."
        ),
    )
    source_reference: Optional[str] = Field(
        default=None,
        description=(
            "External source citation if this finding references another paper. "
            "E.g., 'Smith et al. (2022) doi:10.1000/xyz123'."
        ),
    )
    suggested_action: Optional[str] = Field(
        default=None,
        description="What action this specific piece of evidence suggests.",
    )


# ═══════════════════════════════════════════════════════════════════════
# Category-Level Findings
# ═══════════════════════════════════════════════════════════════════════

class CategoryFindings(BaseModel):
    """Aggregated findings for one investigation category."""

    category: FraudCategory = Field(..., description="The investigation category.")
    severity: RiskLevel = Field(
        ...,
        description="Overall severity assessment for this category.",
    )
    confidence: ConfidenceLevel = Field(
        ...,
        description="Overall confidence in this category's assessment.",
    )
    summary: str = Field(
        ...,
        description=(
            "A 2-5 sentence narrative summary of the findings in this category. "
            "Should tell a coherent story, not just list evidence items."
        ),
    )
    findings: List[EvidenceItem] = Field(
        default_factory=list,
        description="All individual evidence items found in this category.",
    )
    total_checks_performed: int = Field(
        default=0,
        description="How many distinct checks/tests were performed in this category.",
    )
    flagged_checks: int = Field(
        default=0,
        description="How many checks returned flagged/suspicious results.",
    )
    category_risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Numeric risk score for this category (0-100).",
    )


# ═══════════════════════════════════════════════════════════════════════
# Recommendation
# ═══════════════════════════════════════════════════════════════════════

class Recommendation(BaseModel):
    """An actionable recommendation based on investigation findings."""

    priority: int = Field(
        ...,
        ge=1,
        le=5,
        description="Priority order (1 = highest priority).",
    )
    action: str = Field(
        ...,
        description=(
            "The recommended action in clear, imperative language. "
            "E.g., 'Initiate a formal retraction review for the affected figures.'"
        ),
    )
    urgency: RecommendationUrgency = Field(
        ...,
        description="How soon this action should be taken.",
    )
    rationale: str = Field(
        ...,
        description="Why this action is recommended, referencing specific findings.",
    )
    linked_evidence_ids: List[str] = Field(
        default_factory=list,
        description="Evidence item IDs that support this recommendation.",
    )


# ═══════════════════════════════════════════════════════════════════════
# The Full Report
# ═══════════════════════════════════════════════════════════════════════

class InvestigationReport(BaseModel):
    """
    The complete investigation report — the final deliverable of the system.

    This is what the Evidence Synthesizer produces after aggregating all
    specialist agent findings. It serves as both an executive summary for
    decision-makers and a detailed technical appendix for verification.
    """

    # ── Metadata ──
    report_metadata: dict = Field(
        default_factory=lambda: {
            "investigation_id": "",
            "investigation_date": datetime.now().isoformat(),
            "investigator": "Academic Fraud Detection System (CrewAI)",
            "schema_version": "1.0.0",
        },
        description="Report generation metadata.",
    )

    # ── Target Paper ──
    paper_under_investigation: dict = Field(
        default_factory=dict,
        description=(
            "Complete metadata of the paper being investigated: "
            "title, authors (list), doi, journal/conference, year, "
            "abstract, and any identifiers used for lookup."
        ),
    )

    # ── Executive Summary ──
    executive_summary: List[str] = Field(
        default_factory=list,
        description=(
            "3-5 bullet points summarizing the most important findings. "
            "Written for a non-specialist audience (journal editors, university administrators). "
            "Each bullet should be one clear, impactful sentence."
        ),
    )

    # ── Overall Assessment ──
    overall_risk: dict = Field(
        default_factory=lambda: {
            "level": RiskLevel.NONE.value,
            "score": 0.0,
            "confidence": ConfidenceLevel.LOW.value,
        },
        description=(
            "Overall investigation result. 'level' is the qualitative risk tier, "
            "'score' is the numeric 0-100 weighted score, "
            "'confidence' reflects how much data supported this assessment."
        ),
    )

    # Weight breakdown (for transparency)
    risk_scoring_methodology: dict = Field(
        default_factory=lambda: {
            "weights": {
                "plagiarism": 0.25,
                "image_manipulation": 0.25,
                "data_fabrication": 0.30,
                "citation_manipulation": 0.10,
                "peer_review_fraud": 0.10,
            },
            "thresholds": {
                "none": "0-15",
                "low": "16-35",
                "medium": "36-55",
                "high": "56-75",
                "critical": "76-100",
            },
        },
        description="The scoring rubric used to compute the overall risk score.",
    )

    # ── Per-Category Findings ──
    findings_by_category: dict[str, CategoryFindings] = Field(
        default_factory=dict,
        description=(
            "Findings grouped by fraud category. Keys are: "
            "'plagiarism', 'image_manipulation', 'data_fabrication', "
            "'citation_manipulation', 'peer_review_fraud'. "
            "Each value is a CategoryFindings object with severity, confidence, "
            "and the list of individual evidence items."
        ),
    )

    # ── Cross-Category Analysis ──
    cross_correlations: List[str] = Field(
        default_factory=list,
        description=(
            "Cross-category correlations that suggest organized misconduct. "
            "E.g., 'The same two authors appear as central nodes in both the citation ring "
            "and the falsified peer review recommendations, suggesting coordinated fraud.'"
        ),
    )

    # ── Recommendations ──
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description=(
            "Actionable recommendations sorted by priority. "
            "Should range from 'initiate retraction' (critical) to "
            "'seek author clarification' (low)."
        ),
    )

    # ── Limitations ──
    limitations: List[str] = Field(
        default_factory=list,
        description=(
            "Honest assessment of what this investigation could NOT determine due to "
            "data access constraints, tool limitations, or scope boundaries. "
            "Critical for legal defensibility and transparency."
        ),
    )

    # ── Evidence Chain ──
    evidence_chain: List[EvidenceItem] = Field(
        default_factory=list,
        description=(
            "The complete, chronological chain of all evidence items collected during "
            "the investigation. Sorted by discovery order. Includes items from all categories."
        ),
    )

    def summary(self) -> str:
        """Return a one-line summary of the report."""
        level = self.overall_risk.get("level", "unknown")
        score = self.overall_risk.get("score", "N/A")
        title = self.paper_under_investigation.get("title", "Unknown paper")
        return f"[{level.upper()}] score={score}/100 — {title}"
