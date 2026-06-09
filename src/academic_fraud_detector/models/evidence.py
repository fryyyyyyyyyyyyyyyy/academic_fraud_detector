"""
Evidence Chain — tracks the chronological flow of evidence collection
and supports cross-referencing between investigation categories.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class EvidenceStatus(str, Enum):
    """Status of an evidence item in the investigation lifecycle."""
    COLLECTED = "collected"        # Raw finding from a specialist agent
    VERIFIED = "verified"          # Cross-checked by at least one other agent
    REFUTED = "refuted"            # Found to be a false positive
    ESCALATED = "escalated"        # Flagged as high-confidence and serious
    INCLUDED = "included"          # Included in final report
    EXCLUDED = "excluded"          # Excluded from final report (explain why)


class EvidenceNode(BaseModel):
    """
    A single node in the evidence chain — wraps an evidence item
    with its collection context and cross-references.
    """

    evidence_id: str = Field(..., description="Unique ID, e.g. 'EVID-001'.")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this evidence was collected.",
    )
    source_agent: str = Field(
        ...,
        description="Which agent discovered this evidence.",
    )
    category: str = Field(
        ...,
        description="Fraud category: plagiarism, image_manipulation, data_fabrication, etc.",
    )
    status: EvidenceStatus = Field(
        default=EvidenceStatus.COLLECTED,
        description="Current status in the investigation lifecycle.",
    )
    short_description: str = Field(
        ...,
        description="One-line summary of the finding.",
    )
    detail: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full finding details as returned by the specialist agent.",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in this finding (0.0-1.0).",
    )

    # ── Cross-references ──
    corroborated_by: List[str] = Field(
        default_factory=list,
        description="Evidence IDs that support this finding.",
    )
    contradicted_by: List[str] = Field(
        default_factory=list,
        description="Evidence IDs that weaken or refute this finding.",
    )
    related_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence IDs that are thematically related but neither corroborate nor contradict.",
    )

    # ── Action ──
    escalated: bool = Field(
        default=False,
        description="Whether this evidence has been escalated for urgent attention.",
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Why this was escalated, if applicable.",
    )


class EvidenceChain(BaseModel):
    """
    The complete evidence chain for an investigation.

    Tracks all evidence nodes from collection through verification
    to final report inclusion. Supports cross-referencing and status tracking.
    """

    investigation_id: str = Field(..., description="Unique investigation identifier.")
    nodes: List[EvidenceNode] = Field(
        default_factory=list,
        description="All evidence nodes, in chronological order of collection.",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When the evidence chain was initialized.",
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="When the evidence chain was last modified.",
    )

    def add_node(self, node: EvidenceNode) -> None:
        """Add a new evidence node to the chain."""
        self.nodes.append(node)
        self.updated_at = datetime.now()

    def get_by_status(self, status: EvidenceStatus) -> List[EvidenceNode]:
        """Filter evidence nodes by status."""
        return [n for n in self.nodes if n.status == status]

    def get_by_category(self, category: str) -> List[EvidenceNode]:
        """Filter evidence nodes by fraud category."""
        return [n for n in self.nodes if n.category == category]

    def get_escalated(self) -> List[EvidenceNode]:
        """Get all escalated evidence nodes."""
        return [n for n in self.nodes if n.escalated]

    def get_verified(self) -> List[EvidenceNode]:
        """Get all verified evidence nodes."""
        return [n for n in self.nodes if n.status == EvidenceStatus.VERIFIED]

    def get_included(self) -> List[EvidenceNode]:
        """Get evidence nodes included in the final report."""
        return [n for n in self.nodes if n.status == EvidenceStatus.INCLUDED]

    def stats(self) -> dict:
        """Return summary statistics about the evidence chain."""
        if not self.nodes:
            return {"total": 0}
        categories = {}
        statuses = {}
        for node in self.nodes:
            categories[node.category] = categories.get(node.category, 0) + 1
            statuses[node.status.value] = statuses.get(node.status.value, 0) + 1
        return {
            "total": len(self.nodes),
            "by_category": categories,
            "by_status": statuses,
            "escalated_count": len(self.get_escalated()),
            "verified_count": len(self.get_verified()),
            "included_count": len(self.get_included()),
        }
