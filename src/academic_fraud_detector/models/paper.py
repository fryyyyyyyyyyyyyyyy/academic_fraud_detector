"""
Paper metadata model — standardized representation of an academic paper
across different source APIs (arXiv, CrossRef, Semantic Scholar).
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional
from datetime import datetime


class Author(BaseModel):
    """A paper author."""

    full_name: str = Field(..., description="Full name, e.g. 'Jane Smith'.")
    first_name: Optional[str] = Field(default=None, description="Given name.")
    last_name: Optional[str] = Field(default=None, description="Family name.")
    affiliation: Optional[str] = Field(default=None, description="Institutional affiliation.")
    email: Optional[str] = Field(default=None, description="Email if publicly available.")
    orcid: Optional[str] = Field(default=None, description="ORCID identifier.")
    semantic_scholar_id: Optional[str] = Field(
        default=None, description="Semantic Scholar author ID."
    )


class Figure(BaseModel):
    """A figure/image extracted from a paper."""

    figure_number: int = Field(..., description="Figure number in the paper (1-indexed).")
    caption: Optional[str] = Field(default=None, description="Figure caption text.")
    image_url: Optional[str] = Field(default=None, description="URL to the figure image.")
    local_path: Optional[str] = Field(default=None, description="Local file path after download.")
    width: Optional[int] = Field(default=None, description="Image width in pixels.")
    height: Optional[int] = Field(default=None, description="Image height in pixels.")
    format: Optional[str] = Field(default=None, description="Image format (png, jpg, tiff, etc.).")


class Table(BaseModel):
    """A table extracted from a paper."""

    table_number: int = Field(..., description="Table number in the paper (1-indexed).")
    caption: Optional[str] = Field(default=None, description="Table caption text.")
    rows: int = Field(default=0, description="Number of data rows.")
    columns: int = Field(default=0, description="Number of data columns.")
    extracted_data: Optional[List[List[str]]] = Field(
        default=None, description="Table data as list of rows, each row is a list of cell values."
    )


class PaperMetadata(BaseModel):
    """
    Standardized paper metadata, sourced from one or more APIs.

    This is the canonical internal representation. All API-specific
    fields are normalized into this schema.
    """

    # ── Identifiers ──
    title: str = Field(..., description="Paper title.")
    doi: Optional[str] = Field(default=None, description="DOI (Digital Object Identifier).")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv identifier, e.g. '2301.12345'.")
    semantic_scholar_id: Optional[str] = Field(
        default=None, description="Semantic Scholar Corpus ID or Paper ID."
    )
    pubmed_id: Optional[str] = Field(default=None, description="PubMed ID if applicable.")

    # ── Bibliographic ──
    authors: List[Author] = Field(default_factory=list, description="List of authors in order.")
    abstract: Optional[str] = Field(default=None, description="Paper abstract.")
    year: Optional[int] = Field(default=None, description="Publication year.")
    journal: Optional[str] = Field(
        default=None, description="Journal or conference name."
    )
    publisher: Optional[str] = Field(default=None, description="Publisher name.")
    keywords: List[str] = Field(default_factory=list, description="Author keywords or MeSH terms.")

    # ── Content ──
    full_text: Optional[str] = Field(
        default=None, description="Full paper text if extractable (PDF/HTML)."
    )
    pdf_url: Optional[str] = Field(default=None, description="URL to the paper PDF.")
    figures: List[Figure] = Field(
        default_factory=list, description="Figures/images extracted from the paper."
    )
    tables: List[Table] = Field(
        default_factory=list, description="Tables extracted from the paper."
    )

    # ── Bibliometrics ──
    citation_count: Optional[int] = Field(
        default=None, description="Total citation count (from Semantic Scholar)."
    )
    reference_count: Optional[int] = Field(
        default=None, description="Number of references in the paper."
    )
    references: List[str] = Field(
        default_factory=list,
        description="List of DOIs or paper IDs that this paper cites.",
    )
    influential_citation_count: Optional[int] = Field(
        default=None,
        description="Count of influential citations (Semantic Scholar metric).",
    )

    # ── Source Tracking ──
    source_api: Optional[str] = Field(
        default=None,
        description="Which API provided this data: 'arxiv', 'crossref', 'semantic_scholar'.",
    )
    retrieved_at: Optional[datetime] = Field(
        default=None, description="When this metadata was retrieved."
    )

    def short_citation(self) -> str:
        """Return a short citation string like 'Smith et al. (2023)'."""
        if not self.authors:
            return f"Unknown ({self.year or 'n.d.'})"
        first_author = self.authors[0].last_name or self.authors[0].full_name.split()[-1]
        if len(self.authors) == 1:
            return f"{first_author} ({self.year or 'n.d.'})"
        elif len(self.authors) == 2:
            second = self.authors[1].last_name or self.authors[1].full_name.split()[-1]
            return f"{first_author} & {second} ({self.year or 'n.d.'})"
        else:
            return f"{first_author} et al. ({self.year or 'n.d.'})"
