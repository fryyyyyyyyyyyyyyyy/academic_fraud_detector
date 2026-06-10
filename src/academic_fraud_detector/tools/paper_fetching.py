"""
Paper fetching tools — search and retrieve paper metadata from
arXiv, CrossRef, and Semantic Scholar APIs, or load from local PDF files.
"""

import json
import os
import logging
import re
from typing import Any, Optional, Literal

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..utils.api_client import safe_request
from ..utils.mineru_client import (
    MinerUConfigError,
    MinerUError,
    extract_pdf_markdown_with_mineru_assets,
)
from ..utils.text_extraction import (
    create_unique_image_output_dir,
    extract_pdf_text,
    extract_pdf_images,
)
from ..utils.table_extraction import extract_pdf_tables, extract_numeric_values, extract_p_values, extract_means_and_sds

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# arXiv Search Tool
# ═══════════════════════════════════════════════════════════════════════

class ArxivSearchInput(BaseModel):
    """Input schema for arXiv search."""

    query: str = Field(
        ...,
        description=(
            "Search query for arXiv API. Supports field prefixes: "
            "ti: (title), au: (author), abs: (abstract), cat: (category), "
            "all: (everywhere). Example: 'ti:transformer AND au:vaswani'."
        ),
    )
    max_results: int = Field(default=20, description="Maximum number of results.", ge=1, le=100)
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = Field(
        default="relevance", description="Sort criterion."
    )


class ArxivSearchTool(BaseTool):
    """Search academic papers on arXiv."""

    name: str = "arxiv_search"
    description: str = (
        "Search academic papers on arXiv by title, author, abstract keywords, or category. "
        "Returns paper metadata (title, authors, abstract, PDF URL, publication date). "
        "Use this to find candidate papers for plagiarism comparison or to retrieve the "
        "target paper's metadata if you have an arXiv ID."
    )
    args_schema: type[BaseModel] = ArxivSearchInput

    def _run(self, query: str, max_results: int = 20, sort_by: str = "relevance") -> str:
        """Execute arXiv search via the official API."""
        try:
            import arxiv

            sort_map = {
                "relevance": arxiv.SortCriterion.Relevance,
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                "submittedDate": arxiv.SortCriterion.SubmittedDate,
            }

            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_map.get(sort_by, arxiv.SortCriterion.Relevance),
            )

            results = []
            for paper in client.results(search):
                results.append({
                    "title": paper.title,
                    "authors": [str(a) for a in paper.authors],
                    "abstract": (paper.summary[:800] + "...") if len(paper.summary) > 800 else paper.summary,
                    "pdf_url": paper.pdf_url,
                    "published": paper.published.isoformat() if paper.published else None,
                    "arxiv_id": paper.entry_id.split("/")[-1] if "/" in paper.entry_id else paper.entry_id,
                    "primary_category": paper.primary_category,
                    "categories": list(paper.categories),
                })

            return json.dumps({
                "source": "arxiv",
                "query": query,
                "result_count": len(results),
                "results": results,
            }, ensure_ascii=False)

        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return json.dumps({"error": str(e), "source": "arxiv"}, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# CrossRef Search Tool
# ═══════════════════════════════════════════════════════════════════════

class CrossrefSearchInput(BaseModel):
    """Input schema for CrossRef search."""

    query: str = Field(..., description="Search query (title, author, or keywords).")
    rows: int = Field(default=20, description="Number of results.", ge=1, le=100)
    filter_type: Optional[str] = Field(
        default=None,
        description="Publication type filter: 'journal-article', 'proceedings-article', 'book-chapter', etc.",
    )


class CrossrefSearchTool(BaseTool):
    """Search the CrossRef database for academic papers."""

    name: str = "crossref_search"
    description: str = (
        "Search the CrossRef database for academic papers by title, author, DOI, "
        "or keywords. Covers a much broader range than arXiv — journals, conferences, "
        "books. Returns title, authors, DOI, journal, and year. "
        "Use this when searching for papers outside arXiv's scope or when you have a DOI."
    )
    args_schema: type[BaseModel] = CrossrefSearchInput

    def _run(self, query: str, rows: int = 20, filter_type: Optional[str] = None) -> str:
        """Execute CrossRef search."""
        try:
            base_url = "https://api.crossref.org/works"
            params: dict = {"query": query, "rows": rows}
            if filter_type:
                params["filter"] = f"type:{filter_type}"

            resp = safe_request(base_url, params=params, api_name="crossref")
            items = resp.json().get("message", {}).get("items", [])

            results = []
            for item in items:
                results.append({
                    "title": item.get("title", [""])[0] if item.get("title") else "",
                    "doi": item.get("DOI"),
                    "journal": (
                        item.get("container-title", [""])[0]
                        if item.get("container-title")
                        else None
                    ),
                    "year": (
                        item.get("published-print", {}).get("date-parts", [[None]])[0][0]
                        or item.get("published-online", {}).get("date-parts", [[None]])[0][0]
                    ),
                    "authors": [
                        f"{a.get('given', '')} {a.get('family', '')}"
                        for a in item.get("author", [])
                    ],
                    "publisher": item.get("publisher"),
                    "type": item.get("type"),
                })

            return json.dumps({
                "source": "crossref",
                "query": query,
                "result_count": len(results),
                "total_results": resp.json().get("message", {}).get("total-results", 0),
                "results": results,
            }, ensure_ascii=False)

        except Exception as e:
            logger.error(f"CrossRef search failed: {e}")
            return json.dumps({"error": str(e), "source": "crossref"}, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# Semantic Scholar Search Tool
# ═══════════════════════════════════════════════════════════════════════

class SemanticScholarSearchInput(BaseModel):
    """Input schema for Semantic Scholar search."""

    query: str = Field(..., description="Search query (title, author, or keywords).")
    limit: int = Field(default=20, description="Maximum results.", ge=1, le=100)
    year_range: Optional[str] = Field(
        default=None,
        description="Year range filter, e.g. '2020-2025'.",
    )
    fields_of_study: Optional[str] = Field(
        default=None,
        description="Comma-separated fields: 'Computer Science,Medicine,Biology'.",
    )


class SemanticScholarSearchTool(BaseTool):
    """Search papers on Semantic Scholar."""

    name: str = "semantic_scholar_search"
    description: str = (
        "Search Semantic Scholar's database. Provides enriched metadata including "
        "citation counts, reference lists, and influential citation data. "
        "Useful for finding related work, citation analysis, and identifying "
        "high-impact papers in a field. S2's citation graph is also accessible."
    )
    args_schema: type[BaseModel] = SemanticScholarSearchInput

    def _run(
        self,
        query: str,
        limit: int = 20,
        year_range: Optional[str] = None,
        fields_of_study: Optional[str] = None,
    ) -> str:
        """Execute Semantic Scholar search."""
        try:
            base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params: dict = {
                "query": query,
                "limit": limit,
                "fields": (
                    "paperId,title,authors,abstract,year,externalIds,"
                    "citationCount,referenceCount,fieldsOfStudy,"
                    "publicationTypes,journal,openAccessPdf"
                ),
            }
            if year_range:
                params["year"] = year_range
            if fields_of_study:
                params["fieldsOfStudy"] = fields_of_study

            resp = safe_request(base_url, params=params, api_name="semantic_scholar")
            data = resp.json().get("data", [])

            results = []
            for paper in data:
                results.append({
                    "paperId": paper.get("paperId"),
                    "title": paper.get("title"),
                    "authors": [
                        a.get("name", "") for a in paper.get("authors", [])
                    ],
                    "abstract": paper.get("abstract"),
                    "year": paper.get("year"),
                    "citationCount": paper.get("citationCount"),
                    "referenceCount": paper.get("referenceCount"),
                    "fieldsOfStudy": paper.get("fieldsOfStudy", []),
                    "doi": (
                        paper.get("externalIds", {}).get("DOI")
                        if paper.get("externalIds")
                        else None
                    ),
                    "arxivId": (
                        paper.get("externalIds", {}).get("ArXiv")
                        if paper.get("externalIds")
                        else None
                    ),
                    "journal": (
                        paper.get("journal", {}).get("name")
                        if paper.get("journal")
                        else None
                    ),
                })

            return json.dumps({
                "source": "semantic_scholar",
                "query": query,
                "result_count": len(results),
                "results": results,
            }, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return json.dumps({"error": str(e), "source": "semantic_scholar"}, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# Paper Lookup by ID Tool
# ═══════════════════════════════════════════════════════════════════════

class PaperLookupInput(BaseModel):
    """Input schema for paper lookup by identifier."""

    identifier: str = Field(..., description="Paper identifier (DOI, arXiv ID, or S2 Paper ID).")
    identifier_type: Literal["doi", "arxiv", "semantic_scholar", "title"] = Field(
        default="doi",
        description="Type of identifier provided.",
    )


class PaperLookupTool(BaseTool):
    """Retrieve detailed metadata for a specific paper by ID."""

    name: str = "paper_lookup"
    description: str = (
        "Retrieve complete metadata for a specific paper using its DOI, arXiv ID, "
        "or Semantic Scholar Paper ID. Returns full author list, abstract, citation counts, "
        "and reference list. Use this as the first step when investigating a known paper."
    )
    args_schema: type[BaseModel] = PaperLookupInput

    def _run(self, identifier: str, identifier_type: str = "doi") -> str:
        """Look up paper by identifier."""
        try:
            if identifier_type == "doi":
                return self._lookup_doi(identifier)
            elif identifier_type == "arxiv":
                return self._lookup_arxiv(identifier)
            elif identifier_type == "semantic_scholar":
                return self._lookup_s2(identifier)
            else:
                # Try Semantic Scholar title search
                search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
                resp = safe_request(
                    search_url,
                    params={"query": identifier, "limit": 1, "fields": self._s2_fields()},
                    api_name="semantic_scholar",
                )
                data = resp.json().get("data", [])
                if data:
                    return json.dumps(self._format_s2(data[0]), ensure_ascii=False)
                return json.dumps({"error": f"Paper not found by title: {identifier}"})
        except Exception as e:
            logger.error(f"Paper lookup failed: {e}")
            return json.dumps({"error": str(e)})

    def _s2_fields(self) -> str:
        return (
            "paperId,title,authors,abstract,year,externalIds,"
            "citationCount,referenceCount,fieldsOfStudy,journal,"
            "references.title,references.authors,references.paperId,"
            "citations.title,citations.authors,citations.paperId"
        )

    def _lookup_doi(self, doi: str) -> str:
        """Look up by DOI via CrossRef + Semantic Scholar."""
        # CrossRef for bibliographic metadata
        crossref_url = f"https://api.crossref.org/works/{doi}"
        resp = safe_request(crossref_url, api_name="crossref")
        item = resp.json().get("message", {})

        result = {
            "title": item.get("title", [""])[0] if item.get("title") else "",
            "doi": doi,
            "authors": [
                f"{a.get('given', '')} {a.get('family', '')}"
                for a in item.get("author", [])
            ],
            "abstract": item.get("abstract", ""),
            "year": (
                item.get("published-print", {}).get("date-parts", [[None]])[0][0]
                or item.get("published-online", {}).get("date-parts", [[None]])[0][0]
            ),
            "journal": (
                item.get("container-title", [""])[0]
                if item.get("container-title")
                else None
            ),
            "publisher": item.get("publisher"),
            "references": [],
        }

        # Enrich with Semantic Scholar data
        try:
            s2_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
            s2_resp = safe_request(
                s2_url,
                params={"fields": self._s2_fields()},
                api_name="semantic_scholar",
            )
            s2_data = s2_resp.json()
            if s2_data.get("paperId"):
                result.update(self._format_s2(s2_data))
        except Exception:
            pass  # S2 enrichment is optional

        return json.dumps(result, ensure_ascii=False)

    def _lookup_arxiv(self, arxiv_id: str) -> str:
        """Look up by arXiv ID."""
        try:
            import arxiv

            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(client.results(search))
            result = {
                "title": paper.title,
                "authors": [str(a) for a in paper.authors],
                "abstract": paper.summary,
                "year": paper.published.year if paper.published else None,
                "arxiv_id": arxiv_id,
                "pdf_url": paper.pdf_url,
                "categories": list(paper.categories),
            }
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"arXiv lookup failed: {e}")
            return json.dumps({"error": str(e)})

    def _lookup_s2(self, paper_id: str) -> str:
        """Look up by Semantic Scholar Paper ID."""
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        resp = safe_request(
            url,
            params={"fields": self._s2_fields()},
            api_name="semantic_scholar",
        )
        data = resp.json()
        return json.dumps(self._format_s2(data), ensure_ascii=False)

    def _format_s2(self, data: dict) -> dict:
        """Format Semantic Scholar API response into common schema."""
        return {
            "semantic_scholar_id": data.get("paperId"),
            "title": data.get("title"),
            "authors": [a.get("name", "") for a in data.get("authors", [])],
            "abstract": data.get("abstract"),
            "year": data.get("year"),
            "citation_count": data.get("citationCount"),
            "reference_count": data.get("referenceCount"),
            "fields_of_study": data.get("fieldsOfStudy", []),
            "doi": (
                data.get("externalIds", {}).get("DOI")
                if data.get("externalIds")
                else None
            ),
            "journal": (
                data.get("journal", {}).get("name")
                if data.get("journal")
                else None
            ),
            "references": [
                {
                    "paperId": r.get("paperId"),
                    "title": r.get("title"),
                    "authors": [a.get("name", "") for a in r.get("authors", [])],
                }
                for r in data.get("references", [])
            ],
        }


# ═══════════════════════════════════════════════════════════════════════
# Local Paper Loader Tool
# ═══════════════════════════════════════════════════════════════════════

class LocalPaperLoaderInput(BaseModel):
    """Input schema for loading a local PDF paper."""

    file_path: str = Field(..., description="Absolute or relative path to the local PDF file.")
    max_pages: int = Field(
        default=200,
        ge=1,
        le=1000,
        description="Maximum number of pages to extract text from.",
    )
    extract_images: bool = Field(
        default=True,
        description="Whether to extract embedded images from the PDF.",
    )
    extract_tables: bool = Field(
        default=True,
        description="Whether to extract tables from the PDF.",
    )
    image_min_size: int = Field(
        default=100,
        ge=0,
        le=1000,
        description="Minimum image dimension (px) to keep. 0 = keep all sizes. Smaller images are ignored.",
    )
    supplementary_paths: str = Field(
        default="",
        description=(
            "Optional. JSON-encoded list of paths to supplementary PDF files "
            "(e.g., S1 File, S2 File). Example: '[\"/path/to/s1_file.pdf\", \"/path/to/s2_file.pdf\"]'. "
            "If empty, auto-detects supplementary files in the same directory as the main paper."
        ),
    )


class LocalPaperLoaderTool(BaseTool):
    """
    Load and process a local PDF paper for investigation.

    This is the primary data source when investigating a paper from a
    user-uploaded PDF file. It extracts:
    - Full text content (all pages)
    - Embedded images/figures (saved to disk)
    - Tables with numeric data
    - Pre-extracted p-values, means, SDs, and other statistics

    All extracted data is structured as JSON for downstream agent analysis.
    When a MinerU API key is configured, text extraction uses MinerU VLM
    Markdown first; otherwise it falls back to local PyMuPDF extraction.
    """

    name: str = "local_paper_loader"
    description: str = (
        "Load a local PDF file and extract all content needed for investigation: "
        "full text, embedded images/figures (saved to disk), tables, and pre-extracted "
        "statistical values (p-values, means, SDs, etc.). "
        "ALSO supports loading supplementary materials (S1 File, S2 File, etc.) "
        "from the same directory — CRITICAL for accessing the underlying raw data "
        "(e.g., bar chart values in supplementary spreadsheets/tables). "
        "Use this as the FIRST step when investigating a user-uploaded PDF. "
        "Returns structured JSON with text content, image file paths (for the image "
        "forensics tools), table data, and extracted numeric statistics. "
        "When MINERU_API_KEY is configured, converts PDF text to Markdown via "
        "MinerU VLM first; otherwise falls back to local PyMuPDF extraction."
    )
    args_schema: type[BaseModel] = LocalPaperLoaderInput

    @staticmethod
    def _count_pdf_pages(pdf_bytes: bytes, max_pages: Optional[int] = None) -> int:
        """Count PDF pages with PyMuPDF without changing extracted text."""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_count = doc.page_count
            doc.close()
            if max_pages:
                page_count = min(page_count, max_pages)
            return int(page_count)
        except Exception as e:
            logger.debug(f"PDF page counting failed: {e}")
            return 0

    @staticmethod
    def _pymupdf_images_with_source(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Mark PyMuPDF-extracted images without changing existing metadata."""
        for image in images:
            image.setdefault("source", "pymupdf")
        return images

    @staticmethod
    def _mineru_images_with_source(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Mark MinerU-extracted images without changing existing metadata."""
        for image in images:
            image.setdefault("source", "mineru")
        return images

    def _extract_mineru_first(
        self,
        pdf_bytes: bytes,
        file_name: str,
        max_pages: int,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Extract MinerU Markdown/assets first, then fall back to PyMuPDF text."""
        try:
            result = extract_pdf_markdown_with_mineru_assets(
                pdf_bytes,
                file_name=file_name,
                max_pages=max_pages,
            )
            if result.markdown.strip():
                mineru_meta = {
                    "used": True,
                    "cache_dir": result.cache_dir,
                    "full_md_path": result.full_md_path,
                    "raw_full_md_path": result.raw_full_md_path,
                    "zip_path": result.zip_path,
                    "image_count": len(result.images),
                }
                return result.markdown, result.images, mineru_meta
            logger.warning("MinerU returned empty Markdown for %s; falling back to PyMuPDF", file_name)
            reason = "empty_markdown"
        except MinerUConfigError:
            logger.info("MinerU API key not configured; using PyMuPDF for %s", file_name)
            reason = "not_configured"
        except MinerUError as e:
            logger.warning("MinerU extraction failed for %s; falling back to PyMuPDF: %s", file_name, e)
            reason = type(e).__name__
        except Exception as e:
            logger.warning(
                "Unexpected MinerU extraction error for %s; falling back to PyMuPDF: %s",
                file_name,
                e,
            )
            reason = "unexpected_error"

        fallback_text = extract_pdf_text(pdf_bytes, max_pages=max_pages)
        return fallback_text, [], {"used": False, "fallback": "pymupdf", "reason": reason}

    def _extract_text_mineru_first(
        self,
        pdf_bytes: bytes,
        file_name: str,
        max_pages: int,
    ) -> str:
        """Extract PDF text as MinerU Markdown first, then fall back to PyMuPDF."""
        text, _, _ = self._extract_mineru_first(pdf_bytes, file_name, max_pages)
        return text

    def _page_count_from_text_or_pdf(
        self,
        text: str,
        pdf_bytes: bytes,
        max_pages: Optional[int] = None,
    ) -> int:
        """Prefer existing [Page N] markers; otherwise count pages from PDF bytes."""
        page_markers = re.findall(r'\[Page (\d+)\]', text)
        if page_markers:
            count = int(page_markers[-1])
            return min(count, max_pages) if max_pages else count
        return self._count_pdf_pages(pdf_bytes, max_pages=max_pages)

    def _run(
        self,
        file_path: str,
        max_pages: int = 200,
        extract_images: bool = True,
        extract_tables: bool = True,
        image_min_size: int = 100,
        supplementary_paths: str = "",
    ) -> str:
        """Load and process a local PDF file."""
        # Validate file exists
        if not os.path.exists(file_path):
            return json.dumps({
                "error": f"File not found: {file_path}",
                "full_text_available": False,
            })

        file_path = os.path.abspath(file_path)
        file_size = os.path.getsize(file_path)

        result = {
            "source": "local_pdf",
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size_bytes": file_size,
            "full_text_available": False,
            "full_text": None,
            "full_text_length_chars": 0,
            "page_count": 0,
            "images": [],
            "image_output_dir": None,
            "tables": [],
            "panels": [],
            "mineru": {"used": False},
            "pre_extracted_stats": {
                "p_values": [],
                "means_and_sds": [],
                "numeric_values": [],
            },
            "error": None,
        }

        # ── Read PDF bytes ──
        try:
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()
        except Exception as e:
            result["error"] = f"Failed to read file: {e}"
            return json.dumps(result)

        mineru_images: list[dict[str, Any]] = []

        # ── Extract text ──
        try:
            full_text, mineru_images, mineru_meta = self._extract_mineru_first(
                pdf_bytes,
                file_name=result["file_name"],
                max_pages=max_pages,
            )
            result["mineru"] = mineru_meta
            if full_text.strip():
                result["full_text_available"] = True
                result["full_text"] = full_text
                result["full_text_length_chars"] = len(full_text)
                result["page_count"] = self._page_count_from_text_or_pdf(
                    full_text,
                    pdf_bytes,
                    max_pages=max_pages,
                )

                # Pre-extract statistical values from text
                result["pre_extracted_stats"]["p_values"] = [
                    round(v, 6) for v in extract_p_values(full_text)
                ]
                result["pre_extracted_stats"]["means_and_sds"] = extract_means_and_sds(full_text)
        except Exception as e:
            logger.warning(f"Text extraction failed: {e}")
            result["full_text"] = f"Text extraction error: {e}"

        # ── Extract images ──
        if extract_images:
            if result.get("mineru", {}).get("used"):
                result["images"] = self._mineru_images_with_source(mineru_images)
                result["image_output_dir"] = result["mineru"].get("cache_dir")
                if not result["images"]:
                    logger.info(
                        "MinerU succeeded for %s but returned no cached images; "
                        "skipping PyMuPDF image extraction by design.",
                        result["file_name"],
                    )
            else:
                image_output_dir = create_unique_image_output_dir(
                    prefix="pymupdf_images",
                    source_name=result["file_name"],
                )
                result["image_output_dir"] = str(image_output_dir)
                try:
                    images = extract_pdf_images(
                        pdf_bytes,
                        output_dir=str(image_output_dir),
                        min_size=image_min_size,
                        max_pages=max_pages,
                    )
                    result["images"] = self._pymupdf_images_with_source(images)
                except Exception as e:
                    logger.warning(f"Image extraction failed: {e}")
                    result["images"] = []

        # ── Extract tables ──
        if extract_tables:
            try:
                tables = extract_pdf_tables(pdf_bytes)
                for table in tables:
                    table["numeric_values"] = extract_numeric_values(table.get("data", []))
                result["tables"] = tables

                # Also collect all numeric values
                for table in tables:
                    result["pre_extracted_stats"]["numeric_values"].extend(
                        table.get("numeric_values", [])
                    )
            except Exception as e:
                logger.warning(f"Table extraction failed: {e}")
                result["tables"] = []

        # ── Split composite figures into individual panels ──
        # This is CRITICAL for detecting partial image reuse (e.g., a single
        # panel from a multi-panel figure reused in another figure). Without
        # panel splitting, cross-image comparison tools compare entire composite
        # images and miss partial overlaps.
        result["panels"] = []
        if extract_images and result["images"]:
            try:
                from ..utils.figure_splitter import extract_all_panels_from_images
                panel_output_dir = result.get("image_output_dir")
                if not panel_output_dir:
                    panel_output_dir = str(create_unique_image_output_dir(
                        prefix="panels",
                        source_name=result["file_name"],
                    ))
                    result["image_output_dir"] = panel_output_dir
                panels_data = extract_all_panels_from_images(
                    result["images"],
                    output_dir=panel_output_dir,
                    min_panel_size=80,
                )
                # Flatten panel paths for easy consumption by forensic tools
                all_panel_paths = []
                for entry in panels_data:
                    for panel in entry.get("panels", []):
                        if panel.get("filepath"):
                            all_panel_paths.append(panel["filepath"])
                result["panels"] = all_panel_paths
                composite_count = sum(1 for e in panels_data if e.get("is_composite"))
                logger.info(
                    f"Panel splitting complete: {len(result['images'])} figures → "
                    f"{len(all_panel_paths)} panels ({composite_count} composite figures split)"
                )
            except Exception as e:
                logger.warning(f"Panel splitting failed (non-fatal): {e}")
                result["panels"] = []

        # ── Load supplementary materials ──
        # Supplementary files (S1 File, S2 File, etc.) often contain the raw
        # numerical data behind bar charts. Without loading these, statistical
        # tests like GRIM, Benford, and cross-figure data comparison cannot
        # function because the data is only in the supplementary files.
        result["supplementary_files"] = []
        supp_paths = self._resolve_supplementary_paths(file_path, supplementary_paths)
        if supp_paths:
            logger.info(f"Loading {len(supp_paths)} supplementary file(s): {supp_paths}")
            for supp_path in supp_paths:
                try:
                    supp_result = self._load_single_pdf(
                        supp_path, max_pages, extract_images, extract_tables, image_min_size
                    )
                    # Merge statistics from supplementary files
                    result["supplementary_files"].append({
                        "file_path": supp_path,
                        "file_name": os.path.basename(supp_path),
                        "stats": supp_result.get("pre_extracted_stats", {}),
                        "tables": supp_result.get("tables", []),
                        "text_length": supp_result.get("full_text_length_chars", 0),
                        "image_output_dir": supp_result.get("image_output_dir"),
                        "mineru": supp_result.get("mineru", {"used": False}),
                    })
                    # Merge numeric data
                    supp_stats = supp_result.get("pre_extracted_stats", {})
                    result["pre_extracted_stats"]["p_values"].extend(
                        supp_stats.get("p_values", [])
                    )
                    result["pre_extracted_stats"]["means_and_sds"].extend(
                        supp_stats.get("means_and_sds", [])
                    )
                    result["pre_extracted_stats"]["numeric_values"].extend(
                        supp_stats.get("numeric_values", [])
                    )
                    # Merge tables
                    result["tables"].extend(supp_result.get("tables", []))
                    # Merge panels if any
                    if supp_result.get("panels"):
                        result["panels"].extend(supp_result["panels"])
                except Exception as e:
                    logger.warning(f"Failed to load supplementary file {supp_path}: {e}")
                    result["supplementary_files"].append({
                        "file_path": supp_path,
                        "error": str(e),
                    })

        # ── Summary ──
        supp_info = (
            f", {len(result['supplementary_files'])} supplementary files"
            if result["supplementary_files"] else ""
        )
        result["_summary"] = (
            f"PDF loaded: {os.path.basename(file_path)} "
            f"({result['page_count']} pages, {len(result['full_text']) if result['full_text'] else 0} chars text, "
            f"{len(result['images'])} images, {len(result['tables'])} tables, "
            f"{len(result['panels'])} individual panels{supp_info}, "
            f"{len(result['pre_extracted_stats']['p_values'])} p-values, "
            f"{len(result['pre_extracted_stats']['means_and_sds'])} mean±SD pairs)"
        )

        logger.info(result["_summary"])
        return json.dumps(result, ensure_ascii=False)

    def _load_single_pdf(
        self,
        file_path: str,
        max_pages: int,
        extract_images: bool,
        extract_tables: bool,
        image_min_size: int,
    ) -> dict:
        """Load a single PDF and return its extracted content as a dict (no JSON)."""
        from ..utils.table_extraction import extract_pdf_tables, extract_numeric_values, extract_p_values, extract_means_and_sds
        from ..utils.figure_splitter import extract_all_panels_from_images

        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        result = {
            "full_text_available": False,
            "full_text": None,
            "full_text_length_chars": 0,
            "page_count": 0,
            "images": [],
            "image_output_dir": None,
            "tables": [],
            "panels": [],
            "mineru": {"used": False},
            "pre_extracted_stats": {
                "p_values": [],
                "means_and_sds": [],
                "numeric_values": [],
            },
        }

        mineru_images: list[dict[str, Any]] = []

        # Text
        try:
            text, mineru_images, mineru_meta = self._extract_mineru_first(
                pdf_bytes,
                file_name=os.path.basename(file_path),
                max_pages=max_pages,
            )
            result["mineru"] = mineru_meta
            if text.strip():
                result["full_text_available"] = True
                result["full_text"] = text
                result["full_text_length_chars"] = len(text)
                result["page_count"] = self._page_count_from_text_or_pdf(
                    text,
                    pdf_bytes,
                    max_pages=max_pages,
                )
                result["pre_extracted_stats"]["p_values"] = [
                    round(v, 6) for v in extract_p_values(text)
                ]
                result["pre_extracted_stats"]["means_and_sds"] = extract_means_and_sds(text)
        except Exception as e:
            logger.warning(f"Supp text extraction failed: {e}")

        # Images
        if extract_images:
            if result.get("mineru", {}).get("used"):
                result["images"] = self._mineru_images_with_source(mineru_images)
                result["image_output_dir"] = result["mineru"].get("cache_dir")
                if not result["images"]:
                    logger.info(
                        "MinerU succeeded for supplementary file %s but returned no cached images; "
                        "skipping PyMuPDF image extraction by design.",
                        os.path.basename(file_path),
                    )
            else:
                image_output_dir = create_unique_image_output_dir(
                    prefix="pymupdf_images",
                    source_name=os.path.basename(file_path),
                )
                result["image_output_dir"] = str(image_output_dir)
                try:
                    pymupdf_images = extract_pdf_images(
                        pdf_bytes,
                        output_dir=str(image_output_dir),
                        min_size=image_min_size,
                        max_pages=max_pages,
                    )
                    result["images"] = self._pymupdf_images_with_source(pymupdf_images)
                except Exception as e:
                    logger.warning(f"Supp image extraction failed: {e}")
                    result["images"] = []

        # Tables
        if extract_tables:
            try:
                tables = extract_pdf_tables(pdf_bytes)
                for t in tables:
                    t["numeric_values"] = extract_numeric_values(t.get("data", []))
                result["tables"] = tables
                for t in tables:
                    result["pre_extracted_stats"]["numeric_values"].extend(
                        t.get("numeric_values", [])
                    )
            except Exception as e:
                logger.warning(f"Supp table extraction failed: {e}")

        # Panels
        if extract_images and result["images"]:
            try:
                panel_output_dir = result.get("image_output_dir")
                if not panel_output_dir:
                    panel_output_dir = str(create_unique_image_output_dir(
                        prefix="panels",
                        source_name=os.path.basename(file_path),
                    ))
                    result["image_output_dir"] = panel_output_dir
                panels_data = extract_all_panels_from_images(
                    result["images"],
                    output_dir=panel_output_dir,
                    min_panel_size=80,
                )
                for entry in panels_data:
                    for panel in entry.get("panels", []):
                        if panel.get("filepath"):
                            result["panels"].append(panel["filepath"])
            except Exception as e:
                logger.warning(f"Supp panel splitting failed: {e}")

        return result

    @staticmethod
    def _resolve_supplementary_paths(main_path: str, supplementary_paths: str) -> list:
        """
        Resolve the list of supplementary file paths.

        If supplementary_paths is non-empty, parse it as JSON.
        Otherwise, auto-detect supplementary files in the same directory as
        the main paper by looking for files matching common naming patterns.
        """
        import glob

        paths = []

        # 1. Explicit paths from the agent
        if supplementary_paths and supplementary_paths.strip():
            try:
                parsed = json.loads(supplementary_paths)
                if isinstance(parsed, list):
                    paths = [p for p in parsed if os.path.isfile(p)]
            except json.JSONDecodeError:
                # Maybe it's a single path
                if os.path.isfile(supplementary_paths):
                    paths = [supplementary_paths]

        # 2. Auto-detect supplementary files in the same directory
        if not paths:
            main_dir = os.path.dirname(os.path.abspath(main_path))
            main_name = os.path.splitext(os.path.basename(main_path))[0]

            # Common supplementary file naming patterns
            patterns = [
                # Same directory, S* File pattern: pone.0313446.s001.pdf, etc.
                os.path.join(main_dir, f"{main_name}.s*.pdf"),
                # Supporting Information
                os.path.join(main_dir, "*supplement*.pdf"),
                os.path.join(main_dir, "*suppl*.pdf"),
                os.path.join(main_dir, "*supporting*.pdf"),
                # S1 File, S2 File, etc.
                os.path.join(main_dir, "*S[0-9]*File*.pdf"),
                os.path.join(main_dir, "*S[0-9]*.pdf"),
                # Appendix
                os.path.join(main_dir, "*appendix*.pdf"),
                # Data file
                os.path.join(main_dir, "*data*.pdf"),
            ]

            found = set()
            for pattern in patterns:
                for match in glob.glob(pattern):
                    abs_path = os.path.abspath(match)
                    if abs_path != os.path.abspath(main_path) and abs_path not in found:
                        found.add(abs_path)
                        paths.append(abs_path)

            if found:
                logger.info(
                    f"Auto-detected {len(found)} supplementary file(s): "
                    f"{[os.path.basename(p) for p in paths]}"
                )

        return paths
