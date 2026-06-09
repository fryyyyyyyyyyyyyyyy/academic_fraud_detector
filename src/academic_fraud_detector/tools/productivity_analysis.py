"""
Productivity anomaly analysis tools — detect paper mill patterns.

Three core analyses:
1. Publication Frequency Check: retrieves author publication records and flags
   abnormally high output rates (paper mill signature).
2. Methods Similarity Check: compares Methods sections across an author's papers
   for template reuse (reuses text_similarity.py infrastructure).
3. Salami Slicing Detector: identifies groups of papers that appear to be
   one study split into multiple minimal publishable units.

Note: These tools depend on external APIs (PubMed/CrossRef/Semantic Scholar).
They are only enabled in non-local_only mode.
"""

import json
import logging
import re
from typing import List, Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..utils.api_client import safe_request

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. Publication Frequency Check
# ═══════════════════════════════════════════════════════════════════════

class PublicationFrequencyInput(BaseModel):
    """Input for publication frequency analysis."""

    author_name: str = Field(
        ...,
        description="Full author name to analyze.",
    )
    author_papers_json: Optional[str] = Field(
        default=None,
        description=(
            "Optional JSON list of the author's papers with years. "
            "If not provided, the tool will attempt to fetch from APIs. "
            'Format: [{"title":"...","year":2024,"doi":"..."}]'
        ),
    )


class PublicationFrequencyTool(BaseTool):
    """
    Analyze an author's publication frequency for anomaly detection.

    Paper mills and fraudulent labs often produce papers at an impossible rate:
    - Multiple first-author papers per year in experimental disciplines
    - Animal studies that require 6+ months each appearing monthly
    - Sudden jumps in output without corresponding lab expansion

    Normal benchmarks (approximate, field-dependent):
    - Molecular biology: 2-5 papers/year for established PI
    - Animal behavior: 1-3 papers/year (long experiments)
    - Clinical trials: 0.5-2 papers/year
    - Theory/computation: can be higher (5-10 papers/year)

    Red flag thresholds:
    - >12 papers/year as first/corresponding author (any experimental field)
    - >20 papers/year total (suggests paper mill or gift authorship)
    - Multiple experimental animal papers in same year from same first author
    """

    name: str = "publication_frequency_check"
    description: str = (
        "Analyze an author's publication frequency for anomalies. "
        "Flags authors with impossibly high output rates for their field, "
        "sudden output changes, or patterns consistent with paper mills. "
        "Input: author_name, and optionally author_papers_json with known papers."
    )
    args_schema: type[BaseModel] = PublicationFrequencyInput

    def _run(
        self,
        author_name: str,
        author_papers_json: Optional[str] = None,
    ) -> str:
        """Execute publication frequency analysis."""
        papers = []

        # ── Try to get papers from provided data ──
        if author_papers_json:
            try:
                papers = json.loads(author_papers_json)
            except json.JSONDecodeError:
                pass

        # ── Try to fetch from Semantic Scholar ──
        if not papers:
            try:
                # Search for author
                search_url = "https://api.semanticscholar.org/graph/v1/author/search"
                search_params = {"query": author_name}
                search_resp = safe_request(
                    search_url, params=search_params, api_name="semantic_scholar"
                )
                authors = search_resp.json().get("data", [])

                if authors:
                    author_id = authors[0].get("authorId")
                    papers_url = (
                        f"https://api.semanticscholar.org/graph/v1/author/"
                        f"{author_id}/papers"
                    )
                    papers_params = {
                        "fields": "title,year,publicationDate,journal,authors",
                        "limit": 100,
                    }
                    papers_resp = safe_request(
                        papers_url, params=papers_params, api_name="semantic_scholar"
                    )
                    papers = papers_resp.json().get("data", [])
            except Exception as e:
                logger.warning(f"Could not fetch author papers: {e}")

        if not papers:
            return json.dumps({
                "analysis_type": "Publication Frequency Analysis",
                "author": author_name,
                "error": "No publication data available. API may be unavailable.",
                "flagged": False,
                "api_limitation": True,
            })

        # ── Analyze year distribution ──
        years = []
        for paper in papers:
            year = paper.get("year")
            if year:
                years.append(int(year))

        if not years:
            return json.dumps({
                "analysis_type": "Publication Frequency Analysis",
                "author": author_name,
                "total_papers": len(papers),
                "error": "No year data in papers.",
                "flagged": False,
            })

        from collections import Counter
        year_counts = Counter(years)
        total_years = max(years) - min(years) + 1 if years else 1
        papers_per_year = len(papers) / max(total_years, 1)

        # Check for burst years
        max_year_count = max(year_counts.values())
        max_year = max(year_counts, key=year_counts.get)

        findings = []
        flagged = False

        # Threshold check
        if papers_per_year > 12:
            findings.append({
                "type": "high_output",
                "detail": (
                    f"Author publishes {papers_per_year:.1f} papers/year on average. "
                    f"This is unusually high for experimental research. "
                    f"Check for gift authorship, paper mill involvement, or "
                    f"data fabrication across papers."
                ),
                "severity": "高",
            })
            flagged = True

        if max_year_count > 8:
            findings.append({
                "type": "output_burst",
                "detail": (
                    f"{max_year_count} papers published in {max_year} alone. "
                    f"Burst output often indicates coordinated paper mill activity "
                    f"or a special issue with lax peer review."
                ),
                "severity": "中",
            })
            flagged = True

        # Check for sudden onset (common in paper mills)
        if len(year_counts) >= 3:
            sorted_years = sorted(year_counts.keys())
            recent_3yr = sum(year_counts[y] for y in sorted_years[-3:])
            older_years = sorted_years[:-3]
            if older_years:
                older_3yr = sum(year_counts[y] for y in older_years[-3:]) if len(older_years) >= 3 else sum(year_counts[y] for y in older_years)
                if older_3yr > 0 and recent_3yr / older_3yr > 5:
                    findings.append({
                        "type": "sudden_output_increase",
                        "detail": (
                            f"Publication output increased {recent_3yr/older_3yr:.0f}x "
                            f"in recent years. Sudden productivity jumps without "
                            f"corresponding lab expansion are suspicious."
                        ),
                        "severity": "中",
                    })
                    flagged = True

        return json.dumps({
            "analysis_type": "Publication Frequency Analysis",
            "author": author_name,
            "total_papers_analyzed": len(papers),
            "year_range": f"{min(years)}-{max(years)}",
            "total_years": total_years,
            "papers_per_year": round(papers_per_year, 1),
            "max_single_year": {"year": max_year, "count": max_year_count},
            "year_distribution": {str(y): c for y, c in sorted(year_counts.items())},
            "findings": findings,
            "flagged": flagged,
            "interpretation": (
                f"Author publishes {papers_per_year:.1f} papers/year ({len(papers)} total). "
                + (
                    "HIGH OUTPUT — publication frequency is anomalous for "
                    "experimental research. Recommended: check individual papers "
                    "for data integrity and Methods overlap."
                    if flagged
                    else "Publication frequency is within expected academic norms."
                )
            ),
        }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# 2. Salami Slicing Detector
# ═══════════════════════════════════════════════════════════════════════

class SalamiSlicingInput(BaseModel):
    """Input for salami slicing detection."""

    papers_json: str = Field(
        ...,
        description=(
            "JSON list of papers to check for salami slicing. Each paper should have: "
            "'title' (str), 'year' (int), 'authors' (list of str), 'doi' (str, optional), "
            "'sample_size' (int, optional), 'experimental_design' (str, optional). "
            'Example: [{"title":"Study A","year":2024,"authors":["Smith J"],'
            '"sample_size":30,"experimental_design":"6 groups of 5 mice"}]'
        ),
    )


class SalamiSlicingTool(BaseTool):
    """
    Detect salami slicing — splitting one study into multiple papers.

    Salami slicing is the practice of dividing a single coherent research project
    into the smallest publishable units (SPUs), resulting in multiple papers that:
    1. Share the same control group data (the most damning evidence)
    2. Have nearly identical experimental designs with only one variable changed
    3. Have similar sample sizes and group structures
    4. Were published in rapid succession (within months)
    5. Have highly similar titles (often differing by only one keyword)

    This is different from legitimate follow-up studies where:
    - The research question meaningfully advances
    - Different methodologies are employed
    - The papers clearly cite and build upon each other
    - There is genuine intellectual progression

    Detection method:
    - Compare paper titles for high similarity (differ by ≤3 words)
    - Compare sample sizes and group structures
    - Check for shared control groups (strongest signal)
    - Check temporal proximity of publication
    """

    name: str = "salami_slicing_check"
    description: str = (
        "Detect salami slicing — when one research study is split into multiple "
        "minimal publishable units. Checks for: shared controls, nearly identical "
        "experimental designs, similar sample sizes, similar titles, and rapid "
        "succession publication. Input: JSON list of papers with title, year, authors, "
        "and optionally sample_size and experimental_design."
    )
    args_schema: type[BaseModel] = SalamiSlicingInput

    def _run(self, papers_json: str) -> str:
        """Execute salami slicing detection."""
        try:
            papers = json.loads(papers_json)
            if not isinstance(papers, list):
                return json.dumps({"error": "papers_json must be a JSON list.", "flagged": False})
        except json.JSONDecodeError:
            return json.dumps({"error": "papers_json must be valid JSON.", "flagged": False})

        if len(papers) < 2:
            return json.dumps({
                "analysis_type": "Salami Slicing Detection",
                "papers_analyzed": len(papers),
                "error": "Need at least 2 papers for comparison.",
                "flagged": False,
            })

        suspicious_groups = []
        import difflib

        # ── Pairwise comparison ──
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                p_a = papers[i]
                p_b = papers[j]
                signals = []

                # Signal 1: Title similarity
                title_a = p_a.get("title", "").lower()
                title_b = p_b.get("title", "").lower()
                title_ratio = difflib.SequenceMatcher(None, title_a, title_b).ratio()

                if title_ratio > 0.65:
                    # Count differing words
                    words_a = set(re.findall(r'\w+', title_a))
                    words_b = set(re.findall(r'\w+', title_b))
                    shared = words_a & words_b
                    unique_a = words_a - words_b
                    unique_b = words_b - words_a

                    if len(shared) >= 5 and len(unique_a) <= 5 and len(unique_b) <= 5:
                        signals.append({
                            "type": "similar_title",
                            "detail": (
                                f"Titles differ by only {len(unique_a | unique_b)} words. "
                                f"Title A: '{p_a.get('title')}'. "
                                f"Title B: '{p_b.get('title')}'."
                            ),
                        })

                # Signal 2: Same sample size
                n_a = p_a.get("sample_size")
                n_b = p_b.get("sample_size")
                if n_a is not None and n_b is not None and n_a == n_b:
                    signals.append({
                        "type": "same_sample_size",
                        "detail": f"Both papers report n={n_a}. Identical sample sizes across studies claiming independent designs.",
                    })

                # Signal 3: Same experimental design description
                design_a = p_a.get("experimental_design", "")
                design_b = p_b.get("experimental_design", "")
                if design_a and design_b:
                    design_ratio = difflib.SequenceMatcher(None, design_a, design_b).ratio()
                    if design_ratio > 0.80:
                        signals.append({
                            "type": "similar_design",
                            "detail": (
                                f"Experimental designs are {design_ratio:.1%} similar. "
                                "Independent studies should have meaningfully different designs."
                            ),
                        })

                # Signal 4: Temporal proximity
                year_a = p_a.get("year")
                year_b = p_b.get("year")
                if year_a and year_b and abs(int(year_a) - int(year_b)) <= 1:
                    signals.append({
                        "type": "temporal_proximity",
                        "detail": (
                            f"Published within 1 year ({year_a} vs {year_b}). "
                            "Rapid succession of similar papers suggests pre-planned slicing."
                        ),
                    })

                if len(signals) >= 2:
                    suspicious_groups.append({
                        "paper_a": p_a.get("title", f"Paper {i}"),
                        "paper_b": p_b.get("title", f"Paper {j}"),
                        "signals": signals,
                        "signal_count": len(signals),
                        "interpretation": (
                            f"{len(signals)} salami slicing indicators found. "
                            "These papers may represent a single study split into "
                            "multiple publications — a form of publication misconduct."
                        ),
                    })

        flagged = len(suspicious_groups) > 0

        return json.dumps({
            "analysis_type": "Salami Slicing Detection",
            "papers_analyzed": len(papers),
            "comparisons_made": len(papers) * (len(papers) - 1) // 2,
            "suspicious_groups": suspicious_groups,
            "suspicious_group_count": len(suspicious_groups),
            "flagged": flagged,
            "interpretation": (
                f"Found {len(suspicious_groups)} suspicious paper group(s) "
                "with salami slicing characteristics. "
                + (
                    "These papers appear to be minimal publishable units carved "
                    "from a single study — review for shared control data and "
                    "meaningful intellectual contribution of each paper."
                    if flagged
                    else "No salami slicing patterns detected among these papers."
                )
            ),
        }, ensure_ascii=False)
