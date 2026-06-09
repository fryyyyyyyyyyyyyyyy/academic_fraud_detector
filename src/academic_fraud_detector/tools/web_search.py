"""
Web search tool for agents — enables real-time literature verification.

This tool gives agents the ability to search the web for academic papers,
verify citation existence, and check if cited papers actually support
the claims made about them.

Uses DuckDuckGo (no API key required) or falls back to direct HTTP requests.

Key use cases:
1. "Smith et al. (2020) showed that X" → search for Smith et al. 2020, verify it exists
2. Check if a cited paper's topic matches the claim being made
3. Find retraction notices or PubPeer comments about a paper
4. Verify if a reagent/antibody catalog number actually exists
"""

import json
import logging
import re
from typing import List, Optional
from urllib.parse import quote_plus

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# Academic Web Search Tool
# ═══════════════════════════════════════════════════════════════════════

class AcademicSearchInput(BaseModel):
    """Input for academic web search."""

    query: str = Field(
        ...,
        description=(
            "Search query for finding academic papers. Be specific: include "
            "paper title, author names, year, or DOI. "
            "Examples: 'Smith et al. 2020 RTA408 oxidative stress glaucoma', "
            "'10.1371/journal.pone.0313446', "
            "'Anti-β-actin antibody ab8227 Abcam'."
        ),
    )
    search_type: str = Field(
        default="general",
        description=(
            "Type of search: 'general' (DuckDuckGo web search), "
            "'scholar' (Google Scholar via serpapi-like), "
            "'paper' (search for specific paper existence)."
        ),
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results to return.",
    )


class AcademicWebSearchTool(BaseTool):
    """
    Search the web for academic papers, reagents, or scientific claims.

    This tool enables REAL-TIME verification that goes beyond the agent's
    training data cutoff:

    - **Citation verification**: Does "Smith et al. (2020)" actually exist?
    - **Reagent verification**: Does "Anti-X antibody (Cat# YZ1234, Vendor Z)" exist?
    - **Claim verification**: Find sources that support or contradict a scientific claim.
    - **Retraction checks**: Search for retraction notices or PubPeer discussions.
    - **Recent papers**: Find papers published after the model's training cutoff.

    The tool returns search results with titles, snippets, and URLs. The agent
    should then use its LLM reasoning to evaluate whether the search results
    support or contradict the claim being investigated.
    """

    name: str = "academic_web_search"
    description: str = (
        "Search the web in real-time to verify academic claims, citations, and reagents. "
        "CRITICAL: Use this tool whenever you need to verify if a cited paper exists, "
        "if a reagent/antibody catalog number is real, or if a scientific claim is "
        "supported by published literature. "
        "Input: query (search terms), search_type ('general'/'scholar'/'paper'), max_results."
        "Returns: list of search results with title, snippet, and URL for each."
    )
    args_schema: type[BaseModel] = AcademicSearchInput

    def _run(
        self,
        query: str,
        search_type: str = "general",
        max_results: int = 5,
    ) -> str:
        """Execute academic web search."""
        results = []

        # ── Try DuckDuckGo via duckduckgo_search library first ──
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(
                    f"{query} site:scholar.google.com OR site:pubmed.ncbi.nlm.nih.gov OR site:semanticscholar.org OR site:researchgate.net",
                    max_results=max_results,
                ))
                for r in ddg_results:
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", "")[:300],
                        "url": r.get("href", ""),
                        "source": "duckduckgo",
                    })
        except ImportError:
            logger.debug("duckduckgo_search not installed, trying requests fallback")
        except Exception as e:
            logger.debug(f"DuckDuckGo search failed: {e}")

        # ── Fallback: DuckDuckGo HTML ──
        if not results:
            try:
                results = self._search_ddg_html(query, max_results)
            except Exception as e:
                logger.warning(f"DDG HTML fallback failed: {e}")

        # ── Last resort: CrossRef for papers ──
        if not results and search_type in ("paper", "scholar"):
            try:
                crossref_results = self._search_crossref(query, max_results)
                if crossref_results:
                    results = crossref_results
            except Exception as e:
                logger.warning(f"CrossRef search failed: {e}")

        # ── Format response ──
        if not results:
            return json.dumps({
                "search_type": search_type,
                "query": query,
                "results": [],
                "total_found": 0,
                "flagged": False,
                "interpretation": (
                    f"No search results found for '{query}'. "
                    "This could mean the paper/reagent does not exist, "
                    "OR the search query needs refinement. Treat with caution."
                ),
            }, ensure_ascii=False)

        return json.dumps({
            "search_type": search_type,
            "query": query,
            "results": results,
            "total_found": len(results),
            "flagged": len(results) == 0,  # Zero results = potentially fake citation/reagent
            "interpretation": (
                f"Found {len(results)} search result(s) for '{query}'. "
                "Review the results — if searching for a specific paper and "
                "no results appear to match, the citation may be fabricated."
            ),
            "agent_guidance": (
                "YOU (the agent) should now use your LLM reasoning to evaluate: "
                "1. Do any of these results actually match the claimed paper/reagent? "
                "2. Is the topic/content of the found papers relevant to the citation claim? "
                "3. Does the existence/finding support or refute the claim being investigated? "
                "Report your reasoning clearly in your findings."
            ),
        }, ensure_ascii=False)

    def _search_ddg_html(self, query: str, max_results: int) -> list:
        """Fallback: search DuckDuckGo via HTML scraping."""
        url = "https://html.duckduckgo.com/html/"
        resp = requests.post(
            url,
            data={"q": query, "kl": "us-en"},
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()

        results = []
        # Parse the HTML results
        html = resp.text
        # Extract result blocks using regex
        result_blocks = re.findall(
            r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>.*?'
            r'<a class="result__snippet"[^>]*>([^<]+)</a>',
            html, re.DOTALL
        )

        for i, (url, title, snippet) in enumerate(result_blocks):
            if i >= max_results:
                break
            results.append({
                "title": title.strip(),
                "snippet": snippet.strip()[:300],
                "url": url.strip(),
                "source": "duckduckgo_html",
            })

        return results

    def _search_crossref(self, query: str, max_results: int) -> list:
        """Search CrossRef for papers matching the query."""
        crossref_url = "https://api.crossref.org/works"
        params = {
            "query": query,
            "rows": max_results,
            "sort": "relevance",
        }
        resp = requests.get(crossref_url, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("message", {}).get("items", [])[:max_results]:
            title = item.get("title", [""])[0] if item.get("title") else "Unknown"
            doi = item.get("DOI", "")
            year = item.get("created", {}).get("date-parts", [[None]])[0][0]
            publisher = item.get("publisher", "")

            results.append({
                "title": title,
                "snippet": f"DOI: {doi} | Published: {year} | Publisher: {publisher}",
                "url": f"https://doi.org/{doi}" if doi else "",
                "doi": doi,
                "year": year,
                "source": "crossref",
            })

        return results


# ═══════════════════════════════════════════════════════════════════════
# Citation Existence Checker
# ═══════════════════════════════════════════════════════════════════════

class CitationCheckInput(BaseModel):
    """Input for citation existence check."""

    citation_text: str = Field(
        ...,
        description=(
            "The full citation text to verify. Include as much info as available: "
            "authors, year, title keywords, DOI if known. "
            "Example: 'Smith, J., et al. (2020). RTA408 alleviates oxidative "
            "stress in retinal ganglion cells. Journal of Neuroscience, 40(15), 3120-3135.'"
        ),
    )
    claim_about_citation: Optional[str] = Field(
        default=None,
        description=(
            "What does the target paper CLAIM this cited paper says? "
            "Example: 'Previous studies demonstrated that RTA408 reduces ROS levels [15].' "
            "We will check if the cited paper actually supports this claim."
        ),
    )


class CitationExistenceCheckTool(BaseTool):
    """
    Verify that a cited reference actually exists and is relevant to the claim.

    This is the core of "citation farming" detection. Many fraudulent papers cite
    references that:
    1. Don't exist at all (fabricated citations)
    2. Exist but are about completely different topics (phantom support)
    3. Exist and are real, but actually contradict the claim (citation distortion)

    The tool:
    1. Searches the web for the cited paper using its title/authors/DOI
    2. Returns search results showing if the paper exists
    3. Provides snippets so the agent can evaluate topic relevance

    COMBINED WITH LLM REASONING: After getting search results, the agent should
    use its own knowledge and reasoning to:
    - Evaluate if the cited paper's topic matches the claim
    - Check if the claim is something that type of paper COULD support
    - Flag citations where the cited paper clearly doesn't support the claim
    """

    name: str = "citation_existence_check"
    description: str = (
        "Verify if a cited reference actually exists and is relevant. "
        "CRITICAL FOR FRAUD DETECTION: many papers cite non-existent references "
        "or misrepresent what cited papers actually say. "
        "Use this on SUSPICIOUS citations — ones that seem too convenient, "
        "have vague formatting, or support controversial claims. "
        "Input: citation_text (full citation to verify), "
        "claim_about_citation (what the target paper claims this citation supports). "
        "After getting results, use your LLM reasoning to judge relevance."
    )
    args_schema: type[BaseModel] = CitationCheckInput

    def _run(
        self,
        citation_text: str,
        claim_about_citation: Optional[str] = None,
    ) -> str:
        """Execute citation existence check."""
        import difflib

        # ── Extract searchable components from citation ──
        # Try to extract: authors, year, title keywords
        author_match = re.search(r'([A-Z][a-z]+(?:\s+(?:et\s+al\.?|[A-Z][a-z]+))?)', citation_text)
        year_match = re.search(r'\b(19|20)\d{2}\b', citation_text)
        doi_match = re.search(r'10\.\d{4,}/[^\s]+', citation_text)

        # Extract significant words for the search query
        # Remove common citation noise words
        noise_words = {"the", "and", "for", "that", "with", "from", "this", "was", "are", "has", "had", "were", "been"}
        title_words = re.findall(r'\b[a-zA-Z]{4,}\b', citation_text)
        keywords = [w for w in title_words if w.lower() not in noise_words][:10]

        # ── Build search queries ──
        queries = []

        # Most specific: DOI lookup
        if doi_match:
            queries.append(doi_match.group(0))

        # Author + year + key terms
        if author_match and year_match:
            queries.append(
                f"{author_match.group(0)} {year_match.group(0)} {' '.join(keywords[:4])}"
            )

        # Title-like search
        queries.append(' '.join(keywords[:6]))

        # ── Execute searches ──
        all_results = []
        search_errors = []

        for q in queries[:2]:  # Try at most 2 queries to avoid rate limits
            try:
                # Use CrossRef first (most reliable for academic papers)
                crossref_url = "https://api.crossref.org/works"
                params = {"query": q, "rows": 3, "sort": "relevance"}
                resp = requests.get(
                    crossref_url, params=params, headers=HEADERS, timeout=15
                )
                if resp.status_code == 200:
                    items = resp.json().get("message", {}).get("items", [])
                    for item in items:
                        paper_title = item.get("title", [""])[0] if item.get("title") else ""
                        paper_doi = item.get("DOI", "")
                        paper_year = item.get("created", {}).get("date-parts", [[None]])[0][0]
                        paper_authors = [
                            f"{a.get('given','')} {a.get('family','')}"
                            for a in item.get("author", [])[:3]
                        ]

                        # Compute relevance to the citation text
                        relevance = difflib.SequenceMatcher(
                            None,
                            citation_text.lower()[:200],
                            (paper_title + ' '.join(paper_authors)).lower()
                        ).ratio()

                        all_results.append({
                            "title": paper_title,
                            "doi": paper_doi,
                            "year": paper_year,
                            "authors_sample": paper_authors,
                            "relevance_to_citation": round(relevance, 3),
                            "source": "crossref",
                        })
            except Exception as e:
                search_errors.append(f"CrossRef search '{q[:50]}...': {e}")

        # ── If claim provided, evaluate claim-citation alignment ──
        claim_assessment = None
        if claim_about_citation and all_results:
            best_match = all_results[0]  # Most relevant result
            best_title = best_match.get("title", "")

            # Keyword overlap between claim and found paper title
            claim_words = set(re.findall(r'\b[a-z]{4,}\b', claim_about_citation.lower()))
            title_words_set = set(re.findall(r'\b[a-z]{4,}\b', best_title.lower()))
            overlap = claim_words & title_words_set
            overlap_ratio = len(overlap) / max(len(claim_words), 1)

            claim_assessment = {
                "claim": claim_about_citation[:200],
                "best_match_title": best_title,
                "keyword_overlap_ratio": round(overlap_ratio, 3),
                "assessment": (
                    "LIKELY_SUPPORTED" if overlap_ratio > 0.20
                    else "POSSIBLY_SUPPORTED" if overlap_ratio > 0.10
                    else "LIKELY_UNSUPPORTED" if overlap_ratio > 0.05
                    else "VERY_UNLIKELY_SUPPORTED"
                ),
                "agent_guidance": (
                    "Based on keyword overlap, evaluate whether the cited paper's "
                    "title and topic actually match what the citing paper claims. "
                    "Low overlap (<0.10) strongly suggests the citation does NOT "
                    "support the claim — this is a 'phantom citation' red flag. "
                    "Use your LLM knowledge of the research area to make a final judgment."
                ),
            }

        # ── Overall assessment ──
        paper_found = len(all_results) > 0
        highly_relevant = any(
            r.get("relevance_to_citation", 0) > 0.5 for r in all_results
        )

        if not paper_found:
            severity = "严重 — 引用文献可能不存在"
        elif not highly_relevant:
            severity = "高 — 引用文献存在但与引用内容关系不大"
        elif claim_assessment and claim_assessment["assessment"] in (
            "LIKELY_UNSUPPORTED", "VERY_UNLIKELY_SUPPORTED"
        ):
            severity = "中 — 引用文献存在但可能不支持所声称的观点"
        else:
            severity = "低 — 引用文献存在且与引用内容相关"

        return json.dumps({
            "analysis_type": "Citation Existence Check",
            "citation_text": citation_text[:300],
            "claim_about_citation": claim_about_citation[:200] if claim_about_citation else None,
            "paper_found": paper_found,
            "results_count": len(all_results),
            "top_results": all_results[:5],
            "claim_assessment": claim_assessment,
            "search_errors": search_errors[:3],
            "severity": severity,
            "flagged": not paper_found or not highly_relevant,
            "interpretation": (
                f"{'FOUND' if paper_found else 'NOT FOUND'}: {severity}. "
                + (
                    "The cited paper could not be verified online — this is a strong "
                    "indicator of citation fabrication. The agent should flag this "
                    "citation as potentially fraudulent."
                    if not paper_found
                    else "The cited paper exists. Now use LLM reasoning to assess "
                    "whether it actually supports the specific claim being made."
                )
            ),
            "agent_llm_guidance": (
                "⚠️ IMPORTANT — YOU MUST NOW USE YOUR LLM REASONING CAPABILITIES: "
                "1. Look at the found paper titles above. Do they match the citation's topic? "
                "2. Based on your training knowledge of this research area, does the "
                "   citation claim make sense? Would this type of paper support this claim? "
                "3. If the paper was NOT found via CrossRef (the most comprehensive "
                "   academic database), it likely DOES NOT EXIST — flag as fabricated citation. "
                "4. If the paper exists but has low keyword overlap with the claim, "
                "   flag as 'citation distortion' — citing a paper that doesn't "
                "   actually support the claim."
            ),
        }, ensure_ascii=False)
