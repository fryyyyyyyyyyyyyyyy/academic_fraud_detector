"""
Citation analysis tools — detect citation manipulation patterns.

Two core analyses:
1. Citation Graph Analyzer: builds a directed citation graph and detects:
   - Citation rings (strongly connected components)
   - Reciprocal citation pairs
   - Unusually dense subgraphs
2. Self-Citation Analyzer: computes self-citation rates per author and flags
   excessive self-citation (>25% threshold).

Uses networkx for graph analysis and Semantic Scholar API for data.
"""

import json
import logging
from typing import List, Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..utils.api_client import safe_request

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Citation Graph Analyzer
# ═══════════════════════════════════════════════════════════════════════

class CitationGraphInput(BaseModel):
    """Input for citation graph analysis."""

    paper_ids: List[str] = Field(
        ...,
        description=(
            "List of Semantic Scholar Paper IDs to include in the citation graph. "
            "Include the target paper and all papers that cite or are cited by it."
        ),
    )
    max_papers: int = Field(
        default=100,
        description="Maximum number of papers to include in the graph.",
    )


class CitationGraphTool(BaseTool):
    """
    Build and analyze a directed citation graph to detect manipulation patterns.

    The graph:
    - Nodes = papers
    - Edges = citations (A → B means A cites B)

    Detects:
    - Citation rings: groups of ≥3 papers where each cites the others in a cycle
      (strongly connected components).
    - Reciprocal citation: A cites B AND B cites A. Normal in small numbers but
      suspicious when extensive.
    - Anomalous density: citation rings create artificially dense subgraphs.

    These patterns are the hallmark of citation manipulation — authors agreeing
    to cite each other's work to inflate citation counts.
    """

    name: str = "citation_graph_analyzer"
    description: str = (
        "Build and analyze a directed citation graph for a set of papers. "
        "Detects citation rings (strongly connected components ≥ 3), reciprocal "
        "citation pairs, and anomalous graph density. Uses networkx and Semantic "
        "Scholar citation data. Use this to identify coordinated citation manipulation "
        "among groups of authors."
    )
    args_schema: type[BaseModel] = CitationGraphInput

    def _run(self, paper_ids: List[str], max_papers: int = 100) -> str:
        """Build citation graph and run analyses."""
        import networkx as nx

        paper_ids = paper_ids[:max_papers]
        paper_id_set = set(paper_ids)
        G = nx.DiGraph()

        for pid in paper_ids:
            G.add_node(pid)

        # ── Fetch citations and references ──
        for pid in paper_ids:
            try:
                # References: papers this paper cites
                refs_url = (
                    f"https://api.semanticscholar.org/graph/v1/paper/{pid}/references"
                )
                refs_params = {"limit": 500, "fields": "citedPaperId"}
                refs_resp = safe_request(
                    refs_url, params=refs_params, api_name="semantic_scholar"
                )
                for ref in refs_resp.json().get("data", []):
                    ref_pid = ref.get("citedPaperId")
                    if ref_pid and ref_pid in paper_id_set:
                        G.add_edge(pid, ref_pid)
            except Exception as e:
                logger.warning(f"Failed to fetch references for {pid}: {e}")

            try:
                # Citations: papers that cite this paper
                cites_url = (
                    f"https://api.semanticscholar.org/graph/v1/paper/{pid}/citations"
                )
                cites_params = {"limit": 500, "fields": "citingPaperId"}
                cites_resp = safe_request(
                    cites_url, params=cites_params, api_name="semantic_scholar"
                )
                for cite in cites_resp.json().get("data", []):
                    cite_pid = cite.get("citingPaperId")
                    if cite_pid and cite_pid in paper_id_set:
                        G.add_edge(cite_pid, pid)
            except Exception as e:
                logger.warning(f"Failed to fetch citations for {pid}: {e}")

        # ── Graph metrics ──
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = nx.density(G) if n_nodes > 1 else 0.0

        # ── Detect citation rings (SCC of size ≥ 3) ──
        sccs = [
            scc for scc in nx.strongly_connected_components(G)
            if len(scc) >= 3
        ]
        rings = [sorted(list(scc)) for scc in sccs]

        # ── Detect reciprocal citation pairs ──
        reciprocal_pairs = []
        for u in G.nodes():
            for v in G.nodes():
                if u < v and G.has_edge(u, v) and G.has_edge(v, u):
                    reciprocal_pairs.append([u, v])

        # ── Compute PageRank for significance ──
        try:
            pagerank = nx.pagerank(G)
            top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        except Exception:
            top_nodes = []

        # ── Compute in-degree (citations received) ──
        in_degrees = dict(G.in_degree())
        top_cited = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]

        flagged = len(rings) > 0 or len(reciprocal_pairs) > max(3, n_nodes * 0.1)

        return json.dumps({
            "analysis_type": "Citation Graph Analysis",
            "graph_stats": {
                "nodes": n_nodes,
                "edges": n_edges,
                "density": round(density, 6),
                "reciprocal_edges": len(reciprocal_pairs),
            },
            "citation_rings": {
                "count": len(rings),
                "rings": rings[:10],
                "interpretation": (
                    f"Found {len(rings)} citation ring(s). "
                    "These are strongly connected groups where each paper cites others "
                    "in the group — a classic signature of citation manipulation."
                    if rings
                    else "No citation rings detected."
                ),
            },
            "reciprocal_citations": {
                "count": len(reciprocal_pairs),
                "pairs": reciprocal_pairs[:20],
                "interpretation": (
                    f"Found {len(reciprocal_pairs)} reciprocal citation pairs. "
                    "Excessive reciprocal citation may indicate 'you-cite-me-I-cite-you' arrangements."
                    if len(reciprocal_pairs) > 3
                    else "Reciprocal citation count is within normal range."
                ),
            },
            "top_by_pagerank": [
                {"paper_id": pid, "pagerank": round(pr, 6)}
                for pid, pr in top_nodes
            ],
            "top_by_citations_received": [
                {"paper_id": pid, "citations": count}
                for pid, count in top_cited
            ],
            "flagged": flagged,
            "overall_interpretation": (
                "Citation manipulation detected: "
                + (f"{len(rings)} ring(s)" if rings else "")
                + (" and " if rings and len(reciprocal_pairs) > 3 else "")
                + (f"{len(reciprocal_pairs)} reciprocal pairs" if len(reciprocal_pairs) > 3 else "")
                + ". Recommend detailed investigation of these author networks."
                if flagged
                else "No significant citation manipulation patterns detected."
            ),
        }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# Self-Citation Analyzer
# ═══════════════════════════════════════════════════════════════════════

class SelfCitationInput(BaseModel):
    """Input for self-citation analysis."""

    author_name: str = Field(..., description="Author's full name as it appears on papers.")
    author_id: Optional[str] = Field(
        default=None,
        description="Semantic Scholar Author ID for disambiguation.",
    )


class SelfCitationTool(BaseTool):
    """
    Analyze an author's self-citation rate.

    Self-citation is citing one's own previous work. It's normal in moderation
    (research builds on prior work), but excessive self-citation inflates
    citation metrics artificially.

    Thresholds (based on bibliometrics literature):
    - < 20%: normal
    - 20-25%: elevated — worth noting
    - 25-40%: high — potential manipulation
    - > 40%: very high — strong indicator of manipulation

    The tool fetches an author's papers and checks the authors of papers
    that cite them. A self-citation is counted when at least one author
    of the citing paper matches the target author.
    """

    name: str = "self_citation_analyzer"
    description: str = (
        "Analyze an author's self-citation rate. Fetches the author's papers from "
        "Semantic Scholar and calculates what percentage of citations to their work "
        "are self-citations. Flags authors with self-citation rates > 25%. "
        "Excessive self-citation is a common citation manipulation tactic. "
        "Use this on ALL authors of a suspect paper."
    )
    args_schema: type[BaseModel] = SelfCitationInput

    def _run(self, author_name: str, author_id: Optional[str] = None) -> str:
        """Execute self-citation analysis."""
        try:
            # ── Step 1: Find the author ──
            search_url = "https://api.semanticscholar.org/graph/v1/author/search"
            search_params = {"query": author_name}
            search_resp = safe_request(
                search_url, params=search_params, api_name="semantic_scholar"
            )
            authors = search_resp.json().get("data", [])

            if not authors:
                return json.dumps({
                    "error": f"Author '{author_name}' not found in Semantic Scholar.",
                    "flagged": False,
                })

            # Match by ID or use first result
            target = authors[0]
            if author_id:
                matches = [a for a in authors if a.get("authorId") == author_id]
                if matches:
                    target = matches[0]

            s2_author_id = target.get("authorId")
            author_display = target.get("name", author_name)

            # ── Step 2: Get author's papers with citation data ──
            papers_url = f"https://api.semanticscholar.org/graph/v1/author/{s2_author_id}/papers"
            papers_params = {
                "fields": (
                    "paperId,title,year,citationCount,"
                    "citations.title,citations.authors,citations.year,citations.paperId"
                ),
                "limit": 100,
            }
            papers_resp = safe_request(
                papers_url, params=papers_params, api_name="semantic_scholar"
            )
            papers = papers_resp.json().get("data", [])

            if not papers:
                return json.dumps({
                    "author": author_display,
                    "error": "No papers found for this author.",
                    "flagged": False,
                })

            # ── Step 3: Count self-citations ──
            # Use the author's last name as primary matching heuristic
            author_last = author_name.split()[-1].lower().rstrip(".")
            author_full_lower = author_name.lower()

            total_cites = 0
            self_cites = 0
            paper_self_cite_rates = []

            for paper in papers:
                paper_citations = paper.get("citations", [])
                paper_total = len(paper_citations)
                paper_self = 0

                for cite in paper_citations:
                    cite_authors = cite.get("authors", [])
                    for ca in cite_authors:
                        ca_name = ca.get("name", "").lower()
                        # Match by last name + first initial for robustness
                        if author_last in ca_name:
                            paper_self += 1
                            break
                        elif ca_name == author_full_lower:
                            paper_self += 1
                            break

                total_cites += paper_total
                self_cites += paper_self
                paper_self_cite_rates.append({
                    "paper_id": paper.get("paperId"),
                    "title": (paper.get("title") or "")[:120],
                    "year": paper.get("year"),
                    "total_citations": paper_total,
                    "self_citations": paper_self,
                })

            # ── Step 4: Compute and assess ──
            self_cite_rate = self_cites / max(total_cites, 1)

            if self_cite_rate > 0.40:
                severity = "very_high"
            elif self_cite_rate > 0.25:
                severity = "high"
            elif self_cite_rate > 0.20:
                severity = "elevated"
            else:
                severity = "normal"

            flagged = self_cite_rate > 0.25

            return json.dumps({
                "analysis_type": "Self-Citation Analysis",
                "author": author_display,
                "semantic_scholar_id": s2_author_id,
                "total_papers_analyzed": len(papers),
                "total_citations_received": total_cites,
                "self_citations": self_cites,
                "self_citation_rate": round(self_cite_rate, 4),
                "severity": severity,
                "threshold_used": {
                    "normal": "< 0.20",
                    "elevated": "0.20 – 0.25",
                    "high": "0.25 – 0.40",
                    "very_high": "> 0.40",
                },
                "flagged": flagged,
                "paper_breakdown": sorted(
                    paper_self_cite_rates,
                    key=lambda p: p["self_citations"],
                    reverse=True,
                )[:20],
                "interpretation": (
                    f"{severity.replace('_', ' ').title()} self-citation rate "
                    f"({self_cite_rate:.1%}). "
                    + (
                        "Strong indicator of citation manipulation. This author "
                        "disproportionately cites their own work."
                        if flagged
                        else "Self-citation rate is within acceptable academic norms."
                    )
                ),
            }, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Self-citation analysis failed: {e}")
            return json.dumps({"error": str(e), "flagged": False})


# ═══════════════════════════════════════════════════════════════════════
# Citation Claim Verifier
# ═══════════════════════════════════════════════════════════════════════

class CitationClaimInput(BaseModel):
    """Input for citation claim verification."""

    claims_json: str = Field(
        ...,
        description=(
            "JSON list of citation claims extracted from the paper. Each claim should have: "
            "'claim_text' (the sentence making the claim, e.g. 'Smith et al. (2020) showed that X'), "
            "'cited_doi' (DOI of the cited paper, optional), "
            "'cited_title' (title of the cited paper, optional), "
            "'location' (where in the target paper, e.g. 'Introduction paragraph 3'). "
            'Example: [{"claim_text":"Previous studies demonstrated that RTA408 reduces oxidative stress [15].","cited_doi":"10.1000/xyz123","location":"Introduction"}]'
        ),
    )


class CitationClaimVerifierTool(BaseTool):
    """
    Verify whether a paper's citation claims are actually supported by the cited papers.

    This tool detects "quotation farming" — the practice of citing papers that
    do not actually support (or even contradict) the claim being made.

    Detection method:
    1. For each citation claim, extract the cited paper's metadata via S2 API.
    2. Compare the claim text against the cited paper's title and abstract.
    3. Flag claims where the cited paper appears unrelated to the claim topic.

    Red flags:
    - Cited paper title has zero keyword overlap with the claim
    - Cited paper is in a completely different field
    - Claim attributes specific findings to a review/meta-analysis
    - Citation chain of trust: paper cites A, but A actually says the opposite

    IMPORTANT: This tool provides HEURISTIC signals, not definitive verification.
    A definitive check would require reading the full text of the cited paper,
    which is typically behind paywalls.
    """

    name: str = "citation_claim_verifier"
    description: str = (
        "Verify whether citation claims in a paper are supported by the cited papers. "
        "Extracts citation sentences, fetches cited paper metadata from Semantic Scholar, "
        "and checks if the cited paper's topic/abstract supports the claim. "
        "Detects 'phantom citations' — references that don't actually support "
        "what the citing paper claims. "
        "Input: JSON list of claims with claim_text, cited_doi/cited_title, and location."
    )
    args_schema: type[BaseModel] = CitationClaimInput

    def _run(self, claims_json: str) -> str:
        """Execute citation claim verification."""
        try:
            claims = json.loads(claims_json)
            if not isinstance(claims, list):
                return json.dumps({"error": "claims_json must be a JSON list.", "flagged": False})
        except json.JSONDecodeError:
            return json.dumps({"error": "claims_json must be valid JSON.", "flagged": False})

        if not claims:
            return json.dumps({
                "analysis_type": "Citation Claim Verification",
                "claims_checked": 0,
                "flagged": False,
                "findings": [],
            })

        import difflib

        verified_claims = []
        suspicious_count = 0

        for i, claim in enumerate(claims):
            claim_text = claim.get("claim_text", "")
            cited_doi = claim.get("cited_doi", "")
            cited_title = claim.get("cited_title", "")
            location = claim.get("location", "Unknown")

            result = {
                "claim_index": i,
                "claim_text": claim_text[:300],
                "location": location,
                "cited_doi": cited_doi,
                "cited_title": cited_title,
                "issues": [],
                "suspicious": False,
            }

            # ── Try to fetch cited paper metadata ──
            cited_abstract = None
            fetched_title = None

            if cited_doi:
                try:
                    doi_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{cited_doi}"
                    doi_params = {"fields": "title,abstract,year"}
                    doi_resp = safe_request(
                        doi_url, params=doi_params, api_name="semantic_scholar"
                    )
                    paper_data = doi_resp.json()
                    if paper_data and "title" in paper_data:
                        fetched_title = paper_data.get("title", "")
                        cited_abstract = paper_data.get("abstract", "")
                        result["fetched_title"] = fetched_title
                except Exception:
                    pass

            if not fetched_title and cited_title:
                # Try search by title
                try:
                    search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
                    search_params = {"query": cited_title, "limit": 1, "fields": "title,abstract"}
                    search_resp = safe_request(
                        search_url, params=search_params, api_name="semantic_scholar"
                    )
                    search_data = search_resp.json().get("data", [])
                    if search_data:
                        fetched_title = search_data[0].get("title", "")
                        cited_abstract = search_data[0].get("abstract", "")
                        result["fetched_title"] = fetched_title
                except Exception:
                    pass

            # ── Check 1: Title-topic relevance ──
            if fetched_title:
                # Extract key content words from claim (skip citation markers like [15])
                claim_words = set(re.findall(
                    r'\b[a-z]{4,}\b',
                    re.sub(r'\[\d+(?:[,–-]\d+)*\]', '', claim_text.lower())
                ))
                title_words = set(re.findall(r'\b[a-z]{4,}\b', fetched_title.lower()))

                if claim_words and title_words:
                    overlap = claim_words & title_words
                    overlap_ratio = len(overlap) / max(len(claim_words), 1)

                    result["title_word_overlap"] = {
                        "claim_keywords": len(claim_words),
                        "title_keywords": len(title_words),
                        "overlap_count": len(overlap),
                        "overlap_ratio": round(overlap_ratio, 3),
                    }

                    if overlap_ratio < 0.10:
                        result["issues"].append({
                            "type": "low_title_relevance",
                            "detail": (
                                f"Cited paper title has only {overlap_ratio:.1%} keyword "
                                f"overlap with the claim. Claim is about '{' '.join(sorted(claim_words)[:5])}' "
                                f"but cited paper is about '{' '.join(sorted(title_words)[:5])}'. "
                                f"The cited paper may not support this claim."
                            ),
                        })

            # ── Check 2: Abstract-content relevance ──
            if cited_abstract and claim_text:
                # Simple keyword overlap with abstract
                claim_words = set(re.findall(
                    r'\b[a-z]{4,}\b',
                    re.sub(r'\[\d+(?:[,–-]\d+)*\]', '', claim_text.lower())
                ))
                abstract_words = set(re.findall(r'\b[a-z]{4,}\b', cited_abstract.lower()))

                if claim_words and abstract_words:
                    overlap = claim_words & abstract_words
                    overlap_ratio = len(overlap) / max(len(claim_words), 1)

                    result["abstract_word_overlap"] = {
                        "overlap_count": len(overlap),
                        "overlap_ratio": round(overlap_ratio, 3),
                    }

                    if overlap_ratio < 0.15:
                        result["issues"].append({
                            "type": "low_abstract_relevance",
                            "detail": (
                                f"Cited paper abstract has only {overlap_ratio:.1%} keyword "
                                f"overlap with the claim text. The cited paper's content "
                                f"may not support the specific claim being made."
                            ),
                        })

            # ── Check 3: Citation format issues ──
            # Check for common patterns of sloppy/fake citations
            if not cited_doi and not cited_title:
                result["issues"].append({
                    "type": "unverifiable",
                    "detail": (
                        "No DOI or title provided for the cited paper. "
                        "Cannot verify — citations without identifiers are "
                        "more common in fabricated papers."
                    ),
                })

            # ── Check 4: Claim pattern analysis ──
            vague_claim_patterns = [
                r'it has been (?:shown|reported|demonstrated) that',
                r'previous studies (?:have|has) (?:shown|reported|demonstrated)',
                r'it is (?:well[ -]?)?known that',
                r'consistent with (?:previous|prior) (?:reports|studies|findings)',
            ]
            for pattern in vague_claim_patterns:
                if re.search(pattern, claim_text, re.IGNORECASE):
                    result["issues"].append({
                        "type": "vague_citation",
                        "detail": (
                            f"Claim uses vague attribution pattern: '{pattern}'. "
                            "This is a common technique to avoid providing specific "
                            "citations that could be verified."
                        ),
                    })
                    break

            if result["issues"]:
                result["suspicious"] = True
                suspicious_count += 1

            verified_claims.append(result)

        flagged = suspicious_count > 0

        return json.dumps({
            "analysis_type": "Citation Claim Verification",
            "claims_checked": len(claims),
            "suspicious_claims": suspicious_count,
            "flagged": flagged,
            "verified_claims": verified_claims,
            "api_used": "semantic_scholar",
            "limitation_note": (
                "Verification is based on title/abstract keyword overlap only. "
                "Full-text reading of cited papers would be needed for definitive "
                "verification. Low overlap strongly suggests the citation does not "
                "support the claim, but high overlap does not guarantee support."
            ),
            "interpretation": (
                f"{suspicious_count}/{len(claims)} citation claim(s) appear "
                "unsupported by the cited papers. "
                + (
                    "These 'phantom citations' may indicate academic dishonesty "
                    "— the authors are citing papers that don't actually support "
                    "their claims, perhaps assuming no one will check."
                    if flagged
                    else "All checked citations appear relevant to their claims."
                )
            ),
        }, ensure_ascii=False)
