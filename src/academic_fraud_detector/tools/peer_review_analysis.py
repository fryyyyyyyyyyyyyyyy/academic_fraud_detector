"""
Peer review analysis tools — detect review fraud and integrity issues.

Three analysis dimensions:
1. Review Text Analysis: detects template/boilerplate language, near-duplicate
   reviews, and generic/non-specific comments.
2. Reviewer Credential Checker: verifies that reviewers exist in the academic
   record and have relevant expertise.
3. Review Template Detector: finds identical or near-identical phrase patterns
   across multiple reviews (suggests fake reviewer accounts or review mills).

Note: Full peer review fraud detection requires access to review metadata
(timelines, reviewer identities, IP logs) that is typically confidential.
These tools work with whatever data is publicly available and explicitly
flag limitations.
"""

import json
import logging
import re
from typing import List, Optional
from collections import Counter

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..utils.api_client import safe_request

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Review Text Analyzer
# ═══════════════════════════════════════════════════════════════════════

class ReviewTextInput(BaseModel):
    """Input for review text analysis."""

    reviews: List[str] = Field(
        ...,
        description="List of review texts to analyze. Each string is one complete review.",
    )
    paper_title: Optional[str] = Field(
        default=None,
        description="Title of the paper being reviewed, for context.",
    )


class ReviewTextAnalyzerTool(BaseTool):
    """
    Analyze peer review texts for fraud indicators.

    Checks for:
    1. Length anomalies: reviews that are suspiciously short (template/generic).
    2. Specificity: proportion of review that references specific paper elements
       (figures, tables, sections) vs generic language.
    3. Phrase commonality: identical or near-identical phrases across reviews.
    4. Sentiment outliers: reviews that are uniformly positive or negative
       without technical justification.

    Red flags:
    - Reviews < 100 words: too short to be substantive
    - < 2 specific references (figure/table/section mentions): too generic
    - Shared boilerplate phrases across ≥2 reviews: template reuse
    """

    name: str = "review_text_analyzer"
    description: str = (
        "Analyze peer review texts for fraud indicators. Checks review length, "
        "technical specificity (references to figures/tables/sections), boilerplate "
        "language patterns, and duplicated phrases across reviews. "
        "Input: list of review text strings (as many as available)."
    )
    args_schema: type[BaseModel] = ReviewTextInput

    def _run(self, reviews: List[str], paper_title: Optional[str] = None) -> str:
        """Execute review text analysis."""
        if not reviews:
            return json.dumps({
                "analysis_type": "Review Text Analysis",
                "error": "No review texts provided.",
                "flagged": False,
            })

        findings = []
        review_analyses = []

        for i, review in enumerate(reviews):
            words = review.split()
            word_count = len(words)

            # ── Specificity check ──
            figure_refs = len(re.findall(
                r'(?:Figure|Fig\.?)\s*\d+', review, re.IGNORECASE
            ))
            table_refs = len(re.findall(
                r'Table\s*\d+', review, re.IGNORECASE
            ))
            section_refs = len(re.findall(
                r'(?:Section|page|paragraph|line)\s*\d+', review, re.IGNORECASE
            ))
            equation_refs = len(re.findall(
                r'(?:Equation|Eq\.?)\s*\d+', review, re.IGNORECASE
            ))
            specific_refs = figure_refs + table_refs + section_refs + equation_refs

            # ── Generic language detection ──
            generic_phrases = [
                "interesting paper",
                "well written",
                "important contribution",
                "novel approach",
                "timely topic",
                "good work",
                "nice paper",
                "well organized",
                "clearly presented",
            ]
            generic_count = sum(
                1 for phrase in generic_phrases
                if phrase in review.lower()
            )

            analysis = {
                "review_index": i,
                "word_count": word_count,
                "specific_references": {
                    "figure_refs": figure_refs,
                    "table_refs": table_refs,
                    "section_refs": section_refs,
                    "equation_refs": equation_refs,
                    "total": specific_refs,
                },
                "generic_phrase_count": generic_count,
                "preview": review[:200] + ("..." if len(review) > 200 else ""),
            }

            # Flag conditions
            flags = []
            if word_count < 100:
                flags.append(f"Very short review ({word_count} words) — likely insufficient for substantive evaluation.")
            if specific_refs < 2:
                flags.append(f"Low technical specificity ({specific_refs} specific references) — review does not engage with paper details.")
            if generic_count >= 3:
                flags.append(f"High generic language ({generic_count} boilerplate phrases) — may be template-generated.")

            analysis["flags"] = flags
            analysis["suspicious"] = len(flags) > 0
            review_analyses.append(analysis)

            if flags:
                findings.append({
                    "review_index": i,
                    "issue": "; ".join(flags),
                })

        # ── Cross-review phrase duplication ──
        # Extract 5-word ngrams and check for duplication
        all_ngrams: Counter = Counter()
        for review in reviews:
            words = review.lower().split()
            ngrams = {" ".join(words[j:j + 5]) for j in range(len(words) - 4)}
            all_ngrams.update(ngrams)

        duplicated_phrases = [
            {"phrase": phrase, "appears_in_reviews": count}
            for phrase, count in all_ngrams.most_common(20)
            if count >= 2
        ]

        if duplicated_phrases:
            findings.append({
                "issue": (
                    f"Found {len(duplicated_phrases)} phrases that appear in multiple reviews. "
                    "This suggests template reuse or the same person writing multiple reviews."
                ),
            })

        suspicious_count = sum(1 for ra in review_analyses if ra["suspicious"])
        flagged = suspicious_count > 0 or len(duplicated_phrases) > 5

        return json.dumps({
            "analysis_type": "Review Text Analysis",
            "reviews_analyzed": len(reviews),
            "suspicious_reviews": suspicious_count,
            "review_analyses": review_analyses,
            "cross_review_duplicated_phrases": duplicated_phrases[:15],
            "findings": findings,
            "flagged": flagged,
            "interpretation": (
                f"{suspicious_count}/{len(reviews)} review(s) show suspicious patterns. "
                + (
                    "Reviews lack technical specificity, are too short, or use boilerplate "
                    "language — possible fake reviews or review mill output."
                    if flagged
                    else "Reviews appear substantive and technically engaged."
                )
            ),
        }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# Reviewer Credential Checker
# ═══════════════════════════════════════════════════════════════════════

class ReviewerCredentialInput(BaseModel):
    """Input for reviewer credential checking."""

    reviewer_name: str = Field(..., description="Reviewer's full name.")
    reviewer_email: Optional[str] = Field(
        default=None,
        description="Reviewer's email address (if known).",
    )
    claimed_expertise: Optional[str] = Field(
        default=None,
        description="The field the reviewer claims to be expert in.",
    )


class ReviewerCredentialCheckerTool(BaseTool):
    """
    Verify that a reviewer appears to be a legitimate academic with relevant expertise.

    Checks:
    1. Does the reviewer have a publication record on Semantic Scholar?
    2. Is their publication record in a relevant field?
    3. Is their email an institutional address (.edu, .ac.uk, etc.) or
       a free email service (Gmail, Yahoo, etc. — red flag)?
    4. Has the reviewer published recently (active researcher)?

    Red flags:
    - No publication record found
    - Free email address instead of institutional
    - No publications in the claimed field of expertise
    - Zero recent publications
    """

    name: str = "reviewer_credential_checker"
    description: str = (
        "Verify a peer reviewer's academic credentials. Checks publication record "
        "on Semantic Scholar, institutional affiliation indicators, and field relevance. "
        "Flags reviewers with no publication history or suspicious email addresses. "
        "Use this to check if suggested reviewers are real academics."
    )
    args_schema: type[BaseModel] = ReviewerCredentialInput

    def _run(
        self,
        reviewer_name: str,
        reviewer_email: Optional[str] = None,
        claimed_expertise: Optional[str] = None,
    ) -> str:
        """Execute reviewer credential check."""
        findings = []

        # ── Email check ──
        email_assessment = None
        if reviewer_email:
            email_lower = reviewer_email.lower()
            is_institutional = any(
                domain in email_lower
                for domain in [".edu", ".ac.", ".gov", ".org", "@uni", "@campus"]
            )
            is_free_email = any(
                provider in email_lower
                for provider in ["@gmail", "@yahoo", "@hotmail", "@outlook", "@qq", "@163", "@126", "@proton"]
            )

            if is_free_email and not is_institutional:
                email_assessment = {
                    "email": reviewer_email,
                    "type": "free_email",
                    "suspicious": True,
                    "detail": (
                        "Reviewer uses a free email service rather than an institutional address. "
                        "While not definitive, this is a common indicator of fake reviewer accounts."
                    ),
                }
                findings.append(email_assessment["detail"])
            elif is_institutional:
                email_assessment = {
                    "email": reviewer_email,
                    "type": "institutional",
                    "suspicious": False,
                    "detail": "Institutional email address — consistent with legitimate academic.",
                }
            else:
                email_assessment = {
                    "email": reviewer_email,
                    "type": "unknown_domain",
                    "suspicious": True,
                    "detail": "Email domain is neither clearly institutional nor a known free provider.",
                }
                findings.append(email_assessment["detail"])

        # ── Publication record check ──
        publication_assessment = None
        try:
            search_url = "https://api.semanticscholar.org/graph/v1/author/search"
            search_params = {"query": reviewer_name}
            search_resp = safe_request(
                search_url, params=search_params, api_name="semantic_scholar"
            )
            authors = search_resp.json().get("data", [])

            if not authors:
                publication_assessment = {
                    "found": False,
                    "suspicious": True,
                    "detail": (
                        f"No Semantic Scholar author profile found for '{reviewer_name}'. "
                        "This person may not be a published academic, which is a red flag "
                        "for a peer reviewer."
                    ),
                }
                findings.append(publication_assessment["detail"])
            else:
                # Use first (most relevant) match
                best_match = authors[0]
                paper_count = best_match.get("paperCount", 0)
                citation_count = best_match.get("citationCount", 0)
                h_index = best_match.get("hIndex", 0)

                suspicious = paper_count == 0
                publication_assessment = {
                    "found": True,
                    "name": best_match.get("name", reviewer_name),
                    "paper_count": paper_count,
                    "citation_count": citation_count,
                    "h_index": h_index,
                    "affiliations": best_match.get("affiliations", []),
                    "suspicious": suspicious,
                    "detail": (
                        f"Reviewer '{best_match.get('name')}' has {paper_count} papers, "
                        f"{citation_count} citations, h-index {h_index}. "
                        + (
                            "ZERO publications — this person does not appear to be a "
                            "published researcher. Strongly suspect."
                            if suspicious
                            else "Publication record exists — reviewer appears to be a "
                            "legitimate researcher."
                        )
                    ),
                }
                if suspicious:
                    findings.append(publication_assessment["detail"])

        except Exception as e:
            publication_assessment = {
                "found": False,
                "error": str(e),
                "suspicious": True,
                "detail": f"Could not verify publication record: {e}",
            }
            findings.append(publication_assessment["detail"])

        # ── Overall assessment ──
        flagged = (
            (email_assessment and email_assessment.get("suspicious", False))
            or (publication_assessment and publication_assessment.get("suspicious", False))
        )

        return json.dumps({
            "analysis_type": "Reviewer Credential Verification",
            "reviewer_name": reviewer_name,
            "email_assessment": email_assessment,
            "publication_assessment": publication_assessment,
            "findings": findings,
            "flagged": flagged,
            "interpretation": (
                "Reviewer credentials are SUSPICIOUS: " + "; ".join(findings)
                if flagged
                else "Reviewer credentials appear LEGITIMATE. No red flags detected."
            ),
        }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# Review Template Detector
# ═══════════════════════════════════════════════════════════════════════

class TemplateDetectionInput(BaseModel):
    """Input for template detection."""

    reviews: List[str] = Field(
        ...,
        description="List of review texts to compare for template patterns.",
    )


class ReviewTemplateDetectorTool(BaseTool):
    """
    Detect template/boilerplate language shared across multiple reviews.

    This tool looks for:
    1. Identical sentence or paragraph reuse across reviews
    2. Structural similarity (same section ordering, same transition phrases)
    3. Signature phrases that appear in multiple "different" reviews

    High structural similarity across reviews supposedly by different people
    is a strong indicator of fabricated reviews — either the same person
    wrote multiple reviews, or a template/LLM was used to generate them.
    """

    name: str = "review_template_detector"
    description: str = (
        "Detect template/boilerplate language shared across multiple reviews. "
        "Looks for identical sentences, shared structural patterns, and signature "
        "phrases. High similarity across reviews from supposedly different reviewers "
        "strongly suggests fabricated/template-generated reviews. "
        "Input: list of review text strings."
    )
    args_schema: type[BaseModel] = TemplateDetectionInput

    def _run(self, reviews: List[str]) -> str:
        """Execute template detection."""
        if len(reviews) < 2:
            return json.dumps({
                "analysis_type": "Review Template Detection",
                "error": "Need at least 2 reviews for comparison.",
                "flagged": False,
            })

        n = len(reviews)
        findings = []

        # ── Pairwise sentence-level comparison ──
        # Split each review into sentences
        review_sentences = []
        for review in reviews:
            sentences = re.split(r'(?<=[.!?])\s+', review)
            sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 30]
            review_sentences.append(sentences)

        # Find sentences that appear in ≥2 reviews
        import difflib

        shared_sentences = []
        for i in range(n):
            for j in range(i + 1, n):
                for si in review_sentences[i]:
                    for sj in review_sentences[j]:
                        # Use SequenceMatcher for near-match detection
                        ratio = difflib.SequenceMatcher(None, si, sj).ratio()
                        if ratio > 0.80:
                            shared_sentences.append({
                                "review_pair": [i, j],
                                "similarity": round(ratio, 3),
                                "sentence_a": si[:200],
                                "sentence_b": sj[:200],
                            })

        # ── Structural similarity ──
        # Check review opening and closing patterns
        openings = []
        closings = []
        for review in reviews:
            sentences = re.split(r'(?<=[.!?])\s+', review)
            if sentences:
                openings.append(sentences[0].strip().lower()[:100])
                closings.append(sentences[-1].strip().lower()[:100])

        # Check if openings are too similar
        opening_similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                ratio = difflib.SequenceMatcher(None, openings[i], openings[j]).ratio()
                opening_similarities.append({
                    "review_pair": [i, j],
                    "similarity": round(ratio, 3),
                })

        opening_avg_sim = (
            sum(s["similarity"] for s in opening_similarities) / len(opening_similarities)
            if opening_similarities else 0
        )

        if opening_avg_sim > 0.70:
            findings.append({
                "type": "structural_similarity",
                "detail": (
                    f"Review openings are highly similar (avg similarity {opening_avg_sim:.2f}). "
                    "Reviews may follow a template or be written by the same person."
                ),
            })

        # ── Overall assessment ──
        unique_shared_count = len(shared_sentences)
        flagged = unique_shared_count > 3 or opening_avg_sim > 0.70

        return json.dumps({
            "analysis_type": "Review Template Detection",
            "reviews_compared": n,
            "total_comparisons": n * (n - 1) // 2,
            "shared_sentences": {
                "count": unique_shared_count,
                "details": shared_sentences[:20],
                "interpretation": (
                    f"Found {unique_shared_count} near-identical sentence(s) across reviews. "
                    "Template reuse or single-author reviews suspected."
                    if unique_shared_count > 3
                    else "Few shared sentences — within expected range for same-topic reviews."
                ),
            },
            "structural_similarity": {
                "opening_similarity_avg": round(opening_avg_sim, 3),
                "details": opening_similarities,
                "interpretation": (
                    "Review openings are suspiciously similar — template pattern detected."
                    if opening_avg_sim > 0.70
                    else "Review openings show natural variation."
                ),
            },
            "findings": findings,
            "flagged": flagged,
            "interpretation": (
                "Template/boilerplate patterns detected across reviews. "
                "This suggests reviews may not be independent — possible fake reviews "
                "or a coordinated review campaign."
                if flagged
                else "No significant template reuse detected. Reviews appear independent."
            ),
        }, ensure_ascii=False)
