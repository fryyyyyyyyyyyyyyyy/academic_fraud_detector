"""
Text similarity tools — semantic and lexical plagiarism detection.

Semantic: uses sentence-transformers embeddings for paraphrase detection.
Lexical: uses difflib SequenceMatcher for verbatim copy detection.

Together they catch the full spectrum from exact copy to sophisticated rewording.
"""

import json
import logging
import re
from typing import Optional

from crewai.tools import BaseTool, tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Lazy-loaded model (loaded once on first use)
_similarity_model = None


def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load the sentence-transformer model."""
    global _similarity_model
    if _similarity_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformer model: {model_name}")
            _similarity_model = SentenceTransformer(model_name)
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
    return _similarity_model


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences with intelligent boundary detection."""
    # Handle common abbreviation patterns before splitting.
    # Use multiple simple replacements instead of variable-width look-behind
    # (Python's re does not support variable-width look-behind).
    abbreviations = [
        (r'\be\.g\.', 'e<DOT>g<DOT>'),
        (r'\bi\.e\.', 'i<DOT>e<DOT>'),
        (r'\bet al\.', 'et al<DOT>'),
        (r'\bvs\.', 'vs<DOT>'),
        (r'\bFig\.', 'Fig<DOT>'),
        (r'\bEq\.', 'Eq<DOT>'),
        (r'\bDr\.', 'Dr<DOT>'),
        (r'\bProf\.', 'Prof<DOT>'),
        (r'\bMr\.', 'Mr<DOT>'),
        (r'\bMs\.', 'Ms<DOT>'),
        (r'\bvol\.', 'vol<DOT>'),
        (r'\bno\.', 'no<DOT>'),
        (r'\betc\.', 'etc<DOT>'),
    ]
    for pattern, replacement in abbreviations:
        text = re.sub(pattern, replacement, text)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Restore abbreviation dots
    sentences = [s.replace("<DOT>", ".") for s in sentences]
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# ═══════════════════════════════════════════════════════════════════════
# Semantic Similarity Tool
# ═══════════════════════════════════════════════════════════════════════

class SemanticSimilarityInput(BaseModel):
    """Input for semantic similarity comparison."""

    text_a: str = Field(..., description="First text — the target/source paper text.")
    text_b: str = Field(..., description="Second text — the comparison paper text.")
    threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold above which to flag.",
    )
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformer model name.",
    )


class SemanticSimilarityTool(BaseTool):
    """
    Compute semantic similarity between two texts using sentence embeddings.

    Detects both verbatim AND paraphrased plagiarism:
    - Sentence-level comparison: each sentence in text_a vs each in text_b.
    - Full-passage fallback if text is too short for sentence splitting.
    - Returns flagged sentence pairs above the threshold.
    """

    name: str = "semantic_similarity_check"
    description: str = (
        "Compute semantic similarity between two text passages using sentence-transformer "
        "embeddings. Detects both verbatim and paraphrased plagiarism by comparing sentence-level "
        "embeddings. Returns cosine similarity scores and aligned passages above threshold. "
        "Use this as the primary tool for detecting sophisticated, paraphrased plagiarism."
    )
    args_schema: type[BaseModel] = SemanticSimilarityInput

    def _run(
        self,
        text_a: str,
        text_b: str,
        threshold: float = 0.80,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> str:
        """Compute semantic similarity between two texts."""
        try:
            model = _get_model(model_name)
        except Exception as e:
            return json.dumps({"error": str(e)})

        sentences_a = _split_sentences(text_a)
        sentences_b = _split_sentences(text_b)

        # Full-passage fallback for short texts
        if len(sentences_a) < 3 or len(sentences_b) < 3:
            return self._full_passage_compare(text_a, text_b, threshold, model)

        # Sentence-level comparison
        try:
            from sentence_transformers import util
            emb_a = model.encode(sentences_a, convert_to_tensor=True, show_progress_bar=False)
            emb_b = model.encode(sentences_b, convert_to_tensor=True, show_progress_bar=False)
            cosine_scores = util.cos_sim(emb_a, emb_b)
        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
            return json.dumps({"error": f"Embedding failed: {e}"})

        matches = []
        for i in range(len(sentences_a)):
            for j in range(len(sentences_b)):
                score = float(cosine_scores[i][j])
                if score > threshold:
                    matches.append({
                        "source_sentence": sentences_a[i][:300],
                        "target_sentence": sentences_b[j][:300],
                        "similarity": round(score, 4),
                        "source_index": i,
                        "target_index": j,
                    })

        overall_max = float(cosine_scores.max())
        overall_mean = float(cosine_scores.mean())

        return json.dumps({
            "overall_similarity": round(overall_mean, 4),
            "max_similarity": round(overall_max, 4),
            "flagged": len(matches) > 0,
            "match_count": len(matches),
            "total_pairs_compared": f"{len(sentences_a)} × {len(sentences_b)}",
            "threshold_used": threshold,
            "matches": sorted(matches, key=lambda m: m["similarity"], reverse=True)[:15],
            "method": "sentence-level-embedding",
        }, ensure_ascii=False)

    def _full_passage_compare(
        self,
        text_a: str,
        text_b: str,
        threshold: float,
        model,
    ) -> str:
        """Fallback: compare entire passages as single embeddings."""
        from sentence_transformers import util

        emb_a = model.encode(text_a, convert_to_tensor=True)
        emb_b = model.encode(text_b, convert_to_tensor=True)
        score = float(util.cos_sim(emb_a, emb_b))

        return json.dumps({
            "overall_similarity": round(score, 4),
            "max_similarity": round(score, 4),
            "flagged": score > threshold,
            "match_count": 1 if score > threshold else 0,
            "matches": [],
            "threshold_used": threshold,
            "method": "full-passage",
            "note": "Text too short for sentence-level comparison; using full-passage embedding.",
        }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# Lexical Plagiarism Tool
# ═══════════════════════════════════════════════════════════════════════
#
# We export BOTH the raw function (for testing / direct use) AND
# the CrewAI @tool-decorated version (for agent integration).

def _lexical_plagiarism_impl(
    text_a: str,
    text_b: str,
    min_match_length: int = 50,
) -> str:
    """
    Core implementation of lexical plagiarism detection using difflib.

    Separated from the @tool decorator so it can be tested independently
    without requiring the full CrewAI runtime.
    """
    import difflib

    # Normalize: lowercase both texts
    text_a_norm = text_a.lower()
    text_b_norm = text_b.lower()

    matcher = difflib.SequenceMatcher(None, text_a_norm, text_b_norm)
    blocks = matcher.get_matching_blocks()

    significant_matches = []
    total_matched_chars = 0
    for block in blocks:
        if block.size >= min_match_length:
            matched_text = text_a[block.a:block.a + min(block.size, 300)]
            significant_matches.append({
                "text_a_position": block.a,
                "text_b_position": block.b,
                "match_length": block.size,
                "matched_text": matched_text,
            })
            total_matched_chars += block.size

    ratio = matcher.ratio()

    return json.dumps({
        "overall_lexical_ratio": round(ratio, 4),
        "total_matched_characters": total_matched_chars,
        "significant_match_count": len(significant_matches),
        "min_match_length_used": min_match_length,
        "matches": sorted(significant_matches, key=lambda m: m["match_length"], reverse=True)[:20],
        "flagged": ratio > 0.60 or len(significant_matches) > 0,
        "severity_heuristic": (
            "critical" if ratio > 0.80 else
            "high" if ratio > 0.60 else
            "medium" if ratio > 0.40 else
            "low" if ratio > 0.20 else
            "none"
        ),
    }, ensure_ascii=False)


# CrewAI @tool wrapper (for agent integration)
lexical_plagiarism_check = tool("lexical_plagiarism_check")(_lexical_plagiarism_impl)
